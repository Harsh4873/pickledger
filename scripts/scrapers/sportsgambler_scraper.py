#!/usr/bin/env python3
"""SportsGambler scraper for NBA, NBA Summer, WNBA, MLB, FIFA, CFB, and NFL picks."""
from __future__ import annotations
import argparse, json, re, sys, unicodedata
from datetime import date, datetime
from typing import Any
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

NBA_URL = "https://www.sportsgambler.com/betting-tips/basketball/nba-predictions/"
# Dedicated NBA listing dropped JSON-LD SportsEvent cards; current tips live in
# pml-game rows on that page and, on mixed days, the basketball hub. Keep both
# so a CMS shuffle still finds matchup links. One working page is enough: NBA
# is optional at the provider runner, like CFB/NFL.
NBA_URLS = (
    NBA_URL,
    "https://www.sportsgambler.com/betting-tips/basketball/",
)
NBA_SUMMER_URLS = (
    NBA_URL,
    "https://www.sportsgambler.com/betting-tips/basketball/",
)
WNBA_URL = "https://www.sportsgambler.com/betting-tips/basketball/wnba-predictions/"
MLB_URL = "https://www.sportsgambler.com/betting-tips/baseball/"
FIFA_WORLD_CUP_URL = "https://www.sportsgambler.com/betting-tips/football/fifa-world-cup-predictions/"
# Dedicated NCAAF listing. `/betting-tips/ncaaf/` currently redirects here;
# keep both so a later CMS shuffle still finds the JSON-LD tip cards.
# Detail pages live under `/betting-tips/ncaaf/...` — that path is the
# strict league filter when a mixed American-football listing is reused.
CFB_URLS = (
    "https://www.sportsgambler.com/betting-tips/american-football/ncaa-college-football-predictions/",
    "https://www.sportsgambler.com/betting-tips/ncaaf/",
    "https://www.sportsgambler.com/betting-tips/american-football/",
)
CFB_DETAIL_PATH = "/ncaaf/"
# Dedicated NFL listing. Detail pages live under `/betting-tips/nfl/...` —
# that path is the strict league filter when a mixed American-football listing
# is reused, so NCAAF cards cannot leak into the NFL bucket.
NFL_URLS = (
    "https://www.sportsgambler.com/betting-tips/american-football/nfl-predictions/",
    "https://www.sportsgambler.com/betting-tips/nfl/",
    "https://www.sportsgambler.com/betting-tips/american-football/",
)
NFL_DETAIL_PATH = "/nfl/"
BLOCK_SIGNALS = (
    "attention required",
    "just a moment",
    "performing security verification",
    "sorry, you have been blocked",
    "cloudflare",
)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _parse_date(raw: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None

def _split_tip_odds(raw: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)\s*@\s*([+-]?\d+(?:\.\d+)?)$", _norm(raw))
    return (_norm(m.group(1)), _norm(m.group(2))) if m else (_norm(raw), "")

def _json_ld(soup: BeautifulSoup) -> list[Any]:
    out = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (tag.string or tag.get_text("\n", strip=True)).strip()
        if raw:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    return out

def _iter_nodes(v: Any):
    if isinstance(v, dict):
        yield v
        for child in v.values():
            yield from _iter_nodes(child)
    elif isinstance(v, list):
        for item in v:
            yield from _iter_nodes(item)

def _matchup_from_node(node: dict) -> str:
    name = _norm(node.get("name", ""))
    if name:
        return name
    comps = node.get("competitor")
    if isinstance(comps, list):
        teams = [_norm(t.get("name", "")) for t in comps if isinstance(t, dict) and _norm(t.get("name", ""))]
        if len(teams) >= 2:
            return f"{teams[0]} vs {teams[1]}"
    return ""

def _matchup_key(raw: str) -> tuple[str, str] | None:
    teams = re.split(r"\s+(?:vs\.?|@)\s+", _norm(raw), maxsplit=1, flags=re.IGNORECASE)
    if len(teams) != 2:
        return None
    aliases = {"turkiye": "turkey"}
    normalized = []
    for team in teams:
        text = unicodedata.normalize("NFKD", team)
        text = "".join(char for char in text if not unicodedata.combining(char))
        key = re.sub(r"[^a-z0-9]+", "", text.lower())
        normalized.append(aliases.get(key, key))
    normalized.sort()
    return (normalized[0], normalized[1]) if all(normalized) else None

def _expected_matchup_whitelist(expected_matchups: list[str] | None) -> dict[tuple[str, str], str]:
    raw_matchups = [_norm(matchup) for matchup in expected_matchups or [] if _norm(matchup)]
    if not raw_matchups:
        raise ValueError("a valid official matchup whitelist is required")
    expected: dict[tuple[str, str], str] = {}
    for matchup in raw_matchups:
        key = _matchup_key(matchup)
        if not key:
            raise ValueError(f"invalid official matchup whitelist entry: {matchup}")
        expected[key] = matchup
    return expected

SPORTSGAMBLER_ORIGIN = "https://www.sportsgambler.com"


def _absolute_href(href: str) -> str:
    return urljoin(SPORTSGAMBLER_ORIGIN, _norm(href))


def _matchup_from_detail_slug(detail_url: str) -> str:
    slug = re.search(
        r"/(?:nfl|ncaaf|basketball)/(.+?)-vs-(.+?)-prediction",
        _norm(detail_url),
    )
    if not slug:
        return ""
    return " vs ".join(team.replace("-", " ") for team in slug.groups())


def _listing_matchup_for_expected(
    matchup: str,
    detail_url: str,
    expected: dict[tuple[str, str], str],
    *,
    use_slug_fallback: bool,
) -> str:
    """Keep listing names when they already match; use the detail slug only for
    abbreviated football/pml cards whose JSON-LD/row text is too short."""
    matchup_key = _matchup_key(matchup)
    if matchup_key in expected:
        return matchup
    if not use_slug_fallback:
        return ""
    slug_matchup = _matchup_from_detail_slug(detail_url)
    slug_key = _matchup_key(slug_matchup)
    if slug_key in expected:
        return expected[slug_key]
    return ""


def _pml_league_label(meta_text: str) -> str:
    text = _norm(meta_text).upper()
    if "WNBA" in text:
        return "WNBA"
    if "SUMMER" in text:
        return "NBA SUMMER"
    if re.search(r"\bNCAA", text) or "COLLEGE" in text:
        return "NCAAB"
    if re.search(r"\bNBA\b", text):
        return "NBA"
    return ""


def _pml_matches_target(date_text: str, target: date | None) -> bool:
    """Drop other-day pml cards for the same teams (NBA series leftovers)."""
    if target is None:
        return True
    text = _norm(date_text)
    if not text:
        return True
    numeric = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", text)
    if numeric:
        first, second = int(numeric.group(1)), int(numeric.group(2))
        raw_year = numeric.group(3)
        years = [target.year] if not raw_year else (
            [2000 + int(raw_year)] if len(raw_year) == 2 else [int(raw_year)]
        )
        for year_value in years:
            for month, day in ((second, first), (first, second)):
                try:
                    if date(year_value, month, day) == target:
                        return True
                except ValueError:
                    continue
        return False
    weekday = target.strftime("%A").lower()
    aliases = {
        "monday": {"monday", "mon"},
        "tuesday": {"tuesday", "tue", "tues"},
        "wednesday": {"wednesday", "wed"},
        "thursday": {"thursday", "thu", "thur", "thurs"},
        "friday": {"friday", "fri"},
        "saturday": {"saturday", "sat"},
        "sunday": {"sunday", "sun"},
    }
    tokens = set(re.findall(r"[a-z]+", text.lower()))
    present = tokens & set().union(*aliases.values())
    if not present:
        return True
    return bool(present & aliases[weekday])


def _collect_pml_articles(
    soup: BeautifulSoup,
    league: str,
    expected: dict[tuple[str, str], str],
    href_contains: str | None,
    seen: set[str],
    target: date | None = None,
) -> list[dict]:
    """Current SportsGambler basketball listings use pml-game rows, not JSON-LD."""
    articles: list[dict] = []
    for game in soup.select("div.pml-game"):
        meta = game.select_one(".pml-meta")
        league_label = _pml_league_label(meta.get_text(" ", strip=True) if meta else "")
        if league_label and league_label != league:
            continue
        teams = [_norm(node.get_text(" ", strip=True)) for node in game.select(".pml-teams")]
        teams = [team for team in teams if team]
        matchup = f"{teams[0]} vs {teams[1]}" if len(teams) >= 2 else ""
        row = game.find_parent("div", class_=lambda value: bool(value and "pml-row" in value))
        tip_link = row.select_one("p.pml-tip-text a[href]") if row else None
        if tip_link is None:
            tip_link = game.find_next("p", class_=lambda value: bool(value and "pml-tip-text" in value))
            if tip_link is not None:
                tip_link = tip_link.select_one("a[href]")
        detail_url = _absolute_href(tip_link.get("href") if tip_link is not None else "")
        if href_contains and href_contains not in detail_url:
            continue
        official = _listing_matchup_for_expected(
            matchup,
            detail_url,
            expected,
            use_slug_fallback=True,
        )
        date_text = ""
        if meta is not None:
            spans = [_norm(span.get_text(" ", strip=True)) for span in meta.find_all("span", recursive=False)]
            date_text = next((span for span in spans if span and not span.startswith("-")), "")
        if not detail_url or not official or detail_url in seen:
            continue
        if not _pml_matches_target(date_text, target):
            continue
        seen.add(detail_url)
        articles.append({"url": detail_url, "matchup": official, "date": date_text})
    return articles

def scrape_basketball(
    target: date | None,
    url: str | tuple[str, ...],
    league: str,
    expected_matchups: list[str] | None = None,
    *,
    href_contains: str | None = None,
    require_complete_listings: bool = True,
) -> list[dict]:
    expected = _expected_matchup_whitelist(expected_matchups)
    articles, seen = [], set()
    listing_urls = (url,) if isinstance(url, str) else url
    listing_failures: list[str] = []
    loaded_listing = False
    for listing_url in dict.fromkeys(listing_urls):
        try:
            response = requests.get(listing_url, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:
            listing_failures.append(f"{listing_url}: {exc}")
            continue
        status = getattr(response, "status_code", 200)
        html = response.text
        if status != 200:
            listing_failures.append(f"{listing_url}: HTTP {status}")
            continue
        if any(signal in html[:12000].lower() for signal in BLOCK_SIGNALS):
            listing_failures.append(f"{listing_url}: provider challenge page")
            continue
        loaded_listing = True
        soup = BeautifulSoup(html, "html.parser")
        for obj in _json_ld(soup):
            for node in _iter_nodes(obj):
                item = node.get("item")
                if not isinstance(item, dict) or item.get("@type") != "SportsEvent":
                    continue
                detail_url = _absolute_href(item.get("url", ""))
                matchup = _matchup_from_node(item)
                if href_contains and href_contains not in detail_url:
                    continue
                official = _listing_matchup_for_expected(
                    matchup,
                    detail_url,
                    expected,
                    use_slug_fallback=href_contains in {CFB_DETAIL_PATH, NFL_DETAIL_PATH},
                )
                if not detail_url or not official or detail_url in seen:
                    continue
                seen.add(detail_url)
                articles.append({"url": detail_url, "matchup": official, "date": item.get("startDate", "")})
        articles.extend(_collect_pml_articles(soup, league, expected, href_contains, seen, target))
    if not loaded_listing:
        raise RuntimeError(
            f"unable to load {league} prediction listing(s): "
            f"{'; '.join(listing_failures[:2]) or 'unknown transport failure'}"
        )
    if listing_failures and require_complete_listings:
        raise RuntimeError(
            f"incomplete {league} listing coverage: "
            f"{'; '.join(listing_failures[:2])}"
        )

    rows = []
    missing: list[str] = []
    blocked: list[str] = []
    for art in articles:
        try:
            response = requests.get(art["url"], headers=HEADERS, timeout=30)
            status = getattr(response, "status_code", 200)
            html = response.text
            if status != 200 or any(signal in html[:12000].lower() for signal in BLOCK_SIGNALS):
                reason = f"HTTP {status}" if status != 200 else "provider challenge page"
                blocked.append(f"{art['url']}: {reason}")
                continue
            detail = BeautifulSoup(html, "html.parser")
        except Exception:
            missing.append(art["url"])
            continue
        prediction = ""
        for cont in detail.select("div.tpbot_container"):
            tip_link = cont.select_one("a.tpbot_tip")
            if not tip_link:
                continue
            title_el = cont.select_one(".tpbot_title")
            if title_el and "prediction" not in title_el.get_text().lower():
                continue
            spans = tip_link.select("span")
            prediction = _norm(spans[-1].get_text(" ", strip=True)) if spans else _norm(tip_link.get_text(" ", strip=True))
            if prediction:
                break
        if not prediction:
            missing.append(art["url"])
            continue
        tip, odds = _split_tip_odds(prediction)
        if not tip:
            missing.append(art["url"])
            continue
        rows.append({"datetime": art["date"], "league": league, "matchup": art["matchup"], "tip": tip, "odds": odds, "href": art["url"]})
    if blocked:
        message = (
            f"partial {league} scrape: parsed {len(rows)} of {len(articles)} listed prediction page(s); "
            f"blocked {len(blocked)} detail page(s) (Cloudflare/HTTP): {'; '.join(blocked[:3])}"
        )
        if missing:
            message += f"; missing {', '.join(missing[:3])}"
        raise RuntimeError(message)
    if missing:
        raise RuntimeError(
            f"partial {league} scrape: parsed {len(rows)} of {len(articles)} listed prediction page(s); "
            f"missing {', '.join(missing[:3])}"
        )
    return rows

def scrape_nba(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    return scrape_basketball(
        target,
        NBA_URLS,
        "NBA",
        expected_matchups,
        require_complete_listings=False,
    )

def scrape_nba_summer(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    return scrape_basketball(target, NBA_SUMMER_URLS, "NBA SUMMER", expected_matchups)

def scrape_wnba(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    return scrape_basketball(target, WNBA_URL, "WNBA", expected_matchups)

def scrape_fifa_world_cup(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    return scrape_basketball(target, FIFA_WORLD_CUP_URL, "FIFA WC", expected_matchups)

def scrape_cfb(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    """NCAAF tip cards from the dedicated college listing, never NFL."""
    return scrape_basketball(
        target,
        CFB_URLS,
        "CFB",
        expected_matchups,
        href_contains=CFB_DETAIL_PATH,
        require_complete_listings=False,
    )

def scrape_nfl(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    """NFL tip cards from the dedicated NFL listing, never NCAAF."""
    return scrape_basketball(
        target,
        NFL_URLS,
        "NFL",
        expected_matchups,
        href_contains=NFL_DETAIL_PATH,
        require_complete_listings=False,
    )

def scrape_mlb(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    expected = _expected_matchup_whitelist(expected_matchups)
    response = requests.get(MLB_URL, headers=HEADERS, timeout=30)
    status = getattr(response, "status_code", 200)
    html = response.text
    if status != 200:
        raise RuntimeError(f"unable to load MLB prediction listing(s): {MLB_URL}: HTTP {status}")
    if any(signal in html[:12000].lower() for signal in BLOCK_SIGNALS):
        raise RuntimeError(f"unable to load MLB prediction listing(s): {MLB_URL}: provider challenge page")
    soup = BeautifulSoup(html, "html.parser")
    rows, seen = [], set()
    for item in soup.select("div.tipbox_item"):
        title_spans = item.select(".tipsbox_title h3 > span")
        matchup = _norm(title_spans[0].get_text(" ", strip=True)) if title_spans else ""
        meta_spans = item.select(".tipsbox_title .tipsbox_meta span")
        date_text = _norm(meta_spans[0].get_text(" ", strip=True)) if meta_spans else ""
        league_text = _norm(" ".join(s.get_text(" ", strip=True) for s in meta_spans[1:])).lstrip("-").strip()
        if league_text.upper() != "MLB":
            continue
        tip_spans = item.select(".tipbox_tip span")
        prediction = _norm(tip_spans[-1].get_text(" ", strip=True)) if tip_spans else ""
        tip, odds = _split_tip_odds(prediction)
        if not matchup or not tip:
            continue
        if _matchup_key(matchup) not in expected:
            continue
        key = (matchup, tip)
        if key in seen:
            continue
        seen.add(key)
        anchor = item.get("id", "").strip()
        href = MLB_URL + (f"#{anchor}" if anchor else "")
        rows.append({"datetime": date_text, "league": "MLB", "matchup": matchup, "tip": tip, "odds": odds, "href": href})
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", "-s", default="nba")
    ap.add_argument("--date", "-d", default=None)
    ap.add_argument("--expected-matchup", action="append", default=[])
    args = ap.parse_args()
    expected_matchups = [
        _norm(matchup)
        for matchup in args.expected_matchup
        if _matchup_key(matchup)
    ]
    if not expected_matchups or len(expected_matchups) != len(args.expected_matchup):
        print("Error: SportsGambler requires a valid official matchup whitelist.", file=sys.stderr)
        sys.exit(1)
    sport = args.sport.strip().lower()
    target = _parse_date(args.date) if args.date else None
    try:
        if sport in ("nba", "basketball"):
            rows = scrape_nba(target, expected_matchups)
        elif sport in ("nba_summer", "nba_summer_league", "summer_league"):
            rows = scrape_nba_summer(target, expected_matchups)
        elif sport == "wnba":
            rows = scrape_wnba(target, expected_matchups)
        elif sport in ("mlb", "baseball"):
            rows = scrape_mlb(target, expected_matchups)
        elif sport in ("fifa", "fifa_world_cup", "football", "soccer", "world_cup"):
            rows = scrape_fifa_world_cup(target, expected_matchups)
        elif sport in ("cfb", "ncaaf", "college_football", "ncaa"):
            rows = scrape_cfb(target, expected_matchups)
        elif sport == "nfl":
            rows = scrape_nfl(target, expected_matchups)
        else:
            raise ValueError(
                "supported sports: nba/basketball, nba_summer, wnba, mlb/baseball, "
                "fifa_world_cup/soccer, cfb/ncaaf/college_football, nfl"
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if not rows:
        print("No picks found.")
        sys.exit(0)
    for r in rows:
        print("\n" + "━" * 32)
        print(f"Match:          {r['matchup']}")
        print(f"Date/Time:      {r['datetime'] or '[not found]'}")
        print(f"League:         {r['league']}")
        print(f"Tip:            {r['tip']}")
        print(f"Odds:           {r['odds'] or '[not found]'}")
        print(f"Source URL:     {r['href']}")
        print("━" * 32)

if __name__ == "__main__":
    main()
