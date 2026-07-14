#!/usr/bin/env python3
"""SportsGambler scraper for NBA, WNBA, MLB, and FIFA World Cup picks."""
from __future__ import annotations
import argparse, json, re, sys, unicodedata
from datetime import date, datetime
from typing import Any
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

NBA_URL = "https://www.sportsgambler.com/betting-tips/basketball/nba-predictions/"
WNBA_URL = "https://www.sportsgambler.com/betting-tips/basketball/wnba-predictions/"
MLB_URL = "https://www.sportsgambler.com/betting-tips/baseball/"
FIFA_WORLD_CUP_URL = "https://www.sportsgambler.com/betting-tips/football/fifa-world-cup-predictions/"

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

def scrape_basketball(
    target: date | None,
    url: str,
    league: str,
    expected_matchups: list[str] | None = None,
) -> list[dict]:
    expected = _expected_matchup_whitelist(expected_matchups)
    html = requests.get(url, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    articles, seen = [], set()
    for obj in _json_ld(soup):
        for node in _iter_nodes(obj):
            item = node.get("item")
            if not isinstance(item, dict) or item.get("@type") != "SportsEvent":
                continue
            url = _norm(item.get("url", ""))
            matchup = _matchup_from_node(item)
            if not url or not matchup or url in seen:
                continue
            matchup_key = _matchup_key(matchup)
            if matchup_key not in expected:
                continue
            seen.add(url)
            articles.append({"url": url, "matchup": matchup, "date": item.get("startDate", "")})

    rows = []
    missing: list[str] = []
    for art in articles:
        try:
            detail = BeautifulSoup(requests.get(art["url"], headers=HEADERS, timeout=30).text, "html.parser")
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
    if missing:
        raise RuntimeError(
            f"partial {league} scrape: parsed {len(rows)} of {len(articles)} listed prediction page(s); "
            f"missing {', '.join(missing[:3])}"
        )
    return rows

def scrape_nba(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    return scrape_basketball(target, NBA_URL, "NBA", expected_matchups)

def scrape_wnba(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    return scrape_basketball(target, WNBA_URL, "WNBA", expected_matchups)

def scrape_fifa_world_cup(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    return scrape_basketball(target, FIFA_WORLD_CUP_URL, "FIFA WC", expected_matchups)

def scrape_mlb(target: date | None, expected_matchups: list[str] | None = None) -> list[dict]:
    expected = _expected_matchup_whitelist(expected_matchups)
    html = requests.get(MLB_URL, headers=HEADERS, timeout=30).text
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
        elif sport == "wnba":
            rows = scrape_wnba(target, expected_matchups)
        elif sport in ("mlb", "baseball"):
            rows = scrape_mlb(target, expected_matchups)
        elif sport in ("fifa", "fifa_world_cup", "football", "soccer", "world_cup"):
            rows = scrape_fifa_world_cup(target, expected_matchups)
        else:
            raise ValueError("supported sports: nba/basketball, wnba, mlb/baseball, fifa_world_cup/soccer")
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
