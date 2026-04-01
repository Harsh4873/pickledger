#!/usr/bin/env python3
"""
Scores24.live Prediction Scraper (Olostep-backed)
=================================================
Fetches prediction data from scores24.live through the Olostep API.
If a specific matchup isn't on the main index, automatically
scans sub-league directories and follows more listing pages to hunt it down.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from html import unescape
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

def _load_local_env() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in (".env", ".env.local"):
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError:
            continue


_load_local_env()

def _default_playwright_browsers_path() -> str:
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if configured:
        return configured
    darwin_cache = os.path.expanduser("~/Library/Caches/ms-playwright")
    if sys.platform == "darwin" and os.path.isdir(darwin_cache):
        return darwin_cache
    # Fall back to package-local browsers for environments like Render.
    return "0"


os.environ["PLAYWRIGHT_BROWSERS_PATH"] = _default_playwright_browsers_path()

try:
    from camoufox.sync_api import Camoufox
    from playwright.sync_api import TimeoutError as PwTimeout
except Exception:
    Camoufox = None

    class PwTimeout(Exception):
        pass

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEAGUE_TO_SPORT = {
    "nba": "basketball", "nfl": "american-football", "mlb": "baseball",
    "nhl": "ice-hockey", "premier league": "soccer", "premier-league": "soccer",
    "la liga": "soccer", "la-liga": "soccer", "serie a": "soccer",
    "serie-a": "soccer", "bundesliga": "soccer", "ligue 1": "soccer",
    "ligue-1": "soccer", "champions league": "soccer", "champions-league": "soccer",
    "europa league": "soccer", "europa-league": "soccer",
    "fa cup": "soccer", "fa-cup": "soccer", "atp": "tennis", "wta": "tennis",
}

VALID_SPORTS = [
    "soccer", "basketball", "tennis", "ice-hockey", "volleyball", "handball",
    "baseball", "american-football", "rugby", "cricket", "mma", "boxing",
    "snooker", "futsal", "table-tennis", "waterpolo", "badminton", "darts",
    "csgo", "dota2", "lol", "horse-racing",
]

SUGGESTIVE_PHRASES = [
    "confident", "certain", "sure", "guaranteed", "safe bet", "strong pick",
    "can't lose", "well-liked", "highly recommended", "best bet", "lock",
    "must bet", "no doubt", "clearly", "obvious",
]

BASE = os.environ.get("SCORES24_BASE_URL", "https://scores24.live").rstrip("/")
SPORT_TO_ESPNSLUG = {
    "nba": ("basketball", "nba"),
    "nhl": ("hockey", "nhl"),
    "mlb": ("baseball", "mlb"),
}
TEAM_ABBREVIATION_ALIASES = {
    "WAS": {"WSH"},
    "WSH": {"WAS"},
    "NOP": {"NO"},
    "NO": {"NOP"},
    "GSW": {"GS"},
    "GS": {"GSW"},
    "PHX": {"PHO"},
    "PHO": {"PHX"},
    "SAS": {"SA"},
    "SA": {"SAS"},
    "NYK": {"NY"},
    "NY": {"NYK"},
    "BKN": {"BRK"},
    "BRK": {"BKN"},
}
LEAGUE_LISTING_HINTS = {
    "nba": [
        f"{BASE}/en/basketball",
        f"{BASE}/en/basketball/l-usa-nba",
    ],
    "mlb": [
        f"{BASE}/en/baseball",
        f"{BASE}/en/baseball/l-usa-mlb",
    ],
}
MAX_LISTING_PAGES = 24
SCORES24_BACKEND = os.environ.get("SCORES24_BACKEND", "auto").strip().lower() or "auto"
OLOSTEP_API_KEY = os.environ.get("OLOSTEP_API_KEY", "").strip()
OLOSTEP_API_BASE = os.environ.get("OLOSTEP_API_BASE", "https://api.olostep.com").rstrip("/")
OLOSTEP_COUNTRY = os.environ.get("OLOSTEP_COUNTRY", "").strip()
try:
    OLOSTEP_WAIT_MS = max(0, int(os.environ.get("OLOSTEP_WAIT_BEFORE_SCRAPING_MS", "2500")))
except ValueError:
    OLOSTEP_WAIT_MS = 2500
ALLOW_OLOSTEP_AUTO = os.environ.get("ALLOW_OLOSTEP_AUTO", "").strip().lower() in {"1", "true", "yes", "on"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def resolve_sport(raw: str) -> str:
    key = raw.lower().strip()
    if key.replace(" ", "-") in VALID_SPORTS:
        return key.replace(" ", "-")
    if key in LEAGUE_TO_SPORT:
        return LEAGUE_TO_SPORT[key]
    if key.replace(" ", "-") in LEAGUE_TO_SPORT:
        return LEAGUE_TO_SPORT[key.replace(" ", "-")]
    return key.replace(" ", "-")


def date_variants(date_str: str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return []
    day = str(dt.day)
    return [
        f"{day} {dt.strftime('%b')}",     
        f"{day} {dt.strftime('%B')}",     
        dt.strftime("%d.%m.%y"),          
        dt.strftime("%Y-%m-%d"),          
    ]


def _normalize_match_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _team_code_aliases(value: str) -> set[str]:
    code = re.sub(r"[^A-Za-z]", "", str(value or "")).upper()
    if not code:
        return set()
    return {code, *TEAM_ABBREVIATION_ALIASES.get(code, set())}


def _team_matches_text(team_text: str, candidate_text: str) -> bool:
    team_norm = _normalize_match_text(team_text)
    cand_norm = _normalize_match_text(candidate_text)
    if not team_norm or not cand_norm:
        return False

    team_aliases = _team_code_aliases(team_text)
    cand_aliases = _team_code_aliases(candidate_text)
    if team_aliases and cand_aliases and team_aliases & cand_aliases:
        return True

    if team_norm == cand_norm:
        return True
    if len(team_norm) > 3 and (team_norm in cand_norm or cand_norm in team_norm):
        return True

    team_last = team_norm.split()[-1] if team_norm.split() else ""
    if len(team_last) >= 3:
        cand_tokens = cand_norm.split()
        if team_last in cand_tokens:
            return True
    return False


def _scoreboard_date_key(date_str: str | None) -> str:
    if date_str:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y%m%d")
            except ValueError:
                continue
    return datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d")


def _fetch_scoreboard(sport: str, league: str, yyyymmdd: str) -> dict | None:
    url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard?dates={yyyymmdd}"
    req = Request(url, headers={"User-Agent": "Scores24Scraper/1.0"})
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (HTTPError, OSError, TimeoutError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def fetch_daily_matchups(sport_slug: str, date_str: str | None) -> list[dict[str, str]]:
    mapping = SPORT_TO_ESPNSLUG.get(sport_slug)
    if not mapping:
        return []

    board = _fetch_scoreboard(mapping[0], mapping[1], _scoreboard_date_key(date_str))
    if not board:
        return []

    matchups: list[dict[str, str]] = []
    for event in board.get("events", []):
        comps = event.get("competitions", []) if isinstance(event, dict) else []
        if not comps:
            continue
        comp0 = comps[0]
        competitors = comp0.get("competitors", []) if isinstance(comp0, dict) else []
        if len(competitors) != 2:
            continue

        home_comp = next((comp for comp in competitors if comp.get("homeAway") == "home"), competitors[0])
        away_comp = next((comp for comp in competitors if comp.get("homeAway") == "away"), competitors[1])

        def _competitor_label(comp: dict) -> str:
            team = comp.get("team", {}) if isinstance(comp, dict) else {}
            for field in ("displayName", "shortDisplayName", "name", "abbreviation"):
                value = str(team.get(field, "")).strip()
                if value:
                    return value
            return ""

        home = _competitor_label(home_comp)
        away = _competitor_label(away_comp)
        if not home or not away:
            continue

        matchups.append({
            "home": home,
            "away": away,
            "home_code": str((home_comp.get("team", {}) or {}).get("abbreviation", "")).strip(),
            "away_code": str((away_comp.get("team", {}) or {}).get("abbreviation", "")).strip(),
            "event_id": str(event.get("id") or ""),
        })

    return matchups


def _schedule_team_matches(team_text: str, candidate_text: str) -> bool:
    team_norm = _normalize_match_text(team_text)
    cand_norm = _normalize_match_text(candidate_text)
    if not team_norm or not cand_norm:
        return False

    team_aliases = _team_code_aliases(team_text)
    cand_aliases = _team_code_aliases(candidate_text)
    if team_aliases and cand_aliases and team_aliases & cand_aliases:
        return True

    if team_norm == cand_norm:
        return True
    if len(team_norm) > 3 and team_norm in cand_norm:
        return True
    if len(cand_norm) > 3 and cand_norm in team_norm:
        return True

    team_tokens = team_norm.split()
    cand_tokens = cand_norm.split()
    if len(team_tokens) >= 2 and len(cand_tokens) >= 2:
        return " ".join(team_tokens[-2:]) == " ".join(cand_tokens[-2:])
    return False


def _fetch_mlb_schedule_http(date_str: str) -> list[dict[str, str]]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    req = Request(url, headers={"User-Agent": "Scores24Scraper/1.0"})
    with urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    games: list[dict[str, str]] = []
    for day in data.get("dates", []) if isinstance(data, dict) else []:
        for game in day.get("games", []) if isinstance(day, dict) else []:
            teams = game.get("teams", {}) if isinstance(game, dict) else {}
            away = ((teams.get("away") or {}).get("team") or {}).get("name")
            home = ((teams.get("home") or {}).get("team") or {}).get("name")
            game_datetime = str(game.get("gameDate") or "").strip()
            away_name = str(away or "").strip()
            home_name = str(home or "").strip()
            if not away_name or not home_name:
                continue
            games.append({
                "away_name": away_name,
                "home_name": home_name,
                "game_datetime": game_datetime,
            })
    return games


def _validate_mlb_picks_against_schedule(picks, target_date_str):
    """
    Filter/flag MLB picks that don't match real statsapi schedule.
    Returns picks with 'start_time' populated from statsapi if matched,
    or with 'start_time' = '' and 'unverified' = True if not matched.
    """
    try:
        try:
            import statsapi  # type: ignore
        except Exception:
            statsapi = None
        from datetime import datetime, timedelta

        base = datetime.strptime(target_date_str, "%Y-%m-%d")
        # Check yesterday, today, tomorrow to handle UTC offset.
        real_games: list[dict[str, str]] = []
        for delta in (0, -1, 1):
            d = base + timedelta(days=delta)
            if statsapi is not None:
                try:
                    day_games = statsapi.schedule(date=d.strftime("%m/%d/%Y"), sportId=1)
                except Exception:
                    day_games = _fetch_mlb_schedule_http(d.strftime("%Y-%m-%d"))
            else:
                day_games = _fetch_mlb_schedule_http(d.strftime("%Y-%m-%d"))
            for game in day_games or []:
                away_name = str(game.get("away_name") or "").strip()
                home_name = str(game.get("home_name") or "").strip()
                if not away_name or not home_name:
                    continue
                real_games.append({
                    "away_name": away_name,
                    "home_name": home_name,
                    "game_datetime": str(game.get("game_datetime") or "").strip(),
                })

        result = []
        for pick in picks:
            away_name = str(pick.get("away_team") or "").strip()
            home_name = str(pick.get("home_team") or "").strip()
            if (not away_name or not home_name) and " vs " in str(pick.get("matchup") or ""):
                left, right = [part.strip() for part in str(pick.get("matchup") or "").split(" vs ", 1)]
                home_name = home_name or left
                away_name = away_name or right

            matched_time = ""
            for game in real_games:
                away_match = _schedule_team_matches(away_name, str(game.get("away_name") or ""))
                home_match = _schedule_team_matches(home_name, str(game.get("home_name") or ""))
                reverse_away_match = _schedule_team_matches(away_name, str(game.get("home_name") or ""))
                reverse_home_match = _schedule_team_matches(home_name, str(game.get("away_name") or ""))
                if (away_match and home_match) or (reverse_away_match and reverse_home_match):
                    matched_time = str(game.get("game_datetime") or "").strip()
                    break

            if matched_time:
                pick["start_time"] = matched_time
                pick["unverified"] = False
            else:
                pick["start_time"] = ""
                pick["unverified"] = True
                print(f"[WARN] No statsapi match for: {away_name or '?'} vs {home_name or '?'}")
            result.append(pick)
        return result

    except Exception as e:
        print(f"[WARN] Schedule validation failed: {e}")
        return picks


def _prediction_matches_matchup(pred: dict, matchup: dict[str, str]) -> bool:
    home = str(pred.get("homeTeam", "") or "").strip()
    away = str(pred.get("awayTeam", "") or "").strip()
    if not home or not away:
        return False
    direct = _team_matches_text(matchup.get("home", ""), home) and _team_matches_text(matchup.get("away", ""), away)
    reverse = _team_matches_text(matchup.get("home", ""), away) and _team_matches_text(matchup.get("away", ""), home)
    return direct or reverse


def _card_matches_matchup(card: dict, matchup: dict[str, str]) -> bool:
    home = str(card.get("home", "") or "").strip()
    away = str(card.get("away", "") or "").strip()
    href = str(card.get("href", "") or "").strip()
    direct = bool(home and away and _team_matches_text(matchup.get("home", ""), home) and _team_matches_text(matchup.get("away", ""), away))
    reverse = bool(home and away and _team_matches_text(matchup.get("home", ""), away) and _team_matches_text(matchup.get("away", ""), home))
    if direct or reverse:
        return True
    blob = f"{home} {away} {href}"
    return _team_matches_text(matchup.get("home", ""), blob) and _team_matches_text(matchup.get("away", ""), blob)


def _matches_requested_date(date_str: str, card: dict, variants: list[str]) -> bool:
    combined = f"{card.get('isoDate','')} {card.get('visDate','')}".lower()
    if any(v.lower() in combined for v in variants):
        return True
    # Some cards expose UTC startDate while listing/day filters are local (ET).
    iso = (card.get("isoDate") or "").strip()
    if not iso:
        return False
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d") == date_str
    except Exception:
        return False


def _build_scraped_pick(pred: dict, card: dict, url: str) -> dict:
    card_copy = dict(card or {})
    home = str(pred.get("homeTeam") or card_copy.get("home", "") or "").strip()
    away = str(pred.get("awayTeam") or card_copy.get("away", "") or "").strip()
    matchup = f"{home} vs {away}" if home and away else str(pred.get("matchTitle", "") or "").strip()
    return {
        "pred": pred,
        "card": card_copy,
        "url": url,
        "matchup": matchup,
        "away_team": away,
        "home_team": home,
        "start_time": str(card_copy.get("start_time") or card_copy.get("isoDate") or "").strip(),
        "unverified": False,
    }


def _emit_scraped_predictions(entries: list[dict], requested_league: str | None, target_date_str: str | None, sport_label: str, stats: dict) -> None:
    if _requested_league_key(requested_league) == "mlb":
        effective_date = (target_date_str or "").strip() or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        entries = _validate_mlb_picks_against_schedule(entries, effective_date)

    for entry in entries:
        if entry.get("unverified"):
            stats["unverified"] = stats.get("unverified", 0) + 1
            continue
        card = dict(entry.get("card") or {})
        start_time = str(entry.get("start_time") or "").strip()
        if start_time:
            card["start_time"] = start_time
        print_prediction(entry.get("pred") or {}, card, sport_label, str(entry.get("url") or ""))
        stats["loaded"] += 1

def guess_urls(sport_slug: str, date_str: str, matchup_str: str) -> list[str]:
    if not date_str or "vs" not in matchup_str.lower(): return []
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        url_date = dt.strftime("%d-%m-%Y")
    except:
        return []

    parts = matchup_str.lower().split("vs")
    p1 = re.sub(r'[^a-z0-9]+', '-', parts[0].strip())
    p2 = re.sub(r'[^a-z0-9]+', '-', parts[1].strip())

    base_path = f"{BASE}/en/{sport_slug}/m-{url_date}"
    
    return [
        f"{base_path}-{p1}-{p2}-prediction",
        f"{base_path}-{p2}-{p1}-prediction",
        f"{base_path}-{p1}-{p2}",
        f"{base_path}-{p2}-{p1}"
    ]


def _probe_date_candidates(date_str: str | None) -> list[str]:
    if not date_str:
        return []
    try:
        base = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return [date_str]
    return [
        base.strftime("%Y-%m-%d"),
        (base + timedelta(days=1)).strftime("%Y-%m-%d"),
        (base - timedelta(days=1)).strftime("%Y-%m-%d"),
    ]


def _guess_matchup_prediction_urls(sport_slug: str, matchup: dict[str, str], date_str: str | None) -> list[str]:
    matchup_str = f"{matchup.get('home', '')} vs {matchup.get('away', '')}".strip()
    if " vs " not in matchup_str:
        return []
    guessed: list[str] = []
    for probe_date in _probe_date_candidates(date_str):
        guessed.extend(guess_urls(sport_slug, probe_date, matchup_str))
    return _dedupe_prediction_links(guessed)


def listing_url_candidates(sport_slug: str, requested_sport: str | None = None) -> list[str]:
    """Try multiple known Scores24 listing URL patterns."""
    requested_key = (requested_sport or "").strip().lower().replace(" ", "-")
    candidates = list(LEAGUE_LISTING_HINTS.get(requested_key, [])) + [
        f"{BASE}/en/predictions/{sport_slug}",
        f"{BASE}/en/{sport_slug}/predictions",
        f"{BASE}/en/{sport_slug}",
    ]
    seen = set()
    out = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _looks_like_cloudflare_block(text: str) -> bool:
    blob = (text or "").lower()
    signals = [
        "attention required",
        "just a moment",
        "sorry, you have been blocked",
        "performing security verification",
        "cf-error-details",
        "cloudflare",
    ]
    return any(sig in blob for sig in signals)


def _olostep_markdown_content(scrape_obj: dict) -> str:
    result = scrape_obj.get("result") if isinstance(scrape_obj, dict) else {}
    markdown = result.get("markdown_content") if isinstance(result, dict) else ""
    return markdown if isinstance(markdown, str) else ""


def _log_cloudflare_block_once(url: str, warned_urls: set[str], label: str) -> None:
    if url in warned_urls:
        return
    warned_urls.add(url)
    print(f"{label}: ❌ Cloudflare block detected for {url}")


def scan_suggestive(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for phrase in SUGGESTIVE_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\b"
        m = re.search(rf".{{0,40}}{pattern}.{{0,40}}", lower)
        if m:
            hits.append(m.group(0).strip())
    return hits


def should_use_olostep() -> bool:
    if SCORES24_BACKEND == "olostep":
        return True
    if SCORES24_BACKEND in {"playwright", "browser", "camoufox"}:
        return False
    if ALLOW_OLOSTEP_AUTO and OLOSTEP_API_KEY:
        return True
    return Camoufox is None


def _olostep_headers() -> dict[str, str]:
    if not OLOSTEP_API_KEY:
        raise RuntimeError("OLOSTEP_API_KEY is not set")
    return {
        "Authorization": f"Bearer {OLOSTEP_API_KEY}",
        "Content-Type": "application/json",
    }


def _olostep_scrape(url: str, formats: list[str] | None = None) -> dict:
    payload: dict[str, object] = {
        "url_to_scrape": url,
        "formats": formats or ["markdown"],
    }
    if OLOSTEP_WAIT_MS:
        payload["wait_before_scraping"] = OLOSTEP_WAIT_MS
    if OLOSTEP_COUNTRY:
        payload["country"] = OLOSTEP_COUNTRY

    req = Request(
        f"{OLOSTEP_API_BASE}/v1/scrapes",
        headers=_olostep_headers(),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    with urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("unexpected Olostep response shape")
    return data


def _collect_markdown_links(markdown: str) -> list[str]:
    matches = re.findall(r"\((https?://[^)\s]+|/[^)\s]+)\)", markdown or "")
    out: list[str] = []
    for match in matches:
        link = (match or "").strip()
        if not link or link.startswith("data:"):
            continue
        out.append(link)
    return out


def _normalize_prediction_url(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    return href if href.startswith("http") else urljoin(f"{BASE}/", href.lstrip("/"))


def _normalize_prediction_page_url(url: str) -> str:
    full_url = _normalize_prediction_url(url)
    if not full_url:
        return ""
    parts = urlsplit(full_url)
    path = parts.path.rstrip("/")
    if re.search(r"/m-\d{2}-\d{2}-\d{4}-", path) and not path.endswith("-prediction"):
        path = f"{path}-prediction"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _is_prediction_page_url(url: str) -> bool:
    normalized = _normalize_prediction_page_url(url)
    if not normalized:
        return False
    return bool(re.search(r"/m-\d{2}-\d{2}-\d{4}-", normalized))


def _dedupe_prediction_links(raw_links: list[str]) -> list[str]:
    seen_links = set()
    normalized_links: list[str] = []
    for href in raw_links:
        norm = _normalize_prediction_page_url(href)
        if not norm or norm in seen_links:
            continue
        seen_links.add(norm)
        normalized_links.append(norm)
    return normalized_links


def _url_matches_matchup(url: str, matchup_parts: list[str]) -> bool:
    blob = (url or "").lower().replace("-", " ")
    return all(part in blob for part in matchup_parts)


def _target_date_slug(date_str: str | None) -> str:
    if not date_str:
        return ""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d-%m-%Y")
    except ValueError:
        return ""


def _target_date_slugs(date_str: str | None) -> list[str]:
    if not date_str:
        return []
    try:
        base = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        slug = _target_date_slug(date_str)
        return [slug] if slug else []
    return [
        base.strftime("%d-%m-%Y"),
        (base + timedelta(days=1)).strftime("%d-%m-%Y"),
    ]


def _href_matches_requested_date(url: str, date_str: str | None, variants: list[str]) -> bool:
    blob = str(url or "").lower()
    if any(v.lower().replace(" ", "-") in blob for v in variants):
        return True
    return any(slug and slug in blob for slug in _target_date_slugs(date_str))


def _extract_prediction_urls(scrape_obj: dict, date_str: str | None, strict_date: bool = True) -> list[str]:
    result = scrape_obj.get("result") if isinstance(scrape_obj, dict) else {}
    links_on_page = result.get("links_on_page") if isinstance(result, dict) else []
    markdown = result.get("markdown_content") if isinstance(result, dict) else ""

    raw_links: list[str] = []
    if isinstance(links_on_page, list):
        for item in links_on_page:
            if isinstance(item, str):
                raw_links.append(item)
            elif isinstance(item, dict):
                href = item.get("url") or item.get("href")
                if isinstance(href, str):
                    raw_links.append(href)

    raw_links.extend(_collect_markdown_links(markdown if isinstance(markdown, str) else ""))

    date_slugs = _target_date_slugs(date_str)
    filtered: list[str] = []
    seen = set()
    for href in raw_links:
        full_url = _normalize_prediction_page_url(href)
        if not full_url:
            continue
        low = full_url.lower()
        if "/m-" not in low and "-prediction" not in low:
            continue
        if strict_date and date_slugs and not any(
            f"/m-{slug}-" in low or f"/m-{slug}" in low or f"/{slug}-" in low or f"/{slug}" in low
            for slug in date_slugs
        ):
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        filtered.append(full_url)
    return filtered


def _extract_subleague_urls(scrape_obj: dict, requested_key: str | None = None) -> list[str]:
    result = scrape_obj.get("result") if isinstance(scrape_obj, dict) else {}
    links_on_page = result.get("links_on_page") if isinstance(result, dict) else []
    markdown = result.get("markdown_content") if isinstance(result, dict) else ""
    raw_links: list[str] = []
    if isinstance(links_on_page, list):
        for item in links_on_page:
            if isinstance(item, str):
                raw_links.append(item)
            elif isinstance(item, dict):
                href = item.get("url") or item.get("href")
                if isinstance(href, str):
                    raw_links.append(href)
    raw_links.extend(_collect_markdown_links(markdown if isinstance(markdown, str) else ""))

    out: list[str] = []
    seen = set()
    for href in raw_links:
        full_url = _normalize_prediction_url(href)
        if not full_url:
            continue
        low = full_url.lower()
        if "/l-" not in low or "/predictions" not in low:
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        out.append(full_url)
    requested = _requested_league_key(requested_key)
    if requested == "nba":
        out = [url for url in out if "nba" in url.lower() or "usa-nba" in url.lower()]
    elif requested == "mlb":
        out = [url for url in out if "mlb" in url.lower() or "usa-mlb" in url.lower()]
    return out


def _normalize_md_text(text: str) -> str:
    text = unescape(text or "")
    text = text.replace("\r", "")
    return re.sub(r"[ \t]+", " ", text)


def _first_match(patterns: list[str], text: str, flags: int = re.IGNORECASE) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            return " ".join((match.group(1) or "").split()).strip()
    return ""


def _extract_matchup(markdown: str) -> tuple[str, str]:
    patterns = [
        r"\[([A-Z][A-Za-z0-9 .&'\-()]+)\]\(/en/[^)]+/t-[^)]+\)\s*\\-\s*\[([A-Z][A-Za-z0-9 .&'\-()]+)\]\(/en/[^)]+/t-[^)]+\)",
        r"^([A-Z][A-Za-z0-9 .&'\-()]+)\s+vs\.?\s+([A-Z][A-Za-z0-9 .&'\-()]+?)\s+Prediction(?:\s+Today)?\b",
        r"([A-Z][A-Za-z0-9 .&'\-()]+)\s*-\s*([A-Z][A-Za-z0-9 .&'\-()]+)\s+prediction",
    ]
    for pattern in patterns:
        match = re.search(pattern, markdown, re.IGNORECASE | re.MULTILINE)
        if match:
            return " ".join(match.group(1).split()).strip(), " ".join(match.group(2).split()).strip()
    return "", ""


def _extract_title_matchup(markdown: str) -> tuple[str, str]:
    compact = (markdown or "").splitlines()[0] if markdown else ""
    match = re.search(
        r"^\s*([A-Z][A-Za-z0-9 .&'\-()]+?)\s+vs\.?\s+([A-Z][A-Za-z0-9 .&'\-()]+?)(?:\s+(?:Prediction|Live Stream)\b|$)",
        compact,
        re.IGNORECASE,
    )
    if not match:
        return "", ""
    return " ".join(match.group(1).split()).strip(), " ".join(match.group(2).split()).strip()


def _slug_similarity(candidate: str, target: str) -> int:
    cand = re.sub(r"[^a-z0-9]+", " ", candidate.lower()).strip()
    targ = re.sub(r"[^a-z0-9]+", " ", target.lower()).strip()
    if not cand or not targ:
        return 0
    score = 0
    cand_tokens = cand.split()
    targ_tokens = targ.split()
    cand_set = set(cand_tokens)
    targ_set = set(targ_tokens)
    score += len(cand_set & targ_set) * 5
    if cand.endswith(targ) or targ.endswith(cand):
        score += 4
    if cand.startswith(targ) or targ.startswith(cand):
        score += 3
    return score


def _extract_matchup_from_url(url: str, markdown: str) -> tuple[str, str]:
    full_url = _normalize_prediction_page_url(url)
    match = re.search(r"/m-\d{2}-\d{2}-\d{4}-(.+?)(?:-prediction)?/?$", full_url)
    if not match:
        return "", ""
    slug = match.group(1).strip("-")
    tokens = [tok for tok in slug.split("-") if tok]
    if len(tokens) < 2:
        return "", ""

    hinted_home, hinted_away = _extract_title_matchup(markdown)
    best_score = -1
    best_pair = ("", "")
    for idx in range(1, len(tokens)):
        home = " ".join(tok.capitalize() for tok in tokens[:idx])
        away = " ".join(tok.capitalize() for tok in tokens[idx:])
        score = 0
        if hinted_home or hinted_away:
            score += _slug_similarity(home, hinted_home)
            score += _slug_similarity(away, hinted_away)
        score += min(idx, len(tokens) - idx)
        if score > best_score:
            best_score = score
            best_pair = (home, away)
    return best_pair


def _requested_league_key(value: str | None) -> str:
    return re.sub(r"[\s_]+", "-", str(value or "").strip().lower())


def _url_matches_expected_matchup(url: str, matchup: dict[str, str]) -> bool:
    url_home, url_away = _extract_matchup_from_url(url, "")
    if url_home and url_away:
        direct = _team_matches_text(matchup.get("home", ""), url_home) and _team_matches_text(matchup.get("away", ""), url_away)
        reverse = _team_matches_text(matchup.get("home", ""), url_away) and _team_matches_text(matchup.get("away", ""), url_home)
        if direct or reverse:
            return True
    blob = (url or "").replace("-", " ")
    return _team_matches_text(matchup.get("home", ""), blob) and _team_matches_text(matchup.get("away", ""), blob)


def _extract_date_text(markdown: str) -> str:
    compact = markdown[:2500]
    dotted_match = re.search(r"\b(\d{2}\.\d{2}\.\d{2})\b[\s\S]{0,60}\b(\d{2}:\d{2})\b", compact)
    if dotted_match:
        return f"{dotted_match.group(1)} {dotted_match.group(2)}"
    patterns = [
        r"\b([A-Z][a-z]+ \d{1,2}, \d{4}(?:, \d{1,2}:\d{2}(?:\s*[AP]M)?)?)\b",
        r"\b(\d{1,2} [A-Z][a-z]{2,8}(?: \d{4})?(?:, \d{2}:\d{2})?)\b",
        r"\b(\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2})?)\b",
    ]
    return _first_match(patterns, markdown, 0)


def _extract_tip(markdown: str) -> str:
    line_patterns = [
        r"Our choice\s+([^\n]+)",
        r"Our choice[:\s]*([^\n]+)",
        r"(Total (?:goals|points) (?:Over|Under)\s*\([\d.]+\))",
        r"([A-Za-z0-9 .&'\-]+ Handicap\s*\([+-]?[\d.]+\))",
        r"([A-Za-z0-9 .&'\-]+ Total (?:goals|points) (?:Over|Under)\s*\([\d.]+\))",
        r"(Both Teams To Score\s*\((?:Yes|No)\))",
        r"([A-Za-z0-9 .&'\-]+ to win)",
        r"((?:Over|Under)\s*\([\d.]+\))",
    ]
    tip = _first_match(line_patterns, markdown)
    tip = re.sub(r"\s+at odds of\s+[^\s]+\*?", "", tip, flags=re.IGNORECASE)
    return tip.rstrip(" .")


def _extract_confidence(markdown: str) -> str:
    candidates = re.findall(r"\b(\d{1,3})%\b", markdown or "")
    for candidate in candidates:
        try:
            value = int(candidate)
        except ValueError:
            continue
        if 30 <= value <= 100:
            return f"{value}%"
    return ""


def _decimal_to_american(value: float) -> str:
    if value <= 1:
        return ""
    if value >= 2:
        return f"+{int(round((value - 1) * 100))}"
    return str(int(round(-100 / (value - 1))))


def _fractional_to_american(text: str) -> str:
    try:
        numerator, denominator = text.split("/", 1)
        num = float(numerator)
        den = float(denominator)
        if den == 0:
            return ""
        decimal = 1 + (num / den)
        return _decimal_to_american(decimal)
    except (ValueError, ZeroDivisionError):
        return ""


def _extract_odds(markdown: str) -> str:
    american = _first_match([r"at odds of\s*([+-]\d{3,4})\b"], markdown)
    if american:
        return american
    fractional = _first_match([r"at odds of\s*(\d+/\d+)\*?"], markdown)
    if fractional:
        return _fractional_to_american(fractional)
    decimal = _first_match([r"at odds of\s*(\d\.\d{1,2})\b"], markdown)
    if decimal:
        try:
            return _decimal_to_american(float(decimal))
        except ValueError:
            return ""
    return ""


def _extract_league(markdown: str, fallback: str) -> str:
    breadcrumb = re.findall(r"\[([A-Za-z0-9 .&'\-]+)\]\(/en/[^)]+/l-[^)]+/predictions\)", markdown or "")
    if breadcrumb:
        return " ".join(breadcrumb[-1].split()).strip()
    league = _first_match(
        [
            r"(?:League|Tournament|Competition)[:\s]*([^\n]+)",
            r"Breadcrumbs?[^\n]*\n([^\n]+)",
        ],
        markdown,
    )
    return league or fallback


def _extract_olostep_prediction(scrape_obj: dict, sport_label: str, source_url: str = "") -> dict:
    result = scrape_obj.get("result") if isinstance(scrape_obj, dict) else {}
    markdown = result.get("markdown_content") if isinstance(result, dict) else ""
    markdown = _normalize_md_text(markdown if isinstance(markdown, str) else "")
    home, away = _extract_matchup(markdown)
    if not home or not away or len(away.split()) == 1:
        url_home, url_away = _extract_matchup_from_url(source_url, markdown)
        if url_home and url_away:
            home = url_home
            away = url_away
    return {
        "homeTeam": home,
        "awayTeam": away,
        "date": _extract_date_text(markdown),
        "tip": _extract_tip(markdown),
        "primaryOdds": _extract_odds(markdown),
        "confidence": _extract_confidence(markdown),
        "votePcts": [],
        "totalVotes": "",
        "trends": [],
        "oddsTable": [],
        "pageText": markdown[:10000],
        "league": _extract_league(markdown, sport_label),
    }


def _scrape_prediction_with_retry(url: str, sport_label: str, attempts: int = 2) -> dict:
    last_pred: dict = {}
    for _ in range(max(1, attempts)):
        scrape = _olostep_scrape(url, ["markdown"])
        pred = _extract_olostep_prediction(scrape, sport_label, url)
        last_pred = pred
        if pred.get("tip"):
            return pred
    return last_pred


def run_with_olostep(args) -> int:
    if args.url and _is_prediction_page_url(args.url):
        direct_url = _normalize_prediction_page_url(args.url)
        pred = _scrape_prediction_with_retry(direct_url, "Unknown")
        if not pred.get("tip"):
            print(f"⚠️  Loaded but extraction failed: {direct_url}")
            return 1
        print_prediction(pred, {"league": pred.get("league", "")}, "Unknown", direct_url)
        return 0

    sport_slug = resolve_sport(args.sport) if args.sport else "Unknown"
    sport_label = (args.sport or sport_slug).upper()
    requested_league = _requested_league_key(args.sport or sport_slug)
    listing_urls = [args.url] if args.url else listing_url_candidates(sport_slug, requested_league)
    listing_url = listing_urls[0]
    matchup_parts = [p.strip().lower() for p in args.matchup.split("vs") if p.strip()] if args.matchup else []
    expected_matchups = fetch_daily_matchups(requested_league, args.date) if requested_league in SPORT_TO_ESPNSLUG else []
    strict_date = requested_league not in SPORT_TO_ESPNSLUG
    print(f"Sport:          {sport_slug if args.sport else 'Unknown'}")
    if args.date:
        print(f"Date requested: {args.date}")
    if args.matchup:
        print(f"Matchup:        {args.matchup}")
    print(f"Listing URL:    {listing_url}")
    print("Backend:        olostep")

    filtered: list[dict] = []
    aggregate_links: list[dict] = []
    queued_subleague_urls: list[str] = []
    warned_cloudflare_urls: set[str] = set()
    used_listing_url = ""
    for cand in listing_urls:
        try:
            scrape = _olostep_scrape(cand, ["markdown"])
        except Exception:
            continue
        markdown = _olostep_markdown_content(scrape)
        if _looks_like_cloudflare_block(markdown):
            _log_cloudflare_block_once(cand, warned_cloudflare_urls, "Listing page status")
            continue
        normalized_links = _dedupe_prediction_links(_extract_prediction_urls(scrape, args.date, strict_date=strict_date))
        if matchup_parts and normalized_links:
            normalized_links = [href for href in normalized_links if _url_matches_matchup(href, matchup_parts)]
        elif expected_matchups and normalized_links:
            normalized_links = [
                href for href in normalized_links
                if any(_url_matches_expected_matchup(href, matchup) for matchup in expected_matchups)
            ]

        subleague_urls = _extract_subleague_urls(scrape, requested_league)
        if isinstance(subleague_urls, list):
            queued_subleague_urls.extend(str(url).strip() for url in subleague_urls if str(url).strip())
        if not normalized_links:
            continue
        used_listing_url = cand
        for href in normalized_links:
            aggregate_links.append({
                "href": href,
                "home": "",
                "away": "",
                "isoDate": "",
                "visDate": "",
                "visTime": "",
                "start_time": "",
                "league": sport_label,
                "confidence": "",
            })

    if queued_subleague_urls and (matchup_parts or expected_matchups):
        ordered_subleague_urls = list(dict.fromkeys(queued_subleague_urls))
        if requested_league:
            ordered_subleague_urls.sort(key=lambda url: (0 if requested_league in url.lower() else 1, url))
        for subleague_url in ordered_subleague_urls[:20]:
            try:
                sub_scrape = _olostep_scrape(subleague_url, ["markdown"])
            except Exception:
                continue
            sub_markdown = _olostep_markdown_content(sub_scrape)
            if _looks_like_cloudflare_block(sub_markdown):
                _log_cloudflare_block_once(subleague_url, warned_cloudflare_urls, "Subleague page status")
                continue
            sub_links = _dedupe_prediction_links(_extract_prediction_urls(sub_scrape, args.date, strict_date=strict_date))
            if matchup_parts:
                sub_links = [href for href in sub_links if _url_matches_matchup(href, matchup_parts)]
            elif expected_matchups:
                sub_links = [
                    href for href in sub_links
                    if any(_url_matches_expected_matchup(href, matchup) for matchup in expected_matchups)
                ]
            for href in sub_links:
                aggregate_links.append({
                    "href": href,
                    "home": "",
                    "away": "",
                    "isoDate": "",
                    "visDate": "",
                    "visTime": "",
                    "start_time": "",
                    "league": sport_label,
                    "confidence": "",
                })
    filtered = _dedupe_cards(aggregate_links)

    if used_listing_url and used_listing_url != listing_url:
        print(f"Listing URL fallback: {used_listing_url}")
    if not filtered and not expected_matchups:
        print("Listing page status: ❌ No prediction links found via Olostep")
        return 1
    print("Listing page status: ✅ Page loaded")

    if args.matchup:
        filtered = [
            card for card in filtered
            if _url_matches_matchup(card["href"], matchup_parts)
        ]
        print(f"Matches found for request on listing: {len(filtered)}")

    if not filtered and not expected_matchups:
        return 0

    if expected_matchups and not args.matchup:
        stats = {"loaded": 0, "404": 0, "no_data": 0, "unverified": 0}
        discovered_urls = [str(card.get("href", "")).strip() for card in filtered if str(card.get("href", "")).strip()]

        def _resolve_matchup(matchup: dict[str, str]) -> tuple[dict[str, str], str, dict]:
            candidate_urls = [
                href for href in discovered_urls
                if _url_matches_expected_matchup(href, matchup)
            ]
            guessed_urls = _guess_matchup_prediction_urls(sport_slug, matchup, args.date)
            candidate_urls = _dedupe_prediction_links(candidate_urls + guessed_urls)

            for attempts in (1, 2):
                for full_url in candidate_urls:
                    try:
                        pred = _scrape_prediction_with_retry(full_url, sport_label, attempts=attempts)
                    except Exception:
                        continue
                    if not pred.get("tip"):
                        continue
                    if not (_prediction_matches_matchup(pred, matchup) or _url_matches_expected_matchup(full_url, matchup)):
                        continue
                    return matchup, full_url, pred
            return matchup, "", {}

        resolved: list[tuple[dict[str, str], str, dict]] = [({}, "", {}) for _ in expected_matchups]
        max_workers = max(1, min(len(expected_matchups), 4 if requested_league == "nba" else 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_resolve_matchup, matchup): idx
                for idx, matchup in enumerate(expected_matchups)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    resolved[idx] = future.result()
                except Exception:
                    resolved[idx] = (expected_matchups[idx], "", {})

        entries: list[dict] = []
        for matchup, chosen_url, pred in resolved:
            if not chosen_url:
                stats["no_data"] += 1
                continue

            entries.append(_build_scraped_pick(
                pred,
                {
                    "league": pred.get("league", sport_label),
                    "home": matchup.get("home", ""),
                    "away": matchup.get("away", ""),
                    "isoDate": "",
                    "visDate": "",
                    "visTime": "",
                    "start_time": "",
                    "confidence": "",
                },
                chosen_url,
            ))

        _emit_scraped_predictions(entries, requested_league, args.date, sport_label, stats)

        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("SUMMARY")
        print(f"Expected slate games:          {len(expected_matchups)}")
        print(f"Individual pages loaded:        {stats['loaded']}")
        print(f"Individual pages 404'd:         {stats['404']}")
        print(f"Matchups with no data:          {stats['no_data']}")
        if stats.get("unverified"):
            print(f"Unverified MLB picks skipped:   {stats['unverified']}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return 0 if stats["loaded"] else 1

    stats = {"loaded": 0, "404": 0, "no_data": 0, "unverified": 0}
    entries: list[dict] = []
    for card in filtered:
        full_url = card["href"]
        pred = _scrape_prediction_with_retry(full_url, sport_label)
        if not pred.get("tip"):
            print(f"\n⚠️  Loaded but no prediction data found on page: {full_url}")
            stats["no_data"] += 1
            continue
        if expected_matchups and not any(_prediction_matches_matchup(pred, matchup) for matchup in expected_matchups):
            continue
        entry_card = dict(card)
        entry_card["league"] = pred.get("league", sport_label)
        entries.append(_build_scraped_pick(pred, entry_card, full_url))

    _emit_scraped_predictions(entries, requested_league, args.date, sport_label, stats)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("SUMMARY")
    print(f"Total matches extracted:        {len(filtered)}")
    print(f"Individual pages loaded:        {stats['loaded']}")
    print(f"Individual pages 404'd:         {stats['404']}")
    if stats.get("unverified"):
        print(f"Unverified MLB picks skipped:   {stats['unverified']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BROWSER HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_browser():
    proxy_conf = None
    launch_args = {
        "headless": False,
        "humanize": True,
    }

    proxy_server = os.environ.get("PLAYWRIGHT_PROXY_SERVER", "").strip()
    if proxy_server:
        proxy_conf = {"server": proxy_server}
        proxy_user = os.environ.get("PLAYWRIGHT_PROXY_USERNAME", "").strip()
        proxy_pass = os.environ.get("PLAYWRIGHT_PROXY_PASSWORD", "").strip()
        if proxy_user:
            proxy_conf["username"] = proxy_user
        if proxy_pass:
            proxy_conf["password"] = proxy_pass
        launch_args["proxy"] = proxy_conf

    return Camoufox(**launch_args)


def make_context(browser):
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        timezone_id="America/New_York",
    )
    # Reduce obvious automation fingerprints; does not bypass hard blocks by itself.
    ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
    return ctx


def load_page(ctx, url: str, wait_ms: int = 6000):
    page = ctx.new_page()
    try:
        resp = page.goto(url, timeout=25000, wait_until="domcontentloaded")
        status = resp.status if resp else 0

        # Cloudflare checks can transiently return 403 first; allow one reload pass.
        if status in (403, 429):
            page.wait_for_timeout(4500)
            try:
                resp2 = page.reload(timeout=25000, wait_until="domcontentloaded")
                if resp2:
                    status = resp2.status
            except Exception:
                pass

        if status == 404:
            page.close()
            return None, status

        # Treat Cloudflare block pages as hard failures, even if status is 200.
        title = ""
        body_head = ""
        try:
            title = page.title()
        except Exception:
            pass
        try:
            body_head = page.evaluate("() => (document.body?.innerText || '').slice(0, 1200)")
        except Exception:
            pass
        if _looks_like_cloudflare_block(f"{title}\n{body_head}"):
            page.close()
            return None, 403

        page.wait_for_timeout(wait_ms)
        return page, status
    except PwTimeout:
        page.close()
        return None, 408
    except Exception:
        page.close()
        return None, 500


def pause_for_manual_cloudflare(ctx, url: str):
    return


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LISTING PAGE & DEEP SEARCH PARSERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEEP_SEARCH_JS = """
() => {
    const links = document.querySelectorAll('a[href*="/l-"]');
    const urls = new Set();
    for (const a of links) {
        const href = a.getAttribute('href');
        if (href && href.includes('/predictions')) {
            urls.add(href.startsWith('http') ? href : new URL(href, window.location.origin).href);
        }
    }
    return Array.from(urls);
}
"""

DIRECT_PREDICTION_LINKS_JS = r"""
() => {
    const links = document.querySelectorAll('a[href*="/m-"]');
    const urls = new Set();
    for (const a of links) {
        const href = a.getAttribute('href');
        if (!href) continue;
        const abs = href.startsWith('http') ? href : new URL(href, window.location.origin).href;
        if (/-prediction$/i.test(abs) || /\/m-\d{2}-\d{2}-\d{4}-/i.test(abs)) {
            urls.add(abs);
        }
    }
    return Array.from(urls);
}
"""

def extract_subleague_links(page):
    try:
        return page.evaluate(DEEP_SEARCH_JS)
    except Exception as e:
        return []


def extract_prediction_links(page):
    try:
        return page.evaluate(DIRECT_PREDICTION_LINKS_JS)
    except Exception:
        return []


LISTING_JS = """
() => {
    const cards = document.querySelectorAll('span[data-testid="PredictionCard"]');
    return Array.from(cards).map(card => {
        const link = card.querySelector('a[itemprop="url"]');
        const href = link ? link.getAttribute('href') : '';

        const homeMeta = card.querySelector('p[itemprop="homeTeam"] meta[itemprop="name"]');
        const awayMeta = card.querySelector('p[itemprop="awayTeam"] meta[itemprop="name"]');
        const home = homeMeta ? homeMeta.content : (card.querySelector('p[itemprop="homeTeam"]')?.textContent?.trim() || '');
        const away = awayMeta ? awayMeta.content : (card.querySelector('p[itemprop="awayTeam"]')?.textContent?.trim() || '');

        const dateMeta = card.querySelector('meta[itemprop="startDate"]');
        const isoDate = dateMeta ? dateMeta.content : '';

        let startTime = '';
        const timeCandidates = [
            card.querySelector('time[datetime]'),
            card.querySelector('[datetime]'),
            card.querySelector('[data-time]'),
            card.querySelector('[data-start-time]'),
            dateMeta,
        ].filter(Boolean);
        for (const node of timeCandidates) {
            const value =
                node.getAttribute('datetime') ||
                node.getAttribute('data-time') ||
                node.getAttribute('data-start-time') ||
                node.getAttribute('content') ||
                '';
            if (value) {
                startTime = value.trim();
                break;
            }
        }

        const dateSpans = card.querySelectorAll('span');
        let visDate = '';
        let visTime = '';
        for (const sp of dateSpans) {
            const t = sp.textContent.trim();
            if (/^\\d{1,2}\\s+[A-Za-z]{3}/.test(t)) visDate = t;
            if (/^\\d{2}:\\d{2}$/.test(t)) visTime = t;
        }
        if (!startTime) {
            if (visDate && visTime) startTime = `${visDate} ${visTime}`;
            else if (isoDate) startTime = isoDate;
            else if (visDate) startTime = visDate;
            else if (visTime) startTime = visTime;
        }

        const league = card.querySelector('span')
            ? [...card.querySelectorAll('span')].find(s => {
                  const txt = s.textContent.trim();
                  return txt.length > 1 && txt.length < 40 && !/\\d/.test(txt) && txt !== 'Prediction';
              })?.textContent?.trim() || ''
            : '';

        let confidence = '';
        const allSpans = card.querySelectorAll('span');
        for (const sp of allSpans) {
            const t = sp.textContent.trim();
            if (/^\\d{1,3}%$/.test(t) || /^[-+]\\d{3,4}$/.test(t)) {
                confidence = t;
            }
        }

        return { href, home, away, isoDate, visDate, visTime, start_time: startTime, league, confidence };
    });
}
"""

def extract_listing_cards(page):
    try:
        return page.evaluate(LISTING_JS)
    except Exception as e:
        return []


def hydrate_listing_page(page, rounds: int = 12):
    """Scroll/click to reveal cards that load lazily on listing pages."""
    stale_rounds = 0
    for _ in range(rounds):
        try:
            before = page.evaluate("() => document.querySelectorAll('span[data-testid=\"PredictionCard\"]').length")
        except Exception:
            break
        try:
            page.evaluate("() => window.scrollBy(0, Math.max(800, Math.floor(window.innerHeight * 0.85)))")
            page.wait_for_timeout(900)
            clicked_more = page.evaluate(
                """() => {
                    const controls = Array.from(document.querySelectorAll('button, a, [role="button"]'));
                    let clicked = 0;
                    for (const el of controls) {
                        const text = (el.textContent || '').trim();
                        if (!/show\\s+more|load\\s+more|more\\s+predictions|more\\s+games|see\\s+more/i.test(text)) continue;
                        try {
                            el.click();
                            clicked += 1;
                            if (clicked >= 4) break;
                        } catch (err) {
                            continue;
                        }
                    }
                    return clicked;
                }"""
            )
            if clicked_more:
                page.wait_for_timeout(1200)
            after = page.evaluate("() => document.querySelectorAll('span[data-testid=\"PredictionCard\"]').length")
        except Exception:
            break
        if after <= before:
            stale_rounds += 1
            if stale_rounds >= 2:
                break
        else:
            stale_rounds = 0


def _normalize_listing_card(card: dict, source_url: str) -> dict:
    normalized = dict(card or {})
    href = _normalize_prediction_page_url(normalized.get("href", ""))
    if href:
        normalized["href"] = href
    normalized["start_time"] = str(normalized.get("start_time") or normalized.get("isoDate") or "").strip()
    normalized["sourceListingUrl"] = source_url
    return normalized


def _collect_listing_cards(page, source_url: str) -> tuple[list[dict], list[str]]:
    cards: list[dict] = []
    subleague_urls: list[str] = []
    try:
        page_cards = extract_listing_cards(page)
        if isinstance(page_cards, list):
            cards.extend(_normalize_listing_card(card, source_url) for card in page_cards if isinstance(card, dict))
    except Exception:
        pass

    try:
        direct_links = extract_prediction_links(page)
        for href in direct_links if isinstance(direct_links, list) else []:
            cards.append(_normalize_listing_card({
                "href": href,
                "home": "",
                "away": "",
                "isoDate": "",
                "visDate": "",
                "visTime": "",
                "start_time": "",
                "league": "",
                "confidence": "",
            }, source_url))
    except Exception:
        pass

    try:
        subleague_urls = extract_subleague_links(page)
    except Exception:
        subleague_urls = []

    return cards, subleague_urls


def _dedupe_cards(cards: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for card in cards:
        href = _normalize_prediction_page_url(str(card.get("href", "") or ""))
        if not href or href in seen:
            continue
        seen.add(href)
        copy = dict(card)
        copy["href"] = href
        deduped.append(copy)
    return deduped


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INDIVIDUAL PREDICTION PAGE PARSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREDICTION_JS = """
() => {
    const result = {};
    result.matchTitle = document.title || '';

    const dateEl = document.querySelector('span[data-testid="MatchHeaderHeadDate"]');
    result.date = dateEl ? dateEl.textContent.trim() : '';

    const homeMeta = document.querySelector('meta[itemprop="homeTeam"]');
    const awayMeta = document.querySelector('meta[itemprop="awayTeam"]');
    result.homeTeam = homeMeta ? homeMeta.content : '';
    result.awayTeam = awayMeta ? awayMeta.content : '';

    result.tip = '';
    const allText = document.body.innerText;
    const ourChoiceMatch = allText.match(/Our choice[:\\s]*([^\\n]+)/i);
    if (ourChoiceMatch) {
        result.tip = ourChoiceMatch[1].trim();
    }
    if (!result.tip) {
        const tipPatterns = [
            /(?:prediction|tip|pick)[:\\s]+((?:over|under|total|handicap|win|draw|home|away)[^\\n]{0,60})/i,
            /(Total goals (?:Over|Under) \\([\\d.]+\\))/i,
            /((?:Home|Away|Draw)\\s+(?:Win|Team))/i,
            /(Handicap\\s+[-+]?[\\d.]+)/i,
        ];
        for (const pat of tipPatterns) {
            const m = allText.match(pat);
            if (m) { result.tip = m[1].trim(); break; }
        }
    }

    const valueSpans = document.querySelectorAll('span.value');
    result.allOdds = Array.from(valueSpans).map(s => s.textContent.trim()).filter(t => /^[-+]?\\d/.test(t));
    result.primaryOdds = result.allOdds.length > 0 ? result.allOdds[0] : '';

    result.confidence = '';
    const spans = document.querySelectorAll('span, div');
    for (const s of spans) {
        const t = s.textContent.trim();
        if (/^\\d{1,3}%$/.test(t) && parseInt(t) > 30 && parseInt(t) <= 100) {
            result.confidence = t;
            break;
        }
    }

    result.userVotes = '';
    const voteTotalEl = [...document.querySelectorAll('div')].find(d => /total votes cast/i.test(d.textContent) && d.textContent.length < 50);
    result.totalVotes = voteTotalEl ? voteTotalEl.textContent.trim() : '';
    
    const votePcts = [];
    const allDivs = document.querySelectorAll('div');
    let foundVoteSection = false;
    for (const d of allDivs) {
        const t = d.textContent.trim();
        if (/total votes cast/i.test(t)) foundVoteSection = true;
        if (foundVoteSection && /^\\d{1,3}%$/.test(t)) {
            votePcts.push(t);
            if (votePcts.length >= 2) break;
        }
    }
    if (votePcts.length < 2) {
        for (const d of allDivs) {
            const t = d.textContent.trim();
            if (/^\\d{1,3}%$/.test(t) && !votePcts.includes(t)) {
                votePcts.push(t);
                if (votePcts.length >= 2) break;
            }
        }
    }
    result.votePcts = votePcts;

    const trendEls = document.querySelectorAll('[data-testid="TrendContent"]');
    result.trends = Array.from(trendEls).map(el => el.textContent.trim());
    if (result.trends.length === 0) {
        const lis = document.querySelectorAll('li');
        result.trends = Array.from(lis)
            .map(l => l.textContent.trim())
            .filter(t => t.length > 20 && t.length < 200 && /\\d/.test(t));
    }

    result.oddsTable = [];
    const oddsLinks = document.querySelectorAll('a');
    for (const a of oddsLinks) {
        const divs = a.querySelectorAll('div');
        if (divs.length >= 2) {
            const label = divs[0].textContent.trim();
            const value = divs[divs.length - 1].textContent.trim();
            if (/^(W[12]|Draw|X|Over|Under|1|2|Handicap)/i.test(label) && /^[-+]?\\d/.test(value)) {
                result.oddsTable.push({ market: label, odds: value });
            }
        }
    }

    result.pageText = allText.substring(0, 10000);
    return result;
}
"""

def extract_prediction(page):
    try:
        return page.evaluate(PREDICTION_JS)
    except Exception as e:
        return {"error": str(e)}


def _prediction_matches_requested_sport(pred: dict, card: dict, requested_sport: str | None) -> bool:
    requested = _requested_league_key(requested_sport)
    if requested not in {"nba", "mlb"}:
        return True

    league_text = " ".join(
        str(value or "").strip().lower()
        for value in (
            pred.get("league"),
            card.get("league"),
        )
        if str(value or "").strip()
    )
    if not league_text:
        return True

    tokens = {
        "nba": ("nba", "basketball"),
        "mlb": ("mlb", "baseball"),
    }[requested]
    return any(token in league_text for token in tokens)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT FORMATTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fmt(label: str, value: str, fallback: str = "[not found on page]") -> str:
    return f"{label:<15}{value if value else fallback}"


def print_prediction(pred: dict, card: dict, sport_label: str, url: str):
    home = pred.get("homeTeam") or card.get("home", "")
    away = pred.get("awayTeam") or card.get("away", "")
    match_str = f"{home} vs {away}" if home and away else pred.get("matchTitle", "[unknown]")

    date_str = str(card.get("start_time") or "").strip() or pred.get("date") or card.get("visDate", "")
    if not card.get("start_time") and card.get("visTime"):
        date_str = f"{date_str}, {card['visTime']}" if date_str else card["visTime"]

    league = card.get("league") or sport_label
    tip = pred.get("tip", "")
    odds = pred.get("primaryOdds", "")
    confidence = pred.get("confidence") or card.get("confidence", "")

    vote_parts = pred.get("votePcts", [])
    total_votes = pred.get("totalVotes", "")
    user_vote = ""
    if len(vote_parts) >= 2:
        user_vote = f"{vote_parts[0]} vs {vote_parts[1]}"
        if total_votes:
            user_vote += f" ({total_votes})"

    trends = pred.get("trends", [])
    odds_table = pred.get("oddsTable", [])
    warnings = scan_suggestive(pred.get("pageText", ""))

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(fmt("Match:", match_str))
    print(fmt("Date/Time:", date_str))
    print(fmt("League:", league))
    print(fmt("Tip:", tip))
    print(fmt("Odds:", odds))
    print(fmt("Confidence:", confidence))
    print(fmt("User vote:", user_vote))

    if trends:
        print("Stat trends:")
        for t in trends:
            print(f"  • {t}")
    else:
        print(fmt("Stat trends:", ""))

    if odds_table:
        print("Best odds:")
        for row in odds_table:
            print(f"  {row['market']:<12} {row['odds']}")

    print(fmt("Source URL:", url))
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for w in warnings:
        print(f"⚠️  Suggestive language detected: \"{w}\"")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN WORKFLOW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(description="Scores24.live Prediction Scraper")
    ap.add_argument("--sport", "-s", help="Sport or league name")
    ap.add_argument("--date", "-d", help="Date in YYYY-MM-DD format")
    ap.add_argument("--matchup", "-m", help="Specific matchup string")
    ap.add_argument("--url", "-u", help="Direct URL to a specific prediction page")
    args = ap.parse_args()

    if not args.url and not args.sport:
        print("Error: provide --url OR --sport.")
        sys.exit(1)

    sport_slug = resolve_sport(args.sport) if args.sport else "Unknown"
    sport_label = (args.sport or sport_slug).upper()
    if args.sport and sport_slug not in VALID_SPORTS:
        print(f"❌ '{args.sport}' → '{sport_slug}' is not a valid sport.")
        print(f"   Available: {', '.join(VALID_SPORTS)}")
        sys.exit(1)

    if should_use_olostep():
        try:
            sys.exit(run_with_olostep(args))
        except HTTPError as exc:
            status = getattr(exc, "code", "?")
            body = ""
            try:
                if exc.fp is not None:
                    body = exc.read(300).decode("utf-8", errors="replace").replace("\n", " ")
            except Exception:
                body = ""
            print(f"Olostep scrape failed (HTTP {status}): {body or exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"Olostep scrape failed: {exc}")
            sys.exit(1)

    if Camoufox is None:
        print("Camoufox is not installed. Add it to the environment before running the browser scraper.")
        sys.exit(1)

    with make_browser() as browser:
        ctx = make_context(browser)
        try:
            # ── DIRECT URL OVERRIDE ──
            if args.url and _is_prediction_page_url(args.url):
                print(f"Direct Prediction URL provided: {args.url}")
                page, status = load_page(ctx, args.url, wait_ms=4000)
                if not page:
                    print(f"❌ 404 — page does not exist: {args.url}")
                    return
                
                pred = extract_prediction(page)
                page.close()
                
                if "error" in pred:
                    print(f"⚠️  Loaded but extraction failed: {args.url}")
                    return
                    
                print_prediction(pred, {}, "Unknown", args.url)
                return

            listing_urls = [args.url] if args.url else listing_url_candidates(sport_slug, args.sport)
            listing_url = listing_urls[0]
            requested_league = _requested_league_key(args.sport or sport_slug)
            expected_matchups = fetch_daily_matchups(requested_league, args.date) if requested_league in SPORT_TO_ESPNSLUG else []
            print(f"Sport:          {sport_slug if args.sport else 'Unknown'}")
            if args.date:    print(f"Date requested: {args.date}")
            if args.matchup: print(f"Matchup:        {args.matchup}")
            print(f"Listing URL:    {listing_url}")

            listing_page = None
            status = 0
            used_listing_url = ""
            for cand in listing_urls:
                lp, st = load_page(ctx, cand)
                if lp:
                    listing_page = lp
                    status = st
                    used_listing_url = cand
                    break
                status = st

            if not listing_page:
                if status == 403:
                    print("Listing page status: ❌ Cloudflare blocked this runtime (status 403)")
                    print("Hint: Configure PLAYWRIGHT_PROXY_SERVER (and optional PLAYWRIGHT_PROXY_USERNAME/PLAYWRIGHT_PROXY_PASSWORD) for Render.")
                    pause_for_manual_cloudflare(ctx, used_listing_url or listing_url)
                else:
                    print(f"Listing page status: ❌ Page failed (status {status})")
                return
            if used_listing_url and used_listing_url != listing_url:
                print(f"Listing URL fallback: {used_listing_url}")
            print("Listing page status: ✅ Page loaded")
            hydrate_listing_page(listing_page)

            cards = extract_listing_cards(listing_page)

            filtered = []
            variants = date_variants(args.date) if args.date else []
            matchup_parts = [p.strip().lower() for p in args.matchup.split("vs")] if args.matchup else []

            for c in cards:
                if variants:
                    if not _matches_requested_date(args.date, c, variants): continue
                if matchup_parts:
                    c_data = f"{c.get('home','')} {c.get('away','')} {c.get('href','')}".lower()
                    if not all(p in c_data for p in matchup_parts): continue
                elif expected_matchups and not any(_card_matches_matchup(c, matchup) for matchup in expected_matchups):
                    continue
                filtered.append(c)

            # If card extraction is blocked/empty, fall back to direct prediction links.
            if not filtered:
                direct_links = extract_prediction_links(listing_page)
                for href in direct_links:
                    c = {"href": href, "home": "", "away": "", "isoDate": "", "visDate": "", "visTime": "", "start_time": "", "league": sport_label, "confidence": ""}
                    if variants:
                        if not _href_matches_requested_date(href, args.date, variants):
                            continue
                    if matchup_parts:
                        blob = href.lower()
                        if not all(re.sub(r'[^a-z0-9]+', '-', p) in blob for p in matchup_parts):
                            continue
                    elif expected_matchups and not any(_url_matches_expected_matchup(href, matchup) for matchup in expected_matchups):
                        continue
                    filtered.append(c)

            # ── DEEP SEARCH LOGIC ──
            if not filtered and args.matchup:
                print(f"\n🔍 '{args.matchup}' not found on main listing.")
                print("Initiating Deep Search across sub-leagues. This may take a minute...")
                
                sub_links = extract_subleague_links(listing_page)
                sub_links = list(set(sub_links))[:30] # Limit to 30 leagues to prevent hanging
                
                print(f"Found {len(sub_links)} sub-leagues to check.")

                found_in_deepSearch = False
                for idx, sl_url in enumerate(sub_links, 1):
                    sys.stdout.write(f"\\rScanning sub-league {idx}/{len(sub_links)}...")
                    sys.stdout.flush()
                    
                    sl_page, sl_status = load_page(ctx, sl_url, wait_ms=1000)
                    if not sl_page: continue
                    
                    sl_cards = extract_listing_cards(sl_page)
                    
                    for c in sl_cards:
                        c_data = f"{c.get('home','')} {c.get('away','')} {c.get('href','')}".lower()
                        if all(p in c_data for p in matchup_parts):
                            filtered.append(c)
                            found_in_deepSearch = True
                    
                    sl_page.close()
                    if found_in_deepSearch:
                        print(f"\\n✅ Matchup found in sub-league routing!")
                        break

                if not found_in_deepSearch:
                    print("\\n❌ Deep search scanning complete. Matchup not found on listing pages.")
                    if args.date and "vs" in args.matchup.lower():
                        print("🔮 Engaging URL Prediction Engine for hidden match...")
                        guessed = guess_urls(sport_slug, args.date, args.matchup)
                        found_guess = False
                        for gurl in guessed:
                            gpage, gstatus = load_page(ctx, gurl, wait_ms=2000)
                            if gpage and gstatus == 200:
                                gpred = extract_prediction(gpage)
                                if gpred and not "error" in gpred and (gpred.get("tip") or gpred.get("confidence")):
                                    print(f"\\n✅ Prediction Engine Success! Extracted hidden page: {gurl}")
                                    print_prediction(gpred, {}, "Unknown", gurl)
                                    found_guess = True
                                    gpage.close()
                                    return
                            if gpage: gpage.close()
                        if not found_guess:
                            print("❌ Prediction Engine failed. Check your team names/date spelling.")
                    
            else:
                print(f"Matches found for request on listing: {len(filtered)}")

            listing_page.close()

            if not filtered:
                return

            # ── EXTRACT DATA ──
            stats = {"loaded": 0, "404": 0, "no_data": 0, "unverified": 0}
            entries: list[dict] = []

            for card in filtered:
                href = card.get("href", "")
                if not href: continue
                full_url = href if href.startswith("http") else f"{BASE}{href}"

                pred_page, pred_status = load_page(ctx, full_url, wait_ms=4000)
                if not pred_page:
                    print(f"\n❌ 404 — page does not exist: {full_url}")
                    stats["404"] += 1; continue

                pred = extract_prediction(pred_page)
                pred_page.close()

                if "error" in pred:
                    print(f"\n⚠️  Loaded but extraction failed on: {full_url}")
                    stats["no_data"] += 1; continue

                if not pred.get("tip") and not pred.get("primaryOdds") and not pred.get("oddsTable"):
                    print(f"\n⚠️  Loaded but no prediction data found on page: {full_url}")
                    stats["no_data"] += 1; continue

                if expected_matchups and not any(
                    _prediction_matches_matchup(pred, matchup) or _url_matches_expected_matchup(full_url, matchup)
                    for matchup in expected_matchups
                ):
                    continue

                if not _prediction_matches_requested_sport(pred, card, args.sport):
                    continue

                entries.append(_build_scraped_pick(pred, card, full_url))

            _emit_scraped_predictions(entries, requested_league, args.date, sport_label, stats)

            print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("SUMMARY")
            print(f"Total matches extracted:        {len(filtered)}")
            print(f"Individual pages loaded:        {stats['loaded']}")
            print(f"Individual pages 404'd:         {stats['404']}")
            if stats.get("unverified"):
                print(f"Unverified MLB picks skipped:   {stats['unverified']}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        finally:
            ctx.close()

if __name__ == "__main__":
    main()
