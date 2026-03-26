#!/usr/bin/env python3
"""
Scores24.live Prediction Scraper (Olostep-backed)
=================================================
Fetches prediction data from scores24.live through the Olostep API.
If a specific matchup isn't on the main index, automatically
scans sub-league directories and follows more listing pages to hunt it down.
"""

import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from html import unescape
from urllib.error import HTTPError
from urllib.parse import urljoin
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
    from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout
except Exception:
    sync_playwright = None

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
    "nba": [f"{BASE}/en/basketball/l-usa-nba"],
    "nhl": [f"{BASE}/en/ice-hockey/l-usa-nhl"],
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
    return True


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
    if re.search(r"/m-\d{2}-\d{2}-\d{4}-", full_url) and not full_url.endswith("-prediction"):
        return f"{full_url}-prediction"
    return full_url


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


def _extract_subleague_urls(scrape_obj: dict) -> list[str]:
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
    if args.url and not args.url.endswith("/predictions") and not args.url.endswith("/predictions/"):
        direct_url = _normalize_prediction_page_url(args.url)
        pred = _scrape_prediction_with_retry(direct_url, "Unknown")
        if not pred.get("tip"):
            print(f"⚠️  Loaded but extraction failed: {direct_url}")
            return 1
        print_prediction(pred, {"league": pred.get("league", "")}, "Unknown", direct_url)
        return 0

    sport_slug = resolve_sport(args.sport) if args.sport else "Unknown"
    sport_label = (args.sport or sport_slug).upper()
    listing_urls = [args.url] if args.url else listing_url_candidates(sport_slug, args.sport)
    listing_url = listing_urls[0]
    matchup_parts = [p.strip().lower() for p in args.matchup.split("vs") if p.strip()] if args.matchup else []
    expected_matchups = fetch_daily_matchups(sport_slug, args.date) if sport_slug in SPORT_TO_ESPNSLUG else []
    strict_date = sport_slug not in SPORT_TO_ESPNSLUG
    print(f"Sport:          {sport_slug if args.sport else 'Unknown'}")
    if args.date:
        print(f"Date requested: {args.date}")
    if args.matchup:
        print(f"Matchup:        {args.matchup}")
    print(f"Listing URL:    {listing_url}")
    print("Backend:        olostep")

    filtered: list[dict] = []
    aggregate_links: list[dict] = []
    used_listing_url = ""
    for cand in listing_urls:
        scrape = _olostep_scrape(cand, ["markdown"])
        normalized_links = _dedupe_prediction_links(_extract_prediction_urls(scrape, args.date, strict_date=strict_date))
        if matchup_parts and normalized_links:
            normalized_links = [href for href in normalized_links if _url_matches_matchup(href, matchup_parts)]

        # Only fan out into sub-leagues when the main listing did not already
        # expose the direct prediction links we need.
        if not normalized_links:
            aggregated: list[str] = []
            subleague_urls = _extract_subleague_urls(scrape)[:20]
            for subleague_url in subleague_urls:
                try:
                    sub_scrape = _olostep_scrape(subleague_url, ["markdown"])
                    sub_links = _dedupe_prediction_links(_extract_prediction_urls(sub_scrape, args.date, strict_date=strict_date))
                    if matchup_parts:
                        sub_links = [href for href in sub_links if _url_matches_matchup(href, matchup_parts)]
                    aggregated.extend(sub_links)
                    if matchup_parts and sub_links:
                        break
                except Exception:
                    continue
            normalized_links = _dedupe_prediction_links(aggregated)
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
                "league": sport_label,
                "confidence": "",
            })
    filtered = _dedupe_cards(aggregate_links)

    if used_listing_url and used_listing_url != listing_url:
        print(f"Listing URL fallback: {used_listing_url}")
    if not filtered:
        print("Listing page status: ❌ No prediction links found via Olostep")
        return 1
    print("Listing page status: ✅ Page loaded")

    if args.matchup:
        filtered = [
            card for card in filtered
            if _url_matches_matchup(card["href"], matchup_parts)
        ]
        print(f"Matches found for request on listing: {len(filtered)}")

    if not filtered:
        return 0

    stats = {"loaded": 0, "404": 0, "no_data": 0}
    for card in filtered:
        full_url = card["href"]
        pred = _scrape_prediction_with_retry(full_url, sport_label)
        if not pred.get("tip"):
            print(f"\n⚠️  Loaded but no prediction data found on page: {full_url}")
            stats["no_data"] += 1
            continue
        if expected_matchups and not any(_prediction_matches_matchup(pred, matchup) for matchup in expected_matchups):
            continue
        stats["loaded"] += 1
        print_prediction(pred, {"league": pred.get("league", sport_label)}, sport_label, full_url)

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("SUMMARY")
    print(f"Total matches extracted:        {len(filtered)}")
    print(f"Individual pages loaded:        {stats['loaded']}")
    print(f"Individual pages 404'd:         {stats['404']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BROWSER HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_context(pw):
    proxy_conf = None
    launch_args = {
        "headless": True,
        "args": [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
        ],
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

    browser = pw.chromium.launch(**launch_args)
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        timezone_id="America/New_York",
    )
    # Reduce obvious automation fingerprints; does not bypass hard blocks by itself.
    ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
    return ctx


def load_page(ctx, url: str, wait_ms: int = 3000):
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

        const dateSpans = card.querySelectorAll('span');
        let visDate = '';
        let visTime = '';
        for (const sp of dateSpans) {
            const t = sp.textContent.trim();
            if (/^\\d{1,2}\\s+[A-Za-z]{3}/.test(t)) visDate = t;
            if (/^\\d{2}:\\d{2}$/.test(t)) visTime = t;
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

        return { href, home, away, isoDate, visDate, visTime, league, confidence };
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT FORMATTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fmt(label: str, value: str, fallback: str = "[not found on page]") -> str:
    return f"{label:<15}{value if value else fallback}"


def print_prediction(pred: dict, card: dict, sport_label: str, url: str):
    home = pred.get("homeTeam") or card.get("home", "")
    away = pred.get("awayTeam") or card.get("away", "")
    match_str = f"{home} vs {away}" if home and away else pred.get("matchTitle", "[unknown]")

    date_str = pred.get("date") or card.get("visDate", "")
    if card.get("visTime"):
        date_str = f"{date_str}, {card['visTime']}"

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

    with sync_playwright() as pw:
        ctx = make_context(pw)

        # ── DIRECT URL OVERRIDE ──
        if args.url and not args.url.endswith("/predictions") and not args.url.endswith("/predictions/"):
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
            filtered.append(c)

        # If card extraction is blocked/empty, fall back to direct prediction links.
        if not filtered:
            direct_links = extract_prediction_links(listing_page)
            for href in direct_links:
                c = {"href": href, "home": "", "away": "", "isoDate": "", "visDate": "", "visTime": "", "league": sport_label, "confidence": ""}
                if variants:
                    blob = href.lower()
                    if not any(v.lower().replace(" ", "-") in blob for v in variants):
                        continue
                if matchup_parts:
                    blob = href.lower()
                    if not all(re.sub(r'[^a-z0-9]+', '-', p) in blob for p in matchup_parts):
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
        stats = {"loaded": 0, "404": 0, "no_data": 0}

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

            stats["loaded"] += 1
            print_prediction(pred, card, sport_label, full_url)

        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("SUMMARY")
        print(f"Total matches extracted:        {len(filtered)}")
        print(f"Individual pages loaded:        {stats['loaded']}")
        print(f"Individual pages 404'd:         {stats['404']}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == "__main__":
    main()
