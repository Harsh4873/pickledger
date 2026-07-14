"""
wnba_schedule.py — Fetch, merge, and expose WNBA game schedules from ESPN + BallDontLie.
"""

from __future__ import annotations

import datetime
import os
import sys
from dataclasses import dataclass

import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from config import BDL_API_KEY
except ImportError:
    BDL_API_KEY = None
if not BDL_API_KEY:
    BDL_API_KEY = os.getenv("BDL_API_KEY", "")
try:
    from .wnba_teams import WNBA_TEAM_MAP, get_team_by_abbr
except ImportError:
    from wnba_teams import WNBA_TEAM_MAP, get_team_by_abbr

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class WNBAGame:
    bdl_game_id: int | None
    espn_game_id: str | None
    home_abbr: str
    away_abbr: str
    date_str: str       # "YYYY-MM-DD"
    start_time: str     # "HH:MM ET" or "TBD"
    status: str         # "scheduled" | "in_progress" | "final"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESPN_SCOREBOARD_URL = (
    "http://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
)
BDL_GAMES_URL = "https://api.balldontlie.io/wnba/v1/games"

# ESPN status.type.name → our canonical status string
_ESPN_STATUS_MAP = {
    "STATUS_SCHEDULED": "scheduled",
    "STATUS_IN_PROGRESS": "in_progress",
    "STATUS_FINAL": "final",
}

# Known alternate abbreviations ESPN/BDL may return.
# Mirrors _ABBR_ALIASES in wnba_teams.py — kept local so this module needs
# only the public imports (WNBA_TEAM_MAP, get_team_by_abbr).
_ABBR_ALIASES: dict[str, str] = {
    "LVA":  "LV",
    "CONN": "CON",
    "NYL":  "NY",
    "WSH":  "WAS",
    "PHO":  "PHX",
    "GS":   "GSV",
}


def _normalize_abbr(raw: str) -> str:
    """Return the canonical team abbreviation for a raw API value."""
    raw = raw.strip().upper()
    canonical = _ABBR_ALIASES.get(raw, raw)
    # Final guard: only return it if it's actually in our team map
    if canonical in WNBA_TEAM_MAP:
        return canonical
    return raw


# ---------------------------------------------------------------------------
# ESPN fetcher
# ---------------------------------------------------------------------------

def _parse_espn_time(iso_date: str) -> str:
    """Convert an ISO-8601 UTC timestamp to 'HH:MM ET', or 'TBD'."""
    try:
        # ESPN dates look like "2026-06-01T00:00Z" or "2026-06-01T23:30:00Z"
        iso_date = iso_date.replace("Z", "+00:00")
        utc_dt = datetime.datetime.fromisoformat(iso_date)
        # ET = UTC-5 (EST) or UTC-4 (EDT).  Use a fixed -4 offset during the
        # WNBA season (May–Oct) which falls entirely within US Eastern Daylight.
        et_offset = datetime.timezone(datetime.timedelta(hours=-4))
        et_dt = utc_dt.astimezone(et_offset)
        # Midnight UTC → 8:00 PM ET previous day; ESPN sometimes sends midnight
        # as a placeholder for TBD.  Treat 00:00 UTC literally (20:00 ET) —
        # callers can refine later if needed.
        return et_dt.strftime("%H:%M") + " ET"
    except (ValueError, TypeError):
        return "TBD"


def fetch_espn_schedule(date_str: str) -> list[WNBAGame]:
    """
    Fetch WNBA games for *date_str* (YYYY-MM-DD) from the ESPN scoreboard API.
    Returns a list of WNBAGame with bdl_game_id=None.
    On any HTTP error or empty/malformed response, returns [].
    """
    try:
        yyyymmdd = date_str.replace("-", "")
        resp = requests.get(
            ESPN_SCOREBOARD_URL,
            params={"dates": yyyymmdd},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[WNBA] ESPN schedule fetch error for {date_str}: {exc}")
        return []

    events = data.get("events", [])
    if not events:
        return []

    games: list[WNBAGame] = []
    for event in events:
        try:
            espn_id = str(event["id"])
            start_time = _parse_espn_time(event.get("date", ""))

            # Status
            raw_status = (
                event.get("status", {})
                     .get("type", {})
                     .get("name", "STATUS_SCHEDULED")
            )
            status = _ESPN_STATUS_MAP.get(raw_status, "scheduled")

            # Competitors
            competitors = event.get("competitions", [{}])[0].get("competitors", [])
            home_abbr = ""
            away_abbr = ""
            for comp in competitors:
                abbr = _normalize_abbr(comp.get("team", {}).get("abbreviation", ""))
                if comp.get("homeAway") == "home":
                    home_abbr = abbr
                else:
                    away_abbr = abbr

            if not home_abbr or not away_abbr:
                continue

            games.append(
                WNBAGame(
                    bdl_game_id=None,
                    espn_game_id=espn_id,
                    home_abbr=home_abbr,
                    away_abbr=away_abbr,
                    date_str=date_str,
                    start_time=start_time,
                    status=status,
                )
            )
        except (KeyError, IndexError, TypeError):
            # Skip malformed events silently
            continue

    return games


# ---------------------------------------------------------------------------
# BallDontLie fetcher
# ---------------------------------------------------------------------------

def _parse_bdl_status(raw: str) -> str:
    """Map BDL status strings to our canonical values."""
    raw_lower = raw.strip().lower() if raw else ""
    if raw_lower in ("final", "completed"):
        return "final"
    if raw_lower in ("in progress", "in_progress", "live"):
        return "in_progress"
    return "scheduled"


def fetch_bdl_schedule(start_date: str, end_date: str) -> list[WNBAGame]:
    """
    Fetch WNBA games from BallDontLie for the date range [start_date, end_date].
    Both parameters are 'YYYY-MM-DD'.  Paginates through all pages.
    Returns [] on missing key, HTTP error, or empty response.
    """
    if not BDL_API_KEY:
        return []

    games: list[WNBAGame] = []
    cursor: str | None = None

    while True:
        try:
            params: dict[str, str | int] = {
                "start_date": start_date,
                "end_date": end_date,
                "per_page": 100,
            }
            if cursor is not None:
                params["cursor"] = cursor

            resp = requests.get(
                BDL_GAMES_URL,
                headers={"Authorization": BDL_API_KEY},
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"[WNBA] BDL schedule fetch error: {exc}")
            return []

        for game in data.get("data", []):
            try:
                bdl_id = int(game["id"])
                # BDL date may be "YYYY-MM-DD" or a full ISO timestamp
                game_date = str(game.get("date", ""))[:10]
                home_abbr = _normalize_abbr(
                    game.get("home_team", {}).get("abbreviation", "")
                )
                away_abbr = _normalize_abbr(
                    game.get("visitor_team", {}).get("abbreviation", "")
                )
                status = _parse_bdl_status(game.get("status", ""))

                if not home_abbr or not away_abbr:
                    continue

                games.append(
                    WNBAGame(
                        bdl_game_id=bdl_id,
                        espn_game_id=None,
                        home_abbr=home_abbr,
                        away_abbr=away_abbr,
                        date_str=game_date,
                        start_time="TBD",  # BDL doesn't reliably provide tip-off times
                        status=status,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        # Pagination
        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = str(next_cursor)

    return games


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_schedules(
    espn_games: list[WNBAGame],
    bdl_games: list[WNBAGame],
) -> list[WNBAGame]:
    """
    Start from the ESPN list (more reliable for schedule data).
    For each ESPN game, find a matching BDL game by (home_abbr, away_abbr, date_str)
    and fill in bdl_game_id when found.  Returns the merged list.
    """
    # Build a lookup: (home, away, date) → bdl_game_id
    bdl_lookup: dict[tuple[str, str, str], int] = {}
    for g in bdl_games:
        key = (g.home_abbr, g.away_abbr, g.date_str)
        if g.bdl_game_id is not None:
            bdl_lookup[key] = g.bdl_game_id

    merged: list[WNBAGame] = []
    for g in espn_games:
        key = (g.home_abbr, g.away_abbr, g.date_str)
        if key in bdl_lookup:
            g.bdl_game_id = bdl_lookup[key]
        merged.append(g)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_todays_wnba_games(date_str: str | None = None) -> list[WNBAGame]:
    """Fetch and merge WNBA games for *date_str* or today from ESPN + BDL."""
    today = date_str or datetime.date.today().isoformat()  # "YYYY-MM-DD"

    espn_games = fetch_espn_schedule(today)
    bdl_games = fetch_bdl_schedule(today, today)
    result = merge_schedules(espn_games, bdl_games)

    if not result:
        print("[WNBA] No games today.")
    return result


def get_upcoming_wnba_games(days_ahead: int = 7) -> list[WNBAGame]:
    """Fetch and merge WNBA games from today through today + *days_ahead*."""
    today = datetime.date.today()
    end = today + datetime.timedelta(days=days_ahead)

    start_str = today.isoformat()
    end_str = end.isoformat()

    # ESPN scoreboard only supports a single date, so iterate each day
    espn_games: list[WNBAGame] = []
    current = today
    while current <= end:
        espn_games.extend(fetch_espn_schedule(current.isoformat()))
        current += datetime.timedelta(days=1)

    bdl_games = fetch_bdl_schedule(start_str, end_str)
    result = merge_schedules(espn_games, bdl_games)

    if not result:
        print("[WNBA] No games today.")
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Today's games
    todays = get_todays_wnba_games()

    if not todays:
        # Check upcoming week to decide if it's truly off-season
        upcoming = get_upcoming_wnba_games(7)
        if not upcoming:
            print("[WNBA] Off-season \u2014 season opens May 8, 2026.")
            sys.exit(0)
    else:
        for g in todays:
            print(
                f"[WNBA] {g.away_abbr} @ {g.home_abbr} "
                f"| {g.date_str} {g.start_time} "
                f"| Status: {g.status}"
            )

    # Upcoming 7 days
    upcoming = get_upcoming_wnba_games(7)
    print(f"Upcoming 7 days: {len(upcoming)} games found.")
