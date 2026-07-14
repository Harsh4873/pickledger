"""
wnba_teams.py — Fetch, cache, and merge WNBA team data from BallDontLie + ESPN APIs.
Builds WNBA_TEAM_MAP keyed by team abbreviation (15 teams for the 2026 season).
"""

import json
import os
import sys

import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------------------------
# API key: try config.py first, then fall back to environment variable
# ---------------------------------------------------------------------------
try:
    from config import BDL_API_KEY
except ImportError:
    BDL_API_KEY = None

if not BDL_API_KEY:
    BDL_API_KEY = os.getenv("BDL_API_KEY", "")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BDL_TEAMS_URL = "https://api.balldontlie.io/wnba/v1/teams"
ESPN_TEAMS_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams"

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wnba")
)
os.makedirs(DATA_DIR, exist_ok=True)
BDL_CACHE_PATH = os.path.join(DATA_DIR, "wnba_teams.json")
ESPN_CACHE_PATH = os.path.join(DATA_DIR, "wnba_teams_espn.json")

# Canonical list of 15 WNBA teams for the 2026 season
ALL_TEAM_ABBRS = [
    "ATL", "CHI", "CON", "DAL", "GSV", "IND",
    "LA", "LV", "MIN", "NY", "PHX", "POR",
    "SEA", "TOR", "WAS",
]

# 2026 expansion teams that may not appear in API responses yet
EXPANSION_TEAMS = {
    "POR": {
        "full_name": "Portland Fire",
        "city": "Portland",
        "conference": "West",
    },
    "TOR": {
        "full_name": "Toronto Tempo",
        "city": "Toronto",
        "conference": "East",
    },
}

# ---------------------------------------------------------------------------
# Helpers: abbreviation normalisation
# ---------------------------------------------------------------------------
# BDL and ESPN sometimes use slightly different abbreviations.
# Map known alternate abbreviations to the canonical ones we use.
_ABBR_ALIASES = {
    "LVA": "LV",
    "CONN": "CON",
    "NYL": "NY",
    "WSH": "WAS",
    "PHO": "PHX",
    "GS": "GSV",
}


def _normalize_abbr(raw: str) -> str:
    """Return canonical abbreviation for a raw API abbreviation."""
    raw = raw.strip().upper()
    return _ABBR_ALIASES.get(raw, raw)


# ---------------------------------------------------------------------------
# Fetch functions
# ---------------------------------------------------------------------------

def _fetch_bdl_teams() -> dict | None:
    """Fetch team list from BallDontLie WNBA API. Returns raw JSON dict or None."""
    if not BDL_API_KEY:
        print("[WNBA] WARNING: BDL_API_KEY is not set — skipping BallDontLie fetch.")
        return None
    try:
        resp = requests.get(
            BDL_TEAMS_URL,
            headers={"Authorization": BDL_API_KEY},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"[WNBA] ERROR fetching BallDontLie teams: {exc}")
        return None


def _fetch_espn_teams() -> dict | None:
    """Fetch team list from ESPN public API. Returns raw JSON dict or None."""
    try:
        resp = requests.get(ESPN_TEAMS_URL, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        print(f"[WNBA] ERROR fetching ESPN teams: {exc}")
        return None


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Build merged team map
# ---------------------------------------------------------------------------

def _build_team_map(bdl_data: dict | None, espn_data: dict | None) -> dict:
    """
    Merge BDL + ESPN payloads into a dict keyed by canonical abbreviation.
    Returns WNBA_TEAM_MAP with entries for all 15 teams.
    """
    # ---- Index BDL teams by canonical abbreviation ----
    bdl_by_abbr: dict[str, dict] = {}
    if bdl_data:
        teams_list = bdl_data.get("data", bdl_data.get("teams", []))
        if isinstance(bdl_data, list):
            teams_list = bdl_data
        for team in teams_list:
            abbr = _normalize_abbr(team.get("abbreviation", ""))
            bdl_by_abbr[abbr] = team

    # ---- Index ESPN teams by canonical abbreviation ----
    espn_by_abbr: dict[str, dict] = {}
    if espn_data:
        # ESPN nests: sports[0].leagues[0].teams[*].team
        try:
            espn_teams = espn_data["sports"][0]["leagues"][0]["teams"]
        except (KeyError, IndexError):
            espn_teams = []
        for entry in espn_teams:
            team = entry.get("team", entry)
            abbr = _normalize_abbr(team.get("abbreviation", ""))
            espn_by_abbr[abbr] = team

    # ---- Merge into canonical map ----
    team_map: dict[str, dict] = {}

    # Start with every abbreviation present in either API
    all_abbrs = set(ALL_TEAM_ABBRS) | set(bdl_by_abbr.keys()) | set(espn_by_abbr.keys())

    for abbr in sorted(all_abbrs):
        bdl_team = bdl_by_abbr.get(abbr)
        espn_team = espn_by_abbr.get(abbr)

        # Determine fields, preferring BDL for name/city, ESPN for ESPN-id
        bdl_id = int(bdl_team["id"]) if bdl_team and "id" in bdl_team else None
        espn_id = str(espn_team["id"]) if espn_team and "id" in espn_team else None

        full_name = (
            (bdl_team.get("full_name") if bdl_team else None)
            or (espn_team.get("displayName") if espn_team else None)
            or EXPANSION_TEAMS.get(abbr, {}).get("full_name", abbr)
        )
        city = (
            (bdl_team.get("city") if bdl_team else None)
            or (espn_team.get("location") if espn_team else None)
            or EXPANSION_TEAMS.get(abbr, {}).get("city", "")
        )
        conference = (
            (bdl_team.get("conference") if bdl_team else None)
            or EXPANSION_TEAMS.get(abbr, {}).get("conference", "")
        )
        # ESPN sometimes puts conference in a different spot
        if not conference and espn_team:
            # Try to find from groups or other nested structure
            groups = espn_team.get("groups", {})
            if isinstance(groups, dict) and groups.get("isConference"):
                conference = groups.get("name", "")

        team_map[abbr] = {
            "bdl_id": bdl_id,
            "espn_id": espn_id,
            "full_name": full_name,
            "city": city,
            "conference": conference,
        }

    # ---- Ensure expansion teams are present (hardcode if missing) ----
    for abbr, defaults in EXPANSION_TEAMS.items():
        if abbr not in team_map:
            team_map[abbr] = {
                "bdl_id": None,
                "espn_id": None,
                **defaults,
            }

    # ---- Keep only the canonical 15 (drop any unexpected extras) ----
    final_map = {a: team_map[a] for a in ALL_TEAM_ABBRS if a in team_map}

    return final_map


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refresh_teams() -> dict:
    """Re-fetch both APIs, rebuild WNBA_TEAM_MAP, overwrite cache files."""
    global WNBA_TEAM_MAP  # noqa: PLW0603

    bdl_data = _fetch_bdl_teams()
    espn_data = _fetch_espn_teams()

    # Cache raw responses (only if fetch succeeded)
    if bdl_data is not None:
        _save_json(BDL_CACHE_PATH, bdl_data)
    if espn_data is not None:
        _save_json(ESPN_CACHE_PATH, espn_data)

    WNBA_TEAM_MAP = _build_team_map(bdl_data, espn_data)

    print(f"[WNBA] Teams refreshed: {len(WNBA_TEAM_MAP)} teams loaded.")
    return WNBA_TEAM_MAP


def get_team_by_abbr(abbr: str) -> dict:
    """
    Return team dict for the given abbreviation.
    Raises KeyError with a helpful message if not found.
    """
    abbr = abbr.strip().upper()
    abbr = _ABBR_ALIASES.get(abbr, abbr)
    if abbr not in WNBA_TEAM_MAP:
        available = ", ".join(sorted(WNBA_TEAM_MAP.keys()))
        raise KeyError(
            f"Team abbreviation '{abbr}' not found in WNBA_TEAM_MAP. "
            f"Available: {available}"
        )
    return WNBA_TEAM_MAP[abbr]


# ---------------------------------------------------------------------------
# Initialise map from cache (or empty) on import
# ---------------------------------------------------------------------------
_cached_bdl = _load_json(BDL_CACHE_PATH)
_cached_espn = _load_json(ESPN_CACHE_PATH)
WNBA_TEAM_MAP: dict[str, dict] = _build_team_map(_cached_bdl, _cached_espn)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    refresh_teams()

    print("\nWNBA_TEAM_MAP keys:")
    for abbr in sorted(WNBA_TEAM_MAP.keys()):
        entry = WNBA_TEAM_MAP[abbr]
        print(f"  {abbr}: {entry['full_name']} (BDL={entry['bdl_id']}, ESPN={entry['espn_id']})")

    assert len(WNBA_TEAM_MAP) == 15, (
        f"Expected 15 teams, got {len(WNBA_TEAM_MAP)}: "
        f"{sorted(WNBA_TEAM_MAP.keys())}"
    )
    print("\nPASS: All 15 teams loaded.")
