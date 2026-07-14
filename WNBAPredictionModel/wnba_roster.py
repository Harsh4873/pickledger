"""
wnba_roster.py — Fetch, merge, cache, and expose WNBA team rosters from
ESPN + BallDontLie.

Public API:
    get_all_rosters()          → dict[team_abbr, list[player_dict]]
    get_roster(team_abbr)      → list[player_dict]
    refresh_rosters()          → force re-fetch + cache
    fetch_espn_roster(id)      → raw ESPN roster for one team
    fetch_bdl_active_players() → flat list of BDL active players

Cache: data/wnba_rosters.json (24 h TTL).
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import BDL_API_KEY
try:
    from .wnba_teams import WNBA_TEAM_MAP
except ImportError:
    from wnba_teams import WNBA_TEAM_MAP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ESPN_ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/"
    "{espn_team_id}/roster"
)
BDL_ACTIVE_PLAYERS_URL = "https://api.balldontlie.io/wnba/v1/players/active"

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wnba")
)
os.makedirs(DATA_DIR, exist_ok=True)
ROSTER_CACHE_PATH = os.path.join(DATA_DIR, "wnba_rosters.json")

CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours

# BDL / ESPN abbreviation aliases — mirrors wnba_teams.py, kept local so this
# module only needs the public WNBA_TEAM_MAP import.
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
    if not raw:
        return ""
    raw = raw.strip().upper()
    canonical = _ABBR_ALIASES.get(raw, raw)
    if canonical in WNBA_TEAM_MAP:
        return canonical
    return raw


def _normalize_name(name: str) -> str:
    """Lowercase + strip whitespace for name matching."""
    return (name or "").strip().lower()


# ---------------------------------------------------------------------------
# ESPN fetcher
# ---------------------------------------------------------------------------

def fetch_espn_roster(espn_team_id: str) -> list[dict]:
    """
    Fetch the ESPN roster for a given team ID.

    Returns a list of player dicts with keys:
        espn_id, name, position, jersey, status

    On any HTTP error or missing field we skip the player / return []
    rather than raise.
    """
    if not espn_team_id:
        return []

    url = ESPN_ROSTER_URL.format(espn_team_id=espn_team_id)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[WNBA] ESPN roster fetch error for team {espn_team_id}: {exc}")
        return []

    # ESPN usually returns a flat `athletes` array for WNBA rosters, but
    # some endpoints group by position.  Handle both shapes.
    raw_athletes = data.get("athletes", []) or []
    athletes: list[dict] = []
    if raw_athletes and isinstance(raw_athletes[0], dict) and "items" in raw_athletes[0]:
        # Grouped form: [{ "position": "...", "items": [athlete, ...] }, ...]
        for group in raw_athletes:
            athletes.extend(group.get("items", []) or [])
    else:
        athletes = raw_athletes

    players: list[dict] = []
    for athlete in athletes:
        try:
            # `id` is required — skip silently otherwise
            if "id" not in athlete:
                continue

            espn_id = str(athlete["id"])
            full_name = athlete.get("fullName") or athlete.get("displayName") or ""
            display_name = athlete.get("displayName") or full_name
            jersey = str(athlete.get("jersey", "")) if athlete.get("jersey") is not None else ""

            position = ""
            pos_field = athlete.get("position")
            if isinstance(pos_field, dict):
                position = pos_field.get("abbreviation", "") or ""

            status = ""
            status_field = athlete.get("status")
            if isinstance(status_field, dict):
                status = status_field.get("type", "") or ""

            if not display_name:
                # No name at all — skip
                continue

            players.append(
                {
                    "espn_id": espn_id,
                    "name": display_name,
                    "position": position,
                    "jersey": jersey,
                    "status": status,
                }
            )
        except (KeyError, TypeError, AttributeError):
            # Skip any malformed athlete record silently
            continue

    return players


# ---------------------------------------------------------------------------
# BallDontLie fetcher
# ---------------------------------------------------------------------------

def fetch_bdl_active_players() -> list[dict]:
    """
    Fetch all active WNBA players from BallDontLie.

    Returns a flat list of dicts with keys:
        bdl_id, name, position, team_abbr

    On missing key: prints a warning and returns [].
    On HTTP error: returns [].
    """
    if not BDL_API_KEY:
        print("[WNBA] WARNING: BDL_API_KEY is not set — skipping BDL active players fetch.")
        return []

    players: list[dict] = []
    cursor: str | None = None

    while True:
        try:
            params: dict[str, str | int] = {"per_page": 100}
            if cursor is not None:
                params["cursor"] = cursor

            resp = requests.get(
                BDL_ACTIVE_PLAYERS_URL,
                headers={"Authorization": BDL_API_KEY},
                params=params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as exc:
            print(f"[WNBA] BDL active players fetch error: {exc}")
            return []

        for player in data.get("data", []) or []:
            try:
                bdl_id = int(player["id"])
                first = (player.get("first_name") or "").strip()
                last = (player.get("last_name") or "").strip()
                full_name = f"{first} {last}".strip()
                if not full_name:
                    continue

                position = player.get("position") or ""

                team_info = player.get("team") or {}
                team_abbr = _normalize_abbr(team_info.get("abbreviation", ""))

                players.append(
                    {
                        "bdl_id": bdl_id,
                        "name": full_name,
                        "position": position,
                        "team_abbr": team_abbr,
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue

        # Pagination
        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            break
        cursor = str(next_cursor)

    return players


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def build_rosters(espn_all: dict, bdl_all: list) -> dict:
    """
    Merge ESPN rosters (primary source, grouped by team) with the flat BDL
    active-player list.

    For each ESPN player we attempt to find a BDL player with the same
    normalized name.  When a match is found we add `bdl_id` to the player
    dict.

    Parameters
    ----------
    espn_all : {team_abbr: [player_dict, ...]}
    bdl_all  : flat list of BDL player dicts

    Returns
    -------
    {team_abbr: [merged_player_dict, ...]}
    """
    # Build a lookup: normalized_name → bdl_id.  A name could in theory be
    # duplicated across the league; if so the last wins — good enough for
    # the soft merge the task requires.
    bdl_by_name: dict[str, int] = {}
    for bdl_player in bdl_all or []:
        key = _normalize_name(bdl_player.get("name", ""))
        if not key:
            continue
        bdl_id = bdl_player.get("bdl_id")
        if bdl_id is None:
            continue
        bdl_by_name[key] = bdl_id

    merged: dict[str, list[dict]] = {}
    for team_abbr, players in (espn_all or {}).items():
        team_players: list[dict] = []
        for player in players or []:
            # Copy so we never mutate the caller's dicts
            merged_player = dict(player)
            name_key = _normalize_name(merged_player.get("name", ""))
            if name_key and name_key in bdl_by_name:
                merged_player["bdl_id"] = bdl_by_name[name_key]
            team_players.append(merged_player)
        merged[team_abbr] = team_players

    return merged


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def _load_cache() -> dict | None:
    """Return the cached payload dict if it exists and is fresh (< 24h)."""
    if not os.path.exists(ROSTER_CACHE_PATH):
        return None

    try:
        with open(ROSTER_CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[WNBA] Could not read roster cache: {exc}")
        return None

    last_updated = payload.get("last_updated")
    if not last_updated:
        return None

    try:
        updated_dt = datetime.datetime.fromisoformat(last_updated)
    except ValueError:
        return None

    # Treat naïve timestamps as UTC for comparison
    now = datetime.datetime.now(updated_dt.tzinfo) if updated_dt.tzinfo else datetime.datetime.now()
    age = (now - updated_dt).total_seconds()
    if age < 0 or age > CACHE_TTL_SECONDS:
        return None

    return payload


def _save_cache(rosters: dict) -> None:
    """Persist the roster dict with a `last_updated` ISO timestamp."""
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "last_updated": datetime.datetime.now().isoformat(),
        "rosters": rosters,
    }
    try:
        with open(ROSTER_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"[WNBA] Could not write roster cache: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_all_rosters() -> dict:
    """
    Return the full {team_abbr: [player_dict, ...]} map.

    Loads from the 24-hour cache when available; otherwise fetches ESPN
    rosters for every team, fetches BDL active players once, merges them,
    and writes the result to disk.
    """
    cached = _load_cache()
    if cached is not None:
        rosters = cached.get("rosters")
        if isinstance(rosters, dict):
            return rosters

    espn_all: dict[str, list[dict]] = {}
    for team_abbr, team_info in WNBA_TEAM_MAP.items():
        espn_id = team_info.get("espn_id")
        if not espn_id:
            # Expansion / unknown team without an ESPN id — record empty
            espn_all[team_abbr] = []
            continue

        try:
            espn_all[team_abbr] = fetch_espn_roster(str(espn_id))
        except Exception as exc:  # noqa: BLE001 — one team must not kill the batch
            print(f"[WNBA] Unexpected error fetching roster for {team_abbr}: {exc}")
            espn_all[team_abbr] = []

        # Be polite to ESPN between requests
        time.sleep(0.3)

    bdl_all = fetch_bdl_active_players()
    rosters = build_rosters(espn_all, bdl_all)

    _save_cache(rosters)
    return rosters


def get_roster(team_abbr: str) -> list[dict]:
    """Return the merged roster list for *team_abbr*, or [] if not found."""
    if not team_abbr:
        return []
    abbr = _normalize_abbr(team_abbr)
    rosters = get_all_rosters()
    return rosters.get(abbr, [])


def refresh_rosters() -> dict:
    """Delete the cache file (if any) and force a fresh fetch."""
    if os.path.exists(ROSTER_CACHE_PATH):
        try:
            os.remove(ROSTER_CACHE_PATH)
        except OSError as exc:
            print(f"[WNBA] Could not delete roster cache: {exc}")

    rosters = get_all_rosters()
    print(f"[WNBA] Rosters refreshed for all {len(rosters)} teams.")
    return rosters


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rosters = refresh_rosters()

    ind_roster = get_roster("IND")
    print(f"\nIND roster ({len(ind_roster)} players):")
    for player in ind_roster:
        jersey = player.get("jersey") or "--"
        name = player.get("name", "")
        position = player.get("position") or "?"
        print(f"  {jersey} {name} ({position})")

    # Soft check — Caitlin Clark could be injured/traded/etc.
    has_clark = any(
        _normalize_name(p.get("name", "")) == "caitlin clark"
        for p in ind_roster
    )
    if has_clark:
        print("PASS: Caitlin Clark found on IND roster.")
    else:
        print("WARN: Caitlin Clark not found — check ESPN data.")

    total_players = sum(len(players) for players in rosters.values())
    print(f"\nTotal players across all {len(rosters)} teams: {total_players}")
