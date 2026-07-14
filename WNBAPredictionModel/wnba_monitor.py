import datetime
import json
import os
import time
from typing import Any

from WNBAPredictionModel.wnba_injuries import get_injury_report
from WNBAPredictionModel.wnba_roster import refresh_rosters
from WNBAPredictionModel.wnba_stats import get_all_team_stats
from WNBAPredictionModel.wnba_teams import refresh_teams

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "wnba")
LOG_PATH = os.path.join(DATA_DIR, "wnba_monitor.log")
os.makedirs(DATA_DIR, exist_ok=True)

INJURIES_CACHE = os.path.join(DATA_DIR, "wnba_injuries.json")
ROSTERS_CACHE = os.path.join(DATA_DIR, "wnba_rosters.json")
STATS_CACHE = os.path.join(DATA_DIR, "wnba_team_stats_2026.json")
TEAMS_CACHE = os.path.join(DATA_DIR, "wnba_teams_espn.json")


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as file_obj:
        file_obj.write(line + "\n")


def _age_minutes(path: str) -> float | None:
    if not os.path.exists(path):
        return None
    mtime = os.path.getmtime(path)
    return (time.time() - mtime) / 60.0


def _read_cache_payload(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stats_cache_path(season: int) -> str:
    if season == 2026:
        return STATS_CACHE
    return os.path.join(DATA_DIR, f"wnba_team_stats_{season}.json")


def refresh_injuries(force: bool = False) -> None:
    age = _age_minutes(INJURIES_CACHE)
    if not force and age is not None and age < 45:
        log(f"Injuries fresh ({age:.1f} min old) — skipping.")
        return
    log("Refreshing WNBA injuries...")
    report = get_injury_report(force_refresh=True)
    players = len(report) if isinstance(report, dict) else 0
    log(f"Injuries refreshed ({players} players cached).")


def refresh_rosters_wrapper(force: bool = False) -> None:
    age = _age_minutes(ROSTERS_CACHE)
    if not force and age is not None and age < 24 * 60:
        log(f"Rosters fresh ({age / 60:.1f} hours old) — skipping.")
        return
    log("Refreshing WNBA rosters...")
    rosters = refresh_rosters()
    teams = len(rosters) if isinstance(rosters, dict) else 0
    log(f"Rosters refreshed ({teams} teams cached).")


def refresh_stats(force: bool = False, season: int = 2026) -> None:
    stats_cache = _stats_cache_path(season)
    age = _age_minutes(stats_cache)
    if not force and age is not None and age < 6 * 60:
        log(f"Stats fresh ({age / 60:.1f} hours old) — skipping.")
        return
    log(f"Refreshing WNBA stats for season {season}...")
    profiles = get_all_team_stats(season=season, force_refresh=True)
    teams = len(profiles) if isinstance(profiles, dict) else 0
    payload = _read_cache_payload(stats_cache)
    updated_at = payload.get("last_updated")
    if updated_at:
        log(f"Stats refreshed ({teams} teams cached, last_updated={updated_at}).")
    else:
        log(f"Stats refreshed ({teams} teams cached).")


def refresh_teams_wrapper(force: bool = False) -> None:
    age = _age_minutes(TEAMS_CACHE)
    if not force and age is not None and age < 7 * 24 * 60:
        log(f"Teams fresh ({age / (24 * 60):.1f} days old) — skipping.")
        return
    log("Refreshing WNBA teams...")
    teams = refresh_teams()
    count = len(teams) if isinstance(teams, dict) else 0
    log(f"Teams refreshed ({count} teams cached).")


def refresh_all_wnba_data(force: bool = False, season: int = 2026) -> None:
    log(f"Starting WNBA monitor refresh (force={force}, season={season})")
    try:
        refresh_teams_wrapper(force=force)
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR refreshing teams: {exc}")
    try:
        refresh_rosters_wrapper(force=force)
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR refreshing rosters: {exc}")
    try:
        refresh_injuries(force=force)
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR refreshing injuries: {exc}")
    try:
        refresh_stats(force=force, season=season)
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR refreshing stats: {exc}")
    log("WNBA monitor refresh complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WNBA data freshness monitor")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh all WNBA data, ignoring staleness TTLs",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Season year (default 2026)",
    )
    args = parser.parse_args()

    refresh_all_wnba_data(force=args.force, season=args.season)
