from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

try:
    from mlb_inning_fetcher import (
        API_BASE,
        LIVE_FEED_BASE,
        STATS_TTL_SECONDS,
        api_get_json,
        cache_get,
        cache_set,
        log_warning,
        safe_int,
    )
except ImportError:
    from .mlb_inning_fetcher import (
        API_BASE,
        LIVE_FEED_BASE,
        STATS_TTL_SECONDS,
        api_get_json,
        cache_get,
        cache_set,
        log_warning,
        safe_int,
    )


MLB_AVG_SCORELESS = {
    1: 0.620,
    2: 0.720,
    3: 0.700,
    4: 0.720,
    5: 0.710,
    6: 0.720,
    7: 0.740,
    8: 0.750,
    9: 0.760,
}

HISTORY_TEAM_MAX_WORKERS = 8


def fetch_team_histories(games: list[dict[str, Any]], target_date: str | None = None) -> dict[str, dict[int, dict[str, float]]]:
    histories: dict[str, dict[int, dict[str, float]]] = {}
    team_requests: list[tuple[int, str, str]] = []
    seen_names: set[str] = set()
    for game in games:
        game_date = target_date or str(game.get("game_date") or "")
        for side in ("away", "home"):
            team_id = safe_int(game.get(f"{side}_team_id"))
            team_name = str(game.get(f"{side}_team") or "")
            if not team_id or not team_name or team_name in seen_names:
                continue
            seen_names.add(team_name)
            team_requests.append((team_id, team_name, game_date))

    if not team_requests:
        return histories

    workers = max(1, min(HISTORY_TEAM_MAX_WORKERS, len(team_requests)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_team = {
            executor.submit(fetch_team_history, team_id, team_name, game_date): team_name
            for team_id, team_name, game_date in team_requests
        }
        for future in as_completed(future_to_team):
            team_name = future_to_team[future]
            try:
                histories[team_name] = future.result()
            except Exception as exc:
                log_warning(f"{team_name} history lookup failed: {exc}; using league inning defaults")
                histories[team_name] = _league_default_history()
    return histories


def fetch_team_history(team_id: int, team_name: str, target_date: str) -> dict[int, dict[str, float]]:
    cache_key = f"history_{team_id}_{target_date}"
    cached = cache_get(cache_key, STATS_TTL_SECONDS)
    if cached is not None:
        return _int_keyed_history(cached)

    inning_runs = _collect_recent_inning_runs(team_id, target_date)
    if len(inning_runs) < 15:
        log_warning(f"{team_name} has only {len(inning_runs)} recent games; using league inning defaults")
        history = _league_default_history()
    else:
        history = _summarize_inning_runs(inning_runs[-30:])

    cache_set(cache_key, history)
    return history


def _collect_recent_inning_runs(team_id: int, target_date: str) -> list[dict[int, int]]:
    season = datetime.strptime(target_date, "%Y-%m-%d").year
    records: list[dict[int, int]] = []
    for lookup_season in (season, season - 1):
        if len(records) >= 30:
            break
        schedule = _team_schedule(team_id, lookup_season)
        game_ids = [
            safe_int(game.get("gamePk"))
            for game in _finished_games_before(schedule, target_date)
            if safe_int(game.get("gamePk"))
        ]
        needed = max(0, 30 - len(records))
        with ThreadPoolExecutor(max_workers=8) as executor:
            inning_records = list(
                executor.map(
                    lambda game_id: _team_inning_runs(game_id, team_id),
                    game_ids[:needed],
                )
            )
        for inning_record in inning_records:
            if inning_record:
                records.append(inning_record)
            if len(records) >= 30:
                break
    return records


def _team_schedule(team_id: int, season: int) -> dict[str, Any]:
    try:
        return api_get_json(
            f"{API_BASE}/schedule",
            params={
                "sportId": 1,
                "teamId": team_id,
                "startDate": f"{season}-03-01",
                "endDate": f"{season}-11-30",
            },
            cache_key=f"team_schedule_{team_id}_{season}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}


def _finished_games_before(schedule_payload: dict[str, Any], target_date: str) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for day in schedule_payload.get("dates") or []:
        games.extend(day.get("games") or [])
    cutoff = datetime.strptime(target_date, "%Y-%m-%d").date()
    final_games = []
    for game in games:
        game_date = _parse_game_date(game.get("gameDate") or game.get("officialDate"))
        status = str(((game.get("status") or {}).get("detailedState")) or "")
        if game_date and game_date < cutoff and "final" in status.lower():
            final_games.append(game)
    final_games.sort(key=lambda item: (str(item.get("gameDate") or ""), safe_int(item.get("gamePk"))), reverse=True)
    return final_games


def _team_inning_runs(game_id: int, team_id: int) -> dict[int, int]:
    try:
        feed = api_get_json(
            f"{LIVE_FEED_BASE}/game/{game_id}/feed/live",
            cache_key=f"feed_{game_id}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}

    teams = ((feed.get("gameData") or {}).get("teams") or {})
    side = ""
    if safe_int((teams.get("home") or {}).get("id")) == team_id:
        side = "home"
    elif safe_int((teams.get("away") or {}).get("id")) == team_id:
        side = "away"
    if not side:
        return {}

    innings = (((feed.get("liveData") or {}).get("linescore") or {}).get("innings") or [])
    by_inning = {inning: 0 for inning in range(1, 10)}
    for inning_payload in innings:
        inning_number = safe_int(inning_payload.get("num"))
        if 1 <= inning_number <= 9:
            by_inning[inning_number] = safe_int(((inning_payload.get(side) or {}).get("runs")), 0)
    return by_inning


def _summarize_inning_runs(records: list[dict[int, int]]) -> dict[int, dict[str, float]]:
    history: dict[int, dict[str, float]] = {}
    for inning in range(1, 10):
        values = [safe_int(record.get(inning), 0) for record in records]
        if not values:
            history[inning] = _default_inning_stats(inning)
            continue
        scoreless = sum(1 for value in values if value == 0)
        avg_runs = sum(values) / len(values)
        history[inning] = {
            "scoreless_rate": round(scoreless / len(values), 4),
            "avg_runs": round(avg_runs, 4),
            "std_runs": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            "sample_games": float(len(values)),
        }
    return history


def _league_default_history() -> dict[int, dict[str, float]]:
    return {inning: _default_inning_stats(inning) for inning in range(1, 10)}


def _default_inning_stats(inning: int) -> dict[str, float]:
    scoreless_rate = MLB_AVG_SCORELESS[inning]
    return {
        "scoreless_rate": scoreless_rate,
        "avg_runs": round(max(0.0, 1.0 - scoreless_rate), 4),
        "std_runs": 0.70,
        "sample_games": 0.0,
    }


def _int_keyed_history(payload: dict[str, Any]) -> dict[int, dict[str, float]]:
    history: dict[int, dict[str, float]] = {}
    for inning in range(1, 10):
        value = payload.get(str(inning), payload.get(inning, _default_inning_stats(inning)))
        history[inning] = {
            "scoreless_rate": float(value.get("scoreless_rate", MLB_AVG_SCORELESS[inning])),
            "avg_runs": float(value.get("avg_runs", max(0.0, 1.0 - MLB_AVG_SCORELESS[inning]))),
            "std_runs": float(value.get("std_runs", 0.70)),
            "sample_games": float(value.get("sample_games", 0.0)),
        }
    return history


def _parse_game_date(raw_value: Any):
    raw = str(raw_value or "")[:10]
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None
