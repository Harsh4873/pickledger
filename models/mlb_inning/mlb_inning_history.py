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

# Shrinkage constants for the small-sample rates extracted from the
# 30-game window: bullpen per-inning rates see ~30 samples (K=10 keeps
# ~75% observed weight); a starter only appears in ~5-6 of those games.
BULLPEN_SHRINK_K = 10.0


def fetch_team_histories(games: list[dict[str, Any]], target_date: str | None = None) -> dict[str, dict[int, dict[str, float]]]:
    """Back-compat offense-only view of :func:`fetch_team_contexts`."""
    return {
        team_name: context.get("offense") or _league_default_history()
        for team_name, context in fetch_team_contexts(games, target_date).items()
    }


def fetch_team_contexts(games: list[dict[str, Any]], target_date: str | None = None) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
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
        return contexts

    workers = max(1, min(HISTORY_TEAM_MAX_WORKERS, len(team_requests)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_team = {
            executor.submit(fetch_team_context, team_id, team_name, game_date): team_name
            for team_id, team_name, game_date in team_requests
        }
        for future in as_completed(future_to_team):
            team_name = future_to_team[future]
            try:
                contexts[team_name] = future.result()
            except Exception as exc:
                log_warning(f"{team_name} history lookup failed: {exc}; using league inning defaults")
                contexts[team_name] = _league_default_context()
    return contexts


def fetch_team_history(team_id: int, team_name: str, target_date: str) -> dict[int, dict[str, float]]:
    return fetch_team_context(team_id, team_name, target_date).get("offense") or _league_default_history()


def fetch_team_context(team_id: int, team_name: str, target_date: str) -> dict[str, Any]:
    """Offense, bullpen, and starter per-inning context from recent games.

    All three come out of the same last-30 cached game feeds the offense
    history has always used, so the added bullpen/starter extraction
    costs no extra round-trips:

    - ``offense``: per-inning scoreless_rate/avg/std (unchanged shape),
    - ``bullpen_scoreless_by_inning``: innings 7-9 runs-allowed scoreless
      rates (relievers own those innings), shrunk toward league average,
    - ``starters``: per starter_id, how often each inning they completed
      was scoreless, plus their average outs per start — this is what
      finally feeds the probability layer's ``inning_scoreless_rates``
      and starter-pull taper, which were dead code paths before.
    """
    cache_key = f"history_v2_{team_id}_{target_date}"
    cached = cache_get(cache_key, STATS_TTL_SECONDS)
    if cached is not None:
        return _normalized_context(cached)

    details = _collect_recent_game_details(team_id, target_date)
    if len(details) < 15:
        log_warning(f"{team_name} has only {len(details)} recent games; using league inning defaults")
        context = _league_default_context()
    else:
        recent = details[-30:]
        context = {
            "offense": _summarize_inning_runs([detail["scored"] for detail in recent]),
            **_summarize_bullpen_allowed(recent),
            "starters": _summarize_starters(recent),
        }

    cache_set(cache_key, context)
    return _normalized_context(context)


def _collect_recent_game_details(team_id: int, target_date: str) -> list[dict[str, Any]]:
    season = datetime.strptime(target_date, "%Y-%m-%d").year
    records: list[dict[str, Any]] = []
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
            detail_records = list(
                executor.map(
                    lambda game_id: _team_game_detail(game_id, team_id),
                    game_ids[:needed],
                )
            )
        for detail in detail_records:
            if detail:
                records.append(detail)
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


def _team_game_detail(game_id: int, team_id: int) -> dict[str, Any]:
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
    opponent_side = "away" if side == "home" else "home"

    innings = (((feed.get("liveData") or {}).get("linescore") or {}).get("innings") or [])
    scored = {inning: 0 for inning in range(1, 10)}
    allowed = {inning: 0 for inning in range(1, 10)}
    for inning_payload in innings:
        inning_number = safe_int(inning_payload.get("num"))
        if 1 <= inning_number <= 9:
            scored[inning_number] = safe_int(((inning_payload.get(side) or {}).get("runs")), 0)
            allowed[inning_number] = safe_int(((inning_payload.get(opponent_side) or {}).get("runs")), 0)

    # First pitcher used is the starter; their outs bound which innings'
    # runs-allowed can be attributed to them.
    box_team = (((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {}
    pitcher_ids = [safe_int(pid) for pid in (box_team.get("pitchers") or []) if safe_int(pid)]
    starter_id = pitcher_ids[0] if pitcher_ids else 0
    starter_outs = 0
    if starter_id:
        starter_entry = (box_team.get("players") or {}).get(f"ID{starter_id}") or {}
        pitching = (starter_entry.get("stats") or {}).get("pitching") or {}
        starter_outs = safe_int(pitching.get("outs"))
        if starter_outs <= 0:
            # "6.1" innings-pitched notation → 6 innings + 1 out = 19 outs.
            ip_text = str(pitching.get("inningsPitched") or "")
            if "." in ip_text:
                whole, _, partial = ip_text.partition(".")
                starter_outs = safe_int(whole) * 3 + safe_int(partial)
            else:
                starter_outs = safe_int(ip_text) * 3

    return {
        "scored": scored,
        "allowed": allowed,
        "starter_id": starter_id,
        "starter_outs": starter_outs,
    }


def _summarize_bullpen_allowed(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Runs-allowed scoreless rates for the bullpen innings (7-9)."""
    by_inning: dict[str, float] = {}
    rates: list[float] = []
    samples = 0
    for inning in (7, 8, 9):
        values = [safe_int((record.get("allowed") or {}).get(inning), 0) for record in records]
        if not values:
            continue
        samples = max(samples, len(values))
        scoreless = sum(1 for value in values if value == 0)
        shrunk = (scoreless + MLB_AVG_SCORELESS[inning] * BULLPEN_SHRINK_K) / (len(values) + BULLPEN_SHRINK_K)
        by_inning[str(inning)] = round(shrunk, 4)
        rates.append(shrunk)
    return {
        "bullpen_scoreless_by_inning": by_inning,
        "bullpen_scoreless_rate": round(sum(rates) / len(rates), 4) if rates else None,
        "bullpen_samples": samples,
    }


def _summarize_starters(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-starter scoreless outcomes for the innings they fully pitched."""
    starters: dict[str, dict[str, Any]] = {}
    for record in records:
        starter_id = safe_int(record.get("starter_id"))
        starter_outs = safe_int(record.get("starter_outs"))
        if not starter_id or starter_outs <= 0:
            continue
        entry = starters.setdefault(str(starter_id), {"innings": {}, "starts": 0, "outs_total": 0})
        entry["starts"] += 1
        entry["outs_total"] += starter_outs
        completed = min(starter_outs // 3, 6)
        allowed = record.get("allowed") or {}
        for inning in range(1, completed + 1):
            inning_entry = entry["innings"].setdefault(str(inning), {"scoreless": 0, "n": 0})
            inning_entry["n"] += 1
            if safe_int(allowed.get(inning), 0) == 0:
                inning_entry["scoreless"] += 1
    for entry in starters.values():
        entry["avg_outs"] = round(entry["outs_total"] / entry["starts"], 2) if entry["starts"] else 0.0
        entry.pop("outs_total", None)
    return starters


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


def _league_default_context() -> dict[str, Any]:
    return {
        "offense": _league_default_history(),
        "bullpen_scoreless_by_inning": {},
        "bullpen_scoreless_rate": None,
        "bullpen_samples": 0,
        "starters": {},
    }


def _normalized_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Restore int offense keys after a JSON cache round-trip; the bullpen
    and starter tables stay string-keyed (the probability layer tries both)."""
    payload = payload if isinstance(payload, dict) else {}
    return {
        "offense": _int_keyed_history(payload.get("offense") or {}),
        "bullpen_scoreless_by_inning": payload.get("bullpen_scoreless_by_inning") or {},
        "bullpen_scoreless_rate": payload.get("bullpen_scoreless_rate"),
        "bullpen_samples": safe_int(payload.get("bullpen_samples"), 0),
        "starters": payload.get("starters") or {},
    }


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
