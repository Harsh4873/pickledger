from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

try:
    from mlb_inning_fetcher import (
        API_BASE,
        DEFAULT_BATTER,
        DEFAULT_PITCHER,
        STATS_TTL_SECONDS,
        api_get_json,
        log_warning,
        safe_float,
        safe_int,
    )
except ImportError:
    from .mlb_inning_fetcher import (
        API_BASE,
        DEFAULT_BATTER,
        DEFAULT_PITCHER,
        STATS_TTL_SECONDS,
        api_get_json,
        log_warning,
        safe_float,
        safe_int,
    )


MIN_MATCHUP_PA = 10


def compute_matchup_threats(games: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    _prefetch_matchup_stats(games)
    threats: dict[str, dict[str, Any]] = {}
    for game in games:
        game_id = str(game.get("game_id") or "")
        if not game_id:
            continue
        innings: dict[int, dict[str, float]] = {}
        for inning in range(1, 10):
            away_threat = _half_inning_threat(
                game.get("away_lineup") or [],
                game.get("home_pitcher") or {},
                inning,
            )
            home_threat = _half_inning_threat(
                game.get("home_lineup") or [],
                game.get("away_pitcher") or {},
                inning,
            )
            innings[inning] = {
                "away_threat": away_threat,
                "home_threat": home_threat,
            }
        threats[game_id] = {"game_id": game_id, "innings": innings}
    return threats


def _prefetch_matchup_stats(games: list[dict[str, Any]]) -> None:
    pairs: set[tuple[int, int]] = set()
    for game in games:
        for lineup_key, pitcher_key in (
            ("away_lineup", "home_pitcher"),
            ("home_lineup", "away_pitcher"),
        ):
            pitcher_id = safe_int((game.get(pitcher_key) or {}).get("id"))
            if not pitcher_id:
                continue
            for batter in (game.get(lineup_key) or [])[:9]:
                batter_id = safe_int(batter.get("player_id"))
                if batter_id:
                    pairs.add((batter_id, pitcher_id))
    if not pairs:
        return

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda pair: _batter_vs_pitcher_stat(pair[0], pair[1]), sorted(pairs)))


def _half_inning_threat(lineup: list[dict[str, Any]], pitcher: dict[str, Any], inning: int) -> float:
    batters = _expected_batters(lineup, inning)
    if not batters:
        return _threat_score(DEFAULT_BATTER["obp"], DEFAULT_BATTER["slg"])

    obp_values: list[float] = []
    slg_values: list[float] = []
    pitcher_id = safe_int(pitcher.get("id"))
    for batter in batters:
        matchup = _batter_vs_pitcher_stat(safe_int(batter.get("player_id")), pitcher_id)
        plate_appearances = safe_int(matchup.get("plateAppearances"))
        if plate_appearances >= MIN_MATCHUP_PA:
            obp = safe_float(matchup.get("obp"), safe_float(batter.get("obp"), DEFAULT_BATTER["obp"]))
            slg = safe_float(matchup.get("slg"), safe_float(batter.get("slg"), DEFAULT_BATTER["slg"]))
        else:
            batter_obp = safe_float(batter.get("obp"), DEFAULT_BATTER["obp"])
            batter_slg = safe_float(batter.get("slg"), DEFAULT_BATTER["slg"])
            pitcher_obp = safe_float(pitcher.get("opponent_obp"), DEFAULT_PITCHER["opponent_obp"])
            pitcher_slg = safe_float(pitcher.get("opponent_slg"), DEFAULT_PITCHER["opponent_slg"])
            obp = (batter_obp * 0.60) + (pitcher_obp * 0.40)
            slg = (batter_slg * 0.60) + (pitcher_slg * 0.40)
        obp_values.append(obp)
        slg_values.append(slg)

    matchup_obp = sum(obp_values) / len(obp_values)
    matchup_slg = sum(slg_values) / len(slg_values)
    return round(_threat_score(matchup_obp, matchup_slg), 4)


def _expected_batters(lineup: list[dict[str, Any]], inning: int) -> list[dict[str, Any]]:
    if not lineup:
        return []
    ordered = sorted(lineup, key=lambda player: safe_int(player.get("batting_order"), 99))[:9]
    if len(ordered) < 9:
        while len(ordered) < 9:
            ordered.append(
                {
                    "batting_order": len(ordered) + 1,
                    "player_id": None,
                    "name": "Unknown",
                    **DEFAULT_BATTER,
                }
            )
    start_index = ((inning - 1) * 3) % 9
    return [ordered[(start_index + offset) % 9] for offset in range(3)]


def _batter_vs_pitcher_stat(batter_id: int, pitcher_id: int) -> dict[str, Any]:
    if not batter_id or not pitcher_id:
        return {}
    try:
        payload = api_get_json(
            f"{API_BASE}/people/{batter_id}/stats",
            params={"stats": "vsPlayer", "opposingPlayerId": pitcher_id, "group": "hitting"},
            cache_key=f"matchup_{batter_id}_vs_{pitcher_id}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}

    splits = []
    for stat_group in payload.get("stats") or []:
        splits.extend(stat_group.get("splits") or [])
    if not splits:
        return {}
    return (splits[0].get("stat") or {})


def _threat_score(obp: float, slg: float) -> float:
    return (obp * 0.60) + (slg * 0.40)
