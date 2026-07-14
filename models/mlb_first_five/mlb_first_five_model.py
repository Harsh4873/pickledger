from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
MLB_INNING_DIR = REPO_ROOT / "models" / "mlb_inning"
OUTPUT_PATH = BASE_DIR / "mlb_first_five_output.json"

if str(MLB_INNING_DIR) not in sys.path:
    sys.path.insert(0, str(MLB_INNING_DIR))

from mlb_inning_fetcher import (  # noqa: E402
    API_BASE,
    DEFAULT_BATTER,
    DEFAULT_PITCHER,
    SCHEDULE_TTL_SECONDS,
    STATS_TTL_SECONDS,
    api_get_json,
    cache_get,
    cache_set,
    fetch_todays_games,
    log_warning,
    normalize_date,
    safe_float,
    safe_int,
)

try:
    from mlb_first_five_environment import (  # noqa: E402
        blend_park_run_delta,
        parse_wind,
        wind_run_delta,
    )
except ImportError:
    from .mlb_first_five_environment import (  # noqa: E402
        blend_park_run_delta,
        parse_wind,
        wind_run_delta,
    )


LEAGUE_TEAM_F5_RUNS = 2.25
LEAGUE_F5_TOTAL = LEAGUE_TEAM_F5_RUNS * 2.0
LEAGUE_ERA = 4.20
LEAGUE_WHIP = 1.30
LEAGUE_OPS_ALLOWED = 0.730
LINEUP_THREAT_BASELINE = (DEFAULT_BATTER["obp"] * 0.58) + (DEFAULT_BATTER["slg"] * 0.42)
F5_SIDE_SIGMA = 3.10
F5_TOTAL_SIGMA = 1.85
ASSUMED_PRICE = -110
ASSUMED_BREAKEVEN = 0.5238
F5_TOTAL_USER_LINE_ODDS = {
    3.5: -170,
    4.5: -130,
    5.5: -170,
}
TOTAL_BET_GAP = 0.65
TOTAL_LEAN_GAP = 0.50
SIDE_BET_MARGIN = 0.90
SIDE_LEAN_MARGIN = 0.55
SIDE_MIN_STARTER_SAMPLE = 5
TOTAL_MIN_STARTER_SAMPLE = 2


def run_mlb_first_five_model(target_date: str | date | None = None) -> list[dict[str, Any]]:
    model_date = normalize_date(target_date)
    games = fetch_todays_games(model_date)
    if not games:
        output = {"date": model_date, "model": "MLBFirstFive", "picks": []}
        _write_output(output)
        print(f"[MLBFirstFive] No eligible MLB games found for {model_date}.")
        return []

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(games)))) as executor:
        picks = list(executor.map(lambda game: _project_game(game, model_date), games))

    output = {
        "date": model_date,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": "MLBFirstFive",
        "picks": picks,
        "method": {
            "market": "first_five",
            "weighting": {
                "current_pitcher_form": "primary",
                "current_pitcher_vs_opponent": "highest matchup override",
                "batter_vs_pitcher": "lineup weighted with current-season PA first",
                "older_years": "shrunken prior-season pitcher/team context",
                "venue": "park run environment plus team venue experience",
            },
            "assumed_price": ASSUMED_PRICE,
        },
    }
    _write_output(output)
    print(f"[MLBFirstFive] {len(picks)} games processed. Output saved to {OUTPUT_PATH.name}")
    return picks


def _project_game(game: dict[str, Any], model_date: str) -> dict[str, Any]:
    away_team = str(game.get("away_team") or "Away Team")
    home_team = str(game.get("home_team") or "Home Team")
    away_team_id = safe_int(game.get("away_team_id"))
    home_team_id = safe_int(game.get("home_team_id"))
    venue_id = safe_int(game.get("venue_id"))
    venue_name = str(game.get("venue_name") or "")

    with ThreadPoolExecutor(max_workers=6) as executor:
        away_team_future = executor.submit(_team_f5_profile, away_team_id, model_date, venue_id)
        home_team_future = executor.submit(_team_f5_profile, home_team_id, model_date, venue_id)
        away_pitcher_future = executor.submit(
            _pitcher_f5_profile,
            safe_int((game.get("away_pitcher") or {}).get("id")),
            home_team_id,
            venue_id,
            model_date,
            game.get("away_pitcher") or {},
        )
        home_pitcher_future = executor.submit(
            _pitcher_f5_profile,
            safe_int((game.get("home_pitcher") or {}).get("id")),
            away_team_id,
            venue_id,
            model_date,
            game.get("home_pitcher") or {},
        )
        away_lineup_future = executor.submit(
            _lineup_matchup_delta,
            game.get("away_lineup") or [],
            game.get("home_pitcher") or {},
            model_date,
        )
        home_lineup_future = executor.submit(
            _lineup_matchup_delta,
            game.get("home_lineup") or [],
            game.get("away_pitcher") or {},
            model_date,
        )

        away_team_profile = away_team_future.result()
        home_team_profile = home_team_future.result()
        away_pitcher_profile = away_pitcher_future.result()
        home_pitcher_profile = home_pitcher_future.result()
        away_lineup_delta = away_lineup_future.result()
        home_lineup_delta = home_lineup_future.result()

    venue_profile = _venue_f5_profile(venue_id, model_date)

    # Blend the learned per-venue F5 delta with the static park-factor prior
    # (helps when the learned sample is thin, e.g. early-season).
    park_blend = blend_park_run_delta(
        learned_delta=safe_float(venue_profile.get("run_delta_per_team"), 0.0),
        learned_games=safe_int(venue_profile.get("games"), 0),
        venue_id=venue_id,
    )
    # Wind signal from the live-feed weather string.
    wind_string = ((game.get("weather") or {}).get("wind") or "")
    parsed_wind = parse_wind(wind_string)
    wind_delta = wind_run_delta(wind_string)
    environment_profile = {
        **venue_profile,
        "park_blend": park_blend,
        "park_run_delta_per_team": park_blend["final_delta"],
        "wind_run_delta_per_team": wind_delta,
        "wind_mph": parsed_wind.get("mph", 0.0),
        "wind_direction": parsed_wind.get("direction") or "",
        "weather_raw": wind_string,
    }
    travel = game.get("travel") if isinstance(game.get("travel"), dict) else {}
    away_travel = travel.get("away") if isinstance(travel.get("away"), dict) else {}
    home_travel = travel.get("home") if isinstance(travel.get("home"), dict) else {}

    away_runs, away_factors = _project_team_runs(
        offense_profile=away_team_profile,
        opposing_pitcher_profile=home_pitcher_profile,
        lineup_delta=away_lineup_delta,
        venue_profile=environment_profile,
        travel_profile=away_travel,
    )
    home_runs, home_factors = _project_team_runs(
        offense_profile=home_team_profile,
        opposing_pitcher_profile=away_pitcher_profile,
        lineup_delta=home_lineup_delta,
        venue_profile=environment_profile,
        travel_profile=home_travel,
    )

    total_runs = _clamp(away_runs + home_runs, 2.0, 8.5)
    home_diff = home_runs - away_runs
    home_lead_probability = _clamp(_normal_cdf(home_diff / F5_SIDE_SIGMA), 0.05, 0.95)
    away_lead_probability = _clamp(1.0 - home_lead_probability, 0.05, 0.95)

    top_picks = [
        _side_pick(
            home_team=home_team,
            away_team=away_team,
            home_runs=home_runs,
            away_runs=away_runs,
            home_probability=home_lead_probability,
            away_probability=away_lead_probability,
        ),
        _total_pick(total_runs),
    ]
    _apply_pick_guardrails(top_picks, away_pitcher_profile, home_pitcher_profile)
    top_picks.sort(key=lambda pick: (_decision_rank(pick.get("decision")), -safe_float(pick.get("edge_pct"), -99.0)))

    game_projection = {
        "game_id": str(game.get("game_id") or ""),
        "game_start_time": str(game.get("game_start_time") or ""),
        "game_order": game.get("game_order", 0),
        "matchup": f"{away_team} @ {home_team}",
        "home_team": home_team,
        "away_team": away_team,
        "venue_id": venue_id,
        "venue_name": venue_name,
        "home_pitcher": (game.get("home_pitcher") or {}).get("name") or "TBD",
        "away_pitcher": (game.get("away_pitcher") or {}).get("name") or "TBD",
        "projected_first_five": {
            "away_runs": round(away_runs, 2),
            "home_runs": round(home_runs, 2),
            "total_runs": round(total_runs, 2),
            "home_lead_probability": round(home_lead_probability, 4),
            "away_lead_probability": round(away_lead_probability, 4),
        },
        "top_picks": top_picks,
        "features": {
            "away_offense": away_factors,
            "home_offense": home_factors,
            "away_pitcher": away_pitcher_profile,
            "home_pitcher": home_pitcher_profile,
            "venue": environment_profile,
            "travel": {
                "away": away_travel,
                "home": home_travel,
            },
            "away_lineup_matchup": away_lineup_delta,
            "home_lineup_matchup": home_lineup_delta,
        },
    }
    game_projection["notes"] = _game_notes(game_projection)
    return game_projection


def _project_team_runs(
    offense_profile: dict[str, Any],
    opposing_pitcher_profile: dict[str, Any],
    lineup_delta: dict[str, Any],
    venue_profile: dict[str, Any],
    travel_profile: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    offense_current = safe_float(offense_profile.get("current_for"), LEAGUE_TEAM_F5_RUNS)
    offense_recent = safe_float(offense_profile.get("recent_for"), offense_current)
    offense_prior = safe_float(offense_profile.get("prior_for"), LEAGUE_TEAM_F5_RUNS)
    offense_venue = safe_float(offense_profile.get("venue_for"), offense_current)

    pitcher_current = safe_float(opposing_pitcher_profile.get("current_f5_allowed"), opposing_pitcher_profile.get("stat_expected"))
    pitcher_recent = safe_float(opposing_pitcher_profile.get("recent_f5_allowed"), pitcher_current)
    pitcher_stat = safe_float(opposing_pitcher_profile.get("stat_expected"), LEAGUE_TEAM_F5_RUNS)
    current_vs_opponent = safe_float(opposing_pitcher_profile.get("current_vs_opponent"), pitcher_current)
    prior_vs_opponent = safe_float(opposing_pitcher_profile.get("prior_vs_opponent"), pitcher_current)
    venue_pitcher = safe_float(opposing_pitcher_profile.get("venue_f5_allowed"), pitcher_current)

    projected = LEAGUE_TEAM_F5_RUNS
    projected += 0.26 * (pitcher_current - LEAGUE_TEAM_F5_RUNS)
    projected += 0.16 * (pitcher_recent - LEAGUE_TEAM_F5_RUNS)
    projected += 0.14 * (pitcher_stat - LEAGUE_TEAM_F5_RUNS)
    projected += _sample_weight(opposing_pitcher_profile.get("current_vs_opponent_starts"), 2, 0.20) * (current_vs_opponent - pitcher_current)
    projected += _sample_weight(opposing_pitcher_profile.get("prior_vs_opponent_starts"), 3, 0.08) * (prior_vs_opponent - pitcher_current)
    projected += _sample_weight(opposing_pitcher_profile.get("venue_starts"), 5, 0.06) * (venue_pitcher - pitcher_current)
    projected += 0.18 * (offense_current - LEAGUE_TEAM_F5_RUNS)
    projected += 0.10 * (offense_recent - offense_current)
    projected += 0.05 * (offense_prior - LEAGUE_TEAM_F5_RUNS)
    projected += _sample_weight(offense_profile.get("venue_games"), 8, 0.07) * (offense_venue - offense_current)
    projected += safe_float(lineup_delta.get("run_delta"), 0.0)
    # Park delta — prefers the blended (learned + static) value when set,
    # falls back to the legacy learned-only field for older callers.
    park_delta = safe_float(
        venue_profile.get("park_run_delta_per_team"),
        safe_float(venue_profile.get("run_delta_per_team"), 0.0),
    )
    projected += park_delta
    # Wind delta — already in F5 runs/team units, capped at ±0.45.
    wind_delta = safe_float(venue_profile.get("wind_run_delta_per_team"), 0.0)
    projected += wind_delta
    # Pitcher rest — short-rest starters yield more F5 runs; extra-long
    # layoffs add a small rust bump. The modifier is in runs scored
    # AGAINST the opposing pitcher, so we add it to this team's projection.
    rest_modifier = safe_float(opposing_pitcher_profile.get("rest_runs_modifier"), 0.0)
    projected += rest_modifier
    travel_profile = travel_profile if isinstance(travel_profile, dict) else {}
    travel_delta = safe_float(travel_profile.get("travel_run_delta"), 0.0)
    projected += travel_delta

    factors = {
        "projected_runs": round(_clamp(projected, 0.55, 5.25), 2),
        "team_current_f5_runs": offense_current,
        "team_recent_f5_runs": offense_recent,
        "team_prior_f5_runs": offense_prior,
        "team_venue_f5_runs": offense_venue,
        "pitcher_current_f5_allowed": pitcher_current,
        "pitcher_recent_f5_allowed": pitcher_recent,
        "pitcher_current_vs_opponent": current_vs_opponent,
        "pitcher_prior_vs_opponent": prior_vs_opponent,
        "pitcher_venue_f5_allowed": venue_pitcher,
        "pitcher_rest_days": opposing_pitcher_profile.get("rest_days"),
        "pitcher_rest_runs_modifier": rest_modifier,
        "pitcher_rest_label": str(opposing_pitcher_profile.get("rest_label") or ""),
        "lineup_delta": safe_float(lineup_delta.get("run_delta"), 0.0),
        "venue_delta": safe_float(venue_profile.get("run_delta_per_team"), 0.0),
        "park_run_delta_per_team": park_delta,
        "wind_run_delta_per_team": wind_delta,
        "wind_mph": safe_float(venue_profile.get("wind_mph"), 0.0),
        "wind_direction": str(venue_profile.get("wind_direction") or ""),
        "travel_run_delta": travel_delta,
        "travel_fatigue_index": safe_float(travel_profile.get("travel_fatigue_index"), 0.0),
        "travel_days_since_previous_game": travel_profile.get("days_since_previous_game"),
        "travel_distance_miles": travel_profile.get("distance_miles"),
        "travel_timezone_shift_hours": travel_profile.get("timezone_shift_hours"),
        "travel_direction": str(travel_profile.get("travel_direction") or ""),
        "travel_label": str(travel_profile.get("label") or ""),
    }
    return factors["projected_runs"], factors


def _team_f5_profile(team_id: int, model_date: str, venue_id: int) -> dict[str, Any]:
    if not team_id:
        return _default_team_profile()
    cache_key = f"first_five_team_{team_id}_{venue_id}_{model_date}"
    cached = cache_get(cache_key, STATS_TTL_SECONDS)
    if cached is not None:
        return cached

    season = _season_for_date(model_date)
    current_records = _team_f5_records_for_season(team_id, season, model_date)
    prior_records: list[dict[str, Any]] = []
    for prior_season in (season - 1, season - 2):
        prior_records.extend(_team_f5_records_for_season(team_id, prior_season, f"{season}-12-31"))
        if len(prior_records) >= 90:
            break

    current_records.sort(key=lambda record: str(record.get("date") or ""))
    recent_records = current_records[-24:]
    venue_records = [
        record
        for record in (current_records + prior_records)
        if venue_id and safe_int(record.get("venue_id")) == venue_id
    ][-36:]

    profile = {
        "current_for": _shrunk_mean([r["for"] for r in current_records], LEAGUE_TEAM_F5_RUNS, 24),
        "current_allowed": _shrunk_mean([r["against"] for r in current_records], LEAGUE_TEAM_F5_RUNS, 24),
        "recent_for": _shrunk_mean([r["for"] for r in recent_records], LEAGUE_TEAM_F5_RUNS, 12),
        "recent_allowed": _shrunk_mean([r["against"] for r in recent_records], LEAGUE_TEAM_F5_RUNS, 12),
        "prior_for": _shrunk_mean([r["for"] for r in prior_records], LEAGUE_TEAM_F5_RUNS, 50),
        "venue_for": _shrunk_mean([r["for"] for r in venue_records], LEAGUE_TEAM_F5_RUNS, 10),
        "current_games": len(current_records),
        "recent_games": len(recent_records),
        "prior_games": len(prior_records),
        "venue_games": len(venue_records),
    }
    cache_set(cache_key, profile)
    return profile


def _team_f5_records_for_season(team_id: int, season: int, before_date: str) -> list[dict[str, Any]]:
    schedule = _team_schedule_with_linescore(team_id, season)
    cutoff = datetime.strptime(before_date, "%Y-%m-%d").date()
    records: list[dict[str, Any]] = []
    for game in _schedule_games(schedule):
        status = str(((game.get("status") or {}).get("detailedState")) or "")
        game_date = _parse_game_date(game.get("officialDate") or game.get("gameDate"))
        if not game_date or game_date >= cutoff or "final" not in status.lower():
            continue
        record = _first_five_record_from_schedule_game(game, team_id)
        if record:
            records.append(record)
    return records


def _pitcher_f5_profile(
    pitcher_id: int,
    opponent_team_id: int,
    venue_id: int,
    model_date: str,
    pitcher: dict[str, Any],
) -> dict[str, Any]:
    if not pitcher_id:
        return _default_pitcher_profile(pitcher)
    cache_key = f"first_five_pitcher_{pitcher_id}_{opponent_team_id}_{venue_id}_{model_date}"
    cached = cache_get(cache_key, STATS_TTL_SECONDS)
    if cached is not None:
        return cached

    season = _season_for_date(model_date)
    current_splits = _pitcher_start_splits(pitcher_id, season, model_date)
    prior_splits: list[dict[str, Any]] = []
    for prior_season in (season - 1, season - 2, season - 3):
        prior_splits.extend(_pitcher_start_splits(pitcher_id, prior_season, f"{season}-12-31"))

    current_records = [_pitcher_start_record(split, fetch_linescore=True) for split in current_splits[:16]]
    current_records = [record for record in current_records if record]
    current_records.sort(key=lambda record: str(record.get("date") or ""))
    recent_records = current_records[-6:]

    current_vs_opponent = [
        record
        for record in current_records
        if safe_int(record.get("opponent_id")) == opponent_team_id
    ]

    prior_vs_splits = [
        split for split in prior_splits
        if safe_int(((split.get("opponent") or {}).get("id"))) == opponent_team_id
    ][:12]
    prior_vs_records = [_pitcher_start_record(split, fetch_linescore=True) for split in prior_vs_splits]
    prior_vs_records = [record for record in prior_vs_records if record]

    prior_general_records = [_pitcher_start_record(split, fetch_linescore=False) for split in prior_splits[:30]]
    prior_general_records = [record for record in prior_general_records if record]

    venue_records = [
        record
        for record in current_records
        if venue_id and safe_int(record.get("venue_id")) == venue_id
    ]
    if venue_id and len(venue_records) < 4:
        for split in prior_splits[:24]:
            record = _pitcher_start_record(split, fetch_linescore=True)
            if record and safe_int(record.get("venue_id")) == venue_id:
                venue_records.append(record)
            if len(venue_records) >= 8:
                break

    rest_days = _pitcher_rest_days(current_records, model_date)
    rest_runs_modifier, rest_label = _pitcher_rest_runs_modifier(rest_days)

    profile = {
        "name": str(pitcher.get("name") or "TBD"),
        "stat_expected": _pitcher_stat_expected(pitcher),
        "current_f5_allowed": _shrunk_mean([r["f5_allowed"] for r in current_records], _pitcher_stat_expected(pitcher), 6),
        "recent_f5_allowed": _shrunk_mean([r["f5_allowed"] for r in recent_records], _pitcher_stat_expected(pitcher), 4),
        "prior_f5_allowed": _shrunk_mean([r["f5_allowed"] for r in prior_general_records], _pitcher_stat_expected(pitcher), 16),
        "current_vs_opponent": _shrunk_mean([r["f5_allowed"] for r in current_vs_opponent], _pitcher_stat_expected(pitcher), 1.5),
        "prior_vs_opponent": _shrunk_mean([r["f5_allowed"] for r in prior_vs_records], _pitcher_stat_expected(pitcher), 2.5),
        "venue_f5_allowed": _shrunk_mean([r["f5_allowed"] for r in venue_records], _pitcher_stat_expected(pitcher), 4),
        "current_starts": len(current_records),
        "recent_starts": len(recent_records),
        "prior_starts": len(prior_general_records),
        "current_vs_opponent_starts": len(current_vs_opponent),
        "prior_vs_opponent_starts": len(prior_vs_records),
        "venue_starts": len(venue_records),
        "rest_days": rest_days,
        "rest_runs_modifier": rest_runs_modifier,
        "rest_label": rest_label,
        "team_bullpen": pitcher.get("team_bullpen") if isinstance(pitcher.get("team_bullpen"), dict) else {},
    }
    cache_set(cache_key, profile)
    return profile


def _pitcher_rest_days(current_records: list[dict[str, Any]], model_date: str) -> int | None:
    """Days between this pitcher's last start and the model date.

    Convention: ``rest_days = days_between_starts - 1`` (i.e. 4 days rest =
    5 days between starts, which is the normal modern MLB cycle). Returns
    ``None`` when there's no prior start in the current-season log.
    """
    if not current_records:
        return None
    try:
        target = datetime.strptime(str(model_date)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None

    last_start_date = None
    for record in reversed(current_records):  # records are date-ascending
        raw_date = str(record.get("date") or "")[:10]
        if not raw_date:
            continue
        try:
            last_start_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
            break
        except ValueError:
            continue
    if last_start_date is None or last_start_date >= target:
        return None
    return max(0, (target - last_start_date).days - 1)


def _pitcher_rest_runs_modifier(rest_days: int | None) -> tuple[float, str]:
    """Translate days-rest into a per-team F5 runs-allowed adjustment.

    Empirical splits from MLB starter performance by rest:
      - 0-2 days rest → emergency / opener territory; +0.30 runs allowed
      - 3 days rest  → "short rest"; +0.20 runs allowed
      - 4-6 days     → normal modern cycle; 0
      - 7-9 days     → mild rust / long layoff; +0.05 runs allowed
      - 10+ days     → IL return or skip; +0.15 runs allowed
      - None         → first start of season / unknown; 0 (no signal)

    Returned adj is in F5 runs/team units (positive = MORE runs scored
    AGAINST this pitcher). The picks layer adds it to the opposing
    team's projected_runs.
    """
    if rest_days is None:
        return 0.0, "rest unknown"
    if rest_days <= 2:
        return 0.30, f"emergency/opener ({rest_days}d rest) +0.30"
    if rest_days == 3:
        return 0.20, f"short rest ({rest_days}d) +0.20"
    if rest_days <= 6:
        return 0.0, f"normal rest ({rest_days}d)"
    if rest_days <= 9:
        return 0.05, f"long layoff ({rest_days}d) +0.05"
    return 0.15, f"extended layoff ({rest_days}d) +0.15"


def _lineup_matchup_delta(lineup: list[dict[str, Any]], pitcher: dict[str, Any], model_date: str) -> dict[str, Any]:
    pitcher_id = safe_int(pitcher.get("id"))
    if not lineup or not pitcher_id:
        return {
            "run_delta": 0.0,
            "threat_score": round(LINEUP_THREAT_BASELINE, 4),
            "current_bvp_pa": 0,
            "older_bvp_pa": 0,
            "sampled_batters": 0,
        }

    season = _season_for_date(model_date)
    ordered = sorted(lineup, key=lambda batter: safe_int(batter.get("batting_order"), 99))[:9]
    if len(ordered) < 9:
        ordered = ordered + [{"batting_order": len(ordered) + 1, **DEFAULT_BATTER} for _ in range(9 - len(ordered))]

    threat_values: list[float] = []
    weights: list[float] = []
    current_pa_total = 0
    older_pa_total = 0

    for batter in ordered:
        order = safe_int(batter.get("batting_order"), 9)
        lineup_weight = 1.22 if order <= 4 else 1.0 if order <= 7 else 0.84
        batter_obp = safe_float(batter.get("obp"), DEFAULT_BATTER["obp"])
        batter_slg = safe_float(batter.get("slg"), DEFAULT_BATTER["slg"])
        pitcher_obp = safe_float(pitcher.get("opponent_obp"), DEFAULT_PITCHER["opponent_obp"])
        pitcher_slg = safe_float(pitcher.get("opponent_slg"), DEFAULT_PITCHER["opponent_slg"])
        fallback_threat = ((batter_obp * 0.62) + (pitcher_obp * 0.38)) * 0.58
        fallback_threat += ((batter_slg * 0.62) + (pitcher_slg * 0.38)) * 0.42

        splits = _batter_vs_pitcher_splits(safe_int(batter.get("player_id")), pitcher_id)
        current_threats: list[tuple[float, int]] = []
        older_threats: list[tuple[float, int]] = []
        for split in splits:
            stat = split.get("stat") or {}
            pa = safe_int(stat.get("plateAppearances"))
            if pa <= 0:
                continue
            threat = _threat_from_stat(stat, fallback_threat)
            split_season = safe_int(split.get("season"))
            if split_season == season:
                current_threats.append((threat, pa))
                current_pa_total += pa
            elif split_season:
                older_threats.append((threat, pa))
                older_pa_total += pa

        current_threat, current_pa = _weighted_threat(current_threats, fallback_threat)
        older_threat, older_pa = _weighted_threat(older_threats, fallback_threat)
        current_weight = min(0.35, current_pa / 20.0) if current_pa else 0.0
        older_weight = min(0.15, older_pa / 60.0) if older_pa else 0.0
        if current_weight + older_weight > 0.45:
            scale = 0.45 / (current_weight + older_weight)
            current_weight *= scale
            older_weight *= scale

        blended = (
            fallback_threat * (1.0 - current_weight - older_weight)
            + current_threat * current_weight
            + older_threat * older_weight
        )
        threat_values.append(blended)
        weights.append(lineup_weight)

    threat_score = _weighted_average(threat_values, weights, LINEUP_THREAT_BASELINE)
    run_delta = _clamp((threat_score - LINEUP_THREAT_BASELINE) * 2.35, -0.25, 0.25)
    return {
        "run_delta": round(run_delta, 3),
        "threat_score": round(threat_score, 4),
        "current_bvp_pa": current_pa_total,
        "older_bvp_pa": older_pa_total,
        "sampled_batters": len(ordered),
    }


def _venue_f5_profile(venue_id: int, model_date: str) -> dict[str, Any]:
    if not venue_id:
        return {"f5_total": LEAGUE_F5_TOTAL, "run_delta_per_team": 0.0, "games": 0}
    cache_key = f"first_five_venue_{venue_id}_{model_date}"
    cached = cache_get(cache_key, STATS_TTL_SECONDS)
    if cached is not None:
        return cached

    season = _season_for_date(model_date)
    records: list[float] = []
    for lookup_season in (season, season - 1, season - 2):
        schedule = _league_schedule_with_linescore(lookup_season)
        cutoff = datetime.strptime(model_date, "%Y-%m-%d").date()
        for game in _schedule_games(schedule):
            game_date = _parse_game_date(game.get("officialDate") or game.get("gameDate"))
            status = str(((game.get("status") or {}).get("detailedState")) or "")
            game_venue_id = safe_int((game.get("venue") or {}).get("id"))
            if not game_date or game_date >= cutoff or "final" not in status.lower() or game_venue_id != venue_id:
                continue
            innings = (((game.get("linescore") or {}).get("innings")) or [])[:5]
            if len(innings) < 5:
                continue
            total = sum(safe_int((inning.get("home") or {}).get("runs")) + safe_int((inning.get("away") or {}).get("runs")) for inning in innings)
            records.append(float(total))
        if len(records) >= 60:
            break

    f5_total = _shrunk_mean(records[-60:], LEAGUE_F5_TOTAL, 24)
    profile = {
        "f5_total": f5_total,
        "run_delta_per_team": round(_clamp((f5_total - LEAGUE_F5_TOTAL) / 2.0, -0.18, 0.18), 3),
        "games": len(records[-60:]),
    }
    cache_set(cache_key, profile)
    return profile


def _side_pick(
    home_team: str,
    away_team: str,
    home_runs: float,
    away_runs: float,
    home_probability: float,
    away_probability: float,
) -> dict[str, Any]:
    if home_probability >= away_probability:
        team = home_team
        probability = home_probability
        diff = home_runs - away_runs
    else:
        team = away_team
        probability = away_probability
        diff = away_runs - home_runs
    edge_pct = (probability - ASSUMED_BREAKEVEN) * 100.0
    return {
        "market": "f5_side",
        "pick": f"{team} F5 ML",
        "team": team,
        "probability": round(probability, 4),
        "edge_pct": round(edge_pct, 2),
        "decision": _decision(probability, edge_pct),
        "confidence": _confidence(probability, edge_pct),
        "assumed_odds": ASSUMED_PRICE,
        "projected_margin": round(diff, 2),
        "model_prediction": f"{team} +{diff:.2f} F5 runs",
    }


def _total_pick(total_runs: float) -> dict[str, Any]:
    if total_runs >= LEAGUE_F5_TOTAL:
        side = "Over"
        line = _nearest_f5_total_line(_half_line_below(total_runs))
        probability = _clamp(1.0 - _normal_cdf((line - total_runs) / F5_TOTAL_SIGMA), 0.05, 0.95)
    else:
        side = "Under"
        line = _nearest_f5_total_line(_half_line_above(total_runs))
        probability = _clamp(_normal_cdf((line - total_runs) / F5_TOTAL_SIGMA), 0.05, 0.95)
    assumed_odds = F5_TOTAL_USER_LINE_ODDS[line]
    edge_pct = (probability - _american_implied_probability(assumed_odds)) * 100.0
    projection_gap = abs(total_runs - line)
    return {
        "market": "f5_total",
        "pick": f"{side} {line:.1f} F5",
        "team": "",
        "probability": round(probability, 4),
        "edge_pct": round(edge_pct, 2),
        "decision": _decision(probability, edge_pct),
        "confidence": _confidence(probability, edge_pct),
        "assumed_odds": assumed_odds,
        "vegas_line": line,
        "projection_gap": round(projection_gap, 2),
        "model_prediction": f"{total_runs:.2f} F5 total",
    }


def _pitcher_start_splits(pitcher_id: int, season: int, before_date: str) -> list[dict[str, Any]]:
    if not pitcher_id:
        return []
    try:
        payload = api_get_json(
            f"{API_BASE}/people/{pitcher_id}/stats",
            params={"stats": "gameLog", "group": "pitching", "season": season},
            cache_key=f"pitcher_gamelog_{pitcher_id}_{season}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return []

    cutoff = datetime.strptime(before_date, "%Y-%m-%d").date()
    splits = []
    for split in ((payload.get("stats") or [{}])[0].get("splits") or []):
        game_date = _parse_game_date(split.get("date"))
        stat = split.get("stat") or {}
        if not game_date or game_date >= cutoff:
            continue
        if safe_int(stat.get("gamesStarted")) <= 0:
            continue
        splits.append(split)
    splits.sort(key=lambda item: str(item.get("date") or ""), reverse=True)
    return splits


def _pitcher_start_record(split: dict[str, Any], fetch_linescore: bool) -> dict[str, Any] | None:
    stat = split.get("stat") or {}
    game = split.get("game") or {}
    game_pk = safe_int(game.get("gamePk"))
    team_id = safe_int(((split.get("team") or {}).get("id")))
    opponent_id = safe_int(((split.get("opponent") or {}).get("id")))
    f5_allowed = _estimated_pitcher_f5_allowed(stat)
    venue_id = 0

    if fetch_linescore and game_pk:
        schedule_game = _schedule_game_by_pk(game_pk)
        if schedule_game:
            venue_id = safe_int((schedule_game.get("venue") or {}).get("id"))
            side = _team_side_from_schedule_game(schedule_game, team_id)
            if side:
                opponent_side = "away" if side == "home" else "home"
                linescore_value = _first_five_runs(schedule_game, opponent_side)
                if linescore_value is not None:
                    f5_allowed = float(linescore_value)

    return {
        "date": str(split.get("date") or ""),
        "game_pk": game_pk,
        "team_id": team_id,
        "opponent_id": opponent_id,
        "venue_id": venue_id,
        "f5_allowed": _clamp(f5_allowed, 0.0, 9.0),
    }


def _team_schedule_with_linescore(team_id: int, season: int) -> dict[str, Any]:
    try:
        return api_get_json(
            f"{API_BASE}/schedule",
            params={
                "sportId": 1,
                "teamId": team_id,
                "startDate": f"{season}-03-01",
                "endDate": f"{season}-11-30",
                "hydrate": "linescore",
            },
            cache_key=f"team_schedule_linescore_{team_id}_{season}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}


def _league_schedule_with_linescore(season: int) -> dict[str, Any]:
    try:
        return api_get_json(
            f"{API_BASE}/schedule",
            params={
                "sportId": 1,
                "startDate": f"{season}-03-01",
                "endDate": f"{season}-11-30",
                "hydrate": "linescore",
            },
            cache_key=f"league_schedule_linescore_{season}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}


def _schedule_game_by_pk(game_pk: int) -> dict[str, Any]:
    if not game_pk:
        return {}
    try:
        payload = api_get_json(
            f"{API_BASE}/schedule",
            params={"sportId": 1, "gamePk": game_pk, "hydrate": "linescore"},
            cache_key=f"schedule_game_linescore_{game_pk}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}
    games = _schedule_games(payload)
    return games[0] if games else {}


def _batter_vs_pitcher_splits(batter_id: int, pitcher_id: int) -> list[dict[str, Any]]:
    if not batter_id or not pitcher_id:
        return []
    try:
        payload = api_get_json(
            f"{API_BASE}/people/{batter_id}/stats",
            params={"stats": "vsPlayer", "opposingPlayerId": pitcher_id, "group": "hitting"},
            cache_key=f"first_five_bvp_{batter_id}_vs_{pitcher_id}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return []
    splits: list[dict[str, Any]] = []
    for stat_group in payload.get("stats") or []:
        display_name = str(((stat_group.get("type") or {}).get("displayName")) or "")
        if display_name.lower() == "vsplayertotal":
            continue
        splits.extend(stat_group.get("splits") or [])
    return splits


def _first_five_record_from_schedule_game(game: dict[str, Any], team_id: int) -> dict[str, Any] | None:
    side = _team_side_from_schedule_game(game, team_id)
    if not side:
        return None
    other_side = "away" if side == "home" else "home"
    f5_for = _first_five_runs(game, side)
    f5_against = _first_five_runs(game, other_side)
    if f5_for is None or f5_against is None:
        return None
    return {
        "date": str(game.get("officialDate") or str(game.get("gameDate") or "")[:10]),
        "game_pk": safe_int(game.get("gamePk")),
        "venue_id": safe_int((game.get("venue") or {}).get("id")),
        "for": float(f5_for),
        "against": float(f5_against),
    }


def _first_five_runs(game: dict[str, Any], side: str) -> int | None:
    innings = (((game.get("linescore") or {}).get("innings")) or [])[:5]
    if len(innings) < 5:
        return None
    return sum(safe_int((inning.get(side) or {}).get("runs"), 0) for inning in innings)


def _team_side_from_schedule_game(game: dict[str, Any], team_id: int) -> str:
    teams = game.get("teams") or {}
    if safe_int((((teams.get("home") or {}).get("team") or {}).get("id"))) == team_id:
        return "home"
    if safe_int((((teams.get("away") or {}).get("team") or {}).get("id"))) == team_id:
        return "away"
    return ""


def _schedule_games(schedule_payload: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for day in schedule_payload.get("dates") or []:
        games.extend(day.get("games") or [])
    return games


def _estimated_pitcher_f5_allowed(stat: dict[str, Any]) -> float:
    runs = safe_float(stat.get("runs"), LEAGUE_TEAM_F5_RUNS)
    innings = _parse_innings_pitched(stat.get("inningsPitched"))
    if innings <= 0:
        return _clamp(runs, 0.0, 8.0)
    return _clamp((runs / innings) * min(5.0, innings), 0.0, 8.0)


def _pitcher_stat_expected(pitcher: dict[str, Any]) -> float:
    era_component = safe_float(pitcher.get("era"), DEFAULT_PITCHER["era"]) / LEAGUE_ERA
    whip_component = safe_float(pitcher.get("whip"), DEFAULT_PITCHER["whip"]) / LEAGUE_WHIP
    ops_component = (
        safe_float(pitcher.get("opponent_obp"), DEFAULT_PITCHER["opponent_obp"])
        + safe_float(pitcher.get("opponent_slg"), DEFAULT_PITCHER["opponent_slg"])
    ) / LEAGUE_OPS_ALLOWED
    multiplier = (era_component * 0.40) + (whip_component * 0.30) + (ops_component * 0.30)
    return round(_clamp(LEAGUE_TEAM_F5_RUNS * multiplier, 0.75, 4.75), 3)


def _threat_from_stat(stat: dict[str, Any], default: float) -> float:
    obp = safe_float(stat.get("obp"), DEFAULT_BATTER["obp"])
    slg = safe_float(stat.get("slg"), DEFAULT_BATTER["slg"])
    if obp <= 0 and slg <= 0:
        return default
    return (obp * 0.58) + (slg * 0.42)


def _weighted_threat(values: list[tuple[float, int]], default: float) -> tuple[float, int]:
    total_pa = sum(max(0, pa) for _, pa in values)
    if total_pa <= 0:
        return default, 0
    return sum(threat * pa for threat, pa in values) / total_pa, total_pa


def _weighted_average(values: list[float], weights: list[float], default: float) -> float:
    if not values or not weights or len(values) != len(weights):
        return default
    total_weight = sum(max(0.0, weight) for weight in weights)
    if total_weight <= 0:
        return default
    return sum(value * max(0.0, weight) for value, weight in zip(values, weights)) / total_weight


def _shrunk_mean(values: list[float], default: float, sample_scale: float) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return round(default, 3)
    mean = sum(clean) / len(clean)
    weight = min(1.0, len(clean) / max(1.0, sample_scale))
    return round((mean * weight) + (default * (1.0 - weight)), 3)


def _sample_weight(sample: Any, full_sample: float, max_weight: float) -> float:
    return min(max_weight, max_weight * (safe_float(sample, 0.0) / max(1.0, full_sample)))


def _apply_pick_guardrails(
    picks: list[dict[str, Any]],
    away_pitcher_profile: dict[str, Any],
    home_pitcher_profile: dict[str, Any],
) -> None:
    min_starts = min(
        safe_int(away_pitcher_profile.get("current_starts")),
        safe_int(home_pitcher_profile.get("current_starts")),
    )
    for pick in picks:
        probability = safe_float(pick.get("probability"), 0.0)
        edge_pct = safe_float(pick.get("edge_pct"), -99.0)
        market = str(pick.get("market") or "")

        if market == "f5_total":
            gap = safe_float(pick.get("projection_gap"), 0.0)
            reasons: list[str] = []
            synthetic_total_line = pick.get("vegas_line") is not None
            if min_starts < TOTAL_MIN_STARTER_SAMPLE and gap < 1.20:
                reasons.append("thin starter sample")
            if gap < TOTAL_LEAN_GAP:
                reasons.append("projection too close to F5 line")
                pick["decision"] = "PASS"
            elif gap < TOTAL_BET_GAP or reasons:
                pick["decision"] = "LEAN" if probability >= 0.55 and edge_pct > 2.0 else "PASS"
            else:
                pick["decision"] = _decision(probability, edge_pct)
            if synthetic_total_line and pick["decision"] == "BET":
                pick["decision"] = "LEAN"
                reasons.append("model-generated F5 total line; capped at LEAN until real market line is available")
            if reasons or gap < TOTAL_BET_GAP:
                pick["guardrail"] = ", ".join(reasons or ["line gap below bet threshold"])
            pick["confidence"] = _confidence_for_decision(pick["decision"], probability, edge_pct)
            continue

        if market == "f5_side":
            margin = safe_float(pick.get("projected_margin"), 0.0)
            reasons = []
            if min_starts < SIDE_MIN_STARTER_SAMPLE:
                reasons.append("thin starter sample")
            if margin < SIDE_LEAN_MARGIN:
                reasons.append("margin too small")
                pick["decision"] = "PASS"
            elif margin < SIDE_BET_MARGIN or reasons:
                pick["decision"] = "LEAN" if probability >= 0.56 and edge_pct > 3.0 else "PASS"
            else:
                pick["decision"] = "BET" if probability >= 0.60 and edge_pct >= 7.0 else _decision(probability, edge_pct)
            if reasons or margin < SIDE_BET_MARGIN:
                pick["guardrail"] = ", ".join(reasons or ["margin below bet threshold"])
            pick["confidence"] = _confidence_for_decision(pick["decision"], probability, edge_pct)


def _decision(probability: float, edge_pct: float) -> str:
    if probability >= 0.56 and edge_pct >= 3.2:
        return "BET"
    if probability >= 0.535 and edge_pct >= 1.1:
        return "LEAN"
    return "PASS"


def _confidence(probability: float, edge_pct: float) -> str:
    if probability >= 0.60 or edge_pct >= 7.5:
        return "High"
    if probability >= 0.56 or edge_pct >= 3.2:
        return "Medium"
    return "Low"


def _confidence_for_decision(decision: str, probability: float, edge_pct: float) -> str:
    if str(decision or "").upper() == "BET":
        return _confidence(probability, edge_pct)
    if str(decision or "").upper() == "LEAN":
        return "Medium"
    return "Low"


def _decision_rank(decision: Any) -> int:
    return {"BET": 0, "LEAN": 1, "PASS": 2}.get(str(decision or "").upper(), 3)


def _half_line_below(value: float) -> float:
    return _clamp(math.floor(value - 0.5) + 0.5, 2.5, 8.5)


def _half_line_above(value: float) -> float:
    return _clamp(math.ceil(value - 0.5) + 0.5, 2.5, 8.5)


def _nearest_f5_total_line(value: float) -> float:
    return min(F5_TOTAL_USER_LINE_ODDS, key=lambda line: abs(line - value))


def _american_implied_probability(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(float(odds)) / (abs(float(odds)) + 100.0)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_innings_pitched(value: Any) -> float:
    raw = str(value or "0").strip()
    if not raw:
        return 0.0
    whole, dot, frac = raw.partition(".")
    try:
        innings = float(int(whole or "0"))
    except ValueError:
        return safe_float(value, 0.0)
    if dot:
        if frac == "1":
            innings += 1.0 / 3.0
        elif frac == "2":
            innings += 2.0 / 3.0
        else:
            innings += safe_float(f"0.{frac}", 0.0)
    return innings


def _season_for_date(model_date: str) -> int:
    return datetime.strptime(model_date, "%Y-%m-%d").year


def _parse_game_date(raw_value: Any):
    raw = str(raw_value or "")[:10]
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _default_team_profile() -> dict[str, Any]:
    return {
        "current_for": LEAGUE_TEAM_F5_RUNS,
        "current_allowed": LEAGUE_TEAM_F5_RUNS,
        "recent_for": LEAGUE_TEAM_F5_RUNS,
        "recent_allowed": LEAGUE_TEAM_F5_RUNS,
        "prior_for": LEAGUE_TEAM_F5_RUNS,
        "venue_for": LEAGUE_TEAM_F5_RUNS,
        "current_games": 0,
        "recent_games": 0,
        "prior_games": 0,
        "venue_games": 0,
    }


def _default_pitcher_profile(pitcher: dict[str, Any]) -> dict[str, Any]:
    expected = _pitcher_stat_expected(pitcher)
    return {
        "name": str(pitcher.get("name") or "TBD"),
        "stat_expected": expected,
        "current_f5_allowed": expected,
        "recent_f5_allowed": expected,
        "prior_f5_allowed": expected,
        "current_vs_opponent": expected,
        "prior_vs_opponent": expected,
        "venue_f5_allowed": expected,
        "current_starts": 0,
        "recent_starts": 0,
        "prior_starts": 0,
        "current_vs_opponent_starts": 0,
        "prior_vs_opponent_starts": 0,
        "venue_starts": 0,
        "rest_days": pitcher.get("rest_days"),
        "rest_runs_modifier": safe_float(pitcher.get("rest_runs_modifier"), 0.0),
        "rest_label": str(pitcher.get("rest_label") or "rest unknown"),
        "team_bullpen": pitcher.get("team_bullpen") if isinstance(pitcher.get("team_bullpen"), dict) else {},
    }


def _game_notes(game_projection: dict[str, Any]) -> str:
    projected = game_projection.get("projected_first_five") or {}
    features = game_projection.get("features") or {}
    away_lineup = features.get("away_lineup_matchup") or {}
    home_lineup = features.get("home_lineup_matchup") or {}
    venue = features.get("venue") or {}
    travel = features.get("travel") if isinstance(features.get("travel"), dict) else {}
    away_travel = travel.get("away") if isinstance(travel.get("away"), dict) else {}
    home_travel = travel.get("home") if isinstance(travel.get("home"), dict) else {}
    travel_note = ""
    if away_travel or home_travel:
        travel_note = (
            f" Travel away/home: {away_travel.get('label') or 'n/a'} "
            f"(fatigue {safe_float(away_travel.get('travel_fatigue_index'), 0.0):.2f}) / "
            f"{home_travel.get('label') or 'n/a'} "
            f"(fatigue {safe_float(home_travel.get('travel_fatigue_index'), 0.0):.2f})."
        )
    return (
        f"F5 projection {game_projection.get('away_team')} {projected.get('away_runs')}, "
        f"{game_projection.get('home_team')} {projected.get('home_runs')} "
        f"(total {projected.get('total_runs')}). "
        f"BvP PA away/home: {safe_int(away_lineup.get('current_bvp_pa'))}+{safe_int(away_lineup.get('older_bvp_pa'))}/"
        f"{safe_int(home_lineup.get('current_bvp_pa'))}+{safe_int(home_lineup.get('older_bvp_pa'))}; "
        f"venue F5 total {safe_float(venue.get('f5_total'), LEAGUE_F5_TOTAL):.2f} over {safe_int(venue.get('games'))} games."
        f"{travel_note}"
    )


def _write_output(payload: dict[str, Any]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MLB first-five projection model.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    return parser.parse_args()


def _date_or_today(raw_date: str) -> str:
    if raw_date:
        return normalize_date(raw_date)
    return datetime.today().strftime("%Y-%m-%d")


if __name__ == "__main__":
    args = _parse_args()
    run_mlb_first_five_model(_date_or_today(args.date))
