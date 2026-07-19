#!/usr/bin/env python3
"""MLB-team-only consensus publication gate.

The three MLB team publishers still generate their native projections, but
their public BET/LEAN decisions must clear this post-calibration gate before
the static board or rankings can treat them as bettable output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from player_props.schema import edge_basis_mode, kelly
from scripts.devig import no_vig_selected_probability
from scripts.pick_calibration import american_implied_probability, normalize_probability


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTCOME_LEDGER_PATH = REPO_ROOT / "data" / "calibration" / "outcome_ledger.json"
MLB_TEAM_MODEL_KEYS = {"mlb_new", "mlb_first_five", "mlb_inning"}
MLB_TEAM_CONSENSUS_VERSION = "mlb_team_consensus_v1.1.0"
MLB_TEAM_RANKING_EPOCH_PREFIX = f"MLB:{MLB_TEAM_CONSENSUS_VERSION}"
MIN_WALK_FORWARD_SAMPLES = 30
VALIDATION_LEAN_MODELS = {"mlb_new", "mlb_inning"}
VALIDATION_LEAN_LIMIT = 3

MODEL_BET_TYPE_DEFAULTS = {
    "mlb_new": "h2h",
    "mlb_first_five": "f5_side",
    "mlb_inning": "no_run_inning",
}

PUBLICATION_THRESHOLDS = {
    "mlb_new": {"lean_edge": 3.0, "bet_edge": 7.0, "lean_prob": 0.53, "bet_prob": 0.56, "lean_signals": 3, "bet_signals": 4},
    "mlb_first_five": {"lean_edge": 3.0, "bet_edge": 7.0, "lean_prob": 0.54, "bet_prob": 0.58, "lean_signals": 4, "bet_signals": 5},
    "mlb_inning": {"lean_edge": 5.0, "bet_edge": 10.0, "lean_prob": 0.55, "bet_prob": 0.60, "lean_signals": 4, "bet_signals": 5},
}

VALIDATION_LEAN_THRESHOLDS = {
    "mlb_new": {"raw_prob": 0.57, "raw_edge": 6.0, "signals": 3},
    "mlb_inning": {"raw_prob": 0.60, "raw_edge": 8.0, "signals": 2},
}

# A validation-fallback LEAN may bypass structural blockers (empty
# walk-forward ledger, assumed pricing) — that's its purpose — but it must
# never publish a pick the system's own calibrated estimate says is -EV at
# the pick's own price. The published validation-lean pool graded 21-21 at
# an implied 54.55% breakeven, and 33 of the 42 decided picks carried a
# negative calibrated edge at publication time.
VALIDATION_LEAN_MIN_PRICE_MARGIN = 0.01


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _bet_type(pick: dict[str, Any], model_key: str) -> str:
    for field in ("market", "market_type", "bet_type"):
        value = str(pick.get(field) or "").strip().lower()
        if value:
            return value
    return MODEL_BET_TYPE_DEFAULTS.get(model_key, "other")


def _game_key(pick: dict[str, Any]) -> str:
    for field in ("game_id", "matchup", "game"):
        value = str(pick.get(field) or "").strip()
        if value:
            return value.lower()
    return ""


def _game_lookup(bucket: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for game in bucket.get("games") or []:
        if not isinstance(game, dict):
            continue
        for field in ("game_id", "matchup", "game"):
            value = str(game.get(field) or "").strip()
            if value:
                lookup[value.lower()] = game
    return lookup


def _line_is_assumed(pick: dict[str, Any]) -> bool:
    pricing = str(pick.get("pricing_type") or "").strip().lower()
    odds_source = str(pick.get("odds_source") or "").strip().lower()
    line_source = str(pick.get("line_source") or "").strip().lower()
    return (
        pricing == "assumed"
        or odds_source == "default_assumed"
        or line_source in {"in_house_projection", "in_house_probability_baseline", "model_generated"}
        or pick.get("market_priced") is False
    )


def _reliable_market_price(pick: dict[str, Any], model_key: str) -> bool:
    if model_key in {"mlb_first_five", "mlb_inning"} and _line_is_assumed(pick):
        return False
    if normalize_probability(pick.get("market_pick_prob")) is not None:
        return True
    if normalize_probability(pick.get("market_probability")) is not None:
        return True
    if normalize_probability(pick.get("market_implied_probability")) is not None:
        return True
    odds = _number(pick.get("odds"))
    if odds is None:
        return False
    if model_key == "mlb_new":
        return True
    return not _line_is_assumed(pick)


def _selected_side_implied_probability(pick: dict[str, Any], model_key: str) -> float | None:
    """Edge baseline for the gate, stamped with its provenance as ``edge_basis``.

    The executable price's implied probability is the EV breakeven; a verified
    no-vig fair probability can only raise the bar (it protects assumed prices
    from minting phantom edge and never loosens a real vigged price).
    """
    fair = no_vig_selected_probability(pick)
    executable = None
    if _reliable_market_price(pick, model_key):
        executable = american_implied_probability(pick.get("odds"))
    if executable is not None and fair is not None and edge_basis_mode() == "no_vig":
        pick["edge_basis"] = "no_vig"
        return max(executable, fair)
    if executable is not None:
        pick["edge_basis"] = "assumed" if _line_is_assumed(pick) else "vigged"
        return executable
    if fair is not None and edge_basis_mode() == "no_vig":
        pick["edge_basis"] = "no_vig"
        return fair
    for field in ("market_pick_prob", "market_probability", "market_implied_probability"):
        probability = normalize_probability(pick.get(field))
        if probability is not None:
            pick["edge_basis"] = "market_field"
            return probability
    return None


def _calibrated_probability(pick: dict[str, Any]) -> float | None:
    for field in ("calibrated_probability", "probability", "model_probability", "predicted_probability"):
        probability = normalize_probability(pick.get(field))
        if probability is not None:
            return probability
    return None


def _raw_probability(pick: dict[str, Any]) -> float | None:
    for field in ("raw_probability", "model_probability", "predicted_probability", "probability"):
        probability = normalize_probability(pick.get(field))
        if probability is not None:
            return probability
    return None


def _calibrated_edge(pick: dict[str, Any], implied: float | None) -> float | None:
    # Recompute from the calibrated probability against the gate's own
    # baseline; a stored pick["edge"] may predate the latest price capture
    # and is only trusted when no probability is available.
    probability = _calibrated_probability(pick)
    if probability is not None and implied is not None:
        return (probability - implied) * 100.0
    edge = _number(pick.get("edge"))
    if edge is not None:
        return edge
    return None


def _raw_edge(pick: dict[str, Any]) -> float | None:
    for field in ("raw_edge", "edge_pp", "model_edge", "edge"):
        edge = _number(pick.get(field))
        if edge is not None:
            return edge
    return None


BRIER_SANITY_CEILING = 0.26
MIN_BRIER_SAMPLES = 15


def _record_clv(record: dict[str, Any]) -> float | None:
    """CLV vs the journaled near-closing anchor price, when one exists."""
    date_iso = str(record.get("date") or "").strip()
    odds = _number(record.get("odds"))
    if not date_iso or odds is None or odds == 0 or -100.0 < odds < 100.0:
        return None
    try:
        from scripts.build_profit_desk import (
            _closing_ledger_for_date,
            american_to_decimal,
            canonical_market_identity,
        )
    except Exception:
        return None
    ledger = _closing_ledger_cache_get(date_iso, _closing_ledger_for_date)
    if not ledger:
        return None
    identity = canonical_market_identity(
        record, mode="team", sport=str(record.get("sport") or ""), date_iso=date_iso
    )
    row = ledger.get(identity)
    if row is None:
        return None
    entry_decimal = american_to_decimal(odds)
    closing_decimal = _number(row.get("decimalOdds"))
    if not entry_decimal or not closing_decimal:
        return None
    return entry_decimal / closing_decimal - 1.0


_CLOSING_LEDGER_CACHE: dict[str, dict[str, Any]] = {}


def _closing_ledger_cache_get(date_iso: str, loader) -> dict[str, Any]:
    if date_iso not in _CLOSING_LEDGER_CACHE:
        try:
            _CLOSING_LEDGER_CACHE[date_iso] = loader(date_iso)
        except Exception:
            _CLOSING_LEDGER_CACHE[date_iso] = {}
    return _CLOSING_LEDGER_CACHE[date_iso]


def _walk_forward_performance(ledger_path: Path = OUTCOME_LEDGER_PATH) -> dict[tuple[str, str], dict[str, Any]]:
    ledger = _read_json(ledger_path) or {}
    records = ledger.get("records") if isinstance(ledger.get("records"), list) else []
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        model_key = str(record.get("model_key") or "").strip()
        if model_key not in MLB_TEAM_MODEL_KEYS or record.get("result") not in {"win", "loss"}:
            continue
        bet_type = str(record.get("bet_type") or MODEL_BET_TYPE_DEFAULTS.get(model_key, "other")).strip().lower()
        key = (model_key, bet_type)
        group = groups.setdefault(
            key,
            {
                "samples": 0, "wins": 0, "losses": 0, "profit": 0.0, "stake": 0.0,
                "brier_sum": 0.0, "brier_n": 0, "clv_sum": 0.0, "clv_n": 0,
            },
        )
        group["samples"] += 1
        won = record.get("result") == "win"
        if won:
            group["wins"] += 1
        else:
            group["losses"] += 1
        group["profit"] += float(record.get("profit") or 0.0)
        group["stake"] += abs(
            float(record.get("stake_units") or record.get("units") or record.get("raw_units") or 1.0)
        )
        probability = normalize_probability(
            record.get("calibrated_probability") if record.get("calibrated_probability") is not None
            else record.get("probability")
        )
        if probability is not None:
            group["brier_sum"] += (probability - (1.0 if won else 0.0)) ** 2
            group["brier_n"] += 1
        clv = _record_clv(record)
        if clv is not None:
            group["clv_sum"] += clv
            group["clv_n"] += 1
    for group in groups.values():
        samples = int(group["samples"] or 0)
        stake = float(group["stake"] or 0.0)
        group["win_rate"] = (float(group["wins"]) / samples) if samples else None
        group["roi"] = (float(group["profit"]) / stake) if stake else None
        group["brier"] = (group["brier_sum"] / group["brier_n"]) if group["brier_n"] else None
        group["avg_clv"] = (group["clv_sum"] / group["clv_n"]) if group["clv_n"] else None
        # Profitability OR consistently beating the close qualifies a group;
        # a Brier score worse than the sanity ceiling disqualifies it because
        # its probabilities cannot be trusted regardless of a lucky record.
        # (Calibration research: well-calibrated probabilities, not raw
        # accuracy, are what separate +ROI from -ROI model selection.)
        profitable = float(group.get("profit") or 0.0) > 0
        beats_close = group["avg_clv"] is not None and group["clv_n"] >= 10 and group["avg_clv"] > 0
        brier_sane = (
            group["brier"] is None
            or group["brier_n"] < MIN_BRIER_SAMPLES
            or group["brier"] <= BRIER_SANITY_CEILING
        )
        group["qualified"] = (
            samples >= MIN_WALK_FORWARD_SAMPLES and (profitable or beats_close) and brier_sane
        )
    return groups


def _performance_for(
    performance: dict[tuple[str, str], dict[str, Any]],
    model_key: str,
    bet_type: str,
) -> dict[str, Any]:
    exact = performance.get((model_key, bet_type))
    if exact:
        return exact
    return performance.get((model_key, MODEL_BET_TYPE_DEFAULTS.get(model_key, "other")), {"samples": 0, "qualified": False})


def _matching_game(pick: dict[str, Any], lookup: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = _game_key(pick)
    return lookup.get(key, {}) if key else {}


def _add_signal(
    signals: list[dict[str, Any]],
    name: str,
    detail: str,
    strength: float = 1.0,
    *,
    category: str = "general",
    impact: str = "support",
) -> None:
    signals.append({
        "name": name,
        "detail": detail,
        "strength": round(float(strength), 3),
        "category": category,
        "impact": impact,
    })


def _add_missing_signal(signals: list[dict[str, Any]], name: str, detail: str, category: str) -> None:
    _add_signal(signals, name, detail, 0.0, category=category, impact="missing")


def _add_risk_signal(
    signals: list[dict[str, Any]],
    name: str,
    detail: str,
    strength: float = -1.0,
    *,
    category: str,
) -> None:
    _add_signal(signals, name, detail, strength, category=category, impact="risk")


def _supporting_signal_names(signals: list[dict[str, Any]]) -> set[str]:
    return {
        str(signal.get("name") or "")
        for signal in signals
        if str(signal.get("name") or "")
        and _number(signal.get("strength")) is not None
        and float(signal.get("strength") or 0.0) > 0
        and str(signal.get("impact") or "support") == "support"
    }


def _factor_categories(signals: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    categories: dict[str, dict[str, list[str]]] = {}
    for signal in signals:
        category = str(signal.get("category") or "general")
        name = str(signal.get("name") or "").strip()
        if not name:
            continue
        impact = str(signal.get("impact") or "support")
        bucket = categories.setdefault(category, {"support": [], "risk": [], "missing": []})
        if impact == "risk":
            target = "risk"
        elif impact == "missing":
            target = "missing"
        else:
            target = "support"
        if name not in bucket[target]:
            bucket[target].append(name)
    return categories


def _travel_context_for(game: dict[str, Any], side_prefix: str) -> dict[str, Any]:
    travel = game.get("travel") if isinstance(game.get("travel"), dict) else {}
    context = travel.get(side_prefix) if side_prefix and isinstance(travel.get(side_prefix), dict) else {}
    if context:
        return context
    features = game.get("features") if isinstance(game.get("features"), dict) else {}
    nested = features.get("travel") if isinstance(features.get("travel"), dict) else {}
    return nested.get(side_prefix) if side_prefix and isinstance(nested.get(side_prefix), dict) else {}


def _add_travel_signals(signals: list[dict[str, Any]], context: dict[str, Any], label: str = "selected side") -> None:
    if not isinstance(context, dict) or not context:
        _add_missing_signal(signals, "travel_context_missing", "travel/rest geography not available", "travel_rest")
        return
    if context.get("available") is False:
        _add_missing_signal(
            signals,
            "travel_context_missing",
            str(context.get("reason") or "travel/rest geography not available"),
            "travel_rest",
        )
        return

    fatigue = _number(context.get("travel_fatigue_index")) or 0.0
    distance = _number(context.get("distance_miles"))
    tz_shift = _number(context.get("timezone_shift_hours"))
    days_since = context.get("days_since_previous_game")
    detail = f"{label}: {context.get('label') or 'travel context available'}"
    _add_signal(signals, "travel_rest_context", detail, 1.0, category="travel_rest")
    if distance is not None and distance >= 1200:
        _add_signal(signals, "long_distance_travel", f"{label}: {round(distance)} miles since previous game", 0.7, category="travel_rest")
    if tz_shift is not None and tz_shift >= 2:
        _add_risk_signal(signals, "eastward_timezone_risk", f"{label}: {int(tz_shift)} hour eastward shift", -0.7, category="travel_rest")
    if days_since is not None and _number(days_since) is not None and float(days_since) <= 1:
        _add_signal(signals, "short_rest_schedule", f"{label}: {int(float(days_since))} day(s) since previous game", 0.6, category="travel_rest")
    if fatigue >= 0.45:
        _add_risk_signal(signals, "travel_fatigue_risk", f"{label}: fatigue index {fatigue:.2f}", -fatigue, category="travel_rest")


def _add_bullpen_signals(signals: list[dict[str, Any]], pitcher: dict[str, Any], label: str) -> None:
    bullpen = pitcher.get("team_bullpen") if isinstance(pitcher.get("team_bullpen"), dict) else {}
    if not bullpen:
        _add_missing_signal(signals, "bullpen_workload_missing", f"{label}: bullpen workload not available", "bullpen")
        return
    games_inspected = int(_number(bullpen.get("games_inspected")) or 0)
    fatigue = _number(bullpen.get("fatigue_index")) or 0.0
    unavailable = bullpen.get("unavailable_today")
    unavailable_count = len(unavailable) if isinstance(unavailable, list) else int(_number(bullpen.get("unavailable_today_count")) or 0)
    if games_inspected > 0:
        _add_signal(signals, "bullpen_workload", f"{label}: bullpen workload checked over {games_inspected} game(s)", 1.0, category="bullpen")
    else:
        _add_missing_signal(signals, "bullpen_workload_missing", f"{label}: no recent bullpen games inspected", "bullpen")
    if fatigue > 0:
        detail = f"{label}: fatigue index {fatigue:.2f}, unavailable arms {unavailable_count}"
        if fatigue >= 0.35:
            _add_risk_signal(signals, "bullpen_fatigue_risk", detail, -fatigue, category="bullpen")
        else:
            _add_signal(signals, "bullpen_freshness", detail, 0.5, category="bullpen")
    else:
        _add_signal(signals, "bullpen_freshness", f"{label}: no bullpen fatigue detected", 0.6, category="bullpen")


def _f5_signals(pick: dict[str, Any], game: dict[str, Any], signals: list[dict[str, Any]]) -> None:
    features = game.get("features") if isinstance(game.get("features"), dict) else {}
    market = str(pick.get("market") or "").lower()
    side_team = str(pick.get("team") or "").strip()
    away_team = str(pick.get("away_team") or game.get("away_team") or "").strip()
    home_team = str(pick.get("home_team") or game.get("home_team") or "").strip()
    side_prefix = "away" if side_team and side_team == away_team else "home" if side_team and side_team == home_team else ""
    venue = features.get("venue") if isinstance(features.get("venue"), dict) else {}

    prefixes = [side_prefix] if side_prefix else ["away", "home"]
    for prefix in prefixes:
        offense = features.get(f"{prefix}_offense") if isinstance(features.get(f"{prefix}_offense"), dict) else {}
        lineup = features.get(f"{prefix}_lineup_matchup") if isinstance(features.get(f"{prefix}_lineup_matchup"), dict) else {}
        pitcher_key = "home_pitcher" if prefix == "away" else "away_pitcher"
        pitcher = features.get(pitcher_key) if isinstance(features.get(pitcher_key), dict) else {}
        label = "away offense" if prefix == "away" else "home offense"

        if pitcher and int(pitcher.get("current_starts") or 0) >= 3:
            _add_signal(signals, "starting_pitcher", f"{label}: opposing starter sample available", category="starting_pitcher")
        else:
            _add_missing_signal(signals, "starter_sample_missing", f"{label}: thin or missing starter sample", "starting_pitcher")
        if pitcher and int(pitcher.get("recent_starts") or 0) >= 3:
            _add_signal(signals, "starter_recent_form", f"{label}: recent starter form included", 0.8, category="starting_pitcher")
        if pitcher and (int(pitcher.get("current_vs_opponent_starts") or 0) + int(pitcher.get("prior_vs_opponent_starts") or 0)) > 0:
            _add_signal(signals, "starter_vs_opponent_history", f"{label}: starter matchup history included", 0.7, category="starting_pitcher")
        if pitcher and int(pitcher.get("venue_starts") or 0) > 0:
            _add_signal(signals, "starter_venue_history", f"{label}: starter venue history included", 0.6, category="starting_pitcher")
        if pitcher:
            _add_bullpen_signals(signals, pitcher, label)

        if lineup and int(lineup.get("sampled_batters") or 0) >= 7:
            _add_signal(signals, "lineup_offense", f"{label}: lineup matchup covers expected hitters", category="lineup_matchup")
        else:
            _add_missing_signal(signals, "lineup_depth_missing", f"{label}: expected lineup depth missing", "lineup_matchup")
        bvp_pa = int(lineup.get("current_bvp_pa") or 0) + int(lineup.get("older_bvp_pa") or 0) if lineup else 0
        if bvp_pa >= 10:
            _add_signal(signals, "batter_pitcher_history", f"{label}: {bvp_pa} batter-vs-pitcher PA included", 0.7, category="lineup_matchup")
        if lineup and lineup.get("threat_score") is not None:
            _add_signal(signals, "lineup_threat_quality", f"{label}: lineup threat score included", 0.6, category="lineup_matchup")

        if offense and (offense.get("team_current_f5_runs") is not None or offense.get("team_recent_f5_runs") is not None):
            _add_signal(signals, "team_offense_form", f"{label}: current/recent F5 offense included", 0.8, category="team_form")
        if offense and offense.get("team_venue_f5_runs") is not None:
            _add_signal(signals, "team_venue_form", f"{label}: venue-specific team offense included", 0.5, category="team_form")
        if offense and (offense.get("pitcher_rest_days") is not None or offense.get("pitcher_rest_label")):
            _add_signal(signals, "travel_rest_schedule", f"{label}: starter rest context present", category="travel_rest")

        travel_context = _travel_context_for(game, prefix)
        if not travel_context and offense:
            travel_context = {
                "available": offense.get("travel_label") not in {None, ""},
                "label": offense.get("travel_label"),
                "travel_fatigue_index": offense.get("travel_fatigue_index"),
                "distance_miles": offense.get("travel_distance_miles"),
                "timezone_shift_hours": offense.get("travel_timezone_shift_hours"),
                "days_since_previous_game": offense.get("travel_days_since_previous_game"),
            }
        _add_travel_signals(signals, travel_context, label)

    if venue and (int(venue.get("games") or 0) >= 20 or venue.get("park_blend") or venue.get("park_factor") is not None):
        _add_signal(signals, "park_weather", "park run-environment context present", category="park_weather")
        _add_signal(signals, "park_factor", "park factor or learned venue run delta included", 0.8, category="park_weather")
    else:
        _add_missing_signal(signals, "park_factor_missing", "park factor context not available", "park_weather")
    if venue and (venue.get("wind_mph") is not None or venue.get("weather_raw")):
        _add_signal(signals, "wind_weather", "wind/weather flight context included", 0.8, category="park_weather")
    else:
        _add_missing_signal(signals, "wind_weather_missing", "wind/weather context not available", "park_weather")
    if market == "f5_total" and _number(pick.get("line")) is not None and _number(pick.get("edge")) is not None:
        _add_signal(signals, "run_environment_gap", "projected F5 total differs from market line", category="model_edge")
    if market == "f5_side" and _number(pick.get("edge")) is not None:
        _add_signal(signals, "projected_margin_gap", "projected F5 side edge is available", 0.7, category="model_edge")


def _inning_signals(pick: dict[str, Any], game: dict[str, Any], signals: list[dict[str, Any]]) -> None:
    inning = int(_number(pick.get("inning")) or 0)
    edge_pp = _number(pick.get("edge_pp")) or _number(pick.get("raw_edge")) or 0.0
    if edge_pp >= 3.0:
        _add_signal(signals, "inning_baseline_edge", "scoreless probability beats inning baseline", category="inning_baseline")
    if inning and inning <= 6 and game.get("home_pitcher") and game.get("away_pitcher"):
        _add_signal(signals, "starting_pitcher", "starter inning profile is applicable", category="starting_pitcher")
        for side in ("home", "away"):
            pitcher = game.get(f"{side}_pitcher_context") if isinstance(game.get(f"{side}_pitcher_context"), dict) else {}
            if pitcher:
                _add_signal(signals, "starter_run_prevention", f"{side}: ERA/WHIP run-prevention context included", 0.7, category="starting_pitcher")
            else:
                _add_missing_signal(signals, "starter_context_missing", f"{side}: starter context not available", "starting_pitcher")
    if inning >= 7:
        _add_signal(signals, "bullpen_condition", "late inning depends on bullpen workload/fatigue model", category="bullpen")
        for side in ("home", "away"):
            pitcher = game.get(f"{side}_pitcher_context") if isinstance(game.get(f"{side}_pitcher_context"), dict) else {}
            _add_bullpen_signals(signals, pitcher, side)
    if game.get("venue_factor") is not None:
        _add_signal(signals, "park_weather", "venue factor included in inning projection", category="park_weather")
        _add_signal(signals, "park_factor", "park run factor included", 0.8, category="park_weather")
    venue = game.get("venue") if isinstance(game.get("venue"), dict) else {}
    weather = game.get("weather") if isinstance(game.get("weather"), dict) else {}
    if venue and venue.get("name"):
        _add_signal(signals, "venue_context", f"venue context included: {venue.get('name')}", 0.5, category="park_weather")
    if weather and (weather.get("wind") or weather.get("temp") or weather.get("condition")):
        _add_signal(signals, "wind_weather", "weather context included for inning run environment", 0.6, category="park_weather")
    if isinstance(game.get("full_inning_table"), dict) and len(game.get("full_inning_table") or {}) >= 6:
        _add_signal(signals, "matchup_structure", "full inning table available for matchup shape", category="inning_baseline")
    for side in ("away", "home"):
        _add_travel_signals(signals, _travel_context_for(game, side), f"{side} team")


def _base_signals(
    pick: dict[str, Any],
    model_key: str,
    bucket: dict[str, Any],
    game: dict[str, Any],
    probability: float | None,
    edge: float | None,
    implied: float | None,
    performance: dict[str, Any],
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    if implied is not None and edge is not None and edge > 0:
        _add_signal(signals, "market_price", "calibrated probability beats selected-side market price", edge / 10.0, category="market_price")
    calibration = pick.get("calibration") if isinstance(pick.get("calibration"), dict) else {}
    if calibration.get("applied") is True and int(calibration.get("samples") or 0) >= 30:
        _add_signal(signals, "probability_calibration", "active calibration group has enough samples", category="calibration")
    else:
        _add_missing_signal(signals, "probability_calibration_missing", "active calibration bucket missing or thin", "calibration")
    if performance.get("qualified") is True:
        _add_signal(signals, "walk_forward_validation", "model family has positive decided history in the gated era input", category="walk_forward")
    if model_key == "mlb_new":
        artifact_status = bucket.get("artifact_status") if isinstance(bucket.get("artifact_status"), dict) else {}
        if artifact_status.get("ready") is True or str(bucket.get("model_stack") or "").lower() == "v2":
            _add_signal(signals, "model_stack_ready", "MLB full-game v2 artifacts are ready", category="model_stack")
        else:
            _add_missing_signal(signals, "model_stack_not_ready", "MLB full-game v2 artifact readiness not confirmed", "model_stack")
        if probability is not None and probability >= 0.55:
            _add_signal(signals, "team_strength", "model probability has meaningful separation from coin-flip", category="team_form")
        for field in ("away_team", "home_team", "team"):
            if pick.get(field):
                _add_signal(signals, "team_identity_context", "selected MLB team context present", 0.4, category="team_form")
                break
    elif model_key == "mlb_first_five":
        _f5_signals(pick, game, signals)
    elif model_key == "mlb_inning":
        _inning_signals(pick, game, signals)
    return signals


def evaluate_mlb_team_pick(
    pick: dict[str, Any],
    model_key: str,
    bucket: dict[str, Any] | None = None,
    *,
    performance: dict[tuple[str, str], dict[str, Any]] | None = None,
    game_lookup: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    bucket = bucket or {}
    performance = performance or _walk_forward_performance()
    lookup = game_lookup if game_lookup is not None else _game_lookup(bucket)
    bet_type = _bet_type(pick, model_key)
    thresholds = PUBLICATION_THRESHOLDS.get(model_key, PUBLICATION_THRESHOLDS["mlb_new"])
    implied = _selected_side_implied_probability(pick, model_key)
    probability = _calibrated_probability(pick)
    raw_probability = _raw_probability(pick)
    edge = _calibrated_edge(pick, implied)
    raw_decision = str((pick.get("pregame_snapshot") or {}).get("decision") if isinstance(pick.get("pregame_snapshot"), dict) else pick.get("decision") or "").upper()
    family_performance = _performance_for(performance, model_key, bet_type)
    game = _matching_game(pick, lookup)
    signals = _base_signals(pick, model_key, bucket, game, probability, edge, implied, family_performance)

    hard_blockers: list[str] = []
    if raw_decision == "PASS":
        hard_blockers.append("raw_model_abstained")
    if probability is None:
        hard_blockers.append("missing_calibrated_probability")
    if not _reliable_market_price(pick, model_key):
        hard_blockers.append("missing_reliable_market_price")
    if implied is None:
        hard_blockers.append("missing_selected_side_implied_probability")
    if edge is None or edge <= 0:
        hard_blockers.append("non_positive_calibrated_edge")
    calibration = pick.get("calibration") if isinstance(pick.get("calibration"), dict) else {}
    if calibration and int(calibration.get("samples") or 0) < 30:
        hard_blockers.append("insufficient_calibration_samples")
    if not family_performance.get("qualified"):
        hard_blockers.append("failed_walk_forward_validation")
    if model_key == "mlb_new" and bucket.get("warnings"):
        hard_blockers.append("model_artifact_warning")
    if model_key in {"mlb_first_five", "mlb_inning"} and _line_is_assumed(pick):
        hard_blockers.append("unsupported_assumed_price")

    signal_count = len(_supporting_signal_names(signals))
    decision = "PASS"
    actionability = "research_signal"
    if not hard_blockers and probability is not None and edge is not None:
        if (
            edge >= thresholds["bet_edge"]
            and probability >= thresholds["bet_prob"]
            and signal_count >= thresholds["bet_signals"]
        ):
            decision = "BET"
            actionability = "bettable"
        elif (
            edge >= thresholds["lean_edge"]
            and probability >= thresholds["lean_prob"]
            and signal_count >= thresholds["lean_signals"]
        ):
            decision = "LEAN"
            actionability = "lean"

    rejection_reason = None
    if decision == "PASS":
        reasons = hard_blockers or [
            f"edge_signal_threshold_not_met(edge={edge}, probability={probability}, signals={signal_count})"
        ]
        rejection_reason = "; ".join(str(reason) for reason in reasons)

    consensus_score = 0.0
    if probability is not None and edge is not None:
        consensus_score = round(max(0.0, edge) + max(0.0, probability - 0.5) * 100.0 + signal_count * 2.5, 3)

    return {
        "model_key": model_key,
        "bet_type": bet_type,
        "decision": decision,
        "actionability": actionability,
        "consensus_passed": decision in {"BET", "LEAN"},
        "consensus_score": consensus_score,
        "consensus_rejection_reason": rejection_reason,
        "hard_blockers": hard_blockers,
        "signals": signals,
        "signal_count": signal_count,
        "factor_categories": _factor_categories(signals),
        "market_no_vig_probability": implied,
        "selected_side_implied_probability": implied,
        "raw_model_probability": raw_probability,
        "calibrated_model_probability": probability,
        "calibrated_edge": edge,
        "walk_forward": family_performance,
    }


def _stake_units(pick: dict[str, Any], decision: str) -> float:
    if decision == "PASS":
        return 0.0
    raw_units = _number(pick.get("raw_units"))
    if raw_units is None:
        raw_units = _number((pick.get("pregame_snapshot") or {}).get("units")) if isinstance(pick.get("pregame_snapshot"), dict) else None
    if raw_units is None:
        raw_units = _number(pick.get("units"))
    if raw_units is None or raw_units <= 0:
        return 0.25 if decision == "LEAN" else 0.5
    return round(min(1.5, raw_units if decision == "BET" else raw_units * 0.6), 2)


def _kelly_stake_fields(pick: dict[str, Any], result: dict[str, Any], decision: str) -> dict[str, Any]:
    """Quarter-Kelly stake suggestion at the actual price.

    Display-only guidance on a 100u bankroll: the flat-record staking in
    ``units`` and the Profit Desk ledgers are unchanged.
    """
    probability = normalize_probability(result.get("calibrated_model_probability"))
    odds = _number(pick.get("odds"))
    if probability is None or odds is None or odds == 0 or -100.0 < odds < 100.0:
        return {}
    full, quarter = kelly(probability, int(round(odds)))
    recommended = 0.0 if decision == "PASS" else round(min(2.0, quarter * 100.0), 2)
    return {
        "full_kelly": full,
        "quarter_kelly": quarter,
        "recommended_units": recommended,
    }


def _raw_snapshot_decision(pick: dict[str, Any]) -> str:
    snapshot = pick.get("pregame_snapshot") if isinstance(pick.get("pregame_snapshot"), dict) else {}
    return str(snapshot.get("decision") or pick.get("decision") or "").strip().upper()


def _has_publishable_price(pick: dict[str, Any]) -> bool:
    return (
        _number(pick.get("odds")) is not None
        or _number(pick.get("assumed_odds")) is not None
        or normalize_probability(pick.get("market_implied_probability")) is not None
        or normalize_probability(pick.get("market_pick_prob")) is not None
        or normalize_probability(pick.get("market_probability")) is not None
    )


def _own_price_implied_probability(pick: dict[str, Any]) -> float | None:
    """Breakeven probability at the price the pick will actually be graded
    against — the assumed price counts here, unlike the strict lane's
    reliable-market check, because that is the ledger's settlement price."""
    implied = normalize_probability(pick.get("market_implied_probability"))
    if implied is not None:
        return implied
    return american_implied_probability(pick.get("odds")) or american_implied_probability(
        pick.get("assumed_odds")
    )


def _validation_candidate_score(
    pick: dict[str, Any],
    result: dict[str, Any],
    model_key: str,
) -> float | None:
    if model_key not in VALIDATION_LEAN_MODELS:
        return None
    if str(result.get("decision") or "").upper() != "PASS":
        return None
    if _raw_snapshot_decision(pick) not in {"BET", "LEAN"}:
        return None
    if not _has_publishable_price(pick):
        return None

    thresholds = VALIDATION_LEAN_THRESHOLDS.get(model_key, {})
    raw_probability = result.get("raw_model_probability")
    raw_probability_f = float(raw_probability) if isinstance(raw_probability, (int, float)) else None
    raw_edge = _raw_edge(pick)
    signal_count = int(result.get("signal_count") or 0)
    if raw_probability_f is None or raw_edge is None:
        return None
    if raw_probability_f < float(thresholds.get("raw_prob") or 0.0):
        return None
    if raw_edge < float(thresholds.get("raw_edge") or 0.0):
        return None
    if signal_count < int(thresholds.get("signals") or 0):
        return None

    # Calibrated-EV gate: the fallback exists to accumulate walk-forward
    # samples, but raw thresholds alone anti-selected — they promoted picks
    # whose calibrated edge against their own settlement price was already
    # negative. Raw model enthusiasm never overrides a knowably -EV price:
    # the calibrated edge must be positive AND the calibrated probability
    # must clear the pick's own price with margin.
    calibrated_edge = _number(result.get("calibrated_edge"))
    if calibrated_edge is None or calibrated_edge <= 0:
        return None
    calibrated_probability = _number(result.get("calibrated_model_probability"))
    price_implied = _own_price_implied_probability(pick)
    if calibrated_probability is None or price_implied is None:
        return None
    if calibrated_probability < price_implied + VALIDATION_LEAN_MIN_PRICE_MARGIN:
        return None

    return round(
        max(0.0, raw_edge)
        + max(0.0, raw_probability_f - 0.5) * 100.0
        + signal_count * 2.5,
        3,
    )


def _promote_validation_leans(
    model_key: str,
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]],
) -> int:
    if model_key not in VALIDATION_LEAN_MODELS:
        return 0
    if any(str(pick.get("decision") or "").upper() in {"BET", "LEAN"} for pick, _ in evaluated):
        return 0

    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for pick, result in evaluated:
        score = _validation_candidate_score(pick, result, model_key)
        if score is not None:
            candidates.append((score, pick, result))
    candidates.sort(key=lambda item: item[0], reverse=True)

    promoted = 0
    seen_games: set[str] = set()
    for score, pick, result in candidates:
        game_key = _game_key(pick) or str(pick.get("matchup") or pick.get("game") or pick.get("pick") or "")
        if game_key in seen_games:
            continue
        seen_games.add(game_key)
        strict_reason = str(result.get("consensus_rejection_reason") or pick.get("consensus_rejection_reason") or "")
        pick.update({
            "decision": "LEAN",
            "units": 0.25,
            "actionability": "validation_lean",
            "consensus_passed": True,
            "consensus_qualified": False,
            "primary_consensus_passed": False,
            "consensus_publication_mode": "validation_fallback",
            "validation_lean": True,
            "validation_score": score,
            "validation_reason": (
                f"validation fallback: {model_key} had zero strict BET/LEAN publications; "
                "published top raw model signal as a small LEAN for tracking"
            ),
            "consensus_rejection_reason": strict_reason,
            "consensus_hard_blockers": result.get("hard_blockers") or [],
        })
        promoted += 1
        if promoted >= VALIDATION_LEAN_LIMIT:
            break
    return promoted


def apply_mlb_team_consensus_to_payload(
    payload: dict[str, Any],
    *,
    performance: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    models = payload.get("models")
    if not isinstance(models, dict):
        return payload
    performance = performance or _walk_forward_performance()
    for model_key, bucket in models.items():
        if model_key not in MLB_TEAM_MODEL_KEYS or not isinstance(bucket, dict):
            continue
        lookup = _game_lookup(bucket)
        bucket["consensus_required"] = True
        bucket["consensus_gate_version"] = MLB_TEAM_CONSENSUS_VERSION
        bucket["ranking_epoch"] = f"{MLB_TEAM_RANKING_EPOCH_PREFIX}:{model_key}"
        picks = bucket.get("picks") if isinstance(bucket.get("picks"), list) else []
        evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            result = evaluate_mlb_team_pick(
                pick,
                str(model_key),
                bucket,
                performance=performance,
                game_lookup=lookup,
            )
            evaluated.append((pick, result))
            decision = result["decision"]
            pick.update({
                "consensus_required": True,
                "consensus_gate_version": MLB_TEAM_CONSENSUS_VERSION,
                "mlb_team_consensus_version": MLB_TEAM_CONSENSUS_VERSION,
                "consensus_passed": result["consensus_passed"],
                "consensus_qualified": result["consensus_passed"],
                "primary_consensus_passed": result["consensus_passed"],
                "consensus_publication_mode": "strict" if result["consensus_passed"] else "research",
                "consensus_score": result["consensus_score"],
                "consensus_rejection_reason": result["consensus_rejection_reason"],
                "consensus_signal_count": result["signal_count"],
                "consensus_signals": result["signals"],
                "consensus_factor_categories": result["factor_categories"],
                "consensus_hard_blockers": result["hard_blockers"],
                "market_no_vig_probability": result["market_no_vig_probability"],
                "selected_side_implied_probability": result["selected_side_implied_probability"],
                "raw_model_probability": result["raw_model_probability"],
                "calibrated_model_probability": result["calibrated_model_probability"],
                "calibrated_edge": result["calibrated_edge"],
                "walk_forward_samples": int((result["walk_forward"] or {}).get("samples") or 0),
                "walk_forward_roi": (result["walk_forward"] or {}).get("roi"),
                "actionability": result["actionability"],
                "decision": decision,
                "units": _stake_units(pick, decision),
                **_kelly_stake_fields(pick, result, decision),
                "ml_rank_epoch": f"{MLB_TEAM_RANKING_EPOCH_PREFIX}:{model_key}",
                "ranking_epoch": f"{MLB_TEAM_RANKING_EPOCH_PREFIX}:{model_key}",
                "model_epoch": f"{MLB_TEAM_RANKING_EPOCH_PREFIX}:{model_key}",
            })
        promoted = _promote_validation_leans(str(model_key), evaluated)
        if promoted:
            bucket["validation_leans_published"] = promoted
            bucket["validation_publication_mode"] = "fallback_when_strict_gate_empty"
        summary: dict[str, int] = {}
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            reason = str(pick.get("consensus_rejection_reason") or "")
            if reason:
                first = reason.split(";", 1)[0]
                summary[first] = summary.get(first, 0) + 1
        bucket["consensus_rejection_reasons"] = summary
    for alias in ("mlb_new", "mlb_first_five", "mlb_inning"):
        if alias in models:
            payload[alias] = models[alias]
    return payload
