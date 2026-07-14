from __future__ import annotations

from typing import Any

try:
    from mlb_inning_history import MLB_AVG_SCORELESS
    from mlb_inning_fetcher import DEFAULT_PITCHER, safe_float
    from mlb_inning_bullpen import compute_fatigue_shift
except ImportError:
    from .mlb_inning_history import MLB_AVG_SCORELESS
    from .mlb_inning_fetcher import DEFAULT_PITCHER, safe_float
    from .mlb_inning_bullpen import compute_fatigue_shift


THREAT_BASELINE = 0.270
THREAT_SPAN = 0.130
THREAT_ADJUSTMENT_LIMIT = 0.15
# Innings 1-8 only. The home half of the 9th is unplayed when the home team
# is ahead, so projecting it as a bettable inning is misleading.
ELIGIBLE_INNINGS = range(1, 9)

# League-average full-inning scoreless rates = product of two half-inning
# averages. Used as the baseline an inning's scoreless probability has to
# beat by EDGE thresholds before it becomes a real LEAN/BET.
MLB_AVG_FULL_SCORELESS = {
    inning: round(MLB_AVG_SCORELESS[inning] * MLB_AVG_SCORELESS[inning], 4)
    for inning in range(1, 10)
}

# Recent grading showed +7pp was not enough separation from baseline: committed
# BET rows were running break-even. Keep +3pp as a research LEAN, but require a
# double-digit edge before calling an inning a BET.
INNING_BET_EDGE = 0.10  # need +10 percentage points over baseline to BET
INNING_LEAN_EDGE = 0.03  # +3 pp over baseline qualifies for LEAN


def compute_inning_probabilities(
    game: dict[str, Any],
    team_histories: dict[str, dict[int, dict[str, float]]],
    matchup_threats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    game_id = str(game.get("game_id") or "")
    away_team = str(game.get("away_team") or "Away Team")
    home_team = str(game.get("home_team") or "Home Team")
    game_threats = (matchup_threats.get(game_id) or {}).get("innings") or {}
    venue_factor = _venue_factor(game)

    full_inning_table: dict[str, float] = {}
    edge_table: dict[str, float] = {}
    for inning in ELIGIBLE_INNINGS:
        inning_threats = game_threats.get(inning) or game_threats.get(str(inning)) or {}
        away_half_scoreless = _half_scoreless_probability(
            historical_scoreless_rate=_history_rate(team_histories, away_team, inning),
            threat_score=safe_float(inning_threats.get("away_threat"), THREAT_BASELINE),
            inning=inning,
            opposing_pitcher=game.get("home_pitcher") or {},
            venue_factor=venue_factor,
        )
        home_half_scoreless = _half_scoreless_probability(
            historical_scoreless_rate=_history_rate(team_histories, home_team, inning),
            threat_score=safe_float(inning_threats.get("home_threat"), THREAT_BASELINE),
            inning=inning,
            opposing_pitcher=game.get("away_pitcher") or {},
            venue_factor=venue_factor,
        )
        full_probability = _clamp(away_half_scoreless * home_half_scoreless, 0.01, 0.98)
        full_inning_table[str(inning)] = round(full_probability, 3)
        edge_table[str(inning)] = round(full_probability - MLB_AVG_FULL_SCORELESS[inning], 4)

    # Rank by edge over baseline, then probability — only emit picks that
    # actually clear the edge gates.
    candidates = sorted(
        ((int(inning), full_inning_table[inning], edge_table[inning]) for inning in full_inning_table),
        key=lambda item: (item[2], item[1]),
        reverse=True,
    )
    top_picks: list[dict[str, Any]] = []
    for inning_num, probability, edge in candidates:
        decision = _decision_for_edge(probability, edge)
        if decision == "PASS":
            continue
        top_picks.append({
            "inning": inning_num,
            "probability_scoreless": probability,
            "baseline": MLB_AVG_FULL_SCORELESS[inning_num],
            "edge_pp": round(edge * 100.0, 2),
            "decision": decision,
            "confidence": _confidence_for_decision(decision, probability, edge),
            "label": f"Inning {inning_num} - No Run Scored ({probability:.1%}, edge {edge*100:+.1f}pp)",
        })
        if len(top_picks) >= 2:
            break

    return {
        "game_id": game_id,
        "game_start_time": str(game.get("game_start_time") or ""),
        "game_order": game.get("game_order", 0),
        "matchup": f"{home_team} vs {away_team}",
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher": (game.get("home_pitcher") or {}).get("name") or "TBD",
        "away_pitcher": (game.get("away_pitcher") or {}).get("name") or "TBD",
        "home_pitcher_context": _pitcher_context(game.get("home_pitcher") or {}),
        "away_pitcher_context": _pitcher_context(game.get("away_pitcher") or {}),
        "travel": game.get("travel") if isinstance(game.get("travel"), dict) else {},
        "weather": game.get("weather") if isinstance(game.get("weather"), dict) else {},
        "venue": {
            "id": game.get("venue_id"),
            "name": game.get("venue_name"),
            "run_factor": round(venue_factor, 3),
        },
        "venue_factor": round(venue_factor, 3),
        "top_2_picks": top_picks,
        "full_inning_table": full_inning_table,
        "edge_table": edge_table,
        "baseline_table": {str(i): MLB_AVG_FULL_SCORELESS[i] for i in ELIGIBLE_INNINGS},
    }


def _half_scoreless_probability(
    historical_scoreless_rate: float,
    threat_score: float,
    inning: int,
    opposing_pitcher: dict[str, Any],
    venue_factor: float = 1.0,
) -> float:
    """Probability the offense fails to score in this half-inning.

    Pre-patch this used only the team's per-inning history × matchup-threat,
    plus a small late-inning ERA bump. Now it also folds in the actual
    starter's per-inning scoreless rate (when known) for innings 1-6, swaps
    in a bullpen baseline for innings 7-9, and applies a park-factor scalar.
    """
    threat_adjustment = _clamp(
        (threat_score - THREAT_BASELINE) / THREAT_SPAN,
        -THREAT_ADJUSTMENT_LIMIT,
        THREAT_ADJUSTMENT_LIMIT,
    )
    probability = historical_scoreless_rate * (1.0 - threat_adjustment)

    # Innings 1-6: blend in the actual starter's per-inning scoreless rate.
    starter_inning_rate = _starter_scoreless_rate_for_inning(opposing_pitcher, inning)
    if inning <= 6 and starter_inning_rate is not None:
        probability = (probability * 0.55) + (starter_inning_rate * 0.45)
    elif inning >= 7:
        # Innings 7-9 are pitched by the bullpen — use the team's bullpen
        # scoreless rate when available, otherwise the league late-inning
        # baseline. The starter's late-inning ERA is no longer relevant.
        bullpen_rate = _bullpen_scoreless_rate(opposing_pitcher, inning)
        probability = (probability * 0.60) + (bullpen_rate * 0.40)

    # Park factor — hitter-friendly parks (>1.0) reduce scoreless prob.
    if venue_factor and venue_factor != 1.0:
        probability /= max(venue_factor, 0.6)

    return _clamp(probability, 0.05, 0.98)


def _starter_scoreless_rate_for_inning(pitcher: dict[str, Any], inning: int) -> float | None:
    """Best-effort per-inning scoreless rate for the listed starter."""
    if not isinstance(pitcher, dict):
        return None
    by_inning = pitcher.get("inning_scoreless_rates") or pitcher.get("scoreless_rate_by_inning")
    if isinstance(by_inning, dict):
        for key in (inning, str(inning)):
            value = by_inning.get(key)
            if value is None:
                continue
            try:
                return _clamp(float(value), 0.05, 0.98)
            except (TypeError, ValueError):
                continue
    # Fallback: derive a flat per-inning rate from ERA. ERA 3.00 -> ~78%
    # half-inning scoreless; ERA 5.50 -> ~64%.
    era = safe_float(pitcher.get("era"), DEFAULT_PITCHER["era"])
    derived = _clamp(0.92 - (era - 3.0) * 0.045, 0.55, 0.86)
    return derived


def _bullpen_scoreless_rate(opposing_pitcher: dict[str, Any], inning: int) -> float:
    """Bullpen scoreless rate fallback for late innings (7-9), with a
    fatigue shrink when the team's top arms are likely unavailable.

    The clean per-inning bullpen rate is computed first (per-inning table,
    flat fallback, or league baseline). Then ``team_bullpen.fatigue_index``
    — produced by ``mlb_inning_bullpen.fetch_bullpen_workload`` — applies a
    downward shift up to 12pp at full fatigue. A manager who burned 4 of
    8 arms in the last two days can't run them today and gets stuck with
    mop-up arms in the 7th-9th, so the team's late-inning scoreless rate
    drops materially.
    """
    bullpen = (opposing_pitcher or {}).get("team_bullpen") or {}
    base_rate = MLB_AVG_SCORELESS[inning]

    if isinstance(bullpen, dict):
        per_inning = bullpen.get("scoreless_rate_by_inning") or {}
        per_inning_value: float | None = None
        for key in (inning, str(inning)):
            value = per_inning.get(key)
            if value is None:
                continue
            try:
                per_inning_value = _clamp(float(value), 0.05, 0.98)
                break
            except (TypeError, ValueError):
                continue

        if per_inning_value is not None:
            base_rate = per_inning_value
        else:
            flat = bullpen.get("scoreless_rate")
            if flat is not None:
                try:
                    base_rate = _clamp(float(flat), 0.05, 0.98)
                except (TypeError, ValueError):
                    pass

        # Fatigue shrink — relievers used yesterday or back-to-back are
        # unavailable today, forcing the manager into worse arms.
        fatigue_shift = compute_fatigue_shift(bullpen.get("fatigue_index"))
        if fatigue_shift > 0:
            base_rate -= fatigue_shift

    return _clamp(base_rate, 0.05, 0.98)


def _venue_factor(game: dict[str, Any]) -> float:
    """Park run-scoring factor; 1.0 = neutral, >1 hitter-friendly, <1 pitcher-friendly."""
    venue = game.get("venue") or {}
    if isinstance(venue, dict):
        for key in ("run_factor", "park_factor", "scoring_factor"):
            value = venue.get(key)
            if value is None:
                continue
            try:
                return _clamp(float(value), 0.80, 1.25)
            except (TypeError, ValueError):
                continue
    direct = game.get("venue_run_factor") or game.get("park_factor")
    if direct is not None:
        try:
            return _clamp(float(direct), 0.80, 1.25)
        except (TypeError, ValueError):
            pass
    return 1.0


def _pitcher_context(pitcher: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(pitcher, dict):
        return {}
    bullpen = pitcher.get("team_bullpen") if isinstance(pitcher.get("team_bullpen"), dict) else {}
    return {
        "name": pitcher.get("name") or "TBD",
        "era": safe_float(pitcher.get("era"), DEFAULT_PITCHER["era"]),
        "whip": safe_float(pitcher.get("whip"), DEFAULT_PITCHER["whip"]),
        "opponent_obp": safe_float(pitcher.get("opponent_obp"), DEFAULT_PITCHER["opponent_obp"]),
        "opponent_slg": safe_float(pitcher.get("opponent_slg"), DEFAULT_PITCHER["opponent_slg"]),
        "team_bullpen": {
            "lookback_games": bullpen.get("lookback_games"),
            "games_inspected": bullpen.get("games_inspected"),
            "fatigue_index": safe_float(bullpen.get("fatigue_index"), 0.0),
            "effective_unavailable_count": safe_float(bullpen.get("effective_unavailable_count"), 0.0),
            "unavailable_today_count": len(bullpen.get("unavailable_today") or []),
            "back_to_back_arms_count": len(bullpen.get("back_to_back_arms") or []),
            "high_leverage_used_count": len(bullpen.get("high_leverage_used_pitcher_ids") or []),
        },
    }


def _history_rate(team_histories: dict[str, dict[int, dict[str, float]]], team_name: str, inning: int) -> float:
    team_history = team_histories.get(team_name) or {}
    inning_stats = team_history.get(inning) or team_history.get(str(inning)) or {}
    return _clamp(safe_float(inning_stats.get("scoreless_rate"), MLB_AVG_SCORELESS[inning]), 0.05, 0.98)


def _decision_for_edge(probability: float, edge: float) -> str:
    """Only label as BET/LEAN when the inning beats the league baseline.

    Pre-patch the model emitted a "High" confidence label any time the
    probability cleared 50% even though the league baseline for early
    innings is already 38-44% — i.e., 50% wasn't actually informative.
    """
    if edge >= INNING_BET_EDGE and probability >= 0.50:
        return "BET"
    if edge >= INNING_LEAN_EDGE and probability >= 0.45:
        return "LEAN"
    return "PASS"


def _confidence_for_decision(decision: str, probability: float, edge: float) -> str:
    decision_upper = str(decision or "").upper()
    if decision_upper == "BET":
        if edge >= 0.10 and probability >= 0.55:
            return "High"
        return "Medium"
    if decision_upper == "LEAN":
        return "Medium"
    return "Low"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
