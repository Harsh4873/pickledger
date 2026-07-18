from __future__ import annotations

from copy import deepcopy

from scripts.mlb_team_consensus import (
    MLB_TEAM_CONSENSUS_VERSION,
    apply_mlb_team_consensus_to_payload,
    evaluate_mlb_team_pick,
)


GOOD_PERFORMANCE = {
    ("mlb_new", "h2h"): {"samples": 80, "wins": 48, "losses": 32, "profit": 9.2, "stake": 80.0, "roi": 0.115, "qualified": True},
    ("mlb_new", "totals"): {"samples": 80, "wins": 45, "losses": 35, "profit": 4.0, "stake": 80.0, "roi": 0.05, "qualified": True},
    ("mlb_first_five", "f5_side"): {"samples": 90, "wins": 55, "losses": 35, "profit": 6.0, "stake": 90.0, "roi": 0.067, "qualified": True},
    ("mlb_first_five", "f5_total"): {"samples": 90, "wins": 55, "losses": 35, "profit": 6.0, "stake": 90.0, "roi": 0.067, "qualified": True},
    ("mlb_inning", "no_run_inning"): {"samples": 90, "wins": 55, "losses": 35, "profit": 6.0, "stake": 90.0, "roi": 0.067, "qualified": True},
}


def _mlb_new_pick(**overrides):
    pick = {
        "source": "MLB Model",
        "sport": "MLB",
        "pick": "Mets ML (Braves vs Mets)",
        "market_type": "h2h",
        "team": "Mets",
        "probability": 0.62,
        "calibrated_probability": 0.62,
        "raw_probability": 0.68,
        "market_pick_prob": 0.52,
        "edge": 10.0,
        "odds": -105,
        "units": 0.8,
        "decision": "BET",
        "calibration": {"applied": True, "samples": 88, "key": "model:mlb_new|bet:h2h"},
        "pregame_snapshot": {"decision": "BET", "units": 0.8, "probability": 0.68},
    }
    pick.update(overrides)
    return pick


def _f5_pick(**overrides):
    pick = {
        "source": "MLB First Five",
        "sport": "MLB",
        "date": "2026-06-25",
        "game_id": "game-1",
        "pick": "Away F5 ML",
        "market": "f5_side",
        "team": "Away",
        "away_team": "Away",
        "home_team": "Home",
        "probability": 0.61,
        "calibrated_probability": 0.61,
        "raw_probability": 0.68,
        "edge": 9.0,
        "odds": -102,
        "market_priced": True,
        "pricing_type": "market",
        "odds_source": "sportsbook",
        "line_source": "sportsbook",
        "units": 0.7,
        "decision": "BET",
        "calibration": {"applied": True, "samples": 92, "key": "model:mlb_first_five|bet:f5_side"},
        "pregame_snapshot": {"decision": "BET", "units": 0.7, "probability": 0.68},
    }
    pick.update(overrides)
    return pick


def _f5_bucket(pick):
    return {
        "ok": True,
        "picks": [pick],
        "games": [{
            "game_id": "game-1",
            "away_team": "Away",
            "home_team": "Home",
            "features": {
                "away_lineup_matchup": {"sampled_batters": 9, "current_bvp_pa": 6, "older_bvp_pa": 8, "threat_score": 0.381},
                "away_offense": {
                    "pitcher_rest_days": 5,
                    "pitcher_rest_label": "normal rest",
                    "team_current_f5_runs": 2.6,
                    "team_recent_f5_runs": 2.9,
                    "team_venue_f5_runs": 2.4,
                    "travel_fatigue_index": 0.0,
                    "travel_label": "same venue",
                },
                "home_pitcher": {
                    "current_starts": 8,
                    "recent_starts": 5,
                    "current_vs_opponent_starts": 1,
                    "venue_starts": 2,
                    "team_bullpen": {"games_inspected": 2, "fatigue_index": 0.08, "unavailable_today": []},
                },
                "travel": {
                    "away": {"available": True, "label": "same venue", "travel_fatigue_index": 0.0, "distance_miles": 0, "timezone_shift_hours": 0, "days_since_previous_game": 1}
                },
                "venue": {"games": 44, "park_blend": {"final_delta": 0.05}, "wind_mph": 9.0},
            },
        }],
    }


def _inning_pick(**overrides):
    pick = {
        "source": "MLB Inning",
        "sport": "MLB",
        "date": "2026-06-25",
        "game_id": "game-2",
        "pick": "Inning 1 - No Run Scored",
        "market": "no_run_inning",
        "inning": 1,
        "probability": 0.62,
        "calibrated_probability": 0.62,
        "raw_probability": 0.70,
        "edge": 11.0,
        "edge_pp": 12.0,
        "odds": -110,
        "market_priced": True,
        "pricing_type": "market",
        "odds_source": "sportsbook",
        "line_source": "sportsbook",
        "units": 0.6,
        "decision": "BET",
        "calibration": {"applied": True, "samples": 120, "key": "model:mlb_inning|bet:no_run_inning"},
        "pregame_snapshot": {"decision": "BET", "units": 0.6, "probability": 0.70},
    }
    pick.update(overrides)
    return pick


def _inning_bucket(pick):
    return {
        "ok": True,
        "picks": [pick],
        "games": [{
            "game_id": "game-2",
            "home_pitcher": "Home SP",
            "away_pitcher": "Away SP",
            "home_pitcher_context": {"team_bullpen": {"games_inspected": 2, "fatigue_index": 0.05, "unavailable_today_count": 0}},
            "away_pitcher_context": {"team_bullpen": {"games_inspected": 2, "fatigue_index": 0.1, "unavailable_today_count": 0}},
            "travel": {
                "home": {"available": True, "label": "same venue", "travel_fatigue_index": 0.0, "distance_miles": 0, "timezone_shift_hours": 0, "days_since_previous_game": 1},
                "away": {"available": True, "label": "900 mi; 1h east", "travel_fatigue_index": 0.28, "distance_miles": 900, "timezone_shift_hours": 1, "days_since_previous_game": 1},
            },
            "weather": {"wind": "8 mph, Out To LF", "temp": "82"},
            "venue_factor": 0.96,
            "full_inning_table": {str(index): 0.55 for index in range(1, 9)},
        }],
    }


def test_mlb_new_can_publish_when_market_calibration_and_validation_agree():
    pick = _mlb_new_pick()
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_new",
        {"artifact_status": {"ready": True}, "model_stack": "v2"},
        performance=GOOD_PERFORMANCE,
    )

    assert result["decision"] == "BET"
    assert result["consensus_passed"] is True
    assert {signal["name"] for signal in result["signals"]} >= {
        "market_price",
        "probability_calibration",
        "walk_forward_validation",
        "model_stack_ready",
    }


def test_missing_market_price_blocks_mlb_new_even_with_high_probability():
    pick = _mlb_new_pick(odds=None, market_pick_prob=None, edge=14.0, probability=0.69)
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_new",
        {"artifact_status": {"ready": True}, "model_stack": "v2"},
        performance=GOOD_PERFORMANCE,
    )

    assert result["decision"] == "PASS"
    assert "missing_reliable_market_price" in result["hard_blockers"]


def test_bad_walk_forward_history_blocks_high_probability_pick():
    pick = _mlb_new_pick(probability=0.72, calibrated_probability=0.72, edge=16.0)
    bad_performance = {("mlb_new", "h2h"): {"samples": 80, "wins": 34, "losses": 46, "profit": -8.0, "stake": 80.0, "roi": -0.1, "qualified": False}}
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_new",
        {"artifact_status": {"ready": True}, "model_stack": "v2"},
        performance=bad_performance,
    )

    assert result["decision"] == "PASS"
    assert "failed_walk_forward_validation" in result["hard_blockers"]


def test_dry_mlb_new_bucket_publishes_top_raw_signal_as_validation_lean():
    pick = _mlb_new_pick(
        pick="Under 8.0 (Dodgers vs Padres)",
        market_type="totals",
        probability=0.5104,
        calibrated_probability=0.5104,
        raw_probability=0.6381,
        edge=-1.34,
        raw_edge=11.43,
        odds=-110,
        assumed_odds=-110,
        pregame_snapshot={"decision": "BET", "units": 0.06, "probability": 0.6381},
    )
    payload = {
        "date": "2026-06-28",
        "models": {
            "mlb_new": {
                "ok": True,
                "artifact_status": {"ready": True},
                "model_stack": "v2",
                "picks": [pick],
            },
        },
    }

    gated = apply_mlb_team_consensus_to_payload(payload, performance=GOOD_PERFORMANCE)
    published = gated["models"]["mlb_new"]["picks"][0]

    assert published["decision"] == "LEAN"
    assert published["actionability"] == "validation_lean"
    assert published["units"] == 0.25
    assert published["consensus_passed"] is True
    assert published["primary_consensus_passed"] is False
    assert published["consensus_publication_mode"] == "validation_fallback"
    assert published["validation_lean"] is True
    assert "non_positive_calibrated_edge" in published["consensus_rejection_reason"]


def test_dry_inning_bucket_publishes_validation_lean_for_raw_inning_edge():
    pick = _inning_pick(
        probability=0.521,
        calibrated_probability=0.521,
        raw_probability=0.643,
        edge=-2.45,
        raw_edge=12.41,
        edge_pp=12.41,
        odds=-120,
        assumed_odds=-120,
        market_implied_probability=0.545455,
        pricing_type="user_assumed",
        odds_source="user_assumed_no_run_inning_-120",
        line_source="user_assumed_no_run_inning_price",
        pregame_snapshot={"decision": "BET", "units": 0.6, "probability": 0.643},
    )
    bad_performance = {
        ("mlb_inning", "no_run_inning"): {
            "samples": 173,
            "wins": 86,
            "losses": 87,
            "profit": 0.0,
            "stake": 173.0,
            "roi": 0.0,
            "qualified": False,
        }
    }
    payload = {
        "date": "2026-06-28",
        "models": {"mlb_inning": {"ok": True, "picks": [pick], "games": []}},
    }

    gated = apply_mlb_team_consensus_to_payload(payload, performance=bad_performance)
    published = gated["models"]["mlb_inning"]["picks"][0]

    assert published["decision"] == "LEAN"
    assert published["actionability"] == "validation_lean"
    assert published["consensus_publication_mode"] == "validation_fallback"
    assert "failed_walk_forward_validation" in published["consensus_rejection_reason"]


def test_first_five_uses_baseball_context_when_real_market_price_exists():
    pick = _f5_pick()
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_first_five",
        _f5_bucket(pick),
        performance=GOOD_PERFORMANCE,
    )

    assert result["decision"] == "BET"
    assert {signal["name"] for signal in result["signals"]} >= {
        "starting_pitcher",
        "starter_recent_form",
        "lineup_offense",
        "batter_pitcher_history",
        "team_offense_form",
        "bullpen_workload",
        "travel_rest_schedule",
        "travel_rest_context",
        "park_weather",
        "wind_weather",
    }
    assert result["factor_categories"]["travel_rest"]["support"]
    assert result["factor_categories"]["bullpen"]["support"]


def test_first_five_assumed_price_stays_research_only():
    pick = _f5_pick(market_priced=False, pricing_type="assumed", odds_source="default_assumed", line_source="in_house_projection")
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_first_five",
        _f5_bucket(pick),
        performance=GOOD_PERFORMANCE,
    )

    assert result["decision"] == "PASS"
    assert "unsupported_assumed_price" in result["hard_blockers"]


def test_first_five_user_assumed_total_ladder_is_evaluable():
    pick = _f5_pick(
        market="f5_total",
        team="",
        pick="Over 4.5 F5",
        line=4.5,
        odds=-130,
        assumed_odds=-130,
        market_implied_probability=0.565217,
        market_priced=True,
        pricing_type="user_assumed",
        odds_source="user_assumed_f5_total_4.5",
        line_source="user_assumed_f5_total_ladder",
        probability=0.64,
        calibrated_probability=0.64,
        edge=7.48,
        pregame_snapshot={"decision": "BET", "units": 0.7, "probability": 0.68},
    )
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_first_five",
        _f5_bucket(pick),
        performance=GOOD_PERFORMANCE,
    )

    assert result["decision"] == "BET"
    assert "unsupported_assumed_price" not in result["hard_blockers"]
    assert "missing_reliable_market_price" not in result["hard_blockers"]


def test_inning_model_uses_inning_baseline_and_context_with_real_market():
    # Edge is derived from the calibrated probability against the executable
    # price, so the stored (stale) edge field cannot decide publication:
    # 0.63 vs -110 implied 0.5238 is a 10.6pp edge, clearing the 10pp BET bar.
    pick = _inning_pick(probability=0.63, calibrated_probability=0.63)
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_inning",
        _inning_bucket(pick),
        performance=GOOD_PERFORMANCE,
    )

    assert result["decision"] == "BET"
    assert pick["edge_basis"] == "vigged"
    assert {signal["name"] for signal in result["signals"]} >= {
        "inning_baseline_edge",
        "starting_pitcher",
        "park_weather",
        "matchup_structure",
        "travel_rest_context",
    }
    assert "inning_baseline" in result["factor_categories"]
    assert "park_weather" in result["factor_categories"]


def test_missing_and_risk_signals_do_not_inflate_support_count():
    pick = _f5_pick(probability=0.61, calibrated_probability=0.61, edge=9.0)
    bucket = _f5_bucket(pick)
    features = bucket["games"][0]["features"]
    features["away_offense"]["travel_fatigue_index"] = 0.65
    features["away_offense"]["travel_label"] = "1d since previous game; 2300 mi; 3h east"
    features["travel"] = {
        "away": {
            "available": True,
            "label": "1d since previous game; 2300 mi; 3h east",
            "travel_fatigue_index": 0.65,
            "distance_miles": 2300,
            "timezone_shift_hours": 3,
            "days_since_previous_game": 1,
        }
    }

    result = evaluate_mlb_team_pick(pick, "mlb_first_five", bucket, performance=GOOD_PERFORMANCE)

    signal_names = {signal["name"] for signal in result["signals"]}
    assert "travel_fatigue_risk" in signal_names
    assert "eastward_timezone_risk" in signal_names
    assert result["factor_categories"]["travel_rest"]["risk"]
    assert result["signal_count"] == len({
        signal["name"]
        for signal in result["signals"]
        if signal.get("impact") == "support" and signal.get("strength", 0) > 0
    })


def test_inning_assumed_price_stays_research_only():
    pick = _inning_pick(market_priced=False, pricing_type="assumed", odds_source="default_assumed", line_source="in_house_probability_baseline")
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_inning",
        _inning_bucket(pick),
        performance=GOOD_PERFORMANCE,
    )

    assert result["decision"] == "PASS"
    assert "unsupported_assumed_price" in result["hard_blockers"]


def test_inning_user_assumed_minus_120_price_is_evaluable():
    pick = _inning_pick(
        odds=-120,
        assumed_odds=-120,
        market_implied_probability=0.545455,
        market_priced=True,
        pricing_type="user_assumed",
        odds_source="user_assumed_no_run_inning_-120",
        line_source="user_assumed_no_run_inning_price",
        probability=0.66,
        calibrated_probability=0.66,
        edge=11.45,
        pregame_snapshot={"decision": "BET", "units": 0.6, "probability": 0.70},
    )
    result = evaluate_mlb_team_pick(
        pick,
        "mlb_inning",
        _inning_bucket(pick),
        performance=GOOD_PERFORMANCE,
    )

    assert result["decision"] == "BET"
    assert "unsupported_assumed_price" not in result["hard_blockers"]
    assert "missing_reliable_market_price" not in result["hard_blockers"]


def test_payload_gate_only_touches_three_mlb_team_models():
    mlb_pick = _mlb_new_pick()
    wnba_pick = {"source": "WNBA Model", "sport": "WNBA", "pick": "Tempo ML", "decision": "BET", "units": 1}
    payload = {
        "date": "2026-06-25",
        "models": {
            "mlb_new": {"ok": True, "artifact_status": {"ready": True}, "picks": [mlb_pick]},
            "wnba": {"ok": True, "picks": [deepcopy(wnba_pick)]},
        },
    }

    gated = apply_mlb_team_consensus_to_payload(payload, performance=GOOD_PERFORMANCE)

    assert gated["models"]["mlb_new"]["consensus_gate_version"] == MLB_TEAM_CONSENSUS_VERSION
    assert gated["models"]["mlb_new"]["picks"][0]["consensus_required"] is True
    assert "consensus_required" not in gated["models"]["wnba"]["picks"][0]
    assert gated["models"]["wnba"]["picks"][0] == wnba_pick
