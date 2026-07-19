"""Tests for the MLB First Five park-factor + wind/weather signal."""
from __future__ import annotations


def test_park_factor_neutral_for_unknown_venue():
    from models.mlb_first_five.mlb_first_five_environment import (
        park_factor,
        park_factor_run_delta,
    )
    assert park_factor(99999) == 1.0
    assert park_factor(None) == 1.0
    assert park_factor_run_delta(99999) == 0.0


def test_park_factor_lookup_known_venues():
    """Coors should be the most extreme hitter park; Petco the most
    pitcher-friendly. Both should produce material run deltas."""
    from models.mlb_first_five.mlb_first_five_environment import (
        park_factor,
        park_factor_run_delta,
    )
    coors = park_factor(19)
    petco = park_factor(2680)
    assert coors >= 1.10
    assert petco <= 0.95
    assert park_factor_run_delta(19) > 0.10
    assert park_factor_run_delta(2680) < -0.05


def test_parse_wind_extracts_mph_and_direction():
    from models.mlb_first_five.mlb_first_five_environment import parse_wind

    assert parse_wind("10 mph, Out to RF") == {"mph": 10.0, "direction": "out"}
    assert parse_wind("12 MPH, In from CF") == {"mph": 12.0, "direction": "in"}
    assert parse_wind("8 mph, L to R")["direction"] == "cross"
    assert parse_wind("Calm")["direction"] == "calm"
    assert parse_wind("")["direction"] == ""
    assert parse_wind("")["mph"] == 0.0


def test_wind_run_delta_signs_and_caps():
    """Outward wind boosts F5 runs; inward suppresses; both capped."""
    from models.mlb_first_five.mlb_first_five_environment import wind_run_delta

    assert wind_run_delta("10 mph, Out to CF") > 0
    assert wind_run_delta("15 mph, Out to RF") > wind_run_delta("8 mph, Out to RF")
    assert wind_run_delta("12 mph, In from CF") < 0
    assert wind_run_delta("Calm") == 0.0
    assert wind_run_delta("5 mph, L to R") == 0.0  # cross winds ~0
    # Cap test — extreme reading is bounded.
    assert wind_run_delta("60 mph, Out to CF") <= 0.45
    assert wind_run_delta("60 mph, In from CF") >= -0.45


def test_blend_park_run_delta_prefers_static_when_sample_thin():
    """A thin learned sample should be pulled toward the static park-factor
    prior; a fat sample should hand off entirely to the learned delta."""
    from models.mlb_first_five.mlb_first_five_environment import blend_park_run_delta

    # 4 games of learned data at Coors (vid=19) — should still tilt strongly
    # toward the static +0.18 hitter prior.
    thin = blend_park_run_delta(learned_delta=0.0, learned_games=4, venue_id=19)
    assert thin["blend_weight_learned"] < 0.5
    assert thin["final_delta"] > 0.10  # static prior dominates

    # 30 games — fully trust the learned signal.
    fat = blend_park_run_delta(learned_delta=-0.05, learned_games=30, venue_id=19)
    assert fat["blend_weight_learned"] == 1.0
    assert fat["final_delta"] == -0.05  # learned wins outright


def test_blend_handles_unknown_venue_gracefully():
    from models.mlb_first_five.mlb_first_five_environment import blend_park_run_delta
    result = blend_park_run_delta(learned_delta=0.05, learned_games=10, venue_id=None)
    assert result["static_delta"] == 0.0
    assert result["park_factor"] == 1.0


def test_model_generated_f5_total_is_capped_at_lean():
    from models.mlb_first_five.mlb_first_five_model import _apply_pick_guardrails

    picks = [{
        "market": "f5_total",
        "probability": 0.64,
        "edge_pct": 11.6,
        "projection_gap": 1.1,
        "vegas_line": 4.5,
        "decision": "BET",
    }]

    _apply_pick_guardrails(
        picks,
        {"current_starts": 7},
        {"current_starts": 8},
    )

    assert picks[0]["decision"] == "LEAN"
    assert "model-generated F5 total line" in picks[0]["guardrail"]
