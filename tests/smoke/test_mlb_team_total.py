"""MLB Team Total model tests: candidate selection, pick rows, grading.

The team-total market is the one MLB prop with real DraftKings prices in
the repo's odds feed, so these rows are the first in-house MLB prop rows
that can become financially measurable end-to-end.
"""
from __future__ import annotations

import pytest


def test_best_candidate_uses_nearest_market_line_not_extreme_ladder():
    """The candidate must sit at the half-line nearest the projection —
    the extreme ladder lines always beat a flat -110 on paper, but the
    book never prices them there, so that edge would be phantom."""
    from models.mlb_team_total.mlb_team_total_model import _best_candidate

    high = _best_candidate(5.9)
    assert high["line"] == 5.5
    assert high["direction"] == "Over"
    assert 0.5 <= high["probability"] < 0.62

    low = _best_candidate(4.05)
    assert low["line"] == 4.5
    assert low["direction"] == "Under"

    # Clamped to the ladder at the extremes.
    assert _best_candidate(1.8)["line"] == 2.5
    assert _best_candidate(8.4)["line"] == 6.5


def test_median_shift_biases_toward_unders_near_the_line():
    """A projection sitting exactly on the line should lean Under — team
    run distributions are right-skewed so the median sits below the mean."""
    from models.mlb_team_total.mlb_team_total_model import _best_candidate

    assert _best_candidate(4.5)["direction"] == "Under"


def test_decision_thresholds():
    from models.mlb_team_total.mlb_team_total_model import _decision

    assert _decision(0.58, 5.5) == "BET"
    assert _decision(0.555, 3.2) == "LEAN"
    # The structural league-average Under (~0.546, +2.2pp) must PASS.
    assert _decision(0.546, 2.2) == "PASS"
    assert _decision(0.53, 10.0) == "PASS"  # probability short of LEAN floor


def test_pitching_multiplier_directionality():
    """A bad starter with a bad bullpen should inflate the opposing
    offense's run expectation; an ace with a good pen should suppress it."""
    from models.mlb_team_total.mlb_team_total_model import _pitching_multiplier

    bad = _pitching_multiplier(
        {"era": 5.80, "expected_outs": 14.0},
        {"starters": {}, "bullpen_scoreless_rate": 0.68},
    )
    ace = _pitching_multiplier(
        {"era": 2.60, "expected_outs": 19.0},
        {"starters": {}, "bullpen_scoreless_rate": 0.82},
    )
    assert bad["multiplier"] > 1.0 > ace["multiplier"]


def test_pick_rows_carry_market_and_assumed_price():
    from pickgrader_server import _mlb_team_total_pick_rows, MLB_TEAM_TOTAL_USER_ASSUMED_ODDS

    payload = {
        "date": "2026-07-19",
        "picks": [{
            "game_id": "747",
            "game_start_time": "2026-07-19T20:10Z",
            "matchup": "Colorado Rockies vs Cincinnati Reds",
            "home_team": "Colorado Rockies",
            "away_team": "Cincinnati Reds",
            "team_totals": [{
                "team": "Colorado Rockies",
                "side": "home",
                "projected_runs": 5.62,
                "offense_runs": 5.1,
                "pitching": {"multiplier": 1.06},
                "park_factor": 1.18,
                "line": 4.5,
                "direction": "Over",
                "probability": 0.646,
                "edge_pp": 12.2,
                "decision": "BET",
                "pick": "Colorado Rockies Team Total Over 4.5",
                "opposing_pitcher": "Away SP",
            }],
        }],
    }
    rows = _mlb_team_total_pick_rows(payload)
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "MLB Team Total"
    assert row["market"] == "team_total"
    assert row["pick"] == "Colorado Rockies Team Total Over 4.5"
    assert row["odds"] == MLB_TEAM_TOTAL_USER_ASSUMED_ODDS
    assert row["pricing_type"] == "user_assumed"
    assert row["line"] == 4.5
    assert row["direction"] == "over"
    assert row["market_implied_probability"] == pytest.approx(110.0 / 210.0, abs=1e-4)


def _graded_game(home: str, home_score: int, away: str, away_score: int) -> dict:
    return {
        "competitors": [
            {"raw": {"team": {"displayName": home}}, "score": home_score, "homeAway": "home", "linescores": []},
            {"raw": {"team": {"displayName": away}}, "score": away_score, "homeAway": "away", "linescores": []},
        ],
    }


def test_team_total_picks_grade_through_existing_named_total_path():
    from pickgrader_server import grade_pick

    game = _graded_game("Colorado Rockies", 6, "Cincinnati Reds", 3)
    over_pick = {
        "sport": "MLB",
        "pick": "Colorado Rockies Team Total Over 4.5",
        "team": "Colorado Rockies",
        "market": "team_total",
    }
    under_pick = {
        "sport": "MLB",
        "pick": "Cincinnati Reds Team Total Under 3.5",
        "team": "Cincinnati Reds",
        "market": "team_total",
    }
    losing_over = {
        "sport": "MLB",
        "pick": "Cincinnati Reds Team Total Over 4.5",
        "team": "Cincinnati Reds",
        "market": "team_total",
    }
    assert grade_pick(over_pick, game) == "win"
    assert grade_pick(under_pick, game) == "win"
    assert grade_pick(losing_over, game) == "loss"
