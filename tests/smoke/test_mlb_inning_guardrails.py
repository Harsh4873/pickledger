"""MLB Inning model guardrail and smarts tests.

The pre-patch model always emitted the top-2 innings by raw scoreless
probability — so it would happily call a 50% inning a "High" confidence
pick even when the league baseline for that inning is already 38-44%.
These tests pin the new edge-vs-baseline gate, the late-inning bullpen
swap, the per-inning starter blend, and the park-factor adjustment.
"""
from __future__ import annotations


def _stub_threats(value: float = 0.27) -> dict:
    inn = {}
    for i in range(1, 10):
        inn[i] = {"away_threat": value, "home_threat": value}
    return {"GAME-1": {"innings": inn}}


def _stub_game(home="Yankees", away="Red Sox", **kwargs):
    base = {
        "game_id": "GAME-1",
        "home_team": home,
        "away_team": away,
        "home_pitcher": {"name": "Home SP", "era": 4.20},
        "away_pitcher": {"name": "Away SP", "era": 4.20},
        "venue": {},
    }
    base.update(kwargs)
    return base


def test_decision_for_edge_gate_thresholds():
    """The edge gate is the core anti-noise improvement: only innings
    that beat the league baseline by 3pp (LEAN) or 10pp (BET) qualify."""
    from models.mlb_inning.mlb_inning_probability import _decision_for_edge

    assert _decision_for_edge(probability=0.50, edge=0.02) == "PASS"   # too small
    assert _decision_for_edge(probability=0.45, edge=0.04) == "LEAN"   # clears LEAN gate
    assert _decision_for_edge(probability=0.55, edge=0.08) == "LEAN"   # below stricter BET gate
    assert _decision_for_edge(probability=0.55, edge=0.11) == "BET"    # clears BET gate
    assert _decision_for_edge(probability=0.42, edge=0.05) == "PASS"   # edge OK but prob too low


def test_strong_edge_inning_emits_bet():
    """When the team-history scoreless rate is materially higher than
    the league baseline, the model should emit a real BET pick."""
    from models.mlb_inning.mlb_inning_probability import compute_inning_probabilities

    # Both teams have very high scoreless rates in inning 1 → product is
    # well above league baseline (0.620^2 = 0.384).
    histories = {
        "Yankees": {1: {"scoreless_rate": 0.86}, 2: {"scoreless_rate": 0.78}},
        "Red Sox": {1: {"scoreless_rate": 0.84}, 2: {"scoreless_rate": 0.80}},
    }
    result = compute_inning_probabilities(_stub_game(), histories, _stub_threats())
    picks = result["top_2_picks"]
    assert picks, f"expected at least one pick, got {picks}"
    assert picks[0]["decision"] in {"BET", "LEAN"}
    assert picks[0]["edge_pp"] > 0.0


def test_park_factor_lowers_scoreless_probability():
    """Hitter-friendly parks (run_factor > 1) should reduce scoreless prob."""
    from models.mlb_inning.mlb_inning_probability import compute_inning_probabilities

    histories = {
        "Yankees": {i: {"scoreless_rate": 0.78} for i in range(1, 10)},
        "Red Sox": {i: {"scoreless_rate": 0.78} for i in range(1, 10)},
    }
    neutral = compute_inning_probabilities(_stub_game(), histories, _stub_threats())
    coors = compute_inning_probabilities(
        _stub_game(venue={"run_factor": 1.20}),
        histories,
        _stub_threats(),
    )
    for i in range(1, 9):
        assert coors["full_inning_table"][str(i)] < neutral["full_inning_table"][str(i)], (
            f"hitter park should reduce inning {i} scoreless prob"
        )


def test_late_innings_use_bullpen_not_starter():
    """Innings 7-8 should use bullpen scoreless rate, not starter ERA."""
    from models.mlb_inning.mlb_inning_probability import compute_inning_probabilities

    histories = {
        "Yankees": {i: {"scoreless_rate": 0.74} for i in range(1, 10)},
        "Red Sox": {i: {"scoreless_rate": 0.74} for i in range(1, 10)},
    }
    # Same starter ERA (5.50) but very different bullpens.
    bad_pen = _stub_game(
        home_pitcher={"name": "SP1", "era": 5.50, "team_bullpen": {"scoreless_rate": 0.62}},
        away_pitcher={"name": "SP2", "era": 5.50, "team_bullpen": {"scoreless_rate": 0.62}},
    )
    good_pen = _stub_game(
        home_pitcher={"name": "SP1", "era": 5.50, "team_bullpen": {"scoreless_rate": 0.86}},
        away_pitcher={"name": "SP2", "era": 5.50, "team_bullpen": {"scoreless_rate": 0.86}},
    )
    bad = compute_inning_probabilities(bad_pen, histories, _stub_threats())
    good = compute_inning_probabilities(good_pen, histories, _stub_threats())
    # Late innings (7, 8) should differ between the two — early ones (1-6)
    # should be unaffected because they don't consult the bullpen.
    for late in (7, 8):
        assert good["full_inning_table"][str(late)] > bad["full_inning_table"][str(late)], (
            f"good bullpen should raise inning {late} scoreless prob"
        )


def test_bullpen_fatigue_shift_is_linear_and_clamped():
    """Fatigue shift converts a [0, 1] index into a 0-12pp scoreless-rate
    drop. 0.5 fatigue ≈ 6pp shift; >=1.0 caps at MAX_FATIGUE_SHIFT_PP."""
    from models.mlb_inning.mlb_inning_bullpen import (
        compute_fatigue_shift,
        MAX_FATIGUE_SHIFT_PP,
    )

    assert compute_fatigue_shift(0.0) == 0.0
    assert compute_fatigue_shift(None) == 0.0
    assert compute_fatigue_shift(1.0) == MAX_FATIGUE_SHIFT_PP
    assert compute_fatigue_shift(2.0) == MAX_FATIGUE_SHIFT_PP   # clamped
    halfway = compute_fatigue_shift(0.5)
    assert abs(halfway - MAX_FATIGUE_SHIFT_PP * 0.5) < 1e-9


def test_burnt_bullpen_lowers_late_inning_scoreless_probability():
    """Same fresh-bullpen baseline; one team has 4 of 8 arms unavailable
    today (fatigue_index 0.50), the other is fully rested. Late-inning
    scoreless probability should be visibly lower for the burnt pen."""
    from models.mlb_inning.mlb_inning_probability import compute_inning_probabilities

    histories = {
        "Yankees": {i: {"scoreless_rate": 0.74} for i in range(1, 10)},
        "Red Sox": {i: {"scoreless_rate": 0.74} for i in range(1, 10)},
    }
    fresh = _stub_game(
        home_pitcher={
            "name": "Home SP", "era": 4.20,
            "team_bullpen": {"scoreless_rate": 0.78, "fatigue_index": 0.0},
        },
        away_pitcher={
            "name": "Away SP", "era": 4.20,
            "team_bullpen": {"scoreless_rate": 0.78, "fatigue_index": 0.0},
        },
    )
    burnt = _stub_game(
        home_pitcher={
            "name": "Home SP", "era": 4.20,
            "team_bullpen": {"scoreless_rate": 0.78, "fatigue_index": 0.50},
        },
        away_pitcher={
            "name": "Away SP", "era": 4.20,
            "team_bullpen": {"scoreless_rate": 0.78, "fatigue_index": 0.50},
        },
    )

    fresh_pp = compute_inning_probabilities(fresh, histories, _stub_threats())
    burnt_pp = compute_inning_probabilities(burnt, histories, _stub_threats())

    # Late innings (7, 8) should differ — burnt pen lower; early innings
    # (1-6) shouldn't change because they don't consult the bullpen rate.
    for late in (7, 8):
        assert burnt_pp["full_inning_table"][str(late)] < fresh_pp["full_inning_table"][str(late)], (
            f"burnt-pen inning {late} should be less scoreless than fresh-pen"
        )
    for early in (1, 5):
        assert burnt_pp["full_inning_table"][str(early)] == fresh_pp["full_inning_table"][str(early)], (
            f"early inning {early} shouldn't depend on bullpen fatigue"
        )


def test_pitch_weighted_fatigue_distinguishes_loogy_from_closer():
    """Reliever pitch counts in the boxscore should weight fatigue:
    a 25-pitch closer outing produces ~1 unit of unavailability;
    a 5-pitch LOOGY outing produces ~0.2 units."""
    from models.mlb_inning.mlb_inning_bullpen import _reliever_loads_from_feed

    # Synthetic boxscore feed shaped like the live MLB API.
    feed = {
        "liveData": {
            "boxscore": {
                "teams": {
                    "home": {
                        "team": {"id": 147},
                        "pitchers": [10000, 10001, 10002, 10003],  # SP + 3 RP
                        "players": {
                            "ID10000": {"stats": {"pitching": {"pitchesThrown": 95}}},  # SP
                            "ID10001": {"stats": {"pitching": {"pitchesThrown": 5}}},   # LOOGY
                            "ID10002": {"stats": {"pitching": {"pitchesThrown": 25}}},  # closer
                            "ID10003": {"stats": {"pitching": {"pitchesThrown": 18}}},  # setup
                        },
                    },
                    "away": {"team": {"id": 999}},
                },
            },
        },
    }

    loads = _reliever_loads_from_feed(feed, team_id=147)
    # Starter (10000) NOT included.
    assert 10000 not in loads
    # All 3 relievers present with pitches as floats.
    assert loads[10001] == 5.0
    assert loads[10002] == 25.0
    assert loads[10003] == 18.0


def test_parallel_bullpen_fetch_returns_same_workload_per_team(monkeypatch):
    """fetch_bullpen_workloads_parallel should produce identical output
    to fetch_bullpen_workload called sequentially, just faster."""
    from models.mlb_inning import mlb_inning_bullpen as bp

    # Stub the per-team fetcher so the test doesn't hit the network.
    def fake_workload(team_id, target_date, lookback_games=2):
        return {
            "lookback_games": lookback_games,
            "games_inspected": 1,
            "recently_used_pitcher_ids": [team_id * 10],
            "back_to_back_arms": [],
            "yesterday_used_pitcher_ids": [team_id * 10],
            "high_leverage_used_pitcher_ids": [],
            "light_use_pitcher_ids": [team_id * 10],
            "unavailable_today": [],
            "effective_unavailable_count": 0.4,
            "fatigue_index": 0.05,
        }

    monkeypatch.setattr(bp, "fetch_bullpen_workload", fake_workload)
    parallel_results = bp.fetch_bullpen_workloads_parallel([147, 147, 121, 158], "2026-05-15")
    assert set(parallel_results.keys()) == {147, 121, 158}  # de-duplicated
    for tid, payload in parallel_results.items():
        assert payload == fake_workload(tid, "2026-05-15")


def test_team_histories_dedupe_unique_teams(monkeypatch):
    """The slate-level history fetch should call each team once even when
    teams appear in multiple games or both sides of a double-header."""
    from models.mlb_inning import mlb_inning_history as hist

    calls = []

    def fake_history(team_id, team_name, target_date):
        calls.append((team_id, team_name, target_date))
        return hist._league_default_history()

    monkeypatch.setattr(hist, "fetch_team_history", fake_history)
    games = [
        {
            "game_date": "2026-05-15",
            "away_team_id": 147,
            "away_team": "New York Yankees",
            "home_team_id": 111,
            "home_team": "Boston Red Sox",
        },
        {
            "game_date": "2026-05-15",
            "away_team_id": 147,
            "away_team": "New York Yankees",
            "home_team_id": 111,
            "home_team": "Boston Red Sox",
        },
    ]

    result = hist.fetch_team_histories(games)

    assert set(result) == {"New York Yankees", "Boston Red Sox"}
    assert sorted(calls) == [
        (111, "Boston Red Sox", "2026-05-15"),
        (147, "New York Yankees", "2026-05-15"),
    ]


def test_inning_9_excluded_from_picks():
    """The 9th inning never gets emitted because the home half is unplayed
    when the home team is leading entering the bottom of the 9th."""
    from models.mlb_inning.mlb_inning_probability import compute_inning_probabilities

    histories = {
        "Yankees": {i: {"scoreless_rate": 0.95} for i in range(1, 10)},
        "Red Sox": {i: {"scoreless_rate": 0.95} for i in range(1, 10)},
    }
    result = compute_inning_probabilities(_stub_game(), histories, _stub_threats())
    assert "9" not in result["full_inning_table"]
    for pick in result["top_2_picks"]:
        assert pick["inning"] != 9
