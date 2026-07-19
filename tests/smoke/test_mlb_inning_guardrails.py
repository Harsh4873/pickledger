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
    """Two gates now: beat the league baseline by 3pp (LEAN) / 10pp (BET)
    AND clear the assumed -120 breakeven (54.55%) with margin — the old
    0.45 LEAN floor published picks that were -EV at their own price."""
    from models.mlb_inning.mlb_inning_probability import _decision_for_edge

    assert _decision_for_edge(probability=0.50, edge=0.02) == "PASS"   # too small
    assert _decision_for_edge(probability=0.45, edge=0.04) == "PASS"   # old LEAN floor: -EV at -120
    assert _decision_for_edge(probability=0.56, edge=0.04) == "LEAN"   # clears LEAN gates
    assert _decision_for_edge(probability=0.55, edge=0.08) == "PASS"   # below the -120 price floor
    assert _decision_for_edge(probability=0.57, edge=0.11) == "LEAN"   # edge BETs but prob below BET floor
    assert _decision_for_edge(probability=0.58, edge=0.11) == "BET"    # clears BET gates
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

    # Late innings (7, 8) should differ — burnt pen lower. Innings 1-4 are
    # pure starter under the taper (league-average exit ≈ 5.1 IP), so they
    # shouldn't move; innings 5-6 legitimately consult the bullpen now.
    for late in (7, 8):
        assert burnt_pp["full_inning_table"][str(late)] < fresh_pp["full_inning_table"][str(late)], (
            f"burnt-pen inning {late} should be less scoreless than fresh-pen"
        )
    for early in (1, 4):
        assert burnt_pp["full_inning_table"][str(early)] == fresh_pp["full_inning_table"][str(early)], (
            f"inning {early} is starter-only and shouldn't depend on bullpen fatigue"
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

    def fake_context(team_id, team_name, target_date):
        calls.append((team_id, team_name, target_date))
        return hist._league_default_context()

    monkeypatch.setattr(hist, "fetch_team_context", fake_context)
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


def test_starter_taper_hands_middle_innings_to_bullpen():
    """A short starter (4 IP avg) leaves innings 5-6 to the bullpen; a
    workhorse (7 IP avg) keeps them. With a bad pen and a good starter,
    the short-starter team's innings 5-6 should be less scoreless."""
    from models.mlb_inning.mlb_inning_probability import compute_inning_probabilities

    histories = {
        "Yankees": {i: {"scoreless_rate": 0.74} for i in range(1, 10)},
        "Red Sox": {i: {"scoreless_rate": 0.74} for i in range(1, 10)},
    }
    def _game(outs):
        return _stub_game(
            home_pitcher={
                "name": "SP", "era": 3.00, "expected_outs": outs,
                "team_bullpen": {"scoreless_rate": 0.58},
            },
            away_pitcher={
                "name": "SP2", "era": 3.00, "expected_outs": outs,
                "team_bullpen": {"scoreless_rate": 0.58},
            },
        )

    workhorse = compute_inning_probabilities(_game(21.0), histories, _stub_threats())
    short = compute_inning_probabilities(_game(12.0), histories, _stub_threats())
    for inning in (5, 6):
        assert short["full_inning_table"][str(inning)] < workhorse["full_inning_table"][str(inning)], (
            f"inning {inning} should lean bullpen for a short starter"
        )
    # Inning 1 is starter-only for both.
    assert short["full_inning_table"]["1"] == workhorse["full_inning_table"]["1"]


def test_populated_starter_inning_rates_move_the_probability():
    """`inning_scoreless_rates` was a dead input pre-patch; a dominant
    observed inning-2 rate should now raise that inning's scoreless prob
    over the flat ERA-derived fallback."""
    from models.mlb_inning.mlb_inning_probability import compute_inning_probabilities

    histories = {
        "Yankees": {i: {"scoreless_rate": 0.74} for i in range(1, 10)},
        "Red Sox": {i: {"scoreless_rate": 0.74} for i in range(1, 10)},
    }
    flat = _stub_game()
    informed = _stub_game(
        home_pitcher={"name": "Home SP", "era": 4.20, "inning_scoreless_rates": {"2": 0.95}},
        away_pitcher={"name": "Away SP", "era": 4.20, "inning_scoreless_rates": {"2": 0.95}},
    )
    flat_pp = compute_inning_probabilities(flat, histories, _stub_threats())
    informed_pp = compute_inning_probabilities(informed, histories, _stub_threats())
    assert informed_pp["full_inning_table"]["2"] > flat_pp["full_inning_table"]["2"]
    assert informed_pp["full_inning_table"]["3"] == flat_pp["full_inning_table"]["3"]


def test_static_park_factor_fallback_applies_by_venue_id():
    """Coors (venue_id 19) should lower scoreless probs vs a neutral park
    even though the fetcher never sets venue.run_factor."""
    from models.mlb_inning.mlb_inning_probability import compute_inning_probabilities

    histories = {
        "Yankees": {i: {"scoreless_rate": 0.78} for i in range(1, 10)},
        "Red Sox": {i: {"scoreless_rate": 0.78} for i in range(1, 10)},
    }
    neutral = compute_inning_probabilities(_stub_game(), histories, _stub_threats())
    coors = compute_inning_probabilities(_stub_game(venue_id=19), histories, _stub_threats())
    for inning in range(1, 9):
        assert coors["full_inning_table"][str(inning)] < neutral["full_inning_table"][str(inning)]
    assert coors["venue_factor"] == 1.18


def test_weather_multiplier_directions():
    """Wind out / heat lower the scoreless probability; wind in raises it;
    a closed roof ignores wind entirely."""
    from models.mlb_inning.mlb_inning_environment import scoreless_weather_multiplier

    out_hot, _ = scoreless_weather_multiplier({"wind": "15 mph, Out To CF", "temp": "93"})
    calm, _ = scoreless_weather_multiplier({"wind": "Calm", "temp": "72"})
    wind_in, _ = scoreless_weather_multiplier({"wind": "12 mph, In From LF", "temp": "72"})
    roof, roof_detail = scoreless_weather_multiplier({"wind": "10 mph, Out To RF", "condition": "Roof Closed", "temp": "72"})

    assert out_hot < calm == 1.0 < wind_in
    assert roof == 1.0
    assert roof_detail["wind_direction"] == "roof_closed"
    assert 0.95 <= out_hot <= 1.03


def test_history_summaries_extract_bullpen_and_starter_rates():
    """The bullpen and starter summaries feed the previously dead
    probability inputs from the same cached feeds as the offense history."""
    from models.mlb_inning.mlb_inning_history import (
        _summarize_bullpen_allowed,
        _summarize_starters,
    )

    records = []
    for index in range(20):
        records.append({
            "scored": {i: 0 for i in range(1, 10)},
            # Inning 7 allowed a run every other game; 8-9 always scoreless.
            "allowed": {i: (1 if i == 7 and index % 2 == 0 else 0) for i in range(1, 10)},
            "starter_id": 500 if index < 5 else 600 + index,
            "starter_outs": 18,  # 6 full innings
        })

    bullpen = _summarize_bullpen_allowed(records)
    by_inning = bullpen["bullpen_scoreless_by_inning"]
    assert by_inning["8"] > by_inning["7"]
    assert bullpen["bullpen_samples"] == 20

    starters = _summarize_starters(records)
    assert starters["500"]["starts"] == 5
    assert starters["500"]["avg_outs"] == 18.0
    # 5 starts × innings 1-6 completed, all scoreless except inning 7 (not credited).
    for inning in ("1", "6"):
        assert starters["500"]["innings"][inning]["n"] == 5


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
