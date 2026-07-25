"""Smoke tests for the MLS Dixon-Coles model (v2).

The properties asserted here are the ones that would silently produce a
plausible-looking but wrong model: as-of fitting (no leakage through the
prediction date), push-aware market pricing that sums to one, the fixed
price cap and unpriced/short-history downgrades on decisions, workbook
parsing with the real quirks (BOM, dd/mm/yyyy, closing-price preference),
duplicate-safe merging of the two match sources, and the shape of the
emitted bucket (which the shared grader and the frontend split consume).
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MLSPredictionModel import mls_model  # noqa: E402
from MLSPredictionModel.mls_core import (  # noqa: E402
    FitConfig,
    VectorScaling,
    conditional_probability,
    devig,
    fit_ratings,
    handicap_split,
    one_x_two,
    score_grid,
    total_split,
)
from MLSPredictionModel.mls_data import (  # noqa: E402
    MlsMatch,
    merge_matches,
    parse_football_data_csv,
)

STRONG, WEAK, MID_A, MID_B = "1001", "1002", "1003", "1004"


def _synthetic_history(through: date, weeks: int = 60) -> list[MlsMatch]:
    """Deterministic round-robin: STRONG beats everyone, MID teams split."""
    teams = [STRONG, WEAK, MID_A, MID_B]
    matches: list[MlsMatch] = []
    day = through - timedelta(weeks=weeks)
    week = 0
    while day < through:
        pairings = [
            (teams[week % 4], teams[(week + 1) % 4]),
            (teams[(week + 2) % 4], teams[(week + 3) % 4]),
        ]
        for home, away in pairings:
            if home == away:
                continue
            # Scores are venue-aware so the fit has a real home edge to find.
            if STRONG in (home, away):
                home_goals, away_goals = (3, 0) if home == STRONG else (1, 2)
            elif WEAK in (home, away):
                home_goals, away_goals = (1, 2) if home == WEAK else (3, 0)
            else:
                home_goals, away_goals = ((1, 1), (2, 1))[week % 2]
            matches.append(MlsMatch(date=day, home=home, away=away,
                                    home_goals=home_goals, away_goals=away_goals))
        week += 1
        day += timedelta(days=7)
    return matches


def test_fit_uses_only_matches_before_as_of():
    as_of = date(2026, 6, 1)
    history = _synthetic_history(as_of)
    # After as_of, WEAK suddenly wins every game; the fit must not notice.
    future = [
        MlsMatch(date=as_of + timedelta(days=offset), home=WEAK, away=STRONG,
                 home_goals=5, away_goals=0)
        for offset in range(0, 21, 7)
    ]
    baseline = fit_ratings(history, as_of, FitConfig())
    with_future = fit_ratings(history + future, as_of, FitConfig())
    assert with_future.attack == pytest.approx(baseline.attack)
    assert with_future.defense == pytest.approx(baseline.defense)
    assert baseline.attack[STRONG] > baseline.attack[WEAK]
    assert baseline.matches_used == with_future.matches_used


def test_fit_recovers_strength_ordering_and_home_edge():
    as_of = date(2026, 6, 1)
    fit = fit_ratings(_synthetic_history(as_of), as_of, FitConfig())
    strong_rating = fit.attack[STRONG] + fit.defense[STRONG]
    weak_rating = fit.attack[WEAK] + fit.defense[WEAK]
    assert strong_rating > weak_rating
    lam, mu = fit.rates(STRONG, WEAK)
    assert lam > mu
    # Unknown team rates league-average, and known() reports it.
    assert not fit.known("9999")
    neutral_lam, neutral_mu = fit.rates("9999", "9998")
    assert neutral_lam > neutral_mu  # only home advantage separates them


def test_score_grid_market_masses_are_push_aware():
    grid = score_grid(1.55, 1.20, -0.05)
    assert grid.sum() == pytest.approx(1.0)
    outcomes = one_x_two(grid)
    assert sum(outcomes.values()) == pytest.approx(1.0)

    integer_total = total_split(grid, 3.0)
    assert integer_total["push"] > 0
    assert sum(integer_total.values()) == pytest.approx(1.0)
    half_total = total_split(grid, 2.5)
    assert half_total["push"] == pytest.approx(0.0)

    # Integer handicap keeps push mass; the conditional probability is what a
    # two-way price implies.
    level = handicap_split(grid, "home", -1.0)
    assert level["push"] > 0
    assert sum(level.values()) == pytest.approx(1.0)
    conditional = conditional_probability(level["win"], level["loss"])
    assert 0.0 < conditional < 1.0

    # Quarter lines settle as the stake-weighted average of the two halves.
    quarter = handicap_split(grid, "home", -0.75)
    half_low = handicap_split(grid, "home", -0.5)
    half_high = handicap_split(grid, "home", -1.0)
    assert quarter["win"] == pytest.approx((half_low["win"] + half_high["win"]) / 2)
    assert quarter["push"] == pytest.approx((half_low["push"] + half_high["push"]) / 2)

    # Home/away symmetry: away at +line mirrors home at -line.
    away_side = handicap_split(grid, "away", 1.0)
    assert away_side["win"] == pytest.approx(level["loss"])
    assert away_side["loss"] == pytest.approx(level["win"])


def test_devig_and_vector_scaling():
    devigged = devig([0.55, 0.30, 0.25])
    assert sum(devigged) == pytest.approx(1.0)
    assert devigged[0] == pytest.approx(0.5)
    identity = VectorScaling()
    probabilities = {"home": 0.5, "draw": 0.3, "away": 0.2}
    assert identity.apply(probabilities) == pytest.approx(probabilities)


def test_football_data_parser_handles_real_quirks():
    csv_text = (
        "﻿Country,League,Season,Date,Time,Home,Away,HG,AG,Res,"
        "PSCH,PSCD,PSCA,MaxCH,MaxCD,MaxCA,AvgCH,AvgCD,AvgCA,B365CH,B365CD,B365CA\n"
        "USA,MLS,2026,05/07/2026,00:30,Inter Miami,FC Dallas,2,1,H,"
        "1.80,3.90,4.20,1.85,4.0,4.4,1.78,3.8,4.1,1.8,3.9,4.2\n"
        "USA,MLS,2026,06/07/2026,00:30,Los Angeles Galaxy,Austin FC,1,1,D,"
        ",,,2.30,3.60,3.20,2.25,3.5,3.1,2.28,3.55,3.15\n"
    )
    matches = parse_football_data_csv(csv_text)
    assert len(matches) == 2
    first = matches[0]
    assert first.date == date(2026, 7, 5)  # dd/mm parsed, not mm/dd
    assert (first.home, first.away) == ("20232", "185")
    assert first.close_source == "pinnacle_close"
    assert first.close_home == pytest.approx(1.80)
    # Pinnacle absent on the second row: preference falls back to the average.
    assert matches[1].close_source == "average_close"
    assert matches[1].close_home == pytest.approx(2.25)

    # A name outside the franchise map poisons ratings silently, so the
    # training path must refuse it; non-strict (runtime refresh) skips it.
    renamed = csv_text.replace("Inter Miami", "Miami Beach United")
    with pytest.raises(ValueError):
        parse_football_data_csv(renamed)
    assert len(parse_football_data_csv(renamed, strict=False)) == 1


def test_merge_matches_dedupes_across_date_skew():
    base = [MlsMatch(date=date(2026, 7, 20), home="182", away="183", home_goals=1, away_goals=0)]
    skewed_duplicate = [MlsMatch(date=date(2026, 7, 21), home="182", away="183", home_goals=1, away_goals=0)]
    fresh = [MlsMatch(date=date(2026, 7, 22), home="184", away="185", home_goals=2, away_goals=2)]
    merged = merge_matches(base, skewed_duplicate + fresh)
    assert len(merged) == 2
    assert merged[0].date == date(2026, 7, 20)


@pytest.fixture()
def serving_fixture(monkeypatch, tmp_path):
    target = date(2026, 7, 25)
    history = _synthetic_history(target)
    artifact = {
        "model_version": "mls_dixon_coles_v2.0-test",
        "trained_at": "2026-07-24",
        "data_through": (target - timedelta(days=2)).isoformat(),
        "config": {"half_life_days": 365.0, "ridge": 0.02},
        "calibration": {
            "vector_scaling": {"gamma": 1.0, "bias_home": 0.0, "bias_away": 0.0},
            "total": {"intercept": 0.0, "slope": 1.0},
        },
        "market_blend_weight": 0.6,
        "gates": {
            "moneyline": {"bet_blended_probability": 0.60, "lean_blended_probability": 0.55},
            "grid_markets": {"bet_blended_probability": 0.58, "lean_blended_probability": 0.545},
            "max_decided_implied_probability": 0.7143,
        },
        "min_effective_games": 8.0,
    }
    artifact_path = tmp_path / "mls_model.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    monkeypatch.setattr(mls_model, "MODEL_PATH", artifact_path)
    monkeypatch.setattr(mls_model, "load_matches", lambda: history)
    monkeypatch.setattr(mls_model, "fetch_football_data_matches", _raise_network_down)
    return target


def _raise_network_down() -> list[MlsMatch]:
    raise RuntimeError("offline test")


def _event(event_id, home_id, away_id, odds, state="pre"):
    return {
        "id": event_id,
        "date": "2026-07-25T23:30Z",
        "status": {"type": {"state": state, "completed": state == "post"}},
        "competitions": [{
            "competitors": [
                {"homeAway": "home", "team": {"id": home_id, "displayName": f"Team {home_id}"},
                 "score": "2" if state == "post" else "0"},
                {"homeAway": "away", "team": {"id": away_id, "displayName": f"Team {away_id}"},
                 "score": "1" if state == "post" else "0"},
            ],
            "venue": {"fullName": "Test Park"},
            "odds": [odds] if odds else [],
        }],
    }


FULL_ODDS = {
    "overUnder": 2.5,
    "moneyline": {
        "home": {"close": {"odds": "-130"}},
        "away": {"close": {"odds": "+340"}},
        "draw": {"close": {"odds": "+270"}},
    },
    "total": {
        "over": {"close": {"odds": "-115"}},
        "under": {"close": {"odds": "-105"}},
    },
    "pointSpread": {
        "home": {"close": {"odds": "-125", "line": "-0.5"}},
        "away": {"close": {"odds": "-105", "line": "+0.5"}},
    },
}
JUICED_ODDS = {
    "overUnder": 3.5,
    "moneyline": {
        "home": {"close": {"odds": "-400"}},
        "away": {"close": {"odds": "+900"}},
        "draw": {"close": {"odds": "+500"}},
    },
    "total": {
        "over": {"close": {"odds": "-110"}},
        "under": {"close": {"odds": "-110"}},
    },
}


class FakeEspnClient:
    def __init__(self, events):
        self.events = events
        self.scoreboard_dates: list[str] = []

    def scoreboard(self, date_iso):
        self.scoreboard_dates.append(date_iso)
        if date_iso == "2026-07-25":
            return {"events": self.events}
        return {"events": []}


def test_serving_bucket_contract(serving_fixture):
    events = [
        _event("g1", STRONG, WEAK, FULL_ODDS),
        _event("g2", MID_A, MID_B, None),
        _event("g3", MID_B, MID_A, JUICED_ODDS),
        _event("g4", STRONG, MID_A, FULL_ODDS, state="post"),
    ]
    bucket = mls_model.generate_mls_picks("2026-07-25", client=FakeEspnClient(events))
    assert bucket["ok"] is True
    assert bucket["date"] == "2026-07-25"
    assert bucket["calibration_excluded"] is True
    assert bucket["meta"]["model_version"] == "mls_dixon_coles_v2.0-test"
    assert "unavailable" in bucket["meta"]["workbook_refresh"]

    by_game: dict[str, dict[str, dict]] = {}
    for pick in bucket["picks"]:
        by_game.setdefault(pick["game_id"], {})[pick["market"]] = pick
        assert pick["source"] == "MLS Model"
        assert pick["sport"] == "MLS"
        assert pick["market_type"] in {"soccer_moneyline", "soccer_total", "soccer_handicap"}
        assert 0.0 < pick["probability"] < 1.0
        assert pick["decision"] in {"BET", "LEAN", "PASS"}
        assert (pick["units"] > 0) == (pick["decision"] != "PASS")
        assert pick["date"] == "2026-07-25"
        assert pick["matchup"] == pick["game"]

    # Completed games never produce rows.
    assert "g4" not in by_game

    # Fully priced game: all three markets, blend arithmetic verified.
    priced = by_game["g1"]
    assert set(priced) == {"moneyline", "total", "spread"}
    moneyline = priced["moneyline"]
    assert moneyline["team"] == f"Team {STRONG}"
    blended = 0.4 * moneyline["model_probability"] + 0.6 * moneyline["market_probability"]
    assert moneyline["probability"] == pytest.approx(blended, abs=1e-3)
    assert moneyline["odds"] == -130
    assert priced["total"]["line"] == 2.5
    assert priced["total"]["push_probability"] == pytest.approx(0.0)
    assert priced["spread"]["line"] in (-0.5, 0.5)

    # STRONG crushes WEAK every meeting: the blended probability must clear
    # the BET gate at a -130 price (implied 0.565 < cap).
    assert moneyline["decision"] == "BET"

    # Unpriced game: moneyline only (no fabricated total/spread lines), and
    # the decision is capped at LEAN even at overwhelming model confidence.
    unpriced = by_game["g2"]
    assert set(unpriced) == {"moneyline"}
    assert unpriced["moneyline"]["odds"] is None
    assert unpriced["moneyline"]["market_probability"] is None
    assert unpriced["moneyline"]["edge"] is None
    assert unpriced["moneyline"]["decision"] in {"LEAN", "PASS"}

    # Heavy juice: -400 implied 0.80 breaches the fixed cap, so the moneyline
    # can never publish decided, regardless of confidence.
    juiced = by_game["g3"]
    assert juiced["moneyline"]["odds"] == -400
    assert juiced["moneyline"]["decision"] == "PASS"
    # The same game's fairly-priced total is still allowed to decide.
    assert juiced["total"]["odds"] in (-110,)

    summaries = {summary["game_id"] for summary in bucket["games"]}
    assert summaries == {"g1", "g2", "g3"}
    for summary in bucket["games"]:
        three_way = (summary["home_win_probability"] + summary["draw_probability"]
                     + summary["away_win_probability"])
        assert three_way == pytest.approx(1.0, abs=2e-2)

    ratings = {rating["team_id"] for rating in bucket["team_ratings"]}
    assert {STRONG, WEAK, MID_A, MID_B} <= ratings


def test_serving_short_history_passes_everything(serving_fixture):
    # A brand-new team (no rated history) must force PASS on its game.
    events = [_event("g9", "7777", STRONG, FULL_ODDS)]
    bucket = mls_model.generate_mls_picks("2026-07-25", client=FakeEspnClient(events))
    assert bucket["picks"], "rows are still published for transparency"
    assert all(pick["decision"] == "PASS" for pick in bucket["picks"])


def test_serving_empty_slate(serving_fixture):
    bucket = mls_model.generate_mls_picks("2026-07-25", client=FakeEspnClient([]))
    assert bucket["ok"] is True
    assert bucket["picks"] == []
    assert "No MLS games" in bucket["note"]
