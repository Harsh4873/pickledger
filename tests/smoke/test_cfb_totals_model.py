"""Smoke tests for the CFB market-residual totals model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from CFBTotalsModel import cfb_totals_model
from CFBTotalsModel.cfb_totals_core import (
    FEATURE_NAMES,
    MARKET_FEATURES,
    TeamState,
    build_dataset,
    feature_row,
    matrix,
)

ARTIFACT_DIR = Path(cfb_totals_model.__file__).resolve().parent / "artifacts"


def _game(game_id, start, home_score, away_score, *, home_line=-3.0, total_line=55.0,
          season=2024, completed=True):
    return {
        "game_id": game_id,
        "season": season,
        "week": 5,
        "start_time": start,
        "completed": completed,
        "neutral_site": False,
        "conference_game": True,
        "home_team_id": "H",
        "away_team_id": "A",
        "home_team": "Home State",
        "away_team": "Away Tech",
        "home_division": "fbs",
        "away_division": "fbs",
        "home_score": home_score,
        "away_score": away_score,
        "home_line": home_line,
        "total_line": total_line,
    }


def test_market_features_are_declared_not_hidden():
    """This model is a market-residual model: the line is an INPUT, on purpose.

    The originator forbids market features; this one requires them. The contract
    difference must be explicit so the two are never conflated.
    """
    assert MARKET_FEATURES == ["home_line", "total_line"]
    for name in MARKET_FEATURES:
        assert name in FEATURE_NAMES


def test_dataset_target_is_distance_from_the_posted_total():
    rows = [_game("g1", "2024-09-07T16:00Z", 30, 24, total_line=48.0)]
    records = build_dataset(rows)
    assert len(records) == 1
    # 30 + 24 = 54 actual vs a posted 48 -> +6
    assert records[0]["total_residual"] == pytest.approx(6.0)
    assert records[0]["game_total"] == pytest.approx(54.0)


def test_features_never_see_their_own_game_or_later_games():
    """Strict as-of: a row's features must be built BEFORE its result is folded in."""
    rows = [
        _game("g1", "2024-09-07T16:00Z", 40, 10),
        _game("g2", "2024-09-14T16:00Z", 3, 60),
    ]
    records = build_dataset(rows)
    assert len(records) == 2
    first, second = records
    # The first game is the teams' debut, so form sits at the league prior.
    assert first["features"]["home_offense_ewma"] == pytest.approx(28.0)
    assert first["features"]["home_games_log"] == pytest.approx(0.0)
    # The second game must reflect g1 having been played -- and only g1.
    assert second["features"]["home_games_log"] > 0.0
    assert second["features"]["home_offense_ewma"] != pytest.approx(28.0)


def test_matrix_column_order_follows_the_feature_contract():
    home, away = TeamState(), TeamState()
    row = {"features": feature_row(_game("g", "2024-09-07T16:00Z", 0, 0), home, away)}
    vector = matrix([row])[0]
    assert len(vector) == len(FEATURE_NAMES)
    assert vector[FEATURE_NAMES.index("total_line")] == pytest.approx(55.0)


def test_artifact_metadata_records_threshold_evidence():
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_class"] == "market_residual"
    assert metadata["market_free"] is False
    cert = metadata["threshold_certification"]
    assert cert["candidates"], "training must record every threshold it considered"
    # Every candidate carries the evidence used to accept or reject it.
    for row in cert["candidates"]:
        for key in ("threshold", "n", "rate", "p_value_vs_break_even",
                    "seasons_beating_break_even", "qualifies"):
            assert key in row
    # A qualifying threshold must actually clear the stated bar.
    if cert["selected_threshold"] is not None:
        evidence = cert["selected_evidence"]
        assert evidence["n"] >= 200
        assert evidence["p_value_vs_break_even"] < 0.05
        assert evidence["rate"] > metadata["break_even"]
        assert metadata["shadow_mode"] is False
    else:
        assert metadata["shadow_mode"] is True


def test_only_picks_above_the_certified_threshold_are_actionable(monkeypatch):
    """Below-threshold games must be an explicit PASS, never a silent bet."""
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    threshold = (metadata.get("threshold_certification") or {}).get("selected_threshold")
    if threshold is None:
        pytest.skip("no threshold certified in the shipped artifact")

    slate = [
        {
            "game_id": "big", "event_id": "big", "season": 2026, "week": 2,
            "start_time": "2026-09-05T16:00Z", "date": "2026-09-05", "completed": False,
            "neutral_site": False, "conference_game": False,
            "home_team_id": "H", "away_team_id": "A",
            "home_team": "Home State", "away_team": "Away Tech",
            "home_line": -7.0, "total_line": 50.0,
            "home_moneyline": -250, "away_moneyline": 200,
            "total_odds_over": -110, "total_odds_under": -110,
            "odds_source": "test",
        },
    ]
    monkeypatch.setattr(cfb_totals_model, "load_live_slate",
                        lambda _d, **_k: [dict(slate[0])])
    monkeypatch.setattr(cfb_totals_model, "load_history", lambda _s: [])
    monkeypatch.setattr(cfb_totals_model, "known_fbs_ids", lambda _h: {"H", "A"})
    monkeypatch.setattr(cfb_totals_model, "features_for_slate",
                        lambda _h, s: [{"game": g,
                                        "features": feature_row(g, TeamState(), TeamState())}
                                       for g in s])

    # Force a deviation just under the threshold -> the totals row must PASS.
    class Small:
        def predict(self, _x):
            return [threshold - 0.5]

    monkeypatch.setattr(cfb_totals_model, "_load_artifacts",
                        lambda: ({"total_residual_model": Small(),
                                  "margin_residual_model": Small()}, metadata))
    payload = cfb_totals_model.generate_cfb_totals_picks("2026-09-05")
    totals_rows = [p for p in payload["picks"] if p["market"] == "totals"]
    assert [p["decision"] for p in totals_rows] == ["PASS"]
    assert totals_rows[0]["units"] == 0.0
    # Spread and moneyline are published for visibility but never actionable
    # while their own certification has not qualified.
    for row in payload["picks"]:
        if row["market"] != "totals":
            assert row["decision"] == "PASS"
            assert row["units"] == 0.0
            assert "UNCERTIFIED" in row["evidence"]

    # Force a deviation past the threshold -> the totals row must BET.
    class Big:
        def predict(self, _x):
            return [threshold + 1.0]

    monkeypatch.setattr(cfb_totals_model, "_load_artifacts",
                        lambda: ({"total_residual_model": Big(),
                                  "margin_residual_model": Big()}, metadata))
    payload = cfb_totals_model.generate_cfb_totals_picks("2026-09-05")
    pick = next(p for p in payload["picks"] if p["market"] == "totals")
    assert pick["decision"] == "BET"
    assert pick["selection"] == "Over"
    assert pick["units"] == 0.5
    assert pick["sport"] == "CFB"
    assert pick["grade_supported"] is True
    assert pick["certified"] is True


def test_all_three_team_markets_are_published(monkeypatch):
    """The dashboard should see totals, spread and moneyline rows per game."""
    metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
    game = {
        "game_id": "g1", "event_id": "g1", "season": 2026, "week": 2,
        "start_time": "2026-09-05T16:00Z", "date": "2026-09-05", "completed": False,
        "neutral_site": False, "conference_game": False,
        "home_team_id": "H", "away_team_id": "A",
        "home_team": "Home State", "away_team": "Away Tech",
        "home_line": -7.0, "total_line": 50.0,
        "home_moneyline": -250, "away_moneyline": 200,
        "total_odds_over": -110, "total_odds_under": -110,
        "spread_odds_home": -108, "spread_odds_away": -112,
        "odds_source": "test",
    }
    monkeypatch.setattr(cfb_totals_model, "load_live_slate", lambda _d, **_k: [dict(game)])
    monkeypatch.setattr(cfb_totals_model, "load_history", lambda _s: [])
    monkeypatch.setattr(cfb_totals_model, "known_fbs_ids", lambda _h: {"H", "A"})
    monkeypatch.setattr(cfb_totals_model, "features_for_slate",
                        lambda _h, s: [{"game": g,
                                        "features": feature_row(g, TeamState(), TeamState())}
                                       for g in s])

    class Flat:
        def predict(self, _x):
            return [1.0]

    monkeypatch.setattr(cfb_totals_model, "_load_artifacts",
                        lambda: ({"total_residual_model": Flat(),
                                  "margin_residual_model": Flat()}, metadata))
    payload = cfb_totals_model.generate_cfb_totals_picks("2026-09-05")
    markets = {p["market"] for p in payload["picks"]}
    assert markets == {"totals", "spread", "h2h"}
    # Nothing may claim to be shadow -- these rows are meant to render.
    assert all(p["shadow_mode"] is False for p in payload["picks"])
    # Every row states which evidence bucket it belongs to.
    assert all(p["market_status"] for p in payload["picks"])


def test_model_is_registered_for_the_pipeline():
    import pickgrader_server as server

    assert callable(server.run_cfb_totals_model)
    assert server.SPORT_TO_ESPNSLUG["CFB"] == ("football", "college-football")

    from scripts import refresh_model_cache

    jobs = refresh_model_cache._model_jobs("2026-09-05")
    assert "cfb_totals" in jobs
    # the originator must still be registered independently
    assert "cfb" in jobs


def test_moneyline_is_read_from_espns_current_shape():
    """Regression: ESPN moved the moneyline, which silently emptied the slate."""
    from CFBPredictionModel.cfb_core import _moneyline_for

    legacy = {"homeTeamOdds": {"moneyLine": -155}}
    assert _moneyline_for(legacy, "home") == -155

    current = {"moneyline": {"home": {"close": {"odds": "-160"}},
                             "away": {"close": {"odds": "+140"}}}}
    assert _moneyline_for(current, "home") == -160
    assert _moneyline_for(current, "away") == 140

    # A book that has pulled the market reports "OFF" -- that is not a price.
    off = {"moneyline": {"home": {"close": {"odds": "OFF"}}}}
    assert _moneyline_for(off, "home") is None
