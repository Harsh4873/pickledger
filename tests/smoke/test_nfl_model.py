"""NFL live model: no-lookahead features, publication integrity, and contracts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _game(gameday, season, week, home, away, hs, as_, spread, total, gtype="REG", **kw):
    row = {
        "game_id": f"{season}_{week:02d}_{away}_{home}", "season": str(season), "week": str(week),
        "game_type": gtype, "gameday": gameday, "gametime": "13:00", "home_team": home, "away_team": away,
        "home_score": "" if hs is None else str(hs), "away_score": "" if as_ is None else str(as_),
        "result": "" if hs is None else str(hs - as_), "spread_line": "" if spread is None else str(spread),
        "total_line": "" if total is None else str(total), "home_rest": "7", "away_rest": "7", "div_game": "0",
        "roof": "outdoors", "location": "Home", "home_qb_id": "", "away_qb_id": "",
        "home_moneyline": "-150", "away_moneyline": "130",
        "home_spread_odds": "-108", "away_spread_odds": "-112", "over_odds": "-105", "under_odds": "-115",
    }
    row.update(kw)
    return row


def _stats(game_id, home, away, home_epa=0.2, away_epa=-0.2):
    return {game_id: {
        home: {"epa_pp": home_epa, "pass_pp": home_epa, "rush_pp": home_epa, "cpoe": 2.0},
        away: {"epa_pp": away_epa, "pass_pp": away_epa, "rush_pp": away_epa, "cpoe": -2.0},
    }}


def test_feature_builder_never_sees_the_game_being_predicted():
    """The as-of pass must emit a game's features BEFORE folding its score
    into team state — a 60-point blowout cannot leak into its own row."""
    from NFLPredictionModel.nfl_core import build_dataset

    rows = [
        _game("2024-09-08", 2024, 1, "KC", "BAL", 27, 20, -3.0, 46.5),
        _game("2024-09-15", 2024, 2, "KC", "CIN", 60, 0, -3.0, 47.5),
    ]
    stats = {**_stats("2024_01_BAL_KC", "KC", "BAL"), **_stats("2024_02_CIN_KC", "KC", "CIN", home_epa=0.9)}
    records = build_dataset(rows, first_season=2024, team_stats=stats)
    week2 = next(r for r in records if r["game_id"].startswith("2024_02"))
    # Week 1 (27-20, +0.2 EPA/play) is in KC's state; week 2's own 60-0 and
    # 0.9 EPA/play are not.
    assert 0 < week2["features"]["epa_off_diff"] < 0.2
    assert week2["features"]["elo_diff"] > 48.0  # week-1 win plus home field
    assert week2["margin_residual"] == (60 - 0) - (-3.0)
    assert week2["home_spread_odds"] == -108 and week2["under_odds"] == -115


def test_slate_features_use_only_prior_history_and_keep_unpriced_games():
    from NFLPredictionModel.nfl_core import features_for_date

    rows = [
        _game("2026-09-06", 2026, 1, "KC", "BAL", 30, 10, -3.0, 46.5),
        _game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0),
        _game("2026-09-13", 2026, 2, "DEN", "LV", None, None, None, None),
    ]
    slate = features_for_date(rows, "2026-09-13", _stats("2026_01_BAL_KC", "KC", "BAL"))
    assert [entry["priced"] for entry in slate] == [True, False]
    features = slate[0]["features"]
    assert features["spread_line"] == -4.5
    assert features["epa_off_diff"] > 0  # week-1 EPA feeds the EWMA
    assert features["games_min"] == 0.0  # CIN has no EPA history yet
    assert slate[1]["spread_line"] is None and slate[1]["features"]["spread_line"] == 0.0


def test_kickoff_is_converted_from_eastern_wall_clock():
    from NFLPredictionModel.nfl_core import kickoff_iso, kickoff_utc

    game = _game("2026-09-13", 2026, 1, "KC", "BAL", None, None, -3.0, 46.5, gametime="16:25")
    assert kickoff_iso(game) == "2026-09-13T20:25Z"
    assert kickoff_utc(_game("2026-12-13", 2026, 14, "KC", "BAL", None, None, -3.0, 46.5)).hour == 18


class _Const:
    def __init__(self, value, proba=False):
        self.value = value
        self.proba = proba

    def predict(self, X):
        return [self.value] * len(X)

    def predict_proba(self, X):
        return [[1.0 - self.value, self.value]] * len(X)


def _artifacts(*, total_residual=-2.0, spread_residual=0.4, p_home=0.61, policy=None):
    metadata = json.loads((ROOT / "NFLPredictionModel" / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    if policy is not None:
        metadata = {**metadata, "decision_policy": policy}
    return {
        "ml": _Const(p_home, proba=True),
        "ml_free": _Const(0.55, proba=True),
        "spread": _Const(spread_residual),
        "total": _Const(total_residual),
        "metadata": metadata,
    }


UNDER_GATE = {
    "h2h": {"mode": "research_only", "reason": "test"},
    "spread": {"mode": "research_only", "reason": "test", "segments": []},
    "totals": {"mode": "segment_gate", "segments": [
        {"direction": "under", "min_residual": 1.0, "decision": "LEAN", "units": 0.25, "max_juice": -125},
    ]},
}


def _serve(monkeypatch, rows, date_iso, now, artifacts, stats=None):
    from NFLPredictionModel import nfl_model

    monkeypatch.setattr(nfl_model, "load_games", lambda refresh=True: rows)
    monkeypatch.setattr(nfl_model, "_load_artifacts", lambda: artifacts)
    monkeypatch.setattr(nfl_model, "load_team_stats", lambda seasons, **_kw: (stats or {}, [2025, 2026]))
    return nfl_model.generate_nfl_picks(date_iso, now=now)


def test_slate_publishes_research_rows_and_stakes_only_validated_segments(monkeypatch):
    rows = [
        _game("2026-09-06", 2026, 1, "KC", "BAL", 30, 10, -3.0, 46.5),
        _game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0),
    ]
    payload = _serve(monkeypatch, rows, "2026-09-13", datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
                     _artifacts(policy=UNDER_GATE))
    assert payload["ok"] is True
    assert payload["coverage"] == {
        "official_games": 1, "pregame_games": 1, "started_games": 0, "unpriced_games": 0,
        "forecast_games": 1, "staked_rows": 1, "team_stats_seasons": [2025, 2026],
    }
    by_market = {pick["market"]: pick for pick in payload["picks"]}
    assert set(by_market) == {"h2h", "spread", "totals"}
    for pick in payload["picks"]:
        assert pick["calibration_excluded"] is True
        assert pick["shadow_mode"] is False
        assert pick["start_time"] == "2026-09-13T17:00Z"
        assert pick["model_version"].startswith("nfl_")
    ml = by_market["h2h"]
    assert ml["team"] == "KC" and ml["odds"] == -150 and ml["opposite_odds"] == 130
    assert ml["decision"] == "PASS" and ml["units"] == 0
    assert ml["decision_reason"].startswith("research_only")
    assert ml["market_priced"] is True and ml["odds_source"] == "nflverse_posted_lines"
    spread = by_market["spread"]
    assert spread["pick"].startswith("KC +4.5") and spread["odds"] == -108
    assert spread["decision"] == "PASS" and spread["units"] == 0
    total = by_market["totals"]
    assert total["pick"].startswith("Under 48") and total["direction"] == "under"
    assert total["odds"] == -115 and total["opposite_odds"] == -105
    assert total["decision"] == "LEAN" and total["units"] == 0.25
    assert total["decision_reason"] == "segment:under:1"
    assert total["probability"] > 0.52
    assert "1 staked row(s)" in payload["note"]


def test_over_side_and_heavy_juice_never_stake(monkeypatch):
    rows = [
        _game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0, under_odds="-130"),
        _game("2026-09-13", 2026, 2, "DEN", "LV", None, None, 3.0, 41.0),
    ]
    artifacts = _artifacts(total_residual=-3.0, policy=UNDER_GATE)
    payload = _serve(monkeypatch, rows, "2026-09-13", datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc), artifacts)
    totals = {pick["matchup"]: pick for pick in payload["picks"] if pick["market"] == "totals"}
    juiced = totals["CIN @ KC"]
    assert juiced["decision"] == "PASS" and juiced["decision_reason"] == "juice_above_cap:-130"
    assert totals["LV @ DEN"]["decision"] == "LEAN"

    over_payload = _serve(monkeypatch, rows, "2026-09-13", datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
                          _artifacts(total_residual=4.0, policy=UNDER_GATE))
    for pick in over_payload["picks"]:
        if pick["market"] == "totals":
            assert pick["direction"] == "over"
            assert pick["decision"] == "PASS"
            assert pick["decision_reason"] == "no_segment_for_direction:over"
    assert over_payload["coverage"]["staked_rows"] == 0
    assert "research (PASS)" in over_payload["note"]


def test_started_games_are_skipped_and_explained(monkeypatch):
    rows = [
        _game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0, gametime="13:00"),
        _game("2026-09-13", 2026, 2, "DEN", "LV", None, None, 3.0, 41.0, gametime="20:20"),
    ]
    # 15:42 ET: the 1pm game has kicked off, the night game has not.
    payload = _serve(monkeypatch, rows, "2026-09-13", datetime(2026, 9, 13, 19, 42, tzinfo=timezone.utc),
                     _artifacts(policy=UNDER_GATE))
    assert payload["coverage"]["started_games"] == 1
    assert payload["coverage"]["pregame_games"] == 1
    assert {pick["matchup"] for pick in payload["picks"]} == {"LV @ DEN"}
    assert "1 started game(s) skipped" in payload["note"]


def test_unpriced_games_publish_as_unpriced_research_rows(monkeypatch):
    rows = [_game("2026-09-13", 2026, 2, "KC", "CIN", None, None, None, None,
                  home_moneyline="", away_moneyline="")]
    payload = _serve(monkeypatch, rows, "2026-09-13", datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc),
                     _artifacts(policy=UNDER_GATE))
    assert payload["coverage"]["unpriced_games"] == 1
    assert [pick["market"] for pick in payload["picks"]] == ["h2h"]
    ml = payload["picks"][0]
    assert ml["odds"] is None and ml["market_priced"] is False and ml["pricing_type"] == "unpriced"
    assert ml["probability_source"] == "market_free_logistic"
    assert ml["decision"] == "PASS" and ml["units"] == 0
    assert "without posted lines" in payload["note"]


def test_missing_team_stats_disable_segment_gates(monkeypatch):
    from NFLPredictionModel import nfl_model

    rows = [_game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0)]
    monkeypatch.setattr(nfl_model, "load_games", lambda refresh=True: rows)
    monkeypatch.setattr(nfl_model, "_load_artifacts", lambda: _artifacts(policy=UNDER_GATE))
    monkeypatch.setattr(nfl_model, "load_team_stats", lambda seasons, **_kw: ({}, []))
    payload = nfl_model.generate_nfl_picks("2026-09-13", now=datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc))
    total = next(pick for pick in payload["picks"] if pick["market"] == "totals")
    assert total["decision"] == "PASS"
    assert total["decision_reason"] == "research_only:team_stats_unavailable"
    assert "segment gates disabled" in payload["note"]


def test_no_game_day_is_explicit(monkeypatch):
    rows = [_game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0)]
    payload = _serve(monkeypatch, rows, "2026-09-16", datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc),
                     _artifacts(policy=UNDER_GATE))
    assert payload["picks"] == []
    assert payload["coverage"]["official_games"] == 0
    assert "active slate: 0 game(s)" in payload["note"]


def test_missing_artifacts_fail_visibly(monkeypatch):
    from NFLPredictionModel import nfl_model

    monkeypatch.setattr(nfl_model, "load_games", lambda refresh=True: [_game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0)])
    monkeypatch.setattr(nfl_model, "_load_artifacts", lambda: None)
    payload = nfl_model.generate_nfl_picks("2026-09-13")
    assert payload["ok"] is False
    assert "artifacts" in payload["error"]


def test_live_publication_across_surfaces():
    """NFL picks must be visible and receive a primary board filter."""
    data_ts = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    assert "SHADOW_SPORTS" not in data_ts
    assert "nfl: { h2h: 'NFL ML'" in data_ts
    main_ts = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    assert "'NFL'" in main_ts.split("PRIMARY_FILTERS")[1][:100]
    assert "NFL: ['football', 'nfl']" in main_ts


def test_registration_and_grading_slug():
    import pickgrader_server as server
    from scripts.market_odds import SPORT_LEAGUES, TEAM_MODEL_BUCKET_KEYS
    from scripts.merge_model_cache_payload import DEPLOYED_MODEL_KEYS
    from scripts.pick_calibration import CALIBRATION_EXCLUDED_MODEL_KEYS
    from scripts.team_prop_model_evaluator import SUPPORTED_MODEL_KEYS
    from scripts.team_prop_pregame_ledger import TEAM_PROP_MODEL_KEYS
    from scripts.refresh_model_cache import _model_jobs

    assert server.SPORT_TO_ESPNSLUG["NFL"] == ("football", "nfl")
    assert callable(server.run_nfl_model)
    assert SPORT_LEAGUES["NFL"] == ("football", "nfl")
    assert "nfl" in TEAM_MODEL_BUCKET_KEYS
    assert "nfl" in DEPLOYED_MODEL_KEYS
    assert "nfl" in TEAM_PROP_MODEL_KEYS
    assert "nfl" in SUPPORTED_MODEL_KEYS
    assert "nfl" in CALIBRATION_EXCLUDED_MODEL_KEYS
    assert "nfl" in _model_jobs("2026-09-13")
    from scripts import site_upcheck
    assert "nfl" in site_upcheck.REQUIRED_MODEL_KEYS


def test_shared_calibration_layer_leaves_nfl_rows_alone():
    from scripts.pick_calibration import apply_calibration_to_payload

    active = {"version": "test", "global": {"intercept": -0.4, "slope": 0.8}, "groups": {}}
    pick = {"sport": "NFL", "probability": 0.58, "raw_probability": 0.58, "decision": "LEAN", "units": 0.25,
            "odds": -110, "calibration_excluded": True}
    payload = {"models": {"nfl": {"picks": [pick]}}}
    apply_calibration_to_payload(payload, active)
    assert pick["probability"] == 0.58
    assert pick["decision"] == "LEAN" and pick["units"] == 0.25
    assert "calibration" not in pick


def test_artifacts_metadata_contract():
    meta = json.loads((ROOT / "NFLPredictionModel" / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["model_version"].startswith("nfl_")
    assert meta["games"] > 5000
    assert meta["feature_names"], "feature contract missing"
    assert set(meta["head_features"]) == {"moneyline", "moneyline_free", "spread", "total"}
    assert meta["walk_forward"], "walk-forward report missing"
    # Market parity: the moneyline head must not be materially worse than the
    # spread-implied market probability it is anchored to.
    assert meta["oof_ml_brier"] <= meta["market_reference_brier"] + 0.002
    policy = meta["decision_policy"]
    assert policy["h2h"]["mode"] == "research_only"
    assert policy["qualification_bar"]["max_tier"] == "LEAN"
    for market in ("spread", "totals"):
        for segment in policy[market]["segments"]:
            evidence = segment["walk_forward"]
            assert segment["decision"] == "LEAN"
            assert evidence["picks"] >= policy["qualification_bar"]["min_picks"]
            assert evidence["flat_roi"] >= policy["qualification_bar"]["min_flat_roi"]
            assert evidence["seasons_positive"] / evidence["seasons"] >= policy["qualification_bar"]["min_season_share_positive"]
    for name in ("nfl_ml.joblib", "nfl_ml_free.joblib", "nfl_spread.joblib", "nfl_total.joblib"):
        assert (ROOT / "NFLPredictionModel" / "artifacts" / name).exists()
    assert not (ROOT / "NFLPredictionModel" / "artifacts" / "nfl_ml_isotonic.joblib").exists()


@pytest.mark.parametrize("season_share,expected", [(0.5, False), (0.7, True)])
def test_segment_qualification_bar(season_share, expected):
    from NFLPredictionModel.nfl_train import _qualifies

    stats = {"picks": 150, "flat_roi": 0.05, "seasons": 10, "seasons_positive": int(season_share * 10)}
    assert _qualifies(stats) is expected
    assert _qualifies({**stats, "picks": 40}) is False
    assert _qualifies({**stats, "flat_roi": 0.01}) is False
