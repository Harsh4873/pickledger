"""Holdout refits stay off the stake path and ship only a better Brier."""
from __future__ import annotations

from scripts.inhouse_holdout_refit import (
    FIRST_FIVE_FEATURE_V2,
    SHIPPED_FITS,
    apply_platt,
    score_model,
    shipped_artifact,
)


def _records(model_key: str, *, dates: int, per_date: int, probability: float, win_rate: float):
    records = []
    index = 0
    wins_per_date = int(round(per_date * win_rate))
    for day_index in range(dates):
        month = 7 + day_index // 28
        day = (day_index % 28) + 1
        slate = f"2026-{month:02d}-{day:02d}"
        for slot in range(per_date):
            records.append({
                "model_key": model_key,
                "model_version": f"{model_key}_v1",
                "stable_id": f"{model_key}-{index}",
                "revision": 1,
                "slate_date": slate,
                "raw_probability": probability,
                "result": "win" if slot < wins_per_date else "loss",
            })
            index += 1
    return records


def test_overconfident_model_ships_a_shrink_and_keeps_the_side():
    records = _records("mlb_new", dates=40, per_date=4, probability=0.85, win_rate=0.5)
    result = score_model(records, "mlb_new")
    assert result["ship"] is True
    assert result["slope"] > 0
    assert result["holdout_brier_new"] < result["holdout_brier_old"]
    adjusted = apply_platt(0.85, result)
    assert 0.5 < adjusted < 0.85


def test_already_sharp_model_is_not_replaced():
    records = _records("nfl", dates=40, per_date=4, probability=0.75, win_rate=0.75)
    result = score_model(records, "nfl")
    assert result["ship"] is False
    assert result["new_version"] == "not shipped"


def test_negative_slope_is_not_shipped(monkeypatch):
    from scripts import inhouse_holdout_refit as refit

    monkeypatch.setattr(refit, "fit_platt", lambda rows, **_kwargs: {"slope": -0.2, "intercept": 0.0})
    records = _records("wnba", dates=40, per_date=4, probability=0.7, win_rate=0.7)
    result = score_model(records, "wnba")
    assert result["ship"] is False
    assert "negative slope" in result["reason"]


def test_first_five_feature_v2_stays_unshipped_on_the_record():
    assert FIRST_FIVE_FEATURE_V2["ship"] is False
    assert round(FIRST_FIVE_FEATURE_V2["holdout_brier_old"], 3) == 0.250
    assert round(FIRST_FIVE_FEATURE_V2["holdout_brier_new"], 3) == 0.322
    assert "mlb_first_five" not in SHIPPED_FITS
    result = score_model([], "mlb_first_five")
    assert result["ship"] is False
    assert result["feature_v2"]["holdout_brier_new"] > result["feature_v2"]["holdout_brier_old"]


def test_inning_refit_does_not_change_the_published_stake():
    from pickgrader_server import _mlb_inning_pick_rows

    rows = _mlb_inning_pick_rows({
        "date": "2026-06-12",
        "picks": [{
            "game_id": "1",
            "matchup": "Home vs Away",
            "home_team": "Home",
            "away_team": "Away",
            "top_2_picks": [{
                "inning": 1,
                "probability_scoreless": 0.55,
                "baseline": 0.44,
                "edge_pp": 11.0,
                "decision": "BET",
                "confidence": "High",
            }],
        }],
    })
    inning = rows[0]
    assert inning["model_version"] == "mlb_inning_platt_2026-10-09"
    assert inning["model_epoch"] == "mlb_inning_v2_2026-08-25"
    assert inning["decision"] == "PASS"
    assert inning["units"] == 0.0
    assert inning["assumed_odds"] == -120
    assert inning["odds"] is None
    assert set(SHIPPED_FITS) == {"mlb_inning"}
    artifact = shipped_artifact([{
        "ship": True,
        "model": "mlb_inning",
        "new_version": "mlb_inning_platt_2026-10-09",
        "old_version": "mlb_inning_v2_2026-08-25",
        "slope": SHIPPED_FITS["mlb_inning"]["slope"],
        "intercept": SHIPPED_FITS["mlb_inning"]["intercept"],
        "holdout_start": "2026-09-09",
        "holdout_brier_old": 0.2550,
        "holdout_brier_new": 0.2511,
    }])
    assert "units" not in artifact["models"]["mlb_inning"]
