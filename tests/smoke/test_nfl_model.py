"""NFL shadow model: no-lookahead features, shadow containment, contracts."""
from __future__ import annotations


def _game(gameday, season, week, home, away, hs, as_, spread, total, gtype="REG", **kw):
    row = {
        "game_id": f"{season}_{week}_{away}_{home}", "season": str(season), "week": str(week),
        "game_type": gtype, "gameday": gameday, "home_team": home, "away_team": away,
        "home_score": "" if hs is None else str(hs), "away_score": "" if as_ is None else str(as_),
        "result": "" if hs is None else str(hs - as_), "spread_line": str(spread),
        "total_line": str(total), "home_rest": "7", "away_rest": "7", "div_game": "0",
        "roof": "outdoors", "home_qb_id": "", "away_qb_id": "",
        "home_moneyline": "-150", "away_moneyline": "130",
    }
    row.update(kw)
    return row


def test_feature_builder_never_sees_the_game_being_predicted():
    """The as-of pass must emit a game's features BEFORE folding its score
    into team state — a 40-point blowout cannot leak into its own row."""
    from NFLPredictionModel.nfl_core import build_dataset

    rows = [
        _game("2024-09-08", 2024, 1, "KC", "BAL", 27, 20, -3.0, 46.5),
        _game("2024-09-15", 2024, 2, "KC", "CIN", 60, 0, -3.0, 47.5),
    ]
    records = build_dataset(rows, first_season=2024)
    week2 = next(r for r in records if r["game_id"].startswith("2024_2"))
    # KC's offense EWMA entering week 2 reflects only the 27-point week 1,
    # pulled toward the 22-point league prior — nowhere near 60.
    assert week2["features"]["home_off_ewma"] < 30.0
    assert week2["margin_residual"] == (60 - 0) - (-3.0)


def test_slate_features_use_only_prior_history():
    from NFLPredictionModel.nfl_core import features_for_date

    rows = [
        _game("2026-09-06", 2026, 1, "KC", "BAL", 30, 10, -3.0, 46.5),
        _game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0),
    ]
    slate = features_for_date(rows, "2026-09-13")
    assert len(slate) == 1
    features = slate[0]["features"]
    assert features["spread_line"] == -4.5
    # Week 1 result feeds the EWMA; the pending game itself contributes nothing.
    assert features["home_off_ewma"] > 22.0


def test_shadow_rows_carry_decisions_but_are_flagged():
    from NFLPredictionModel.nfl_model import _ml_decision, _residual_decision, _row_base

    base = _row_base(_game("2026-09-13", 2026, 2, "KC", "CIN", None, None, -4.5, 48.0), "2026-09-13")
    assert base["shadow_mode"] is True
    assert base["sport"] == "NFL"
    assert base["market_priced"] is True
    assert _ml_decision(0.06, 0.60) == "BET"
    assert _ml_decision(0.03, 0.53) == "LEAN"
    assert _ml_decision(0.01, 0.70) == "PASS"
    # Spread/total stay LEAN-capped in shadow no matter the disagreement.
    assert _residual_decision(8.0) == "LEAN"
    assert _residual_decision(3.0) == "LEAN"
    assert _residual_decision(1.0) == "PASS"


def test_shadow_containment_across_surfaces():
    """Shadow picks must be invisible: site pick load, parlay legs, and
    profit-desk records all skip shadow rows; ledger/grading do not."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    data_ts = (root / "src" / "data.ts").read_text(encoding="utf-8")
    assert "const SHADOW_SPORTS = new Set(['NFL'])" in data_ts
    assert "!SHADOW_SPORTS.has(pick.sport)" in data_ts
    assert "nfl: { h2h: 'NFL ML'" in data_ts
    parlay = (root / "scripts" / "build_parlay_cards.py").read_text(encoding="utf-8")
    assert 'pick.get("shadow_mode") is True' in parlay
    profit = (root / "scripts" / "build_profit_desk.py").read_text(encoding="utf-8")
    assert 'record.get("shadow_mode") is True' in profit
    main_ts = (root / "src" / "main.ts").read_text(encoding="utf-8")
    assert "'MLS']" in main_ts and "'NFL'" not in main_ts.split("PRIMARY_FILTERS")[1][:80]


def test_registration_and_grading_slug():
    import pickgrader_server as server
    from scripts.market_odds import SPORT_LEAGUES, TEAM_MODEL_BUCKET_KEYS
    from scripts.merge_model_cache_payload import DEPLOYED_MODEL_KEYS
    from scripts.team_prop_pregame_ledger import TEAM_PROP_MODEL_KEYS
    from scripts.refresh_model_cache import _model_jobs

    assert server.SPORT_TO_ESPNSLUG["NFL"] == ("football", "nfl")
    assert callable(server.run_nfl_model)
    assert SPORT_LEAGUES["NFL"] == ("football", "nfl")
    assert "nfl" in TEAM_MODEL_BUCKET_KEYS
    assert "nfl" in DEPLOYED_MODEL_KEYS
    assert "nfl" in TEAM_PROP_MODEL_KEYS
    assert "nfl" in _model_jobs("2026-09-13")
    from scripts import site_upcheck
    assert "nfl" not in site_upcheck.REQUIRED_MODEL_KEYS  # soft launch


def test_artifacts_metadata_contract():
    import json
    from pathlib import Path

    meta_path = Path(__file__).resolve().parents[2] / "NFLPredictionModel" / "artifacts" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["model_version"].startswith("nfl_")
    assert meta["games"] > 5000
    assert meta["feature_names"], "feature contract missing"
    assert meta["walk_forward"], "walk-forward report missing"
    assert 0.15 < meta["oof_ml_brier"] < 0.26
