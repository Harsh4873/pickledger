"""CFB shadow model, pipeline, settlement, and containment contracts."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _game(game_id: str, date: str, home_score: int, away_score: int, *, fbs: bool = True) -> dict:
    return {
        "game_id": game_id,
        "season": 2025,
        "week": int(game_id[-1]),
        "start_time": f"{date}T17:00:00Z",
        "completed": True,
        "neutral_site": False,
        "conference_game": True,
        "home_team_id": "1",
        "away_team_id": str(int(game_id[-1]) + 1),
        "home_team": "Home State",
        "away_team": "Away Tech",
        "home_division": "fbs" if fbs else "fcs",
        "away_division": "fbs",
        "home_score": float(home_score),
        "away_score": float(away_score),
        "home_line": -3.5,
        "total_line": 52.5,
    }


def test_originator_is_strictly_as_of_and_market_free():
    from CFBPredictionModel.cfb_core import FEATURE_NAMES, build_dataset

    rows = [
        _game("g1", "2025-08-30", 70, 0),
        _game("g2", "2025-09-06", 28, 21),
    ]
    records = build_dataset(rows)
    assert len(records) == 2
    assert records[0]["features"]["home_offense_ewma"] == 28.0
    assert records[1]["features"]["home_offense_ewma"] > 28.0
    assert not {"home_line", "spread", "total_line", "moneyline"} & set(FEATURE_NAMES)


def test_training_population_filters_non_fbs_and_missing_lines():
    from CFBPredictionModel.cfb_core import build_dataset

    fcs = _game("g1", "2025-08-30", 30, 10, fbs=False)
    missing_line = _game("g2", "2025-09-06", 28, 21)
    missing_line["total_line"] = None
    assert build_dataset([fcs, missing_line]) == []


def test_shadow_serving_emits_exactly_three_stable_market_rows(monkeypatch):
    from CFBPredictionModel import cfb_model
    from CFBPredictionModel.cfb_core import FEATURE_NAMES

    features = {name: 0.0 for name in FEATURE_NAMES}
    entry = {
        "features": features,
        "game": {
            "game_id": "401900001",
            "event_id": "401900001",
            "home_team_id": "1",
            "away_team_id": "2",
            "home_team": "Home State Wildcats",
            "away_team": "Away Tech Owls",
            "start_time": "2026-09-05T17:00:00Z",
            "neutral_site": False,
            "home_line": -3.5,
            "total_line": 52.5,
            "home_moneyline": -155,
            "away_moneyline": 135,
            "odds_source": "espn_scoreboard:DraftKings",
        },
    }
    monkeypatch.setattr(cfb_model, "serving_rows", lambda _date: [entry])
    payload = cfb_model.generate_cfb_picks("2026-09-05")
    assert payload["ok"] is True
    assert payload["shadow_mode"] is True
    assert len(payload["games"]) == 1
    assert len(payload["picks"]) == 3
    assert {pick["source"] for pick in payload["picks"]} == {"CFB ML", "CFB Spread", "CFB Total"}
    assert {pick["market"] for pick in payload["picks"]} == {"h2h", "spread", "totals"}
    for pick in payload["picks"]:
        assert pick["shadow_mode"] is True
        assert pick["actionability"] == "research_signal"
        assert pick["espn_event_id"] == "401900001"
        assert pick["home_team_id"] == "1"
        assert pick["away_team_id"] == "2"
        assert 0 <= pick["push_probability"] < 1


def test_artifact_records_walk_forward_calibration_and_feature_contract():
    metadata = json.loads((ROOT / "CFBPredictionModel" / "artifacts" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_version"].startswith("cfb_")
    assert metadata["games"] > 5000
    assert metadata["walk_forward"]
    assert metadata["selected_family"] in {"ridge", "hist_gradient_boosting"}
    assert metadata["residual_distribution"]["kind"] == "bivariate_gaussian_oof"
    assert metadata["residual_distribution"]["samples"] > 3000
    assert set(metadata["calibration"]) == {"moneyline", "spread", "total"}
    assert metadata["market_features"] == []
    assert metadata["promotion_status"] == "not_qualified"
    assert (ROOT / "CFBPredictionModel" / "artifacts" / "cfb_model.joblib").stat().st_size > 1000


def test_registration_is_shared_but_cfb_is_not_a_core_freshness_requirement():
    import pickgrader_server as server
    from scripts import site_upcheck
    from scripts.market_odds import SPORT_LEAGUES, TEAM_MODEL_BUCKET_KEYS
    from scripts.merge_external_feed_cache_payload import REQUIRED_TEAM_MODEL_KEYS
    from scripts.merge_model_cache_payload import DEPLOYED_MODEL_KEYS, MODEL_ALIAS_KEYS
    from scripts.pick_calibration import CALIBRATION_EXCLUDED_MODEL_KEYS
    from scripts.refresh_model_cache import _model_jobs
    from scripts.team_prop_model_evaluator import SUPPORTED_MODEL_KEYS
    from scripts.team_prop_pregame_ledger import TEAM_PROP_MODEL_KEYS

    assert server.SPORT_TO_ESPNSLUG["CFB"] == ("football", "college-football")
    assert callable(server.run_cfb_model)
    assert SPORT_LEAGUES["CFB"] == ("football", "college-football")
    assert "cfb" in TEAM_MODEL_BUCKET_KEYS
    assert "cfb" in DEPLOYED_MODEL_KEYS
    assert "cfb" in MODEL_ALIAS_KEYS
    assert "cfb" in TEAM_PROP_MODEL_KEYS
    assert "cfb" in SUPPORTED_MODEL_KEYS
    assert "cfb" in CALIBRATION_EXCLUDED_MODEL_KEYS
    assert "cfb" in _model_jobs("2026-09-05")
    assert "cfb" not in site_upcheck.REQUIRED_MODEL_KEYS
    assert "cfb" not in REQUIRED_TEAM_MODEL_KEYS


def test_shadow_rows_are_contained_from_every_public_surface():
    data_ts = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    parlay = (ROOT / "scripts" / "build_parlay_cards.py").read_text(encoding="utf-8")
    profit = (ROOT / "scripts" / "build_profit_desk.py").read_text(encoding="utf-8")
    main_ts = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    assert "if (bucket.shadow_mode === true) continue;" in data_ts
    assert "if (rawRecord.shadow_mode === true) continue;" in data_ts
    assert "if pick.get(\"shadow_mode\") is True:" in parlay
    assert "if record.get(\"shadow_mode\") is True:" in profit
    assert "'CFB'" not in main_ts.split("PRIMARY_FILTERS")[1][:120]


def test_pass_rows_enter_forecast_audit_ledger(tmp_path):
    from scripts.team_prop_pregame_ledger import (
        capture_team_prop_pregame_snapshots,
        load_team_prop_pregame_ledger,
        stamp_team_prop_pregame_timing,
    )

    payload = {
        "date": "2026-09-05",
        "generatedAt": "2026-09-05T12:00:00Z",
        "models": {
            "cfb": {
                "ok": True,
                "model_version": "cfb_v1",
                "picks": [{
                    "game_id": "401900001",
                    "sport": "CFB",
                    "date": "2026-09-05",
                    "pick": "Away Tech +3.5 (Away Tech @ Home State)",
                    "market": "spread",
                    "decision": "PASS",
                    "start_time": "2026-09-05T17:00:00Z",
                    "odds": -110,
                    "pricing_type": "assumed",
                    "features": {"elo_diff": 20.0},
                }],
            }
        },
    }
    stamp_team_prop_pregame_timing(payload, published_at=payload["generatedAt"])
    summary = capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)
    assert summary == {"added": 1, "unchanged": 0, "team_picks": 1}
    records = load_team_prop_pregame_ledger(tmp_path)["records"]
    assert len(records) == 1
    assert records[0]["model_key"] == "cfb"
    assert records[0]["financial_eligible"] is False


def test_cfb_pass_ledger_rows_are_graded_for_forecast_evaluation():
    from scripts.auto_grade_picks import _pending_certified_team_prop_candidate

    record = {
        "id": "cfb-pass-record",
        "model_key": "cfb",
        "result": "pending",
        "decision": "PASS",
        "certification": {"status": "certified"},
        "pregame_snapshot": {
            "decision": "PASS",
            "date": "2026-09-05",
            "sport": "CFB",
            "pick": "Away Tech +3.5 (Away Tech @ Home State)",
        },
    }
    candidate = _pending_certified_team_prop_candidate(record)
    assert candidate is not None
    assert candidate[1]["decision"] == "PASS"


def _graded_game(home_score: int, away_score: int) -> dict:
    return {
        "competitors": [
            {"score": home_score, "raw": {"team": {"id": "1", "displayName": "Home State Wildcats", "abbreviation": "HST"}}},
            {"score": away_score, "raw": {"team": {"id": "2", "displayName": "Away Tech Owls", "abbreviation": "ATO"}}},
        ]
    }


def test_generic_grader_settles_cfb_moneyline_spread_total_and_pushes():
    import pickgrader_server as server

    game = _graded_game(30, 27)
    assert server.grade_pick({"sport": "CFB", "pick": "Home State Wildcats ML"}, game) == "win"
    assert server.grade_pick({"sport": "CFB", "pick": "Away Tech Owls +2.5"}, game) == "loss"
    assert server.grade_pick({"sport": "CFB", "pick": "Home State Wildcats -3"}, game) == "push"
    assert server.grade_pick({"sport": "CFB", "pick": "Over 57"}, game) == "push"
    assert server.grade_pick({"sport": "CFB", "pick": "Under 57.5"}, game) == "win"


def test_cfb_scoreboard_fetch_uses_fbs_group_and_large_limit(monkeypatch):
    import pickgrader_server as server

    seen: dict[str, str] = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"events": []}

    def fake_get(url, **_kwargs):
        seen["url"] = url
        return Response()

    monkeypatch.setattr(server.requests, "get", fake_get)
    assert server.fetch_scoreboard("football", "college-football", "20260905") == {"events": []}
    assert "limit=1000" in seen["url"]
    assert "groups=80" in seen["url"]


def test_cfb_training_workflow_is_manual_and_isolated():
    workflow = (ROOT / ".github" / "workflows" / "cfb-train.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow
    assert "group: cfb-train" in workflow
    assert "CFBPredictionModel/requirements.txt" in workflow
