from __future__ import annotations

import json
from pathlib import Path

from scripts.pick_calibration import (
    apply_calibration_to_payload,
    build_outcome_ledger,
    calibration_group_key,
)
from scripts.train_pick_calibration import run_training, should_promote


def _pick(**overrides):
    pick = {
        "id": "pick-1",
        "date": "2026-06-01",
        "source": "Test Model",
        "sport": "MLB",
        "pick": "Pitcher Over 5.5 Strikeouts",
        "stat_key": "pitcher_strikeouts",
        "probability": 0.7,
        "edge": 17.62,
        "odds": -110,
        "units": 1.0,
        "decision": "BET",
        "result": "pending",
        "feature_detail": {"pitch_types": {"sweeper": 0.31}},
    }
    pick.update(overrides)
    return pick


def test_calibration_preserves_snapshot_and_bounds_units():
    active = {
        "version": "test-v1",
        "minimum_group_samples": 30,
        "global": {"intercept": -1.25, "slope": 1.0, "samples": 100},
        "groups": {},
    }
    payload = {"models": {"mlb_player_props": {"picks": [_pick()]}}}

    apply_calibration_to_payload(payload, active)
    adjusted = payload["models"]["mlb_player_props"]["picks"][0]

    assert adjusted["pregame_snapshot"]["probability"] == 0.7
    assert adjusted["pregame_snapshot"]["feature_detail"]["pitch_types"]["sweeper"] == 0.31
    assert "result" not in adjusted["pregame_snapshot"]
    assert adjusted["raw_probability"] == 0.7
    assert adjusted["probability"] < 0.7
    assert adjusted["units"] <= 1.25
    assert adjusted["calibration"]["version"] == "test-v1"


def test_calibration_can_downgrade_a_bet_without_overwriting_original_decision():
    active = {
        "version": "test-v2",
        "minimum_group_samples": 30,
        "global": {"intercept": -3.0, "slope": 1.0, "samples": 100},
        "groups": {},
    }
    payload = {"models": {"mlb_player_props": {"picks": [_pick()]}}}

    apply_calibration_to_payload(payload, active)
    adjusted = payload["models"]["mlb_player_props"]["picks"][0]

    assert adjusted["decision"] == "PASS"
    assert adjusted["units"] == 0
    assert adjusted["pregame_snapshot"]["decision"] == "BET"


def test_calibration_downgrades_bet_to_lean_when_adjusted_edge_is_midrange():
    active = {
        "version": "test-v2b",
        "minimum_group_samples": 30,
        "global": {"intercept": -0.55, "slope": 1.0, "samples": 100},
        "groups": {},
    }
    payload = {"models": {"wnba_player_props": {"picks": [_pick(sport="WNBA")]}}}

    apply_calibration_to_payload(payload, active)
    adjusted = payload["models"]["wnba_player_props"]["picks"][0]

    assert 3 <= adjusted["edge"] < 7
    assert adjusted["decision"] == "LEAN"
    assert adjusted["pregame_snapshot"]["decision"] == "BET"


def test_calibration_leaves_units_alone_when_no_edge_or_market_price_exists():
    active = {
        "version": "test-v3",
        "minimum_group_samples": 30,
        "global": {"intercept": -0.2, "slope": 1.0, "samples": 100},
        "groups": {},
    }
    pick = _pick(edge=None, odds=None, units=1.13)
    payload = {"models": {"nba": {"picks": [pick]}}}

    apply_calibration_to_payload(payload, active)

    assert payload["models"]["nba"]["picks"][0]["units"] == 1.13


def test_ml_player_props_skip_old_calibration_and_are_not_ledger_trainable(tmp_path: Path):
    active = {
        "version": "test-ml-skip",
        "minimum_group_samples": 1,
        "global": {"intercept": -3.0, "slope": 1.0, "samples": 100},
        "groups": {},
    }
    pick = _pick(
        probability_source="player_props_ml_v1",
        ml_probability=0.7,
        ml_calibration_excluded=True,
        result="win",
    )
    pick["ranking_updated_at"] = "2026-06-20T20:42:53.275777Z"
    payload = {"date": "2026-06-20", "models": {"mlb_player_props": {"picks": [pick]}}}

    apply_calibration_to_payload(payload, active)
    adjusted = payload["models"]["mlb_player_props"]["picks"][0]

    assert adjusted["probability"] == 0.7
    assert "calibration" not in adjusted

    props_dir = tmp_path / "data" / "player_props_cache"
    model_dir = tmp_path / "data" / "model_cache"
    props_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (props_dir / "2026-06-20.json").write_text(json.dumps(payload), encoding="utf-8")
    ledger = build_outcome_ledger(tmp_path)
    assert ledger["summary"]["total_picks"] == 1
    assert ledger["summary"]["trainable_decided_picks"] == 0
    assert ledger["records"][0]["raw_probability"] == 0.7
    assert ledger["records"][0]["calibration_eligible"] is False
    assert ledger["records"][0]["calibration_exclusion_reason"] == "ml_owned_probability"


def test_player_prop_ledger_forgets_pre_ml_records(tmp_path: Path):
    props_dir = tmp_path / "data" / "player_props_cache"
    model_dir = tmp_path / "data" / "model_cache"
    props_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    legacy = _pick(result="win", probability_source="legacy_projection")
    payload = {"date": "2026-06-15", "models": {"mlb_player_props": {"picks": [legacy]}}}
    (props_dir / "2026-06-15.json").write_text(json.dumps(payload), encoding="utf-8")

    ledger = build_outcome_ledger(tmp_path)

    assert ledger["summary"]["total_picks"] == 0


def test_ledger_rebuild_rebases_calibration_checkpoint_after_reset(tmp_path: Path):
    from scripts.pick_calibration import rebuild_outcome_ledger

    calibration_dir = tmp_path / "data" / "calibration"
    props_dir = tmp_path / "data" / "player_props_cache"
    model_dir = tmp_path / "data" / "model_cache"
    calibration_dir.mkdir(parents=True)
    props_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (calibration_dir / "state.json").write_text(json.dumps({
        "last_evaluated_decided_count": 500,
        "last_trainable_decided_count": 400,
    }), encoding="utf-8")

    ledger, _ = rebuild_outcome_ledger(tmp_path)
    state = json.loads((calibration_dir / "state.json").read_text(encoding="utf-8"))

    assert ledger["summary"]["decided_picks"] == 0
    assert state["last_evaluated_decided_count"] == 0
    assert state["last_trainable_decided_count"] == 0


def test_pick_level_calibration_exclusion_skips_adjustment_and_training(tmp_path: Path):
    active = {
        "version": "test-v4",
        "minimum_group_samples": 1,
        "global": {"intercept": -3.0, "slope": 1.0, "samples": 100},
        "groups": {},
    }
    pick = _pick(
        source="SportsGamblerFIFAWorldCup",
        sport="FIFA WC",
        pick="Switzerland Asian Hcp -1.75",
        calibration_excluded=True,
        result="win",
    )
    payload = {"date": "2026-06-01", "models": {"sportsgambler_fifa_world_cup": {"picks": [pick]}}}

    apply_calibration_to_payload(payload, active)
    assert payload["models"]["sportsgambler_fifa_world_cup"]["picks"][0]["probability"] == 0.7
    assert "calibration" not in payload["models"]["sportsgambler_fifa_world_cup"]["picks"][0]

    model_dir = tmp_path / "data" / "model_cache"
    props_dir = tmp_path / "data" / "player_props_cache"
    model_dir.mkdir(parents=True)
    props_dir.mkdir(parents=True)
    (model_dir / "2026-06-01.json").write_text(json.dumps(payload), encoding="utf-8")
    ledger = build_outcome_ledger(tmp_path)
    assert ledger["summary"]["total_picks"] == 0


def test_universal_ledger_deduplicates_and_keeps_exact_pregame_context(tmp_path: Path):
    model_dir = tmp_path / "data" / "model_cache"
    props_dir = tmp_path / "data" / "player_props_cache"
    model_dir.mkdir(parents=True)
    props_dir.mkdir(parents=True)
    pick = _pick(result="win")
    payload = {"date": "2026-06-01", "models": {"mlb_player_props": {"picks": [pick]}}}
    (model_dir / "2026-06-01.json").write_text(json.dumps(payload), encoding="utf-8")
    (model_dir / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
    ledger = build_outcome_ledger(tmp_path)

    assert ledger["summary"]["total_picks"] == 1
    assert ledger["summary"]["decided_picks"] == 1
    record = ledger["records"][0]
    assert record["raw_probability"] == 0.7
    assert record["outcome"] == 1
    assert record["pregame_snapshot"]["feature_detail"]["pitch_types"]["sweeper"] == 0.31
    assert "result" not in record["pregame_snapshot"]


def test_player_prop_snapshots_remain_available_to_outcome_ledger(tmp_path: Path):
    model_dir = tmp_path / "data" / "model_cache"
    props_dir = tmp_path / "data" / "player_props_cache"
    snapshots_dir = tmp_path / "data" / "player_props_snapshots" / "2026-06-21"
    model_dir.mkdir(parents=True)
    props_dir.mkdir(parents=True)
    snapshots_dir.mkdir(parents=True)

    def prop(pick_id: str, player_id: str) -> dict:
        return {
            "id": pick_id,
            "scope": "player",
            "source": "MLB Season Props",
            "sport": "MLB",
            "date": "2026-06-21",
            "game_id": "1",
            "player_id": player_id,
            "stat_key": "hits",
            "selection": "Over",
            "line": 0.5,
            "pick": f"Player {player_id} Over 0.5 Hits",
            "matchup": "Away @ Home",
            "market_priced": True,
            "probability_source": "player_props_ml_v1",
            "probability": 0.61,
            "odds": -110,
            "decision": "BET",
            "result": "pending",
        }

    old_payload = {
        "date": "2026-06-21",
        "generatedAt": "2026-06-21T18:00:00Z",
        "models": {"mlb_player_props_season": {"picks": [prop("old", "10")]}},
    }
    current_payload = {
        "date": "2026-06-21",
        "generatedAt": "2026-06-21T20:00:00Z",
        "models": {"mlb_player_props_season": {"picks": [prop("new", "20")]}},
    }
    (snapshots_dir / "snapshot.json").write_text(json.dumps(old_payload), encoding="utf-8")
    (props_dir / "2026-06-21.json").write_text(json.dumps(current_payload), encoding="utf-8")

    ledger = build_outcome_ledger(tmp_path)

    assert {record["pick"] for record in ledger["records"]} == {
        "Player 10 Over 0.5 Hits",
        "Player 20 Over 0.5 Hits",
    }
    assert ledger["summary"]["total_picks"] == 2


def test_model_cache_pass_decisions_stay_out_of_outcome_ledger(tmp_path: Path):
    model_dir = tmp_path / "data" / "model_cache"
    props_dir = tmp_path / "data" / "player_props_cache"
    model_dir.mkdir(parents=True)
    props_dir.mkdir(parents=True)
    downgraded = _pick(result="win", decision="PASS")
    downgraded["pregame_snapshot"] = {"decision": "BET", "probability": 0.7, "sport": "MLB"}
    native_pass = _pick(id="pick-2", pick="Cubs ML", decision="PASS", result="loss")
    kept = _pick(id="pick-3", pick="Guardians ML", decision="LEAN", result="win")
    payload = {"date": "2026-07-01", "models": {"scores24_mlb": {"picks": [downgraded, native_pass, kept]}}}
    (model_dir / "2026-07-01.json").write_text(json.dumps(payload), encoding="utf-8")

    ledger = build_outcome_ledger(tmp_path)

    assert ledger["summary"]["total_picks"] == 1
    assert ledger["records"][0]["pick"] == "Guardians ML"
    assert ledger["records"][0]["decision"] == "LEAN"


def test_player_prop_pass_abstentions_remain_in_outcome_ledger(tmp_path: Path):
    model_dir = tmp_path / "data" / "model_cache"
    props_dir = tmp_path / "data" / "player_props_cache"
    model_dir.mkdir(parents=True)
    props_dir.mkdir(parents=True)
    abstention = _pick(
        date="2026-06-21",
        decision="PASS",
        units=0,
        result="win",
        probability_source="player_props_ml_v1",
        ml_calibration_excluded=True,
    )
    payload = {
        "date": "2026-06-21",
        "generatedAt": "2026-06-21T18:00:00Z",
        "models": {"mlb_player_props": {"picks": [abstention]}},
    }
    (props_dir / "2026-06-21.json").write_text(json.dumps(payload), encoding="utf-8")

    ledger = build_outcome_ledger(tmp_path)

    assert ledger["summary"]["total_picks"] == 1
    assert ledger["records"][0]["decision"] == "PASS"
    assert ledger["summary"]["trainable_decided_picks"] == 0


def test_certified_team_record_excludes_recalibration_pass_downgrades():
    from scripts.pick_calibration import _certified_team_record

    def certified(decision: str) -> dict:
        return {
            "id": f"team-{decision.lower()}",
            "slate_date": "2026-07-01",
            "model_key": "mlb_new",
            "calibration_eligible": True,
            "certification": {"status": "certified"},
            "raw_probability": 0.62,
            "observed_american_odds": -115,
            "result": "win",
            "stake": 1.0,
            "decision": decision,
            "raw_decision": "BET",
            "sport": "MLB",
            "source": "MLB Team Model",
            "market": "moneyline",
            "pregame_snapshot": {"decision": "BET", "pick": "Cubs ML"},
        }

    assert _certified_team_record(certified("BET")) is not None
    assert _certified_team_record(certified("PASS")) is None


def test_trainer_waits_for_100_new_decisions_then_force_evaluates(tmp_path: Path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    group = calibration_group_key("mlb_new", "MLB", "moneyline")
    records = []
    for index in range(60):
        outcome = 1 if index % 2 == 0 else 0
        records.append(
            {
                "id": str(index),
                "date": f"2026-05-{(index % 28) + 1:02d}",
                "raw_probability": 0.65 if outcome else 0.55,
                "market_implied_probability": 0.52,
                "raw_units": 1,
                "profit": 0.91 if outcome else -1,
                "outcome": outcome,
                "calibration_group": group,
                "calibration_eligible": True,
            }
        )
    ledger = {
        "summary": {"decided_picks": 60, "trainable_decided_picks": 60},
        "records": records,
    }
    (calibration_dir / "outcome_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    waiting = run_training(calibration_dir)
    evaluated = run_training(calibration_dir, force=True)

    assert waiting["evaluated"] is False
    assert waiting["new_decisions"] == 60
    assert evaluated["evaluated"] is True
    assert (calibration_dir / "active.json").exists()
    assert (calibration_dir / "challenger.json").exists()
    assert json.loads((calibration_dir / "state.json").read_text())["last_evaluated_decided_count"] == 60


def test_existing_champion_waits_for_enough_unseen_trainable_rows(tmp_path: Path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    group = calibration_group_key("mlb_new", "MLB", "moneyline")
    records = [
        {
            "id": str(index),
            "date": f"2026-06-{(index % 28) + 1:02d}",
            "raw_probability": 0.6,
            "market_implied_probability": 0.52,
            "raw_units": 1,
            "profit": 0.91 if index % 2 == 0 else -1,
            "outcome": 1 if index % 2 == 0 else 0,
            "calibration_group": group,
            "calibration_eligible": True,
        }
        for index in range(110)
    ]
    (calibration_dir / "outcome_ledger.json").write_text(
        json.dumps({"summary": {"decided_picks": 200}, "records": records}),
        encoding="utf-8",
    )
    (calibration_dir / "active.json").write_text(
        json.dumps({
            "training_contract_version": 2,
            "global": {"intercept": 0, "slope": 1, "samples": 100},
            "groups": {},
        }),
        encoding="utf-8",
    )
    (calibration_dir / "state.json").write_text(
        json.dumps({
            "training_contract_version": 2,
            "last_evaluated_decided_count": 100,
            "last_trainable_decided_count": 100,
        }),
        encoding="utf-8",
    )

    result = run_training(calibration_dir)

    assert result["evaluated"] is False
    assert result["new_trainable_decisions"] == 10


def test_training_contract_change_replaces_incompatible_active_mapping(tmp_path: Path):
    calibration_dir = tmp_path / "calibration"
    calibration_dir.mkdir()
    group = calibration_group_key("mlb_new", "MLB", "moneyline")
    records = [
        {
            "id": str(index),
            "date": f"2026-06-{(index % 28) + 1:02d}",
            "raw_probability": 0.55,
            "market_implied_probability": 0.52,
            "raw_units": 1,
            "profit": 0.91 if index % 2 == 0 else -1,
            "outcome": 1 if index % 2 == 0 else 0,
            "calibration_group": group,
            "calibration_eligible": True,
        }
        for index in range(60)
    ]
    (calibration_dir / "outcome_ledger.json").write_text(
        json.dumps({"summary": {"decided_picks": 60}, "records": records}),
        encoding="utf-8",
    )
    (calibration_dir / "active.json").write_text(
        json.dumps({
            "training_contract_version": 1,
            "version": "contaminated-v1",
            "global": {"intercept": 2.5, "slope": 2.5, "samples": 1000},
            "groups": {},
        }),
        encoding="utf-8",
    )
    (calibration_dir / "state.json").write_text(
        json.dumps({
            "training_contract_version": 1,
            "last_evaluated_decided_count": 60,
            "last_trainable_decided_count": 60,
        }),
        encoding="utf-8",
    )

    result = run_training(calibration_dir)
    active = json.loads((calibration_dir / "active.json").read_text())

    assert result["evaluated"] is True
    assert result["training_contract_changed"] is True
    assert active["training_contract_version"] == 2
    assert active["version"] != "contaminated-v1"


def test_champion_challenger_gate_requires_quality_and_roi_safety():
    champion = {"brier_score": 0.25, "calibration_error": 0.1, "log_loss": 0.7, "roi": 0.1}
    good = {"brier_score": 0.24, "calibration_error": 0.08, "log_loss": 0.69, "roi": 0.09}
    bad_roi = {"brier_score": 0.24, "calibration_error": 0.08, "log_loss": 0.69, "roi": 0.0}

    assert should_promote(champion, good)[0] is True
    assert should_promote(champion, bad_roi)[0] is False


def test_downgrade_exempt_model_keeps_model_decision_and_units():
    # mlb_inning settles at an assumed price only; calibration adjusts the
    # displayed probability but must not veto the model's own decision.
    active = {
        "version": "test-v3",
        "minimum_group_samples": 30,
        "global": {"intercept": -3.0, "slope": 1.0, "samples": 100},
        "groups": {},
    }
    payload = {"models": {"mlb_inning": {"picks": [_pick(market="no_run_inning", decision="LEAN", units=0.25)]}}}

    apply_calibration_to_payload(payload, active)
    adjusted = payload["models"]["mlb_inning"]["picks"][0]

    assert adjusted["decision"] == "LEAN"
    assert adjusted["units"] == 0.25
    assert adjusted["probability"] < 0.7
    assert adjusted["calibration"]["applied"] is True
