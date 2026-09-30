"""Synthetic fixtures exercise the gate; none are written to production data."""
import json

import pytest

from scripts.frozen_staking_candidate import load_freeze, candidate_fingerprint
from scripts.frozen_staking_approval import evaluate, write_reviewed
from scripts.model_stake_policy import apply_stake_policy


def record(freeze, index=0, **overrides):
    snapshot = {
        "staking_candidate_fingerprint": freeze["candidate_fingerprint"],
        "model_version": freeze["fitted_version"],
        "market_updated_at": "2026-10-02T15:50:00Z",
        "pricing_type": "market", "odds_source": "posted_market", "odds": -110,
        "market_no_vig_selected_probability": .5,
    }
    return {
        "id": str(index), "game_id": str(index), "model_key": freeze["model_key"],
        "model_version": freeze["fitted_version"], "market": freeze["markets"][0],
        "published_at": "2026-10-02T16:00:00Z", "game_start_time": "2026-10-02T20:00:00Z",
        "pregame_snapshot": snapshot, "decision": "PASS", "stake": 0,
        "shadow_decision": "BET", "shadow_units": .5, "raw_probability": .6,
        "result": "win", "financial_eligible": True, "observed_american_odds": -110,
        "certification": {"status": "certified", "certified": True, "immutable": True, "pregame": True},
        **overrides,
    }


@pytest.mark.parametrize("model", ["nhl", "mls"])
def test_empty_prospective_holdout_never_writes(model, tmp_path):
    freeze = load_freeze(model)
    assert candidate_fingerprint(freeze) == freeze["candidate_fingerprint"]
    policy = tmp_path / "approvals.json"
    policy.write_text('{"schema_version":1,"approvals":[]}')
    report = evaluate({"records": []}, freeze)
    assert all(m["blocker"] == "insufficient_priced_settled" and m["roi"] is None for m in report["markets"])
    assert not write_reviewed(freeze, report, policy)
    assert json.loads(policy.read_text())["approvals"] == []


@pytest.mark.parametrize("model", ["nhl", "mls"])
def test_evidence_and_exact_approval_lifecycle(model, tmp_path):
    freeze = load_freeze(model)
    report = evaluate({"records": [record(freeze, i) for i in range(100)]}, freeze)
    assert report["markets"][0]["clears_gate"]
    policy = tmp_path / "approvals.json"
    policy.write_text('{"schema_version":1,"approvals":[]}')
    assert write_reviewed(freeze, report, policy)
    approvals = json.loads(policy.read_text())
    pick = {
        **record(freeze)["pregame_snapshot"], "market": freeze["markets"][0],
        "decision": "BET", "units": .5, "game_start_time": "2026-10-02T20:00:00Z",
        "prediction_model_version": "unrelated-serving-hash", "market_priced": True,
    }
    payload = {"publishedAt": "2026-10-02T16:00:00Z", "models": {model: {"picks": [pick]}}}
    apply_stake_policy(payload, model_keys={model}, approvals={"approvals": []})
    assert pick["decision"] == "PASS" and pick["shadow_units"] == .5
    apply_stake_policy(payload, model_keys={model}, approvals={"approvals": []})
    assert pick["shadow_units"] == .5
    apply_stake_policy(payload, model_keys={model}, approvals=approvals)
    assert pick["decision"] == "BET" and pick["units"] == .5
    assert pick["calibration_excluded"] is False
    pick["market"] = "unapproved_market"
    apply_stake_policy(payload, model_keys={model}, approvals=approvals)
    assert pick["decision"] == "PASS"
    pick["market"] = freeze["markets"][0]
    pick["model_version"] = "different_fitted_version"
    apply_stake_policy(payload, model_keys={model}, approvals=approvals)
    assert pick["decision"] == "PASS"
    pick["model_version"] = freeze["fitted_version"]
    pick["staking_candidate_fingerprint"] = "changed"
    apply_stake_policy(payload, model_keys={model}, approvals=approvals)
    assert pick["decision"] == "PASS" and pick["units"] == 0
    pick["staking_candidate_fingerprint"] = freeze["candidate_fingerprint"]
    pick["odds"] = None
    apply_stake_policy(payload, model_keys={model}, approvals=approvals)
    assert pick["decision"] == "PASS" and pick["units"] == 0


@pytest.mark.parametrize("model", ["nhl", "mls"])
def test_duplicates_wrong_versions_and_unpriced_rows_cannot_clear(model):
    freeze = load_freeze(model)
    repeated = [record(freeze) for _ in range(120)]
    report = evaluate({"records": repeated}, freeze)["markets"][0]
    assert report["independently_priced_settled"] == 1
    assert not report["clears_gate"]
    for change in [{"financial_eligible": False}, {"published_at": "2026-09-30T12:00:00Z"},
                   {"certification": {"status": "uncertified"}}]:
        report = evaluate({"records": [record(freeze, i, **change) for i in range(100)]}, freeze)["markets"][0]
        assert report["independently_priced_settled"] == 0
        assert not report["clears_gate"]
    rows = [record(freeze, i) for i in range(100)]
    for r in rows:
        r["pregame_snapshot"]["market_updated_at"] = "2026-10-02T21:00:00Z"
    report = evaluate({"records": rows}, freeze)["markets"][0]
    assert report["paired_model"]["samples"] == 0


def test_regression_and_used_selection_window_block_approval():
    freeze = load_freeze("mls")
    rows = [record(freeze, i, raw_probability=.1) for i in range(100)]
    report = evaluate({"records": rows}, freeze)["markets"][0]
    assert report["blocker"] == "calibration_regression_or_unproven"
    freeze["holdout"]["unused_during_selection"] = False
    report = evaluate({"records": [record(freeze, i) for i in range(100)]}, freeze)["markets"][0]
    assert report["blocker"] == "holdout_not_unused"


@pytest.mark.parametrize("approved", [False, True])
def test_mls_calibration_ledger_eligibility_tracks_policy(approved, tmp_path):
    from scripts.team_prop_pregame_ledger import capture_team_prop_pregame_snapshots, load_team_prop_pregame_ledger
    from scripts.pick_calibration import _certified_team_record
    freeze = load_freeze("mls")
    row = record(freeze)
    pick = {**row["pregame_snapshot"], "game_id": "mls-game", "date": "2026-10-02",
            "game_start_time": row["game_start_time"], "market": "total", "pick": "Over 2.5",
            "decision": "BET" if approved else "PASS", "units": .5 if approved else 0,
            "shadow_decision": "BET", "shadow_units": .5, "probability": .6,
            "calibration_excluded": not approved, "prediction_model_version": "serving-hash",
            "certification_timing": {"trusted": True, "published_at": row["published_at"],
                                     "data_as_of": row["published_at"], "source": "mls-model-generate"}}
    capture_team_prop_pregame_snapshots({"date": "2026-10-02", "models": {"mls": {"picks": [pick]}}}, repo_root=tmp_path)
    saved = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert saved["model_version"] == freeze["fitted_version"]
    assert saved["calibration_eligible"] is approved
    assert (_certified_team_record(saved) is not None) is approved
