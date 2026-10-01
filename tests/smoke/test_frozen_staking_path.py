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
    if freeze["model_key"] == "mls":
        snapshot["line"] = -0.5 if overrides.get("market", freeze["markets"][0]) == "spread" else 2.5
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
    assert all(m["status"] == "awaiting_holdout_evidence" and m["independently_priced_settled"] == 0
               for m in report["markets"])
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
    if model == "mls":
        pick["line"] = 2.25
        apply_stake_policy(payload, model_keys={model}, approvals=approvals)
        assert pick["decision"] == "PASS" and pick["units"] == 0
        assert pick["calibration_excluded"] is True
        pick["line"] = 2.5
        apply_stake_policy(payload, model_keys={model}, approvals=approvals)
        assert pick["decision"] == "BET"
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


def test_frozen_candidate_uses_first_eligible_publication_by_instant():
    freeze = load_freeze("mls")
    wrong_version = record(freeze, model_version="older-version")
    earlier = record(freeze, published_at="2026-10-02T17:00:00+02:00")
    later = record(freeze, published_at="2026-10-02T16:00:00Z", result="loss")
    for row in (wrong_version, earlier, later):
        row["pregame_snapshot"]["market_updated_at"] = "2026-10-02T14:50:00Z"
    report = evaluate({"records": [wrong_version, later, earlier]}, freeze)["markets"][0]
    assert report["independently_priced_settled"] == 1
    assert report["profit_units"] > 0  # The 15:00Z win precedes the 16:00Z loss.
    assert report["candidate_exclusions"] == {
        "duplicate_event_market": 1, "fitted_version_mismatch": 1,
    }


def test_candidate_exclusions_are_specific_to_each_market():
    freeze = load_freeze("mls")
    earlier = record(freeze, 1, published_at="2026-09-30T12:00:00Z")
    later = record(freeze, 2, market="spread")
    reports = evaluate({"records": [earlier, later]}, freeze)["markets"]
    assert reports[0]["candidate_exclusions"] == {"before_holdout_start": 1}
    assert reports[1]["candidate_exclusions"] == {}


@pytest.mark.parametrize("market,line", [("spread", -0.25), ("total", 2.25)])
def test_mls_quarter_lines_cannot_be_graded_as_full_binary_settlements(market, line):
    from scripts.auto_grade_picks import _pending_certified_team_prop_candidate
    from scripts.pick_calibration import _certified_team_record
    from scripts.settlement_support import binary_settlement_supported

    freeze = load_freeze("mls")
    quarter = record(freeze, market=market, result="pending")
    label = f"Home {line:+g} (Away @ Home)" if market == "spread" else f"Over {line:g} (Away @ Home)"
    quarter["pregame_snapshot"].update(line=line, pick=label,
                                        date="2026-10-02", sport="MLS")
    assert binary_settlement_supported(quarter) is False
    assert _pending_certified_team_prop_candidate(quarter) is None

    # A historic full-win label on the same fractional wager is also excluded
    # from the frozen financial and calibration gates.
    quarter["result"] = "win"
    quarter["calibration_eligible"] = True
    quarter["decision"] = "BET"
    assert _certified_team_record(quarter) is None
    report = next(row for row in evaluate({"records": [quarter]}, freeze)["markets"] if row["market"] == market)
    assert report["independently_priced_settled"] == 0
    assert report["candidate_exclusions"] == {"unsupported_fractional_settlement": 1}

    half = record(freeze, 2, market=market)
    report = next(row for row in evaluate({"records": [quarter, half]}, freeze)["markets"] if row["market"] == market)
    assert report["independently_priced_settled"] == 1


def test_no_vig_benchmark_is_retained_in_canonical_price_fields(tmp_path):
    from scripts.team_prop_pregame_ledger import capture_team_prop_pregame_snapshots, load_team_prop_pregame_ledger

    freeze = load_freeze("mls")
    row = record(freeze)
    pick = {**row["pregame_snapshot"], "game_id": "priced-game", "date": "2026-10-02",
            "game_start_time": row["game_start_time"], "market": "total", "pick": "Over 2.5",
            "line": 2.5, "decision": "PASS", "shadow_decision": "BET", "shadow_units": .5,
            "probability": .6, "market_priced": True,
            "certification_timing": {"trusted": True, "published_at": row["published_at"],
                                     "data_as_of": row["published_at"], "source": "mls-model-generate"}}
    capture_team_prop_pregame_snapshots({"date": "2026-10-02", "models": {"mls": {"picks": [pick]}}}, repo_root=tmp_path)
    saved = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert saved["price"]["market_no_vig_selected_probability"] == .5
    assert saved["market_probability"] == .5


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


def test_mls_final_approval_controls_calibration_after_earlier_snapshot_flag(tmp_path):
    from scripts.team_prop_pregame_ledger import (
        capture_team_prop_pregame_snapshots, load_team_prop_pregame_ledger, write_team_prop_pregame_ledger,
    )
    from scripts.pick_calibration import _certified_team_record, build_outcome_ledger

    freeze = load_freeze("mls")
    row = record(freeze)
    pick = {**row["pregame_snapshot"], "game_id": "approved-game", "date": "2026-10-02",
            "game_start_time": row["game_start_time"], "market": "total", "pick": "Over 2.5",
            "line": 2.5, "decision": "BET", "units": .5, "probability": .6,
            "shadow_decision": "BET", "shadow_units": .5, "calibration_excluded": False,
            "market_priced": True,
            "certification_timing": {"trusted": True, "published_at": row["published_at"],
                                     "data_as_of": row["published_at"], "source": "mls-model-generate"}}
    pick["pregame_snapshot"] = {**pick, "calibration_excluded": True}
    capture_team_prop_pregame_snapshots({"date": "2026-10-02", "models": {"mls": {"picks": [pick]}}}, repo_root=tmp_path)
    saved = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert saved["pregame_snapshot"]["calibration_excluded"] is True
    assert saved["calibration_eligible"] is True
    assert _certified_team_record(saved) is not None
    ledger = load_team_prop_pregame_ledger(tmp_path)
    ledger["records"][0]["result"] = "win"
    write_team_prop_pregame_ledger(ledger, repo_root=tmp_path)
    calibration = build_outcome_ledger(tmp_path)
    assert calibration["summary"]["trainable_decided_picks"] == 1
    assert calibration["records"][0]["model_key"] == "mls"
