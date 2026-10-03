"""Synthetic fixtures exercise the gate; none are written to production data."""
import json

import pytest

from scripts.frozen_staking_candidate import load_freeze as load_committed_freeze, candidate_fingerprint
from scripts.frozen_staking_approval import evaluate, write_reviewed
from scripts.model_stake_policy import apply_stake_policy


def load_freeze(model):
    """Use a synthetic clock for fixture games, preserving real content identity."""
    freeze = load_committed_freeze(model)
    freeze["frozen_at"] = "2026-09-30T19:00:00Z"
    freeze["holdout"]["starts_at"] = "2026-10-01T00:00:00Z"
    return freeze


@pytest.mark.parametrize("model", ["nhl", "mls"])
def test_committed_freeze_matches_artifact_and_has_valid_window(model):
    from scripts.price_clock import aware_time

    freeze = load_committed_freeze(model)
    assert candidate_fingerprint(freeze) == freeze["candidate_fingerprint"]
    assert freeze["frozen_rule"].endswith(":" + freeze["candidate_fingerprint"])
    assert aware_time(freeze["frozen_at"]) < aware_time(freeze["holdout"]["starts_at"])


def test_mls_retrain_cannot_reuse_previous_freeze_or_observed_window(tmp_path):
    from scripts.frozen_staking_candidate import ROOT, candidate_matches

    freeze = load_committed_freeze("mls")
    old = json.loads((ROOT / freeze["supersedes"]).read_text())
    assert old["candidate_fingerprint"] != freeze["candidate_fingerprint"]
    assert not candidate_matches(record(old)["pregame_snapshot"], "mls")
    report = evaluate({"records": [record(freeze, i) for i in range(100)]}, freeze)
    assert all(row["independently_priced_settled"] == 0 for row in report["markets"])
    assert not write_reviewed(freeze, report, tmp_path / "approvals.json")
    with pytest.raises(ValueError, match="candidate changed"):
        write_reviewed(old, report, tmp_path / "approvals.json")


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
def test_first_graded_shadow_publication_starts_accruing(model, monkeypatch, tmp_path):
    from scripts import auto_grade_picks
    from scripts.team_prop_pregame_ledger import (
        capture_team_prop_pregame_snapshots, load_team_prop_pregame_ledger,
    )

    freeze = load_freeze(model)
    source = record(freeze)
    market = freeze["markets"][0]
    pick = {**source["pregame_snapshot"],
            "game_id": f"{model}-holdout-game", "date": "2026-10-02",
            "game_start_time": source["game_start_time"], "sport": model.upper(),
            "market": market, "model_version": freeze["fitted_version"],
            "pick": ("Over 2.5 (Away @ Home)" if model == "mls"
                     else "Home ML (Away @ Home)"),
            "decision": "PASS", "units": 0, "shadow_decision": "BET",
            "shadow_units": .5, "raw_probability": .6, "probability": .6,
            "market_priced": True,
            "certification_timing": {"trusted": True, "published_at": source["published_at"],
                                     "data_as_of": source["published_at"]}}
    # An old result marker in the pregame image must not override the grader's
    # later, top-level ledger settlement.
    pick["pregame_snapshot"] = {**pick, "result": "pending"}
    capture_team_prop_pregame_snapshots(
        {"date": "2026-10-02", "models": {model: {"picks": [pick]}}}, repo_root=tmp_path)
    before = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert before["certification"]["status"] == "certified"
    assert before["financial_eligible"] is True
    assert before["observed_american_odds"] == -110
    assert before["price"]["market_updated_at"] == "2026-10-02T15:50:00Z"
    assert before["result"] == "pending"

    def official_grade(candidates, _existing, _year):
        assert len(candidates) == 1
        return {"graded": {candidates[0]["id"]: "win"}, "startTimes": {}}

    monkeypatch.setattr(auto_grade_picks.pickgrader_server, "auto_grade", official_grade)
    assert auto_grade_picks.grade_certified_team_prop_snapshots(tmp_path)["graded"] == 1
    ledger = load_team_prop_pregame_ledger(tmp_path)
    report = next(row for row in evaluate(ledger, freeze)["markets"] if row["market"] == market)
    assert ledger["records"][0]["result"] == "win"
    assert ledger["records"][0]["pregame_snapshot"]["result"] == "pending"
    assert report["status"] == "accruing"
    assert report["independently_priced_settled"] == 1
    assert report["pending_actionable"] == 0
    assert report["clears_gate"] is False


@pytest.mark.parametrize("model", ["nhl", "mls"])
def test_frozen_holdout_uses_the_ledger_price_clock_and_odds(model):
    freeze = load_freeze(model)
    row = record(freeze)
    row["pregame_snapshot"]["market_updated_at"] = "2026-09-30T12:00:00Z"
    row["price"] = {
        "odds": -125, "pricing_type": "market", "odds_source": "posted_market",
        "market_updated_at": "2026-10-02T15:50:00Z",
        "market_no_vig_selected_probability": 0.53,
    }
    row["observed_american_odds"] = -125
    row["market_probability"] = 0.53
    report = evaluate({"records": [row]}, freeze)["markets"][0]
    assert report["independently_priced_settled"] == 1
    assert report["profit_units"] == 0.4
    assert report["observed_market"]["brier"] == 0.2209

    row["price"]["market_updated_at"] = "2026-10-02T21:00:00Z"
    late = evaluate({"records": [row]}, freeze)["markets"][0]
    assert late["independently_priced_settled"] == 0
    assert late["candidate_exclusions"] == {"invalid_quote_clock": 1}

    untimed = record(freeze)
    untimed["price"] = {
        "odds": -125, "pricing_type": "market", "odds_source": "posted_market",
        "market_no_vig_selected_probability": 0.53,
    }
    untimed["observed_american_odds"] = -125
    untimed["pregame_snapshot"]["odds"] = -110
    borrowed = evaluate({"records": [untimed]}, freeze)["markets"][0]
    assert borrowed["independently_priced_settled"] == 1
    assert borrowed["profit_units"] == 0.4
    assert borrowed["candidate_exclusions"] == {}


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


@pytest.mark.parametrize("market,line,label", [
    ("total", 2.25, "Over 2.25 (Away @ Home)"),
    ("total", 2.75, "Under 2.75 (Away @ Home)"),
    ("spread", -0.25, "Home -0.25 (Away @ Home)"),
    ("spread", 0.75, "Home +0.75 (Away @ Home)"),
])
def test_mls_fractional_cache_result_is_retracted_as_research_only(monkeypatch, market, line, label):
    from scripts.auto_grade_picks import grade_payload

    def unexpected_grade(*_args, **_kwargs):
        raise AssertionError("fractional MLS row reached binary grader")

    monkeypatch.setattr("scripts.auto_grade_picks.pickgrader_server.auto_grade", unexpected_grade)
    pick = {"sport": "MLS", "market": market, "line": line, "pick": label,
            "decision": "PASS", "result": "win", "calibration_excluded": False}
    payload = {"date": "2026-10-02", "models": {"mls": {"picks": [pick]}}}
    assert grade_payload(payload) > 0
    assert pick["result"] == "pending"
    assert pick["grade_supported"] is False
    assert pick["settlement_exclusion_reason"] == "unsupported_fractional_settlement"
    assert pick["calibration_excluded"] is True
    assert "Research only" in pick["grade_note"]


def test_mls_line_conflict_cannot_pass_binary_settlement_gate():
    from scripts.settlement_support import settlement_exclusion_reason

    row = {"model_key": "mls", "market": "total", "line": 2.5,
           "pick": "Over 2.25 (Away @ Home)"}
    assert settlement_exclusion_reason(row) == "conflicting_mls_line"


def test_fractional_ledger_records_explicit_calibration_exclusion(tmp_path):
    from scripts.team_prop_pregame_ledger import capture_team_prop_pregame_snapshots, load_team_prop_pregame_ledger

    freeze = load_freeze("mls")
    row = record(freeze)
    pick = {**row["pregame_snapshot"], "game_id": "quarter-game", "date": "2026-10-02",
            "game_start_time": row["game_start_time"], "market": "total",
            "pick": "Over 2.25 (Away @ Home)", "line": 2.25, "sport": "MLS",
            "decision": "BET", "units": .5, "probability": .6,
            "calibration_excluded": False, "market_priced": True,
            "certification_timing": {"trusted": True, "published_at": row["published_at"],
                                     "data_as_of": row["published_at"]}}
    capture_team_prop_pregame_snapshots(
        {"date": "2026-10-02", "models": {"mls": {"picks": [pick]}}}, repo_root=tmp_path)
    saved = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert saved["settlement_exclusion_reason"] == "unsupported_fractional_settlement"
    assert saved["calibration_eligible"] is False
    assert saved["calibration_eligibility_reason"] == "unsupported_fractional_settlement"


def test_fractional_ledger_retracts_prior_binary_result(tmp_path):
    from scripts.auto_grade_picks import grade_certified_team_prop_snapshots
    from scripts.team_prop_pregame_ledger import load_team_prop_pregame_ledger, write_team_prop_pregame_ledger

    quarter = record(load_freeze("mls"), result="win", calibration_eligible=True)
    quarter["pregame_snapshot"].update(
        line=2.25, pick="Over 2.25 (Away @ Home)", sport="MLS", date="2026-10-02")
    write_team_prop_pregame_ledger({"records": [quarter]}, repo_root=tmp_path)
    summary = grade_certified_team_prop_snapshots(tmp_path)
    saved = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert summary["candidates"] == 0
    assert summary["changed"] is True
    assert saved["result"] == "pending"
    assert saved["calibration_eligible"] is False
    assert saved["settlement_exclusion_reason"] == "unsupported_fractional_settlement"


def test_no_vig_benchmark_is_retained_in_canonical_price_fields(tmp_path):
    from scripts.team_prop_pregame_ledger import capture_team_prop_pregame_snapshots, load_team_prop_pregame_ledger

    freeze = load_freeze("mls")
    row = record(freeze)
    pick = {**row["pregame_snapshot"], "game_id": "priced-game", "date": "2026-10-02",
            "game_start_time": row["game_start_time"], "market": "total", "pick": "Over 2.5",
            "line": 2.5, "decision": "PASS", "shadow_decision": "BET", "shadow_units": .5,
            "probability": .6, "market_priced": True,
            "certification_timing": {"trusted": True, "published_at": row["published_at"],
                                     "data_as_of": row["published_at"], "source": "mls-model-generate"},
            "market_odds_captured_at": "2026-10-02T15:50:00Z"}
    capture_team_prop_pregame_snapshots({"date": "2026-10-02", "models": {"mls": {"picks": [pick]}}}, repo_root=tmp_path)
    saved = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert saved["price"]["market_no_vig_selected_probability"] == .5
    assert saved["price"]["market_odds_captured_at"] == "2026-10-02T15:50:00Z"
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


def _mls_approval(freeze, market):
    return {
        "model_key": "mls",
        "model_version": freeze["fitted_version"],
        "market": market,
        "variant": "base",
        "approved": True,
        "frozen_rule": freeze["frozen_rule"],
        "holdout": {
            "unused_during_selection": True,
            "independently_priced_settled": 100,
            "roi": 0.1,
            "clustered_lower_95": 0.01,
            "calibration_no_material_regression": True,
        },
    }


@pytest.mark.parametrize("market,line,label", [
    ("total", 2.25, "Over 2.25 (Away @ Home)"),
    ("total", 2.75, "Under 2.75 (Away @ Home)"),
    ("spread", -0.25, "Home -0.25 (Away @ Home)"),
    ("spread", 0.75, "Home +0.75 (Away @ Home)"),
])
def test_approved_mls_quarter_lines_cannot_stake(market, line, label):
    """A .25/.75 total or handicap splits the stake across two lines, so a
    final score can be a half win or half loss. A forged market approval
    must not turn that into a binary BET, even from the pick text alone."""
    freeze = load_freeze("mls")
    base = record(freeze)["pregame_snapshot"]
    pick = {
        **base, "market": market, "line": line, "pick": label,
        "decision": "BET", "units": 0.5, "market_priced": True,
        "game_start_time": "2026-10-02T20:00:00Z",
    }
    text_only = {key: value for key, value in pick.items() if key != "line"}
    approvals = {"approvals": [_mls_approval(freeze, market)]}
    payload = {"publishedAt": "2026-10-02T16:00:00Z", "models": {"mls": {"picks": [pick, text_only]}}}
    apply_stake_policy(payload, model_keys={"mls"}, approvals=approvals)
    assert pick["decision"] == "PASS" and pick["units"] == 0
    assert pick["calibration_excluded"] is True
    assert text_only["decision"] == "PASS" and text_only["units"] == 0

    half_line = 2.5 if market == "total" else -0.5
    half_label = "Over 2.5 (Away @ Home)" if market == "total" else "Home -0.5 (Away @ Home)"
    half = {
        **base, "market": market, "line": half_line, "pick": half_label,
        "decision": "BET", "units": 0.5, "market_priced": True,
        "game_start_time": "2026-10-02T20:00:00Z",
    }
    apply_stake_policy(
        {"publishedAt": "2026-10-02T16:00:00Z", "models": {"mls": {"picks": [half]}}},
        model_keys={"mls"}, approvals=approvals,
    )
    assert half["decision"] == "BET" and half["units"] == 0.5


def test_nhl_holdout_uses_market_retrieved_at_and_rejects_a_missing_clock():
    freeze = load_freeze("nhl")
    row = record(freeze)
    row["pregame_snapshot"].pop("market_updated_at", None)
    row["price"] = {
        "odds": -110,
        "pricing_type": "market",
        "odds_source": "draftkings",
        "market_retrieved_at": "2026-10-02T15:50:00Z",
        "market_no_vig_selected_probability": 0.5,
    }
    row["observed_american_odds"] = -110
    priced = evaluate({"records": [row]}, freeze)["markets"][0]
    assert priced["independently_priced_settled"] == 1

    row["price"].pop("market_retrieved_at")
    missed = evaluate({"records": [row]}, freeze)["markets"][0]
    assert missed["independently_priced_settled"] == 0
    assert missed["candidate_exclusions"] == {"invalid_quote_clock": 1}
