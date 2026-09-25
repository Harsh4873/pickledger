from __future__ import annotations

from scripts.model_stake_policy import apply_stake_policy
from scripts.price_clock import observed_quote_timing


def _approval(**overrides):
    approval = {
        "model_key": "mlb_new", "model_version": "mlb:v1", "market": "totals",
        "variant": "base", "approved": True, "frozen_rule": "v1: totals at observed odds",
        "holdout": {
            "unused_during_selection": True,
            "independently_priced_settled": 100,
            "roi": 0.08, "clustered_lower_95": 0.01,
            "calibration_no_material_regression": True,
        },
    }
    approval.update(overrides)
    return approval


def test_policy_demotes_unapproved_pick_and_preserves_shadow_decision():
    pick = {"decision": "BET", "units": 0.5, "market": "totals", "prediction_model_version": "mlb:v1"}
    payload = {"models": {"mlb_new": {"picks": [pick]}}}
    assert apply_stake_policy(payload, approvals={"approvals": []}, model_keys={"mlb_new"}) == 1
    assert pick["decision"] == "PASS" and pick["units"] == 0
    assert pick["shadow_decision"] == "BET" and pick["shadow_units"] == 0.5


def test_policy_requires_exact_version_market_and_valid_holdout():
    pick = {"decision": "LEAN", "units": 0.25, "market": "totals", "prediction_model_version": "mlb:v1"}
    payload = {"models": {"mlb_new": {"picks": [pick]}}}
    invalid = _approval(holdout={**_approval()["holdout"], "clustered_lower_95": -0.01})
    assert apply_stake_policy(payload, approvals={"approvals": [invalid]}, model_keys={"mlb_new"}) == 1
    pick.update(decision="LEAN", units=0.25)
    assert apply_stake_policy(payload, approvals={"approvals": [_approval()]}, model_keys={"mlb_new"}) == 0
    assert pick["decision"] == "LEAN" and pick["units"] == 0.25


def test_policy_preserves_retained_pregame_bet_after_start():
    pick = {
        "decision": "BET", "units": 0.5, "market": "totals",
        "game_start_time": "2026-09-25T19:00:00Z",
        "certification_timing": {"published_at": "2026-09-25T18:00:00Z"},
    }
    payload = {"publishedAt": "2026-09-25T20:00:00Z", "models": {"mlb_new": {"picks": [pick]}}}
    assert apply_stake_policy(payload, approvals={"approvals": []}, model_keys={"mlb_new"}) == 0
    assert pick["decision"] == "BET" and pick["units"] == 0.5
    pick["certification_timing"]["published_at"] = "2026-09-25T20:00:00Z"
    assert apply_stake_policy(payload, approvals={"approvals": []}, model_keys={"mlb_new"}) == 1
    assert pick["decision"] == "PASS" and pick["units"] == 0


def test_price_clock_rejects_missing_late_stale_and_post_start_quotes():
    published = "2026-09-24T20:00:00Z"
    start = "2026-09-24T22:00:00Z"
    assert observed_quote_timing({}, published_at=published, start_at=start) == "missing_quote_timestamp"
    assert observed_quote_timing({"market_updated_at": "2026-09-24T20:06:00Z"}, published_at=published, start_at=start) == "quote_after_publication"
    assert observed_quote_timing({"market_updated_at": "2026-09-23T19:59:00Z"}, published_at=published, start_at=start) == "stale_quote"
    assert observed_quote_timing({"market_updated_at": start}, published_at=published, start_at=start) == "post_start"
    assert observed_quote_timing({"market_updated_at": "2026-09-24T19:59:00Z"}, published_at=published, start_at=start) is None
