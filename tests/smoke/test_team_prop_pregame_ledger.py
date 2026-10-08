from __future__ import annotations

import copy
import json

import pytest

from scripts.pick_calibration import build_outcome_ledger
from scripts.team_prop_pregame_ledger import (
    capture_team_prop_pregame_snapshots,
    load_team_prop_pregame_ledger,
    stamp_team_prop_pregame_timing,
    write_team_prop_pregame_ledger,
)


def _payload(
    *,
    probability: float = 0.62,
    pricing_type: str = "market",
    decision: str = "BET",
) -> dict:
    assumed = pricing_type != "market"
    pick = {
        "source": "MLB Model",
        "sport": "MLB",
        "date": "2026-07-10",
        "game_id": "game-1",
        "game_start_time": "2026-07-10T23:00:00Z",
        "matchup": "Away @ Home",
        "home_team": "Home",
        "away_team": "Away",
        "market_type": "h2h",
        "team": "Home",
        "pick": "Home ML (Away @ Home)",
        "probability": probability,
        "decision": decision,
        "units": 1.0,
        "odds": -110,
        "market_pick_prob": 0.52381,
        "market_priced": True,
        "pricing_type": pricing_type,
        "odds_source": "user_assumed_price" if assumed else "sportsbook_observed",
        "market_updated_at": "2026-07-10T19:55:00Z",
        "features": {"home_starter_era": 3.2, "away_starter_era": 4.1},
    }
    return {
        "date": "2026-07-10",
        "generatedAt": "2026-07-10T20:00:00Z",
        "models": {"mlb_new": {"model_version": "mlb-v2", "picks": [pick]}},
    }


def test_capture_certifies_first_publication_and_appends_material_revision(tmp_path):
    payload = _payload()
    assert stamp_team_prop_pregame_timing(payload) == 1

    first = capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)
    repeated = capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)

    assert first == {"added": 1, "unchanged": 0, "team_picks": 1}
    assert repeated == {"added": 0, "unchanged": 1, "team_picks": 1}
    ledger = load_team_prop_pregame_ledger(tmp_path)
    record = ledger["records"][0]
    assert record["certification"] == {
        "status": "certified",
        "reason": "trusted_per_pick_pregame_timestamp",
        "certified": True,
        "immutable": True,
        "pregame": True,
    }
    assert record["financial_eligible"] is True
    assert record["market_benchmark_eligible"] is True
    assert record["calibration_eligible"] is True
    assert record["observed_american_odds"] == -110.0
    assert record["feature_snapshot"] == {
        "home_starter_era": 3.2,
        "away_starter_era": 4.1,
    }

    revised = _payload(probability=0.66)
    stamp_team_prop_pregame_timing(revised, published_at="2026-07-10T20:30:00Z")
    assert capture_team_prop_pregame_snapshots(revised, repo_root=tmp_path)["added"] == 1
    records = load_team_prop_pregame_ledger(tmp_path)["records"]
    assert [item["revision"] for item in records] == [1, 2]
    assert records[1]["supersedes_id"] == records[0]["id"]
    assert records[0]["snapshot_hash"] != records[1]["snapshot_hash"]


def test_capture_keeps_pass_for_probability_evaluation_without_staking(tmp_path):
    payload = _payload(decision="PASS")
    payload["models"]["mlb_new"]["picks"][0]["units"] = 0
    assert stamp_team_prop_pregame_timing(payload) == 1

    summary = capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)

    assert summary == {"added": 1, "unchanged": 0, "team_picks": 1}
    record = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert record["decision"] == "PASS"
    assert record["stake"] == 0


def test_quote_clock_refresh_does_not_append_or_rewrite_prior_evidence(tmp_path):
    payload = _payload()
    stamp_team_prop_pregame_timing(payload)
    pick = payload["models"]["mlb_new"]["picks"][0]
    # Some generators retain their clock inside the raw snapshot as well.
    pick["pregame_snapshot"] = copy.deepcopy(pick)
    capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)
    before = load_team_prop_pregame_ledger(tmp_path)
    revised = copy.deepcopy(payload)
    pick = revised["models"]["mlb_new"]["picks"][0]
    for target in (pick, pick["pregame_snapshot"]):
        target["market_updated_at"] = "2026-07-10T20:25:00Z"
        target["certification_timing"]["published_at"] = "2026-07-10T20:30:00Z"
        target["certification_timing"]["data_as_of"] = "2026-07-10T20:30:00Z"
    assert capture_team_prop_pregame_snapshots(revised, repo_root=tmp_path)["added"] == 0
    assert load_team_prop_pregame_ledger(tmp_path) == before


def test_price_returning_to_an_earlier_value_still_appends_a_revision(tmp_path):
    payload = _payload()
    stamp_team_prop_pregame_timing(payload)
    pick = payload["models"]["mlb_new"]["picks"][0]
    for odds, minute in [(-110, 55), (-115, 56), (-110, 57)]:
        pick.update(odds=odds, market_updated_at=f"2026-07-10T19:{minute}:00Z")
        assert capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)["added"] == 1
    assert [row["price"]["odds"] for row in load_team_prop_pregame_ledger(tmp_path)["records"]] == [-110, -115, -110]
    pick["market_updated_at"] = "2026-07-10T19:58:00Z"
    assert capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)["added"] == 0
@pytest.mark.parametrize("change", ["odds", "probability", "features", "decision", "provenance", "quote_eligibility"])
def test_clock_churn_filter_keeps_material_changes_and_evidence_transitions(tmp_path, change):
    payload = _payload()
    stamp_team_prop_pregame_timing(payload)
    capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)
    before = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    pick = payload["models"]["mlb_new"]["picks"][0]
    pick["market_updated_at"] = "2026-07-10T19:56:00Z"
    if change == "odds":
        pick["odds"] = -115
    elif change == "probability":
        pick["probability"] += 0.01
    elif change == "features":
        pick["features"]["home_starter_era"] += 0.00000001
    elif change == "decision":
        pick["decision"] = "LEAN"
    elif change == "provenance":
        pick["odds_source"] = "another_sportsbook"
    else:
        pick["market_updated_at"] = "2026-07-08T19:55:00Z"
    assert capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)["added"] == 1
    records = load_team_prop_pregame_ledger(tmp_path)["records"]
    assert records[0] == before
    if change == "quote_eligibility":
        assert records[1]["financial_eligibility_reason"] == "stale_quote"


def test_assumed_and_untrusted_rows_never_become_financial_or_calibration_evidence(tmp_path):
    assumed = _payload(pricing_type="user_assumed")
    stamp_team_prop_pregame_timing(assumed)
    capture_team_prop_pregame_snapshots(assumed, repo_root=tmp_path)

    untrusted = _payload(probability=0.64)
    capture_team_prop_pregame_snapshots(untrusted, repo_root=tmp_path)

    records = load_team_prop_pregame_ledger(tmp_path)["records"]
    assert records[0]["certification"]["status"] == "certified"
    assert records[0]["financial_eligible"] is False
    assert records[0]["calibration_eligible"] is False
    assert records[1]["certification"]["status"] == "uncertified"
    assert records[1]["calibration_eligible"] is False


def test_universal_calibration_ledger_uses_only_certified_real_price_team_rows(tmp_path):
    model_dir = tmp_path / "data" / "model_cache"
    model_dir.mkdir(parents=True)
    legacy = _payload()
    legacy["models"]["mlb_new"]["picks"][0]["result"] = "win"
    (model_dir / "2026-07-10.json").write_text(json.dumps(legacy), encoding="utf-8")

    payload = _payload()
    stamp_team_prop_pregame_timing(payload)
    capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)
    certified = load_team_prop_pregame_ledger(tmp_path)
    certified["records"][0]["result"] = "win"
    write_team_prop_pregame_ledger(certified, repo_root=tmp_path)

    ledger = build_outcome_ledger(tmp_path)

    assert ledger["summary"] == {
        "total_picks": 1,
        "decided_picks": 1,
        "trainable_decided_picks": 1,
        "pending_picks": 0,
    }
    record = ledger["records"][0]
    assert record["cache_type"] == "team_prop_pregame_ledger"
    assert record["stake_units"] == 1.0
    assert record["profit"] == 100 / 110
    assert record["calibration_eligible"] is True


def test_assumed_line_source_with_observed_market_odds_stays_financial(tmp_path):
    # The line may come from an assumed ladder (e.g. F5 total 4.5) while the
    # odds at that line are observed sportsbook prices; line-selection
    # provenance must not disqualify an executable observed price.
    payload = _payload()
    pick = payload["models"]["mlb_new"]["picks"][0]
    pick["odds_source"] = "posted_market"
    pick["line_source"] = "user_assumed_f5_total_ladder"
    stamp_team_prop_pregame_timing(payload)
    capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)

    record = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert record["financial_eligible"] is True
    assert record["calibration_eligible"] is True


def test_market_probability_alone_never_proves_an_executable_price(tmp_path):
    # 259 MLS handicap records carried a model-published market_probability
    # and no odds provenance, yet were certified observed_executable_price.
    payload = _payload()
    pick = payload["models"]["mlb_new"]["picks"][0]
    pick.pop("market_priced")
    pick.pop("pricing_type")
    pick.pop("odds_source")
    pick["market_probability"] = 0.55
    stamp_team_prop_pregame_timing(payload)
    capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)

    record = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert record["certification"]["status"] == "certified"
    assert record["financial_eligible"] is False
    assert record["financial_eligibility_reason"] == "unverified_price_provenance"
    assert record["observed_american_odds"] is None
    assert record["calibration_eligible"] is False


def test_unreplaced_placeholder_odds_are_not_financial_but_replaced_ones_are(tmp_path):
    placeholder = _payload()
    pick = placeholder["models"]["mlb_new"]["picks"][0]
    pick["odds"] = 100
    pick["model_assumed_odds"] = 100
    stamp_team_prop_pregame_timing(placeholder)
    capture_team_prop_pregame_snapshots(placeholder, repo_root=tmp_path)

    replaced = _payload(probability=0.6)
    pick = replaced["models"]["mlb_new"]["picks"][0]
    pick["odds"] = -110
    pick["model_assumed_odds"] = -110  # the posted price happened to equal the placeholder
    pick["assumed_odds_replaced"] = True
    stamp_team_prop_pregame_timing(replaced)
    capture_team_prop_pregame_snapshots(replaced, repo_root=tmp_path)

    records = load_team_prop_pregame_ledger(tmp_path)["records"]
    assert records[0]["financial_eligible"] is False
    assert records[0]["financial_eligibility_reason"] == "assumed_or_proxy_price"
    assert records[1]["financial_eligible"] is True
    assert records[1]["observed_american_odds"] == -110.0


def test_frozen_legacy_pick_never_inherits_fresh_bucket_fingerprint():
    from scripts.team_prop_pregame_ledger import _model_version
    bucket = {'prediction_model_version': 'nfl:new-artifact', 'model_version': 'legacy-bucket'}
    assert _model_version('nfl', bucket, {'model_version': 'legacy-pick'}) == 'legacy-pick'
    assert _model_version('nfl', bucket, {'prediction_model_version': 'nfl:old-artifact'}) == 'nfl:old-artifact'
    assert _model_version('nfl', bucket, {}) != 'nfl:new-artifact'
    assert _model_version(
        'nfl',
        {'prediction_model_version': 'nfl:new-artifact', 'model_version': 'nfl_v1_epa_elo_market_anchored'},
        {'prediction_model_version': 'nfl:churn', 'model_version': 'nfl_v1_epa_elo_market_anchored'},
    ) == 'nfl_v1_epa_elo_market_anchored'


def test_stamp_does_not_overwrite_trusted_generation_clock():
    payload = _payload()
    pick = payload["models"]["mlb_new"]["picks"][0]
    pick["certification_timing"] = {
        "trusted": True,
        "published_at": "2026-07-10T18:00:00Z",
        "data_as_of": "2026-07-10T18:00:00Z",
        "source": "nfl-model-generate",
    }
    assert stamp_team_prop_pregame_timing(payload, published_at="2026-07-10T20:42:00Z") == 0
    assert pick["certification_timing"]["published_at"] == "2026-07-10T18:00:00Z"


def test_shadow_decision_is_stored_for_demoted_nfl_rows(tmp_path):
    payload = _payload(decision="PASS")
    nfl_pick = {
        **payload["models"]["mlb_new"]["picks"][0],
        "sport": "NFL",
        "market": "totals",
        "decision": "PASS",
        "units": 0,
        "shadow_decision": "BET",
        "shadow_units": 0.5,
        "source_decision": "BET",
        "model_version": "nfl_v1_epa_elo_market_anchored",
        "prediction_model_version": "nfl:deadbeefdeadbeef",
    }
    payload["models"] = {"nfl": {"picks": [nfl_pick]}}
    stamp_team_prop_pregame_timing(payload)
    capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)
    record = load_team_prop_pregame_ledger(tmp_path)["records"][0]
    assert record["model_key"] == "nfl"
    assert record["model_version"] == "nfl_v1_epa_elo_market_anchored"
    assert record["decision"] == "PASS"
    assert record["raw_decision"] == "BET"
    assert record["shadow_decision"] == "BET"
    assert record["shadow_units"] == 0.5
