from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.model_stake_policy import _approved
from scripts.nfl_staking_approval import (
    evaluate_frozen_holdout,
    load_freeze,
    maybe_write_approval,
)
from scripts.team_prop_pregame_ledger import (
    backfill_team_prop_pregame_from_cache,
    refresh_trusted_publication_clock,
)


ROOT = Path(__file__).resolve().parents[2]
FITTED = "nfl_v1_epa_elo_market_anchored"


def _freeze(**holdout_overrides):
    freeze = json.loads((ROOT / "data" / "calibration" / "nfl_staking_freeze.json").read_text())
    freeze["holdout"] = {**freeze["holdout"], **holdout_overrides}
    return freeze


def _record(index: int, *, result: str = "win") -> dict:
    return {
        "id": f"nfl-{index}",
        "model_key": "nfl",
        "model_version": FITTED,
        "market": "totals",
        "published_at": "2026-09-27T16:00:00Z",
        "data_as_of": "2026-09-27T16:00:00Z",
        "game_start_time": "2026-09-27T20:00:00Z",
        "game_id": f"game-{index}",
        "decision": "PASS",
        "stake": 0,
        "shadow_decision": "BET",
        "shadow_units": 0.5,
        "result": result,
        "raw_probability": 0.60,
        "market_probability": 0.52,
        "financial_eligible": True,
        "certification": {"status": "certified", "certified": True, "immutable": True, "pregame": True},
        "observed_american_odds": -110,
        "odds": -110,
        "pricing_type": "market",
        "odds_source": "posted_market",
        "pregame_snapshot": {
            "decision": "PASS",
            "shadow_decision": "BET",
            "shadow_units": 0.5,
            "odds": -110,
            "pricing_type": "market",
            "odds_source": "posted_market",
            "market_updated_at": "2026-09-27T15:50:00Z",
            "model_version": FITTED,
        },
    }


def test_committed_freeze_matches_fitted_nfl_v1_and_unused_holdout_starts_today():
    freeze = load_freeze()
    assert freeze["fitted_version"] == FITTED
    assert freeze["market"] == "totals"
    assert freeze["holdout"]["unused_during_selection"] is True
    assert freeze["holdout"]["starts_at"] >= "2026-09-27T00:00:00Z"
    assert freeze["selection_window"]["unused_during_selection"] is False
    approvals = json.loads((ROOT / "data" / "calibration" / "staking_approvals.json").read_text())
    assert approvals["approvals"] == []


def test_holdout_below_100_does_not_write_an_approval(tmp_path):
    freeze = _freeze()
    report = evaluate_frozen_holdout({"records": [_record(1), _record(2, result="loss")]}, freeze)
    assert report["independently_priced_settled"] == 2
    assert report["clears_gate"] is False
    assert report["blocker"] == "insufficient_priced_settled"
    policy = tmp_path / "staking_approvals.json"
    policy.write_text(json.dumps({"schema_version": 1, "approvals": []}) + "\n")
    assert maybe_write_approval(freeze, report, policy_path=policy, write=True) is False
    assert json.loads(policy.read_text())["approvals"] == []


def test_holdout_uses_the_captured_price_and_quote_clock():
    row = _record(1)
    # The first publication snapshot precedes the market overlay. The ledger
    # price is the quote that was actually attached to this publication.
    row["pregame_snapshot"].update(
        odds=-110, market_updated_at="2026-09-27T12:00:00Z",
        market_probability=0.40,
    )
    row["price"] = {
        "odds": -125, "pricing_type": "market", "odds_source": "posted_market",
        "market_updated_at": "2026-09-27T15:50:00Z",
        "market_no_vig_selected_probability": 0.53,
    }
    row["observed_american_odds"] = -125
    report = evaluate_frozen_holdout({"records": [row]}, _freeze())
    assert report["independently_priced_settled"] == 1
    assert report["profit_units"] == 0.4
    assert report["observed_market"]["samples"] == 1
    assert report["observed_market"]["brier"] == 0.2209

    row["pregame_snapshot"].update(pricing_type="user_assumed", odds_source="user_assumed")
    replaced = evaluate_frozen_holdout({"records": [row]}, _freeze())
    assert replaced["independently_priced_settled"] == 1
    assert replaced["profit_units"] == 0.4

    # A valid old snapshot cannot certify a newer price with no quote clock.
    row["price"].pop("market_updated_at")
    missing = evaluate_frozen_holdout({"records": [row]}, _freeze())
    assert missing["independently_priced_settled"] == 0
    assert missing["exclusions"] == {"missing_quote_timestamp": 1}


def test_holdout_uses_the_ledger_settlement_when_snapshot_is_pending():
    row = _record(1)
    row["pregame_snapshot"]["result"] = "pending"
    report = evaluate_frozen_holdout({"records": [row]}, _freeze())
    assert report["independently_priced_settled"] == 1
    assert report["pending_actionable"] == 0


def test_stale_or_post_start_price_cannot_supply_paired_calibration():
    row = _record(1)
    row["price"] = {
        "odds": -125, "pricing_type": "market", "odds_source": "posted_market",
        "market_updated_at": "2026-09-27T21:00:00Z",
        "market_no_vig_selected_probability": 0.53,
    }
    row["observed_american_odds"] = -125
    report = evaluate_frozen_holdout({"records": [row]}, _freeze())
    assert report["independently_priced_settled"] == 0
    assert report["model"]["samples"] == 1
    assert report["paired_model"]["samples"] == 0
    assert report["observed_market"]["samples"] == 0
    assert report["exclusions"] == {"post_start": 1}

    row["price"]["market_updated_at"] = "2026-09-27T15:50:00Z"
    row["price"]["odds"] = 0
    row["observed_american_odds"] = 0
    invalid_odds = evaluate_frozen_holdout({"records": [row]}, _freeze())
    assert invalid_odds["paired_model"]["samples"] == 0
    assert invalid_odds["exclusions"] == {"missing_verified_american_price": 1}


def test_holdout_clears_only_with_real_gate_fields(tmp_path):
    freeze = _freeze()
    records = [_record(i) for i in range(100)]
    report = evaluate_frozen_holdout({"records": records}, freeze)
    assert report["independently_priced_settled"] == 100
    assert report["roi"] > 0
    assert report["clustered_lower_95"] > 0
    assert report["calibration_no_material_regression"] is True
    assert report["clears_gate"] is True
    entry_ok = {
        "model_key": "nfl",
        "model_version": FITTED,
        "market": "totals",
        "variant": "base",
        "approved": True,
        "frozen_rule": freeze["frozen_rule"],
        "holdout": {
            "unused_during_selection": True,
            "independently_priced_settled": 100,
            "roi": report["roi"],
            "clustered_lower_95": report["clustered_lower_95"],
            "calibration_no_material_regression": True,
        },
    }
    assert _approved(entry_ok)
    policy = tmp_path / "staking_approvals.json"
    policy.write_text(json.dumps({"schema_version": 1, "approvals": []}) + "\n")
    assert maybe_write_approval(freeze, report, policy_path=policy, write=False) is False
    assert json.loads(policy.read_text())["approvals"] == []
    assert maybe_write_approval(freeze, report, policy_path=policy, write=True) is True
    written = json.loads(policy.read_text())["approvals"]
    assert len(written) == 1
    assert _approved(written[0])
    assert written[0]["model_version"] == FITTED


def test_mined_selection_window_never_counts_as_unused_holdout():
    freeze = _freeze(starts_at="2012-01-01T00:00:00Z", unused_during_selection=False)
    records = [_record(i) for i in range(100)]
    for row in records:
        row["published_at"] = "2024-12-01T18:00:00Z"
        row["data_as_of"] = "2024-12-01T18:00:00Z"
        row["game_start_time"] = "2024-12-01T21:00:00Z"
        row["pregame_snapshot"]["market_updated_at"] = "2024-12-01T17:50:00Z"
    report = evaluate_frozen_holdout({"records": records}, freeze)
    assert report["independently_priced_settled"] == 100
    assert report["clears_gate"] is False
    assert report["blocker"] == "holdout_not_unused"


def test_refresh_clock_advances_only_while_still_pregame():
    pick = {
        "game_start_time": "2026-09-27T20:00:00Z",
        "certification_timing": {
            "trusted": True,
            "published_at": "2026-09-27T16:00:00Z",
            "data_as_of": "2026-09-27T16:00:00Z",
            "source": "nfl-model-generate",
        },
    }
    payload = {"models": {"nfl": {"picks": [pick]}}}
    advanced = refresh_trusted_publication_clock(
        payload, now=datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc)
    )
    assert advanced == 1
    assert pick["certification_timing"]["published_at"].startswith("2026-09-27T17:00")
    assert pick["certification_timing"]["data_as_of"] == "2026-09-27T16:00:00Z"
    started = refresh_trusted_publication_clock(
        payload, now=datetime(2026, 9, 27, 21, 0, tzinfo=timezone.utc)
    )
    assert started == 0
    assert pick["certification_timing"]["published_at"].startswith("2026-09-27T17:00")


def test_nba_retained_clock_is_not_rewritten_by_a_later_run():
    pick = {
        "game_start_time": "2026-09-27T23:00:00Z",
        "certification_timing": {
            "trusted": True,
            "published_at": "2026-09-27T16:00:00Z",
            "data_as_of": "2026-09-27T16:00:00Z",
        },
    }
    payload = {"models": {"nba": {"picks": [pick]}}}
    advanced = refresh_trusted_publication_clock(
        payload,
        now=datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc),
        this_run_generated_at="2026-09-27T17:00:00+00:00",
    )
    assert advanced == 0
    assert pick["certification_timing"]["published_at"] == "2026-09-27T16:00:00Z"
    this_run = {
        "game_start_time": "2026-09-27T23:00:00Z",
        "certification_timing": {
            "trusted": True,
            "published_at": "2026-09-27T16:00:00Z",
            "data_as_of": "2026-09-27T16:00:00Z",
        },
    }
    this_run_payload = {"models": {"nba": {"picks": [this_run]}}}
    assert refresh_trusted_publication_clock(
        this_run_payload,
        now=datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc),
        this_run_generated_at="2026-09-27T16:00:00Z",
    ) == 1
    assert this_run["certification_timing"]["published_at"].startswith("2026-09-27T17:00")


def test_backfill_captures_missing_nfl_cache_rows_only(tmp_path):
    cache = tmp_path / "data" / "model_cache"
    cache.mkdir(parents=True)
    payload = {
        "date": "2026-09-24",
        "generatedAt": "2026-09-24T16:00:00Z",
        "models": {
            "nfl": {
                "picks": [{
                    "sport": "NFL",
                    "date": "2026-09-24",
                    "game_id": "2026_03_ATL_GB",
                    "matchup": "ATL @ GB",
                    "home_team": "GB",
                    "away_team": "ATL",
                    "market": "totals",
                    "pick": "Under 44 (ATL @ GB)",
                    "decision": "PASS",
                    "shadow_decision": "LEAN",
                    "shadow_units": 0.25,
                    "units": 0,
                    "probability": 0.55,
                    "raw_probability": 0.55,
                    "odds": -110,
                    "pricing_type": "market",
                    "odds_source": "nflverse_posted_lines",
                    "market_priced": True,
                    "market_retrieved_at": "2026-09-24T15:50:00Z",
                    "game_start_time": "2026-09-25T00:15:00Z",
                    "model_version": FITTED,
                    "features": {"elo_diff": 1.0, "total_line": 44.0},
                    "certification_timing": {
                        "trusted": True,
                        "published_at": "2026-09-24T16:00:00Z",
                        "data_as_of": "2026-09-24T16:00:00Z",
                        "source": "nfl-model-generate",
                    },
                }]
            },
            "nba": {
                "picks": [{
                    "sport": "NBA",
                    "date": "2026-09-24",
                    "game_id": "nba-1",
                    "matchup": "A @ B",
                    "market": "h2h",
                    "pick": "B ML",
                    "decision": "BET",
                    "units": 1,
                    "probability": 0.6,
                    "odds": -110,
                    "pricing_type": "market",
                    "odds_source": "posted_market",
                    "market_priced": True,
                    "market_updated_at": "2026-09-24T15:00:00Z",
                    "game_start_time": "2026-09-24T23:00:00Z",
                }]
            },
        },
    }
    (cache / "2026-09-24.json").write_text(json.dumps(payload))
    summary = backfill_team_prop_pregame_from_cache(cache, repo_root=tmp_path, model_keys={"nfl"})
    assert summary["added"] == 1
    ledger = json.loads((tmp_path / "data" / "calibration" / "team_prop_pregame_ledger.json").read_text())
    assert [row["model_key"] for row in ledger["records"]] == ["nfl"]
    assert ledger["records"][0]["shadow_decision"] == "LEAN"
    repeated = backfill_team_prop_pregame_from_cache(cache, repo_root=tmp_path, model_keys={"nfl"})
    assert repeated["added"] == 0
