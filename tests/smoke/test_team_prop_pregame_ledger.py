from __future__ import annotations

import json

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


def test_capture_leaves_pass_only_in_raw_team_cache_diagnostics(tmp_path):
    payload = _payload(decision="PASS")
    assert stamp_team_prop_pregame_timing(payload) == 1

    summary = capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)

    assert summary == {"added": 0, "unchanged": 0, "team_picks": 0}
    assert payload["models"]["mlb_new"]["picks"][0]["decision"] == "PASS"
    assert load_team_prop_pregame_ledger(tmp_path)["records"] == []


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
