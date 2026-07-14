from __future__ import annotations

import json

from scripts.team_prop_model_evaluator import evaluate_team_prop_ledger, load_ledger


def _certification(*, financial: bool = False, benchmark: bool = False) -> dict:
    return {
        "certified": True,
        "immutable": True,
        "pregame": True,
        "financial_eligible": financial,
        "market_benchmark_eligible": benchmark,
    }


def _record(**overrides) -> dict:
    record = {
        "snapshot_id": "record-1",
        "model_key": "fifa_world_cup",
        "model_version": "fifa-v1",
        "market": "moneyline",
        "probability": 0.70,
        "result": "win",
        "snapshot_at": "2026-06-10T15:00:00Z",
        "decision": "BET",
        "units": 1.0,
        "certification": _certification(financial=True, benchmark=True),
        "observed_american_odds": -110,
        "market_probability": 0.52381,
        "price_source": "sportsbook_observed",
        "home_unit_ratings": {"attack": 80},
        "away_unit_ratings": {"attack": 74},
        "home_tournament_form": {"games": 2},
        "away_tournament_form": {"games": 2},
        "venue_profile": {"games": 3},
        "raw_projected_home_goals": 1.6,
        "market_total_line": 2.5,
    }
    record.update(overrides)
    return record


def test_evaluator_segments_models_versions_and_markets_with_reproducible_metrics():
    fifa_total = _record(
        snapshot_id="fifa-total",
        market="total",
        probability=0.25,
        result="loss",
        snapshot_at="2026-06-11T15:00:00Z",
        observed_american_odds=120,
        market_probability=0.454545,
    )
    fifa_new_version = _record(
        snapshot_id="fifa-v2",
        model_version="fifa-v2",
        probability=0.60,
        result="win",
        snapshot_at="2026-06-12T15:00:00Z",
    )
    payload = {"schema_version": 1, "records": [_record(), fifa_total, fifa_new_version]}

    report = evaluate_team_prop_ledger(payload, bins=2)

    assert report["record_quality"]["certified_evaluable_records"] == 3
    assert report["overall"]["model_metrics"] == {
        "settled_records": 3,
        "wins": 2,
        "losses": 1,
        "hit_rate": 0.666667,
        "mean_probability": 0.516667,
        "brier_score": 0.104167,
        "log_loss": 0.385061,
    }
    assert {(item["model_key"], item["model_version"], item["market"]) for item in report["segments"]} == {
        ("fifa_world_cup", "fifa-v1", "moneyline"),
        ("fifa_world_cup", "fifa-v1", "total"),
        ("fifa_world_cup", "fifa-v2", "moneyline"),
    }
    assert report["overall"]["calibration"]["bins"][0]["records"] == 1
    assert report["overall"]["calibration"]["bins"][1]["records"] == 2
    roi = report["overall"]["real_price_roi"]
    assert roi["priced_settled_actionable_records"] == 3
    assert roi["stake_units"] == 3.0
    assert roi["profit_units"] == 0.818182
    assert roi["roi"] == 0.272727
    benchmark = report["overall"]["market_benchmark"]
    assert benchmark["priced_or_observed_records"] == 3
    assert benchmark["probability_sources"] == {"observed_american_odds": 3}


def test_evaluator_accepts_canonical_status_certification_and_nested_price():
    record = _record(
        certification={"status": "certified"},
        observed_american_odds=None,
        market_probability=None,
        financial_eligible=True,
        market_benchmark_eligible=True,
        price={"odds": -120, "pricing_type": "market", "odds_source": "sportsbook_observed"},
    )

    report = evaluate_team_prop_ledger({"records": [record]})

    assert report["record_quality"]["certified_evaluable_records"] == 1
    assert report["overall"]["market_benchmark"]["probability_sources"] == {
        "observed_american_odds": 1,
    }


def test_evaluator_never_uses_assumed_or_proxy_prices_as_financial_evidence():
    assumed_f5 = _record(
        snapshot_id="assumed-f5",
        model_key="mlb_first_five",
        model_version="f5-v1",
        market="f5_total",
        probability=0.60,
        result="win",
        certification=_certification(financial=True, benchmark=True),
        observed_american_odds=-110,
        market_probability=0.52381,
        price_source="user_assumed_f5_total_4.5",
        pricing_type="user_assumed",
    )
    unpriced_summer = _record(
        snapshot_id="unpriced-summer",
        model_key="nba_summer",
        model_version="nba_summer_v1.0.0",
        market="h2h",
        probability=0.61,
        result="loss",
        certification=_certification(),
        odds=None,
        market_probability=None,
        price_source="unpriced",
    )
    report = evaluate_team_prop_ledger({"records": [assumed_f5, unpriced_summer]})

    roi = report["overall"]["real_price_roi"]
    assert roi["priced_settled_actionable_records"] == 0
    assert roi["roi"] is None
    assert roi["excluded"] == {
        "assumed_or_proxy_price": 1,
        "not_explicitly_financial_eligible": 1,
    }
    assert report["overall"]["market_benchmark"]["priced_or_observed_records"] == 0


def test_evaluator_uses_only_latest_certified_revision_per_stable_market_slot():
    first = _record(
        snapshot_id="revision-1",
        stable_id="same-game-market",
        revision=1,
        probability=0.80,
        result="loss",
    )
    latest = _record(
        snapshot_id="revision-2",
        stable_id="same-game-market",
        revision=2,
        probability=0.60,
        result="win",
        snapshot_at="2026-06-10T16:00:00Z",
    )

    report = evaluate_team_prop_ledger({"records": [first, latest]})

    assert report["record_quality"]["certified_revision_records"] == 2
    assert report["record_quality"]["certified_evaluable_records"] == 1
    assert report["record_quality"]["superseded_revisions_excluded"] == 1
    assert report["overall"]["model_metrics"]["settled_records"] == 1
    assert report["overall"]["model_metrics"]["mean_probability"] == 0.6


def test_evaluator_rejects_uncertified_rows_and_reports_feature_snapshot_gaps():
    certified = _record(
        snapshot_id="f5-with-features",
        model_key="mlb_first_five",
        model_version="f5-v1",
        market="f5_side",
        pregame_snapshot={
            "features": {
                "away_offense": {"runs": 2.0},
                "home_offense": {"runs": 2.1},
                "away_pitcher": {"era": 3.2},
                "home_pitcher": {"era": 3.1},
                "away_lineup_matchup": {"delta": 0.1},
                "home_lineup_matchup": {"delta": 0.1},
                "venue": {"run_delta": 0.0},
                "travel": {"away": {}, "home": {}},
            },
        },
    )
    uncertified = _record(
        snapshot_id="bad-time",
        certification={"certified": True, "immutable": True, "pregame": False},
    )
    report = evaluate_team_prop_ledger({"records": [certified, uncertified]})

    assert report["record_quality"]["certified_evaluable_records"] == 1
    assert report["record_quality"]["exclusions"] == {"uncertified:missing_pregame_flag": 1}
    feature_audit = report["feature_contract_audit"]
    assert len(feature_audit) == 1
    groups = {group["name"]: group for group in feature_audit[0]["feature_groups"]}
    assert groups["away_offense"]["availability_rate"] == 1.0
    assert groups["travel"]["availability_rate"] == 1.0


def test_explicit_ledger_path_loads_only_the_supplied_fixture(tmp_path):
    fixture = {"schema_version": 7, "records": [_record()]}
    path = tmp_path / "certified-ledger.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    assert load_ledger(path) == fixture
