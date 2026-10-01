from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone


def test_all_model_exact_prices_and_fresh_holdout_are_separate():
    rows = [_record(snapshot_id=str(i), model_key=key, market='total', line=3.5,
                    snapshot_at='2026-09-20T12:00:00Z')
            for i, key in enumerate(['mls', 'wnba', 'mlb_team_total', 'nba', 'nba_playoffs', 'tennis', 'ipl'])]
    rows.append(_record(snapshot_id='next', model_key='mls', market='total', line=4.5,
                        observed_american_odds=-145, snapshot_at='2026-09-21T12:00:00Z'))
    report = evaluate_team_prop_ledger({'records': rows}, forward_since='2026-09-21T00:00:00Z')
    assert report['record_quality']['certified_evaluable_records'] == 8
    mls = [r for r in report['exact_market_prices'] if r['model_key'] == 'mls']
    assert {(r['line'], r['offered_american_odds']) for r in mls} == {('3.5', '-110.0'), ('4.5', '-145.0')}
    forward = next(r for r in report['forward_holdout'] if r['model_key'] == 'mls')
    assert forward['records'] == 1 and forward['status'] == 'insufficient_samples'
    assert forward['promotion_approved'] is False

from scripts.team_prop_model_evaluator import evaluate_team_prop_ledger, load_ledger


def test_canonical_nested_snapshot_uses_prediction_fingerprint_over_serving_label():
    row = _record(model_key='nfl', model_version='nfl:artifact-hash',
        pregame_snapshot={'model_version': 'legacy-serving-label',
                          'prediction_model_version': 'nfl:artifact-hash'})
    report = evaluate_team_prop_ledger({'records': [row]})
    assert report['segments'][0]['model_version'] == 'nfl:artifact-hash'
    assert report['exact_market_prices'][0]['model_version'] == 'nfl:artifact-hash'
    row['snapshot_hash'] = 'immutable-hash'
    del row['pregame_snapshot']['prediction_model_version']
    report = evaluate_team_prop_ledger({'records': [row]})
    assert report['segments'][0]['model_version'] == 'nfl:artifact-hash'


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
    when = datetime.fromisoformat(str(record["snapshot_at"]).replace("Z", "+00:00"))
    record.setdefault("published_at", when.isoformat().replace("+00:00", "Z"))
    record.setdefault("game_start_time", (when + timedelta(hours=4)).isoformat().replace("+00:00", "Z"))
    record.setdefault("market_updated_at", (when - timedelta(minutes=5)).isoformat().replace("+00:00", "Z"))
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
        price={"odds": -120, "pricing_type": "market", "odds_source": "sportsbook_observed",
               "market_updated_at": "2026-06-10T14:55:00Z"},
    )

    report = evaluate_team_prop_ledger({"records": [record]})

    assert report["record_quality"]["certified_evaluable_records"] == 1
    assert report["overall"]["market_benchmark"]["probability_sources"] == {
        "observed_american_odds": 1,
    }


def test_evaluator_uses_captured_quote_for_roi_benchmark_and_exact_price():
    record = _record(
        model_key="nfl",
        pregame_snapshot={
            "odds": -110, "pricing_type": "user_assumed",
            "market_updated_at": "2026-06-10T14:00:00Z",
            "market_no_vig_selected_probability": 0.8,
        },
        observed_american_odds=140,
        price={
            "odds": 140, "pricing_type": "market", "odds_source": "sportsbook_observed",
            "market_updated_at": "2026-06-10T14:58:00Z",
            "market_no_vig_selected_probability": 0.42,
        },
    )

    report = evaluate_team_prop_ledger({"records": [record]})

    assert report["overall"]["real_price_roi"]["profit_units"] == 1.4
    assert report["overall"]["market_benchmark"]["brier_score"] == 0.3364
    assert report["overall"]["market_benchmark"]["probability_sources"] == {"observed_no_vig": 1}
    assert report["exact_market_prices"][0]["offered_american_odds"] == "140.0"

    del record["price"]["market_no_vig_selected_probability"]
    report = evaluate_team_prop_ledger({"records": [record]})
    assert report["overall"]["market_benchmark"]["probability_sources"] == {"observed_american_odds": 1}
    assert report["overall"]["market_benchmark"]["mean_probability"] == 0.416667


def test_evaluator_borrows_missing_quote_clock_but_not_snapshot_odds():
    record = _record(
        model_key="nfl",
        market_updated_at="2026-06-10T20:00:00Z",
        pregame_snapshot={
            "odds": -110, "pricing_type": "market",
            "market_updated_at": "2026-06-10T14:55:00Z",
            "market_no_vig_selected_probability": 0.55,
        },
        observed_american_odds=140,
        price={"odds": 140, "pricing_type": "market", "odds_source": "sportsbook_observed"},
    )

    report = evaluate_team_prop_ledger({"records": [record]})
    roi = report["overall"]["real_price_roi"]
    assert roi["priced_settled_actionable_records"] == 1
    assert roi["profit_units"] == 1.4
    assert "missing_quote_timestamp" not in roi["excluded"]
    assert report["overall"]["market_benchmark"]["priced_or_observed_records"] == 1
    assert report["overall"]["market_benchmark"]["probability_sources"] == {"observed_american_odds": 1}

    record["pregame_snapshot"].pop("market_updated_at")
    record["market_updated_at"] = "2026-06-10T14:55:00Z"
    report = evaluate_team_prop_ledger({"records": [record]})
    assert report["overall"]["real_price_roi"]["priced_settled_actionable_records"] == 1
    assert "missing_quote_timestamp" not in report["overall"]["real_price_roi"]["excluded"]

    record["market_updated_at"] = None
    report = evaluate_team_prop_ledger({"records": [record]})
    assert report["overall"]["real_price_roi"]["excluded"] == {"missing_quote_timestamp": 1}
    assert report["overall"]["market_benchmark"]["priced_or_observed_records"] == 0

    record["pregame_snapshot"]["market_odds_captured_at"] = "2026-06-10T14:55:00Z"
    report = evaluate_team_prop_ledger({"records": [record]})
    assert report["overall"]["real_price_roi"]["priced_settled_actionable_records"] == 1
    assert report["overall"]["real_price_roi"]["profit_units"] == 1.4

    record["price"]["market_updated_at"] = "2026-06-10T15:06:00Z"
    report = evaluate_team_prop_ledger({"records": [record]})
    assert report["overall"]["real_price_roi"]["excluded"] == {"quote_after_publication": 1}
    assert report["overall"]["market_benchmark"]["priced_or_observed_records"] == 0

    record["price"]["market_updated_at"] = "2026-06-10T14:58:00Z"
    record["observed_american_odds"] = -110
    report = evaluate_team_prop_ledger({"records": [record]})
    assert report["overall"]["real_price_roi"]["excluded"] == {"missing_verified_american_price": 1}
    assert report["overall"]["market_benchmark"]["priced_or_observed_records"] == 0

    record["observed_american_odds"] = None
    del record["price"]["odds"]
    report = evaluate_team_prop_ledger({"records": [record]})
    assert report["overall"]["real_price_roi"]["excluded"] == {"missing_verified_american_price": 1}
    assert report["overall"]["market_benchmark"]["priced_or_observed_records"] == 0


def test_evaluator_uses_ledger_settlement_over_pending_pregame_image():
    record = _record(
        model_key="nfl", result="win",
        pregame_snapshot={"result": "pending", "outcome": "pending"},
    )

    report = evaluate_team_prop_ledger({"records": [record]})

    assert report["overall"]["model_metrics"]["settled_records"] == 1
    assert report["overall"]["result_counts"] == {"win": 1}
    assert report["overall"]["real_price_roi"]["priced_settled_actionable_records"] == 1


def test_quote_borrow_preserves_source_priority_across_aliases():
    from scripts.price_clock import QUOTE_FIELDS, borrow_missing_quote_clocks, observed_quote_timing

    good = "2026-06-10T14:55:00Z"
    for field in QUOTE_FIELDS:
        for bad, reason in (("2026-06-10T23:00:00Z", "post_start"), ("invalid", "missing_quote_timestamp")):
            original = {"odds": 140, field: bad}
            context = borrow_missing_quote_clocks(original, {"market_updated_at": good})
            assert context == original
            assert context is not original
            assert observed_quote_timing(
                context, published_at="2026-06-10T15:00:00Z", start_at="2026-06-10T22:00:00Z",
            ) == reason

    context = borrow_missing_quote_clocks(
        {"odds": 140}, {"market_odds_captured_at": good},
        {"market_updated_at": "2026-06-10T23:00:00Z", "odds": -110},
    )
    assert context == {"odds": 140, "market_odds_captured_at": good}
    assert observed_quote_timing(
        context, published_at="2026-06-10T15:00:00Z", start_at="2026-06-10T22:00:00Z",
    ) is None


def test_observed_odds_at_model_selected_line_remain_financial_evidence():
    record = _record(
        model_key="mlb_first_five",
        price={
            "odds": -110, "pricing_type": "market", "odds_source": "posted_market",
            "line_source": "user_assumed_f5_total_ladder",
            "market_updated_at": "2026-06-10T14:55:00Z",
        },
    )

    report = evaluate_team_prop_ledger({"records": [record]})

    assert report["overall"]["real_price_roi"]["priced_settled_actionable_records"] == 1
    assert report["overall"]["market_benchmark"]["priced_or_observed_records"] == 1


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
