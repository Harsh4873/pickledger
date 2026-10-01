from __future__ import annotations

import json

from scripts.inhouse_model_scorecard import build_scorecard
from scripts.model_scorecard_stats import clustered_roi_interval


def test_prop_scorecard_uses_first_pregame_publication_and_observed_price(tmp_path):
    snapshots = tmp_path / "snapshots" / "2026-09-24"
    snapshots.mkdir(parents=True)
    base = {
        "id": "prop-1", "date": "2026-09-24", "game_id": "game-1",
        "start_time": "2026-09-24T22:00:00Z", "stat_key": "hits",
        "selection": "Over", "line": 0.5, "probability": 0.60,
        "market_no_vig_selected_probability": 0.52,
        "market_priced": True, "pricing_type": "market",
        "odds_source": "posted_market", "market_updated_at": "2026-09-24T19:00:00Z",
        "odds": -110, "decision": "LEAN", "units": 0.25,
        "ml_model_version": "prop-v1", "result": "pending",
    }
    first = {"generatedAt": "2026-09-24T20:00:00Z", "models": {"mlb_player_props": {"picks": [base]}}}
    later = {"generatedAt": "2026-09-24T21:00:00Z", "models": {"mlb_player_props": {"picks": [
        {**base, "probability": 0.99, "odds": -180, "market_updated_at": "2026-09-24T20:59:00Z"},
    ]}}}
    (snapshots / "a.json").write_text(json.dumps(first))
    (snapshots / "b.json").write_text(json.dumps(later))
    outcomes = {"records": [{"cache_type": "player_props_cache", "result": "win", "pregame_snapshot": {"id": "prop-1"}}]}
    report = build_scorecard({"records": []}, outcomes, tmp_path / "snapshots")
    card = next(c for c in report["scorecards"] if c.get("model_version") == "prop-v1")
    assert card["forecasts"] == 1
    assert card["model"]["brier"] == 0.16
    assert card["market_comparison"]["observed_no_vig"]["brier"] == 0.2304
    assert card["priced_settled_bets"] == 1
    assert card["stake_units"] == 0.25
    assert card["profit_units"] == 0.227273
    assert card["status"] == "insufficient_priced_sample"
    assert report["forward_since"] == "2026-09-22T00:00:00Z"


def test_prop_scorecard_excludes_assumed_and_post_start_prices(tmp_path):
    snapshots = tmp_path / "snapshots" / "2026-09-24"
    snapshots.mkdir(parents=True)
    picks = [
        {"id": "assumed", "start_time": "2026-09-24T22:00:00Z", "game_id": "g1",
         "stat_key": "rbis", "probability": 0.9, "decision": "BET", "units": 1,
         "odds": -110, "market_priced": True, "odds_source": "user_assumed", "market_updated_at": "2026-09-24T19:00:00Z"},
        {"id": "late", "start_time": "2026-09-24T22:00:00Z", "game_id": "g2",
         "stat_key": "rbis", "probability": 0.9, "decision": "BET", "units": 1,
         "odds": -110, "market_priced": True, "odds_source": "posted_market", "market_updated_at": "2026-09-24T22:10:00Z"},
    ]
    (snapshots / "a.json").write_text(json.dumps({
        "generatedAt": "2026-09-24T21:00:00Z", "models": {"mlb_player_props": {"picks": picks}},
    }))
    outcomes = {"records": [
        {"cache_type": "player_props_cache", "result": "win", "pregame_snapshot": {"id": name}}
        for name in ("assumed", "late")
    ]}
    report = build_scorecard({"records": []}, outcomes, tmp_path / "snapshots")
    card = next(c for c in report["scorecards"] if c.get("market") == "rbis")
    assert card["priced_settled_bets"] == 0
    assert card["exclusions"] == {"assumed_or_proxy_price": 1, "post_start": 1}


def test_roi_interval_clusters_same_game_and_is_deterministic():
    rows = [("game-1", 1.0, 0.9), ("game-1", 1.0, -1.0), ("game-2", 1.0, 0.9)]
    result = clustered_roi_interval(rows, samples=200)
    assert result == clustered_roi_interval(rows, samples=200)
    assert result["event_clusters"] == 2
    assert result["lower_95"] <= result["upper_95"]


def test_shadow_pass_is_graded_without_becoming_a_staked_bet(tmp_path):
    snapshots = tmp_path / "snapshots" / "2026-09-24"
    snapshots.mkdir(parents=True)
    pick = {
        "id": "shadow-1", "date": "2026-09-24", "game_id": "game-1",
        "start_time": "2026-09-24T22:00:00Z", "stat_key": "hits",
        "probability": 0.6, "market_priced": True, "pricing_type": "market",
        "odds_source": "posted_market", "market_updated_at": "2026-09-24T19:00:00Z",
        "odds": -110, "decision": "PASS", "units": 0,
        "shadow_decision": "BET", "shadow_units": 0.5,
        "ml_model_version": "prop-v1",
    }
    (snapshots / "a.json").write_text(json.dumps({
        "publishedAt": "2026-09-24T20:00:00Z",
        "models": {"mlb_player_props": {"picks": [pick]}},
    }))
    outcomes = {"records": [{
        "cache_type": "player_props_cache", "result": "win",
        "pregame_snapshot": {"id": "shadow-1"},
    }]}
    report = build_scorecard({"records": []}, outcomes, tmp_path / "snapshots")
    card = next(c for c in report["scorecards"] if c.get("model_version") == "prop-v1")
    assert card["priced_settled_bets"] == 0
    assert card["shadow_priced_settled"] == 1
    assert card["shadow_stake_units"] == 0.5
    assert card["shadow_profit_units"] == 0.454545


def test_team_scorecard_uses_captured_ledger_quote_instead_of_snapshot_price(tmp_path):
    record = {
        "model_key": "nfl", "model_version": "nfl-v1", "market": "h2h",
        "probability": 0.6, "result": "win", "decision": "BET", "units": 1,
        "snapshot_at": "2026-09-24T20:00:00Z",
        "published_at": "2026-09-24T20:00:00Z",
        "game_start_time": "2026-09-24T22:00:00Z",
        "certification": {"certified": True, "immutable": True, "pregame": True,
                          "financial_eligible": True, "market_benchmark_eligible": True},
        "observed_american_odds": 140,
        "pregame_snapshot": {
            "odds": -110, "pricing_type": "user_assumed",
            "market_updated_at": "2026-09-24T19:00:00Z",
            "market_no_vig_selected_probability": 0.8,
        },
        "price": {
            "odds": 140, "pricing_type": "market", "odds_source": "sportsbook_observed",
            "market_updated_at": "2026-09-24T19:58:00Z",
            "market_no_vig_selected_probability": 0.42,
        },
    }

    report = build_scorecard({"records": [record]}, {"records": []}, tmp_path / "snapshots")
    card = next(c for c in report["scorecards"] if c.get("model_key") == "nfl" and c.get("market") == "h2h")

    assert card["profit_units"] == 1.4
    assert card["market_comparison"]["observed_market_brier"] == 0.3364
    assert card["market_comparison"]["probability_sources"] == {"observed_no_vig": 1}


def test_prop_scorecard_matches_reused_id_by_market_side(tmp_path):
    snapshots = tmp_path / "snapshots" / "2026-09-24"
    snapshots.mkdir(parents=True)
    base = {
        "id": "same-player-market-id", "date": "2026-09-24", "game_id": "game-1",
        "player_id": "player-1", "stat_key": "hits", "line": 0.5,
        "start_time": "2026-09-24T22:00:00Z", "probability": 0.6,
        "market_priced": True, "pricing_type": "market", "odds_source": "posted_market",
        "market_updated_at": "2026-09-24T19:00:00Z", "odds": -110,
        "ml_model_version": "prop-v1", "result": "pending",
    }
    over = {**base, "selection": "Over", "decision": "BET", "units": 1}
    under = {**base, "selection": "Under", "decision": "PASS", "units": 0}
    (snapshots / "a.json").write_text(json.dumps({
        "generatedAt": "2026-09-24T20:00:00Z",
        "models": {"mlb_player_props": {"picks": [over, under]}},
    }))
    outcomes = {"records": [
        {"cache_type": "player_props_cache", "model_key": "mlb_player_props",
         "result": result, "pregame_snapshot": pick}
        for pick, result in ((over, "loss"), (under, "win"))
    ]}

    report = build_scorecard({"records": []}, outcomes, tmp_path / "snapshots")
    card = next(c for c in report["scorecards"] if c.get("model_version") == "prop-v1")

    assert card["forecasts"] == 2
    assert card["result_counts"] == {"loss": 1, "win": 1}
    assert card["priced_settled_bets"] == 1
    assert card["profit_units"] == -1

    # An older outcome row with only the reused id cannot identify a side.
    ambiguous = {"records": [{"cache_type": "player_props_cache", "result": "win",
                               "pregame_snapshot": {"id": "same-player-market-id"}}]}
    report = build_scorecard({"records": []}, ambiguous, tmp_path / "snapshots")
    card = next(c for c in report["scorecards"] if c.get("model_version") == "prop-v1")
    assert card["result_counts"] == {"pending": 2}
    assert card["priced_settled_bets"] == 0


def test_prop_scorecard_keeps_unjoined_snapshot_result_pending(tmp_path):
    snapshots = tmp_path / "snapshots" / "2026-09-24"
    snapshots.mkdir(parents=True)
    pick = {
        "id": "unjoined-prop", "date": "2026-09-24", "game_id": "game-1",
        "stat_key": "hits", "selection": "Over", "line": 0.5,
        "start_time": "2026-09-24T22:00:00Z", "probability": 0.6,
        "market_priced": True, "pricing_type": "market", "odds_source": "posted_market",
        "market_updated_at": "2026-09-24T19:00:00Z", "odds": -110,
        "decision": "BET", "units": 1, "result": "win", "ml_model_version": "prop-v1",
    }
    (snapshots / "a.json").write_text(json.dumps({
        "generatedAt": "2026-09-24T20:00:00Z", "models": {"mlb_player_props": {"picks": [pick]}},
    }))

    report = build_scorecard({"records": []}, {"records": []}, tmp_path / "snapshots")
    card = next(c for c in report["scorecards"] if c.get("model_version") == "prop-v1")

    assert card["result_counts"] == {"pending": 1}
    assert card["model"]["samples"] == 0
    assert card["priced_settled_bets"] == 0
    assert card["exclusions"] == {"unsettled": 1}
