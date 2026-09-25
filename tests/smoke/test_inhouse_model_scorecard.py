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
