"""Smoke tests for the in-house NHL goal-rate model and its feed wiring."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _game(**overrides):
    game = {
        "game_id": "1",
        "home_team": "Boston Bruins",
        "away_team": "Philadelphia Flyers",
        "home_abbrev": "BOS",
        "away_abbrev": "PHI",
        "start_time": "2026-09-22T23:00:00Z",
        "status": "STATUS_SCHEDULED",
        "season_type": "PRE",
    }
    game.update(overrides)
    return game


def test_ratings_are_a_full_regular_season_and_contain_no_prices():
    artifact = json.loads((ROOT / "NHLPredictionModel" / "artifacts" / "nhl_ratings.json").read_text(encoding="utf-8"))
    assert artifact["model_version"] == "nhl_poisson_v1"
    assert artifact["prior_season"] == "20252026"
    assert artifact["standings_date"] == "2026-04-17"
    assert artifact["game_type_id"] == 2
    assert artifact["decision_policy"]["mode"] == "research_only"
    assert len(artifact["teams"]) == 32
    assert all(team["games_played"] == 82 for team in artifact["teams"].values())
    blob = json.dumps(artifact).lower()
    assert "odds" not in blob
    assert "moneyline" not in blob
    league = artifact["league"]
    assert 2.5 < league["goals_per_game"] < 4.0
    assert league["home_goals_per_game"] > league["away_goals_per_game"]
    assert 0.0 < league["ot_home_win_rate"] < 1.0


def test_pregame_moneyline_is_unpriced_research(monkeypatch):
    from NHLPredictionModel import nhl_model

    payload = nhl_model.generate_nhl_picks("2026-09-22", games=[_game()])
    assert payload["ok"] is True
    assert payload["coverage"]["staked_rows"] == 0
    assert len(payload["picks"]) == 1
    pick = payload["picks"][0]
    assert pick["sport"] == "NHL"
    assert pick["source"] == "NHL Model"
    assert pick["market"] == "h2h"
    assert pick["decision"] == "PASS"
    assert pick["units"] == 0
    assert pick["odds"] is None
    assert pick["market_priced"] is False
    assert pick["calibration_excluded"] is True
    assert pick["decision_reason"] == "research_only:preseason"
    assert pick["probability"] >= 0.5
    assert "score" not in pick


def test_started_games_and_empty_slates_publish_nothing():
    from NHLPredictionModel import nhl_model

    started = nhl_model.generate_nhl_picks("2026-09-21", games=[_game(status="STATUS_IN_PROGRESS")])
    assert started["picks"] == []
    assert started["coverage"]["started_games"] == 1
    assert "started game" in started["note"]
    empty = nhl_model.generate_nhl_picks("2026-09-27", games=[])
    assert empty["ok"] is True
    assert empty["picks"] == []
    assert "0 game(s)" in empty["note"]


def test_lines_are_used_only_when_observed():
    from NHLPredictionModel import nhl_model

    bare = nhl_model.generate_nhl_picks("2026-09-22", games=[_game()])
    assert {pick["market"] for pick in bare["picks"]} == {"h2h"}
    quoted = nhl_model.generate_nhl_picks("2026-09-22", games=[_game(
        spread_line=-1.5,
        home_spread_odds=-110,
        away_spread_odds=-110,
        total_line=6.5,
        over_odds=-105,
        under_odds=-115,
        home_moneyline=-140,
        away_moneyline=120,
    )])
    markets = {pick["market"]: pick for pick in quoted["picks"]}
    assert set(markets) == {"h2h", "spread", "totals"}
    assert markets["h2h"]["odds"] in {-140, 120}
    assert markets["spread"]["market_line"] == -1.5
    assert markets["spread"]["odds"] == -110
    assert markets["totals"]["line"] == 6.5
    assert markets["totals"]["decision"] == "PASS"
    assert markets["totals"]["units"] == 0
    other_line = nhl_model.generate_nhl_picks("2026-09-22", games=[_game(spread_line=-1.0)])
    assert {pick["market"] for pick in other_line["picks"]} == {"h2h"}


def test_stronger_offense_raises_home_win_probability():
    from NHLPredictionModel.nhl_core import load_ratings, project_game

    ratings = load_ratings()
    base = project_game(ratings, "BOS", "PHI")
    boosted = json.loads(json.dumps(ratings))
    boosted["teams"]["BOS"]["goals_for_per_game"] = 5.0
    stronger = project_game(boosted, "BOS", "PHI")
    assert stronger["moneyline_home_probability"] > base["moneyline_home_probability"]


def test_registration_across_model_cache_feeds_and_board():
    import pickgrader_server as server
    from scripts.market_odds import SPORT_LEAGUES, TEAM_MODEL_BUCKET_KEYS
    from scripts.merge_external_feed_cache_payload import EXTERNAL_FEED_MODEL_KEYS
    from scripts.merge_model_cache_payload import DEPLOYED_MODEL_KEYS, UNPRICED_STAKE_DEMOTION_KEYS
    from scripts.pick_calibration import CALIBRATION_EXCLUDED_MODEL_KEYS
    from scripts.refresh_external_feeds import FEED_RUNNERS
    from scripts.refresh_model_cache import _model_jobs
    from scripts.team_prop_model_evaluator import SUPPORTED_MODEL_KEYS
    from scripts.team_prop_pregame_ledger import TEAM_PROP_MODEL_KEYS

    assert callable(server.run_nhl_model)
    assert server.SPORT_TO_ESPNSLUG["NHL"] == ("hockey", "nhl")
    assert "nhl" in _model_jobs("2026-09-22")
    assert "nhl" in DEPLOYED_MODEL_KEYS
    assert "nhl" in TEAM_MODEL_BUCKET_KEYS
    assert "nhl" in UNPRICED_STAKE_DEMOTION_KEYS
    assert "nhl" in CALIBRATION_EXCLUDED_MODEL_KEYS
    assert "nhl" in TEAM_PROP_MODEL_KEYS
    assert "nhl" in SUPPORTED_MODEL_KEYS
    assert SPORT_LEAGUES["NHL"] == ("hockey", "nhl")
    for key in ("scores24_nhl", "forebet_nhl", "sportytrader_nhl", "sportsgambler_nhl"):
        assert key in EXTERNAL_FEED_MODEL_KEYS
    assert "scores24_nhl" in FEED_RUNNERS
    assert "forebet_nhl" in FEED_RUNNERS
    data_ts = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    main_ts = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    assert "nhl: { h2h: 'NHL ML'" in data_ts
    assert "forebet_nhl: 'ForebetNHL'" in data_ts
    assert "scores24_nhl: 'Scores24NHL'" in data_ts
    assert "'NHL'" in main_ts.split("PRIMARY_FILTERS")[1][:120]
    assert "NHL: ['hockey', 'nhl']" in main_ts
    workflow = (ROOT / ".github" / "workflows" / "external-feed-refresh.yml").read_text(encoding="utf-8")
    assert "forebet_nhl" in workflow
    assert "scores24_nhl" not in workflow
