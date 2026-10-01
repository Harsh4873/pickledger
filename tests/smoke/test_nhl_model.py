"""Smoke tests for the in-house NHL goal-rate model and its feed wiring."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

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
    assert artifact["model_version"] == "nhl_poisson_v2_20252026_shadow_ev_v1"
    assert artifact["prior_season"] == "20252026"
    assert artifact["standings_date"] == "2026-04-17"
    assert artifact["game_type_id"] == 2
    assert artifact["decision_policy"]["mode"] == "research_only"
    assert artifact["decision_policy"]["shadow_candidate"]["validated"] is False
    assert artifact["decision_policy"]["shadow_candidate"]["markets"] == ["h2h", "spread", "totals"]
    assert len(artifact["teams"]) == 32
    assert all(team["games_played"] == 82 for team in artifact["teams"].values())
    blob = json.dumps(artifact).lower()
    assert "odds" not in blob
    assert "moneyline" not in blob
    league = artifact["league"]
    assert 2.5 < league["goals_per_game"] < 4.0
    assert league["home_goals_per_game"] > league["away_goals_per_game"]
    assert 0.0 < league["ot_home_win_rate"] < 1.0


def test_preseason_does_not_publish_a_side_from_the_regular_season_prior():
    from NHLPredictionModel import nhl_model

    bare = nhl_model.generate_nhl_picks("2026-09-22", games=[_game()])
    assert bare["ok"] is True
    assert bare["picks"] == []
    assert bare["preseason_blocked"] == 1
    assert bare["games"][0]["side_published"] is False
    assert bare["games"][0]["side_block"] == "preseason_prior_not_validated"
    assert "not a validated preseason model" in bare["note"]

    priced = nhl_model.generate_nhl_picks("2026-09-22", games=[_game(
        spread_line=-1.5,
        home_spread_odds=-110,
        away_spread_odds=-110,
        total_line=6.5,
        over_odds=-105,
        under_odds=-115,
        home_moneyline=-140,
        away_moneyline=120,
        team_totals={
            "home": {"line": 3.5, "over_odds": -120, "under_odds": 100, "team": "Boston Bruins", "mean": 3.3, "mean_source": "nhl_standings_goals_for"},
        },
        player_props=[{
            "player": "David Pastrnak",
            "stat": "shots_on_goal",
            "stat_label": "Shots on Goal",
            "line": 1.5,
            "mean": 4.0,
            "mean_source": "nhl_skater_summary",
            "over_odds": -115,
            "under_odds": -105,
        }],
    )])
    assert priced["picks"] == []
    assert priced["games"][0]["prices_seen"] == ["h2h", "spread", "totals", "team_total", "player_props"]
    assert priced["games"][0]["side_published"] is False
    assert "home_win_probability" not in priced["games"][0]


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

    bare = nhl_model.generate_nhl_picks("2026-09-22", games=[_game(season_type="REG")])
    assert bare["picks"] == []
    assert bare["markets_skipped"]["h2h"] == 1
    quoted = nhl_model.generate_nhl_picks("2026-09-22", games=[_game(
        season_type="REG",
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
    assert markets["spread"]["market_line"] == markets["spread"]["line"] == 1.5
    assert markets["spread"]["odds"] == -110
    assert markets["totals"]["line"] == 6.5
    assert markets["totals"]["decision"] == "PASS"
    assert markets["totals"]["units"] == 0
    other_line = nhl_model.generate_nhl_picks("2026-09-22", games=[_game(
        season_type="REG",
        spread_line=-1.0,
        home_spread_odds=-110,
        away_spread_odds=-110,
        home_moneyline=-140,
        away_moneyline=120,
    )])
    assert {pick["market"] for pick in other_line["picks"]} == {"h2h"}


def test_priced_regular_season_candidates_are_shadow_only_and_certifiable(tmp_path):
    from NHLPredictionModel import nhl_model
    from scripts.model_stake_policy import _policy_version
    from scripts.team_prop_pregame_ledger import capture_team_prop_pregame_snapshots, load_team_prop_pregame_ledger

    clock = datetime(2026, 10, 10, 18, tzinfo=timezone.utc)
    game = _game(
        season_type="REG", start_time="2026-10-10T23:00:00Z",
        odds_source="draftkings", market_retrieved_at="2026-10-10T18:00:00Z",
        home_moneyline=-140, away_moneyline=120,
        spread_line=-1.5, home_spread_odds=-110, away_spread_odds=-110,
        total_line=6.5, over_odds=-105, under_odds=-115,
    )
    result = nhl_model.generate_nhl_picks("2026-10-10", games=[game], now=clock)
    rows = {row["market"]: row for row in result["picks"]}
    assert set(rows) == {"h2h", "spread", "totals"}
    assert rows["h2h"]["side"] == "away"  # price value can favor the model's lower-probability side
    assert rows["h2h"]["model_home_win_probability"] > 0.5
    assert rows["spread"]["shadow_decision"] == "BET"
    assert rows["spread"]["shadow_units"] == 0.5
    assert rows["spread"]["source_decision"] == "BET"
    assert rows["spread"]["source_units"] == 0.5
    assert rows["spread"]["staking_policy"] == "awaiting_approved_holdout"
    assert rows["spread"]["market_retrieved_at"] == "2026-10-10T18:00:00Z"
    assert rows["spread"]["expected_value"] > 0.1
    assert rows["spread"]["market_implied_probability"] == 0.5
    assert all(row["decision"] == "PASS" and row["units"] == 0 for row in rows.values())
    assert result["coverage"]["staked_rows"] == 0
    assert result["shadow_candidate_rows"] >= 1

    rows["spread"]["prediction_model_version"] = "nhl:unrelated-serving-hash"
    assert _policy_version(rows["spread"], result) == result["model_version"]
    payload = {"date": "2026-10-10", "generatedAt": clock.isoformat(), "models": {"nhl": result}}
    captured = capture_team_prop_pregame_snapshots(payload, repo_root=tmp_path)
    assert captured["added"] == 3
    records = load_team_prop_pregame_ledger(tmp_path)["records"]
    spread = next(record for record in records if record["market"] == "spread")
    assert spread["model_version"] == result["model_version"]
    assert spread["certification"]["status"] == "certified"
    assert spread["financial_eligible"] is True
    assert spread["shadow_decision"] == "BET"
    assert spread["shadow_units"] == 0.5
    # Once the real pregame quote has a settled result, the frozen holdout
    # evaluator must count the published zero-stake shadow action.
    from scripts.frozen_staking_approval import evaluate
    from scripts.frozen_staking_candidate import load_freeze
    spread["result"] = "win"
    holdout = evaluate({"records": records}, load_freeze("nhl"))
    spread_status = next(row for row in holdout["markets"] if row["market"] == "spread")
    assert spread_status["independently_priced_settled"] == 1


def test_shadow_rule_needs_a_complete_fresh_quote_and_team_rating():
    from NHLPredictionModel import nhl_model

    clock = datetime(2026, 10, 10, 18, tzinfo=timezone.utc)
    game = _game(
        season_type="REG", start_time="2026-10-10T23:00:00Z",
        odds_source="draftkings", market_retrieved_at="2026-10-10T18:00:00Z",
        spread_line=-1.5, home_spread_odds=-110, away_spread_odds=-110,
    )
    stale = nhl_model.generate_nhl_picks(
        "2026-10-10", games=[{**game, "market_retrieved_at": "2026-10-08T18:00:00Z"}], now=clock,
    )["picks"][0]
    assert stale["decision_reason"] == "research_only:stale_quote"
    assert "shadow_decision" not in stale
    one_sided = nhl_model.generate_nhl_picks(
        "2026-10-10", games=[{**game, "home_spread_odds": None}], now=clock,
    )["picks"][0]
    assert one_sided["decision_reason"] == "research_only:incomplete_two_sided_quote"
    assert "shadow_decision" not in one_sided
    unknown = nhl_model.generate_nhl_picks(
        "2026-10-10", games=[{**game, "away_abbrev": "UNKNOWN"}], now=clock,
    )["picks"][0]
    assert unknown["decision_reason"] == "research_only:team_ratings_unavailable"
    assert "shadow_decision" not in unknown
    preseason = nhl_model.generate_nhl_picks(
        "2026-10-10", games=[{**game, "season_type": "PRE"}], now=clock,
    )
    assert preseason["picks"] == []
    assert preseason["coverage"]["staked_rows"] == 0
    no_clock = nhl_model.generate_nhl_picks(
        "2026-10-10",
        games=[{key: value for key, value in game.items() if key != "market_retrieved_at"}],
        now=clock,
    )["picks"][0]
    assert no_clock["decision"] == "PASS" and no_clock["units"] == 0
    assert no_clock["decision_reason"] == "research_only:missing_quote_timestamp"
    assert "shadow_decision" not in no_clock
    assert "market_retrieved_at" not in no_clock


def test_integer_totals_keep_push_mass_out_of_the_under_probability():
    from NHLPredictionModel import nhl_model
    from NHLPredictionModel.nhl_core import load_ratings, project_game, total_outcomes

    projection = project_game(load_ratings(), "BOS", "PHI")
    over, under, push = total_outcomes(projection["lambda_home"], projection["lambda_away"], 6.0)
    assert push > 0.0
    assert abs(over + under + push - 1.0) < 1e-10
    row = next(pick for pick in nhl_model.generate_nhl_picks(
        "2026-10-10", games=[_game(season_type="REG", total_line=6.0, over_odds=-110, under_odds=-110)],
    )["picks"] if pick["market"] == "totals")
    expected = over if row["direction"] == "over" else under
    assert abs(row["probability"] - expected) < 0.0001
    assert abs(row["push_probability"] - push) < 0.0001


def test_team_total_and_player_props_publish_only_with_price_and_observed_mean():
    from NHLPredictionModel import nhl_model
    from NHLPredictionModel.nhl_observed import lookup_player_mean, rates_from_summaries, stat_key_from_market

    assert stat_key_from_market("David Pastrnak Shots on Goal") == "shots"
    assert stat_key_from_market("CAR Hurricanes: Team Total Goals") is None
    assert stat_key_from_market("First to 5 Shots on Goal") is None
    rates = rates_from_summaries(
        [{
            "skaterFullName": "David Pastrnak",
            "gamesPlayed": 77,
            "goals": 29,
            "assists": 71,
            "points": 100,
            "shots": 261,
        }],
        [],
        season="20252026",
    )
    looked = lookup_player_mean(rates, "David Pastrnak", "shots")
    assert looked is not None
    assert abs(looked["mean"] - (261 / 77)) < 1e-9
    assert lookup_player_mean(rates_from_summaries(
        [{"skaterFullName": "Cup Of Coffee", "gamesPlayed": 4, "goals": 2, "assists": 0, "points": 2, "shots": 8}],
        [],
        season="20252026",
    ), "Cup Of Coffee", "shots") is None

    missing = nhl_model.generate_nhl_picks("2026-10-10", games=[_game(
        season_type="REG",
        home_moneyline=-140,
        away_moneyline=120,
        team_totals={"home": {"line": 3.5, "team": "Boston Bruins"}},
        player_props=[{
            "player": "David Pastrnak",
            "stat": "shots_on_goal",
            "stat_label": "Shots on Goal",
            "line": 3.5,
            "over_odds": -115,
            "under_odds": -105,
        }],
    )], player_rates={})
    assert {pick["market"] for pick in missing["picks"]} == {"h2h"}
    assert missing["player_props_without_prior"] == 1
    assert "team_total" not in {pick["market"] for pick in missing["picks"]}

    quoted = nhl_model.generate_nhl_picks("2026-10-10", games=[_game(
        season_type="REG",
        home_moneyline=-140,
        away_moneyline=120,
        odds_source="draftkings",
        team_totals={
            "home": {
                "line": 3.5,
                "over_odds": -120,
                "under_odds": 100,
                "team": "Boston Bruins",
                "mean": 3.317073,
                "mean_source": "nhl_standings_goals_for",
                "mean_games": 82,
            },
        },
        player_props=[{
            "player": "David Pastrnak",
            "team": "BOS",
            "stat": "shots_on_goal",
            "stat_label": "Shots on Goal",
            "line": 3.5,
            "over_odds": -115,
            "under_odds": -105,
        }],
    )], player_rates=rates)
    markets = [pick["market"] for pick in quoted["picks"]]
    assert "team_total" in markets
    team_row = next(pick for pick in quoted["picks"] if pick["market"] == "team_total")
    assert team_row["mean_source"] == "nhl_standings_goals_for"
    assert team_row["odds"] in {-120, 100}
    assert team_row["decision"] == "PASS"
    assert team_row["units"] == 0
    assert team_row["market_priced"] is True
    prop = next(pick for pick in quoted["picks"] if pick["market"] == "player_props")
    assert prop["direction"] == "under"
    assert prop["odds"] == -105
    assert prop["units"] == 0
    assert prop["mean_source"] == "nhl_skater_summary"
    assert abs(prop["mean"] - (261 / 77)) < 1e-4
    assert "score" not in prop


def test_team_total_uses_the_balanced_posted_line_not_the_first_alternate():
    from NHLPredictionModel.nhl_markets import choose_balanced_line, fetch_pregame_quotes

    chosen = choose_balanced_line([
        {"line": 0.5, "over_odds": -4500, "under_odds": 1200},
        {"line": 3.5, "over_odds": -125, "under_odds": -115},
        {"line": 4.5, "over_odds": 195, "under_odds": -270},
        {"line": 2.5, "over_odds": -190, "under_odds": 145},
    ])
    assert chosen is not None
    assert chosen["line"] == 3.5

    def fetch(url: str):
        if not url.endswith("/leagues/42133"):
            return {"markets": [], "selections": []}
        return {
            "events": [{
                "id": "reg",
                "status": "NOT_STARTED",
                "startEventDate": "2026-10-10T23:00:00.0000000Z",
                "participants": [
                    {"name": "BOS Bruins", "venueRole": "Home", "type": "Team", "metadata": {"shortName": "BOS"}},
                    {"name": "PHI Flyers", "venueRole": "Away", "type": "Team", "metadata": {"shortName": "PHI"}},
                ],
            }],
            "markets": [{"id": "tt", "eventId": "reg", "name": "BOS Bruins: Team Total Goals", "marketType": {"name": "Team Total Goals"}}],
            "selections": [
                {"marketId": "tt", "label": "Over", "outcomeType": "Over", "points": 0.5, "displayOdds": {"american": "-4500"}, "participants": [{"type": "Team", "venueRole": "Home", "name": "BOS Bruins"}]},
                {"marketId": "tt", "label": "Under", "outcomeType": "Under", "points": 0.5, "displayOdds": {"american": "+1200"}, "participants": [{"type": "Team", "venueRole": "Home", "name": "BOS Bruins"}]},
                {"marketId": "tt", "label": "Over", "outcomeType": "Over", "points": 3.5, "displayOdds": {"american": "+115"}, "participants": [{"type": "Team", "venueRole": "Home", "name": "BOS Bruins"}]},
                {"marketId": "tt", "label": "Under", "outcomeType": "Under", "points": 3.5, "displayOdds": {"american": "-155"}, "participants": [{"type": "Team", "venueRole": "Home", "name": "BOS Bruins"}]},
            ],
        }

    games = [_game(season_type="REG", start_time="2026-10-10T23:00:00Z")]
    fetch_pregame_quotes(games, fetch_json=fetch)
    assert games[0]["team_totals"]["home"]["line"] == 3.5
    assert games[0]["team_totals"]["home"]["over_odds"] == 115
    assert games[0]["team_totals"]["home"]["alternate_lines"] == 2


def test_draftkings_board_attaches_only_posted_prices():
    from NHLPredictionModel.nhl_markets import american_odds, fetch_pregame_quotes

    assert american_odds("\u2212245") == -245
    assert american_odds("+195") == 195
    assert american_odds("even") is None

    def fetch(url: str):
        if url.endswith("/leagues/26150"):
            return {
                "events": [{
                    "id": "34707252",
                    "status": "NOT_STARTED",
                    "startEventDate": "2026-09-22T23:00:00.0000000Z",
                    "participants": [
                        {"name": "WPG Jets", "venueRole": "Home", "metadata": {"shortName": "WPG"}},
                        {"name": "COL Avalanche", "venueRole": "Away", "metadata": {"shortName": "COL"}},
                    ],
                }, {
                    "id": "later",
                    "status": "NOT_STARTED",
                    "startEventDate": "2026-10-03T01:10:00.0000000Z",
                    "participants": [
                        {"name": "WPG Jets", "venueRole": "Home", "metadata": {"shortName": "WPG"}},
                        {"name": "COL Avalanche", "venueRole": "Away", "metadata": {"shortName": "COL"}},
                    ],
                }, {
                    "id": "34707245",
                    "status": "IN_PROGRESS",
                    "participants": [
                        {"name": "CBJ Blue Jackets", "venueRole": "Home", "metadata": {"shortName": "CBJ"}},
                        {"name": "DET Red Wings", "venueRole": "Away", "metadata": {"shortName": "DET"}},
                    ],
                }],
                "markets": [
                    {"id": "ml", "eventId": "34707252", "name": "Moneyline"},
                    {"id": "pl", "eventId": "34707252", "name": "Puck Line"},
                    {"id": "tot", "eventId": "34707252", "name": "Total"},
                    {"id": "live", "eventId": "34707245", "name": "Moneyline"},
                    {"id": "later-ml", "eventId": "later", "name": "Moneyline"},
                ],
                "selections": [
                    {"marketId": "ml", "outcomeType": "Home", "displayOdds": {"american": "\u2212238"}},
                    {"marketId": "ml", "outcomeType": "Away", "displayOdds": {"american": "+195"}},
                    {"marketId": "pl", "outcomeType": "Home", "points": -1.5, "displayOdds": {"american": "+114"}},
                    {"marketId": "pl", "outcomeType": "Away", "points": 1.5, "displayOdds": {"american": "\u2212135"}},
                    {"marketId": "tot", "outcomeType": "Over", "points": 5.5, "displayOdds": {"american": "\u2212115"}},
                    {"marketId": "tot", "outcomeType": "Under", "points": 5.5, "displayOdds": {"american": "\u2212105"}},
                    {"marketId": "live", "outcomeType": "Home", "displayOdds": {"american": "\u2212150"}},
                    {"marketId": "later-ml", "outcomeType": "Home", "displayOdds": {"american": "\u2212205"}},
                    {"marketId": "later-ml", "outcomeType": "Away", "displayOdds": {"american": "+170"}},
                ],
            }
        if "/events/34707252/categories" in url:
            return {"markets": [{"id": "ml2", "name": "Moneyline"}], "selections": []}
        return None

    games = [
        _game(home_abbrev="WPG", away_abbrev="COL", home_team="Winnipeg Jets", away_team="Colorado Avalanche"),
        _game(home_abbrev="CBJ", away_abbrev="DET", home_team="Columbus Blue Jackets", away_team="Detroit Red Wings"),
    ]
    summary = fetch_pregame_quotes(games, fetch_json=fetch)
    assert summary["matched_games"] == 1
    assert games[0]["home_moneyline"] == -238
    assert games[0]["away_moneyline"] == 195
    assert games[0]["spread_line"] == -1.5
    assert games[0]["total_line"] == 5.5
    assert games[0]["over_odds"] == -115
    assert datetime.fromisoformat(games[0]["market_retrieved_at"].replace("Z", "+00:00")).tzinfo is not None
    assert "team_totals" not in games[0]
    assert "player_props" not in games[0]
    assert "home_moneyline" not in games[1]


def test_draftkings_does_not_pair_lines_from_different_numbers():
    from NHLPredictionModel.nhl_markets import fetch_pregame_quotes

    def fetch(url: str):
        if not url.endswith("/leagues/42133"):
            return {"markets": [], "selections": []}
        return {
            "events": [{
                "id": "game", "status": "NOT_STARTED", "startEventDate": "2026-10-10T23:00:00Z",
                "participants": [
                    {"venueRole": "Home", "metadata": {"shortName": "BOS"}},
                    {"venueRole": "Away", "metadata": {"shortName": "PHI"}},
                ],
            }],
            "markets": [
                {"id": "ml", "eventId": "game", "name": "Moneyline"},
                {"id": "pl", "eventId": "game", "name": "Puck Line"},
                {"id": "total", "eventId": "game", "name": "Total"},
            ],
            "selections": [
                {"marketId": "ml", "outcomeType": "Home", "displayOdds": {"american": "-140"}},
                {"marketId": "ml", "outcomeType": "Away", "displayOdds": {"american": "+120"}},
                {"marketId": "pl", "outcomeType": "Home", "points": -1.5, "displayOdds": {"american": "+110"}},
                {"marketId": "pl", "outcomeType": "Away", "points": 2.5, "displayOdds": {"american": "-135"}},
                {"marketId": "total", "outcomeType": "Over", "points": 5.5, "displayOdds": {"american": "-110"}},
                {"marketId": "total", "outcomeType": "Under", "points": 6.5, "displayOdds": {"american": "-110"}},
            ],
        }

    games = [_game(season_type="REG", start_time="2026-10-10T23:00:00Z")]
    fetch_pregame_quotes(games, fetch_json=fetch)
    assert games[0]["home_moneyline"] == -140
    assert "spread_line" not in games[0]
    assert "total_line" not in games[0]


def test_training_maps_next_completed_spring_and_rejects_duplicate_teams():
    from NHLPredictionModel.nhl_train import train

    rows = [{
        "teamAbbrev": {"default": f"T{index:02d}"},
        "teamName": {"default": f"Team {index}"},
        "date": "2027-04-17", "gameTypeId": 2,
        "gamesPlayed": 82, "goalFor": 250, "goalAgainst": 250,
        "homeGamesPlayed": 41, "homeGoalsFor": 130, "homeGoalsAgainst": 120,
        "homeWins": 25, "homeRegulationWins": 20, "homeOtLosses": 5,
    } for index in range(32)]
    artifact = train({"standings": rows})
    assert artifact["prior_season"] == "20262027"
    assert artifact["model_version"] == "nhl_poisson_v2_20262027_shadow_ev_v1"
    rows[1]["teamAbbrev"] = rows[0]["teamAbbrev"]
    with pytest.raises(RuntimeError, match="duplicate standings team"):
        train({"standings": rows})


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
    from scripts.merge_model_cache_payload import DEPLOYED_MODEL_KEYS, UNPRICED_STAKE_DEMOTION_KEYS
    from scripts.model_versions import MODEL_DIRECTORIES
    from scripts.pick_calibration import CALIBRATION_EXCLUDED_MODEL_KEYS
    from scripts.refresh_model_cache import _model_jobs
    from scripts.team_prop_model_evaluator import FEATURE_CONTRACTS, SUPPORTED_MODEL_KEYS
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
    assert "nhl" in FEATURE_CONTRACTS
    assert MODEL_DIRECTORIES["nhl"] == "NHLPredictionModel"
    assert SPORT_LEAGUES["NHL"] == ("hockey", "nhl")
    assert "nhl" in server._EXTERNAL_FEED_OPTIONAL_SPORTS
    data_ts = (ROOT / "src" / "data.ts").read_text(encoding="utf-8")
    main_ts = (ROOT / "src" / "main.ts").read_text(encoding="utf-8")
    assert "nhl: 'NHL Model'" in data_ts
    assert "nhl: { h2h: 'NHL ML'" in data_ts
    assert "team_total: 'NHL Team Total'" in data_ts
    assert "player_props: 'NHL Player Prop'" in data_ts
    assert "'NHL'" in main_ts.split("PRIMARY_FILTERS")[1][:120]
    assert "NHL: ['hockey', 'nhl']" in main_ts
    workflow = (ROOT / ".github" / "workflows" / "model-cache-refresh.yml").read_text(encoding="utf-8")
    assert ",nhl,tennis" in workflow or ",nhl," in workflow
    refresh = (ROOT / "scripts" / "refresh_model_cache.py").read_text(encoding="utf-8")
    assert "nhl" in refresh
    assert "run_nhl_model" in refresh


def test_nhl_generator_promotes_only_the_approved_market(monkeypatch):
    from NHLPredictionModel import nhl_model
    from scripts import model_stake_policy
    from scripts.frozen_staking_candidate import load_freeze
    freeze = load_freeze("nhl")
    # Synthetic approval fixture, isolated from the production approval file.
    approval = {"model_key": "nhl", "model_version": freeze["fitted_version"],
                "market": "spread", "variant": "base", "approved": True,
                "frozen_rule": freeze["frozen_rule"],
                "holdout": {"unused_during_selection": True, "independently_priced_settled": 100,
                            "roi": .1, "clustered_lower_95": .01, "calibration_no_material_regression": True}}
    monkeypatch.setattr(model_stake_policy, "load_approvals", lambda: {"approvals": [approval]})
    game = _game(season_type="REG", start_time="2026-10-10T23:00:00Z",
                 odds_source="draftkings", market_retrieved_at="2026-10-10T18:00:00Z",
                 home_moneyline=-140, away_moneyline=120,
                 spread_line=-1.5, home_spread_odds=-110, away_spread_odds=-110,
                 total_line=6.5, over_odds=-105, under_odds=-115)
    result = nhl_model.generate_nhl_picks("2026-10-10", games=[game], now=datetime(2026, 10, 10, 18, tzinfo=timezone.utc))
    rows = {row["market"]: row for row in result["picks"]}
    assert rows["spread"]["decision"] == "BET" and rows["spread"]["units"] == .5
    assert rows["spread"]["actionability"] == "approved"
    assert rows["spread"]["decision_reason"] == "approved_holdout"
    assert rows["h2h"]["decision"] == rows["totals"]["decision"] == "PASS"
    assert result["coverage"]["staked_rows"] == 1
