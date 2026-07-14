from __future__ import annotations

from FIFAWorldCupPredictionModel.fifa_world_cup_model import (
    club_power,
    generate_fifa_world_cup_picks,
    poisson_probabilities,
    spread_cover_probability,
)
from scripts.pick_calibration import apply_calibration_to_payload, build_outcome_ledger


def _record(wins: int, ties: int, losses: int, goals_for: int, goals_against: int, rank: int):
    games = wins + ties + losses
    return {
        "record": {
            "items": [{
                "stats": [
                    {"name": "gamesPlayed", "value": games},
                    {"name": "wins", "value": wins},
                    {"name": "ties", "value": ties},
                    {"name": "losses", "value": losses},
                    {"name": "points", "value": (wins * 3) + ties},
                    {"name": "pointsFor", "value": goals_for},
                    {"name": "pointsAgainst", "value": goals_against},
                    {"name": "rank", "value": rank},
                ]
            }]
        }
    }


def _roster(prefix: str):
    positions = ["Goalkeeper", *["Defender"] * 4, *["Midfielder"] * 3, *["Forward"] * 3]
    return {
        "athletes": [
            {
                "id": f"{prefix}-{index}",
                "displayName": f"{prefix} Player {index}",
                "age": 27,
                "position": {"name": position},
                "status": {"type": "active"},
                "injuries": [],
            }
            for index, position in enumerate(positions)
        ]
    }


class FakeFifaClient:
    def scoreboard(self, _date_iso):
        return {
            "events": [{
                "id": "wc-smoke",
                "date": "2026-06-13T20:00Z",
                "status": {"type": {"state": "pre"}},
                "competitions": [{
                    "competitors": [
                        {"homeAway": "home", "team": {"id": "1", "displayName": "Strongland", "abbreviation": "STR"}},
                        {"homeAway": "away", "team": {"id": "2", "displayName": "Weakland", "abbreviation": "WEA"}},
                    ],
                    "odds": [{
                        "overUnder": 2.5,
                        "moneyline": {
                            "home": {"close": {"odds": "-110"}},
                            "away": {"close": {"odds": "+320"}},
                            "draw": {"close": {"odds": "+250"}},
                        },
                        "total": {
                            "over": {"close": {"odds": "-105"}},
                            "under": {"close": {"odds": "-115"}},
                        },
                    }],
                }],
            }]
        }

    def roster(self, team_id):
        return _roster("strong" if team_id == "1" else "weak")

    def athlete(self, athlete_id):
        strong = str(athlete_id).startswith("strong")
        return {
            "athlete": {"team": {"id": "900" if strong else "901", "displayName": "Elite Club" if strong else "Small Club"}},
            "league": {"slug": "eng.1" if strong else "aus.1", "name": "Premier League" if strong else "A-League"},
        }

    def club(self, _league_slug, club_id):
        return {
            "team": _record(24, 8, 6, 78, 34, 2)
            if club_id == "900"
            else _record(7, 8, 15, 31, 53, 11)
        }


def test_fifa_model_is_player_centric_and_emits_team_toggle_rows():
    result = generate_fifa_world_cup_picks("2026-06-13", client=FakeFifaClient(), max_workers=4)

    assert result["ok"] is True
    assert result["calibration_excluded"] is True
    assert len(result["player_rankings"]) == 22
    assert result["team_ratings"][0]["team"] == "Strongland"
    assert len(result["picks"]) == 2
    moneyline, total = result["picks"]
    assert moneyline["pick"].startswith("Strongland ML")
    assert moneyline["sport"] == "FIFA WC"
    assert moneyline["market_type"] == "soccer_moneyline"
    assert "no head-to-head" in moneyline["model_basis"]
    assert moneyline["home_unit_ratings"]["attack"] > moneyline["away_unit_ratings"]["attack"]
    assert total["market_type"] == "soccer_total"


def test_fifa_club_and_poisson_layers_behave_directionally():
    strong = club_power("eng.1", _record(25, 7, 6, 80, 30, 1))
    weak = club_power("aus.1", _record(7, 8, 15, 29, 54, 12))
    probabilities = poisson_probabilities(home_xg=2.2, away_xg=0.7)
    home_cover = spread_cover_probability(home_xg=2.2, away_xg=0.7, side="home", line=-0.5)
    away_cover = spread_cover_probability(home_xg=2.2, away_xg=0.7, side="away", line=0.5)

    assert strong > weak
    assert probabilities["home_win"] > probabilities["draw"] > probabilities["away_win"]
    assert home_cover > away_cover
    assert abs(sum(probabilities.values()) - 1.0) < 1e-9


def test_fifa_model_layers_tournament_form_venue_and_spread_market():
    class ContextClient(FakeFifaClient):
        def scoreboard(self, date_iso):
            if date_iso == "2026-06-12":
                return {
                    "events": [
                        {
                            "id": "history-hot",
                            "date": "2026-06-12T20:00Z",
                            "status": {"type": {"state": "post", "completed": True}},
                            "competitions": [{
                                "venue": {"id": "hot", "fullName": "Goal Dome", "address": {"city": "Test City"}},
                                "competitors": [
                                    {"homeAway": "home", "score": "5", "team": {"id": "1", "displayName": "Strongland", "abbreviation": "STR"}},
                                    {"homeAway": "away", "score": "1", "team": {"id": "2", "displayName": "Weakland", "abbreviation": "WEA"}},
                                ],
                            }],
                        },
                        {
                            "id": "history-low",
                            "date": "2026-06-12T23:00Z",
                            "status": {"type": {"state": "post", "completed": True}},
                            "competitions": [{
                                "venue": {"id": "low", "fullName": "Quiet Park", "address": {"city": "Test City"}},
                                "competitors": [
                                    {"homeAway": "home", "score": "1", "team": {"id": "10", "displayName": "Other A", "abbreviation": "OTA"}},
                                    {"homeAway": "away", "score": "0", "team": {"id": "11", "displayName": "Other B", "abbreviation": "OTB"}},
                                ],
                            }],
                        },
                    ]
                }
            if date_iso != "2026-06-13":
                return {"events": []}
            return {
                "events": [{
                    "id": "wc-context",
                    "date": "2026-06-13T20:00Z",
                    "status": {"type": {"state": "pre"}},
                    "competitions": [{
                        "venue": {"id": "hot", "fullName": "Goal Dome", "address": {"city": "Test City"}},
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {"id": "1", "displayName": "Strongland", "abbreviation": "STR"},
                                "records": [{"summary": "1-0-0"}],
                            },
                            {
                                "homeAway": "away",
                                "team": {"id": "2", "displayName": "Weakland", "abbreviation": "WEA"},
                                "records": [{"summary": "0-0-1"}],
                            },
                        ],
                        "odds": [{
                            "overUnder": 2.5,
                            "moneyline": {
                                "home": {"close": {"odds": "-140"}},
                                "away": {"close": {"odds": "+390"}},
                                "draw": {"close": {"odds": "+300"}},
                            },
                            "total": {
                                "over": {"close": {"odds": "-110"}},
                                "under": {"close": {"odds": "-110"}},
                            },
                            "pointSpread": {
                                "home": {"close": {"line": "-1.5", "odds": "+130"}},
                                "away": {"close": {"line": "+1.5", "odds": "-160"}},
                            },
                        }],
                    }],
                }]
            }

    result = generate_fifa_world_cup_picks("2026-06-13", client=ContextClient(), max_workers=4)

    assert result["ok"] is True
    assert len(result["picks"]) == 3
    markets = {pick["market"]: pick for pick in result["picks"]}
    assert markets["spread"]["market_type"] == "soccer_handicap"
    assert markets["spread"]["line"] in {-1.5, 1.5}
    assert result["games"][0]["venue_profile"]["goal_multiplier"] > 1.0
    assert result["team_ratings"][0]["tournament_form"]["record"] == "1-0-0"
    assert result["tournament_context"]["completed_games"] == 2


def test_fifa_model_passes_when_player_profile_coverage_is_incomplete():
    class IncompleteClient(FakeFifaClient):
        def athlete(self, athlete_id):
            if str(athlete_id).startswith("weak"):
                return {}
            return super().athlete(athlete_id)

    result = generate_fifa_world_cup_picks("2026-06-13", client=IncompleteClient(), max_workers=4)

    assert result["team_ratings"][-1]["roster_ready"] is False
    assert {pick["decision"] for pick in result["picks"]} == {"PASS"}


def test_fifa_bucket_is_excluded_from_calibration_and_training_ledger(tmp_path):
    pick = {
        "source": "FIFA Model",
        "sport": "FIFA WC",
        "pick": "Strongland ML (Weakland @ Strongland)",
        "market_type": "soccer_moneyline",
        "probability": 0.6,
        "edge": 5.0,
        "units": 0.5,
        "decision": "BET",
        "result": "win",
    }
    payload = {"date": "2026-06-13", "models": {"fifa_world_cup": {"ok": True, "picks": [pick]}}}
    active = {
        "version": "test",
        "minimum_group_samples": 1,
        "global": {"intercept": -3.0, "slope": 1.0, "samples": 100},
        "groups": {},
    }

    apply_calibration_to_payload(payload, active)
    assert payload["models"]["fifa_world_cup"]["picks"][0]["probability"] == 0.6
    assert "calibration" not in payload["models"]["fifa_world_cup"]["picks"][0]

    model_dir = tmp_path / "data" / "model_cache"
    props_dir = tmp_path / "data" / "player_props_cache"
    model_dir.mkdir(parents=True)
    props_dir.mkdir(parents=True)
    (model_dir / "2026-06-13.json").write_text(__import__("json").dumps(payload), encoding="utf-8")
    ledger = build_outcome_ledger(tmp_path)
    assert ledger["summary"]["total_picks"] == 0
