from __future__ import annotations

from copy import deepcopy
from typing import Any


class FakeDocumentSnapshot:
    def __init__(self, doc_id: str, data: dict[str, Any] | None):
        self.id = doc_id
        self._data = deepcopy(data) if data is not None else None
        self.exists = data is not None

    def to_dict(self) -> dict[str, Any] | None:
        return deepcopy(self._data) if self._data is not None else None


class FakeDocumentReference:
    def __init__(self, store: dict[str, dict[str, Any]], doc_id: str):
        self._store = store
        self._doc_id = doc_id

    def get(self) -> FakeDocumentSnapshot:
        return FakeDocumentSnapshot(self._doc_id, self._store.get(self._doc_id))

    def set(self, payload: dict[str, Any], merge: bool = False) -> None:
        if merge:
            current = self._store.setdefault(self._doc_id, {})
            current.update(deepcopy(payload))
            return
        self._store[self._doc_id] = deepcopy(payload)


class FakeCollectionReference:
    def __init__(self, store: dict[str, dict[str, Any]]):
        self._store = store

    def stream(self) -> list[FakeDocumentSnapshot]:
        return [
            FakeDocumentSnapshot(doc_id, data)
            for doc_id, data in self._store.items()
        ]

    def document(self, doc_id: str) -> FakeDocumentReference:
        return FakeDocumentReference(self._store, doc_id)


class FakeFirestoreClient:
    def __init__(self, users: dict[str, dict[str, Any]]):
        self.users = users

    def collection(self, name: str) -> FakeCollectionReference:
        assert name == "users"
        return FakeCollectionReference(self.users)


def test_background_grader_preserves_existing_record(monkeypatch):
    import pickgrader_server

    record = {"wins": 604, "losses": 531, "pushes": 0}
    users = {
        "test-pickledger-smoke": {
            "record": deepcopy(record),
            "picks": [
                {
                    "id": "smoke-pick",
                    "sport": "NBA",
                    "date": "Jan 1",
                    "pick": "Lakers ML (Lakers vs Celtics)",
                }
            ],
            "results": {"smoke-pick": "pending"},
            "startTimes": {},
            "ledger": {
                "addedPicks": [],
                "results": {"smoke-pick": "pending"},
                "gameTimes": {},
            },
        }
    }

    monkeypatch.setattr(
        pickgrader_server,
        "_get_firestore_client",
        lambda: FakeFirestoreClient(users),
    )
    monkeypatch.setattr(
        pickgrader_server,
        "auto_grade",
        lambda picks, existing, year: {
            "graded": {"smoke-pick": "win"},
            "startTimes": {"smoke-pick": "2025-01-01T20:00:00Z"},
            "summary": {"attempted": 1, "updated": 1, "remaining": 0},
        },
    )

    summary = pickgrader_server.run_background_grade_all_users()

    assert summary["graded_users"] == 1
    assert not summary["errors"]
    user_doc = users["test-pickledger-smoke"]
    assert user_doc["record"] == record
    assert user_doc["results"]["smoke-pick"] == "win"
    assert user_doc["ledger"]["results"]["smoke-pick"] == "win"
    assert user_doc["startTimes"]["smoke-pick"] == "2025-01-01T20:00:00Z"
    assert "lastGraded" in user_doc


def test_grade_pick_moneyline_result_without_network():
    import pickgrader_server

    game = {
        "competitors": [
            {
                "raw": {
                    "team": {
                        "displayName": "Los Angeles Lakers",
                        "shortDisplayName": "Lakers",
                        "name": "Lakers",
                        "abbreviation": "LAL",
                    }
                },
                "score": 112,
                "homeAway": "home",
                "linescores": [],
            },
            {
                "raw": {
                    "team": {
                        "displayName": "Boston Celtics",
                        "shortDisplayName": "Celtics",
                        "name": "Celtics",
                        "abbreviation": "BOS",
                    }
                },
                "score": 100,
                "homeAway": "away",
                "linescores": [],
            },
        ],
        "startTime": "2025-01-01T20:00:00Z",
        "eventId": "smoke",
    }
    pick = {
        "id": "smoke-pick",
        "sport": "NBA",
        "pick": "Lakers ML (Lakers vs Celtics)",
    }

    assert pickgrader_server.grade_pick(pick, game) == "win"
    assert pickgrader_server.grade_pick(
        {**pick, "pick": "Lakers to Win (Lakers vs Celtics)"},
        game,
    ) == "win"


def test_soccer_three_way_moneyline_loses_on_draw():
    import pickgrader_server

    game = {
        "competitors": [
            {
                "raw": {"team": {"displayName": "Brazil", "name": "Brazil", "abbreviation": "BRA"}},
                "score": 1,
                "homeAway": "home",
                "linescores": [],
            },
            {
                "raw": {"team": {"displayName": "Morocco", "name": "Morocco", "abbreviation": "MAR"}},
                "score": 1,
                "homeAway": "away",
                "linescores": [],
            },
        ],
        "startTime": "2026-06-13T22:00:00Z",
        "eventId": "wc-smoke",
    }
    pick = {
        "id": "wc-moneyline",
        "sport": "FIFA WC",
        "pick": "Brazil ML (Morocco @ Brazil)",
        "market_type": "soccer_moneyline",
    }

    assert pickgrader_server.grade_pick(pick, game) == "loss"


def test_scores24_comma_totals_and_wnba_handicap_grade_correctly():
    import pickgrader_server

    mlb_game = {
        "competitors": [
            {"raw": {"team": {"displayName": "Boston Red Sox"}}, "score": 3},
            {"raw": {"team": {"displayName": "Toronto Blue Jays"}}, "score": 4},
        ],
    }
    wnba_game = {
        "competitors": [
            {"raw": {"team": {"displayName": "Indiana Fever"}}, "score": 101},
            {"raw": {"team": {"displayName": "Atlanta Dream"}}, "score": 108},
        ],
    }

    assert pickgrader_server.grade_pick(
        {
            "sport": "MLB",
            "pick": "Total Under (8,5) (Toronto Blue Jays @ Boston Red Sox)",
            "tip": "Total Under (8,5)",
        },
        mlb_game,
    ) == "win"
    assert pickgrader_server.grade_pick(
        {
            "sport": "WNBA",
            "pick": "Indiana Fever (W) Handicap (+7,5) (Atlanta Dream @ Indiana Fever)",
            "tip": "Indiana Fever (W) Handicap (+7,5)",
            "away_team": "Atlanta Dream",
            "home_team": "Indiana Fever",
        },
        wnba_game,
    ) == "win"
    assert pickgrader_server.grade_pick(
        {
            "sport": "MLB",
            "pick": "Toronto Blue Jays Total Over (3,5) (Toronto Blue Jays @ Boston Red Sox)",
            "tip": "Toronto Blue Jays Total Over (3,5)",
            "away_team": "Toronto Blue Jays",
            "home_team": "Boston Red Sox",
        },
        mlb_game,
    ) == "win"
    assert pickgrader_server.grade_pick(
        {
            "sport": "WNBA",
            "pick": "Indiana Fever (W) Total points Under (101,5) (Atlanta Dream @ Indiana Fever)",
            "tip": "Indiana Fever (W) Total points Under (101,5)",
            "away_team": "Atlanta Dream",
            "home_team": "Indiana Fever",
        },
        wnba_game,
    ) == "win"


def test_fifa_world_cup_moneyline_total_and_handicap_results():
    import pickgrader_server

    draw = {
        "competitors": [
            {"raw": {"team": {"displayName": "Qatar"}}, "score": 1},
            {"raw": {"team": {"displayName": "Switzerland"}}, "score": 1},
        ],
    }
    scotland_win = {
        "competitors": [
            {"raw": {"team": {"displayName": "Haiti"}}, "score": 0},
            {"raw": {"team": {"displayName": "Scotland"}}, "score": 1},
        ],
    }
    brazil_draw = {
        "competitors": [
            {"raw": {"team": {"displayName": "Brazil"}}, "score": 1},
            {"raw": {"team": {"displayName": "Morocco"}}, "score": 1},
        ],
    }

    assert pickgrader_server.grade_pick(
        {"sport": "FIFA WC", "pick": "Switzerland -1.5 (Switzerland @ Qatar)", "market_type": "soccer_handicap"},
        draw,
    ) == "loss"
    assert pickgrader_server.grade_pick(
        {"sport": "FIFA WC", "pick": "Scotland ML (Scotland @ Haiti)", "market_type": "soccer_moneyline"},
        scotland_win,
    ) == "win"
    assert pickgrader_server.grade_pick(
        {"sport": "FIFA WC", "pick": "Under 2.5 Goals (Brazil vs Morocco)", "market_type": "soccer_total"},
        brazil_draw,
    ) == "win"


def test_auto_grade_accepts_iso_dates_and_pushes_canceled_games(monkeypatch):
    import pickgrader_server

    scoreboard = {
        "events": [
            {
                "id": "canceled-smoke",
                "competitions": [
                    {
                        "date": "2026-06-08T20:00:00Z",
                        "status": {"type": {"completed": False, "name": "STATUS_CANCELED"}},
                        "competitors": [
                            {
                                "score": "0",
                                "homeAway": "home",
                                "team": {
                                    "displayName": "Los Angeles Lakers",
                                    "shortDisplayName": "Lakers",
                                    "name": "Lakers",
                                    "abbreviation": "LAL",
                                },
                            },
                            {
                                "score": "0",
                                "homeAway": "away",
                                "team": {
                                    "displayName": "Boston Celtics",
                                    "shortDisplayName": "Celtics",
                                    "name": "Celtics",
                                    "abbreviation": "BOS",
                                },
                            },
                        ],
                    }
                ],
            }
        ]
    }
    monkeypatch.setattr(pickgrader_server, "fetch_scoreboard", lambda *_: scoreboard)

    result = pickgrader_server.auto_grade(
        [
            {
                "id": "iso-date-pick",
                "sport": "NBA",
                "date": "2026-06-08",
                "pick": "Lakers ML (Lakers vs Celtics)",
            }
        ],
        {},
        2026,
    )

    assert pickgrader_server.parse_pick_date("2026-06-08", 2026) == "20260608"
    assert result["graded"] == {"iso-date-pick": "push"}
    assert result["startTimes"] == {"iso-date-pick": "2026-06-08T20:00:00Z"}


def test_grade_mlb_first_five_markets_without_network():
    import pickgrader_server

    game = {
        "competitors": [
            {
                "raw": {
                    "team": {
                        "displayName": "Boston Red Sox",
                        "shortDisplayName": "Red Sox",
                        "name": "Red Sox",
                        "abbreviation": "BOS",
                    }
                },
                "score": 4,
                "homeAway": "home",
                "linescores": [
                    {"value": 1}, {"value": 0}, {"value": 1}, {"value": 0}, {"value": 0},
                    {"value": 2}, {"value": 0}, {"value": 0}, {"value": 0},
                ],
            },
            {
                "raw": {
                    "team": {
                        "displayName": "Tampa Bay Rays",
                        "shortDisplayName": "Rays",
                        "name": "Rays",
                        "abbreviation": "TB",
                    }
                },
                "score": 3,
                "homeAway": "away",
                "linescores": [
                    {"value": 0}, {"value": 0}, {"value": 0}, {"value": 1}, {"value": 0},
                    {"value": 0}, {"value": 1}, {"value": 1}, {"value": 0},
                ],
            },
        ],
        "startTime": "2026-05-10T17:35:00Z",
        "eventId": "mlb-f5-smoke",
    }

    assert pickgrader_server.grade_pick(
        {"sport": "MLB", "pick": "Boston Red Sox F5 ML", "team": "Boston Red Sox", "market": "f5_side"},
        game,
    ) == "win"
    assert pickgrader_server.grade_pick(
        {"sport": "MLB", "pick": "Under 4.5 F5", "market": "f5_total"},
        game,
    ) == "win"


def test_grade_structured_wnba_player_prop_from_boxscore():
    import pickgrader_server

    summary = {
        "boxscore": {
            "players": [{
                "statistics": [{
                    "labels": ["MIN", "PTS", "REB", "AST"],
                    "athletes": [{
                        "athlete": {"displayName": "Brittney Sykes"},
                        "stats": ["34", "24", "5", "7"],
                    }],
                }],
            }],
        },
    }
    pick = {
        "scope": "player",
        "sport": "WNBA",
        "player_name": "Brittney Sykes",
        "stat_key": "points",
        "selection": "OVER",
        "line": 20.5,
        "pick": "Brittney Sykes points OVER 20.5 vs Tempo",
    }

    assert pickgrader_server.parse_player_prop_pick(pick)["stat_key"] == "points"
    assert pickgrader_server.parse_nba_player_prop_pick(pick["pick"])["stat_key"] == "points"
    assert pickgrader_server.grade_player_prop_pick(pick, {}, summary) == "win"


def test_grade_aneesah_morrow_total_rebounds_from_boxscore():
    import pickgrader_server

    summary = {
        "boxscore": {
            "players": [{
                "statistics": [{
                    "labels": ["MIN", "PTS", "REB", "AST"],
                    "athletes": [{
                        "athlete": {"displayName": "Aneesah Morrow"},
                        "stats": ["21", "8", "5", "2"],
                    }],
                }],
            }],
        },
    }
    pick = {
        "scope": "player",
        "sport": "WNBA",
        "player_name": "Aneesah Morrow",
        "stat_key": "totalRebounds",
        "selection": "Over",
        "line": 10.5,
        "pick": "Aneesah Morrow Over 10.5 Rebounds",
    }
    high_scoring_game = {"competitors": [{"score": 80}, {"score": 75}]}

    assert pickgrader_server.parse_player_prop_pick(pick)["stat_key"] == "rebounds"
    assert pickgrader_server.grade_player_prop_pick(pick, high_scoring_game, summary) == "loss"
    assert pickgrader_server.grade_pick(pick, high_scoring_game) == "pending"


def test_grade_expanded_wnba_combo_props_from_boxscore():
    import pickgrader_server

    summary = {
        "boxscore": {
            "players": [{
                "statistics": [{
                    "labels": ["MIN", "PTS", "REB", "AST", "3PM", "STL", "BLK"],
                    "athletes": [{
                        "athlete": {"displayName": "Napheesa Collier"},
                        "stats": ["35", "22", "9", "5", "3", "2", "1"],
                    }],
                }],
            }],
        },
    }

    assert pickgrader_server.grade_player_prop_pick(
        {
            "scope": "player",
            "sport": "WNBA",
            "player_name": "Napheesa Collier",
            "stat_key": "points_rebounds_assists",
            "selection": "Over",
            "line": 34.5,
            "pick": "Napheesa Collier Over 34.5 Points + Rebounds + Assists",
        },
        {},
        summary,
    ) == "win"
    assert pickgrader_server.grade_player_prop_pick(
        {
            "scope": "player",
            "sport": "WNBA",
            "player_name": "Napheesa Collier",
            "stat_key": "steals_blocks",
            "selection": "Over",
            "line": 2.5,
            "pick": "Napheesa Collier Over 2.5 Steals + Blocks",
        },
        {},
        summary,
    ) == "win"
    assert pickgrader_server.grade_player_prop_pick(
        {
            "scope": "player",
            "sport": "WNBA",
            "player_name": "Napheesa Collier",
            "stat_key": "three_pointers_made",
            "selection": "Under",
            "line": 3.5,
            "pick": "Napheesa Collier Under 3.5 3PM",
        },
        {},
        summary,
    ) == "win"


def test_grade_wnba_three_point_made_from_espn_made_attempted_format():
    import pickgrader_server

    summary = {
        "boxscore": {
            "players": [{
                "statistics": [{
                    "labels": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST"],
                    "athletes": [
                        {
                            "athlete": {"displayName": "Breanna Stewart"},
                            "stats": ["33", "29", "10-16", "0-3", "9-11", "9", "4"],
                        },
                        {
                            "athlete": {"displayName": "Paige Bueckers"},
                            "stats": ["33", "15", "5-15", "1-5", "4-4", "7", "6"],
                        },
                    ],
                }],
            }],
        },
    }
    assert pickgrader_server.grade_player_prop_pick(
        {
            "scope": "player",
            "sport": "WNBA",
            "player_name": "Breanna Stewart",
            "stat_key": "three_pointers_made",
            "selection": "Over",
            "line": 0.5,
            "pick": "Breanna Stewart Over 0.5 3-Point Field Goals",
        },
        {},
        summary,
    ) == "loss"
    assert pickgrader_server.grade_player_prop_pick(
        {
            "scope": "player",
            "sport": "WNBA",
            "player_name": "Paige Bueckers",
            "stat_key": "three_pointers_made",
            "selection": "Over",
            "line": 1.5,
            "pick": "Paige Bueckers Over 1.5 3-Point Field Goals",
        },
        {},
        summary,
    ) == "loss"


def test_grade_wnba_three_point_dnp_counts_as_zero():
    import pickgrader_server

    summary = {
        "boxscore": {
            "players": [{
                "statistics": [{
                    "labels": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST"],
                    "athletes": [
                        {
                            "athlete": {"displayName": "Skylar Diggins"},
                            "stats": [],
                        },
                    ],
                }],
            }],
        },
    }
    assert pickgrader_server.grade_player_prop_pick(
        {
            "scope": "player",
            "sport": "WNBA",
            "player_name": "Skylar Diggins",
            "stat_key": "three_pointers_made",
            "selection": "Under",
            "line": 1.5,
            "pick": "Skylar Diggins Under 1.5 3-Point Field Goals",
        },
        {},
        summary,
    ) == "win"


def test_grade_external_threshold_player_prop():
    import pickgrader_server

    summary = {
        "boxscore": {
            "players": [{
                "statistics": [{
                    "labels": ["IP", "H", "K"],
                    "athletes": [{
                        "athlete": {"displayName": "Shohei Ohtani"},
                        "stats": ["6.0", "4", "7"],
                    }],
                }],
            }],
        },
    }
    pick = {
        "scope": "player",
        "sport": "MLB",
        "pick": "Shohei Ohtani 7+ Strikeouts (Pirates vs Dodgers)",
    }

    assert pickgrader_server.parse_player_prop_pick(pick)["selection"] == "AT_LEAST"
    assert pickgrader_server.grade_player_prop_pick(pick, {}, summary) == "win"


def test_grade_structured_mlb_player_props_from_boxscore():
    import pickgrader_server

    summary = {
        "boxscore": {
            "players": [{
                "statistics": [
                    {
                        "labels": ["H-AB", "H", "R", "RBI", "K"],
                        "athletes": [{
                            "athlete": {"displayName": "Otto Lopez"},
                            "stats": ["2-4", "2", "1", "2", "1"],
                        }],
                    },
                    {
                        "labels": ["IP", "H", "K"],
                        "athletes": [{
                            "athlete": {"displayName": "Sandy Alcantara"},
                            "stats": ["6.0", "5", "7"],
                        }],
                    },
                ],
            }],
        },
    }
    hitter = {
        "scope": "player",
        "sport": "MLB",
        "player_name": "Otto Lopez",
        "stat_key": "hits",
        "selection": "OVER",
        "line": 0.5,
        "pick": "Otto Lopez hits OVER 0.5 vs Pirates",
    }
    pitcher = {
        "scope": "player",
        "sport": "MLB",
        "player_name": "Sandy Alcantara",
        "stat_key": "strikeouts",
        "selection": "OVER",
        "line": 5.5,
        "pick": "Sandy Alcantara strikeouts OVER 5.5 vs Pirates",
    }
    hrr = {
        "scope": "player",
        "sport": "MLB",
        "player_name": "Otto Lopez",
        "stat_key": "hits_runs_rbis",
        "selection": "OVER",
        "line": 3.5,
        "pick": "Otto Lopez Over 3.5 Hits + Runs + RBIs",
    }

    assert pickgrader_server.grade_player_prop_pick(hitter, {}, summary) == "win"
    assert pickgrader_server.grade_player_prop_pick(pitcher, {}, summary) == "win"
    assert pickgrader_server.grade_player_prop_pick(hrr, {}, summary) == "win"


def test_grade_expanded_mlb_player_props_from_boxscore():
    import pickgrader_server

    summary = {
        "boxscore": {
            "players": [{
                "statistics": [
                    {
                        "labels": ["H-AB", "H", "R", "RBI", "BB", "K", "2B", "3B", "HR", "SB"],
                        "athletes": [{
                            "athlete": {"displayName": "Otto Lopez"},
                            "stats": ["3-5", "3", "2", "1", "1", "0", "1", "0", "1", "1"],
                        }],
                    },
                    {
                        "labels": ["IP", "H", "ER", "BB", "K"],
                        "athletes": [{
                            "athlete": {"displayName": "Sandy Alcantara"},
                            "stats": ["6.1", "4", "2", "1", "8"],
                        }],
                    },
                ],
            }],
        },
    }

    assert pickgrader_server.grade_player_prop_pick(
        {"scope": "player", "sport": "MLB", "player_name": "Otto Lopez", "stat_key": "total_bases", "selection": "Over", "line": 6.5, "pick": "Otto Lopez Over 6.5 Total Bases"},
        {},
        summary,
    ) == "win"
    assert pickgrader_server.grade_player_prop_pick(
        {"scope": "player", "sport": "MLB", "player_name": "Otto Lopez", "stat_key": "singles", "selection": "Under", "line": 1.5, "pick": "Otto Lopez Under 1.5 Singles"},
        {},
        summary,
    ) == "win"
    assert pickgrader_server.grade_player_prop_pick(
        {"scope": "player", "sport": "MLB", "player_name": "Otto Lopez", "stat_key": "batter_walks", "selection": "Over", "line": 0.5, "pick": "Otto Lopez Over 0.5 Walks"},
        {},
        summary,
    ) == "win"
    assert pickgrader_server.grade_player_prop_pick(
        {"scope": "player", "sport": "MLB", "player_name": "Sandy Alcantara", "stat_key": "pitcher_outs_recorded", "selection": "Over", "line": 18.5, "pick": "Sandy Alcantara Over 18.5 Outs Recorded"},
        {},
        summary,
    ) == "win"
    assert pickgrader_server.grade_player_prop_pick(
        {"scope": "player", "sport": "MLB", "player_name": "Sandy Alcantara", "stat_key": "pitcher_earned_runs_allowed", "selection": "Under", "line": 2.5, "pick": "Sandy Alcantara Under 2.5 Earned Runs Allowed"},
        {},
        summary,
    ) == "win"


def test_grade_mlb_props_from_official_live_feed_when_espn_omits_stats():
    import pickgrader_server

    live_feed = {
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {}},
                    "home": {
                        "players": {
                            "ID123": {
                                "person": {"fullName": "Zack Gelof"},
                                "stats": {
                                    "batting": {
                                        "hits": 3,
                                        "runs": 2,
                                        "rbi": 2,
                                        "doubles": 0,
                                        "triples": 0,
                                        "homeRuns": 1,
                                        "totalBases": 6,
                                        "stolenBases": 1,
                                    }
                                },
                            }
                        }
                    },
                }
            }
        }
    }

    assert pickgrader_server.grade_player_prop_pick(
        {"sport": "MLB", "player_name": "Zack Gelof", "stat_key": "total_bases", "selection": "Over", "line": 1.5, "pick": "Zack Gelof Over 1.5 Total Bases"},
        {},
        None,
        live_feed,
    ) == "win"
    assert pickgrader_server.grade_player_prop_pick(
        {"sport": "MLB", "player_name": "Zack Gelof", "stat_key": "stolen_bases", "selection": "Over", "line": 0.5, "pick": "Zack Gelof Over 0.5 Stolen Bases"},
        {},
        None,
        live_feed,
    ) == "win"


def test_final_mlb_player_prop_with_listed_inactive_player_pushes():
    import pickgrader_server

    live_feed = {
        "gameData": {"status": {"abstractGameState": "Final", "codedGameState": "F"}},
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {}},
                    "home": {
                        "players": {
                            "ID691723": {
                                "person": {"id": 691723, "fullName": "Coby Mayo"},
                                "stats": {"batting": {}},
                            }
                        }
                    },
                }
            }
        },
    }

    assert pickgrader_server.grade_player_prop_pick(
        {
            "sport": "MLB",
            "player_id": "691723",
            "player_name": "Coby Mayo",
            "stat_key": "hits",
            "selection": "Under",
            "line": 0.5,
            "pick": "Coby Mayo Under 0.5 Hits",
        },
        {},
        None,
        live_feed,
    ) == "push"


def test_auto_grade_mlb_player_props_from_official_feed_without_espn_completed(monkeypatch):
    import pickgrader_server

    pick = {
        "id": "mlb-prop-1",
        "sport": "MLB",
        "date": "2026-06-16",
        "decision": "BET",
        "pick": "Zack Gelof Over 0.5 RBIs",
        "player_name": "Zack Gelof",
        "matchup": "Pittsburgh Pirates @ Athletics",
        "result": "pending",
    }
    live_feed = {
        "gameData": {"status": {"abstractGameState": "Final", "codedGameState": "F"}},
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {}},
                    "home": {
                        "players": {
                            "ID123": {
                                "person": {"fullName": "Zack Gelof"},
                                "stats": {"batting": {"rbi": 1}},
                            }
                        }
                    },
                }
            }
        },
    }

    monkeypatch.setattr(pickgrader_server, "fetch_scoreboard", lambda *args, **kwargs: {"events": []})
    monkeypatch.setattr(pickgrader_server, "fetch_mlb_schedule", lambda *_: {"dates": [{"games": [{"gamePk": 999, "teams": {"away": {"team": {"name": "Pittsburgh Pirates"}}, "home": {"team": {"name": "Athletics"}}}}]}]})
    monkeypatch.setattr(pickgrader_server, "fetch_mlb_live_feed", lambda *_: live_feed)

    result = pickgrader_server.auto_grade([pick], {}, 2026)

    assert result["graded"]["mlb-prop-1"] == "win"


def test_find_mlb_game_pk_matches_structured_matchup():
    import pickgrader_server

    schedule = {
        "dates": [{
            "games": [{
                "gamePk": 823939,
                "teams": {
                    "away": {"team": {"name": "Tampa Bay Rays"}},
                    "home": {"team": {"name": "Los Angeles Dodgers"}},
                },
            }],
        }],
    }
    pick = {"away_team": "Tampa Bay Rays", "home_team": "Los Angeles Dodgers"}

    assert pickgrader_server.find_mlb_game_pk(schedule, pick) == "823939"
