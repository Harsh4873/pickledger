from __future__ import annotations

import pytest

from player_props.mlb import _espn_event_market, _game_market_index, _market_index
from player_props.schema import normalize_name


ALL_STAR_GAME = {
    "away_team": "American League All-Stars",
    "home_team": "National League All-Stars",
}


def _all_star_scoreboard() -> dict:
    return {
        "events": [
            {
                "id": "all-star-event",
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "away",
                                "team": {"displayName": "American All-Stars"},
                            },
                            {
                                "homeAway": "home",
                                "team": {"displayName": "National All-Stars"},
                            },
                        ],
                        "odds": [
                            {"provider": {"id": "100", "name": "DraftKings"}}
                        ],
                    }
                ],
            }
        ]
    }


def _market_side(odds: int, type_name: str = "Total Hits") -> dict:
    return {
        "athlete": {
            "$ref": (
                "http://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/"
                "seasons/2026/athletes/123?lang=en&region=us"
            )
        },
        "type": {"name": type_name},
        "odds": {
            "american": {"value": f"{odds:+d}"},
            "total": {"value": "0.5"},
        },
        "current": {"target": {"value": 0.5, "displayValue": "0.5"}},
        "lastUpdated": "2026-07-14T12:00:00Z",
    }


def _milestone_market(type_name: str, threshold: float = 1.0, odds: int = -115) -> dict:
    return {
        "athlete": {
            "$ref": (
                "http://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/"
                "seasons/2026/athletes/123?lang=en&region=us"
            )
        },
        "type": {"name": type_name},
        "odds": {
            "american": {"value": f"{odds:+d}"},
            "total": {"value": f"{threshold:g}+"},
        },
        "current": {
            "target": {"value": threshold, "displayValue": f"{threshold:g}+"}
        },
        "lastUpdated": "2026-07-14T12:00:00Z",
    }


class AllStarMarketClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def mlb_espn_summary(self, event_id: str) -> dict:
        self.calls.append(("summary", event_id))
        return {
            "rosters": [
                {
                    "roster": [
                        {"athlete": {"id": "123", "displayName": "All Star Hitter"}}
                    ]
                }
            ]
        }

    def mlb_espn_prop_bets(self, event_id: str, provider_id: str) -> dict:
        self.calls.append(("props", event_id, provider_id))
        return {"items": [_market_side(-120), _market_side(100)]}


def test_all_star_aliases_select_the_draftkings_event_and_parse_markets():
    scoreboard = _all_star_scoreboard()

    assert _espn_event_market(scoreboard, ALL_STAR_GAME) == (
        "all-star-event",
        "100",
        "DraftKings via ESPN",
    )

    client = AllStarMarketClient()
    market_index = _game_market_index(client, scoreboard, ALL_STAR_GAME)

    assert client.calls == [
        ("summary", "all-star-event"),
        ("props", "all-star-event", "100"),
    ]
    assert market_index[normalize_name("All Star Hitter")] == [
        {
            "stat_key": "hits",
            "stat_label": "Hits",
            "market_athlete_id": "123",
            "line": 0.5,
            "over_odds": -120,
            "under_odds": 100,
            "market_role": "batter",
            "grade_supported": True,
            "market_format": "total",
            "market_type": "Total Hits",
            "market_source": "DraftKings via ESPN",
            "market_updated_at": "2026-07-14T12:00:00Z",
        }
    ]


def test_all_star_aliases_do_not_make_ordinary_team_matching_fuzzy():
    scoreboard = _all_star_scoreboard()
    scoreboard["events"][0]["competitions"][0]["competitors"] = [
        {"homeAway": "away", "team": {"displayName": "Boston Sox"}},
        {"homeAway": "home", "team": {"displayName": "Chicago Cubs"}},
    ]

    assert (
        _espn_event_market(
            scoreboard,
            {"away_team": "Boston Red Sox", "home_team": "Chicago Cubs"},
        )
        is None
    )


@pytest.mark.parametrize(
    ("type_name", "stat_key", "stat_label", "market_role"),
    [
        ("Hits Milestones", "hits", "Hits", "batter"),
        ("Strikeouts Thrown Milestones", "strikeouts", "Strikeouts", "pitcher"),
        ("Total Bases Milestones", "total_bases", "Total Bases", "batter"),
        ("RBIs Milestones", "rbis", "RBIs", "batter"),
        ("Home Runs Milestones", "home_runs", "Home Runs", "batter"),
        ("Singles Milestones", "singles", "Singles", "batter"),
        ("Doubles Milestones", "doubles", "Doubles", "batter"),
        ("Triples Milestones", "triples", "Triples", "batter"),
        ("Runs Milestones", "runs", "Runs", "batter"),
        ("Stolen Bases Milestones", "stolen_bases", "Stolen Bases", "batter"),
        ("Walks (Batter) Milestones", "batter_walks", "Walks", "batter"),
        (
            "Strikeouts (Batter) Milestones",
            "batter_strikeouts",
            "Batter Strikeouts",
            "batter",
        ),
        (
            "Hits + Runs + RBIs Milestones",
            "hits_runs_rbis",
            "Hits + Runs + RBIs",
            "batter",
        ),
    ],
)
def test_live_all_star_plural_milestone_types_parse_as_posted_markets(
    type_name: str,
    stat_key: str,
    stat_label: str,
    market_role: str,
):
    market_index = _market_index(
        [_milestone_market(type_name)],
        {"123": "All Star Hitter"},
        "DraftKings via ESPN",
    )

    market = market_index[normalize_name("All Star Hitter")][0]
    assert market["stat_key"] == stat_key
    assert market["stat_label"] == stat_label
    assert market["market_role"] == market_role
    assert market["grade_supported"] is True
    assert market["market_format"] == "milestone"
    assert market["market_type"] == type_name
    assert market["market_threshold"] == "1+"
    assert market["line"] == 0.5
    assert market["over_odds"] == -115
    assert "under_odds" not in market


@pytest.mark.parametrize(
    ("type_name", "stat_key", "stat_label", "market_role"),
    [
        ("Total Singles Hit", "singles", "Singles", "batter"),
        ("Total Doubles Hit", "doubles", "Doubles", "batter"),
        (
            "Earned Runs Allowed",
            "pitcher_earned_runs_allowed",
            "Earned Runs Allowed",
            "pitcher",
        ),
    ],
)
def test_live_provider_total_aliases_parse_as_two_sided_markets(
    type_name: str,
    stat_key: str,
    stat_label: str,
    market_role: str,
):
    market_index = _market_index(
        [_market_side(-120, type_name), _market_side(100, type_name)],
        {"123": "All Star Hitter"},
        "DraftKings via ESPN",
    )

    market = market_index[normalize_name("All Star Hitter")][0]
    assert market["stat_key"] == stat_key
    assert market["stat_label"] == stat_label
    assert market["market_role"] == market_role
    assert market["grade_supported"] is True
    assert market["market_format"] == "total"
    assert market["market_type"] == type_name
    assert market["line"] == 0.5
    assert market["over_odds"] == -120
    assert market["under_odds"] == 100
