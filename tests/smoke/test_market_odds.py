from __future__ import annotations

import pytest

from scripts import market_odds
from scripts.merge_model_cache_payload import _preserve_pick_metadata


DATE = "2026-07-11"


def scoreboard_event(
    *,
    event_id: str = "401",
    state: str = "pre",
    home: str = "Pittsburgh Pirates",
    away: str = "Milwaukee Brewers",
    home_id: str = "23",
    away_id: str = "8",
    home_ml: str = "+109",
    away_ml: str = "-132",
    draw_ml: str | None = None,
    total_line: float = 9.0,
    over: str = "-107",
    under: str = "-112",
    spread: float = 1.5,
    spread_home: str = "-136",
    spread_away: str = "+113",
) -> dict:
    moneyline = {
        "home": {"close": {"odds": home_ml}},
        "away": {"close": {"odds": away_ml}},
    }
    if draw_ml is not None:
        moneyline["draw"] = {"close": {"odds": draw_ml}}
    return {
        "id": event_id,
        "date": f"{DATE}T20:05Z",
        "status": {"type": {"state": state}},
        "competitions": [
            {
                "competitors": [
                    {
                        "homeAway": "home",
                        "team": {"id": home_id, "displayName": home, "abbreviation": home[:3].upper()},
                    },
                    {
                        "homeAway": "away",
                        "team": {"id": away_id, "displayName": away, "abbreviation": away[:3].upper()},
                    },
                ],
                "odds": [
                    {
                        "provider": {"displayName": "DraftKings"},
                        "overUnder": total_line,
                        "spread": spread,
                        "moneyline": moneyline,
                        "total": {
                            "over": {"close": {"odds": over}},
                            "under": {"close": {"odds": under}},
                        },
                        "pointSpread": {
                            "home": {"close": {"odds": spread_home}},
                            "away": {"close": {"odds": spread_away}},
                        },
                    }
                ],
            }
        ],
    }


def make_fetch(events: list[dict], prop_items: list[dict] | None = None):
    def fetch(url: str, params: dict) -> dict:
        if "propBets" in url:
            return {"items": prop_items or []}
        return {"events": events}

    return fetch


def build_book(events: list[dict], prop_items: list[dict] | None = None, sport: str = "MLB"):
    return market_odds.fetch_market_odds_for_date(
        DATE, [sport], make_fetch(events, prop_items)
    )


def payload_with(bucket_key: str, picks: list[dict]) -> dict:
    return {"date": DATE, "models": {bucket_key: {"ok": True, "picks": picks}}}


def test_scoreboard_parse_skips_live_games_and_prefers_close_prices():
    book = build_book([
        scoreboard_event(event_id="live", state="in"),
        scoreboard_event(event_id="pre1"),
    ])
    games = book["MLB"]
    assert [game["eventId"] for game in games] == ["pre1"]
    markets = games[0]["markets"]
    assert markets["moneyline"] == {"home": 109, "away": -132, "draw": None}
    assert markets["total"] == {"line": 9.0, "over": -107, "under": -112}
    assert markets["spread"] == {"homeLine": 1.5, "home": -136, "away": 113}


def test_reversed_matchup_still_prices_the_correct_side():
    book = build_book([scoreboard_event()])
    payload = payload_with(
        "sportsgambler_mlb",
        [
            {
                "date": DATE,
                "sport": "MLB",
                "pick": "Milwaukee Brewers to Win (Pittsburgh Pirates vs Milwaukee Brewers)",
                # The feed lists the matchup reversed relative to reality.
                "away_team": "Pittsburgh Pirates",
                "home_team": "Milwaukee Brewers",
                "odds": 102,
                "decision": "BET",
            }
        ],
    )
    summary = market_odds.apply_market_odds_to_payload(payload, book)
    pick = payload["models"]["sportsgambler_mlb"]["picks"][0]
    assert summary["attached"] == 1
    assert pick["selected_odds"] == -132  # Milwaukee is the real away side
    assert pick["opposite_odds"] == 109
    assert pick["odds"] == 102  # scraped executable price is never replaced
    assert "assumed_odds_replaced" not in pick
    assert pick["market_odds_provider"] == "espn_scoreboard:DraftKings"


def test_totals_require_an_exact_line_match():
    book = build_book([scoreboard_event(total_line=9.0)])
    payload = payload_with(
        "scores24_mlb",
        [
            {
                "date": DATE,
                "sport": "MLB",
                "pick": "Over 8.5 (Milwaukee Brewers @ Pittsburgh Pirates)",
                "away_team": "Milwaukee Brewers",
                "home_team": "Pittsburgh Pirates",
                "line": 8.5,
                "odds": -110,
                "decision": "BET",
            }
        ],
    )
    summary = market_odds.apply_market_odds_to_payload(payload, book)
    assert summary["attached"] == 0
    assert "market_over_odds" not in payload["models"]["scores24_mlb"]["picks"][0]


def test_assumed_model_price_is_replaced_even_when_market_priced_flag_lies():
    book = build_book([scoreboard_event()])
    payload = payload_with(
        "mlb_new",
        [
            {
                "date": DATE,
                "sport": "MLB",
                "pick": "Over 9.0 (Milwaukee Brewers vs Pittsburgh Pirates)",
                "away_team": "Milwaukee Brewers",
                "home_team": "Pittsburgh Pirates",
                "line": 9.0,
                "odds": -110,
                "assumed_odds": -110,
                "market_priced": True,
                "pricing_type": "user_assumed",
                "decision": "BET",
            }
        ],
    )
    summary = market_odds.apply_market_odds_to_payload(payload, book)
    pick = payload["models"]["mlb_new"]["picks"][0]
    assert summary["replacedAssumed"] == 1
    assert pick["odds"] == -107
    assert pick["model_assumed_odds"] == -110
    assert pick["assumed_odds_replaced"] is True
    assert pick["pricing_type"] == "market"
    assert "assumed_odds" not in pick
    assert pick["market_over_odds"] == -107
    assert pick["market_under_odds"] == -112


def test_three_way_market_publishes_exact_no_vig_instead_of_two_way_pair():
    book = build_book(
        [
            scoreboard_event(
                home="Norway",
                away="England",
                home_ml="+280",
                away_ml="-105",
                draw_ml="+265",
            )
        ],
        sport="FIFA WC",
    )
    payload = payload_with(
        "fifa_world_cup",
        [
            {
                "date": DATE,
                "sport": "FIFA WC",
                "pick": "England ML (England @ Norway)",
                "away_team": "England",
                "home_team": "Norway",
                "team": "England",
                "odds": -110,
                "assumed_odds": -110,
                "decision": "BET",
            }
        ],
    )
    market_odds.apply_market_odds_to_payload(payload, book)
    pick = payload["models"]["fifa_world_cup"]["picks"][0]
    assert pick["market_draw_odds"] == 265
    assert "selected_odds" not in pick
    implied = {
        "home": market_odds._implied(280),
        "away": market_odds._implied(-105),
        "draw": market_odds._implied(265),
    }
    expected = implied["away"] / sum(implied.values())
    assert pick["market_no_vig_selected_probability"] == pytest.approx(expected, abs=1e-6)
    assert pick["odds"] == -105  # replaced with the real observed price


def f5_prop_items() -> list[dict]:
    def item(type_name: str, price: str, team_id: str | None = None, line: float | None = None) -> dict:
        odds: dict = {"american": {"value": price}}
        if line is not None:
            odds["total"] = {"value": str(line)}
        row = {"type": {"name": type_name}, "odds": odds}
        if team_id:
            row["team"] = {"$ref": f"http://example/teams/{team_id}?lang=en"}
        return row

    return [
        item("1st 5 Innings Moneyline", "-140", team_id="8"),
        item("1st 5 Innings Moneyline", "+110", team_id="23"),
        item("1st 5 Innings Run Line", "-115", team_id="8", line=-0.5),
        item("1st 5 Innings Run Line", "-105", team_id="23", line=0.5),
        item("1st 5 Innings Total Runs", "-145", line=4.5),
        item("1st 5 Innings Total Runs", "+114", line=4.5),
    ]


def test_first_five_markets_price_moneyline_and_ordered_total_pairs():
    book = build_book([scoreboard_event()], f5_prop_items())
    payload = payload_with(
        "mlb_first_five",
        [
            {
                "date": DATE,
                "sport": "MLB",
                "pick": "Milwaukee Brewers F5 ML",
                "away_team": "Milwaukee Brewers",
                "home_team": "Pittsburgh Pirates",
                "team": "Milwaukee Brewers",
                "odds": -110,
                "assumed_odds": -110,
                "decision": "BET",
            },
            {
                "date": DATE,
                "sport": "MLB",
                "pick": "Over 4.5 F5 (Milwaukee Brewers @ Pittsburgh Pirates)",
                "away_team": "Milwaukee Brewers",
                "home_team": "Pittsburgh Pirates",
                "line": 4.5,
                "direction": "Over",
                "odds": -130,
                "assumed_odds": -130,
                "decision": "BET",
            },
        ],
    )
    summary = market_odds.apply_market_odds_to_payload(payload, book)
    ml_pick, total_pick = payload["models"]["mlb_first_five"]["picks"]
    assert summary["replacedAssumed"] == 2
    assert ml_pick["odds"] == -140  # away team id 8 in the fixture
    assert ml_pick["selected_odds"] == -140
    assert ml_pick["opposite_odds"] == 110
    assert total_pick["odds"] == -145
    assert total_pick["market_over_odds"] == -145
    assert total_pick["market_under_odds"] == 114


def test_replaced_price_survives_a_merge_against_a_regenerated_assumed_pick():
    current_bucket = {
        "picks": [
            {
                "date": DATE,
                "sport": "MLB",
                "pick": "Over 9.0 (Milwaukee Brewers vs Pittsburgh Pirates)",
                "odds": -107,
                "model_assumed_odds": -110,
                "assumed_odds_replaced": True,
                "pricing_type": "market",
                "odds_source": "posted_market",
                "price_source": "espn_scoreboard:DraftKings",
                "market_priced": True,
                "market_over_odds": -107,
                "market_under_odds": -112,
                "market_updated_at": f"{DATE}T15:00:00Z",
                "market_odds_provider": "espn_scoreboard:DraftKings",
                "market_odds_captured_at": f"{DATE}T15:00:00Z",
            }
        ]
    }
    regenerated_bucket = {
        "picks": [
            {
                "date": DATE,
                "sport": "MLB",
                "pick": "Over 9.0 (Milwaukee Brewers vs Pittsburgh Pirates)",
                "odds": -110,
                "assumed_odds": -110,
                "pricing_type": "user_assumed",
            }
        ]
    }
    merged = _preserve_pick_metadata(current_bucket, regenerated_bucket)
    pick = merged["picks"][0]
    assert pick["odds"] == -107
    assert pick["pricing_type"] == "market"
    assert pick["market_over_odds"] == -107
    assert pick["assumed_odds_replaced"] is True
    assert pick["market_odds_captured_at"] == f"{DATE}T15:00:00Z"


def test_postgame_slates_and_foreign_dates_are_never_touched():
    book = build_book([scoreboard_event(state="in")])
    assert book == {}
    payload = payload_with(
        "scores24_mlb",
        [
            {
                "date": "2026-07-10",
                "sport": "MLB",
                "pick": "Pittsburgh Pirates ML",
                "away_team": "Milwaukee Brewers",
                "home_team": "Pittsburgh Pirates",
                "odds": -130,
                "decision": "BET",
            }
        ],
    )
    summary = market_odds.apply_market_odds_to_payload(payload, {"MLB": []})
    assert summary == {"attached": 0, "replacedAssumed": 0, "picksSeen": 0}
