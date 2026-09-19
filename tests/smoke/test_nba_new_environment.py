"""Tests for NBA New venue environment + parser fatigue/ML fallbacks."""
from __future__ import annotations


def test_lookup_venue_env_known_teams():
    from NBAPredictionModel.venue_environment import lookup_venue_env

    assert lookup_venue_env("Nuggets").altitude_feet >= 5000
    assert lookup_venue_env("Jazz").altitude_feet >= 4000
    assert lookup_venue_env("Heat").altitude_feet < 100
    assert lookup_venue_env("LA Lakers").timezone_offset_hours == -8
    assert lookup_venue_env("Trail Blazers").timezone_offset_hours == -8


def test_timezone_delta_signs_correctly():
    from NBAPredictionModel.venue_environment import timezone_delta_hours

    # Lakers (Pacific -8) at Heat (Eastern -5) → away is +3h east shift
    assert timezone_delta_hours(home_team="Heat", away_team="Lakers") == 3
    # Heat at Lakers → away is -3h west (easier on body clock)
    assert timezone_delta_hours(home_team="Lakers", away_team="Heat") == -3
    # Knicks at Celtics → 0
    assert timezone_delta_hours(home_team="Celtics", away_team="Knicks") == 0


def test_travel_fatigue_only_penalizes_eastward_trips():
    from NBAPredictionModel.venue_environment import travel_fatigue_adjustment

    east_3h, _ = travel_fatigue_adjustment(home_team="Heat", away_team="Lakers")
    east_2h, _ = travel_fatigue_adjustment(home_team="Heat", away_team="Nuggets")
    west_3h, _ = travel_fatigue_adjustment(home_team="Lakers", away_team="Heat")
    same_zone, _ = travel_fatigue_adjustment(home_team="Celtics", away_team="Knicks")

    # Eastward trips boost the home team's win prob; westward and same
    # zone produce zero adjustment.
    assert east_3h > 0
    assert east_2h > 0
    assert east_3h > east_2h
    assert west_3h == 0.0
    assert same_zone == 0.0
    # Cap at +2.0%
    assert east_3h <= 0.02


def test_altitude_bonus_only_for_denver_utah():
    from NBAPredictionModel.venue_environment import altitude_home_bonus

    nuggets_adj, _ = altitude_home_bonus("Nuggets")
    jazz_adj, _ = altitude_home_bonus("Jazz")
    heat_adj, _ = altitude_home_bonus("Heat")
    knicks_adj, _ = altitude_home_bonus("Knicks")
    suns_adj, _ = altitude_home_bonus("Suns")  # 1086 ft — below threshold

    assert nuggets_adj > 0
    assert jazz_adj > 0
    assert heat_adj == 0.0
    assert knicks_adj == 0.0
    assert suns_adj == 0.0


def test_nba_fatigue_multiplier_reads_rest_line_buffer():
    """The parser buffers the **Rest:** line per game; the helper should
    return 0.55 when the picked team is on the second leg of a B2B."""
    import pickgrader_server as ps

    ps._NBA_REST_LINE_BUFFER.clear()
    ps._NBA_REST_LINE_BUFFER[("Pistons", "Grizzlies")] = (
        "- **Rest:** [Grizzlies B2B] vs [Pistons Rested]"
    )

    # Picking the away team (Grizzlies, on B2B) should reduce stake.
    away_pick_mult = ps._nba_fatigue_multiplier("Pistons", "Grizzlies", "Grizzlies")
    assert away_pick_mult == 0.55

    # Picking the home team (Pistons, rested) should not reduce stake.
    home_pick_mult = ps._nba_fatigue_multiplier("Pistons", "Grizzlies", "Pistons")
    assert home_pick_mult == 1.0


def test_nba_fatigue_multiplier_3_in_4_is_lighter_penalty():
    import pickgrader_server as ps

    ps._NBA_REST_LINE_BUFFER.clear()
    ps._NBA_REST_LINE_BUFFER[("Heat", "Lakers")] = (
        "- **Rest:** [Lakers 3-in-4-nights] vs [Heat Rested]"
    )
    mult = ps._nba_fatigue_multiplier("Heat", "Lakers", "Lakers")
    assert mult == 0.75


def test_nba_fatigue_multiplier_no_rest_line_returns_none():
    import pickgrader_server as ps

    ps._NBA_REST_LINE_BUFFER.clear()
    assert ps._nba_fatigue_multiplier("Heat", "Lakers", "Lakers") is None


def test_nba_schedule_falls_back_to_espn_scoreboard(monkeypatch):
    import importlib
    import sys
    import types

    static = types.ModuleType("nba_api.stats.static")
    static.teams = types.SimpleNamespace(get_teams=lambda: [])
    endpoints = types.ModuleType("nba_api.stats.endpoints")
    endpoints.commonteamroster = types.SimpleNamespace()
    endpoints.leaguegamefinder = types.SimpleNamespace()
    endpoints.leaguedashteamstats = types.SimpleNamespace()
    endpoints.scoreboardv2 = types.SimpleNamespace(ScoreboardV2=object)

    monkeypatch.setitem(sys.modules, "nba_api", types.ModuleType("nba_api"))
    monkeypatch.setitem(sys.modules, "nba_api.stats", types.ModuleType("nba_api.stats"))
    monkeypatch.setitem(sys.modules, "nba_api.stats.static", static)
    monkeypatch.setitem(sys.modules, "nba_api.stats.endpoints", endpoints)
    monkeypatch.delitem(sys.modules, "NBAPredictionModel.live_data", raising=False)

    live_data = importlib.import_module("NBAPredictionModel.live_data")

    class BrokenScoreboard:
        def __init__(self, **_kwargs):
            raise TimeoutError("stats.nba.com timed out")

    payload = {
        "events": [
            {
                "id": "401999001",
                "competitions": [
                    {
                        "venue": {"fullName": "Frost Bank Center"},
                        "status": {"type": {"shortDetail": "8:30 PM"}},
                        "competitors": [
                            {"homeAway": "away", "team": {"id": "18", "name": "Knicks"}},
                            {"homeAway": "home", "team": {"id": "24", "name": "Spurs"}},
                        ],
                    }
                ],
            }
        ]
    }

    monkeypatch.setattr(live_data.scoreboardv2, "ScoreboardV2", BrokenScoreboard)
    monkeypatch.setattr(live_data, "_fetch_espn_json", lambda _url: payload)

    games = live_data.fetch_todays_games("2026-06-13")

    assert games == [
        {
            "game_id": "401999001",
            "home_team_id": "24",
            "away_team_id": "18",
            "home_team": "Spurs",
            "away_team": "Knicks",
            "game_status": "8:30 PM",
            "arena": "Frost Bank Center",
            "schedule_source": "ESPN scoreboard fallback",
        }
    ]


def test_nba_schedule_falls_back_when_scoreboard_row_is_incomplete(monkeypatch):
    import importlib
    import sys
    import types

    import pandas as pd

    static = types.ModuleType("nba_api.stats.static")
    static.teams = types.SimpleNamespace(get_teams=lambda: [])
    endpoints = types.ModuleType("nba_api.stats.endpoints")
    endpoints.commonteamroster = types.SimpleNamespace()
    endpoints.leaguegamefinder = types.SimpleNamespace()
    endpoints.leaguedashteamstats = types.SimpleNamespace()
    endpoints.scoreboardv2 = types.SimpleNamespace(ScoreboardV2=object)

    monkeypatch.setitem(sys.modules, "nba_api", types.ModuleType("nba_api"))
    monkeypatch.setitem(sys.modules, "nba_api.stats", types.ModuleType("nba_api.stats"))
    monkeypatch.setitem(sys.modules, "nba_api.stats.static", static)
    monkeypatch.setitem(sys.modules, "nba_api.stats.endpoints", endpoints)
    monkeypatch.delitem(sys.modules, "NBAPredictionModel.live_data", raising=False)

    live_data = importlib.import_module("NBAPredictionModel.live_data")

    class IncompleteScoreboard:
        def __init__(self, **_kwargs):
            pass

        def get_data_frames(self):
            return [
                pd.DataFrame(
                    [
                        {
                            "GAME_ID": "bad-row",
                            "HOME_TEAM_ID": None,
                            "VISITOR_TEAM_ID": 1610612752,
                            "GAME_STATUS_TEXT": "8:30 pm ET",
                            "ARENA_NAME": "",
                        }
                    ]
                )
            ]

    payload = {
        "events": [
            {
                "id": "401999002",
                "competitions": [
                    {
                        "venue": {"fullName": "Frost Bank Center"},
                        "status": {"type": {"shortDetail": "8:30 PM"}},
                        "competitors": [
                            {"homeAway": "away", "team": {"id": "18", "name": "Knicks"}},
                            {"homeAway": "home", "team": {"id": "24", "name": "Spurs"}},
                        ],
                    }
                ],
            }
        ]
    }

    monkeypatch.setattr(live_data.scoreboardv2, "ScoreboardV2", IncompleteScoreboard)
    monkeypatch.setattr(live_data, "_fetch_espn_json", lambda _url: payload)

    games = live_data.fetch_todays_games("2026-06-13")

    assert games[0]["away_team"] == "Knicks"
    assert games[0]["home_team"] == "Spurs"
    assert games[0]["schedule_source"] == "ESPN scoreboard fallback"


def _nba_new_output(model_total: float = 214.0) -> str:
    return "\n".join([
        "GAME: Grizzlies @ Pistons (7:30 pm ET)",
        "**Winner:** Pistons (Model Prob: 66.0%)",
        "**Spread:** Pistons by 5.0 points",
        "**Model Confidence:** 66.0%",
        "**Decision: BET**",
        f"- **Total:** {model_total:.1f} O/U",
        "**O/U Decision: BET UNDER**",
    ])


def test_nba_new_no_market_row_is_provisional_and_labelled_unpriced(monkeypatch):
    """Confidence alone must never mint a placeable stake: the row is labelled
    unpriced so the refresh pipeline demotes it unless a posted price attaches
    (the lone live 2026 row published BET, odds null, 1.13u, graded a loss)."""
    import pickgrader_server as ps
    from scripts.merge_model_cache_payload import demote_unpriced_team_model_picks

    monkeypatch.setattr(ps, "_sl_get_spread", lambda h, a, league: (None, None, None))
    monkeypatch.setattr(ps, "_sl_get_ml", lambda h, a, league: (None, None))
    monkeypatch.setattr(ps, "_sl_get_total", lambda h, a, league: (None, None))
    monkeypatch.setattr(ps, "_nba_fatigue_multiplier", lambda *args: None)

    picks = ps._parse_nba_output(_nba_new_output(), source_label="NBA New")
    side = [p for p in picks if p.get("market_type") != "totals"][0]
    assert side["decision"] in ("LEAN", "BET")
    assert side["odds"] is None
    assert side["pricing_type"] == "unpriced" and side["market_priced"] is False
    assert side["decision_basis"] == "model_conviction_pending_price"
    # No total row is invented when there is no market total.
    assert not [p for p in picks if p.get("market_type") == "totals"]

    payload = {"models": {"nba": {"picks": picks}}}
    assert demote_unpriced_team_model_picks(payload) == 1
    assert payload["models"]["nba"]["picks"][0]["decision"] == "PASS"
    assert payload["models"]["nba"]["picks"][0]["units"] == 0


def test_nba_new_total_and_spread_without_a_price_are_labelled_assumed(monkeypatch):
    import pickgrader_server as ps
    from scripts.market_odds import _looks_assumed

    monkeypatch.setattr(ps, "_sl_get_spread", lambda h, a, league: (-4.5, 4.5, None))
    monkeypatch.setattr(ps, "_sl_get_ml", lambda h, a, league: (None, None))
    monkeypatch.setattr(ps, "_sl_get_total", lambda h, a, league: (220.5, None))
    monkeypatch.setattr(ps, "_nba_fatigue_multiplier", lambda *args: None)

    picks = ps._parse_nba_output(_nba_new_output(model_total=214.0), source_label="NBA New")
    spread = [p for p in picks if p.get("market_line") == -4.5][0]
    assert spread["odds"] == -110 and spread["assumed_odds"] == -110
    assert spread["pricing_type"] == "assumed" and spread["market_priced"] is False
    assert _looks_assumed(spread) is True

    total = [p for p in picks if p.get("market_type") == "totals"][0]
    assert total["odds"] == -110 and total["assumed_odds"] == -110
    assert total["pricing_type"] == "assumed" and total["market_priced"] is False
    assert total["direction"] == "Under"
    # PASS totals are 0u research; staked totals scale with Kelly.
    assert (total["units"] > 0) == (total["decision"] != "PASS")

    priced = ps._parse_nba_output(_nba_new_output(model_total=205.0), source_label="NBA New")
    monkeypatch.setattr(ps, "_sl_get_total", lambda h, a, league: (220.5, -108))
    priced = ps._parse_nba_output(_nba_new_output(model_total=205.0), source_label="NBA New")
    priced_total = [p for p in priced if p.get("market_type") == "totals"][0]
    assert priced_total["odds"] == -108 and "assumed_odds" not in priced_total
    assert priced_total["decision"] == "BET" and priced_total["units"] > 0


def test_nba_buckets_are_certified_and_frozen_like_other_team_models():
    from scripts.pick_calibration import GLOBAL_FALLBACK_EXEMPT_MODEL_KEYS
    from scripts.refresh_model_cache import KICKOFF_FROZEN_MODEL_KEYS
    from scripts.team_prop_pregame_ledger import TEAM_PROP_MODEL_KEYS

    for key in ("nba", "nba_playoffs"):
        assert key in TEAM_PROP_MODEL_KEYS
        assert key in KICKOFF_FROZEN_MODEL_KEYS
        assert key in GLOBAL_FALLBACK_EXEMPT_MODEL_KEYS
