from __future__ import annotations


def test_mlb_travel_context_flags_long_eastward_no_rest(monkeypatch):
    from models.mlb_inning import mlb_inning_fetcher as fetcher

    monkeypatch.setattr(
        fetcher,
        "_recent_team_games",
        lambda team_id, game_date, lookback_days: [{
            "officialDate": "2026-06-24",
            "gameDate": "2026-06-24T02:10:00Z",
            "gamePk": 1,
            "status": {"detailedState": "Final"},
            "venue": {"id": 22, "name": "Dodger Stadium"},
        }],
    )

    context = fetcher._team_travel_context(147, "2026-06-25", 3, "Fenway Park")

    assert context["available"] is True
    assert context["travel_direction"] == "east"
    assert context["timezone_shift_hours"] == 3
    assert context["distance_miles"] > 2500
    assert context["travel_fatigue_index"] >= 0.6
    assert context["travel_run_delta"] < 0


def test_mlb_travel_context_same_venue_has_no_fatigue(monkeypatch):
    from models.mlb_inning import mlb_inning_fetcher as fetcher

    monkeypatch.setattr(
        fetcher,
        "_recent_team_games",
        lambda team_id, game_date, lookback_days: [{
            "officialDate": "2026-06-24",
            "gamePk": 2,
            "status": {"detailedState": "Final"},
            "venue": {"id": 2395, "name": "Oracle Park"},
        }],
    )

    context = fetcher._team_travel_context(137, "2026-06-25", 2395, "Oracle Park")

    assert context["available"] is True
    assert context["same_venue"] is True
    assert context["distance_miles"] == 0.0
    assert context["travel_fatigue_index"] == 0.0
    assert context["travel_run_delta"] == 0.0
