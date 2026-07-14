from __future__ import annotations

import datetime as dt


def _event(
    event_id: str,
    date: str,
    away: tuple[str, str, int, str],
    home: tuple[str, str, int, str],
    state: str = "post",
    completed: bool = True,
) -> dict:
    away_name, away_abbr, away_score, away_record = away
    home_name, home_abbr, home_score, home_record = home
    status_name = {
        "pre": "STATUS_SCHEDULED",
        "in": "STATUS_IN_PROGRESS",
        "post": "STATUS_FINAL",
    }[state]
    return {
        "id": event_id,
        "date": date,
        "name": f"{away_name} at {home_name}",
        "competitions": [{
            "id": event_id,
            "date": date,
            "neutralSite": True,
            "venue": {"fullName": "Thomas & Mack Center"},
            "notes": [{"headline": "NBA Summer League - Las Vegas"}],
            "status": {
                "type": {
                    "name": status_name,
                    "state": state,
                    "completed": completed,
                    "description": "Final" if completed else "Scheduled",
                    "shortDetail": "Final" if completed else "7/9 - 9:00 PM EDT",
                },
            },
            "competitors": [
                {
                    "homeAway": "away",
                    "score": away_score,
                    "records": [{"summary": away_record}],
                    "team": {
                        "displayName": away_name,
                        "name": away_name.split()[-1],
                        "abbreviation": away_abbr,
                    },
                },
                {
                    "homeAway": "home",
                    "score": home_score,
                    "records": [{"summary": home_record}],
                    "team": {
                        "displayName": home_name,
                        "name": home_name.split()[-1],
                        "abbreviation": home_abbr,
                    },
                },
            ],
        }],
    }


def _payload(events: list[dict], calendar: list[str] | None = None) -> dict:
    return {
        "leagues": [{
            "name": "NBA Summer League",
            "slug": "nba-summer",
            "calendar": calendar or [],
        }],
        "events": events,
    }


def test_nba_summer_model_skips_started_games_and_emits_pregame_pick(monkeypatch):
    from NBASummerPredictionModel import summer_model

    target = "2026-07-09"
    payloads = {
        "2026-07-05": _payload([
            _event(
                "hist-1",
                "2026-07-05T23:00Z",
                ("Oklahoma City Thunder", "OKC", 70, "0-1"),
                ("Utah Jazz", "UTAH", 88, "1-0"),
            )
        ]),
        "2026-07-06": _payload([
            _event(
                "hist-2",
                "2026-07-06T23:00Z",
                ("Oklahoma City Thunder", "OKC", 77, "0-2"),
                ("Atlanta Hawks", "ATL", 82, "1-1"),
            ),
            _event(
                "hist-3",
                "2026-07-06T23:00Z",
                ("Memphis Grizzlies", "MEM", 100, "1-1"),
                ("Utah Jazz", "UTAH", 109, "2-0"),
            ),
        ]),
        target: _payload(
            [
                _event(
                    "started",
                    "2026-07-09T21:00Z",
                    ("Atlanta Hawks", "ATL", 38, "1-1"),
                    ("Memphis Grizzlies", "MEM", 27, "1-1"),
                    state="in",
                    completed=False,
                ),
                _event(
                    "pregame",
                    "2026-07-10T01:00Z",
                    ("Oklahoma City Thunder", "OKC", 0, "0-2"),
                    ("Utah Jazz", "UTAH", 0, "2-0"),
                    state="pre",
                    completed=False,
                ),
            ],
            calendar=[
                "2026-07-05T07:00Z",
                "2026-07-06T07:00Z",
                "2026-07-09T07:00Z",
            ],
        ),
    }

    monkeypatch.setattr(summer_model, "_request_scoreboard", lambda date: payloads[date])

    result = summer_model.generate_nba_summer_picks(
        target,
        now_utc=dt.datetime(2026, 7, 9, 22, 0, tzinfo=dt.timezone.utc),
    )

    assert result["ok"] is True
    assert result["slate_games"] == 2
    assert result["eligible_games"] == 1
    assert result["skipped_started_games"] == 1
    assert len(result["picks"]) == 1
    pick = result["picks"][0]
    assert pick["source"] == "NBA Summer League"
    assert pick["sport"] == "NBA SUMMER"
    assert pick["team"] == "Utah Jazz"
    assert pick["decision"] in {"BET", "LEAN"}
    assert pick["units"] > 0
    assert pick["probability"] > 0.6
    assert "game already started" in result["games"][0]["skipped_reason"]


def test_nba_summer_model_returns_empty_ok_for_no_slate(monkeypatch):
    from NBASummerPredictionModel import summer_model

    monkeypatch.setattr(summer_model, "_request_scoreboard", lambda _date: _payload([]))

    result = summer_model.generate_nba_summer_picks("2026-07-08")

    assert result["ok"] is True
    assert result["picks"] == []
    assert result["slate_games"] == 0
