from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


def _write_markets(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"sport": "MLB", "athlete_id": f"athlete-{index}"}
        for index in range(count)
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def _read_output_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row(athlete_id: str) -> dict:
    return {
        "sport": "MLB",
        "season": 2026,
        "date": "2026-06-01",
        "event_id": f"event-{athlete_id}",
        "athlete_id": athlete_id,
        "stat_key": "hits",
        "actual": 1.0,
        "source": "test",
    }


def test_outcome_history_retries_and_accepts_tiny_partial_failure(monkeypatch, tmp_path, capsys):
    from scripts import build_player_prop_outcome_history as history

    markets = tmp_path / "market_history.jsonl"
    output = tmp_path / "outcome_history.jsonl.gz"
    _write_markets(markets, 50)
    calls: dict[str, int] = {}

    def fake_fetch(sport: str, athlete_id: str, season: int):
        calls[athlete_id] = calls.get(athlete_id, 0) + 1
        if athlete_id == "athlete-49":
            return sport, season, athlete_id, [], "espn 500"
        return sport, season, athlete_id, [_row(athlete_id)], None

    monkeypatch.setattr(history, "_fetch", fake_fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_player_prop_outcome_history.py",
            "--markets",
            str(markets),
            "--output",
            str(output),
            "--seasons",
            "2026",
            "--sports",
            "MLB",
            "--max-workers",
            "8",
        ],
    )

    assert history.main() == 0
    assert calls["athlete-49"] == 2
    assert len(_read_output_rows(output)) == 49
    stdout = capsys.readouterr().out
    assert "retrying 1 failed profile" in stdout
    assert "accepted partial history refresh" in stdout
    assert '"ok": true' in stdout


def test_outcome_history_fails_when_retry_failures_exceed_threshold(monkeypatch, tmp_path, capsys):
    from scripts import build_player_prop_outcome_history as history

    markets = tmp_path / "market_history.jsonl"
    output = tmp_path / "outcome_history.jsonl.gz"
    _write_markets(markets, 10)

    def fake_fetch(sport: str, athlete_id: str, season: int):
        if athlete_id in {"athlete-8", "athlete-9"}:
            return sport, season, athlete_id, [], "espn 500"
        return sport, season, athlete_id, [_row(athlete_id)], None

    monkeypatch.setattr(history, "_fetch", fake_fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_player_prop_outcome_history.py",
            "--markets",
            str(markets),
            "--output",
            str(output),
            "--seasons",
            "2026",
            "--sports",
            "MLB",
            "--max-workers",
            "8",
            "--max-failure-rate",
            "0.02",
        ],
    )

    assert history.main() == 1
    assert len(_read_output_rows(output)) == 8
    stdout = capsys.readouterr().out
    assert "too many profile failures after retry" in stdout
    assert '"ok": false' in stdout


def test_wnba_outcome_history_writes_three_pointers_made_and_attempts():
    from scripts import build_player_prop_outcome_history as history

    payload = {
        "names": [
            "minutes",
            "points",
            "totalRebounds",
            "assists",
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
        ],
        "events": {
            "game-1": {
                "gameDate": "2026-06-12T23:30Z",
                "opponent": {"id": "20"},
                "team": {"id": "10"},
                "atVs": "@",
            }
        },
        "seasonTypes": [
            {
                "displayName": "2026 Regular Season",
                "categories": [
                    {
                        "type": "event",
                        "events": [{"eventId": "game-1", "stats": ["31", "17", "4", "6", "3-8"]}],
                    }
                ],
            }
        ],
    }

    rows = history._event_rows("WNBA", "athlete-1", 2026, payload)
    threes = [row for row in rows if row["stat_key"] == "three_pointers_made"]

    assert len(threes) == 1
    assert threes[0]["actual"] == 3.0
    assert threes[0]["three_pointers_attempted"] == 8.0
    assert threes[0]["minutes"] == 31.0
    assert threes[0]["usage"] == 31.0
    assert threes[0]["opponent_id"] == "20"
    assert threes[0]["team_id"] == "10"
    assert threes[0]["home_away"] == "@"
