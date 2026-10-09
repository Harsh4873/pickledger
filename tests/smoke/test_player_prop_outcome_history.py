from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest


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


@pytest.fixture
def recorded_outcomes():
    return json.loads((Path(__file__).resolve().parents[1] / "fixtures/player_prop_outcomes.json").read_text())


def test_refresh_of_identical_recorded_profiles_keeps_gzip_bytes_and_mtime(tmp_path, monkeypatch, recorded_outcomes):
    from scripts import build_player_prop_outcome_history as history

    output = tmp_path / "outcomes.jsonl.gz"
    history._write(output, recorded_outcomes)
    before = output.read_bytes(), output.stat().st_mtime_ns
    markets = tmp_path / "profiles.jsonl"
    markets.write_text("".join(json.dumps(row) + "\n" for row in recorded_outcomes))
    calls = []

    def replay(sport, athlete_id, season):
        calls.append((sport, athlete_id, season))
        return sport, season, athlete_id, [
            row.copy() for row in recorded_outcomes
            if (row["sport"], row["athlete_id"], row["season"]) == (sport, athlete_id, season)
        ], None

    monkeypatch.setattr(history, "_fetch", replay)
    monkeypatch.setattr(sys, "argv", [
        "history", "--refresh", "--markets", str(markets), "--output", str(output),
        "--sports", ",".join(sorted({r["sport"] for r in recorded_outcomes})),
        "--seasons", ",".join(str(s) for s in sorted({r["season"] for r in recorded_outcomes})),
    ])
    assert history.main() == 0
    assert calls
    assert (output.read_bytes(), output.stat().st_mtime_ns) == before


def test_changed_history_matches_previous_gzip_writer_at_fixed_clock(tmp_path, monkeypatch, recorded_outcomes):
    from scripts import build_player_prop_outcome_history as history

    monkeypatch.setattr(gzip.time, "time", lambda: 1791504000)
    output = tmp_path / "new" / "history.jsonl.gz"
    previous = tmp_path / "old" / "history.jsonl.gz"
    previous.parent.mkdir()
    # Withhold a real row, then replay the complete profile including duplicates.
    history._write(output, recorded_outcomes[:-1])
    rows = recorded_outcomes + recorded_outcomes[:2]
    expected = sorted({
        (r["sport"], r["season"], r["event_id"], r["athlete_id"], r["stat_key"]): r for r in rows
    }.values(), key=lambda r: (r["date"], r["sport"], r["event_id"], r["athlete_id"], r["stat_key"]))
    with gzip.open(previous, "wt", encoding="utf-8", compresslevel=9) as stream:
        for row in expected:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    assert history._write(output, rows, existing=recorded_outcomes[:-1])
    assert output.read_bytes() == previous.read_bytes()


def test_interrupted_history_write_preserves_committed_gzip(tmp_path, monkeypatch, recorded_outcomes):
    from scripts import build_player_prop_outcome_history as history

    output = tmp_path / "history.jsonl.gz"
    history._write(output, recorded_outcomes[:1])
    before = output.read_bytes()
    original = json.dumps
    count = 0

    def interrupted(row, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("write interrupted")
        return original(row, **kwargs)

    monkeypatch.setattr(history.json, "dumps", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        history._write(output, recorded_outcomes)
    assert output.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [output.name]


@pytest.mark.parametrize("extra", ["{\n", "[]\n"])
def test_invalid_existing_lines_are_repaired_even_without_new_profiles(tmp_path, monkeypatch, recorded_outcomes, extra):
    from scripts import build_player_prop_outcome_history as history

    output = tmp_path / "history.jsonl.gz"
    history._write(output, recorded_outcomes)
    with gzip.open(output, "at", encoding="utf-8") as stream:
        stream.write(extra)
    markets = tmp_path / "profiles.jsonl"
    markets.write_text("")
    monkeypatch.setattr(sys, "argv", ["history", "--markets", str(markets), "--output", str(output)])
    assert history.main() == 0
    with gzip.open(output, "rt", encoding="utf-8") as stream:
        assert len(stream.readlines()) == len(recorded_outcomes)
