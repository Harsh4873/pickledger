from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import build_player_prop_market_history as market_history


@pytest.mark.parametrize("sports", ["NFL", "NFL,CFB"])
def test_empty_batches_normalize_once_without_rewriting_the_corpus(monkeypatch, tmp_path, sports):
    # Daily Refresh 37857732736 logged zero market rows for both sports on
    # September 27 and 28. Replay those empty batches without network IO.
    output = tmp_path / "history.jsonl"
    monkeypatch.setattr(market_history, "_parse_args", lambda: SimpleNamespace(
        start="2026-09-27", end="2026-09-28", sports=sports, output=output,
        no_resume=False, max_workers=2, max_output_bytes=90_000_000,
    ))
    monkeypatch.setattr(market_history, "_scoreboard", lambda *_: {"events": []})
    monkeypatch.setattr(market_history, "_cfb_snapshot_rows", lambda *_: [])
    writes = []
    write = market_history._write_rows
    def counted(*args, **kwargs):
        writes.append(args[0])
        return write(*args, **kwargs)
    monkeypatch.setattr(market_history, "_write_rows", counted)
    assert market_history.main() == 1
    assert output.read_bytes() == b""
    assert writes == [output]


def test_market_history_prunes_whole_oldest_dates_below_repository_limit(tmp_path):
    path = tmp_path / "market_history.jsonl"
    rows = [
        {
            "sport": "MLB",
            "date": "2026-08-19",
            "event_id": "game-1",
            "athlete_id": "player-1",
            "stat_key": "hits",
            "line": 1.5,
            "market_format": "total",
            "over_outcome": 1,
        },
        {
            "sport": "WNBA",
            "date": "2026-08-20",
            "event_id": "game-2",
            "athlete_id": "player-2",
            "stat_key": "points",
            "line": 18.5,
            "market_format": "total",
            "over_outcome": 0,
        },
    ]

    newest_row_size = len((json.dumps(rows[1], sort_keys=True) + "\n").encode("utf-8"))
    kept = market_history._write_rows(path, rows, max_bytes=newest_row_size)

    loaded, completed = market_history._load_existing(path)
    assert kept == [rows[1]]
    assert loaded == [rows[1]]
    assert completed == {("WNBA", "2026-08-20")}
    assert path.stat().st_size <= newest_row_size
