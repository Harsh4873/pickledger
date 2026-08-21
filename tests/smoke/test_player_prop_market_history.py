from __future__ import annotations

import json

from scripts import build_player_prop_market_history as market_history


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
