"""NBA DraftKings prop price archive: pregame-only capture and graded shards."""

from __future__ import annotations

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import archive_nba_prop_prices as archive


ROOT = Path(__file__).resolve().parents[2]
TIP = "2026-10-21T23:30Z"


def _item(athlete: str, name: str, line: float, odds: str, updated: str = "2026-10-21T18:00Z", display: str = "") -> dict:
    return {
        "athlete": {"$ref": f"http://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/seasons/2027/athletes/{athlete}?lang=en"},
        "type": {"id": "1", "name": name},
        "odds": {"american": {"value": odds, "open": "+100"}, "total": {"value": str(line)}},
        "lastUpdated": updated,
        "current": {"target": {"value": line, "displayValue": display or str(line)}},
        "open": {"target": {"value": line + 1, "displayValue": str(line + 1)}},
    }


def _items() -> list[dict]:
    return [
        _item("1", "Total Points", 24.5, "-115"),
        _item("1", "Total Points", 24.5, "-105"),
        _item("1", "Total Points and Rebounds", 31.5, "+100"),
        _item("1", "Total Points and Rebounds", 31.5, "-120"),
        _item("2", "Total 3-Point Field Goals", 2.5, "+120"),
        _item("2", "Total 3-Point Field Goals", 2.5, "-150"),
        # In-game quote (updated after tip) must never enter the archive.
        _item("3", "Total Assists", 6.5, "-110", updated="2026-10-22T00:10Z"),
        _item("3", "Total Assists", 6.5, "-110", updated="2026-10-22T00:10Z"),
        # One-sided milestone ladders are skipped.
        _item("1", "Points Milestones", 25.0, "-110", display="25+"),
    ]


def _event(state: str = "pre", completed: bool = False, season_type: int = 2) -> dict:
    return {
        "id": "401900001",
        "date": TIP,
        "season": {"year": 2027, "type": season_type},
        "competitions": [{
            "date": TIP,
            "status": {"type": {"state": state, "completed": completed}},
            "competitors": [
                {"homeAway": "home", "team": {"id": "10"}},
                {"homeAway": "away", "team": {"id": "20"}},
            ],
        }],
    }


class FakeClient:
    def __init__(self, event: dict, items: list[dict] | None):
        self.event, self.items = event, items

    def basketball_scoreboard(self, league, date_iso):
        assert league == "nba"
        return {"events": [self.event]}

    def basketball_espn_prop_bets(self, league, event_id, provider_id="100"):
        if self.items is None:
            raise RuntimeError("Direct API request failed: 404 Client Error")
        return {"items": self.items}

    def _get(self, url, params=None):
        return _summary()


def _summary() -> dict:
    keys = ["minutes", "points", "rebounds", "assists",
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "steals", "blocks"]
    return {"boxscore": {"players": [{"statistics": [{"keys": keys, "athletes": [
        {"athlete": {"id": "1"}, "stats": ["34", "27", "6", "5", "3-8", "1", "0"]},
        {"athlete": {"id": "2"}, "stats": ["30", "12", "3", "2", "2-6", "0", "1"]},
        {"athlete": {"id": "3"}, "stats": ["28", "8", "2", "9", "0-1", "2", "0"]},
        {"athlete": {"id": "4"}, "didNotPlay": True, "stats": []},
    ]}]}]}}


def test_parse_quotes_pairs_totals_keeps_milestones_one_sided_and_drops_in_game():
    quotes = archive.parse_quotes(_items(), start_time=TIP)
    milestones = [q for q in quotes if q["market_format"] == "milestone"]
    assert [(q["athlete_id"], q["stat_key"], q["line"], q["under_odds"]) for q in milestones] == [
        ("1", "points", 24.5, None)
    ]
    by_key = {(q["athlete_id"], q["stat_key"]): q for q in quotes if q["market_format"] == "total"}
    assert set(by_key) == {("1", "points"), ("1", "points_rebounds"), ("2", "three_pointers_made")}
    points = by_key[("1", "points")]
    assert (points["over_odds"], points["under_odds"], points["line"]) == (-115, -105, 24.5)
    assert points["pregame"] is True and points["market_format"] == "total"
    assert 0.5 < points["over_implied"] < 0.54 and points["open_line"] == 25.5


def test_capture_writes_deterministic_pregame_shard_and_keeps_first_seen(tmp_path):
    client = FakeClient(_event(), _items())
    early = datetime(2026, 10, 21, 11, 30, tzinfo=timezone.utc)
    result = archive.capture_date(client, "2026-10-21", now=early, price_dir=tmp_path)
    assert result["captured_events"] == 1 and result["quotes"] == 4 and result["changed"]
    path = tmp_path / "2026-10-21.jsonl.gz"
    first_bytes = path.read_bytes()
    rows = archive.read_shard(path)
    assert {row["season_type"] for row in rows} == {"regular"}
    assert all(row["pregame"] and row["home_team_id"] == "10" for row in rows)
    # Re-writing identical content is a no-op (stable gzip header, sorted rows).
    assert archive.write_shard(path, rows) is False and path.read_bytes() == first_bytes
    later = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)
    archive.capture_date(client, "2026-10-21", now=later, price_dir=tmp_path)
    merged = archive.read_shard(path)
    assert len(merged) == 4
    assert all(row["first_retrieved_at"] == "2026-10-21T11:30:00Z" for row in merged)
    assert all(row["retrieved_at"] == "2026-10-21T18:00:00Z" for row in merged)
    # After tip nothing new is captured.
    after = datetime(2026, 10, 22, 0, 0, tzinfo=timezone.utc)
    assert archive.capture_date(client, "2026-10-21", now=after, price_dir=tmp_path)["quotes"] == 0


def test_grade_uses_archived_quotes_then_falls_back_to_capture(tmp_path, monkeypatch):
    prices, history = tmp_path / "prices", tmp_path / "history"
    archive.capture_date(FakeClient(_event(), _items()), "2026-10-21",
                         now=datetime(2026, 10, 21, 12, tzinfo=timezone.utc), price_dir=prices)
    done = _event(state="post", completed=True)
    monkeypatch.setattr(archive, "DirectApiClient", lambda **_: FakeClient(done, _items()))
    result = archive.grade_date("2026-10-21", client=FakeClient(done, _items()),
                                price_dir=prices, history_dir=history)
    rows = archive.read_shard(history / "2026-10-21.jsonl.gz")
    assert result["graded_events"] == 1 and len(rows) == 4
    milestone = [r for r in rows if r["market_format"] == "milestone"]
    assert [(r["line"], r["actual"], r["over_outcome"]) for r in milestone] == [(24.5, 27.0, 1)]
    outcomes = {(r["athlete_id"], r["stat_key"]): (r["actual"], r["over_outcome"]) for r in rows
                if r["market_format"] == "total"}
    assert outcomes == {("1", "points"): (27.0, 1), ("1", "points_rebounds"): (33.0, 1),
                        ("2", "three_pointers_made"): (2.0, 0)}
    assert {r["provenance"] for r in rows} == {"espn_archived_pregame"}
    assert {r["season"] for r in rows} == {2027}
    # Already graded events are skipped on the next run.
    assert archive.grade_date("2026-10-21", client=FakeClient(done, _items()),
                              price_dir=prices, history_dir=history)["rows"] == 0

    # ESPN archive missing (404): grade from our own pregame capture instead.
    monkeypatch.setattr(archive, "DirectApiClient", lambda **_: FakeClient(done, None))
    archive.grade_date("2026-10-21", client=FakeClient(done, None), price_dir=prices,
                       history_dir=tmp_path / "fallback")
    fallback = archive.read_shard(tmp_path / "fallback" / "2026-10-21.jsonl.gz")
    assert len(fallback) == 4 and {r["provenance"] for r in fallback} == {"pickledger_pregame_capture"}


def test_preseason_rows_are_stamped_for_trainer_exclusion(tmp_path, monkeypatch):
    done = _event(state="post", completed=True, season_type=1)
    monkeypatch.setattr(archive, "DirectApiClient", lambda **_: FakeClient(done, _items()))
    archive.grade_date("2026-10-08", client=FakeClient(done, _items()), price_dir=tmp_path, history_dir=tmp_path)
    rows = archive.read_shard(tmp_path / "2026-10-08.jsonl.gz")
    assert rows and {r["season_type"] for r in rows} == {"preseason"}


def test_nba_season_labels_follow_espn_end_year():
    assert archive.nba_season_for_date("2026-10-21") == 2027
    assert archive.nba_season_for_date("2026-04-10") == 2026


def test_committed_nba_shards_stay_small_and_pregame():
    for base in (archive.PRICE_DIR, archive.HISTORY_DIR):
        for path in sorted(base.glob("*.jsonl.gz")):
            assert path.stat().st_size < 5_000_000, path
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    assert row["sport"] == "NBA"
                    assert row["market_updated_at"] < row["start_time"] or row.get("pregame") is True


def test_daily_props_refresh_runs_and_commits_the_nba_archive():
    workflow = (ROOT / ".github/workflows/player-props-refresh.yml").read_text()
    assert "python scripts/archive_nba_prop_prices.py" in workflow
    assert "continue-on-error: true" in workflow.split("Archive NBA prop prices", 1)[1].split("run:", 1)[0]
    commit_step = workflow.split("Commit player-props cache if changed", 1)[1]
    assert "data/nba_prop_prices" in commit_step
    assert 'cp -R "$GENERATED_NBA_PRICES"/. data/nba_prop_prices/' in commit_step
    assert 'cp -R "$GENERATED_NBA_HISTORY"/. data/player_props_training/nba_market_history/' in commit_step
