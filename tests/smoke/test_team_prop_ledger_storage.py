from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import team_prop_pregame_ledger as ledger


def legacy(root, payload):
    path = root / ledger.LEDGER_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def row(record_id, day="2026-10-01", **kwargs):
    return {"id": record_id, "slate_date": day, "pregame_snapshot": {"probability": 0.6123456789012345}, **kwargs}


def shard_bytes(root):
    return {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in (root / ledger.SHARD_RELATIVE_PATH).glob("*.json")}


def test_migration_preserves_interleaved_insertion_order_metadata_and_every_record(tmp_path):
    payload = {
        "schema_version": 1, "kind": "team_prop_pregame_snapshot_ledger", "updated_at": "2026-10-08T12:00:00Z",
        "extra_metadata": {"unicode": "é"},
        "records": [row("z", "2026-10-03"), row("a"), row("y", "2026-10-03", result="win"), row("b", "2026-09-30")],
    }
    path = legacy(tmp_path, payload)
    before = ledger.load_team_prop_pregame_ledger(tmp_path)
    assert ledger.write_team_prop_pregame_ledger(before, tmp_path)
    assert not path.exists()
    assert ledger.load_team_prop_pregame_ledger(tmp_path) == payload == before
    contents = shard_bytes(tmp_path)
    assert not ledger.write_team_prop_pregame_ledger(before, tmp_path)
    assert shard_bytes(tmp_path) == contents
    for name, (data, _) in contents.items():
        if name != "index.json":
            assert len(data.splitlines()) == len(json.loads(data)) + 2


def test_date_fallbacks_and_oversized_days_split_without_reordering(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "SHARD_MAX_BYTES", 400)
    rows = [
        row("slate", "2026-10-01", game_start_time="2026-10-02T01:00:00Z"),
        row("game", "", game_start_time="2026-10-02T01:00:00Z"),
        row("published", "invalid", game_start_time="invalid", published_at="2026-10-03T00:00:00Z"),
        row("undated", "../../escape"),
        *[row(str(i)) for i in range(8)],
    ]
    ledger.write_team_prop_pregame_ledger({"records": rows}, tmp_path)
    assert ledger.load_team_prop_pregame_ledger(tmp_path)["records"] == rows
    files = shard_bytes(tmp_path)
    assert {"2026-10-01.json", "2026-10-01-002.json", "2026-10-02.json", "2026-10-03.json", "undated.json"} <= files.keys()
    assert all(len(data) <= 400 for name, (data, _) in files.items() if name != "index.json")


def test_grading_rewrites_only_the_affected_day(tmp_path):
    ledger.write_team_prop_pregame_ledger({"records": [row("a"), row("b", "2026-10-02")]}, tmp_path)
    before = shard_bytes(tmp_path)
    payload = ledger.load_team_prop_pregame_ledger(tmp_path)
    payload["records"][0]["result"] = "loss"
    assert ledger.write_team_prop_pregame_ledger(payload, tmp_path)
    after = shard_bytes(tmp_path)
    assert {name for name in before if before[name] != after[name]} == {"2026-10-01.json"}
    assert ledger.load_team_prop_pregame_ledger(tmp_path) == payload


def test_late_monolith_unions_history_and_keeps_results_and_retractions(tmp_path):
    primary = [row("a", result="win"), row("b"), row("c", result="pending", settlement_exclusion_reason="unsupported_fractional_settlement")]
    ledger.write_team_prop_pregame_ledger({"records": primary}, tmp_path)
    late = [row("a", result="pending"), row("b", result="loss"), row("c", result="win", settlement_exclusion_reason="unsupported_fractional_settlement"), row("d")]
    path = legacy(tmp_path, {"updated_at": "2026-10-08T16:00:00Z", "records": late})
    expected = [primary[0], late[1], primary[2], late[3]]
    merged = ledger.load_team_prop_pregame_ledger(tmp_path)
    assert merged["records"] == expected
    # Even a stale caller cannot drop rows or reset attached results.
    ledger.write_team_prop_pregame_ledger({"records": [row("a")]}, tmp_path)
    assert ledger.load_team_prop_pregame_ledger(tmp_path)["records"] == expected
    assert not path.exists()


def test_merge_ties_favor_primary_and_explicit_grade_time_breaks_ties():
    a = row("a", result="win", graded_at="2026-10-07T12:00:00Z")
    b = {**a, "result": "loss"}
    merged = ledger.merge_team_prop_pregame_ledgers({"records": [a]}, {"records": [b], "updated_at": "2026-10-08T12:00:00Z"})
    assert merged["records"] == [a]
    b["graded_at"] = "2026-10-07T13:00:00Z"
    assert ledger.merge_team_prop_pregame_ledgers({"records": [a]}, {"records": [b]})["records"] == [b]


@pytest.mark.parametrize("failed_file", ["index.json", "2026-10-03.json"])
def test_interrupted_migration_keeps_legacy_and_is_retryable(tmp_path, monkeypatch, failed_file):
    payload = {"records": [row("a", "2026-10-03"), row("b"), row("c", "2026-10-03")]}
    path = legacy(tmp_path, payload)
    original = path.read_bytes()
    atomic = ledger._atomic_write_changed

    def fail_index(target, contents):
        if target.name == failed_file:
            raise OSError("simulated disk failure")
        return atomic(target, contents)

    monkeypatch.setattr(ledger, "_atomic_write_changed", fail_index)
    with pytest.raises(OSError, match="disk failure"):
        ledger.write_team_prop_pregame_ledger(payload, tmp_path)
    assert path.read_bytes() == original
    assert ledger.load_team_prop_pregame_ledger(tmp_path)["records"] == payload["records"]
    monkeypatch.setattr(ledger, "_atomic_write_changed", atomic)
    ledger.write_team_prop_pregame_ledger(ledger.load_team_prop_pregame_ledger(tmp_path), tmp_path)
    assert not path.exists()
    assert ledger.load_team_prop_pregame_ledger(tmp_path)["records"] == payload["records"]


def test_failed_atomic_replace_does_not_damage_existing_shard(tmp_path, monkeypatch):
    ledger.write_team_prop_pregame_ledger({"records": [row("a")]}, tmp_path)
    before = shard_bytes(tmp_path)
    payload = ledger.load_team_prop_pregame_ledger(tmp_path)
    payload["records"][0]["result"] = "win"

    def fail_replace(*args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failure"):
        ledger.write_team_prop_pregame_ledger(payload, tmp_path)
    assert shard_bytes(tmp_path) == before
    assert not list((tmp_path / ledger.SHARD_RELATIVE_PATH).glob(".*.tmp"))


@pytest.mark.parametrize("damage", ["missing", "malformed", "truncated_records"])
def test_shard_damage_cannot_silently_discard_evidence(tmp_path, damage):
    ledger.write_team_prop_pregame_ledger({"records": [row("a")]}, tmp_path)
    path = tmp_path / ledger.SHARD_RELATIVE_PATH / "2026-10-01.json"
    if damage == "missing":
        path.unlink()
    elif damage == "malformed":
        path.write_text("{")
    else:
        path.write_text("[]")
    with pytest.raises(ValueError):
        ledger.load_team_prop_pregame_ledger(tmp_path)
    with pytest.raises(ValueError):
        ledger.write_team_prop_pregame_ledger({"records": []}, tmp_path)


def test_malformed_legacy_read_contract_but_writer_will_not_delete_it(tmp_path):
    path = legacy(tmp_path, {})
    path.write_text("{")
    assert ledger.load_team_prop_pregame_ledger(tmp_path)["records"] == []
    with pytest.raises(ValueError):
        ledger.write_team_prop_pregame_ledger({"records": []}, tmp_path)
    assert path.read_text() == "{"


def test_cli_restore_merges_new_remote_rows_and_grades_and_is_idempotent(tmp_path):
    saved = tmp_path / "saved"
    remote = tmp_path / "remote"
    ledger.write_team_prop_pregame_ledger({"records": [row("a"), row("generated")]}, saved)
    ledger.write_team_prop_pregame_ledger({"records": [row("a", result="win"), row("remote")]}, remote)
    # Also cover a concurrent old-code publication on the remote checkout.
    legacy(remote, {"records": [row("a"), row("legacy")]})
    command = [sys.executable, "-m", "scripts.team_prop_pregame_ledger", "--migrate", "--repo-root", str(remote), "--merge-from", str(saved)]
    for changed in (True, False):
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        assert json.loads(result.stdout)["changed"] is changed
        assert ledger.load_team_prop_pregame_ledger(remote)["records"] == [row("a", result="win"), row("remote"), row("legacy"), row("generated")]


def test_anonymous_legacy_occurrences_and_empty_ledgers_round_trip(tmp_path):
    records = [{"legacy": "no id"}, {"legacy": "no id"}, None]
    ledger.write_team_prop_pregame_ledger({"records": records}, tmp_path)
    assert ledger.load_team_prop_pregame_ledger(tmp_path)["records"] == records
    assert not ledger.write_team_prop_pregame_ledger({"records": copy.deepcopy(records)}, tmp_path)
    empty = tmp_path / "empty"
    assert ledger.write_team_prop_pregame_ledger({"records": []}, empty)
    assert not ledger.write_team_prop_pregame_ledger({"records": []}, empty)
