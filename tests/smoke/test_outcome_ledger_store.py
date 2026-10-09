"""Storage and migration contracts using the committed outcome history."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from scripts import outcome_ledger_store as store


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def published():
    return store.load_outcome_ledger(ROOT)


def subset(published, rows):
    return store.merge_outcome_ledgers(
        {"schema_version": published["schema_version"], "records": []},
        {"schema_version": published["schema_version"], "records": rows},
    )


def legacy(root, payload):
    path = root / store.LEDGER_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def files(root):
    return {p.name: (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in (root / store.SHARD_RELATIVE_PATH).glob("*.json")}


def test_real_history_roundtrip_and_idempotence(tmp_path, published):
    path = legacy(tmp_path, published)
    canonical = path.read_bytes()
    assert store.write_outcome_ledger(store.load_outcome_ledger(tmp_path), tmp_path)
    loaded = store.load_outcome_ledger(tmp_path)
    assert loaded == published
    assert (json.dumps(loaded, indent=2, sort_keys=True) + "\n").encode() == canonical
    assert not path.exists()
    before = files(tmp_path)
    assert not store.write_outcome_ledger(loaded, tmp_path)
    assert files(tmp_path) == before


def test_only_changed_day_is_rewritten(tmp_path, published):
    store.write_outcome_ledger(published, tmp_path)
    before = files(tmp_path)
    changed = subset(published, published["records"][:-1])
    assert store.write_outcome_ledger(changed, tmp_path)
    after = files(tmp_path)
    day = published["records"][-1]["date"]
    assert {name for name in before if before[name] != after.get(name)} == {"index.json", f"{day}.json"}
    assert store.load_outcome_ledger(tmp_path) == changed


def test_interrupted_initial_migration_preserves_legacy(tmp_path, published, monkeypatch):
    path = legacy(tmp_path, published)
    write = store._atomic_write_changed
    def fail_index(target, rendered):
        if target.name == "index.json":
            raise OSError("index write interrupted")
        return write(target, rendered)
    monkeypatch.setattr(store, "_atomic_write_changed", fail_index)
    with pytest.raises(OSError, match="interrupted"):
        store.write_outcome_ledger(published, tmp_path)
    assert path.exists()
    assert store.load_outcome_ledger(tmp_path) == published
    monkeypatch.setattr(store, "_atomic_write_changed", write)
    assert store.write_outcome_ledger(published, tmp_path)
    assert store.load_outcome_ledger(tmp_path) == published


@pytest.mark.parametrize("corruption", ["missing", "truncated", "count", "duplicate", "path", "index"])
def test_indexed_history_fails_closed(tmp_path, published, corruption):
    sample = subset(published, published["records"][:3])
    store.write_outcome_ledger(sample, tmp_path)
    directory = tmp_path / store.SHARD_RELATIVE_PATH
    index_path = directory / "index.json"
    index = json.loads(index_path.read_text())
    shard = directory / index["shards"][0]["file"]
    if corruption == "missing":
        shard.unlink()
    elif corruption == "truncated":
        shard.write_text("[")
    elif corruption == "count":
        index["record_count"] += 1
        index_path.write_text(json.dumps(index))
    elif corruption == "duplicate":
        rows = json.loads(shard.read_text())
        rows[1] = rows[0]
        shard.write_text(json.dumps(rows))
    elif corruption == "path":
        index["shards"][0]["file"] = "../outcome_ledger.json"
        index_path.write_text(json.dumps(index))
    else:
        index_path.unlink()
    with pytest.raises(ValueError):
        store.load_outcome_ledger(tmp_path)


def test_oversized_day_splits_and_roundtrips(tmp_path, published, monkeypatch):
    rows = published["records"][:10]
    sample = subset(published, rows)
    largest = max(len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()) for row in rows)
    monkeypatch.setattr(store, "SHARD_MAX_BYTES", largest + 6)
    store.write_outcome_ledger(sample, tmp_path)
    paths = list((tmp_path / store.SHARD_RELATIVE_PATH).glob("*.json"))
    assert any("-002.json" in p.name for p in paths)
    assert all(p.stat().st_size <= store.SHARD_MAX_BYTES for p in paths if p.name != "index.json")
    assert store.load_outcome_ledger(tmp_path) == sample


@pytest.mark.parametrize("incoming_shards", [False, True])
def test_migrate_merge_from_and_late_legacy_are_idempotent(tmp_path, published, monkeypatch, incoming_shards):
    current = subset(published, published["records"][:4])
    incoming = subset(published, published["records"][2:7])
    expected = store.merge_outcome_ledgers(current, incoming)
    source = tmp_path / "source"
    if incoming_shards:
        store.write_outcome_ledger(incoming, source)
    else:
        legacy(source, incoming)
    store.write_outcome_ledger(current, tmp_path)
    monkeypatch.setattr(sys, "argv", ["outcome_ledger_store", "--migrate", "--repo-root", str(tmp_path), "--merge-from", str(source)])
    assert store.main() == 0
    assert store.load_outcome_ledger(tmp_path) == expected
    before = files(tmp_path)
    assert store.main() == 0
    assert files(tmp_path) == before
    legacy(tmp_path, incoming)
    assert store.load_outcome_ledger(tmp_path) == expected
    assert store.main() == 0
    assert not (tmp_path / store.LEDGER_RELATIVE_PATH).exists()
    assert files(tmp_path) == before


def test_invalid_legacy_is_not_removed(tmp_path, published):
    path = legacy(tmp_path, published)
    path.write_text("{")
    with pytest.raises(ValueError):
        store.write_outcome_ledger(published, tmp_path)
    assert path.read_text() == "{"


def test_noncanonical_rows_are_rejected(tmp_path, published):
    payload = copy.deepcopy(published)
    payload["records"].reverse()
    with pytest.raises(ValueError, match="canonical"):
        store.write_outcome_ledger(payload, tmp_path)


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_auto_grade_due_step_reads_both_formats(tmp_path, published, sharded, checkpoint):
    if sharded:
        store.write_outcome_ledger(published, tmp_path)
    else:
        legacy(tmp_path, published)
    state = json.loads((ROOT / "data/calibration/state.json").read_text()) if checkpoint else {}
    if checkpoint:
        (tmp_path / "data/calibration/state.json").write_text(json.dumps(state))
    workflow = yaml.load((ROOT / ".github/workflows/auto-grade.yml").read_text(), Loader=yaml.BaseLoader)
    command = next(step["run"] for step in workflow["jobs"]["auto-grade"]["steps"] if step.get("id") == "calibration-due")
    output = tmp_path / "output"
    env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
           "PYTHONPATH": str(ROOT), "GITHUB_OUTPUT": str(output)}
    subprocess.run(["bash", "-e", "-c", command], cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    due = published["summary"]["decided_picks"] - int(state.get("last_evaluated_decided_count") or 0) >= 100
    assert output.read_text() == f"due={str(due).lower()}\n"


def test_publication_pathspec_stages_shards_and_legacy_deletion(tmp_path, published):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    path = legacy(tmp_path, subset(published, published["records"][:3]))
    subprocess.run(["git", "add", str(path)], cwd=tmp_path, check=True)
    store.write_outcome_ledger(store.load_outcome_ledger(tmp_path), tmp_path)
    for _ in range(2):
        subprocess.run(["git", "add", "-A", "data/calibration/outcome_ledger*"], cwd=tmp_path, check=True)
        paths = subprocess.check_output(["git", "ls-files"], cwd=tmp_path, text=True).splitlines()
        assert all(name.startswith("data/calibration/outcome_ledger/") for name in paths)
        assert len(paths) == 2
