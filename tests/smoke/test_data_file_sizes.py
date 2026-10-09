import json
from pathlib import Path
import subprocess

import pytest

from scripts import check_data_file_sizes as guard
from scripts.check_data_file_sizes import MAX_JSON_BYTES, MAX_LEDGER_SHARD_BYTES, oversized_data_files


def test_repository_data_json_stays_below_hosting_limits():
    assert oversized_data_files() == []


def test_size_guard_includes_tracked_and_new_files_but_ignores_deleted_and_ignored(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("data/ignored.json\n")
    for name, size in (
        ("tracked.json", MAX_JSON_BYTES + 1),
        ("new.json", MAX_JSON_BYTES + 1),
        ("allowed.json", MAX_JSON_BYTES),
        ("deleted.json", 1),
        ("ignored.json", MAX_JSON_BYTES + 1),
        ("calibration/team_prop_pregame_ledger/2026-10-08.json", MAX_LEDGER_SHARD_BYTES + 1),
        ("calibration/outcome_ledger/2026-10-08.json", MAX_LEDGER_SHARD_BYTES + 1),
    ):
        path = tmp_path / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(size)
    subprocess.run(["git", "add", "data/tracked.json", "data/deleted.json"], cwd=tmp_path, check=True)
    (tmp_path / "data/deleted.json").unlink()
    assert {name for name, _, _ in oversized_data_files(tmp_path)} == {
        "data/tracked.json", "data/new.json", "data/calibration/team_prop_pregame_ledger/2026-10-08.json",
        "data/calibration/outcome_ledger/2026-10-08.json",
    }


@pytest.fixture
def sparse_data(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    relative = "data/calibration/outcome_ledger/2026-10-08.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    # Use a complete committed shard; only its filesystem presence changes.
    path.write_bytes((Path(__file__).resolve().parents[2] / relative).read_bytes())
    size = path.stat().st_size
    subprocess.run(["git", "add", relative], cwd=tmp_path, check=True)
    oid = subprocess.check_output(["git", "rev-parse", f":{relative}"], cwd=tmp_path, text=True).strip()
    subprocess.run(["git", "update-index", "--skip-worktree", relative], cwd=tmp_path, check=True)
    path.unlink()
    metadata = tmp_path / "tree.json"
    metadata.write_text(json.dumps({"truncated": False, "tree": [
        {"path": relative, "sha": oid, "type": "blob", "size": size},
    ]}))
    return relative, size, metadata


@pytest.mark.parametrize("use_metadata", [False, True])
def test_size_guard_checks_omitted_shards(tmp_path, monkeypatch, sparse_data, use_metadata):
    relative, size, metadata = sparse_data
    monkeypatch.setattr(guard, "MAX_LEDGER_SHARD_BYTES", size - 1)
    if use_metadata:
        original = subprocess.check_output
        def without_blob_reads(command, **kwargs):
            assert "cat-file" not in command
            return original(command, **kwargs)
        monkeypatch.setattr(subprocess, "check_output", without_blob_reads)
    assert oversized_data_files(tmp_path, git_tree=metadata if use_metadata else None) == [
        (relative, size, size - 1),
    ]


@pytest.mark.parametrize("fault", ["truncated", "missing", "stale", "invalid-size"])
def test_size_guard_rejects_incomplete_or_mismatched_tree(tmp_path, sparse_data, fault):
    relative, _, metadata = sparse_data
    payload = json.loads(metadata.read_text())
    if fault == "truncated":
        payload["truncated"] = True
    elif fault == "missing":
        payload["tree"] = []
    elif fault == "stale":
        payload["tree"][0]["sha"] = "0" * 40
    else:
        payload["tree"][0]["size"] = None
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        oversized_data_files(tmp_path, git_tree=metadata)
    assert not (tmp_path / relative).exists()


def test_size_guard_uses_worktree_size_for_present_sparse_file(tmp_path, monkeypatch, sparse_data):
    relative, _, metadata = sparse_data
    source = Path(__file__).resolve().parents[2] / "data/calibration/active.json"
    (tmp_path / relative).write_bytes(source.read_bytes())
    size = source.stat().st_size
    monkeypatch.setattr(guard, "MAX_LEDGER_SHARD_BYTES", size - 1)
    assert oversized_data_files(tmp_path, git_tree=metadata) == [(relative, size, size - 1)]
