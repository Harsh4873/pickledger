import subprocess

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
    ):
        path = tmp_path / "data" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            stream.truncate(size)
    subprocess.run(["git", "add", "data/tracked.json", "data/deleted.json"], cwd=tmp_path, check=True)
    (tmp_path / "data/deleted.json").unlink()
    assert {name for name, _, _ in oversized_data_files(tmp_path)} == {
        "data/tracked.json", "data/new.json", "data/calibration/team_prop_pregame_ledger/2026-10-08.json",
    }
