from __future__ import annotations

import json
import copy
import os
from pathlib import Path
import re
import subprocess
import sys
import shutil

import pytest
import yaml

from scripts.team_prop_pregame_ledger import SHARD_RELATIVE_PATH, write_team_prop_pregame_ledger
from scripts.merge_pick_outcome_ledger import merge_outcome_ledgers
from scripts.outcome_ledger_store import load_outcome_ledger, write_outcome_ledger


ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def evaluate_condition(condition, values, cancelled=False):
    expression = condition.removeprefix("${{").removesuffix("}}").strip()
    expression = expression.replace("cancelled()", repr(cancelled))
    for key in sorted(values, key=len, reverse=True):
        expression = expression.replace(key, repr(values[key]))
    expression = expression.replace("&&", " and ").replace("||", " or ")
    expression = re.sub(r"!(?!=)", " not ", expression).strip()
    return eval(expression, {"__builtins__": {}}, {})


def test_model_workflow_preserves_written_results_before_reporting_failure():
    doc = workflow("model-cache-refresh.yml")
    steps = doc["jobs"]["refresh"]["steps"]
    by_name = {step["name"]: step for step in steps}
    refresh = by_name["Refresh model caches"]
    artifact = by_name["Preserve generated model cache"]
    commit = by_name["Commit cache JSON if changed"]
    report = by_name["Report model refresh failure after publication"]
    assert refresh["id"] == "refresh-models"
    assert refresh["continue-on-error"] == "true"
    assert artifact["uses"] == "actions/upload-artifact@v4"
    assert "data/model_cache/latest.json" in artifact["with"]["path"]
    assert "data/calibration/team_prop_pregame_ledger/" in artifact["with"]["path"]
    assert steps.index(refresh) < steps.index(artifact) < steps.index(commit) < steps.index(report)
    assert steps.index(by_name["Deploy updated model cache"]) < steps.index(report)
    assert "continue-on-error" not in commit
    assert "continue-on-error" not in report
    assert "exit 1" in report["run"]
    assert doc["on"]["workflow_call"]["outputs"]["published"]["value"] == "${{ jobs.refresh.outputs.published }}"
    for publishable in ("true", "false", ""):
        assert evaluate_condition(commit["if"], {"steps.refresh-models.outputs.publishable": publishable}) is (publishable == "true")
    assert evaluate_condition(artifact["if"], {"steps.refresh-models.outputs.cache_written": "true"})
    assert not evaluate_condition(artifact["if"], {"steps.refresh-models.outputs.cache_written": "false"})
    assert evaluate_condition(report["if"], {"steps.refresh-models.outcome": "failure"})
    assert not evaluate_condition(report["if"], {"steps.refresh-models.outcome": "success"})


@pytest.mark.parametrize("results,published,expected", [
    (("success", "success", "success"), "true", True),
    (("failure", "success", "success"), "", True),
    (("success", "failure", "failure"), "", True),
    (("failure", "failure", "success"), "", True),
    (("failure", "failure", "failure"), "true", True),
    (("failure", "failure", "failure"), "false", False),
    (("skipped", "skipped", "skipped"), "", False),
])
def test_daily_deploy_keeps_healthy_publications_visible(results, published, expected):
    daily = workflow("daily-refresh.yml")
    condition = daily["jobs"]["deploy"]["if"]
    values = {f"needs.{key}.result": value for key, value in zip(("models", "props", "feeds"), results)}
    values["needs.models.outputs.published"] = published
    assert evaluate_condition(condition, values) is expected
    assert not evaluate_condition(condition, values, cancelled=True)
    assert daily["jobs"]["props"]["needs"] == "models"
    assert daily["jobs"]["feeds"]["needs"] == "props"


def test_optional_consensus_check_cannot_hold_the_cache_writer():
    props = workflow("player-props-refresh.yml")
    steps = props["jobs"]["refresh"]["steps"]
    assert not any("train_player_prop_consensus_ml.py" in step.get("run", "") for step in steps)
    for step in steps:
        if step.get("id") in {"market-history", "outcome-history", "train-props"}:
            assert step["continue-on-error"] == "true"
            assert 1 <= int(step["timeout-minutes"]) <= 5
    check = workflow("player-prop-consensus-check.yml")
    assert check["permissions"] == {"contents": "read"}
    assert check["on"]["workflow_run"] == {"workflows": ["Daily Refresh"], "types": ["completed"], "branches": ["main"]}
    assert check["concurrency"]["group"] != "pick-cache-writer"
    commands = "\n".join(step.get("run", "") for step in check["jobs"]["verify"]["steps"])
    assert "python scripts/train_player_prop_consensus_ml.py --verify-only" in commands
    assert "git push" not in commands and "gh workflow run" not in commands


@pytest.mark.parametrize("failed_pushes,expected_exit,attempts,delays", [
    (0, 0, 1, []),
    (3, 0, 4, [10, 20, 30]),
    (5, 1, 5, [10, 20, 30, 40]),
])
def test_commit_step_recovers_from_remote_errors_and_stays_bounded(tmp_path, failed_pushes, expected_exit, attempts, delays):
    """Execute the actual workflow shell with isolated fake git/python/sleep commands."""
    steps = workflow("model-cache-refresh.yml")["jobs"]["refresh"]["steps"]
    command = next(step["run"] for step in steps if step["name"] == "Commit cache JSON if changed")
    cache = tmp_path / "data/model_cache/latest.json"
    cache.parent.mkdir(parents=True)
    cache.write_text('{"date":"2026-10-07","models":{}}')
    write_team_prop_pregame_ledger({"records": [{"id": "generated", "slate_date": "2026-10-07"}]}, tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = f"#!{sys.executable}\n" + '''import json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
root = pathlib.Path(os.environ["FIXTURE_ROOT"])
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps([name, *args]) + "\\n")
if name == "git":
    if args[0] == "status":
        print(" M data/model_cache/latest.json")
    elif args[0] == "diff":
        sys.exit(1)
    elif args[0] == "push":
        path = root / "push-count"
        count = int(path.read_text()) + 1 if path.exists() else 1
        path.write_text(str(count))
        sys.exit(1 if count <= int(os.environ["FAILED_PUSHES"]) else 0)
elif name == "python" and args[0] == "-c":
    print("2026-10-07")
'''
    for name in ("git", "python", "sleep"):
        path = bin_dir / name
        path.write_text(stub)
        path.chmod(0o755)
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    output = tmp_path / "github-output"
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "FIXTURE_ROOT": str(tmp_path), "FAILED_PUSHES": str(failed_pushes),
           "RUNNER_TEMP": str(runner_temp), "GITHUB_ACTOR": "fixture", "GITHUB_OUTPUT": str(output)}
    result = subprocess.run(["bash", "-e", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == expected_exit, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert sum(call[:2] == ["git", "push"] for call in calls) == attempts
    assert sum(call[:2] == ["python", "scripts/merge_model_cache_payload.py"] for call in calls) == attempts
    assert [call[1] for call in calls if call[0] == "python" and call[1].startswith("scripts/build_")] == ["scripts/build_parlay_cards.py", "scripts/build_profit_desk.py"] * attempts
    assert [int(call[1]) for call in calls if call[0] == "sleep"] == delays
    assert (runner_temp / "model-cache-latest.json").read_bytes() == cache.read_bytes()
    for path in (tmp_path / SHARD_RELATIVE_PATH).glob("*.json"):
        assert (runner_temp / "team-prop-pregame-ledger" / SHARD_RELATIVE_PATH / path.name).read_bytes() == path.read_bytes()
    assert sum(call[:5] == ["python", "-m", "scripts.team_prop_pregame_ledger", "--migrate", "--merge-from"] for call in calls) == attempts
    assert all("-A" in call and "data/calibration/team_prop_pregame_ledger*" in call for call in calls if call[:2] == ["git", "add"])
    if expected_exit == 0:
        assert "changed=true" in output.read_text()
    else:
        assert not output.exists()


def test_calibration_backup_excludes_team_history(tmp_path):
    command = next(step["run"] for step in workflow("calibration-refresh.yml")["jobs"]["calibrate"]["steps"] if step.get("id") == "commit-calibration")
    # Execute the actual backup loop without the publication commands.
    backup_loop = command[command.index("for artifact in data/calibration/*;"):command.index("for attempt in")]
    calibration = tmp_path / "data/calibration"
    calibration.mkdir(parents=True)
    (calibration / "active.json").write_text('{"version":1}')
    (calibration / "outcome_ledger.json").write_text('{"records":[]}')
    (calibration / "outcome_ledger").mkdir()
    (calibration / "outcome_ledger/index.json").write_text('{}')
    (calibration / "team_prop_pregame_ledger.json").write_text('{"records":[]}')
    (calibration / "team_prop_pregame_ledger").mkdir()
    (calibration / "team_prop_pregame_ledger/index.json").write_text('{}')
    generated = tmp_path / "generated"
    generated.mkdir()
    subprocess.run(["bash", "-e", "-c", backup_loop], cwd=tmp_path, env={**os.environ, "GENERATED_CALIBRATION": str(generated)}, check=True)
    assert [path.name for path in generated.iterdir()] == ["active.json"]


def test_publication_pathspec_stages_shards_and_removes_legacy(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    legacy = tmp_path / "data/calibration/team_prop_pregame_ledger.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"records":[]}')
    subprocess.run(["git", "add", str(legacy)], cwd=tmp_path, check=True)
    write_team_prop_pregame_ledger({"records": [{"id": "new", "slate_date": "2026-10-08"}]}, tmp_path)
    for _ in range(2):
        # Also works once the monolith is no longer tracked at all.
        subprocess.run(["git", "add", "-A", "data/calibration/team_prop_pregame_ledger*"], cwd=tmp_path, check=True)
        paths = subprocess.check_output(["git", "ls-files"], cwd=tmp_path, text=True).splitlines()
        assert paths == ["data/calibration/team_prop_pregame_ledger/2026-10-08.json", "data/calibration/team_prop_pregame_ledger/index.json"]


@pytest.fixture(scope="module")
def publication_outcomes():
    # Reuse published rows, prices and grades; only simulate the pre-grade state.
    published = load_outcome_ledger(ROOT)
    props = [row for row in published["records"] if row["cache_type"] == "player_props_cache" and row["result"] in {"win", "loss"}]
    team = next(row for row in published["records"] if row["cache_type"] == "team_prop_pregame_ledger" and row["result"] == "win")
    push = next(row for row in published["records"] if row["result"] == "push")
    def pending(row):
        return {**row, "result": "pending", "outcome": None, "profit": None}
    current = {"schema_version": published["schema_version"], "records": [props[0], team, push, props[1], pending(props[2])]}
    generated = {"schema_version": published["schema_version"], "records": [pending(props[0]), pending(team), pending(push), props[2], props[3]]}
    expected = sorted([props[0], team, push, props[1], props[2], props[3]], key=lambda row: (row["date"], row["model_key"], row["id"]))
    return current, generated, expected


def test_outcome_union_preserves_whole_rows_grades_order_and_is_idempotent(publication_outcomes):
    current, generated, expected = copy.deepcopy(publication_outcomes)
    merged = merge_outcome_ledgers(current, generated)
    assert merged["records"] == expected
    assert merged["summary"] == {
        "total_picks": len(expected),
        "decided_picks": sum(row["result"] in {"win", "loss"} for row in expected),
        "pending_picks": 0,
        "trainable_decided_picks": sum(row["result"] in {"win", "loss"} and row["raw_probability"] is not None and row["calibration_eligible"] is True for row in expected),
    }
    assert merge_outcome_ledgers(merged, generated) == merged
    assert "updated_at" not in merged
    # Current rows remain authoritative when both copies have a result.
    old = copy.deepcopy(current)
    old["records"][0]["pregame_snapshot"].pop("ranking_updated_at", None)
    assert merge_outcome_ledgers(current, old)["records"] == sorted(current["records"], key=lambda row: (row["date"], row["model_key"], row["id"]))


def test_outcome_union_takes_fresh_generated_row_when_both_are_pending(publication_outcomes):
    current, _, _ = copy.deepcopy(publication_outcomes)
    stale = {**current["records"][4], "pregame_snapshot": {**current["records"][4]["pregame_snapshot"], "odds": -110}}
    fresh = {**current["records"][4], "pregame_snapshot": {**current["records"][4]["pregame_snapshot"], "odds": -125}}
    excluded = {**current["records"][1], "result": "pending", "settlement_exclusion_reason": "fractional_line"}
    current["records"][4] = stale
    current["records"][1] = excluded
    generated = {"schema_version": current["schema_version"], "records": [fresh, {**excluded, "settlement_exclusion_reason": None}]}
    rows = {row["id"]: row for row in merge_outcome_ledgers(current, generated)["records"]}
    assert rows[fresh["id"]]["pregame_snapshot"]["odds"] == -125
    # A grader retraction on the current row is never undone by a saved copy.
    assert rows[excluded["id"]]["settlement_exclusion_reason"] == "fractional_line"


@pytest.mark.parametrize("failure", ["schema", "missing-id", "duplicate"])
def test_outcome_union_rejects_ambiguous_input(publication_outcomes, failure):
    current, generated, _ = copy.deepcopy(publication_outcomes)
    if failure == "schema":
        generated["schema_version"] += 1
    elif failure == "missing-id":
        generated["records"][0].pop("id")
    else:
        generated["records"].append(generated["records"][0])
    with pytest.raises(ValueError):
        merge_outcome_ledgers(current, generated)


@pytest.mark.parametrize("filename,job,step_id,failed_pushes,attempts,exit_code", [
    ("calibration-refresh.yml", "calibrate", "commit-calibration", 0, 1, 0),
    ("calibration-refresh.yml", "calibrate", "commit-calibration", 2, 3, 0),
    ("calibration-refresh.yml", "calibrate", "commit-calibration", 5, 5, 1),
    ("player-props-refresh.yml", "refresh", "commit-props", 0, 1, 0),
    ("player-props-refresh.yml", "refresh", "commit-props", 2, 3, 0),
    ("player-props-refresh.yml", "refresh", "commit-props", 3, 3, 1),
])
@pytest.mark.parametrize("generated_shards,current_shards", [(False, False), (True, False), (False, True), (True, True)])
def test_outcome_publication_recovers_rows_and_grades_after_reset(
    tmp_path, publication_outcomes, filename, job, step_id, failed_pushes, attempts, exit_code,
    generated_shards, current_shards,
):
    command = next(step["run"] for step in workflow(filename)["jobs"][job]["steps"] if step.get("id") == step_id)
    current, generated, expected = publication_outcomes
    ledger = Path("data/calibration/outcome_ledger.json")
    remote = tmp_path / "remote"
    for root, payload, sharded in ((tmp_path, generated, generated_shards), (remote, current, current_shards)):
        (root / ledger).parent.mkdir(parents=True)
        if sharded:
            ordered = {**payload, "records": sorted(payload["records"], key=lambda row: (row["date"], row["model_key"], row["id"]))}
            write_outcome_ledger(ordered, root)
        else:
            (root / ledger).write_text(json.dumps(payload))
    trained = (ROOT / "data/calibration/state.json").read_bytes()
    (tmp_path / "data/calibration/state.json").write_bytes(trained)
    cache = json.loads((ROOT / "data/player_props_cache/latest.json").read_text())
    cache_date = cache["date"]
    cache_dir = tmp_path / "data/player_props_cache"
    cache_dir.mkdir()
    (cache_dir / f"{cache_date}.json").write_text(json.dumps(cache))
    snapshots = Path("data/player_props_snapshots")
    archived = sorted((ROOT / snapshots).glob("20??-??-??/*.json"))[-1]
    fresh_snapshot = json.dumps(cache)
    fresh_path = snapshots / cache_date / "generated.json"
    (tmp_path / fresh_path).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / fresh_path).write_text(fresh_snapshot)
    existing_path = archived.relative_to(ROOT)
    for root in (tmp_path, remote):
        (root / existing_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archived, root / existing_path)
    # A saved snapshot predates the grades present on the remote branch.
    stale = json.loads(archived.read_text())
    for bucket in stale["models"].values():
        for row in bucket.get("picks", []):
            row["result"] = "pending"
    (tmp_path / existing_path).write_text(json.dumps(stale))
    for relative in ("data/player_props_training/market_history_2026.jsonl", "data/player_props_training/outcome_history_2022_2026.jsonl.gz", "player_props/artifacts/metadata.json"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    runner = tmp_path / "runner"
    runner.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = f"#!{sys.executable}\n" + '''import json, os, pathlib, runpy, shutil, sys
name, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
root = pathlib.Path(os.environ["FIXTURE_ROOT"])
with (root / "calls.jsonl").open("a") as stream:
    stream.write(json.dumps([name, *args]) + "\\n")
if name == "git":
    if args[0] == "status":
        print(" M data/calibration/outcome_ledger.json")
    elif args[0] == "diff":
        sys.exit(1)
    elif args[0] == "reset":
        shutil.rmtree(root / "data")
        shutil.copytree(root / "remote/data", root / "data")
    elif args[0] == "push":
        path = root / "push-count"
        count = int(path.read_text()) + 1 if path.exists() else 1
        path.write_text(str(count))
        sys.exit(1 if count <= int(os.environ["FAILED_PUSHES"]) else 0)
elif args[0] == "scripts/merge_pick_outcome_ledger.py":
    sys.argv = [args[0], args[1], "--ledger", str(root / "data/calibration/outcome_ledger.json")]
    runpy.run_path(str(pathlib.Path(os.environ["SOURCE_ROOT"]) / args[0]), run_name="__main__")
elif args[0] == "scripts/merge_player_props_cache_payload.py":
    cache = root / "data/player_props_cache/latest.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args[1], cache)
elif args[0] == "-c":
    print(os.environ["CACHE_DATE"])
'''
    for name in ("git", "python"):
        path = bin_dir / name
        path.write_text(stub)
        path.chmod(0o755)
    command = command.replace("${{ steps.target-date.outputs.date }}", cache_date)
    env = {**os.environ, "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
           "FIXTURE_ROOT": str(tmp_path), "SOURCE_ROOT": str(ROOT), "FAILED_PUSHES": str(failed_pushes),
           "RUNNER_TEMP": str(runner), "GITHUB_ACTOR": "fixture", "CACHE_DATE": cache_date,
           "GITHUB_OUTPUT": str(tmp_path / "github-output")}
    result = subprocess.run(["bash", "-e", "-c", command], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == exit_code, result.stdout + result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert sum(call[:2] == ["git", "push"] for call in calls) == attempts
    resets = attempts if job == "refresh" else attempts - 1
    assert sum(call[:2] == ["git", "reset"] for call in calls) == resets
    assert sum(call[:2] == ["python", "scripts/merge_pick_outcome_ledger.py"] for call in calls) == resets
    actual = load_outcome_ledger(tmp_path)
    expected_records = expected if resets else generated["records"]
    assert sorted(actual["records"], key=lambda row: row["id"]) == sorted(expected_records, key=lambda row: row["id"])
    if job == "refresh":
        assert (tmp_path / fresh_path).read_text() == fresh_snapshot
        assert (tmp_path / existing_path).read_bytes() == archived.read_bytes()
        assert [call[1] for call in calls if call[0] == "python" and call[1].startswith("scripts/build_")] == ["scripts/build_parlay_cards.py", "scripts/build_profit_desk.py"] * attempts
    else:
        assert (tmp_path / "data/calibration/state.json").read_bytes() == trained
