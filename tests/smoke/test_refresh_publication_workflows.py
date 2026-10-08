from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml


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
    ledger = tmp_path / "data/calibration/team_prop_pregame_ledger.json"
    cache.parent.mkdir(parents=True)
    ledger.parent.mkdir(parents=True)
    cache.write_text('{"date":"2026-10-07","models":{}}')
    ledger.write_text('{}')
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
    assert [int(call[1]) for call in calls if call[0] == "sleep"] == delays
    assert (runner_temp / "model-cache-latest.json").read_bytes() == cache.read_bytes()
    assert (runner_temp / "team-prop-pregame-ledger.json").read_bytes() == ledger.read_bytes()
    if expected_exit == 0:
        assert "changed=true" in output.read_text()
    else:
        assert not output.exists()
