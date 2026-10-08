"""A transient upstream outage must not erase an already published slate."""
from __future__ import annotations

import json

import pytest

from scripts.merge_model_cache_payload import merge_payload
from scripts.refresh_model_cache import _is_transient_model_error, _run_model_job_with_retries


@pytest.mark.parametrize("error", [
    "502 Server Error: Bad Gateway for url: schedule",
    "429 Client Error: Too Many Requests",
    "503 Server Error: Service Unavailable",
    "504 Server Error: Gateway Timeout",
    "Connection reset by peer",
])
def test_temporary_upstream_errors_are_retryable(error):
    assert _is_transient_model_error({"ok": False, "error": error})


def test_transient_error_retries_and_returns_recovered_model(monkeypatch):
    from scripts import refresh_model_cache

    responses = iter([{"ok": False, "error": "502 Server Error: Bad Gateway"}, {"ok": True, "picks": []}])
    sleeps = []
    monkeypatch.setattr(refresh_model_cache.time, "sleep", sleeps.append)
    assert _run_model_job_with_retries("mlb_new", lambda: next(responses)) == {"ok": True, "picks": []}
    assert sleeps == [3]


def test_invalid_model_is_not_retried(monkeypatch):
    from scripts import refresh_model_cache

    monkeypatch.setattr(refresh_model_cache.time, "sleep", lambda _seconds: pytest.fail("unexpected retry"))
    failure = {"ok": False, "error": "model artifact is invalid"}
    assert _run_model_job_with_retries("cfb", lambda: failure) == failure


def test_failed_refresh_keeps_same_date_picks_and_failure_visible(tmp_path):
    pick = {"source": "MLB", "sport": "MLB", "date": "2026-09-06", "pick": "Home ML", "result": "win"}
    current = {"date": "2026-09-06", "models": {"mlb_new": {"ok": True, "picks": [pick], "games": 1}}}
    (tmp_path / "latest.json").write_text(json.dumps(current))
    generated = {"date": "2026-09-06", "models": {"mlb_new": {"ok": False, "error": "502 Bad Gateway"}}}
    merged = merge_payload(generated, tmp_path)
    bucket = merged["models"]["mlb_new"]
    assert bucket["picks"] == [pick]
    assert bucket["games"] == 1
    assert bucket["ok"] is False
    assert bucket["error"] == "502 Bad Gateway"
    assert bucket["preserved_after_refresh_error"] is True
    assert merged["mlb_new"] == bucket


def test_failed_refresh_never_carries_yesterdays_model_picks(tmp_path):
    current = {"date": "2026-09-05", "models": {"mlb_new": {"ok": True, "picks": [{"pick": "Old pick"}]}}}
    (tmp_path / "latest.json").write_text(json.dumps(current))
    generated = {"date": "2026-09-06", "models": {"mlb_new": {"ok": False, "error": "502 Bad Gateway"}}}
    bucket = merge_payload(generated, tmp_path)["models"]["mlb_new"]
    assert bucket["ok"] is False
    assert not bucket.get("picks")


def test_later_success_clears_failed_refresh_marker(tmp_path):
    current = {"date": "2026-09-06", "models": {"cfb": {"ok": False, "picks": [], "preserved_after_refresh_error": True}}}
    (tmp_path / "latest.json").write_text(json.dumps(current))
    generated = {"date": "2026-09-06", "models": {"cfb": {"ok": True, "picks": []}}}
    bucket = merge_payload(generated, tmp_path)["models"]["cfb"]
    assert bucket["ok"] is True
    assert "preserved_after_refresh_error" not in bucket

def test_partial_model_failures_exit_nonzero(monkeypatch, tmp_path):
    """A mixed ok/error refresh must not exit 0 (Auditor A model-cache-partial-exit0)."""
    from types import SimpleNamespace

    from scripts import refresh_model_cache as rmc

    monkeypatch.setattr(rmc, "MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        rmc,
        "_parse_args",
        lambda: SimpleNamespace(
            date="2026-10-01",
            models="ok_model,bad_model",
            max_workers=1,
            skip_firestore=True,
        ),
    )
    monkeypatch.setattr(
        rmc,
        "_model_jobs",
        lambda _date: {
            "ok_model": lambda: {"ok": True, "picks": [{"id": "a"}]},
            "bad_model": lambda: {"ok": False, "error": "boom", "picks": []},
        },
    )
    monkeypatch.setattr(rmc, "_run_model_job_with_retries", lambda _key, job: job())
    written = {}

    def _write(date_iso, payload):
        written["date"] = date_iso
        written["errors"] = list(payload.get("errors") or [])
        written["models"] = dict(payload.get("models") or {})
        (tmp_path / f"{date_iso}.json").write_text("{}")
        return payload

    monkeypatch.setattr(rmc, "_write_json_cache", _write)
    monkeypatch.setattr(rmc.server, "_write_admin_picks_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(rmc.server, "_parse_model_date_arg", lambda _raw: ("2026-10-01", None))

    assert rmc.main() == 1
    assert written["date"] == "2026-10-01"
    assert written["errors"] and "bad_model" in written["errors"][0]
    assert written["models"]["ok_model"]["ok"] is True
    assert written["models"]["bad_model"]["ok"] is False


def _stub_refresh_jobs(monkeypatch, tmp_path, jobs: dict):
    from types import SimpleNamespace

    from scripts import refresh_model_cache as rmc

    monkeypatch.setattr(rmc, "MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        rmc,
        "_parse_args",
        lambda: SimpleNamespace(
            date="2026-10-01",
            models=",".join(jobs),
            max_workers=1,
            skip_firestore=True,
        ),
    )
    monkeypatch.setattr(rmc, "_model_jobs", lambda _date: {key: (lambda value=value: value) for key, value in jobs.items()})
    monkeypatch.setattr(rmc, "_run_model_job_with_retries", lambda _key, job: job())
    written: dict = {}

    def _write(date_iso, payload):
        written["date"] = date_iso
        written["errors"] = list(payload.get("errors") or [])
        written["models"] = dict(payload.get("models") or {})
        (tmp_path / f"{date_iso}.json").write_text("{}")
        return payload

    monkeypatch.setattr(rmc, "_write_json_cache", _write)
    monkeypatch.setattr(rmc.server, "_write_admin_picks_cache", lambda *_a, **_k: None)
    monkeypatch.setattr(rmc.server, "_parse_model_date_arg", lambda _raw: ("2026-10-01", None))
    return rmc, written


def test_nba_timeout_soft_fails_so_other_models_still_publish(monkeypatch, tmp_path):
    """NBA New wall-clock timeout must not block mlb/nfl/nhl publication."""
    rmc, written = _stub_refresh_jobs(
        monkeypatch,
        tmp_path,
        {
            "mlb_new": {"ok": True, "picks": [{"id": "mlb"}]},
            "nfl": {"ok": True, "picks": [{"id": "nfl"}]},
            "nhl": {"ok": True, "picks": [{"id": "nhl"}]},
            "nba": {"ok": False, "error": "NBA New timed out (15 min limit)", "picks": []},
        },
    )

    assert rmc.main() == 0
    assert written["models"]["mlb_new"]["ok"] is True
    assert written["models"]["nfl"]["ok"] is True
    assert written["models"]["nhl"]["ok"] is True
    assert written["models"]["nba"]["ok"] is False
    assert any("nba:" in line and "timed out" in line for line in written["errors"])


def test_nba_non_timeout_error_still_exits_nonzero(monkeypatch, tmp_path):
    rmc, written = _stub_refresh_jobs(
        monkeypatch,
        tmp_path,
        {
            "mlb_new": {"ok": True, "picks": [{"id": "mlb"}]},
            "nba": {"ok": False, "error": "NBA New parser found no predictions", "picks": []},
        },
    )

    assert rmc.main() == 1
    assert written["models"]["mlb_new"]["ok"] is True
    assert written["models"]["nba"]["ok"] is False


def test_non_nba_timeout_still_exits_nonzero(monkeypatch, tmp_path):
    rmc, written = _stub_refresh_jobs(
        monkeypatch,
        tmp_path,
        {
            "mlb_new": {"ok": True, "picks": [{"id": "mlb"}]},
            "nhl": {"ok": False, "error": "NHL Model timed out (8 min limit)", "picks": []},
        },
    )

    assert rmc.main() == 1
    assert written["models"]["nhl"]["ok"] is False


def test_nba_timeout_alone_still_exits_nonzero(monkeypatch, tmp_path):
    """A refresh that only selected NBA and timed out must not look healthy."""
    rmc, written = _stub_refresh_jobs(
        monkeypatch,
        tmp_path,
        {"nba": {"ok": False, "error": "NBA New timed out (15 min limit)", "picks": []}},
    )

    assert rmc.main() == 1
    assert written["models"]["nba"]["ok"] is False


def test_nba_model_timeout_honors_env_override(monkeypatch):
    import subprocess

    import pickgrader_server as server

    captured = {}
    monkeypatch.setenv("PICKLEDGER_NBA_MODEL_TIMEOUT_SECONDS", "900")
    monkeypatch.setattr(server, "_espn_event_count_for_date", lambda *_a, **_k: 5)
    monkeypatch.setattr(server, "_resolve_python_bin", lambda *_a, **_k: "python")
    monkeypatch.setattr(server, "_nba_model_extra_args", lambda *_a, **_k: ["--date", "2026-10-05"])

    def fake_run_script(*_args, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired("run_live.py", kwargs["timeout"])

    monkeypatch.setattr(server, "_run_script", fake_run_script)
    result = server.run_nba_model("2026-10-05", "new")
    assert captured["timeout"] == 900
    assert result["ok"] is False
    assert "timed out" in result["error"]
    assert "15 min" in result["error"]


def test_nba_model_timeout_default_remains_seven_minutes(monkeypatch):
    import subprocess

    import pickgrader_server as server

    captured = {}
    monkeypatch.delenv("PICKLEDGER_NBA_MODEL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(server, "_espn_event_count_for_date", lambda *_a, **_k: 5)
    monkeypatch.setattr(server, "_resolve_python_bin", lambda *_a, **_k: "python")
    monkeypatch.setattr(server, "_nba_model_extra_args", lambda *_a, **_k: ["--date", "2026-10-05"])

    def fake_run_script(*_args, **kwargs):
        captured["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired("run_live.py", kwargs["timeout"])

    monkeypatch.setattr(server, "_run_script", fake_run_script)
    result = server.run_nba_model("2026-10-05", "new")
    assert captured["timeout"] == 420
    assert "7 min" in result["error"]


def test_actions_raises_nba_timeout_without_blanket_raise():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "model-cache-refresh.yml"
    ).read_text(encoding="utf-8")
    assert "PICKLEDGER_NBA_MODEL_TIMEOUT_SECONDS: 900" in workflow
    assert "PICKLEDGER_MLB_MODEL_TIMEOUT_SECONDS: 1200" in workflow
    assert "PICKLEDGER_MLB_INNING_TIMEOUT_SECONDS: 240" in workflow
    source = (
        Path(__file__).resolve().parents[2] / "scripts" / "refresh_model_cache.py"
    ).read_text(encoding="utf-8")
    assert 'SOFT_FAIL_TIMEOUT_MODEL_KEYS = frozenset({"nba"})' in source


@pytest.mark.parametrize("failure", [
    {"ok": False, "error": "NBA New timed out (15 min limit)"},
    {"ok": False, "error": "NHL Model timed out (8 min limit)"},
    {"ok": False, "error": "ReadTimeout after bounded HTTP retries", "retryable": False},
])
def test_exhausted_model_or_fetch_budget_is_never_retried(monkeypatch, failure):
    from scripts import refresh_model_cache as rmc

    calls = []
    monkeypatch.setattr(rmc.time, "sleep", lambda _seconds: pytest.fail("Unexpected model retry"))
    assert rmc._run_model_job_with_retries("nba", lambda: calls.append(1) or failure) == failure
    assert calls == [1]


@pytest.mark.parametrize("jobs,exit_code,status,publishable", [
    ({"nba": {"ok": False, "error": "NBA New timed out (15 min limit)"}}, 1, "failed", "false"),
    ({"nhl": {"ok": True, "picks": []}}, 0, "success", "true"),
    ({"nhl": {"ok": True}, "nba": {"ok": False, "error": "bad parser"}}, 1, "partial", "true"),
    ({"nhl": {"ok": True}, "nba": {"ok": False, "error": "fetch unavailable",
                                       "error_kind": "upstream_unavailable", "retryable": False}}, 0, "partial", "true"),
])
def test_refresh_outputs_publish_healthy_siblings_and_report_actual_status(monkeypatch, tmp_path, capsys,
                                                                         jobs, exit_code, status, publishable):
    rmc, written = _stub_refresh_jobs(monkeypatch, tmp_path, jobs)
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert rmc.main() == exit_code
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values == {"cache_written": "true", "publishable": publishable, "cache_date": "2026-10-01"}
    summary = json.loads("{" + capsys.readouterr().out.rsplit("\n{", 1)[1])
    assert summary["ok"] is (exit_code == 0)
    assert summary["status"] == status
    assert summary["successful_models"] == sum(bool(job.get("ok")) for job in jobs.values())
    assert written["models"] == jobs


def test_cache_write_failure_cannot_authorize_publication(monkeypatch, tmp_path):
    rmc, _ = _stub_refresh_jobs(monkeypatch, tmp_path, {"nhl": {"ok": True}})
    output = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    def fail_write(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(rmc, "_write_json_cache", fail_write)
    with pytest.raises(OSError, match="disk full"):
        rmc.main()
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert values["cache_written"] == "false"
    assert values["publishable"] == "false"
