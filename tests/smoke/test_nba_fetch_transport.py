from __future__ import annotations

import subprocess

import pytest
import requests

from scripts import run_nba_model as runner


URL = "https://stats.nba.com/stats/teamplayeronoffsummary"


@pytest.fixture(autouse=True)
def isolated_ci_transport(monkeypatch):
    monkeypatch.setenv("CI", "true")
    for setting in ("CONNECT_TIMEOUT", "READ_TIMEOUT", "ATTEMPTS"):
        monkeypatch.delenv(f"PICKLEDGER_NBA_HTTP_{setting}", raising=False)


def response(status=200, body=b'{"resultSets": []}'):
    result = requests.Response()
    result.status_code = status
    result._content = body
    result._content_consumed = True
    result.url = URL
    return result


@pytest.mark.parametrize("original,expected", [
    (None, (5, 15)), (60, (5, 15)), ((None, None), (5, 15)),
    ((2, 3), (2, 3)), ((10, 10), (5, 10)),
])
def test_nba_requests_cap_both_timeouts_without_lengthening_shorter_limits(original, expected):
    assert runner._bounded_timeout(original) == expected


@pytest.mark.parametrize("ci,read", [(None, 30), ("false", 30), ("0", 30), ("true", 15), ("1", 15)])
def test_transport_defaults_preserve_local_read_allowance(monkeypatch, ci, read):
    if ci is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", ci)
    calls = []
    monkeypatch.setattr(requests.Session, "send", lambda *_a, **kw: calls.append(kw) or response())
    with runner.BoundedNBAStatsSession() as session:
        session.get(URL, timeout=60)
        assert session.max_attempts == 2
    assert calls[0]["timeout"] == (5, read)


@pytest.mark.parametrize("attempts", [1, 3])
def test_environment_controls_timeout_bounds_and_retry_budget(monkeypatch, attempts):
    monkeypatch.setenv("PICKLEDGER_NBA_HTTP_CONNECT_TIMEOUT", "7.5")
    monkeypatch.setenv("PICKLEDGER_NBA_HTTP_READ_TIMEOUT", "45")
    monkeypatch.setenv("PICKLEDGER_NBA_HTTP_ATTEMPTS", str(attempts))
    calls, sleeps = [], []

    def send(_session, _request, **kwargs):
        calls.append(kwargs["timeout"])
        raise requests.ReadTimeout("fixture outage")

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with runner.BoundedNBAStatsSession() as session:
        with pytest.raises(runner.NBAFetchUnavailable, match=f"stopped after {attempts} attempt"):
            session.get(URL, timeout=60)
        assert runner._bounded_timeout((2, 3), session.connect_timeout, session.read_timeout) == (2, 3)
    assert calls == [(7.5, 45)] * attempts
    assert sleeps == [1] * (attempts - 1)


@pytest.mark.parametrize("value", ["", "nope", "0", "-1", "nan", "inf", "-inf", "1e999"])
def test_invalid_environment_settings_warn_and_keep_finite_defaults(monkeypatch, capsys, value):
    for setting in ("CONNECT_TIMEOUT", "READ_TIMEOUT", "ATTEMPTS"):
        monkeypatch.setenv(f"PICKLEDGER_NBA_HTTP_{setting}", value)
    with runner.BoundedNBAStatsSession() as session:
        assert (session.connect_timeout, session.read_timeout, session.max_attempts) == (5, 15, 2)
    assert capsys.readouterr().err.count("Invalid PICKLEDGER_NBA_HTTP_") == 3


def test_fractional_attempt_count_is_rejected(monkeypatch, capsys):
    monkeypatch.setenv("PICKLEDGER_NBA_HTTP_ATTEMPTS", "1.5")
    with runner.BoundedNBAStatsSession() as session:
        assert session.max_attempts == 2
    assert "Invalid PICKLEDGER_NBA_HTTP_ATTEMPTS" in capsys.readouterr().err


def test_success_preserves_response_bytes_headers_and_parameters(monkeypatch):
    expected = response(body=b'{"resultSets":[{"name":"fixture","rowSet":[]}]}')
    calls = []

    def send(_session, request, **kwargs):
        calls.append((request, kwargs))
        return expected

    monkeypatch.setattr(requests.Session, "send", send)
    with runner.BoundedNBAStatsSession() as session:
        actual = session.get(URL, params={"Season": "2025-26", "TeamID": 123},
                             headers={"User-Agent": "existing-nba-header", "Referer": "https://www.nba.com/"},
                             timeout=60)
    assert actual is expected
    assert actual.content == expected.content
    request, options = calls[0]
    assert request.url == URL + "?Season=2025-26&TeamID=123"
    assert request.headers["User-Agent"] == "existing-nba-header"
    assert request.headers["Referer"] == "https://www.nba.com/"
    assert options["timeout"] == (5, 15)
    assert len(calls) == 1


def test_exhausted_request_stops_model_before_its_broad_retry_handlers(monkeypatch, capsys):
    calls, sleeps = [], []

    def send(_session, _request, **kwargs):
        calls.append(kwargs["timeout"])
        raise requests.ReadTimeout("blocked upstream")

    monkeypatch.setattr(requests.Session, "send", send)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with runner.BoundedNBAStatsSession() as session, pytest.raises(runner.NBAFetchUnavailable) as exc:
        try:
            session.get(URL, timeout=60)
        except Exception:
            pytest.fail("Model exception handler must not retry an exhausted transport")
    assert calls == [(5, 15), (5, 15)]
    assert sleeps == [1]
    assert runner.FAILURE_MARKER in str(exc.value)
    assert "stats.nba.com/stats/teamplayeronoffsummary ReadTimeout" in str(exc.value)
    assert "attempt 2/2" in capsys.readouterr().err


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_can_recover_with_one_http_retry(monkeypatch, status):
    good = response()
    responses = iter([response(status), good])
    monkeypatch.setattr(requests.Session, "send", lambda *_a, **_kw: next(responses))
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: None)
    with runner.BoundedNBAStatsSession() as session:
        assert session.get(URL) is good


@pytest.mark.parametrize("status,body,reason", [
    (403, b"Forbidden", "HTTP 403"),
    (200, b"<html>unavailable</html>", "invalid JSON (HTTP 200)"),
])
def test_block_pages_fail_once_with_clear_diagnostics(monkeypatch, status, body, reason):
    calls = []
    monkeypatch.setattr(requests.Session, "send", lambda *_a, **_kw: calls.append(1) or response(status, body))
    monkeypatch.setattr(runner.time, "sleep", lambda _seconds: pytest.fail("Unexpected retry"))
    with runner.BoundedNBAStatsSession() as session, pytest.raises(runner.NBAFetchUnavailable, match=runner.FAILURE_MARKER) as exc:
        session.get(URL)
    assert calls == [1]
    assert reason in str(exc.value)


def test_nba_session_does_not_change_other_hosts(monkeypatch):
    options = []
    expected = response(body=b"non-JSON resource")
    monkeypatch.setattr(requests.Session, "send", lambda *_a, **kw: options.append(kw) or expected)
    with runner.BoundedNBAStatsSession() as session:
        assert session.get("https://example.test/resource", timeout=60) is expected
    assert options[0]["timeout"] == 60


def test_bootstrap_installs_session_without_changing_model_args(monkeypatch, capsys):
    from nba_api.stats.library.http import NBAStatsHTTP

    monkeypatch.setattr(NBAStatsHTTP, "_session", None)
    monkeypatch.setattr(runner.sys, "path", list(runner.sys.path))
    monkeypatch.setattr(runner.sys, "argv", ["wrapper", "--date", "2026-10-07", "--variant", "new", "--no-log"])

    def execute(path, run_name):
        assert path.endswith("NBAPredictionModel/run_live.py")
        assert run_name == "__main__"
        assert runner.sys.argv[1:] == ["--date", "2026-10-07", "--variant", "new", "--no-log"]
        assert isinstance(NBAStatsHTTP.get_session(), runner.BoundedNBAStatsSession)
        raise runner.NBAFetchUnavailable(runner.FAILURE_MARKER + " fixture outage")

    monkeypatch.setattr(runner.runpy, "run_path", execute)
    assert runner.main() == 75
    assert "fixture outage" in capsys.readouterr().err


def test_parent_discards_partial_stdout_when_fetch_fails(monkeypatch):
    import pickgrader_server as server

    monkeypatch.setattr(server, "_espn_event_count_for_date", lambda *_a: 1)
    monkeypatch.setattr(server, "_parse_nba_output", lambda *_a, **_kw: pytest.fail("Partial results must not be parsed"))
    monkeypatch.setattr(server, "_save_admin_picks_doc", lambda *_a, **_kw: pytest.fail("Failed run must not be saved"))
    invoked = []

    def run(*args, **kwargs):
        invoked.append((args, kwargs))
        return "partial output\n" + runner.FAILURE_MARKER + " stats.nba.com/stats/scoreboardv2 ReadTimeout"

    monkeypatch.setattr(server, "_run_script", run)
    result = server.run_nba_model("2026-10-07")
    assert result["ok"] is False
    assert result["error_kind"] == "upstream_unavailable"
    assert result["retryable"] is False
    assert invoked[0][0][1].endswith("scripts/run_nba_model.py")
    assert invoked[0][0][2] == server.NBA_MODEL_DIR


def test_parent_preserves_child_diagnostics_on_outer_timeout(monkeypatch):
    import pickgrader_server as server

    monkeypatch.setattr(server, "_espn_event_count_for_date", lambda *_a: 1)

    def run(*_args, **kwargs):
        raise subprocess.TimeoutExpired("fixture", kwargs["timeout"], output=b"Fetching injuries...\n",
                                        stderr=b"[nba-fetch] stats.nba.com/stats/scoreboardv2 attempt 1/2\n")

    monkeypatch.setattr(server, "_run_script", run)
    result = server.run_nba_model("2026-10-07")
    assert result["ok"] is False
    assert result["retryable"] is False
    assert "Fetching injuries" in result["diagnostic"]
    assert "stats.nba.com" in result["diagnostic"]
