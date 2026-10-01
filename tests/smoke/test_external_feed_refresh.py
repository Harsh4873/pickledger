from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

from scripts import refresh_external_feeds as refresh


def _configure(monkeypatch, tmp_path, feeds, *, date="2026-09-06"):
    monkeypatch.setattr(refresh, "MODEL_CACHE_DIR", tmp_path)
    monkeypatch.setattr(refresh, "FEED_RUNNERS", feeds)
    monkeypatch.setattr(refresh, "apply_market_odds_to_payload", lambda *_args: None)
    monkeypatch.setattr(refresh, "apply_calibration_to_payload", lambda *_args: None)
    monkeypatch.setattr(
        refresh,
        "_parse_args",
        lambda: SimpleNamespace(date=date, feeds=",".join(feeds), sports="mlb,cfb", skip_firestore=True),
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def _write_previous(tmp_path, bucket, *, date="2026-09-06"):
    payload = {"date": date, "models": {}, "external_feeds": {"forebet_mlb": bucket}}
    (tmp_path / "latest.json").write_text(json.dumps(payload))


def test_refresh_publishes_outage_diagnostics_without_redating_last_good_picks(monkeypatch, tmp_path):
    prior = {
        "ok": True,
        "date": "2026-09-05",
        "updatedAt": "2026-09-05T14:00:00Z",
        "picks": [{"id": "verified", "date": "2026-09-05", "decision": "PASS", "units": 0}],
    }
    _write_previous(tmp_path, prior, date="2026-09-05")
    _configure(
        monkeypatch, tmp_path,
        {"forebet_mlb": lambda *_args: {"ok": False, "error": "Listing blocked by Cloudflare"}},
    )
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert refresh.main() == 1

    published = json.loads((tmp_path / "latest.json").read_text())
    bucket = published["external_feeds"]["forebet_mlb"]
    assert bucket["picks"] == prior["picks"]
    assert bucket["date"] == "2026-09-05"
    assert bucket["updatedAt"] == prior["updatedAt"]
    assert bucket["lastSuccessAt"] == prior["updatedAt"]
    assert bucket["lastAttemptDate"] == "2026-09-06"
    assert bucket["refreshStatus"] == "error"
    assert "Cloudflare" in bucket["lastError"]
    assert published["external_feed_errors"] == ["forebet_mlb: Listing blocked by Cloudflare"]
    assert "| forebet_mlb | error | 2026-09-05 | 1 |" in summary.read_text()


def test_forebet_cross_date_block_reports_attempted_url_with_retained_snapshot(monkeypatch, tmp_path):
    from scripts.source_health import source_issues

    prior = {
        "ok": True, "date": "2026-09-05", "updatedAt": "2026-09-05T14:00:00Z",
        "picks": [{"pick": "Prior verified pick", "date": "2026-09-05"}],
        "meta": {"officialMatchups": 1, "matchedPicks": 1, "blockedUrls": 0},
    }
    _write_previous(tmp_path, prior, date="2026-09-05")
    url = "https://www.forebet.com/en/baseball/usa/mlb"
    _configure(monkeypatch, tmp_path, {"forebet_mlb": lambda *_args: {
        "ok": False, "date": "2026-09-06", "picks": [], "error": "Cloudflare blocked",
        "meta": {"officialMatchups": 2, "matchedPicks": 0, "blockedUrls": 1,
                 "blockedUrl": url, "missingMatchups": ["A @ B", "C @ D"]},
    }})

    assert refresh.main() == 1
    bucket = json.loads((tmp_path / "latest.json").read_text())["external_feeds"]["forebet_mlb"]
    assert bucket["date"] == "2026-09-05"
    assert bucket["lastAttemptDate"] == "2026-09-06"
    assert bucket["lastAttemptMeta"]["blockedUrl"] == url
    assert f"Forebet listing blocked: {url}" in source_issues("forebet_mlb", bucket, "2026-09-06")


def test_forebet_same_day_cloudflare_retry_keeps_picks_but_reports_failure(monkeypatch, tmp_path):
    from scripts.source_health import source_issues

    date = "2026-09-06"
    prior = {
        "ok": True, "date": date, "updatedAt": "2026-09-06T14:00:00Z",
        "refreshStatus": "ok",
        "picks": [{"pick": "Cubs ML", "matchup": "Cubs @ Padres", "date": date}],
        "meta": {"officialMatchups": 1, "expectedMatchups": 1,
                 "matchedPicks": 1, "missingMatchups": [],
                 "updatedAt": "2026-09-06T14:00:00Z", "date": date},
    }
    _write_previous(tmp_path, prior)
    _configure(monkeypatch, tmp_path, {"forebet_mlb": lambda *_args: {
        "ok": False, "date": date, "picks": [],
        "error": "ForebetMLB: listing blocked by Cloudflare",
        "meta": {"officialMatchups": 1, "expectedMatchups": 1,
                 "matchedPicks": 0, "missingMatchups": ["Cubs @ Padres"], "blockedUrls": 1},
    }})

    assert refresh.main() == 1
    published = json.loads((tmp_path / "latest.json").read_text())
    for container in (published["models"], published["external_feeds"], published):
        bucket = container["forebet_mlb"]
        assert bucket["picks"] == prior["picks"]
        assert bucket["ok"] is False
        assert bucket["refreshStatus"] == "error"
        assert bucket["updatedAt"] == prior["updatedAt"]
        assert bucket["meta"]["updatedAt"] == prior["meta"]["updatedAt"]
        assert bucket["lastSuccessAt"] == prior["updatedAt"]
        assert bucket["lastAttemptDate"] == date
        assert "Cloudflare" in bucket["lastError"]
        assert source_issues("forebet_mlb", bucket, date)


def test_forebet_legacy_partial_coverage_is_not_recertified_on_failed_retry(monkeypatch, tmp_path):
    date = "2026-09-06"
    prior = {
        "ok": True, "date": date, "updatedAt": "2026-09-06T14:00:00Z",
        "picks": [{"pick": "Cubs ML", "matchup": "Cubs @ Padres", "date": date}],
        "meta": {"officialMatchups": 2, "expectedMatchups": 1,
                 "matchedPicks": 1, "unpublishedMatchups": ["Giants @ Dodgers"]},
    }
    _write_previous(tmp_path, prior)
    _configure(monkeypatch, tmp_path, {"forebet_mlb": lambda *_args: {
        "ok": False, "date": date, "picks": [], "error": "Cloudflare",
        "meta": {"officialMatchups": 2, "missingMatchups": ["Cubs @ Padres", "Giants @ Dodgers"]},
    }})

    assert refresh.main() == 1
    bucket = json.loads((tmp_path / "latest.json").read_text())["external_feeds"]["forebet_mlb"]
    assert bucket["ok"] is False
    assert bucket["meta"]["expectedMatchups"] == 2
    assert bucket["meta"]["matchedPicks"] == 1
    assert bucket["meta"]["missingMatchups"] == ["Giants @ Dodgers"]
    assert "lastSuccessAt" not in bucket


def test_forebet_partial_success_result_is_demoted_to_incomplete(monkeypatch, tmp_path):
    date = "2026-09-06"
    _configure(monkeypatch, tmp_path, {"forebet_mlb": lambda *_args: {
        "ok": True, "date": date,
        "picks": [{"pick": "Cubs ML", "matchup": "Cubs @ Padres", "date": date}],
        "meta": {"officialMatchups": 2, "expectedMatchups": 1,
                 "matchedPicks": 1, "unpublishedMatchups": ["Giants @ Dodgers"]},
    }})

    assert refresh.main() == 1
    bucket = json.loads((tmp_path / "latest.json").read_text())["external_feeds"]["forebet_mlb"]
    assert bucket["ok"] is False
    assert bucket["meta"]["expectedMatchups"] == 2
    assert bucket["meta"]["missingMatchups"] == ["Giants @ Dodgers"]
    assert "lastSuccessAt" not in bucket
    assert bucket["refreshStatus"] == "error"


def test_tennistonic_failed_retry_retains_pick_with_full_official_gap(monkeypatch, tmp_path):
    from scripts.source_health import source_issues

    date = "2026-09-06"
    prior = {
        "ok": True, "date": date, "updatedAt": "2026-09-06T14:00:00Z",
        "picks": [{"pick": "Krejcikova ML", "matchup": "Krejcikova vs Havlickova", "date": date}],
        "meta": {"officialMatchups": 2, "expectedMatchups": 1,
                 "matchedPicks": 1, "unpublishedMatchups": ["Alcaraz vs Paul"]},
    }
    (tmp_path / "latest.json").write_text(json.dumps({
        "date": date, "models": {}, "external_feeds": {"tennistonic_tennis": prior},
    }))
    _configure(monkeypatch, tmp_path, {"tennistonic_tennis": lambda *_args: {
        "ok": False, "date": date, "picks": [], "error": "TennisTonic incomplete slate (403)",
        "meta": {"officialMatchups": 2, "expectedMatchups": 2,
                 "matchedPicks": 0, "missingMatchups": ["Krejcikova vs Havlickova", "Alcaraz vs Paul"],
                 "unavailableMatchups": ["Alcaraz vs Paul"]},
    }})

    assert refresh.main() == 1
    bucket = json.loads((tmp_path / "latest.json").read_text())["external_feeds"]["tennistonic_tennis"]
    assert bucket["picks"] == prior["picks"]
    assert bucket["ok"] is False
    assert bucket["meta"]["officialMatchups"] == 2
    assert bucket["meta"]["expectedMatchups"] == 2
    assert bucket["meta"]["matchedPicks"] == 1
    assert bucket["meta"]["missingMatchups"] == ["Alcaraz vs Paul"]
    assert bucket["refreshStatus"] == "error"
    assert source_issues("tennistonic_tennis", bucket, date)


def test_separate_forebet_runs_keep_prior_same_day_error_visible(monkeypatch, tmp_path):
    date = "2026-09-06"
    _configure(monkeypatch, tmp_path, {"forebet_mlb": lambda *_args: {
        "ok": False, "date": date, "picks": [], "error": "Cloudflare blocked MLB",
        "meta": {"officialMatchups": 1, "expectedMatchups": 1,
                 "matchedPicks": 0, "missingMatchups": ["Cubs @ Padres"]},
    }})
    assert refresh.main() == 1

    _configure(monkeypatch, tmp_path, {"forebet_wnba": lambda *_args: {
        "ok": True, "date": date, "picks": [],
        "meta": {"officialMatchups": 0, "expectedMatchups": 0,
                 "matchedPicks": 0, "missingMatchups": []},
    }})
    assert refresh.main() == 0
    published = json.loads((tmp_path / "latest.json").read_text())
    assert published["external_feeds"]["forebet_wnba"]["refreshStatus"] == "ok"
    assert published["external_feeds"]["forebet_mlb"]["refreshStatus"] == "error"
    assert published["external_feed_errors"] == ["forebet_mlb: Cloudflare blocked MLB"]


def test_forebet_and_tennistonic_resume_distinct_verified_partial_picks():
    date = "2026-09-06"
    for feed, separator in (("forebet_mlb", " @ "), ("tennistonic_tennis", " vs ")):
        first_match = f"A{separator}B"
        second_match = f"C{separator}D"
        third_match = f"E{separator}F"
        previous = {
            "ok": False, "date": date, "picks": [
                {"pick": "A ML", "matchup": first_match, "date": date}],
            "meta": {"feed": feed, "officialMatchups": 3, "expectedMatchups": 3,
                     "matchedPicks": 1, "missingMatchups": [second_match, third_match]},
        }
        attempt = {
            "ok": False, "date": date, "picks": [
                {"pick": "C ML", "matchup": second_match, "date": date}],
            "error": "soft timeout",
            "meta": {"feed": feed, "officialMatchups": 3, "expectedMatchups": 3,
                     "matchedPicks": 1, "missingMatchups": [first_match, third_match]},
        }
        record = (refresh._record_forebet_attempt if feed.startswith("forebet_")
                  else refresh._record_tennistonic_attempt)
        bucket = record(previous, attempt, date, "2026-09-06T17:00:00Z")
        assert bucket["ok"] is False
        assert bucket["refreshStatus"] == "error"
        assert {pick["matchup"] for pick in bucket["picks"]} == {first_match, second_match}
        assert bucket["meta"]["expectedMatchups"] == 3
        assert bucket["meta"]["matchedPicks"] == 2
        assert bucket["meta"]["missingMatchups"] == [third_match]
        assert bucket["meta"]["resumedPicks"] == 1


def test_refresh_publishes_todays_partial_scores24_cfb_instead_of_yesterday(monkeypatch, tmp_path):
    yesterday = {
        "ok": True,
        "date": "2026-09-10",
        "updatedAt": "2026-09-10T19:12:03Z",
        "picks": [{"pick": "Miami Under", "date": "2026-09-10", "decision": "PASS", "units": 0}],
    }
    (tmp_path / "latest.json").write_text(
        json.dumps({"date": "2026-09-10", "models": {}, "external_feeds": {"scores24_cfb": yesterday}})
    )
    _configure(
        monkeypatch,
        tmp_path,
        {
            "scores24_cfb": lambda *_args: {
                "ok": False,
                "date": "2026-09-11",
                "error": "Scores24CFB scrape timed out after 180s with 4 matched pick(s) of 5 official matchup(s)",
                "picks": [
                    {"pick": "Louisville ML", "date": "2026-09-11", "source": "Scores24CFB"},
                    {"pick": "Stanford ML", "date": "2026-09-11", "source": "Scores24CFB"},
                    {"pick": "Ole Miss ML", "date": "2026-09-11", "source": "Scores24CFB"},
                    {"pick": "Alabama ML", "date": "2026-09-11", "source": "Scores24CFB"},
                ],
            }
        },
        date="2026-09-11",
    )

    assert refresh.main() == 1

    published = json.loads((tmp_path / "latest.json").read_text())
    bucket = published["external_feeds"]["scores24_cfb"]
    assert bucket["date"] == "2026-09-11"
    assert bucket["ok"] is False
    assert bucket["refreshStatus"] == "error"
    assert bucket["lastAttemptDate"] == "2026-09-11"
    assert len(bucket["picks"]) == 4
    assert all(pick["date"] == "2026-09-11" for pick in bucket["picks"])
    assert "timed out" in bucket["lastError"]


def test_failed_same_day_retry_keeps_larger_partial_until_completion():
    date_iso = "2026-09-11"
    first = {
        "ok": False,
        "date": date_iso,
        "picks": [
            {"pick": "Louisville ML", "date": date_iso, "source": "Scores24CFB"},
            {"pick": "Stanford ML", "date": date_iso, "source": "Scores24CFB"},
        ],
        "error": "first run timed out",
    }
    partial = refresh._record_feed_attempt(None, first, date_iso, "2026-09-11T12:00:00Z")
    smaller = refresh._record_feed_attempt(
        partial,
        {
            "ok": False,
            "date": date_iso,
            "picks": [{"pick": "Louisville ML", "date": date_iso, "source": "Scores24CFB"}],
            "error": "second run was blocked",
        },
        date_iso,
        "2026-09-11T17:00:00Z",
    )
    assert smaller["ok"] is False
    assert len(smaller["picks"]) == 2
    assert smaller["lastError"] == "second run was blocked"
    assert smaller["lastAttemptAt"] == "2026-09-11T17:00:00Z"

    completed = refresh._record_feed_attempt(
        smaller,
        {"ok": True, "date": date_iso, "picks": [*first["picks"], {"pick": "Alabama ML", "date": date_iso}]},
        date_iso,
        "2026-09-11T18:00:00Z",
    )
    assert completed["ok"] is True
    assert completed["refreshStatus"] == "ok"
    assert len(completed["picks"]) == 3
    assert "lastError" not in completed


def test_retry_demotes_legacy_ok_football_bucket_with_missing_official_games():
    date_iso = "2026-09-11"
    old_partial = {
        "ok": True,
        "date": date_iso,
        "picks": [
            {"pick": "Louisville ML", "date": date_iso, "source": "Scores24CFB"},
            {"pick": "Stanford ML", "date": date_iso, "source": "Scores24CFB"},
        ],
        "meta": {"feed": "scores24_cfb", "officialMatchups": 3,
                 "expectedMatchups": 2, "matchedPicks": 2, "missingMatchups": []},
    }
    retried = refresh._record_feed_attempt(
        old_partial,
        {"ok": False, "date": date_iso, "picks": [], "error": "later run timed out"},
        date_iso,
        "2026-09-11T17:00:00Z",
    )
    assert retried["ok"] is False
    assert len(retried["picks"]) == 2
    assert retried["meta"]["expectedMatchups"] == 3
    assert retried["meta"]["matchedPicks"] == 2
    assert retried["lastError"] == "later run timed out"
    checkpoint_retry = refresh._record_feed_attempt(
        old_partial,
        {"ok": False, "date": date_iso, "picks": old_partial["picks"],
         "meta": {"checkpointedPicks": 2}, "error": "hard timeout"},
        date_iso,
        "2026-09-11T17:30:00Z",
    )
    assert checkpoint_retry["ok"] is False
    assert checkpoint_retry["meta"]["expectedMatchups"] == 3
    assert checkpoint_retry["meta"]["matchedPicks"] == 2
    prior_without_official_count = {
        **old_partial,
        "meta": {"feed": "scores24_cfb", "expectedMatchups": 2,
                 "matchedPicks": 2, "missingMatchups": []},
    }
    explicit_failure = refresh._record_feed_attempt(
        prior_without_official_count,
        {"ok": False, "date": date_iso, "picks": [], "error": "slate incomplete",
         "meta": {"feed": "scores24_cfb", "officialMatchups": 3,
                  "expectedMatchups": 3, "matchedPicks": 0,
                  "missingMatchups": ["Alabama @ Auburn"]}},
        date_iso,
        "2026-09-11T18:00:00Z",
    )
    assert explicit_failure["ok"] is False
    assert len(explicit_failure["picks"]) == 2
    assert explicit_failure["meta"]["expectedMatchups"] == 3
    assert explicit_failure["meta"]["matchedPicks"] == 2


def test_optional_timeout_salvages_checkpoint_instead_of_redating_yesterday(tmp_path):
    from scripts.scrapers import scores24_optional_publish as optional

    checkpoint_dir = tmp_path / "state"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "scores24-cfb-2026-09-11.json").write_text(
        json.dumps(
            {
                "sport": "cfb",
                "date": "2026-09-11",
                "picks": [
                    {
                        "source": "Scores24CFB",
                        "pick": "Louisville ML",
                        "date": "2026-09-11",
                        "away_team": "Villanova Wildcats",
                        "home_team": "Louisville Cardinals",
                    },
                    {
                        "source": "Scores24CFB",
                        "pick": "Stanford ML",
                        "date": "2026-09-11",
                        "away_team": "Miami Hurricanes",
                        "home_team": "Stanford Cardinal",
                    },
                    {
                        "source": "Scores24CFB",
                        "pick": "Ole Miss ML",
                        "date": "2026-09-11",
                        "away_team": "Kentucky Wildcats",
                        "home_team": "Ole Miss Rebels",
                    },
                    {
                        "source": "Scores24CFB",
                        "pick": "Alabama ML",
                        "date": "2026-09-11",
                        "away_team": "South Florida Bulls",
                        "home_team": "Alabama Crimson Tide",
                    },
                ],
            }
        )
    )
    yesterday = {
        "ok": True,
        "date": "2026-09-10",
        "updatedAt": "2026-09-10T19:12:03Z",
        "picks": [{"pick": "Miami Under", "date": "2026-09-10", "source": "Scores24CFB"}],
    }
    cache_path = tmp_path / "2026-09-11.json"
    cache_path.write_text(
        json.dumps(
            {
                "date": "2026-09-11",
                "models": {"scores24_mlb": {"ok": True, "date": "2026-09-11", "picks": [{"pick": "Cubs ML"}]}},
                "external_feeds": {
                    "scores24_mlb": {"ok": True, "date": "2026-09-11", "picks": [{"pick": "Cubs ML"}]},
                    "scores24_cfb": yesterday,
                },
            }
        )
    )

    bucket = optional.apply_optional_timeout_to_cache(
        cache_path,
        "scores24_cfb",
        "2026-09-11",
        180,
        checkpoint_dir=str(checkpoint_dir),
        now_iso="2026-09-11T12:00:00Z",
    )
    published = json.loads(cache_path.read_text())
    assert bucket["date"] == "2026-09-11"
    assert bucket["ok"] is False
    assert bucket["refreshStatus"] == "error"
    assert len(bucket["picks"]) == 4
    assert published["external_feeds"]["scores24_mlb"]["ok"] is True
    assert published["external_feeds"]["scores24_cfb"]["picks"][0]["pick"] == "Louisville ML"


def test_optional_timeout_without_checkpoint_keeps_yesterday_date(tmp_path):
    from scripts.scrapers import scores24_optional_publish as optional

    yesterday = {
        "ok": True,
        "date": "2026-09-10",
        "updatedAt": "2026-09-10T19:12:03Z",
        "picks": [{"pick": "Miami Under", "date": "2026-09-10", "source": "Scores24CFB"}],
    }
    cache_path = tmp_path / "2026-09-11.json"
    cache_path.write_text(
        json.dumps({"date": "2026-09-11", "models": {}, "external_feeds": {"scores24_cfb": yesterday}})
    )

    bucket = optional.apply_optional_timeout_to_cache(
        cache_path,
        "scores24_cfb",
        "2026-09-11",
        180,
        checkpoint_dir=str(tmp_path / "empty"),
        now_iso="2026-09-11T12:00:00Z",
    )
    assert bucket["date"] == "2026-09-10"
    assert bucket["picks"] == yesterday["picks"]
    assert bucket["lastAttemptDate"] == "2026-09-11"
    assert bucket["refreshStatus"] == "error"
    assert "timed out" in bucket["lastError"]


def test_optional_feed_hard_timeout_kills_child_and_returns_soft_fail(tmp_path):
    from scripts.scrapers import scores24_optional_publish as optional

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "refresh_external_feeds.py").write_text("import time\ntime.sleep(30)\n")
    cache_path = tmp_path / "2026-09-11.json"
    cache_path.write_text(json.dumps({"date": "2026-09-11", "models": {}, "external_feeds": {}}))

    started = time.monotonic()
    rc = optional.run_optional_scores24_feed(
        python_bin=sys.executable,
        repo=str(repo),
        date_iso="2026-09-11",
        feed_key="scores24_cfb",
        sports="cfb",
        timeout_seconds=0.5,
        cache_path=str(cache_path),
        checkpoint_dir=str(tmp_path / "state"),
    )
    elapsed = time.monotonic() - started
    assert rc == 0
    assert elapsed < 8
    bucket = json.loads(cache_path.read_text())["external_feeds"]["scores24_cfb"]
    assert bucket["refreshStatus"] == "error"
    assert bucket["lastAttemptDate"] == "2026-09-11"
    assert bucket["ok"] is False
    assert "timed out" in bucket["lastError"]


def test_optional_nfl_budget_keeps_checkpoint_resume_and_outer_kill_switch(monkeypatch, tmp_path):
    from scripts.scrapers import scores24_optional_publish as optional

    observed = {}

    class FinishedProcess:
        def wait(self, timeout):
            observed["outer_timeout"] = timeout
            return 0

    def start(_cmd, *, env, start_new_session):
        observed["env"] = env
        observed["new_session"] = start_new_session
        return FinishedProcess()

    monkeypatch.setattr(optional.subprocess, "Popen", start)
    assert optional.run_optional_scores24_feed(
        python_bin=sys.executable,
        repo=str(tmp_path),
        date_iso="2026-09-27",
        feed_key="scores24_nfl",
        sports="nfl",
        timeout_seconds=420,
        cache_path=str(tmp_path / "cache.json"),
        checkpoint_dir=str(tmp_path / "state"),
    ) == 0
    assert observed["outer_timeout"] == 420
    assert observed["new_session"] is True
    assert observed["env"]["SCORES24_SCRAPE_TIMEOUT_SECONDS"] == "400.0"
    assert observed["env"]["SCORES24_CHECKPOINT_DIR"] == str(tmp_path / "state")


def test_optional_nonzero_exit_salvages_same_day_checkpoint(tmp_path):
    from scripts.scrapers import scores24_optional_publish as optional

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "refresh_external_feeds.py").write_text("raise SystemExit(1)\n")
    state = tmp_path / "state"
    state.mkdir()
    (state / "scores24-nfl-2026-09-11.json").write_text(json.dumps({
        "sport": "nfl", "date": "2026-09-11",
        "picks": [{"source": "Scores24NFL", "date": "2026-09-11", "pick": "Bears ML"}],
    }))
    cache_path = tmp_path / "2026-09-11.json"
    cache_path.write_text(json.dumps({
        "date": "2026-09-11", "models": {},
        "external_feeds": {"scores24_nfl": {
            "ok": False, "date": "2026-09-10",
            "picks": [{"source": "Scores24NFL", "date": "2026-09-10", "pick": "Chiefs ML"}],
        }},
    }))

    rc = optional.run_optional_scores24_feed(
        python_bin=sys.executable, repo=str(repo), date_iso="2026-09-11",
        feed_key="scores24_nfl", sports="nfl", timeout_seconds=2,
        cache_path=str(cache_path), checkpoint_dir=str(state),
    )
    bucket = json.loads(cache_path.read_text())["external_feeds"]["scores24_nfl"]
    # Optional feeds soft-fail: nonzero child exit still returns 0 after salvage.
    assert rc == 0
    assert bucket["ok"] is False
    assert bucket["date"] == "2026-09-11"
    assert bucket["picks"][0]["pick"] == "Bears ML"
    assert bucket["lastAttemptDate"] == "2026-09-11"


def test_optional_nonzero_exit_keeps_child_written_timeout_bucket(tmp_path):
    """Child soft-timeout already wrote today; outer salvage must not clobber it."""
    from scripts.scrapers import scores24_optional_publish as optional

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "refresh_external_feeds.py").write_text("raise SystemExit(1)\n")
    cache_path = tmp_path / "2026-09-30.json"
    child_bucket = {
        "ok": False,
        "date": "2026-09-30",
        "error": "Scores24Tennis stopped before finishing the 2026-09-30 slate (timeout); 2 pick(s) checkpointed.",
        "lastAttemptDate": "2026-09-30",
        "lastError": "Scores24Tennis stopped before finishing the 2026-09-30 slate (timeout); 2 pick(s) checkpointed.",
        "picks": [
            {"source": "Scores24Tennis", "date": "2026-09-30", "pick": "Rune ML",
             "away_team": "A", "home_team": "B"},
            {"source": "Scores24Tennis", "date": "2026-09-30", "pick": "Cerundolo ML",
             "away_team": "C", "home_team": "D"},
        ],
        "meta": {
            "feed": "scores24_tennis",
            "officialMatchups": 45,
            "expectedMatchups": 45,
            "matchedPicks": 2,
            "checkpointedPicks": 2,
            "timedOut": True,
            "unattemptedMatchups": ["X vs Y"],
        },
        "refreshStatus": "error",
    }
    cache_path.write_text(json.dumps({
        "date": "2026-09-30",
        "models": {"scores24_tennis": child_bucket},
        "external_feeds": {"scores24_tennis": child_bucket},
    }))

    rc = optional.run_optional_scores24_feed(
        python_bin=sys.executable, repo=str(repo), date_iso="2026-09-30",
        feed_key="scores24_tennis", sports="tennis", timeout_seconds=2,
        cache_path=str(cache_path), checkpoint_dir=str(tmp_path / "state"),
    )
    bucket = json.loads(cache_path.read_text())["external_feeds"]["scores24_tennis"]
    assert rc == 0
    assert bucket["ok"] is False
    assert len(bucket["picks"]) == 2
    assert bucket["meta"]["officialMatchups"] == 45
    assert bucket["meta"]["expectedMatchups"] == 45
    assert bucket["meta"]["timedOut"] is True
    assert "exited with code" not in str(bucket.get("lastError") or "")
    assert "timeout" in str(bucket.get("lastError") or "").lower() or "timeout" in str(bucket.get("error") or "").lower()


def test_optional_hard_timeout_kills_descendant_process_group(tmp_path):
    from scripts.scrapers import scores24_optional_publish as optional

    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    marker = tmp_path / "orphan-wrote.txt"
    spawned = tmp_path / "spawned.txt"
    child_code = (
        "import pathlib,time; time.sleep(1.5); "
        f"pathlib.Path({str(marker)!r}).write_text('orphan')"
    )
    (repo / "scripts" / "refresh_external_feeds.py").write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"open({str(spawned)!r}, 'w').write('yes')\n"
        "time.sleep(30)\n"
    )
    cache_path = tmp_path / "2026-09-11.json"
    cache_path.write_text(json.dumps({"date": "2026-09-11", "models": {}, "external_feeds": {}}))

    assert optional.run_optional_scores24_feed(
        python_bin=sys.executable, repo=str(repo), date_iso="2026-09-11",
        feed_key="scores24_cfb", sports="cfb", timeout_seconds=0.8,
        cache_path=str(cache_path), checkpoint_dir=str(tmp_path / "state"),
    ) == 0
    assert spawned.exists()
    time.sleep(1.6)
    assert not marker.exists()


def test_optional_timeout_salvage_does_not_leak_checkpoint_env(monkeypatch, tmp_path):
    monkeypatch.delenv("SCORES24_CHECKPOINT_DIR", raising=False)
    from scripts.scrapers import scores24_optional_publish as optional

    checkpoint_dir = tmp_path / "state"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "scores24-cfb-2026-09-11.json").write_text(
        json.dumps(
            {
                "sport": "cfb",
                "date": "2026-09-11",
                "picks": [
                    {
                        "source": "Scores24CFB",
                        "pick": "Louisville ML",
                        "date": "2026-09-11",
                    }
                ],
            }
        )
    )
    cache_path = tmp_path / "2026-09-11.json"
    cache_path.write_text(json.dumps({"date": "2026-09-11", "models": {}, "external_feeds": {}}))

    optional.apply_optional_timeout_to_cache(
        cache_path,
        "scores24_cfb",
        "2026-09-11",
        180,
        checkpoint_dir=str(checkpoint_dir),
        now_iso="2026-09-11T12:00:00Z",
    )
    assert os.environ.get("SCORES24_CHECKPOINT_DIR") in {None, ""}
    published = json.loads(cache_path.read_text())
    assert len(published["external_feeds"]["scores24_cfb"]["picks"]) == 1


def test_refresh_recovers_and_clears_previous_source_error(monkeypatch, tmp_path):
    _write_previous(tmp_path, {
        "ok": True, "date": "2026-09-06", "picks": [],
        "refreshStatus": "error", "lastError": "Previous source outage",
    })
    previous = json.loads((tmp_path / "latest.json").read_text())
    previous["external_feed_errors"] = ["forebet_mlb: Previous source outage"]
    (tmp_path / "latest.json").write_text(json.dumps(previous))
    _configure(monkeypatch, tmp_path, {"forebet_mlb": lambda *_args: {"ok": True, "picks": []}})

    assert refresh.main() == 0

    published = json.loads((tmp_path / "latest.json").read_text())
    assert published["external_feed_errors"] == []
    bucket = published["external_feeds"]["forebet_mlb"]
    assert bucket["refreshStatus"] == "ok"
    assert bucket["lastSuccessAt"] == bucket["lastAttemptAt"]
    assert "lastError" not in bucket


def test_refresh_exposes_failed_source_that_has_never_published(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, {"forebet_mlb": lambda *_args: {"ok": False, "error": "Timed out"}})

    assert refresh.main() == 1

    published = json.loads((tmp_path / "latest.json").read_text())
    for bucket in (
        published["external_feeds"]["forebet_mlb"],
        published["models"]["forebet_mlb"],
        published["forebet_mlb"],
    ):
        assert bucket["ok"] is False
        assert bucket["picks"] == []
        assert bucket["refreshStatus"] == "error"
        assert bucket["lastError"] == "Timed out"


def test_refresh_reports_sport_failure_while_publishing_valid_other_sport(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path, {
        "sportytrader": lambda *_args: {
            "ok": True,
            "picks": [{"sport": "MLB", "pick": "Cubs ML", "decision": "BET", "units": 1}],
            "errors": ["cfb: Official CFB slate unavailable"],
            "meta": {"sportErrors": {"cfb": "Official CFB slate unavailable"}},
        },
    })

    assert refresh.main() == 0

    published = json.loads((tmp_path / "latest.json").read_text())
    assert published["external_feed_errors"] == ["sportytrader_cfb: Official CFB slate unavailable"]
    assert published["sportytrader_cfb"]["refreshStatus"] == "error"
    mlb = published["sportytrader_mlb"]
    assert mlb["refreshStatus"] == "ok"
    assert mlb["errors"] == []
    assert published["sportytrader_cfb"]["errors"] == ["Official CFB slate unavailable"]
    assert mlb["picks"][0]["pick"] == "Cubs ML"
    assert mlb["picks"][0]["decision"] == "PASS"
    assert mlb["picks"][0]["units"] == 0


def test_non_nba_split_source_retry_marks_retained_cfb_snapshot_degraded(monkeypatch, tmp_path):
    from scripts.source_health import source_issues

    date = "2026-09-06"
    prior = {
        "ok": True, "date": date, "updatedAt": "2026-09-06T14:00:00Z",
        "picks": [{"pick": "Auburn ML", "sport": "CFB", "date": date}],
        "refreshStatus": "ok",
    }
    (tmp_path / "latest.json").write_text(json.dumps({
        "date": date, "models": {}, "external_feeds": {"sportytrader_cfb": prior},
    }))
    _configure(monkeypatch, tmp_path, {"sportytrader": lambda *_args: {
        "ok": True, "picks": [{"sport": "MLB", "pick": "Cubs ML", "date": date}],
        "meta": {"sportErrors": {"cfb": "CFB listing unavailable"}},
    }})

    assert refresh.main() == 0
    published = json.loads((tmp_path / "latest.json").read_text())
    bucket = published["external_feeds"]["sportytrader_cfb"]
    assert bucket["picks"] == prior["picks"]
    assert bucket["ok"] is False
    assert bucket["refreshStatus"] == "error"
    assert bucket["lastSuccessAt"] == prior["updatedAt"]
    assert bucket["lastError"] == "CFB listing unavailable"
    assert source_issues("sportytrader_cfb", bucket, date)
    assert published["external_feeds"]["sportytrader_mlb"]["refreshStatus"] == "ok"


def test_workflow_publishes_diagnostics_before_marking_total_outage_failed():
    workflow = (refresh.REPO_ROOT / ".github/workflows/external-feed-refresh.yml").read_text()
    assert "id: refresh-feeds\n" in workflow
    assert "continue-on-error: true" in workflow
    assert workflow.index("Report refresh failure after publishing diagnostics") > workflow.index("Deploy updated external feeds")
    assert "if: steps.refresh-feeds.outcome == 'failure'" in workflow
