from __future__ import annotations

from scripts.forebet_retry import FOREBET_KEYS, retryable_forebet_keys


def _complete(date: str, official: int = 0) -> dict:
    return {"ok": True, "date": date, "refreshStatus": "ok", "picks": [],
            "meta": {"officialMatchups": official, "matchedPicks": official,
                     "missingMatchups": [], "blockedUrls": 0}}


def test_retry_selector_is_sport_scoped_and_accepts_verified_off_day():
    date = "2026-09-30"
    feeds = {key: _complete(date) for key in FOREBET_KEYS}
    feeds["forebet_mlb"] = {**_complete(date, 2), "meta": {
        "officialMatchups": 2, "matchedPicks": 1,
        "missingMatchups": ["Missing official matchup"], "blockedUrls": 0}}
    feeds["forebet_cfb"] = {**_complete(date), "refreshStatus": "error",
                            "lastAttemptDate": date, "lastError": "Cloudflare"}
    feeds["forebet_nfl"] = _complete("2026-09-29")

    assert retryable_forebet_keys({"external_feeds": feeds}, date) == [
        "forebet_mlb", "forebet_cfb", "forebet_nfl",
    ]


def test_retry_selector_requires_a_certified_denominator_and_missing_bucket():
    date = "2026-09-30"
    feeds = {key: _complete(date) for key in FOREBET_KEYS if key != "forebet_mls"}
    feeds["forebet_wnba"]["meta"].pop("officialMatchups")

    assert retryable_forebet_keys({"external_feeds": feeds}, date) == [
        "forebet_mls", "forebet_wnba",
    ]


def test_forebet_retry_workflow_redispatches_after_active_refresh():
    from pathlib import Path

    workflow = (Path(__file__).resolve().parents[2] / ".github/workflows/forebet-retry.yml").read_text(encoding="utf-8")
    assert "workflow_run:" in workflow
    assert 'workflows: ["External Feed Refresh", "Daily Refresh"]' in workflow
    assert "github.event.workflow_run.name == 'Daily Refresh'" in workflow
    assert '.workflowName == "Daily Refresh"' in workflow
    assert "github.event.workflow_run.event != 'workflow_dispatch'" in workflow
    assert "timeout-minutes: 35" in workflow
    assert "gh run watch" in workflow
    assert "--exit-status" in workflow
    assert workflow.count('python scripts/forebet_retry.py --date "$TARGET_DATE"') >= 2
    assert "git fetch origin main" in workflow
    assert "git reset --hard origin/main" in workflow
    assert 'gh workflow run external-feed-refresh.yml --ref main -f date="$TARGET_DATE" -f feeds="$FEEDS"' in workflow
    assert "retry remains due" not in workflow
    assert "::error::External feed refresh is still active after waiting" in workflow
    assert "exit 1" in workflow
