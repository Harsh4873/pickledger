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
