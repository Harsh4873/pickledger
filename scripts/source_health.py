"""Report source degradation independently of whether a cache was published."""
import argparse
import json
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_issues(key, bucket, day):
    issues = []
    meta = bucket.get("meta") or {}
    if key.startswith("forebet_"):
        has_attempt_meta = bucket.get("lastAttemptDate") == day and "lastAttemptMeta" in bucket
        attempt_meta = bucket.get("lastAttemptMeta") if has_attempt_meta else {}
        if not isinstance(attempt_meta, dict):
            attempt_meta = {}
        if bucket.get("refreshStatus") in {"error", "incomplete"}:
            issues.append(f"latest Forebet attempt failed: {bucket.get('lastError') or 'unknown error'}")
        if str(bucket.get("date") or "") != day:
            issues.append(f"Forebet snapshot dated {bucket.get('date') or 'unknown'}, expected {day}")
        blocked = attempt_meta.get("blockedUrls") if has_attempt_meta else meta.get("blockedUrls")
        if blocked:
            url = attempt_meta.get("blockedUrl") if has_attempt_meta else meta.get("blockedUrl")
            issues.append(f"Forebet listing blocked: {url or str(blocked) + ' URL(s)'}")
    if key == "tennistonic_tennis":
        if bucket.get("refreshStatus") == "error":
            issues.append(f"latest TennisTonic attempt failed: {bucket.get('lastError') or 'unknown error'}")
        if str(bucket.get("date") or "") != day:
            issues.append(f"TennisTonic snapshot dated {bucket.get('date') or 'unknown'}, expected {day}")
        if meta.get("unavailableMatchups") or meta.get("unattemptedMatchups"):
            issues.append("TennisTonic official slate has unavailable or unattempted matches")
    if key.startswith(("sportytrader_", "sportsgambler_")) and key.endswith(
        ("_mlb", "_wnba", "_cfb", "_nfl", "_fifa_world_cup")
    ):
        if bucket.get("refreshStatus") == "error":
            issues.append(f"latest provider attempt failed: {bucket.get('lastError') or 'unknown error'}")
    if bucket.get("ok") is False or bucket.get("errors") or bucket.get("error"):
        issues.append(str(bucket.get("error") or bucket.get("errors") or "refresh failed"))
    expected = meta.get("expectedMatchups", meta.get("officialMatchups"))
    if key.startswith("forebet_") and isinstance(meta.get("officialMatchups"), int):
        expected = meta["officialMatchups"]
    matched = meta.get("matchedPicks")
    if isinstance(expected, (float, int)) and isinstance(matched, (float, int)) and matched < expected:
        issues.append(f"partial provider coverage: {matched}/{expected}")
    if meta.get("missingMatchups"):
        issues.append(f"{len(meta['missingMatchups'])} expected matchups missing")
    if key == "tennis":
        through = meta.get("ratingsThrough")
        age = (date.fromisoformat(day) - date.fromisoformat(through)).days if through else None
        if age is None or age > 7:
            issues.append(f"ratings stale: through {through or 'unknown'}, age {age} days")
        if meta.get("catchUpErrors"):
            issues.append(f"{len(meta['catchUpErrors'])} result downloads failed")
        unknown, official = meta.get("unknownPlayers", 0), meta.get("officialMatchups", 0)
        if unknown:
            issues.append(f"{unknown} unrated players across {official} scheduled matches")
        archive = meta.get("archiveThrough")
        if archive and (date.fromisoformat(day) - date.fromisoformat(archive)).days > 7:
            issues.append(f"archive stale through {archive}; official-result fallback does not refresh all ranking/points inputs")
    if bucket.get("games", 0) and not bucket.get("picks"):
        issues.append(str(bucket.get("note") or "scheduled games with no published candidates"))
    diagnostics = bucket.get("primary_source_diagnostics", bucket.get("diagnostics", []))
    unpriced = sum(int(row.get("unpriced_market_rows") or 0) for row in diagnostics)
    if unpriced:
        issues.append(f"{unpriced} primary-feed market rows lack usable prices; excluded")
    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    args = parser.parse_args()
    lines = ["| Source | Rows | State | Detail |", "|---|---:|---|---|"]
    for directory in ("model_cache", "player_props_cache"):
        payload = json.loads((ROOT / "data" / directory / "latest.json").read_text())
        day = args.date or payload["date"]
        for key, bucket in sorted(payload.get("models", {}).items()):
            issues = source_issues(key, bucket, day)
            count = len(bucket.get("picks") or [])
            detail = "; ".join(issues) or str(bucket.get("note") or "refresh complete")
            lines.append(f"| {key} | {count} | {'degraded' if issues else 'available'} | {detail.replace('|', '/')} |")
            for issue in issues:
                print(f"::warning title=Source health: {key}::{issue.replace(chr(10), ' ')}")
    summary = "\n".join(lines) + "\n"
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as output:
            output.write(summary)


if __name__ == "__main__":
    main()
