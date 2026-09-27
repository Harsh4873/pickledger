#!/usr/bin/env python3
"""Run a soft-fail Scores24 feed with a hard timeout and checkpoint salvage.

Used by the Scores24 and tennis publishers for optional feeds. A Camoufox
request may hang; this wrapper kills the process group and promotes any
same-day checkpointed picks without delaying other feeds beyond the timeout.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.merge_external_feed_cache_payload import _demote_scraped_feed_picks  # noqa: E402
from scripts.refresh_external_feeds import (  # noqa: E402
    _previous_feed_bucket,
    _record_feed_attempt,
    _today_result_picks,
)
from scripts.scrapers.scores24_scraper import (  # noqa: E402
    load_checkpoint_picks,
    sport_key_for_feed,
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    staged = path.with_name(path.name + ".tmp")
    staged.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    staged.replace(path)


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def timeout_result(
    feed_key: str,
    date_iso: str,
    timeout_seconds: float,
    picks: list[dict[str, Any]],
) -> dict[str, Any]:
    return failure_result(
        feed_key,
        date_iso,
        f"Optional source refresh timed out after {timeout_seconds:.0f}s with {len(picks)} matched pick(s)",
        picks,
        timed_out=True,
    )


def failure_result(
    feed_key: str,
    date_iso: str,
    error: str,
    picks: list[dict[str, Any]],
    *,
    timed_out: bool = False,
) -> dict[str, Any]:
    return {
        "ok": False,
        "date": date_iso,
        "picks": picks,
        "error": error,
        "meta": {
            "date": date_iso,
            "feed": feed_key,
            "matchedPicks": len(picks),
            "checkpointedPicks": len(picks),
            **({"timedOut": True} if timed_out else {}),
        },
    }


def apply_optional_failure_to_cache(
    cache_path: Path,
    feed_key: str,
    date_iso: str,
    error: str,
    *,
    checkpoint_dir: str | None = None,
    now_iso: str | None = None,
    timed_out: bool = False,
) -> dict[str, Any]:
    """Record an optional failure without redatestamping yesterday's rows.

    If a same-day checkpoint has matched picks, those become today's (incomplete)
    bucket. Otherwise lastAttemptDate moves to today and the previous snapshot
    keeps its own date.
    """
    payload = _read_json(cache_path) or {"date": date_iso, "models": {}, "external_feeds": {}}
    previous = _previous_feed_bucket(payload, feed_key)
    sport_key = sport_key_for_feed(feed_key)
    checkpoint_picks = (
        load_checkpoint_picks(sport_key, date_iso, checkpoint_dir=checkpoint_dir)
        if sport_key
        else []
    )
    previous_picks = (
        _today_result_picks(previous, date_iso)
        if isinstance(previous, dict) and previous.get("date") == date_iso
        else []
    )
    matched_count = max(len(checkpoint_picks), len(previous_picks))
    if timed_out:
        error = f"{error} with {matched_count} matched pick(s)"
    if (
        isinstance(previous, dict)
        and previous.get("date") == date_iso
        and len(previous_picks) >= len(checkpoint_picks)
    ):
        # The child may already have written a richer partial result. Keep its
        # official-slate metadata while recording this outer failure.
        checkpoint_picks = []
    result = failure_result(
        feed_key, date_iso, error, checkpoint_picks, timed_out=timed_out,
    )
    now = now_iso or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    bucket = _record_feed_attempt(previous, result, date_iso, now)
    feeds = payload.get("external_feeds") if isinstance(payload.get("external_feeds"), dict) else {}
    feeds = dict(feeds)
    feeds[feed_key] = bucket
    payload["external_feeds"] = feeds
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    models = dict(models)
    models[feed_key] = bucket
    payload["models"] = models
    payload[feed_key] = bucket
    # Soft-fail writes can bypass the later merge demotion when a publisher
    # commits the cache file directly; demote here so BET tips never look tracked.
    payload = _demote_scraped_feed_picks(payload)
    demoted = payload.get("models", {}).get(feed_key) if isinstance(payload.get("models"), dict) else None
    if isinstance(demoted, dict):
        bucket = demoted
        feeds = payload.get("external_feeds") if isinstance(payload.get("external_feeds"), dict) else feeds
        models = payload.get("models") if isinstance(payload.get("models"), dict) else models
    _write_json_atomic(cache_path, payload)
    return bucket


def apply_optional_timeout_to_cache(
    cache_path: Path,
    feed_key: str,
    date_iso: str,
    timeout_seconds: float,
    *,
    checkpoint_dir: str | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    return apply_optional_failure_to_cache(
        cache_path,
        feed_key,
        date_iso,
        f"Optional source refresh timed out after {timeout_seconds:.0f}s",
        checkpoint_dir=checkpoint_dir,
        now_iso=now_iso,
        timed_out=True,
    )


def run_optional_scores24_feed(
    *,
    python_bin: str,
    repo: str,
    date_iso: str,
    feed_key: str,
    sports: str,
    timeout_seconds: float,
    cache_path: str,
    checkpoint_dir: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> int:
    """Refresh one optional feed; never raise, never block the caller with hang."""
    timeout = max(1.0, float(timeout_seconds))
    inner_timeout = max(30.0, timeout - 20.0)
    cmd = [
        python_bin,
        os.path.join(repo, "scripts/refresh_external_feeds.py"),
        "--date",
        date_iso,
        "--feeds",
        feed_key,
        "--sports",
        sports,
        "--skip-firestore",
    ]
    env = os.environ.copy()
    env["SCORES24_SCRAPE_TIMEOUT_SECONDS"] = str(inner_timeout)
    if checkpoint_dir:
        env["SCORES24_CHECKPOINT_DIR"] = checkpoint_dir
    if extra_env:
        env.update(extra_env)

    try:
        proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    except OSError as exc:
        try:
            apply_optional_failure_to_cache(
                Path(cache_path), feed_key, date_iso,
                f"Optional source refresh could not start: {exc}",
                checkpoint_dir=checkpoint_dir,
            )
        except (OSError, ValueError) as cache_exc:
            print(f"Could not record optional launch failure: {cache_exc}", file=sys.stderr)
        print(f"Optional {feed_key} scrape could not start: {exc}", file=sys.stderr)
        return 1
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        try:
            apply_optional_timeout_to_cache(
                Path(cache_path),
                feed_key,
                date_iso,
                timeout,
                checkpoint_dir=checkpoint_dir,
            )
        except (OSError, ValueError) as exc:
            print(f"Could not record optional timeout diagnostics: {exc}", file=sys.stderr)
        print(
            f"Optional {feed_key} scrape timed out after {timeout:.0f}s; "
            "continuing with other feeds.",
            file=sys.stderr,
        )
        return 0
    if returncode:
        try:
            apply_optional_failure_to_cache(
                Path(cache_path), feed_key, date_iso,
                f"Optional source refresh exited with code {returncode}",
                checkpoint_dir=checkpoint_dir,
            )
        except (OSError, ValueError) as exc:
            print(f"Could not record optional failure diagnostics: {exc}", file=sys.stderr)
    return int(returncode or 0)


def main() -> int:
    return run_optional_scores24_feed(
        python_bin=os.environ["PYTHON_BIN"],
        repo=os.environ["TEMP_REPO"],
        date_iso=os.environ["DATE_ISO"],
        feed_key=os.environ["OPTIONAL_FEED_KEY"],
        sports=os.environ["PUBLISH_SPORTS"],
        timeout_seconds=float(os.environ.get("OPTIONAL_FEED_TIMEOUT") or "180"),
        cache_path=os.environ["SCORES24_CACHE_FILE"],
        checkpoint_dir=os.environ.get("SCORES24_CHECKPOINT_DIR") or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
