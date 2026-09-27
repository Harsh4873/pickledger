#!/usr/bin/env python3
"""Offline Camoufox readiness check and bounded Scores24 browser warmup."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


def camoufox_enabled() -> bool:
    return os.environ.get("SCORES24_CAMOUFOX_FALLBACK", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }


def camoufox_profile_path() -> Path:
    configured = os.environ.get("SCORES24_CAMOUFOX_PROFILE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    state_root = os.environ.get("SCORES24_STATE_ROOT", "").strip()
    root = (
        Path(state_root).expanduser()
        if state_root
        else Path.home() / ".cache" / "pickledger-scores24"
    )
    return root / "camoufox-profile"


def camoufox_binary() -> tuple[str | None, str | None]:
    """Resolve an installed browser without Camoufox's automatic download."""
    try:
        from camoufox.pkgman import camoufox_path, launch_path

        browser_dir = camoufox_path(download_if_missing=False)
        binary = Path(launch_path(browser_dir))
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise FileNotFoundError(f"Camoufox executable is unavailable at {binary}")
        return str(binary), None
    except Exception as exc:
        return None, f"{exc}. Install the browser with {sys.executable} -m camoufox fetch"


def _warmup_child() -> int:
    binary, reason = camoufox_binary()
    if binary is None:
        print(f"Scores24 Camoufox warmup skipped: {reason}", file=sys.stderr)
        return 1

    from camoufox.sync_api import Camoufox

    options = {"headless": True, "humanize": True, "executable_path": binary}
    profile = camoufox_profile_path()
    try:
        profile.mkdir(parents=True, exist_ok=True)
        with Camoufox(persistent_context=True, user_data_dir=str(profile), **options) as context:
            page = context.new_page()
            try:
                page.goto("about:blank", timeout=5000)
            finally:
                page.close()
        return 0
    except Exception as persistent_error:
        try:
            with Camoufox(**options) as browser:
                context = browser.new_context(
                    locale="en-US", timezone_id="America/Chicago", no_viewport=True
                )
                page = context.new_page()
                try:
                    page.goto("about:blank", timeout=5000)
                finally:
                    page.close()
                    context.close()
            return 0
        except Exception as fallback_error:
            print(
                "Scores24 Camoufox warmup skipped: "
                f"persistent launch failed ({persistent_error}); "
                f"temporary launch failed ({fallback_error})",
                file=sys.stderr,
            )
            return 1


def warmup(timeout_seconds: float) -> int:
    if not camoufox_enabled():
        print("Scores24 Camoufox warmup skipped: SCORES24_CAMOUFOX_FALLBACK is disabled.")
        return 0
    binary, reason = camoufox_binary()
    if binary is None:
        print(f"Scores24 Camoufox warmup skipped: {reason}", file=sys.stderr)
        return 1
    try:
        proc = subprocess.Popen(
            [sys.executable, __file__, "child"],
            start_new_session=True,
        )
    except OSError as exc:
        print(f"Scores24 Camoufox warmup skipped: could not start browser check: {exc}", file=sys.stderr)
        return 1
    try:
        result = proc.wait(timeout=max(1.0, timeout_seconds))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        print(
            f"Scores24 Camoufox warmup timed out after {timeout_seconds:g}s; using curl_cffi.",
            file=sys.stderr,
        )
        return 1
    if result == 0:
        print("Scores24 Camoufox warmup complete.")
    return int(result or 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("warmup", "child"))
    args = parser.parse_args()
    if args.command == "child":
        return _warmup_child()
    try:
        timeout = float(os.environ.get("SCORES24_CAMOUFOX_WARMUP_TIMEOUT_SECONDS", "20"))
    except ValueError:
        timeout = 20.0
    return warmup(timeout)


if __name__ == "__main__":
    raise SystemExit(main())
