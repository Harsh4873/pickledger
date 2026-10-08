#!/usr/bin/env python3
"""Run the unchanged NBA entry point with bounded stats.nba.com transport."""

from __future__ import annotations

import runpy
import math
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests


FAILURE_MARKER = "NBA_FETCH_UNAVAILABLE:"
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 15.0
MAX_ATTEMPTS = 2
RETRY_STATUSES = {429, 500, 502, 503, 504}


class NBAFetchUnavailable(SystemExit):
    """Stop this child before model-level exception handlers retry or fill gaps."""


def _positive_setting(name: str, default, parse):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = parse(value)
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    except (ValueError, OverflowError):
        pass
    print(f"[nba-fetch] Invalid {name}={value!r}; using {default}", file=sys.stderr, flush=True)
    return default


def _bounded_timeout(value: object, connect_timeout: float = CONNECT_TIMEOUT,
                     read_timeout: float = READ_TIMEOUT) -> tuple[float, float]:
    connect, read = value if isinstance(value, tuple) and len(value) == 2 else (value, value)

    def bound(part: object, limit: float) -> float:
        return min(float(part), limit) if isinstance(part, (int, float)) and part > 0 else limit

    return bound(connect, connect_timeout), bound(read, read_timeout)


class BoundedNBAStatsSession(requests.Session):
    """Keep NBA request headers/parameters and successful response bytes intact."""

    def __init__(self):
        super().__init__()
        ci = os.environ.get("CI", "").strip().lower() not in {"", "0", "false", "no"}
        self.connect_timeout = _positive_setting("PICKLEDGER_NBA_HTTP_CONNECT_TIMEOUT", CONNECT_TIMEOUT, float)
        # Keep nba_api's 30s read allowance locally; hosted runners fail faster.
        self.read_timeout = _positive_setting("PICKLEDGER_NBA_HTTP_READ_TIMEOUT", READ_TIMEOUT if ci else 30.0, float)
        self.max_attempts = _positive_setting("PICKLEDGER_NBA_HTTP_ATTEMPTS", MAX_ATTEMPTS, int)

    def request(self, method: str, url: str, **kwargs) -> requests.Response:
        endpoint = urlsplit(url)
        if endpoint.hostname != "stats.nba.com" or method.upper() != "GET":
            return super().request(method, url, **kwargs)
        kwargs["timeout"] = _bounded_timeout(kwargs.get("timeout"), self.connect_timeout, self.read_timeout)
        label = f"{endpoint.hostname}{endpoint.path}"
        for attempt in range(1, self.max_attempts + 1):
            started = time.monotonic()
            print(f"[nba-fetch] {label} attempt {attempt}/{self.max_attempts}", file=sys.stderr, flush=True)
            response = None
            try:
                response = super().request(method, url, **kwargs)
                response.raise_for_status()
                # HTML block pages sometimes arrive with HTTP 200. Do not let
                # those trigger the model's per-team retry/fallback loops.
                response.json()
            except (requests.RequestException, ValueError) as exc:
                retryable = response is None or response.status_code in RETRY_STATUSES
                reason = f"HTTP {response.status_code}" if response is not None else type(exc).__name__
                if response is not None and response.ok:
                    reason = f"invalid JSON (HTTP {response.status_code})"
                print(f"[nba-fetch] {label} {reason} after {time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)
                if response is not None:
                    response.close()
                if retryable and attempt < self.max_attempts:
                    time.sleep(1)
                    continue
                # SystemExit intentionally crosses broad `except Exception`
                # blocks in the frozen model. No partial predictions are used.
                raise NBAFetchUnavailable(f"{FAILURE_MARKER} {label} {reason}; stopped after {attempt} attempt(s)") from None
            print(f"[nba-fetch] {label} HTTP {response.status_code} in {time.monotonic() - started:.1f}s", file=sys.stderr, flush=True)
            return response
        raise AssertionError("NBA request attempts exhausted without a result")


def main() -> int:
    from nba_api.stats.library.http import NBAStatsHTTP

    model_dir = Path(__file__).resolve().parents[1] / "NBAPredictionModel"
    entrypoint = model_dir / "run_live.py"
    sys.path.insert(0, str(model_dir))
    sys.argv[0] = str(entrypoint)
    # nba_api exposes this session hook. Only this isolated model child uses it.
    with BoundedNBAStatsSession() as session:
        NBAStatsHTTP.set_session(session)
        try:
            runpy.run_path(str(entrypoint), run_name="__main__")
        except NBAFetchUnavailable as exc:
            print(str(exc), file=sys.stderr, flush=True)
            return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
