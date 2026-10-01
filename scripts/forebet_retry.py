#!/usr/bin/env python3
"""Select Forebet sports whose current official slate still needs a retry."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
FOREBET_KEYS = ("forebet_mls", "forebet_mlb", "forebet_wnba", "forebet_cfb", "forebet_nfl")


def retryable_forebet_keys(payload: dict[str, Any], target_date: str) -> list[str]:
    feeds = payload.get("external_feeds") if isinstance(payload.get("external_feeds"), dict) else {}
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    retry: list[str] = []
    for key in FOREBET_KEYS:
        bucket = feeds[key] if key in feeds else models[key] if key in models else payload.get(key)
        if not isinstance(bucket, dict):
            retry.append(key)
            continue
        if str(bucket.get("date") or "") != target_date:
            retry.append(key)
            continue
        if bucket.get("ok") is not True or bucket.get("refreshStatus") in {"error", "incomplete"}:
            retry.append(key)
            continue
        meta = bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}
        official = meta.get("officialMatchups")
        matched = meta.get("matchedPicks")
        if (not isinstance(official, int) or isinstance(official, bool)
                or not isinstance(matched, int) or isinstance(matched, bool)
                or matched != official or meta.get("missingMatchups") or meta.get("blockedUrls")):
            retry.append(key)
    return retry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=datetime.now(ZoneInfo("America/Chicago")).date().isoformat())
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "data/model_cache/latest.json")
    args = parser.parse_args()
    try:
        payload = json.loads(args.cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    print(",".join(retryable_forebet_keys(payload if isinstance(payload, dict) else {}, args.date)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
