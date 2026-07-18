#!/usr/bin/env python3
"""Capture near-closing prices for imminent pregame picks.

The Profit Desk's CLV uses the last pregame price capture as its closing
observation, but the scheduled cache refreshes stop hours before night games.
This script runs on a tight cron: whenever a tracked pick starts within the
capture window it re-attaches current ESPN/DraftKings prices to the day's
model cache and journals the refreshed selected-side prices into an
append-only closing-lines ledger (``data/closing_lines/YYYY-MM-DD.json``)
that ``build_profit_desk`` prefers over the cache's own last capture.

Team markets are refreshed here; player props keep their generator-owned
capture cadence. When ``ODDS_API_KEY`` is set, sharp-book prices from The
Odds API are journaled next to the anchor rows as an optional alternative
closing baseline — they are enrichment only and nothing requires them.
The script exits quickly with no network traffic when nothing starts soon.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_profit_desk import (  # noqa: E402
    _closing_from_record,
    _parse_timestamp,
    canonical_market_identity,
)
from scripts.cache_manifest import write_cache_manifest  # noqa: E402
from scripts.market_odds import apply_market_odds_to_payload  # noqa: E402

MODEL_CACHE_DIR = REPO_ROOT / "data" / "model_cache"
CLOSING_LINES_DIR = REPO_ROOT / "data" / "closing_lines"
DEFAULT_WINDOW_MINUTES = 75


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _iter_picks(payload: dict[str, Any]):
    models = payload.get("models")
    if not isinstance(models, dict):
        return
    for bucket in models.values():
        if not isinstance(bucket, dict):
            continue
        for pick in bucket.get("picks") or []:
            if isinstance(pick, dict):
                yield pick


def _start_time(pick: dict[str, Any]) -> datetime | None:
    for field in ("game_start_time", "start_time", "event_start_time"):
        parsed = _parse_timestamp(pick.get(field))
        if parsed is not None:
            return parsed
    return None


def _imminent_picks(
    payload: dict[str, Any], now: datetime, window: timedelta
) -> list[dict[str, Any]]:
    imminent: list[dict[str, Any]] = []
    for pick in _iter_picks(payload):
        start = _start_time(pick)
        if start is None:
            continue
        if now <= start <= now + window:
            imminent.append(pick)
    return imminent


def _ledger_rows_for(
    picks: list[dict[str, Any]], date_iso: str, run_started_at: datetime
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pick in picks:
        captured = _parse_timestamp(pick.get("market_odds_captured_at"))
        if captured is None or captured < run_started_at:
            continue
        closing = _closing_from_record(pick)
        if closing is None:
            continue
        sport = str(pick.get("sport") or "").strip().upper()
        rows.append(
            {
                "marketIdentity": canonical_market_identity(
                    pick, mode="team", sport=sport, date_iso=date_iso
                ),
                "sport": sport,
                "matchup": str(pick.get("matchup") or pick.get("game") or ""),
                "pick": str(pick.get("pick") or ""),
                "startTime": str(
                    pick.get("game_start_time") or pick.get("start_time") or ""
                ),
                "role": "anchor",
                **closing,
            }
        )
    return rows


def _append_ledger(date_iso: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    path = CLOSING_LINES_DIR / f"{date_iso}.json"
    payload = _read_json(path) or {"date": date_iso, "rows": []}
    existing = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    seen = {
        (str(row.get("marketIdentity")), str(row.get("capturedAt")), str(row.get("provider")))
        for row in existing
        if isinstance(row, dict)
    }
    added = 0
    for row in rows:
        key = (str(row.get("marketIdentity")), str(row.get("capturedAt")), str(row.get("provider")))
        if key in seen:
            continue
        existing.append(row)
        seen.add(key)
        added += 1
    if added:
        payload["rows"] = existing
        payload["updatedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        _write_json(path, payload)
    return added


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture near-closing prices for imminent picks.")
    parser.add_argument("--date", default="", help="Slate date YYYY-MM-DD (default: today UTC).")
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=DEFAULT_WINDOW_MINUTES,
        help="Look-ahead window for imminent starts.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report imminent picks without fetching or writing."
    )
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    date_iso = args.date.strip() or now.date().isoformat()
    window = timedelta(minutes=max(5, args.window_minutes))

    cache_path = MODEL_CACHE_DIR / f"{date_iso}.json"
    payload = _read_json(cache_path)
    if payload is None:
        print(f"[closing-lines] no model cache for {date_iso}; nothing to do")
        return 0

    imminent = _imminent_picks(payload, now, window)
    if not imminent:
        print(f"[closing-lines] no picks starting within {window} of {now.isoformat()}")
        return 0
    print(f"[closing-lines] {len(imminent)} pick(s) start within {window}")
    if args.dry_run:
        return 0

    run_started_at = now
    stats = apply_market_odds_to_payload(payload)
    print(f"[closing-lines] market attach: {stats}")

    rows = _ledger_rows_for(imminent, date_iso, run_started_at)

    try:
        from scripts.odds_api import journal_sharp_rows

        rows.extend(journal_sharp_rows(imminent, date_iso, now=now))
    except Exception as exc:  # pragma: no cover - sharp lines are optional enrichment
        print(f"[closing-lines] sharp-line module skipped: {exc}")

    added = _append_ledger(date_iso, rows)
    print(f"[closing-lines] journaled {added} new closing row(s)")

    if added or stats.get("attached"):
        _write_json(cache_path, payload)
        latest = _read_json(MODEL_CACHE_DIR / "latest.json")
        if latest is not None and str(latest.get("date") or "") == date_iso:
            _write_json(MODEL_CACHE_DIR / "latest.json", payload)
        write_cache_manifest(MODEL_CACHE_DIR)
        print(f"[closing-lines] refreshed {cache_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
