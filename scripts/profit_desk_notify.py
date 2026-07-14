#!/usr/bin/env python3
"""Send a webhook alert when new live Profit Desk picks are published.

The notify workflow runs whenever ``data/profit_desk/latest.json`` changes on
main.  It compares the pushed artifact with the previous commit's version and
alerts only for live picks that were not already announced for the same slate
date, so repeated cache refreshes never re-ping an unchanged card.

Without a configured webhook secret the script logs the message it would have
sent and exits successfully, so the workflow is safe to ship before any
destination exists.  Set the ``PROFIT_DESK_WEBHOOK_URL`` repository secret to
a Discord-compatible webhook URL to turn alerts on.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Mapping

SITE_URL = "https://harsh.bet/pickledger/"
WEBHOOK_ENV_VARS = ("PROFIT_DESK_WEBHOOK_URL", "DISCORD_WEBHOOK_URL")


def _text(value: Any) -> str:
    return str(value or "").strip()


def _live_rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    portfolio = payload.get("portfolio")
    if not isinstance(portfolio, Mapping):
        return []
    return [row for row in portfolio.get("live") or [] if isinstance(row, dict)]


def _row_key(row: Mapping[str, Any]) -> str:
    return _text(row.get("id")) or "|".join(
        _text(row.get(field)) for field in ("sourceKey", "marketIdentity", "pick")
    )


def _format_odds(value: Any) -> str:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return "—"
    return f"+{number}" if number > 0 else str(number)


def build_notification(
    current: Mapping[str, Any] | None,
    previous: Mapping[str, Any] | None,
) -> str | None:
    """Return the alert text, or None when nothing new needs announcing."""

    date = _text((current or {}).get("date"))
    live = _live_rows(current)
    if not date or not live:
        return None
    previous_keys = (
        {_row_key(row) for row in _live_rows(previous)}
        if _text((previous or {}).get("date")) == date
        else set()
    )
    fresh = [row for row in live if _row_key(row) not in previous_keys]
    if not fresh:
        return None
    lines = [
        f"Profit Desk — {date}: {len(fresh)} new live pick{'s' if len(fresh) != 1 else ''}"
    ]
    for row in fresh:
        stake = row.get("stakeUnits")
        lane = _text(row.get("lane")).upper() or "LIVE"
        sport = _text(row.get("sport")) or "?"
        start = _text(row.get("startTime"))
        lines.append(
            f"• {_text(row.get('pick')) or 'Unnamed pick'} — {stake}u {lane} "
            f"at {_format_odds(row.get('oddsAmerican'))} ({sport}"
            + (f", starts {start}" if start else "")
            + ")"
        )
    lines.append(f"Card: {SITE_URL}")
    return "\n".join(lines)


def _send_webhook(url: str, message: str) -> bool:
    body = json.dumps({"content": message}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "profit-desk-notify"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return 200 <= response.status < 300
    except Exception as exc:  # pragma: no cover - network resilience
        print(f"[notify] webhook delivery failed: {exc}")
        return False


def _read_json(path: str) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, help="Path to the pushed latest.json.")
    parser.add_argument(
        "--previous",
        default="",
        help="Path to the previous commit's latest.json ('null' file or missing is fine).",
    )
    args = parser.parse_args()

    current = _read_json(args.current)
    previous = _read_json(args.previous) if args.previous else None
    message = build_notification(current, previous)
    if message is None:
        print("[notify] no new live picks; nothing to send")
        return 0
    print(f"[notify] message:\n{message}")
    webhook = next((os.environ.get(name) for name in WEBHOOK_ENV_VARS if os.environ.get(name)), "")
    if not webhook:
        print(
            "[notify] no webhook configured; set the PROFIT_DESK_WEBHOOK_URL "
            "repository secret to enable alerts"
        )
        return 0
    delivered = _send_webhook(webhook, message)
    print(f"[notify] delivered={delivered}")
    # Never fail the workflow over a notification: the artifact publish that
    # triggered this run is already complete.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
