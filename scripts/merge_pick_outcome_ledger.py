#!/usr/bin/env python3
"""Preserve computed outcome rows and concurrent grades during publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pick_calibration import write_json_if_changed  # noqa: E402


SETTLED_RESULTS = {"win", "loss", "push", "void", "cancelled"}


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
        for row in records
    ):
        raise ValueError("Outcome ledger requires records with stable string ids")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Outcome ledger contains duplicate ids")
    return records


def merge_outcome_ledgers(current: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    """Union whole rows by id; keep current settled rows, else take generated rows."""
    if current.get("schema_version") != generated.get("schema_version"):
        raise ValueError("Cannot merge different outcome ledger schemas")
    rows = {row["id"]: row for row in _records(current)}
    for incoming in _records(generated):
        existing = rows.get(incoming["id"])
        # A settled or retracted current row is never replaced. Otherwise the
        # run's freshly computed row wins, as the publication previously did.
        if existing is None or (
            existing.get("result") not in SETTLED_RESULTS
            and not existing.get("settlement_exclusion_reason")
        ):
            rows[incoming["id"]] = incoming
    records = sorted(rows.values(), key=lambda row: (
        str(row.get("date") or ""), str(row.get("model_key") or ""), row["id"],
    ))
    decided = [row for row in records if row.get("result") in {"win", "loss"}]
    return {
        **generated,
        **current,
        "records": records,
        "summary": {
            "total_picks": len(records),
            "decided_picks": len(decided),
            "trainable_decided_picks": sum(
                row.get("raw_probability") is not None and row.get("calibration_eligible") is True
                for row in decided
            ),
            "pending_picks": sum(row.get("result") in {"", "pending"} for row in records),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path, help="Saved outcome ledger from before the reset")
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / "data/calibration/outcome_ledger.json")
    args = parser.parse_args()
    generated = json.loads(args.generated.read_text())
    current = json.loads(args.ledger.read_text())
    merged = merge_outcome_ledgers(current, generated)
    changed = write_json_if_changed(args.ledger, merged)
    print(json.dumps({**merged["summary"], "changed": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
