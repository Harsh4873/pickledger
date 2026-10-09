#!/usr/bin/env python3
"""Preserve computed outcome rows and concurrent grades during publication."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.outcome_ledger_store import (  # noqa: E402
    load_outcome_ledger,
    merge_outcome_ledgers,
    write_outcome_ledger,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path, help="Saved outcome ledger from before the reset")
    parser.add_argument("--ledger", type=Path, default=REPO_ROOT / "data/calibration/outcome_ledger.json")
    args = parser.parse_args()
    generated = load_outcome_ledger(path=args.generated)
    current = load_outcome_ledger(path=args.ledger)
    if generated is None or current is None:
        parser.error("Both current and saved outcome ledgers are required")
    merged = merge_outcome_ledgers(current, generated)
    changed = write_outcome_ledger(merged, path=args.ledger)
    print(json.dumps({**merged["summary"], "changed": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
