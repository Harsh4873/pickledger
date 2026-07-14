#!/usr/bin/env python3
"""Rebuild the deduplicated universal pick outcome ledger."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.pick_calibration import rebuild_outcome_ledger  # noqa: E402


def main() -> int:
    ledger, changed = rebuild_outcome_ledger()
    print(json.dumps({**ledger["summary"], "changed": changed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
