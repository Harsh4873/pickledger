#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipl.ipl_model import run_ipl_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IPL model and emit JSON only.")
    parser.add_argument("--team1", default="")
    parser.add_argument("--team2", default="")
    parser.add_argument("--venue", default="")
    parser.add_argument("--toss-winner", default="")
    parser.add_argument("--toss-decision", default="")
    parser.add_argument("--db-path", default="")
    return parser.parse_args()


def _clean(value: str) -> str | None:
    text = str(value or "").strip()
    return text or None


def main() -> int:
    args = parse_args()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_ipl_model(
                team1=_clean(args.team1),
                team2=_clean(args.team2),
                venue=_clean(args.venue),
                toss_winner=_clean(args.toss_winner),
                toss_decision=_clean(args.toss_decision),
                db_path=_clean(args.db_path),
            )
        print(json.dumps(result))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
