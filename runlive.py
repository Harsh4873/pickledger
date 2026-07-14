#!/usr/bin/env python3
"""Run the live NBA variants, then append the WNBA branch."""

from __future__ import annotations

import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NBA_MODEL_DIR = os.path.join(BASE_DIR, "NBAPredictionModel")
PROJECT_PYTHON = os.path.join(BASE_DIR, ".venv", "bin", "python")

try:
    sys.path.insert(0, os.path.join(BASE_DIR, "WNBAPredictionModel"))
    from wnba_picks import generate_wnba_picks
    try:
        from config import RUN_WNBA as _CONFIG_RUN_WNBA
    except ImportError:
        _CONFIG_RUN_WNBA = True
    RUN_WNBA = os.environ.get("PICKLEDGER_RUN_WNBA", str(_CONFIG_RUN_WNBA)).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    WNBA_AVAILABLE = True
except ImportError as e:
    print(f"[WNBA] Module not available: {e}")
    WNBA_AVAILABLE = False
    RUN_WNBA = False


def _run_nba_variant(variant: str, passthrough_args: list[str]) -> int:
    python_bin = PROJECT_PYTHON if os.path.exists(PROJECT_PYTHON) else sys.executable
    cmd = [python_bin, "run_live.py", "--variant", variant, *passthrough_args]
    result = subprocess.run(
        cmd,
        cwd=NBA_MODEL_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    return result.returncode


def main() -> int:
    passthrough_args = sys.argv[1:]

    exit_codes = [
        _run_nba_variant("old", passthrough_args),
        _run_nba_variant("new", passthrough_args),
    ]

    # ── WNBA ──────────────────────────────────────────────────────────
    if RUN_WNBA and WNBA_AVAILABLE:
        print("\n[WNBA] Generating picks...")
        try:
            wnba_picks = generate_wnba_picks(echo=False)
            if wnba_picks:
                for pick in wnba_picks:
                    print(pick["output_line"])
            else:
                print("[WNBA] No picks generated today.")
        except Exception as e:
            print(f"[WNBA] Pick generation failed: {e}")
    elif not RUN_WNBA:
        print("[WNBA] Off-season — skipping (set RUN_WNBA=True to override).")

    return next((code for code in exit_codes if code), 0)


if __name__ == "__main__":
    raise SystemExit(main())
