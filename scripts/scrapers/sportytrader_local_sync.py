#!/usr/bin/env python3
"""Local SportyTrader sync via macOS launchd — uses the same refresh+merge path as Actions.

Writes Pages cache through ``scripts.refresh_external_feeds`` (retain/demote
included). Does **not** write an orphan ``sportytrader_manual_feed.json``.
Set ``SPORTYTRADER_DEBUG_ORPHAN=true`` to also dump that debug file after a
successful refresh.
"""

from __future__ import annotations

import json
import os
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _load_local_env() -> None:
    base_dir = Path(__file__).resolve().parent
    for filename in (".env", ".env.local"):
        path = base_dir / filename
        if not path.exists():
            continue
        try:
            for raw_line in path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


_load_local_env()

if not _env_flag("ENABLE_SPORTYTRADER_LOCALSYNC"):
    print("SportyTrader launchd sync disabled. Set ENABLE_SPORTYTRADER_LOCALSYNC=true to enable it.")
    raise SystemExit(0)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts import refresh_external_feeds as refresh  # noqa: E402


def main() -> int:
    date_str = datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    # Mirror Actions SportyTrader coverage (incl. CFB/NFL soft sports).
    sports = "nba,nba_summer,mlb,wnba,fifa_world_cup,cfb,nfl"

    args = Namespace(date=date_str, feeds="sportytrader", sports=sports, skip_firestore=True)
    original_parse_args = refresh._parse_args
    try:
        refresh._parse_args = lambda: args
        code = refresh.main()
    finally:
        refresh._parse_args = original_parse_args
    if code != 0:
        print("SportyTrader local sync: refresh_external_feeds failed", file=sys.stderr)
        return code

    if _env_flag("SPORTYTRADER_DEBUG_ORPHAN"):
        # Debug-only dump — not the Pages publish contract.
        import pickgrader_server as p

        result = p.run_sportytrader_scraper(
            date_str,
            ["nba", "nba_summer", "mlb", "wnba", "fifa_world_cup", "cfb", "nfl"],
        )
        payload = {
            "updated_at": datetime.now().isoformat(),
            "date": date_str,
            "leagues": sports,
            "note": "Debug orphan only; Pages cache comes from refresh_external_feeds.",
            "picks": result.get("picks", []),
            "ok": result.get("ok"),
        }
        Path("sportytrader_manual_feed.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print("Wrote debug orphan sportytrader_manual_feed.json")
    print(f"SportyTrader local sync published via refresh_external_feeds for {date_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
