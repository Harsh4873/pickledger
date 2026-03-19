#!/usr/bin/env python3
"""Local sync script for Scores24 picks via macOS launchd."""

import json
from datetime import datetime
from pathlib import Path

import pickgrader_server as p

date_str = datetime.now().strftime("%Y-%m-%d")
result = p.run_scores24_scraper(["nba", "nhl", "mlb"], date_str)
if not result.get("ok"):
    raise SystemExit(result.get("error") or "Scores24 sync failed")

payload = {
    "updated_at": datetime.now().isoformat(),
    "date": date_str,
    "note": "Auto-synced locally via macOS launchd.",
    "picks": result.get("picks", []),
}

Path("scores24_manual_feed.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(f"Synced {len(payload['picks'])} Scores24 pick(s) for {date_str}")
if result.get("errors"):
    print("Non-fatal sport errors:", "; ".join(result["errors"]))
