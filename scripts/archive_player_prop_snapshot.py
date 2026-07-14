#!/usr/bin/env python3
"""Archive an immutable copy of every published player-prop snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "player_props_cache" / "latest.json"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "player_props_snapshots"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected an object in {path}")
    return payload


def archive_snapshot(payload: dict[str, Any], output_dir: Path) -> Path:
    date_iso = str(payload.get("date") or "").strip()
    generated_at = str(payload.get("generatedAt") or payload.get("updatedAt") or "").strip()
    if not date_iso or not generated_at:
        raise SystemExit("Player-prop snapshot requires date and generatedAt")
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:12]
    stamp = generated_at.replace(":", "-").replace(".", "-").replace("+", "_")
    target = output_dir / date_iso / f"{stamp}_{digest}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    target = archive_snapshot(_read(args.input), args.output_dir)
    print(f"[player-props] archived immutable snapshot {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
