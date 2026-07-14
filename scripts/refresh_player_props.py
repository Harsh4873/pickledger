#!/usr/bin/env python3
"""Generate isolated in-house player-props cache artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "player_props_cache"
sys.path.insert(0, str(REPO_ROOT))

from player_props import generate_payload  # noqa: E402
from scripts.merge_player_props_cache_payload import PUBLIC_PLAYER_PROP_MODEL_KEYS  # noqa: E402
from scripts.pick_calibration import apply_calibration_to_payload  # noqa: E402


def _default_central_date() -> str:
    return datetime.now(ZoneInfo("America/Chicago")).date().isoformat()


def _target_date(raw: str | None) -> str:
    value = str(raw or "").strip()
    return value or _default_central_date()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh direct-API player-props cache.")
    default_date = _default_central_date()
    parser.add_argument("--date", default=default_date, help="Target date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    temp.replace(path)


def main() -> int:
    args = _parse_args()
    target_date = _target_date(args.date)
    payload = apply_calibration_to_payload(generate_payload(target_date))
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    payload = {
        **payload,
        "models": {
            key: bucket
            for key, bucket in models.items()
            if key in PUBLIC_PLAYER_PROP_MODEL_KEYS
        },
    }
    output_dir = args.output_dir.resolve()
    _write_json(output_dir / f"{target_date}.json", payload)
    _write_json(output_dir / "latest.json", payload)
    files = sorted(path.name for path in output_dir.glob("20??-??-??.json"))
    _write_json(output_dir / "index.json", {"files": files})

    failed = False
    for model_name, model in payload["models"].items():
        picks = model.get("picks") or []
        ok = bool(model.get("ok"))
        failed = failed or not ok
        print(f"[player-props] {model_name}: {'ok' if ok else 'error'} ({len(picks)} pick(s))")
        for error in model.get("errors") or []:
            print(f"[player-props] {model_name} warning: {error}")
    print(f"[player-props] wrote {output_dir / f'{target_date}.json'}")
    print(f"[player-props] wrote {output_dir / 'latest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
