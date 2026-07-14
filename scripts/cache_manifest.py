"""Maintain the static model-cache manifest consumed by GitHub Pages."""

from __future__ import annotations

import json
from pathlib import Path


def write_cache_manifest(cache_dir: Path) -> None:
    files = sorted(path.name for path in cache_dir.glob("20??-??-??.json"))
    payload = {"files": files}
    (cache_dir / "index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
