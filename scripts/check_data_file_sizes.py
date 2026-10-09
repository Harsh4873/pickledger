#!/usr/bin/env python3
"""Keep versioned JSON data below repository hosting limits without reading it."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 50_000_000
MAX_LEDGER_SHARD_BYTES = 10_000_000
LEDGER_DIRECTORIES = {
    Path("data/calibration/team_prop_pregame_ledger"),
    Path("data/calibration/outcome_ledger"),
}


def oversized_data_files(repo_root: Path = REPO_ROOT) -> list[tuple[str, int, int]]:
    """Check tracked and new nonignored files, excluding worktree deletions."""
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "data/"],
        cwd=repo_root,
    ).decode().split("\0")
    oversized = []
    for name in sorted(set(paths)):
        relative = Path(name)
        path = repo_root / relative
        if relative.suffix != ".json" or not path.is_file():
            continue
        limit = MAX_LEDGER_SHARD_BYTES if relative.parent in LEDGER_DIRECTORIES else MAX_JSON_BYTES
        size = path.stat().st_size
        if size > limit:
            oversized.append((name, size, limit))
    return oversized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    oversized = oversized_data_files(args.repo_root)
    for name, size, limit in oversized:
        print(f"{name}: {size:,} bytes exceeds {limit:,}; shard this data before publishing")
    if not oversized:
        print("Data JSON sizes are within limits (50 MB per file; 10 MB per calibration ledger shard).")
    return int(bool(oversized))


if __name__ == "__main__":
    raise SystemExit(main())
