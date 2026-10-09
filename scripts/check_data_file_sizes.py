#!/usr/bin/env python3
"""Keep versioned JSON data below repository hosting limits without reading it."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_JSON_BYTES = 50_000_000
MAX_LEDGER_SHARD_BYTES = 10_000_000
LEDGER_DIRECTORIES = {
    Path("data/calibration/team_prop_pregame_ledger"),
    Path("data/calibration/outcome_ledger"),
}


def _sparse_sizes(repo_root: Path, objects: dict[str, str], git_tree: Path | None) -> dict[str, int]:
    if not objects:
        return {}
    if git_tree is not None:
        # A blob:none checkout does not have omitted blobs. GitHub's recursive
        # tree supplies their sizes without fetching their contents. Match each
        # path AND object id against the index, including on pull requests.
        payload = json.loads(git_tree.read_text(encoding="utf-8"))
        if payload.get("truncated") is not False or not isinstance(payload.get("tree"), list):
            raise ValueError("Expected a complete Git tree for sparse data size checks")
        entries = {
            (entry["path"], entry["sha"]): entry.get("size")
            for entry in payload["tree"] if entry.get("type") == "blob"
        }
        sizes = {name: entries.get((name, oid)) for name, oid in objects.items()}
    else:
        output = subprocess.check_output(
            ["git", "cat-file", "--batch-check=%(objectname) %(objectsize)"],
            input="".join(f"{oid}\n" for oid in objects.values()).encode(), cwd=repo_root,
        ).decode().splitlines()
        by_object = {oid: int(size) for oid, size in (line.split() for line in output) if size.isdigit()}
        sizes = {name: by_object.get(oid) for name, oid in objects.items()}
    for name, size in sizes.items():
        if type(size) is not int or size < 0:
            raise ValueError(f"Missing or invalid indexed blob size: {name}")
    return sizes


def oversized_data_files(
    repo_root: Path = REPO_ROOT, *, git_tree: Path | None = None,
) -> list[tuple[str, int, int]]:
    """Check worktree files and sparse index entries, excluding real deletions."""
    if git_tree is None and repo_root.resolve() == REPO_ROOT:
        configured = os.environ.get("PICKLEDGER_DATA_GIT_TREE")
        git_tree = Path(configured) if configured else None
    entries = subprocess.check_output(
        ["git", "ls-files", "-t", "--stage", "--cached", "--others", "--exclude-standard", "-z", "--", "data/"],
        cwd=repo_root,
    ).decode().split("\0")
    sizes = {}
    sparse = {}
    for entry in filter(None, entries):
        if entry.startswith("? "):
            name = entry[2:]
        else:
            header, name = entry.split("\t", 1)
        if Path(name).suffix != ".json":
            continue
        path = repo_root / name
        if path.is_file():
            sizes[name] = path.stat().st_size
        elif entry.startswith("S "):
            _, _, oid, stage = header.split()
            if stage != "0":
                raise ValueError(f"Unmerged sparse data file: {name}")
            sparse[name] = oid
    sizes.update(_sparse_sizes(repo_root, sparse, git_tree))
    oversized = []
    for name, size in sorted(sizes.items()):
        relative = Path(name)
        limit = MAX_LEDGER_SHARD_BYTES if relative.parent in LEDGER_DIRECTORIES else MAX_JSON_BYTES
        if size > limit:
            oversized.append((name, size, limit))
    return oversized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--git-tree", type=Path, help="Complete GitHub Git tree JSON for omitted blobs; object ids must match the index")
    args = parser.parse_args()
    oversized = oversized_data_files(args.repo_root, git_tree=args.git_tree)
    for name, size, limit in oversized:
        print(f"{name}: {size:,} bytes exceeds {limit:,}; shard this data before publishing")
    if not oversized:
        print("Data JSON sizes are within limits (50 MB per file; 10 MB per calibration ledger shard).")
    return int(bool(oversized))


if __name__ == "__main__":
    raise SystemExit(main())
