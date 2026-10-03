#!/usr/bin/env python3
"""Dry-check the GitHub Pages publish set. Does not deploy.

harsh.bet/pickledger serves the Pages artifact built from main. The board
reads the merged model artifact at data/model_cache/latest.json (written by
merge_model_cache_payload.py). The daily brief is harsh.html together with
data/bet_briefs/latest.json and data/personal_ledger.json. Those paths have
to be copied into dist/ by .github/workflows/deploy-pages.yml. This command
reads that workflow, runs its publish copies into a temporary directory, and
fails if any required output is missing.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "deploy-pages.yml"

# Lines that must appear in deploy-pages.yml. The test fails when one is dropped.
REQUIRED_WORKFLOW_LINES = (
    "cp -R data/model_cache dist/data/",
    "cp -R data/bet_briefs dist/data/",
    "cp data/personal_ledger.json dist/data/",
    "cp harsh.html dist/",
    "test -f dist/data/model_cache/latest.json",
    "test -f dist/data/bet_briefs/latest.json",
    "test -f dist/data/personal_ledger.json",
    "test -f dist/harsh.html",
    "path: dist",
)

# Files the live board and the Harsh page request. Paths are relative to dist/.
REQUIRED_OUTPUTS = (
    "data/model_cache/latest.json",
    "data/model_cache/index.json",
    "data/bet_briefs/latest.json",
    "data/bet_briefs/index.json",
    "data/personal_ledger.json",
    "harsh.html",
)

BOARD_LINK = 'href="./harsh.html"'
MODEL_FETCH = "./data/model_cache/latest.json"


def workflow_text(repo: Path = REPO_ROOT) -> str:
    return (repo / ".github" / "workflows" / "deploy-pages.yml").read_text(encoding="utf-8")


def missing_workflow_lines(text: str) -> list[str]:
    return [line for line in REQUIRED_WORKFLOW_LINES if line not in text]


def publish_copy_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if stripped.startswith("cp ") and "dist" in stripped:
            lines.append(stripped)
    return lines


def _apply_cp(line: str, repo: Path, dest: Path) -> None:
    """Apply one `cp` / `cp -R` line from the workflow, with dist/ mapped to dest."""
    parts = line.split()
    if not parts or parts[0] != "cp":
        raise SystemExit(f"unrecognized publish copy: {line}")
    args = parts[1:]
    recursive = False
    if args and args[0] == "-R":
        recursive = True
        args = args[1:]
    if len(args) != 2:
        raise SystemExit(f"unrecognized publish copy: {line}")
    src_rel, dest_rel = args
    src = repo / src_rel
    if dest_rel == "dist":
        target_dir = dest
    elif dest_rel.startswith("dist/"):
        suffix = dest_rel[len("dist/"):]
        target_dir = dest / suffix if suffix else dest
    else:
        raise SystemExit(f"publish copy does not land in dist/: {line}")
    target_dir.mkdir(parents=True, exist_ok=True)
    if recursive:
        shutil.copytree(src, target_dir / src.name)
    else:
        shutil.copy2(src, target_dir / src.name)


def assemble_from_workflow(repo: Path, dest: Path, text: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for line in publish_copy_lines(text):
        _apply_cp(line, repo, dest)


def missing_outputs(dest: Path) -> list[str]:
    return [rel for rel in REQUIRED_OUTPUTS if not (dest / rel).is_file()]


def dry_check(repo: Path = REPO_ROOT) -> dict[str, object]:
    text = workflow_text(repo)
    missing_lines = missing_workflow_lines(text)
    index_html = (repo / "index.html").read_text(encoding="utf-8")
    data_ts = (repo / "src" / "data.ts").read_text(encoding="utf-8")
    link_ok = BOARD_LINK in index_html
    model_ok = MODEL_FETCH in data_ts
    with tempfile.TemporaryDirectory(prefix="pages-publish-") as tmp:
        dest = Path(tmp) / "dist"
        if not missing_lines:
            assemble_from_workflow(repo, dest, text)
        absent = missing_outputs(dest) if dest.exists() else list(REQUIRED_OUTPUTS)
    report = {
        "workflow": ".github/workflows/deploy-pages.yml",
        "missingWorkflowLines": missing_lines,
        "missingOutputs": absent,
        "boardLinksHarsh": link_ok,
        "boardFetchesMergedModel": model_ok,
        "deployed": False,
    }
    if missing_lines or absent or not link_ok or not model_ok:
        raise SystemExit("Pages publish set is incomplete: " + str(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-check",
        action="store_true",
        help="Copy the workflow publish set into a temp directory and exit. Does not deploy.",
    )
    args = parser.parse_args()
    if not args.dry_check:
        parser.error("pass --dry-check. This script does not deploy.")
    report = dry_check()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
