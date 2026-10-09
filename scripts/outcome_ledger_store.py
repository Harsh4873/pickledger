#!/usr/bin/env python3
"""Daily storage for the canonical, deduplicated pick outcome ledger."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_RELATIVE_PATH = Path("data/calibration/outcome_ledger.json")
SHARD_RELATIVE_PATH = LEDGER_RELATIVE_PATH.with_suffix("")
SHARD_MAX_BYTES = 8_000_000
_SHARD_NAME = re.compile(r"(?:\d{4}-\d{2}-\d{2}|undated)(?:-\d{3,})?\.json")
SETTLED_RESULTS = {"win", "loss", "push", "void", "cancelled"}


def _sort_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(str(row.get(key) or "") for key in ("date", "model_key", "id"))


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list) or any(
        not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
        for row in records
    ):
        raise ValueError("Outcome ledger requires records with stable string ids")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Outcome ledger contains duplicate ids")
    return records


def merge_outcome_ledgers(current: dict[str, Any], generated: dict[str, Any]) -> dict[str, Any]:
    """Union whole rows by id; keep current settled rows, else take generated rows."""
    if current.get("schema_version") != generated.get("schema_version"):
        raise ValueError("Cannot merge different outcome ledger schemas")
    rows = {row["id"]: row for row in _records(current)}
    for incoming in _records(generated):
        existing = rows.get(incoming["id"])
        # Preserve concurrent grades and grader retractions during publication.
        if existing is None or (
            existing.get("result") not in SETTLED_RESULTS
            and not existing.get("settlement_exclusion_reason")
        ):
            rows[incoming["id"]] = incoming
    records = sorted(rows.values(), key=_sort_key)
    decided = [row for row in records if row.get("result") in {"win", "loss"}]
    return {
        **generated,
        **current,
        "records": records,
        "summary": {
            "total_picks": len(records),
            "decided_picks": len(decided),
            "trainable_decided_picks": sum(
                row.get("raw_probability") is not None and row.get("calibration_eligible") is True
                for row in decided
            ),
            "pending_picks": sum(row.get("result") in {"", "pending"} for row in records),
        },
    }


def _paths(repo_root: Path | str, path: Path | str | None) -> tuple[Path, Path]:
    target = Path(path) if path is not None else Path(repo_root) / LEDGER_RELATIVE_PATH
    if target.suffix == ".json":
        return target, target.with_suffix("")
    return Path(str(target) + ".json"), target


def _legacy(path: Path, *, strict: bool = False) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        if strict:
            raise
        return None
    if strict and (not isinstance(payload, dict) or not isinstance(payload.get("records"), list)):
        raise ValueError(f"Invalid legacy outcome ledger: {path}")
    return payload if isinstance(payload, dict) else None


def load_outcome_ledger(
    repo_root: Path | str = REPO_ROOT, *, path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Read daily shards or the old file without changing the payload shape.

    A late legacy publication is unioned using the publication retry rules.
    Missing/malformed legacy-only files retain the old optional-reader behavior;
    missing or corrupt indexed evidence raises rather than dropping history.
    ``path`` accepts either a legacy filename or a shard directory.
    """
    legacy_path, directory = _paths(repo_root, path)
    index_path = directory / "index.json"
    if not index_path.exists():
        # Initial migration keeps the complete legacy file until the index is
        # installed. Ignore any partial shards while that file still owns it.
        if legacy_path.exists():
            return _legacy(legacy_path)
        if any(directory.glob("*.json")):
            raise ValueError(f"Missing outcome ledger index: {index_path}")
        return None
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if (
        not isinstance(index, dict) or index.get("storage_version") != 1
        or not isinstance(index.get("metadata"), dict) or "records" in index["metadata"]
        or not isinstance(index.get("shards"), list)
        or type(index.get("record_count")) is not int or index["record_count"] < 0
    ):
        raise ValueError(f"Invalid outcome ledger index: {index_path}")
    records = []
    seen = set()
    for shard in index["shards"]:
        if (
            not isinstance(shard, dict) or not isinstance(shard.get("file"), str)
            or not _SHARD_NAME.fullmatch(shard["file"]) or shard["file"] in seen
            or type(shard.get("record_count")) is not int or shard["record_count"] < 0
        ):
            raise ValueError(f"Invalid outcome ledger shard entry: {index_path}")
        seen.add(shard["file"])
        shard_path = directory / shard["file"]
        if not shard_path.is_file():
            raise ValueError(f"Missing outcome ledger shard: {shard_path}")
        rows = json.loads(shard_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or len(rows) != shard["record_count"]:
            raise ValueError(f"Incomplete outcome ledger shard: {shard_path}")
        records.extend(rows)
    payload = {**index["metadata"], "records": records}
    _records(payload)
    if len(records) != index["record_count"]:
        raise ValueError(f"Incomplete outcome ledger history: {index_path}")
    records.sort(key=_sort_key)
    legacy = _legacy(legacy_path, strict=True)
    return merge_outcome_ledgers(payload, legacy) if legacy is not None and legacy != payload else payload


def _atomic_write_changed(path: Path, rendered: bytes) -> bool:
    if path.exists() and path.read_bytes() == rendered:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def write_outcome_ledger(
    payload: dict[str, Any], repo_root: Path | str = REPO_ROOT, *, path: Path | str | None = None,
) -> bool:
    """Replace the canonical payload, writing only changed daily shards.

    Rebuilds remain authoritative, including removals and grade corrections.
    Callers preserving concurrent history must load/merge before writing; the
    migration and retry commands do this explicitly. The legacy file is removed
    only after every shard and the index has been successfully written.
    """
    records = _records(payload)
    if records != sorted(records, key=_sort_key):
        raise ValueError("Outcome ledger records must be in canonical (date, model_key, id) order")
    legacy_path, directory = _paths(repo_root, path)
    _legacy(legacy_path, strict=True)
    buckets: dict[str, list[bytes]] = defaultdict(list)
    for record in records:
        try:
            day = date.fromisoformat(str(record.get("date") or "")).isoformat()
        except ValueError:
            day = "undated"
        line = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(line) + 5 > SHARD_MAX_BYTES:
            raise ValueError("One outcome ledger record exceeds the shard size limit")
        buckets[day].append(line)
    rendered_shards = {}
    shards = []
    for day, lines in sorted(buckets.items()):
        parts: list[list[bytes]] = [[]]
        size = 4
        for line in lines:
            if size + len(line) + 2 > SHARD_MAX_BYTES and parts[-1]:
                parts.append([])
                size = 4
            parts[-1].append(line)
            size += len(line) + 2
        for part, contents in enumerate(parts, 1):
            name = f"{day}.json" if part == 1 else f"{day}-{part:03d}.json"
            rendered_shards[name] = b"[\n" + b",\n".join(contents) + b"\n]\n"
            shards.append({"file": name, "record_count": len(contents)})
    index = {
        "storage_version": 1,
        "metadata": {key: value for key, value in payload.items() if key != "records"},
        "record_count": len(records),
        "shards": shards,
    }
    changed = False
    for name, rendered in rendered_shards.items():
        changed |= _atomic_write_changed(directory / name, rendered)
    changed |= _atomic_write_changed(directory / "index.json", (json.dumps(index, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    for old in directory.glob("*.json"):
        if _SHARD_NAME.fullmatch(old.name) and old.name not in rendered_shards:
            old.unlink()
            changed = True
    if legacy_path.exists():
        legacy_path.unlink()
        changed = True
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--migrate", action="store_true", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--merge-from", type=Path, help="Union a saved repository root before migrating")
    args = parser.parse_args()
    legacy_path, _ = _paths(args.repo_root, None)
    _legacy(legacy_path, strict=True)
    payload = load_outcome_ledger(args.repo_root)
    if args.merge_from is not None:
        incoming_legacy, _ = _paths(args.merge_from, None)
        _legacy(incoming_legacy, strict=True)
        incoming = load_outcome_ledger(args.merge_from)
        if incoming is None:
            parser.error("No outcome ledger at --merge-from root")
        payload = merge_outcome_ledgers(payload, incoming) if payload is not None else incoming
    if payload is None:
        parser.error("No outcome ledger to migrate")
    changed = write_outcome_ledger(payload, args.repo_root)
    print(json.dumps({"records": len(payload["records"]), "changed": changed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
