"""Advance the committed tennis ratings snapshot from published workbooks."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .tennis_core import RATINGS_PATH, RatingEngine
from .tennis_model import catch_up_ratings


ARCHIVE_REFRESH_LAG_DAYS = 3


def refresh_archive_snapshot(before: str, *, path: Path = RATINGS_PATH, download: bool = True) -> dict:
    """Persist only workbook results; ESPN fallback stays in the serving replay.

    The fallback lacks some archive inputs and may overlap a later workbook
    release. Persisting it would cause that release to be skipped by date.
    """
    target = date.fromisoformat(before)
    engine = RatingEngine.load(path)
    if engine is None or not engine.last_date:
        raise ValueError(f"tennis ratings snapshot unavailable: {path}")
    previous = engine.last_date
    lag = (target - date.fromisoformat(previous)).days
    result = {"snapshotThrough": previous, "target": before, "lagDays": lag, "updated": False}
    if lag <= ARCHIVE_REFRESH_LAG_DAYS:
        return result

    replay = catch_up_ratings(
        engine, previous, download=download, before=before, include_fallback=False
    )
    result["archiveThrough"] = replay.get("archive_through", previous)
    result["workbookMatches"] = replay["applied"]
    if replay["applied"] == 0 or engine.last_date <= previous:
        return result

    # A failed write must leave the previous usable artifact in place.
    temporary = path.with_name(f".{path.stem}.tmp.gz")
    try:
        engine.save(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    result["snapshotThrough"] = engine.last_date
    result["updated"] = True
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Replay workbook results before this ISO date")
    args = parser.parse_args()
    print(json.dumps(refresh_archive_snapshot(args.date), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
