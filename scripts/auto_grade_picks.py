#!/usr/bin/env python3
"""Grade committed static pick caches against ESPN scoreboards."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = REPO_ROOT / "data" / "model_cache"
PLAYER_PROPS_CACHE_DIR = REPO_ROOT / "data" / "player_props_cache"
PLAYER_PROPS_SNAPSHOT_DIR = REPO_ROOT / "data" / "player_props_snapshots"
sys.path.insert(0, str(REPO_ROOT))

import pickgrader_server  # noqa: E402
from scripts.scrapers.tennis_scraper import grade_tennis_picks, is_tennis_pick  # noqa: E402


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sync_top_level_model_aliases(payload: dict[str, Any]) -> int:
    """Re-point the vestigial top-level model aliases at their graded bucket.

    model_cache files carry each in-house team model twice: the canonical
    ``models[<key>]`` bucket that the static site and the outcome ledger read,
    and a legacy top-level alias (``payload[<alias>]``) written by the merge and
    consensus steps for external/legacy consumers. Grading only touches
    ``models[]``, so the top-level alias goes stale. Re-point each existing alias
    at its graded bucket so the duplicate always mirrors the graded record.
    """
    models = payload.get("models")
    if not isinstance(models, dict):
        return 0
    try:
        from scripts.merge_model_cache_payload import MODEL_ALIAS_TO_MODEL_KEY
    except Exception:
        return 0
    changed = 0
    for alias_key, model_key in MODEL_ALIAS_TO_MODEL_KEY.items():
        bucket = models.get(model_key)
        if not isinstance(bucket, dict) or alias_key not in payload:
            continue
        current = payload.get(alias_key)
        if current is bucket:
            continue
        if json.dumps(current, sort_keys=True, default=str) == json.dumps(bucket, sort_keys=True, default=str):
            continue
        payload[alias_key] = bucket
        changed += 1
    return changed


def _iter_pick_lists(payload: dict[str, Any]) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    direct = payload.get("picks")
    if isinstance(direct, list):
        yield "picks", [pick for pick in direct if isinstance(pick, dict)]

    models = payload.get("models")
    if not isinstance(models, dict):
        return
    for model_key, bucket in models.items():
        if not isinstance(bucket, dict) or not isinstance(bucket.get("picks"), list):
            continue
        yield str(model_key), [pick for pick in bucket["picks"] if isinstance(pick, dict)]


def _grade_id(scope: str, index: int, pick: dict[str, Any]) -> str:
    existing = str(pick.get("id") or "").strip()
    if existing:
        return existing
    raw = json.dumps(
        [
            scope,
            index,
            pick.get("source"),
            pick.get("sport"),
            pick.get("date"),
            pick.get("pick"),
            pick.get("matchup") or pick.get("game"),
        ],
        sort_keys=True,
        default=str,
    )
    return f"grade-{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _apply_tennis_grades(
    candidates: list[dict[str, Any]],
    refs: dict[str, dict[str, Any]],
) -> int:
    """Grade intercepted tennis picks against the ESPN winner flag."""
    graded = grade_tennis_picks(candidates)
    changed = 0
    for grade_id, pick in refs.items():
        entry = graded.get(grade_id) if isinstance(graded, dict) else None
        if not isinstance(entry, dict):
            continue
        result = str(entry.get("result") or "pending").lower()
        if result in {"win", "loss", "push"} and pick.get("result") != result:
            pick["result"] = result
            changed += 1
        start_time = str(entry.get("start_time") or "").strip()
        if start_time and pick.get("start_time") != start_time:
            pick["start_time"] = start_time
            pick["game_start_time"] = start_time
            changed += 1
    return changed


def grade_payload(payload: dict[str, Any], *, ml_player_props_only: bool = False) -> int:
    fallback_date = str(payload.get("date") or payload.get("slate_date") or payload.get("as_of") or "").strip()
    fallback_timestamp = payload.get("generatedAt") or payload.get("updatedAt")
    pending: list[dict[str, Any]] = []
    refs: dict[str, dict[str, Any]] = {}
    tennis_pending: list[dict[str, Any]] = []
    tennis_refs: dict[str, dict[str, Any]] = {}
    changed = 0
    is_ml_era_pick = None
    if ml_player_props_only:
        from player_props.era import is_ml_era_pick as is_ml_era_pick_predicate

        is_ml_era_pick = is_ml_era_pick_predicate

    for scope, picks in _iter_pick_lists(payload):
        for index, pick in enumerate(picks):
            changed += pickgrader_server.apply_external_pick_metadata(pick)
            if ml_player_props_only and is_ml_era_pick is not None and not is_ml_era_pick(pick, fallback_timestamp):
                continue
            if str(pick.get("decision") or "").strip().upper() not in {"BET", "LEAN"}:
                continue
            if pick.get("grade_supported") is False:
                continue
            grade_id = _grade_id(scope, index, pick)
            candidate = dict(pick)
            candidate["id"] = grade_id
            candidate["date"] = str(candidate.get("date") or fallback_date)
            candidate["result"] = "pending"
            # Tennis is player-vs-player: ESPN's tennis JSON is structurally
            # different from the team scoreboards the shared engine parses, so
            # grade it in an isolated winner-flag path and keep it out of the
            # team grader entirely.
            if is_tennis_pick(pick):
                tennis_pending.append(candidate)
                tennis_refs[grade_id] = pick
                continue
            pending.append(candidate)
            refs[grade_id] = pick

    if tennis_pending:
        changed += _apply_tennis_grades(tennis_pending, tennis_refs)

    if not pending:
        return changed

    response = pickgrader_server.auto_grade(pending, {}, datetime.now().year)
    grades = response.get("graded") if isinstance(response, dict) else {}
    start_times = response.get("startTimes") if isinstance(response, dict) else {}
    unsupported = response.get("unsupported") if isinstance(response, dict) else {}
    anomalies = response.get("gradeAnomalies") if isinstance(response, dict) else []
    grades = grades if isinstance(grades, dict) else {}
    start_times = start_times if isinstance(start_times, dict) else {}
    unsupported = unsupported if isinstance(unsupported, dict) else {}

    for grade_id, pick in refs.items():
        result = str(grades.get(grade_id) or "pending").lower()
        if result in {"win", "loss", "push"} and pick.get("result") != result:
            pick["result"] = result
            changed += 1
        reason = str(unsupported.get(grade_id) or "").strip()
        if reason and pick.get("grade_supported") is not False:
            pick["grade_supported"] = False
            pick["grade_note"] = f"Auto-grading unsupported: {reason}"
            changed += 1
        start_time = str(start_times.get(grade_id) or "").strip()
        if start_time and pick.get("start_time") != start_time:
            pick["start_time"] = start_time
            pick["game_start_time"] = start_time
            changed += 1

    if isinstance(anomalies, list) and anomalies:
        detail = ", ".join(
            f"{row.get('id')}={row.get('reason')}" for row in anomalies[:10] if isinstance(row, dict)
        )
        print(f"[auto-grade] {len(anomalies)} grading anomalie(s): {detail}")
    return changed


def grade_file(path: Path, *, ml_player_props_only: bool = False) -> int:
    payload = _read_json(path)
    if not payload:
        print(f"[auto-grade] skipped unreadable {path.relative_to(REPO_ROOT)}")
        return 0
    changed = grade_payload(payload, ml_player_props_only=ml_player_props_only)
    if path.parent == MODEL_CACHE_DIR:
        changed += _sync_top_level_model_aliases(payload)
    if changed:
        _write_json(path, payload)
    print(f"[auto-grade] {path.relative_to(REPO_ROOT)}: {changed} update(s)")
    return changed


def _team_prop_ledger_storage() -> tuple[Any, Any] | None:
    """Load the optional certified team-prop ledger without coupling legacy grading to it."""
    try:
        module = importlib.import_module("scripts.team_prop_pregame_ledger")
    except ModuleNotFoundError as exc:
        if exc.name == "scripts.team_prop_pregame_ledger":
            return None
        raise

    load = getattr(module, "load_team_prop_pregame_ledger", None)
    write = getattr(module, "write_team_prop_pregame_ledger", None)
    if not callable(load) or not callable(write):
        raise RuntimeError("team-prop pregame ledger does not expose the required storage functions")
    return load, write


def _certified_team_prop_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    certification = record.get("certification")
    if isinstance(certification, dict) and str(certification.get("status") or "").strip().lower() == "certified":
        return True
    return record.get("certified_pregame") is True


def _pending_certified_team_prop_candidate(record: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Build a BET/LEAN grading candidate for a certified ledger record."""
    if not _certified_team_prop_record(record):
        return None
    if str(record.get("result") or "pending").strip().lower() != "pending":
        return None

    snapshot = record.get("pregame_snapshot")
    if not isinstance(snapshot, dict):
        snapshot = record.get("snapshot")
    if not isinstance(snapshot, dict):
        return None

    decision_markers = {
        str(value).strip().upper()
        for value in (record.get("decision"), record.get("raw_decision"), snapshot.get("decision"))
        if str(value or "").strip()
    }
    # Historical ledgers may already contain PASS snapshots (including a
    # calibrated PASS whose raw snapshot still says BET). Never send any such
    # row to the grader: only actual BET/LEAN publications are tracked picks.
    if not decision_markers or "PASS" in decision_markers or not decision_markers <= {"BET", "LEAN"}:
        return None
    decision = str(
        record.get("decision") or snapshot.get("decision") or record.get("raw_decision") or ""
    ).strip().upper()

    record_id = str(record.get("id") or record.get("snapshot_id") or "").strip()
    if not record_id:
        return None

    candidate = copy.deepcopy(snapshot)
    candidate["id"] = record_id
    candidate["decision"] = decision
    candidate["result"] = "pending"
    for key in (
        "date",
        "game_date",
        "slate_date",
        "sport",
        "pick",
        "selection",
        "matchup",
        "game",
        "away_team",
        "home_team",
        "market_type",
        "market",
        "grade_supported",
    ):
        if candidate.get(key) in {None, ""} and record.get(key) not in {None, ""}:
            candidate[key] = copy.deepcopy(record[key])

    if not str(candidate.get("date") or candidate.get("game_date") or candidate.get("slate_date") or "").strip():
        return None
    if not str(candidate.get("sport") or "").strip() or not str(candidate.get("pick") or candidate.get("selection") or "").strip():
        return None
    return record_id, candidate


def grade_certified_team_prop_snapshots(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Grade certified BET/LEAN team-prop snapshots.

    This is intentionally separate from ``grade_payload``: existing cache files
    keep their historical BET/LEAN-only grading contract, while the immutable
    ledger supplies the explicit pregame certification required for grading.
    """
    storage = _team_prop_ledger_storage()
    if storage is None:
        return {
            "available": False,
            "candidates": 0,
            "graded": 0,
            "start_times": 0,
            "changed": False,
        }

    load, write = storage
    payload = load(repo_root=repo_root)
    if not isinstance(payload, dict):
        raise RuntimeError("team-prop pregame ledger returned an invalid payload")
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("team-prop pregame ledger is missing its records list")

    candidates: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        candidate_row = _pending_certified_team_prop_candidate(record)
        if candidate_row is None:
            continue
        record_id, candidate = candidate_row
        if record_id in records_by_id:
            continue
        records_by_id[record_id] = record
        candidates.append(candidate)

    if not candidates:
        return {
            "available": True,
            "candidates": 0,
            "graded": 0,
            "start_times": 0,
            "changed": False,
        }

    response = pickgrader_server.auto_grade(candidates, {}, datetime.now().year)
    grades = response.get("graded") if isinstance(response, dict) else {}
    start_times = response.get("startTimes") if isinstance(response, dict) else {}
    grades = grades if isinstance(grades, dict) else {}
    start_times = start_times if isinstance(start_times, dict) else {}

    changed = False
    graded = 0
    updated_start_times = 0
    for record_id, record in records_by_id.items():
        result = str(grades.get(record_id) or "pending").strip().lower()
        if result in {"win", "loss", "push"} and record.get("result") != result:
            record["result"] = result
            changed = True
            graded += 1
        start_time = str(start_times.get(record_id) or "").strip()
        if start_time:
            if record.get("start_time") != start_time:
                record["start_time"] = start_time
                changed = True
                updated_start_times += 1
            if record.get("game_start_time") != start_time:
                record["game_start_time"] = start_time
                changed = True

    persisted = write(payload, repo_root=repo_root) if changed else False
    return {
        "available": True,
        "candidates": len(candidates),
        "graded": graded,
        "start_times": updated_start_times,
        "changed": bool(persisted),
    }


def main() -> int:
    from scripts.build_parlay_cards import rebuild_parlay_cards
    from scripts.build_profit_desk import rebuild_profit_desk
    from scripts.pick_calibration import rebuild_outcome_ledger

    total = 0
    for cache_dir in (MODEL_CACHE_DIR, PLAYER_PROPS_CACHE_DIR):
        for path in sorted(cache_dir.glob("20??-??-??.json")):
            total += grade_file(path, ml_player_props_only=cache_dir == PLAYER_PROPS_CACHE_DIR)

        latest = _read_json(cache_dir / "latest.json")
        latest_date = str(latest.get("date") or "") if latest else ""
        latest_source = cache_dir / f"{latest_date}.json"
        if latest_date and latest_source.exists():
            shutil.copyfile(latest_source, cache_dir / "latest.json")

    for path in sorted(PLAYER_PROPS_SNAPSHOT_DIR.glob("20??-??-??/*.json")):
        total += grade_file(path, ml_player_props_only=True)

    certified_summary = grade_certified_team_prop_snapshots()
    if certified_summary["available"]:
        print(
            "[auto-grade] certified team-prop ledger: "
            f"{certified_summary['graded']} result(s), "
            f"{certified_summary['start_times']} start time(s), "
            f"{certified_summary['candidates']} candidate(s), "
            f"changed={certified_summary['changed']}"
        )

    ledger, ledger_changed = rebuild_outcome_ledger()
    print(
        "[auto-grade] outcome ledger: "
        f"{ledger['summary']['total_picks']} pick(s), "
        f"{ledger['summary']['decided_picks']} decided, "
        f"changed={ledger_changed}"
    )
    parlay_changed = rebuild_parlay_cards(all_dates=True)
    print(f"[auto-grade] parlay cards: {parlay_changed} file update(s)")
    profit_desk_changed = rebuild_profit_desk(all_dates=True)
    print(f"[auto-grade] Profit Desk: {profit_desk_changed} file update(s)")
    print(f"[auto-grade] complete: {total} update(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
