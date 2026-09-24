#!/usr/bin/env python3
"""Grade settled Profit Desk picks against results already stored in the repo.

The desk publishes a live card and then leaves that settled record outside the
next model run. This grader does not call ESPN and does not invent a result.
A row is scored only when the dated model cache, the dated player-prop cache,
or an agreeing immutable player-prop snapshot already says win, loss, or push.

The checked-in report is what the next publish reads. ``seal_published_picks``
stamps that report onto every outgoing pick and refuses to ship a pick that
still has no id and no report link.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_profit_desk import (
    FIRST_LIVE_DATE,
    _iter_records,
    _read_json,
    _record_date,
    _result,
    _text,
    canonical_market_identity,
)


REPORT_RELATIVE_PATH = "data/loss_feedback/desk_grade.json"
LINK_FIELD = "lossFeedback"
SETTLED_RESULTS = {"win", "loss", "push"}


class LossFeedbackLinkError(RuntimeError):
    """A pick was about to ship without a link to the settled desk record."""


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _write_json_if_changed(path: Path, payload: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _json_text(payload)
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return False
    path.write_text(rendered, encoding="utf-8")
    return True


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _hits_from_payload(
    payload: Mapping[str, Any] | None,
    *,
    mode: str,
    date_iso: str,
    origin: str,
    path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if not isinstance(payload, Mapping):
        return hits
    for context in _iter_records(payload, mode):
        record = context.record
        if _record_date(record, context.fallback_date) != date_iso:
            continue
        identity = canonical_market_identity(
            record,
            mode=context.mode,
            sport=_text(record.get("sport")),
            date_iso=date_iso,
        )
        hits.append(
            {
                "source_key": context.source_key,
                "identity": identity,
                "result": _result(record.get("result")),
                "source_pick_id": _text(record.get("id")) or None,
                "origin": origin,
                "path": _relative(path, repo_root),
            }
        )
    return hits


def _hits_from_file(
    path: Path,
    *,
    mode: str,
    date_iso: str,
    origin: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return _hits_from_payload(
        _read_json(path),
        mode=mode,
        date_iso=date_iso,
        origin=origin,
        path=path,
        repo_root=repo_root,
    )


def _choose(hits: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Collapse observations of one market. Disagreeing settled results stay ungraded."""

    if not hits:
        return None
    settled = [hit for hit in hits if hit["result"] in SETTLED_RESULTS]
    if not settled:
        return {**hits[-1], "status": "pending", "result": None}
    distinct = {hit["result"] for hit in settled}
    if len(distinct) != 1:
        return {**settled[-1], "status": "conflict", "result": None}
    return {**settled[-1], "status": "graded"}


def index_settled_sources(
    model_dir: Path,
    player_dir: Path,
    snapshot_dir: Path | None,
    date_iso: str,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, dict[tuple[str, str], dict[str, Any]]]:
    """Index actual settled results for one slate.

    The dated cache wins when it already has a settled result. Snapshots are
    used only when the cache row is missing or still pending, and only when
    every settled snapshot of that market agrees.
    """

    cache_hits = _hits_from_file(
        Path(model_dir) / f"{date_iso}.json",
        mode="team",
        date_iso=date_iso,
        origin="model_cache",
        repo_root=repo_root,
    ) + _hits_from_file(
        Path(player_dir) / f"{date_iso}.json",
        mode="player",
        date_iso=date_iso,
        origin="player_props_cache",
        repo_root=repo_root,
    )
    snapshot_hits: list[dict[str, Any]] = []
    if snapshot_dir is not None:
        day_dir = Path(snapshot_dir) / date_iso
        if day_dir.is_dir():
            for path in sorted(day_dir.glob("*.json")):
                snapshot_hits.extend(
                    _hits_from_file(
                        path,
                        mode="player",
                        date_iso=date_iso,
                        origin="player_props_snapshot",
                        repo_root=repo_root,
                    )
                )

    grouped_cache: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_snapshot: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for hit in cache_hits:
        grouped_cache[(hit["source_key"], hit["identity"])].append(hit)
    for hit in snapshot_hits:
        grouped_snapshot[(hit["source_key"], hit["identity"])].append(hit)

    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    by_source_pick_id: dict[tuple[str, str], dict[str, Any]] = {}
    for key in set(grouped_cache) | set(grouped_snapshot):
        cache_choice = _choose(grouped_cache.get(key, []))
        snapshot_choice = _choose(grouped_snapshot.get(key, []))
        if cache_choice and cache_choice["status"] == "graded":
            chosen = cache_choice
        elif cache_choice and cache_choice["status"] == "conflict":
            chosen = cache_choice
        elif snapshot_choice and snapshot_choice["status"] in {"graded", "conflict"}:
            chosen = snapshot_choice
        else:
            chosen = cache_choice or snapshot_choice
        if not chosen:
            continue
        by_identity[key] = chosen
        source_pick_id = chosen.get("source_pick_id")
        if chosen["status"] == "graded" and source_pick_id:
            by_source_pick_id[(chosen["source_key"], source_pick_id)] = chosen
    return {"by_identity": by_identity, "by_source_pick_id": by_source_pick_id}


def _live_rows(profit_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(profit_dir.glob("20??-??-??.json")):
        if path.stem < FIRST_LIVE_DATE:
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        portfolio = payload.get("portfolio") if isinstance(payload.get("portfolio"), dict) else {}
        for row in portfolio.get("live") or []:
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _published_desk_record(profit_dir: Path) -> dict[str, Any] | None:
    files = sorted(profit_dir.glob("20??-??-??.json"))
    if not files:
        return None
    payload = _read_json(files[-1]) or {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    record = summary.get("liveRecordToDate")
    return dict(record) if isinstance(record, dict) else None


def _lookup_hit(
    row: Mapping[str, Any],
    indexed: Mapping[str, Mapping[tuple[str, str], dict[str, Any]]],
) -> dict[str, Any] | None:
    identity_key = (_text(row.get("sourceKey")), _text(row.get("marketIdentity")))
    hit = indexed["by_identity"].get(identity_key)
    if hit is not None:
        return hit
    source_pick_id = _text(row.get("sourcePickId"))
    if not source_pick_id:
        return None
    return indexed["by_source_pick_id"].get((_text(row.get("sourceKey")), source_pick_id))


def _profit_units(result: str | None, stake: float | None, decimal_odds: float | None) -> float | None:
    if result == "push":
        return 0.0
    if result not in {"win", "loss"} or stake is None or stake <= 0 or decimal_odds is None:
        return None
    if result == "win":
        return stake * (decimal_odds - 1.0)
    return -stake


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def grade_desk_loss_feedback(
    repo_root: Path = REPO_ROOT,
    *,
    profit_dir: Path | None = None,
    model_dir: Path | None = None,
    player_dir: Path | None = None,
    snapshot_dir: Path | None = None,
) -> dict[str, Any]:
    """Score live desk picks. Rows with no settled source result stay ungraded."""

    root = Path(repo_root)
    desk_dir = Path(profit_dir) if profit_dir is not None else root / "data" / "profit_desk"
    models = Path(model_dir) if model_dir is not None else root / "data" / "model_cache"
    props = Path(player_dir) if player_dir is not None else root / "data" / "player_props_cache"
    snapshots = Path(snapshot_dir) if snapshot_dir is not None else root / "data" / "player_props_snapshots"
    if not snapshots.is_dir():
        snapshots = None  # type: ignore[assignment]

    indexes: dict[str, dict[str, dict[tuple[str, str], dict[str, Any]]]] = {}
    picks: list[dict[str, Any]] = []
    for row in _live_rows(desk_dir):
        date_iso = _text(row.get("date"))
        if date_iso not in indexes:
            indexes[date_iso] = index_settled_sources(
                models,
                props,
                snapshots,
                date_iso,
                repo_root=root,
            )
        hit = _lookup_hit(row, indexes[date_iso])
        desk_result = _result(row.get("result"))
        stake = _number(row.get("stakeUnits"))
        decimal_odds = _number(row.get("decimalOdds"))
        graded = bool(hit and hit.get("status") == "graded" and hit.get("result") in SETTLED_RESULTS)
        result = hit.get("result") if graded and hit else None
        if hit and hit.get("status") == "conflict":
            reason = "source_result_conflict"
        elif hit and hit.get("status") == "pending":
            reason = "source_result_pending"
        elif not graded:
            reason = "source_row_missing"
        else:
            reason = None
        profit = _profit_units(result, stake, decimal_odds) if graded else None
        picks.append(
            {
                "deskId": _text(row.get("id")) or None,
                "date": date_iso,
                "sourceKey": _text(row.get("sourceKey")) or None,
                "sourcePickId": (hit or {}).get("source_pick_id") or _text(row.get("sourcePickId")) or None,
                "pick": _text(row.get("pick")) or None,
                "sport": _text(row.get("sport")) or None,
                "market": _text(row.get("market")) or None,
                "lane": _text(row.get("lane")) or None,
                "stakeUnits": stake,
                "decimalOdds": decimal_odds,
                "oddsAmerican": row.get("oddsAmerican"),
                "deskResult": desk_result,
                "result": result,
                "graded": graded,
                "resultSource": (hit or {}).get("origin") if graded else None,
                "resultPath": (hit or {}).get("path") if graded else None,
                "ungradedReason": reason,
                "profitUnits": round(profit, 4) if profit is not None else None,
                "disagreesWithDesk": bool(
                    graded and desk_result in SETTLED_RESULTS and desk_result != result
                ),
            }
        )

    return _report(picks, published=_published_desk_record(desk_dir))


def _report(picks: list[dict[str, Any]], *, published: dict[str, Any] | None) -> dict[str, Any]:
    wins = losses = pushes = 0
    net = 0.0
    staked = 0.0
    ungraded = 0
    by_source: dict[str, dict[str, Any]] = {}
    for row in picks:
        source_key = row.get("sourceKey") or "unknown"
        bucket = by_source.setdefault(
            source_key,
            {"sourceKey": source_key, "graded": 0, "ungraded": 0, "wins": 0, "losses": 0, "pushes": 0, "netUnits": 0.0, "stakedUnits": 0.0},
        )
        if not row["graded"]:
            ungraded += 1
            bucket["ungraded"] += 1
            continue
        bucket["graded"] += 1
        profit = row.get("profitUnits")
        if row["result"] == "win":
            wins += 1
            bucket["wins"] += 1
        elif row["result"] == "loss":
            losses += 1
            bucket["losses"] += 1
        elif row["result"] == "push":
            pushes += 1
            bucket["pushes"] += 1
        if profit is not None and row["result"] in {"win", "loss"}:
            net += profit
            stake = row.get("stakeUnits") or 0.0
            staked += stake
            bucket["netUnits"] += profit
            bucket["stakedUnits"] += stake

    for bucket in by_source.values():
        bucket["netUnits"] = round(bucket["netUnits"], 4)
        bucket["stakedUnits"] = round(bucket["stakedUnits"], 4)
        bucket["roi"] = round(bucket["netUnits"] / bucket["stakedUnits"], 6) if bucket["stakedUnits"] else None

    dates = [row["date"] for row in picks if row.get("date")]
    source_graded = {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pending": ungraded,
        "settled": wins + losses,
        "netUnits": round(net, 4),
        "stakedUnits": round(staked, 4),
        "roi": round(net / staked, 6) if staked else None,
        "sinceDate": FIRST_LIVE_DATE,
        "throughDate": max(dates) if dates else None,
    }
    identity = [
        {
            "deskId": row.get("deskId"),
            "result": row.get("result"),
            "profitUnits": row.get("profitUnits"),
            "resultSource": row.get("resultSource"),
        }
        for row in picks
    ]
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:16]
    disagreements = [row for row in picks if row["disagreesWithDesk"]]
    ungraded_rows = [row for row in picks if not row["graded"]]
    return {
        "schemaVersion": 1,
        "kind": "desk_loss_feedback",
        "reportId": f"desk-grade-{digest}",
        "reportPath": REPORT_RELATIVE_PATH,
        "method": "dated_cache_then_agreeing_snapshots",
        "policy": (
            "Score a live Profit Desk pick only when a dated model cache, dated player-prop cache, "
            "or agreeing immutable snapshot already records win, loss, or push. "
            "Do not invent a result when those rows are missing, pending, or in conflict."
        ),
        "summary": {
            "deskRows": len(picks),
            "graded": len(picks) - ungraded,
            "ungraded": ungraded,
            "disagreements": len(disagreements),
            "publishedDeskRecord": published,
            "sourceGradedRecord": source_graded,
        },
        "bySource": [by_source[key] for key in sorted(by_source)],
        "disagreements": disagreements,
        "ungradedPicks": ungraded_rows,
        "picks": picks,
    }


def write_desk_grade_report(
    repo_root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    profit_dir: Path | None = None,
    model_dir: Path | None = None,
    player_dir: Path | None = None,
    snapshot_dir: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    report = grade_desk_loss_feedback(
        repo_root,
        profit_dir=profit_dir,
        model_dir=model_dir,
        player_dir=player_dir,
        snapshot_dir=snapshot_dir,
    )
    path = Path(output_path) if output_path is not None else Path(repo_root) / REPORT_RELATIVE_PATH
    changed = _write_json_if_changed(path, report)
    return report, changed


def load_desk_grade_report(repo_root: Path = REPO_ROOT) -> dict[str, Any] | None:
    payload = _read_json(Path(repo_root) / REPORT_RELATIVE_PATH)
    if not payload or not _text(payload.get("reportId")):
        return None
    return payload


def _iter_published_picks(payload: Mapping[str, Any]):
    seen: set[int] = set()

    def emit(picks: Any):
        if not isinstance(picks, list):
            return
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            marker = id(pick)
            if marker in seen:
                continue
            seen.add(marker)
            yield pick

    yield from emit(payload.get("picks"))
    for container_key in ("models", "external_feeds"):
        container = payload.get(container_key)
        if not isinstance(container, dict):
            continue
        for bucket in container.values():
            if isinstance(bucket, dict):
                yield from emit(bucket.get("picks"))
    for key, bucket in payload.items():
        if key in {"models", "external_feeds", "picks"} or not isinstance(bucket, dict):
            continue
        yield from emit(bucket.get("picks"))


def stable_pick_id(scope: str, pick: Mapping[str, Any]) -> str:
    """Stable id for a pick that shipped without one. List position is not part of it."""

    raw = json.dumps(
        [
            scope,
            pick.get("source"),
            pick.get("sport"),
            pick.get("date") or pick.get("game_date") or pick.get("slate_date"),
            pick.get("pick"),
            pick.get("matchup") or pick.get("game"),
            pick.get("player_name") or pick.get("player"),
            pick.get("market_type") or pick.get("market"),
            pick.get("line"),
            pick.get("selection") or pick.get("direction"),
        ],
        sort_keys=True,
        default=str,
    )
    return "pick-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def loss_feedback_link(report: Mapping[str, Any]) -> dict[str, Any]:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    record = summary.get("sourceGradedRecord") if isinstance(summary.get("sourceGradedRecord"), dict) else {}
    return {
        "reportId": _text(report.get("reportId")),
        "reportPath": REPORT_RELATIVE_PATH,
        "throughDate": record.get("throughDate"),
        "settled": record.get("settled"),
        "wins": record.get("wins"),
        "losses": record.get("losses"),
        "pushes": record.get("pushes"),
        "netUnits": record.get("netUnits"),
        "stakedUnits": record.get("stakedUnits"),
        "roi": record.get("roi"),
    }


def attach_loss_feedback_link(payload: dict[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """Stamp a stable id and the settled-record link onto every pick about to ship."""

    link = loss_feedback_link(report)
    if not link["reportId"]:
        raise LossFeedbackLinkError("loss-feedback report has no reportId")
    used = {_text(pick.get("id")) for pick in _iter_published_picks(payload) if _text(pick.get("id"))}
    for pick in _iter_published_picks(payload):
        if not _text(pick.get("id")):
            scope = _text(pick.get("source")) or _text(pick.get("sport")) or "pick"
            candidate = stable_pick_id(scope, pick)
            suffix = 2
            while candidate in used:
                candidate = f"{stable_pick_id(scope, pick)}-{suffix}"
                suffix += 1
            pick["id"] = candidate
            used.add(candidate)
        pick[LINK_FIELD] = dict(link)
    return payload


def require_loss_feedback_link(payload: Mapping[str, Any]) -> None:
    """Refuse publication when any pick has no id or no settled-record link."""

    missing = []
    for pick in _iter_published_picks(payload):
        link = pick.get(LINK_FIELD) if isinstance(pick.get(LINK_FIELD), dict) else {}
        if not _text(pick.get("id")) or not _text(link.get("reportId")):
            missing.append(_text(pick.get("pick")) or _text(pick.get("matchup")) or "unlabeled pick")
    if missing:
        preview = ", ".join(missing[:5])
        raise LossFeedbackLinkError(
            f"{len(missing)} pick(s) would ship with no link to the settled desk record: {preview}"
        )


def seal_published_picks(payload: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Write the graded report, stamp the link, and fail closed if a pick is still unlinked."""

    report, _changed = write_desk_grade_report(repo_root)
    attach_loss_feedback_link(payload, report)
    require_loss_feedback_link(payload)
    return payload


def main() -> int:
    report, changed = write_desk_grade_report()
    record = report["summary"]["sourceGradedRecord"]
    print(
        "[desk-loss-feedback] "
        f"graded={report['summary']['graded']} ungraded={report['summary']['ungraded']} "
        f"record={record['wins']}-{record['losses']}-{record['pushes']} "
        f"net={record['netUnits']} roi={record['roi']} "
        f"report={REPORT_RELATIVE_PATH} changed={changed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
