#!/usr/bin/env python3
"""Re-settle NHL player props with the fixed hockey prop grader.

Before the fix, NHL O/U player props fell through to the team grader and were
settled as full-game totals. This rewrites ONLY the ``result`` field of NHL
``player_props`` rows (certified team-prop ledger + model_cache buckets) from
real ESPN box scores. Rows whose player cannot be found go back to pending.
Re-running is a no-op. Forecast fields are never touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pickgrader_server  # noqa: E402
from scripts.team_prop_pregame_ledger import (  # noqa: E402
    load_team_prop_pregame_ledger,
    write_team_prop_pregame_ledger,
)

SummaryFetch = Callable[[str], "dict[str, Any] | None"]


def _is_nhl_prop(row: Any) -> bool:
    return (
        isinstance(row, dict)
        and str(row.get("market") or "").lower() == "player_props"
        and (str(row.get("model_key") or "") == "nhl" or str(row.get("sport") or "").upper() == "NHL")
    )


def _slate(row: dict[str, Any]) -> str:
    snap = row.get("pregame_snapshot") if isinstance(row.get("pregame_snapshot"), dict) else {}
    return str(row.get("slate_date") or row.get("date") or snap.get("date") or "")


def _final(summary: dict[str, Any] | None) -> bool:
    try:
        status = summary["header"]["competitions"][0]["status"]["type"]
    except (KeyError, IndexError, TypeError):
        return False
    return bool(status.get("completed")) and str(status.get("name") or "") == "STATUS_FINAL"


def _candidate(row: dict[str, Any]) -> dict[str, Any]:
    snap = row.get("pregame_snapshot") if isinstance(row.get("pregame_snapshot"), dict) else {}
    candidate = {**snap, **{k: v for k, v in row.items() if k != "pregame_snapshot" and v not in (None, "")}}
    candidate["sport"] = "NHL"
    candidate["market"] = "player_props"
    for key in ("player_name", "player", "stat", "stat_label", "direction", "line", "market_line"):
        if snap.get(key) not in (None, ""):
            candidate[key] = snap[key]
    return candidate


def settle(row: dict[str, Any], summaries: dict[str, Any], fetch: SummaryFetch) -> tuple[str | None, str | None]:
    """Return (result, anomaly). None result means the game is not final/known."""
    snap = row.get("pregame_snapshot") if isinstance(row.get("pregame_snapshot"), dict) else {}
    event_id = str(row.get("game_id") or snap.get("game_id") or "").strip()
    if not event_id:
        return None, "missing_event_id"
    if event_id not in summaries:
        summaries[event_id] = fetch(event_id)
    summary = summaries[event_id]
    if not _final(summary):
        return None, "game_not_final"
    candidate = _candidate(row)
    if pickgrader_server.parse_player_prop_pick(candidate) is None:
        return "pending", "nhl_player_prop_unparsed"
    result = pickgrader_server.grade_player_prop_pick(candidate, {}, summary)
    return result, candidate.get("grade_anomaly")


def _apply(row: dict[str, Any], summaries: dict[str, Any], fetch: SummaryFetch, since: str, report: dict[str, Any]) -> bool:
    if not _is_nhl_prop(row) or _slate(row) < since:
        return False
    result, anomaly = settle(row, summaries, fetch)
    if anomaly:
        report["anomalies"][anomaly] = report["anomalies"].get(anomaly, 0) + 1
    if result is None:
        return False
    old = str(row.get("result") or "pending")
    if old == result:
        return False
    report["transitions"][f"{old}->{result}"] = report["transitions"].get(f"{old}->{result}", 0) + 1
    row["result"] = result
    return True


def default_fetch(event_id: str) -> dict[str, Any] | None:
    url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary?event={event_id}"
    return pickgrader_server._fetch_json_url(url) or pickgrader_server.fetch_event_summary("hockey", "nhl", event_id)


def regrade(repo_root: Path = ROOT, *, since: str = "2026-09-29", write: bool = False,
            fetch: SummaryFetch = default_fetch, summaries: dict[str, Any] | None = None) -> dict[str, Any]:
    summaries = {} if summaries is None else summaries
    report: dict[str, Any] = {"ledger": {"transitions": {}, "anomalies": {}},
                              "model_cache": {"transitions": {}, "anomalies": {}}, "files_changed": []}
    ledger = load_team_prop_pregame_ledger(repo_root=repo_root)
    changed = False
    for row in ledger.get("records", []):
        changed |= _apply(row, summaries, fetch, since, report["ledger"])
    if changed and write:
        write_team_prop_pregame_ledger(ledger, repo_root=repo_root)
        report["files_changed"].append("data/calibration/team_prop_pregame_ledger")
    cache_dir = Path(repo_root) / "data" / "model_cache"
    for path in sorted(cache_dir.glob("20??-??-??.json")):
        if path.stem < since:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        file_changed = False
        buckets = []
        models = payload.get("models")
        if isinstance(models, dict) and isinstance(models.get("nhl"), dict):
            buckets.append(models["nhl"])
        if isinstance(payload.get("nhl"), dict) and payload.get("nhl") is not (models or {}).get("nhl"):
            buckets.append(payload["nhl"])
        for bucket in buckets:
            for pick in bucket.get("picks") or []:
                file_changed |= _apply(pick, summaries, fetch, since, report["model_cache"])
        if file_changed and write:
            path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            report["files_changed"].append(str(path.relative_to(repo_root)))
    if write:
        latest = cache_dir / "latest.json"
        if latest.exists():
            latest_date = str(json.loads(latest.read_text(encoding="utf-8")).get("date") or "")
            source = cache_dir / f"{latest_date}.json"
            if latest_date and source.exists() and source.read_bytes() != latest.read_bytes():
                latest.write_bytes(source.read_bytes())
                report["files_changed"].append(str(latest.relative_to(repo_root)))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default="2026-09-29")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--summary-cache", type=Path, help="Directory of cached ESPN summaries (<event>.json)")
    args = parser.parse_args()
    fetch: SummaryFetch = default_fetch
    if args.summary_cache:
        args.summary_cache.mkdir(parents=True, exist_ok=True)

        def fetch(event_id: str) -> dict[str, Any] | None:  # noqa: F811
            cached = args.summary_cache / f"{event_id}.json"
            if cached.exists():
                return json.loads(cached.read_text(encoding="utf-8"))
            summary = default_fetch(event_id)
            if _final(summary):
                cached.write_text(json.dumps(summary), encoding="utf-8")
            return summary
    print(json.dumps(regrade(since=args.since, write=args.write, fetch=fetch), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
