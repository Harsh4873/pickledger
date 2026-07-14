#!/usr/bin/env python3
"""Refresh scheduled external pick feeds for GitHub Actions and Pages cache."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = REPO_ROOT / "data" / "model_cache"
sys.path.insert(0, str(REPO_ROOT))

import pickgrader_server as server  # noqa: E402
from scripts.cache_manifest import write_cache_manifest  # noqa: E402
from scripts.market_odds import apply_market_odds_to_payload  # noqa: E402
from scripts.merge_external_feed_cache_payload import merge_payload  # noqa: E402
from scripts.pick_calibration import apply_calibration_to_payload  # noqa: E402
from scripts.scrapers.scores24_scraper import (  # noqa: E402
    run_scores24_fifa_world_cup,
    run_scores24_mlb,
    run_scores24_wnba,
)


FEED_RUNNERS: dict[str, Callable[[str, list[str]], dict[str, Any]]] = {
    "sportytrader": server.run_sportytrader_scraper,
    "sportsgambler": server.run_sportsgambler_scraper,
    "scores24_wnba": run_scores24_wnba,
    "scores24_mlb": run_scores24_mlb,
    "scores24_fifa_world_cup": run_scores24_fifa_world_cup,
}
SPLIT_PROVIDER_FEEDS = {"sportytrader", "sportsgambler"}
SPLIT_PROVIDER_MODEL_KEYS = {
    "sportytrader": (
        "sportytrader_nba",
        "sportytrader_mlb",
        "sportytrader_wnba",
        "sportytrader_fifa_world_cup",
    ),
    "sportsgambler": (
        "sportsgambler_nba",
        "sportsgambler_mlb",
        "sportsgambler_wnba",
        "sportsgambler_fifa_world_cup",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external feed scrapers and publish cache artifacts.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    parser.add_argument(
        "--feeds",
        default="sportytrader,sportsgambler",
        help="Comma-separated feeds to refresh, or 'all'.",
    )
    parser.add_argument("--sports", default="nba,mlb,wnba,fifa_world_cup", help="Comma-separated sports passed to each feed scraper.")
    parser.add_argument("--skip-firestore", action="store_true", help="Write JSON only; useful for local checks.")
    return parser.parse_args()


def _csv_values(raw: str) -> list[str]:
    return [item.strip().lower() for item in str(raw or "").split(",") if item.strip()]


def _selected_feed_keys(raw: str) -> list[str]:
    value = str(raw or "").strip().lower()
    if value == "all":
        return list(FEED_RUNNERS)
    keys = _csv_values(value)
    unknown = [key for key in keys if key not in FEED_RUNNERS]
    if unknown:
        raise SystemExit(f"Unknown feed key(s): {', '.join(unknown)}")
    return keys or list(FEED_RUNNERS)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _base_cache_payload(date_iso: str) -> dict[str, Any]:
    date_path = MODEL_CACHE_DIR / f"{date_iso}.json"
    payload = _read_json(date_path)
    if payload:
        return payload
    latest = _read_json(MODEL_CACHE_DIR / "latest.json")
    if latest and str(latest.get("date") or "") == date_iso:
        return latest
    return {
        "date": date_iso,
        "models": {},
    }


def _normalize_feed_result(
    feed_key: str,
    result: Any,
    date_iso: str,
    sports: list[str],
    now_iso: str,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        result = {"ok": False, "error": str(result)}

    normalized = dict(result)
    picks = normalized.get("picks")
    if not isinstance(picks, list):
        picks = []
    meta = normalized.get("meta") if isinstance(normalized.get("meta"), dict) else {}
    normalized["date"] = str(normalized.get("date") or date_iso)
    normalized["updatedAt"] = now_iso
    normalized["generatedAt"] = now_iso
    normalized["generatedBy"] = "github-actions:external-feed-refresh"
    normalized["picks"] = picks
    normalized["meta"] = {
        **meta,
        "updatedAt": now_iso,
        "date": date_iso,
        "from": "github-actions",
        "leagues": ",".join(sports),
        "feed": feed_key,
    }
    if "note" not in normalized:
        normalized["note"] = f"Scheduled {feed_key} refresh returned {len(picks)} pick(s)."
    return normalized


def _empty_split_bucket(
    feed_key: str,
    split_key: str,
    result: dict[str, Any],
    date_iso: str,
    sports: list[str],
    now_iso: str,
) -> dict[str, Any]:
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    return {
        **result,
        "picks": [],
        "date": str(result.get("date") or date_iso),
        "updatedAt": now_iso,
        "generatedAt": now_iso,
        "generatedBy": "github-actions:external-feed-refresh",
        "meta": {
            **meta,
            "updatedAt": now_iso,
            "date": date_iso,
            "from": "github-actions",
            "leagues": ",".join(sports),
            "feed": split_key,
            "provider": feed_key,
        },
    }


def _split_provider_result(
    feed_key: str,
    result: dict[str, Any],
    date_iso: str,
    sports: list[str],
    now_iso: str,
) -> dict[str, dict[str, Any]]:
    if feed_key not in SPLIT_PROVIDER_FEEDS:
        return {feed_key: result}

    registered_keys = set(SPLIT_PROVIDER_MODEL_KEYS.get(feed_key, ()))
    split_keys = {
        key
        for sport in sports
        for key in (server.external_feed_model_key(feed_key, sport),)
        if key in registered_keys
    }
    split_keys.discard(feed_key)
    buckets = {
        split_key: _empty_split_bucket(feed_key, split_key, result, date_iso, sports, now_iso)
        for split_key in sorted(split_keys)
    }

    for raw_pick in result.get("picks") or []:
        if not isinstance(raw_pick, dict):
            continue
        split_key = server.external_feed_model_key(feed_key, raw_pick.get("sport"))
        if split_key == feed_key:
            split_key = f"{feed_key}_unknown"
        bucket = buckets.setdefault(
            split_key,
            _empty_split_bucket(feed_key, split_key, result, date_iso, sports, now_iso),
        )
        pick = dict(raw_pick)
        pick["source"] = server.external_feed_source_label(feed_key, pick.get("sport"))
        bucket["picks"].append(pick)

    for split_key, bucket in buckets.items():
        bucket["note"] = f"Scheduled {split_key} refresh returned {len(bucket['picks'])} pick(s)."
        bucket["meta"] = {
            **(bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}),
            "pick_count": len(bucket["picks"]),
        }
    return buckets


def _write_json_cache(date_iso: str, payload: dict[str, Any]) -> None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged = merge_payload(payload, MODEL_CACHE_DIR)
    # Attach real pregame market prices so scraped picks carry a verifiable
    # two-sided baseline next to their own posted odds.
    apply_market_odds_to_payload(merged)
    for target in (MODEL_CACHE_DIR / f"{date_iso}.json", MODEL_CACHE_DIR / "latest.json"):
        with target.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    write_cache_manifest(MODEL_CACHE_DIR)


def main() -> int:
    args = _parse_args()
    date_iso, _ = server._parse_model_date_arg(args.date or None)  # noqa: SLF001
    feeds = _selected_feed_keys(args.feeds)
    sports = _csv_values(args.sports) or ["nba", "mlb", "wnba", "fifa_world_cup"]
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    print(f"[external-feeds] refreshing {', '.join(feeds)} for {date_iso} sports={','.join(sports)}")
    payload = _base_cache_payload(date_iso)
    payload["date"] = date_iso
    payload["updatedAt"] = now_iso
    payload["externalFeedsUpdatedAt"] = now_iso
    payload.setdefault("models", {})

    errors: list[str] = []
    success_count = 0
    results: dict[str, Any] = {}
    for feed_key in feeds:
        try:
            raw_result = FEED_RUNNERS[feed_key](date_iso, sports)
        except Exception as exc:  # pragma: no cover - defensive for scheduled jobs
            raw_result = {"ok": False, "error": str(exc)}

        result = _normalize_feed_result(feed_key, raw_result, date_iso, sports, now_iso)
        split_results = _split_provider_result(feed_key, result, date_iso, sports, now_iso)
        ok = bool(result.get("ok"))
        pick_count = len(result.get("picks") or [])
        print(f"[external-feeds] {feed_key}: {'ok' if ok else 'error'} ({pick_count} pick(s))")
        if ok:
            success_count += 1
            results.update(split_results)
            if feed_key in SPLIT_PROVIDER_FEEDS:
                payload["models"].pop(feed_key, None)
                payload.pop(feed_key, None)
            for split_key, split_result in split_results.items():
                payload["models"][split_key] = split_result
                payload[split_key] = split_result
        else:
            errors.append(f"{feed_key}: {result.get('error') or 'unknown error'}")

    if errors:
        payload["external_feed_errors"] = errors
    else:
        payload.pop("external_feed_errors", None)
    external_feeds = payload.get("external_feeds") if isinstance(payload.get("external_feeds"), dict) else {}
    external_feeds = dict(external_feeds)
    for feed_key in feeds:
        if feed_key in SPLIT_PROVIDER_FEEDS:
            external_feeds.pop(feed_key, None)
    payload["external_feeds"] = {**external_feeds, **results}

    apply_calibration_to_payload(payload)
    _write_json_cache(date_iso, payload)
    if args.skip_firestore:
        print("[external-feeds] skipped Firestore write")
    else:
        server._write_admin_picks_cache(date_iso, payload)  # noqa: SLF001
        print(f"[external-feeds] wrote Firestore admin_picks/{date_iso}")
    print(f"[external-feeds] wrote {MODEL_CACHE_DIR / f'{date_iso}.json'}")
    print(f"[external-feeds] wrote {MODEL_CACHE_DIR / 'latest.json'}")
    print(json.dumps({"ok": success_count > 0, "date": date_iso, "feeds": feeds, "errors": errors}, indent=2))
    return 0 if success_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
