#!/usr/bin/env python3
"""Refresh scheduled external pick feeds for GitHub Actions and Pages cache."""

from __future__ import annotations

import argparse
import json
import os
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
from scripts.scrapers.forebet_scraper import (  # noqa: E402
    run_forebet_cfb,
    run_forebet_mlb,
    run_forebet_mls,
    run_forebet_nba,
    run_forebet_nhl,
    run_forebet_nfl,
    run_forebet_wnba,
)
from scripts.scrapers.scores24_scraper import (  # noqa: E402
    run_scores24_cfb,
    run_scores24_fifa_world_cup,
    run_scores24_mlb,
    run_scores24_nba,
    run_scores24_nba_summer,
    run_scores24_nfl,
    run_scores24_wnba,
)
from scripts.scrapers.tennis_scraper import (  # noqa: E402
    run_scores24_tennis,
    run_tennistonic_tennis,
)


FEED_RUNNERS: dict[str, Callable[[str, list[str]], dict[str, Any]]] = {
    "sportytrader": server.run_sportytrader_scraper,
    "sportsgambler": server.run_sportsgambler_scraper,
    "scores24_nba_summer": run_scores24_nba_summer,
    "scores24_nba": run_scores24_nba,
    "scores24_wnba": run_scores24_wnba,
    "scores24_mlb": run_scores24_mlb,
    "scores24_fifa_world_cup": run_scores24_fifa_world_cup,
    "scores24_cfb": run_scores24_cfb,
    "scores24_nfl": run_scores24_nfl,
    "forebet_mls": run_forebet_mls,
    "forebet_mlb": run_forebet_mlb,
    "forebet_wnba": run_forebet_wnba,
    "forebet_cfb": run_forebet_cfb,
    "forebet_nfl": run_forebet_nfl,
    "forebet_nhl": run_forebet_nhl,
    "forebet_nba": run_forebet_nba,
    "tennistonic_tennis": run_tennistonic_tennis,
    "scores24_tennis": run_scores24_tennis,
}
SPLIT_PROVIDER_FEEDS = {"sportytrader", "sportsgambler"}
SPLIT_PROVIDER_MODEL_KEYS = {
    "sportytrader": (
        "sportytrader_nba",
        "sportytrader_nba_summer",
        "sportytrader_mlb",
        "sportytrader_wnba",
        "sportytrader_fifa_world_cup",
        "sportytrader_cfb",
        "sportytrader_nfl",
    ),
    "sportsgambler": (
        "sportsgambler_nba",
        "sportsgambler_nba_summer",
        "sportsgambler_mlb",
        "sportsgambler_wnba",
        "sportsgambler_fifa_world_cup",
        "sportsgambler_cfb",
        "sportsgambler_nfl",
    ),
}
NON_NBA_SPLIT_FEEDS = {
    f"{provider}_{sport}"
    for provider in SPLIT_PROVIDER_FEEDS
    for sport in ("nba", "mlb", "wnba", "fifa_world_cup", "cfb", "nfl")
}


def _runtime_origin() -> str:
    return "github-actions" if os.environ.get("GITHUB_ACTIONS", "").strip().lower() == "true" else "local"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run external feed scrapers and publish cache artifacts.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    parser.add_argument(
        "--feeds",
        default="sportytrader,sportsgambler",
        help="Comma-separated feeds to refresh, or 'all'.",
    )
    parser.add_argument(
        "--sports",
        default="nba,mlb,wnba,cfb,nfl",
        help="Comma-separated sports passed to each feed scraper.",
    )
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
    previous_feeds = {}
    if latest:
        keys = set(FEED_RUNNERS)
        for split_keys in SPLIT_PROVIDER_MODEL_KEYS.values():
            keys.update(split_keys)
        previous_feeds = {
            key: bucket
            for key in keys
            if (bucket := _previous_feed_bucket(latest, key)) is not None
        }
    return {
        "date": date_iso,
        "models": {},
        "external_feeds": previous_feeds,
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
    origin = _runtime_origin()
    normalized["generatedBy"] = f"{origin}:external-feed-refresh"
    normalized["picks"] = picks
    normalized["meta"] = {
        **meta,
        "updatedAt": now_iso,
        "date": date_iso,
        "from": origin,
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
    origin = _runtime_origin()
    return {
        **result,
        "picks": [],
        "date": str(result.get("date") or date_iso),
        "updatedAt": now_iso,
        "generatedAt": now_iso,
        "generatedBy": f"{origin}:external-feed-refresh",
        "meta": {
            **meta,
            "updatedAt": now_iso,
            "date": date_iso,
            "from": origin,
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
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    sport_errors = meta.get("sportErrors") if isinstance(meta.get("sportErrors"), dict) else {}
    no_preview_evidence = meta.get("noPreviewEvidence") or {}

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
        sport_key = ""
        prefix = f"{feed_key}_"
        if split_key.startswith(prefix):
            sport_key = split_key[len(prefix):]
        sport_error = sport_errors.get(sport_key)
        # Provider-level diagnostics contain all requested sports. Keep the
        # split bucket's error list scoped so a CFB outage does not label the
        # successfully published MLB feed as failed.
        if result.get("ok"):
            bucket["errors"] = [sport_error] if sport_error else []
            if not sport_error:
                bucket.pop("error", None)
        if sport_error:
            bucket["ok"] = False
            bucket["error"] = sport_error
        bucket["note"] = f"Scheduled {split_key} refresh returned {len(bucket['picks'])} pick(s)."
        if not sport_error and not bucket["picks"] and no_preview_evidence.get(sport_key):
            reasons = "; ".join(item["reason"] for item in no_preview_evidence[sport_key])
            bucket["note"] = f"No previews published for {split_key} on {date_iso}: {reasons}"
            bucket["meta"]["providerStatus"] = "no_previews_published"
        bucket["meta"] = {
            **(bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}),
            "pick_count": len(bucket["picks"]),
        }
    return buckets


def _today_result_picks(result: dict[str, Any], date_iso: str) -> list[dict[str, Any]]:
    result_date = str(result.get("date") or date_iso).strip()
    picks: list[dict[str, Any]] = []
    for pick in result.get("picks") if isinstance(result.get("picks"), list) else []:
        if not isinstance(pick, dict) or not pick.get("pick"):
            continue
        if str(pick.get("date") or result_date) != date_iso:
            continue
        picks.append(pick)
    return picks


def _normalize_forebet_coverage(result: dict[str, Any], date_iso: str) -> dict[str, Any]:
    """Use the full official slate, including games Forebet did not publish."""
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    official = meta.get("officialMatchups")
    if not isinstance(official, int) or isinstance(official, bool):
        return result
    count = len(_today_result_picks(result, date_iso))
    missing = list(meta.get("missingMatchups") or [])
    for matchup in meta.get("unpublishedMatchups") or []:
        if matchup not in missing:
            missing.append(matchup)
    result = dict(result)
    result["meta"] = {**meta, "expectedMatchups": official,
                      "matchedPicks": count, "missingMatchups": missing}
    if result.get("ok") and (count != official or missing or meta.get("blockedUrls")):
        result["ok"] = False
        result["error"] = f"Forebet incomplete official slate coverage: {count}/{official}"
    return result


def _mark_failed_retained_bucket(
    bucket: dict[str, Any], result: dict[str, Any], date_iso: str, now_iso: str,
) -> dict[str, Any]:
    """Separate the last real snapshot clock from an unsuccessful retry."""
    if result.get("ok"):
        return bucket
    bucket["ok"] = False
    bucket["refreshStatus"] = "error"
    bucket["lastError"] = str(result.get("error") or "Source refresh incomplete")
    bucket["lastAttemptAt"] = now_iso
    bucket["lastAttemptDate"] = date_iso
    if str(bucket.get("date") or "") == date_iso:
        meta = bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}
        attempt_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        official = attempt_meta.get("officialMatchups")
        if not isinstance(official, int) or isinstance(official, bool):
            official = meta.get("officialMatchups")
        retained = _today_result_picks(bucket, date_iso)
        retained_matchups = {str(pick.get("matchup") or "") for pick in retained}
        missing = attempt_meta.get("missingMatchups", meta.get("missingMatchups", []))
        missing = [name for name in missing if name not in retained_matchups] if isinstance(missing, list) else []
        bucket["meta"] = {
            **meta,
            **({"officialMatchups": official, "expectedMatchups": official}
               if isinstance(official, int) else {}),
            "matchedPicks": len(retained),
            "missingMatchups": missing,
            "updatedAt": bucket.get("updatedAt", meta.get("updatedAt")),
            "date": date_iso,
        }
    return bucket


def _resume_same_day_partial_picks(previous: Any, result: dict[str, Any], date_iso: str) -> dict[str, Any]:
    """Carry verified matchup picks across incomplete attempts for one date."""
    if (not isinstance(previous, dict) or result.get("ok")
            or str(previous.get("date") or "") != date_iso
            or str(result.get("date") or "") != date_iso):
        return result
    old = _today_result_picks(previous, date_iso)
    new = _today_result_picks(result, date_iso)
    if not old or not new:
        return result
    carried = {
        str(pick.get("matchup")): pick
        for pick in old
        if pick.get("matchup")
    }
    current = {
        str(pick.get("matchup")): pick
        for pick in new
        if pick.get("matchup")
    }
    if not carried or not current or len(carried) != len(old) or len(current) != len(new):
        return result
    resumed = len(carried.keys() - current.keys())
    if not resumed:
        return result
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    return {
        **result,
        "picks": list({**carried, **current}.values()),
        "meta": {**meta, "resumedPicks": resumed},
    }


def _record_forebet_attempt(
    previous: Any, result: dict[str, Any], date_iso: str, now_iso: str,
) -> dict[str, Any]:
    if isinstance(previous, dict) and str(previous.get("date") or "") == date_iso:
        previous = _normalize_forebet_coverage(previous, date_iso)
    result = _resume_same_day_partial_picks(previous, result, date_iso)
    bucket = _mark_failed_retained_bucket(
        _record_feed_attempt(previous, result, date_iso, now_iso), result, date_iso, now_iso,
    )
    if result.get("ok"):
        bucket.pop("lastAttemptMeta", None)
    else:
        attempt_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
        bucket["lastAttemptMeta"] = {
            key: attempt_meta[key]
            for key in ("officialMatchups", "matchedPicks", "missingMatchups", "blockedUrls", "blockedUrl")
            if key in attempt_meta
        }
    return bucket


def _record_research_feed_attempt(
    previous: Any, result: dict[str, Any], date_iso: str, now_iso: str,
) -> dict[str, Any]:
    if isinstance(previous, dict) and str(previous.get("date") or "") == date_iso:
        prior_meta = previous.get("meta") if isinstance(previous.get("meta"), dict) else {}
        if (prior_meta.get("unavailableMatchups") or prior_meta.get("unattemptedMatchups")
                or prior_meta.get("timedOut") or prior_meta.get("interrupted")):
            previous = {**previous, "ok": False}
            previous.pop("lastSuccessAt", None)
    return _mark_failed_retained_bucket(
        _record_feed_attempt(previous, result, date_iso, now_iso), result, date_iso, now_iso,
    )


def _record_tennistonic_attempt(
    previous: Any, result: dict[str, Any], date_iso: str, now_iso: str,
) -> dict[str, Any]:
    return _record_research_feed_attempt(
        previous, _resume_same_day_partial_picks(previous, result, date_iso), date_iso, now_iso,
    )


def _incomplete_scores24_football_bucket(bucket: Any, date_iso: str) -> bool:
    """True for same-day optional Scores24 CFB/NFL/NBA/tennis buckets still filling.

    Tennis soft-timeouts used to set expectedMatchups=len(picks), which hid the
    gap versus officialMatchups and let salvage treat a 2-pick checkpoint as
    complete. Count tennis (and CFB/NFL/NBA) incomplete whenever the official slate
    outruns matched picks, missing rows remain, or timedOut/interrupted is set.
    """
    if not isinstance(bucket, dict) or str(bucket.get("date") or "") != date_iso:
        return False
    meta = bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}
    picks = _today_result_picks(bucket, date_iso)
    feed = str(meta.get("feed") or "")
    sources = {str(pick.get("source") or "") for pick in picks}
    is_optional = (
        feed in {"scores24_cfb", "scores24_nfl", "scores24_nba", "scores24_tennis"}
        or bool(sources & {"Scores24CFB", "Scores24NFL", "Scores24NBA", "Scores24Tennis"})
    )
    if not is_optional:
        return False
    if meta.get("timedOut") or meta.get("interrupted"):
        return True
    unattempted = meta.get("unattemptedMatchups")
    if isinstance(unattempted, list) and unattempted:
        return True
    official = meta.get("officialMatchups")
    expected = meta.get("expectedMatchups")
    missing = meta.get("missingMatchups")
    return (
        isinstance(official, int) and official > len(picks)
        or isinstance(expected, int) and expected > len(picks)
        or isinstance(missing, list) and bool(missing)
    )


def _record_feed_attempt(
    previous: Any,
    result: dict[str, Any],
    date_iso: str,
    now_iso: str,
) -> dict[str, Any]:
    """Retain the last successful snapshot without hiding a later failure.

    Same-day failed retries keep the larger same-day partial snapshot. A newer
    day's partial scrape (matched picks for date_iso, even when ok=False)
    replaces yesterday's successful snapshot so an optional CFB hang cannot
    leave yesterday's bucket as the live research feed.
    """
    if result.get("ok"):
        bucket = dict(result)
        bucket.pop("lastError", None)
        bucket["lastSuccessAt"] = now_iso
        bucket["refreshStatus"] = "ok"
        bucket["lastAttemptAt"] = now_iso
        bucket["lastAttemptDate"] = date_iso
        return bucket

    today_picks = _today_result_picks(result, date_iso)
    previous_date = str((previous or {}).get("date") or "").strip() if isinstance(previous, dict) else ""
    previous_picks = previous.get("picks") if isinstance(previous, dict) else []
    previous_today_picks = (
        _today_result_picks(previous, date_iso)
        if isinstance(previous, dict) and previous_date == date_iso
        else []
    )
    previous_incomplete = _incomplete_scores24_football_bucket(previous, date_iso)
    result_incomplete = _incomplete_scores24_football_bucket(result, date_iso)
    previous_same_day_ok = (
        isinstance(previous, dict)
        and previous.get("ok")
        and not previous_incomplete
        and not result_incomplete
        and previous_date == date_iso
        and isinstance(previous_picks, list)
        and len(previous_picks) >= len(today_picks)
    )
    if (
        str(result.get("date") or date_iso).strip() == date_iso
        and today_picks
        and not previous_same_day_ok
        and len(today_picks) >= len(previous_today_picks)
    ):
        bucket = dict(result)
        bucket["picks"] = today_picks
        bucket["date"] = date_iso
        bucket["ok"] = False
        if previous_incomplete:
            prior_meta = previous.get("meta") if isinstance(previous.get("meta"), dict) else {}
            result_meta = bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}
            meta = {**prior_meta, **result_meta}
            if isinstance(meta.get("officialMatchups"), int):
                meta["expectedMatchups"] = meta["officialMatchups"]
                meta["matchedPicks"] = len(today_picks)
            bucket["meta"] = meta
        bucket["refreshStatus"] = "error"
        bucket["lastError"] = str(result.get("error") or "Source refresh incomplete")
        bucket["lastAttemptAt"] = now_iso
        bucket["lastAttemptDate"] = date_iso
        return bucket

    # A failed fetch must not erase already published picks, or redatestamp
    # yesterday's rows as today's. A prior bucket can be ok=false after a
    # blocked retry and still hold those rows (lastSuccessAt stays set).
    # Attempt freshness is separate from the date of the last collected snapshot.
    previous_published = isinstance(previous_picks, list) and any(
        isinstance(pick, dict) and pick.get("pick") for pick in previous_picks
    )
    has_previous = isinstance(previous, dict) and (
        previous.get("ok") or bool(previous_today_picks) or previous_published
    )
    bucket = dict(previous if has_previous else result)
    if has_previous and previous.get("ok") and not previous_incomplete:
        bucket.setdefault("lastSuccessAt", previous.get("updatedAt") or previous.get("generatedAt"))
    if previous_today_picks and (not previous.get("ok") or previous_incomplete or result_incomplete):
        bucket["ok"] = False
        if previous_incomplete or result_incomplete:
            bucket.pop("lastSuccessAt", None)
            previous_meta = bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}
            result_meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
            meta = {**previous_meta, **result_meta}
            official = meta.get("officialMatchups")
            if isinstance(official, int):
                meta["expectedMatchups"] = official
                meta["matchedPicks"] = len(previous_today_picks)
            bucket["meta"] = meta
    bucket["refreshStatus"] = "error"
    bucket["lastError"] = str(result.get("error") or "Source refresh failed")
    bucket["lastAttemptAt"] = now_iso
    bucket["lastAttemptDate"] = date_iso
    return bucket


def _previous_feed_bucket(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    for container_key in ("external_feeds", "models"):
        container = payload.get(container_key)
        if isinstance(container, dict) and isinstance(container.get(key), dict):
            return container[key]
    bucket = payload.get(key)
    return bucket if isinstance(bucket, dict) else None


def _write_run_summary(date_iso: str, results: dict[str, Any], errors: list[str]) -> None:
    """Expose partial outages in Actions even when other feeds published."""
    for error in errors:
        if _runtime_origin() == "github-actions":
            escaped = error.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
            print(f"::warning::{escaped}")
        else:
            print(f"[external-feeds] warning: {error}")
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    lines = [
        f"### External sources for {date_iso}",
        "",
        "| Source | Latest attempt | Published date | Picks | Detail |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for key, bucket in sorted(results.items()):
        detail = str(bucket.get("lastError") or bucket.get("note") or "").replace("|", "\\|")
        detail = " ".join(detail.splitlines())
        lines.append(
            f"| {key} | {bucket['refreshStatus']} | {bucket.get('date') or '—'} | "
            f"{len(bucket.get('picks') or [])} | {detail} |"
        )
    with Path(summary_path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _write_json_cache(date_iso: str, payload: dict[str, Any]) -> dict[str, Any]:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    merged = merge_payload(payload, MODEL_CACHE_DIR)
    # Attach real pregame market prices so scraped picks carry a verifiable
    # two-sided baseline next to their own posted odds. Calibration then runs
    # against those observed prices; it is idempotent because it restarts from
    # each pick's raw probability.
    apply_market_odds_to_payload(merged)
    apply_calibration_to_payload(merged)
    merged["publishedAt"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    for target in (MODEL_CACHE_DIR / f"{date_iso}.json", MODEL_CACHE_DIR / "latest.json"):
        with target.open("w", encoding="utf-8") as handle:
            json.dump(merged, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
    write_cache_manifest(MODEL_CACHE_DIR)
    return merged


def _forebet_cloudflare_block(feed_key: str, result: dict[str, Any]) -> bool:
    """Recognize the scraper's explicit listing-block result, before retention."""
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    blocked = meta.get("blockedUrls")
    error = str(result.get("error") or "").lower()
    return (
        feed_key.startswith("forebet_")
        and result.get("ok") is False
        and isinstance(blocked, int) and blocked > 0
        and error.endswith((": listing fetch blocked by cloudflare", ": listing blocked by cloudflare"))
    )


def main() -> int:
    args = _parse_args()
    date_iso, _ = server._parse_model_date_arg(args.date or None)  # noqa: SLF001
    feeds = _selected_feed_keys(args.feeds)
    sports = _csv_values(args.sports) or [
        "nba",
        "nba_summer",
        "mlb",
        "wnba",
        "fifa_world_cup",
        "cfb",
        "nfl",
    ]
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    print(f"[external-feeds] refreshing {', '.join(feeds)} for {date_iso} sports={','.join(sports)}")
    payload = _base_cache_payload(date_iso)
    payload["date"] = date_iso
    payload["updatedAt"] = now_iso
    payload["externalFeedsUpdatedAt"] = now_iso
    payload.setdefault("models", {})

    errors: list[str] = []
    success_count = 0
    forebet_blocks = 0
    results: dict[str, Any] = {}
    for feed_key in feeds:
        try:
            raw_result = FEED_RUNNERS[feed_key](date_iso, sports)
        except Exception as exc:  # pragma: no cover - defensive for scheduled jobs
            raw_result = {"ok": False, "error": str(exc)}

        result = _normalize_feed_result(feed_key, raw_result, date_iso, sports, now_iso)
        if feed_key.startswith("forebet_"):
            result = _normalize_forebet_coverage(result, date_iso)
        split_results = _split_provider_result(feed_key, result, date_iso, sports, now_iso)
        ok = bool(result.get("ok"))
        pick_count = len(result.get("picks") or [])
        print(f"[external-feeds] {feed_key}: {'ok' if ok else 'error'} ({pick_count} pick(s))")
        if ok:
            success_count += 1
        elif _forebet_cloudflare_block(feed_key, result):
            forebet_blocks += 1
        if feed_key in SPLIT_PROVIDER_FEEDS:
            payload["models"].pop(feed_key, None)
            payload.pop(feed_key, None)
        for split_key, split_result in split_results.items():
            if not split_result.get("ok"):
                errors.append(f"{split_key}: {split_result.get('error') or 'unknown error'}")
            record_attempt = (
                _record_forebet_attempt if split_key.startswith("forebet_")
                else _record_tennistonic_attempt if split_key == "tennistonic_tennis"
                else _record_research_feed_attempt if split_key in NON_NBA_SPLIT_FEEDS
                else _record_feed_attempt
            )
            bucket = record_attempt(
                _previous_feed_bucket(payload, split_key), split_result, date_iso, now_iso,
            )
            results[split_key] = bucket
            payload["models"][split_key] = bucket
            payload[split_key] = bucket

    external_feeds = payload.get("external_feeds") if isinstance(payload.get("external_feeds"), dict) else {}
    external_feeds = dict(external_feeds)
    for feed_key in feeds:
        if feed_key in SPLIT_PROVIDER_FEEDS:
            external_feeds.pop(feed_key, None)
    payload["external_feeds"] = {**external_feeds, **results}
    # Local publishers invoke Forebet and TennisTonic one feed at a time. A
    # later successful sibling must not erase an earlier same-day source error.
    for key, bucket in payload["external_feeds"].items():
        if key in results or not (key.startswith("forebet_") or key == "tennistonic_tennis"):
            continue
        if (isinstance(bucket, dict) and bucket.get("refreshStatus") == "error"
                and bucket.get("lastAttemptDate") == date_iso):
            errors.append(f"{key}: {bucket.get('lastError') or 'source refresh incomplete'}")
    # An explicit empty list clears repaired errors during merge.
    payload["external_feed_errors"] = errors

    _write_run_summary(date_iso, results, errors)
    payload = _write_json_cache(date_iso, payload)
    if args.skip_firestore:
        print("[external-feeds] skipped Firestore write")
    else:
        server._write_admin_picks_cache(date_iso, payload)  # noqa: SLF001
        print(f"[external-feeds] wrote Firestore admin_picks/{date_iso}")
    print(f"[external-feeds] wrote {MODEL_CACHE_DIR / f'{date_iso}.json'}")
    print(f"[external-feeds] wrote {MODEL_CACHE_DIR / 'latest.json'}")
    # Forebet blocks hosted runners. A targeted retry must publish its warnings
    # without failing solely because it selected no other providers. Local
    # publishers still need a failed exit status to detect an unsuccessful fetch.
    warning_only = _runtime_origin() == "github-actions" and forebet_blocks == len(feeds)
    print(json.dumps({"ok": success_count > 0, "warning_only": warning_only,
                      "date": date_iso, "feeds": feeds, "errors": errors}, indent=2))
    return 0 if success_count or warning_only else 1


if __name__ == "__main__":
    raise SystemExit(main())
