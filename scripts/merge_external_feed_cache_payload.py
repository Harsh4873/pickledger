#!/usr/bin/env python3
"""Merge generated external feed cache payloads into the latest cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_manifest import write_cache_manifest  # noqa: E402


MODEL_CACHE_DIR = Path("data/model_cache")
EXTERNAL_FEED_MODEL_KEYS = {
    "sportytrader",
    "sportytrader_nba",
    "sportytrader_nba_summer",
    "sportytrader_mlb",
    "sportytrader_wnba",
    "sportytrader_fifa_world_cup",
    "sportsgambler",
    "sportsgambler_nba",
    "sportsgambler_nba_summer",
    "sportsgambler_mlb",
    "sportsgambler_wnba",
    "sportsgambler_fifa_world_cup",
    "scores24_nba_summer",
    "scores24_wnba",
    "scores24_mlb",
    "scores24_fifa_world_cup",
}
SPLIT_EXTERNAL_FEED_LEGACY_KEYS = {"sportytrader", "sportsgambler"}
EXTERNAL_FEED_SPORT_KEYS = {
    "NBA": "nba",
    "NBA SUMMER": "nba_summer",
    "WNBA": "wnba",
    "MLB": "mlb",
    "FIFA WC": "fifa_world_cup",
}
EXTERNAL_FEED_SOURCE_LABELS = {
    "sportytrader": {
        "NBA": "SportyTraderNBA",
        "NBA SUMMER": "SportyTraderNBASummer",
        "WNBA": "SportyTraderWNBA",
        "MLB": "SportyTraderMLB",
        "FIFA WC": "SportyTraderFIFAWorldCup",
    },
    "sportsgambler": {
        "NBA": "SportsGamblerNBA",
        "NBA SUMMER": "SportsGamblerNBASummer",
        "WNBA": "SportsGamblerWNBA",
        "MLB": "SportsGamblerMLB",
        "FIFA WC": "SportsGamblerFIFAWorldCup",
    },
}
REQUIRED_TEAM_MODEL_KEYS = {
    "mlb_new",
    "mlb_inning",
    "mlb_first_five",
    "wnba",
    "nba",
    "nba_playoffs",
    "nba_summer",
    "fifa_world_cup",
}
PICK_METADATA_FIELDS = {"result", "start_time", "game_start_time", "pregame_snapshot"}
MARKET_ODDS_METADATA_FIELDS = {
    # Pregame market prices captured by scripts/market_odds.py.  Once a game
    # goes live the attach step skips it, so these captured pregame values
    # must survive later merges instead of being wiped by a regenerated pick.
    "market_odds_provider",
    "market_odds_captured_at",
    "market_updated_at",
    "market_home_odds",
    "market_away_odds",
    "market_draw_odds",
    "market_over_odds",
    "market_under_odds",
    "market_line",
    "selected_odds",
    "opposite_odds",
    "market_no_vig_selected_probability",
    "assumed_odds_replaced",
    "model_assumed_odds",
}
REPLACED_PRICE_FIELDS = ("odds", "pricing_type", "odds_source", "price_source", "market_priced")



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge generated feed cache JSON into data/model_cache.")
    parser.add_argument("generated", help="Path to the generated latest.json from refresh_external_feeds.py.")
    parser.add_argument("--cache-dir", default=str(MODEL_CACHE_DIR), help="Cache directory to update.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _seed_external_feeds_from_latest(latest_payload: dict[str, Any]) -> dict[str, Any]:
    seeded: dict[str, Any] = {}
    external_feeds = latest_payload.get("external_feeds")
    if isinstance(external_feeds, dict):
        seeded.update(external_feeds)
    models = latest_payload.get("models")
    if isinstance(models, dict):
        for key in EXTERNAL_FEED_MODEL_KEYS:
            if key in models and key not in seeded:
                seeded[key] = models[key]
    for key in EXTERNAL_FEED_MODEL_KEYS:
        if key in latest_payload and key not in seeded:
            seeded[key] = latest_payload[key]
    return seeded


def _current_payload(cache_dir: Path, date_iso: str) -> dict[str, Any]:
    date_payload = _read_json(cache_dir / f"{date_iso}.json")
    if date_payload and str(date_payload.get("date") or "") == date_iso:
        return date_payload
    latest_payload = _read_json(cache_dir / "latest.json")
    if latest_payload and str(latest_payload.get("date") or "") == date_iso:
        return latest_payload
    if latest_payload:
        seeded_feeds = _seed_external_feeds_from_latest(latest_payload)
        if seeded_feeds:
            return {"date": date_iso, "models": {}, "external_feeds": seeded_feeds}
    return {"date": date_iso, "models": {}}


def _feed_keys(generated: dict[str, Any]) -> set[str]:
    keys = set(EXTERNAL_FEED_MODEL_KEYS)
    external_feeds = generated.get("external_feeds")
    if isinstance(external_feeds, dict):
        keys.update(str(key) for key in external_feeds)
    return keys


def _legacy_feed_keys_replaced_by(generated: dict[str, Any]) -> set[str]:
    generated_keys: set[str] = set()
    models = generated.get("models")
    if isinstance(models, dict):
        generated_keys.update(str(key) for key in models)
    external_feeds = generated.get("external_feeds")
    if isinstance(external_feeds, dict):
        generated_keys.update(str(key) for key in external_feeds)
    generated_keys.update(str(key) for key in generated if key in EXTERNAL_FEED_MODEL_KEYS)
    return {
        legacy_key
        for legacy_key in SPLIT_EXTERNAL_FEED_LEGACY_KEYS
        if any(key.startswith(f"{legacy_key}_") for key in generated_keys)
    }


def _canonical_sport_label(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "nba": "NBA",
        "basketball": "NBA",
        "nba_summer": "NBA SUMMER",
        "nba_summer_league": "NBA SUMMER",
        "summer_league": "NBA SUMMER",
        "wnba": "WNBA",
        "mlb": "MLB",
        "baseball": "MLB",
        "fifa": "FIFA WC",
        "fifa_wc": "FIFA WC",
        "fifa_world_cup": "FIFA WC",
        "world_cup": "FIFA WC",
        "soccer": "FIFA WC",
        "football": "FIFA WC",
    }
    if normalized in aliases:
        return aliases[normalized]
    upper = raw.upper()
    if upper == "FIFA WORLD CUP":
        return "FIFA WC"
    return upper if upper in EXTERNAL_FEED_SPORT_KEYS else ""


def _split_feed_key(provider_key: str, sport: Any) -> str:
    sport_label = _canonical_sport_label(sport)
    sport_key = EXTERNAL_FEED_SPORT_KEYS.get(sport_label)
    return f"{provider_key}_{sport_key}" if sport_key else provider_key


def _split_source_label(provider_key: str, sport: Any) -> str:
    sport_label = _canonical_sport_label(sport)
    return EXTERNAL_FEED_SOURCE_LABELS.get(provider_key, {}).get(sport_label, provider_key)


def _split_legacy_bucket(provider_key: str, bucket: Any) -> dict[str, Any]:
    if not isinstance(bucket, dict):
        return {}
    split: dict[str, Any] = {}
    for raw_pick in bucket.get("picks") or []:
        if not isinstance(raw_pick, dict):
            continue
        split_key = _split_feed_key(provider_key, raw_pick.get("sport"))
        if split_key == provider_key:
            continue
        split_bucket = split.setdefault(
            split_key,
            {
                **bucket,
                "picks": [],
                "meta": {
                    **(bucket.get("meta") if isinstance(bucket.get("meta"), dict) else {}),
                    "feed": split_key,
                    "provider": provider_key,
                },
            },
        )
        pick = dict(raw_pick)
        pick["source"] = _split_source_label(provider_key, pick.get("sport"))
        split_bucket["picks"].append(pick)
    return split


def _split_legacy_buckets(provider_key: str, buckets: dict[str, Any]) -> dict[str, Any]:
    split: dict[str, Any] = {}
    for split_key, split_bucket in _split_legacy_bucket(provider_key, buckets.get(provider_key)).items():
        split[split_key] = _preserve_pick_metadata(split.get(split_key), split_bucket)
    return split


def _pick_key(pick: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(pick.get(key) or "").strip().lower()
        for key in ("source", "sport", "date", "pick", "matchup", "game")
    )


def _preserve_pick_metadata(current_bucket: Any, generated_bucket: Any) -> Any:
    if not isinstance(current_bucket, dict) or not isinstance(generated_bucket, dict):
        return generated_bucket
    current_picks = current_bucket.get("picks")
    generated_picks = generated_bucket.get("picks")
    if not isinstance(current_picks, list) or not isinstance(generated_picks, list):
        return generated_bucket
    def _kept_fields(pick: dict[str, Any]) -> dict[str, Any]:
        kept = {
            field: pick[field]
            for field in (*PICK_METADATA_FIELDS, *MARKET_ODDS_METADATA_FIELDS)
            if field in pick
        }
        if pick.get("assumed_odds_replaced") is True:
            # A real captured price must not be reverted to a regenerated
            # assumed price after the game has started.
            kept.update({field: pick[field] for field in REPLACED_PRICE_FIELDS if field in pick})
        return kept

    metadata = {
        _pick_key(pick): _kept_fields(pick)
        for pick in current_picks
        if isinstance(pick, dict)
    }
    merged = dict(generated_bucket)
    merged["picks"] = [
        {**pick, **metadata.get(_pick_key(pick), {})} if isinstance(pick, dict) else pick
        for pick in generated_picks
    ]
    return merged


def _merge_feed_buckets(
    current_buckets: dict[str, Any],
    generated_buckets: dict[str, Any],
    feed_keys: set[str],
) -> dict[str, Any]:
    merged = dict(current_buckets)
    for key in feed_keys:
        if key in generated_buckets:
            merged[key] = _preserve_pick_metadata(current_buckets.get(key), generated_buckets[key])
    return merged


def merge_payload(generated: dict[str, Any], cache_dir: Path) -> dict[str, Any]:
    date_iso = str(generated.get("date") or "").strip()
    if not date_iso:
        raise SystemExit("Generated external feed cache is missing date")

    current = _current_payload(cache_dir, date_iso)
    merged = dict(current)
    for key in ("date", "updatedAt", "externalFeedsUpdatedAt", "external_feed_errors"):
        if key in generated:
            merged[key] = generated[key]

    feed_keys = _feed_keys(generated)
    replaced_legacy_keys = _legacy_feed_keys_replaced_by(generated)
    for key in replaced_legacy_keys:
        merged.pop(key, None)

    current_models = current.get("models") if isinstance(current.get("models"), dict) else {}
    generated_models = generated.get("models") if isinstance(generated.get("models"), dict) else {}
    models = dict(current_models)
    for key in replaced_legacy_keys:
        for split_key, split_bucket in _split_legacy_buckets(key, current_models).items():
            models[split_key] = _preserve_pick_metadata(models.get(split_key), split_bucket)
        models.pop(key, None)
    for key in feed_keys:
        if key in replaced_legacy_keys:
            continue
        if key in generated_models:
            models[key] = _preserve_pick_metadata(models.get(key), generated_models[key])
    merged["models"] = models

    current_external = current.get("external_feeds") if isinstance(current.get("external_feeds"), dict) else {}
    generated_external = generated.get("external_feeds") if isinstance(generated.get("external_feeds"), dict) else {}
    current_external = {
        key: value
        for key, value in current_external.items()
        if key not in replaced_legacy_keys
    }
    for key in replaced_legacy_keys:
        source_buckets = current.get("external_feeds") if isinstance(current.get("external_feeds"), dict) else {}
        for split_key, split_bucket in _split_legacy_buckets(key, source_buckets).items():
            current_external[split_key] = _preserve_pick_metadata(current_external.get(split_key), split_bucket)
    generated_external = {
        key: value
        for key, value in generated_external.items()
        if key not in replaced_legacy_keys
    }
    if current_external or generated_external:
        merged["external_feeds"] = _merge_feed_buckets(current_external, generated_external, feed_keys)

    for key in feed_keys:
        if key in replaced_legacy_keys:
            continue
        if key in generated:
            merged[key] = _preserve_pick_metadata(current.get(key), generated[key])
    for key in replaced_legacy_keys:
        for split_key, split_bucket in _split_legacy_buckets(key, current).items():
            if split_key in merged:
                merged[split_key] = _preserve_pick_metadata(split_bucket, merged[split_key])
            else:
                merged[split_key] = split_bucket

    return merged


def write_merged_payload(merged: dict[str, Any], cache_dir: Path) -> bool:
    date_iso = str(merged["date"])
    models = merged.get("models") if isinstance(merged.get("models"), dict) else {}
    latest_updated = all(
        isinstance(models.get(key), dict) and models[key].get("ok") is True
        for key in REQUIRED_TEAM_MODEL_KEYS
    )

    _write_json(cache_dir / f"{date_iso}.json", merged)
    if latest_updated:
        _write_json(cache_dir / "latest.json", merged)
    write_cache_manifest(cache_dir)
    return latest_updated


def main() -> int:
    args = _parse_args()
    generated_path = Path(args.generated)
    cache_dir = Path(args.cache_dir)
    generated = _read_json(generated_path)
    if not generated:
        raise SystemExit(f"Could not read generated external feed cache: {generated_path}")

    merged = merge_payload(generated, cache_dir)
    date_iso = str(merged["date"])
    latest_updated = write_merged_payload(merged, cache_dir)
    print(json.dumps({
        "date": date_iso,
        "models": sorted((merged.get("models") or {}).keys()),
        "generated": str(generated_path),
        "latestUpdated": latest_updated,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
