#!/usr/bin/env python3
"""Merge generated player-prop models while preserving committed grades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from cache_manifest import write_cache_manifest
except ModuleNotFoundError:  # pragma: no cover - exercised when tests import by file path
    from scripts.cache_manifest import write_cache_manifest


PLAYER_PROPS_CACHE_DIR = Path("data/player_props_cache")
PLAYER_PROPS_SNAPSHOT_DIR = Path("data/player_props_snapshots")
CONSENSUS_METADATA_PATH = Path("player_props/artifacts/player_props_consensus_metadata.json")
PUBLIC_PLAYER_PROP_MODEL_KEYS = {
    "nba_player_props",
    "mlb_player_props",
    "wnba_player_props",
}
PICK_METADATA_FIELDS = {"result", "start_time", "game_start_time", "pregame_snapshot"}
MARKET_METADATA_FIELDS = {"start_time", "game_start_time", "pregame_snapshot"}
_CONSENSUS_MODELS: list[str] | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _consensus_models() -> list[str]:
    global _CONSENSUS_MODELS
    if _CONSENSUS_MODELS is not None:
        return _CONSENSUS_MODELS
    metadata = _read_json(CONSENSUS_METADATA_PATH) or {}
    model_map = metadata.get("models") if isinstance(metadata.get("models"), dict) else {}
    _CONSENSUS_MODELS = [f"{name}: {description}" for name, description in sorted(model_map.items())]
    return _CONSENSUS_MODELS


def _consensus_models_for_sport(sport: str) -> list[str]:
    sport_prefix = f"{sport.strip().lower()}_"
    return [label for label in _consensus_models() if label.lower().startswith(sport_prefix)]


def _ensure_consensus_fields(pick: dict[str, Any]) -> dict[str, Any]:
    if str(pick.get("ml_model_version") or "").strip() != "player_props_consensus_v2.0.0":
        return pick
    models = _consensus_models()
    sport = str(pick.get("sport") or "").strip().upper()
    applicable_models = _consensus_models_for_sport(sport)
    if models:
        pick.setdefault("consensus_model_count", len(models))
        pick.setdefault("consensus_models", models)
    if applicable_models:
        pick.setdefault("consensus_applicable_models", applicable_models)
        pick.setdefault("consensus_record_models", applicable_models)
    model_names = ", ".join(label.split(":", 1)[0] for label in models)
    if model_names:
        factors = pick.get("key_factors")
        if not isinstance(factors, list):
            factors = []
        if not any("Four-model consensus suite active" in str(factor) for factor in factors):
            pick["key_factors"] = [f"Four-model consensus suite active: {model_names}", *factors]
    reason = str(pick.get("reason") or "")
    if reason.startswith("The 2026 season model and roster-aware history model qualify this market"):
        pick["reason"] = reason.replace(
            "The 2026 season model and roster-aware history model qualify this market",
            f"The active four-model consensus suite qualifies this market through the {sport} season and roster-aware history voters",
            1,
        )
    return pick


def _pick_key(pick: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(pick.get(key) or "").strip().lower()
        for key in ("id", "source", "sport", "date", "pick", "matchup", "ml_rank_epoch", "ranking_epoch", "model_epoch")
    )


def _market_key(pick: dict[str, Any]) -> tuple[str, ...]:
    primary = tuple(
        str(pick.get(key) or "").strip().lower()
        for key in ("sport", "date", "game_id", "player_id", "stat_key", "selection", "line")
    )
    if all(primary[:6]) and primary[6]:
        return primary
    return tuple(
        str(pick.get(key) or "").strip().lower()
        for key in ("source", "sport", "date", "pick", "matchup")
    )


def _sport_from_model_key(model_key: str) -> str:
    value = str(model_key or "").strip().lower()
    if value == "wnba_3pm":
        return "WNBA"
    if value.startswith("mlb_player_props"):
        return "MLB"
    if value.startswith("wnba_player_props"):
        return "WNBA"
    if value.startswith("nba_player_props"):
        return "NBA"
    return ""


def _sport_model_key(sport: str) -> str:
    return f"{str(sport or '').strip().lower()}_player_props"


def _sport_source(sport: str) -> str:
    return f"{str(sport or '').strip().upper()}PlayerProps"


def _model_source(model_key: str, sport: str) -> str:
    if str(model_key or "").strip().lower() == "wnba_3pm":
        return "WNBA3PM"
    return _sport_source(sport)


def _same_sport_model_bucket(model_key: str, bucket_key: str) -> bool:
    if "wnba_3pm" in {
        str(model_key or "").strip().lower(),
        str(bucket_key or "").strip().lower(),
    }:
        return str(model_key or "").strip().lower() == str(bucket_key or "").strip().lower()
    sport = _sport_from_model_key(model_key)
    return bool(sport and _sport_from_model_key(bucket_key) == sport)


def _consensus_qualified_player_prop(pick: dict[str, Any]) -> bool:
    mode = str(pick.get("ml_probability_mode") or "").strip()
    return (
        pick.get("consensus_qualified") is True
        or pick.get("precision_qualified") is True
        or mode == "four_model_consensus_gate"
    )


def _tracked_decision(pick: dict[str, Any]) -> bool:
    return str(pick.get("decision") or "").strip().upper() in {"BET", "LEAN"}


def _carry_forward_allowed(pick: Any, date_iso: str) -> bool:
    if not isinstance(pick, dict):
        return False
    if str(pick.get("date") or "").strip() != date_iso:
        return False
    if str(pick.get("scope") or "").strip().lower() != "player":
        return False
    return bool(
        pick.get("market_priced") is True
        and str(pick.get("probability_source") or "").strip() == "player_props_ml_v1"
        and str(pick.get("pick") or "").strip()
        and _tracked_decision(pick)
        and _consensus_qualified_player_prop(pick)
    )


def _snapshot_buckets(date_iso: str, model_key: str, snapshot_dir: Path) -> list[dict[str, Any]]:
    buckets: list[dict[str, Any]] = []
    for path in sorted((snapshot_dir / date_iso).glob("*.json")):
        snapshot = _read_json(path)
        if not snapshot or str(snapshot.get("date") or "").strip() != date_iso:
            continue
        models = snapshot.get("models") if isinstance(snapshot.get("models"), dict) else {}
        for bucket_key, bucket in models.items():
            if isinstance(bucket, dict) and _same_sport_model_bucket(model_key, str(bucket_key)):
                count = sum(
                    1
                    for pick in bucket.get("picks") or []
                    if _carry_forward_allowed(pick, date_iso)
                )
                if count:
                    buckets.append(bucket)
    return buckets


def _current_buckets(current_models: dict[str, Any], model_key: str) -> list[dict[str, Any]]:
    return [
        bucket
        for bucket_key, bucket in current_models.items()
        if isinstance(bucket, dict) and _same_sport_model_bucket(model_key, str(bucket_key))
    ]


def _normalized_player_prop_id(pick: dict[str, Any]) -> str:
    existing = str(pick.get("id") or "").strip()
    for suffix in ("_season", "_all_time", "_hot_l10", "_matchup_h2h"):
        if existing.endswith(suffix):
            return f"{existing[:-len(suffix)]}_consensus"
    return existing or "_".join(_market_key(pick))


def _normalize_carried_pick(pick: dict[str, Any], generated_bucket: dict[str, Any]) -> dict[str, Any]:
    generated_model_key = str(generated_bucket.get("model_key") or "").strip()
    sport = str(pick.get("sport") or _sport_from_model_key(generated_model_key)).strip().upper()
    model_key = generated_model_key or _sport_model_key(sport)
    source = _model_source(model_key, sport)
    rank_epoch = str(generated_bucket.get("ranking_epoch") or pick.get("ml_rank_epoch") or pick.get("ranking_epoch") or "")
    normalized = _ensure_consensus_fields(dict(pick))
    if "supporting_variant" not in normalized and normalized.get("model_variant"):
        normalized["supporting_variant"] = normalized.get("model_variant")
    if "supporting_variant_label" not in normalized and normalized.get("model_variant_label"):
        normalized["supporting_variant_label"] = normalized.get("model_variant_label")
    normalized.update(
        {
            "id": _normalized_player_prop_id(normalized),
            "source": source,
            "model_key": model_key,
            "ranking_model": source,
            "published_model": source,
            "ml_rank_epoch": rank_epoch,
            "ranking_epoch": rank_epoch,
            "model_epoch": rank_epoch,
            "preserved_from_prior_refresh": True,
        }
    )
    return normalized


def _rank_pick_sort_key(pick: dict[str, Any]) -> tuple[int, float, float, float, int, str]:
    decision_rank = {"BET": 0, "LEAN": 1}
    return (
        decision_rank.get(str(pick.get("decision") or "").strip().upper(), 2),
        -float(pick.get("ml_expected_value") or 0.0),
        -float(pick.get("ml_edge") or 0.0),
        -float(pick.get("ml_probability") or pick.get("probability") or 0.0),
        int(pick.get("ml_rank") or pick.get("rank") or 9999),
        str(pick.get("id") or ""),
    )


def _snapshot_market_keys(date_iso: str, model_key: str, snapshot_dir: Path) -> set[tuple[str, ...]]:
    snapshot_keys: set[tuple[str, ...]] = set()
    for path in sorted((snapshot_dir / date_iso).glob("*.json")):
        snapshot = _read_json(path)
        if not snapshot or str(snapshot.get("date") or "").strip() != date_iso:
            continue
        models = snapshot.get("models") if isinstance(snapshot.get("models"), dict) else {}
        for bucket_key, bucket in models.items():
            if not isinstance(bucket, dict) or not _same_sport_model_bucket(model_key, str(bucket_key)):
                continue
            for pick in bucket.get("picks") or []:
                if isinstance(pick, dict) and _carry_forward_allowed(pick, date_iso):
                    snapshot_keys.add(_market_key(pick))
    return snapshot_keys


def _rank_merged_picks(
    pinned: list[dict[str, Any]],
    fresh: list[dict[str, Any]],
    *,
    required_market_keys: set[tuple[str, ...]] | None = None,
) -> list[dict[str, Any]]:
    required_market_keys = required_market_keys or set()
    pinned_sorted = sorted(pinned, key=_rank_pick_sort_key)
    fresh_sorted = sorted(fresh, key=_rank_pick_sort_key)
    selected: list[dict[str, Any]] = []
    known_markets: set[tuple[str, ...]] = set()

    for pick in pinned_sorted:
        key = _market_key(pick)
        if key in known_markets:
            continue
        if required_market_keys and key in required_market_keys:
            selected.append(pick)
            known_markets.add(key)

    for picks in (pinned_sorted, fresh_sorted):
        for pick in picks:
            key = _market_key(pick)
            if key in known_markets:
                continue
            selected.append(pick)
            known_markets.add(key)

    ranked = sorted(selected, key=_rank_pick_sort_key)
    for index, pick in enumerate(ranked, start=1):
        pick["ml_rank"] = index
        pick["model_rank"] = index
        pick["rank"] = index
    return ranked


def _preserve_pick_metadata(
    source_buckets: list[Any],
    generated_bucket: Any,
    date_iso: str,
    *,
    snapshot_dir: Path = PLAYER_PROPS_SNAPSHOT_DIR,
) -> Any:
    if not isinstance(generated_bucket, dict):
        return generated_bucket
    generated_picks = generated_bucket.get("picks")
    if not isinstance(generated_picks, list):
        return generated_bucket
    model_key = str(generated_bucket.get("model_key") or "")
    source_picks = [
        pick
        for bucket in source_buckets
        if isinstance(bucket, dict) and isinstance(bucket.get("picks"), list)
        for pick in bucket.get("picks") or []
        if isinstance(pick, dict)
    ]
    metadata = {
        _pick_key(pick): {field: pick[field] for field in PICK_METADATA_FIELDS if field in pick}
        for pick in source_picks
    }
    metadata_by_market = {
        _market_key(pick): {field: pick[field] for field in MARKET_METADATA_FIELDS if field in pick}
        for pick in source_picks
    }
    merged = dict(generated_bucket)
    generated_with_metadata = [
        _ensure_consensus_fields({
            **pick,
            **metadata_by_market.get(_market_key(pick), {}),
            **metadata.get(_pick_key(pick), {}),
        }) if isinstance(pick, dict) else pick
        for pick in generated_picks
    ]
    fresh_picks: list[dict[str, Any]] = [
        pick
        for pick in generated_with_metadata
        if isinstance(pick, dict)
    ]
    pinned_picks: list[dict[str, Any]] = []
    pinned_markets: set[tuple[str, ...]] = set()
    for pick in source_picks:
        if not _carry_forward_allowed(pick, date_iso):
            continue
        key = _market_key(pick)
        if key in pinned_markets:
            continue
        pinned_picks.append(_normalize_carried_pick(pick, merged))
        pinned_markets.add(key)
    fresh_only = [pick for pick in fresh_picks if _market_key(pick) not in pinned_markets]
    required_market_keys = _snapshot_market_keys(date_iso, model_key, snapshot_dir)
    merged["picks"] = _rank_merged_picks(
        pinned_picks,
        fresh_only,
        required_market_keys=required_market_keys,
    )
    return merged


def merge_payload(
    generated: dict[str, Any],
    cache_dir: Path,
    snapshot_dir: Path = PLAYER_PROPS_SNAPSHOT_DIR,
    *,
    include_current: bool = True,
) -> dict[str, Any]:
    date_iso = str(generated.get("date") or "").strip()
    if not date_iso:
        raise SystemExit("Generated player-props cache is missing date")
    current = _read_json(cache_dir / f"{date_iso}.json") or {}
    current_models = current.get("models") if isinstance(current.get("models"), dict) else {}
    generated_models = generated.get("models") if isinstance(generated.get("models"), dict) else {}
    public_generated_models = {
        key: bucket
        for key, bucket in generated_models.items()
        if key in PUBLIC_PLAYER_PROP_MODEL_KEYS
    }

    merged = dict(generated)
    merged["models"] = {
        key: _preserve_pick_metadata(
            (_current_buckets(current_models, key) if include_current else []) + _snapshot_buckets(date_iso, key, snapshot_dir),
            bucket,
            date_iso,
            snapshot_dir=snapshot_dir,
        )
        for key, bucket in public_generated_models.items()
    }
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", help="Generated player-props latest.json")
    parser.add_argument("--cache-dir", default=str(PLAYER_PROPS_CACHE_DIR))
    parser.add_argument("--snapshot-dir", default=str(PLAYER_PROPS_SNAPSHOT_DIR))
    parser.add_argument("--ignore-current-cache", action="store_true")
    args = parser.parse_args()

    generated = _read_json(Path(args.generated))
    if not generated:
        raise SystemExit(f"Could not read generated player-props cache: {args.generated}")
    cache_dir = Path(args.cache_dir)
    merged = merge_payload(
        generated,
        cache_dir,
        Path(args.snapshot_dir),
        include_current=not args.ignore_current_cache,
    )
    date_iso = str(merged["date"])
    _write_json(cache_dir / f"{date_iso}.json", merged)
    _write_json(cache_dir / "latest.json", merged)
    write_cache_manifest(cache_dir)
    print(json.dumps({"date": date_iso, "models": sorted(merged["models"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
