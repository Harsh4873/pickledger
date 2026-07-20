#!/usr/bin/env python3
"""Immutable, certified pregame snapshots for in-house team-model picks.

This ledger is intentionally separate from the legacy calibration outcome
ledger.  It captures the first cache publication of a team pick and every
material revision, while retaining the exact pregame pick object that was
published.  Graders can later attach outcomes by record id without having to
reconstruct timing or price provenance from a mutable cache file.

Only a refresh-stamped *per-pick* timestamp can certify a record.  A root
payload timestamp is preserved as context for old cache files, but is not
enough to certify an old row retroactively.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_RELATIVE_PATH = Path("data") / "calibration" / "team_prop_pregame_ledger.json"
SCHEMA_VERSION = 1
TIMING_FIELD = "certification_timing"

# These are the in-house, game/team-model buckets published through
# refresh_model_cache.  Player-prop and external-feed buckets deliberately do
# not enter this ledger; their existing snapshot/calibration contracts remain
# unchanged.
TEAM_PROP_MODEL_KEYS = {
    "mlb_new",
    "mlb_inning",
    "mlb_first_five",
    "mlb_team_total",
    "wnba",
    "nba_summer",
    "fifa_world_cup",
    "mls",
    "nfl",
}
FIFA_MODEL_KEYS = {"fifa_world_cup", "mls"}  # soccer models share the evaluation exclusion
TRACKED_TEAM_DECISIONS = {"BET", "LEAN"}

_TIMESTAMP_FIELDS = (
    "game_start_time",
    "start_time",
    "startTime",
    "scheduled_start_time",
    "event_start_time",
)
_NON_EXECUTABLE_MARKERS = (
    "assumed",
    "proxy",
    "synthetic",
    "model_output",
    "model_generated",
    "in_house",
    "default",
    "baseline",
    "unpriced",
)
_SNAPSHOT_EXCLUDED_FIELDS = {
    "result",
    "outcome",
    "profit",
    "certification",
    "calibration_eligible",
    "financial_eligible",
    "market_benchmark_eligible",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ledger_path(repo_root: Path | str | None = None) -> Path:
    return Path(repo_root or REPO_ROOT) / LEDGER_RELATIVE_PATH


def _empty_ledger() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "team_prop_pregame_snapshot_ledger",
        "records": [],
    }


def load_team_prop_pregame_ledger(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Load the canonical team-pick snapshot ledger without mutating it.

    A missing or malformed file behaves like a new empty ledger.  Consumers
    can rely on ``records`` always being a list.
    """

    path = _ledger_path(repo_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_ledger()
    if not isinstance(payload, dict):
        return _empty_ledger()

    loaded = dict(payload)
    loaded.setdefault("schema_version", SCHEMA_VERSION)
    loaded.setdefault("kind", "team_prop_pregame_snapshot_ledger")
    if not isinstance(loaded.get("records"), list):
        loaded["records"] = []
    return loaded


def write_team_prop_pregame_ledger(
    payload: dict[str, Any],
    repo_root: Path | str | None = None,
) -> bool:
    """Persist a canonical ledger payload and return whether its bytes changed.

    The writer never creates or rewrites individual records; append-only
    behavior is enforced by :func:`capture_team_prop_pregame_snapshots`.
    """

    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("team pregame ledger payload must contain a records list")

    normalized = dict(payload)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("kind", "team_prop_pregame_snapshot_ledger")
    rendered = json.dumps(normalized, indent=2, sort_keys=True, default=str) + "\n"
    path = _ledger_path(repo_root)
    try:
        if path.read_text(encoding="utf-8") == rendered:
            return False
    except OSError:
        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)
    return True


def stamp_team_prop_pregame_timing(
    payload: dict[str, Any],
    *,
    published_at: str | None = None,
    data_as_of: str | None = None,
    source: str = "model-cache-refresh",
) -> int:
    """Stamp freshly generated in-house team picks with trusted timing.

    This is called only by the live refresh workflow.  The marker is excluded
    from immutable snapshot hashing, so an unchanged re-run does not create a
    false revision merely because the refresh time changed.
    """

    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    published = str(published_at or payload.get("generatedAt") or _utc_now())
    as_of = str(data_as_of or published)
    stamped = 0
    for model_key, bucket in models.items():
        if str(model_key) not in TEAM_PROP_MODEL_KEYS or not isinstance(bucket, dict):
            continue
        picks = bucket.get("picks") if isinstance(bucket.get("picks"), list) else []
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            pick[TIMING_FIELD] = {
                "trusted": True,
                "published_at": published,
                "data_as_of": as_of,
                "source": source,
            }
            stamped += 1
    return stamped


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _norm(value: Any) -> str:
    return _text(value).lower()


def _parse_timestamp(value: Any) -> datetime | None:
    raw = _text(value)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _game_lookup(bucket: Mapping[str, Any], pick: Mapping[str, Any]) -> dict[str, Any]:
    games = bucket.get("games") if isinstance(bucket.get("games"), list) else []
    pick_game_id = _text(pick.get("game_id") or pick.get("gamePk") or pick.get("event_id"))
    pick_matchup = _norm(pick.get("matchup") or pick.get("game"))
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = _text(game.get("game_id") or game.get("gamePk") or game.get("event_id"))
        if pick_game_id and game_id and pick_game_id == game_id:
            return game
        matchup = _norm(game.get("matchup") or game.get("game"))
        if pick_matchup and matchup and pick_matchup == matchup:
            return game
    return {}


def _game_start_time(pick: Mapping[str, Any], game: Mapping[str, Any]) -> str | None:
    for field in _TIMESTAMP_FIELDS:
        value = _first_value(pick.get(field), game.get(field))
        if value not in (None, ""):
            return _text(value)
    return None


def _slug_without_line(value: Any) -> str:
    text = _norm(value)
    text = re.sub(r"(?<![a-z])[-+]?\d+(?:\.\d+)?", "#", text)
    return re.sub(r"\s+", " ", text).strip()


def _selection_identity(pick: Mapping[str, Any]) -> str:
    market = _norm(pick.get("market") or pick.get("market_type") or pick.get("bet_type"))
    explicit = _text(
        _first_value(
            pick.get("selection"),
            pick.get("direction"),
            pick.get("team"),
            pick.get("side"),
        )
    )
    inning = _text(pick.get("inning"))
    if explicit:
        return f"{market}:{_norm(explicit)}:{inning}".rstrip(":")
    return f"{market}:{_slug_without_line(pick.get('pick'))}:{inning}".rstrip(":")


def _hash(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _pregame_snapshot(pick: Mapping[str, Any]) -> dict[str, Any]:
    existing = pick.get("pregame_snapshot")
    if isinstance(existing, dict):
        return copy.deepcopy(existing)
    return {
        key: copy.deepcopy(value)
        for key, value in pick.items()
        if key not in _SNAPSHOT_EXCLUDED_FIELDS and key != TIMING_FIELD
    }


def _tracked_team_decision(pick: Mapping[str, Any]) -> str:
    """Return the current tracked decision, falling back to the raw snapshot."""

    snapshot = pick.get("pregame_snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else {}
    return _text(_first_value(pick.get("decision"), snapshot.get("decision"))).upper()


def _feature_context(pick: Mapping[str, Any], game: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    for key in ("feature_snapshot", "features", "feature_detail"):
        value = pick.get(key)
        if isinstance(value, dict):
            return copy.deepcopy(value), f"pick.{key}"

    game_context = {
        key: copy.deepcopy(game[key])
        for key in (
            "features",
            "projected_first_five",
            "full_inning_table",
            "edge_table",
            "venue",
            "weather",
            "travel",
            "home_pitcher_context",
            "away_pitcher_context",
        )
        if key in game
    }
    if game_context:
        return game_context, "game_context"

    return {
        key: copy.deepcopy(pick.get(key))
        for key in (
            "raw_probability",
            "probability",
            "model_probability",
            "predicted_probability",
            "model_prediction",
            "edge",
            "model_epoch",
        )
        if key in pick
    }, "pick_fallback"


def _raw_probability(pick: Mapping[str, Any], snapshot: Mapping[str, Any]) -> float | None:
    for field in ("raw_probability", "model_probability", "predicted_probability", "probability", "prob"):
        value = _number(_first_value(pick.get(field), snapshot.get(field)))
        if value is not None:
            return value / 100.0 if value > 1 and value <= 100 else value
    return None


def _displayed_probability(pick: Mapping[str, Any]) -> float | None:
    for field in ("probability", "calibrated_probability", "raw_probability", "model_probability", "predicted_probability", "prob"):
        value = _number(pick.get(field))
        if value is not None:
            return value / 100.0 if value > 1 and value <= 100 else value
    return None


def _model_version(model_key: str, bucket: Mapping[str, Any], pick: Mapping[str, Any]) -> str:
    value = _first_value(
        pick.get("model_version"),
        pick.get("model_epoch"),
        bucket.get("model_version"),
        bucket.get("model_epoch"),
        bucket.get("model_stack"),
        bucket.get("consensus_gate_version"),
    )
    return _text(value) or model_key


def _price_fields(pick: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "odds",
        "assumed_odds",
        "line",
        "market_line",
        "market_total_line",
        "market_pick_prob",
        "market_probability",
        "market_implied_probability",
        "pricing_type",
        "price_source",
        "odds_source",
        "line_source",
        "market_total_source",
        "market_priced",
    )
    return {field: copy.deepcopy(pick.get(field)) for field in fields if field in pick}


def _price_marker_text(price: Mapping[str, Any]) -> str:
    # Only odds-provenance fields decide executability. line_source /
    # market_total_source describe how the LINE was selected (e.g. an
    # assumed total ladder); when the odds at that line are observed
    # sportsbook prices the bet is still executable at those prices.
    return " ".join(
        _norm(price.get(field))
        for field in ("pricing_type", "price_source", "odds_source")
        if price.get(field) not in (None, "")
    )


def _price_eligibility(pick: Mapping[str, Any], price: Mapping[str, Any]) -> tuple[bool, str, bool, str]:
    """Return financial and market-benchmark eligibility with reasons."""

    markers = _price_marker_text(price)
    if price.get("market_priced") is False:
        financial = False
        financial_reason = "market_priced_false"
    elif any(marker in markers for marker in _NON_EXECUTABLE_MARKERS):
        financial = False
        financial_reason = "assumed_or_proxy_price"
    elif _number(price.get("odds")) is None:
        financial = False
        financial_reason = "missing_observed_odds"
    else:
        has_market_probability = any(
            _number(price.get(field)) is not None
            for field in ("market_pick_prob", "market_probability", "market_implied_probability")
        )
        explicit_market = price.get("market_priced") is True or _norm(price.get("pricing_type")) in {
            "market",
            "sportsbook",
            "bookmaker",
            "observed",
            "executable",
        }
        financial = bool(explicit_market or has_market_probability)
        financial_reason = "observed_executable_price" if financial else "unverified_price_provenance"

    has_observed_probability = any(
        _number(price.get(field)) is not None
        for field in ("market_pick_prob", "market_probability", "market_implied_probability")
    )
    benchmark = financial
    if benchmark:
        benchmark_reason = (
            "observed_market_probability"
            if has_observed_probability
            else "observed_executable_odds"
        )
    elif not financial:
        benchmark_reason = financial_reason
    else:
        benchmark_reason = "missing_observed_market_probability"
    return financial, financial_reason, benchmark, benchmark_reason


def _certification(
    pick: Mapping[str, Any],
    bucket: Mapping[str, Any],
    payload: Mapping[str, Any],
    game_start_time: str | None,
) -> tuple[dict[str, Any], str | None, str | None]:
    timing = pick.get(TIMING_FIELD) if isinstance(pick.get(TIMING_FIELD), dict) else {}
    root_time = _first_value(payload.get("generatedAt"), payload.get("updatedAt"), bucket.get("generatedAt"))
    published_at = _text(_first_value(timing.get("published_at"), root_time)) or None
    data_as_of = _text(_first_value(timing.get("data_as_of"), root_time)) or None

    if timing.get("trusted") is not True:
        return (
            {
                "status": "uncertified",
                "reason": "missing_trusted_per_pick_timing",
                "certified": False,
                "immutable": True,
                "pregame": False,
            },
            published_at,
            data_as_of,
        )
    published_dt = _parse_timestamp(timing.get("published_at"))
    as_of_dt = _parse_timestamp(timing.get("data_as_of"))
    start_dt = _parse_timestamp(game_start_time)
    if published_dt is None or as_of_dt is None:
        return (
            {"status": "uncertified", "reason": "invalid_trusted_timestamp", "certified": False, "immutable": True, "pregame": False},
            published_at,
            data_as_of,
        )
    if start_dt is None:
        return (
            {"status": "uncertified", "reason": "missing_or_invalid_game_start_time", "certified": False, "immutable": True, "pregame": False},
            published_at,
            data_as_of,
        )
    if as_of_dt > published_dt:
        return (
            {"status": "uncertified", "reason": "data_as_of_after_publication", "certified": False, "immutable": True, "pregame": False},
            published_at,
            data_as_of,
        )
    if published_dt >= start_dt:
        return (
            {"status": "uncertified", "reason": "published_at_not_before_game_start", "certified": False, "immutable": True, "pregame": False},
            published_at,
            data_as_of,
        )
    return (
        {
            "status": "certified",
            "reason": "trusted_per_pick_pregame_timestamp",
            "certified": True,
            "immutable": True,
            "pregame": True,
        },
        published_at,
        data_as_of,
    )


def _stable_id(
    model_key: str,
    slate_date: str,
    game_id: str,
    matchup: str,
    game_start_time: str | None,
    market: str,
    selection: str,
) -> str:
    identity = {
        "model_key": model_key,
        "slate_date": slate_date,
        "game": game_id or matchup,
        "game_start_time": game_start_time or "",
        "market": market,
        "selection": selection,
    }
    return "team-pregame-" + _hash(identity)[:24]


def _snapshot_record(
    *,
    payload: Mapping[str, Any],
    model_key: str,
    bucket: Mapping[str, Any],
    pick: Mapping[str, Any],
    existing: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    game = _game_lookup(bucket, pick)
    snapshot = _pregame_snapshot(pick)
    game_start_time = _game_start_time(pick, game)
    game_id = _text(_first_value(pick.get("game_id"), pick.get("gamePk"), pick.get("event_id"), game.get("game_id"), game.get("gamePk"), game.get("event_id")))
    matchup = _text(_first_value(pick.get("matchup"), pick.get("game"), game.get("matchup"), game.get("game")))
    away_team = _text(_first_value(pick.get("away_team"), game.get("away_team")))
    home_team = _text(_first_value(pick.get("home_team"), game.get("home_team")))
    slate_date = _text(_first_value(pick.get("date"), pick.get("game_date"), pick.get("slate_date"), payload.get("date")))
    market = _norm(pick.get("market") or pick.get("market_type") or pick.get("bet_type")) or "other"
    selection = _selection_identity(pick)
    stable_id = _stable_id(model_key, slate_date, game_id, matchup or f"{away_team}@{home_team}", game_start_time, market, selection)
    feature_context, feature_hash_source = _feature_context(pick, game)
    feature_hash = _hash(feature_context)
    price = _price_fields(pick)
    certification, published_at, data_as_of = _certification(pick, bucket, payload, game_start_time)
    raw_probability = _raw_probability(pick, snapshot)
    displayed_probability = _displayed_probability(pick)
    financial_eligible, financial_reason, benchmark_eligible, benchmark_reason = _price_eligibility(pick, price)

    if model_key in FIFA_MODEL_KEYS:
        calibration_eligible = False
        calibration_reason = "fifa_evaluation_excluded"
    elif certification["status"] != "certified":
        calibration_eligible = False
        calibration_reason = f"uncertified:{certification['reason']}"
    elif not financial_eligible:
        calibration_eligible = False
        calibration_reason = f"nonfinancial:{financial_reason}"
    elif raw_probability is None:
        calibration_eligible = False
        calibration_reason = "missing_raw_probability"
    else:
        calibration_eligible = True
        calibration_reason = "certified_executable_price"

    immutable = {
        "model_key": model_key,
        "model_version": _model_version(model_key, bucket, pick),
        "game_id": game_id,
        "game_start_time": game_start_time,
        "slate_date": slate_date,
        "market": market,
        "selection": selection,
        "raw_probability": raw_probability,
        "displayed_probability": displayed_probability,
        "decision": _text(pick.get("decision")),
        "stake": _number(pick.get("units")),
        "price": price,
        "feature_hash": feature_hash,
        "feature_snapshot": feature_context,
        "pregame_snapshot": snapshot,
    }
    snapshot_hash = _hash(immutable)
    same_slot = [record for record in existing if record.get("stable_id") == stable_id]
    prior = max(same_slot, key=lambda record: int(record.get("revision") or 0), default=None)
    revision = int(prior.get("revision") or 0) + 1 if prior else 1
    record_id = "team-pregame-rev-" + _hash({"stable_id": stable_id, "revision": revision, "snapshot_hash": snapshot_hash})[:24]

    return {
        "id": record_id,
        "stable_id": stable_id,
        "revision": revision,
        "supersedes_id": prior.get("id") if prior else None,
        "snapshot_hash": snapshot_hash,
        "model_key": model_key,
        "model_version": immutable["model_version"],
        "source": _text(pick.get("source")) or model_key,
        "sport": _text(pick.get("sport")),
        "slate_date": slate_date,
        "game_id": game_id or None,
        "matchup": matchup or None,
        "away_team": away_team or None,
        "home_team": home_team or None,
        "game_start_time": game_start_time,
        "published_at": published_at,
        "data_as_of": data_as_of,
        "raw_probability": raw_probability,
        "displayed_probability": displayed_probability,
        "raw_decision": _text(snapshot.get("decision")) or None,
        "decision": immutable["decision"] or None,
        "raw_stake": _number(snapshot.get("units")),
        "stake": immutable["stake"],
        "market": market,
        "selection": selection,
        "pick": _text(snapshot.get("pick") or pick.get("pick")),
        "price": price,
        "observed_american_odds": _number(price.get("odds")) if financial_eligible else None,
        "market_probability": _first_value(
            price.get("market_pick_prob"),
            price.get("market_probability"),
            price.get("market_implied_probability"),
        ) if benchmark_eligible else None,
        "feature_hash": feature_hash,
        "feature_hash_source": feature_hash_source,
        "feature_snapshot": feature_context,
        "certification": certification,
        "financial_eligible": financial_eligible,
        "financial_eligibility_reason": financial_reason,
        "market_benchmark_eligible": benchmark_eligible,
        "market_benchmark_eligibility_reason": benchmark_reason,
        "calibration_eligible": calibration_eligible,
        "calibration_eligibility_reason": calibration_reason,
        "pregame_snapshot": snapshot,
    }


def capture_team_prop_pregame_snapshots(
    payload: dict[str, Any],
    *,
    repo_root: Path | str | None = None,
) -> dict[str, int]:
    """Append first publications and material revisions from a model cache.

    The returned counters make the refresh/merge integration observable while
    keeping this module independent from graders and evaluators.
    """

    if not isinstance(payload, dict):
        return {"added": 0, "unchanged": 0, "team_picks": 0}
    ledger = load_team_prop_pregame_ledger(repo_root)
    records = ledger["records"]
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    added = 0
    unchanged = 0
    team_picks = 0
    seen_in_payload: set[tuple[str, str]] = set()

    for raw_model_key, bucket in models.items():
        model_key = str(raw_model_key)
        if model_key not in TEAM_PROP_MODEL_KEYS or not isinstance(bucket, dict):
            continue
        picks = bucket.get("picks") if isinstance(bucket.get("picks"), list) else []
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            # PASS rows remain in the raw model cache for diagnostics, but they
            # were never tracked wagers and therefore cannot enter the
            # certified snapshot/grading/calibration pipeline.
            if _tracked_team_decision(pick) not in TRACKED_TEAM_DECISIONS:
                continue
            team_picks += 1
            record = _snapshot_record(
                payload=payload,
                model_key=model_key,
                bucket=bucket,
                pick=pick,
                existing=records,
            )
            key = (str(record["stable_id"]), str(record["snapshot_hash"]))
            if key in seen_in_payload or any(
                current.get("stable_id") == record["stable_id"]
                and current.get("snapshot_hash") == record["snapshot_hash"]
                for current in records
                if isinstance(current, dict)
            ):
                unchanged += 1
                seen_in_payload.add(key)
                continue
            records.append(record)
            seen_in_payload.add(key)
            added += 1

    if added:
        ledger["updated_at"] = _utc_now()
        write_team_prop_pregame_ledger(ledger, repo_root)
    return {"added": added, "unchanged": unchanged, "team_picks": team_picks}


__all__ = [
    "TEAM_PROP_MODEL_KEYS",
    "capture_team_prop_pregame_snapshots",
    "load_team_prop_pregame_ledger",
    "stamp_team_prop_pregame_timing",
    "write_team_prop_pregame_ledger",
]
