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

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.price_clock import QUOTE_FIELDS, observed_quote_timing
from scripts.settlement_support import settlement_exclusion_reason


REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER_RELATIVE_PATH = Path("data") / "calibration" / "team_prop_pregame_ledger.json"
SHARD_RELATIVE_PATH = LEDGER_RELATIVE_PATH.with_suffix("")
SHARD_MAX_BYTES = 8_000_000
_SHARD_NAME = re.compile(r"(?:\d{4}-\d{2}-\d{2}|undated)(?:-\d{3,})?\.json")
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
    # nba / nba_playoffs added 2026-09-19: without trusted per-pick timing
    # 0 of 5,135 ledger records were NBA, so the October book would have
    # started uncertified. Their rows carry aware game_start_time values.
    "nba",
    "nba_playoffs",
    "nba_summer",
    "fifa_world_cup",
    "mls",
    "nfl",
    "cfb",
    "nhl",
    "tennis",
    "ipl",
}
FIFA_MODEL_KEYS = {"fifa_world_cup"}
TRACKED_TEAM_DECISIONS = {"BET", "LEAN"}
FORECAST_AUDIT_MODEL_KEYS = TEAM_PROP_MODEL_KEYS

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


def _load_legacy(path: Path, *, strict: bool = False) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_ledger()
    except (OSError, json.JSONDecodeError):
        if strict:
            raise
        return _empty_ledger()
    if strict and (not isinstance(payload, dict) or not isinstance(payload.get("records"), list)):
        raise ValueError(f"Invalid legacy team ledger: {path}")
    if not isinstance(payload, dict):
        return _empty_ledger()
    loaded = dict(payload)
    loaded.setdefault("schema_version", SCHEMA_VERSION)
    loaded.setdefault("kind", "team_prop_pregame_snapshot_ledger")
    if not isinstance(loaded.get("records"), list):
        loaded["records"] = []
    return loaded


def _record_precedence(record: Any) -> tuple:
    if not isinstance(record, dict):
        return (False, False, "", False)
    # The grader can retract a binary grade for unsupported fractional lines.
    # Such corrections must survive a merge with an older settled copy.
    result = str(record.get("result") or "pending").strip().lower()
    corrected = bool(record.get("settlement_exclusion_reason")) and result == "pending"
    settled = result in {"win", "loss", "push", "void", "cancelled"}
    timestamps = [
        parsed.isoformat() for field in ("graded_at", "result_updated_at", "updated_at")
        if (parsed := _parse_timestamp(record.get(field))) is not None
    ]
    return (corrected, settled, max(timestamps, default=""), bool(record.get("start_time")))


def merge_team_prop_pregame_ledgers(primary: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Union by id in first-seen order, retaining corrections and attached grades.

    Explicit record update times break grade ties; otherwise the primary copy
    wins. Publication-level updated_at is not a grade clock. No record fields
    are combined or regenerated. Anonymous legacy rows retain their occurrences.
    """
    merged = {**incoming, **primary}
    if str(incoming.get("updated_at") or "") > str(primary.get("updated_at") or ""):
        merged.update({key: value for key, value in incoming.items() if key != "records"})
    records: list[Any] = []
    positions: dict[tuple, int] = {}
    for payload in (primary, incoming):
        anonymous: Counter = Counter()
        for record in payload["records"]:
            record_id = record.get("id") if isinstance(record, dict) else None
            if record_id:
                key = ("id", str(record_id))
            else:
                fingerprint = _hash(record)
                key = ("anonymous", fingerprint, anonymous[fingerprint])
                anonymous[fingerprint] += 1
            if key not in positions:
                positions[key] = len(records)
                records.append(record)
            elif _record_precedence(record) > _record_precedence(records[positions[key]]):
                records[positions[key]] = record
    merged["records"] = records
    return merged


def load_team_prop_pregame_ledger(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Return the original payload shape and global insertion order.

    Legacy-only missing/malformed files retain their empty-ledger behavior.
    Shard corruption or a missing indexed shard raises instead of silently
    turning incomplete certified evidence into an empty or truncated ledger.
    """
    legacy_path = _ledger_path(repo_root)
    directory = legacy_path.with_suffix("")
    index_path = directory / "index.json"
    paths = {path.name: path for path in directory.glob("*.json") if _SHARD_NAME.fullmatch(path.name)}
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else None
    if index is not None:
        if (
            not isinstance(index, dict) or index.get("storage_version") != 1
            or not isinstance(index.get("metadata"), dict) or not isinstance(index.get("shards"), list)
            or type(index.get("record_count")) is not int or index["record_count"] < 0
        ):
            raise ValueError(f"Invalid team ledger index: {index_path}")
        for shard in index["shards"]:
            if shard["file"] not in paths:
                raise ValueError(f"Missing team ledger shard: {shard['file']}")
    entries = []
    for name, path in sorted(paths.items()):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"Invalid team ledger shard: {path}")
        for row in rows:
            if not isinstance(row, dict) or type(row.get("sequence")) is not int or row["sequence"] < 0 or "record" not in row:
                raise ValueError(f"Invalid team ledger entry: {path}")
            entries.append((row["sequence"], name, row["record"]))
    if index is not None:
        sequences = {sequence for sequence, _, _ in entries}
        if any(sequence not in sequences for sequence in range(index["record_count"])):
            raise ValueError(f"Incomplete team ledger history: {index_path}")
    sharded = {**_empty_ledger(), **(index["metadata"] if index else {})}
    sharded["records"] = [row for _, _, row in sorted(entries, key=lambda entry: entry[:2])]
    legacy = _load_legacy(legacy_path)
    # Before the first index is installed, the complete monolith still owns
    # ordering; this also makes interrupted initial migrations safe to retry.
    if index is None and legacy_path.exists():
        return merge_team_prop_pregame_ledgers(legacy, sharded)
    return merge_team_prop_pregame_ledgers(sharded, legacy)


def _shard_date(record: Any) -> str:
    if isinstance(record, dict):
        for field in ("slate_date", "game_start_time", "published_at"):
            candidate = str(record.get(field) or "")[:10]
            try:
                return date.fromisoformat(candidate).isoformat()
            except ValueError:
                continue
    return "undated"


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


def write_team_prop_pregame_ledger(
    payload: dict[str, Any],
    repo_root: Path | str | None = None,
) -> bool:
    """Atomically replace changed daily shards, then the index; report changes.

    Sequence numbers live outside each untouched record to retain insertion
    order across dates. A busy date splits into numbered parts before 8 MB.
    Existing history is unioned in, including any late legacy publication.
    The monolith is removed only after every shard and the index succeeds.
    """

    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("team pregame ledger payload must contain a records list")

    normalized = dict(payload)
    normalized.setdefault("schema_version", SCHEMA_VERSION)
    normalized.setdefault("kind", "team_prop_pregame_snapshot_ledger")
    path = _ledger_path(repo_root)
    _load_legacy(path, strict=True)  # Never delete unreadable legacy evidence.
    normalized = merge_team_prop_pregame_ledgers(normalized, load_team_prop_pregame_ledger(repo_root))
    directory = path.with_suffix("")
    buckets: dict[str, list[bytes]] = defaultdict(list)
    for sequence, record in enumerate(normalized["records"]):
        line = json.dumps({"sequence": sequence, "record": record}, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        if len(line) + 5 > SHARD_MAX_BYTES:
            raise ValueError("One team ledger record exceeds the shard size limit")
        buckets[_shard_date(record)].append(line)
    rendered_shards: dict[str, bytes] = {}
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
        "metadata": {key: value for key, value in normalized.items() if key != "records"},
        "record_count": len(normalized["records"]),
        "shards": shards,
    }
    changed = False
    for name, rendered in rendered_shards.items():
        changed |= _atomic_write_changed(directory / name, rendered)
    changed |= _atomic_write_changed(directory / "index.json", (json.dumps(index, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8"))
    for old in directory.glob("*.json"):
        if _SHARD_NAME.fullmatch(old.name) and old.name not in rendered_shards:
            old.unlink()
            changed = True
    if path.exists():
        path.unlink()
        changed = True
    return changed


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
            existing = pick.get(TIMING_FIELD)
            if isinstance(existing, dict) and existing.get("trusted") is True:
                # NFL (and any other generator) may stamp at forecast time so a
                # later multi-model write cannot move the clock past kickoff.
                continue
            pick[TIMING_FIELD] = {
                "trusted": True,
                "published_at": published,
                "data_as_of": as_of,
                "source": source,
            }
            stamped += 1
    return stamped


def refresh_trusted_publication_clock(
    payload: dict[str, Any],
    *,
    now: datetime | None = None,
    this_run_generated_at: str | None = None,
) -> int:
    """Advance published_at to this pregame write after odds attach.

    Games that have already started keep their generation clock. Advancing
    those timestamps would un-certify forecasts that were produced before
    kickoff while a slower sibling model was still running.

    NFL generate-time stamps are advanced while the game is still pregame so
    the quote-to-publication window matches the live DraftKings overlay.
    Other in-house buckets only move clocks that were stamped with this run's
    generatedAt (retained rows keep their original publication).
    """

    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)
    else:
        clock = clock.astimezone(timezone.utc)
    clock_iso = clock.isoformat().replace("+00:00", "Z")
    this_run = str(this_run_generated_at or "").strip()
    models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
    advanced = 0
    for model_key, bucket in models.items():
        if str(model_key) not in TEAM_PROP_MODEL_KEYS or not isinstance(bucket, dict):
            continue
        picks = bucket.get("picks") if isinstance(bucket.get("picks"), list) else []
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            timing = pick.get(TIMING_FIELD)
            if not isinstance(timing, dict) or timing.get("trusted") is not True:
                continue
            game = _game_lookup(bucket, pick)
            start = _parse_timestamp(_game_start_time(pick, game))
            if start is None or start <= clock:
                continue
            if str(model_key) != "nfl":
                if not this_run or str(timing.get("published_at") or "") != this_run:
                    continue
            timing["published_at"] = clock_iso
            as_of = _parse_timestamp(timing.get("data_as_of"))
            if as_of is not None and as_of > clock:
                timing["data_as_of"] = clock_iso
            advanced += 1
    return advanced


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


def _fitted_artifact_version(model_key: str, bucket: Mapping[str, Any], pick: Mapping[str, Any]) -> str:
    """Stable fitted artifact label, when the pick still carries one.

    NFL and NHL serving also stamp content hashes onto
    ``prediction_model_version``. Those hashes are useful for audit but split
    staking scorecards when unrelated server code changes.
    """

    if str(model_key) not in {"nfl", "nhl", "mls"}:
        return ""
    for source in (pick, bucket):
        value = _text(source.get("model_version"))
        if (str(model_key) == "nfl" and value.startswith("nfl_v")) or (
            str(model_key) == "nhl" and value.startswith("nhl_poisson_v")
        ) or (
            str(model_key) == "mls" and value.startswith("mls_dixon_coles_v")
        ):
            return value
    return ""


def _model_version(model_key: str, bucket: Mapping[str, Any], pick: Mapping[str, Any]) -> str:
    fitted = _fitted_artifact_version(model_key, bucket, pick)
    if fitted:
        return fitted
    value = _first_value(
        pick.get("prediction_model_version"),
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
        "model_assumed_odds",
        "assumed_odds_replaced",
        "line",
        "market_line",
        "market_total_line",
        "market_pick_prob",
        "market_no_vig_selected_probability",
        "market_probability",
        "market_implied_probability",
        "pricing_type",
        "price_source",
        "odds_source",
        "line_source",
        "market_total_source",
        "market_priced",
        "market_updated_at",
        "market_odds_captured_at",
        "market_retrieved_at",
        "odds_updated_at",
        "price_updated_at",
    )
    return {field: copy.deepcopy(pick.get(field)) for field in fields if field in pick}


def _odds_still_assumed(price: Mapping[str, Any]) -> bool:
    """The published odds equal the model's own placeholder and were never replaced.

    Mirrors ``market_odds._looks_assumed`` so the ledger and the attach step
    agree about the same row; 17 records were certified at exactly +100 with
    ``model_assumed_odds: 100`` sitting in the record.
    """

    if price.get("assumed_odds_replaced") is True:
        return False
    odds = _number(price.get("odds"))
    if odds is None:
        return False
    return any(
        _number(price.get(field)) is not None and _number(price.get(field)) == odds
        for field in ("assumed_odds", "model_assumed_odds")
    )


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
    elif _odds_still_assumed(price):
        financial = False
        financial_reason = "assumed_or_proxy_price"
    else:
        # Only explicit odds provenance proves the price was executable. A
        # model-published market probability is benchmark context, never proof
        # that the odds were posted: 259 MLS handicap records with no odds
        # provenance were being certified as observed_executable_price.
        explicit_market = price.get("market_priced") is True or _norm(price.get("pricing_type")) in {
            "market",
            "sportsbook",
            "bookmaker",
            "observed",
            "executable",
        }
        financial = bool(explicit_market)
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
    raw_decision = _text(
        _first_value(
            snapshot.get("shadow_decision"),
            pick.get("shadow_decision"),
            snapshot.get("source_decision"),
            pick.get("source_decision"),
            snapshot.get("decision"),
        )
    ) or None
    shadow_decision = _text(_first_value(snapshot.get("shadow_decision"), pick.get("shadow_decision"))) or None
    shadow_units = _number(_first_value(snapshot.get("shadow_units"), pick.get("shadow_units")))
    financial_eligible, financial_reason, benchmark_eligible, benchmark_reason = _price_eligibility(pick, price)
    settlement_reason = settlement_exclusion_reason({
        **pick, "model_key": model_key, "market": market,
        "pregame_snapshot": snapshot, "price": price,
    })
    if financial_eligible:
        clock_reason = observed_quote_timing(pick, published_at=published_at, start_at=game_start_time)
        if clock_reason:
            financial_eligible = benchmark_eligible = False
            financial_reason = benchmark_reason = clock_reason

    if settlement_reason:
        calibration_eligible = False
        calibration_reason = settlement_reason
    elif model_key in FIFA_MODEL_KEYS:
        calibration_eligible = False
        calibration_reason = "fifa_evaluation_excluded"
    elif model_key == "mls" and pick.get("calibration_excluded", True):
        calibration_eligible = False
        calibration_reason = "awaiting_approved_holdout"
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
        "raw_decision": raw_decision,
        "decision": immutable["decision"] or None,
        "raw_stake": _number(_first_value(snapshot.get("shadow_units"), snapshot.get("source_units"), snapshot.get("units"))),
        "stake": immutable["stake"],
        "shadow_decision": shadow_decision,
        "shadow_units": shadow_units,
        **({"result": "pending"} if model_key in {"nhl", "mls"} else {}),
        "market": market,
        "selection": selection,
        "pick": _text(snapshot.get("pick") or pick.get("pick")),
        "price": price,
        "observed_american_odds": _number(price.get("odds")) if financial_eligible else None,
        "market_probability": _first_value(
            price.get("market_no_vig_selected_probability"),
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
        **({"settlement_exclusion_reason": settlement_reason} if settlement_reason else {}),
        "calibration_eligible": calibration_eligible,
        "calibration_eligibility_reason": calibration_reason,
        "pregame_snapshot": snapshot,
    }


def _material_signature(record: Mapping[str, Any]) -> str:
    """Ignore quote/publication clock churn only when evidence eligibility agrees.

    Stored snapshots and their original hashes remain exact audit evidence.
    Features, probabilities, prices, model versions and provenance stay exact;
    no floating-point rounding or blanket timestamp removal is applied.
    """
    material = {key: record.get(key) for key in (
        "model_key", "model_version", "game_id", "game_start_time", "slate_date",
        "market", "selection", "raw_probability", "displayed_probability",
        "decision", "stake", "feature_hash", "feature_snapshot", "certification",
        "financial_eligible", "financial_eligibility_reason", "market_benchmark_eligible",
        "market_benchmark_eligibility_reason", "calibration_eligible", "calibration_eligibility_reason",
        "settlement_exclusion_reason",
    )}
    for field in ("price", "pregame_snapshot"):
        value = record.get(field)
        material[field] = {
            key: item for key, item in value.items()
            if key not in QUOTE_FIELDS
        } if isinstance(value, dict) else value
        if isinstance(value, dict) and isinstance(value.get(TIMING_FIELD), dict):
            material[field][TIMING_FIELD] = {
                key: item for key, item in value[TIMING_FIELD].items()
                if key not in {"published_at", "data_as_of"}
            }
    return _hash(material)


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
    material_signatures: dict[str, str] = {}

    for raw_model_key, bucket in models.items():
        model_key = str(raw_model_key)
        if model_key not in TEAM_PROP_MODEL_KEYS or not isinstance(bucket, dict):
            continue
        picks = bucket.get("picks") if isinstance(bucket.get("picks"), list) else []
        for pick in picks:
            if not isinstance(pick, dict):
                continue
            # Capture PASS forecasts for unbiased probability evaluation. They
            # remain zero-stake research and are excluded from financial ROI.
            decision = _tracked_team_decision(pick)
            if decision not in TRACKED_TEAM_DECISIONS and not (
                model_key in FORECAST_AUDIT_MODEL_KEYS and decision == "PASS"
            ):
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
            material = _material_signature(record)
            same_slot = [
                current for current in records
                if isinstance(current, dict) and current.get("stable_id") == record["stable_id"]
            ]
            prior = max(same_slot, key=lambda current: int(current.get("revision") or 0), default=None)
            clock_only = False
            if prior is not None:
                prior_id = str(prior.get("id"))
                if prior_id not in material_signatures:
                    material_signatures[prior_id] = _material_signature(prior)
                clock_only = material_signatures[prior_id] == material
            # Compare clock churn only to the latest revision: a real price
            # moving away and then back to an earlier value is still material.
            if key in seen_in_payload or clock_only or any(
                current.get("snapshot_hash") == record["snapshot_hash"] for current in same_slot
            ):
                unchanged += 1
                seen_in_payload.add(key)
                continue
            records.append(record)
            material_signatures[str(record["id"])] = material
            seen_in_payload.add(key)
            added += 1

    if added:
        ledger["updated_at"] = _utc_now()
        write_team_prop_pregame_ledger(ledger, repo_root)
    return {"added": added, "unchanged": unchanged, "team_picks": team_picks}


def backfill_team_prop_pregame_from_cache(
    cache_dir: Path | None = None,
    *,
    repo_root: Path | str | None = None,
    model_keys: set[str] | None = None,
) -> dict[str, int]:
    """Capture missing snapshots from dated model-cache files.

    Timestamps are taken from each cache payload as published. This never
    invents a pregame clock for a row that was first written after kickoff.
    """

    root = Path(repo_root or REPO_ROOT)
    cache = Path(cache_dir) if cache_dir is not None else root / "data" / "model_cache"
    wanted = set(model_keys or TEAM_PROP_MODEL_KEYS)
    added = unchanged = team_picks = files = 0
    for path in sorted(cache.glob("20??-??-??.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        filtered = {
            key: bucket
            for key, bucket in models.items()
            if str(key) in wanted and isinstance(bucket, dict)
        }
        if not filtered:
            continue
        summary = capture_team_prop_pregame_snapshots(
            {**payload, "models": filtered},
            repo_root=root,
        )
        added += int(summary.get("added") or 0)
        unchanged += int(summary.get("unchanged") or 0)
        team_picks += int(summary.get("team_picks") or 0)
        files += 1
    return {"files": files, "added": added, "unchanged": unchanged, "team_picks": team_picks}


__all__ = [
    "TEAM_PROP_MODEL_KEYS",
    "backfill_team_prop_pregame_from_cache",
    "capture_team_prop_pregame_snapshots",
    "load_team_prop_pregame_ledger",
    "merge_team_prop_pregame_ledgers",
    "refresh_trusted_publication_clock",
    "stamp_team_prop_pregame_timing",
    "write_team_prop_pregame_ledger",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate and merge certified team ledger history into daily shards.")
    parser.add_argument("--migrate", action="store_true", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--merge-from", type=Path, help="Saved repository root to union after resyncing a workflow checkout.")
    args = parser.parse_args()
    ledger = load_team_prop_pregame_ledger(args.repo_root)
    if args.merge_from:
        # Do not silently skip an unreadable legacy file in a saved checkout.
        _load_legacy(_ledger_path(args.merge_from), strict=True)
        ledger = merge_team_prop_pregame_ledgers(ledger, load_team_prop_pregame_ledger(args.merge_from))
    changed = write_team_prop_pregame_ledger(ledger, args.repo_root)
    directory = args.repo_root / SHARD_RELATIVE_PATH
    sizes = [path.stat().st_size for path in directory.glob("*.json") if _SHARD_NAME.fullmatch(path.name)]
    print(json.dumps({
        "changed": changed, "records": len(ledger["records"]), "shards": len(sizes),
        "largest_shard_bytes": max(sizes, default=0),
        "total_bytes": sum(sizes) + (directory / "index.json").stat().st_size,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
