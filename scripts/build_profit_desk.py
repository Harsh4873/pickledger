#!/usr/bin/env python3
"""Build the decision-first Profit Desk from committed pick caches.

Policy v2 qualifies picks through two evidence lanes, both anchored on the
observed price rather than model scores or consensus:

- EDGE lane (1.0u): the segment (source + model era + market family +
  direction + probability band) must beat the market baseline after
  hierarchical shrinkage and an uncertainty penalty, with strict sample,
  distinct-date, and chronological-stability gates.
- VALUE lane (0.5u): the source as a whole must show a positive shrunk
  residual against its own posted prices (a conservative flat-ROI test:
  one-sided quotes are measured against their vigged break-even, never
  a fabricated no-vig number), with volume, distinct-date, stability, and
  probability-of-profit gates.

The selection policy version is owned by this engine and stamped on every
candidate; upstream feeds only need to supply real prices, timestamps, and
graded results.  Every estimate starts at the market baseline and adds only
a conservatively shrunk estimate of *prior* residuals (outcome minus
baseline) dated strictly before the slate.

The resulting files are deterministic for a given set of inputs and are safe
to rebuild because dates before ``ENGINE_CUTOVER_DATE`` are never written.
Live stakes only exist for slates on or after ``FIRST_LIVE_DATE``; earlier
slates rebuild as zero-stake research so no live record can be backfilled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = REPO_ROOT / "data" / "model_cache"
PLAYER_PROPS_CACHE_DIR = REPO_ROOT / "data" / "player_props_cache"
PROFIT_DESK_DIR = REPO_ROOT / "data" / "profit_desk"
CLOSING_LINES_DIR = REPO_ROOT / "data" / "closing_lines"

ENGINE_VERSION = "profit_desk_v2_live"
POLICY_VERSION = "profit_desk_policy_v2"
ENGINE_CUTOVER_DATE = "2026-07-10"
FIRST_LIVE_DATE = "2026-07-11"

VISIBLE_DECISIONS = {"BET", "LEAN"}
MAX_PRICE_AGE_HOURS = 24.0

# EDGE lane: strict segment-level market-alpha gates (unchanged from v1).
MIN_SOURCE_SAMPLES = 100
MIN_SEGMENT_SAMPLES = 40
MIN_DISTINCT_DATES = 20
MIN_PROBABILITY_POSITIVE_EV = 0.80
MIN_CONSERVATIVE_PROBABILITY_MARGIN = 0.02
EDGE_STAKE_UNITS = 1.0

# VALUE lane: source-level flat-ROI gates against the source's own posted
# prices.  Thresholds are grounded in the July 2026 evidence audit: the one
# genuinely positive source (322 rows, 20 dates, +7.5% flat ROI) clears them
# while marginal (+2.4% ROI, Pr 0.67) and negative sources do not.
VALUE_MIN_SOURCE_SAMPLES = 150
VALUE_MIN_SOURCE_DATES = 15
VALUE_MIN_PROBABILITY_POSITIVE_EV = 0.70
VALUE_STAKE_UNITS = 0.5

MAX_PER_MODE = 3

# Zero-centered source prior, then a segment prior centered on the source.
SOURCE_PRIOR_ROWS = 40.0
SEGMENT_PRIOR_ROWS = 25.0
MIN_RESIDUAL_VARIANCE = 0.04
LOWER_BOUND_Z = 1.2815515655446004  # one-sided 90% lower bound

# Feeds whose whole purpose is republishing bookmaker-posted odds.  Their
# records carry a real scraped price but no per-record provenance markers,
# so provenance is declared here at the source level (Tier C: posted,
# one-sided).  Model feeds that invent assumed prices are NOT listed.
SCRAPED_ODDS_SOURCE_PREFIXES = ("scores24_", "sportsgambler_", "sportytrader_", "covers_", "forebet_")

_NON_EXECUTABLE_MARKERS = (
    "assumed",
    "synthetic",
    "proxy",
    "fallback",
    "default",
    "estimated",
    "derived",
    "model price",
    "model_price",
)
_DATE_FILE_RE = re.compile(r"^20\d\d-\d\d-\d\d$")
_DIRECTION_RE = re.compile(r"\b(over|under|yes|no)\b", re.IGNORECASE)
_LINE_RE = re.compile(r"(?<![A-Za-z])([+-]?\d+(?:[.,]\d+)?)")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return " ".join(
        "".join(char.lower() if char.isalnum() else " " for char in _text(value)).split()
    )


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_probability(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 < number < 1.0:
        return None
    return number


def american_to_decimal(odds: Any) -> float | None:
    number = _number(odds)
    if number is None or number == 0 or -100.0 < number < 100.0:
        return None
    if number > 0:
        return 1.0 + number / 100.0
    return 1.0 + 100.0 / abs(number)


def implied_probability(odds: Any) -> float | None:
    decimal = american_to_decimal(odds)
    return (1.0 / decimal) if decimal else None


def _american_int(value: Any) -> int | None:
    number = _number(value)
    decimal = american_to_decimal(number)
    if number is None or decimal is None:
        return None
    return int(round(number))


def _parse_timestamp(value: Any) -> datetime | None:
    raw = _text(value)
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _stable_hash(value: Any, length: int = 20) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:length]


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_if_changed(path: Path, payload: Mapping[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = _json_text(payload)
    if path.exists() and path.read_text(encoding="utf-8") == rendered:
        return False
    path.write_text(rendered, encoding="utf-8")
    return True


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _result(value: Any) -> str:
    result = _norm(value)
    if result in {"win", "won", "w"}:
        return "win"
    if result in {"loss", "lost", "l"}:
        return "loss"
    if result in {"push", "void", "p"}:
        return "push"
    return "pending"


def _record_date(record: Mapping[str, Any], fallback: str) -> str:
    return _text(_first(record, "date", "game_date", "slate_date", "Date") or fallback)


def _pick_text(record: Mapping[str, Any]) -> str:
    return _text(_first(record, "pick", "selection", "prop", "bet"))


def _game_label(record: Mapping[str, Any]) -> str:
    explicit = _text(_first(record, "matchup", "game", "event"))
    if explicit:
        return explicit
    away = _text(record.get("away_team"))
    home = _text(record.get("home_team"))
    return f"{away} @ {home}" if away and home else ""


def canonical_game_key(record: Mapping[str, Any], sport: str, date_iso: str) -> str:
    """Return an order-insensitive game identity shared across sources."""

    away = _norm(record.get("away_team"))
    home = _norm(record.get("home_team"))
    teams = [team for team in (away, home) if team]
    label = _game_label(record)
    if len(teams) < 2 and label:
        normalized = re.sub(r"\s+(?:@|vs\.?|v\.)\s+", " @ ", label, flags=re.IGNORECASE)
        parts = [_norm(part) for part in normalized.split(" @ ") if _norm(part)]
        if len(parts) == 2:
            teams = parts
    if len(teams) == 2:
        return f"{date_iso}:{_norm(sport)}:{'|'.join(sorted(teams))}"
    game_id = _norm(_first(record, "game_id", "event_id", "gamePk"))
    if game_id:
        return f"{date_iso}:{_norm(sport)}:id:{game_id}"
    return f"{date_iso}:{_norm(sport)}:unknown:{_stable_hash(label or _pick_text(record), 12)}"


def _direction(record: Mapping[str, Any]) -> str:
    explicit = _norm(_first(record, "direction", "selection"))
    if explicit in {"over", "under", "yes", "no"}:
        return explicit
    match = _DIRECTION_RE.search(_pick_text(record))
    return match.group(1).lower() if match else "side"


def _line(record: Mapping[str, Any], direction: str) -> float | None:
    for key in ("line", "market_line", "market_total_line", "spread", "handicap"):
        number = _number(record.get(key))
        if number is not None:
            return number
    if direction != "side":
        direction_match = re.search(
            rf"\b{re.escape(direction)}\b[^0-9+-]*([+-]?\d+(?:[.,]\d+)?)",
            _pick_text(record),
            flags=re.IGNORECASE,
        )
        if direction_match:
            return _number(direction_match.group(1).replace(",", "."))
    else:
        # Several team feeds embed the handicap only in the pick label (for
        # example, "Seattle -1.5"). Parse it only for spread-like markets so
        # a moneyline price such as "+145" cannot become a fake handicap.
        family = _market_family(record)
        pick = _pick_text(record)
        spread_like = any(
            marker in family
            for marker in (
                "spread",
                "handicap",
                "run line",
                "runline",
                "puck line",
                "puckline",
            )
        ) or bool(
            re.search(
                r"\b(?:spread|handicap|run\s*line|puck\s*line)\b",
                pick,
                flags=re.IGNORECASE,
            )
        )
        if spread_like:
            pick_without_parentheticals = re.sub(r"\([^)]*\)", " ", pick)
            side_line = re.search(
                r"(?<![A-Za-z0-9])([+-]\d+(?:[.,]\d+)?)\b",
                pick_without_parentheticals,
            )
            if side_line:
                return _number(side_line.group(1).replace(",", "."))
    return None


def _market_family(record: Mapping[str, Any]) -> str:
    return _norm(
        _first(record, "stat_key", "market_type", "market", "stat_label", "bet_type")
        or "market"
    )


def _player(record: Mapping[str, Any]) -> str:
    return _text(_first(record, "player_name", "player", "athlete_name"))


def _selected_side(record: Mapping[str, Any], direction: str) -> str:
    if direction != "side":
        return direction
    explicit = _text(_first(record, "team", "side", "selection"))
    if explicit:
        return _norm(explicit)
    pick = _pick_text(record)
    pick = re.sub(r"\([^)]*(?:@|\bvs\.?\b)[^)]*\)", "", pick, flags=re.IGNORECASE)
    pick = re.sub(r"\b(?:moneyline|ml|to win|wins?|cover)\b", " ", pick, flags=re.IGNORECASE)
    pick = _LINE_RE.sub(" ", pick)
    return _norm(pick)


def canonical_market_identity(
    record: Mapping[str, Any], *, mode: str, sport: str, date_iso: str
) -> str:
    """Identity includes prop direction and line, so opposing props never merge."""

    direction = _direction(record)
    line = _line(record, direction)
    parts = {
        "date": date_iso,
        "game": canonical_game_key(record, sport, date_iso),
        "mode": mode,
        "player": _norm(_player(record)) if mode == "player" else "",
        "market": _market_family(record),
        "direction": direction,
        "line": round(line, 4) if line is not None else None,
        "side": _selected_side(record, direction),
    }
    return "market:" + _stable_hash(parts, 24)


# ---------------------------------------------------------------------------
# Market verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoVigProbability:
    probability: float | None
    verified: bool
    method: str
    inputs: dict[str, Any] = field(default_factory=dict)


def _mapping_layers(record: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    layers: list[tuple[str, Mapping[str, Any]]] = [("pick", record)]
    for name in ("pregame_snapshot", "price", "market_snapshot"):
        nested = record.get(name)
        if isinstance(nested, Mapping):
            layers.append((name, nested))
    return layers


def derive_no_vig_probability(record: Mapping[str, Any]) -> NoVigProbability:
    """Use explicit no-vig data or normalize a complete observed market.

    Regular ``market_probability`` and ``selected_side_implied_probability``
    are deliberately ignored: a single vigged side cannot verify fair value.
    """

    direction = _direction(record)
    layers = _mapping_layers(record)

    for layer_name, layer in layers:
        for key in (
            "market_no_vig_selected_probability",
            "no_vig_selected_probability",
            "market_no_vig_probability",
            "no_vig_probability",
        ):
            probability = normalize_probability(layer.get(key))
            if probability is not None:
                return NoVigProbability(
                    probability, True, f"explicit:{layer_name}.{key}", {"field": key}
                )

        over = normalize_probability(
            _first(layer, "market_no_vig_over_probability", "no_vig_over_probability")
        )
        under = normalize_probability(
            _first(layer, "market_no_vig_under_probability", "no_vig_under_probability")
        )
        if direction == "over" and over is not None:
            return NoVigProbability(over, True, f"explicit:{layer_name}.no_vig_over", {})
        if direction == "under":
            if under is not None:
                return NoVigProbability(under, True, f"explicit:{layer_name}.no_vig_under", {})
            if over is not None:
                return NoVigProbability(1.0 - over, True, f"explicit_complement:{layer_name}.no_vig_over", {})

    # Generic selected/opposite pair.
    for layer_name, layer in layers:
        selected = _american_int(_first(layer, "selected_odds", "market_selected_odds", "odds"))
        opposite = _american_int(
            _first(layer, "opposite_odds", "market_opposite_odds", "other_side_odds")
        )
        if selected is not None and opposite is not None:
            probabilities = [implied_probability(selected), implied_probability(opposite)]
            hold = sum(value for value in probabilities if value is not None)
            if hold > 0 and probabilities[0] is not None:
                probability = probabilities[0] / hold
                return NoVigProbability(
                    probability,
                    True,
                    f"derived_two_sided:{layer_name}.selected_opposite",
                    {"selectedOdds": selected, "oppositeOdds": opposite, "hold": round(hold, 6)},
                )

    # Directional over/under or yes/no pairs.
    pair_specs = (
        ("over", "market_over_odds", "market_under_odds"),
        ("over", "over_odds", "under_odds"),
        ("yes", "market_yes_odds", "market_no_odds"),
        ("yes", "yes_odds", "no_odds"),
    )
    for layer_name, layer in layers:
        for positive_side, positive_key, negative_key in pair_specs:
            positive_odds = _american_int(layer.get(positive_key))
            negative_odds = _american_int(layer.get(negative_key))
            if positive_odds is None or negative_odds is None:
                continue
            positive_implied = implied_probability(positive_odds)
            negative_implied = implied_probability(negative_odds)
            if positive_implied is None or negative_implied is None:
                continue
            hold = positive_implied + negative_implied
            positive_probability = positive_implied / hold
            selected_probability = (
                positive_probability if direction == positive_side else 1.0 - positive_probability
            )
            if direction not in {positive_side, "under" if positive_side == "over" else "no"}:
                continue
            return NoVigProbability(
                selected_probability,
                True,
                f"derived_two_sided:{layer_name}.{positive_key}+{negative_key}",
                {
                    positive_key: positive_odds,
                    negative_key: negative_odds,
                    "hold": round(hold, 6),
                },
            )

    # Home/away markets; include draw when supplied so a 3-way market is not
    # incorrectly treated as two-way.
    for layer_name, layer in layers:
        home_odds = _american_int(_first(layer, "market_home_odds", "home_odds"))
        away_odds = _american_int(_first(layer, "market_away_odds", "away_odds"))
        if home_odds is None or away_odds is None:
            continue
        entries = [("home", home_odds), ("away", away_odds)]
        draw_odds = _american_int(_first(layer, "market_draw_odds", "draw_odds"))
        if draw_odds is not None:
            entries.append(("draw", draw_odds))
        implied = [(side, implied_probability(price)) for side, price in entries]
        if any(value is None for _, value in implied):
            continue
        hold = sum(float(value) for _, value in implied)
        selected_team = _norm(_first(record, "team", "side", "selection"))
        home_team = _norm(record.get("home_team"))
        away_team = _norm(record.get("away_team"))
        selected_slot = "draw" if selected_team == "draw" else (
            "home" if selected_team and selected_team == home_team else (
                "away" if selected_team and selected_team == away_team else ""
            )
        )
        if not selected_slot:
            continue
        selected_implied = next(float(value) for side, value in implied if side == selected_slot)
        return NoVigProbability(
            selected_implied / hold,
            True,
            f"derived_complete_market:{layer_name}.home_away" + ("_draw" if draw_odds else ""),
            {"hold": round(hold, 6), "outcomes": len(entries)},
        )

    return NoVigProbability(None, False, "unverified_single_side", {})


def _is_scraped_odds_source(source_key: str) -> bool:
    return source_key.startswith(SCRAPED_ODDS_SOURCE_PREFIXES)


def _price_provenance(
    record: Mapping[str, Any], source_key: str = ""
) -> tuple[bool, str | None, str, list[str]]:
    odds = _american_int(record.get("odds"))
    blockers: list[str] = []
    if odds is None:
        return False, None, "missing", ["missing_executable_odds"]

    marker_values: list[str] = []
    source: str | None = None
    source_field = ""
    for layer_name, layer in _mapping_layers(record):
        for key in (
            "pricing_type",
            "price_source",
            "odds_source",
            "line_source",
            "market_source",
            "market_total_source",
        ):
            value = _text(layer.get(key))
            if value:
                marker_values.append(value.lower())
                if source is None and key in {"price_source", "odds_source", "market_source", "line_source"}:
                    source = value
                    source_field = f"{layer_name}.{key}"
    marker_text = " ".join(marker_values)
    assumed_odds = _american_int(record.get("assumed_odds"))
    if record.get("market_priced") is False or assumed_odds == odds or any(
        marker in marker_text for marker in _NON_EXECUTABLE_MARKERS
    ):
        blockers.append("assumed_or_non_executable_price")

    explicit_market = record.get("market_priced") is True or any(
        marker in marker_text
        for marker in ("market", "sportsbook", "bookmaker", "posted", "observed", "executable")
    )
    if not explicit_market and not blockers and _is_scraped_odds_source(source_key):
        # Odds-republishing feeds post real bookmaker prices without stamping
        # per-record provenance; the source registry supplies it instead.
        explicit_market = True
        if source is None:
            source = f"scraped_feed:{source_key}"
            source_field = "source_registry.scraped_odds"
    if not explicit_market:
        blockers.append("unverified_price_provenance")
    if source is None and record.get("market_priced") is True:
        source = "explicit market_priced flag"
        source_field = "pick.market_priced"
    return not blockers, source, source_field or "unverified", blockers


def _timing(
    record: Mapping[str, Any],
    bucket: Mapping[str, Any] | None = None,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp_value: Any = None
    timestamp_field = ""
    for layer_name, layer in _mapping_layers(record):
        for key in (
            "market_updated_at",
            "odds_updated_at",
            "price_updated_at",
            "snapshot_at",
            "data_as_of",
            "published_at",
        ):
            if layer.get(key) not in (None, ""):
                timestamp_value = layer.get(key)
                timestamp_field = f"{layer_name}.{key}"
                break
        if timestamp_value is not None:
            break
    certification = record.get("certification_timing")
    if timestamp_value is None and isinstance(certification, Mapping):
        timestamp_value = _first(certification, "data_as_of", "published_at")
        timestamp_field = "pick.certification_timing.data_as_of"
    if timestamp_value in (None, ""):
        # Scraped feeds record when the whole feed was captured, not each
        # row.  The capture time is an upper bound on when the price was
        # observed, so using it can only make freshness stricter.
        for container_name, container in (("bucket", bucket), ("payload", payload)):
            if not isinstance(container, Mapping):
                continue
            for key in ("updatedAt", "generatedAt"):
                if container.get(key) not in (None, ""):
                    timestamp_value = container.get(key)
                    timestamp_field = f"{container_name}.{key}"
                    break
            if timestamp_value not in (None, ""):
                break

    start_value = _first(record, "game_start_time", "start_time", "event_start_time")
    timestamp = _parse_timestamp(timestamp_value)
    start = _parse_timestamp(start_value)
    blockers: list[str] = []
    age_hours: float | None = None
    if timestamp is None:
        blockers.append("missing_or_invalid_price_timestamp")
    if start is None:
        blockers.append("missing_or_invalid_game_start_time")
    if timestamp is not None and start is not None:
        age_hours = (start - timestamp).total_seconds() / 3600.0
        if age_hours < 0:
            blockers.append("price_not_pregame")
        elif age_hours > MAX_PRICE_AGE_HOURS:
            blockers.append("stale_price")
    return {
        "timestamp": _text(timestamp_value) or None,
        "timestampField": timestamp_field or None,
        "startTime": _text(start_value) or None,
        "ageHours": round(age_hours, 3) if age_hours is not None else None,
        "freshPregame": not blockers,
        "maxAgeHours": MAX_PRICE_AGE_HOURS,
        "blockers": blockers,
    }


def _grade_support(record: Mapping[str, Any], source_key: str, mode: str) -> tuple[bool, str]:
    flags = [record.get(key) for key in ("grade_supported", "grading_supported", "gradable")]
    if False in flags:
        return False, "explicit_false"
    if True in flags:
        return True, "explicit_true"
    # Repository auto-grading demonstrably settles every cached feed,
    # including the scraped odds feeds (hundreds of graded rows each), so
    # the default is supported unless a record explicitly opts out.
    return True, "repository_grader_default"


def _certified_price(record: Mapping[str, Any]) -> bool:
    certification = record.get("certification")
    if isinstance(certification, Mapping):
        status = _norm(certification.get("status"))
        if status == "certified" and certification.get("pregame") is not False:
            return True
    if record.get("certified_pregame") is True:
        return True
    timing = record.get("certification_timing")
    return isinstance(timing, Mapping) and timing.get("trusted") is True


def _price_tier(
    record: Mapping[str, Any],
    *,
    executable: bool,
    no_vig: NoVigProbability,
) -> tuple[str, str]:
    """Classify price provenance without treating one posted side as fair value."""

    if not executable:
        return "D", "assumed_proxy_or_synthetic"
    if no_vig.verified and _certified_price(record):
        return "A", "certified_executable"
    if no_vig.verified:
        return "B", "posted_two_sided"
    return "C", "posted_one_sided"


# ---------------------------------------------------------------------------
# Records, evidence keys, and trailing estimates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordContext:
    payload: Mapping[str, Any]
    bucket: Mapping[str, Any]
    record: Mapping[str, Any]
    source_key: str
    source: str
    mode: str
    fallback_date: str


def _iter_records(payload: Mapping[str, Any] | None, mode: str) -> Iterable[RecordContext]:
    if not isinstance(payload, Mapping):
        return
    fallback_date = _text(_first(payload, "date", "slate_date"))
    models = payload.get("models")
    if isinstance(models, Mapping):
        buckets = models.items()
    elif isinstance(payload.get("picks"), list):
        buckets = [(_text(payload.get("source_key")) or mode, payload)]
    else:
        buckets = []
    for raw_source_key, raw_bucket in buckets:
        if not isinstance(raw_bucket, Mapping) or raw_bucket.get("ok") is False:
            continue
        source_key = _text(raw_source_key)
        for raw_record in raw_bucket.get("picks") or []:
            if not isinstance(raw_record, Mapping):
                continue
            record_mode = "player" if _norm(raw_record.get("scope")) == "player" else mode
            source = _text(raw_record.get("source")) or source_key
            yield RecordContext(
                payload=payload,
                bucket=raw_bucket,
                record=raw_record,
                source_key=source_key,
                source=source,
                mode=record_mode,
                fallback_date=fallback_date,
            )


def _version(context: RecordContext, kind: str) -> str:
    if kind != "model":
        # The selection policy is this engine, not an upstream field; v1
        # demanded a per-pick policy stamp that no pipeline could supply,
        # which structurally blocked every candidate forever.
        return POLICY_VERSION
    keys = (
        "model_version",
        "ml_model_version",
        "model_epoch",
        "ranking_model_version",
        "engine_version",
    )
    for mapping in (context.record, context.bucket, context.payload):
        value = _first(mapping, *keys)
        if value not in (None, ""):
            return _text(value)
    # Scraped feeds have no model; the source identity is the stable era.
    return f"source_identity:{context.source_key}"


def _probability_band(probability: float) -> str:
    if probability < 0.45:
        return "lt_0.45"
    if probability < 0.50:
        return "0.45_0.50"
    if probability < 0.55:
        return "0.50_0.55"
    if probability < 0.60:
        return "0.55_0.60"
    return "gte_0.60"


def _evidence_keys(
    context: RecordContext, probability: float, market_family: str, direction: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    # The source pool tracks the strategy across model retrains; the
    # chronological-halves gate catches an era regression.  The segment pool
    # stays era-aware so the strict EDGE lane never mixes model versions.
    model_version = _version(context, "model")
    source_key = (
        context.mode,
        context.source_key,
        _norm(context.record.get("sport")) or "unknown_sport",
    )
    segment_key = source_key + (
        model_version,
        market_family,
        direction,
        _probability_band(probability),
    )
    return source_key, segment_key


def _key_text(key: Sequence[str]) -> str:
    return "|".join(str(value).replace("|", "/") for value in key)


@dataclass(frozen=True)
class EvidenceRow:
    row_id: str
    date: str
    source_key: tuple[str, ...]
    segment_key: tuple[str, ...]
    result: str
    outcome: float
    market_probability: float
    residual: float
    profit_units: float


@dataclass
class Aggregate:
    rows: list[EvidenceRow] = field(default_factory=list)

    @property
    def samples(self) -> int:
        return len(self.rows)

    @property
    def dates(self) -> set[str]:
        return {row.date for row in self.rows}

    @property
    def wins(self) -> int:
        return sum(row.result == "win" for row in self.rows)

    @property
    def losses(self) -> int:
        return sum(row.result == "loss" for row in self.rows)

    @property
    def net_units(self) -> float:
        return sum(row.profit_units for row in self.rows)

    @property
    def residual_sum(self) -> float:
        return sum(row.residual for row in self.rows)

    @property
    def chronological_half_net_units(self) -> tuple[float, float]:
        ordered = sorted(self.rows, key=lambda row: (row.date, row.row_id))
        midpoint = len(ordered) // 2
        first = ordered[:midpoint]
        second = ordered[midpoint:]
        return (
            sum(row.profit_units for row in first),
            sum(row.profit_units for row in second),
        )


def source_value_stats(source: Aggregate) -> dict[str, Any]:
    """Source-level VALUE-lane statistics shared by estimates and report cards."""

    source_alpha = source.residual_sum / (source.samples + SOURCE_PRIOR_ROWS)
    if source.samples:
        source_variance = sum(
            (row.residual - source_alpha) ** 2 for row in source.rows
        ) / source.samples
    else:
        source_variance = 0.25
    source_std_error = math.sqrt(
        max(MIN_RESIDUAL_VARIANCE, source_variance)
        / (source.samples + SOURCE_PRIOR_ROWS)
    )
    source_z = source_alpha / source_std_error if source_std_error else 0.0
    probability_positive = 0.5 * (1.0 + math.erf(source_z / math.sqrt(2.0)))
    flat_roi = source.net_units / source.samples if source.samples else None
    first_half, second_half = source.chronological_half_net_units
    return {
        "alpha": source_alpha,
        "alphaStdError": source_std_error,
        "probabilityPositiveEv": probability_positive,
        "flatRoi": flat_roi,
        "flatNetUnits": source.net_units,
        "firstHalfFlatNetUnits": first_half,
        "secondHalfFlatNetUnits": second_half,
        "halvesNonnegative": first_half >= 0.0 and second_half >= 0.0,
        "samples": source.samples,
        "distinctDates": len(source.dates),
        "wins": source.wins,
        "losses": source.losses,
    }


class EvidenceBook:
    """Verified, settled, strictly-prior market residuals."""

    def __init__(self, rows: Iterable[EvidenceRow] = ()) -> None:
        unique: dict[str, EvidenceRow] = {}
        for row in rows:
            unique.setdefault(row.row_id, row)
        self.rows = list(unique.values())
        self.by_source: dict[tuple[str, ...], Aggregate] = defaultdict(Aggregate)
        self.by_segment: dict[tuple[str, ...], Aggregate] = defaultdict(Aggregate)
        for row in self.rows:
            self.by_source[row.source_key].rows.append(row)
            self.by_segment[row.segment_key].rows.append(row)

    @classmethod
    def build(
        cls,
        date_iso: str,
        team_history: Iterable[Mapping[str, Any]],
        prop_history: Iterable[Mapping[str, Any]],
    ) -> "EvidenceBook":
        rows: list[EvidenceRow] = []
        for mode, payloads in (("team", team_history), ("player", prop_history)):
            for payload in payloads:
                for context in _iter_records(payload, mode):
                    record = context.record
                    record_date = _record_date(record, context.fallback_date)
                    if not record_date or record_date >= date_iso:
                        continue
                    if _text(record.get("decision")).upper() not in VISIBLE_DECISIONS:
                        continue
                    if record.get("shadow_mode") is True:
                        continue
                    result = _result(record.get("result"))
                    if result not in {"win", "loss"}:
                        continue
                    odds = _american_int(record.get("odds"))
                    decimal = american_to_decimal(odds)
                    executable, _, _, price_blockers = _price_provenance(
                        record, context.source_key
                    )
                    timing = _timing(record, context.bucket, context.payload)
                    grade_supported, _ = _grade_support(record, context.source_key, context.mode)
                    no_vig = derive_no_vig_probability(record)
                    if (
                        odds is None
                        or decimal is None
                        or not executable
                        or price_blockers
                        or not timing["freshPregame"]
                        or not grade_supported
                    ):
                        continue
                    # Two-sided rows measure against true no-vig; one-sided
                    # rows measure against their own vigged break-even, which
                    # is the stricter baseline (it includes the hold).
                    baseline = (
                        no_vig.probability
                        if no_vig.verified and no_vig.probability is not None
                        else implied_probability(odds)
                    )
                    if baseline is None:
                        continue
                    market_family = _market_family(record)
                    direction = _direction(record)
                    source_key, segment_key = _evidence_keys(
                        context, baseline, market_family, direction
                    )
                    outcome = 1.0 if result == "win" else 0.0
                    profit = decimal - 1.0 if result == "win" else -1.0
                    row_identity = {
                        "date": record_date,
                        "source": source_key,
                        "market": canonical_market_identity(
                            record,
                            mode=context.mode,
                            sport=_text(record.get("sport")),
                            date_iso=record_date,
                        ),
                    }
                    rows.append(
                        EvidenceRow(
                            row_id=_stable_hash(row_identity, 32),
                            date=record_date,
                            source_key=source_key,
                            segment_key=segment_key,
                            result=result,
                            outcome=outcome,
                            market_probability=baseline,
                            residual=outcome - baseline,
                            profit_units=profit,
                        )
                    )
        return cls(rows)

    def estimate(
        self,
        source_key: tuple[str, ...],
        segment_key: tuple[str, ...],
        market_probability: float,
        decimal_odds: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        source = self.by_source[source_key]
        segment = self.by_segment[segment_key]
        source_alpha = source.residual_sum / (source.samples + SOURCE_PRIOR_ROWS)
        alpha = (
            segment.residual_sum + SEGMENT_PRIOR_ROWS * source_alpha
        ) / (segment.samples + SEGMENT_PRIOR_ROWS)

        # Residual variance includes a conservative floor, so even a perfect
        # finite record retains uncertainty.
        if segment.samples:
            empirical_variance = sum(
                (row.residual - alpha) ** 2 for row in segment.rows
            ) / segment.samples
        else:
            empirical_variance = 0.25
        residual_variance = max(MIN_RESIDUAL_VARIANCE, empirical_variance)
        alpha_std_error = math.sqrt(
            residual_variance / (segment.samples + SEGMENT_PRIOR_ROWS)
        )

        probability = min(0.99, max(0.01, market_probability + alpha))
        lower_probability = min(
            0.99,
            max(0.01, market_probability + alpha - LOWER_BOUND_Z * alpha_std_error),
        )
        break_even = 1.0 / decimal_odds
        z_score = (market_probability + alpha - break_even) / alpha_std_error
        probability_positive_ev = 0.5 * (1.0 + math.erf(z_score / math.sqrt(2.0)))
        expected_value = probability * decimal_odds - 1.0
        conservative_ev = lower_probability * decimal_odds - 1.0

        # VALUE lane: the source-level shrunk residual against posted prices.
        # Its EV question is "does this source beat its own break-even",
        # so the candidate anchor is the break-even probability, not no-vig.
        source_stats = source_value_stats(source)
        source_std_error = source_stats["alphaStdError"]
        source_probability_positive = source_stats["probabilityPositiveEv"]
        value_probability = min(0.99, max(0.01, break_even + source_alpha))
        value_lower_probability = min(
            0.99,
            max(0.01, break_even + source_alpha - LOWER_BOUND_Z * source_std_error),
        )
        source_flat_roi = source_stats["flatRoi"]
        source_first_half = source_stats["firstHalfFlatNetUnits"]
        source_second_half = source_stats["secondHalfFlatNetUnits"]

        estimate = {
            "marketProbability": round(market_probability, 6),
            "alpha": round(alpha, 6),
            "sourceAlpha": round(source_alpha, 6),
            "alphaStdError": round(alpha_std_error, 6),
            "probability": round(probability, 6),
            "lowerProbability": round(lower_probability, 6),
            "breakEvenProbability": round(break_even, 6),
            "expectedValue": round(expected_value, 6),
            "conservativeExpectedValue": round(conservative_ev, 6),
            "probabilityPositiveEv": round(probability_positive_ev, 6),
            "method": "market_baseline_plus_hierarchically_shrunk_prior_residual",
            "value": {
                "alpha": round(source_alpha, 6),
                "alphaStdError": round(source_std_error, 6),
                "probability": round(value_probability, 6),
                "lowerProbability": round(value_lower_probability, 6),
                "expectedValue": round(value_probability * decimal_odds - 1.0, 6),
                "conservativeExpectedValue": round(
                    value_lower_probability * decimal_odds - 1.0, 6
                ),
                "probabilityPositiveEv": round(source_probability_positive, 6),
                "method": "source_flat_roi_shrunk_residual_vs_break_even",
            },
        }
        flat_roi = segment.net_units / segment.samples if segment.samples else None
        first_half_net, second_half_net = segment.chronological_half_net_units
        evidence = {
            "sourceEvidenceKey": _key_text(source_key),
            "segmentEvidenceKey": _key_text(segment_key),
            "modelVersion": segment_key[3],
            "policyVersion": POLICY_VERSION,
            "sourceSamples": source.samples,
            "segmentSamples": segment.samples,
            "sourceDistinctDates": len(source.dates),
            "segmentDistinctDates": len(segment.dates),
            "distinctDates": len(segment.dates),
            "wins": segment.wins,
            "losses": segment.losses,
            "pushes": 0,
            "flatNetUnits": round(segment.net_units, 4),
            "flatRoi": round(flat_roi, 6) if flat_roi is not None else None,
            "firstHalfFlatNetUnits": round(first_half_net, 4),
            "secondHalfFlatNetUnits": round(second_half_net, 4),
            "chronologicalHalvesNonnegative": (
                first_half_net >= 0.0 and second_half_net >= 0.0
            ),
            "sourceWins": source.wins,
            "sourceLosses": source.losses,
            "sourceFlatNetUnits": round(source.net_units, 4),
            "sourceFlatRoi": (
                round(source_flat_roi, 6) if source_flat_roi is not None else None
            ),
            "sourceFirstHalfFlatNetUnits": round(source_first_half, 4),
            "sourceSecondHalfFlatNetUnits": round(source_second_half, 4),
            "sourceChronologicalHalvesNonnegative": (
                source_first_half >= 0.0 and source_second_half >= 0.0
            ),
            "priorOnly": True,
        }
        return estimate, evidence


# ---------------------------------------------------------------------------
# Candidate construction and selection
# ---------------------------------------------------------------------------


@dataclass
class RawCandidate:
    context: RecordContext
    date: str
    sport: str
    pick: str
    game: str
    canonical_game: str
    market_family: str
    market_identity: str
    direction: str
    line: float | None
    player: str
    odds: int | None
    decimal_odds: float | None
    price: dict[str, Any]
    price_tier: str
    price_tier_label: str
    no_vig: NoVigProbability
    grade_supported: bool
    grade_support_source: str
    base_blockers: list[str]


def _raw_candidate(context: RecordContext, date_iso: str) -> RawCandidate:
    record = context.record
    sport = _text(record.get("sport"))
    odds = _american_int(record.get("odds"))
    decimal = american_to_decimal(odds)
    executable, price_source, price_source_field, price_blockers = _price_provenance(
        record, context.source_key
    )
    timing = _timing(record, context.bucket, context.payload)
    no_vig = derive_no_vig_probability(record)
    price_tier, price_tier_label = _price_tier(
        record, executable=executable, no_vig=no_vig
    )
    grade_supported, grade_support_source = _grade_support(
        record, context.source_key, context.mode
    )
    blockers = list(price_blockers) + list(timing["blockers"])
    if not grade_supported:
        blockers.append("unsupported_grading")
    market_family = _market_family(record)
    direction = _direction(record)
    if price_tier == "A":
        public_price_quality = "verified_no_vig"
    elif price_tier == "B":
        public_price_quality = "verified_two_sided"
    elif price_tier == "C":
        public_price_quality = "one_sided"
    elif odds is None:
        public_price_quality = "missing"
    elif "stale_price" in timing["blockers"]:
        public_price_quality = "stale"
    else:
        public_price_quality = "assumed"
    price = {
        "observedExecutable": executable,
        "oddsAmerican": odds,
        "decimalOdds": round(decimal, 6) if decimal is not None else None,
        "source": price_source,
        "sourceField": price_source_field,
        "timestamp": timing["timestamp"],
        "timestampField": timing["timestampField"],
        "startTime": timing["startTime"],
        "ageHours": timing["ageHours"],
        "freshPregame": timing["freshPregame"],
        "maxAgeHours": timing["maxAgeHours"],
        "noVigVerified": no_vig.verified,
        "noVigMethod": no_vig.method,
        "noVigInputs": no_vig.inputs,
        "tier": price_tier,
        "tierLabel": price_tier_label,
        "eligibleForAlphaEstimate": price_tier in {"A", "B"},
        # Stable reader-facing aliases used by the static Profit Desk.
        "quality": public_price_quality,
        "updatedAt": timing["timestamp"],
        "fresh": timing["freshPregame"],
        "twoSided": no_vig.verified,
        "noVigProbability": (
            round(no_vig.probability, 6)
            if no_vig.probability is not None
            else None
        ),
        "breakEvenProbability": (
            round(1.0 / decimal, 6) if decimal is not None else None
        ),
    }
    return RawCandidate(
        context=context,
        date=date_iso,
        sport=sport,
        pick=_pick_text(record),
        game=_game_label(record),
        canonical_game=canonical_game_key(record, sport, date_iso),
        market_family=market_family,
        market_identity=canonical_market_identity(
            record, mode=context.mode, sport=sport, date_iso=date_iso
        ),
        direction=direction,
        line=_line(record, direction),
        player=_player(record),
        odds=odds,
        decimal_odds=decimal,
        price=price,
        price_tier=price_tier,
        price_tier_label=price_tier_label,
        no_vig=no_vig,
        grade_supported=grade_supported,
        grade_support_source=grade_support_source,
        base_blockers=list(dict.fromkeys(blockers)),
    )


def _dedupe_raw_candidates(candidates: Iterable[RawCandidate]) -> list[tuple[RawCandidate, list[RawCandidate]]]:
    grouped: dict[str, list[RawCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.market_identity].append(candidate)
    winners: list[tuple[RawCandidate, list[RawCandidate]]] = []
    for identity, group in grouped.items():
        # A genuinely executable price beats a proxy.  Among equally verified
        # copies, the highest decimal payout is the better executable price.
        ordered = sorted(
            group,
            key=lambda candidate: (
                0 if candidate.price_tier in {"A", "B"} else (
                    1 if candidate.price_tier == "C" else 2
                ),
                -(candidate.decimal_odds or 0.0),
                candidate.context.source_key,
                candidate.pick,
            ),
        )
        winners.append((ordered[0], ordered))
    return sorted(winners, key=lambda pair: (pair[0].context.mode, pair[0].market_identity))


def _candidate_payload(
    raw: RawCandidate,
    duplicates: Sequence[RawCandidate],
    evidence_book: EvidenceBook,
    *,
    live_slate: bool = True,
) -> dict[str, Any]:
    context = raw.context
    record = context.record
    model_version = _version(context, "model")
    policy_version = POLICY_VERSION
    structural_blockers = list(dict.fromkeys(raw.base_blockers))
    edge_blockers: list[str] = []
    value_blockers: list[str] = []
    estimate: dict[str, Any] | None = None
    evidence: dict[str, Any]

    baseline = (
        raw.no_vig.probability
        if raw.no_vig.verified and raw.no_vig.probability is not None
        else (1.0 / raw.decimal_odds if raw.decimal_odds is not None else None)
    )
    if raw.decimal_odds is not None and baseline is not None and raw.price_tier != "D":
        source_key, segment_key = _evidence_keys(
            context, baseline, raw.market_family, raw.direction
        )
        estimate, evidence = evidence_book.estimate(
            source_key, segment_key, baseline, raw.decimal_odds
        )
        # EDGE lane: strict segment-level market-alpha qualification.
        if raw.price_tier not in {"A", "B"}:
            edge_blockers.append("edge_requires_two_sided_price")
        if evidence["sourceSamples"] < MIN_SOURCE_SAMPLES:
            edge_blockers.append("edge_insufficient_source_samples")
        if evidence["segmentSamples"] < MIN_SEGMENT_SAMPLES:
            edge_blockers.append("edge_insufficient_segment_samples")
        if evidence["distinctDates"] < MIN_DISTINCT_DATES:
            edge_blockers.append("edge_insufficient_distinct_prior_dates")
        if not evidence["chronologicalHalvesNonnegative"]:
            edge_blockers.append("edge_negative_chronological_evidence_half")
        if estimate["probabilityPositiveEv"] < MIN_PROBABILITY_POSITIVE_EV:
            edge_blockers.append("edge_probability_positive_ev_below_0.80")
        if (
            estimate["lowerProbability"]
            < estimate["breakEvenProbability"] + MIN_CONSERVATIVE_PROBABILITY_MARGIN
        ):
            edge_blockers.append("edge_conservative_probability_margin_below_0.02")
        if estimate["conservativeExpectedValue"] <= 0.0:
            edge_blockers.append("edge_non_positive_conservative_ev")
        # VALUE lane: source-level flat-ROI qualification at posted prices.
        value_estimate = estimate["value"]
        if evidence["sourceSamples"] < VALUE_MIN_SOURCE_SAMPLES:
            value_blockers.append("value_insufficient_source_samples")
        if evidence["sourceDistinctDates"] < VALUE_MIN_SOURCE_DATES:
            value_blockers.append("value_insufficient_distinct_prior_dates")
        if not evidence["sourceChronologicalHalvesNonnegative"]:
            value_blockers.append("value_negative_chronological_evidence_half")
        if (evidence["sourceFlatRoi"] or 0.0) <= 0.0:
            value_blockers.append("value_non_positive_flat_roi")
        if value_estimate["probabilityPositiveEv"] < VALUE_MIN_PROBABILITY_POSITIVE_EV:
            value_blockers.append("value_probability_positive_ev_below_0.70")
    else:
        source_key = (
            context.mode,
            context.source_key,
            _norm(record.get("sport")) or "unknown_sport",
        )
        evidence = {
            "sourceEvidenceKey": _key_text(source_key),
            "segmentEvidenceKey": None,
            "modelVersion": model_version,
            "policyVersion": policy_version,
            "sourceSamples": evidence_book.by_source[source_key].samples,
            "segmentSamples": 0,
            "sourceDistinctDates": len(evidence_book.by_source[source_key].dates),
            "segmentDistinctDates": 0,
            "distinctDates": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "flatNetUnits": 0.0,
            "flatRoi": None,
            "firstHalfFlatNetUnits": 0.0,
            "secondHalfFlatNetUnits": 0.0,
            "chronologicalHalvesNonnegative": False,
            "sourceWins": 0,
            "sourceLosses": 0,
            "sourceFlatNetUnits": 0.0,
            "sourceFlatRoi": None,
            "sourceFirstHalfFlatNetUnits": 0.0,
            "sourceSecondHalfFlatNetUnits": 0.0,
            "sourceChronologicalHalvesNonnegative": False,
            "priorOnly": True,
        }
        edge_blockers.append("edge_no_usable_price_baseline")
        value_blockers.append("value_no_usable_price_baseline")

    edge_qualified = not structural_blockers and not edge_blockers
    value_qualified = not structural_blockers and not value_blockers
    qualified = edge_qualified or value_qualified
    if edge_qualified:
        tier = "edge"
        lane = "edge"
        stake_units = EDGE_STAKE_UNITS if live_slate else 0.0
    elif value_qualified:
        tier = "value"
        lane = "value"
        stake_units = VALUE_STAKE_UNITS if live_slate else 0.0
    elif not structural_blockers:
        tier = "watch"
        lane = None
        stake_units = 0.0
    else:
        tier = "avoid"
        lane = None
        stake_units = 0.0
    live_qualified = qualified and live_slate
    # Qualified candidates show a clean card; the lanes they did NOT clear
    # stay inspectable in laneBlockers.
    blockers = (
        []
        if qualified
        else list(dict.fromkeys(structural_blockers + edge_blockers + value_blockers))
    )
    shadow_qualified = qualified
    candidate_id = "profit-" + _stable_hash(
        {
            "date": raw.date,
            "market": raw.market_identity,
            "source": context.source_key,
            "odds": raw.odds,
            "modelVersion": model_version,
            "policyVersion": policy_version,
        },
        24,
    )
    raw_probability = normalize_probability(
        _first(
            record,
            "raw_probability",
            "ml_raw_probability",
            "model_probability",
            "probability",
        )
    )
    return {
        "id": candidate_id,
        "date": raw.date,
        "mode": context.mode,
        "sport": raw.sport,
        "sourceKey": context.source_key,
        "source": context.source,
        "modelVersion": model_version,
        "policyVersion": policy_version,
        "pick": raw.pick,
        "decision": _text(record.get("decision")).upper(),
        "result": _result(record.get("result")),
        "game": raw.game,
        "canonicalGame": raw.canonical_game,
        "market": _text(_first(record, "market_type", "market", "stat_label")) or raw.market_family,
        "marketFamily": raw.market_family,
        "marketIdentity": raw.market_identity,
        "player": raw.player or None,
        "direction": raw.direction,
        "line": raw.line,
        "oddsAmerican": raw.odds,
        "decimalOdds": round(raw.decimal_odds, 6) if raw.decimal_odds is not None else None,
        "rawModelProbabilityIgnored": round(raw_probability, 6) if raw_probability is not None else None,
        "gradeSupported": raw.grade_supported,
        "gradeSupportSource": raw.grade_support_source,
        "price": raw.price,
        "estimate": estimate,
        "evidence": {**evidence, "cutoffExclusive": raw.date},
        "tier": tier,
        "lane": lane,
        "blockers": blockers,
        "laneBlockers": {
            "structural": structural_blockers,
            "edge": edge_blockers,
            "value": value_blockers,
        },
        "shadowQualified": shadow_qualified,
        "edgeQualified": edge_qualified,
        "valueQualified": value_qualified,
        "liveQualified": live_qualified,
        "stakeUnits": round(stake_units, 2),
        "duplicateCount": len(duplicates),
        "duplicateSources": sorted(
            {candidate.context.source for candidate in duplicates}
        ),
        "consensusSources": sorted(
            {candidate.context.source for candidate in duplicates}
        ) if len({candidate.context.source for candidate in duplicates}) > 1 else [],
        "dedupeRule": "exact_market_identity_best_executable_price",
    }


def _portfolio_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    estimate = candidate.get("estimate") or {}
    lane = _text(candidate.get("lane"))
    if lane == "value":
        lane_estimate = estimate.get("value") or {}
    else:
        lane_estimate = estimate
    return (
        0 if lane == "edge" else 1,
        -float(lane_estimate.get("conservativeExpectedValue") or -999),
        -float(lane_estimate.get("probabilityPositiveEv") or 0),
        -int((candidate.get("evidence") or {}).get("segmentSamples") or 0),
        _text(candidate.get("id")),
    )


def select_portfolio(candidates: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Select the ranked card from lane-qualified candidates."""

    qualified = [
        candidate for candidate in candidates if candidate.get("tier") in {"edge", "value"}
    ]
    ordered = sorted(qualified, key=_portfolio_sort_key)
    selected: list[dict[str, Any]] = []
    mode_counts: dict[str, int] = defaultdict(int)
    used_games: set[str] = set()
    for candidate in ordered:
        mode = _text(candidate.get("mode"))
        game = _text(candidate.get("canonicalGame"))
        if mode_counts[mode] >= MAX_PER_MODE or game in used_games:
            continue
        selected.append({
            **dict(candidate),
            "portfolioSelected": True,
            "rank": len(selected) + 1,
        })
        mode_counts[mode] += 1
        used_games.add(game)
    live = [candidate for candidate in selected if candidate.get("liveQualified")]
    return {
        "team": [candidate for candidate in selected if candidate.get("mode") == "team"],
        "player": [candidate for candidate in selected if candidate.get("mode") == "player"],
        "all": selected,
        "shadow": [],
        "live": live,
    }


def _source_report_cards(
    evidence_book: EvidenceBook, candidates: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per-source VALUE-lane gate progress for the qualification leaderboard.

    Every number is already computed by the evidence book; this only lays the
    gates out so the site can show how far each source is from earning a
    stake, instead of a bare rejection.
    """

    labels: dict[tuple[str, ...], str] = {}
    candidates_today: dict[tuple[str, ...], int] = defaultdict(int)
    live_today: dict[tuple[str, ...], int] = defaultdict(int)
    for candidate in candidates:
        key = (
            _text(candidate.get("mode")),
            _text(candidate.get("sourceKey")),
            _norm(candidate.get("sport")) or "unknown_sport",
        )
        labels.setdefault(key, _text(candidate.get("source")) or key[1])
        candidates_today[key] += 1
        if candidate.get("liveQualified"):
            live_today[key] += 1

    keys = set(evidence_book.by_source) | set(candidates_today)
    cards: list[dict[str, Any]] = []
    for key in keys:
        if len(key) != 3:
            continue
        stats = source_value_stats(evidence_book.by_source[key])
        probability_positive = round(stats["probabilityPositiveEv"], 6)
        flat_roi = stats["flatRoi"]
        gates = {
            "sourceSamples": {
                "required": VALUE_MIN_SOURCE_SAMPLES,
                "actual": stats["samples"],
                "passed": stats["samples"] >= VALUE_MIN_SOURCE_SAMPLES,
            },
            "distinctPriorDates": {
                "required": VALUE_MIN_SOURCE_DATES,
                "actual": stats["distinctDates"],
                "passed": stats["distinctDates"] >= VALUE_MIN_SOURCE_DATES,
            },
            "positiveFlatRoi": {
                "required": 0.0,
                "actual": round(flat_roi, 6) if flat_roi is not None else None,
                "passed": (flat_roi or 0.0) > 0.0,
            },
            "stableChronologicalHalves": {
                "required": True,
                "actual": stats["halvesNonnegative"],
                "passed": bool(stats["halvesNonnegative"] and stats["samples"]),
            },
            "probabilityPositiveEv": {
                "required": VALUE_MIN_PROBABILITY_POSITIVE_EV,
                "actual": probability_positive,
                "passed": probability_positive >= VALUE_MIN_PROBABILITY_POSITIVE_EV,
            },
        }
        gates_passed = sum(1 for gate in gates.values() if gate["passed"])
        cards.append(
            {
                "mode": key[0],
                "sourceKey": key[1],
                "sport": key[2],
                "source": labels.get(key) or key[1],
                "samples": stats["samples"],
                "distinctDates": stats["distinctDates"],
                "wins": stats["wins"],
                "losses": stats["losses"],
                "flatNetUnits": round(stats["flatNetUnits"], 4),
                "flatRoi": round(flat_roi, 6) if flat_roi is not None else None,
                "alpha": round(stats["alpha"], 6),
                "probabilityPositiveEv": probability_positive,
                "gates": gates,
                "gatesPassed": gates_passed,
                "gatesTotal": len(gates),
                "evidenceQualified": gates_passed == len(gates),
                "candidatesToday": candidates_today.get(key, 0),
                "liveToday": live_today.get(key, 0),
            }
        )
    cards.sort(
        key=lambda card: (
            -int(card["gatesPassed"]),
            -float(card["probabilityPositiveEv"]),
            -int(card["samples"]),
            str(card["sourceKey"]),
        )
    )
    return cards


def _flat_record(
    candidates: Iterable[Mapping[str, Any]], *, stake_weighted: bool = False
) -> dict[str, Any]:
    wins = losses = pushes = pending = 0
    net = 0.0
    staked = 0.0
    clv_values: list[float] = []
    for candidate in candidates:
        result = _result(candidate.get("result"))
        decimal = _number(candidate.get("decimalOdds"))
        stake = _number(candidate.get("stakeUnits")) if stake_weighted else 1.0
        if stake is None or stake <= 0.0:
            stake = 0.0 if stake_weighted else 1.0
        closing = candidate.get("closing")
        if isinstance(closing, Mapping):
            clv = _number(closing.get("clv"))
            if clv is not None:
                clv_values.append(clv)
        if result == "win" and decimal is not None:
            wins += 1
            net += stake * (decimal - 1.0)
            staked += stake
        elif result == "loss":
            losses += 1
            net -= stake
            staked += stake
        elif result == "push":
            pushes += 1
        else:
            pending += 1
    settled = wins + losses
    denominator = staked if stake_weighted else float(settled)
    return {
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "pending": pending,
        "settled": settled,
        "netUnits": round(net, 4),
        "stakedUnits": round(staked, 4),
        "roi": round(net / denominator, 6) if denominator else None,
        "clvCount": len(clv_values),
        "avgClv": (
            round(sum(clv_values) / len(clv_values), 6) if clv_values else None
        ),
    }


def _deterministic_generated_at(
    date_iso: str,
    team_payload: Mapping[str, Any] | None,
    prop_payload: Mapping[str, Any] | None,
) -> str:
    timestamps: list[tuple[datetime, str]] = []
    for payload in (team_payload, prop_payload):
        if not isinstance(payload, Mapping):
            continue
        raw = _text(_first(payload, "updatedAt", "generatedAt"))
        parsed = _parse_timestamp(raw)
        if parsed is not None:
            timestamps.append((parsed, parsed.isoformat().replace("+00:00", "Z")))
    if timestamps:
        return max(timestamps, key=lambda item: item[0])[1]
    return f"{date_iso}T00:00:00Z"


def build_profit_desk_payload(
    date_iso: str,
    team_payload: Mapping[str, Any] | None,
    prop_payload: Mapping[str, Any] | None,
    *,
    team_history: Iterable[Mapping[str, Any]] | None = None,
    prop_history: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one deterministic Profit Desk slate payload.

    Supplied history is still filtered by each record's own date, preventing a
    fixture or caller from accidentally introducing same-date/future leakage.
    """

    if team_history is None:
        team_history = _payloads_before(MODEL_CACHE_DIR, date_iso)
    if prop_history is None:
        prop_history = _payloads_before(PLAYER_PROPS_CACHE_DIR, date_iso)
    evidence_book = EvidenceBook.build(date_iso, team_history, prop_history)

    raw_candidates: list[RawCandidate] = []
    input_count = 0
    for mode, payload in (("team", team_payload), ("player", prop_payload)):
        for context in _iter_records(payload, mode):
            record = context.record
            if _record_date(record, context.fallback_date) != date_iso:
                continue
            if _text(record.get("decision")).upper() not in VISIBLE_DECISIONS:
                continue
            if record.get("shadow_mode") is True:
                continue
            input_count += 1
            raw_candidates.append(_raw_candidate(context, date_iso))

    live_slate = date_iso >= FIRST_LIVE_DATE
    candidates = [
        _candidate_payload(winner, duplicates, evidence_book, live_slate=live_slate)
        for winner, duplicates in _dedupe_raw_candidates(raw_candidates)
    ]
    candidates.sort(
        key=lambda candidate: (
            {"edge": 0, "value": 1, "watch": 2, "avoid": 3}.get(_text(candidate.get("tier")), 4),
            *_portfolio_sort_key(candidate)[1:],
            _text(candidate.get("mode")),
            _text(candidate.get("id")),
        )
    )
    portfolio = select_portfolio(candidates)
    qualified = sum(candidate["shadowQualified"] for candidate in candidates)
    edge_qualified = sum(candidate["edgeQualified"] for candidate in candidates)
    value_qualified = sum(candidate["valueQualified"] for candidate in candidates)
    live_qualified = sum(candidate["liveQualified"] for candidate in candidates)
    watchlist = sum(candidate["tier"] == "watch" for candidate in candidates)
    observed = sum(candidate["price"]["observedExecutable"] for candidate in candidates)

    mode_summary: dict[str, Any] = {}
    for mode in ("team", "player"):
        rows = [candidate for candidate in candidates if candidate["mode"] == mode]
        mode_evidence_rows = sum(
            1 for row in evidence_book.rows if row.source_key[0] == mode
        )
        mode_observed = sum(candidate["price"]["observedExecutable"] for candidate in rows)
        mode_live = [candidate for candidate in portfolio[mode] if candidate.get("liveQualified")]
        mode_summary[mode] = {
            "candidates": len(rows),
            "candidateCount": len(rows),
            "candidatesEvaluated": len(rows),
            "observedPriceCandidates": mode_observed,
            "shadowQualified": sum(candidate["shadowQualified"] for candidate in rows),
            "researchQualified": sum(candidate["shadowQualified"] for candidate in rows),
            "edgeQualified": sum(candidate["edgeQualified"] for candidate in rows),
            "valueQualified": sum(candidate["valueQualified"] for candidate in rows),
            "watchlist": sum(candidate["tier"] == "watch" for candidate in rows),
            "avoid": sum(candidate["tier"] == "avoid" for candidate in rows),
            "selected": len(portfolio[mode]),
            "portfolioCandidates": len(portfolio[mode]),
            "liveQualified": len(mode_live),
            "evidenceRows": mode_evidence_rows,
        }

    live_record = _flat_record(portfolio["live"], stake_weighted=True)
    research_record = _flat_record(portfolio["all"])
    source_cards = _source_report_cards(evidence_book, candidates)
    policy = {
        "version": POLICY_VERSION,
        "status": "LIVE" if live_slate else "RESEARCH_BACKFILL",
        "statusLabel": "live" if live_slate else "research backfill",
        "mode": "live" if live_slate else "research_backfill",
        "firstLiveDate": FIRST_LIVE_DATE,
        "liveStaking": live_slate,
        "gates": {
            "structural": {
                "observedExecutableOdds": True,
                "freshPregameTimestamp": True,
                "maximumPriceAgeHours": MAX_PRICE_AGE_HOURS,
                "gradeSupported": True,
                "selectionPolicyVersion": POLICY_VERSION,
            },
            "edgeLane": {
                "stakeUnits": EDGE_STAKE_UNITS,
                "requiresTwoSidedNoVigPrice": True,
                "minimumSourceSamples": MIN_SOURCE_SAMPLES,
                "minimumSegmentSamples": MIN_SEGMENT_SAMPLES,
                "minimumDistinctPriorDates": MIN_DISTINCT_DATES,
                "minimumProbabilityPositiveEv": MIN_PROBABILITY_POSITIVE_EV,
                "minimumConservativeProbabilityMargin": MIN_CONSERVATIVE_PROBABILITY_MARGIN,
                "chronologicalEvidenceHalvesMustBeNonnegative": True,
                "segmentsNeverMixModelVersions": True,
            },
            "valueLane": {
                "stakeUnits": VALUE_STAKE_UNITS,
                "baseline": "posted price break-even (vig included), never a fabricated no-vig",
                "minimumSourceSamples": VALUE_MIN_SOURCE_SAMPLES,
                "minimumDistinctPriorDates": VALUE_MIN_SOURCE_DATES,
                "minimumProbabilityPositiveEv": VALUE_MIN_PROBABILITY_POSITIVE_EV,
                "requiresPositiveFlatRoi": True,
                "chronologicalEvidenceHalvesMustBeNonnegative": True,
            },
            "portfolio": {
                "maximumPerMode": MAX_PER_MODE,
                "maximumPerCanonicalGame": 1,
            },
        },
        "notes": [
            "EDGE picks stake 1.0u after strict segment-level market-alpha gates.",
            "VALUE picks stake 0.5u after source-level flat-ROI gates at posted prices.",
            "Raw model probability and consensus are display context only and never create edge.",
            "Evidence uses settled, executable-priced rows dated strictly before the target slate.",
            f"Live staking begins {FIRST_LIVE_DATE}; earlier slates rebuild as zero-stake research.",
        ],
    }
    summary = {
        "inputPicks": input_count,
        "candidateCount": len(candidates),
        "candidatesEvaluated": len(candidates),
        "deduplicatedPicks": input_count - len(candidates),
        "observedPriceCandidates": observed,
        "shadowQualified": qualified,
        "researchQualified": qualified,
        "edgeQualified": edge_qualified,
        "valueQualified": value_qualified,
        "watchlist": watchlist,
        "avoid": sum(candidate["tier"] == "avoid" for candidate in candidates),
        "selected": len(portfolio["all"]),
        "portfolioCandidates": len(portfolio["all"]),
        "shadowPortfolioCandidates": 0,
        "livePortfolioCandidates": len(portfolio["live"]),
        "liveQualified": live_qualified,
        "evidenceRows": len(evidence_book.rows),
        "sourcesTracked": len(source_cards),
        "modes": mode_summary,
        "shadowRecord": research_record,
        "researchRecord": research_record,
        "liveRecord": live_record,
    }
    notices = [
        "Qualified picks carry real flat stakes: EDGE 1.0u, VALUE 0.5u; everything else stays 0u.",
        "The market price is the baseline; historical residual alpha is shrunk and uncertainty-adjusted.",
        "A 3-0 streak is still insufficient: the VALUE lane needs 150 source rows over 15 dates with stable halves.",
        "Opposing Over/Under selections remain separate markets and receive no consensus bonus.",
    ]
    if not live_slate:
        notices.append(
            f"This slate predates the {FIRST_LIVE_DATE} live cutover, so every stake is 0u research."
        )
    if not qualified:
        notices.append("No candidates cleared a qualification lane on this slate; zero action is a valid result.")

    return {
        "schemaVersion": 2,
        "date": date_iso,
        "generatedAt": _deterministic_generated_at(date_iso, team_payload, prop_payload),
        "engineVersion": ENGINE_VERSION,
        "phase": "live" if live_slate else "research_backfill",
        "cutoverDate": ENGINE_CUTOVER_DATE,
        "firstLiveDate": FIRST_LIVE_DATE,
        "policy": policy,
        "summary": summary,
        "portfolio": portfolio,
        "candidates": candidates,
        "sources": source_cards,
        "notices": notices,
    }


# ---------------------------------------------------------------------------
# Rebuild / CLI
# ---------------------------------------------------------------------------


def _payloads_before(directory: Path, date_iso: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for path in sorted(directory.glob("20??-??-??.json")):
        if path.stem >= date_iso:
            continue
        payload = _read_json(path)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _closing_from_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract the last pregame-captured price for the record's own side.

    Captured pregame prices are preserved by the merge layer once a game goes
    live, so the record's final market fields are the closest observation to
    the closing line this pipeline has.  A capture stamped after the start
    time is never trusted as closing.
    """

    captured_raw = _first(record, "market_odds_captured_at", "market_updated_at")
    captured = _parse_timestamp(captured_raw)
    start = _parse_timestamp(
        _first(record, "game_start_time", "start_time", "event_start_time")
    )
    if captured is None or (start is not None and captured > start):
        return None
    direction = _direction(record)
    selected = _american_int(record.get("selected_odds"))
    opposite = _american_int(record.get("opposite_odds"))
    if selected is None and direction in {"over", "under"}:
        over = _american_int(_first(record, "market_over_odds", "over_odds"))
        under = _american_int(_first(record, "market_under_odds", "under_odds"))
        if over is not None and under is not None:
            selected = over if direction == "over" else under
            opposite = under if direction == "over" else over
    if selected is None:
        return None
    decimal = american_to_decimal(selected)
    no_vig: float | None = None
    if opposite is not None:
        selected_implied = implied_probability(selected)
        opposite_implied = implied_probability(opposite)
        if selected_implied and opposite_implied:
            no_vig = selected_implied / (selected_implied + opposite_implied)
    explicit_no_vig = normalize_probability(
        record.get("market_no_vig_selected_probability")
    )
    if explicit_no_vig is not None:
        no_vig = explicit_no_vig
    return {
        "oddsAmerican": selected,
        "decimalOdds": round(decimal, 6) if decimal is not None else None,
        "noVigProbability": round(no_vig, 6) if no_vig is not None else None,
        "capturedAt": _text(captured_raw) or None,
        "provider": _text(record.get("market_odds_provider")) or None,
    }


def _closing_ledger_for_date(date_iso: str) -> dict[str, dict[str, Any]]:
    """Latest anchor-provider closing row per market identity for the slate.

    Rows come from the near-close capture cron (``capture_closing_lines``).
    Sharp-book rows (``role == 'sharp'``) are journaled for analysis but the
    primary CLV baseline stays on the anchor provider so the metric is
    consistent across picks with and without sharp coverage.
    """

    payload = _read_json(CLOSING_LINES_DIR / f"{date_iso}.json")
    rows = payload.get("rows") if isinstance(payload, dict) else None
    latest: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if _text(row.get("role") or "anchor") != "anchor":
            continue
        identity = _text(row.get("marketIdentity"))
        captured = _parse_timestamp(row.get("capturedAt"))
        if not identity or captured is None:
            continue
        start = _parse_timestamp(row.get("startTime"))
        if start is not None and captured > start:
            continue
        current = latest.get(identity)
        current_captured = _parse_timestamp(current.get("capturedAt")) if current else None
        if current_captured is None or captured > current_captured:
            latest[identity] = row
    return latest


def _grade_sync_maps(
    model_dir: Path, player_dir: Path, date_iso: str
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], dict[str, Any]]]:
    """Map (source_key, market_identity) to settled results and closing prices."""

    results: dict[tuple[str, str], str] = {}
    closings: dict[tuple[str, str], dict[str, Any]] = {}
    ledger = _closing_ledger_for_date(date_iso)
    for mode, directory in (("team", model_dir), ("player", player_dir)):
        payload = _read_json(directory / f"{date_iso}.json")
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
            key = (context.source_key, identity)
            result = _result(record.get("result"))
            if result != "pending":
                results[key] = result
            closing = _closing_from_record(record)
            ledger_row = ledger.get(identity)
            if ledger_row is not None:
                ledger_captured = _parse_timestamp(ledger_row.get("capturedAt"))
                record_captured = (
                    _parse_timestamp(closing.get("capturedAt")) if closing else None
                )
                if ledger_captured is not None and (
                    record_captured is None or ledger_captured > record_captured
                ):
                    closing = {
                        "oddsAmerican": ledger_row.get("oddsAmerican"),
                        "decimalOdds": ledger_row.get("decimalOdds"),
                        "noVigProbability": ledger_row.get("noVigProbability"),
                        "capturedAt": _text(ledger_row.get("capturedAt")) or None,
                        "provider": _text(ledger_row.get("provider")) or None,
                    }
            if closing is not None:
                closings.setdefault(key, closing)
    return results, closings


def _result_map_for_date(
    model_dir: Path, player_dir: Path, date_iso: str
) -> dict[tuple[str, str], str]:
    """Map (source_key, market_identity) to the latest settled result."""

    return _grade_sync_maps(model_dir, player_dir, date_iso)[0]


def _sync_artifact_results(
    destination: Path, model_dir: Path, player_dir: Path
) -> int:
    """Refresh result and closing fields on frozen artifacts as caches grade.

    Selection is never re-run here: candidates, stakes, and ranks stay exactly
    as published; only each pick's settled outcome, its closing-price
    observation, and the derived records move.  This keeps the live record
    prospective while letting it settle.
    """

    changed = 0
    for path in sorted(destination.glob("20??-??-??.json")):
        if path.stem < ENGINE_CUTOVER_DATE:
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        portfolio = payload.get("portfolio") if isinstance(payload.get("portfolio"), dict) else {}
        rows = [
            row
            for row in payload.get("candidates") or []
            if isinstance(row, dict)
        ] + [
            row
            for bucket in portfolio.values()
            for row in (bucket if isinstance(bucket, list) else [])
            if isinstance(row, dict)
        ]
        needs_sync = [
            row
            for row in rows
            if _result(row.get("result")) == "pending" or "closing" not in row
        ]
        if not needs_sync:
            continue
        results, closings = _grade_sync_maps(model_dir, player_dir, path.stem)
        if not results and not closings:
            continue
        updated = False
        for row in needs_sync:
            key = (_text(row.get("sourceKey")), _text(row.get("marketIdentity")))
            result = results.get(key)
            if result and result != _result(row.get("result")):
                row["result"] = result
                updated = True
            # Closing attaches once, only after the pick settles, so the value
            # can never drift while a game is still being priced.
            if "closing" not in row and _result(row.get("result")) in {"win", "loss", "push"}:
                closing = closings.get(key)
                if closing is not None:
                    entry_decimal = _number(row.get("decimalOdds"))
                    closing_decimal = _number(closing.get("decimalOdds"))
                    clv = (
                        round(entry_decimal / closing_decimal - 1.0, 6)
                        if entry_decimal and closing_decimal
                        else None
                    )
                    row["closing"] = {**closing, "clv": clv}
                    updated = True
        if not updated:
            continue
        summary = payload.get("summary")
        if isinstance(summary, dict) and isinstance(portfolio, dict):
            live_rows = [row for row in portfolio.get("live") or [] if isinstance(row, dict)]
            all_rows = [row for row in portfolio.get("all") or [] if isinstance(row, dict)]
            summary["liveRecord"] = _flat_record(live_rows, stake_weighted=True)
            research_record = _flat_record(all_rows)
            summary["shadowRecord"] = research_record
            summary["researchRecord"] = research_record
        if _write_json_if_changed(path, payload):
            changed += 1
    return changed


def _live_rows_from_artifacts(
    destination: Path, *, before: str | None = None, through: str | None = None
) -> list[dict[str, Any]]:
    live_rows: list[dict[str, Any]] = []
    for path in sorted(destination.glob("20??-??-??.json")):
        if path.stem < FIRST_LIVE_DATE:
            continue
        if before is not None and path.stem >= before:
            continue
        if through is not None and path.stem > through:
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        portfolio = payload.get("portfolio") if isinstance(payload.get("portfolio"), dict) else {}
        live_rows.extend(
            row for row in portfolio.get("live") or [] if isinstance(row, dict)
        )
    return live_rows


def _cumulative_live_record(
    live_rows: Sequence[Mapping[str, Any]], through_date: str
) -> dict[str, Any]:
    """Stake-weighted record of every live pick from FIRST_LIVE_DATE onward."""

    record = _flat_record(live_rows, stake_weighted=True)
    record["sinceDate"] = FIRST_LIVE_DATE
    record["throughDate"] = through_date
    return record


def _target_dates(
    *,
    date_iso: str | None,
    all_dates: bool,
    model_cache_dir: Path,
    player_cache_dir: Path,
) -> list[str]:
    if date_iso:
        return [date_iso]
    dates = {
        path.stem
        for directory in (model_cache_dir, player_cache_dir)
        for path in directory.glob("20??-??-??.json")
        if _DATE_FILE_RE.match(path.stem)
    }
    if all_dates:
        return sorted(dates)
    latest_dates: list[str] = []
    for directory in (model_cache_dir, player_cache_dir):
        latest = _read_json(directory / "latest.json") or {}
        value = _text(_first(latest, "date", "slate_date"))
        if value:
            latest_dates.append(value)
    if latest_dates:
        return [max(latest_dates)]
    return [max(dates)] if dates else []


def rebuild_profit_desk(
    *,
    date_iso: str | None = None,
    all_dates: bool = False,
    model_cache_dir: Path | str | None = None,
    player_cache_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    today_iso: str | None = None,
) -> int:
    """Write dated, latest, and index files; return the changed-file count."""

    model_dir = Path(model_cache_dir) if model_cache_dir is not None else MODEL_CACHE_DIR
    player_dir = Path(player_cache_dir) if player_cache_dir is not None else PLAYER_PROPS_CACHE_DIR
    destination = Path(output_dir) if output_dir is not None else PROFIT_DESK_DIR
    today = today_iso or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    targets = _target_dates(
        date_iso=date_iso,
        all_dates=all_dates,
        model_cache_dir=model_dir,
        player_cache_dir=player_dir,
    )
    changed = 0
    for target in targets:
        if target < ENGINE_CUTOVER_DATE:
            print(f"[profit-desk] skipped {target}: predates cutover {ENGINE_CUTOVER_DATE}")
            continue
        if target < today and (destination / f"{target}.json").exists():
            # A published past slate is frozen: re-running selection against
            # its since-evolved caches would rewrite history with hindsight.
            # Results and closing prices still flow via the sync pass below.
            print(f"[profit-desk] skipped {target}: published artifact is frozen (sync-only)")
            continue
        team_payload = _read_json(model_dir / f"{target}.json")
        prop_payload = _read_json(player_dir / f"{target}.json")
        if team_payload is None and prop_payload is None:
            print(f"[profit-desk] skipped {target}: no committed source cache")
            continue
        payload = build_profit_desk_payload(
            target,
            team_payload,
            prop_payload,
            team_history=_payloads_before(model_dir, target),
            prop_history=_payloads_before(player_dir, target),
        )
        prior_live_rows = _live_rows_from_artifacts(destination, before=target)
        payload["summary"]["liveRecordToDate"] = _cumulative_live_record(
            prior_live_rows + list(payload["portfolio"]["live"]), target
        )
        if _write_json_if_changed(destination / f"{target}.json", payload):
            changed += 1
        print(
            f"[profit-desk] {target}: {payload['summary']['candidateCount']} candidate(s), "
            f"{payload['summary']['researchQualified']} qualified, "
            f"{payload['summary']['liveQualified']} live"
        )

    changed += _sync_artifact_results(destination, model_dir, player_dir)

    files = sorted(path.name for path in destination.glob("20??-??-??.json"))
    if files:
        # Absorb result syncs into the newest artifact's cumulative record.
        latest_date = files[-1].removesuffix(".json")
        latest_dated = _read_json(destination / files[-1])
        if latest_dated is not None and isinstance(latest_dated.get("summary"), dict):
            cumulative = _cumulative_live_record(
                _live_rows_from_artifacts(destination, through=latest_date), latest_date
            )
            if latest_dated["summary"].get("liveRecordToDate") != cumulative:
                latest_dated["summary"]["liveRecordToDate"] = cumulative
                if _write_json_if_changed(destination / files[-1], latest_dated):
                    changed += 1
    manifest = {
        "engineVersion": ENGINE_VERSION,
        "cutoverDate": ENGINE_CUTOVER_DATE,
        "firstLiveDate": FIRST_LIVE_DATE,
        "files": files,
    }
    if _write_json_if_changed(destination / "index.json", manifest):
        changed += 1
    if files:
        latest_payload = _read_json(destination / files[-1])
        if latest_payload is not None and _write_json_if_changed(
            destination / "latest.json", latest_payload
        ):
            changed += 1
    return changed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Target date in YYYY-MM-DD format.")
    parser.add_argument(
        "--all", action="store_true", help="Build all cache dates at or after the cutover."
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    changed = rebuild_profit_desk(date_iso=args.date, all_dates=args.all)
    print(f"[profit-desk] complete: {changed} file update(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
