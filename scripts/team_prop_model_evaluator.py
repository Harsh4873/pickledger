#!/usr/bin/env python3
"""Evaluate certified immutable in-house team-prop prediction snapshots.

This module intentionally reads the pre-game ledger rather than the mutable
``data/model_cache`` files.  It is an audit tool: it does not train, calibrate,
grade, alter pick decisions, or infer financial results from assumed prices.

Supported model keys are limited to the in-house team markets under review:
``mlb_new``, ``mlb_first_five``, ``mlb_inning``, ``fifa_world_cup``, and
``nba_summer``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
SUPPORTED_MODEL_KEYS = (
    "mlb_new",
    "mlb_first_five",
    "mlb_inning",
    "fifa_world_cup",
    "nba_summer",
)
UNVERSIONED = "unversioned"


# These are audit-only serving contracts distilled from the currently deployed
# model inputs.  They deliberately test whether the immutable pre-game record
# retained the inputs needed to replay/audit a prediction; they do not modify
# any model formula or synthesize missing values.
FEATURE_CONTRACTS: dict[str, dict[str, Any]] = {
    "mlb_new": {
        "contract_version": "mlb_new_v2_serving_groups",
        "groups": (
            ("market_moneyline", ("market_home_ml", "market_away_ml", "market_home_vigfree_prob")),
            ("market_total", ("market_total_line", "totals_line")),
            ("team_strength", ("home_games_played", "home_season_win_pct", "home_runs_scored_season")),
            ("starting_pitchers", ("home_starter_era", "away_starter_era", "home_starter_ip")),
            ("lineup_proxy", ("home_lineup_ops_proxy", "away_lineup_ops_proxy")),
            ("bullpen", ("home_bullpen_era_30d", "away_bullpen_era_30d")),
            ("environment", ("park_factor_runs", "temperature_f", "wind_speed_mph")),
        ),
    },
    "mlb_first_five": {
        "contract_version": "mlb_first_five_serving_groups",
        "groups": (
            ("away_offense", ("features.away_offense",)),
            ("home_offense", ("features.home_offense",)),
            ("away_pitcher", ("features.away_pitcher",)),
            ("home_pitcher", ("features.home_pitcher",)),
            ("lineup_matchups", ("features.away_lineup_matchup", "features.home_lineup_matchup")),
            ("venue", ("features.venue",)),
            ("travel", ("features.travel",)),
        ),
    },
    "mlb_inning": {
        "contract_version": "mlb_inning_serving_groups",
        "groups": (
            ("home_pitcher", ("home_pitcher_context", "features.home_pitcher_context")),
            ("away_pitcher", ("away_pitcher_context", "features.away_pitcher_context")),
            ("lineup_or_matchup", ("home_lineup", "away_lineup", "matchup_threats", "features.matchup_threats")),
            ("team_inning_history", ("team_histories", "features.team_histories", "inning_history")),
            ("bullpen", ("home_pitcher_context.team_bullpen", "away_pitcher_context.team_bullpen")),
            ("environment", ("weather", "venue", "features.weather", "features.venue")),
            ("travel", ("travel", "features.travel")),
        ),
    },
    "fifa_world_cup": {
        "contract_version": "fifa_world_cup_player_power_serving_groups",
        "groups": (
            ("home_squad_units", ("home_unit_ratings",)),
            ("away_squad_units", ("away_unit_ratings",)),
            ("home_tournament_form", ("home_tournament_form",)),
            ("away_tournament_form", ("away_tournament_form",)),
            ("venue_context", ("venue_profile",)),
            ("goal_projection", ("raw_projected_home_goals", "projected_home_goals")),
            ("market_context", ("market_total_line", "market_probability")),
        ),
    },
    "nba_summer": {
        "contract_version": "nba_summer_v1_serving_groups",
        "groups": (
            ("sample_size", ("sample_games", "projection.min_sample")),
            ("team_profiles", ("projection.home_profile", "projection.away_profile")),
            ("recent_form", ("projection.home_profile.recent_margin", "projection.away_profile.recent_margin")),
            ("rest", ("projection.home_rest_days", "projection.away_rest_days")),
            ("site_context", ("neutral_site", "venue", "tournament")),
            ("market_context", ("market_pick_odds", "market_pick_prob", "has_market_price")),
        ),
    },
}


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return None


def _nested_value(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _snapshot_contexts(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return immutable snapshot contexts before the ledger's derived row."""
    contexts: list[Mapping[str, Any]] = []
    for key in ("feature_snapshot", "pregame_snapshot", "snapshot", "immutable_record"):
        candidate = record.get(key)
        if isinstance(candidate, Mapping):
            contexts.append(candidate)
    price = record.get("price")
    if isinstance(price, Mapping):
        contexts.append(price)
    contexts.append(record)
    return contexts


def _value_from_contexts(record: Mapping[str, Any], *names: str) -> Any:
    for context in _snapshot_contexts(record):
        value = _mapping_value(context, *names)
        if value is not None:
            return value
    return None


def _path_available(record: Mapping[str, Any], path: str) -> bool:
    for context in _snapshot_contexts(record):
        value = _nested_value(context, path)
        if value not in (None, ""):
            return True
    return False


def _normalise_probability(value: Any) -> float | None:
    probability = _as_number(value)
    if probability is None or not 0.0 <= probability <= 1.0:
        return None
    return probability


def _model_key(record: Mapping[str, Any]) -> str:
    value = _value_from_contexts(record, "model_key", "modelKey")
    return str(value or "").strip().lower()


def _model_version(record: Mapping[str, Any]) -> str:
    value = _value_from_contexts(record, "model_version", "modelVersion", "version")
    return str(value or UNVERSIONED).strip() or UNVERSIONED


def _market(record: Mapping[str, Any]) -> str:
    value = _value_from_contexts(record, "market", "market_type", "marketType", "bet_type")
    return str(value or "unclassified").strip().lower() or "unclassified"


def _record_identifier(record: Mapping[str, Any], index: int) -> str:
    value = _value_from_contexts(record, "snapshot_id", "record_id", "id")
    return str(value or f"record-{index:06d}")


def _timestamp_value(record: Mapping[str, Any]) -> str | None:
    value = _value_from_contexts(
        record,
        "snapshot_at",
        "prediction_created_at",
        "pregame_generated_at",
        "generated_at",
        "generatedAt",
        "created_at",
        "published_at",
    )
    return str(value).strip() if value not in (None, "") else None


def _timestamp_key(value: str | None) -> tuple[int, datetime, str]:
    if not value:
        return (1, datetime.max.replace(tzinfo=timezone.utc), "")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return (1, datetime.max.replace(tzinfo=timezone.utc), value)
        return (0, parsed.astimezone(timezone.utc), value)
    except ValueError:
        return (1, datetime.max.replace(tzinfo=timezone.utc), value)


def _outcome(record: Mapping[str, Any]) -> int | None:
    value = _value_from_contexts(record, "outcome", "result")
    if value in (1, "1", True):
        return 1
    if value in (0, "0", False):
        return 0
    text = str(value or "").strip().lower()
    if text == "win":
        return 1
    if text == "loss":
        return 0
    return None


def _result_label(record: Mapping[str, Any]) -> str:
    value = _value_from_contexts(record, "result", "outcome")
    return str(value or "pending").strip().lower() or "pending"


def _model_probability(record: Mapping[str, Any]) -> tuple[float | None, str | None]:
    for field in ("raw_probability", "model_probability", "predicted_probability", "probability"):
        probability = _normalise_probability(_value_from_contexts(record, field))
        if probability is not None:
            return probability, field
    return None, None


def _certification_mapping(record: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("certification", "pregame_certification", "eligibility"):
        candidate = record.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _explicit_bool(record: Mapping[str, Any], certification: Mapping[str, Any], *names: str) -> bool | None:
    for source in (certification, record):
        for name in names:
            if name in source:
                value = source[name]
                if isinstance(value, bool):
                    return value
    return None


def certification_status(record: Mapping[str, Any], ledger: Mapping[str, Any] | None = None) -> tuple[bool, str]:
    """Require explicit pre-game, immutable, certified evidence for every row."""
    certification = _certification_mapping(record)
    if str(certification.get("status") or "").strip().lower() == "certified":
        return True, "certified"
    certified = _explicit_bool(record, certification, "certified", "is_certified", "certified_pregame")
    immutable = _explicit_bool(record, certification, "immutable", "is_immutable", "immutable_snapshot")
    pregame = _explicit_bool(record, certification, "pregame", "pre_game", "is_pregame", "pregame_verified")

    # A canonical ledger may guarantee all records are certified at the
    # payload level.  This is intentionally an explicit opt-in, never a
    # fallback for arbitrary JSON supplied via --ledger-path.
    ledger_certified = False
    if isinstance(ledger, Mapping):
        ledger_certified = bool(
            _mapping_value(
                ledger,
                "certified_immutable_pregame_only",
                "records_are_certified_immutable_pregame",
            )
        )
    if ledger_certified and certified is None and immutable is None and pregame is None:
        return True, "ledger_certified"
    if certified is not True:
        return False, "missing_certified_flag"
    if immutable is not True:
        return False, "missing_immutable_flag"
    if pregame is not True:
        return False, "missing_pregame_flag"
    return True, "certified"


def _financial_eligible(record: Mapping[str, Any]) -> bool:
    certification = _certification_mapping(record)
    value = _explicit_bool(record, certification, "financial_eligible", "real_price_eligible")
    return value is True


def _market_benchmark_eligible(record: Mapping[str, Any]) -> bool:
    certification = _certification_mapping(record)
    explicit = _explicit_bool(record, certification, "market_benchmark_eligible", "observed_market_eligible")
    return explicit is True or _financial_eligible(record)


def _price_provenance_is_disallowed(record: Mapping[str, Any]) -> bool:
    text_values: list[str] = []
    for context in _snapshot_contexts(record):
        for key in ("pricing_type", "odds_source", "line_source", "price_source", "price_provenance"):
            value = context.get(key)
            if value not in (None, ""):
                text_values.append(str(value).lower())
    blocked = ("assumed", "proxy", "synthetic", "model_generated", "default", "unpriced", "unknown")
    return any(any(marker in value for marker in blocked) for value in text_values)


def _american_odds(record: Mapping[str, Any]) -> float | None:
    """Return a verified observed American price, never an assumed/proxy one."""
    if not _financial_eligible(record) or _price_provenance_is_disallowed(record):
        return None
    for context in _snapshot_contexts(record):
        for key in ("observed_american_odds", "actual_american_odds", "american_odds", "real_american_odds"):
            odds = _as_number(context.get(key))
            if odds is not None and odds != 0 and abs(odds) >= 100:
                return odds
        odds = _as_number(context.get("odds"))
        odds_format = str(context.get("odds_format") or context.get("price_format") or "").strip().lower()
        pricing_type = str(context.get("pricing_type") or "").strip().lower()
        if (
            odds is not None
            and odds != 0
            and abs(odds) >= 100
            and (
                odds_format in {"american", "us"}
                or pricing_type in {"market", "sportsbook", "bookmaker", "observed", "executable"}
            )
        ):
            return odds
    return None


def _market_probability(record: Mapping[str, Any]) -> tuple[float | None, str | None]:
    if not _market_benchmark_eligible(record) or _price_provenance_is_disallowed(record):
        return None, None
    odds = _american_odds(record)
    if odds is not None:
        implied = 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)
        return implied, "observed_american_odds"
    for field in ("market_probability", "market_implied_probability", "market_pick_prob"):
        probability = _normalise_probability(_value_from_contexts(record, field))
        if probability is not None:
            return probability, field
    return None, None


def _stake(record: Mapping[str, Any]) -> float | None:
    value = _value_from_contexts(record, "stake_units", "units", "stake")
    stake = _as_number(value)
    return stake if stake is not None and stake > 0 else None


def _decision_is_actionable(record: Mapping[str, Any]) -> bool:
    decision = str(_value_from_contexts(record, "decision") or "").strip().upper()
    return decision in {"BET", "LEAN"}


def _binary_metrics(rows: Iterable[dict[str, Any]], probability_key: str) -> dict[str, Any]:
    items = list(rows)
    if not items:
        return {
            "settled_records": 0,
            "wins": 0,
            "losses": 0,
            "hit_rate": None,
            "mean_probability": None,
            "brier_score": None,
            "log_loss": None,
        }
    probabilities = [float(item[probability_key]) for item in items]
    outcomes = [int(item["outcome"]) for item in items]
    total = len(items)
    wins = sum(outcomes)
    epsilon = 1e-15
    brier = sum((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes)) / total
    log_loss = -sum(
        outcome * math.log(min(1.0 - epsilon, max(epsilon, probability)))
        + (1 - outcome) * math.log(1.0 - min(1.0 - epsilon, max(epsilon, probability)))
        for probability, outcome in zip(probabilities, outcomes)
    ) / total
    return {
        "settled_records": total,
        "wins": wins,
        "losses": total - wins,
        "hit_rate": _round(wins / total),
        "mean_probability": _round(sum(probabilities) / total),
        "brier_score": _round(brier),
        "log_loss": _round(log_loss),
    }


def _calibration_bins(rows: Iterable[dict[str, Any]], bins: int) -> dict[str, Any]:
    items = list(rows)
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(bins)]
    for item in items:
        probability = float(item["probability"])
        index = min(bins - 1, int(probability * bins))
        buckets[index].append(item)
    rendered: list[dict[str, Any]] = []
    calibration_error = 0.0
    for index, members in enumerate(buckets):
        lower = index / bins
        upper = (index + 1) / bins
        if not members:
            rendered.append({
                "lower": _round(lower),
                "upper": _round(upper),
                "records": 0,
                "mean_probability": None,
                "outcome_rate": None,
                "gap": None,
            })
            continue
        mean_probability = sum(float(item["probability"]) for item in members) / len(members)
        outcome_rate = sum(int(item["outcome"]) for item in members) / len(members)
        gap = outcome_rate - mean_probability
        calibration_error += len(members) / len(items) * abs(gap)
        rendered.append({
            "lower": _round(lower),
            "upper": _round(upper),
            "records": len(members),
            "mean_probability": _round(mean_probability),
            "outcome_rate": _round(outcome_rate),
            "gap": _round(gap),
        })
    return {"bins": rendered, "expected_calibration_error": _round(calibration_error) if items else None}


def _roi(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    excluded = Counter()
    for item in rows:
        record = item["record"]
        if not _decision_is_actionable(record):
            excluded["not_actionable"] += 1
            continue
        if not _financial_eligible(record):
            excluded["not_explicitly_financial_eligible"] += 1
            continue
        if _price_provenance_is_disallowed(record):
            excluded["assumed_or_proxy_price"] += 1
            continue
        odds = _american_odds(record)
        if odds is None:
            excluded["missing_verified_american_price"] += 1
            continue
        stake = _stake(record)
        if stake is None:
            excluded["missing_positive_stake"] += 1
            continue
        result = _result_label(record)
        if result not in {"win", "loss", "push"}:
            excluded["unsettled"] += 1
            continue
        profit = 0.0 if result == "push" else (stake * (odds / 100.0) if odds > 0 else stake * (100.0 / abs(odds)))
        if result == "loss":
            profit = -stake
        eligible.append({"stake": stake, "profit": profit, "result": result})
    stake_total = sum(item["stake"] for item in eligible)
    profit_total = sum(item["profit"] for item in eligible)
    return {
        "priced_settled_actionable_records": len(eligible),
        "stake_units": _round(stake_total),
        "profit_units": _round(profit_total),
        "roi": _round(profit_total / stake_total) if stake_total else None,
        "excluded": dict(sorted(excluded.items())),
        "note": "ROI uses only explicitly certified observed American prices and explicit positive stakes.",
    }


def _feature_audit(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[(item["model_key"], item["model_version"])].append(item)
    audits: list[dict[str, Any]] = []
    for (model_key, model_version), members in sorted(grouped.items()):
        contract = FEATURE_CONTRACTS.get(model_key)
        if not contract:
            continue
        raw_records = [item["record"] for item in members]
        groups: list[dict[str, Any]] = []
        for name, paths in contract["groups"]:
            available = sum(any(_path_available(record, path) for path in paths) for record in raw_records)
            groups.append({
                "name": name,
                "accepted_paths": list(paths),
                "available_records": available,
                "missing_records": len(raw_records) - available,
                "availability_rate": _round(available / len(raw_records)) if raw_records else None,
            })
        prediction_times = [item["prediction_at"] for item in members if item["prediction_at"]]
        audits.append({
            "model_key": model_key,
            "model_version": model_version,
            "contract_version": contract["contract_version"],
            "certified_records": len(raw_records),
            "records_with_prediction_timestamp": len(prediction_times),
            "records_missing_prediction_timestamp": len(raw_records) - len(prediction_times),
            "feature_groups": groups,
        })
    return audits


def _group_report(records: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    binary = [item for item in records if item["outcome"] is not None and item["probability"] is not None]
    market_binary: list[dict[str, Any]] = []
    for item in binary:
        market_probability, market_source = _market_probability(item["record"])
        if market_probability is not None:
            market_binary.append({**item, "market_probability": market_probability, "market_probability_source": market_source})
    chronology = sorted(records, key=lambda item: (_timestamp_key(item["prediction_at"]), item["record_id"]))
    times = [item["prediction_at"] for item in chronology if item["prediction_at"]]
    result_labels = Counter(_result_label(item["record"]) for item in records)
    return {
        "records": len(records),
        "settled_binary_records": len(binary),
        "result_counts": dict(sorted(result_labels.items())),
        "first_prediction_at": times[0] if times else None,
        "last_prediction_at": times[-1] if times else None,
        "records_missing_prediction_timestamp": len(records) - len(times),
        "model_metrics": _binary_metrics(binary, "probability"),
        "calibration": _calibration_bins(binary, bins),
        "market_benchmark": {
            **_binary_metrics(market_binary, "market_probability"),
            "priced_or_observed_records": len(market_binary),
            "probability_sources": dict(sorted(Counter(item["market_probability_source"] for item in market_binary).items())),
            "note": "Benchmark excludes assumed, proxy, synthetic, and unpriced market data.",
        },
        "real_price_roi": _roi(records),
    }


def _latest_certified_revisions(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep one final pregame revision per stable market slot for metrics."""

    selected: dict[str, dict[str, Any]] = {}
    passthrough: list[dict[str, Any]] = []
    for item in records:
        stable_id = str(item["record"].get("stable_id") or "").strip()
        if not stable_id:
            passthrough.append(item)
            continue
        current = selected.get(stable_id)
        candidate_key = (
            int(item["record"].get("revision") or 0),
            _timestamp_key(item["prediction_at"]),
            item["record_id"],
        )
        if current is None:
            selected[stable_id] = item
            continue
        current_key = (
            int(current["record"].get("revision") or 0),
            _timestamp_key(current["prediction_at"]),
            current["record_id"],
        )
        if candidate_key > current_key:
            selected[stable_id] = item
    final = passthrough + list(selected.values())
    return final, len(records) - len(final)


def evaluate_team_prop_ledger(ledger: Mapping[str, Any], *, bins: int = 10) -> dict[str, Any]:
    """Build a deterministic chronological report from a certified ledger payload."""
    if not isinstance(ledger, Mapping):
        raise ValueError("Team-prop pregame ledger must be a JSON object")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    raw_records = ledger.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("Team-prop pregame ledger must contain a records list")

    exclusion_counts = Counter()
    certified: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            exclusion_counts["non_object_record"] += 1
            continue
        model_key = _model_key(raw_record)
        if model_key not in SUPPORTED_MODEL_KEYS:
            exclusion_counts["out_of_scope_model"] += 1
            continue
        is_certified, reason = certification_status(raw_record, ledger)
        if not is_certified:
            exclusion_counts[f"uncertified:{reason}"] += 1
            continue
        probability, probability_field = _model_probability(raw_record)
        if probability is None:
            exclusion_counts["missing_or_invalid_model_probability"] += 1
            continue
        certified.append({
            "record": raw_record,
            "record_id": _record_identifier(raw_record, index),
            "model_key": model_key,
            "model_version": _model_version(raw_record),
            "market": _market(raw_record),
            "prediction_at": _timestamp_value(raw_record),
            "probability": probability,
            "probability_field": probability_field,
            "outcome": _outcome(raw_record),
        })

    certified_revisions = len(certified)
    certified, superseded_revisions = _latest_certified_revisions(certified)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in certified:
        groups[(item["model_key"], item["model_version"], item["market"])].append(item)
    segment_reports = [
        {
            "model_key": model_key,
            "model_version": model_version,
            "market": market,
            **_group_report(group_records, bins),
        }
        for (model_key, model_version, market), group_records in sorted(groups.items())
    ]
    all_records_report = _group_report(certified, bins)
    return {
        "schema_version": SCHEMA_VERSION,
        "ledger_schema_version": ledger.get("schema_version"),
        "scope": list(SUPPORTED_MODEL_KEYS),
        "calibration_bins": bins,
        "record_quality": {
            "input_records": len(raw_records),
            "certified_revision_records": certified_revisions,
            "certified_evaluable_records": len(certified),
            "superseded_revisions_excluded": superseded_revisions,
            "excluded_records": sum(exclusion_counts.values()),
            "exclusions": dict(sorted(exclusion_counts.items())),
        },
        "overall": all_records_report,
        "segments": segment_reports,
        "feature_contract_audit": _feature_audit(certified),
    }


def load_ledger(path: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load an explicit ledger JSON, or lazily ask the canonical ledger module."""
    if path is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read team-prop ledger {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Team-prop ledger JSON must be an object")
        return payload
    try:
        from scripts.team_prop_pregame_ledger import load_team_prop_pregame_ledger
    except ImportError as exc:  # pragma: no cover - only useful before deployment wiring exists.
        raise ValueError(
            "Canonical team-prop pregame ledger is unavailable; pass --ledger-path to an immutable certified ledger JSON."
        ) from exc
    payload = load_team_prop_pregame_ledger(repo_root=repo_root) if repo_root else load_team_prop_pregame_ledger()
    if not isinstance(payload, dict):
        raise ValueError("Canonical team-prop pregame ledger returned an invalid payload")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-path", type=Path, help="Explicit immutable certified ledger JSON.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; stdout when omitted.")
    parser.add_argument("--bins", type=int, default=10, help="Equal-width calibration bins (default: 10).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        ledger = load_ledger(args.ledger_path)
        report = evaluate_team_prop_ledger(ledger, bins=args.bins)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
