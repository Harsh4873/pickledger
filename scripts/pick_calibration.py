#!/usr/bin/env python3
"""Shared pick snapshots, outcome-ledger helpers, and probability calibration."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator

from player_props.era import is_ml_era_pick
from scripts.team_prop_pregame_ledger import (
    TEAM_PROP_MODEL_KEYS,
    load_team_prop_pregame_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = REPO_ROOT / "data" / "calibration"
ACTIVE_CALIBRATION_PATH = CALIBRATION_DIR / "active.json"
LEDGER_PATH = CALIBRATION_DIR / "outcome_ledger.json"
CALIBRATION_SCHEMA_VERSION = 1
MIN_GROUP_SAMPLES = 30
CALIBRATION_EXCLUDED_MODEL_KEYS = {"fifa_world_cup", "mls"}
# Research models with no real market (settlement at an assumed price only)
# keep their calibrated probabilities for display, but the model's own
# decision and stake publish untouched — there is no executable price for
# the edge-based downgrade to protect.
DECISION_DOWNGRADE_EXEMPT_MODEL_KEYS = {"mlb_inning"}
ML_OWNED_PROBABILITY_SOURCE = "player_props_ml_v1"

SNAPSHOT_EXCLUDED_FIELDS = {
    "result",
    "pregame_snapshot",
    "raw_probability",
    "raw_edge",
    "raw_units",
    "calibrated_probability",
    "calibration",
}


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_if_changed(path: Path, payload: dict[str, Any]) -> bool:
    rendered = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    try:
        if path.read_text(encoding="utf-8") == rendered:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return True


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
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
    if 1 < number <= 100:
        number /= 100
    if not 0 <= number <= 1:
        return None
    return min(0.999, max(0.001, number))


def pick_probability(pick: dict[str, Any], *, raw: bool = False) -> float | None:
    fields = (
        ("raw_probability", "model_probability", "predicted_probability", "probability", "prob", "confidence")
        if raw
        else ("probability", "calibrated_probability", "raw_probability", "model_probability", "predicted_probability", "prob")
    )
    for field in fields:
        probability = normalize_probability(pick.get(field))
        if probability is not None:
            return probability
    return None


def american_implied_probability(value: Any) -> float | None:
    odds = _number(value)
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def market_probability(pick: dict[str, Any]) -> float | None:
    for field in ("market_pick_prob", "market_probability", "market_implied_probability"):
        probability = normalize_probability(pick.get(field))
        if probability is not None:
            return probability
    return american_implied_probability(pick.get("odds")) or american_implied_probability(pick.get("assumed_odds"))


def infer_bet_type(pick: dict[str, Any]) -> str:
    for field in ("stat_key", "market_type", "market", "bet_type", "prop_type"):
        value = str(pick.get(field) or "").strip().lower()
        if value:
            return value.replace(" ", "_")
    text = " ".join(str(pick.get(field) or "") for field in ("pick", "selection")).lower()
    if "strikeout" in text:
        return "strikeouts"
    if "first five" in text or "f5" in text:
        return "first_five"
    if "moneyline" in text or " ml " in f" {text} " or text.endswith(" ml"):
        return "moneyline"
    if "over" in text or "under" in text or "total" in text:
        return "total"
    if "spread" in text or "run line" in text:
        return "spread"
    return "other"


def calibration_group_key(model_key: str, sport: str, bet_type: str, source: str = "") -> str:
    return "|".join(
        (
            f"model:{str(model_key or 'unknown').strip().lower()}",
            f"source:{str(source or model_key or 'unknown').strip().lower()}",
            f"sport:{str(sport or 'unknown').strip().lower()}",
            f"bet:{str(bet_type or 'other').strip().lower()}",
        )
    )


def make_pregame_snapshot(pick: dict[str, Any]) -> dict[str, Any]:
    existing = pick.get("pregame_snapshot")
    if isinstance(existing, dict):
        return copy.deepcopy(existing)
    return {
        key: copy.deepcopy(value)
        for key, value in pick.items()
        if key not in SNAPSHOT_EXCLUDED_FIELDS
    }


def _logit(probability: float) -> float:
    value = min(0.999, max(0.001, probability))
    return math.log(value / (1 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def calibrated_probability(probability: float, parameters: dict[str, Any]) -> float:
    intercept = _number(parameters.get("intercept"))
    slope = _number(parameters.get("slope"))
    intercept = intercept if intercept is not None else 0.0
    slope = slope if slope is not None else 1.0
    return min(0.99, max(0.01, _sigmoid(intercept + slope * _logit(probability))))


def load_active_calibration(path: Path = ACTIVE_CALIBRATION_PATH) -> dict[str, Any] | None:
    payload = read_json(path)
    if not payload or not isinstance(payload.get("global"), dict):
        return None
    return payload


def _calibration_parameters(
    active: dict[str, Any],
    model_key: str,
    source: str,
    sport: str,
    bet_type: str,
) -> tuple[str, dict[str, Any]]:
    group_key = calibration_group_key(model_key, sport, bet_type, source)
    groups = active.get("groups") if isinstance(active.get("groups"), dict) else {}
    group = groups.get(group_key)
    if isinstance(group, dict) and int(group.get("samples") or 0) >= int(active.get("minimum_group_samples") or MIN_GROUP_SAMPLES):
        return group_key, group
    return "global", active.get("global") or {}


def apply_calibration_to_pick(
    pick: dict[str, Any],
    model_key: str,
    active: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pick["pregame_snapshot"] = make_pregame_snapshot(pick)
    if not active:
        return pick

    snapshot = pick["pregame_snapshot"]
    raw_probability = pick_probability(snapshot, raw=True)
    if raw_probability is None:
        return pick

    sport = str(snapshot.get("sport") or pick.get("sport") or "unknown")
    source = str(snapshot.get("source") or pick.get("source") or model_key)
    bet_type = infer_bet_type(snapshot)
    key, parameters = _calibration_parameters(active, model_key, source, sport, bet_type)
    adjusted = calibrated_probability(raw_probability, parameters)
    probability_delta = adjusted - raw_probability

    raw_edge = _number(snapshot.get("edge"))
    raw_units = _number(snapshot.get("units"))
    implied = market_probability(snapshot)
    if implied is not None:
        adjusted_edge = (adjusted - implied) * 100
    elif raw_edge is not None:
        adjusted_edge = raw_edge + probability_delta * 100
    else:
        adjusted_edge = None

    pick["raw_probability"] = round(raw_probability, 6)
    pick["calibrated_probability"] = round(adjusted, 6)
    pick["probability"] = round(adjusted, 4)
    if raw_edge is not None:
        pick["raw_edge"] = round(raw_edge, 4)
    if adjusted_edge is not None:
        pick["edge"] = round(adjusted_edge, 2)
    downgrade_exempt = model_key in DECISION_DOWNGRADE_EXEMPT_MODEL_KEYS
    if raw_units is not None:
        pick["raw_units"] = round(raw_units, 4)
        if downgrade_exempt or adjusted_edge is None:
            pick["units"] = round(raw_units, 4)
        elif adjusted_edge <= 0:
            pick["units"] = 0
        else:
            baseline_edge = max(0.01, raw_edge or adjusted_edge or 0.01)
            ratio = max(0.5, min(1.25, max(0.0, adjusted_edge or 0.0) / baseline_edge))
            pick["units"] = round(raw_units * ratio, 2)

    raw_decision = str(snapshot.get("decision") or pick.get("decision") or "").strip().upper()
    if not downgrade_exempt and raw_decision in {"BET", "LEAN"} and adjusted_edge is not None:
        if adjusted_edge < 3:
            pick["decision"] = "PASS"
            pick["units"] = 0
        elif raw_decision == "BET" and adjusted_edge < 7:
            pick["decision"] = "LEAN"

    pick["calibration"] = {
        "applied": True,
        "version": str(active.get("version") or "unknown"),
        "key": key,
        "samples": int(parameters.get("samples") or 0),
        "probability_delta": round(probability_delta, 6),
    }
    return pick


def apply_calibration_to_payload(
    payload: dict[str, Any],
    active: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active = active if active is not None else load_active_calibration()
    models = payload.get("models")
    if isinstance(models, dict):
        for model_key, bucket in models.items():
            if str(model_key) in CALIBRATION_EXCLUDED_MODEL_KEYS:
                continue
            if not isinstance(bucket, dict) or not isinstance(bucket.get("picks"), list):
                continue
            for pick in bucket["picks"]:
                if (
                    isinstance(pick, dict)
                    and not pick.get("calibration_excluded")
                    and str(pick.get("probability_source") or "") != ML_OWNED_PROBABILITY_SOURCE
                    and not pick.get("ml_calibration_excluded")
                ):
                    apply_calibration_to_pick(pick, str(model_key), active)
    elif isinstance(payload.get("picks"), list):
        if str(payload.get("model_key") or "") in CALIBRATION_EXCLUDED_MODEL_KEYS:
            return payload
        for pick in payload["picks"]:
            if (
                isinstance(pick, dict)
                and not pick.get("calibration_excluded")
                and str(pick.get("probability_source") or "") != ML_OWNED_PROBABILITY_SOURCE
                and not pick.get("ml_calibration_excluded")
            ):
                apply_calibration_to_pick(pick, str(payload.get("model_key") or "unknown"), active)
    return payload


def _record_id(cache_type: str, date_iso: str, model_key: str, pick: dict[str, Any]) -> str:
    raw = json.dumps(
        [
            cache_type,
            date_iso,
            model_key,
            pick.get("id"),
            pick.get("source"),
            pick.get("sport"),
            pick.get("pick"),
            pick.get("matchup") or pick.get("game"),
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _profit(
    result: str,
    units: float | None,
    odds: float | None,
    implied_probability: float | None = None,
) -> float | None:
    if result == "push":
        return 0.0
    if result not in {"win", "loss"} or units is None:
        return None
    if result == "loss":
        return -units
    if odds is None or odds == 0:
        if implied_probability is not None:
            return units * (1 - implied_probability) / implied_probability
        return units
    return units * (odds / 100 if odds > 0 else 100 / abs(odds))


def ledger_record(
    pick: dict[str, Any],
    *,
    cache_type: str,
    date_iso: str,
    model_key: str,
) -> dict[str, Any]:
    snapshot = make_pregame_snapshot(pick)
    result = str(pick.get("result") or "pending").strip().lower()
    raw_probability = pick_probability(snapshot, raw=True)
    displayed_probability = pick_probability(pick)
    odds = _number(snapshot.get("odds"))
    odds = odds if odds is not None else _number(snapshot.get("assumed_odds"))
    raw_units = _number(snapshot.get("units"))
    units = _number(pick.get("units"))
    stake_units = units if units is not None else raw_units
    sport = str(snapshot.get("sport") or pick.get("sport") or "unknown")
    source = str(snapshot.get("source") or pick.get("source") or model_key)
    bet_type = infer_bet_type(snapshot)
    implied_probability = market_probability(snapshot)
    outcome = 1 if result == "win" else 0 if result == "loss" else None
    probability_source = str(
        snapshot.get("probability_source")
        or pick.get("probability_source")
        or ""
    ).strip()
    ml_calibration_excluded = bool(
        snapshot.get("ml_calibration_excluded") is True
        or pick.get("ml_calibration_excluded") is True
        or probability_source == ML_OWNED_PROBABILITY_SOURCE
    )
    return {
        "id": _record_id(cache_type, date_iso, model_key, pick),
        "date": str(snapshot.get("date") or pick.get("date") or date_iso),
        "cache_type": cache_type,
        "model_key": model_key,
        "source": source,
        "sport": sport,
        "bet_type": bet_type,
        "calibration_group": calibration_group_key(model_key, sport, bet_type, source),
        "probability_source": probability_source,
        "ml_calibration_excluded": ml_calibration_excluded,
        "calibration_eligible": not ml_calibration_excluded,
        "calibration_exclusion_reason": "ml_owned_probability" if ml_calibration_excluded else None,
        "pick": str(snapshot.get("pick") or pick.get("pick") or ""),
        "matchup": str(snapshot.get("matchup") or snapshot.get("game") or pick.get("matchup") or pick.get("game") or ""),
        "raw_probability": round(raw_probability, 6) if raw_probability is not None else None,
        "probability": round(displayed_probability, 6) if displayed_probability is not None else None,
        "market_implied_probability": implied_probability,
        "raw_edge": _number(snapshot.get("edge")),
        "edge": _number(pick.get("edge")),
        "odds": odds,
        "raw_units": raw_units,
        "units": stake_units,
        "stake_units": stake_units,
        "raw_decision": str(snapshot.get("decision") or ""),
        "decision": str(pick.get("decision") or snapshot.get("decision") or ""),
        "result": result,
        "outcome": outcome,
        "profit": _profit(result, stake_units, odds, implied_probability),
        "pregame_snapshot": snapshot,
    }


def _current_decision(pick: dict[str, Any]) -> str:
    """Return the latest decision for a pick, including recalibration downgrades.

    Recalibration may retroactively flip a graded BET/LEAN to PASS.  Team-model
    ledger rows filter on this value so never-wagered PASS rows cannot train
    calibration; player-prop PASS abstentions intentionally stay recorded.
    """

    snapshot = pick.get("pregame_snapshot") if isinstance(pick.get("pregame_snapshot"), dict) else {}
    return str(pick.get("decision") or snapshot.get("decision") or "").strip().upper()


def _certified_team_record(record: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one certified, price-verified team snapshot for calibration.

    The complete team ledger remains the evaluation source.  The universal
    calibration ledger receives only rows that passed its stricter provenance
    contract, so legacy/assumed/proxy prices cannot promote a calibration.
    """

    if record.get("calibration_eligible") is not True:
        return None
    certification = record.get("certification")
    if not isinstance(certification, dict) or certification.get("status") != "certified":
        return None
    model_key = str(record.get("model_key") or "").strip()
    if model_key not in TEAM_PROP_MODEL_KEYS or model_key in CALIBRATION_EXCLUDED_MODEL_KEYS:
        return None
    if _current_decision(record) == "PASS":
        return None
    raw_probability = normalize_probability(record.get("raw_probability"))
    if raw_probability is None:
        return None
    price = record.get("price") if isinstance(record.get("price"), dict) else {}
    odds = _number(record.get("observed_american_odds"))
    odds = odds if odds is not None else _number(price.get("odds"))
    implied_probability = american_implied_probability(odds)
    if odds is None or implied_probability is None:
        return None
    result = str(record.get("result") or "pending").strip().lower()
    outcome = 1 if result == "win" else 0 if result == "loss" else None
    stake_units = _number(record.get("stake"))
    snapshot = record.get("pregame_snapshot") if isinstance(record.get("pregame_snapshot"), dict) else {}
    sport = str(record.get("sport") or snapshot.get("sport") or "unknown")
    source = str(record.get("source") or snapshot.get("source") or model_key)
    bet_type = str(record.get("market") or infer_bet_type(snapshot) or "other")
    displayed_probability = normalize_probability(record.get("displayed_probability"))
    return {
        "id": str(record.get("id") or ""),
        "date": str(record.get("slate_date") or snapshot.get("date") or ""),
        "cache_type": "team_prop_pregame_ledger",
        "model_key": model_key,
        "model_version": str(record.get("model_version") or model_key),
        "source": source,
        "sport": sport,
        "bet_type": bet_type,
        "calibration_group": calibration_group_key(model_key, sport, bet_type, source),
        "pick": str(record.get("pick") or snapshot.get("pick") or ""),
        "matchup": str(record.get("matchup") or snapshot.get("matchup") or snapshot.get("game") or ""),
        "raw_probability": round(raw_probability, 6),
        "probability": round(displayed_probability, 6) if displayed_probability is not None else None,
        "market_implied_probability": implied_probability,
        "raw_edge": _number(snapshot.get("edge")),
        "edge": _number(snapshot.get("edge")),
        "odds": odds,
        "raw_units": stake_units,
        "units": stake_units,
        "stake_units": stake_units,
        "raw_decision": str(record.get("raw_decision") or snapshot.get("decision") or ""),
        "decision": str(record.get("decision") or snapshot.get("decision") or ""),
        "result": result,
        "outcome": outcome,
        "profit": _profit(result, stake_units, odds, implied_probability),
        "calibration_eligible": True,
        "pregame_snapshot": copy.deepcopy(snapshot),
    }


def _iter_bucket_records(
    payload: dict[str, Any],
    *,
    cache_type: str,
    fallback_date: str,
) -> Iterator[dict[str, Any]]:
    models = payload.get("models")
    fallback_timestamp = payload.get("generatedAt") or payload.get("updatedAt")
    if isinstance(models, dict):
        for model_key, bucket in models.items():
            if str(model_key) in CALIBRATION_EXCLUDED_MODEL_KEYS:
                continue
            if cache_type == "model_cache" and str(model_key) in TEAM_PROP_MODEL_KEYS:
                # These buckets are represented only by immutable certified
                # snapshots; mutable legacy cache rows are not training data.
                continue
            if not isinstance(bucket, dict) or not isinstance(bucket.get("picks"), list):
                continue
            for pick in bucket["picks"]:
                if (
                    isinstance(pick, dict)
                    and not pick.get("calibration_excluded")
                    and (cache_type != "player_props_cache" or is_ml_era_pick(pick, fallback_timestamp))
                    and (cache_type != "model_cache" or _current_decision(pick) != "PASS")
                ):
                    yield ledger_record(
                        pick,
                        cache_type=cache_type,
                        date_iso=fallback_date,
                        model_key=str(model_key),
                    )
        return
    for pick in payload.get("picks") or []:
        if (
            isinstance(pick, dict)
            and not pick.get("calibration_excluded")
            and (cache_type != "player_props_cache" or is_ml_era_pick(pick, fallback_timestamp))
            and (cache_type != "model_cache" or _current_decision(pick) != "PASS")
        ):
            yield ledger_record(
                pick,
                cache_type=cache_type,
                date_iso=fallback_date,
                model_key=str(payload.get("model_key") or cache_type),
            )


def build_outcome_ledger(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    records_by_id: dict[str, dict[str, Any]] = {}

    def add_payload(path: Path, *, cache_type: str, fallback_stem: str | None = None) -> None:
        payload = read_json(path)
        if not payload:
            return
        fallback_date = str(payload.get("date") or payload.get("slate_date") or fallback_stem or path.stem)
        for record in _iter_bucket_records(payload, cache_type=cache_type, fallback_date=fallback_date):
            records_by_id[record["id"]] = record

    for path in sorted((repo_root / "data" / "model_cache").glob("20??-??-??.json")):
        add_payload(path, cache_type="model_cache")
    for path in sorted((repo_root / "data" / "player_props_snapshots").glob("20??-??-??/*.json")):
        add_payload(path, cache_type="player_props_cache", fallback_stem=path.parent.name)
    for path in sorted((repo_root / "data" / "player_props_cache").glob("20??-??-??.json")):
        add_payload(path, cache_type="player_props_cache")

    team_ledger = load_team_prop_pregame_ledger(repo_root=repo_root)
    for snapshot_record in team_ledger.get("records") or []:
        if not isinstance(snapshot_record, dict):
            continue
        record = _certified_team_record(snapshot_record)
        if record and record.get("id"):
            records_by_id[str(record["id"])] = record

    records = sorted(
        records_by_id.values(),
        key=lambda record: (str(record.get("date") or ""), str(record.get("model_key") or ""), str(record.get("id") or "")),
    )
    decided = [record for record in records if record.get("result") in {"win", "loss"}]
    trainable = [
        record
        for record in decided
        if record.get("raw_probability") is not None
        and record.get("calibration_eligible") is True
    ]
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "summary": {
            "total_picks": len(records),
            "decided_picks": len(decided),
            "trainable_decided_picks": len(trainable),
            "pending_picks": sum(record.get("result") in {"", "pending"} for record in records),
        },
        "records": records,
    }


def rebuild_outcome_ledger(
    repo_root: Path = REPO_ROOT,
    output_path: Path | None = None,
) -> tuple[dict[str, Any], bool]:
    ledger = build_outcome_ledger(repo_root)
    path = output_path or repo_root / "data" / "calibration" / "outcome_ledger.json"
    changed = write_json_if_changed(path, ledger)

    state_path = repo_root / "data" / "calibration" / "state.json"
    state = read_json(state_path)
    if state:
        decided = int(ledger["summary"]["decided_picks"])
        trainable = int(ledger["summary"]["trainable_decided_picks"])
        state_changed = False
        if int(state.get("last_evaluated_decided_count") or 0) > decided:
            state["last_evaluated_decided_count"] = decided
            state_changed = True
        if int(state.get("last_trainable_decided_count") or 0) > trainable:
            state["last_trainable_decided_count"] = trainable
            state_changed = True
        if state_changed:
            write_json_if_changed(state_path, state)
    return ledger, changed
