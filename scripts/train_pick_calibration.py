#!/usr/bin/env python3
"""Train and safely promote the pick probability calibration layer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_DIR = REPO_ROOT / "data" / "calibration"
sys.path.insert(0, str(REPO_ROOT))

from scripts.pick_calibration import (  # noqa: E402
    CALIBRATION_SCHEMA_VERSION,
    MIN_GROUP_SAMPLES,
    calibrated_probability,
    read_json,
    write_json_if_changed,
)


TRIGGER_INTERVAL = 100
MIN_TRAINABLE_ROWS = 40
TRAINING_CONTRACT_VERSION = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Evaluate even if fewer than 100 new decisions exist.")
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION_DIR)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _identity(samples: int = 0) -> dict[str, Any]:
    return {"intercept": 0.0, "slope": 1.0, "samples": samples}


def _logit(probability: float) -> float:
    value = min(0.999, max(0.001, probability))
    return math.log(value / (1 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1 + exp_value)


def fit_platt(
    rows: list[dict[str, Any]],
    *,
    prior: dict[str, Any] | None = None,
    prior_strength: float = 80.0,
) -> dict[str, Any]:
    prior = prior or _identity()
    prior_intercept = float(prior.get("intercept") or 0)
    prior_slope = float(prior.get("slope") or 1)
    intercept = prior_intercept
    slope = prior_slope
    count = len(rows)
    if not count:
        return _identity()

    for iteration in range(3000):
        intercept_gradient = 0.0
        slope_gradient = 0.0
        for row in rows:
            feature = _logit(float(row["raw_probability"]))
            prediction = _sigmoid(intercept + slope * feature)
            error = prediction - int(row["outcome"])
            intercept_gradient += error
            slope_gradient += error * feature
        intercept_gradient = (intercept_gradient + prior_strength * (intercept - prior_intercept)) / count
        slope_gradient = (slope_gradient + prior_strength * (slope - prior_slope)) / count
        learning_rate = 0.04 / (1 + iteration / 1800)
        intercept -= learning_rate * intercept_gradient
        slope -= learning_rate * slope_gradient
        intercept = max(-3.0, min(3.0, intercept))
        slope = max(0.1, min(3.0, slope))

    return {
        "intercept": round(intercept, 8),
        "slope": round(slope, 8),
        "samples": count,
        "shrinkage_prior_samples": round(prior_strength, 2),
    }


def fit_mapping(rows: list[dict[str, Any]]) -> dict[str, Any]:
    global_parameters = fit_platt(rows, prior=_identity(), prior_strength=120.0)
    groups: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["calibration_group"]), []).append(row)
    for key, group_rows in sorted(grouped.items()):
        if len(group_rows) < MIN_GROUP_SAMPLES:
            continue
        groups[key] = fit_platt(group_rows, prior=global_parameters, prior_strength=80.0)
    return {"global": global_parameters, "groups": groups}


def probability_for(row: dict[str, Any], mapping: dict[str, Any]) -> float:
    groups = mapping.get("groups") if isinstance(mapping.get("groups"), dict) else {}
    parameters = groups.get(row.get("calibration_group"))
    if not isinstance(parameters, dict) or int(parameters.get("samples") or 0) < MIN_GROUP_SAMPLES:
        parameters = mapping.get("global") if isinstance(mapping.get("global"), dict) else _identity()
    return calibrated_probability(float(row["raw_probability"]), parameters)


def _ece(probabilities: list[float], outcomes: list[int], bins: int = 10) -> float:
    total = len(probabilities)
    if not total:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bins - 1 and probability == upper)
        ]
        if not members:
            continue
        mean_probability = sum(probabilities[item] for item in members) / len(members)
        mean_outcome = sum(outcomes[item] for item in members) / len(members)
        error += len(members) / total * abs(mean_probability - mean_outcome)
    return error


def evaluate(rows: list[dict[str, Any]], mapping: dict[str, Any]) -> dict[str, Any]:
    probabilities = [probability_for(row, mapping) for row in rows]
    outcomes = [int(row["outcome"]) for row in rows]
    brier = sum((probability - outcome) ** 2 for probability, outcome in zip(probabilities, outcomes)) / len(rows)
    log_loss = -sum(
        outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability)
        for probability, outcome in zip(probabilities, outcomes)
    ) / len(rows)

    qualified = [
        (row, probability)
        for row, probability in zip(rows, probabilities)
        if row.get("market_implied_probability") is not None
        and probability - float(row["market_implied_probability"]) >= 0.03
    ]
    stake = sum(
        float(row.get("stake_units") or row.get("units") or row.get("raw_units") or 1)
        for row, _ in qualified
    )
    profit = sum(float(row.get("profit") or 0) for row, _ in qualified)
    return {
        "samples": len(rows),
        "brier_score": round(brier, 8),
        "log_loss": round(log_loss, 8),
        "calibration_error": round(_ece(probabilities, outcomes), 8),
        "qualified_bets": len(qualified),
        "roi": round(profit / stake, 8) if stake and len(qualified) >= 10 else None,
    }


def should_promote(champion: dict[str, Any], challenger: dict[str, Any]) -> tuple[bool, list[str]]:
    brier_improved = challenger["brier_score"] <= champion["brier_score"] - 0.00025
    calibration_improved = challenger["calibration_error"] <= champion["calibration_error"] - 0.005
    log_loss_safe = challenger["log_loss"] <= champion["log_loss"] + 0.0025
    champion_roi = champion.get("roi")
    challenger_roi = challenger.get("roi")
    roi_safe = champion_roi is None or challenger_roi is None or challenger_roi >= champion_roi - 0.02
    reasons = [
        f"brier_improved={brier_improved}",
        f"calibration_improved={calibration_improved}",
        f"log_loss_safe={log_loss_safe}",
        f"roi_safe={roi_safe}",
    ]
    return (brier_improved or calibration_improved) and log_loss_safe and roi_safe, reasons


def _trainable_records(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    records = ledger.get("records") if isinstance(ledger.get("records"), list) else []
    return sorted(
        [
            row
            for row in records
            if isinstance(row, dict)
            and row.get("outcome") in {0, 1}
            and isinstance(row.get("raw_probability"), (int, float))
            and row.get("calibration_eligible") is True
            and row.get("ml_calibration_excluded") is not True
            and str(row.get("probability_source") or "") != "player_props_ml_v1"
        ],
        key=lambda row: (str(row.get("date") or ""), str(row.get("id") or "")),
    )


def _active_mapping(active: dict[str, Any] | None, samples: int) -> dict[str, Any]:
    if active and isinstance(active.get("global"), dict):
        return {
            "global": active["global"],
            "groups": active.get("groups") if isinstance(active.get("groups"), dict) else {},
        }
    return {"global": _identity(samples), "groups": {}}


def run_training(calibration_dir: Path, *, force: bool = False) -> dict[str, Any]:
    ledger = read_json(calibration_dir / "outcome_ledger.json")
    if not ledger:
        raise SystemExit("Missing calibration outcome ledger; run rebuild_pick_outcome_ledger.py first")
    state = read_json(calibration_dir / "state.json") or {}
    active = read_json(calibration_dir / "active.json")
    decided_count = int((ledger.get("summary") or {}).get("decided_picks") or 0)
    last_count = int(state.get("last_evaluated_decided_count") or 0)
    new_decisions = max(0, decided_count - last_count)
    contract_changed = bool(state or active) and (
        int(state.get("training_contract_version") or 0) != TRAINING_CONTRACT_VERSION
        or bool(active)
        and int((active or {}).get("training_contract_version") or 0) != TRAINING_CONTRACT_VERSION
    )
    if not force and not contract_changed and new_decisions < TRIGGER_INTERVAL:
        return {
            "evaluated": False,
            "promoted": False,
            "decided_count": decided_count,
            "new_decisions": new_decisions,
            "required_new_decisions": TRIGGER_INTERVAL,
        }

    rows = _trainable_records(ledger)
    if len(rows) < MIN_TRAINABLE_ROWS:
        raise SystemExit(f"Need at least {MIN_TRAINABLE_ROWS} trainable decided picks; found {len(rows)}")
    last_trainable_count = 0 if contract_changed else int(state.get("last_trainable_decided_count") or 0)
    new_trainable_count = max(0, len(rows) - last_trainable_count)
    if active and last_trainable_count and not force and not contract_changed:
        if new_trainable_count < 20:
            return {
                "evaluated": False,
                "promoted": False,
                "decided_count": decided_count,
                "new_decisions": new_decisions,
                "new_trainable_decisions": new_trainable_count,
                "required_new_trainable_decisions": 20,
            }
        holdout_size = new_trainable_count
    else:
        holdout_size = max(20, int(len(rows) * 0.2))
        holdout_size = min(holdout_size, len(rows) - 20)
    training_rows = rows[:-holdout_size]
    holdout_rows = rows[-holdout_size:]

    champion_mapping = _active_mapping(None if contract_changed else active, len(training_rows))
    challenger_mapping = fit_mapping(training_rows)
    champion_metrics = evaluate(holdout_rows, champion_mapping)
    challenger_metrics = evaluate(holdout_rows, challenger_mapping)
    promote, reasons = should_promote(champion_metrics, challenger_metrics)
    now = _utc_now()
    final_mapping = fit_mapping(rows)
    version = f"cal-v{TRAINING_CONTRACT_VERSION}-{decided_count}-{now[:10]}"
    challenger_payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "version": version,
        "evaluated_at": now,
        "decided_count": decided_count,
        "trainable_decided_count": len(rows),
        "holdout": {"champion": champion_metrics, "challenger": challenger_metrics},
        "promotion": {"approved": promote, "reasons": reasons},
        "minimum_group_samples": MIN_GROUP_SAMPLES,
        **final_mapping,
    }
    write_json_if_changed(calibration_dir / "challenger.json", challenger_payload)

    active_changed = False
    if promote:
        active_payload = {
            **challenger_payload,
            "promoted_at": now,
            "promotion": {"approved": True, "reasons": reasons},
        }
        active_changed = write_json_if_changed(calibration_dir / "active.json", active_payload)
    elif not active or contract_changed:
        active_payload = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "training_contract_version": TRAINING_CONTRACT_VERSION,
            "version": f"identity-v{TRAINING_CONTRACT_VERSION}",
            "promoted_at": now,
            "decided_count": decided_count,
            "trainable_decided_count": len(rows),
            "minimum_group_samples": MIN_GROUP_SAMPLES,
            "global": _identity(len(rows)),
            "groups": {},
            "holdout": {"champion": champion_metrics, "challenger": challenger_metrics},
            "promotion": {"approved": False, "reasons": reasons},
        }
        active_changed = write_json_if_changed(calibration_dir / "active.json", active_payload)

    state_payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "training_contract_version": TRAINING_CONTRACT_VERSION,
        "last_evaluated_at": now,
        "last_evaluated_decided_count": decided_count,
        "last_trainable_decided_count": len(rows),
        "trigger_interval": TRIGGER_INTERVAL,
        "last_challenger_version": version,
        "last_challenger_promoted": promote,
    }
    write_json_if_changed(calibration_dir / "state.json", state_payload)
    return {
        "evaluated": True,
        "promoted": promote,
        "active_changed": active_changed,
        "decided_count": decided_count,
        "trainable_decided_count": len(rows),
        "new_decisions": new_decisions,
        "champion": champion_metrics,
        "challenger": challenger_metrics,
        "reasons": reasons,
        "training_contract_changed": contract_changed,
    }


def main() -> int:
    args = _parse_args()
    summary = run_training(args.calibration_dir.resolve(), force=args.force)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
