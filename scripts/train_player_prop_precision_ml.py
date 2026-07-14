#!/usr/bin/env python3
"""Train and chronologically validate the season player-prop precision model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from player_props.precision import (  # noqa: E402
    ARTIFACT_DIR,
    CATEGORICAL_FEATURES,
    DEFAULT_POLICY,
    METADATA_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PRECISION_MODEL_VERSION,
    build_training_features,
)
from player_props.schema import safe_float  # noqa: E402


DEFAULT_HISTORY = REPO_ROOT / "data" / "player_props_training" / "market_history_2026.jsonl"
TARGET_ACCURACY = 0.70
MIN_VALIDATION_PICKS = 100
MIN_HOLDOUT_PICKS = 30


def _read_history(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("over_outcome") in {0, 1}:
            rows.append(row)
    return rows


def _pipeline() -> Any:
    from sklearn.compose import ColumnTransformer  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore
    from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(C=0.10, max_iter=1000, random_state=42)),
    ])


def _frame(rows: list[dict[str, Any]]) -> Any:
    import pandas as pd  # type: ignore

    return pd.DataFrame(rows)


def _fit(rows: Any) -> Any:
    model = _pipeline()
    model.fit(rows[NUMERIC_FEATURES + CATEGORICAL_FEATURES], rows["over_outcome"].astype(int))
    return model


def _candidate_rows(model: Any, frame: Any, policy: dict[str, Any]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    probabilities = model.predict_proba(frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(frame.to_dict("records")):
        if str(row.get("sport") or "").upper() != str(policy["sport"]).upper():
            continue
        if str(row.get("stat_key") or "") != str(policy["stat_key"]):
            continue
        under_odds = row.get("under_odds")
        under_implied = row.get("under_implied")
        if under_odds is None or under_implied is None:
            continue
        under_odds = int(under_odds)
        under_implied = safe_float(under_implied)
        over_implied = safe_float(row.get("over_implied"))
        under_probability = 1.0 - float(probabilities[index])
        under_history = 1.0 - safe_float(row.get("over_rate"))
        under_last5 = 1.0 - safe_float(row.get("over_rate5"))
        if int(row.get("history_count") or 0) < int(policy["minimum_history"]):
            continue
        if under_history < float(policy["minimum_under_history_rate"]):
            continue
        if under_last5 < float(policy["minimum_under_last5_rate"]):
            continue
        if not int(policy["minimum_under_odds"]) <= under_odds <= int(policy["maximum_under_odds"]):
            continue
        if policy.get("require_market_favorite") and under_implied < over_implied:
            continue
        if policy.get("require_mean10_below_line") and safe_float(row.get("mean10")) >= safe_float(row.get("line")):
            continue
        if policy.get("require_model_under") and float(probabilities[index]) >= 0.5:
            continue
        edge = under_probability - under_implied
        if edge < float(policy["minimum_model_edge"]):
            continue
        outcome = 1 - int(row["over_outcome"])
        profit = (100.0 / abs(under_odds)) if outcome else -1.0
        candidates.append({
            **row,
            "selection": "Under",
            "model_probability": under_probability,
            "model_edge": edge,
            "selected_odds": under_odds,
            "selected_outcome": outcome,
            "profit": profit,
        })
    return candidates


def _one_per_game(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_event: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_event.setdefault(str(row.get("event_id") or ""), []).append(row)
    return [
        sorted(
            event_rows,
            key=lambda row: (
                -safe_float(row.get("model_edge")),
                -safe_float(row.get("model_probability")),
                str(row.get("athlete_id") or ""),
            ),
        )[0]
        for event_rows in by_event.values()
    ]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = len(rows)
    wins = sum(int(row.get("selected_outcome") or 0) for row in rows)
    accuracy = wins / samples if samples else None
    roi = sum(safe_float(row.get("profit")) for row in rows) / samples if samples else None
    return {
        "samples": samples,
        "wins": wins,
        "losses": samples - wins,
        "accuracy": accuracy,
        "roi": roi,
        "average_probability": (
            sum(safe_float(row.get("model_probability")) for row in rows) / samples
            if samples
            else None
        ),
        "average_edge": (
            sum(safe_float(row.get("model_edge")) for row in rows) / samples
            if samples
            else None
        ),
    }


def _evaluate(
    features: Any,
    *,
    training_end: date,
    evaluation_start: date,
    evaluation_end: date,
    policy: dict[str, Any],
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    training = features[features["date"] <= training_end.isoformat()]
    evaluation = features[
        (features["date"] >= evaluation_start.isoformat())
        & (features["date"] <= evaluation_end.isoformat())
    ]
    model = _fit(training)
    selections = _one_per_game(_candidate_rows(model, evaluation, policy))
    return model, selections, _metrics(selections)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--target-accuracy", type=float, default=TARGET_ACCURACY)
    args = parser.parse_args()
    history_path = args.history.resolve()
    if not history_path.exists():
        raise SystemExit(f"Missing immutable market history: {history_path}")
    market_rows = _read_history(history_path)
    feature_rows, profiles = build_training_features(market_rows)
    features = _frame([row for row in feature_rows if str(row.get("sport") or "").upper() == "MLB"])
    if features.empty:
        raise SystemExit("No MLB precision training rows were built")

    max_date = date.fromisoformat(str(features["date"].max()))
    holdout_start = max_date - timedelta(days=8)
    validation_end = holdout_start - timedelta(days=1)
    validation_start = validation_end - timedelta(days=20)
    training_end = validation_start - timedelta(days=1)
    policy = dict(DEFAULT_POLICY)

    _, validation_rows, validation = _evaluate(
        features,
        training_end=training_end,
        evaluation_start=validation_start,
        evaluation_end=validation_end,
        policy=policy,
    )
    _, holdout_rows, holdout = _evaluate(
        features,
        training_end=validation_end,
        evaluation_start=holdout_start,
        evaluation_end=max_date,
        policy=policy,
    )
    combined = _metrics(validation_rows + holdout_rows)
    target = float(args.target_accuracy)
    active = bool(
        validation["samples"] >= MIN_VALIDATION_PICKS
        and holdout["samples"] >= MIN_HOLDOUT_PICKS
        and safe_float(validation["accuracy"]) >= target
        and safe_float(holdout["accuracy"]) >= target
        and safe_float(combined["roi"], -1.0) >= 0.0
    )
    final_model = _fit(features)
    training_fingerprint = hashlib.sha256(history_path.read_bytes()).hexdigest()
    metadata = {
        "version": PRECISION_MODEL_VERSION,
        "active": active,
        "supported_sports": ["MLB"] if active else [],
        "training_season": max_date.year,
        "training_end_date": max_date.isoformat(),
        "market_rows": len(market_rows),
        "feature_rows": len(features),
        "model_type": "StandardScaler+OneHotEncoder+LogisticRegression with selective precision gate",
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_accuracy": target,
        "policy": policy,
        "training_window": {
            "end": training_end.isoformat(),
        },
        "validation_window": {
            "start": validation_start.isoformat(),
            "end": validation_end.isoformat(),
        },
        "holdout_window": {
            "start": holdout_start.isoformat(),
            "end": max_date.isoformat(),
        },
        "validation": validation,
        "holdout": holdout,
        "combined_out_of_sample": combined,
        "activation_requirements": {
            "minimum_accuracy": target,
            "minimum_validation_picks": MIN_VALIDATION_PICKS,
            "minimum_holdout_picks": MIN_HOLDOUT_PICKS,
            "minimum_combined_roi": 0.0,
            "chronological_split_required": True,
            "one_pick_per_game": True,
        },
        "wnba": {
            "active": False,
            "reason": "No WNBA policy reached 70% on both chronological validation and holdout.",
        },
        "training_fingerprint": training_fingerprint,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib  # type: ignore

    joblib.dump(
        {
            "model": final_model,
            "profiles": profiles,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        },
        MODEL_PATH,
    )
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    # An inactive artifact is still a successful safety result: inference will
    # abstain instead of falling back to the unvalidated legacy ranker.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
