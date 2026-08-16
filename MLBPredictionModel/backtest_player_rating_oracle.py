"""Walk-forward test for the strict MLB player-lineup oracle dataset.

The control and challenger share the same date-safe team/starter inputs.  The
only incremental inputs are the ordered nine-player batting-rating features.
This is deliberately a research sensitivity test: historical lineup/starter
identity is oracle knowledge and no artifact from this script is loadable by
the public MLB model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from player_rating_features import RATING_SCHEMA
from player_rating_oracle_data import (
    DEFAULT_OUTPUT_PATH,
    ORACLE_DATASET_SCHEMA,
    ORACLE_STATE_UPDATE_POLICY,
)


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DEFAULT_ARTIFACT_PATH = ARTIFACT_DIR / "mlb_player_rating_oracle_challenger.joblib"
DEFAULT_METADATA_PATH = ARTIFACT_DIR / "mlb_player_rating_oracle_challenger_metadata.json"

CONTROL_FEATURES = [
    "market_home_vigfree_prob",
    "team_win_pct_adv",
    "team_run_diff_per_game_adv",
    "team_games_adv",
    "starter_era_adv",
]
PLAYER_FEATURES = [
    "lineup_batting_rating_adv",
    "lineup_power_rating_adv",
    "lineup_rating_reliability_adv",
]
CHALLENGER_FEATURES = [*CONTROL_FEATURES, *PLAYER_FEATURES]
BACKTEST_SCHEMA = "mlb_player_rating_oracle_backtest_v2"


def _build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=400,
        max_depth=4,
        min_samples_leaf=32,
        l2_regularization=0.35,
        validation_fraction=0.12,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
    )


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y_true, (clipped >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, clipped)),
    }


def _bootstrap_log_loss_delta(
    y_true: np.ndarray,
    challenger: np.ndarray,
    control: np.ndarray,
    *,
    samples: int = 1200,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    challenger = np.clip(np.asarray(challenger, dtype=float), 1e-6, 1 - 1e-6)
    control = np.clip(np.asarray(control, dtype=float), 1e-6, 1 - 1e-6)
    challenger_loss = -(y * np.log(challenger) + (1.0 - y) * np.log(1.0 - challenger))
    control_loss = -(y * np.log(control) + (1.0 - y) * np.log(1.0 - control))
    delta = challenger_loss - control_loss
    rng = np.random.default_rng(42)
    means = np.empty(samples, dtype=float)
    for index in range(samples):
        means[index] = float(np.mean(rng.choice(delta, size=len(delta), replace=True)))
    return {
        "mean_delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _validate_and_prepare(frame: pd.DataFrame, *, market_only: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "dataset_schema",
        "state_update_policy",
        "game_pk",
        "game_start_utc",
        "home_win",
        "market_available",
        "oracle_lineup_known",
        "oracle_starter_known",
        *CONTROL_FEATURES,
        *PLAYER_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Dataset is missing required oracle columns: " + ", ".join(missing))
    if set(frame["dataset_schema"].dropna().unique()) != {ORACLE_DATASET_SCHEMA}:
        raise ValueError("Dataset schema is not the strict player-rating oracle schema.")
    if set(frame["state_update_policy"].dropna().unique()) != {ORACLE_STATE_UPDATE_POLICY}:
        raise ValueError("Dataset does not use verified game-end state updates.")
    if frame["game_pk"].duplicated().any():
        raise ValueError("Oracle dataset has duplicate game_pk rows.")

    prepared = frame.copy()
    prepared["game_start_utc"] = pd.to_datetime(prepared["game_start_utc"], utc=True, errors="coerce")
    if prepared["game_start_utc"].isna().any():
        raise ValueError("Oracle dataset has invalid game_start_utc values.")
    for column in [*CONTROL_FEATURES, *PLAYER_FEATURES]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)
    prepared["home_win"] = pd.to_numeric(prepared["home_win"], errors="raise").astype(int)
    prepared["market_available"] = pd.to_numeric(prepared["market_available"], errors="coerce").fillna(0).astype(int)
    prepared = prepared.loc[
        (prepared["oracle_lineup_known"] == 1) & (prepared["oracle_starter_known"] == 1)
    ].copy()
    if market_only:
        prepared = prepared.loc[prepared["market_available"] == 1].copy()
    prepared = prepared.sort_values(["game_start_utc", "game_pk"]).reset_index(drop=True)
    if len(prepared) < 300:
        raise ValueError("Need at least 300 eligible games for the oracle backtest.")

    quality = {
        "source_rows": int(len(frame)),
        "eligible_rows": int(len(prepared)),
        "market_only": bool(market_only),
        "priced_rows": int((prepared["market_available"] == 1).sum()),
        "date_start": str(prepared["game_start_utc"].min().date()),
        "date_end": str(prepared["game_start_utc"].max().date()),
        "duplicate_game_pks": 0,
        "oracle_lineup_known_rate": float((prepared["oracle_lineup_known"] == 1).mean()),
        "state_update_policy": ORACLE_STATE_UPDATE_POLICY,
    }
    return prepared, quality


def _walk_forward_folds(
    frame: pd.DataFrame,
    *,
    folds: int,
    initial_train_fraction: float,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    if not 0.3 <= initial_train_fraction < 0.9:
        raise ValueError("initial_train_fraction must be in [0.3, 0.9).")
    if folds < 1:
        raise ValueError("folds must be at least 1.")

    starts = pd.to_datetime(frame["game_start_utc"], utc=True).dt.normalize()
    unique_dates = np.array(sorted(starts.unique()))
    split = max(1, int(len(unique_dates) * initial_train_fraction))
    validation_dates = unique_dates[split:]
    if len(validation_dates) < folds:
        raise ValueError("Not enough validation dates for requested folds.")

    output: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    for index, chunk in enumerate(np.array_split(validation_dates, folds), start=1):
        if not len(chunk):
            continue
        start = pd.Timestamp(chunk[0])
        end = pd.Timestamp(chunk[-1])
        train = frame.loc[starts < start].copy()
        validation = frame.loc[(starts >= start) & (starts <= end)].copy()
        if train.empty or validation.empty:
            continue
        if train["game_start_utc"].max() >= validation["game_start_utc"].min():
            raise AssertionError("A walk-forward fold leaked a first-pitch timestamp into training.")
        output.append(
            (
                train,
                validation,
                {
                    "fold": index,
                    "training_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                    "training_end": str(train["game_start_utc"].max()),
                    "validation_start": str(validation["game_start_utc"].min()),
                    "validation_end": str(validation["game_start_utc"].max()),
                },
            )
        )
    if not output:
        raise ValueError("No valid walk-forward folds were produced.")
    return output


def run_backtest(
    prepared: pd.DataFrame,
    *,
    folds: int,
    initial_train_fraction: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for train, validation, detail in _walk_forward_folds(
        prepared, folds=folds, initial_train_fraction=initial_train_fraction
    ):
        y_train = train["home_win"].to_numpy(dtype=int)
        y_validation = validation["home_win"].to_numpy(dtype=int)
        control = _build_model().fit(train[CONTROL_FEATURES], y_train)
        challenger = _build_model().fit(train[CHALLENGER_FEATURES], y_train)
        control_probability = control.predict_proba(validation[CONTROL_FEATURES])[:, 1]
        challenger_probability = challenger.predict_proba(validation[CHALLENGER_FEATURES])[:, 1]
        market_probability = validation["market_home_vigfree_prob"].to_numpy(dtype=float)
        fold_results.append(
            {
                **detail,
                "control": _metrics(y_validation, control_probability),
                "challenger": _metrics(y_validation, challenger_probability),
                "market": _metrics(y_validation, market_probability),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "game_pk": validation["game_pk"].to_numpy(),
                    "game_start_utc": validation["game_start_utc"].to_numpy(),
                    "home_win": y_validation,
                    "market_available": validation["market_available"].to_numpy(dtype=int),
                    "control_probability": control_probability,
                    "challenger_probability": challenger_probability,
                    "market_probability": market_probability,
                    "fold": detail["fold"],
                }
            )
        )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    y = prediction_frame["home_win"].to_numpy(dtype=int)
    control_probability = prediction_frame["control_probability"].to_numpy(dtype=float)
    challenger_probability = prediction_frame["challenger_probability"].to_numpy(dtype=float)
    market_probability = prediction_frame["market_probability"].to_numpy(dtype=float)
    control_metrics = _metrics(y, control_probability)
    challenger_metrics = _metrics(y, challenger_probability)
    priced = prediction_frame["market_available"].to_numpy(dtype=int) == 1
    market_metrics = _metrics(y[priced], market_probability[priced]) if priced.any() else None
    market_challenger_metrics = (
        _metrics(y[priced], challenger_probability[priced]) if priced.any() else None
    )
    bootstrap = _bootstrap_log_loss_delta(y, challenger_probability, control_probability)
    deltas = {
        "challenger_minus_control_log_loss": challenger_metrics["log_loss"] - control_metrics["log_loss"],
        "challenger_minus_control_brier": challenger_metrics["brier_score"] - control_metrics["brier_score"],
        "challenger_minus_control_accuracy": challenger_metrics["accuracy"] - control_metrics["accuracy"],
    }
    if market_metrics and market_challenger_metrics:
        deltas["challenger_minus_market_log_loss_priced"] = (
            market_challenger_metrics["log_loss"] - market_metrics["log_loss"]
        )
        deltas["challenger_minus_market_brier_priced"] = (
            market_challenger_metrics["brier_score"] - market_metrics["brier_score"]
        )

    return (
        {
            "validation_rows": int(len(prediction_frame)),
            "validation_date_start": str(prediction_frame["game_start_utc"].min()),
            "validation_date_end": str(prediction_frame["game_start_utc"].max()),
            "control_metrics": control_metrics,
            "challenger_metrics": challenger_metrics,
            "market_metrics_priced": market_metrics,
            "challenger_metrics_priced": market_challenger_metrics,
            "deltas": deltas,
            "paired_log_loss_bootstrap": bootstrap,
            "folds": fold_results,
            "statistically_positive_vs_control": bool(
                deltas["challenger_minus_control_log_loss"] <= -0.002
                and deltas["challenger_minus_control_brier"] <= 0.0
                and bootstrap["ci95_high"] < 0.0
            ),
        },
        prediction_frame,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest strict MLB player-lineup oracle challenger.")
    parser.add_argument("--dataset", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--initial-train-fraction", type=float, default=0.60)
    parser.add_argument("--market-only", action="store_true")
    parser.add_argument("--output-artifact", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--output-metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Oracle dataset not found at {dataset_path}. Run player_rating_oracle_data.py first."
        )
    source = pd.read_csv(dataset_path)
    prepared, quality = _validate_and_prepare(source, market_only=args.market_only)
    evaluation, _ = run_backtest(
        prepared,
        folds=args.folds,
        initial_train_fraction=args.initial_train_fraction,
    )
    metadata: dict[str, Any] = {
        "variant": "player_rating_oracle_challenger",
        "research_only": True,
        "promotion_state": "blocked_oracle_lineup_and_starter_identity",
        "backtest_schema": BACKTEST_SCHEMA,
        "dataset_schema": ORACLE_DATASET_SCHEMA,
        "player_rating_schema": RATING_SCHEMA,
        "control_features": CONTROL_FEATURES,
        "challenger_features": CHALLENGER_FEATURES,
        "quality": quality,
        "walk_forward_evaluation": evaluation,
        "critical_caveat": (
            "Player, pitcher, and team state uses only outcomes with a verified final-play "
            "timestamp before the next game's scheduled first pitch. Lineup and starter "
            "identity still come from the final historical boxscore. This is an oracle-lineup "
            "sensitivity test, not an MLB New production backtest or promotion decision."
        ),
    }
    if not args.no_save:
        artifact_path = Path(args.output_artifact)
        metadata_path = Path(args.output_metadata)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        model = _build_model().fit(prepared[CHALLENGER_FEATURES], prepared["home_win"].to_numpy(dtype=int))
        joblib.dump({"pipeline": model, "metadata": metadata}, artifact_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "research_only": True,
                "training_rows": int(len(prepared)),
                "validation_rows": evaluation["validation_rows"],
                "control": evaluation["control_metrics"],
                "challenger": evaluation["challenger_metrics"],
                "market_priced": evaluation["market_metrics_priced"],
                "deltas": evaluation["deltas"],
                "statistically_positive_vs_control": evaluation[
                    "statistically_positive_vs_control"
                ],
                "promotion_state": metadata["promotion_state"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
