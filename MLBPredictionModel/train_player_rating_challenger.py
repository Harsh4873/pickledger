"""Future trainer for a timestamped MLB player-lineup rating challenger.

This script intentionally never writes a ``*_new`` artifact and is not wired
to ``run_today.py``, the model cache, calibration, consensus, or the public
ledger. It is deliberately gated on an archive of genuinely pregame lineup
and starter snapshots. Final StatsAPI box-score lineups do not meet that
contract and must use ``backtest_player_rating_oracle.py`` only as a labeled
sensitivity test, never as production-comparison training data.

When a snapshot archive exists, run from ``MLBPredictionModel`` with a dataset
that supplies every lineup/starter snapshot timestamp before first pitch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from features_v2 import (
    CATEGORICAL_FEATURES_V2,
    NUMERIC_FEATURES_V2,
    build_feature_frame,
    feature_columns_v2,
)
from historical_data import DATASET_PATH
from player_rating_features import RATING_SCHEMA


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
DEFAULT_ARTIFACT_PATH = ARTIFACT_DIR / "mlb_player_rating_challenger.joblib"
DEFAULT_METADATA_PATH = ARTIFACT_DIR / "mlb_player_rating_challenger_metadata.json"

CHALLENGER_SCHEMA_VERSION = "mlb_new_player_rating_challenger_v1"
PLAYER_RATING_FEATURES = [
    "lineup_batting_rating_adv",
    "lineup_power_rating_adv",
    "lineup_rating_reliability_adv",
]
REQUIRED_RATING_COLUMNS = [
    "home_lineup_rating_available",
    "away_lineup_rating_available",
    *PLAYER_RATING_FEATURES,
]
PREGAME_SNAPSHOT_COLUMNS = [
    "game_start_utc",
    "pregame_lineup_snapshot_utc",
    "pregame_starter_snapshot_utc",
]


def _recency_weights(frame: pd.DataFrame) -> np.ndarray:
    years = pd.to_datetime(frame["game_date"]).dt.year.to_numpy()
    weights = np.ones(len(frame), dtype=float)
    if not len(years):
        return weights
    newest = int(np.max(years))
    weights[years <= newest - 3] = 0.5
    weights[years == newest - 2] = 0.85
    weights[years == newest - 1] = 1.15
    weights[years == newest] = 1.5
    return weights


def _build_pipeline(numeric_features: list[str]) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, CATEGORICAL_FEATURES_V2),
        ]
    )
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_iter=450,
        max_depth=5,
        min_samples_leaf=40,
        l2_regularization=0.35,
        validation_fraction=0.12,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def validate_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    missing = [
        column
        for column in [*REQUIRED_RATING_COLUMNS, *PREGAME_SNAPSHOT_COLUMNS]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(
            "Dataset is missing pregame player-rating/snapshot columns: "
            + ", ".join(missing)
            + ". Final box-score lineups are intentionally rejected; provide a timestamped "
            "pregame lineup/starter snapshot archive instead."
        )
    if "game_date" not in frame.columns or "home_win" not in frame.columns:
        raise ValueError("Dataset must contain game_date and home_win.")

    dates = pd.to_datetime(frame["game_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Dataset contains invalid game_date values.")
    if "game_pk" in frame.columns and frame["game_pk"].duplicated().any():
        duplicates = int(frame["game_pk"].duplicated().sum())
        raise ValueError(f"Dataset has {duplicates} duplicate game_pk rows.")

    game_start = pd.to_datetime(frame["game_start_utc"], utc=True, errors="coerce")
    lineup_snapshot = pd.to_datetime(frame["pregame_lineup_snapshot_utc"], utc=True, errors="coerce")
    starter_snapshot = pd.to_datetime(frame["pregame_starter_snapshot_utc"], utc=True, errors="coerce")
    if game_start.isna().any() or lineup_snapshot.isna().any() or starter_snapshot.isna().any():
        raise ValueError("Dataset contains an invalid pregame snapshot timestamp.")
    if (lineup_snapshot >= game_start).any() or (starter_snapshot >= game_start).any():
        raise ValueError("Every lineup and starter snapshot must precede first pitch.")

    confirmed = (
        (_numeric_series(frame, "home_lineup_rating_available") >= 1.0)
        & (_numeric_series(frame, "away_lineup_rating_available") >= 1.0)
    )
    home_ml = _numeric_series(frame, "home_moneyline", default=np.nan)
    away_ml = _numeric_series(frame, "away_moneyline", default=np.nan)
    priced = home_ml.notna() & away_ml.notna() & (home_ml != 0) & (away_ml != 0)
    return {
        "rows": int(len(frame)),
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "confirmed_lineup_rows": int(confirmed.sum()),
        "confirmed_lineup_rate": float(confirmed.mean()) if len(frame) else 0.0,
        "priced_rows": int(priced.sum()),
        "priced_rate": float(priced.mean()) if len(frame) else 0.0,
        "duplicate_game_pks": 0,
    }


def prepare_challenger_frame(
    dataset: pd.DataFrame,
    *,
    market_only: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    quality = validate_dataset(dataset)
    raw = dataset.copy().reset_index(drop=True)
    raw["game_date"] = pd.to_datetime(raw["game_date"])

    confirmed = (
        (_numeric_series(raw, "home_lineup_rating_available") >= 1.0)
        & (_numeric_series(raw, "away_lineup_rating_available") >= 1.0)
    )
    home_ml = _numeric_series(raw, "home_moneyline", default=np.nan)
    away_ml = _numeric_series(raw, "away_moneyline", default=np.nan)
    priced = home_ml.notna() & away_ml.notna() & (home_ml != 0) & (away_ml != 0)
    eligible = confirmed & (priced if market_only else pd.Series(True, index=raw.index))
    raw = raw.loc[eligible].reset_index(drop=True)
    if raw.empty:
        scope = "confirmed, priced" if market_only else "confirmed"
        raise ValueError(f"No {scope} rows remain for challenger training.")

    base = build_feature_frame(raw).reset_index(drop=True)
    for column in PLAYER_RATING_FEATURES:
        base[column] = _numeric_series(raw, column)
    base["lineup_rating_available"] = np.minimum(
        _numeric_series(raw, "home_lineup_rating_available"),
        _numeric_series(raw, "away_lineup_rating_available"),
    )
    base["market_available"] = priced.loc[eligible].to_numpy(dtype=bool)
    base = base.dropna(subset=["home_win", "game_date"]).sort_values("game_date").reset_index(drop=True)
    if base["game_date"].nunique() < 8:
        raise ValueError("Need at least eight distinct game dates for a walk-forward test.")

    quality["eligible_rows"] = int(len(base))
    quality["eligible_date_start"] = str(base["game_date"].min().date())
    quality["eligible_date_end"] = str(base["game_date"].max().date())
    quality["market_only"] = bool(market_only)
    return base, quality


def walk_forward_folds(
    frame: pd.DataFrame,
    *,
    folds: int,
    validation_year: int | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    """Create date-boundary folds so same-day games never cross train/test."""
    if folds < 1:
        raise ValueError("folds must be at least 1")

    prepared = frame.sort_values("game_date").reset_index(drop=True)
    dates = pd.to_datetime(prepared["game_date"])
    if validation_year is None:
        validation_year = int(dates.dt.year.max())
    validation_dates = sorted(dates.loc[dates.dt.year == validation_year].dt.normalize().unique())
    if len(validation_dates) < folds:
        raise ValueError(
            f"Validation year {validation_year} has only {len(validation_dates)} dates for {folds} folds."
        )

    chunks = [chunk for chunk in np.array_split(np.array(validation_dates), folds) if len(chunk)]
    output: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    for fold_index, chunk in enumerate(chunks, start=1):
        start = pd.Timestamp(chunk[0]).normalize()
        end = pd.Timestamp(chunk[-1]).normalize()
        train = prepared.loc[dates < start].copy()
        validation = prepared.loc[(dates >= start) & (dates <= end)].copy()
        if train.empty or validation.empty:
            continue
        if train["game_date"].max() >= validation["game_date"].min():
            raise AssertionError("Walk-forward split leaked a game date into training.")
        output.append(
            (
                train,
                validation,
                {
                    "fold": fold_index,
                    "training_end": str(train["game_date"].max().date()),
                    "validation_start": str(validation["game_date"].min().date()),
                    "validation_end": str(validation["game_date"].max().date()),
                    "training_rows": int(len(train)),
                    "validation_rows": int(len(validation)),
                },
            )
        )
    if not output:
        raise ValueError("No usable walk-forward folds were created.")
    return output


def _evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "accuracy": float(accuracy_score(y_true, (clipped >= 0.5).astype(int))),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, clipped)),
    }


def _bootstrap_log_loss_delta(
    y_true: np.ndarray,
    challenger: np.ndarray,
    baseline: np.ndarray,
    *,
    samples: int = 1200,
) -> dict[str, float]:
    """Bootstrap challenger-minus-baseline log loss; negative is better."""
    y = np.asarray(y_true, dtype=float)
    if len(y) < 2:
        return {"mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    challenger = np.clip(np.asarray(challenger, dtype=float), 1e-6, 1 - 1e-6)
    baseline = np.clip(np.asarray(baseline, dtype=float), 1e-6, 1 - 1e-6)
    challenger_loss = -(y * np.log(challenger) + (1 - y) * np.log(1 - challenger))
    baseline_loss = -(y * np.log(baseline) + (1 - y) * np.log(1 - baseline))
    delta = challenger_loss - baseline_loss
    rng = np.random.default_rng(42)
    sample_means = np.empty(samples, dtype=float)
    for index in range(samples):
        sample_means[index] = float(np.mean(rng.choice(delta, size=len(delta), replace=True)))
    return {
        "mean_delta": float(np.mean(delta)),
        "ci95_low": float(np.quantile(sample_means, 0.025)),
        "ci95_high": float(np.quantile(sample_means, 0.975)),
    }


def _matrix(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected = frame.copy()
    for column in columns:
        if column not in selected.columns:
            selected[column] = np.nan
    return selected[list(columns)].copy()


def evaluate_walk_forward(
    prepared: pd.DataFrame,
    *,
    folds: int,
    validation_year: int | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    base_features = feature_columns_v2()
    challenger_features = [*base_features, *PLAYER_RATING_FEATURES]
    fold_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []

    for train, validation, fold_metadata in walk_forward_folds(
        prepared, folds=folds, validation_year=validation_year
    ):
        train_y = train["home_win"].astype(int).to_numpy()
        validation_y = validation["home_win"].astype(int).to_numpy()
        base_pipeline = _build_pipeline(NUMERIC_FEATURES_V2)
        challenger_pipeline = _build_pipeline([*NUMERIC_FEATURES_V2, *PLAYER_RATING_FEATURES])
        weights = _recency_weights(train)

        base_pipeline.fit(
            _matrix(train, base_features),
            train_y,
            estimator__sample_weight=weights,
        )
        challenger_pipeline.fit(
            _matrix(train, challenger_features),
            train_y,
            estimator__sample_weight=weights,
        )
        baseline_prob = base_pipeline.predict_proba(_matrix(validation, base_features))[:, 1]
        challenger_prob = challenger_pipeline.predict_proba(
            _matrix(validation, challenger_features)
        )[:, 1]
        market_prob = validation["market_home_vigfree_prob"].to_numpy(dtype=float)

        fold_rows.append(
            {
                **fold_metadata,
                "baseline": _evaluate_probabilities(validation_y, baseline_prob),
                "challenger": _evaluate_probabilities(validation_y, challenger_prob),
                "market": _evaluate_probabilities(validation_y, market_prob),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "game_date": validation["game_date"].to_numpy(),
                    "home_win": validation_y,
                    "baseline_probability": baseline_prob,
                    "challenger_probability": challenger_prob,
                    "market_probability": market_prob,
                    "fold": fold_metadata["fold"],
                }
            )
        )

    predictions = pd.concat(prediction_rows, ignore_index=True)
    actual = predictions["home_win"].to_numpy(dtype=int)
    baseline = predictions["baseline_probability"].to_numpy(dtype=float)
    challenger = predictions["challenger_probability"].to_numpy(dtype=float)
    market = predictions["market_probability"].to_numpy(dtype=float)
    baseline_metrics = _evaluate_probabilities(actual, baseline)
    challenger_metrics = _evaluate_probabilities(actual, challenger)
    market_metrics = _evaluate_probabilities(actual, market)
    bootstrap = _bootstrap_log_loss_delta(actual, challenger, baseline)
    deltas = {
        "challenger_minus_baseline_log_loss": challenger_metrics["log_loss"] - baseline_metrics["log_loss"],
        "challenger_minus_baseline_brier": challenger_metrics["brier_score"] - baseline_metrics["brier_score"],
        "challenger_minus_baseline_accuracy": challenger_metrics["accuracy"] - baseline_metrics["accuracy"],
        "challenger_minus_market_log_loss": challenger_metrics["log_loss"] - market_metrics["log_loss"],
        "challenger_minus_market_brier": challenger_metrics["brier_score"] - market_metrics["brier_score"],
    }
    statistically_positive = (
        deltas["challenger_minus_baseline_log_loss"] <= -0.002
        and deltas["challenger_minus_baseline_brier"] <= 0.0
        and bootstrap["ci95_high"] < 0.0
    )
    report = {
        "validation_rows": int(len(predictions)),
        "validation_date_start": str(predictions["game_date"].min().date()),
        "validation_date_end": str(predictions["game_date"].max().date()),
        "baseline_metrics": baseline_metrics,
        "challenger_metrics": challenger_metrics,
        "market_metrics": market_metrics,
        "deltas": deltas,
        "paired_log_loss_bootstrap": bootstrap,
        "folds": fold_rows,
        "statistically_positive_vs_baseline": bool(statistically_positive),
    }
    return report, predictions


def train_final_artifact(prepared: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    feature_columns = [*feature_columns_v2(), *PLAYER_RATING_FEATURES]
    pipeline = _build_pipeline([*NUMERIC_FEATURES_V2, *PLAYER_RATING_FEATURES])
    pipeline.fit(
        _matrix(prepared, feature_columns),
        prepared["home_win"].astype(int).to_numpy(),
        estimator__sample_weight=_recency_weights(prepared),
    )
    return {
        "pipeline": pipeline,
        "metadata": metadata,
    }


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found at {path}. Build it with "
            "--include-player-ratings before training the challenger."
        )
    return pd.read_csv(path, parse_dates=["game_date"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLB player-rating challenger (research only).")
    parser.add_argument("--dataset", default=str(DATASET_PATH), help="Historical CSV with player-rating columns.")
    parser.add_argument("--folds", type=int, default=3, help="Date-boundary walk-forward folds.")
    parser.add_argument(
        "--validation-year",
        type=int,
        default=None,
        help="Year reserved for sequential walk-forward validation (defaults to newest year).",
    )
    parser.add_argument(
        "--market-only",
        action="store_true",
        help="Train and score only rows with both historical moneylines for a fair market comparison.",
    )
    parser.add_argument("--output-artifact", default=str(DEFAULT_ARTIFACT_PATH))
    parser.add_argument("--output-metadata", default=str(DEFAULT_METADATA_PATH))
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Evaluate only; do not write the research artifact or metadata JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset)
    dataset = _load_dataset(dataset_path)
    prepared, quality = prepare_challenger_frame(dataset, market_only=args.market_only)
    validation_year = args.validation_year or int(pd.to_datetime(prepared["game_date"]).dt.year.max())
    evaluation, _ = evaluate_walk_forward(
        prepared,
        folds=args.folds,
        validation_year=validation_year,
    )

    metadata: dict[str, Any] = {
        "variant": "player_rating_challenger",
        "research_only": True,
        "promotion_state": "blocked_pending_pregame_lineup_snapshot_validation",
        "feature_schema_version": CHALLENGER_SCHEMA_VERSION,
        "player_rating_schema": RATING_SCHEMA,
        "base_feature_schema": "mlb_new_v2",
        "feature_columns": [*feature_columns_v2(), *PLAYER_RATING_FEATURES],
        "training_rows": int(len(prepared)),
        "training_date_start": str(prepared["game_date"].min().date()),
        "training_date_end": str(prepared["game_date"].max().date()),
        "validation_year": validation_year,
        "quality": quality,
        "walk_forward_evaluation": evaluation,
        "critical_caveat": (
            "Historical lineup rows use final-boxscore batting order with current-game "
            "stats removed. Live scoring is deliberately opt-in and must require a "
            "confirmed pregame batting order before this artifact can be promoted."
        ),
    }

    if not args.no_save:
        artifact_path = Path(args.output_artifact)
        metadata_path = Path(args.output_metadata)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(train_final_artifact(prepared, metadata), artifact_path)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary = {
        "research_only": True,
        "training_rows": metadata["training_rows"],
        "validation_rows": evaluation["validation_rows"],
        "baseline": evaluation["baseline_metrics"],
        "challenger": evaluation["challenger_metrics"],
        "market": evaluation["market_metrics"],
        "deltas": evaluation["deltas"],
        "statistically_positive_vs_baseline": evaluation["statistically_positive_vs_baseline"],
        "promotion_state": metadata["promotion_state"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
