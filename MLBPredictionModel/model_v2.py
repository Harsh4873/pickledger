"""
MLB model v2 — modern stack used by the "new" variant.

Architecture:
  * HistGradientBoostingClassifier for moneyline (directly outputs a home-win
    probability). No blended heuristic at inference. Handles missing values
    natively and captures non-linear feature interactions.
  * HistGradientBoostingRegressor for totals, but trained to predict the
    *residual* versus the closing market total line. At inference the
    prediction is `market_total_line + predicted_residual`, keeping us anchored
    to the market consensus.
  * IsotonicRegression calibration on held-out data for moneyline, so the
    output probabilities are actually well-calibrated — no more conservative
    cap that flattens the distribution.

The training entry points are invoked by `train_model_v2.py` and by the
GitHub Actions workflow. They write the artifacts the "new" variant loads:

  artifacts/mlb_moneyline_model_new.joblib
  artifacts/mlb_totals_model_new.joblib
  artifacts/mlb_probability_calibration_new.joblib
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from features_v2 import (
    CATEGORICAL_FEATURES_V2,
    LEAGUE_AVG_TOTAL,
    NUMERIC_FEATURES_V2,
    build_feature_frame,
    feature_columns_v2,
    select_feature_matrix,
    select_training_rows_v2,
)
from model_variants import ARTIFACT_DIR, MLB_MODEL_VARIANTS


BASE_DIR = Path(__file__).resolve().parent


class StaleV2Artifact(Exception):
    """Raised when a v2 artifact file exists on disk but carries legacy
    metadata (missing ``variant="new"``). The caller is expected to fall
    back to the legacy moneyline/totals stack for the ``new`` variant so
    MLB New can still run standalone on a cold start, without waiting for
    ``train_model_v2.py`` or the GitHub Actions training workflow to run.
    """


MONEYLINE_V2_PATH = MLB_MODEL_VARIANTS["new"]["moneyline"]
MONEYLINE_V2_METADATA_PATH = MLB_MODEL_VARIANTS["new"]["moneyline_metadata"]
TOTALS_V2_PATH = MLB_MODEL_VARIANTS["new"]["totals"]
TOTALS_V2_METADATA_PATH = MLB_MODEL_VARIANTS["new"]["totals_metadata"]
CALIBRATION_V2_PATH = MLB_MODEL_VARIANTS["new"]["calibration"]
CALIBRATION_V2_METADATA_PATH = MLB_MODEL_VARIANTS["new"]["calibration_metadata"]


# Minimum observations we need on the validation fold before we trust the
# isotonic fit. Below this we fall back to a light Platt scaling.
MIN_ISOTONIC_ROWS = 400


@dataclass
class TrainingResult:
    artifact: dict[str, Any]
    metadata: dict[str, Any]


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def chronological_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values("game_date").reset_index(drop=True)
    split_idx = max(1, int(len(frame) * (1.0 - validation_fraction)))
    split_idx = min(split_idx, len(frame) - 1)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def _build_preprocessor() -> ColumnTransformer:
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
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES_V2),
            ("categorical", categorical, CATEGORICAL_FEATURES_V2),
        ]
    )


def _build_moneyline_pipeline() -> Pipeline:
    preprocessor = _build_preprocessor()
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


def _build_totals_pipeline() -> Pipeline:
    preprocessor = _build_preprocessor()
    estimator = HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=0.04,
        max_iter=500,
        max_depth=5,
        min_samples_leaf=45,
        l2_regularization=0.25,
        validation_fraction=0.12,
        early_stopping=True,
        n_iter_no_change=30,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("estimator", estimator)])


def _evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    predictions = (clipped >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, clipped)),
        "brier_score": float(brier_score_loss(y_true, clipped)),
    }


def _evaluate_regression(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
    }


def _recency_weights(frame: pd.DataFrame) -> np.ndarray:
    """Upweight recent seasons. Current-season pitching usage + run env shifts
    year over year, so the previous season is worth less than the current one.
    """
    years = pd.to_datetime(frame["game_date"]).dt.year.to_numpy()
    weights = np.ones(len(frame), dtype=float)
    # Anything older than 3 full seasons is a soft prior only.
    weights[years <= np.max(years) - 3] = 0.5
    weights[years == np.max(years) - 2] = 0.85
    weights[years == np.max(years) - 1] = 1.15
    weights[years == np.max(years)] = 1.5
    return weights


def train_moneyline_v2(
    dataset: pd.DataFrame,
    validation_fraction: float = 0.2,
) -> TrainingResult:
    frame = build_feature_frame(dataset)
    frame = select_training_rows_v2(frame)

    train_frame, validation_frame = chronological_split(frame, validation_fraction)

    train_x = select_feature_matrix(train_frame)
    validation_x = select_feature_matrix(validation_frame)
    train_y = train_frame["home_win"].astype(int).to_numpy()
    validation_y = validation_frame["home_win"].astype(int).to_numpy()

    pipeline = _build_moneyline_pipeline()
    sample_weights = _recency_weights(train_frame)
    pipeline.fit(train_x, train_y, estimator__sample_weight=sample_weights)

    raw_probabilities = pipeline.predict_proba(validation_x)[:, 1]

    # Include a market-only baseline so we can see whether the model actually
    # beats just betting the vig-free line.
    market_probabilities = validation_frame["market_home_vigfree_prob"].to_numpy(dtype=float)
    market_probabilities = np.where(
        np.isnan(market_probabilities), 0.5, market_probabilities
    )
    market_metrics = _evaluate_probabilities(validation_y, market_probabilities)
    raw_model_metrics = _evaluate_probabilities(validation_y, raw_probabilities)

    metadata: dict[str, Any] = {
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "training_end_date": str(train_frame["game_date"].max().date()),
        "validation_start_date": str(validation_frame["game_date"].min().date()),
        "feature_columns": feature_columns_v2(),
        "numeric_features": NUMERIC_FEATURES_V2,
        "categorical_features": CATEGORICAL_FEATURES_V2,
        "raw_model_metrics": raw_model_metrics,
        "market_baseline_metrics": market_metrics,
        "seasons": sorted(
            pd.to_datetime(frame["game_date"]).dt.year.dropna().astype(int).unique().tolist()
        ),
        "architecture": "HistGradientBoostingClassifier",
        "variant": "new",
    }

    artifact = {"pipeline": pipeline, "metadata": metadata}
    return TrainingResult(artifact=artifact, metadata=metadata)


def fit_calibration(
    pipeline: Pipeline,
    validation_x: pd.DataFrame,
    validation_y: np.ndarray,
) -> dict[str, Any]:
    raw_probabilities = pipeline.predict_proba(validation_x)[:, 1]

    if len(validation_y) < MIN_ISOTONIC_ROWS:
        # Not enough data for a clean isotonic fit; fall back to Platt-style
        # linear rescale.
        mean_pred = float(np.mean(raw_probabilities))
        mean_actual = float(np.mean(validation_y))
        shift = mean_actual - mean_pred
        metadata = {
            "mode": "platt_shift",
            "variant": "new",
            "shift": shift,
            "validation_rows": int(len(validation_y)),
        }
        return {"mode": "platt_shift", "shift": shift, "metadata": metadata}

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probabilities, validation_y.astype(float))
    calibrated = np.clip(calibrator.predict(raw_probabilities), 0.03, 0.97)
    metadata = {
        "mode": "isotonic",
        "variant": "new",
        "validation_rows": int(len(validation_y)),
        "pre_calibration": _evaluate_probabilities(validation_y, raw_probabilities),
        "post_calibration": _evaluate_probabilities(validation_y, calibrated),
    }
    return {"mode": "isotonic", "calibrator": calibrator, "metadata": metadata}


def apply_calibration(artifact: dict[str, Any], probabilities: np.ndarray) -> np.ndarray:
    if artifact is None:
        return np.clip(probabilities, 0.03, 0.97)
    mode = artifact.get("mode")
    if mode == "isotonic":
        calibrator: IsotonicRegression = artifact["calibrator"]
        return np.clip(calibrator.predict(probabilities), 0.03, 0.97)
    if mode == "platt_shift":
        shifted = probabilities + float(artifact.get("shift", 0.0))
        return np.clip(shifted, 0.03, 0.97)
    return np.clip(probabilities, 0.03, 0.97)


def train_totals_v2(
    dataset: pd.DataFrame,
    validation_fraction: float = 0.2,
) -> TrainingResult:
    frame = build_feature_frame(dataset)
    frame = frame.dropna(subset=["total_runs", "game_date"]).sort_values("game_date").reset_index(drop=True)

    # Residual-to-market training target. The model learns `total_runs -
    # market_total_line` so it stays anchored to Vegas and only deviates when
    # the features justify it.
    frame = frame.copy()
    frame["market_total_line"] = frame["market_total_line"].fillna(LEAGUE_AVG_TOTAL)
    frame["total_residual"] = frame["total_runs"].astype(float) - frame["market_total_line"].astype(float)

    train_frame, validation_frame = chronological_split(frame, validation_fraction)

    train_x = select_feature_matrix(train_frame)
    validation_x = select_feature_matrix(validation_frame)
    train_y = train_frame["total_residual"].to_numpy(dtype=float)
    validation_y = validation_frame["total_runs"].to_numpy(dtype=float)
    validation_market = validation_frame["market_total_line"].to_numpy(dtype=float)

    pipeline = _build_totals_pipeline()
    sample_weights = _recency_weights(train_frame)
    pipeline.fit(train_x, train_y, estimator__sample_weight=sample_weights)

    predicted_residual = pipeline.predict(validation_x)
    predicted_totals = validation_market + predicted_residual

    model_metrics = _evaluate_regression(validation_y, predicted_totals)
    market_metrics = _evaluate_regression(validation_y, validation_market)

    metadata = {
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "training_end_date": str(train_frame["game_date"].max().date()),
        "validation_start_date": str(validation_frame["game_date"].min().date()),
        "feature_columns": feature_columns_v2(),
        "numeric_features": NUMERIC_FEATURES_V2,
        "categorical_features": CATEGORICAL_FEATURES_V2,
        "model_metrics": model_metrics,
        "market_baseline_metrics": market_metrics,
        "architecture": "HistGradientBoostingRegressor (residual-to-market)",
        "target": "total_runs_minus_market_total_line",
        "variant": "new",
    }

    artifact = {"pipeline": pipeline, "metadata": metadata}
    return TrainingResult(artifact=artifact, metadata=metadata)


def save_training_artifacts(
    moneyline_result: TrainingResult,
    totals_result: TrainingResult,
    calibration_artifact: dict[str, Any],
) -> None:
    ensure_artifact_dir()

    joblib.dump(moneyline_result.artifact, MONEYLINE_V2_PATH)
    MONEYLINE_V2_METADATA_PATH.write_text(
        json.dumps(moneyline_result.metadata, indent=2), encoding="utf-8"
    )

    joblib.dump(totals_result.artifact, TOTALS_V2_PATH)
    TOTALS_V2_METADATA_PATH.write_text(
        json.dumps(totals_result.metadata, indent=2), encoding="utf-8"
    )

    joblib.dump(calibration_artifact, CALIBRATION_V2_PATH)
    CALIBRATION_V2_METADATA_PATH.write_text(
        json.dumps(calibration_artifact.get("metadata", {}), indent=2),
        encoding="utf-8",
    )


def _is_v2_artifact(artifact: dict[str, Any]) -> bool:
    """Detect artifacts produced by `train_model_v2.py`.

    The v2 trainer tags metadata with `variant=new` so we can tell a current
    stack apart from an older `_new` file that was produced by the legacy
    LogisticRegression pipeline (which uses a different feature schema and
    would crash at `predict_proba`).
    """
    metadata = artifact.get("metadata") or {}
    architecture = str(metadata.get("architecture") or "")
    return metadata.get("variant") == "new" and "HistGradientBoosting" in architecture


def load_moneyline_v2() -> dict[str, Any]:
    if not MONEYLINE_V2_PATH.exists():
        raise FileNotFoundError(
            "Missing v2 moneyline artifact. Run `python train_model_v2.py` "
            "inside MLBPredictionModel to train it, or trigger the "
            "mlb-train GitHub Actions workflow."
        )
    artifact = joblib.load(MONEYLINE_V2_PATH)
    if not _is_v2_artifact(artifact):
        raise StaleV2Artifact(
            f"Stale pre-v2 moneyline artifact at {MONEYLINE_V2_PATH}. "
            "Retrain with `python train_model_v2.py` (or trigger the mlb-train "
            "GitHub Actions workflow) so the HistGradientBoosting stack and "
            "v2 feature schema are used."
        )
    return artifact


def load_totals_v2() -> dict[str, Any]:
    if not TOTALS_V2_PATH.exists():
        raise FileNotFoundError(
            "Missing v2 totals artifact. Run `python train_model_v2.py` "
            "inside MLBPredictionModel to train it, or trigger the "
            "mlb-train GitHub Actions workflow."
        )
    artifact = joblib.load(TOTALS_V2_PATH)
    if not _is_v2_artifact(artifact):
        raise StaleV2Artifact(
            f"Stale pre-v2 totals artifact at {TOTALS_V2_PATH}. "
            "Retrain with `python train_model_v2.py` (or trigger the mlb-train "
            "GitHub Actions workflow) so the market-residual HistGradient "
            "Boosting regressor and v2 feature schema are used."
        )
    return artifact


def load_calibration_v2() -> dict[str, Any] | None:
    if not CALIBRATION_V2_PATH.exists():
        return None
    return joblib.load(CALIBRATION_V2_PATH)


def predict_moneyline_v2(frame: pd.DataFrame) -> pd.DataFrame:
    artifact = load_moneyline_v2()
    pipeline: Pipeline = artifact["pipeline"]
    features = build_feature_frame(frame)
    matrix = select_feature_matrix(features)
    raw_probabilities = pipeline.predict_proba(matrix)[:, 1]

    calibration = load_calibration_v2()
    calibrated = apply_calibration(calibration, raw_probabilities)

    out = frame.copy().reset_index(drop=True)
    out["raw_home_win_probability"] = raw_probabilities
    out["calibrated_home_win_probability"] = calibrated
    out["calibration_mode"] = (
        calibration.get("mode", "none") if calibration is not None else "none"
    )
    return out


def predict_totals_v2(frame: pd.DataFrame) -> pd.DataFrame:
    artifact = load_totals_v2()
    pipeline: Pipeline = artifact["pipeline"]
    features = build_feature_frame(frame)
    matrix = select_feature_matrix(features)
    predicted_residual = pipeline.predict(matrix)

    out = frame.copy().reset_index(drop=True)
    market_total = pd.to_numeric(
        features["market_total_line"], errors="coerce"
    ).fillna(LEAGUE_AVG_TOTAL).to_numpy()
    out["raw_model_total_runs"] = market_total + predicted_residual
    out["predicted_total_runs"] = out["raw_model_total_runs"]
    return out
