from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ensure_feature_frame,
    feature_columns,
    select_feature_frame,
    select_training_rows,
)
from historical_data import DATASET_PATH, build_historical_dataset


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "mlb_moneyline_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "mlb_moneyline_model_metadata.json"


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def chronological_split(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values("game_date").reset_index(drop=True)
    split_idx = max(1, int(len(frame) * (1.0 - validation_fraction)))
    split_idx = min(split_idx, len(frame) - 1)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    estimator = LogisticRegression(
        C=0.55,
        max_iter=3000,
        solver="lbfgs",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ]
    )


def evaluate_probabilities(target: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
    predictions = (clipped >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "log_loss": float(log_loss(target, clipped)),
        "brier_score": float(brier_score_loss(target, clipped)),
    }


def blend_probabilities(
    model_probabilities: np.ndarray,
    heuristic_probabilities: np.ndarray,
    blend_weight_model: float,
) -> np.ndarray:
    return blend_weight_model * model_probabilities + (1.0 - blend_weight_model) * heuristic_probabilities


def choose_blend_weight(
    target: pd.Series,
    model_probabilities: np.ndarray,
    heuristic_probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    heuristic_metrics = evaluate_probabilities(target, heuristic_probabilities)
    heuristic_accuracy = heuristic_metrics["accuracy"]

    best_alpha = 1.0
    best_metrics = evaluate_probabilities(target, model_probabilities)
    best_priority = (best_metrics["accuracy"] >= heuristic_accuracy, -best_metrics["log_loss"])

    for alpha in [round(step / 20.0, 2) for step in range(1, 20)]:
        blended = blend_probabilities(model_probabilities, heuristic_probabilities, alpha)
        metrics = evaluate_probabilities(target, blended)
        priority = (metrics["accuracy"] >= heuristic_accuracy, -metrics["log_loss"])
        if priority > best_priority or (
            priority == best_priority and metrics["brier_score"] < best_metrics["brier_score"]
        ):
            best_alpha = alpha
            best_metrics = metrics
            best_priority = priority

    return best_alpha, best_metrics


def train_moneyline_model(
    dataset_path: Path = DATASET_PATH,
    *,
    seasons: list[int] | None = None,
    rebuild_dataset: bool = False,
) -> dict[str, Any]:
    if rebuild_dataset or not dataset_path.exists():
        target_seasons = seasons or [2023, 2024, 2025]
        build_historical_dataset(target_seasons, output_path=dataset_path)

    frame = pd.read_csv(dataset_path, parse_dates=["game_date"])
    frame = select_training_rows(frame)

    train_frame, validation_frame = chronological_split(frame)

    train_x = select_feature_frame(train_frame)
    validation_x = select_feature_frame(validation_frame)
    train_y = train_frame["home_win"].astype(int)
    validation_y = validation_frame["home_win"].astype(int)

    pipeline = build_pipeline()
    pipeline.fit(train_x, train_y)

    validation_probabilities = pipeline.predict_proba(validation_x)[:, 1]
    heuristic_probabilities = validation_frame["heuristic_home_win_prob"].to_numpy()

    raw_model_metrics = evaluate_probabilities(validation_y, validation_probabilities)
    heuristic_metrics = evaluate_probabilities(validation_y, heuristic_probabilities)
    blend_weight_model, hybrid_metrics = choose_blend_weight(
        validation_y,
        validation_probabilities,
        heuristic_probabilities,
    )

    metadata = {
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "training_end_date": str(train_frame["game_date"].max().date()),
        "validation_start_date": str(validation_frame["game_date"].min().date()),
        "feature_columns": feature_columns(),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "raw_model_metrics": raw_model_metrics,
        "model_metrics": hybrid_metrics,
        "heuristic_metrics": heuristic_metrics,
        "blend_weight_model": blend_weight_model,
        "seasons": sorted(frame["season"].dropna().astype(int).unique().tolist()),
    }

    ensure_artifact_dir()
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"pipeline": pipeline, "metadata": metadata}


def load_moneyline_model() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Missing MLB moneyline model artifact. Run "
            "`venv/bin/python train_moneyline_model.py` inside MLBPredictionModel first."
        )
    return joblib.load(MODEL_PATH)


def predict_home_win_probability(frame: pd.DataFrame) -> pd.DataFrame:
    artifact = load_moneyline_model()
    pipeline: Pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]
    prepared = ensure_feature_frame(frame)
    features = select_feature_frame(prepared)
    raw_model = pipeline.predict_proba(features)[:, 1]
    heuristic = prepared["heuristic_home_win_prob"].to_numpy()
    blend_weight_model = float(metadata.get("blend_weight_model", 1.0))

    out = prepared.copy()
    out["raw_model_home_win_probability"] = raw_model
    out["heuristic_home_win_probability"] = heuristic
    out["raw_home_win_probability"] = blend_probabilities(raw_model, heuristic, blend_weight_model)
    return out
