from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from feature_engineering import ensure_feature_frame
from historical_data import DATASET_PATH
from model_variants import ARTIFACT_DIR, get_mlb_model_artifacts
from probability_layers import predict_total_runs as heuristic_total_runs


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "mlb_totals_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "mlb_totals_model_metadata.json"
INFERENCE_BLEND_WEIGHT_MODEL = 0.65

TOTALS_NUMERIC_FEATURES = [
    "park_factor_runs",
    "temperature_f",
    "wind_speed_mph",
    "is_dome",
    "home_team_win_pct_shrunk",
    "away_team_win_pct_shrunk",
    "home_form_14d_win_pct_shrunk",
    "away_form_14d_win_pct_shrunk",
    "home_starter_era_shrunk",
    "away_starter_era_shrunk",
    "home_starter_fip_shrunk",
    "away_starter_fip_shrunk",
    "home_starter_reliability",
    "away_starter_reliability",
    "home_bullpen_pitches_1d",
    "away_bullpen_pitches_1d",
    "home_bullpen_pitches_3d",
    "away_bullpen_pitches_3d",
    "home_bullpen_era_30d",
    "away_bullpen_era_30d",
    "home_lineup_ops_proxy_shrunk",
    "away_lineup_ops_proxy_shrunk",
    "home_lineup_obp_proxy_shrunk",
    "away_lineup_obp_proxy_shrunk",
    "home_lineup_slg_proxy_shrunk",
    "away_lineup_slg_proxy_shrunk",
    "heuristic_total_runs",
    # Vegas/market total line is the single strongest available predictor of
    # actual totals. Including it lets the model learn how much to *adjust*
    # around the market consensus using the other signals above, rather than
    # predicting from scratch.
    "market_total_line",
]

TOTALS_CATEGORICAL_FEATURES = [
    "wind_direction",
    "home_starter_hand",
    "away_starter_hand",
]


def ensure_artifact_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def chronological_split(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.sort_values("game_date").reset_index(drop=True)
    split_idx = max(1, int(len(frame) * (1.0 - validation_fraction)))
    split_idx = min(split_idx, len(frame) - 1)
    return frame.iloc[:split_idx].copy(), frame.iloc[split_idx:].copy()


def add_totals_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_feature_frame(frame)
    if "heuristic_total_runs" not in frame.columns:
        frame["heuristic_total_runs"] = frame.apply(lambda row: heuristic_total_runs(row.to_dict()), axis=1)
    return frame


def build_recency_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    years = frame["game_date"].dt.year
    weights = np.ones(len(frame), dtype=float)
    # More recent seasons carry heavier weight because team composition,
    # pitching usage patterns, and run-scoring environment change year to year.
    weights[years == 2023] = 1.0
    weights[years == 2024] = 1.0
    weights[years == 2025] = 1.5
    weights[years == 2026] = 2.0
    return weights


def select_totals_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_totals_features(frame).copy()
    for column in TOTALS_NUMERIC_FEATURES + TOTALS_CATEGORICAL_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
    return frame[TOTALS_NUMERIC_FEATURES + TOTALS_CATEGORICAL_FEATURES].copy()


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), TOTALS_NUMERIC_FEATURES),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                TOTALS_CATEGORICAL_FEATURES,
            ),
        ]
    )
    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=350,
        min_samples_leaf=35,
        l2_regularization=0.1,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_totals(target: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(target, predictions)))
    mae = mean_absolute_error(target, predictions)
    return {
        "rmse": rmse,
        "mae": float(mae),
    }


def blend_totals(
    model_predictions: np.ndarray,
    heuristic_predictions: np.ndarray,
    blend_weight_model: float,
) -> np.ndarray:
    return blend_weight_model * model_predictions + (1.0 - blend_weight_model) * heuristic_predictions


def choose_blend_weight(
    target: pd.Series,
    model_predictions: np.ndarray,
    heuristic_predictions: np.ndarray,
) -> tuple[float, dict[str, float]]:
    heuristic_metrics = evaluate_totals(target, heuristic_predictions)
    best_alpha = 1.0
    best_metrics = evaluate_totals(target, model_predictions)
    heuristic_pair = (heuristic_metrics["mae"], heuristic_metrics["rmse"])
    best_priority = (
        best_metrics["mae"] <= heuristic_pair[0] and best_metrics["rmse"] <= heuristic_pair[1],
        -best_metrics["mae"],
        -best_metrics["rmse"],
    )

    for alpha in [round(step / 20.0, 2) for step in range(1, 20)]:
        blended = blend_totals(model_predictions, heuristic_predictions, alpha)
        metrics = evaluate_totals(target, blended)
        priority = (
            metrics["mae"] <= heuristic_pair[0] and metrics["rmse"] <= heuristic_pair[1],
            -metrics["mae"],
            -metrics["rmse"],
        )
        if priority > best_priority:
            best_alpha = alpha
            best_metrics = metrics
            best_priority = priority

    return best_alpha, best_metrics


def train_totals_model(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path, parse_dates=["game_date"])
    frame = add_totals_features(frame)
    frame = frame.dropna(subset=["total_runs", "game_date"]).sort_values("game_date").reset_index(drop=True)

    train_frame, validation_frame = chronological_split(frame)
    train_x = select_totals_feature_frame(train_frame)
    validation_x = select_totals_feature_frame(validation_frame)
    train_y = train_frame["total_runs"].astype(float)
    validation_y = validation_frame["total_runs"].astype(float)

    train_sample_weights = build_recency_sample_weights(train_frame)

    pipeline = build_pipeline()
    pipeline.fit(train_x, train_y, model__sample_weight=train_sample_weights)

    validation_predictions = pipeline.predict(validation_x)
    heuristic_predictions = validation_frame["heuristic_total_runs"].to_numpy()

    raw_model_metrics = evaluate_totals(validation_y, validation_predictions)
    heuristic_metrics = evaluate_totals(validation_y, heuristic_predictions)
    blend_weight_model, blended_metrics = choose_blend_weight(
        validation_y,
        validation_predictions,
        heuristic_predictions,
    )

    metadata = {
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "training_end_date": str(train_frame["game_date"].max().date()),
        "validation_start_date": str(validation_frame["game_date"].min().date()),
        "training_years": sorted(train_frame["game_date"].dt.year.unique().astype(int).tolist()),
        "validation_years": sorted(validation_frame["game_date"].dt.year.unique().astype(int).tolist()),
        "sample_weighting": {
            "2023": 1.0,
            "2024": 1.0,
            "2025": 1.5,
            "2026": 2.0,
        },
        "feature_columns": TOTALS_NUMERIC_FEATURES + TOTALS_CATEGORICAL_FEATURES,
        "raw_model_metrics": raw_model_metrics,
        "model_metrics": blended_metrics,
        "heuristic_metrics": heuristic_metrics,
        "blend_weight_model": blend_weight_model,
    }

    ensure_artifact_dir()
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"pipeline": pipeline, "metadata": metadata}


def load_totals_model(variant: str | None = None) -> dict[str, Any]:
    model_path = get_mlb_model_artifacts(variant)["totals"]
    if not model_path.exists():
        variant_note = f" for variant={variant!r}" if variant else ""
        raise FileNotFoundError(
            "Missing MLB totals model artifact"
            f"{variant_note}. Run "
            "`venv/bin/python train_totals_model.py` inside MLBPredictionModel first."
        )
    return joblib.load(model_path)


def predict_totals(frame: pd.DataFrame, variant: str | None = None) -> pd.DataFrame:
    artifact = load_totals_model(variant=variant)
    pipeline: Pipeline = artifact["pipeline"]
    metadata = artifact["metadata"]
    prepared = add_totals_features(frame)
    features = select_totals_feature_frame(prepared)
    out = prepared.copy()
    raw_model = pipeline.predict(features)
    heuristic = prepared["heuristic_total_runs"].to_numpy()
    blend_weight_model = max(
        INFERENCE_BLEND_WEIGHT_MODEL,
        float(metadata.get("blend_weight_model", INFERENCE_BLEND_WEIGHT_MODEL)),
    )
    out["raw_model_total_runs"] = raw_model
    out["predicted_total_runs"] = blend_totals(raw_model, heuristic, blend_weight_model)
    return out
