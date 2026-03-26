from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from feature_engineering import select_training_rows
from moneyline_model import (
    ARTIFACT_DIR,
    blend_probabilities,
    chronological_split,
    evaluate_probabilities,
    load_moneyline_model,
    predict_home_win_probability,
)


CALIBRATION_PATH = ARTIFACT_DIR / "mlb_probability_calibration.joblib"
CALIBRATION_METADATA_PATH = ARTIFACT_DIR / "mlb_probability_calibration_metadata.json"
MIN_CALIBRATION_ROWS = 400
CONSERVATIVE_SCALE = 0.82
CONSERVATIVE_FLOOR = 0.08
CONSERVATIVE_CEILING = 0.92


def _conservative_cap(probabilities: np.ndarray) -> np.ndarray:
    centered = 0.5 + (probabilities - 0.5) * CONSERVATIVE_SCALE
    return np.clip(centered, CONSERVATIVE_FLOOR, CONSERVATIVE_CEILING)


def train_moneyline_calibration(dataset_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path, parse_dates=["game_date"])
    frame = select_training_rows(frame)
    _, validation_frame = chronological_split(frame)

    if len(validation_frame) < MIN_CALIBRATION_ROWS:
        metadata = {
            "mode": "conservative_cap",
            "validation_rows": int(len(validation_frame)),
            "scale": CONSERVATIVE_SCALE,
            "floor": CONSERVATIVE_FLOOR,
            "ceiling": CONSERVATIVE_CEILING,
        }
        joblib.dump({"mode": "conservative_cap", "metadata": metadata}, CALIBRATION_PATH)
        CALIBRATION_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"metadata": metadata}

    predictions = predict_home_win_probability(validation_frame)
    raw_probabilities = predictions["raw_home_win_probability"].to_numpy()
    outcomes = validation_frame["home_win"].astype(int).to_numpy()

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_probabilities, outcomes)
    calibrated = calibrator.predict(raw_probabilities)
    calibrated = np.clip(calibrated, CONSERVATIVE_FLOOR, CONSERVATIVE_CEILING)

    metadata = {
        "mode": "fit",
        "method": "isotonic_regression",
        "validation_rows": int(len(validation_frame)),
        "pre_calibration_metrics": evaluate_probabilities(validation_frame["home_win"], raw_probabilities),
        "post_calibration_metrics": evaluate_probabilities(validation_frame["home_win"], calibrated),
        "floor": CONSERVATIVE_FLOOR,
        "ceiling": CONSERVATIVE_CEILING,
    }
    joblib.dump({"mode": "fit", "calibrator": calibrator, "metadata": metadata}, CALIBRATION_PATH)
    CALIBRATION_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata}


def load_calibration_artifact() -> dict[str, Any] | None:
    if not CALIBRATION_PATH.exists():
        return None
    return joblib.load(CALIBRATION_PATH)


def apply_moneyline_calibration(frame: pd.DataFrame) -> pd.DataFrame:
    artifact = load_calibration_artifact()
    out = frame.copy()
    raw_probabilities = out["raw_home_win_probability"].to_numpy()

    if artifact is None or artifact.get("mode") == "conservative_cap":
        out["calibrated_home_win_probability"] = _conservative_cap(raw_probabilities)
        out["calibration_mode"] = "conservative_cap"
        return out

    calibrator: IsotonicRegression = artifact["calibrator"]
    calibrated = calibrator.predict(raw_probabilities)
    out["calibrated_home_win_probability"] = np.clip(calibrated, CONSERVATIVE_FLOOR, CONSERVATIVE_CEILING)
    out["calibration_mode"] = "fit"
    return out
