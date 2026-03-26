from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "mlb_predictions.csv"

LOG_FIELDS = [
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "predicted_winner",
    "predicted_side_is_home",
    "raw_win_probability",
    "calibrated_win_probability",
    "raw_home_win_probability",
    "calibrated_home_win_probability",
    "predicted_total",
    "away_starter_name",
    "home_starter_name",
    "away_starter_era_input",
    "home_starter_era_input",
    "away_starter_fip_input",
    "home_starter_fip_input",
    "temperature_f",
    "wind_speed_mph",
    "wind_direction",
    "park_factor_runs",
    "calibration_mode",
    "timestamp_utc",
    "realized_outcome",
    "home_score",
    "away_score",
    "realized_total_runs",
]


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def append_prediction_rows(rows: Iterable[dict[str, object]]) -> Path:
    ensure_log_dir()
    file_exists = PREDICTION_LOG_PATH.exists()
    with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in LOG_FIELDS}
            writer.writerow(payload)
    return PREDICTION_LOG_PATH


def build_prediction_log_rows(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        calibrated_home = float(prediction.get("calibrated_home_win_probability", prediction["raw_home_win_probability"]))
        raw_home = float(prediction["raw_home_win_probability"])
        predicted_is_home = calibrated_home >= 0.5
        rows.append(
            {
                "game_id": prediction.get("game_pk", ""),
                "game_date": prediction.get("game_date", ""),
                "away_team": prediction.get("away_team", ""),
                "home_team": prediction.get("home_team", ""),
                "predicted_winner": prediction.get("home_team") if predicted_is_home else prediction.get("away_team"),
                "predicted_side_is_home": int(predicted_is_home),
                "raw_win_probability": raw_home if predicted_is_home else 1.0 - raw_home,
                "calibrated_win_probability": calibrated_home if predicted_is_home else 1.0 - calibrated_home,
                "raw_home_win_probability": raw_home,
                "calibrated_home_win_probability": calibrated_home,
                "predicted_total": prediction.get("predicted_total_runs", ""),
                "away_starter_name": prediction.get("away_starter_name", ""),
                "home_starter_name": prediction.get("home_starter_name", ""),
                "away_starter_era_input": prediction.get("away_starter_era_shrunk", prediction.get("away_starter_era", "")),
                "home_starter_era_input": prediction.get("home_starter_era_shrunk", prediction.get("home_starter_era", "")),
                "away_starter_fip_input": prediction.get("away_starter_fip_shrunk", prediction.get("away_starter_fip", "")),
                "home_starter_fip_input": prediction.get("home_starter_fip_shrunk", prediction.get("home_starter_fip", "")),
                "temperature_f": prediction.get("temperature_f", ""),
                "wind_speed_mph": prediction.get("wind_speed_mph", ""),
                "wind_direction": prediction.get("wind_direction", ""),
                "park_factor_runs": prediction.get("park_factor_runs", ""),
                "calibration_mode": prediction.get("calibration_mode", ""),
                "timestamp_utc": timestamp,
                "realized_outcome": "",
                "home_score": "",
                "away_score": "",
                "realized_total_runs": "",
            }
        )
    return rows
