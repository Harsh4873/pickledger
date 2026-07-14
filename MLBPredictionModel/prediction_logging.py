from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
PREDICTION_LOG_PATH = LOG_DIR / "mlb_predictions.csv"
TOTALS_BASELINE = 8.5
MIN_TOTALS_CALIBRATION_OUTCOMES = 50
UNCALIBRATED_TOTALS_CONFIDENCE_CAP = 0.80
TOTALS_CONFIDENCE_SIGMOID_SCALE = 1.25

LOG_FIELDS = [
    "game_id",
    "model_name",
    "model_variant",
    "run_date",
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
    "totals_line",
    "totals_confidence",
    "away_starter_name",
    "home_starter_name",
    "away_starter_era_input",
    "home_starter_era_input",
    "away_starter_fip_input",
    "home_starter_fip_input",
    "away_starter_last_5_starts_era",
    "home_starter_last_5_starts_era",
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


def realized_totals_outcome_count() -> int:
    return sum(
        1
        for row in _read_existing_log_rows()
        if str(row.get("realized_total_runs", "")).strip()
    )


def compute_totals_confidence(predicted_total: object, totals_line: object) -> float | None:
    try:
        predicted = float(predicted_total)
    except (TypeError, ValueError):
        return None

    distance_from_baseline = abs(predicted - TOTALS_BASELINE)
    confidence = 1.0 / (1.0 + math.exp(-TOTALS_CONFIDENCE_SIGMOID_SCALE * distance_from_baseline))
    if realized_totals_outcome_count() < MIN_TOTALS_CALIBRATION_OUTCOMES:
        confidence = min(confidence, UNCALIBRATED_TOTALS_CONFIDENCE_CAP)
    return round(confidence, 4)


def _read_existing_log_rows() -> list[dict[str, str]]:
    if not PREDICTION_LOG_PATH.exists():
        return []
    with PREDICTION_LOG_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _normalize_log_row(row: dict[str, object]) -> dict[str, object]:
    payload = {field: row.get(field, "") for field in LOG_FIELDS}
    if not payload["run_date"]:
        timestamp_utc = str(row.get("timestamp_utc", "")).strip()
        payload["run_date"] = timestamp_utc[:10] if timestamp_utc else ""
    return payload


def _ensure_log_schema() -> None:
    if not PREDICTION_LOG_PATH.exists():
        return

    with PREDICTION_LOG_PATH.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])

    if header == LOG_FIELDS:
        return

    existing_rows = [_normalize_log_row(row) for row in _read_existing_log_rows()]
    with PREDICTION_LOG_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)


def existing_prediction_keys() -> set[tuple[str, str, str]]:
    ensure_log_dir()
    _ensure_log_schema()
    keys: set[tuple[str, str, str]] = set()
    for row in _read_existing_log_rows():
        normalized = _normalize_log_row(row)
        game_id = str(normalized.get("game_id", "")).strip()
        model_variant = str(normalized.get("model_variant", "")).strip().lower()
        run_date = str(normalized.get("run_date", "")).strip()
        if game_id and run_date and model_variant:
            keys.add((game_id, run_date, model_variant))
    return keys


def append_prediction_rows(rows: Iterable[dict[str, object]]) -> Path:
    ensure_log_dir()
    _ensure_log_schema()
    file_exists = PREDICTION_LOG_PATH.exists()
    with PREDICTION_LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            payload = _normalize_log_row(row)
            writer.writerow(payload)
    return PREDICTION_LOG_PATH


def build_prediction_log_rows(
    predictions: list[dict[str, object]],
    *,
    model_name: str = "",
    model_variant: str = "",
) -> list[dict[str, object]]:
    timestamp = datetime.now(timezone.utc).isoformat()
    run_date = timestamp[:10]
    rows: list[dict[str, object]] = []
    for prediction in predictions:
        calibrated_home = float(prediction.get("calibrated_home_win_probability", prediction["raw_home_win_probability"]))
        raw_home = float(prediction["raw_home_win_probability"])
        predicted_is_home = calibrated_home >= 0.5
        totals_line = prediction.get("totals_line", "")
        rows.append(
            {
                "game_id": prediction.get("game_pk", ""),
                "model_name": prediction.get("model_name", model_name),
                "model_variant": prediction.get("model_variant", model_variant),
                "run_date": run_date,
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
                "totals_line": totals_line,
                "totals_confidence": prediction.get(
                    "totals_confidence",
                    compute_totals_confidence(prediction.get("predicted_total_runs", ""), totals_line),
                ),
                "away_starter_name": prediction.get("away_starter_name", ""),
                "home_starter_name": prediction.get("home_starter_name", ""),
                "away_starter_era_input": prediction.get("away_starter_era_shrunk", prediction.get("away_starter_era", "")),
                "home_starter_era_input": prediction.get("home_starter_era_shrunk", prediction.get("home_starter_era", "")),
                "away_starter_fip_input": prediction.get("away_starter_fip_shrunk", prediction.get("away_starter_fip", "")),
                "home_starter_fip_input": prediction.get("home_starter_fip_shrunk", prediction.get("home_starter_fip", "")),
                "away_starter_last_5_starts_era": prediction.get("away_starter_last_5_starts_era", ""),
                "home_starter_last_5_starts_era": prediction.get("home_starter_last_5_starts_era", ""),
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
