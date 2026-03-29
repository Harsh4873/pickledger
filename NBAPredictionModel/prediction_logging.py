import csv
from datetime import datetime, UTC
from pathlib import Path

from data_models import GameContext

PREDICTION_LOG_FIELDS = [
    "game_id",
    "game_date",
    "away_team",
    "home_team",
    "predicted_spread",
    "raw_probability",
    "extremized_probability",
    "calibrated_probability",
    "calibration_flag",
    "is_home",
    "rest_days_home",
    "rest_days_away",
    "back_to_back_home",
    "back_to_back_away",
    "timestamp",
    "actual_home_win",
]


def _default_log_path() -> Path:
    return Path(__file__).resolve().parent / "logs" / "predictions.csv"


def _normalize_row(row: dict) -> dict:
    normalized = {field: row.get(field, "") for field in PREDICTION_LOG_FIELDS}
    normalized["game_id"] = normalized["game_id"] or (
        f"{normalized['game_date']}:{normalized['away_team']}@{normalized['home_team']}"
        if normalized["game_date"] and normalized["away_team"] and normalized["home_team"]
        else ""
    )
    return normalized


def ensure_prediction_log_schema(log_path: Path | None = None) -> Path:
    target_path = log_path or _default_log_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        return target_path

    with target_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fields = reader.fieldnames or []
        rows = [_normalize_row(row) for row in reader]

    if existing_fields == PREDICTION_LOG_FIELDS:
        return target_path

    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target_path


def load_prediction_rows(log_path: Path | None = None) -> tuple[Path, list[dict]]:
    target_path = ensure_prediction_log_schema(log_path)
    if not target_path.exists():
        return target_path, []

    with target_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return target_path, [_normalize_row(row) for row in reader]


def write_prediction_rows(rows: list[dict], log_path: Path | None = None) -> Path:
    target_path = (log_path or _default_log_path()).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(_normalize_row(row) for row in rows)
    return target_path


def append_prediction_log(
    game_ctx: GameContext,
    predicted_spread: float,
    raw_probability: float,
    extremized_probability: float,
    calibrated_probability: float,
    calibration_flag: str = "",
    timestamp: str | None = None,
) -> Path:
    log_path = ensure_prediction_log_schema()

    game_id = game_ctx.game_id or f"{game_ctx.date}:{game_ctx.away_team.name}@{game_ctx.home_team.name}"
    row = {
        "game_id": game_id,
        "game_date": game_ctx.date,
        "away_team": game_ctx.away_team.name,
        "home_team": game_ctx.home_team.name,
        "predicted_spread": f"{predicted_spread:.4f}",
        "raw_probability": f"{raw_probability:.6f}",
        "extremized_probability": f"{extremized_probability:.6f}",
        "calibrated_probability": f"{calibrated_probability:.6f}",
        "calibration_flag": calibration_flag,
        "is_home": int(game_ctx.home_team.is_home),
        "rest_days_home": f"{game_ctx.home_team.team_stats.rest_days:.2f}",
        "rest_days_away": f"{game_ctx.away_team.team_stats.rest_days:.2f}",
        "back_to_back_home": int(game_ctx.home_team.team_stats.back_to_back_flag),
        "back_to_back_away": int(game_ctx.away_team.team_stats.back_to_back_flag),
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
        "actual_home_win": "",
    }

    write_header = not log_path.exists()
    with log_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return log_path
