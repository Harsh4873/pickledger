import csv
from datetime import datetime, UTC
from pathlib import Path

from data_models import GameContext

PREDICTION_LOG_FIELDS = [
    "game_id",
    "predicted_spread",
    "raw_probability",
    "calibrated_probability",
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


def append_prediction_log(
    game_ctx: GameContext,
    predicted_spread: float,
    raw_probability: float,
    calibrated_probability: float,
    timestamp: str | None = None,
) -> Path:
    log_path = _default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    game_id = game_ctx.game_id or f"{game_ctx.date}:{game_ctx.away_team.name}@{game_ctx.home_team.name}"
    row = {
        "game_id": game_id,
        "predicted_spread": f"{predicted_spread:.4f}",
        "raw_probability": f"{raw_probability:.6f}",
        "calibrated_probability": f"{calibrated_probability:.6f}",
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
