import csv
from datetime import datetime, UTC
from pathlib import Path

from data_models import GameContext

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREDICTION_LOG_PATH = REPO_ROOT / "data" / "nba" / "predictions.csv"
LEGACY_PREDICTION_LOG_PATH = Path(__file__).resolve().parent / "logs" / "predictions.csv"

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
    "home_score",
    "away_score",
    "log_source",
    "outcome_source",
]

PREDICTION_VALUE_FIELDS = (
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
    "log_source",
)

OUTCOME_FIELDS = ("actual_home_win", "home_score", "away_score", "outcome_source")


def default_prediction_log_path() -> Path:
    return DEFAULT_PREDICTION_LOG_PATH


def _normalize_team_token(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def prediction_row_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("game_date", "")).strip(),
        _normalize_team_token(row.get("away_team", "")),
        _normalize_team_token(row.get("home_team", "")),
    )


def _normalize_row(row: dict) -> dict:
    normalized = {field: "" if row.get(field) is None else str(row.get(field, "")) for field in PREDICTION_LOG_FIELDS}
    normalized["game_id"] = normalized["game_id"] or (
        f"{normalized['game_date']}:{normalized['away_team']}@{normalized['home_team']}"
        if normalized["game_date"] and normalized["away_team"] and normalized["home_team"]
        else ""
    )
    return normalized


def _has_outcome(row: dict) -> bool:
    return str(row.get("actual_home_win", "")).strip() != ""


def _is_live_source(row: dict) -> bool:
    return str(row.get("log_source", "")).strip().lower() == "live"


def _row_timestamp(row: dict) -> str:
    return str(row.get("timestamp", "")).strip()


def resolve_prediction_log_path(log_path: Path | None = None) -> Path:
    if log_path is not None:
        return Path(log_path)
    if DEFAULT_PREDICTION_LOG_PATH.exists():
        return DEFAULT_PREDICTION_LOG_PATH
    if LEGACY_PREDICTION_LOG_PATH.exists() and not DEFAULT_PREDICTION_LOG_PATH.exists():
        return LEGACY_PREDICTION_LOG_PATH
    return DEFAULT_PREDICTION_LOG_PATH


def ensure_prediction_log_schema(log_path: Path | None = None) -> Path:
    target_path = resolve_prediction_log_path(log_path)
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
    target_path = resolve_prediction_log_path(log_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (_normalize_row(row) for row in rows),
        key=lambda row: (row.get("game_date", ""), row.get("away_team", ""), row.get("home_team", ""), row.get("game_id", "")),
    )
    with target_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    return target_path


def merge_prediction_rows(existing: list[dict], incoming: list[dict]) -> list[dict]:
    """Union prediction rows without inventing or clobbering graded outcomes."""
    merged: dict[tuple[str, str, str], dict] = {}
    for row in existing:
        normalized = _normalize_row(row)
        key = prediction_row_key(normalized)
        if not any(key):
            continue
        merged[key] = normalized
    for row in incoming:
        normalized = _normalize_row(row)
        key = prediction_row_key(normalized)
        if not any(key):
            continue
        current = merged.get(key)
        if current is None:
            merged[key] = normalized
            continue
        merged[key] = _merge_one_prediction_row(current, normalized)
    return list(merged.values())


def _merge_one_prediction_row(existing: dict, incoming: dict) -> dict:
    combined = dict(existing)
    existing_live = _is_live_source(existing)
    incoming_live = _is_live_source(incoming)
    replace_prediction = False
    if incoming_live and not existing_live:
        replace_prediction = True
    elif existing_live and not incoming_live:
        replace_prediction = False
    elif _row_timestamp(incoming) > _row_timestamp(existing):
        replace_prediction = True
    if replace_prediction:
        for field in PREDICTION_VALUE_FIELDS:
            if str(incoming.get(field, "")).strip() != "":
                combined[field] = incoming[field]
        if incoming.get("game_id"):
            combined["game_id"] = incoming["game_id"]
    if not _has_outcome(existing) and _has_outcome(incoming):
        for field in OUTCOME_FIELDS:
            combined[field] = incoming.get(field, "")
    return _normalize_row(combined)


def apply_prediction_outcome(
    row: dict,
    *,
    actual_home_win: int,
    home_score: object = "",
    away_score: object = "",
    outcome_source: str = "",
) -> bool:
    """Fill a realized home-win flag. Never overwrite a recorded result."""
    if _has_outcome(row):
        return False
    if actual_home_win not in (0, 1):
        raise ValueError("actual_home_win must be 0 or 1 from a real final score")
    row["actual_home_win"] = str(int(actual_home_win))
    if home_score != "" and home_score is not None:
        row["home_score"] = str(home_score)
    if away_score != "" and away_score is not None:
        row["away_score"] = str(away_score)
    if outcome_source:
        row["outcome_source"] = outcome_source
    return True


def upsert_prediction_row(row: dict, log_path: Path | None = None) -> Path:
    target_path, rows = load_prediction_rows(log_path)
    merged = merge_prediction_rows(rows, [_normalize_row(row)])
    return write_prediction_rows(merged, log_path=target_path)


def count_graded_predictions(rows: list[dict] | None = None, log_path: Path | None = None) -> dict[str, int]:
    if rows is None:
        _, rows = load_prediction_rows(log_path)
    total = 0
    graded = 0
    platt_eligible = 0
    for row in rows:
        total += 1
        has_raw = str(row.get("raw_probability", "")).strip() != ""
        if _has_outcome(row):
            graded += 1
            if has_raw:
                platt_eligible += 1
    return {
        "total": total,
        "graded": graded,
        "platt_eligible": platt_eligible,
    }


def append_prediction_log(
    game_ctx: GameContext,
    predicted_spread: float,
    raw_probability: float,
    extremized_probability: float,
    calibrated_probability: float,
    calibration_flag: str = "",
    timestamp: str | None = None,
    log_path: Path | None = None,
    log_source: str = "live",
) -> Path:
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
        "home_score": "",
        "away_score": "",
        "log_source": log_source,
        "outcome_source": "",
    }
    return upsert_prediction_row(row, log_path=log_path)
