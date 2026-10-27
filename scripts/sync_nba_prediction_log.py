#!/usr/bin/env python3
"""Persist, backfill, and grade the NBA team-model prediction log.

The NBA Platt wrapper reads ``data/nba/predictions.csv``. GitHub Actions
runners only see that file if it is committed, so this script is invoked
from the model-cache refresh and auto-grade paths.

Outcomes are copied from ESPN final scores or from already-graded NBA cache
rows. Results are never invented.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
NBA_MODEL_DIR = REPO_ROOT / "NBAPredictionModel"
# 2026-27 regular season openers. Earlier cache slates are preseason/test dates.
NBA_REGULAR_SEASON_START = "2026-10-21"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(NBA_MODEL_DIR))

from prediction_logging import (  # noqa: E402
    DEFAULT_PREDICTION_LOG_PATH,
    apply_prediction_outcome,
    count_graded_predictions,
    load_prediction_rows,
    merge_prediction_rows,
    prediction_row_key,
    write_prediction_rows,
)


FetchScoreboard = Callable[[str, str, str], dict[str, Any] | None]


def _normalize_team_token(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def teams_match(left: object, right: object) -> bool:
    left_key = _normalize_team_token(left)
    right_key = _normalize_team_token(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if left_key in right_key or right_key in left_key:
        return len(min(left_key, right_key, key=len)) >= 4
    left_tail = re.sub(r"[^a-z0-9]+", "", str(left or "").split()[-1].lower()) if str(left or "").split() else ""
    right_tail = re.sub(r"[^a-z0-9]+", "", str(right or "").split()[-1].lower()) if str(right or "").split() else ""
    return bool(left_tail and left_tail == right_tail and len(left_tail) >= 3)


def _is_nba_side_pick(pick: dict[str, Any]) -> bool:
    if not isinstance(pick, dict):
        return False
    if str(pick.get("sport") or "").strip().upper() != "NBA":
        return False
    if str(pick.get("market_type") or "").strip().lower() == "totals":
        return False
    text = str(pick.get("pick") or "").strip()
    if re.match(r"^(over|under)\b", text, flags=re.IGNORECASE):
        return False
    if not str(pick.get("home_team") or "").strip() or not str(pick.get("away_team") or "").strip():
        return False
    return bool(str(pick.get("team") or "").strip())


def _coerce_probability(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number < 0.0 or number > 1.0:
        return None
    return number


def published_home_win_probability(pick: dict[str, Any]) -> float | None:
    """Recover home-win probability from a published NBA side pick.

    Live model stdout reports the picked side (>= 0.5). A historical parser
    bug then flipped away picks, so those published values are already the
    home-win probability (typically 0.25 after the uncalibrated cap).
    """
    probability = None
    for field in ("probability", "prob", "calibrated_probability"):
        probability = _coerce_probability(pick.get(field))
        if probability is not None:
            break
    if probability is None:
        return None
    team = str(pick.get("team") or "").strip()
    home = str(pick.get("home_team") or "").strip()
    if teams_match(team, home):
        return probability
    if probability < 0.5:
        return probability
    return 1.0 - probability


def home_predicted_spread(pick: dict[str, Any]) -> float | None:
    try:
        margin = abs(float(pick.get("predicted_spread")))
    except (TypeError, ValueError):
        return None
    team = str(pick.get("team") or "").strip()
    home = str(pick.get("home_team") or "").strip()
    return margin if teams_match(team, home) else -margin


def outcome_from_graded_pick(pick: dict[str, Any]) -> int | None:
    result = str(pick.get("result") or "").strip().lower()
    if result not in {"win", "loss"}:
        return None
    team = str(pick.get("team") or "").strip()
    home = str(pick.get("home_team") or "").strip()
    picked_home = teams_match(team, home)
    if result == "win":
        return 1 if picked_home else 0
    return 0 if picked_home else 1


def _pick_timestamp(pick: dict[str, Any], fallback: str = "") -> str:
    timing = pick.get("certification_timing")
    if isinstance(timing, dict):
        published = str(timing.get("published_at") or "").strip()
        if published:
            return published
    for field in ("published_at", "timestamp", "generatedAt", "updatedAt"):
        value = str(pick.get(field) or "").strip()
        if value:
            return value
    return fallback


def prediction_row_from_cache_pick(pick: dict[str, Any], *, slate_date: str, timestamp: str = "") -> dict[str, str] | None:
    if not _is_nba_side_pick(pick):
        return None
    home_win = published_home_win_probability(pick)
    if home_win is None:
        return None
    away = str(pick.get("away_team") or "").strip()
    home = str(pick.get("home_team") or "").strip()
    game_date = str(pick.get("date") or pick.get("game_date") or slate_date or "").strip()
    if not game_date or not away or not home:
        return None
    if game_date < NBA_REGULAR_SEASON_START:
        return None
    spread = home_predicted_spread(pick)
    spread_text = f"{spread:.4f}" if spread is not None else ""
    probability_text = f"{home_win:.6f}"
    return {
        "game_id": str(pick.get("game_id") or f"{game_date}:{away}@{home}"),
        "game_date": game_date,
        "away_team": away,
        "home_team": home,
        "predicted_spread": spread_text,
        # Published cache probabilities are already capped/possibly flipped.
        # Leave raw empty so Platt waits for live pre-calibration values.
        "raw_probability": "",
        "extremized_probability": "",
        "calibrated_probability": probability_text,
        "calibration_flag": "[CACHE_BACKFILL]",
        "is_home": "1",
        "rest_days_home": "",
        "rest_days_away": "",
        "back_to_back_home": "",
        "back_to_back_away": "",
        "timestamp": _pick_timestamp(pick, timestamp),
        "actual_home_win": "",
        "home_score": "",
        "away_score": "",
        "log_source": "model_cache_backfill",
        "outcome_source": "",
    }


def iter_nba_cache_picks(repo_root: Path = REPO_ROOT) -> Iterable[tuple[str, dict[str, Any]]]:
    cache_dir = repo_root / "data" / "model_cache"
    for path in sorted(cache_dir.glob("20??-??-??.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        slate_date = str(payload.get("date") or path.stem).strip()
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        bucket = models.get("nba") if isinstance(models.get("nba"), dict) else {}
        picks = bucket.get("picks") if isinstance(bucket.get("picks"), list) else []
        timestamp = str(payload.get("generatedAt") or payload.get("updatedAt") or "")
        for pick in picks:
            if isinstance(pick, dict):
                yield slate_date, {**pick, "_cache_timestamp": timestamp}


def backfill_rows_from_model_cache(repo_root: Path = REPO_ROOT) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for slate_date, pick in iter_nba_cache_picks(repo_root):
        row = prediction_row_from_cache_pick(
            pick,
            slate_date=slate_date,
            timestamp=str(pick.get("_cache_timestamp") or ""),
        )
        if row is None:
            continue
        key = prediction_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _competitor_names(competitor: dict[str, Any]) -> list[str]:
    raw = competitor.get("raw") if isinstance(competitor.get("raw"), dict) else competitor
    team = raw.get("team") if isinstance(raw.get("team"), dict) else {}
    names = [
        team.get("displayName"),
        team.get("shortDisplayName"),
        team.get("name"),
        team.get("abbreviation"),
        competitor.get("name"),
    ]
    return [str(name).strip() for name in names if str(name or "").strip()]


def _side_competitor(game: dict[str, Any], side: str) -> dict[str, Any] | None:
    for competitor in game.get("competitors") or []:
        if not isinstance(competitor, dict):
            continue
        if str(competitor.get("homeAway") or "").strip().lower() == side:
            return competitor
    return None


def grade_rows_from_espn(
    rows: list[dict],
    *,
    fetch_scoreboard: FetchScoreboard | None = None,
) -> int:
    pending_dates = sorted({
        str(row.get("game_date") or "").strip()
        for row in rows
        if str(row.get("game_date") or "").strip() and str(row.get("actual_home_win") or "").strip() == ""
    })
    if not pending_dates:
        return 0

    if fetch_scoreboard is None:
        from pickgrader_server import fetch_scoreboard as _fetch_scoreboard

        fetch_scoreboard = _fetch_scoreboard

    from pickgrader_server import get_games

    updated = 0
    games_by_date: dict[str, list[dict[str, Any]]] = {}
    for date_iso in pending_dates:
        try:
            yyyymmdd = datetime.strptime(date_iso, "%Y-%m-%d").strftime("%Y%m%d")
        except ValueError:
            continue
        scoreboard = fetch_scoreboard("basketball", "nba", yyyymmdd)
        if not isinstance(scoreboard, dict):
            continue
        games_by_date[date_iso] = get_games(scoreboard, completed_only=True)

    for row in rows:
        if str(row.get("actual_home_win") or "").strip():
            continue
        games = games_by_date.get(str(row.get("game_date") or "").strip()) or []
        for game in games:
            if not game.get("completed"):
                continue
            home = _side_competitor(game, "home")
            away = _side_competitor(game, "away")
            if home is None or away is None:
                continue
            if not any(teams_match(row.get("home_team"), name) for name in _competitor_names(home)):
                continue
            if not any(teams_match(row.get("away_team"), name) for name in _competitor_names(away)):
                continue
            try:
                home_score = int(home.get("score"))
                away_score = int(away.get("score"))
            except (TypeError, ValueError):
                continue
            if home_score == away_score:
                continue
            if apply_prediction_outcome(
                row,
                actual_home_win=1 if home_score > away_score else 0,
                home_score=home_score,
                away_score=away_score,
                outcome_source="espn_scoreboard",
            ):
                updated += 1
            break
    return updated


def grade_rows_from_cache_results(rows: list[dict], repo_root: Path = REPO_ROOT) -> int:
    outcomes: dict[tuple[str, str, str], tuple[int, dict[str, Any]]] = {}
    for slate_date, pick in iter_nba_cache_picks(repo_root):
        if not _is_nba_side_pick(pick):
            continue
        actual = outcome_from_graded_pick(pick)
        if actual is None:
            continue
        key = (
            str(pick.get("date") or pick.get("game_date") or slate_date).strip(),
            _normalize_team_token(pick.get("away_team")),
            _normalize_team_token(pick.get("home_team")),
        )
        outcomes[key] = (actual, pick)

    updated = 0
    for row in rows:
        actual_and_pick = outcomes.get(prediction_row_key(row))
        if actual_and_pick is None:
            continue
        actual, _pick = actual_and_pick
        if apply_prediction_outcome(row, actual_home_win=actual, outcome_source="graded_pick"):
            updated += 1
    return updated


def merge_from_path(source: Path, log_path: Path | None = None) -> int:
    target_path, current = load_prediction_rows(log_path)
    _, incoming = load_prediction_rows(source)
    merged = merge_prediction_rows(current, incoming)
    write_prediction_rows(merged, log_path=target_path)
    return len(merged) - len(current)


def sync_nba_prediction_log(
    *,
    repo_root: Path = REPO_ROOT,
    log_path: Path | None = None,
    merge_from: Path | None = None,
    backfill_cache: bool = True,
    grade: bool = True,
    fetch_scoreboard: FetchScoreboard | None = None,
) -> dict[str, int]:
    target = Path(log_path) if log_path else (repo_root / "data" / "nba" / "predictions.csv")
    if merge_from is not None:
        merge_from_path(Path(merge_from), log_path=target)

    target_path, rows = load_prediction_rows(target)
    created = 0
    if backfill_cache:
        incoming = backfill_rows_from_model_cache(repo_root)
        before = len(rows)
        rows = merge_prediction_rows(rows, incoming)
        created = len(rows) - before

    espn_graded = 0
    cache_graded = 0
    if grade:
        espn_graded = grade_rows_from_espn(rows, fetch_scoreboard=fetch_scoreboard)
        cache_graded = grade_rows_from_cache_results(rows, repo_root=repo_root)

    write_prediction_rows(rows, log_path=target_path)
    counts = count_graded_predictions(rows)
    counts.update({
        "created": created,
        "espn_graded": espn_graded,
        "cache_graded": cache_graded,
        "log_path_exists": int(target_path.exists()),
    })
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persist and grade the NBA team-model prediction log.")
    parser.add_argument("--merge-from", default="", help="Existing predictions.csv to union into the committed log.")
    parser.add_argument("--no-backfill", action="store_true", help="Do not reconstruct rows from model cache.")
    parser.add_argument("--no-grade", action="store_true", help="Do not attach realized outcomes.")
    parser.add_argument("--log-path", default="", help="Override the committed prediction log path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = sync_nba_prediction_log(
        merge_from=Path(args.merge_from) if args.merge_from else None,
        backfill_cache=not args.no_backfill,
        grade=not args.no_grade,
        log_path=Path(args.log_path) if args.log_path else DEFAULT_PREDICTION_LOG_PATH,
    )
    print(
        "[nba-prediction-log] "
        f"total={summary['total']} graded={summary['graded']} "
        f"platt_eligible={summary['platt_eligible']} created={summary['created']} "
        f"espn_graded={summary['espn_graded']} cache_graded={summary['cache_graded']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
