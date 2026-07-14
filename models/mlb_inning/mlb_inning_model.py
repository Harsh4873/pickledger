from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from mlb_inning_fetcher import fetch_todays_games, normalize_date
    from mlb_inning_history import fetch_team_histories
    from mlb_inning_matchup import compute_matchup_threats
    from mlb_inning_probability import compute_inning_probabilities
except ImportError:
    from .mlb_inning_fetcher import fetch_todays_games, normalize_date
    from .mlb_inning_history import fetch_team_histories
    from .mlb_inning_matchup import compute_matchup_threats
    from .mlb_inning_probability import compute_inning_probabilities


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "mlb_inning_output.json"


def run_mlb_inning_model(target_date: str | date | None = None) -> list[dict[str, Any]]:
    model_date = normalize_date(target_date)
    games = fetch_todays_games(model_date)
    if not games:
        output = {"date": model_date, "model": "MLBInning", "picks": []}
        _write_output(output)
        print(f"[MLBInning] No eligible MLB games found for {model_date}.")
        return []

    team_histories = fetch_team_histories(games, model_date)
    matchup_threats = compute_matchup_threats(games)
    picks = [
        compute_inning_probabilities(game, team_histories, matchup_threats)
        for game in games
    ]

    output = {"date": model_date, "model": "MLBInning", "picks": picks}
    _write_output(output)
    print(f"[MLBInning] {len(picks)} games processed. Output saved to {OUTPUT_PATH.name}")
    return picks


def _write_output(payload: dict[str, Any]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MLB inning scoreless probability model.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    return parser.parse_args()


def _date_or_today(raw_date: str) -> str:
    if raw_date:
        return normalize_date(raw_date)
    return datetime.today().strftime("%Y-%m-%d")


if __name__ == "__main__":
    args = _parse_args()
    run_mlb_inning_model(_date_or_today(args.date))
