from __future__ import annotations

import csv
from pathlib import Path

from mlb_api import StatsAPIClient
from prediction_logging import LOG_FIELDS, PREDICTION_LOG_PATH


def seed_realized_outcomes(log_path: Path = PREDICTION_LOG_PATH) -> int:
    if not log_path.exists():
        raise FileNotFoundError(f"Missing prediction log at {log_path}")

    client = StatsAPIClient()
    rows: list[dict[str, str]] = []
    updates = 0
    with log_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("realized_outcome"):
                rows.append(row)
                continue

            game_date = row.get("game_date", "")
            away_team = row.get("away_team", "")
            home_team = row.get("home_team", "")
            if not game_date or not away_team or not home_team:
                rows.append(row)
                continue

            season = int(game_date[:4])
            matched = None
            for game in client.get_schedule_for_season(season):
                if str(game.get("game_date")) != game_date:
                    continue
                if game.get("away_name") != away_team or game.get("home_name") != home_team:
                    continue
                if "final" not in str(game.get("status", "")).lower():
                    continue
                matched = game
                break

            if matched:
                away_score = str(matched.get("away_score", ""))
                home_score = str(matched.get("home_score", ""))
                row["away_score"] = away_score
                row["home_score"] = home_score
                try:
                    total_runs = int(away_score) + int(home_score)
                    row["realized_total_runs"] = str(total_runs)
                except ValueError:
                    row["realized_total_runs"] = ""
                row["realized_outcome"] = (
                    home_team if int(home_score or 0) > int(away_score or 0) else away_team
                )
                updates += 1

            rows.append(row)

    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return updates


if __name__ == "__main__":
    updated = seed_realized_outcomes()
    print(f"Updated {updated} logged predictions with realized outcomes.")
