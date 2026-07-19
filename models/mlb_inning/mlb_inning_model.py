from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    from mlb_inning_fetcher import fetch_todays_games, normalize_date, safe_float, safe_int
    from mlb_inning_history import fetch_team_contexts, _league_default_history
    from mlb_inning_matchup import compute_matchup_threats
    from mlb_inning_probability import compute_inning_probabilities, era_derived_scoreless_rate
except ImportError:
    from .mlb_inning_fetcher import fetch_todays_games, normalize_date, safe_float, safe_int
    from .mlb_inning_history import fetch_team_contexts, _league_default_history
    from .mlb_inning_matchup import compute_matchup_threats
    from .mlb_inning_probability import compute_inning_probabilities, era_derived_scoreless_rate


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

    team_contexts = fetch_team_contexts(games, model_date)
    team_histories = {
        name: context.get("offense") or _league_default_history()
        for name, context in team_contexts.items()
    }
    for game in games:
        _attach_pitching_context(game, team_contexts)
    matchup_threats = compute_matchup_threats(games)
    picks = [
        compute_inning_probabilities(game, team_histories, matchup_threats)
        for game in games
    ]

    output = {"date": model_date, "model": "MLBInning", "picks": picks}
    _write_output(output)
    print(f"[MLBInning] {len(picks)} games processed. Output saved to {OUTPUT_PATH.name}")
    return picks


# A starter appears in only ~5-6 of the team's last 30 games, so the
# observed per-inning scoreless rates are shrunk toward the ERA-derived
# prior; at 5 completed innings the observation gets 50% weight.
STARTER_RATE_SHRINK_K = 5.0


def _attach_pitching_context(game: dict[str, Any], team_contexts: dict[str, dict[str, Any]]) -> None:
    """Populate the per-inning starter and bullpen rates the probability
    layer reads (``inning_scoreless_rates``, ``expected_outs``,
    ``team_bullpen.scoreless_rate_by_inning``) — previously dead inputs
    that always fell back to flat ERA/league rates."""
    for side in ("home", "away"):
        pitcher = game.get(f"{side}_pitcher")
        context = team_contexts.get(str(game.get(f"{side}_team") or ""))
        if not isinstance(pitcher, dict) or not isinstance(context, dict):
            continue

        starter_record = (context.get("starters") or {}).get(str(safe_int(pitcher.get("id"))))
        if isinstance(starter_record, dict):
            prior = era_derived_scoreless_rate(safe_float(pitcher.get("era"), 4.20))
            rates: dict[str, float] = {}
            samples: dict[str, int] = {}
            for inning_key, entry in (starter_record.get("innings") or {}).items():
                observed_n = safe_int((entry or {}).get("n"))
                if observed_n <= 0:
                    continue
                scoreless = safe_int((entry or {}).get("scoreless"))
                rates[str(inning_key)] = round(
                    (scoreless + prior * STARTER_RATE_SHRINK_K) / (observed_n + STARTER_RATE_SHRINK_K), 4
                )
                samples[str(inning_key)] = observed_n
            if rates:
                pitcher["inning_scoreless_rates"] = rates
                pitcher["inning_rate_samples"] = samples
            avg_outs = safe_float(starter_record.get("avg_outs"))
            if avg_outs > 0:
                pitcher["expected_outs"] = avg_outs

        bullpen_by_inning = context.get("bullpen_scoreless_by_inning") or {}
        if bullpen_by_inning:
            team_bullpen = pitcher.setdefault("team_bullpen", {})
            team_bullpen.setdefault("scoreless_rate_by_inning", bullpen_by_inning)
            if context.get("bullpen_scoreless_rate") is not None:
                team_bullpen.setdefault("scoreless_rate", context["bullpen_scoreless_rate"])
            team_bullpen.setdefault("scoreless_samples", safe_int(context.get("bullpen_samples")))


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
