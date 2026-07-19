"""MLB Team Total model — per-team full-game run totals.

Unlike the inning model (whose market does not exist at the repo's odds
source) the DraftKings-via-ESPN prop feed posts a real two-sided "Team
Total Runs" market for every game, so this model's picks can carry
observed prices, enter the financial outcome ledger, and qualify for
strict publication through walk-forward validation — the path the inning
and F5 models are structurally locked out of.

Projection per team:
- offense expectation from the team's last-30 per-inning scoring history
  (shared cache with the inning model — no extra fetches), shrunk toward
  the league mean,
- opposing pitching multiplier: ERA-derived starter rate over the
  starter's expected share of the game (their average outs per start),
  bullpen runs-allowed rate for the remainder, tempered because the
  offense baseline already reflects league-average pitching,
- static park factor.

Runs are approximated Normal(mu, sigma=3.0) — team run distributions are
right-skewed, so the model is deliberately conservative near the line —
and each team's best half-run ladder line/side is emitted with an edge
computed against the assumed -110 price the row is stamped with until the
real market price replaces it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
MLB_INNING_DIR = BASE_DIR.parent / "mlb_inning"
sys.path.insert(0, str(MLB_INNING_DIR))

from mlb_inning_fetcher import fetch_todays_games, normalize_date, safe_float, safe_int  # noqa: E402
from mlb_inning_history import fetch_team_contexts, MLB_AVG_SCORELESS  # noqa: E402
from mlb_inning_environment import park_run_factor  # noqa: E402
from mlb_inning_probability import era_derived_scoreless_rate  # noqa: E402

OUTPUT_PATH = BASE_DIR / "mlb_team_total_output.json"

LEAGUE_RUNS_PER_GAME = 4.4
LEAGUE_LATE_SCORELESS = 0.75  # league bullpen-inning scoreless baseline
TEAM_RUNS_SIGMA = 3.0
TEAM_TOTAL_LINES = (2.5, 3.5, 4.5, 5.5, 6.5)
ASSUMED_BREAKEVEN = 110.0 / 210.0  # -110 both ways

# Offense sample shrink (sample_games/(sample+K)) and pitching temper —
# the offense baseline already faced league-average pitching, so the
# opponent multiplier applies at less than full strength.
OFFENSE_SHRINK_K = 10.0
PITCHING_TEMPER = 0.7

# Team run distributions are right-skewed: the median sits below the
# mean, so a symmetric normal overprices Overs near the line. Shifting
# the location down before the CDF biases the model toward Unders the
# same way the real distribution does.
MEDIAN_SHIFT = 0.25

# The shrunk league-average projection (~4.15 after the median shift)
# against the modal 4.5 line produces a structural ~0.546 Under for every
# average team; the LEAN gate sits above that artifact so only genuine
# deviations from the line qualify.
BET_PROB = 0.575
BET_EDGE_PP = 5.0
LEAN_PROB = 0.55
LEAN_EDGE_PP = 3.0


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _offense_expected_runs(context: dict[str, Any]) -> tuple[float, float]:
    offense = context.get("offense") or {}
    total = 0.0
    sample = 0.0
    for inning in range(1, 10):
        stats = offense.get(inning) or offense.get(str(inning)) or {}
        total += safe_float(stats.get("avg_runs"), max(0.0, 1.0 - MLB_AVG_SCORELESS[inning]))
        sample = max(sample, safe_float(stats.get("sample_games"), 0.0))
    weight = sample / (sample + OFFENSE_SHRINK_K) if sample > 0 else 0.0
    return (total * weight) + (LEAGUE_RUNS_PER_GAME * (1.0 - weight)), sample


def _pitching_multiplier(pitcher: dict[str, Any], context: dict[str, Any]) -> dict[str, float]:
    """Runs-allowed rate of the opposing pitching staff vs league average."""
    era = safe_float(pitcher.get("era"), 4.20)
    starter_allow = 1.0 - era_derived_scoreless_rate(era)
    league_allow = 1.0 - era_derived_scoreless_rate(4.20)
    starter_ratio = _clamp(starter_allow / league_allow, 0.70, 1.35) if league_allow > 0 else 1.0

    starter_record = ((context.get("starters") or {}).get(str(safe_int(pitcher.get("id")))) or {})
    expected_outs = safe_float(pitcher.get("expected_outs"), safe_float(starter_record.get("avg_outs"), 16.0))
    starter_share = _clamp((expected_outs / 3.0) / 9.0, 0.40, 0.75)

    bullpen_rate = context.get("bullpen_scoreless_rate")
    bullpen_ratio = 1.0
    if bullpen_rate is not None:
        bullpen_ratio = _clamp(
            (1.0 - safe_float(bullpen_rate, LEAGUE_LATE_SCORELESS)) / (1.0 - LEAGUE_LATE_SCORELESS),
            0.70,
            1.35,
        )

    combined = (starter_ratio * starter_share) + (bullpen_ratio * (1.0 - starter_share))
    return {
        "starter_ratio": round(starter_ratio, 4),
        "starter_share": round(starter_share, 4),
        "bullpen_ratio": round(bullpen_ratio, 4),
        "multiplier": round(combined ** PITCHING_TEMPER, 4),
    }


def _decision(probability: float, edge_pp: float) -> str:
    if probability >= BET_PROB and edge_pp >= BET_EDGE_PP:
        return "BET"
    if probability >= LEAN_PROB and edge_pp >= LEAN_EDGE_PP:
        return "LEAN"
    return "PASS"


def _best_candidate(mu: float) -> dict[str, Any]:
    """Candidate at the ladder line nearest the projection.

    The extreme ladder lines always look like huge "edges" against a flat
    -110 assumption (Over 2.5 clears 70%+ for any competent offense) but
    the book prices those lines at -250, not -110 — the market posts its
    total at the line nearest its own projection, so that's the only line
    where the assumed price is a sane placeholder and where the real
    DraftKings price can attach for an exact match.
    """
    line = _clamp(math.floor(mu) + 0.5, TEAM_TOTAL_LINES[0], TEAM_TOTAL_LINES[-1])
    over_probability = 1.0 - _phi((line - (mu - MEDIAN_SHIFT)) / TEAM_RUNS_SIGMA)
    if over_probability >= 0.5:
        direction, probability = "Over", over_probability
    else:
        direction, probability = "Under", 1.0 - over_probability
    return {
        "line": line,
        "direction": direction,
        "probability": round(probability, 4),
        "edge_pp": round((probability - ASSUMED_BREAKEVEN) * 100.0, 2),
    }


def _team_projection(
    team: str,
    side: str,
    context: dict[str, Any],
    opposing_pitcher: dict[str, Any],
    opposing_context: dict[str, Any],
    venue_id: Any,
) -> dict[str, Any]:
    offense_runs, offense_sample = _offense_expected_runs(context)
    pitching = _pitching_multiplier(opposing_pitcher, opposing_context)
    park = park_run_factor(venue_id)
    mu = offense_runs * pitching["multiplier"] * park
    candidate = _best_candidate(mu)
    decision = _decision(candidate["probability"], candidate["edge_pp"])
    return {
        "team": team,
        "side": side,
        "projected_runs": round(mu, 3),
        "offense_runs": round(offense_runs, 3),
        "offense_sample_games": offense_sample,
        "pitching": pitching,
        "park_factor": park,
        "line": candidate["line"],
        "direction": candidate["direction"],
        "probability": candidate["probability"],
        "edge_pp": candidate["edge_pp"],
        "decision": decision,
        "pick": f"{team} Team Total {candidate['direction']} {candidate['line']:g}",
        "opposing_pitcher": str(opposing_pitcher.get("name") or "TBD"),
    }


def run_mlb_team_total_model(target_date: str | date | None = None) -> list[dict[str, Any]]:
    model_date = normalize_date(target_date)
    games = fetch_todays_games(model_date)
    if not games:
        _write_output({"date": model_date, "model": "MLBTeamTotal", "picks": []})
        print(f"[MLBTeamTotal] No eligible MLB games found for {model_date}.")
        return []

    contexts = fetch_team_contexts(games, model_date)
    results: list[dict[str, Any]] = []
    for game in games:
        home_team = str(game.get("home_team") or "Home Team")
        away_team = str(game.get("away_team") or "Away Team")
        home_context = contexts.get(home_team) or {}
        away_context = contexts.get(away_team) or {}
        venue_id = game.get("venue_id")
        results.append({
            "game_id": str(game.get("game_id") or ""),
            "game_date": model_date,
            "game_start_time": str(game.get("game_start_time") or ""),
            "game_order": game.get("game_order", 0),
            "matchup": f"{home_team} vs {away_team}",
            "home_team": home_team,
            "away_team": away_team,
            "venue_id": venue_id,
            "venue_name": str(game.get("venue_name") or ""),
            "team_totals": [
                _team_projection(
                    away_team, "away", away_context,
                    game.get("home_pitcher") or {}, home_context, venue_id,
                ),
                _team_projection(
                    home_team, "home", home_context,
                    game.get("away_pitcher") or {}, away_context, venue_id,
                ),
            ],
        })

    _write_output({"date": model_date, "model": "MLBTeamTotal", "picks": results})
    print(f"[MLBTeamTotal] {len(results)} games processed. Output saved to {OUTPUT_PATH.name}")
    return results


def _write_output(payload: dict[str, Any]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MLB team total runs model.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_mlb_team_total_model(args.date or datetime.today().strftime("%Y-%m-%d"))
