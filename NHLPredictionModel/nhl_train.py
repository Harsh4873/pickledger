"""Fit NHL goal-rate ratings from the NHL standings endpoint.

The endpoint returns the last completed regular season while the next regular
season has not started. Every team must have played a full 82-game schedule.
The writer stores rates only. It does not store scores from a live slate or
any price.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

STANDINGS_URL = "https://api-web.nhle.com/v1/standings/now"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "nhl_ratings.json"
MODEL_FAMILY = "nhl_poisson_v2"
SHADOW_CANDIDATE_VERSION = "v1"


def model_version(prior_season: str) -> str:
    """A new season of fitted ratings must never inherit an older approval."""
    return f"{MODEL_FAMILY}_{prior_season}_shadow_ev_{SHADOW_CANDIDATE_VERSION}"


def decision_policy() -> dict:
    """Prospective research rule; no priced NHL results have qualified it."""
    return {
        "mode": "research_only",
        "reason": "no_as_of_priced_walk_forward_or_unused_holdout",
        "shadow_candidate": {
            "version": f"nhl_shadow_ev_{SHADOW_CANDIDATE_VERSION}",
            "markets": ["h2h", "spread", "totals"],
            "rule": "higher two-way expected value, observed pregame quote, complete price pair",
            "max_juice": -125,
            "lean_min_expected_value": 0.05,
            "bet_min_expected_value": 0.10,
            "lean_units": 0.25,
            "bet_units": 0.5,
            "validated": False,
        },
    }


def _text(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("default") or "").strip()
    return str(value or "").strip()


def _required_number(row: dict, key: str) -> float:
    if key not in row:
        raise KeyError(f"standings row missing {key}")
    return float(row[key])


def train(payload: dict | None = None) -> dict:
    if payload is None:
        request = Request(STANDINGS_URL, headers={"User-Agent": "PickLedger/1.0", "Accept": "application/json"})
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    rows = payload.get("standings") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 32:
        raise RuntimeError(f"expected 32 NHL standings rows, got {0 if not isinstance(rows, list) else len(rows)}")
    teams: dict[str, dict] = {}
    totals = {
        "games": 0.0,
        "goals_for": 0.0,
        "home_games": 0.0,
        "home_goals_for": 0.0,
        "home_goals_against": 0.0,
        "home_ot_wins": 0.0,
        "home_ot_losses": 0.0,
    }
    standings_date = ""
    game_type = None
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("standings row was not an object")
        games = _required_number(row, "gamesPlayed")
        if games != 82:
            raise RuntimeError(f"{_text(row.get('teamAbbrev'))} has gamesPlayed={games}, expected 82")
        abbrev = _text(row.get("teamAbbrev")).upper()
        name = _text(row.get("teamName"))
        if not abbrev or not name:
            raise RuntimeError("standings row missing team abbreviation or name")
        goals_for = _required_number(row, "goalFor")
        goals_against = _required_number(row, "goalAgainst")
        home_games = _required_number(row, "homeGamesPlayed")
        home_goals_for = _required_number(row, "homeGoalsFor")
        home_goals_against = _required_number(row, "homeGoalsAgainst")
        home_wins = _required_number(row, "homeWins")
        home_regulation_wins = _required_number(row, "homeRegulationWins")
        home_ot_losses = _required_number(row, "homeOtLosses")
        if home_wins < home_regulation_wins:
            raise RuntimeError(f"{abbrev} home wins are below regulation wins")
        if abbrev in teams:
            raise RuntimeError(f"duplicate standings team {abbrev}")
        row_date = str(row.get("date") or "")
        if standings_date and row_date != standings_date:
            raise RuntimeError("standings rows have different dates")
        if game_type is not None and row.get("gameTypeId") != game_type:
            raise RuntimeError("standings rows have different game types")
        teams[abbrev] = {
            "abbrev": abbrev,
            "name": name,
            "games_played": int(games),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goals_for_per_game": round(goals_for / games, 6),
            "goals_against_per_game": round(goals_against / games, 6),
        }
        totals["games"] += games
        totals["goals_for"] += goals_for
        totals["home_games"] += home_games
        totals["home_goals_for"] += home_goals_for
        totals["home_goals_against"] += home_goals_against
        totals["home_ot_wins"] += home_wins - home_regulation_wins
        totals["home_ot_losses"] += home_ot_losses
        standings_date = standings_date or row_date
        game_type = row.get("gameTypeId")
    if int(game_type or 0) != 2:
        raise RuntimeError(f"standings gameTypeId={game_type}, expected regular season (2)")
    road_games = totals["games"] - totals["home_games"]
    road_goals = totals["goals_for"] - totals["home_goals_for"]
    if totals["home_games"] <= 0 or road_games <= 0:
        raise RuntimeError("home/road split is empty")
    home_gpg = totals["home_goals_for"] / totals["home_games"]
    away_gpg = road_goals / road_games
    neutral = (home_gpg + away_gpg) / 2.0
    ot_decisions = totals["home_ot_wins"] + totals["home_ot_losses"]
    if ot_decisions <= 0:
        raise RuntimeError("no home overtime or shootout decisions in the standings")
    try:
        as_of = date.fromisoformat(standings_date)
    except ValueError as exc:
        raise RuntimeError(f"invalid standings date {standings_date}") from exc
    # A completed NHL regular season ends in spring. Refuse a stale or
    # future-looking standings date instead of assigning the wrong season.
    if as_of.month not in {4, 5, 6}:
        raise RuntimeError(f"completed standings date {standings_date} is outside spring")
    prior_season = f"{as_of.year - 1}{as_of.year}"
    artifact = {
        "model_version": model_version(prior_season),
        "source": STANDINGS_URL,
        "prior_season": prior_season,
        "standings_date": standings_date,
        "game_type_id": 2,
        "teams_count": len(teams),
        "decision_policy": decision_policy(),
        "league": {
            "goals_per_game": round(totals["goals_for"] / totals["games"], 6),
            "home_goals_per_game": round(home_gpg, 6),
            "away_goals_per_game": round(away_gpg, 6),
            "home_factor": round(home_gpg / neutral, 6),
            "away_factor": round(away_gpg / neutral, 6),
            "ot_home_win_rate": round(totals["home_ot_wins"] / ot_decisions, 6),
            "home_ot_decisions": int(ot_decisions),
        },
        "teams": dict(sorted(teams.items())),
    }
    return artifact


def main() -> None:
    artifact = train()
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {ARTIFACT_PATH} teams={artifact['teams_count']} "
        f"season={artifact['prior_season']} as_of={artifact['standings_date']}"
    )


if __name__ == "__main__":
    main()
