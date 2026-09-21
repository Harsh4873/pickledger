"""Poisson goal rates from the last completed NHL regular season.

Ratings are season goal rates, not prices. Moneyline probability adds the
observed home win rate in overtime and shootouts. A tie that reaches overtime
does not cover a -1.5 puck line. Nothing here mints a line or a price.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "nhl_ratings.json"

# ESPN scoreboard abbreviations that differ from the NHL API.
ABBREV_ALIASES = {
    "LA": "LAK",
    "NJ": "NJD",
    "SJ": "SJS",
    "TB": "TBL",
    "WAS": "WSH",
}

MAX_GOALS = 12


def canonical_abbrev(value: Any) -> str:
    raw = str(value or "").strip().upper()
    return ABBREV_ALIASES.get(raw, raw)


def load_ratings() -> dict[str, Any] | None:
    try:
        payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    teams = payload.get("teams") if isinstance(payload, dict) else None
    league = payload.get("league") if isinstance(payload, dict) else None
    if not isinstance(teams, dict) or not isinstance(league, dict) or not teams:
        return None
    return payload


def _team(ratings: dict[str, Any], abbrev: str) -> dict[str, Any] | None:
    teams = ratings.get("teams") or {}
    row = teams.get(canonical_abbrev(abbrev))
    return row if isinstance(row, dict) else None


def _poisson(lam: float, limit: int = MAX_GOALS) -> list[float]:
    lam = max(0.05, float(lam))
    probability = math.exp(-lam)
    values = [probability]
    for goals in range(1, limit + 1):
        probability *= lam / goals
        values.append(probability)
    return values


def over_probability(lambda_home: float, lambda_away: float, line: float) -> float:
    """P(regulation goals, plus one overtime goal on a tie, exceed the line)."""
    home_pmf = _poisson(lambda_home)
    away_pmf = _poisson(lambda_away)
    probability = 0.0
    mass = 0.0
    for home_goals, home_probability in enumerate(home_pmf):
        for away_goals, away_probability in enumerate(away_pmf):
            weight = home_probability * away_probability
            mass += weight
            goals = home_goals + away_goals + (1 if home_goals == away_goals else 0)
            if goals > line:
                probability += weight
    return probability / mass if mass else 0.5


def project_game(ratings: dict[str, Any], home_abbrev: str, away_abbrev: str) -> dict[str, Any]:
    """Return goal rates and win/cover probabilities for one matchup."""
    league = ratings["league"]
    home = _team(ratings, home_abbrev)
    away = _team(ratings, away_abbrev)
    evidence_ok = home is not None and away is not None
    league_gpg = float(league["goals_per_game"])
    if not evidence_ok:
        lam_home = float(league["home_goals_per_game"])
        lam_away = float(league["away_goals_per_game"])
    else:
        home_factor = float(league["home_factor"])
        away_factor = float(league["away_factor"])
        lam_home = (float(home["goals_for_per_game"]) * float(away["goals_against_per_game"]) / league_gpg) * home_factor
        lam_away = (float(away["goals_for_per_game"]) * float(home["goals_against_per_game"]) / league_gpg) * away_factor
    home_pmf = _poisson(lam_home)
    away_pmf = _poisson(lam_away)
    regulation_home = regulation_away = tie = cover = 0.0
    for home_goals, home_probability in enumerate(home_pmf):
        for away_goals, away_probability in enumerate(away_pmf):
            probability = home_probability * away_probability
            if home_goals > away_goals:
                regulation_home += probability
            elif home_goals < away_goals:
                regulation_away += probability
            else:
                tie += probability
            if home_goals >= away_goals + 2:
                cover += probability
    mass = regulation_home + regulation_away + tie
    if mass > 0:
        regulation_home /= mass
        regulation_away /= mass
        tie /= mass
        cover /= mass
    ot_home = league.get("ot_home_win_rate")
    if isinstance(ot_home, (int, float)) and 0.0 < float(ot_home) < 1.0:
        ot_rate = float(ot_home)
        ot_source = "observed_home_ot_so"
    else:
        ot_rate = None
        ot_source = "unavailable"
    if ot_rate is None:
        moneyline_home = regulation_home
    else:
        moneyline_home = regulation_home + tie * ot_rate
    return {
        "evidence_ok": evidence_ok,
        "lambda_home": lam_home,
        "lambda_away": lam_away,
        "expected_regulation_total": lam_home + lam_away,
        "expected_total_with_ot_goal": lam_home + lam_away + tie,
        "regulation_home_win_probability": regulation_home,
        "regulation_away_win_probability": regulation_away,
        "regulation_tie_probability": tie,
        "moneyline_home_probability": moneyline_home,
        "puckline_home_cover_probability": cover,
        "ot_home_win_rate": ot_rate,
        "ot_rate_source": ot_source,
        "home_games": None if home is None else home.get("games_played"),
        "away_games": None if away is None else away.get("games_played"),
        "home_gf_per_game": None if home is None else home.get("goals_for_per_game"),
        "away_gf_per_game": None if away is None else away.get("goals_for_per_game"),
        "home_ga_per_game": None if home is None else home.get("goals_against_per_game"),
        "away_ga_per_game": None if away is None else away.get("goals_against_per_game"),
    }
