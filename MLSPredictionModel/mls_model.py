"""Algorithmic MLS moneyline, spread, and totals model (FIFA WC engine, league-parameterized).

The model intentionally does not use historical head-to-head results or a
trained estimator. It rates the current tournament squad through each
player's position, availability, current club league, and club table record,
then layers in current tournament form, venue scoring context, and recent
model feedback before converting the ratings into Poisson goal probabilities.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests


ESPN_SITE_API = "https://site.api.espn.com/apis/site/v2/sports/soccer"
ESPN_ATHLETE_API = "https://site.web.api.espn.com/apis/common/v3/sports/soccer/usa.1/athletes"
USER_AGENT = "PickLedgerMLSModel/1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = REPO_ROOT / "data" / "model_cache"
TOURNAMENT_CONTEXT_DAYS = 14
BASE_WORLD_CUP_TOTAL = 3.05  # MLS games average ~3.0-3.2 goals
# Unlike the neutral-site World Cup, MLS home advantage is among the
# largest in world soccer (altitude/travel/turf variance): home teams
# outscore their away baseline by roughly +0.2 and suppress opponents
# by roughly -0.15 goals.
MLS_HOME_GOAL_EDGE = 0.20
MLS_AWAY_GOAL_DRAG = 0.15

LEAGUE_STRENGTH = {
    "eng.1": 92.0,
    "esp.1": 91.0,
    "ita.1": 89.5,
    "ger.1": 89.0,
    "fra.1": 87.5,
    "uefa.champions": 92.0,
    "bra.1": 84.0,
    "por.1": 83.5,
    "ned.1": 82.5,
    "bel.1": 79.5,
    "tur.1": 79.0,
    "arg.1": 79.0,
    "mex.1": 77.5,
    "usa.1": 76.5,
    "sco.1": 75.5,
    "ksa.1": 75.0,
    "jpn.1": 74.0,
    "aus.1": 70.5,
}
POSITION_BASELINE = {
    "goalkeeper": 72.0,
    "defender": 72.0,
    "midfielder": 73.0,
    "forward": 73.0,
}
UNIT_STARTERS = {
    "goalkeeper": 1,
    "defender": 4,
    "midfielder": 3,
    "forward": 3,
}
UNIT_WEIGHTS = {
    "goalkeeper": 0.16,
    "defender": 0.28,
    "midfielder": 0.28,
    "forward": 0.28,
}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _american_implied(odds: Any) -> float | None:
    value = _number(odds)
    if value is None or value == 0:
        return None
    return 100.0 / (value + 100.0) if value > 0 else abs(value) / (abs(value) + 100.0)


def _american_odds(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number != 0 else None


def _position_group(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "goal" in text or text in {"g", "gk"}:
        return "goalkeeper"
    if "def" in text or text in {"d", "cb", "lb", "rb"}:
        return "defender"
    if "mid" in text or text in {"m", "dm", "cm", "am"}:
        return "midfielder"
    return "forward"


def _record_stats(team: dict[str, Any]) -> dict[str, float]:
    record = team.get("record") if isinstance(team.get("record"), dict) else {}
    items = record.get("items") if isinstance(record.get("items"), list) else []
    first = items[0] if items and isinstance(items[0], dict) else {}
    stats = first.get("stats") if isinstance(first.get("stats"), list) else []
    return {
        str(stat.get("name") or ""): float(stat.get("value") or 0)
        for stat in stats
        if isinstance(stat, dict) and _number(stat.get("value")) is not None
    }


def _parse_record_summary(summary: Any) -> dict[str, Any] | None:
    """Parse ESPN soccer records, which are commonly W-D-L in summaries."""
    parts = [part.strip() for part in str(summary or "").split("-")]
    if len(parts) < 3:
        return None
    try:
        wins, draws, losses = (int(float(part)) for part in parts[:3])
    except ValueError:
        return None
    games = wins + draws + losses
    points = (wins * 3) + draws
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points": points,
        "points_per_game": round(points / games, 3) if games else None,
        "summary": f"{wins}-{draws}-{losses}",
    }


def _competitor_record(competitor: dict[str, Any]) -> dict[str, Any]:
    records = competitor.get("records") if isinstance(competitor.get("records"), list) else []
    for record in records:
        if not isinstance(record, dict):
            continue
        parsed = _parse_record_summary(record.get("summary"))
        if parsed:
            return parsed
    return {"games": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0, "points_per_game": None}


def _team_context_key(team: dict[str, Any]) -> str:
    team_id = str(team.get("id") or "").strip()
    if team_id:
        return team_id
    return str(team.get("abbreviation") or team.get("displayName") or team.get("name") or "").strip().lower()


def _fresh_team_record(team: dict[str, Any]) -> dict[str, Any]:
    return {
        "team_id": str(team.get("id") or ""),
        "team": str(team.get("displayName") or team.get("name") or "Unknown"),
        "abbreviation": str(team.get("abbreviation") or ""),
        "games": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "points_per_game": None,
        "goals_for_per_game": None,
        "goals_against_per_game": None,
        "goal_diff_per_game": None,
        "summary": "0-0-0",
    }


def _add_team_result(record: dict[str, Any], goals_for: int, goals_against: int) -> None:
    record["games"] += 1
    record["goals_for"] += goals_for
    record["goals_against"] += goals_against
    if goals_for > goals_against:
        record["wins"] += 1
    elif goals_for < goals_against:
        record["losses"] += 1
    else:
        record["draws"] += 1
    record["points"] = (record["wins"] * 3) + record["draws"]
    games = max(1, int(record["games"]))
    record["points_per_game"] = round(record["points"] / games, 3)
    record["goals_for_per_game"] = round(record["goals_for"] / games, 3)
    record["goals_against_per_game"] = round(record["goals_against"] / games, 3)
    record["goal_diff_per_game"] = round((record["goals_for"] - record["goals_against"]) / games, 3)
    record["summary"] = f"{record['wins']}-{record['draws']}-{record['losses']}"


def _date_range_through(target: date, days: int) -> Iterable[date]:
    start = target - timedelta(days=max(0, days - 1))
    for offset in range((target - start).days + 1):
        yield start + timedelta(days=offset)


def club_power(league_slug: str, club: dict[str, Any] | None) -> float:
    """Return a current club-strength proxy on a roughly 60-96 scale."""
    base = LEAGUE_STRENGTH.get(str(league_slug or "").lower(), 70.0)
    if not isinstance(club, dict):
        return base
    stats = _record_stats(club)
    games = stats.get("gamesPlayed", 0.0)
    if games < 5:
        return base
    points = stats.get("points", (stats.get("wins", 0.0) * 3) + stats.get("ties", 0.0))
    goals_for = stats.get("pointsFor", 0.0)
    goals_against = stats.get("pointsAgainst", 0.0)
    ppg_adjustment = _clamp(((points / games) - 1.45) * 5.0, -4.0, 5.0)
    goal_adjustment = _clamp(((goals_for - goals_against) / games) * 2.0, -3.0, 3.0)
    rank = stats.get("rank", 0.0)
    rank_adjustment = _clamp(2.5 - (rank * 0.22), -2.0, 2.0) if rank else 0.0
    return _clamp(base + ppg_adjustment + goal_adjustment + rank_adjustment, 58.0, 96.0)


def _age_adjustment(age: Any, position: str) -> float:
    value = _number(age)
    if value is None:
        return 0.0
    peak_low, peak_high = (27, 33) if position == "goalkeeper" else (24, 30)
    if peak_low <= value <= peak_high:
        return 1.5
    if value < 20 or value > 36:
        return -2.0
    if value < peak_low - 2 or value > peak_high + 2:
        return -0.75
    return 0.5


def player_power(
    player: dict[str, Any],
    profile: dict[str, Any] | None,
    club: dict[str, Any] | None,
) -> dict[str, Any]:
    """Rank one current squad player from club quality and availability."""
    profile = profile if isinstance(profile, dict) else {}
    athlete = profile.get("athlete") if isinstance(profile.get("athlete"), dict) else {}
    league = profile.get("league") if isinstance(profile.get("league"), dict) else {}
    club_team = athlete.get("team") if isinstance(athlete.get("team"), dict) else {}
    position_data = player.get("position") if isinstance(player.get("position"), dict) else {}
    position = _position_group(position_data.get("name") or position_data.get("abbreviation"))
    league_slug = str(league.get("slug") or "").lower()
    base = club_power(league_slug, club)
    if not league_slug:
        base = POSITION_BASELINE[position]

    injuries = player.get("injuries") if isinstance(player.get("injuries"), list) else []
    status = player.get("status") if isinstance(player.get("status"), dict) else {}
    availability_penalty = min(10.0, len(injuries) * 5.0)
    if str(status.get("type") or "").lower() not in {"", "active"}:
        availability_penalty += 8.0

    rating = _clamp(base + _age_adjustment(player.get("age"), position) - availability_penalty, 45.0, 97.0)
    return {
        "player_id": str(player.get("id") or ""),
        "name": str(player.get("displayName") or player.get("fullName") or "Unknown"),
        "position": position,
        "rating": round(rating, 2),
        "age": player.get("age"),
        "available": availability_penalty == 0,
        "injury_count": len(injuries),
        "club": str(club_team.get("displayName") or club_team.get("name") or "Unknown"),
        "club_id": str(club_team.get("id") or ""),
        "league": str(league.get("name") or "Unknown"),
        "league_slug": league_slug,
        "profile_available": bool(league_slug and club_team.get("id")),
    }


def _weighted_unit(players: Iterable[dict[str, Any]], position: str) -> float:
    ranked = sorted(
        (player for player in players if player.get("position") == position),
        key=lambda player: float(player.get("rating") or 0),
        reverse=True,
    )
    if not ranked:
        return POSITION_BASELINE[position] - 4.0
    starter_count = UNIT_STARTERS[position]
    starters = ranked[:starter_count]
    depth = ranked[starter_count:starter_count + max(2, starter_count)]
    starter_score = sum(float(player["rating"]) for player in starters) / len(starters)
    depth_score = (
        sum(float(player["rating"]) for player in depth) / len(depth)
        if depth
        else starter_score - 4.0
    )
    return round((starter_score * 0.82) + (depth_score * 0.18), 2)


def _team_form_record(team: dict[str, Any], tournament_context: dict[str, Any] | None) -> dict[str, Any]:
    context_records = (
        tournament_context.get("team_records")
        if isinstance(tournament_context, dict) and isinstance(tournament_context.get("team_records"), dict)
        else {}
    )
    context_record = context_records.get(_team_context_key(team)) if isinstance(context_records, dict) else None
    record = dict(context_record) if isinstance(context_record, dict) else _fresh_team_record(team)
    current_record = team.get("current_record") if isinstance(team.get("current_record"), dict) else {}
    current_games = int(current_record.get("games") or 0)
    if current_games >= int(record.get("games") or 0):
        for key in ("games", "wins", "draws", "losses", "points", "points_per_game", "summary"):
            if key in current_record:
                record[key] = current_record[key]
    games = int(record.get("games") or 0)
    if games:
        record["points_per_game"] = round(float(record.get("points") or 0) / games, 3)
        if record.get("goals_for_per_game") is None and record.get("goals_for") is not None:
            record["goals_for_per_game"] = round(float(record.get("goals_for") or 0) / games, 3)
        if record.get("goals_against_per_game") is None and record.get("goals_against") is not None:
            record["goals_against_per_game"] = round(float(record.get("goals_against") or 0) / games, 3)
        if record.get("goal_diff_per_game") is None and record.get("goals_for") is not None:
            record["goal_diff_per_game"] = round(
                (float(record.get("goals_for") or 0) - float(record.get("goals_against") or 0)) / games,
                3,
            )
    return record


def _form_adjustments(record: dict[str, Any]) -> dict[str, float]:
    games = int(record.get("games") or 0)
    if games <= 0:
        return {"overall": 0.0, "attack": 0.0, "defense": 0.0, "xg": 0.0}
    sample_weight = min(1.0, games / 3.0)
    ppg = _number(record.get("points_per_game"))
    gf_pg = _number(record.get("goals_for_per_game"))
    ga_pg = _number(record.get("goals_against_per_game"))
    gd_pg = _number(record.get("goal_diff_per_game"))

    record_adjustment = _clamp(((ppg or 1.25) - 1.35) * 0.75, -0.75, 0.95)
    goal_diff_adjustment = _clamp((gd_pg or 0.0) * 0.35, -0.80, 0.90)
    attack_adjustment = _clamp(((gf_pg or 1.25) - 1.35) * 0.65 + max(0.0, gd_pg or 0.0) * 0.12, -0.90, 1.05)
    defense_adjustment = _clamp((1.25 - (ga_pg or 1.25)) * 0.65 + max(0.0, gd_pg or 0.0) * 0.10, -0.90, 1.05)
    xg_adjustment = _clamp(((gf_pg or 1.25) - 1.35) * 0.045 + ((ga_pg or 1.25) - 1.25) * 0.025, -0.09, 0.11)
    return {
        "overall": round((record_adjustment + goal_diff_adjustment) * sample_weight, 3),
        "attack": round(attack_adjustment * sample_weight, 3),
        "defense": round(defense_adjustment * sample_weight, 3),
        "xg": round(xg_adjustment * sample_weight, 4),
    }


def team_power(
    team: dict[str, Any],
    players: list[dict[str, Any]],
    tournament_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    unit_scores = {
        position: _weighted_unit(players, position)
        for position in UNIT_STARTERS
    }
    base_overall = sum(unit_scores[position] * UNIT_WEIGHTS[position] for position in UNIT_WEIGHTS)
    base_attack = (unit_scores["forward"] * 0.72) + (unit_scores["midfielder"] * 0.28)
    base_defense = (unit_scores["defender"] * 0.68) + (unit_scores["midfielder"] * 0.12) + (unit_scores["goalkeeper"] * 0.20)
    form_record = _team_form_record(team, tournament_context)
    form_adjustments = _form_adjustments(form_record)
    overall = _clamp(base_overall + form_adjustments["overall"], 45.0, 99.0)
    attack = _clamp(base_attack + form_adjustments["attack"], 45.0, 99.0)
    defense = _clamp(base_defense + form_adjustments["defense"], 45.0, 99.0)
    ranked = sorted(players, key=lambda player: float(player.get("rating") or 0), reverse=True)
    top_five = ranked[:5]
    availability = sum(bool(player.get("available")) for player in players) / max(1, len(players))
    profile_coverage = sum(bool(player.get("profile_available")) for player in players) / max(1, len(players))
    position_counts = {
        position: sum(player.get("position") == position for player in players)
        for position in UNIT_STARTERS
    }
    roster_ready = (
        len(players) >= 11
        and profile_coverage >= 0.60
        and all(position_counts[position] >= UNIT_STARTERS[position] for position in UNIT_STARTERS)
    )
    return {
        "team_id": str(team.get("id") or ""),
        "team": str(team.get("displayName") or team.get("name") or "Unknown"),
        "abbreviation": str(team.get("abbreviation") or ""),
        "overall": round(overall, 2),
        "attack": round(attack, 2),
        "midfield": unit_scores["midfielder"],
        "defense": round(defense, 2),
        "goalkeeper": unit_scores["goalkeeper"],
        "base_unit_ratings": {
            "overall": round(base_overall, 2),
            "attack": round(base_attack, 2),
            "midfield": unit_scores["midfielder"],
            "defense": round(base_defense, 2),
            "goalkeeper": unit_scores["goalkeeper"],
        },
        "tournament_form": {
            "games": int(form_record.get("games") or 0),
            "record": str(form_record.get("summary") or "0-0-0"),
            "points_per_game": form_record.get("points_per_game"),
            "goals_for_per_game": form_record.get("goals_for_per_game"),
            "goals_against_per_game": form_record.get("goals_against_per_game"),
            "goal_diff_per_game": form_record.get("goal_diff_per_game"),
        },
        "form_adjustments": form_adjustments,
        "availability": round(availability, 3),
        "profile_coverage": round(profile_coverage, 3),
        "position_counts": position_counts,
        "roster_ready": roster_ready,
        "players_rated": len(players),
        "top_players": [
            {
                "name": player["name"],
                "position": player["position"],
                "rating": player["rating"],
                "club": player["club"],
                "league": player["league"],
            }
            for player in top_five
        ],
        "players": ranked,
    }


def expected_goals(team: dict[str, Any], opponent: dict[str, Any], venue_multiplier: float = 1.0) -> float:
    attack_gap = float(team["attack"]) - float(opponent["defense"])
    overall_gap = float(team["overall"]) - float(opponent["overall"])
    availability_drag = max(0.0, 0.94 - float(team.get("availability") or 0.0)) * 2.0
    form_xg = float((team.get("form_adjustments") or {}).get("xg") or 0.0)
    opponent_form_xg = float((opponent.get("form_adjustments") or {}).get("xg") or 0.0)
    value = 1.22 * math.exp((attack_gap / 32.0) + (overall_gap / 90.0) - availability_drag)
    value *= _clamp(venue_multiplier, 0.82, 1.18)
    value *= _clamp(1.0 + form_xg + (opponent_form_xg * 0.35), 0.88, 1.14)
    return round(_clamp(value, 0.25, 3.60), 3)


def poisson_probabilities(home_xg: float, away_xg: float, max_goals: int = 9) -> dict[str, float]:
    home = [math.exp(-home_xg) * (home_xg ** goals) / math.factorial(goals) for goals in range(max_goals + 1)]
    away = [math.exp(-away_xg) * (away_xg ** goals) / math.factorial(goals) for goals in range(max_goals + 1)]
    home_win = draw = away_win = 0.0
    for home_goals, home_prob in enumerate(home):
        for away_goals, away_prob in enumerate(away):
            probability = home_prob * away_prob
            if home_goals > away_goals:
                home_win += probability
            elif home_goals < away_goals:
                away_win += probability
            else:
                draw += probability
    total = home_win + draw + away_win
    return {
        "home_win": home_win / total,
        "draw": draw / total,
        "away_win": away_win / total,
    }


def total_probability(projected_total: float, line: float, side: str) -> float:
    max_goals = 14
    probabilities = [
        math.exp(-projected_total) * (projected_total ** goals) / math.factorial(goals)
        for goals in range(max_goals + 1)
    ]
    if side == "over":
        return sum(probability for goals, probability in enumerate(probabilities) if goals > line)
    return sum(probability for goals, probability in enumerate(probabilities) if goals < line)


def spread_cover_probability(home_xg: float, away_xg: float, side: str, line: float, max_goals: int = 12) -> float:
    home = [math.exp(-home_xg) * (home_xg ** goals) / math.factorial(goals) for goals in range(max_goals + 1)]
    away = [math.exp(-away_xg) * (away_xg ** goals) / math.factorial(goals) for goals in range(max_goals + 1)]
    cover = push = total = 0.0
    for home_goals, home_prob in enumerate(home):
        for away_goals, away_prob in enumerate(away):
            probability = home_prob * away_prob
            total += probability
            selected_goals, opponent_goals = (home_goals, away_goals) if side == "home" else (away_goals, home_goals)
            adjusted = selected_goals + line
            if abs(adjusted - opponent_goals) < 1e-9:
                push += probability
            elif adjusted > opponent_goals:
                cover += probability
    non_push = max(0.0, total - push)
    if non_push > 0:
        return cover / non_push
    return cover / total if total else 0.5


def _status_state(event: dict[str, Any]) -> str:
    status = event.get("status") if isinstance(event.get("status"), dict) else {}
    status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
    return str(status_type.get("state") or "").lower()


def _event_completed(event: dict[str, Any]) -> bool:
    status = event.get("status") if isinstance(event.get("status"), dict) else {}
    status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
    return _status_state(event) == "post" or bool(status_type.get("completed"))


def _score_value(competitor: dict[str, Any]) -> int | None:
    score = _number(competitor.get("score"))
    return int(score) if score is not None else None


def _venue_key(venue: dict[str, Any]) -> str:
    venue_id = str(venue.get("id") or "").strip()
    if venue_id:
        return venue_id
    return str(venue.get("fullName") or venue.get("name") or "").strip().lower()


def _venue_name(venue: dict[str, Any]) -> str:
    return str(venue.get("fullName") or venue.get("name") or "Unknown venue")


def _venue_city(venue: dict[str, Any]) -> str:
    address = venue.get("address") if isinstance(venue.get("address"), dict) else {}
    return str(address.get("city") or "")


def _build_tournament_context(date_iso: str, client: Any, days: int = TOURNAMENT_CONTEXT_DAYS) -> dict[str, Any]:
    target = datetime.strptime(date_iso, "%Y-%m-%d").date()
    team_records: dict[str, dict[str, Any]] = {}
    venues: dict[str, dict[str, Any]] = {}
    completed_events: set[str] = set()
    total_goals: list[int] = []

    for slate_date in _date_range_through(target, days):
        try:
            scoreboard = client.scoreboard(slate_date.strftime("%Y-%m-%d"))
        except Exception:
            continue
        for event in scoreboard.get("events") if isinstance(scoreboard.get("events"), list) else []:
            if not isinstance(event, dict) or not _event_completed(event):
                continue
            event_id = str(event.get("id") or "")
            if event_id and event_id in completed_events:
                continue
            competitions = event.get("competitions") if isinstance(event.get("competitions"), list) else []
            competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
            competitors = competition.get("competitors") if isinstance(competition.get("competitors"), list) else []
            if len(competitors) < 2:
                continue
            scored = [(competitor, _score_value(competitor)) for competitor in competitors if isinstance(competitor, dict)]
            if len(scored) < 2 or any(score is None for _, score in scored[:2]):
                continue
            if event_id:
                completed_events.add(event_id)
            first_score = int(scored[0][1] or 0)
            second_score = int(scored[1][1] or 0)
            game_total = first_score + second_score
            total_goals.append(game_total)

            for index, (competitor, goals_for) in enumerate(scored[:2]):
                opponent_goals = second_score if index == 0 else first_score
                team = competitor.get("team") if isinstance(competitor.get("team"), dict) else {}
                key = _team_context_key(team)
                if not key:
                    continue
                if key not in team_records:
                    team_records[key] = _fresh_team_record(team)
                _add_team_result(team_records[key], int(goals_for or 0), int(opponent_goals or 0))

            venue = competition.get("venue") if isinstance(competition.get("venue"), dict) else {}
            venue_key = _venue_key(venue)
            if venue_key:
                profile = venues.setdefault(
                    venue_key,
                    {
                        "venue_id": str(venue.get("id") or ""),
                        "venue_name": _venue_name(venue),
                        "city": _venue_city(venue),
                        "games": 0,
                        "goals": 0,
                    },
                )
                profile["games"] += 1
                profile["goals"] += game_total

    tournament_avg = sum(total_goals) / len(total_goals) if total_goals else BASE_WORLD_CUP_TOTAL
    for profile in venues.values():
        games = int(profile.get("games") or 0)
        average = float(profile.get("goals") or 0) / games if games else tournament_avg
        sample_weight = min(1.0, games / 3.0)
        relative_delta = (average - tournament_avg) / max(1.0, tournament_avg)
        multiplier = 1.0 + _clamp(relative_delta * 0.45 * sample_weight, -0.12, 0.12)
        significant = games >= 2 and abs(average - tournament_avg) >= 0.35
        profile["avg_total"] = round(average, 3)
        profile["tournament_avg_total"] = round(tournament_avg, 3)
        profile["goal_multiplier"] = round(multiplier, 4)
        profile["significant"] = significant

    return {
        "team_records": team_records,
        "venue_profiles": venues,
        "completed_games": len(total_goals),
        "overall_avg_total": round(tournament_avg, 3),
        "lookback_days": days,
    }


def _venue_profile_for_game(game: dict[str, Any], tournament_context: dict[str, Any] | None) -> dict[str, Any]:
    venue = game.get("venue") if isinstance(game.get("venue"), dict) else {}
    profiles = (
        tournament_context.get("venue_profiles")
        if isinstance(tournament_context, dict) and isinstance(tournament_context.get("venue_profiles"), dict)
        else {}
    )
    profile = profiles.get(_venue_key(venue)) if isinstance(profiles, dict) else None
    if isinstance(profile, dict):
        scoring_note = "high-scoring" if float(profile.get("goal_multiplier") or 1.0) > 1.015 else "low-scoring"
        if not profile.get("significant"):
            scoring_note = "limited venue sample"
        return {
            **profile,
            "scoring_note": scoring_note,
        }
    return {
        "venue_id": str(venue.get("id") or ""),
        "venue_name": _venue_name(venue),
        "city": _venue_city(venue),
        "games": 0,
        "avg_total": None,
        "tournament_avg_total": (tournament_context or {}).get("overall_avg_total", BASE_WORLD_CUP_TOTAL)
        if isinstance(tournament_context, dict)
        else BASE_WORLD_CUP_TOTAL,
        "goal_multiplier": 1.0,
        "significant": False,
        "scoring_note": "no prior venue sample",
    }


def _feedback_profile(date_iso: str) -> dict[str, Any]:
    stats = {
        "moneyline": {"wins": 0, "losses": 0},
        "total": {"wins": 0, "losses": 0},
        "spread": {"wins": 0, "losses": 0},
    }
    if not MODEL_CACHE_DIR.exists():
        return _feedback_adjustments(stats)
    for path in sorted(MODEL_CACHE_DIR.glob("20*.json")):
        if path.name in {"latest.json", "index.json"} or path.stem >= date_iso:
            continue
        try:
            import json

            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        bucket = (payload.get("models") or {}).get("mls") if isinstance(payload, dict) else {}
        if not isinstance(bucket, dict):
            continue
        for pick in bucket.get("picks") or []:
            if not isinstance(pick, dict):
                continue
            result = str(pick.get("result") or "").lower()
            if result not in {"win", "loss"}:
                continue
            market = str(pick.get("market") or "").lower()
            if market not in stats:
                continue
            stats[market]["wins" if result == "win" else "losses"] += 1
    return _feedback_adjustments(stats)


def _feedback_adjustments(stats: dict[str, dict[str, int]]) -> dict[str, Any]:
    markets: dict[str, dict[str, Any]] = {}
    for market, counts in stats.items():
        wins = int(counts.get("wins") or 0)
        losses = int(counts.get("losses") or 0)
        settled = wins + losses
        hit_rate = wins / settled if settled else None
        markets[market] = {"wins": wins, "losses": losses, "settled": settled, "hit_rate": hit_rate}

    total_hit_rate = markets["total"]["hit_rate"]
    moneyline_hit_rate = markets["moneyline"]["hit_rate"]
    markets["total"]["decision_penalty"] = 0.02 if markets["total"]["settled"] >= 12 and (total_hit_rate or 0.0) < 0.54 else 0.0
    markets["total"]["unit_multiplier"] = 0.85 if markets["total"]["decision_penalty"] else 1.0
    markets["total"]["market_blend"] = 0.18 if markets["total"]["decision_penalty"] else 0.12
    markets["moneyline"]["decision_penalty"] = 0.0
    markets["moneyline"]["unit_multiplier"] = 1.05 if markets["moneyline"]["settled"] >= 8 and (moneyline_hit_rate or 0.0) >= 0.60 else 1.0
    markets["moneyline"]["market_blend"] = 0.0
    markets["spread"]["decision_penalty"] = 0.01
    markets["spread"]["unit_multiplier"] = 0.90
    markets["spread"]["market_blend"] = markets["total"]["market_blend"]
    return {"markets": markets}


class EspnClient:
    def __init__(self, session: requests.Session | None = None, timeout: int = 18):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.timeout = timeout

    def get_json(self, url: str) -> dict[str, Any]:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {}

    def scoreboard(self, date_iso: str) -> dict[str, Any]:
        compact = date_iso.replace("-", "")
        return self.get_json(f"{ESPN_SITE_API}/usa.1/scoreboard?dates={compact}&limit=100")

    def roster(self, team_id: str) -> dict[str, Any]:
        return self.get_json(f"{ESPN_SITE_API}/usa.1/teams/{team_id}/roster")

    def athlete(self, athlete_id: str) -> dict[str, Any]:
        return self.get_json(f"{ESPN_ATHLETE_API}/{athlete_id}")

    def club(self, league_slug: str, club_id: str) -> dict[str, Any]:
        return self.get_json(f"{ESPN_SITE_API}/{league_slug}/teams/{club_id}")


def _parallel_map(items: Iterable[Any], fn, max_workers: int) -> dict[Any, Any]:
    unique = list(dict.fromkeys(items))
    results: dict[Any, Any] = {}
    if not unique:
        return results
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(unique)))) as executor:
        future_map = {executor.submit(fn, item): item for item in unique}
        for future in as_completed(future_map):
            item = future_map[future]
            try:
                results[item] = future.result()
            except Exception:
                results[item] = {}
    return results


def _parse_games(scoreboard: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for event in scoreboard.get("events") if isinstance(scoreboard.get("events"), list) else []:
        competitions = event.get("competitions") if isinstance(event, dict) else []
        competition = competitions[0] if isinstance(competitions, list) and competitions else {}
        competitors = competition.get("competitors") if isinstance(competition, dict) else []
        home = next((item for item in competitors if isinstance(item, dict) and item.get("homeAway") == "home"), None)
        away = next((item for item in competitors if isinstance(item, dict) and item.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        if _status_state(event) == "post":
            continue
        home_team = dict(home.get("team")) if isinstance(home.get("team"), dict) else {}
        away_team = dict(away.get("team")) if isinstance(away.get("team"), dict) else {}
        home_team["current_record"] = _competitor_record(home)
        away_team["current_record"] = _competitor_record(away)
        odds_items = competition.get("odds") if isinstance(competition.get("odds"), list) else []
        venue = competition.get("venue") if isinstance(competition.get("venue"), dict) else {}
        games.append({
            "game_id": str(event.get("id") or ""),
            "start_time": str(event.get("date") or ""),
            "home": home_team,
            "away": away_team,
            "venue": venue,
            "odds": odds_items[0] if odds_items and isinstance(odds_items[0], dict) else {},
        })
    return games


def _closed_market_value(odds: dict[str, Any], market: str, side: str, field: str = "odds") -> Any:
    market_data = odds.get(market) if isinstance(odds.get(market), dict) else {}
    side_data = market_data.get(side) if isinstance(market_data.get(side), dict) else {}
    close = side_data.get("close") if isinstance(side_data.get("close"), dict) else {}
    open_data = side_data.get("open") if isinstance(side_data.get("open"), dict) else {}
    return close.get(field) if close.get(field) not in {"", None} else open_data.get(field)


def _market_probabilities(odds: dict[str, Any]) -> dict[str, float | None]:
    raw = {
        side: _american_implied(_closed_market_value(odds, "moneyline", side))
        for side in ("home", "away", "draw")
    }
    total = sum(value for value in raw.values() if value is not None)
    if total <= 0:
        return raw
    return {
        side: (value / total if value is not None else None)
        for side, value in raw.items()
    }


def _decision(
    probability: float,
    edge: float | None,
    *,
    total: bool = False,
    confidence_penalty: float = 0.0,
) -> str:
    penalty = max(0.0, confidence_penalty)
    if total:
        if probability >= 0.56 + penalty and (edge is None or edge >= 0.025 + (penalty * 0.5)):
            return "BET"
        if probability >= 0.52 + (penalty * 0.5) and (edge is None or edge >= 0.0):
            return "LEAN"
        return "PASS"
    if probability >= 0.48 + (penalty * 0.5) and (edge is None or edge >= 0.025 + (penalty * 0.5)):
        return "BET"
    if probability >= 0.40 + (penalty * 0.5) and (edge is None or edge >= -0.01):
        return "LEAN"
    return "PASS"


def _units(probability: float, edge: float | None, decision: str, multiplier: float = 1.0) -> float:
    if decision == "PASS":
        return 0.0
    value = 0.25 + max(0.0, probability - 0.50) * 2.0 + max(0.0, edge or 0.0) * 3.0
    return round(_clamp(value * _clamp(multiplier, 0.5, 1.15), 0.25, 1.0), 2)


def _top_player_text(team: dict[str, Any]) -> str:
    return ", ".join(
        f"{player['name']} {player['rating']:.1f}"
        for player in team.get("top_players", [])[:3]
    )


def _feedback_market(feedback: dict[str, Any] | None, market: str) -> dict[str, Any]:
    markets = feedback.get("markets") if isinstance(feedback, dict) and isinstance(feedback.get("markets"), dict) else {}
    value = markets.get(market) if isinstance(markets, dict) else None
    return value if isinstance(value, dict) else {}


def _market_blended_xg(
    home_xg: float,
    away_xg: float,
    market_total: float | None,
    blend: float,
) -> tuple[float, float, float, float]:
    raw_total = max(0.01, home_xg + away_xg)
    if market_total is None:
        return home_xg, away_xg, round(raw_total, 2), 0.0
    blended_total = (raw_total * (1.0 - blend)) + (market_total * blend)
    scale = _clamp(blended_total / raw_total, 0.88, 1.12)
    return round(home_xg * scale, 3), round(away_xg * scale, 3), round(blended_total, 2), round(blend, 3)


def _spread_candidates(odds: dict[str, Any], home_xg: float, away_xg: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for side in ("home", "away"):
        line = _number(_closed_market_value(odds, "pointSpread", side, "line"))
        if line is None:
            continue
        odds_value = _american_odds(_closed_market_value(odds, "pointSpread", side))
        probability = spread_cover_probability(home_xg, away_xg, side, line)
        market_probability = _american_implied(odds_value)
        edge = probability - market_probability if market_probability is not None else None
        candidates.append({
            "side": side,
            "line": line,
            "odds": odds_value,
            "probability": probability,
            "market_probability": market_probability,
            "edge": edge,
        })
    return sorted(
        candidates,
        key=lambda item: (
            item["edge"] if item.get("edge") is not None else item["probability"] - 0.50,
            item["probability"],
        ),
        reverse=True,
    )


def _form_factor_text(team: dict[str, Any]) -> str:
    form = team.get("tournament_form") if isinstance(team.get("tournament_form"), dict) else {}
    games = int(form.get("games") or 0)
    if games <= 0:
        return f"{team['team']} tournament form not established"
    return (
        f"{team['team']} form {form.get('record')} "
        f"({float(form.get('points_per_game') or 0):.2f} pts/g, "
        f"{float(form.get('goals_for_per_game') or 0):.2f}-{float(form.get('goals_against_per_game') or 0):.2f} goals/g)"
    )


def _matchup_picks(
    date_iso: str,
    game: dict[str, Any],
    home: dict[str, Any],
    away: dict[str, Any],
    tournament_context: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    venue_profile = _venue_profile_for_game(game, tournament_context)
    venue_multiplier = float(venue_profile.get("goal_multiplier") or 1.0)
    raw_home_xg = expected_goals(home, away, venue_multiplier) + MLS_HOME_GOAL_EDGE
    raw_away_xg = max(0.15, expected_goals(away, home, venue_multiplier) - MLS_AWAY_GOAL_DRAG)
    odds = game.get("odds") if isinstance(game.get("odds"), dict) else {}
    line = _number(odds.get("overUnder")) or 2.5
    total_feedback = _feedback_market(feedback, "total")
    home_xg, away_xg, projected_total, total_market_blend = _market_blended_xg(
        raw_home_xg,
        raw_away_xg,
        line,
        float(total_feedback.get("market_blend") or 0.0),
    )
    probabilities = poisson_probabilities(home_xg, away_xg)
    market = _market_probabilities(odds)
    matchup = f"{away['team']} @ {home['team']}"
    common = {
        "source": "MLS Model",
        "sport": "MLS",
        "league": "Major League Soccer",
        "date": date_iso,
        "game": matchup,
        "matchup": matchup,
        "away_team": away["team"],
        "home_team": home["team"],
        "game_id": game["game_id"],
        "start_time": game["start_time"],
        "game_start_time": game["start_time"],
        "calibration_excluded": True,
        "model_basis": "current squad player power, tournament form, venue scoring, and market context; no head-to-head input",
        "venue": venue_profile.get("venue_name"),
        "venue_id": venue_profile.get("venue_id"),
        "venue_city": venue_profile.get("city"),
        "venue_profile": venue_profile,
        "projected_home_goals": home_xg,
        "projected_away_goals": away_xg,
        "raw_projected_home_goals": raw_home_xg,
        "raw_projected_away_goals": raw_away_xg,
        "projected_total": projected_total,
        "raw_projected_total": round(raw_home_xg + raw_away_xg, 2),
        "market_total_line": line,
        "total_market_blend": total_market_blend,
        "home_unit_ratings": {key: home[key] for key in ("overall", "attack", "midfield", "defense", "goalkeeper")},
        "away_unit_ratings": {key: away[key] for key in ("overall", "attack", "midfield", "defense", "goalkeeper")},
        "home_tournament_form": home.get("tournament_form"),
        "away_tournament_form": away.get("tournament_form"),
    }

    side = "home" if probabilities["home_win"] >= probabilities["away_win"] else "away"
    selected = home if side == "home" else away
    probability = probabilities[f"{side}_win"]
    market_probability = market.get(side)
    edge = probability - market_probability if market_probability is not None else None
    ml_feedback = _feedback_market(feedback, "moneyline")
    ml_decision = _decision(probability, edge, confidence_penalty=float(ml_feedback.get("decision_penalty") or 0.0))
    if not home.get("roster_ready") or not away.get("roster_ready"):
        ml_decision = "PASS"
    ml_odds = _american_odds(_closed_market_value(odds, "moneyline", side))
    ml_reason = (
        f"{selected['team']} owns the stronger current-squad projection. "
        f"Attack {selected['attack']:.1f}, midfield {selected['midfield']:.1f}, "
        f"defense {selected['defense']:.1f}, goalkeeper {selected['goalkeeper']:.1f}. "
        f"Top player ranks: {_top_player_text(selected)}. "
        f"{_form_factor_text(selected)}."
    )
    picks = [{
        **common,
        "pick": f"{selected['team']} ML ({matchup})",
        "team": selected["team"],
        "market": "moneyline",
        "market_type": "soccer_moneyline",
        "odds": ml_odds,
        "probability": round(probability, 4),
        "draw_probability": round(probabilities["draw"], 4),
        "market_probability": round(market_probability, 4) if market_probability is not None else None,
        "edge": round(edge * 100, 2) if edge is not None else None,
        "decision": ml_decision,
        "units": _units(probability, edge, ml_decision, float(ml_feedback.get("unit_multiplier") or 1.0)),
        "reason": ml_reason,
        "key_factors": [
            "Current World Cup roster and availability",
            "Player club-league and club-table power rankings",
            "Current tournament record, goals for, and goals against",
            f"Venue scoring: {venue_profile.get('scoring_note')} at {venue_profile.get('venue_name')}",
            "Attack/midfield/defense/goalkeeper unit matchup",
            "No historical head-to-head input",
        ],
    }]

    over_probability = total_probability(projected_total, line, "over")
    under_probability = total_probability(projected_total, line, "under")
    total_side = "over" if over_probability >= under_probability else "under"
    total_prob = over_probability if total_side == "over" else under_probability
    total_odds = _american_odds(_closed_market_value(odds, "total", total_side))
    total_market_probability = _american_implied(total_odds)
    total_edge = total_prob - total_market_probability if total_market_probability is not None else None
    total_decision = _decision(
        total_prob,
        total_edge,
        total=True,
        confidence_penalty=float(total_feedback.get("decision_penalty") or 0.0),
    )
    if not home.get("roster_ready") or not away.get("roster_ready"):
        total_decision = "PASS"
    picks.append({
        **common,
        "pick": f"{total_side.title()} {line:g} ({matchup})",
        "team": "",
        "market": "total",
        "market_type": "soccer_total",
        "line": line,
        "odds": total_odds,
        "probability": round(total_prob, 4),
        "market_probability": round(total_market_probability, 4) if total_market_probability is not None else None,
        "edge": round(total_edge * 100, 2) if total_edge is not None else None,
        "decision": total_decision,
        "units": _units(total_prob, total_edge, total_decision, float(total_feedback.get("unit_multiplier") or 1.0)),
        "reason": (
            f"Projected goals {away['team']} {away_xg:.2f}, {home['team']} {home_xg:.2f} "
            f"after tournament-form, venue, and market-total blending."
        ),
        "key_factors": [
            f"{away['team']} attack {away['attack']:.1f} vs {home['team']} defense {home['defense']:.1f}",
            f"{home['team']} attack {home['attack']:.1f} vs {away['team']} defense {away['defense']:.1f}",
            _form_factor_text(away),
            _form_factor_text(home),
            f"Venue scoring: {venue_profile.get('scoring_note')} ({venue_profile.get('goal_multiplier'):.3f}x)",
            f"Projected total {projected_total:.2f}",
        ],
    })

    spread_feedback = _feedback_market(feedback, "spread")
    spread_options = _spread_candidates(odds, home_xg, away_xg)
    if spread_options:
        spread = spread_options[0]
        spread_side = str(spread["side"])
        spread_team = home if spread_side == "home" else away
        spread_probability = float(spread["probability"])
        spread_market_probability = spread.get("market_probability")
        spread_edge = spread.get("edge") if isinstance(spread.get("edge"), float) else None
        spread_decision = _decision(
            spread_probability,
            spread_edge,
            total=True,
            confidence_penalty=float(spread_feedback.get("decision_penalty") or 0.0),
        )
        if not home.get("roster_ready") or not away.get("roster_ready"):
            spread_decision = "PASS"
        line_label = f"{float(spread['line']):+g}"
        picks.append({
            **common,
            "pick": f"{spread_team['team']} {line_label} ({matchup})",
            "team": spread_team["team"],
            "market": "spread",
            "market_type": "soccer_handicap",
            "line": spread["line"],
            "odds": spread["odds"],
            "probability": round(spread_probability, 4),
            "market_probability": round(spread_market_probability, 4) if spread_market_probability is not None else None,
            "edge": round(spread_edge * 100, 2) if spread_edge is not None else None,
            "decision": spread_decision,
            "units": _units(spread_probability, spread_edge, spread_decision, float(spread_feedback.get("unit_multiplier") or 1.0)),
            "reason": (
                f"{spread_team['team']} has the best spread-cover edge at {line_label} from the venue-adjusted "
                f"Poisson score grid."
            ),
            "key_factors": [
                f"Projected score {away['team']} {away_xg:.2f}, {home['team']} {home_xg:.2f}",
                f"Spread line {spread_team['team']} {line_label}",
                _form_factor_text(spread_team),
                f"Venue scoring: {venue_profile.get('scoring_note')} ({venue_profile.get('goal_multiplier'):.3f}x)",
            ],
        })

    game_summary = {
        **common,
        "home_win_probability": round(probabilities["home_win"], 4),
        "draw_probability": round(probabilities["draw"], 4),
        "away_win_probability": round(probabilities["away_win"], 4),
    }
    return picks, game_summary


def generate_mls_picks(
    date_str: str | None = None,
    *,
    client: EspnClient | None = None,
    max_workers: int = 24,
) -> dict[str, Any]:
    """Generate a cache-ready MLS model bucket."""
    date_iso = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d") if date_str else datetime.now().strftime("%Y-%m-%d")
    api = client or EspnClient()
    scoreboard = api.scoreboard(date_iso)
    games = _parse_games(scoreboard)
    tournament_context = _build_tournament_context(date_iso, api)
    feedback = _feedback_profile(date_iso)
    if not games:
        return {
            "ok": True,
            "date": date_iso,
            "picks": [],
            "games": [],
            "team_ratings": [],
            "player_rankings": [],
            "calibration_excluded": True,
            "tournament_context": tournament_context,
            "model_feedback": feedback,
            "note": f"No MLS games on ESPN for {date_iso}.",
        }

    teams = {
        str(team.get("id") or ""): team
        for game in games
        for team in (game["home"], game["away"])
        if str(team.get("id") or "")
    }
    rosters = _parallel_map(teams, lambda team_id: api.roster(team_id), max_workers)
    raw_players = {
        team_id: [
            player for player in (rosters.get(team_id, {}).get("athletes") or [])
            if isinstance(player, dict)
        ]
        for team_id in teams
    }
    athlete_ids = [
        str(player.get("id") or "")
        for players in raw_players.values()
        for player in players
        if str(player.get("id") or "")
    ]
    profiles = _parallel_map(athlete_ids, lambda athlete_id: api.athlete(athlete_id), max_workers)

    club_keys: list[tuple[str, str]] = []
    for profile in profiles.values():
        athlete = profile.get("athlete") if isinstance(profile.get("athlete"), dict) else {}
        club = athlete.get("team") if isinstance(athlete.get("team"), dict) else {}
        league = profile.get("league") if isinstance(profile.get("league"), dict) else {}
        league_slug = str(league.get("slug") or "").lower()
        club_id = str(club.get("id") or "")
        # MLS players' clubs ARE usa.1 teams — always rate by club table
        # record (the WC engine skipped same-league clubs because WC
        # squads play for clubs in other leagues).
        if league_slug and club_id:
            club_keys.append((league_slug, club_id))
    clubs = _parallel_map(club_keys, lambda key: api.club(key[0], key[1]), max_workers)

    ratings: dict[str, dict[str, Any]] = {}
    for team_id, team in teams.items():
        players: list[dict[str, Any]] = []
        for raw_player in raw_players.get(team_id, []):
            athlete_id = str(raw_player.get("id") or "")
            profile = profiles.get(athlete_id) if athlete_id else {}
            athlete = profile.get("athlete") if isinstance(profile, dict) and isinstance(profile.get("athlete"), dict) else {}
            club_team = athlete.get("team") if isinstance(athlete.get("team"), dict) else {}
            league = profile.get("league") if isinstance(profile, dict) and isinstance(profile.get("league"), dict) else {}
            club_key = (str(league.get("slug") or "").lower(), str(club_team.get("id") or ""))
            club_payload = clubs.get(club_key) if club_key in clubs else {}
            club_data = club_payload.get("team") if isinstance(club_payload, dict) and isinstance(club_payload.get("team"), dict) else {}
            players.append(player_power(raw_player, profile, club_data))
        ratings[team_id] = team_power(team, players, tournament_context)

    all_players: list[dict[str, Any]] = []
    for rating in ratings.values():
        for player in rating["players"]:
            all_players.append({**player, "national_team": rating["team"]})
    all_players.sort(key=lambda player: float(player.get("rating") or 0), reverse=True)
    for index, player in enumerate(all_players, start=1):
        player["slate_rank"] = index

    picks: list[dict[str, Any]] = []
    game_summaries: list[dict[str, Any]] = []
    for game in games:
        home_id = str(game["home"].get("id") or "")
        away_id = str(game["away"].get("id") or "")
        if home_id not in ratings or away_id not in ratings:
            continue
        game_picks, summary = _matchup_picks(
            date_iso,
            game,
            ratings[home_id],
            ratings[away_id],
            tournament_context,
            feedback,
        )
        picks.extend(game_picks)
        game_summaries.append(summary)

    team_ratings = sorted(ratings.values(), key=lambda team: float(team["overall"]), reverse=True)
    for index, rating in enumerate(team_ratings, start=1):
        rating["slate_rank"] = index
        rating.pop("players", None)
    return {
        "ok": True,
        "date": date_iso,
        "model": "MLSPlayerPower",
        "picks": picks,
        "games": game_summaries,
        "team_ratings": team_ratings,
        "player_rankings": all_players[:75],
        "calibration_excluded": True,
        "tournament_context": {
            "completed_games": tournament_context.get("completed_games"),
            "overall_avg_total": tournament_context.get("overall_avg_total"),
            "lookback_days": tournament_context.get("lookback_days"),
            "venue_count": len(tournament_context.get("venue_profiles") or {}),
            "team_record_count": len(tournament_context.get("team_records") or {}),
        },
        "model_feedback": feedback,
        "schedule_source": "ESPN MLS scoreboard",
        "player_rating_source": "ESPN tournament rosters, player current clubs/leagues, club table records, current World Cup records, and venue scoring history",
        "note": (
            f"Rated {len(all_players)} current-squad players across {len(team_ratings)} teams; "
            f"generated {len(picks)} moneyline/spread/total rows without head-to-head or trained-model inputs."
        ),
    }
