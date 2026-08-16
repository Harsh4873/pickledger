"""Research-only, player-level lineup rating features for MLB.

This module deliberately does *not* implement a player Elo, fantasy-points
rank, or a hand-tuned win-probability adjustment.  It turns the nine confirmed
hitters in a lineup into a dated, empirical-Bayes batting rating that a
separate challenger model can evaluate against the existing MLB New stack.

The production ``new`` artifact does not import this module.  Keeping the
schema separate prevents an unvalidated research feature from changing live
pick output.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


RATING_SCHEMA = "mlb_lineup_player_rating_v1"

# These are deliberately stable, league-level priors rather than a claim that
# the weighted score below is official season-specific wOBA.  The score is an
# interpretable run-value proxy whose only job is to rank a hitter relative to
# the same league baseline before the game begins.
LEAGUE_BATTING_RATING = 0.320
LEAGUE_POWER_RATING = 0.390
BATTING_PRIOR_PLATE_APPEARANCES = 180.0
PRIOR_SEASON_RELIABILITY_PLATE_APPEARANCES = 450.0

# Expected plate appearances fall modestly through the batting order.  Values
# are normalized by ``_weighted_mean`` so their absolute scale is irrelevant.
LINEUP_ORDER_WEIGHTS = (1.15, 1.13, 1.11, 1.08, 1.04, 0.98, 0.94, 0.90, 0.87)
CONFIRMED_LINEUP_SIZE = len(LINEUP_ORDER_WEIGHTS)

_BATTING_COUNT_FIELDS = (
    "runs",
    "doubles",
    "triples",
    "homeRuns",
    "strikeOuts",
    "baseOnBalls",
    "hits",
    "hitByPitch",
    "atBats",
    "stolenBases",
    "groundIntoDoublePlay",
    "groundIntoTriplePlay",
    "plateAppearances",
    "totalBases",
    "rbi",
    "leftOnBase",
    "sacBunts",
    "sacFlies",
    "catchersInterference",
    "pickoffs",
    "flyOuts",
    "groundOuts",
    "airOuts",
    "popOuts",
    "lineOuts",
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-", "-.--"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _stat_payload(entry: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Accept either a StatsAPI ``split`` or its nested ``stat`` object."""
    if not entry:
        return {}
    nested = entry.get("stat")
    return nested if isinstance(nested, Mapping) else entry


def _plate_appearances(stat: Mapping[str, Any]) -> float:
    pa = _as_float(stat.get("plateAppearances"))
    if pa > 0:
        return pa
    return max(
        0.0,
        _as_float(stat.get("atBats"))
        + _as_float(stat.get("baseOnBalls"))
        + _as_float(stat.get("hitByPitch"))
        + _as_float(stat.get("sacFlies"))
        + _as_float(stat.get("sacBunts")),
    )


def _batting_value(stat: Mapping[str, Any]) -> float:
    """Return a simple weighted-on-base run-value proxy per plate appearance."""
    pa = _plate_appearances(stat)
    if pa <= 0:
        return LEAGUE_BATTING_RATING

    hits = _as_float(stat.get("hits"))
    doubles = _as_float(stat.get("doubles"))
    triples = _as_float(stat.get("triples"))
    home_runs = _as_float(stat.get("homeRuns"))
    singles = max(0.0, hits - doubles - triples - home_runs)
    walks = _as_float(stat.get("baseOnBalls"))
    hbp = _as_float(stat.get("hitByPitch"))

    weighted_total = (
        0.69 * walks
        + 0.72 * hbp
        + 0.88 * singles
        + 1.25 * doubles
        + 1.58 * triples
        + 2.03 * home_runs
    )
    return weighted_total / pa


def _power_value(stat: Mapping[str, Any]) -> float:
    at_bats = _as_float(stat.get("atBats"))
    if at_bats <= 0:
        return LEAGUE_POWER_RATING
    return _as_float(stat.get("totalBases")) / at_bats


def _shrink(value: float, prior: float, sample: float, prior_sample: float) -> float:
    if sample <= 0:
        return prior
    weight = sample / (sample + prior_sample)
    return weight * value + (1.0 - weight) * prior


def pregame_batting_stat(
    season_stat: Mapping[str, Any] | None,
    game_stat: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a pregame batting line from a final-game StatsAPI box score.

    Historical feeds expose a player's season-to-date line after the game has
    finished.  Removing that game's batting line makes the value available
    before first pitch.  Live callers pass ``game_stat=None`` because their
    season line is already pregame.
    """
    result = dict(_stat_payload(season_stat))
    if not game_stat:
        return result

    game = _stat_payload(game_stat)
    for key in _BATTING_COUNT_FIELDS:
        result[key] = max(0.0, _as_float(result.get(key)) - _as_float(game.get(key)))
    result["gamesPlayed"] = max(
        0.0,
        _as_float(result.get("gamesPlayed")) - (1.0 if _plate_appearances(game) > 0 else 0.0),
    )
    return result


def batting_rating(
    pregame_batting: Mapping[str, Any] | None,
    prior_batting: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Create a shrinkage-based player batting and power rating.

    Current-season production is pulled toward a prior-season estimate, which
    is itself pulled toward the league baseline.  This prevents a small-sample
    call-up or one-game spike from dominating an otherwise strong lineup.
    """
    current = _stat_payload(pregame_batting)
    prior = _stat_payload(prior_batting)
    current_pa = _plate_appearances(current)
    prior_pa = _plate_appearances(prior)

    prior_reliability = min(1.0, prior_pa / PRIOR_SEASON_RELIABILITY_PLATE_APPEARANCES)
    prior_batting_value = _batting_value(prior)
    prior_power_value = _power_value(prior)
    prior_center_batting = (
        prior_reliability * prior_batting_value
        + (1.0 - prior_reliability) * LEAGUE_BATTING_RATING
    )
    prior_center_power = (
        prior_reliability * prior_power_value
        + (1.0 - prior_reliability) * LEAGUE_POWER_RATING
    )

    current_batting_value = _batting_value(current)
    current_power_value = _power_value(current)
    reliability = current_pa / (current_pa + BATTING_PRIOR_PLATE_APPEARANCES)

    return {
        "batting_rating": _shrink(
            current_batting_value,
            prior_center_batting,
            current_pa,
            BATTING_PRIOR_PLATE_APPEARANCES,
        ),
        "power_rating": _shrink(
            current_power_value,
            prior_center_power,
            current_pa,
            BATTING_PRIOR_PLATE_APPEARANCES,
        ),
        "reliability": reliability,
        "sample_pa": current_pa,
        "prior_sample_pa": prior_pa,
        "has_observation": float(current_pa > 0 or prior_pa > 0),
    }


def _weighted_mean(values: Sequence[float], weights: Sequence[float], default: float) -> float:
    if not values or not weights:
        return default
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return default
    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight)


def lineup_batting_rating(
    players: Sequence[Mapping[str, Any]],
    *,
    source: str,
) -> dict[str, Any]:
    """Aggregate an ordered, confirmed nine-player lineup.

    A missing batting order is intentionally marked unavailable rather than
    replaced with a roster average.  That lets a serving caller withhold the
    challenger while the established team-level model remains available.
    """
    lineup = list(players[:CONFIRMED_LINEUP_SIZE])
    player_count = len(lineup)
    if player_count < CONFIRMED_LINEUP_SIZE:
        return {
            "batting_rating": LEAGUE_BATTING_RATING,
            "power_rating": LEAGUE_POWER_RATING,
            "reliability": 0.0,
            "coverage": player_count / CONFIRMED_LINEUP_SIZE,
            "available": 0.0,
            "player_count": float(player_count),
            "source": source,
        }

    ratings: list[dict[str, float]] = []
    for player in lineup:
        ratings.append(
            batting_rating(
                player.get("pregame_batting"),
                player.get("prior_batting"),
            )
        )

    weights = LINEUP_ORDER_WEIGHTS[:player_count]
    return {
        "batting_rating": _weighted_mean(
            [rating["batting_rating"] for rating in ratings],
            weights,
            LEAGUE_BATTING_RATING,
        ),
        "power_rating": _weighted_mean(
            [rating["power_rating"] for rating in ratings],
            weights,
            LEAGUE_POWER_RATING,
        ),
        "reliability": _weighted_mean(
            [rating["reliability"] for rating in ratings], weights, 0.0
        ),
        "coverage": _weighted_mean(
            [rating["has_observation"] for rating in ratings], weights, 0.0
        ),
        "available": 1.0,
        "player_count": float(player_count),
        "source": source,
    }


def lineup_batting_rating_from_boxscore(
    team_box: Mapping[str, Any] | None,
    prior_stats_by_player: Mapping[int, Mapping[str, Any]] | None,
    *,
    subtract_current_game: bool,
    source: str,
) -> dict[str, Any]:
    """Build a lineup rating from the same StatsAPI boxscore shape live uses."""
    team_box = team_box or {}
    players = team_box.get("players") or {}
    batting_order = list(team_box.get("battingOrder") or [])[:CONFIRMED_LINEUP_SIZE]
    prior_stats_by_player = prior_stats_by_player or {}

    ordered_players: list[dict[str, Any]] = []
    for player_id in batting_order:
        try:
            normalized_id = int(player_id)
        except (TypeError, ValueError):
            continue
        player = players.get(f"ID{normalized_id}") or {}
        season_batting = (player.get("seasonStats") or {}).get("batting") or {}
        game_batting = (player.get("stats") or {}).get("batting") or {}
        ordered_players.append(
            {
                "pregame_batting": pregame_batting_stat(
                    season_batting,
                    game_batting if subtract_current_game else None,
                ),
                "prior_batting": _stat_payload(prior_stats_by_player.get(normalized_id)),
            }
        )

    return lineup_batting_rating(ordered_players, source=source)


def matchup_player_rating_features(
    home_lineup: Mapping[str, Any],
    away_lineup: Mapping[str, Any],
) -> dict[str, float]:
    """Return model-ready, home-minus-away challenger features."""
    home = home_lineup or {}
    away = away_lineup or {}
    return {
        "lineup_batting_rating_adv": _as_float(home.get("batting_rating"), LEAGUE_BATTING_RATING)
        - _as_float(away.get("batting_rating"), LEAGUE_BATTING_RATING),
        "lineup_power_rating_adv": _as_float(home.get("power_rating"), LEAGUE_POWER_RATING)
        - _as_float(away.get("power_rating"), LEAGUE_POWER_RATING),
        "lineup_rating_reliability_adv": _as_float(home.get("reliability"))
        - _as_float(away.get("reliability")),
        "lineup_rating_coverage_min": min(
            _as_float(home.get("coverage")), _as_float(away.get("coverage"))
        ),
        "lineup_rating_available": min(
            _as_float(home.get("available")), _as_float(away.get("available"))
        ),
    }
