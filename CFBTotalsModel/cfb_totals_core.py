"""Data spine and feature builder for the CFB market-residual totals model.

This model is deliberately NOT market-free. Where `CFBPredictionModel` is an
originator that re-derives a game from team form and is then compared to the
closing line, this model takes the posted total as an INPUT and predicts how far
the game will land from it. Walk-forward evidence is what motivated the split:
the market-free originator sits at coin flip against the line (~0.49-0.51
direction), while the residual formulation clears the -110 break-even once a
confidence threshold is applied.

Because the two contracts are incompatible -- one forbids market features, the
other requires them -- this lives in its own package rather than mutating the
originator's audited feature contract.

Team-state construction reuses the originator's strict as-of discipline: a game's
features are emitted BEFORE its own result updates any team state, so no row can
see its own outcome or any later game.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from CFBPredictionModel.cfb_core import (
    _is_fbs_training_game,
    _num,
    _parse_time,
    _text,
    load_schedule_season,
    load_training_rows,
)

# Team-form features (identical in spirit to the originator's) ...
FORM_FEATURES = [
    "home_offense_ewma",
    "home_defense_ewma",
    "away_offense_ewma",
    "away_defense_ewma",
    "net_efficiency_diff",
    "elo_diff",
    "schedule_strength_diff",
    "home_games_log",
    "away_games_log",
    "home_rest_days",
    "away_rest_days",
    "rest_diff",
    "neutral_site",
    "conference_game",
    "week",
    "early_season",
]
# ... plus the market anchors, which are the point of this model.
MARKET_FEATURES = ["home_line", "total_line"]
FEATURE_NAMES = FORM_FEATURES + MARKET_FEATURES

LEAGUE_POINTS = 28.0
LEAGUE_ELO = 1500.0
EWMA_DECAY = 0.5 ** (1.0 / 5.0)
OFFSEASON_DECAY = 0.58
ELO_K = 22.0


class TeamState:
    """Running team form. Mirrors the originator so the two stay comparable."""

    __slots__ = ("offense", "defense", "elo", "opponent_elo", "games", "season", "last_date")

    def __init__(self) -> None:
        self.offense = LEAGUE_POINTS
        self.defense = LEAGUE_POINTS
        self.elo = LEAGUE_ELO
        self.opponent_elo = LEAGUE_ELO
        self.games = 0
        self.season = 0
        self.last_date = None

    def roll_season(self, season: int) -> None:
        if self.season and season != self.season:
            for attr, mean in (("offense", LEAGUE_POINTS), ("defense", LEAGUE_POINTS),
                               ("elo", LEAGUE_ELO), ("opponent_elo", LEAGUE_ELO)):
                setattr(self, attr, mean + (getattr(self, attr) - mean) * OFFSEASON_DECAY)
            self.games = 0
            self.last_date = None
        self.season = season


def _rest_days(state: TeamState, start) -> float:
    if state.last_date is None or start is None:
        return 14.0
    return min(21.0, max(3.0, float((start.date() - state.last_date.date()).days)))


def feature_row(game: Mapping[str, Any], home: TeamState, away: TeamState) -> dict[str, float]:
    start = _parse_time(game.get("start_time"))
    neutral = 1.0 if game.get("neutral_site") is True else 0.0
    home_rest = _rest_days(home, start)
    away_rest = _rest_days(away, start)
    return {
        "home_offense_ewma": home.offense,
        "home_defense_ewma": home.defense,
        "away_offense_ewma": away.offense,
        "away_defense_ewma": away.defense,
        "net_efficiency_diff": (home.offense - home.defense) - (away.offense - away.defense),
        "elo_diff": home.elo - away.elo + (0.0 if neutral else 55.0),
        "schedule_strength_diff": home.opponent_elo - away.opponent_elo,
        "home_games_log": math.log1p(home.games),
        "away_games_log": math.log1p(away.games),
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_diff": home_rest - away_rest,
        "neutral_site": neutral,
        "conference_game": 1.0 if game.get("conference_game") is True else 0.0,
        "week": float(_num(game.get("week"), 1.0) or 1.0),
        "early_season": 1.0 if int(_num(game.get("week"), 1) or 1) <= 4 else 0.0,
        "home_line": float(_num(game.get("home_line"), 0.0) or 0.0),
        "total_line": float(_num(game.get("total_line"), 0.0) or 0.0),
    }


def update_states(game: Mapping[str, Any], home: TeamState, away: TeamState) -> None:
    hs, aw = _num(game.get("home_score")), _num(game.get("away_score"))
    if hs is None or aw is None:
        return
    neutral = game.get("neutral_site") is True
    expected = 1.0 / (1.0 + 10.0 ** (-((home.elo - away.elo) + (0.0 if neutral else 55.0)) / 400.0))
    result = 1.0 if hs > aw else 0.0 if hs < aw else 0.5
    delta = ELO_K * math.log(abs(hs - aw) + 1.0) * 1.35 * (result - expected)
    old_home, old_away = home.elo, away.elo
    home.elo += delta
    away.elo -= delta
    home.opponent_elo = home.opponent_elo * EWMA_DECAY + old_away * (1 - EWMA_DECAY)
    away.opponent_elo = away.opponent_elo * EWMA_DECAY + old_home * (1 - EWMA_DECAY)
    home.offense = home.offense * EWMA_DECAY + hs * (1 - EWMA_DECAY)
    home.defense = home.defense * EWMA_DECAY + aw * (1 - EWMA_DECAY)
    away.offense = away.offense * EWMA_DECAY + aw * (1 - EWMA_DECAY)
    away.defense = away.defense * EWMA_DECAY + hs * (1 - EWMA_DECAY)
    start = _parse_time(game.get("start_time"))
    home.last_date = start or home.last_date
    away.last_date = start or away.last_date
    home.games += 1
    away.games += 1


def build_dataset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Priced FBS-vs-FBS rows with the residual targets, strictly as-of."""
    states: dict[str, TeamState] = {}
    records: list[dict[str, Any]] = []
    for game in sorted(rows, key=lambda r: (r["start_time"], r["game_id"])):
        home_id, away_id = _text(game.get("home_team_id")), _text(game.get("away_team_id"))
        if not home_id or not away_id:
            continue
        season = int(_num(game.get("season"), 0) or 0)
        home = states.setdefault(home_id, TeamState())
        away = states.setdefault(away_id, TeamState())
        home.roll_season(season)
        away.roll_season(season)
        if game.get("completed") is True and _is_fbs_training_game(game):
            hs, aw = _num(game.get("home_score")), _num(game.get("away_score"))
            hl, tl = _num(game.get("home_line")), _num(game.get("total_line"))
            if None not in (hs, aw, hl, tl):
                records.append({
                    "game_id": game["game_id"],
                    "season": season,
                    "features": feature_row(game, home, away),
                    "home_margin": float(hs - aw),
                    "game_total": float(hs + aw),
                    "home_line": float(hl),
                    "total_line": float(tl),
                    # the model's targets: signed distance from each posted line
                    "total_residual": float(hs + aw) - float(tl),
                    # positive => home covered (home_line is negative when favored)
                    "margin_residual": float(hs - aw) + float(hl),
                })
        if game.get("completed") is True:
            update_states(game, home, away)
    return records


def features_for_slate(history: list[dict[str, Any]], slate: list[dict[str, Any]]):
    """Slate features folded only from games that kicked off earlier."""
    target_ids = {_text(g.get("game_id")) for g in slate}
    combined = sorted([*history, *slate],
                      key=lambda r: (_text(r.get("start_time")), _text(r.get("game_id"))))
    states: dict[str, TeamState] = {}
    out: list[dict[str, Any]] = []
    for game in combined:
        home_id, away_id = _text(game.get("home_team_id")), _text(game.get("away_team_id"))
        if not home_id or not away_id:
            continue
        season = int(_num(game.get("season"), 0) or 0)
        home = states.setdefault(home_id, TeamState())
        away = states.setdefault(away_id, TeamState())
        home.roll_season(season)
        away.roll_season(season)
        if _text(game.get("game_id")) in target_ids:
            out.append({"game": game, "features": feature_row(game, home, away)})
            continue
        if game.get("completed") is True:
            update_states(game, home, away)
    return out


def matrix(records: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(r["features"][name]) for name in FEATURE_NAMES] for r in records]


def load_history(season: int) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for year in range(season - 2, season + 1):
        history.extend(load_schedule_season(year, refresh=(year == season)))
    return history


def known_fbs_ids(history: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for g in history:
        if g.get("home_division") == "fbs":
            ids.add(_text(g.get("home_team_id")))
        if g.get("away_division") == "fbs":
            ids.add(_text(g.get("away_team_id")))
    return ids


__all__ = [
    "FEATURE_NAMES", "FORM_FEATURES", "MARKET_FEATURES", "TeamState",
    "build_dataset", "feature_row", "features_for_slate", "matrix",
    "load_history", "known_fbs_ids", "load_training_rows", "update_states",
]
