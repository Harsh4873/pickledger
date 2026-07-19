"""NFL data spine and as-of feature builder.

Data source: the nflverse games dataset (github.com/nflverse/nfldata) —
every game 1999-present with final scores, market spread/total/moneylines,
rest days, roof, and starting QBs. One keyless CSV covers training targets,
market anchors, and the current season's schedule with posted lines.

Features are built strictly as-of each game: the chronological pass updates
team state only AFTER emitting a game's feature row, so nothing a model
trains on could have been known after kickoff. Team strength is an EWMA of
points scored/allowed (half-life ~6 games, decayed extra across seasons) —
a deliberate Phase-1 proxy for play-by-play EPA, which lands in Phase 2.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
GAMES_CSV_PATH = REPO_ROOT / "data" / "nfl" / "games.csv"
GAMES_CSV_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
GAMES_TTL_SECONDS = 24 * 60 * 60

EWMA_DECAY = 0.5 ** (1.0 / 6.0)  # half-life of six games
SEASON_GAP_DECAY = 0.6
LEAGUE_AVG_POINTS = 22.0

FEATURE_NAMES = [
    "home_off_ewma", "home_def_ewma", "away_off_ewma", "away_def_ewma",
    "net_rating_diff", "home_rest", "away_rest", "rest_diff",
    "div_game", "roof_dome", "week", "home_qb_change", "away_qb_change",
    "spread_line", "total_line", "covid_season",
]


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_games(refresh: bool = True) -> list[dict[str, Any]]:
    """Load (and lazily refresh) the nflverse games file."""
    path = GAMES_CSV_PATH
    # Freshness rides on a download stamp, not file mtime — a CI checkout
    # resets mtimes, which would leave in-season results permanently stale.
    stamp = path.with_suffix(".stamp")
    stamp_age = (time.time() - stamp.stat().st_mtime) if stamp.exists() else None
    stale = not path.exists() or stamp_age is None or stamp_age > GAMES_TTL_SECONDS
    if refresh and stale:
        try:
            response = requests.get(GAMES_CSV_URL, timeout=60)
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text, encoding="utf-8")
            stamp.write_text(str(time.time()), encoding="utf-8")
        except Exception as exc:  # keep serving from the cached copy
            print(f"[nfl] games.csv refresh failed ({exc}); using cached copy")
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: (str(r.get("gameday") or ""), str(r.get("game_id") or "")))
    return rows


class TeamState:
    __slots__ = ("off_ewma", "def_ewma", "games", "season", "qb_id")

    def __init__(self) -> None:
        self.off_ewma = LEAGUE_AVG_POINTS
        self.def_ewma = LEAGUE_AVG_POINTS
        self.games = 0
        self.season = 0
        self.qb_id = ""

    def roll_season(self, season: int) -> None:
        if self.season and season != self.season:
            # Shrink accumulated strength toward league average across the
            # offseason — rosters churn, but strength persists partially.
            self.off_ewma = LEAGUE_AVG_POINTS + (self.off_ewma - LEAGUE_AVG_POINTS) * SEASON_GAP_DECAY
            self.def_ewma = LEAGUE_AVG_POINTS + (self.def_ewma - LEAGUE_AVG_POINTS) * SEASON_GAP_DECAY
        self.season = season

    def update(self, points_for: float, points_against: float) -> None:
        self.off_ewma = self.off_ewma * EWMA_DECAY + points_for * (1.0 - EWMA_DECAY)
        self.def_ewma = self.def_ewma * EWMA_DECAY + points_against * (1.0 - EWMA_DECAY)
        self.games += 1


def _feature_row(game: dict[str, Any], home: TeamState, away: TeamState) -> dict[str, Any] | None:
    spread_line = _num(game.get("spread_line"))
    total_line = _num(game.get("total_line"))
    if spread_line is None or total_line is None:
        return None
    season = int(_num(game.get("season"), 0) or 0)
    week = _num(game.get("week"), 1.0) or 1.0
    roof = str(game.get("roof") or "").strip().lower()
    home_qb = str(game.get("home_qb_id") or "").strip()
    away_qb = str(game.get("away_qb_id") or "").strip()
    features = {
        "home_off_ewma": home.off_ewma,
        "home_def_ewma": home.def_ewma,
        "away_off_ewma": away.off_ewma,
        "away_def_ewma": away.def_ewma,
        "net_rating_diff": (home.off_ewma - home.def_ewma) - (away.off_ewma - away.def_ewma),
        "home_rest": _num(game.get("home_rest"), 7.0) or 7.0,
        "away_rest": _num(game.get("away_rest"), 7.0) or 7.0,
        "rest_diff": (_num(game.get("home_rest"), 7.0) or 7.0) - (_num(game.get("away_rest"), 7.0) or 7.0),
        "div_game": _num(game.get("div_game"), 0.0) or 0.0,
        "roof_dome": 1.0 if roof in {"dome", "closed"} else 0.0,
        "week": week,
        "home_qb_change": 1.0 if (home.qb_id and home_qb and home_qb != home.qb_id) else 0.0,
        "away_qb_change": 1.0 if (away.qb_id and away_qb and away_qb != away.qb_id) else 0.0,
        "spread_line": spread_line,
        "total_line": total_line,
        "covid_season": 1.0 if season == 2020 else 0.0,
    }
    return features


def build_dataset(
    rows: list[dict[str, Any]],
    first_season: int = 2002,
    last_season: int | None = None,
) -> list[dict[str, Any]]:
    """Chronological pass emitting one as-of feature record per finished game."""
    states: dict[str, TeamState] = {}
    records: list[dict[str, Any]] = []
    for game in rows:
        season = int(_num(game.get("season"), 0) or 0)
        if season < 1999:
            continue
        game_type = str(game.get("game_type") or "").upper()
        home_key = str(game.get("home_team") or "")
        away_key = str(game.get("away_team") or "")
        if not home_key or not away_key:
            continue
        home = states.setdefault(home_key, TeamState())
        away = states.setdefault(away_key, TeamState())
        home.roll_season(season)
        away.roll_season(season)

        home_score = _num(game.get("home_score"))
        away_score = _num(game.get("away_score"))
        finished = home_score is not None and away_score is not None and str(game.get("result") or "") != ""

        if (
            finished
            and season >= first_season
            and (last_season is None or season <= last_season)
            and game_type in {"REG", "POST", "WC", "DIV", "CON", "SB"}
        ):
            features = _feature_row(game, home, away)
            if features is not None:
                margin = home_score - away_score
                records.append({
                    "game_id": str(game.get("game_id") or ""),
                    "season": season,
                    "features": features,
                    "home_win": 1 if margin > 0 else 0,
                    "margin_residual": margin - features["spread_line"],
                    "total_residual": (home_score + away_score) - features["total_line"],
                })

        # State updates happen strictly after the feature row is emitted.
        if finished:
            home.update(home_score, away_score)
            away.update(away_score, home_score)
            home_qb = str(game.get("home_qb_id") or "").strip()
            away_qb = str(game.get("away_qb_id") or "").strip()
            if home_qb:
                home.qb_id = home_qb
            if away_qb:
                away.qb_id = away_qb
    return records


def features_for_date(rows: list[dict[str, Any]], date_iso: str) -> list[dict[str, Any]]:
    """Feature rows for the given slate date using only prior history."""
    states: dict[str, TeamState] = {}
    slate: list[dict[str, Any]] = []
    for game in rows:
        gameday = str(game.get("gameday") or "")
        home_key = str(game.get("home_team") or "")
        away_key = str(game.get("away_team") or "")
        if not home_key or not away_key:
            continue
        season = int(_num(game.get("season"), 0) or 0)
        home = states.setdefault(home_key, TeamState())
        away = states.setdefault(away_key, TeamState())
        home.roll_season(season)
        away.roll_season(season)
        if gameday == date_iso:
            features = _feature_row(game, home, away)
            if features is not None:
                slate.append({"game": game, "features": features})
            continue
        home_score = _num(game.get("home_score"))
        away_score = _num(game.get("away_score"))
        if gameday < date_iso and home_score is not None and away_score is not None:
            home.update(home_score, away_score)
            away.update(away_score, home_score)
            home_qb = str(game.get("home_qb_id") or "").strip()
            away_qb = str(game.get("away_qb_id") or "").strip()
            if home_qb:
                home.qb_id = home_qb
            if away_qb:
                away.qb_id = away_qb
    return slate


def matrix(records: list[dict[str, Any]]) -> list[list[float]]:
    return [[float(rec["features"][name]) for name in FEATURE_NAMES] for rec in records]
