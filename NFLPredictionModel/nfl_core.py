"""NFL data spine and as-of feature builder.

Data sources (all keyless, all nflverse):

* ``games.csv`` (github.com/nflverse/nfldata) — every game 1999-present with
  final scores, posted spread/total/moneyline prices, rest days, roof, and the
  projected/actual starting QBs.  It is the training target set, the market
  anchor, and the current-season slate with posted lines.
* ``stats_team_week_{season}.csv`` (nflverse-data releases) — per-team,
  per-game passing/rushing EPA, attempts, carries, sacks, and CPOE.  These give
  the EPA/play team-strength features the Phase-1 model lacked.

Features are built strictly as-of each game: the chronological pass emits a
game's feature row BEFORE folding its result into team state, so nothing a
model trains on could have been known after kickoff.  Team strength is an
Elo rating (margin-of-victory adjusted, regressed toward the mean across the
offseason) plus EWMA EPA/play on offense and defense (half-life six games,
halved across the offseason).  Posted market lines are features on purpose:
every head is a market-anchored residual model, which is the only design that
survived the walk-forward tests recorded in ``artifacts/metadata.json``.
"""
from __future__ import annotations

import csv
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "nfl"
GAMES_CSV_PATH = DATA_DIR / "games.csv"
GAMES_CSV_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
TEAM_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_team/"
    "stats_team_week_{season}.csv"
)
GAMES_TTL_SECONDS = 24 * 60 * 60
TEAM_STATS_TTL_SECONDS = 6 * 60 * 60
REQUEST_TIMEOUT = 60

# Kickoff times in games.csv are US/Eastern wall-clock strings ("13:00").
KICKOFF_TZ = ZoneInfo("America/New_York")
DEFAULT_KICKOFF_ET = "13:00"

FIRST_TEAM_STATS_SEASON = 2006
TRAINING_GAME_TYPES = {"REG", "POST", "WC", "DIV", "CON", "SB"}

EWMA_DECAY = 0.5 ** (1.0 / 6.0)  # half-life of six games
SEASON_GAP_DECAY = 0.6            # points EWMA carry-over
EPA_SEASON_GAP_DECAY = 0.5        # EPA EWMA carry-over
ELO_SEASON_REGRESSION = 2.0 / 3.0
LEAGUE_AVG_POINTS = 22.0
ELO_K = 20.0
ELO_HOME_FIELD = 48.0
ELO_POINTS_PER_POINT = 25.0       # ~25 Elo points per point of spread

FEATURE_NAMES = [
    "elo_diff",
    "elo_minus_spread",
    "epa_net_diff",
    "epa_off_diff",
    "epa_def_diff",
    "pass_matchup_home",
    "pass_matchup_away",
    "rush_matchup_home",
    "rush_matchup_away",
    "epa_total_env",
    "cpoe_diff",
    "net_rating_diff",
    "home_qb_change",
    "away_qb_change",
    "rest_diff",
    "div_game",
    "week",
    "roof_dome",
    "games_min",
    "spread_line",
    "total_line",
]
MARKET_FEATURE_NAMES = ["spread_line", "total_line", "elo_minus_spread"]


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _download(url: str, path: Path, stamp: Path) -> bool:
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text, encoding="utf-8")
        stamp.write_text(str(time.time()), encoding="utf-8")
        return True
    except Exception as exc:  # keep serving from the cached copy
        print(f"[nfl] download failed for {url} ({exc}); using cached copy if present")
        return False


def _stale(path: Path, stamp: Path, ttl: float) -> bool:
    # Freshness rides on a download stamp, not file mtime — a CI checkout
    # resets mtimes, which would leave in-season results permanently stale.
    if not path.exists():
        return True
    if not stamp.exists():
        return True
    return (time.time() - stamp.stat().st_mtime) > ttl


def load_games(refresh: bool = True) -> list[dict[str, Any]]:
    """Load (and lazily refresh) the nflverse games file."""
    path = GAMES_CSV_PATH
    stamp = path.with_suffix(".stamp")
    if refresh and _stale(path, stamp, GAMES_TTL_SECONDS):
        _download(GAMES_CSV_URL, path, stamp)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: (str(r.get("gameday") or ""), str(r.get("game_id") or "")))
    return rows


def team_stats_path(season: int) -> Path:
    return DATA_DIR / f"stats_team_week_{season}.csv"


def load_team_stats(
    seasons: Iterable[int],
    *,
    refresh: bool = True,
    required: Iterable[int] = (),
) -> tuple[dict[str, dict[str, dict[str, float]]], list[int]]:
    """Return ``game_id -> team -> per-game EPA stats`` and the seasons loaded.

    Only seasons listed in ``required`` are re-downloaded when stale (the
    current season changes weekly; finished seasons never change).  A season
    that cannot be downloaded and has no cached copy is skipped and reported.
    """

    required_set = set(int(season) for season in required)
    table: dict[str, dict[str, dict[str, float]]] = {}
    loaded: list[int] = []
    for season in sorted(set(int(season) for season in seasons)):
        path = team_stats_path(season)
        stamp = path.with_suffix(".stamp")
        if refresh and (not path.exists() or (season in required_set and _stale(path, stamp, TEAM_STATS_TTL_SECONDS))):
            _download(TEAM_STATS_URL.format(season=season), path, stamp)
        if not path.exists():
            continue
        loaded.append(season)
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                game_id = _text(row.get("game_id"))
                team = _text(row.get("team"))
                if not game_id or not team:
                    continue
                attempts = _num(row.get("attempts"), 0.0) or 0.0
                carries = _num(row.get("carries"), 0.0) or 0.0
                sacks = _num(row.get("sacks_suffered"), 0.0) or 0.0
                plays = attempts + carries + sacks
                if plays < 20:
                    continue
                passing_epa = _num(row.get("passing_epa"), 0.0) or 0.0
                rushing_epa = _num(row.get("rushing_epa"), 0.0) or 0.0
                table.setdefault(game_id, {})[team] = {
                    "epa_pp": (passing_epa + rushing_epa) / plays,
                    "pass_pp": passing_epa / max(1.0, attempts + sacks),
                    "rush_pp": rushing_epa / max(1.0, carries),
                    "cpoe": _num(row.get("passing_cpoe"), 0.0) or 0.0,
                }
    return table, loaded


def kickoff_utc(game: dict[str, Any]) -> datetime | None:
    """Kickoff as an aware UTC datetime from games.csv ``gameday``/``gametime``."""
    gameday = _text(game.get("gameday"))
    if not gameday:
        return None
    clock = _text(game.get("gametime")) or DEFAULT_KICKOFF_ET
    try:
        local = datetime.strptime(f"{gameday} {clock[:5]}", "%Y-%m-%d %H:%M").replace(tzinfo=KICKOFF_TZ)
    except ValueError:
        try:
            local = datetime.strptime(gameday, "%Y-%m-%d").replace(hour=13, tzinfo=KICKOFF_TZ)
        except ValueError:
            return None
    return local.astimezone(timezone.utc)


def kickoff_iso(game: dict[str, Any]) -> str:
    kickoff = kickoff_utc(game)
    return kickoff.strftime("%Y-%m-%dT%H:%MZ") if kickoff else ""


class TeamState:
    __slots__ = (
        "elo", "off_ewma", "def_ewma", "epa_off", "epa_def", "pass_off", "pass_def",
        "rush_off", "rush_def", "cpoe", "games", "epa_games", "season", "qb_id",
    )

    def __init__(self) -> None:
        self.elo = 1500.0
        self.off_ewma = LEAGUE_AVG_POINTS
        self.def_ewma = LEAGUE_AVG_POINTS
        self.epa_off = 0.0
        self.epa_def = 0.0
        self.pass_off = 0.0
        self.pass_def = 0.0
        self.rush_off = 0.0
        self.rush_def = 0.0
        self.cpoe = 0.0
        self.games = 0
        self.epa_games = 0
        self.season = 0
        self.qb_id = ""

    def roll_season(self, season: int) -> None:
        if self.season and season != self.season:
            # Rosters churn but strength persists partially: regress every
            # rating toward league average across the offseason.
            self.elo = 1500.0 + (self.elo - 1500.0) * ELO_SEASON_REGRESSION
            self.off_ewma = LEAGUE_AVG_POINTS + (self.off_ewma - LEAGUE_AVG_POINTS) * SEASON_GAP_DECAY
            self.def_ewma = LEAGUE_AVG_POINTS + (self.def_ewma - LEAGUE_AVG_POINTS) * SEASON_GAP_DECAY
            for name in ("epa_off", "epa_def", "pass_off", "pass_def", "rush_off", "rush_def", "cpoe"):
                setattr(self, name, getattr(self, name) * EPA_SEASON_GAP_DECAY)
            self.games = 0
        self.season = season

    def update_points(self, points_for: float, points_against: float) -> None:
        self.off_ewma = self.off_ewma * EWMA_DECAY + points_for * (1.0 - EWMA_DECAY)
        self.def_ewma = self.def_ewma * EWMA_DECAY + points_against * (1.0 - EWMA_DECAY)
        self.games += 1

    def update_epa(self, mine: dict[str, float], theirs: dict[str, float]) -> None:
        self.epa_off = self.epa_off * EWMA_DECAY + mine["epa_pp"] * (1.0 - EWMA_DECAY)
        self.epa_def = self.epa_def * EWMA_DECAY + theirs["epa_pp"] * (1.0 - EWMA_DECAY)
        self.pass_off = self.pass_off * EWMA_DECAY + mine["pass_pp"] * (1.0 - EWMA_DECAY)
        self.pass_def = self.pass_def * EWMA_DECAY + theirs["pass_pp"] * (1.0 - EWMA_DECAY)
        self.rush_off = self.rush_off * EWMA_DECAY + mine["rush_pp"] * (1.0 - EWMA_DECAY)
        self.rush_def = self.rush_def * EWMA_DECAY + theirs["rush_pp"] * (1.0 - EWMA_DECAY)
        self.cpoe = self.cpoe * EWMA_DECAY + mine["cpoe"] * (1.0 - EWMA_DECAY)
        self.epa_games += 1


def _is_neutral(game: dict[str, Any]) -> bool:
    return _text(game.get("location")).lower() == "neutral"


def _elo_update(home: TeamState, away: TeamState, home_score: float, away_score: float, neutral: bool) -> None:
    home_field = 0.0 if neutral else ELO_HOME_FIELD
    expected_home = 1.0 / (1.0 + 10.0 ** (-(home.elo - away.elo + home_field) / 400.0))
    result = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5
    margin = abs(home_score - away_score)
    if margin > 0:
        winner_elo_gap = (home.elo - away.elo) * (1.0 if home_score > away_score else -1.0)
        multiplier = math.log(margin + 1.0) * (2.2 / (winner_elo_gap * 0.001 + 2.2))
    else:
        multiplier = 1.0
    delta = ELO_K * multiplier * (result - expected_home)
    home.elo += delta
    away.elo -= delta


def _feature_row(game: dict[str, Any], home: TeamState, away: TeamState) -> dict[str, Any] | None:
    """As-of features.  Returns None only when the market anchor is missing."""
    spread_line = _num(game.get("spread_line"))
    total_line = _num(game.get("total_line"))
    features = feature_values(game, home, away, spread_line=spread_line, total_line=total_line)
    if spread_line is None or total_line is None:
        return None
    return features


def feature_values(
    game: dict[str, Any],
    home: TeamState,
    away: TeamState,
    *,
    spread_line: float | None,
    total_line: float | None,
) -> dict[str, Any]:
    """Feature dictionary; market anchors may be None for unpriced slate games."""
    week = _num(game.get("week"), 1.0) or 1.0
    roof = _text(game.get("roof")).lower()
    home_qb = _text(game.get("home_qb_id"))
    away_qb = _text(game.get("away_qb_id"))
    home_rest = _num(game.get("home_rest"), 7.0) or 7.0
    away_rest = _num(game.get("away_rest"), 7.0) or 7.0
    home_field = 0.0 if _is_neutral(game) else ELO_HOME_FIELD
    elo_diff = home.elo - away.elo + home_field
    return {
        "elo_diff": elo_diff,
        "elo_minus_spread": (elo_diff / ELO_POINTS_PER_POINT - spread_line) if spread_line is not None else 0.0,
        "epa_net_diff": (home.epa_off - home.epa_def) - (away.epa_off - away.epa_def),
        "epa_off_diff": home.epa_off - away.epa_off,
        "epa_def_diff": home.epa_def - away.epa_def,
        "pass_matchup_home": home.pass_off - away.pass_def,
        "pass_matchup_away": away.pass_off - home.pass_def,
        "rush_matchup_home": home.rush_off - away.rush_def,
        "rush_matchup_away": away.rush_off - home.rush_def,
        "epa_total_env": home.epa_off + home.epa_def + away.epa_off + away.epa_def,
        "cpoe_diff": home.cpoe - away.cpoe,
        "net_rating_diff": (home.off_ewma - home.def_ewma) - (away.off_ewma - away.def_ewma),
        "home_qb_change": 1.0 if (home.qb_id and home_qb and home_qb != home.qb_id) else 0.0,
        "away_qb_change": 1.0 if (away.qb_id and away_qb and away_qb != away.qb_id) else 0.0,
        "rest_diff": home_rest - away_rest,
        "div_game": _num(game.get("div_game"), 0.0) or 0.0,
        "week": week,
        "roof_dome": 1.0 if roof in {"dome", "closed"} else 0.0,
        "games_min": float(min(home.epa_games, away.epa_games)),
        "spread_line": spread_line if spread_line is not None else 0.0,
        "total_line": total_line if total_line is not None else 0.0,
    }


def _finished(game: dict[str, Any]) -> tuple[float, float] | None:
    home_score = _num(game.get("home_score"))
    away_score = _num(game.get("away_score"))
    if home_score is None or away_score is None or _text(game.get("result")) == "":
        return None
    return home_score, away_score


def _apply_result(
    game: dict[str, Any],
    home: TeamState,
    away: TeamState,
    scores: tuple[float, float],
    team_stats: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Fold a finished game into both team states (strictly after emission)."""
    home_score, away_score = scores
    _elo_update(home, away, home_score, away_score, _is_neutral(game))
    home.update_points(home_score, away_score)
    away.update_points(away_score, home_score)
    stats = team_stats.get(_text(game.get("game_id")), {})
    home_key, away_key = _text(game.get("home_team")), _text(game.get("away_team"))
    if home_key in stats and away_key in stats:
        home.update_epa(stats[home_key], stats[away_key])
        away.update_epa(stats[away_key], stats[home_key])
    home_qb = _text(game.get("home_qb_id"))
    away_qb = _text(game.get("away_qb_id"))
    if home_qb:
        home.qb_id = home_qb
    if away_qb:
        away.qb_id = away_qb


def build_dataset(
    rows: list[dict[str, Any]],
    first_season: int = 2007,
    last_season: int | None = None,
    team_stats: dict[str, dict[str, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Chronological pass emitting one as-of feature record per finished game."""
    team_stats = team_stats or {}
    states: dict[str, TeamState] = {}
    records: list[dict[str, Any]] = []
    for game in rows:
        season = int(_num(game.get("season"), 0) or 0)
        if season < 1999:
            continue
        home_key = _text(game.get("home_team"))
        away_key = _text(game.get("away_team"))
        if not home_key or not away_key:
            continue
        home = states.setdefault(home_key, TeamState())
        away = states.setdefault(away_key, TeamState())
        home.roll_season(season)
        away.roll_season(season)

        scores = _finished(game)
        if (
            scores is not None
            and season >= first_season
            and (last_season is None or season <= last_season)
            and _text(game.get("game_type")).upper() in TRAINING_GAME_TYPES
        ):
            features = _feature_row(game, home, away)
            if features is not None:
                home_score, away_score = scores
                margin = home_score - away_score
                records.append({
                    "game_id": _text(game.get("game_id")),
                    "season": season,
                    "features": features,
                    "home_win": 1 if margin > 0 else 0,
                    "margin": margin,
                    "total": home_score + away_score,
                    "margin_residual": margin - features["spread_line"],
                    "total_residual": (home_score + away_score) - features["total_line"],
                    "home_moneyline": _num(game.get("home_moneyline")),
                    "away_moneyline": _num(game.get("away_moneyline")),
                    "home_spread_odds": _num(game.get("home_spread_odds")),
                    "away_spread_odds": _num(game.get("away_spread_odds")),
                    "over_odds": _num(game.get("over_odds")),
                    "under_odds": _num(game.get("under_odds")),
                })

        # State updates happen strictly after the feature row is emitted.
        if scores is not None:
            _apply_result(game, home, away, scores, team_stats)
    return records


def features_for_date(
    rows: list[dict[str, Any]],
    date_iso: str,
    team_stats: dict[str, dict[str, dict[str, float]]] | None = None,
) -> list[dict[str, Any]]:
    """Feature rows for every game on ``date_iso`` using only prior history.

    Unpriced games (no posted spread or total yet) are still returned with
    ``priced=False`` so the slate is never silently shorter than the schedule.
    """
    team_stats = team_stats or {}
    states: dict[str, TeamState] = {}
    slate: list[dict[str, Any]] = []
    for game in rows:
        gameday = _text(game.get("gameday"))
        home_key = _text(game.get("home_team"))
        away_key = _text(game.get("away_team"))
        if not home_key or not away_key:
            continue
        season = int(_num(game.get("season"), 0) or 0)
        home = states.setdefault(home_key, TeamState())
        away = states.setdefault(away_key, TeamState())
        home.roll_season(season)
        away.roll_season(season)
        if gameday == date_iso:
            spread_line = _num(game.get("spread_line"))
            total_line = _num(game.get("total_line"))
            slate.append({
                "game": game,
                "features": feature_values(game, home, away, spread_line=spread_line, total_line=total_line),
                "priced": spread_line is not None and total_line is not None,
                "spread_line": spread_line,
                "total_line": total_line,
            })
            continue
        scores = _finished(game)
        if gameday < date_iso and scores is not None:
            _apply_result(game, home, away, scores, team_stats)
    return slate


def matrix(records: list[dict[str, Any]], names: list[str] | None = None) -> list[list[float]]:
    names = names or FEATURE_NAMES
    return [[float(rec["features"][name]) for name in names] for rec in records]


def slate_seasons(date_iso: str) -> list[int]:
    """Seasons whose weekly team stats the serving path needs for a slate date."""
    year = int(date_iso[:4])
    month = int(date_iso[5:7])
    season = year if month >= 3 else year - 1
    return [season - 2, season - 1, season]


def ensure_utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


__all__ = [
    "FEATURE_NAMES",
    "MARKET_FEATURE_NAMES",
    "TeamState",
    "build_dataset",
    "ensure_utc",
    "feature_values",
    "features_for_date",
    "kickoff_iso",
    "kickoff_utc",
    "load_games",
    "load_team_stats",
    "matrix",
    "slate_seasons",
    "team_stats_path",
]
