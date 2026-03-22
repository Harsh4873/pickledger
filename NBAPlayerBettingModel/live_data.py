from __future__ import annotations

import os
import sys
import time
import unicodedata
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd
from nba_api.stats.endpoints import (
    commonteamroster,
    leaguedashplayerstats,
    leaguedashteamstats,
    playerdashboardbygamesplits,
    playerdashboardbylastngames,
    scoreboardv2,
)
from nba_api.stats.static import teams

from data_models import OpponentDefenseStats, PlayerSeasonStats

# Import injury report from sibling NBAPredictionModel directory
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NBA_MODEL_DIR = os.path.join(_PARENT_DIR, "NBAPredictionModel")
if _NBA_MODEL_DIR not in sys.path:
    sys.path.insert(0, _NBA_MODEL_DIR)
try:
    from injury_report import fetch_injuries  # type: ignore[import-untyped]
    _HAS_INJURY_REPORT = True
except ImportError:
    _HAS_INJURY_REPORT = False
    def fetch_injuries(*args, **kwargs):  # type: ignore[misc]
        return {}

REQUEST_PAUSE_SECONDS = 0.35
_NBA_TEAMS = teams.get_teams()
_TEAM_BY_ID = {int(team["id"]): team for team in _NBA_TEAMS}

PLAYER_FEATURE_COLUMNS = [
    "mp_per_game",
    "fg_per_game",
    "fga_per_game",
    "fg_percent",
    "x3p_per_game",
    "x3pa_per_game",
    "x3p_percent",
    "x2p_per_game",
    "x2pa_per_game",
    "x2p_percent",
    "e_fg_percent",
    "ft_per_game",
    "fta_per_game",
    "ft_percent",
    "orb_per_game",
    "drb_per_game",
    "trb_per_game",
    "ast_per_game",
    "stl_per_game",
    "blk_per_game",
    "tov_per_game",
    "usage_rate",
    "points_per_game",
    "rebounds_per_game",
    "assists_per_game",
]


def _sleep() -> None:
    time.sleep(REQUEST_PAUSE_SECONDS)


def _normalize_name_for_matching(name: str) -> str:
    """Normalize a player name for fuzzy matching across data sources."""
    text = unicodedata.normalize("NFKD", str(name or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.strip().lower()


def _build_injured_player_set(injuries: dict) -> set[str]:
    """Build a set of normalized player names who are Out or Doubtful."""
    excluded: set[str] = set()
    for _team, players in injuries.items():
        for player in players:
            status = str(player.get("status", "")).strip()
            if status in {"Out", "Doubtful"}:
                excluded.add(_normalize_name_for_matching(player.get("name", "")))
    return excluded


def _is_player_injured(player_name: str, injured_names: set[str]) -> bool:
    """Check if a player name matches any entry in the injured set."""
    norm = _normalize_name_for_matching(player_name)
    if not norm:
        return False
    # Exact match
    if norm in injured_names:
        return True
    # Substring match (handles "first last" vs "first middle last" variants)
    for inj_name in injured_names:
        if norm in inj_name or inj_name in norm:
            return True
        # Last-name + first-initial match
        norm_parts = norm.split()
        inj_parts = inj_name.split()
        if len(norm_parts) >= 2 and len(inj_parts) >= 2:
            if norm_parts[-1] == inj_parts[-1] and norm_parts[0][0] == inj_parts[0][0]:
                return True
    return False


def _parse_date(date_str: str | None) -> datetime:
    if not date_str:
        return datetime.now()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {date_str}")


def get_current_season(date_str: str | None = None) -> str:
    dt = _parse_date(date_str)
    start_year = dt.year if dt.month >= 7 else dt.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _first_frame_with_columns(frames: Iterable[pd.DataFrame], required: set[str]) -> pd.DataFrame:
    for frame in frames:
        if required.issubset(set(frame.columns)):
            return frame.copy()
    for frame in frames:
        if not frame.empty:
            return frame.copy()
    return pd.DataFrame()


def _team_name(team_id: int) -> str:
    team = _TEAM_BY_ID.get(int(team_id), {})
    return str(team.get("nickname") or team.get("full_name") or team_id)


def _team_abbreviation(team_id: int) -> str:
    team = _TEAM_BY_ID.get(int(team_id), {})
    return str(team.get("abbreviation") or team.get("nickname") or team_id)


def _normalize_usage_rate(value: Any) -> float:
    usage = _safe_float(value)
    if usage <= 1.0:
        usage *= 100.0
    return usage


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_position(position: str | None) -> tuple[str, str]:
    raw = (position or "").upper().strip()
    raw = raw.replace(" ", "")
    if not raw:
        return "F", "F"
    first = raw.split("-")[0]
    if first in {"PG", "SG", "G"}:
        return first, "G"
    if first in {"SF", "PF", "F"}:
        return first, "F"
    if first == "C":
        return "C", "C"
    if "C" in raw:
        return "C", "C"
    if "G" in raw:
        return "G", "G"
    return "F", "F"


def _infer_position_bucket(row: pd.Series) -> tuple[str, str]:
    if _safe_float(row.get("ast_per_game")) >= 5.5:
        return "G", "G"
    if _safe_float(row.get("trb_per_game")) >= 8.0:
        return "C", "C"
    return "F", "F"


def _rename_player_stat_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "GP": "games_played",
            "MIN": "mp_per_game",
            "FGM": "fg_per_game",
            "FGA": "fga_per_game",
            "FG_PCT": "fg_percent",
            "FG3M": "x3p_per_game",
            "FG3A": "x3pa_per_game",
            "FG3_PCT": "x3p_percent",
            "FG2M": "x2p_per_game",
            "FG2A": "x2pa_per_game",
            "FG2_PCT": "x2p_percent",
            "EFG_PCT": "e_fg_percent",
            "FTM": "ft_per_game",
            "FTA": "fta_per_game",
            "FT_PCT": "ft_percent",
            "OREB": "orb_per_game",
            "DREB": "drb_per_game",
            "REB": "trb_per_game",
            "AST": "ast_per_game",
            "STL": "stl_per_game",
            "BLK": "blk_per_game",
            "TOV": "tov_per_game",
            "PTS": "points_per_game",
            "USG_PCT": "usage_rate",
        }
    )

    if "x2p_per_game" not in renamed.columns and {"fg_per_game", "x3p_per_game"}.issubset(set(renamed.columns)):
        renamed["x2p_per_game"] = renamed["fg_per_game"] - renamed["x3p_per_game"]
    if "x2pa_per_game" not in renamed.columns and {"fga_per_game", "x3pa_per_game"}.issubset(set(renamed.columns)):
        renamed["x2pa_per_game"] = renamed["fga_per_game"] - renamed["x3pa_per_game"]
    if "x2p_percent" not in renamed.columns:
        attempts = renamed.get("x2pa_per_game", pd.Series(dtype=float)).replace(0, pd.NA)
        renamed["x2p_percent"] = (renamed.get("x2p_per_game", 0.0) / attempts).fillna(0.0)

    if "e_fg_percent" not in renamed.columns:
        fga = renamed.get("fga_per_game", pd.Series(dtype=float)).replace(0, pd.NA)
        fgm = renamed.get("fg_per_game", 0.0)
        fg3m = renamed.get("x3p_per_game", 0.0)
        renamed["e_fg_percent"] = ((fgm + 0.5 * fg3m) / fga).fillna(0.0)

    return renamed


def fetch_todays_games(date_str: str | None = None) -> list[dict[str, Any]]:
    dt = _parse_date(date_str)
    _sleep()
    board = scoreboardv2.ScoreboardV2(game_date=dt.strftime("%m/%d/%Y"))
    header = _first_frame_with_columns(
        board.get_data_frames(),
        {"GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"},
    )
    if header.empty:
        return []

    games: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for _, row in header.iterrows():
        game_id = str(row.get("GAME_ID", "")).strip()
        if not game_id or game_id in seen_ids:
            continue
        seen_ids.add(game_id)

        home_team_id = int(row["HOME_TEAM_ID"])
        away_team_id = int(row["VISITOR_TEAM_ID"])
        games.append(
            {
                "game_id": game_id,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team": _team_name(home_team_id),
                "away_team": _team_name(away_team_id),
                "home_team_abbreviation": _team_abbreviation(home_team_id),
                "away_team_abbreviation": _team_abbreviation(away_team_id),
                "game_status": str(row.get("GAME_STATUS_TEXT", "")).strip(),
            }
        )

    return games


def fetch_player_pool(season: str) -> pd.DataFrame:
    _sleep()
    base = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
    )
    base_df = _rename_player_stat_columns(
        _first_frame_with_columns(base.get_data_frames(), {"PLAYER_ID", "PLAYER_NAME", "TEAM_ID"})
    )

    _sleep()
    advanced = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    )
    advanced_df = _rename_player_stat_columns(
        _first_frame_with_columns(advanced.get_data_frames(), {"PLAYER_ID", "USG_PCT"})
    )

    keep_columns = [
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "games_played",
        "mp_per_game",
        "fg_per_game",
        "fga_per_game",
        "fg_percent",
        "x3p_per_game",
        "x3pa_per_game",
        "x3p_percent",
        "x2p_per_game",
        "x2pa_per_game",
        "x2p_percent",
        "e_fg_percent",
        "ft_per_game",
        "fta_per_game",
        "ft_percent",
        "orb_per_game",
        "drb_per_game",
        "trb_per_game",
        "ast_per_game",
        "stl_per_game",
        "blk_per_game",
        "tov_per_game",
        "points_per_game",
        "usage_rate",
    ]
    available_keep = [column for column in keep_columns if column in base_df.columns]
    player_df = base_df[available_keep].copy()

    if "usage_rate" not in player_df.columns:
        player_df = player_df.merge(
            advanced_df[["PLAYER_ID", "usage_rate"]],
            on="PLAYER_ID",
            how="left",
        )

    player_df["assists_per_game"] = player_df.get("ast_per_game", 0.0)
    player_df["rebounds_per_game"] = player_df.get("trb_per_game", 0.0)
    player_df["team_name"] = player_df["TEAM_ID"].apply(lambda team_id: _team_name(int(team_id)))

    numeric_columns = [column for column in PLAYER_FEATURE_COLUMNS if column in player_df.columns]
    numeric_columns.extend(["games_played"])
    for column in numeric_columns:
        player_df[column] = pd.to_numeric(player_df[column], errors="coerce").fillna(0.0)

    player_df["usage_rate"] = player_df.get("usage_rate", 0.0).apply(_normalize_usage_rate)
    return player_df


def fetch_roster_positions(team_ids: set[int], season: str) -> dict[int, tuple[str, str]]:
    positions: dict[int, tuple[str, str]] = {}
    for team_id in sorted(team_ids):
        _sleep()
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        roster_df = _first_frame_with_columns(
            roster.get_data_frames(),
            {"PLAYER_ID", "PLAYER"},
        )
        if roster_df.empty:
            continue
        for _, row in roster_df.iterrows():
            player_id = int(row.get("PLAYER_ID", 0))
            if not player_id:
                continue
            positions[player_id] = _normalize_position(str(row.get("POSITION", "")))
    return positions


def attach_positions(player_df: pd.DataFrame, team_ids: set[int], season: str) -> pd.DataFrame:
    roster_positions = fetch_roster_positions(team_ids, season)
    df = player_df.copy()

    display_positions: list[str] = []
    buckets: list[str] = []
    for _, row in df.iterrows():
        player_id = int(row.get("PLAYER_ID", 0))
        mapped = roster_positions.get(player_id)
        if mapped is None:
            mapped = _infer_position_bucket(row)
        display_positions.append(mapped[0])
        buckets.append(mapped[1])

    df["position"] = display_positions
    df["position_bucket"] = buckets
    return df


def _extract_dashboard_totals(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {}

    row = frame.iloc[0]
    return {
        "pts": _safe_float(row.get("PTS")),
        "reb": _safe_float(row.get("REB")),
        "ast": _safe_float(row.get("AST")),
    }


@lru_cache(maxsize=256)
def fetch_last_10_averages(player_id: int, season: str) -> dict[str, float]:
    _sleep()
    dashboard = playerdashboardbylastngames.PlayerDashboardByLastNGames(
        player_id=player_id,
        season=season,
        season_type_playoffs="Regular Season",
        per_mode_detailed="PerGame",
        last_n_games=10,
    )
    frame = _first_frame_with_columns(dashboard.get_data_frames(), {"PTS", "REB", "AST"})
    return _extract_dashboard_totals(frame)


@lru_cache(maxsize=256)
def fetch_home_away_splits(player_id: int, season: str) -> dict[str, dict[str, float]]:
    _sleep()
    dashboard = playerdashboardbygamesplits.PlayerDashboardByGameSplits(
        player_id=player_id,
        season=season,
        season_type_playoffs="Regular Season",
        per_mode_detailed="PerGame",
    )
    frame = _first_frame_with_columns(dashboard.get_data_frames(), {"GROUP_VALUE", "PTS", "REB", "AST"})
    if frame.empty:
        return {}

    splits: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        key = str(row.get("GROUP_VALUE", "")).strip().lower()
        if key not in {"home", "road", "away"}:
            continue
        norm_key = "away" if key == "road" else key
        splits[norm_key] = {
            "pts": _safe_float(row.get("PTS")),
            "reb": _safe_float(row.get("REB")),
            "ast": _safe_float(row.get("AST")),
        }
    return splits


def _position_baselines(player_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    baselines: dict[str, dict[str, float]] = {}
    for bucket in ("G", "F", "C"):
        sample = player_df[player_df["position_bucket"] == bucket]
        if sample.empty:
            sample = player_df
        baselines[bucket] = {
            "pts": sample["points_per_game"].mean(),
            "reb": sample["rebounds_per_game"].mean(),
            "ast": sample["assists_per_game"].mean(),
        }
    return baselines


def build_opponent_defense_lookup(
    season: str,
    player_df: pd.DataFrame,
) -> tuple[dict[int, OpponentDefenseStats], dict[str, dict[str, float]], dict[str, float]]:
    _sleep()
    team_stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        per_mode_detailed="PerGame",
        measure_type_detailed_defense="Advanced",
    )
    team_df = _first_frame_with_columns(
        team_stats.get_data_frames(),
        {"TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"},
    )
    if team_df.empty:
        return {}, _position_baselines(player_df), {"league_def_rating": 113.0, "league_pace": 99.0}

    dreb_column = "DREB_PCT" if "DREB_PCT" in team_df.columns else "REB_PCT"
    baselines = _position_baselines(player_df)

    league_def_rating = team_df["DEF_RATING"].astype(float).mean()
    league_pace = team_df["PACE"].astype(float).mean()
    league_dreb_pct = (
        team_df[dreb_column].astype(float).mean()
        if dreb_column in team_df.columns
        else 73.0
    )

    lookup: dict[int, OpponentDefenseStats] = {}
    for _, row in team_df.iterrows():
        team_id = int(row["TEAM_ID"])
        def_rating = _safe_float(row.get("DEF_RATING"), league_def_rating)
        pace = _safe_float(row.get("PACE"), league_pace)
        dreb_pct = _safe_float(row.get(dreb_column), league_dreb_pct)

        def_factor = _clip(def_rating / league_def_rating, 0.90, 1.12)
        pace_factor = _clip(pace / league_pace, 0.92, 1.08)
        rebound_factor = _clip(league_dreb_pct / max(dreb_pct, 1e-6), 0.90, 1.12)
        assist_factor = _clip((def_factor * 0.7) + (pace_factor * 0.3), 0.90, 1.10)

        pts_allowed = {
            bucket: baselines[bucket]["pts"] * def_factor * pace_factor
            for bucket in baselines
        }
        reb_allowed = {
            bucket: baselines[bucket]["reb"] * pace_factor * rebound_factor
            for bucket in baselines
        }
        ast_allowed = {
            bucket: baselines[bucket]["ast"] * pace_factor * assist_factor
            for bucket in baselines
        }

        lookup[team_id] = OpponentDefenseStats(
            team_id=team_id,
            team_name=str(row.get("TEAM_NAME") or _team_name(team_id)),
            team_abbreviation=str(row.get("TEAM_ABBREVIATION") or _team_abbreviation(team_id)),
            def_rating=def_rating,
            pace=pace,
            pts_allowed_by_position=pts_allowed,
            reb_allowed_by_position=reb_allowed,
            ast_allowed_by_position=ast_allowed,
        )

    return lookup, baselines, {"league_def_rating": league_def_rating, "league_pace": league_pace}


def _build_selected_player(
    row: pd.Series,
    game: dict[str, Any],
    season: str,
) -> PlayerSeasonStats:
    player_id = int(row["PLAYER_ID"])
    is_home = int(row["TEAM_ID"]) == int(game["home_team_id"])
    opponent_team_id = int(game["away_team_id"] if is_home else game["home_team_id"])

    last_10 = fetch_last_10_averages(player_id, season)
    splits = fetch_home_away_splits(player_id, season)
    home_split = splits.get("home", {})
    away_split = splits.get("away", {})

    return PlayerSeasonStats(
        player_id=player_id,
        player_name=str(row["PLAYER_NAME"]),
        team_id=int(row["TEAM_ID"]),
        team_name=str(row["team_name"]),
        team_abbreviation=str(row.get("TEAM_ABBREVIATION") or _team_abbreviation(int(row["TEAM_ID"]))),
        opponent_team_id=opponent_team_id,
        opponent_team_name=game["away_team"] if is_home else game["home_team"],
        opponent_team_abbreviation=game["away_team_abbreviation"] if is_home else game["home_team_abbreviation"],
        game_id=str(game["game_id"]),
        away_team_name=str(game["away_team"]),
        home_team_name=str(game["home_team"]),
        position=str(row["position"]),
        position_bucket=str(row["position_bucket"]),
        is_home=is_home,
        games_played=int(_safe_float(row.get("games_played"), 0)),
        mp_per_game=_safe_float(row.get("mp_per_game")),
        fg_per_game=_safe_float(row.get("fg_per_game")),
        fga_per_game=_safe_float(row.get("fga_per_game")),
        fg_percent=_safe_float(row.get("fg_percent")),
        x3p_per_game=_safe_float(row.get("x3p_per_game")),
        x3pa_per_game=_safe_float(row.get("x3pa_per_game")),
        x3p_percent=_safe_float(row.get("x3p_percent")),
        x2p_per_game=_safe_float(row.get("x2p_per_game")),
        x2pa_per_game=_safe_float(row.get("x2pa_per_game")),
        x2p_percent=_safe_float(row.get("x2p_percent")),
        e_fg_percent=_safe_float(row.get("e_fg_percent")),
        ft_per_game=_safe_float(row.get("ft_per_game")),
        fta_per_game=_safe_float(row.get("fta_per_game")),
        ft_percent=_safe_float(row.get("ft_percent")),
        orb_per_game=_safe_float(row.get("orb_per_game")),
        drb_per_game=_safe_float(row.get("drb_per_game")),
        trb_per_game=_safe_float(row.get("trb_per_game")),
        ast_per_game=_safe_float(row.get("ast_per_game")),
        stl_per_game=_safe_float(row.get("stl_per_game")),
        blk_per_game=_safe_float(row.get("blk_per_game")),
        tov_per_game=_safe_float(row.get("tov_per_game")),
        usage_rate=_normalize_usage_rate(row.get("usage_rate")),
        points_per_game=_safe_float(row.get("points_per_game")),
        rebounds_per_game=_safe_float(row.get("rebounds_per_game")),
        assists_per_game=_safe_float(row.get("assists_per_game")),
        last10_points=_safe_float(last_10.get("pts"), _safe_float(row.get("points_per_game"))),
        last10_rebounds=_safe_float(last_10.get("reb"), _safe_float(row.get("rebounds_per_game"))),
        last10_assists=_safe_float(last_10.get("ast"), _safe_float(row.get("assists_per_game"))),
        home_points=_safe_float(home_split.get("pts")) if home_split else None,
        away_points=_safe_float(away_split.get("pts")) if away_split else None,
        home_rebounds=_safe_float(home_split.get("reb")) if home_split else None,
        away_rebounds=_safe_float(away_split.get("reb")) if away_split else None,
        home_assists=_safe_float(home_split.get("ast")) if home_split else None,
        away_assists=_safe_float(away_split.get("ast")) if away_split else None,
    )


def load_props_slate(
    date_str: str | None = None,
    game_ids: set[str] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[PlayerSeasonStats],
    dict[int, OpponentDefenseStats],
    pd.DataFrame,
    dict[str, dict[str, float]],
    dict[str, float],
    str,
]:
    games = fetch_todays_games(date_str)
    if game_ids:
        normalized_ids = {str(game_id).strip() for game_id in game_ids if str(game_id).strip()}
        games = [game for game in games if str(game.get("game_id", "")).strip() in normalized_ids]
    season = get_current_season(date_str)
    player_df = fetch_player_pool(season)

    if not games or player_df.empty:
        return games, [], {}, player_df, {}, {}, season

    team_ids = {
        int(game["home_team_id"])
        for game in games
    } | {
        int(game["away_team_id"])
        for game in games
    }

    player_df = attach_positions(player_df, team_ids, season)
    opponent_lookup, position_baselines, league_meta = build_opponent_defense_lookup(season, player_df)

    qualified = player_df[
        (player_df["TEAM_ID"].isin(team_ids))
        & (player_df["mp_per_game"] >= 20.0)
        & (player_df["games_played"] >= 10.0)
    ].copy()

    # ── Injury filtering: remove Out / Doubtful players ──
    injuries: dict = {}
    injured_names: set[str] = set()
    try:
        injuries = fetch_injuries()
        injured_names = _build_injured_player_set(injuries)
        if injured_names:
            before_count = len(qualified)
            qualified = qualified[
                ~qualified["PLAYER_NAME"].apply(lambda name: _is_player_injured(name, injured_names))
            ].copy()
            removed = before_count - len(qualified)
            if removed > 0:
                print(f"    [injury_filter] Removed {removed} injured player(s) (Out/Doubtful) from props pool.")
    except Exception as exc:
        print(f"    [injury_filter] WARNING: Could not fetch injury report: {exc}")

    selected_players: list[PlayerSeasonStats] = []
    for game in games:
        for team_id in (int(game["away_team_id"]), int(game["home_team_id"])):
            team_players = qualified[qualified["TEAM_ID"] == team_id].sort_values(
                ["mp_per_game"],
                ascending=[False],
            )
            for _, row in team_players.head(5).iterrows():
                selected_players.append(_build_selected_player(row, game, season))

    return (
        games,
        selected_players,
        opponent_lookup,
        player_df,
        position_baselines,
        league_meta,
        season,
    )
