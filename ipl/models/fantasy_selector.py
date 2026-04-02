from __future__ import annotations

import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ipl.data_loader import _default_db_path


MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "fantasy_ranker.pkl"
FEATURES_PATH = MODEL_DIR / "fantasy_ranker_features.json"
MEDIANS_PATH = MODEL_DIR / "fantasy_ranker_medians.json"
POOL_COLUMNS = ["player_name", "team", "role", "is_overseas"]
ROLLING_COLS = [
    "avg_runs_last5",
    "avg_sr_last5",
    "avg_wickets_last5",
    "avg_economy_last5",
    "avg_fours_last5",
    "avg_sixes_last5",
    "matches_played_last5",
]
TRAINING_FEATURES = [
    "avg_runs_last5",
    "avg_sr_last5",
    "avg_wickets_last5",
    "avg_economy_last5",
    "avg_fours_last5",
    "avg_sixes_last5",
    "matches_played_last5",
    "role_encoded",
]
TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
}
ROLE_WEIGHTS = {
    "Wicket-Keeper": 1.08,
    "All-Rounder": 1.12,
    "Batsman": 1.00,
    "Bowler": 0.96,
}
ROLE_ENCODING = {
    "Wicket-Keeper": 3,
    "All-Rounder": 2,
    "Batsman": 1,
    "Bowler": 0,
}
BATTER_ROLES = {"Batsman", "Wicket-Keeper", "All-Rounder"}
BOWLER_ROLES = {"Bowler", "All-Rounder"}

try:
    import joblib
except ImportError:
    import pickle

    class _JoblibCompat:
        @staticmethod
        def dump(obj: Any, path: str | Path) -> None:
            with Path(path).open("wb") as handle:
                pickle.dump(obj, handle)

        @staticmethod
        def load(path: str | Path) -> Any:
            with Path(path).open("rb") as handle:
                return pickle.load(handle)

    joblib = _JoblibCompat()


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    return text or None


def _canonical_team(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    return TEAM_ALIASES.get(text, text)


def _normalize_role(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    role_map = {
        "wicket keeper": "Wicket-Keeper",
        "wicket-keeper": "Wicket-Keeper",
        "wk": "Wicket-Keeper",
        "keeper": "Wicket-Keeper",
        "all rounder": "All-Rounder",
        "all-rounder": "All-Rounder",
        "allrounder": "All-Rounder",
        "batter": "Batsman",
        "bat": "Batsman",
        "batsman": "Batsman",
        "bowler": "Bowler",
    }
    return role_map.get(text.lower().replace("_", " "), text)


def _normalize_decision(value: Any) -> str | None:
    text = _normalize_text(value)
    if text is None:
        return None
    return text.lower()


def _parse_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str).str.replace("/", "-", regex=False), errors="coerce")


def _table_columns(con: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in con.execute(f"PRAGMA table_info({table_name})").fetchall()}


def _placeholders(count: int) -> str:
    return ", ".join(["?"] * count)


def _player_tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def _player_key(name: str) -> str:
    return " ".join(_player_tokens(name))


def _player_prefix(name: str) -> str:
    tokens = _player_tokens(name)
    if not tokens:
        return ""
    head = tokens[:-1] or tokens[:1]
    return "".join(token[0] for token in head if token)


def _resolve_history_names(pool: pd.DataFrame, con: sqlite3.Connection) -> pd.DataFrame:
    if pool.empty:
        return pool.assign(history_player_name=pd.Series(dtype="object"))

    history_stats = pd.read_sql_query(
        """
        SELECT
            player_name,
            player_team,
            COUNT(*) AS team_matches
        FROM ipl_player_match_features
        WHERE player_name IS NOT NULL
          AND TRIM(player_name) <> ''
        GROUP BY player_name, player_team
        """,
        con,
    )
    if history_stats.empty:
        result = pool.copy()
        result["history_player_name"] = result["player_name"]
        return result

    history_stats["player_name"] = history_stats["player_name"].map(_normalize_text)
    history_stats["player_team"] = history_stats["player_team"].map(_canonical_team)
    history_stats = history_stats.dropna(subset=["player_name"])

    totals = history_stats.groupby("player_name", as_index=False)["team_matches"].sum()
    totals = totals.rename(columns={"team_matches": "total_matches"})
    exact_names = set(totals["player_name"])
    total_lookup = dict(zip(totals["player_name"], totals["total_matches"]))
    team_lookup = {
        (row.player_name, row.player_team): int(row.team_matches)
        for row in history_stats.itertuples(index=False)
    }

    candidates_by_last: dict[str, list[str]] = {}
    for history_name in exact_names:
        tokens = _player_tokens(history_name)
        if not tokens:
            continue
        candidates_by_last.setdefault(tokens[-1], []).append(history_name)

    resolved_names: list[str | None] = []
    for row in pool.itertuples(index=False):
        player_name = _normalize_text(row.player_name)
        team_name = _canonical_team(row.team)
        if player_name is None:
            resolved_names.append(None)
            continue
        if player_name in exact_names:
            resolved_names.append(player_name)
            continue

        tokens = _player_tokens(player_name)
        if not tokens:
            resolved_names.append(None)
            continue
        last_name = tokens[-1]
        current_prefix = _player_prefix(player_name)
        current_first = current_prefix[:1]

        scored_candidates: list[tuple[int, int, int, str]] = []
        for candidate in candidates_by_last.get(last_name, []):
            candidate_prefix = _player_prefix(candidate)
            if current_first and candidate_prefix[:1] != current_first:
                continue
            overlap = sum(
                1
                for current_char, candidate_char in zip(current_prefix, candidate_prefix)
                if current_char == candidate_char
            )
            same_team_matches = team_lookup.get((candidate, team_name), 0)
            total_matches = total_lookup.get(candidate, 0)
            exact_prefix = 1 if current_prefix and current_prefix == candidate_prefix else 0
            scored_candidates.append(
                (
                    same_team_matches,
                    exact_prefix,
                    overlap * 100 + total_matches,
                    candidate,
                )
            )

        if scored_candidates:
            scored_candidates.sort(reverse=True)
            resolved_names.append(scored_candidates[0][3])
        else:
            resolved_names.append(None)

    result = pool.copy()
    result["history_player_name"] = resolved_names
    return result


def _rebuild_rolling_features(history_rows: pd.DataFrame) -> pd.DataFrame:
    if history_rows.empty:
        return pd.DataFrame(columns=["player_name", "match_id", "date", *ROLLING_COLS])

    rolling_frame = history_rows.copy()
    rolling_frame["date"] = _parse_dates(rolling_frame["date"])
    rolling_frame = rolling_frame.dropna(subset=["date"]).sort_values(
        ["player_name", "date", "match_id"]
    )

    feature_map = {
        "runs_scored": "avg_runs_last5",
        "strike_rate": "avg_sr_last5",
        "wickets_taken": "avg_wickets_last5",
        "economy_rate": "avg_economy_last5",
        "fours": "avg_fours_last5",
        "sixes": "avg_sixes_last5",
    }
    grouped = rolling_frame.groupby("player_name", sort=False)
    for source_col, target_col in feature_map.items():
        rolling_frame[target_col] = grouped[source_col].transform(
            lambda series: series.rolling(window=5, min_periods=1).mean().shift(1)
        )
    rolling_frame["matches_played_last5"] = grouped.cumcount().clip(upper=5).astype(float)
    return rolling_frame[["player_name", "match_id", "date", *ROLLING_COLS]].reset_index(drop=True)


def _load_history_base(
    con: sqlite3.Connection,
    player_names: list[str] | None = None,
) -> pd.DataFrame:
    params: tuple[Any, ...] = ()
    where_clause = ""
    if player_names:
        where_clause = f"WHERE f.player_name IN ({_placeholders(len(player_names))})"
        params = tuple(player_names)
    return pd.read_sql_query(
        f"""
        SELECT
            f.player_name,
            f.match_id,
            m.date,
            f.runs_scored,
            f.fours,
            f.sixes,
            f.wickets_taken,
            f.strike_rate,
            f.economy_rate
        FROM ipl_player_match_features f
        JOIN ipl_matches m
          ON m.match_id = f.match_id
        {where_clause}
        """,
        con,
        params=params,
    )


def _load_latest_rolling(
    con: sqlite3.Connection,
    player_names: list[str],
) -> pd.DataFrame:
    if not player_names:
        return pd.DataFrame(columns=["player_name", *ROLLING_COLS])

    rolling_columns = _table_columns(con, "ipl_player_rolling_features")
    if "player_name" in rolling_columns:
        rolling = pd.read_sql_query(
            f"""
            SELECT
                r.player_name,
                r.match_id,
                m.date,
                r.avg_runs_last5,
                r.avg_sr_last5,
                r.avg_wickets_last5,
                r.avg_economy_last5,
                r.avg_fours_last5,
                r.avg_sixes_last5,
                r.matches_played_last5
            FROM ipl_player_rolling_features r
            JOIN ipl_matches m
              ON m.match_id = r.match_id
            WHERE r.player_name IN ({_placeholders(len(player_names))})
            """,
            con,
            params=tuple(player_names),
        )
        rolling["date"] = _parse_dates(rolling["date"])
    else:
        base = _load_history_base(con, player_names)
        rolling = _rebuild_rolling_features(base)

    if rolling.empty:
        return pd.DataFrame(columns=["player_name", *ROLLING_COLS])

    rolling = rolling.dropna(subset=["date"]).sort_values(["player_name", "date", "match_id"])
    latest = rolling.groupby("player_name", as_index=False).tail(1)
    return latest[["player_name", *ROLLING_COLS]].reset_index(drop=True)


def _load_career_aggregates(
    con: sqlite3.Connection,
    player_names: list[str],
) -> pd.DataFrame:
    if not player_names:
        return pd.DataFrame()

    return pd.read_sql_query(
        f"""
        WITH hist AS (
            SELECT
                f.player_name,
                f.runs_scored,
                f.balls_faced,
                f.fours,
                f.sixes,
                f.dismissed,
                f.strike_rate,
                f.is_duck,
                f.milestone_50,
                f.milestone_100,
                f.pp_runs,
                f.mid_runs,
                f.death_runs,
                f.overs_bowled,
                f.wickets_taken,
                f.economy_rate,
                f.pp_wickets,
                f.death_wickets,
                f.death_economy,
                COALESCE(p.batting_points, 0) AS batting_points,
                COALESCE(p.bowling_points, 0) AS bowling_points,
                COALESCE(p.sr_points, 0) AS sr_points,
                COALESCE(p.economy_points, 0) AS economy_points,
                COALESCE(p.total_fantasy_points, 0) AS total_fantasy_points
            FROM ipl_player_match_features f
            LEFT JOIN ipl_player_fantasy_points p
              ON p.match_id = f.match_id
             AND p.player_name = f.player_name
            WHERE f.player_name IN ({_placeholders(len(player_names))})
        )
        SELECT
            player_name,
            COUNT(*) AS matches_played_total,
            AVG(total_fantasy_points) AS avg_fantasy_points,
            MAX(total_fantasy_points) AS max_fantasy_points,
            AVG(batting_points) AS avg_batting_points,
            AVG(bowling_points) AS avg_bowling_points,
            AVG(sr_points) AS avg_sr_points,
            AVG(economy_points) AS avg_economy_points,
            AVG(CASE WHEN balls_faced > 0 THEN 1.0 ELSE 0.0 END) AS batting_match_share,
            AVG(CASE WHEN overs_bowled > 0 THEN 1.0 ELSE 0.0 END) AS bowling_match_share,
            AVG(CASE WHEN milestone_50 = 1 THEN 1.0 ELSE 0.0 END) AS fifty_rate,
            AVG(CASE WHEN milestone_100 = 1 THEN 1.0 ELSE 0.0 END) AS hundred_rate,
            AVG(CASE WHEN wickets_taken >= 2 THEN 1.0 ELSE 0.0 END) AS wicket_haul_rate,
            AVG(CASE WHEN is_duck = 1 THEN 1.0 ELSE 0.0 END) AS duck_rate,
            AVG(pp_runs) AS avg_pp_runs,
            AVG(mid_runs) AS avg_mid_runs,
            AVG(death_runs) AS avg_death_runs,
            AVG(pp_wickets) AS avg_pp_wickets,
            AVG(death_wickets) AS avg_death_wickets,
            AVG(death_economy) AS avg_death_economy,
            AVG(runs_scored) AS career_avg_runs,
            AVG(strike_rate) AS career_avg_sr,
            AVG(wickets_taken) AS career_avg_wickets,
            AVG(economy_rate) AS career_avg_economy,
            AVG(fours) AS career_avg_fours,
            AVG(sixes) AS career_avg_sixes
        FROM hist
        GROUP BY player_name
        """,
        con,
        params=tuple(player_names),
    )


def _load_venue_aggregates(
    con: sqlite3.Connection,
    player_names: list[str],
    venue: str | None,
) -> pd.DataFrame:
    venue_name = _normalize_text(venue)
    if venue_name is None or not player_names:
        return pd.DataFrame(columns=["player_name", "venue_matches", "venue_avg_fantasy_points"])

    return pd.read_sql_query(
        f"""
        SELECT
            f.player_name,
            COUNT(*) AS venue_matches,
            AVG(COALESCE(p.total_fantasy_points, 0)) AS venue_avg_fantasy_points
        FROM ipl_player_match_features f
        LEFT JOIN ipl_player_fantasy_points p
          ON p.match_id = f.match_id
         AND p.player_name = f.player_name
        WHERE f.player_name IN ({_placeholders(len(player_names))})
          AND f.venue = ?
        GROUP BY f.player_name
        """,
        con,
        params=tuple(player_names) + (venue_name,),
    )


def _load_role_lookup(con: sqlite3.Connection) -> dict[str, int]:
    rows = con.execute(
        """
        SELECT player_name, role
        FROM ipl_current_squads
        WHERE season='2026'
        """
    ).fetchall()
    lookup: dict[str, int] = {}
    for player_name, role in rows:
        name = _normalize_text(player_name)
        role_name = _normalize_role(role)
        if name is None:
            continue
        lookup[name] = ROLE_ENCODING.get(role_name or "", 1)
    return lookup


def _build_training_frame(con: sqlite3.Connection) -> pd.DataFrame:
    base = pd.read_sql_query(
        """
        SELECT
            f.player_name,
            f.match_id,
            f.season,
            m.date,
            f.runs_scored,
            f.strike_rate,
            f.fours,
            f.sixes,
            f.wickets_taken,
            f.economy_rate,
            p.total_fantasy_points
        FROM ipl_player_match_features f
        JOIN ipl_player_fantasy_points p
          ON p.match_id = f.match_id
         AND p.player_name = f.player_name
        JOIN ipl_matches m
          ON m.match_id = f.match_id
        """,
        con,
    )
    base["date"] = _parse_dates(base["date"])
    base = base.dropna(subset=["date"]).sort_values(["player_name", "date", "match_id"]).reset_index(drop=True)

    rolling_columns = _table_columns(con, "ipl_player_rolling_features")
    if "player_name" in rolling_columns:
        rolling = pd.read_sql_query(
            """
            SELECT
                player_name,
                match_id,
                avg_runs_last5,
                avg_sr_last5,
                avg_wickets_last5,
                avg_economy_last5,
                avg_fours_last5,
                avg_sixes_last5,
                matches_played_last5
            FROM ipl_player_rolling_features
            """,
            con,
        )
    else:
        history_base = base[
            [
                "player_name",
                "match_id",
                "date",
                "runs_scored",
                "fours",
                "sixes",
                "wickets_taken",
                "strike_rate",
                "economy_rate",
            ]
        ]
        rolling = _rebuild_rolling_features(history_base)

    frame = base.merge(
        rolling[["player_name", "match_id", *ROLLING_COLS]],
        on=["player_name", "match_id"],
        how="left",
    )
    role_lookup = _load_role_lookup(con)
    frame["role_encoded"] = frame["player_name"].map(lambda name: role_lookup.get(name, 1))
    frame[TRAINING_FEATURES] = frame[TRAINING_FEATURES].apply(pd.to_numeric, errors="coerce")
    return frame


def _load_model_artifacts() -> tuple[Any, list[str], dict[str, float]]:
    model = joblib.load(MODEL_PATH)
    with FEATURES_PATH.open("r", encoding="utf-8") as handle:
        feature_list = json.load(handle)
    with MEDIANS_PATH.open("r", encoding="utf-8") as handle:
        medians = json.load(handle)
    return model, feature_list, medians


def get_match_player_pool(
    team1: str,
    team2: str,
    db_path: str | Path,
    season: str = "2026",
) -> pd.DataFrame:
    team1_name = _canonical_team(team1)
    team2_name = _canonical_team(team2)
    if team1_name is None or team2_name is None:
        raise ValueError("team1 and team2 must be non-empty strings")

    con = sqlite3.connect(db_path)
    try:
        pool = pd.read_sql_query(
            """
            SELECT player_name, team, role, is_overseas
            FROM ipl_current_squads
            WHERE team IN (?, ?)
              AND season = ?
              AND is_available = 1
            ORDER BY team, player_name
            """,
            con,
            params=(team1_name, team2_name, str(season)),
        )
    finally:
        con.close()

    if pool.empty:
        raise ValueError(f"No available players found for {team1_name} vs {team2_name}")

    pool = pool.copy()
    pool["player_name"] = pool["player_name"].map(_normalize_text)
    pool["team"] = pool["team"].map(_canonical_team)
    pool["role"] = pool["role"].map(_normalize_role)
    pool["is_overseas"] = pool["is_overseas"].fillna(0).astype(int)
    pool = pool.dropna(subset=["player_name", "team"]).drop_duplicates(
        subset=["player_name"], keep="first"
    )

    counts = pool.groupby("team")["player_name"].nunique().to_dict()
    if counts.get(team1_name, 0) == 0:
        raise ValueError(f"No available players found for team: {team1_name}")
    if counts.get(team2_name, 0) == 0:
        raise ValueError(f"No available players found for team: {team2_name}")
    if len(pool) < 18:
        raise ValueError(
            f"Expected at least 18 available players across {team1_name} and {team2_name}, found {len(pool)}"
        )

    return pool[POOL_COLUMNS].reset_index(drop=True)


def split_pool_by_team(
    team1: str,
    team2: str,
    db_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = get_match_player_pool(team1, team2, db_path)
    team1_name = _canonical_team(team1)
    team2_name = _canonical_team(team2)
    team1_df = pool.loc[pool["team"] == team1_name].reset_index(drop=True)
    team2_df = pool.loc[pool["team"] == team2_name].reset_index(drop=True)
    return team1_df, team2_df


def build_prematch_player_snapshot(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    db_path: str | Path,
) -> pd.DataFrame:
    team1_name = _canonical_team(team1)
    team2_name = _canonical_team(team2)
    venue_name = _normalize_text(venue)
    toss_winner_name = _canonical_team(toss_winner)
    toss_decision_name = _normalize_decision(toss_decision)
    if team1_name is None or team2_name is None:
        raise ValueError("team1 and team2 must be non-empty team names")

    pool = get_match_player_pool(team1_name, team2_name, db_path)

    con = sqlite3.connect(db_path)
    try:
        resolved_pool = _resolve_history_names(pool, con)
        history_names = [
            name for name in resolved_pool["history_player_name"].dropna().astype(str).unique().tolist() if name
        ]
        rolling = _load_latest_rolling(con, history_names)
        career = _load_career_aggregates(con, history_names)
        venue_history = _load_venue_aggregates(con, history_names, venue_name)
    finally:
        con.close()

    snapshot = resolved_pool.copy()
    snapshot["opponent_team"] = snapshot["team"].map(
        lambda team_name: team2_name if team_name == team1_name else team1_name
    )
    snapshot["venue"] = venue_name
    snapshot["toss_team_flag"] = snapshot["team"].eq(toss_winner_name).astype(int)
    snapshot["toss_bat_flag"] = 1 if toss_decision_name == "bat" else 0
    snapshot["role_weight"] = snapshot["role"].map(ROLE_WEIGHTS).fillna(1.0)

    snapshot = snapshot.merge(
        rolling.rename(columns={"player_name": "history_player_name"}),
        on="history_player_name",
        how="left",
    )
    snapshot = snapshot.merge(
        career.rename(columns={"player_name": "history_player_name"}),
        on="history_player_name",
        how="left",
    )
    snapshot = snapshot.merge(
        venue_history.rename(columns={"player_name": "history_player_name"}),
        on="history_player_name",
        how="left",
    )

    fill_pairs = {
        "avg_runs_last5": "career_avg_runs",
        "avg_sr_last5": "career_avg_sr",
        "avg_wickets_last5": "career_avg_wickets",
        "avg_economy_last5": "career_avg_economy",
        "avg_fours_last5": "career_avg_fours",
        "avg_sixes_last5": "career_avg_sixes",
    }
    for target_col, source_col in fill_pairs.items():
        snapshot[target_col] = snapshot[target_col].combine_first(snapshot[source_col])

    snapshot["matches_played_total"] = snapshot["matches_played_total"].fillna(0).astype(int)
    snapshot["matches_played_last5"] = snapshot["matches_played_last5"].fillna(
        snapshot["matches_played_total"].clip(upper=5)
    )
    snapshot["avg_fantasy_points"] = snapshot["avg_fantasy_points"].fillna(0.0)
    snapshot["max_fantasy_points"] = snapshot["max_fantasy_points"].fillna(0.0)
    snapshot["venue_matches"] = snapshot["venue_matches"].fillna(0).astype(int)
    snapshot["venue_avg_fantasy_points"] = snapshot["venue_avg_fantasy_points"].fillna(
        snapshot["avg_fantasy_points"]
    )

    zero_fill = [
        "avg_batting_points",
        "avg_bowling_points",
        "avg_sr_points",
        "avg_economy_points",
        "batting_match_share",
        "bowling_match_share",
        "fifty_rate",
        "hundred_rate",
        "wicket_haul_rate",
        "duck_rate",
        "avg_pp_runs",
        "avg_mid_runs",
        "avg_death_runs",
        "avg_pp_wickets",
        "avg_death_wickets",
        "avg_death_economy",
        "avg_runs_last5",
        "avg_sr_last5",
        "avg_wickets_last5",
        "avg_economy_last5",
        "avg_fours_last5",
        "avg_sixes_last5",
        "matches_played_last5",
    ]
    for column in zero_fill:
        snapshot[column] = snapshot[column].fillna(0.0)

    return snapshot[
        [
            "player_name",
            "team",
            "opponent_team",
            "role",
            "is_overseas",
            "venue",
            "toss_team_flag",
            "toss_bat_flag",
            "avg_runs_last5",
            "avg_sr_last5",
            "avg_wickets_last5",
            "avg_economy_last5",
            "avg_fours_last5",
            "avg_sixes_last5",
            "matches_played_last5",
            "matches_played_total",
            "avg_fantasy_points",
            "max_fantasy_points",
            "avg_batting_points",
            "avg_bowling_points",
            "avg_sr_points",
            "avg_economy_points",
            "batting_match_share",
            "bowling_match_share",
            "fifty_rate",
            "hundred_rate",
            "wicket_haul_rate",
            "duck_rate",
            "avg_pp_runs",
            "avg_mid_runs",
            "avg_death_runs",
            "avg_pp_wickets",
            "avg_death_wickets",
            "avg_death_economy",
            "venue_matches",
            "venue_avg_fantasy_points",
            "role_weight",
        ]
    ].reset_index(drop=True)


def train_fantasy_ranker(db_path: str | Path) -> dict[str, float | str | int]:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    con = sqlite3.connect(db_path)
    try:
        training = _build_training_frame(con)
    finally:
        con.close()

    if training.empty:
        raise ValueError("No historical IPL rows available to train fantasy ranker")

    training["date"] = _parse_dates(training["date"])
    training = training.dropna(subset=["date"]).copy()

    train_mask = training["date"] < pd.Timestamp("2025-01-01")
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Temporal split failed; need rows on both sides of 2025-01-01")

    medians = training.loc[train_mask, TRAINING_FEATURES].median(numeric_only=True).to_dict()
    medians = {key: float(value) if pd.notna(value) else 0.0 for key, value in medians.items()}

    x_train = training.loc[train_mask, TRAINING_FEATURES].fillna(medians)
    y_train = training.loc[train_mask, "total_fantasy_points"].astype(float)
    x_test = training.loc[test_mask, TRAINING_FEATURES].fillna(medians)
    y_test = training.loc[test_mask, "total_fantasy_points"].astype(float)

    backend = "gradient_boosting"
    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        backend = "xgboost"
    except Exception:
        model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            max_depth=4,
            random_state=42,
        )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    metrics = {
        "backend": backend,
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }

    joblib.dump(model, MODEL_PATH)
    with FEATURES_PATH.open("w", encoding="utf-8") as handle:
        json.dump(TRAINING_FEATURES, handle, indent=2)
    with MEDIANS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(medians, handle, indent=2)

    return metrics


def rank_match_players(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    db_path: str | Path,
) -> pd.DataFrame:
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists() or not MEDIANS_PATH.exists():
        train_fantasy_ranker(db_path)

    model, feature_list, medians = _load_model_artifacts()
    snapshot = build_prematch_player_snapshot(team1, team2, venue, toss_winner, toss_decision, db_path)

    scoring = snapshot.copy()
    scoring["role_encoded"] = scoring["role"].map(ROLE_ENCODING).fillna(1).astype(int)
    model_input = scoring[feature_list].apply(pd.to_numeric, errors="coerce").fillna(medians)
    predicted_points = model.predict(model_input)
    scoring["predicted_points"] = predicted_points

    from ipl.models.win_predictor import predict_winner

    winner_prediction = predict_winner(team1, team2, venue, toss_winner, toss_decision, db_path)
    predicted_winner = winner_prediction["predicted_winner"]

    scoring["venue_factor"] = np.where(
        (scoring["venue_matches"] >= 3)
        & (scoring["venue_avg_fantasy_points"] > scoring["avg_fantasy_points"]),
        1.05,
        1.00,
    )
    toss_factor = np.ones(len(scoring), dtype=float)
    batter_boost = (
        scoring["role"].isin(BATTER_ROLES)
        & (scoring["toss_team_flag"] == 1)
        & (scoring["toss_bat_flag"] == 1)
    )
    bowler_boost = (
        scoring["role"].isin(BOWLER_ROLES)
        & (scoring["toss_team_flag"] == 0)
        & (scoring["toss_bat_flag"] == 1)
    )
    toss_factor = np.where(batter_boost | bowler_boost, 1.03, toss_factor)
    scoring["toss_factor"] = toss_factor
    scoring["win_factor"] = np.where(scoring["team"] == predicted_winner, 1.06, 0.94)

    scoring["adjusted_score"] = (
        scoring["predicted_points"]
        * scoring["role_weight"]
        * scoring["venue_factor"]
        * scoring["toss_factor"]
        * scoring["win_factor"]
    )

    min_score = float(scoring["adjusted_score"].min())
    max_score = float(scoring["adjusted_score"].max())
    if np.isclose(max_score, min_score):
        scoring["fantasy_probability_pct"] = 50.0
    else:
        scoring["fantasy_probability_pct"] = (
            100.0 * (scoring["adjusted_score"] - min_score) / (max_score - min_score)
        )
    scoring["fantasy_probability_pct"] = scoring["fantasy_probability_pct"].clip(0.0, 100.0)
    scoring["decision"] = np.where(scoring["fantasy_probability_pct"] >= 40.0, "BET", "PASS")
    scoring["captain_candidate_score"] = scoring["adjusted_score"]
    scoring["vice_captain_candidate_score"] = scoring["adjusted_score"] * 0.92

    return scoring[
        [
            "player_name",
            "team",
            "role",
            "adjusted_score",
            "fantasy_probability_pct",
            "decision",
            "captain_candidate_score",
            "vice_captain_candidate_score",
        ]
    ].sort_values(
        ["adjusted_score", "fantasy_probability_pct", "player_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def run_match_fantasy_model(
    team1: str,
    team2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    db_path: str | Path,
) -> dict[str, Any]:
    ranked = rank_match_players(team1, team2, venue, toss_winner, toss_decision, db_path)

    def _ordered(frame: pd.DataFrame, decision: str) -> pd.DataFrame:
        subset = frame.loc[frame["decision"] == decision].copy()
        return subset.sort_values(
            ["adjusted_score", "fantasy_probability_pct", "player_name"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)

    def _enforce_team_cap(
        selected: pd.DataFrame,
        remaining: pd.DataFrame,
        max_per_team: int = 7,
    ) -> pd.DataFrame:
        selected = selected.copy().reset_index(drop=True)
        remaining = remaining.copy().reset_index(drop=True)
        while True:
            counts = selected["team"].value_counts()
            overfull = [team_name for team_name, count in counts.items() if count > max_per_team]
            if not overfull:
                break
            over_team = overfull[0]
            replacement_pool = remaining.loc[remaining["team"] != over_team].sort_values(
                ["adjusted_score", "fantasy_probability_pct", "player_name"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            if replacement_pool.empty:
                raise ValueError(f"Unable to enforce max {max_per_team} players from one team")

            drop_idx = (
                selected.loc[selected["team"] == over_team]
                .sort_values(
                    ["adjusted_score", "fantasy_probability_pct", "player_name"],
                    ascending=[True, True, False],
                    kind="mergesort",
                )
                .index[0]
            )
            replacement = replacement_pool.iloc[0]
            selected = selected.drop(index=drop_idx).reset_index(drop=True)
            remaining = remaining.drop(index=replacement.name).reset_index(drop=True)
            selected = pd.concat([selected, replacement.to_frame().T], ignore_index=True)
        return selected

    bet_rows = _ordered(ranked, "BET")
    pass_rows = _ordered(ranked, "PASS")

    selected = bet_rows.head(11).copy()
    extra_bet = bet_rows.iloc[len(selected) :].copy()
    if len(selected) < 11:
        fill_rows = pass_rows.head(11 - len(selected)).copy()
        selected = pd.concat([selected, fill_rows], ignore_index=True)
        remaining = pd.concat([extra_bet, pass_rows.iloc[len(fill_rows) :].copy()], ignore_index=True)
    else:
        remaining = pd.concat([extra_bet, pass_rows], ignore_index=True)

    selected = _enforce_team_cap(selected, remaining, max_per_team=7)
    selected = selected.sort_values(
        ["adjusted_score", "fantasy_probability_pct", "player_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    if len(selected) != 11:
        raise ValueError(f"Fantasy selector returned {len(selected)} players instead of 11")
    team_counts = selected["team"].value_counts().to_dict()
    if any(count > 7 for count in team_counts.values()):
        raise ValueError(f"Max-7 team rule violated: {team_counts}")

    selected["captain"] = False
    selected["vice_captain"] = False
    selected.loc[selected.index[0], "captain"] = True
    selected.loc[selected.index[1], "vice_captain"] = True

    return {
        "match": f"{_canonical_team(team1)} vs {_canonical_team(team2)}",
        "selected_players": [
            {
                "player_name": row.player_name,
                "team": row.team,
                "role": row.role,
                "fantasy_probability_pct": float(row.fantasy_probability_pct),
                "decision": row.decision,
                "captain": bool(row.captain),
                "vice_captain": bool(row.vice_captain),
            }
            for row in selected.itertuples(index=False)
        ],
    }


if __name__ == "__main__":
    db = _default_db_path()

    print("Training fantasy ranker...")
    start = time.time()
    metrics = train_fantasy_ranker(db)
    print(metrics)
    print(f"Training done in {time.time() - start:.1f}s")

    print("\nRunning sample fantasy selection...")
    result = run_match_fantasy_model(
        "Mumbai Indians",
        "Chennai Super Kings",
        "Wankhede Stadium",
        "Mumbai Indians",
        "bat",
        db,
    )

    print(result["match"])
    for player in result["selected_players"]:
        flags: list[str] = []
        if player["captain"]:
            flags.append("C")
        if player["vice_captain"]:
            flags.append("VC")
        flag_text = f" [{' '.join(flags)}]" if flags else ""
        print(
            f"{player['player_name']} | {player['team']} | {player['role']} | "
            f"{player['fantasy_probability_pct']:.1f}% | {player['decision']}{flag_text}"
        )
