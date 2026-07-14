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
    "Wicket-Keeper": 1.00,
    "All-Rounder": 1.06,
    "Batsman": 1.00,
    "Bowler": 0.98,
}
ROLE_ENCODING = {
    "Wicket-Keeper": 3,
    "All-Rounder": 2,
    "Batsman": 1,
    "Bowler": 0,
}
BATTER_ROLES = {"Batsman", "Wicket-Keeper", "All-Rounder"}
BOWLER_ROLES = {"Bowler", "All-Rounder"}
BOWLER_CREDIT_WICKET_TYPES = {
    "bowled",
    "caught",
    "caught and bowled",
    "hit wicket",
    "lbw",
    "stumped",
}
DREAM11_ROLE_LIMITS = {
    "Wicket-Keeper": (1, 4),
    "Batsman": (3, 6),
    "All-Rounder": (1, 4),
    "Bowler": (3, 6),
}
ROLE_ORDER = ("Wicket-Keeper", "Batsman", "All-Rounder", "Bowler")
NO_MARKET_SOURCE = "none_wired"
LEAN_PRIORITY_EDGE_PCT = 2.0
BET_PRIORITY_EDGE_PCT = 4.0
MIN_CONTEST_UNITS = 0.25
MAX_CONTEST_UNITS = 1.5
MODEL_POINT_WEIGHT_FLOOR = 0.20
MODEL_POINT_WEIGHT_RANGE = 0.20
MODEL_POINT_SAMPLE_MATCHES = 20.0
RECENT_FORM_WEIGHT = 0.60
CAREER_FORM_WEIGHT = 0.40
EXPERIENCE_FACTOR_FLOOR = 0.72
LOW_SAMPLE_CAP_MATCHES = 6.0
LOW_SAMPLE_CAP_BASE_POINTS = 26.0
LOW_SAMPLE_CAP_POINTS_PER_MATCH = 2.5
FAVORED_TEAM_FACTOR = 1.03
UNDERDOG_TEAM_FACTOR = 0.97
TOSS_ROLE_FACTOR = 1.02

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


def _table_exists(con: sqlite3.Connection, table_name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    ).fetchone()
    return row is not None


def _placeholders(count: int) -> str:
    return ", ".join(["?"] * count)


def _player_tokens(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def _player_key(name: str) -> str:
    return " ".join(_player_tokens(name))


def _player_identity_key(name: str) -> str:
    tokens = _player_tokens(name)
    if len(tokens) >= 3 and all(len(token) == 1 for token in tokens[:-1]):
        return f"{''.join(tokens[:-1])} {tokens[-1]}"
    return " ".join(tokens)


def _has_leading_initial_history_match(
    current_tokens: list[str],
    candidate_tokens: list[str],
) -> bool:
    return (
        len(current_tokens) >= 2
        and len(candidate_tokens) > len(current_tokens)
        and candidate_tokens[-len(current_tokens) :] == current_tokens
        and all(len(token) <= 2 for token in candidate_tokens[: -len(current_tokens)])
    )


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

        scored_candidates: list[tuple[int, int, int, int, str]] = []
        for candidate in candidates_by_last.get(last_name, []):
            candidate_tokens = _player_tokens(candidate)
            candidate_prefix = _player_prefix(candidate)
            leading_initial_match = _has_leading_initial_history_match(tokens, candidate_tokens)
            if not leading_initial_match and current_first and candidate_prefix[:1] != current_first:
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
                    1 if leading_initial_match else 0,
                    exact_prefix,
                    overlap * 100 + total_matches,
                    candidate,
                )
            )

        if scored_candidates:
            scored_candidates.sort(reverse=True)
            resolved_names.append(scored_candidates[0][4])
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


def _load_matchup_aggregates(
    con: sqlite3.Connection,
    player_names: list[str],
) -> pd.DataFrame:
    columns = [
        "history_player_name",
        "opponent_team",
        "h2h_batting_balls",
        "h2h_batting_runs",
        "h2h_batting_dismissals",
        "h2h_bowling_balls",
        "h2h_bowling_runs",
        "h2h_bowling_wickets",
    ]
    if not player_names or not _table_exists(con, "ipl_deliveries"):
        return pd.DataFrame(columns=columns)

    wicket_types = tuple(sorted(BOWLER_CREDIT_WICKET_TYPES))
    wicket_placeholders = _placeholders(len(wicket_types))
    batting = pd.read_sql_query(
        f"""
        SELECT
            striker AS history_player_name,
            bowling_team AS opponent_team,
            SUM(CASE WHEN COALESCE(wides, 0) = 0 THEN 1 ELSE 0 END) AS h2h_batting_balls,
            SUM(COALESCE(runs_off_bat, 0)) AS h2h_batting_runs,
            SUM(
                CASE
                    WHEN player_dismissed = striker
                     AND LOWER(COALESCE(wicket_type, '')) IN ({wicket_placeholders})
                    THEN 1 ELSE 0
                END
            ) AS h2h_batting_dismissals
        FROM ipl_deliveries
        WHERE striker IN ({_placeholders(len(player_names))})
          AND bowling_team IS NOT NULL
        GROUP BY striker, bowling_team
        """,
        con,
        params=wicket_types + tuple(player_names),
    )
    bowling = pd.read_sql_query(
        f"""
        SELECT
            bowler AS history_player_name,
            batting_team AS opponent_team,
            SUM(
                CASE
                    WHEN COALESCE(wides, 0) = 0 AND COALESCE(noballs, 0) = 0
                    THEN 1 ELSE 0
                END
            ) AS h2h_bowling_balls,
            SUM(COALESCE(runs_off_bat, 0) + COALESCE(wides, 0) + COALESCE(noballs, 0)) AS h2h_bowling_runs,
            SUM(
                CASE
                    WHEN player_dismissed IS NOT NULL
                     AND LOWER(COALESCE(wicket_type, '')) IN ({wicket_placeholders})
                    THEN 1 ELSE 0
                END
            ) AS h2h_bowling_wickets
        FROM ipl_deliveries
        WHERE bowler IN ({_placeholders(len(player_names))})
          AND batting_team IS NOT NULL
        GROUP BY bowler, batting_team
        """,
        con,
        params=wicket_types + tuple(player_names),
    )

    for frame in (batting, bowling):
        if not frame.empty:
            frame["history_player_name"] = frame["history_player_name"].map(_normalize_text)
            frame["opponent_team"] = frame["opponent_team"].map(_canonical_team)
            numeric_columns = [
                column
                for column in frame.columns
                if column not in {"history_player_name", "opponent_team"}
            ]
            frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            frame = frame.groupby(["history_player_name", "opponent_team"], as_index=False)[
                numeric_columns
            ].sum()
            if "h2h_batting_balls" in frame.columns:
                batting = frame
            else:
                bowling = frame

    if batting.empty and bowling.empty:
        return pd.DataFrame(columns=columns)
    if batting.empty:
        matchup = bowling
    elif bowling.empty:
        matchup = batting
    else:
        matchup = batting.merge(
            bowling,
            on=["history_player_name", "opponent_team"],
            how="outer",
        )
    for column in columns:
        if column not in matchup.columns:
            matchup[column] = 0.0
    return matchup[columns]


def _load_bowling_opportunity(
    con: sqlite3.Connection,
    player_names: list[str],
) -> pd.DataFrame:
    columns = ["history_player_name", "last_match_overs", "last_match_balls_bowled"]
    if not player_names or not _table_exists(con, "ipl_player_match_features"):
        return pd.DataFrame(columns=columns)

    rows = pd.read_sql_query(
        f"""
        SELECT
            f.player_name AS history_player_name,
            f.match_id,
            m.date,
            f.overs_bowled,
            f.balls_bowled
        FROM ipl_player_match_features f
        JOIN ipl_matches m
          ON m.match_id = f.match_id
        WHERE f.player_name IN ({_placeholders(len(player_names))})
        """,
        con,
        params=tuple(player_names),
    )
    if rows.empty:
        return pd.DataFrame(columns=columns)

    rows["date"] = _parse_dates(rows["date"])
    rows = rows.dropna(subset=["date"]).sort_values(["history_player_name", "date", "match_id"])
    latest = rows.groupby("history_player_name", as_index=False).tail(1).copy()
    latest["history_player_name"] = latest["history_player_name"].map(_normalize_text)
    latest["last_match_overs"] = pd.to_numeric(latest["overs_bowled"], errors="coerce").fillna(0.0)
    latest["last_match_balls_bowled"] = pd.to_numeric(
        latest["balls_bowled"], errors="coerce"
    ).fillna(0.0)
    return latest[columns].reset_index(drop=True)


def _add_matchup_and_opportunity_factors(snapshot: pd.DataFrame) -> pd.DataFrame:
    result = snapshot.copy()
    matchup_cols = [
        "h2h_batting_balls",
        "h2h_batting_runs",
        "h2h_batting_dismissals",
        "h2h_bowling_balls",
        "h2h_bowling_runs",
        "h2h_bowling_wickets",
        "last_match_overs",
        "last_match_balls_bowled",
    ]
    for column in matchup_cols:
        if column not in result.columns:
            result[column] = 0.0
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0.0)

    batting_balls = result["h2h_batting_balls"].clip(lower=0.0)
    bowling_balls = result["h2h_bowling_balls"].clip(lower=0.0)
    batting_rpb = np.divide(
        result["h2h_batting_runs"],
        batting_balls,
        out=np.full(len(result), 1.15, dtype=float),
        where=batting_balls.to_numpy(dtype=float) > 0,
    )
    batting_dismissal_rate = np.divide(
        result["h2h_batting_dismissals"],
        batting_balls,
        out=np.full(len(result), 0.045, dtype=float),
        where=batting_balls.to_numpy(dtype=float) > 0,
    )
    bowling_rpb = np.divide(
        result["h2h_bowling_runs"],
        bowling_balls,
        out=np.full(len(result), 1.15, dtype=float),
        where=bowling_balls.to_numpy(dtype=float) > 0,
    )
    bowling_wicket_rate = np.divide(
        result["h2h_bowling_wickets"],
        bowling_balls,
        out=np.full(len(result), 0.045, dtype=float),
        where=bowling_balls.to_numpy(dtype=float) > 0,
    )

    batting_delta = np.clip(
        ((batting_rpb - 1.15) * 0.08) - ((batting_dismissal_rate - 0.045) * 0.90),
        -0.08,
        0.08,
    )
    bowling_delta = np.clip(
        ((1.15 - bowling_rpb) * 0.05) + ((bowling_wicket_rate - 0.045) * 1.10),
        -0.08,
        0.08,
    )
    role = result["role"].fillna("")
    batting_role = role.isin(BATTER_ROLES).to_numpy(dtype=float)
    bowling_role = role.isin(BOWLER_ROLES).to_numpy(dtype=float)
    role_denominator = np.maximum(batting_role + bowling_role, 1.0)
    blended_delta = ((batting_delta * batting_role) + (bowling_delta * bowling_role)) / role_denominator
    evidence = np.minimum(1.0, np.sqrt((batting_balls + bowling_balls).to_numpy(dtype=float) / 72.0))
    result["matchup_evidence_balls"] = (batting_balls + bowling_balls).astype(float)
    result["matchup_factor"] = np.clip(1.0 + (blended_delta * evidence), 0.92, 1.08)

    opportunity_factor = np.ones(len(result), dtype=float)
    bowling_roles = role.isin(BOWLER_ROLES).to_numpy(dtype=bool)
    full_quota = (result["last_match_overs"] >= 3.5).to_numpy(dtype=bool)
    partial_quota = (
        (result["last_match_overs"] >= 2.0) & (result["last_match_overs"] < 3.5)
    ).to_numpy(dtype=bool)
    no_recent_overs = (
        (result["last_match_overs"] == 0)
        & (result["matches_played_total"] > 0)
    ).to_numpy(dtype=bool)
    opportunity_factor = np.where(bowling_roles & full_quota, 1.06, opportunity_factor)
    opportunity_factor = np.where(bowling_roles & partial_quota, 1.02, opportunity_factor)
    opportunity_factor = np.where(bowling_roles & no_recent_overs, 0.95, opportunity_factor)
    result["bowling_opportunity_factor"] = opportunity_factor
    return result


def _numeric_column(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default).astype(float)


def _add_stabilized_point_estimates(
    scoring: pd.DataFrame,
    predicted_points: np.ndarray,
) -> pd.DataFrame:
    result = scoring.copy()
    result["predicted_points"] = pd.Series(predicted_points, index=result.index).astype(float)

    runs = _numeric_column(result, "avg_runs_last5")
    fours = _numeric_column(result, "avg_fours_last5")
    sixes = _numeric_column(result, "avg_sixes_last5")
    wickets = _numeric_column(result, "avg_wickets_last5")
    strike_rate = _numeric_column(result, "avg_sr_last5")
    economy = _numeric_column(result, "avg_economy_last5")
    career_points = _numeric_column(result, "avg_fantasy_points").clip(lower=0.0, upper=95.0)
    matches_total = _numeric_column(result, "matches_played_total")

    batting_points = runs + fours + (sixes * 2.0)
    bowling_points = wickets * 25.0
    strike_rate_points = np.select(
        [
            (batting_points >= 10.0) & (strike_rate > 170.0),
            (batting_points >= 10.0) & (strike_rate > 150.0),
            (batting_points >= 10.0) & (strike_rate >= 130.0),
            (batting_points >= 10.0) & (strike_rate < 70.0) & (strike_rate > 0.0),
        ],
        [6.0, 4.0, 2.0, -2.0],
        default=0.0,
    )
    economy_points = np.select(
        [
            (wickets > 0.0) & (economy > 0.0) & (economy < 6.0),
            (wickets > 0.0) & (economy > 10.0),
        ],
        [2.0, -2.0],
        default=0.0,
    )
    recent_form_points = pd.Series(
        batting_points + bowling_points + strike_rate_points + economy_points,
        index=result.index,
    ).clip(lower=0.0, upper=95.0)

    empirical_points = pd.Series(
        np.where(
            matches_total > 0.0,
            (recent_form_points * RECENT_FORM_WEIGHT) + (career_points * CAREER_FORM_WEIGHT),
            recent_form_points,
        ),
        index=result.index,
    ).clip(lower=0.0, upper=95.0)
    sample_ratio = (matches_total / MODEL_POINT_SAMPLE_MATCHES).clip(lower=0.0, upper=1.0)
    model_weight = MODEL_POINT_WEIGHT_FLOOR + (MODEL_POINT_WEIGHT_RANGE * sample_ratio)
    stabilized_points = (
        result["predicted_points"] * model_weight
        + empirical_points * (1.0 - model_weight)
    ).clip(lower=0.0, upper=95.0)

    low_sample_cap = (
        LOW_SAMPLE_CAP_BASE_POINTS
        + matches_total.clip(lower=0.0, upper=LOW_SAMPLE_CAP_MATCHES)
        * LOW_SAMPLE_CAP_POINTS_PER_MATCH
    )
    stabilized_points = pd.Series(
        np.where(
            matches_total < LOW_SAMPLE_CAP_MATCHES,
            np.minimum(stabilized_points, low_sample_cap),
            stabilized_points,
        ),
        index=result.index,
    )
    experience_factor = (
        EXPERIENCE_FACTOR_FLOOR
        + ((1.0 - EXPERIENCE_FACTOR_FLOOR) * sample_ratio)
    ).clip(lower=EXPERIENCE_FACTOR_FLOOR, upper=1.0)

    result["recent_form_points"] = recent_form_points
    result["empirical_points"] = empirical_points
    result["stabilized_points"] = stabilized_points
    result["experience_factor"] = experience_factor
    return result


def decision_from_priority_edge(priority_edge_pct: float | int | None) -> str:
    try:
        edge = float(priority_edge_pct)
    except (TypeError, ValueError):
        return "PASS"
    if edge >= BET_PRIORITY_EDGE_PCT:
        return "BET"
    if edge >= LEAN_PRIORITY_EDGE_PCT:
        return "LEAN"
    return "PASS"


def contest_units_from_priority_edge(priority_edge_pct: float | int | None) -> float:
    if decision_from_priority_edge(priority_edge_pct) == "PASS":
        return 0.0
    edge = max(float(priority_edge_pct), LEAN_PRIORITY_EDGE_PCT)
    curve = min(1.0, (edge - LEAN_PRIORITY_EDGE_PCT) / 28.0)
    units = MIN_CONTEST_UNITS + (curve * (MAX_CONTEST_UNITS - MIN_CONTEST_UNITS))
    if decision_from_priority_edge(priority_edge_pct) == "LEAN":
        units *= 0.60
    return round(min(MAX_CONTEST_UNITS, max(MIN_CONTEST_UNITS, units)), 2)


def _role_name(value: Any) -> str:
    role = _normalize_role(value) or "Batsman"
    return role if role in DREAM11_ROLE_LIMITS else "Batsman"


def _role_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["role"].map(_role_name).value_counts().to_dict()
    return {role: int(counts.get(role, 0)) for role in ROLE_ORDER}


def _lineup_constraints_summary(
    selected: pd.DataFrame,
    max_per_team: int = 7,
) -> dict[str, Any]:
    role_counts = _role_counts(selected)
    team_counts = {str(team): int(count) for team, count in selected["team"].value_counts().items()}
    role_limits = {
        role: {"min": limits[0], "max": limits[1]}
        for role, limits in DREAM11_ROLE_LIMITS.items()
    }
    role_ok = all(
        limits[0] <= role_counts.get(role, 0) <= limits[1]
        for role, limits in DREAM11_ROLE_LIMITS.items()
    )
    team_ok = all(count <= max_per_team for count in team_counts.values())
    return {
        "role_counts": role_counts,
        "role_limits": role_limits,
        "team_counts": team_counts,
        "max_per_team": max_per_team,
        "satisfied": bool(role_ok and team_ok and len(selected) == 11),
    }


def _select_valid_fantasy_xi(
    ranked: pd.DataFrame,
    max_per_team: int = 7,
) -> pd.DataFrame:
    frame = ranked.copy().reset_index(drop=True)
    frame["_player_identity_key"] = frame["player_name"].map(_player_identity_key)
    frame = frame.sort_values(
        ["adjusted_score", "fantasy_probability_pct", "player_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).drop_duplicates(subset=["_player_identity_key"], keep="first").reset_index(drop=True)
    if len(frame) < 11:
        raise ValueError(f"Expected at least 11 candidate players, found {len(frame)}")

    frame["_role_key"] = frame["role"].map(_role_name)
    teams = sorted(str(team) for team in frame["team"].dropna().unique())
    team_index = {team: index for index, team in enumerate(teams)}
    role_index = {role: index for index, role in enumerate(ROLE_ORDER)}

    available_role_counts = frame["_role_key"].value_counts().to_dict()
    role_minimums = tuple(
        min(DREAM11_ROLE_LIMITS[role][0], int(available_role_counts.get(role, 0)))
        for role in ROLE_ORDER
    )
    role_maximums = tuple(
        min(DREAM11_ROLE_LIMITS[role][1], int(available_role_counts.get(role, 0)))
        for role in ROLE_ORDER
    )

    # State = (players, role_counts_tuple, team_counts_tuple); value = (score, indices)
    zero_roles = tuple(0 for _ in ROLE_ORDER)
    zero_teams = tuple(0 for _ in teams)
    states: dict[tuple[int, tuple[int, ...], tuple[int, ...]], tuple[float, tuple[int, ...]]] = {
        (0, zero_roles, zero_teams): (0.0, ())
    }

    for idx, row in frame.iterrows():
        role_pos = role_index[str(row["_role_key"])]
        team_pos = team_index[str(row["team"])]
        score = float(row["adjusted_score"])
        next_states = dict(states)
        for (count, role_counts, team_counts), (total_score, indices) in states.items():
            if count >= 11:
                continue
            if role_counts[role_pos] >= role_maximums[role_pos]:
                continue
            if team_counts[team_pos] >= max_per_team:
                continue

            new_role_counts = list(role_counts)
            new_team_counts = list(team_counts)
            new_role_counts[role_pos] += 1
            new_team_counts[team_pos] += 1
            new_key = (count + 1, tuple(new_role_counts), tuple(new_team_counts))
            new_score = total_score + score
            previous = next_states.get(new_key)
            if previous is None or new_score > previous[0]:
                next_states[new_key] = (new_score, indices + (idx,))
        states = next_states

    best: tuple[float, tuple[int, ...]] | None = None
    for (count, role_counts, team_counts), candidate in states.items():
        if count != 11:
            continue
        if any(role_counts[index] < role_minimums[index] for index in range(len(ROLE_ORDER))):
            continue
        if any(team_count > max_per_team for team_count in team_counts):
            continue
        if best is None or candidate[0] > best[0]:
            best = candidate

    if best is None:
        selected = frame.head(11).copy()
    else:
        selected = frame.loc[list(best[1])].copy()
    return selected.drop(columns=["_role_key", "_player_identity_key"], errors="ignore").sort_values(
        ["adjusted_score", "fantasy_probability_pct", "player_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


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
    pool["_player_identity_key"] = pool["player_name"].map(_player_identity_key)
    pool = pool.drop_duplicates(subset=["team", "_player_identity_key"], keep="first")

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
        matchup_history = _load_matchup_aggregates(con, history_names)
        bowling_opportunity = _load_bowling_opportunity(con, history_names)
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
    snapshot = snapshot.merge(
        matchup_history,
        on=["history_player_name", "opponent_team"],
        how="left",
    )
    snapshot = snapshot.merge(
        bowling_opportunity,
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

    snapshot = _add_matchup_and_opportunity_factors(snapshot)

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
            "h2h_batting_balls",
            "h2h_batting_runs",
            "h2h_batting_dismissals",
            "h2h_bowling_balls",
            "h2h_bowling_runs",
            "h2h_bowling_wickets",
            "matchup_evidence_balls",
            "matchup_factor",
            "last_match_overs",
            "last_match_balls_bowled",
            "bowling_opportunity_factor",
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
    scoring = _add_stabilized_point_estimates(scoring, predicted_points)

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
    toss_factor = np.where(batter_boost | bowler_boost, TOSS_ROLE_FACTOR, toss_factor)
    scoring["toss_factor"] = toss_factor
    scoring["win_factor"] = np.where(
        scoring["team"] == predicted_winner,
        FAVORED_TEAM_FACTOR,
        UNDERDOG_TEAM_FACTOR,
    )

    scoring["adjusted_score"] = (
        scoring["stabilized_points"]
        * scoring["role_weight"]
        * scoring["venue_factor"]
        * scoring["toss_factor"]
        * scoring["win_factor"]
        * scoring["matchup_factor"]
        * scoring["bowling_opportunity_factor"]
        * scoring["experience_factor"]
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
    ordered_for_baseline = scoring.sort_values(
        ["adjusted_score", "fantasy_probability_pct", "player_name"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(ordered_for_baseline) > 11:
        replacement_probability = float(ordered_for_baseline.loc[11, "fantasy_probability_pct"])
    else:
        replacement_probability = float(ordered_for_baseline["fantasy_probability_pct"].min())
    scoring["selection_baseline_pct"] = replacement_probability
    scoring["priority_edge_pct"] = scoring["fantasy_probability_pct"] - replacement_probability
    scoring["decision"] = scoring["priority_edge_pct"].map(decision_from_priority_edge)
    scoring["units"] = scoring["priority_edge_pct"].map(contest_units_from_priority_edge)
    scoring["market_source"] = NO_MARKET_SOURCE
    scoring["has_market_price"] = False
    scoring["market_probability_pct"] = None
    scoring["market_edge_pct"] = None
    scoring["captain_candidate_score"] = scoring["adjusted_score"] * 2.0
    scoring["vice_captain_candidate_score"] = scoring["adjusted_score"] * 1.5

    return scoring[
        [
            "player_name",
            "team",
            "role",
            "adjusted_score",
            "fantasy_probability_pct",
            "selection_baseline_pct",
            "priority_edge_pct",
            "decision",
            "units",
            "market_source",
            "has_market_price",
            "market_probability_pct",
            "market_edge_pct",
            "matchup_evidence_balls",
            "matchup_factor",
            "last_match_overs",
            "bowling_opportunity_factor",
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
    selected = _select_valid_fantasy_xi(ranked, max_per_team=7)

    if len(selected) != 11:
        raise ValueError(f"Fantasy selector returned {len(selected)} players instead of 11")
    team_counts = selected["team"].value_counts().to_dict()
    if any(count > 7 for count in team_counts.values()):
        raise ValueError(f"Max-7 team rule violated: {team_counts}")
    lineup_constraints = _lineup_constraints_summary(selected, max_per_team=7)

    selected["captain"] = False
    selected["vice_captain"] = False
    selected.loc[selected.index[0], "captain"] = True
    selected.loc[selected.index[1], "vice_captain"] = True
    selected["captain_multiplier"] = 1.0
    selected.loc[selected["captain"], "captain_multiplier"] = 2.0
    selected.loc[selected["vice_captain"], "captain_multiplier"] = 1.5
    selected["captaincy_boost_points"] = selected["adjusted_score"] * (
        selected["captain_multiplier"] - 1.0
    )

    return {
        "match": f"{_canonical_team(team1)} vs {_canonical_team(team2)}",
        "market": {
            "has_market": False,
            "source": NO_MARKET_SOURCE,
            "note": (
                "No Dream11/MyTeam11 ownership, salary, contest-cost, or player-prop "
                "market source is wired; IPL fantasy sizing uses model priority only."
            ),
        },
        "lineup_constraints": lineup_constraints,
        "selected_players": [
            {
                "player_name": row.player_name,
                "team": row.team,
                "role": row.role,
                "adjusted_score": float(row.adjusted_score),
                "fantasy_probability_pct": float(row.fantasy_probability_pct),
                "selection_baseline_pct": float(row.selection_baseline_pct),
                "priority_edge_pct": float(row.priority_edge_pct),
                "decision": row.decision,
                "units": float(row.units),
                "market_source": row.market_source,
                "has_market_price": bool(row.has_market_price),
                "market_probability_pct": row.market_probability_pct,
                "market_edge_pct": row.market_edge_pct,
                "matchup_evidence_balls": float(row.matchup_evidence_balls),
                "matchup_factor": float(row.matchup_factor),
                "last_match_overs": float(row.last_match_overs),
                "bowling_opportunity_factor": float(row.bowling_opportunity_factor),
                "captain": bool(row.captain),
                "vice_captain": bool(row.vice_captain),
                "captain_multiplier": float(row.captain_multiplier),
                "captaincy_boost_points": float(row.captaincy_boost_points),
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
