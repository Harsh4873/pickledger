"""
Feature engineering v2 — modern MLB prediction features.

Key differences from the legacy v1 stack:

  * No heuristic probabilities as model inputs. The heuristic was a blurry
    handcrafted score that duplicated what the model should be learning.
    Including it as a feature made the LogisticRegression mirror the heuristic
    and blocked it from learning the underlying signals.

  * Market-residual features. Vig-free implied probability and market total
    line are first-class inputs. The model learns to nudge *around* the market
    consensus, which is by far the strongest available signal in sports
    prediction.

  * Run-differential based team strength. Pythagorean win % derived from runs
    scored/allowed is a substantially better forward-looking indicator than
    naive W/L record, especially in short samples.

  * Sharper starter pitching profile. Uses K/9, BB/9, HR/9, K-BB% in addition
    to ERA/FIP/WHIP. Those rate stats stabilize much faster than ERA.

  * Handedness matchup features (LvL/LvR/RvL/RvR) instead of raw L/R dummies.

  * Richer bullpen state (ERA and usage in separate 1/3/7 day windows).

  * Environment (park factor × wind, temperature, dome) kept from v1.

  * Clean train / inference split. Every feature is derived from information
    that was available *before* first pitch, so there is no post-game
    leakage.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from market_mechanics import remove_vig


LEAGUE_AVG_RUNS_PER_GAME = 4.5
LEAGUE_AVG_OPS = 0.710
LEAGUE_AVG_OBP = 0.320
LEAGUE_AVG_SLG = 0.390
LEAGUE_AVG_ERA = 4.20
LEAGUE_AVG_FIP = 4.20
LEAGUE_AVG_WHIP = 1.30
LEAGUE_AVG_K_PER_9 = 8.80
LEAGUE_AVG_BB_PER_9 = 3.30
LEAGUE_AVG_HR_PER_9 = 1.15
LEAGUE_AVG_TOTAL = 8.70


# Stabilization priors. These are the sample sizes at which observed current-
# season numbers start to dominate the prior-season / league-average fallback.
TEAM_WIN_PRIOR_GAMES = 45.0
PITCHER_IP_PRIOR = 45.0
LINEUP_GAMES_PRIOR = 40.0


NUMERIC_FEATURES_V2 = [
    # Market inputs (single most important block)
    "market_home_vigfree_prob",
    "market_home_ml",
    "market_away_ml",
    "market_home_line_move",
    "market_total_line",
    # Team strength (differentials, home minus away)
    "win_pct_adv",
    "pythag_win_pct_adv",
    "run_diff_per_game_adv",
    "runs_scored_per_game_adv",
    "runs_allowed_per_game_adv",
    # Form
    "form_7d_win_pct_adv",
    "form_14d_win_pct_adv",
    "form_30d_run_diff_adv",
    # Split records
    "home_home_win_pct",
    "away_away_win_pct",
    # Starters
    "starter_era_adv",
    "starter_fip_adv",
    "starter_whip_adv",
    "starter_k_per_9_adv",
    "starter_bb_per_9_adv",
    "starter_hr_per_9_adv",
    "starter_k_minus_bb_pct_adv",
    "starter_recent_era_adv",
    "starter_reliability_adv",
    # Lineup proxy (shrunk)
    "lineup_ops_adv",
    "lineup_obp_adv",
    "lineup_slg_adv",
    # Bullpen
    "bullpen_era_30d_adv",
    "bullpen_pitches_1d_adv",
    "bullpen_pitches_3d_adv",
    # Rest / travel
    "rest_days_adv",
    "travel_distance_adv",
    "travel_flag_adv",
    # Environment
    "park_factor_runs",
    "park_factor_deviation",
    "temperature_f",
    "wind_speed_mph",
    "wind_out_speed",
    "wind_in_speed",
    "is_dome",
    # Season phase
    "game_month",
    "is_early_season",
]


CATEGORICAL_FEATURES_V2 = [
    "home_starter_hand",
    "away_starter_hand",
    "wind_direction",
    "matchup_handedness",
]


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value in (None, "", "nan"):
            return default
        result = float(value)
        if np.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _shrink(current: float, prior: float, sample: float, prior_sample: float) -> float:
    if sample <= 0 and prior_sample <= 0:
        return prior
    weight = sample / (sample + prior_sample) if (sample + prior_sample) > 0 else 0.0
    weight = max(0.0, min(1.0, weight))
    return weight * current + (1.0 - weight) * prior


def _pythagorean(runs_scored: float, runs_allowed: float, exponent: float = 1.83) -> float:
    rs = max(0.0, runs_scored)
    ra = max(0.0, runs_allowed)
    if rs <= 0 and ra <= 0:
        return 0.5
    return (rs ** exponent) / ((rs ** exponent) + (ra ** exponent))


def _starter_rates(row: Mapping[str, object], side: str) -> dict[str, float]:
    """Derive K/9, BB/9, HR/9, K-BB% from cumulative starter stats.

    Historical rows carry `starter_ip`, `starter_starts`, and the basic rate
    stats. For deeper rates we fall back to league averages when the raw
    counts are missing.
    """
    ip = _as_float(row.get(f"{side}_starter_ip"))
    era = _as_float(row.get(f"{side}_starter_era"), LEAGUE_AVG_ERA)
    fip = _as_float(row.get(f"{side}_starter_fip"), LEAGUE_AVG_FIP)
    whip = _as_float(row.get(f"{side}_starter_whip"), LEAGUE_AVG_WHIP)

    k = _as_float(row.get(f"{side}_starter_strikeouts"))
    bb = _as_float(row.get(f"{side}_starter_walks"))
    hr = _as_float(row.get(f"{side}_starter_home_runs"))
    bf = _as_float(row.get(f"{side}_starter_batters_faced"))

    prior_ip = _as_float(row.get(f"{side}_prior_starter_ip"))
    prior_era = _as_float(row.get(f"{side}_prior_starter_era"), LEAGUE_AVG_ERA)
    prior_fip = _as_float(row.get(f"{side}_prior_starter_fip"), LEAGUE_AVG_FIP)

    era_shrunk = _shrink(era, prior_era, ip, PITCHER_IP_PRIOR)
    fip_shrunk = _shrink(fip, prior_fip, ip, PITCHER_IP_PRIOR)
    whip_shrunk = _shrink(whip, LEAGUE_AVG_WHIP, ip, PITCHER_IP_PRIOR)

    if ip > 0:
        k_per_9 = 9.0 * k / ip
        bb_per_9 = 9.0 * bb / ip
        hr_per_9 = 9.0 * hr / ip
    else:
        k_per_9 = LEAGUE_AVG_K_PER_9
        bb_per_9 = LEAGUE_AVG_BB_PER_9
        hr_per_9 = LEAGUE_AVG_HR_PER_9

    k_per_9 = _shrink(k_per_9, LEAGUE_AVG_K_PER_9, ip, PITCHER_IP_PRIOR)
    bb_per_9 = _shrink(bb_per_9, LEAGUE_AVG_BB_PER_9, ip, PITCHER_IP_PRIOR)
    hr_per_9 = _shrink(hr_per_9, LEAGUE_AVG_HR_PER_9, ip, PITCHER_IP_PRIOR)

    if bf > 0:
        k_minus_bb_pct = (k - bb) / bf
    else:
        # Approximate K-BB% from league-average rates when batters-faced isn't
        # surfaced.
        k_minus_bb_pct = (LEAGUE_AVG_K_PER_9 - LEAGUE_AVG_BB_PER_9) / 38.0

    recent_era = _as_float(
        row.get(f"{side}_starter_recent_era"),
        era_shrunk,
    )

    # Reliability = sample weight on current-season numbers.
    reliability = ip / (ip + PITCHER_IP_PRIOR) if (ip + PITCHER_IP_PRIOR) > 0 else 0.0

    return {
        "era": era_shrunk,
        "fip": fip_shrunk,
        "whip": whip_shrunk,
        "k_per_9": k_per_9,
        "bb_per_9": bb_per_9,
        "hr_per_9": hr_per_9,
        "k_minus_bb_pct": k_minus_bb_pct,
        "recent_era": recent_era,
        "reliability": reliability,
    }


def _lineup_profile(row: Mapping[str, object], side: str) -> dict[str, float]:
    ops = _as_float(row.get(f"{side}_lineup_ops_proxy"), LEAGUE_AVG_OPS)
    obp = _as_float(row.get(f"{side}_lineup_obp_proxy"), LEAGUE_AVG_OBP)
    slg = _as_float(row.get(f"{side}_lineup_slg_proxy"), LEAGUE_AVG_SLG)
    sample = _as_float(row.get(f"{side}_lineup_sample_games"))
    return {
        "ops": _shrink(ops, LEAGUE_AVG_OPS, sample, LINEUP_GAMES_PRIOR),
        "obp": _shrink(obp, LEAGUE_AVG_OBP, sample, LINEUP_GAMES_PRIOR),
        "slg": _shrink(slg, LEAGUE_AVG_SLG, sample, LINEUP_GAMES_PRIOR),
        "sample": sample,
    }


def _team_strength(row: Mapping[str, object], side: str) -> dict[str, float]:
    games = _as_float(row.get(f"{side}_games_played"))
    win_pct = _as_float(row.get(f"{side}_season_win_pct"), 0.5)
    prior = _as_float(row.get(f"{side}_prior_season_win_pct"), 0.5)
    win_pct_shrunk = _shrink(win_pct, prior, games, TEAM_WIN_PRIOR_GAMES)

    # Estimate runs scored / allowed per game. If we don't have cumulative run
    # totals we fall back to league-average.
    runs_scored = _as_float(row.get(f"{side}_runs_scored_season"))
    runs_allowed = _as_float(row.get(f"{side}_runs_allowed_season"))
    if games > 0 and (runs_scored > 0 or runs_allowed > 0):
        rs_pg = runs_scored / games
        ra_pg = runs_allowed / games
    else:
        rs_pg = LEAGUE_AVG_RUNS_PER_GAME
        ra_pg = LEAGUE_AVG_RUNS_PER_GAME

    pythag = _pythagorean(rs_pg, ra_pg)
    pythag_shrunk = _shrink(pythag, 0.5, games, TEAM_WIN_PRIOR_GAMES)

    return {
        "win_pct": win_pct_shrunk,
        "pythag": pythag_shrunk,
        "run_diff_pg": rs_pg - ra_pg,
        "rs_pg": rs_pg,
        "ra_pg": ra_pg,
    }


def _market_signals(row: Mapping[str, object]) -> dict[str, float]:
    home_ml = row.get("home_moneyline")
    away_ml = row.get("away_moneyline")
    home_open = row.get("home_open_moneyline")

    # remove_vig preserves argument order: pass home then away, first returned
    # value is the vig-free home probability.
    if home_ml is not None and away_ml is not None:
        try:
            home_prob, _ = remove_vig(int(home_ml), int(away_ml))
        except (TypeError, ValueError):
            home_prob = 0.5
    else:
        home_prob = 0.5
        home_ml = 0
        away_ml = 0

    # Line move: close minus open (+) means home got longer, market faded home.
    line_move = 0.0
    if home_ml is not None and home_open is not None:
        try:
            line_move = float(home_ml) - float(home_open)
        except (TypeError, ValueError):
            line_move = 0.0

    total_line = _as_float(row.get("market_total_line"), LEAGUE_AVG_TOTAL)

    return {
        "home_vigfree_prob": float(home_prob),
        "home_ml": _as_float(home_ml),
        "away_ml": _as_float(away_ml),
        "line_move": float(line_move),
        "total_line": float(total_line),
    }


def _matchup_handedness(home_hand: str, away_hand: str) -> str:
    h = (home_hand or "U").upper()[:1]
    a = (away_hand or "U").upper()[:1]
    return f"{h}v{a}"


def _wind_components(direction: str, speed: float) -> tuple[float, float]:
    direction = (direction or "unknown").lower()
    if direction == "out":
        return (float(speed), 0.0)
    if direction == "in":
        return (0.0, float(speed))
    return (0.0, 0.0)


def _game_month(row: Mapping[str, object]) -> int:
    game_date = row.get("game_date")
    if game_date is None:
        return 0
    try:
        if hasattr(game_date, "month"):
            return int(game_date.month)
        return int(pd.Timestamp(game_date).month)
    except Exception:
        return 0


def build_feature_row(raw: Mapping[str, object]) -> dict[str, float]:
    home_strength = _team_strength(raw, "home")
    away_strength = _team_strength(raw, "away")
    home_starter = _starter_rates(raw, "home")
    away_starter = _starter_rates(raw, "away")
    home_lineup = _lineup_profile(raw, "home")
    away_lineup = _lineup_profile(raw, "away")
    market = _market_signals(raw)

    wind_out, wind_in = _wind_components(
        str(raw.get("wind_direction") or "unknown"),
        _as_float(raw.get("wind_speed_mph")),
    )
    month = _game_month(raw)

    form_7 = _as_float(raw.get("home_form_7d_win_pct"), 0.5) - _as_float(
        raw.get("away_form_7d_win_pct"), 0.5
    )
    form_14 = _as_float(raw.get("home_form_14d_win_pct"), 0.5) - _as_float(
        raw.get("away_form_14d_win_pct"), 0.5
    )
    form_30_rd = _as_float(raw.get("home_form_30d_run_diff"), 0.0) - _as_float(
        raw.get("away_form_30d_run_diff"), 0.0
    )

    home_home = _as_float(raw.get("home_home_record_win_pct"), home_strength["win_pct"])
    away_away = _as_float(raw.get("away_away_record_win_pct"), away_strength["win_pct"])

    park_factor = _as_float(raw.get("park_factor_runs"), 100.0)

    return {
        # Market block
        "market_home_vigfree_prob": market["home_vigfree_prob"],
        "market_home_ml": market["home_ml"],
        "market_away_ml": market["away_ml"],
        "market_home_line_move": market["line_move"],
        "market_total_line": market["total_line"],
        # Team strength
        "win_pct_adv": home_strength["win_pct"] - away_strength["win_pct"],
        "pythag_win_pct_adv": home_strength["pythag"] - away_strength["pythag"],
        "run_diff_per_game_adv": home_strength["run_diff_pg"] - away_strength["run_diff_pg"],
        "runs_scored_per_game_adv": home_strength["rs_pg"] - away_strength["rs_pg"],
        "runs_allowed_per_game_adv": away_strength["ra_pg"] - home_strength["ra_pg"],
        # Form
        "form_7d_win_pct_adv": form_7,
        "form_14d_win_pct_adv": form_14,
        "form_30d_run_diff_adv": form_30_rd,
        # Splits
        "home_home_win_pct": home_home,
        "away_away_win_pct": away_away,
        # Starters (home minus away where higher-is-better-for-home)
        "starter_era_adv": away_starter["era"] - home_starter["era"],
        "starter_fip_adv": away_starter["fip"] - home_starter["fip"],
        "starter_whip_adv": away_starter["whip"] - home_starter["whip"],
        "starter_k_per_9_adv": home_starter["k_per_9"] - away_starter["k_per_9"],
        "starter_bb_per_9_adv": away_starter["bb_per_9"] - home_starter["bb_per_9"],
        "starter_hr_per_9_adv": away_starter["hr_per_9"] - home_starter["hr_per_9"],
        "starter_k_minus_bb_pct_adv": home_starter["k_minus_bb_pct"] - away_starter["k_minus_bb_pct"],
        "starter_recent_era_adv": away_starter["recent_era"] - home_starter["recent_era"],
        "starter_reliability_adv": home_starter["reliability"] - away_starter["reliability"],
        # Lineup (shrunk)
        "lineup_ops_adv": home_lineup["ops"] - away_lineup["ops"],
        "lineup_obp_adv": home_lineup["obp"] - away_lineup["obp"],
        "lineup_slg_adv": home_lineup["slg"] - away_lineup["slg"],
        # Bullpen
        "bullpen_era_30d_adv": _as_float(raw.get("away_bullpen_era_30d"), LEAGUE_AVG_ERA)
        - _as_float(raw.get("home_bullpen_era_30d"), LEAGUE_AVG_ERA),
        "bullpen_pitches_1d_adv": _as_float(raw.get("away_bullpen_pitches_1d"))
        - _as_float(raw.get("home_bullpen_pitches_1d")),
        "bullpen_pitches_3d_adv": _as_float(raw.get("away_bullpen_pitches_3d"))
        - _as_float(raw.get("home_bullpen_pitches_3d")),
        # Rest / travel
        "rest_days_adv": _as_float(raw.get("home_rest_days"))
        - _as_float(raw.get("away_rest_days")),
        "travel_distance_adv": _as_float(raw.get("away_travel_distance_miles"))
        - _as_float(raw.get("home_travel_distance_miles")),
        "travel_flag_adv": _as_float(raw.get("away_travel_flag"))
        - _as_float(raw.get("home_travel_flag")),
        # Environment
        "park_factor_runs": park_factor,
        "park_factor_deviation": park_factor - 100.0,
        "temperature_f": _as_float(raw.get("temperature_f"), 72.0),
        "wind_speed_mph": _as_float(raw.get("wind_speed_mph")),
        "wind_out_speed": wind_out,
        "wind_in_speed": wind_in,
        "is_dome": _as_float(raw.get("is_dome")),
        "game_month": float(month),
        "is_early_season": 1.0 if month in (3, 4) else 0.0,
        # Categoricals
        "home_starter_hand": str(raw.get("home_starter_hand") or "U"),
        "away_starter_hand": str(raw.get("away_starter_hand") or "U"),
        "wind_direction": str(raw.get("wind_direction") or "unknown"),
        "matchup_handedness": _matchup_handedness(
            str(raw.get("home_starter_hand") or "U"),
            str(raw.get("away_starter_hand") or "U"),
        ),
    }


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply v2 feature engineering to a training/inference dataframe.

    Passes through identifier columns (game_date, home_win, total_runs, etc.)
    so downstream training code can split and label easily.
    """
    if frame.empty:
        columns = NUMERIC_FEATURES_V2 + CATEGORICAL_FEATURES_V2
        return pd.DataFrame(columns=columns)

    features: list[dict[str, object]] = []
    passthrough_cols: Iterable[str] = [
        column
        for column in [
            "game_pk",
            "game_date",
            "season",
            "home_team",
            "away_team",
            "home_win",
            "total_runs",
            "home_score",
            "away_score",
            "home_moneyline",
            "away_moneyline",
            "market_total_line",
        ]
        if column in frame.columns
    ]

    for record in frame.to_dict("records"):
        row = build_feature_row(record)
        for key in passthrough_cols:
            row[key] = record.get(key)
        features.append(row)

    return pd.DataFrame(features)


def feature_columns_v2() -> list[str]:
    return NUMERIC_FEATURES_V2 + CATEGORICAL_FEATURES_V2


def select_training_rows_v2(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.dropna(subset=["home_win", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )


def select_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    columns = feature_columns_v2()
    for column in columns:
        if column not in frame.columns:
            frame[column] = np.nan
    return frame[columns].copy()
