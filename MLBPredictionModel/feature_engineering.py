from __future__ import annotations

from typing import Iterable

import pandas as pd

from probability_layers import heuristic_features, predict_total_runs


# Early-season stabilization constants.
# Team win %: 45 games is roughly a quarter season, enough for current results
# to matter materially but not overwhelm prior-season strength in April/May.
TEAM_WIN_PCT_PRIOR_GAMES = 45.0

# Recent-form windows: these are intentionally conservative so a 3-2 week does
# not look like a true signal yet.
FORM_PRIOR_GAMES = {
    7: 6.0,
    14: 8.0,
    30: 12.0,
}

# Pitcher reliability:
# 45 IP or ~8 starts is the threshold where current-season run prevention starts
# getting most of the weight. Below that, blend aggressively toward prior season
# (or league-average fallback if no prior exists).
PITCHER_RELIABILITY_IP = 45.0
PITCHER_RELIABILITY_STARTS = 8.0

# Lineup proxy stabilization: early lineup OPS from tiny batter samples is noisy,
# so blend toward a league-average offense until the confirmed lineup has enough
# cumulative games behind it.
LINEUP_PRIOR_GAMES = 40.0

LEAGUE_AVG_ERA = 4.20
LEAGUE_AVG_FIP = 4.20
LEAGUE_AVG_WHIP = 1.30
LEAGUE_AVG_OPS = 0.710
LEAGUE_AVG_OBP = 0.320
LEAGUE_AVG_SLG = 0.390


CATEGORICAL_FEATURES = [
    "home_starter_hand",
    "away_starter_hand",
    "wind_direction",
]


NUMERIC_FEATURES = [
    "park_factor_runs",
    "temperature_f",
    "wind_speed_mph",
    "is_dome",
    "team_win_pct_adv",
    "prior_team_win_pct_adv",
    "form_7d_adv",
    "form_14d_adv",
    "form_30d_adv",
    "form_7d_game_count_adv",
    "form_14d_game_count_adv",
    "form_30d_game_count_adv",
    "bullpen_pitches_1d_adv",
    "bullpen_pitches_3d_adv",
    "bullpen_quality_adv",
    "rest_days_adv",
    "travel_distance_adv",
    "travel_flag_adv",
    "starter_era_adv",
    "starter_fip_adv",
    "starter_whip_adv",
    "starter_ip_adv",
    "starter_starts_adv",
    "starter_reliability_adv",
    "home_starter_reliability",
    "away_starter_reliability",
    "prior_starter_era_adv",
    "prior_starter_fip_adv",
    "prior_starter_ip_adv",
    "lineup_ops_adv",
    "lineup_obp_adv",
    "lineup_slg_adv",
    "lineup_sample_adv",
    "heuristic_layer1_home_prob",
    "heuristic_layer2_adj",
    "heuristic_layer3_adj",
    "heuristic_raw_home_prob",
    "heuristic_home_win_prob",
    "heuristic_total_runs",
]


def add_heuristic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    drop_cols = [
        "heuristic_layer1_home_prob",
        "heuristic_layer2_adj",
        "heuristic_layer3_adj",
        "heuristic_raw_home_prob",
        "heuristic_home_win_prob",
        "heuristic_total_runs",
    ]
    existing = [column for column in drop_cols if column in frame.columns]
    if existing:
        frame = frame.drop(columns=existing)
    heuristics = frame.apply(lambda row: heuristic_features(row.to_dict()), axis=1, result_type="expand")
    frame = pd.concat([frame, heuristics], axis=1)
    frame["heuristic_total_runs"] = frame.apply(lambda row: predict_total_runs(row.to_dict()), axis=1)
    return frame


def _blend(current: pd.Series, baseline: pd.Series | float, weight: pd.Series) -> pd.Series:
    return weight * current + (1.0 - weight) * baseline


def _reliability_weight(sample: pd.Series, prior_scale: float) -> pd.Series:
    return (sample / (sample + prior_scale)).clip(lower=0.0, upper=1.0)


def apply_sample_size_shrinkage(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    for side in ("home", "away"):
        games_played = frame[f"{side}_games_played"].fillna(0.0)
        team_weight = _reliability_weight(games_played, TEAM_WIN_PCT_PRIOR_GAMES)
        frame[f"{side}_team_win_pct_shrunk"] = _blend(
            frame[f"{side}_season_win_pct"].fillna(0.5),
            frame[f"{side}_prior_season_win_pct"].fillna(0.5),
            team_weight,
        )

        for window, prior_games in FORM_PRIOR_GAMES.items():
            sample = frame[f"{side}_form_{window}d_games"].fillna(0.0)
            form_weight = _reliability_weight(sample, prior_games)
            frame[f"{side}_form_{window}d_win_pct_shrunk"] = _blend(
                frame[f"{side}_form_{window}d_win_pct"].fillna(0.5),
                frame[f"{side}_team_win_pct_shrunk"].fillna(0.5),
                form_weight,
            )

        ip = frame[f"{side}_starter_ip"].fillna(0.0)
        starts = frame[f"{side}_starter_starts"].fillna(0.0)
        ip_weight = _reliability_weight(ip, PITCHER_RELIABILITY_IP)
        starts_weight = _reliability_weight(starts, PITCHER_RELIABILITY_STARTS)
        reliability = ((ip_weight + starts_weight) / 2.0).clip(lower=0.0, upper=1.0)
        frame[f"{side}_starter_reliability"] = reliability

        prior_era = frame[f"{side}_prior_starter_era"].fillna(LEAGUE_AVG_ERA)
        prior_fip = frame[f"{side}_prior_starter_fip"].fillna(LEAGUE_AVG_FIP)
        prior_whip = LEAGUE_AVG_WHIP
        frame[f"{side}_starter_era_shrunk"] = _blend(
            frame[f"{side}_starter_era"].fillna(LEAGUE_AVG_ERA),
            prior_era,
            reliability,
        )
        frame[f"{side}_starter_fip_shrunk"] = _blend(
            frame[f"{side}_starter_fip"].fillna(LEAGUE_AVG_FIP),
            prior_fip,
            reliability,
        )
        frame[f"{side}_starter_whip_shrunk"] = _blend(
            frame[f"{side}_starter_whip"].fillna(LEAGUE_AVG_WHIP),
            prior_whip,
            reliability,
        )

        lineup_sample = frame[f"{side}_lineup_sample_games"].fillna(0.0)
        lineup_weight = _reliability_weight(lineup_sample, LINEUP_PRIOR_GAMES)
        frame[f"{side}_lineup_ops_proxy_shrunk"] = _blend(
            frame[f"{side}_lineup_ops_proxy"].fillna(LEAGUE_AVG_OPS),
            LEAGUE_AVG_OPS,
            lineup_weight,
        )
        frame[f"{side}_lineup_obp_proxy_shrunk"] = _blend(
            frame[f"{side}_lineup_obp_proxy"].fillna(LEAGUE_AVG_OBP),
            LEAGUE_AVG_OBP,
            lineup_weight,
        )
        frame[f"{side}_lineup_slg_proxy_shrunk"] = _blend(
            frame[f"{side}_lineup_slg_proxy"].fillna(LEAGUE_AVG_SLG),
            LEAGUE_AVG_SLG,
            lineup_weight,
        )

    return frame


def add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = apply_sample_size_shrinkage(frame)
    frame = add_heuristic_columns(frame)
    frame = frame.copy()

    frame["team_win_pct_adv"] = frame["home_team_win_pct_shrunk"] - frame["away_team_win_pct_shrunk"]
    frame["prior_team_win_pct_adv"] = frame["home_prior_season_win_pct"] - frame["away_prior_season_win_pct"]

    frame["form_7d_adv"] = frame["home_form_7d_win_pct_shrunk"] - frame["away_form_7d_win_pct_shrunk"]
    frame["form_14d_adv"] = frame["home_form_14d_win_pct_shrunk"] - frame["away_form_14d_win_pct_shrunk"]
    frame["form_30d_adv"] = frame["home_form_30d_win_pct_shrunk"] - frame["away_form_30d_win_pct_shrunk"]
    frame["form_7d_game_count_adv"] = frame["home_form_7d_games"] - frame["away_form_7d_games"]
    frame["form_14d_game_count_adv"] = frame["home_form_14d_games"] - frame["away_form_14d_games"]
    frame["form_30d_game_count_adv"] = frame["home_form_30d_games"] - frame["away_form_30d_games"]

    frame["bullpen_pitches_1d_adv"] = frame["away_bullpen_pitches_1d"] - frame["home_bullpen_pitches_1d"]
    frame["bullpen_pitches_3d_adv"] = frame["away_bullpen_pitches_3d"] - frame["home_bullpen_pitches_3d"]
    frame["bullpen_quality_adv"] = frame["away_bullpen_era_30d"] - frame["home_bullpen_era_30d"]
    frame["rest_days_adv"] = frame["home_rest_days"] - frame["away_rest_days"]
    frame["travel_distance_adv"] = frame["away_travel_distance_miles"] - frame["home_travel_distance_miles"]
    frame["travel_flag_adv"] = frame["away_travel_flag"] - frame["home_travel_flag"]

    frame["starter_era_adv"] = frame["away_starter_era_shrunk"] - frame["home_starter_era_shrunk"]
    frame["starter_fip_adv"] = frame["away_starter_fip_shrunk"] - frame["home_starter_fip_shrunk"]
    frame["starter_whip_adv"] = frame["away_starter_whip_shrunk"] - frame["home_starter_whip_shrunk"]
    frame["starter_ip_adv"] = frame["home_starter_ip"] - frame["away_starter_ip"]
    frame["starter_starts_adv"] = frame["home_starter_starts"] - frame["away_starter_starts"]
    frame["starter_reliability_adv"] = frame["home_starter_reliability"] - frame["away_starter_reliability"]

    frame["prior_starter_era_adv"] = frame["away_prior_starter_era"] - frame["home_prior_starter_era"]
    frame["prior_starter_fip_adv"] = frame["away_prior_starter_fip"] - frame["home_prior_starter_fip"]
    frame["prior_starter_ip_adv"] = frame["home_prior_starter_ip"] - frame["away_prior_starter_ip"]

    frame["lineup_ops_adv"] = frame["home_lineup_ops_proxy_shrunk"] - frame["away_lineup_ops_proxy_shrunk"]
    frame["lineup_obp_adv"] = frame["home_lineup_obp_proxy_shrunk"] - frame["away_lineup_obp_proxy_shrunk"]
    frame["lineup_slg_adv"] = frame["home_lineup_slg_proxy_shrunk"] - frame["away_lineup_slg_proxy_shrunk"]
    frame["lineup_sample_adv"] = frame["home_lineup_sample_games"] - frame["away_lineup_sample_games"]
    return frame


def feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def ensure_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_model_features(frame)
    for column in feature_columns():
        if column not in frame.columns:
            frame[column] = 0.0
    return frame


def select_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return ensure_feature_frame(frame)[feature_columns()].copy()


def select_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_feature_frame(frame)
    required = ["home_win", "game_date"]
    return frame.dropna(subset=required).sort_values("game_date").reset_index(drop=True)


def summarize_feature_set() -> list[str]:
    return feature_columns()
