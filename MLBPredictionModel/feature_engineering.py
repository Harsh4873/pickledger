from __future__ import annotations

from typing import Iterable

import pandas as pd

from probability_layers import heuristic_features, predict_total_runs


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


def add_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = add_heuristic_columns(frame)
    frame = frame.copy()

    frame["team_win_pct_adv"] = frame["home_season_win_pct"] - frame["away_season_win_pct"]
    frame["prior_team_win_pct_adv"] = frame["home_prior_season_win_pct"] - frame["away_prior_season_win_pct"]

    frame["form_7d_adv"] = frame["home_form_7d_win_pct"] - frame["away_form_7d_win_pct"]
    frame["form_14d_adv"] = frame["home_form_14d_win_pct"] - frame["away_form_14d_win_pct"]
    frame["form_30d_adv"] = frame["home_form_30d_win_pct"] - frame["away_form_30d_win_pct"]
    frame["form_7d_game_count_adv"] = frame["home_form_7d_games"] - frame["away_form_7d_games"]
    frame["form_14d_game_count_adv"] = frame["home_form_14d_games"] - frame["away_form_14d_games"]
    frame["form_30d_game_count_adv"] = frame["home_form_30d_games"] - frame["away_form_30d_games"]

    frame["bullpen_pitches_1d_adv"] = frame["away_bullpen_pitches_1d"] - frame["home_bullpen_pitches_1d"]
    frame["bullpen_pitches_3d_adv"] = frame["away_bullpen_pitches_3d"] - frame["home_bullpen_pitches_3d"]
    frame["bullpen_quality_adv"] = frame["away_bullpen_era_30d"] - frame["home_bullpen_era_30d"]
    frame["rest_days_adv"] = frame["home_rest_days"] - frame["away_rest_days"]
    frame["travel_distance_adv"] = frame["away_travel_distance_miles"] - frame["home_travel_distance_miles"]
    frame["travel_flag_adv"] = frame["away_travel_flag"] - frame["home_travel_flag"]

    frame["starter_era_adv"] = frame["away_starter_era"] - frame["home_starter_era"]
    frame["starter_fip_adv"] = frame["away_starter_fip"] - frame["home_starter_fip"]
    frame["starter_whip_adv"] = frame["away_starter_whip"] - frame["home_starter_whip"]
    frame["starter_ip_adv"] = frame["home_starter_ip"] - frame["away_starter_ip"]
    frame["starter_starts_adv"] = frame["home_starter_starts"] - frame["away_starter_starts"]

    frame["prior_starter_era_adv"] = frame["away_prior_starter_era"] - frame["home_prior_starter_era"]
    frame["prior_starter_fip_adv"] = frame["away_prior_starter_fip"] - frame["home_prior_starter_fip"]
    frame["prior_starter_ip_adv"] = frame["home_prior_starter_ip"] - frame["away_prior_starter_ip"]

    frame["lineup_ops_adv"] = frame["home_lineup_ops_proxy"] - frame["away_lineup_ops_proxy"]
    frame["lineup_obp_adv"] = frame["home_lineup_obp_proxy"] - frame["away_lineup_obp_proxy"]
    frame["lineup_slg_adv"] = frame["home_lineup_slg_proxy"] - frame["away_lineup_slg_proxy"]
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
