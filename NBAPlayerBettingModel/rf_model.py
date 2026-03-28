from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from data_models import OpponentDefenseStats, PlayerSeasonStats, PropPrediction
from market_mechanics import get_recommended_stake

PROP_LABELS = {
    "pts": "Points",
    "reb": "Rebounds",
    "ast": "Assists",
}

TARGET_COLUMNS = {
    "pts": "points_per_game",
    "reb": "rebounds_per_game",
    "ast": "assists_per_game",
}

PROP_CONFIDENCE_PENALTY = {"pts": 0.0, "reb": 0.0, "ast": -5.0}

BASE_FEATURE_COLUMNS = [
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
]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _line_round(value: float) -> float:
    return round(value * 2.0) / 2.0


def estimate_prop_line(player: PlayerSeasonStats, prop_key: str) -> float:
    season_avg = player.season_average_for(prop_key)
    last10_avg = player.last10_average_for(prop_key)
    delta = 0.0
    if season_avg > 0:
        diff_ratio = abs(last10_avg - season_avg) / season_avg
        if diff_ratio > 0.10:
            delta = 0.5 if last10_avg > season_avg else -0.5
    return _line_round(max(0.5, season_avg + delta))


def _player_feature_row(player: PlayerSeasonStats) -> dict[str, float]:
    return {
        "mp_per_game": player.mp_per_game,
        "fg_per_game": player.fg_per_game,
        "fga_per_game": player.fga_per_game,
        "fg_percent": player.fg_percent,
        "x3p_per_game": player.x3p_per_game,
        "x3pa_per_game": player.x3pa_per_game,
        "x3p_percent": player.x3p_percent,
        "x2p_per_game": player.x2p_per_game,
        "x2pa_per_game": player.x2pa_per_game,
        "x2p_percent": player.x2p_percent,
        "e_fg_percent": player.e_fg_percent,
        "ft_per_game": player.ft_per_game,
        "fta_per_game": player.fta_per_game,
        "ft_percent": player.ft_percent,
        "orb_per_game": player.orb_per_game,
        "drb_per_game": player.drb_per_game,
        "trb_per_game": player.trb_per_game,
        "ast_per_game": player.ast_per_game,
        "stl_per_game": player.stl_per_game,
        "blk_per_game": player.blk_per_game,
        "tov_per_game": player.tov_per_game,
        "usage_rate": player.usage_rate,
    }


def _train_models(player_df: pd.DataFrame) -> tuple[dict[str, RandomForestRegressor], dict[str, float]]:
    train_df = player_df.copy()
    available_features = [col for col in BASE_FEATURE_COLUMNS if col in train_df.columns]
    fill_values = train_df[available_features].median(numeric_only=True).to_dict()
    x_train = train_df[available_features].fillna(fill_values).astype(float)

    models: dict[str, RandomForestRegressor] = {}
    for prop_key, target_column in TARGET_COLUMNS.items():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        y_train = train_df[target_column].astype(float)
        model.fit(x_train.values, y_train.values)
        models[prop_key] = model

    return models, {key: float(value) for key, value in fill_values.items()}


def _prediction_std(model: RandomForestRegressor, features: pd.DataFrame) -> float:
    tree_predictions = np.array([tree.predict(features)[0] for tree in model.estimators_], dtype=float)
    if tree_predictions.size == 0:
        return 0.0
    return float(tree_predictions.std())


def _prediction_iqr(model: RandomForestRegressor, features: pd.DataFrame) -> float:
    """Inter-quartile range of tree predictions - more robust than std for confidence."""
    tree_preds = np.array([tree.predict(features)[0] for tree in model.estimators_], dtype=float)
    if tree_preds.size == 0:
        return 0.0
    return float(np.percentile(tree_preds, 75) - np.percentile(tree_preds, 25))


def _matchup_multiplier(
    prop_key: str,
    player: PlayerSeasonStats,
    opponent: OpponentDefenseStats | None,
    position_baselines: dict[str, dict[str, float]],
    league_meta: dict[str, float],
) -> float:
    if opponent is None:
        return 1.0

    position_bucket = player.position_bucket
    baseline = position_baselines.get(position_bucket) or position_baselines.get("F") or {"pts": 20.0, "reb": 7.0, "ast": 4.0}
    allowance = opponent.prop_allowance(prop_key, position_bucket)
    base_allowance = max(0.5, float(baseline[prop_key]))
    positional_factor = allowance / base_allowance

    def_factor = opponent.def_rating / max(league_meta.get("league_def_rating", 113.0), 1e-6)
    pace_factor = opponent.pace / max(league_meta.get("league_pace", 99.0), 1e-6)

    if prop_key == "reb":
        multiplier = (0.45 * positional_factor) + (0.35 * pace_factor) + (0.20 * def_factor)
        return _clip(multiplier, 0.90, 1.12)
    if prop_key == "ast":
        multiplier = (0.50 * positional_factor) + (0.25 * pace_factor) + (0.25 * def_factor)
        return _clip(multiplier, 0.90, 1.12)

    multiplier = (0.55 * positional_factor) + (0.25 * def_factor) + (0.20 * pace_factor)
    return _clip(multiplier, 0.90, 1.12)


def _context_adjustment(player: PlayerSeasonStats, prop_key: str) -> float:
    season_avg = player.season_average_for(prop_key)
    last10_avg = player.last10_average_for(prop_key)
    split_avg = player.split_average_for(prop_key)

    trend_adjustment = (last10_avg - season_avg) * 0.15
    split_adjustment = 0.0
    if split_avg is not None:
        split_adjustment = (split_avg - season_avg) * 0.10
    return trend_adjustment + split_adjustment


def _build_reason(
    player: PlayerSeasonStats,
    prop_key: str,
    direction: str,
    matchup_multiplier: float,
    line: float,
    predicted_value: float,
) -> str:
    season_avg = player.season_average_for(prop_key)
    last10_avg = player.last10_average_for(prop_key)
    split_avg = player.split_average_for(prop_key)

    notes: list[str] = [f"usage sits at {player.usage_rate:.1f}%"]
    if last10_avg > season_avg * 1.05:
        notes.append("last-10 form is running above season baseline")
    elif last10_avg < season_avg * 0.95:
        notes.append("last-10 form has cooled off from the season baseline")

    if split_avg is not None:
        if split_avg > season_avg * 1.05:
            notes.append("the venue split is stronger than the season average")
        elif split_avg < season_avg * 0.95:
            notes.append("the venue split is softer than the season average")

    if matchup_multiplier > 1.03:
        notes.append(f"{player.opponent_team_abbreviation} profile as a favorable matchup")
    elif matchup_multiplier < 0.97:
        notes.append(f"{player.opponent_team_abbreviation} profile as a tougher matchup")

    action = "above" if direction == "OVER" else "below"
    joined = ", ".join(notes[:3])
    return f"RF projects {predicted_value:.1f}, which lands {action} the {line:.1f} line because {joined}."


def build_prop_predictions(
    players: list[PlayerSeasonStats],
    opponent_lookup: dict[int, OpponentDefenseStats],
    player_df: pd.DataFrame,
    position_baselines: dict[str, dict[str, float]],
    league_meta: dict[str, float],
) -> list[PropPrediction]:
    if not players or player_df.empty:
        return []

    models, fill_values = _train_models(player_df)
    predictions: list[PropPrediction] = []

    for player in players:
        features = pd.DataFrame([_player_feature_row(player)], columns=BASE_FEATURE_COLUMNS).fillna(fill_values)
        opponent = opponent_lookup.get(player.opponent_team_id)

        for prop_key, model in models.items():
            line = estimate_prop_line(player, prop_key)
            base_prediction = float(model.predict(features)[0])
            # Season-level training rows do not map cleanly to one opponent, so matchup
            # context is applied after the RF base prediction using tonight's defense.
            matchup_multiplier = _matchup_multiplier(prop_key, player, opponent, position_baselines, league_meta)
            adjusted_prediction = base_prediction * matchup_multiplier
            adjusted_prediction += _context_adjustment(player, prop_key)
            adjusted_prediction = max(0.0, adjusted_prediction)

            direction = "OVER" if adjusted_prediction >= line else "UNDER"
            line_denominator = max(line, 0.5)
            edge_pct = abs(adjusted_prediction - line) / line_denominator * 100.0

            tree_iqr = _prediction_iqr(model, features)
            sigma = max(tree_iqr / 1.35 + 1.5, 1.0)
            if direction == "OVER":
                true_prob = 1 - norm.cdf(line, loc=adjusted_prediction, scale=sigma)
            else:
                true_prob = norm.cdf(line, loc=adjusted_prediction, scale=sigma)
            true_prob = float(np.clip(true_prob, 0.51, 0.74))
            iqr_confidence = 50.0 + 50.0 * math.exp(-tree_iqr / 3.0)
            edge_boost = _clip(edge_pct * 0.5, 0.0, 8.0)
            raw_confidence = _clip(
                iqr_confidence + edge_boost + PROP_CONFIDENCE_PENALTY[prop_key],
                0.0,
                100.0,
            )
            # Cap confidence until calibration data is sufficient.
            # PropPrediction.decision uses confidence >= 65.0 as a gate - the cap must sit above that
            # threshold so BET picks can still fire, but below the overconfident 78%+ range.
            _CALIBRATED_SAMPLE_THRESHOLD = 50
            _props_log_path = os.path.join(os.path.dirname(__file__), "logs", "props_predictions.csv")
            try:
                import pandas as _pd

                _log_size = len(_pd.read_csv(_props_log_path)) if os.path.exists(_props_log_path) else 0
            except Exception:
                _log_size = 0

            is_calibrated = _log_size >= _CALIBRATED_SAMPLE_THRESHOLD
            if not is_calibrated:
                confidence = _clip(raw_confidence, 0.0, 75.0)
                calibration_flag = "[UNCALIBRATED]"
            else:
                confidence = raw_confidence
                calibration_flag = ""
            full_kelly, quarter_kelly = get_recommended_stake(odds=-110, model_prob=true_prob)

            decision = "BET" if edge_pct >= 8.0 and quarter_kelly > 0 and confidence >= 65.0 else "PASS"
            reason = _build_reason(player, prop_key, direction, matchup_multiplier, line, adjusted_prediction)
            if calibration_flag:
                reason = f"{calibration_flag} {reason}"

            predictions.append(
                PropPrediction(
                    player_id=player.player_id,
                    player_name=player.player_name,
                    team_abbreviation=player.team_abbreviation,
                    opponent_team_abbreviation=player.opponent_team_abbreviation,
                    opponent_team_name=player.opponent_team_name,
                    position=player.position,
                    game_id=player.game_id,
                    away_team_name=player.away_team_name,
                    home_team_name=player.home_team_name,
                    prop_key=prop_key,
                    prop_label=PROP_LABELS[prop_key],
                    line=line,
                    predicted_value=adjusted_prediction,
                    direction=direction,
                    edge_pct=edge_pct,
                    true_prob=true_prob,
                    confidence=confidence,
                    full_kelly=full_kelly,
                    quarter_kelly=quarter_kelly,
                    decision=decision,
                    reason=reason,
                )
            )

    return predictions
