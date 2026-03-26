from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibration import apply_moneyline_calibration
from feature_engineering import select_training_rows
from moneyline_model import chronological_split, evaluate_probabilities, predict_home_win_probability
from totals_model import predict_totals


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "mlb_historical_dataset_2023_2024.csv"
REPORT_PATH = BASE_DIR / "mlb_backtest_report.json"
SAMPLE_PATH = BASE_DIR / "mlb_backtest_sample.csv"


def american_profit(odds: float) -> float:
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    return 0.0


def side_accuracy(frame: pd.DataFrame, probability_column: str) -> float:
    predicted_home = frame[probability_column] >= 0.5
    correct = np.where(predicted_home, frame["home_win"] == 1, frame["home_win"] == 0)
    return float(np.mean(correct))


def side_roi(frame: pd.DataFrame, probability_column: str) -> dict[str, float]:
    predicted_home = frame[probability_column] >= 0.5
    odds = np.where(predicted_home, frame["home_moneyline"], frame["away_moneyline"])
    valid = ~pd.isna(odds)
    if not np.any(valid):
        return {"roi": 0.0, "bets": 0, "profit_units": 0.0}

    outcomes = np.where(predicted_home, frame["home_win"] == 1, frame["home_win"] == 0)
    profit = 0.0
    bets = 0
    for won, line, include in zip(outcomes, odds, valid):
        if not include:
            continue
        bets += 1
        line = float(line)
        profit += american_profit(line) if won else -1.0

    return {
        "roi": float(profit / bets) if bets else 0.0,
        "bets": int(bets),
        "profit_units": float(profit),
    }


def calibration_buckets(frame: pd.DataFrame, probability_column: str) -> list[dict[str, Any]]:
    predicted_home = frame[probability_column] >= 0.5
    confidence = np.where(predicted_home, frame[probability_column], 1.0 - frame[probability_column])
    correct = np.where(predicted_home, frame["home_win"] == 1, frame["home_win"] == 0)
    bucket_edges = np.arange(0.50, 1.01, 0.05)

    out = []
    for start, end in zip(bucket_edges[:-1], bucket_edges[1:]):
        mask = (confidence >= start) & (confidence < end if end < 1.0 else confidence <= end)
        if not np.any(mask):
            continue
        out.append(
            {
                "bucket": f"{int(start * 100)}-{int(end * 100)}",
                "count": int(np.sum(mask)),
                "avg_confidence": float(np.mean(confidence[mask])),
                "win_rate": float(np.mean(correct[mask])),
            }
        )
    return out


def totals_metrics(frame: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    residuals = frame[prediction_column] - frame["total_runs"]
    return {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def build_backtest(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    frame = pd.read_csv(dataset_path, parse_dates=["game_date"])
    frame = select_training_rows(frame)
    _, validation_frame = chronological_split(frame)

    model_frame = predict_home_win_probability(validation_frame)
    model_frame = apply_moneyline_calibration(model_frame)
    model_frame = predict_totals(model_frame)
    model_frame["heuristic_predicted_winner"] = np.where(
        model_frame["heuristic_home_win_probability"] >= 0.5,
        model_frame["home_team"],
        model_frame["away_team"],
    )
    model_frame["hybrid_predicted_winner"] = np.where(
        model_frame["calibrated_home_win_probability"] >= 0.5,
        model_frame["home_team"],
        model_frame["away_team"],
    )
    model_frame["heuristic_correct"] = np.where(
        model_frame["heuristic_home_win_probability"] >= 0.5,
        model_frame["home_win"] == 1,
        model_frame["home_win"] == 0,
    )
    model_frame["hybrid_correct"] = np.where(
        model_frame["calibrated_home_win_probability"] >= 0.5,
        model_frame["home_win"] == 1,
        model_frame["home_win"] == 0,
    )
    model_frame["side_probability_delta"] = (
        model_frame["calibrated_home_win_probability"] - model_frame["heuristic_home_win_probability"]
    ).abs()

    heuristic_side = {
        "accuracy": side_accuracy(model_frame, "heuristic_home_win_probability"),
        **evaluate_probabilities(model_frame["home_win"], model_frame["heuristic_home_win_probability"].to_numpy()),
        **side_roi(model_frame, "heuristic_home_win_probability"),
        "calibration_by_bucket": calibration_buckets(model_frame, "heuristic_home_win_probability"),
    }
    hybrid_side = {
        "accuracy": side_accuracy(model_frame, "calibrated_home_win_probability"),
        **evaluate_probabilities(model_frame["home_win"], model_frame["calibrated_home_win_probability"].to_numpy()),
        **side_roi(model_frame, "calibrated_home_win_probability"),
        "calibration_by_bucket": calibration_buckets(model_frame, "calibrated_home_win_probability"),
    }
    heuristic_totals = totals_metrics(model_frame, "heuristic_total_runs")
    hybrid_totals = totals_metrics(model_frame, "predicted_total_runs")

    report = {
        "dataset": str(dataset_path.name),
        "seasons_used": sorted(int(season) for season in frame["season"].dropna().unique()),
        "validation_rows": int(len(model_frame)),
        "date_range": {
            "start": str(model_frame["game_date"].min().date()),
            "end": str(model_frame["game_date"].max().date()),
        },
        "moneyline_roi_strategy": "1 unit on the predicted side for games with matched market odds",
        "calibration_mode": str(model_frame["calibration_mode"].iloc[0]),
        "heuristic": heuristic_side,
        "hybrid_model": {
            **hybrid_side,
            "delta_vs_heuristic": {
                "accuracy_points": float(hybrid_side["accuracy"] - heuristic_side["accuracy"]),
                "log_loss_improvement": float(heuristic_side["log_loss"] - hybrid_side["log_loss"]),
                "brier_improvement": float(heuristic_side["brier_score"] - hybrid_side["brier_score"]),
                "roi_delta": float(hybrid_side["roi"] - heuristic_side["roi"]),
            },
        },
        "totals_model": {
            "heuristic_baseline": heuristic_totals,
            "hybrid_model": hybrid_totals,
            "delta_vs_heuristic": {
                "rmse_improvement": float(heuristic_totals["rmse"] - hybrid_totals["rmse"]),
                "mae_improvement": float(heuristic_totals["mae"] - hybrid_totals["mae"]),
            },
        },
    }

    sample = model_frame.sort_values(["side_probability_delta", "game_date"], ascending=[False, True])[
        [
            "game_date",
            "away_team",
            "home_team",
            "home_win",
            "home_moneyline",
            "away_moneyline",
            "heuristic_home_win_probability",
            "heuristic_predicted_winner",
            "heuristic_correct",
            "raw_model_home_win_probability",
            "raw_home_win_probability",
            "calibrated_home_win_probability",
            "hybrid_predicted_winner",
            "hybrid_correct",
            "side_probability_delta",
            "heuristic_total_runs",
            "raw_model_total_runs",
            "predicted_total_runs",
            "total_runs",
        ]
    ].head(50)

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sample.to_csv(SAMPLE_PATH, index=False)
    return report


if __name__ == "__main__":
    report = build_backtest()
    print(json.dumps(report, indent=2))
