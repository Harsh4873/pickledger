"""Train and certify the market-free CFB originator with season walk-forward tests."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from cfb_core import FEATURE_NAMES, build_dataset, load_training_rows, matrix
    from cfb_model import _probabilities
except ImportError:
    from .cfb_core import FEATURE_NAMES, build_dataset, load_training_rows, matrix
    from .cfb_model import _probabilities

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_VERSION = "cfb_v1_market_free_bivariate"
FIRST_SEASON = 2017
LAST_SEASON = 2025
WALK_FORWARD_SEASONS = range(2021, 2026)


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=18.0))


def _hist() -> Any:
    return HistGradientBoostingRegressor(
        max_depth=4,
        max_iter=260,
        learning_rate=0.045,
        l2_regularization=5.0,
        min_samples_leaf=30,
        random_state=17,
    )


FAMILIES: dict[str, Callable[[], Any]] = {"ridge": _ridge, "hist_gradient_boosting": _hist}


def _brier(truth: list[int], probabilities: list[float]) -> float:
    return float(np.mean([(probability - outcome) ** 2 for probability, outcome in zip(probabilities, truth)]))


def _push_possible(line: float) -> bool:
    return abs(line - round(line)) < 1e-9


def _fit_calibrator(probabilities: list[float], truth: list[int]) -> IsotonicRegression:
    if len(set(truth)) < 2:
        raise RuntimeError("calibrator truth lacks both outcome classes")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    calibrator.fit(probabilities, truth)
    return calibrator


def train(first_season: int = FIRST_SEASON, last_season: int = LAST_SEASON) -> dict[str, Any]:
    records = build_dataset(load_training_rows(first_season, last_season))
    if len(records) < 4500:
        raise SystemExit(f"CFB dataset too small ({len(records)} priced FBS games); refusing to train")

    family_reports: dict[str, list[dict[str, Any]]] = {}
    for family_name, factory in FAMILIES.items():
        report: list[dict[str, Any]] = []
        for season in WALK_FORWARD_SEASONS:
            if season > last_season:
                continue
            train_rows = [row for row in records if row["season"] < season]
            test_rows = [row for row in records if row["season"] == season]
            if len(train_rows) < 1000 or not test_rows:
                continue
            margin_model = factory().fit(matrix(train_rows), [row["home_margin"] for row in train_rows])
            total_model = factory().fit(matrix(train_rows), [row["game_total"] for row in train_rows])
            margin_predictions = margin_model.predict(matrix(test_rows))
            total_predictions = total_model.predict(matrix(test_rows))
            report.append(
                {
                    "season": season,
                    "games": len(test_rows),
                    "margin_mae": round(mean_absolute_error([row["home_margin"] for row in test_rows], margin_predictions), 5),
                    "total_mae": round(mean_absolute_error([row["game_total"] for row in test_rows], total_predictions), 5),
                }
            )
        family_reports[family_name] = report

    def family_score(name: str) -> float:
        rows = family_reports[name]
        return float(np.mean([row["margin_mae"] + row["total_mae"] for row in rows])) if rows else math.inf

    selected_family = min(FAMILIES, key=family_score)
    selected_factory = FAMILIES[selected_family]

    residual_margin: list[float] = []
    residual_total: list[float] = []
    calibration_input: dict[str, list[float]] = {"moneyline": [], "spread": [], "total": []}
    calibration_truth: dict[str, list[int]] = {"moneyline": [], "spread": [], "total": []}
    selected_walk_forward: list[dict[str, Any]] = []
    for season in WALK_FORWARD_SEASONS:
        if season > last_season:
            continue
        train_rows = [row for row in records if row["season"] < season]
        test_rows = [row for row in records if row["season"] == season]
        if len(train_rows) < 1000 or not test_rows:
            continue
        margin_model = selected_factory().fit(matrix(train_rows), [row["home_margin"] for row in train_rows])
        total_model = selected_factory().fit(matrix(train_rows), [row["game_total"] for row in train_rows])
        train_margin_predictions = margin_model.predict(matrix(train_rows))
        train_total_predictions = total_model.predict(matrix(train_rows))
        prior_margin = residual_margin or [
            row["home_margin"] - float(prediction)
            for row, prediction in zip(train_rows, train_margin_predictions)
        ]
        prior_total = residual_total or [
            row["game_total"] - float(prediction)
            for row, prediction in zip(train_rows, train_total_predictions)
        ]
        sigma_margin = max(1.0, float(np.std(prior_margin, ddof=1)))
        sigma_total = max(1.0, float(np.std(prior_total, ddof=1)))
        test_margin_predictions = margin_model.predict(matrix(test_rows))
        test_total_predictions = total_model.predict(matrix(test_rows))

        spread_hits = spread_graded = total_hits = total_graded = 0
        for row, margin_prediction, total_prediction in zip(test_rows, test_margin_predictions, test_total_predictions):
            margin_prediction = float(margin_prediction)
            total_prediction = float(total_prediction)
            home_margin = float(row["home_margin"])
            game_total = float(row["game_total"])
            home_line = float(row["home_line"])
            total_line = float(row["total_line"])

            ml_home, _, ml_away = _probabilities(margin_prediction, 0.0, sigma_margin, push_possible=False)
            home_win = 1 if home_margin > 0 else 0
            calibration_input["moneyline"].extend([ml_home, ml_away])
            calibration_truth["moneyline"].extend([home_win, 1 - home_win])

            spread_home, spread_push, spread_away = _probabilities(
                margin_prediction, -home_line, sigma_margin, push_possible=_push_possible(home_line)
            )
            actual_spread = home_margin + home_line
            if abs(actual_spread) > 1e-9:
                non_push = max(1e-9, 1.0 - spread_push)
                calibration_input["spread"].extend([spread_home / non_push, spread_away / non_push])
                home_cover = 1 if actual_spread > 0 else 0
                calibration_truth["spread"].extend([home_cover, 1 - home_cover])
                spread_hits += int((spread_home > spread_away) == bool(home_cover))
                spread_graded += 1

            total_over, total_push, total_under = _probabilities(
                total_prediction, total_line, sigma_total, push_possible=_push_possible(total_line)
            )
            actual_total = game_total - total_line
            if abs(actual_total) > 1e-9:
                non_push = max(1e-9, 1.0 - total_push)
                calibration_input["total"].extend([total_over / non_push, total_under / non_push])
                over = 1 if actual_total > 0 else 0
                calibration_truth["total"].extend([over, 1 - over])
                total_hits += int((total_over > total_under) == bool(over))
                total_graded += 1

            residual_margin.append(home_margin - margin_prediction)
            residual_total.append(game_total - total_prediction)

        selected_walk_forward.append(
            {
                "season": season,
                "games": len(test_rows),
                "margin_mae": round(mean_absolute_error([row["home_margin"] for row in test_rows], test_margin_predictions), 5),
                "total_mae": round(mean_absolute_error([row["game_total"] for row in test_rows], test_total_predictions), 5),
                "spread_direction_rate": round(spread_hits / spread_graded, 5) if spread_graded else None,
                "total_direction_rate": round(total_hits / total_graded, 5) if total_graded else None,
            }
        )

    calibrators = {
        market: _fit_calibrator(calibration_input[market], calibration_truth[market])
        for market in ("moneyline", "spread", "total")
    }
    final_margin = selected_factory().fit(matrix(records), [row["home_margin"] for row in records])
    final_total = selected_factory().fit(matrix(records), [row["game_total"] for row in records])
    residual_array = np.array([residual_margin, residual_total])
    covariance = np.cov(residual_array).tolist()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"margin_model": final_margin, "total_model": final_total, "calibrators": calibrators},
        ARTIFACT_DIR / "cfb_model.joblib",
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_source": "SportsDataverse ESPN-derived schedules and resolved betting releases",
        "train_window": [first_season, last_season],
        "games": len(records),
        "population": "FBS-vs-FBS games with posted spread and total",
        "feature_names": FEATURE_NAMES,
        "market_features": [],
        "selected_family": selected_family,
        "family_walk_forward": family_reports,
        "walk_forward": selected_walk_forward,
        "residual_distribution": {
            "kind": "bivariate_gaussian_oof",
            "samples": len(residual_margin),
            "margin_sigma": round(float(np.std(residual_margin, ddof=1)), 6),
            "total_sigma": round(float(np.std(residual_total, ddof=1)), 6),
            "correlation": round(float(np.corrcoef(residual_array)[0, 1]), 6),
            "covariance": [[round(float(value), 6) for value in row] for row in covariance],
        },
        "calibration": {
            market: {
                "method": "isotonic_on_season_walk_forward_predictions",
                "samples": len(calibration_truth[market]),
                "raw_brier": round(_brier(calibration_truth[market], calibration_input[market]), 6),
                "calibrated_brier": round(
                    _brier(
                        calibration_truth[market],
                        [float(value) for value in calibrators[market].predict(calibration_input[market])],
                    ),
                    6,
                ),
            }
            for market in ("moneyline", "spread", "total")
        },
        "shadow_mode": True,
        "promotion_status": "not_qualified",
        "financial_backtest": {
            "moneyline": "unavailable_no_complete_historical_two_sided_prices",
            "spread": "forecast_direction_only",
            "total": "forecast_direction_only",
        },
        "notes": "Sportsbook lines are evaluation/pricing only and are absent from FEATURE_NAMES; live shadow evidence is required before promotion.",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = train()
    print(json.dumps({key: result[key] for key in ("model_version", "games", "selected_family")}, indent=2))
    for row in result["walk_forward"]:
        print(row)
