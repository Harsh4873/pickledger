"""Train the NFL model heads with a strict walk-forward-by-season protocol.

Three heads, mirroring MLBPredictionModel v2's proven stack:
  1. moneyline  — HistGradientBoostingClassifier -> isotonic calibration,
     where the isotonic layer is fit ONLY on out-of-sample walk-forward
     predictions (train <= season N-1, predict season N) so the calibration
     never sees its own training data.
  2. spread     — HistGradientBoostingRegressor on (margin - spread_line).
  3. total      — HistGradientBoostingRegressor on (total - total_line).

Walk-forward report: for each season 2015-2025, train on everything prior
and score that season. Artifacts + metadata land in artifacts/.

Run locally or via a manual workflow — never in the daily cron.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

try:
    from nfl_core import FEATURE_NAMES, build_dataset, load_games, matrix
except ImportError:
    from .nfl_core import FEATURE_NAMES, build_dataset, load_games, matrix

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_VERSION = "nfl_v0_games_ewma"
FIRST_TRAIN_SEASON = 2002
WALK_FORWARD_SEASONS = range(2015, 2026)

MARGIN_RESIDUAL_SIGMA = 13.2
TOTAL_RESIDUAL_SIGMA = 13.5


def _classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=300, random_state=7)


def _regressor() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_depth=4, learning_rate=0.06, max_iter=300, random_state=7)


def _brier(y_true: list[int], probs: list[float]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, y_true)) / max(1, len(y_true))


def train(first_season: int = FIRST_TRAIN_SEASON, last_season: int = 2025) -> dict:
    records = build_dataset(load_games(), first_season=first_season, last_season=last_season)
    if len(records) < 500:
        raise SystemExit(f"dataset too small ({len(records)} games) — refusing to train")

    oof_probs: list[float] = []
    oof_truth: list[int] = []
    walk_forward: list[dict] = []
    for season in WALK_FORWARD_SEASONS:
        if season > last_season:
            continue
        train_recs = [r for r in records if r["season"] < season]
        test_recs = [r for r in records if r["season"] == season]
        if len(train_recs) < 500 or not test_recs:
            continue
        clf = _classifier().fit(matrix(train_recs), [r["home_win"] for r in train_recs])
        probs = [float(p) for p in clf.predict_proba(matrix(test_recs))[:, 1]]
        truth = [r["home_win"] for r in test_recs]
        oof_probs.extend(probs)
        oof_truth.extend(truth)

        spread_reg = _regressor().fit(matrix(train_recs), [r["margin_residual"] for r in train_recs])
        spread_pred = [float(v) for v in spread_reg.predict(matrix(test_recs))]
        cover_hits = sum(
            1 for pred, rec in zip(spread_pred, test_recs)
            if rec["margin_residual"] != 0 and (pred > 0) == (rec["margin_residual"] > 0)
        )
        cover_graded = sum(1 for rec in test_recs if rec["margin_residual"] != 0)

        total_reg = _regressor().fit(matrix(train_recs), [r["total_residual"] for r in train_recs])
        total_pred = [float(v) for v in total_reg.predict(matrix(test_recs))]
        total_hits = sum(
            1 for pred, rec in zip(total_pred, test_recs)
            if rec["total_residual"] != 0 and (pred > 0) == (rec["total_residual"] > 0)
        )
        total_graded = sum(1 for rec in test_recs if rec["total_residual"] != 0)

        walk_forward.append({
            "season": season,
            "games": len(test_recs),
            "ml_brier": round(_brier(truth, probs), 5),
            "spread_direction_rate": round(cover_hits / cover_graded, 4) if cover_graded else None,
            "total_direction_rate": round(total_hits / total_graded, 4) if total_graded else None,
        })

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso.fit(oof_probs, oof_truth)

    final_ml = _classifier().fit(matrix(records), [r["home_win"] for r in records])
    final_spread = _regressor().fit(matrix(records), [r["margin_residual"] for r in records])
    final_total = _regressor().fit(matrix(records), [r["total_residual"] for r in records])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_ml, ARTIFACT_DIR / "nfl_ml.joblib")
    joblib.dump(iso, ARTIFACT_DIR / "nfl_ml_isotonic.joblib")
    joblib.dump(final_spread, ARTIFACT_DIR / "nfl_spread.joblib")
    joblib.dump(final_total, ARTIFACT_DIR / "nfl_total.joblib")

    market_brier = _brier(
        oof_truth,
        [1.0 / (1.0 + math.exp(-0.16 * r["features"]["spread_line"])) for r in records if r["season"] in set(WALK_FORWARD_SEASONS)],
    ) if records else None

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_window": [first_season, last_season],
        "games": len(records),
        "feature_names": FEATURE_NAMES,
        "margin_residual_sigma": MARGIN_RESIDUAL_SIGMA,
        "total_residual_sigma": TOTAL_RESIDUAL_SIGMA,
        "walk_forward": walk_forward,
        "oof_ml_brier": round(_brier(oof_truth, oof_probs), 5),
        "market_reference_brier": round(market_brier, 5) if market_brier is not None else None,
        "notes": "Phase-1 provisional (games.csv EWMA features); EPA/play features arrive in Phase 2.",
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    meta = train()
    print(json.dumps({k: meta[k] for k in ("model_version", "games", "oof_ml_brier")}, indent=2))
    for row in meta["walk_forward"]:
        print(row)
