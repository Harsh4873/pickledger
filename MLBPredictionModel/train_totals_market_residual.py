"""
train_totals_market_residual.py
--------------------------------
Experimental totals model that predicts the *residual* between actual run
totals and the closing market total line, rather than trying to predict the
raw run total from scratch.

  target  =  actual_total_runs  −  market_total_line
  predict =  market_total_line  +  model_residual_prediction

Intuition: the market total is already an efficient aggregate of pitcher
matchup, park, weather and lineup.  We only need to learn "how far off is
the market, and in which direction?".  A well-calibrated residual model
that is near-zero on average but occasionally catches systematic biases
(e.g. bullpen fatigue, umpire bias, travel effects the market underweights)
can outperform a raw prediction model.

Fallback: if the odds archive has no totals-line data, the script falls back
to using ``heuristic_total_runs`` as the market proxy.  This is noted clearly
in the metadata so you can tell the two conditions apart.

Split / seasons:
  Train  – 2024 regular season
  Val    – 2025 regular season  (for tuning and early reporting)
  Test   – 2026-to-date         (reporting only)

Artifacts saved (distinct names – does NOT touch production):
  MLBPredictionModel/artifacts/mlb_totals_market_residual_model.joblib
  MLBPredictionModel/artifacts/mlb_totals_market_residual_metadata.json

Usage
-----
  cd MLBPredictionModel
  python train_totals_market_residual.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from experimental_splits import load_splits, split_summary
from totals_model import (
    ARTIFACT_DIR,
    TOTALS_CATEGORICAL_FEATURES,
    TOTALS_NUMERIC_FEATURES,
    add_totals_features,
    blend_totals,
    evaluate_totals,
)


# ---------------------------------------------------------------------------
# Artifact paths  (distinct names – do NOT shadow production)
# ---------------------------------------------------------------------------
MODEL_PATH    = ARTIFACT_DIR / "mlb_totals_market_residual_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "mlb_totals_market_residual_metadata.json"


# ---------------------------------------------------------------------------
# Features for the residual model
# ---------------------------------------------------------------------------
# All existing totals features are kept so the model can learn which stats
# predict market mis-pricing. The market total line itself is added so the
# model can condition on it (useful in edge cases where the line is extreme).
RESIDUAL_NUMERIC_FEATURES: list[str] = TOTALS_NUMERIC_FEATURES + ["market_total_line"]
RESIDUAL_CATEGORICAL_FEATURES: list[str] = TOTALS_CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Market totals enrichment
# ---------------------------------------------------------------------------

def _median_float(values: list[float]) -> float | None:
    clean = sorted(v for v in values if v == v)  # filter NaN
    if not clean:
        return None
    mid = len(clean) // 2
    return clean[mid]


def _extract_entry_total_line(entry: dict[str, Any]) -> float | None:
    """Extract the closing over/under line from a single odds archive entry."""
    odds = entry.get("odds") or {}
    # Try multiple candidate keys
    for key in ("totals", "total", "overUnder", "over_under"):
        totals_data = odds.get(key)
        if totals_data is None:
            continue

        if isinstance(totals_data, (int, float)):
            return float(totals_data)

        if isinstance(totals_data, list) and totals_data:
            lines: list[float] = []
            for book in totals_data:
                for src in (
                    book.get("currentLine") or {},
                    book.get("closingLine") or {},
                    book,
                ):
                    for field in ("overUnder", "total", "line", "value"):
                        v = src.get(field)
                        if v is not None:
                            try:
                                lines.append(float(v))
                                break
                            except (TypeError, ValueError):
                                pass
                    if lines and lines[-1] > 0:
                        break
            candidate = _median_float(lines)
            if candidate is not None and candidate > 0:
                return candidate

    return None


def enrich_with_market_total_line(
    frame: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Add ``market_total_line`` column to the frame.

    Returns the enriched frame and a string describing the source:
      "odds_archive"  – real closing totals from the historical odds archive
      "heuristic"     – fallback: heuristic_total_runs used as proxy
      "unavailable"   – neither source available
    """
    from mlb_api import HistoricalOddsArchive

    frame = frame.copy()

    # Try the odds archive first
    try:
        archive = HistoricalOddsArchive()
        index   = archive.build_index()
    except Exception as exc:
        if verbose:
            print(f"  [warn] Could not load odds archive ({exc}); using heuristic fallback.")
        index = {}

    required = {"game_date", "away_abbrev", "home_abbrev"}
    if index and required.issubset(frame.columns):
        total_lines: list[float | None] = []
        for row in frame.itertuples(index=False):
            gdate = str(getattr(row, "game_date", ""))[:10]
            away  = str(getattr(row, "away_abbrev", "")).upper()
            home  = str(getattr(row, "home_abbrev", "")).upper()
            entry = index.get((gdate, away, home))
            if entry is None:
                total_lines.append(None)
            else:
                total_lines.append(_extract_entry_total_line(entry))

        frame["market_total_line"] = [
            x if x is not None else np.nan for x in total_lines
        ]
        n_covered = int(frame["market_total_line"].notna().sum())
        coverage  = round(100 * n_covered / max(len(frame), 1), 1)

        if verbose:
            print(f"  Odds archive totals coverage: {n_covered}/{len(frame)} ({coverage}%)")

        if n_covered >= 20:
            return frame, "odds_archive"

        if verbose:
            print(f"  [warn] Insufficient totals coverage ({n_covered} rows); using heuristic fallback.")

    # Fallback: heuristic_total_runs as market proxy
    if "heuristic_total_runs" in frame.columns:
        frame["market_total_line"] = frame["heuristic_total_runs"].copy()
        if verbose:
            print("  [info] Using heuristic_total_runs as market_total_line proxy.")
        return frame, "heuristic"

    frame["market_total_line"] = np.nan
    return frame, "unavailable"


# ---------------------------------------------------------------------------
# Target: residual
# ---------------------------------------------------------------------------

def compute_residual_target(frame: pd.DataFrame) -> pd.Series:
    """Return ``actual_total_runs − market_total_line``.  Rows with NaN line
    will have NaN residual and are dropped before training."""
    return (frame["total_runs"] - frame["market_total_line"]).rename("residual")


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_frame(raw_df: pd.DataFrame, verbose: bool = False) -> tuple[pd.DataFrame, str]:
    """Apply totals feature engineering then add market total line."""
    frame = add_totals_features(raw_df)
    frame, source = enrich_with_market_total_line(frame, verbose=verbose)
    for col in RESIDUAL_NUMERIC_FEATURES + RESIDUAL_CATEGORICAL_FEATURES:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame, source


def select_residual_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[RESIDUAL_NUMERIC_FEATURES + RESIDUAL_CATEGORICAL_FEATURES].copy()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(
    max_depth: int = 4,
    learning_rate: float = 0.05,
    max_iter: int = 350,
) -> Pipeline:
    preprocessor = ColumnTransformer([
        (
            "numeric",
            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
            RESIDUAL_NUMERIC_FEATURES,
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            RESIDUAL_CATEGORICAL_FEATURES,
        ),
    ])
    model = HistGradientBoostingRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        min_samples_leaf=35,
        l2_regularization=0.1,
        random_state=42,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


# ---------------------------------------------------------------------------
# Sample weights
# ---------------------------------------------------------------------------

def build_sample_weights(frame: pd.DataFrame, season_weights: dict[int, float]) -> np.ndarray:
    year_col = frame["game_date"].dt.year
    return year_col.map(lambda y: season_weights.get(y, 1.0)).to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Blend-weight selection (same structure as production)
# ---------------------------------------------------------------------------

def choose_blend_weight_totals(
    target: pd.Series,
    model_preds: np.ndarray,
    heuristic_preds: np.ndarray,
) -> tuple[float, dict[str, float]]:
    heuristic_metrics = evaluate_totals(target, heuristic_preds)
    h_pair = (heuristic_metrics["mae"], heuristic_metrics["rmse"])

    best_alpha  = 1.0
    best_met    = evaluate_totals(target, model_preds)
    best_prior  = (
        best_met["mae"] <= h_pair[0] and best_met["rmse"] <= h_pair[1],
        -best_met["mae"],
        -best_met["rmse"],
    )

    for alpha in [round(s / 20.0, 2) for s in range(1, 20)]:
        blended  = blend_totals(model_preds, heuristic_preds, alpha)
        met      = evaluate_totals(target, blended)
        priority = (
            met["mae"] <= h_pair[0] and met["rmse"] <= h_pair[1],
            -met["mae"],
            -met["rmse"],
        )
        if priority > best_prior:
            best_alpha  = alpha
            best_met    = met
            best_prior  = priority

    return best_alpha, best_met


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def reconstruct_and_blend(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    blend_weight_model: float,
) -> np.ndarray:
    """Run pipeline, add back market line, blend with heuristic."""
    feat_x      = select_residual_features(frame)
    residual    = pipeline.predict(feat_x)

    market_line = frame["market_total_line"].fillna(
        frame["heuristic_total_runs"]
    ).to_numpy()
    reconstructed = market_line + residual

    heuristic = frame["heuristic_total_runs"].to_numpy()
    return blend_totals(reconstructed, heuristic, blend_weight_model)


def eval_split(
    pipeline:     Pipeline,
    prep_df:      pd.DataFrame,
    blend_weight: float,
    split_name:   str,
) -> dict[str, Any] | None:
    if prep_df.empty:
        return None
    blended = reconstruct_and_blend(pipeline, prep_df, blend_weight)
    actual  = prep_df["total_runs"].to_numpy(dtype=float)
    valid   = ~np.isnan(actual)
    return {
        "rows":          int(valid.sum()),
        "metrics":       evaluate_totals(pd.Series(actual[valid]), blended[valid]),
        "residual_bias": float(np.nanmean(prep_df["total_runs"].to_numpy() - prep_df["market_total_line"].to_numpy())),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_totals_market_residual(
    *,
    train_seasons: tuple[int, ...] = (2024,),
    val_seasons:   tuple[int, ...] = (2025,),
    season_weights: dict[int, float] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the experimental market-residual totals model.

    Parameters
    ----------
    train_seasons:    Years for training data (default: 2024).
    val_seasons:      Years for validation (default: 2025).
    season_weights:   Optional per-year sample weights for training.
    verbose:          Print progress.

    Returns
    -------
    dict with "pipeline" and "metadata" keys.
    """
    splits = load_splits(train_seasons=train_seasons, val_seasons=val_seasons)

    if verbose:
        print("\n=== Experimental Totals (market-residual) ===\n")
        print(split_summary(splits))
        print()

    train_raw = splits["train"]
    val_raw   = splits["val"]
    test_raw  = splits["test"]

    if train_raw.empty:
        raise ValueError(
            f"Training split is empty for seasons={train_seasons}. "
            "Make sure the dataset CSV covers those seasons."
        )

    # ----------------------------------------------------------------
    # Prepare features and targets
    # ----------------------------------------------------------------
    if verbose:
        print("Preparing training data…")
    train_prep, train_source = prepare_frame(train_raw, verbose=verbose)
    train_residual = compute_residual_target(train_prep)

    # Drop rows where we can't compute the residual (no market line)
    train_valid_mask = train_residual.notna() & train_prep["total_runs"].notna()
    train_prep_clean = train_prep[train_valid_mask].copy().reset_index(drop=True)
    train_res_clean  = train_residual[train_valid_mask].reset_index(drop=True)

    if verbose:
        n_dropped = (~train_valid_mask).sum()
        print(f"  Dropped {n_dropped} training rows with missing market line or total runs.")
        print(f"  Training on {len(train_prep_clean)} rows.")

    if len(train_prep_clean) < 50:
        raise ValueError(
            f"Too few training rows ({len(train_prep_clean)}) after filtering. "
            "Check odds archive coverage for training seasons."
        )

    train_x       = select_residual_features(train_prep_clean)
    sw: np.ndarray | None = None
    if season_weights:
        sw = build_sample_weights(train_prep_clean, season_weights)

    # ----------------------------------------------------------------
    # Fit
    # ----------------------------------------------------------------
    if verbose:
        print("Fitting residual regression model…")
    pipeline = build_pipeline()
    if sw is not None:
        pipeline.fit(train_x, train_res_clean, model__sample_weight=sw)
    else:
        pipeline.fit(train_x, train_res_clean)

    # ----------------------------------------------------------------
    # Blend-weight on val
    # ----------------------------------------------------------------
    val_eval: dict[str, Any] | None = None
    blend_weight = 0.65  # default

    if not val_raw.empty:
        if verbose:
            print("Evaluating on val (2025)…")
        val_prep, val_source = prepare_frame(val_raw, verbose=verbose)
        val_actual   = val_prep["total_runs"].to_numpy(dtype=float)
        val_heuristic = val_prep["heuristic_total_runs"].to_numpy()

        val_feat_x   = select_residual_features(val_prep)
        val_residual = pipeline.predict(val_feat_x)
        val_market   = val_prep["market_total_line"].fillna(val_prep["heuristic_total_runs"]).to_numpy()
        val_reconstr = val_market + val_residual

        blend_weight, val_blended_met = choose_blend_weight_totals(
            pd.Series(val_actual), val_reconstr, val_heuristic
        )
        raw_val_met  = evaluate_totals(pd.Series(val_actual), val_reconstr)
        heur_val_met = evaluate_totals(pd.Series(val_actual), val_heuristic)

        val_eval = {
            "rows":             len(val_prep),
            "blend_weight":     blend_weight,
            "raw_metrics":      raw_val_met,
            "blended_metrics":  val_blended_met,
            "heuristic_metrics": heur_val_met,
            "market_line_source": val_source,
        }

        if verbose:
            print(f"  Blend weight: {blend_weight}")
            print(f"  Val raw  (residual reconstr): RMSE={raw_val_met['rmse']:.3f} MAE={raw_val_met['mae']:.3f}")
            print(f"  Val blend:                    RMSE={val_blended_met['rmse']:.3f} MAE={val_blended_met['mae']:.3f}")
            print(f"  Heuristic baseline:           RMSE={heur_val_met['rmse']:.3f} MAE={heur_val_met['mae']:.3f}")

    # ----------------------------------------------------------------
    # Evaluate train and test
    # ----------------------------------------------------------------
    train_eval = eval_split(pipeline, train_prep_clean, blend_weight, "train")

    test_eval: dict[str, Any] | None = None
    if not test_raw.empty:
        if verbose:
            print("Evaluating on 2026 test split…")
        test_prep, test_source = prepare_frame(test_raw, verbose=verbose)
        test_eval  = eval_split(pipeline, test_prep, blend_weight, "test")
        if test_eval:
            test_eval["market_line_source"] = test_source
        if verbose and test_eval:
            print(f"  Test (2026): RMSE={test_eval['metrics']['rmse']:.3f} MAE={test_eval['metrics']['mae']:.3f}")

    # ----------------------------------------------------------------
    # Compute training residual bias (useful diagnostics)
    # ----------------------------------------------------------------
    train_feat_x   = select_residual_features(train_prep_clean)
    train_res_pred = pipeline.predict(train_feat_x)
    residual_bias  = float(np.mean(train_res_clean.to_numpy() - train_res_pred))

    # ----------------------------------------------------------------
    # Persist artifacts
    # ----------------------------------------------------------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "model_type":               "hist_gradient_boosting_residual",
        "approach":                 "market_residual_regression",
        "target":                   "actual_total_runs - market_total_line",
        "reconstruction":           "predicted_total = market_total_line + predicted_residual",
        "train_seasons":            list(train_seasons),
        "val_seasons":              list(val_seasons),
        "season_weights":           season_weights or {},
        "market_line_source_train": train_source,
        "numeric_features":         RESIDUAL_NUMERIC_FEATURES,
        "categorical_features":     RESIDUAL_CATEGORICAL_FEATURES,
        "blend_weight_model":       blend_weight,
        "train_residual_bias":      residual_bias,
        "train_eval":               train_eval,
        "val_eval":                 val_eval,
        "test_eval":                test_eval,
        "production_model_unchanged": True,
    }

    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nArtifact saved → {MODEL_PATH}")
        print(f"Metadata saved → {METADATA_PATH}")

    return {"pipeline": pipeline, "metadata": metadata}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_totals_market_residual(
        train_seasons=(2024,),
        val_seasons=(2025,),
        # 2026 is test-only; no training weight
        season_weights={2024: 1.0},
        verbose=True,
    )
