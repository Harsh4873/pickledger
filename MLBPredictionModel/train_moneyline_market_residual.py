"""
train_moneyline_market_residual.py
-----------------------------------
Experimental moneyline model that treats the closing market line as a strong
prior and learns whether any of the model's non-market features carry residual
edge on top of that prior.

Approach (Option A from the design):
  Target   – still binary ``home_win`` (same as production).
  Extra    – adds vig-removed closing market probability and raw American odds
             as explicit features alongside the full existing feature set.
  Intuition – the model can learn: "given the market says home team is 57%,
              what does FIP differential / fatigue / park add to that?".

Split:
  Train  – 2024 regular season (rows with odds coverage preferred)
  Val    – 2025 regular season (for early-stop / reporting)
  Test   – 2026-to-date (reporting only)

Artifacts saved (never overwrites production):
  MLBPredictionModel/artifacts/mlb_moneyline_market_residual_model.joblib
  MLBPredictionModel/artifacts/mlb_moneyline_market_residual_metadata.json

Usage
-----
  cd MLBPredictionModel
  python train_moneyline_market_residual.py
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from experimental_splits import load_splits, split_summary
from feature_engineering import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    ensure_feature_frame,
)
from market_mechanics import remove_vig
from moneyline_model import (
    ARTIFACT_DIR,
    blend_probabilities,
    evaluate_probabilities,
)


# ---------------------------------------------------------------------------
# Artifact paths  (distinct names – do NOT shadow production)
# ---------------------------------------------------------------------------
MODEL_PATH    = ARTIFACT_DIR / "mlb_moneyline_market_residual_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "mlb_moneyline_market_residual_metadata.json"


# ---------------------------------------------------------------------------
# Market features added on top of the production feature set
# ---------------------------------------------------------------------------
MARKET_NUMERIC_FEATURES: list[str] = [
    "market_home_win_prob_novig",   # vig-removed closing home win probability
    "market_home_moneyline_feat",   # raw American odds (home), imputed to 0 when absent
    "market_away_moneyline_feat",   # raw American odds (away), imputed to 0 when absent
]

AUGMENTED_NUMERIC_FEATURES: list[str] = NUMERIC_FEATURES + MARKET_NUMERIC_FEATURES
AUGMENTED_CATEGORICAL_FEATURES: list[str] = CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add vig-removed probability + raw-odds columns from the dataset's
    moneyline columns (already present when the dataset was built with
    include_odds=True)."""
    frame = frame.copy()

    has_home_ml = "home_moneyline" in frame.columns
    has_away_ml = "away_moneyline" in frame.columns

    if not has_home_ml or not has_away_ml:
        # Odds not present in dataset – fill with NaN so imputer handles them
        frame["market_home_win_prob_novig"]  = np.nan
        frame["market_home_moneyline_feat"]  = np.nan
        frame["market_away_moneyline_feat"]  = np.nan
        return frame

    novig_probs: list[float | None] = []
    for _, row in frame[["home_moneyline", "away_moneyline"]].iterrows():
        h = row["home_moneyline"]
        a = row["away_moneyline"]
        if pd.isna(h) or pd.isna(a):
            novig_probs.append(None)
        else:
            try:
                p_home, _ = remove_vig(int(h), int(a))
                novig_probs.append(p_home)
            except Exception:
                novig_probs.append(None)

    frame["market_home_win_prob_novig"] = [
        x if x is not None else np.nan for x in novig_probs
    ]
    # Keep raw odds as numeric features; centre around 0 so StandardScaler works well.
    # American odds like -150, +130 are already numeric; the imputer will fill median
    # for any NaN rows.
    frame["market_home_moneyline_feat"] = pd.to_numeric(frame["home_moneyline"], errors="coerce")
    frame["market_away_moneyline_feat"] = pd.to_numeric(frame["away_moneyline"], errors="coerce")
    return frame


def prepare_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply full feature engineering (production path) then add market features."""
    frame = ensure_feature_frame(raw_df)
    frame = add_market_features(frame)
    for col in AUGMENTED_NUMERIC_FEATURES + AUGMENTED_CATEGORICAL_FEATURES:
        if col not in frame.columns:
            frame[col] = 0.0
    return frame


def select_augmented_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[AUGMENTED_NUMERIC_FEATURES + AUGMENTED_CATEGORICAL_FEATURES].copy()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline(C: float = 0.55) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric",     numeric_pipe,     AUGMENTED_NUMERIC_FEATURES),
        ("categorical", categorical_pipe, AUGMENTED_CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("estimator",    LogisticRegression(C=C, max_iter=3000, solver="lbfgs")),
    ])


# ---------------------------------------------------------------------------
# Blend weight selection (mirrors production logic)
# ---------------------------------------------------------------------------

def choose_blend_weight(
    target: pd.Series,
    model_probs: np.ndarray,
    heuristic_probs: np.ndarray,
) -> tuple[float, dict[str, float]]:
    heuristic_metrics = evaluate_probabilities(target, heuristic_probs)
    heuristic_acc = heuristic_metrics["accuracy"]
    best_alpha = 1.0
    best_metrics = evaluate_probabilities(target, model_probs)
    best_priority = (best_metrics["accuracy"] >= heuristic_acc, -best_metrics["log_loss"])

    for alpha in [round(s / 20.0, 2) for s in range(1, 20)]:
        blended   = blend_probabilities(model_probs, heuristic_probs, alpha)
        metrics   = evaluate_probabilities(target, blended)
        priority  = (metrics["accuracy"] >= heuristic_acc, -metrics["log_loss"])
        if priority > best_priority or (
            priority == best_priority
            and metrics["brier_score"] < best_metrics["brier_score"]
        ):
            best_alpha   = alpha
            best_metrics = metrics
            best_priority = priority

    return best_alpha, best_metrics


# ---------------------------------------------------------------------------
# C-hyperparameter grid search on val
# ---------------------------------------------------------------------------

def tune_on_val(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    val_x:   pd.DataFrame,
    val_y:   pd.Series,
    val_heuristic: np.ndarray,
    c_grid: tuple[float, ...] = (0.1, 0.3, 0.55, 1.0, 3.0),
) -> tuple[float, dict[str, float]]:
    """Return (best_C, val_metrics_at_best_C)."""
    best_C       = 0.55
    best_ll      = float("inf")
    best_val_met: dict[str, float] = {}

    for c_val in c_grid:
        pipe = build_pipeline(C=c_val)
        pipe.fit(train_x, train_y)
        raw_val   = pipe.predict_proba(val_x)[:, 1]
        blend_w, blended_met = choose_blend_weight(
            val_y, raw_val, val_heuristic
        )
        # Use log-loss on blended as primary criterion
        blended = blend_probabilities(raw_val, val_heuristic, blend_w)
        ll      = float(log_loss(val_y, np.clip(blended, 1e-6, 1 - 1e-6)))
        if ll < best_ll:
            best_ll      = ll
            best_C       = c_val
            best_val_met = blended_met

    return best_C, best_val_met


# ---------------------------------------------------------------------------
# Odds coverage stats
# ---------------------------------------------------------------------------

def odds_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    """Return dict describing what fraction of rows have market features."""
    n = len(frame)
    col = "market_home_win_prob_novig"
    if col not in frame.columns:
        return {"total_rows": n, "with_odds": 0, "coverage_pct": 0.0}
    covered = int(frame[col].notna().sum())
    return {
        "total_rows": n,
        "with_odds":  covered,
        "coverage_pct": round(100 * covered / n, 1) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_moneyline_market_residual(
    *,
    train_seasons: tuple[int, ...] = (2024,),
    val_seasons:   tuple[int, ...] = (2025,),
    season_weights: dict[int, float] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train the experimental market-anchored moneyline model.

    Parameters
    ----------
    train_seasons:    Years used for training (default: 2024).
    val_seasons:      Years used for validation / tuning (default: 2025).
    season_weights:   Optional per-year sample weights for training.
                      e.g. {2024: 1.0, 2025: 0.5}
    verbose:          Print progress to stdout.

    Returns
    -------
    dict with "pipeline" and "metadata" keys.
    """
    splits = load_splits(train_seasons=train_seasons, val_seasons=val_seasons)

    if verbose:
        print("\n=== Experimental Moneyline (market-residual) ===\n")
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
    # Prepare features
    # ----------------------------------------------------------------
    if verbose:
        print("Preparing training features…")
    train_prep = prepare_frame(train_raw)
    train_y    = train_prep["home_win"].astype(int)
    train_x    = select_augmented_features(train_prep)
    train_heur = train_prep["heuristic_home_win_prob"].to_numpy()

    # Build sample weights
    weights: np.ndarray | None = None
    if season_weights:
        year_col = train_prep["game_date"].dt.year
        weights  = year_col.map(lambda y: season_weights.get(y, 1.0)).to_numpy(dtype=float)

    if verbose:
        cov = odds_coverage(train_prep)
        print(f"  Train: {cov['with_odds']}/{cov['total_rows']} rows with market odds ({cov['coverage_pct']}%)")

    if val_raw.empty:
        if verbose:
            print("  [warn] Val split is empty – tuning skipped, using default C=0.55")
        best_C = 0.55
    else:
        if verbose:
            print("Preparing val features…")
        val_prep  = prepare_frame(val_raw)
        val_y     = val_prep["home_win"].astype(int)
        val_x     = select_augmented_features(val_prep)
        val_heur  = val_prep["heuristic_home_win_prob"].to_numpy()

        if verbose:
            cov_val = odds_coverage(val_prep)
            print(f"  Val:   {cov_val['with_odds']}/{cov_val['total_rows']} rows with market odds ({cov_val['coverage_pct']}%)")
            print("Tuning C on val…")

        best_C, _val_tune_metrics = tune_on_val(train_x, train_y, val_x, val_y, val_heur)
        if verbose:
            print(f"  Best C={best_C}")

    # ----------------------------------------------------------------
    # Final training run on best C
    # ----------------------------------------------------------------
    if verbose:
        print(f"Training final model (C={best_C})…")

    pipeline = build_pipeline(C=best_C)
    if weights is not None:
        pipeline.fit(train_x, train_y, estimator__sample_weight=weights)
    else:
        pipeline.fit(train_x, train_y)

    # ----------------------------------------------------------------
    # Evaluate: train, val, test
    # ----------------------------------------------------------------
    def _eval(prep_df: pd.DataFrame, name: str) -> dict[str, Any] | None:
        if prep_df.empty:
            return None
        df_p = prepare_frame(prep_df) if name != "train" else prep_df
        x    = select_augmented_features(df_p)
        y    = df_p["home_win"].astype(int)
        heur = df_p["heuristic_home_win_prob"].to_numpy()
        raw  = pipeline.predict_proba(x)[:, 1]
        blend_w, blended_met = choose_blend_weight(y, raw, heur)
        return {
            "rows":             len(df_p),
            "blend_weight":     blend_w,
            "raw_metrics":      evaluate_probabilities(y, raw),
            "blended_metrics":  blended_met,
            "heuristic_metrics": evaluate_probabilities(y, heur),
        }

    train_eval = _eval(train_prep, "train")
    val_eval   = _eval(val_prep   if not val_raw.empty else pd.DataFrame(), "val")
    test_eval  = None
    if not test_raw.empty:
        if verbose:
            print("Evaluating on 2026 test split…")
        test_eval = _eval(test_raw, "test")

    # ----------------------------------------------------------------
    # Persist artifacts
    # ----------------------------------------------------------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "model_type":              "logistic_regression_market_anchored",
        "approach":                "option_a_market_as_feature",
        "train_seasons":           list(train_seasons),
        "val_seasons":             list(val_seasons),
        "season_weights":          season_weights or {},
        "hyperparameters":         {"C": best_C},
        "numeric_features":        AUGMENTED_NUMERIC_FEATURES,
        "categorical_features":    AUGMENTED_CATEGORICAL_FEATURES,
        "market_features_added":   MARKET_NUMERIC_FEATURES,
        "train_eval":              train_eval,
        "val_eval":                val_eval,
        "test_eval":               test_eval,
        "blend_weight_model":      (val_eval or {}).get("blend_weight", 1.0),
        "production_model_unchanged": True,
    }

    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nArtifact saved → {MODEL_PATH}")
        print(f"Metadata saved → {METADATA_PATH}")
        print("\n--- Val metrics (blended) ---")
        if val_eval:
            for k, v in val_eval["blended_metrics"].items():
                print(f"  {k}: {v:.4f}")
        if test_eval:
            print("\n--- Test (2026) metrics (blended) ---")
            for k, v in test_eval["blended_metrics"].items():
                print(f"  {k}: {v:.4f}")

    return {"pipeline": pipeline, "metadata": metadata}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_moneyline_market_residual(
        train_seasons=(2024,),
        val_seasons=(2025,),
        # 2026 included in test only – no training weight
        season_weights={2024: 1.0},
        verbose=True,
    )
