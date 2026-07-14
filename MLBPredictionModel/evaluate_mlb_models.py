"""
evaluate_mlb_models.py
-----------------------
Champion-vs-challenger evaluation for MLB moneyline and totals models.
Read-only with respect to all artifacts: nothing here trains or saves anything.

Produces a side-by-side table on the 2025 (val) and 2026-to-date (test) splits:

  Moneyline  –  accuracy | log-loss | Brier | simulated ROI
  Totals     –  RMSE | MAE | simulated ROI

Models evaluated (when available):
  1. Market-only baseline  – vig-removed closing probability / closing total line
  2. Production moneyline + calibration
  3. Production totals
  4. Experimental market-residual moneyline  (if artifact exists)
  5. Experimental market-residual totals     (if artifact exists)

ROI assumptions:
  Moneyline – 1-unit flat bet on home team when model probability exceeds
               vig-removed market probability by >= 3 pp.
  Totals    – 1-unit flat bet OVER/UNDER when model total differs from
               the closing market total line by >= 0.4 runs.
              Assumes -110 juice on both sides when book odds not available.

Usage
-----
  cd MLBPredictionModel
  python evaluate_mlb_models.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from calibration import apply_moneyline_calibration, load_calibration_artifact
from experimental_splits import load_splits, split_summary
from feature_engineering import ensure_feature_frame, select_feature_frame
from market_mechanics import convert_american_to_implied, remove_vig
from moneyline_model import (
    ARTIFACT_DIR,
    blend_probabilities,
    evaluate_probabilities,
    load_moneyline_model,
)
from totals_model import (
    MODEL_PATH as PROD_TOTALS_PATH,
    add_totals_features,
    evaluate_totals,
    select_totals_feature_frame,
)

# Experimental artifact paths (written by train_*_market_residual.py scripts)
EXP_ML_PATH     = ARTIFACT_DIR / "mlb_moneyline_market_residual_model.joblib"
EXP_TOTALS_PATH = ARTIFACT_DIR / "mlb_totals_market_residual_model.joblib"

# Betting thresholds
ML_EDGE_THRESHOLD     = 0.03   # minimum edge over market to trigger ML bet
TOTALS_GAP_THRESHOLD  = 0.4    # minimum run gap vs market line to trigger totals bet
DEFAULT_TOTALS_JUICE  = -110   # assumed juice when book odds unavailable


# ---------------------------------------------------------------------------
# Helpers: American odds → payout
# ---------------------------------------------------------------------------

def american_to_profit_per_unit(odds: int) -> float:
    """Return the net profit on a 1-unit winning bet at given American odds."""
    if odds >= 100:
        return odds / 100.0
    return 100.0 / abs(odds)


def _median_int(values: list[int]) -> int | None:
    clean = sorted(v for v in values if isinstance(v, int))
    if not clean:
        return None
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 == 1 else round((clean[mid - 1] + clean[mid]) / 2)


# ---------------------------------------------------------------------------
# Helpers: vig-removed market probability from dataset columns
# ---------------------------------------------------------------------------

def add_vig_removed_prob(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``market_home_win_prob_novig`` column if moneyline odds are present."""
    frame = frame.copy()
    if "home_moneyline" not in frame.columns or "away_moneyline" not in frame.columns:
        frame["market_home_win_prob_novig"] = np.nan
        return frame

    novig_home = np.full(len(frame), np.nan)
    for i, row in enumerate(frame.itertuples(index=False)):
        h = getattr(row, "home_moneyline", None)
        a = getattr(row, "away_moneyline", None)
        if pd.isna(h) or pd.isna(a):
            continue
        try:
            p_home, _ = remove_vig(int(h), int(a))
            novig_home[i] = p_home
        except Exception:
            pass

    frame["market_home_win_prob_novig"] = novig_home
    return frame


# ---------------------------------------------------------------------------
# Helpers: totals market line lookup via HistoricalOddsArchive
# ---------------------------------------------------------------------------

def _extract_entry_totals(entry: dict[str, Any]) -> tuple[float | None, int | None, int | None]:
    """Return (over_under_line, over_odds, under_odds) from an odds archive entry.

    Returns (None, None, None) when totals data is absent.
    """
    odds = entry.get("odds") or {}
    totals_list = odds.get("totals") or odds.get("total") or []
    if not isinstance(totals_list, list) or not totals_list:
        return None, None, None

    lines: list[float] = []
    over_odds:  list[int] = []
    under_odds: list[int] = []

    for book in totals_list:
        current  = book.get("currentLine") or {}
        opening  = book.get("openingLine") or {}
        line_val = (
            current.get("overUnder")
            or current.get("total")
            or book.get("overUnder")
            or book.get("total")
        )
        if line_val is not None:
            try:
                lines.append(float(line_val))
            except (TypeError, ValueError):
                pass

        for src in (current, opening, book):
            ov = src.get("overOdds")
            un = src.get("underOdds")
            if isinstance(ov, int):
                over_odds.append(ov)
            if isinstance(un, int):
                under_odds.append(un)

    if not lines:
        return None, None, None

    lines.sort()
    mid = len(lines) // 2
    median_line = lines[mid]
    return median_line, _median_int(over_odds), _median_int(under_odds)


def enrich_with_market_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``market_total_line``, ``market_over_odds``, ``market_under_odds``
    columns by querying the HistoricalOddsArchive.

    Rows where the archive has no totals data remain NaN.
    """
    from mlb_api import HistoricalOddsArchive

    frame = frame.copy()
    frame["market_total_line"]  = np.nan
    frame["market_over_odds"]   = np.nan
    frame["market_under_odds"]  = np.nan

    try:
        archive = HistoricalOddsArchive()
        index   = archive.build_index()
    except Exception as exc:
        print(f"  [warn] Could not load odds archive: {exc}")
        return frame

    required = {"game_date", "away_abbrev", "home_abbrev"}
    if not required.issubset(frame.columns):
        print("  [warn] Frame missing date/team columns – skipping totals market lookup.")
        return frame

    total_lines:  list[float | None] = []
    over_odds_col: list[int | None]   = []
    under_odds_col: list[int | None]  = []

    for row in frame.itertuples(index=False):
        gdate = str(getattr(row, "game_date", ""))[:10]
        away  = str(getattr(row, "away_abbrev", "")).upper()
        home  = str(getattr(row, "home_abbrev", "")).upper()
        entry = index.get((gdate, away, home))
        if entry is None:
            total_lines.append(None)
            over_odds_col.append(None)
            under_odds_col.append(None)
            continue
        line, ov, un = _extract_entry_totals(entry)
        total_lines.append(line)
        over_odds_col.append(ov)
        under_odds_col.append(un)

    frame["market_total_line"]  = [x if x is not None else np.nan for x in total_lines]
    frame["market_over_odds"]   = [x if x is not None else np.nan for x in over_odds_col]
    frame["market_under_odds"]  = [x if x is not None else np.nan for x in under_odds_col]
    return frame


# ---------------------------------------------------------------------------
# ROI simulators
# ---------------------------------------------------------------------------

def simulate_moneyline_roi(
    model_probs: np.ndarray,
    market_novig: np.ndarray,
    home_win: np.ndarray,
    home_ml_odds: np.ndarray,
    threshold: float = ML_EDGE_THRESHOLD,
) -> dict[str, float]:
    """Simulate flat 1-unit bets on home when model edge >= threshold.

    Returns dict with keys: bets, wins, roi_pct.
    """
    mask = (
        ~np.isnan(model_probs)
        & ~np.isnan(market_novig)
        & ~np.isnan(home_ml_odds)
        & ((model_probs - market_novig) >= threshold)
    )
    if mask.sum() == 0:
        return {"bets": 0, "wins": 0, "roi_pct": float("nan")}

    profits: list[float] = []
    for mp, hw, odds in zip(model_probs[mask], home_win[mask], home_ml_odds[mask]):
        if hw == 1:
            profits.append(american_to_profit_per_unit(int(odds)))
        else:
            profits.append(-1.0)

    bets = len(profits)
    wins = sum(1 for p in profits if p > 0)
    roi  = sum(profits) / bets * 100
    return {"bets": bets, "wins": wins, "roi_pct": round(roi, 2)}


def simulate_totals_roi(
    model_totals: np.ndarray,
    market_lines: np.ndarray,
    actual_totals: np.ndarray,
    over_odds: np.ndarray,
    under_odds: np.ndarray,
    threshold: float = TOTALS_GAP_THRESHOLD,
) -> dict[str, float]:
    """Simulate flat 1-unit bets OVER/UNDER when |model - market| >= threshold."""
    valid = ~np.isnan(model_totals) & ~np.isnan(market_lines) & ~np.isnan(actual_totals)
    if valid.sum() == 0:
        return {"bets": 0, "wins": 0, "roi_pct": float("nan")}

    profits: list[float] = []
    for mt, ml, at, ov, un in zip(
        model_totals[valid], market_lines[valid], actual_totals[valid],
        over_odds[valid], under_odds[valid],
    ):
        diff = mt - ml
        if abs(diff) < threshold:
            continue
        bet_over = diff > 0
        ov_int   = int(ov) if not np.isnan(ov) else DEFAULT_TOTALS_JUICE
        un_int   = int(un) if not np.isnan(un) else DEFAULT_TOTALS_JUICE

        if bet_over:
            won = at > ml
            odds_used = ov_int
        else:
            won = at < ml
            odds_used = un_int

        profits.append(american_to_profit_per_unit(odds_used) if won else -1.0)

    if not profits:
        return {"bets": 0, "wins": 0, "roi_pct": float("nan")}

    bets = len(profits)
    wins = sum(1 for p in profits if p > 0)
    roi  = sum(profits) / bets * 100
    return {"bets": bets, "wins": wins, "roi_pct": round(roi, 2)}


# ---------------------------------------------------------------------------
# Calibration summary (reliability bins)
# ---------------------------------------------------------------------------

def calibration_summary(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 5) -> str:
    """Return a compact reliability-bin string like '(0.45→0.47 0.50→0.52 ...)'."""
    bins   = np.linspace(0, 1, n_bins + 1)
    parts:  list[str] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() < 5:
            continue
        mean_p = probs[mask].mean()
        mean_o = outcomes[mask].mean()
        parts.append(f"{mean_p:.2f}→{mean_o:.2f}")
    return "(" + "  ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def evaluate_split_moneyline(
    split_df: pd.DataFrame,
    split_name: str,
) -> dict[str, dict[str, Any]]:
    """Evaluate all available moneyline models on ``split_df``.

    Returns a dict keyed by model label.
    """
    if split_df.empty:
        print(f"  [{split_name}] empty – skipping moneyline evaluation.")
        return {}

    outcomes = split_df["home_win"].astype(int).to_numpy()

    # ----------------------------------------------------------------
    # Market-only baseline
    # ----------------------------------------------------------------
    split_enriched = add_vig_removed_prob(split_df)
    market_novig   = split_enriched["market_home_win_prob_novig"].to_numpy()
    results: dict[str, dict[str, Any]] = {}

    market_mask = ~np.isnan(market_novig)
    if market_mask.sum() >= 20:
        mkt_probs = np.where(market_mask, market_novig, 0.52)
        results["market_only"] = {
            **evaluate_probabilities(
                pd.Series(outcomes[market_mask]), mkt_probs[market_mask]
            ),
            "n": int(market_mask.sum()),
            "calibration": calibration_summary(mkt_probs[market_mask], outcomes[market_mask].astype(float)),
            "roi": {"bets": 0, "wins": 0, "roi_pct": float("nan"), "note": "market IS the baseline"},
        }
    else:
        print(f"  [{split_name}] insufficient market odds coverage ({market_mask.sum()} rows) – market baseline skipped.")

    # ----------------------------------------------------------------
    # Production model
    # ----------------------------------------------------------------
    try:
        prod_artifact = load_moneyline_model()
        pipeline      = prod_artifact["pipeline"]
        metadata      = prod_artifact["metadata"]
        blend_w       = float(metadata.get("blend_weight_model", 1.0))

        prepared     = ensure_feature_frame(split_df)
        feat_x       = select_feature_frame(prepared)
        raw_model    = pipeline.predict_proba(feat_x)[:, 1]
        heuristic    = prepared["heuristic_home_win_prob"].to_numpy()
        blended      = blend_probabilities(raw_model, heuristic, blend_w)

        # Apply calibration
        calib_frame  = prepared.copy()
        calib_frame["raw_home_win_probability"] = blended
        calibrated_frame = apply_moneyline_calibration(calib_frame)
        cal_probs    = calibrated_frame["calibrated_home_win_probability"].to_numpy()

        home_ml_odds = split_enriched["home_moneyline"].to_numpy() if "home_moneyline" in split_enriched.columns else np.full(len(split_df), np.nan)

        results["prod_blended"] = {
            **evaluate_probabilities(pd.Series(outcomes), blended),
            "n": len(split_df),
            "calibration": calibration_summary(blended, outcomes.astype(float)),
            "roi": simulate_moneyline_roi(blended, market_novig, outcomes, home_ml_odds),
        }
        results["prod_calibrated"] = {
            **evaluate_probabilities(pd.Series(outcomes), cal_probs),
            "n": len(split_df),
            "calibration": calibration_summary(cal_probs, outcomes.astype(float)),
            "roi": simulate_moneyline_roi(cal_probs, market_novig, outcomes, home_ml_odds),
        }
    except FileNotFoundError as exc:
        print(f"  [{split_name}] Production moneyline model not found – {exc}")
    except Exception as exc:
        print(f"  [{split_name}] Error running production moneyline model: {exc}")

    # ----------------------------------------------------------------
    # Experimental market-residual moneyline
    # ----------------------------------------------------------------
    if EXP_ML_PATH.exists():
        try:
            exp_artifact = joblib.load(EXP_ML_PATH)
            exp_pipeline = exp_artifact["pipeline"]
            exp_meta     = exp_artifact["metadata"]
            exp_blend_w  = float(exp_meta.get("blend_weight_model", 1.0))

            # Add market features the experimental model expects
            prepared_exp = ensure_feature_frame(split_df)
            prepared_exp = add_vig_removed_prob(prepared_exp)
            prepared_exp["market_home_win_prob_novig"] = prepared_exp["market_home_win_prob_novig"]

            exp_features = exp_meta.get("numeric_features", []) + exp_meta.get("categorical_features", [])
            for col in exp_features:
                if col not in prepared_exp.columns:
                    prepared_exp[col] = 0.0
            feat_exp = prepared_exp[exp_features].copy()

            exp_raw    = exp_pipeline.predict_proba(feat_exp)[:, 1]
            exp_heur   = prepared_exp["heuristic_home_win_prob"].to_numpy()
            exp_blend  = blend_probabilities(exp_raw, exp_heur, exp_blend_w)

            home_ml_odds = split_enriched["home_moneyline"].to_numpy() if "home_moneyline" in split_enriched.columns else np.full(len(split_df), np.nan)

            results["exp_ml_residual"] = {
                **evaluate_probabilities(pd.Series(outcomes), exp_blend),
                "n": len(split_df),
                "calibration": calibration_summary(exp_blend, outcomes.astype(float)),
                "roi": simulate_moneyline_roi(exp_blend, market_novig, outcomes, home_ml_odds),
            }
        except Exception as exc:
            print(f"  [{split_name}] Error running experimental ML model: {exc}")

    return results


def evaluate_split_totals(
    split_df: pd.DataFrame,
    split_name: str,
) -> dict[str, dict[str, Any]]:
    """Evaluate all available totals models on ``split_df``."""
    if split_df.empty:
        print(f"  [{split_name}] empty – skipping totals evaluation.")
        return {}

    results: dict[str, dict[str, Any]] = {}
    actual_runs = split_df["total_runs"].to_numpy(dtype=float)

    # ----------------------------------------------------------------
    # Market totals enrichment (best-effort via odds archive)
    # ----------------------------------------------------------------
    print(f"  [{split_name}] Fetching market total lines from odds archive…")
    split_enriched = enrich_with_market_totals(split_df)
    market_lines   = split_enriched["market_total_line"].to_numpy()
    over_odds_arr  = split_enriched["market_over_odds"].fillna(DEFAULT_TOTALS_JUICE).to_numpy()
    under_odds_arr = split_enriched["market_under_odds"].fillna(DEFAULT_TOTALS_JUICE).to_numpy()

    market_coverage = int((~np.isnan(market_lines)).sum())
    print(f"  [{split_name}] Market total-line coverage: {market_coverage}/{len(split_df)}")

    # Market-only baseline
    if market_coverage >= 20:
        valid = ~np.isnan(market_lines) & ~np.isnan(actual_runs)
        results["market_only"] = {
            **evaluate_totals(
                pd.Series(actual_runs[valid]), market_lines[valid]
            ),
            "n": int(valid.sum()),
            "roi": {"bets": 0, "wins": 0, "roi_pct": float("nan"), "note": "market IS the baseline"},
        }

    # ----------------------------------------------------------------
    # Production totals model
    # ----------------------------------------------------------------
    try:
        prod_artifact  = joblib.load(PROD_TOTALS_PATH)
        prod_pipeline  = prod_artifact["pipeline"]
        prod_meta      = prod_artifact["metadata"]
        blend_w        = float(prod_meta.get("blend_weight_model", 0.65))
        from totals_model import blend_totals, INFERENCE_BLEND_WEIGHT_MODEL
        blend_w = max(INFERENCE_BLEND_WEIGHT_MODEL, blend_w)

        prepared   = add_totals_features(split_df)
        feat_x     = select_totals_feature_frame(prepared)
        raw_pred   = prod_pipeline.predict(feat_x)
        heur_pred  = prepared["heuristic_total_runs"].to_numpy()
        blended    = blend_totals(raw_pred, heur_pred, blend_w)

        results["prod_totals"] = {
            **evaluate_totals(pd.Series(actual_runs), blended),
            "n": len(split_df),
            "roi": simulate_totals_roi(blended, market_lines, actual_runs, over_odds_arr, under_odds_arr),
        }
    except FileNotFoundError as exc:
        print(f"  [{split_name}] Production totals model not found – {exc}")
    except Exception as exc:
        print(f"  [{split_name}] Error running production totals model: {exc}")

    # ----------------------------------------------------------------
    # Experimental market-residual totals
    # ----------------------------------------------------------------
    if EXP_TOTALS_PATH.exists():
        try:
            exp_artifact  = joblib.load(EXP_TOTALS_PATH)
            exp_pipeline  = exp_artifact["pipeline"]
            exp_meta      = exp_artifact["metadata"]
            exp_blend_w   = float(exp_meta.get("blend_weight_model", 0.65))

            exp_features  = exp_meta.get("numeric_features", []) + exp_meta.get("categorical_features", [])
            prepared_exp  = add_totals_features(split_enriched)
            for col in exp_features:
                if col not in prepared_exp.columns:
                    prepared_exp[col] = 0.0
            feat_exp      = prepared_exp[exp_features].copy()

            # Model predicts residual; reconstruct total
            residual_pred  = exp_pipeline.predict(feat_exp)
            ml_lines_safe  = np.where(np.isnan(market_lines), prepared_exp["heuristic_total_runs"].to_numpy(), market_lines)
            exp_total_pred = ml_lines_safe + residual_pred

            heur_pred_exp  = prepared_exp["heuristic_total_runs"].to_numpy()
            exp_blended    = blend_totals(exp_total_pred, heur_pred_exp, exp_blend_w)

            results["exp_totals_residual"] = {
                **evaluate_totals(pd.Series(actual_runs), exp_blended),
                "n": len(split_df),
                "roi": simulate_totals_roi(exp_blended, market_lines, actual_runs, over_odds_arr, under_odds_arr),
            }
        except Exception as exc:
            print(f"  [{split_name}] Error running experimental totals model: {exc}")

    return results


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------

def _fmt_ml_row(label: str, r: dict[str, Any]) -> str:
    acc   = f"{r.get('accuracy', float('nan')):.3f}"
    ll    = f"{r.get('log_loss', float('nan')):.4f}"
    brier = f"{r.get('brier_score', float('nan')):.4f}"
    roi   = r.get("roi", {})
    roi_s = f"{roi.get('roi_pct', float('nan')):+.1f}% ({roi.get('bets',0)} bets)"
    n     = r.get("n", "?")
    cal   = r.get("calibration", "")
    return f"  {label:<28} n={n:<5}  acc={acc}  ll={ll}  brier={brier}  roi={roi_s}\n  {' '*28} cal={cal}"


def _fmt_tot_row(label: str, r: dict[str, Any]) -> str:
    rmse  = f"{r.get('rmse', float('nan')):.3f}"
    mae   = f"{r.get('mae',  float('nan')):.3f}"
    roi   = r.get("roi", {})
    roi_s = f"{roi.get('roi_pct', float('nan')):+.1f}% ({roi.get('bets',0)} bets)"
    n     = r.get("n", "?")
    return f"  {label:<28} n={n:<5}  rmse={rmse}  mae={mae}  roi={roi_s}"


def print_results_table(
    moneyline_results: dict[str, dict[str, dict[str, Any]]],
    totals_results:    dict[str, dict[str, dict[str, Any]]],
) -> None:
    """Print side-by-side comparison tables."""
    sep = "=" * 80

    print(f"\n{sep}")
    print("MONEYLINE  –  accuracy | log-loss | Brier | simulated ROI")
    print(sep)
    for split_name in ("val", "test"):
        ml_r = moneyline_results.get(split_name, {})
        print(f"\n  [{split_name.upper()}]")
        if not ml_r:
            print("    (no results)")
            continue
        for label, r in ml_r.items():
            print(_fmt_ml_row(label, r))

    print(f"\n{sep}")
    print("TOTALS  –  RMSE | MAE | simulated ROI")
    print(sep)
    for split_name in ("val", "test"):
        tot_r = totals_results.get(split_name, {})
        print(f"\n  [{split_name.upper()}]")
        if not tot_r:
            print("    (no results)")
            continue
        for label, r in tot_r.items():
            print(_fmt_tot_row(label, r))

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== MLB Model Evaluation ===\n")

    try:
        splits = load_splits()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print("Split summary:")
    print(split_summary(splits))
    print()

    ml_results:  dict[str, dict[str, Any]] = {}
    tot_results: dict[str, dict[str, Any]] = {}

    for split_name in ("val", "test"):
        df = splits[split_name]
        print(f"--- Evaluating {split_name.upper()} ({len(df)} rows) ---")

        print(f"  Moneyline…")
        ml_results[split_name] = evaluate_split_moneyline(df, split_name)

        print(f"  Totals…")
        tot_results[split_name] = evaluate_split_totals(df, split_name)
        print()

    print_results_table(ml_results, tot_results)

    # Remind the user about experimental artifacts
    for label, path in [
        ("moneyline market-residual", EXP_ML_PATH),
        ("totals market-residual",    EXP_TOTALS_PATH),
    ]:
        if not path.exists():
            print(f"  [info] Experimental {label} artifact not found at {path}.")
            print(f"         Run the corresponding train_*_market_residual.py script to generate it.\n")


if __name__ == "__main__":
    main()
