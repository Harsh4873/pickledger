"""Train and certify the CFB market-residual totals model.

The trainer does not just fit -- it EARNS a betting threshold. It runs a strict
season walk-forward, then for each candidate confidence threshold measures the
out-of-sample hit rate, tests it against the -110 break-even (not merely against
a coin flip), and checks per-season consistency. Only a threshold that is both
statistically significant and consistent across seasons is certified; if none
qualifies the model ships unpromoted and publishes nothing.

Run:
    python -m CFBTotalsModel.cfb_totals_train
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

from CFBTotalsModel.cfb_totals_core import (
    FEATURE_NAMES,
    build_dataset,
    load_training_rows,
    matrix,
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_VERSION = "cfb_totals_residual_v1"
FIRST_SEASON = 2017
LAST_SEASON = 2025
WALK_FORWARD_SEASONS = range(2021, 2026)

BREAK_EVEN = 0.5238            # -110 two-way
CANDIDATE_THRESHOLDS = (2.0, 3.0, 4.0, 5.0, 6.0)
MIN_SAMPLE = 200               # graded picks needed to take a threshold seriously
MIN_SEASONS_BEATING = 4        # of 5 -- guards against one lucky year
ALPHA = 0.05
MIN_TRAINING_GAMES = 4500
RANDOM_STATE = 17


def _model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_depth=3, max_iter=300, learning_rate=0.04,
        l2_regularization=8.0, min_samples_leaf=40, random_state=RANDOM_STATE,
    )


def _roi(hits: int, n: int) -> float:
    """Units returned per unit staked at -110."""
    return (hits * (100.0 / 110.0) - (n - hits)) / n if n else 0.0


def _certify(oos: list[tuple[int, float, float]], label: str) -> dict[str, Any]:
    """Score every candidate threshold for one market and pick a qualifying one.

    Certification is deliberately strict: a threshold must have a real sample,
    beat the -110 break-even at ALPHA, and do so in most individual seasons. A
    market that cannot clear this returns selected=None, which keeps it
    unactionable rather than letting it publish a coin flip.
    """
    report: list[dict[str, Any]] = []
    for threshold in CANDIDATE_THRESHOLDS:
        picked = [(s, p, a) for s, p, a in oos if abs(p) >= threshold]
        if not picked:
            continue
        hits = sum((p > 0) == (a > 0) for _, p, a in picked)
        n = len(picked)
        p_value = float(stats.binomtest(hits, n, BREAK_EVEN, alternative="greater").pvalue)
        seasons: dict[str, Any] = {}
        for season in WALK_FORWARD_SEASONS:
            rows = [(p, a) for s, p, a in picked if s == season]
            if rows:
                sh = sum((p > 0) == (a > 0) for p, a in rows)
                seasons[str(season)] = {
                    "n": len(rows), "hits": sh, "rate": round(sh / len(rows), 5),
                    "beats_break_even": (sh / len(rows)) > BREAK_EVEN,
                }
        beating = sum(1 for v in seasons.values() if v["beats_break_even"])
        report.append({
            "threshold": threshold, "n": n, "hits": hits, "rate": round(hits / n, 5),
            "p_value_vs_break_even": round(p_value, 6),
            "roi_at_minus_110": round(_roi(hits, n), 5),
            "seasons_beating_break_even": beating,
            "seasons_evaluated": len(seasons),
            "per_season": seasons,
            "qualifies": bool(
                n >= MIN_SAMPLE and p_value < ALPHA and beating >= MIN_SEASONS_BEATING
            ),
        })
    qualifying = [t for t in report if t["qualifies"]]
    selected = max(qualifying, key=lambda t: t["n"]) if qualifying else None
    return {
        "market": label,
        "candidates": report,
        "selected_threshold": selected["threshold"] if selected else None,
        "selected_evidence": (
            {k: selected[k] for k in ("n", "rate", "p_value_vs_break_even",
                                      "roi_at_minus_110", "seasons_beating_break_even")}
            if selected else None
        ),
    }


def train(first_season: int = FIRST_SEASON, last_season: int = LAST_SEASON) -> dict[str, Any]:
    records = build_dataset(load_training_rows(first_season, last_season))
    if len(records) < MIN_TRAINING_GAMES:
        raise SystemExit(
            f"CFB totals dataset too small ({len(records)} priced FBS games); refusing to train"
        )

    # ---- strict walk-forward: predict each season from earlier seasons only ----
    oos_total: list[tuple[int, float, float]] = []
    oos_margin: list[tuple[int, float, float]] = []
    per_season_mae: list[dict[str, Any]] = []
    for season in WALK_FORWARD_SEASONS:
        if season > last_season:
            continue
        train_rows = [r for r in records if r["season"] < season]
        test_rows = [r for r in records if r["season"] == season]
        if len(train_rows) < 1000 or not test_rows:
            continue
        design_train, design_test = matrix(train_rows), matrix(test_rows)
        total_fit = _model().fit(design_train, [r["total_residual"] for r in train_rows])
        margin_fit = _model().fit(design_train, [r["margin_residual"] for r in train_rows])
        total_pred = total_fit.predict(design_test)
        margin_pred = margin_fit.predict(design_test)
        total_err, margin_err = [], []
        for row, t_pred, m_pred in zip(test_rows, total_pred, margin_pred):
            t_actual, m_actual = float(row["total_residual"]), float(row["margin_residual"])
            total_err.append(abs(t_actual - float(t_pred)))
            margin_err.append(abs(m_actual - float(m_pred)))
            if abs(t_actual) > 1e-9:      # landing on the number is a push
                oos_total.append((season, float(t_pred), t_actual))
            if abs(m_actual) > 1e-9:
                oos_margin.append((season, float(m_pred), m_actual))
        per_season_mae.append({
            "season": season,
            "games": len(test_rows),
            "total_residual_mae": round(float(np.mean(total_err)), 5),
            "margin_residual_mae": round(float(np.mean(margin_err)), 5),
        })

    totals_cert = _certify(oos_total, "totals")
    spread_cert = _certify(oos_margin, "spread")

    # ---- final fits on everything, plus residual spreads for probability ----
    design_all = matrix(records)
    final_total = _model().fit(design_all, [r["total_residual"] for r in records])
    final_margin = _model().fit(design_all, [r["margin_residual"] for r in records])
    total_sigma = float(np.std(
        np.array([r["total_residual"] for r in records]) - final_total.predict(design_all), ddof=1))
    margin_sigma = float(np.std(
        np.array([r["margin_residual"] for r in records]) - final_margin.predict(design_all), ddof=1))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "total_residual_model": final_total,
        "margin_residual_model": final_margin,
        "feature_names": FEATURE_NAMES,
    }, ARTIFACT_DIR / "cfb_totals_model.joblib")

    selected = totals_cert["selected_threshold"]
    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model_class": "market_residual",
        "targets": {
            "totals": "game_total - posted_total_line",
            "spread": "home_margin + posted_home_line",
        },
        "data_source": "SportsDataverse ESPN-derived schedules + resolved betting releases",
        "train_window": [first_season, last_season],
        "games": len(records),
        "population": "FBS-vs-FBS games with posted spread and total",
        "feature_names": FEATURE_NAMES,
        "market_features": ["home_line", "total_line"],
        "market_free": False,
        "market_free_note": (
            "This model intentionally consumes the posted line. It is a market-residual "
            "model, NOT an originator, and must not be compared to CFBPredictionModel's "
            "market-free contract."
        ),
        "walk_forward": per_season_mae,
        "residual_sigma": round(total_sigma, 5),
        "margin_residual_sigma": round(margin_sigma, 5),
        "break_even": BREAK_EVEN,
        "threshold_certification": totals_cert,
        "spread_certification": spread_cert,
        "market_status": {
            "totals": "certified" if totals_cert["selected_threshold"] else "uncertified",
            "spread": "certified" if spread_cert["selected_threshold"] else "uncertified",
            "moneyline": "uncertified_derived_from_margin",
        },
        "criteria": {
            "min_sample": MIN_SAMPLE,
            "alpha_vs_break_even": ALPHA,
            "min_seasons_beating_break_even": f"{MIN_SEASONS_BEATING}/{len(list(WALK_FORWARD_SEASONS))}",
        },
        "promotion_status": "qualified_totals_threshold" if selected else "not_qualified",
        "shadow_mode": selected is None,
        "notes": (
            "Only picks whose |predicted deviation| clears their market's CERTIFIED "
            "threshold are actionable; everything else is PASS. Spread and moneyline are "
            "published for visibility but are marked uncertified unless their own "
            "certification qualified -- do not treat an uncertified row as a bet."
        ),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


if __name__ == "__main__":
    meta = train()
    print(json.dumps({
        "model_version": meta["model_version"],
        "games": meta["games"],
        "market_status": meta["market_status"],
        "promotion_status": meta["promotion_status"],
    }, indent=2))
    for cert_key in ("threshold_certification", "spread_certification"):
        cert = meta[cert_key]
        print(f"\n-- {cert['market']} (selected={cert['selected_threshold']}) --")
        for row in cert["candidates"]:
            print(f"  |dev|>={row['threshold']}: n={row['n']:5d} rate={row['rate']:.4f} "
                  f"p={row['p_value_vs_break_even']:.4f} roi={row['roi_at_minus_110']:+.4f} "
                  f"seasons={row['seasons_beating_break_even']}/{row['seasons_evaluated']} "
                  f"qualifies={row['qualifies']}")
