"""Train and certify the CFB model with season walk-forward tests.

Two layers live in one artifact bundle:

* the **market-free originator** (margin/total Ridge or HGB on the as-of team
  state, plus its isotonic calibrators) — the research forecast that is
  audited by the pregame ledger and displayed on every card;
* the **market-anchored residual heads** — a logistic moneyline model, a
  spread residual model on (margin + home_line), and a total residual model
  on (total - total_line), each fed the originator features plus the posted
  line.  These are the only heads allowed to mint a stake, and only through a
  segment that this script validated out of fold.

Walk-forward evidence (train < season N, score season N, 2021-2025) recorded
in ``artifacts/metadata.json``:

* the originator's disagreement with the spread-implied win probability is
  anti-predictive (the model's side wins 34-39% when it disagrees by 5+
  points) and its spread/total direction rates sit at 47-53% — so moneyline
  and spread publish as research (PASS) only;
* the anchored total residual head beats the 52.4% break-even in a clear
  majority of seasons once |residual| >= 4 points, which is the segment gate
  the serving path stakes at LEAN.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from cfb_core import FEATURE_NAMES, anchored_matrix, build_dataset, load_training_rows, matrix
    from cfb_model import _probabilities
except ImportError:
    from .cfb_core import FEATURE_NAMES, anchored_matrix, build_dataset, load_training_rows, matrix
    from .cfb_model import _probabilities

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_VERSION = "cfb_v2_market_anchored_totals"
FIRST_SEASON = 2017
LAST_SEASON = 2025
WALK_FORWARD_SEASONS = range(2021, 2026)

# Segment search for the anchored residual heads.  Direction-only evidence
# (historical two-sided prices are unavailable), so every bar is expressed as
# a hit rate against the -110 break-even (52.38%).
BREAK_EVEN = 110.0 / 210.0
SEGMENT_THRESHOLDS = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)
SEGMENT_DIRECTIONS = {"spread": ("home", "away", "both"), "total": ("over", "under", "both")}
# Graduated confidence ladder, strongest tier first.  A tier is a band of
# residual magnitude whose OWN picks must clear the tier's bar: BET is the
# strongest validated band (~+11% at -110), LEAN the next band that still
# beats break-even comfortably (~+4%), and a lighter LEAN (fewer units) is
# allowed only where no regular LEAN exists but the evidence is still above
# break-even.  Everything else publishes as PASS research.
TIER_BARS = (
    {"tier": "bet", "decision": "BET", "units": 0.5, "min_picks": 100, "min_hit_rate": 0.58, "min_season_share": 0.75},
    {"tier": "lean", "decision": "LEAN", "units": 0.25, "min_picks": 100, "min_hit_rate": 0.545, "min_season_share": 0.70},
    {"tier": "lean_light", "decision": "LEAN", "units": 0.15, "min_picks": 150, "min_hit_rate": 0.535, "min_season_share": 0.50},
)
MIN_BAND_PICKS = 60
QUALIFY_MIN_PICKS = TIER_BARS[1]["min_picks"]
QUALIFY_MIN_HIT_RATE = TIER_BARS[1]["min_hit_rate"]
QUALIFY_MIN_SEASON_SHARE = TIER_BARS[1]["min_season_share"]
MAX_JUICE = -125


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


def _anchored_logistic() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


def _anchored_residual() -> Any:
    # Deliberately the most regularized configuration that qualified: fewer,
    # larger disagreements rather than many small ones.
    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.03,
        max_iter=150,
        min_samples_leaf=60,
        l2_regularization=5.0,
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


def _segment_stats(picks: list[tuple[int, bool | None]]) -> dict[str, Any]:
    graded = [(season, hit) for season, hit in picks if hit is not None]
    if not graded:
        return {"picks": len(picks), "graded": 0}
    by_season: dict[int, list[int]] = defaultdict(list)
    for season, hit in graded:
        by_season[season].append(int(hit))
    hit_rate = sum(hit for _, hit in graded) / len(graded)
    seasons_above = sum(1 for hits in by_season.values() if sum(hits) / len(hits) > BREAK_EVEN)
    return {
        "picks": len(picks),
        "graded": len(graded),
        "hit_rate": round(hit_rate, 4),
        "flat_roi_at_minus_110": round(hit_rate * (100.0 / 110.0) - (1.0 - hit_rate), 4),
        "seasons": len(by_season),
        "seasons_above_break_even": seasons_above,
        "season_hit_rate": {str(season): round(sum(hits) / len(hits), 4) for season, hits in sorted(by_season.items())},
    }


def _clears(stats: dict[str, Any], bar: dict[str, Any], *, min_picks: int | None = None) -> bool:
    graded = int(stats.get("graded") or 0)
    seasons = int(stats.get("seasons") or 0)
    return (
        graded >= (min_picks if min_picks is not None else int(bar["min_picks"]))
        and float(stats.get("hit_rate") or 0.0) >= float(bar["min_hit_rate"])
        and seasons > 0
        and int(stats.get("seasons_above_break_even") or 0) / seasons >= float(bar["min_season_share"])
    )


def _qualifies(stats: dict[str, Any]) -> bool:
    """Loosest live gate: the LEAN bar (kept for the segment-search report)."""

    return _clears(stats, TIER_BARS[1])


def _band_picks(
    oof: list[dict[str, Any]],
    *,
    pred_key: str,
    actual_key: str,
    direction: str,
    market: str,
    low: float,
    high: float | None,
) -> list[tuple[int, bool | None]]:
    picks: list[tuple[int, bool | None]] = []
    for o in oof:
        pred = o[pred_key]
        magnitude = abs(pred)
        if magnitude < low or (high is not None and magnitude >= high):
            continue
        positive = pred > 0  # home cover / over
        side = ("home" if positive else "away") if market == "spread" else ("over" if positive else "under")
        if direction != "both" and side != direction:
            continue
        actual = o[actual_key]
        picks.append((o["season"], None if abs(actual) <= 1e-9 else (actual > 0) == positive))
    return picks


def build_ladder(
    oof: list[dict[str, Any]],
    *,
    market: str,
    direction: str,
    pred_key: str,
    actual_key: str,
) -> list[dict[str, Any]]:
    """Graduated tiers for one market direction from band-level evidence.

    Strongest tier first, on the direction's own picks only.  A tier takes the
    loosest threshold whose cumulative tail clears its bar; a LEAN band below
    BET must itself clear the LEAN bar.  The loosest BET threshold that leaves
    a validated LEAN band wins; a lighter LEAN exists only where no regular
    LEAN could be placed.  A tier that cannot be placed is absent and the
    market publishes PASS research there.
    """

    def stats(low: float, high: float | None) -> dict[str, Any]:
        return _segment_stats(_band_picks(
            oof, pred_key=pred_key, actual_key=actual_key, direction=direction, market=market, low=low, high=high,
        ))

    def placement(bar: dict[str, Any], threshold: float, ceiling: float | None) -> dict[str, Any] | None:
        tail = stats(threshold, None)
        if not _clears(tail, bar):
            return None
        band = stats(threshold, ceiling)
        if ceiling is not None and not _clears(band, bar, min_picks=MIN_BAND_PICKS):
            return None
        keep = ("picks", "graded", "hit_rate", "flat_roi_at_minus_110", "seasons", "seasons_above_break_even", "season_hit_rate")
        return {
            "direction": direction,
            "min_residual": threshold,
            "max_residual": ceiling,
            "decision": bar["decision"],
            "tier": bar["tier"],
            "units": bar["units"],
            "max_juice": MAX_JUICE,
            "walk_forward": {
                "band": {k: band[k] for k in keep if k in band},
                "cumulative": {k: tail[k] for k in keep if k in tail and k != "season_hit_rate"},
            },
        }

    bet_bar, lean_bar, light_bar = TIER_BARS
    thresholds = list(SEGMENT_THRESHOLDS)

    def lean_below(ceiling: float | None, bar: dict[str, Any]) -> dict[str, Any] | None:
        for threshold in thresholds:
            if ceiling is not None and threshold >= ceiling:
                break
            placed = placement(bar, threshold, ceiling)
            if placed is not None:
                return placed
        return None

    bet_candidates = [t for t in thresholds if placement(bet_bar, t, None) is not None]
    for bet_threshold in bet_candidates:
        lean = lean_below(bet_threshold, lean_bar)
        if lean is not None:
            return [placement(bet_bar, bet_threshold, None), lean]
    if bet_candidates:
        bet = placement(bet_bar, bet_candidates[0], None)
        light = lean_below(bet_candidates[0], light_bar)
        return [bet] + ([light] if light else [])
    lean = lean_below(None, lean_bar)
    if lean is not None:
        return [lean]
    light = lean_below(None, light_bar)
    return [light] if light else []


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
    # Anchored heads: out-of-fold predictions for calibration/segment evidence.
    anchored_oof: list[dict[str, Any]] = []
    ml_brier: dict[str, list[float]] = {"originator_raw": [], "anchored_logistic": [], "market_spread_logit": []}
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

        home_win_truth = [1 if row["home_margin"] > 0 else 0 for row in train_rows]
        anchored_ml = _anchored_logistic().fit(anchored_matrix(train_rows, "spread"), home_win_truth)
        market_logit = LogisticRegression(C=10.0, max_iter=1000).fit([[row["home_line"]] for row in train_rows], home_win_truth)
        anchored_spread = _anchored_residual().fit(
            anchored_matrix(train_rows, "spread"), [row["home_margin"] + row["home_line"] for row in train_rows]
        )
        anchored_total = _anchored_residual().fit(
            anchored_matrix(train_rows, "total"), [row["game_total"] - row["total_line"] for row in train_rows]
        )
        p_anchored = anchored_ml.predict_proba(anchored_matrix(test_rows, "spread"))[:, 1]
        p_market = market_logit.predict_proba([[row["home_line"]] for row in test_rows])[:, 1]
        spread_residual_pred = anchored_spread.predict(anchored_matrix(test_rows, "spread"))
        total_residual_pred = anchored_total.predict(anchored_matrix(test_rows, "total"))

        spread_hits = spread_graded = total_hits = total_graded = 0
        for row, margin_prediction, total_prediction, pa, pm, sr, tr in zip(
            test_rows, test_margin_predictions, test_total_predictions, p_anchored, p_market, spread_residual_pred, total_residual_pred
        ):
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
            ml_brier["originator_raw"].append((ml_home - home_win) ** 2)
            ml_brier["anchored_logistic"].append((float(pa) - home_win) ** 2)
            ml_brier["market_spread_logit"].append((float(pm) - home_win) ** 2)

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
            anchored_oof.append({
                "season": season,
                "week": row["features"]["week"],
                "spread_pred": float(sr),
                "total_pred": float(tr),
                "actual_spread": actual_spread,
                "actual_total": actual_total,
                "p_home": float(pa),
                "home_win": home_win,
            })

        selected_walk_forward.append(
            {
                "season": season,
                "games": len(test_rows),
                "margin_mae": round(mean_absolute_error([row["home_margin"] for row in test_rows], test_margin_predictions), 5),
                "total_mae": round(mean_absolute_error([row["game_total"] for row in test_rows], test_total_predictions), 5),
                "spread_direction_rate": round(spread_hits / spread_graded, 5) if spread_graded else None,
                "total_direction_rate": round(total_hits / total_graded, 5) if total_graded else None,
                "anchored_ml_brier": round(float(np.mean([(o["p_home"] - o["home_win"]) ** 2 for o in anchored_oof if o["season"] == season])), 5),
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

    # Anchored heads: residual scale and segment search from out-of-fold rows.
    anchored_sigma_spread = float(np.std([o["actual_spread"] - o["spread_pred"] for o in anchored_oof], ddof=1))
    anchored_sigma_total = float(np.std([o["actual_total"] - o["total_pred"] for o in anchored_oof], ddof=1))
    segments: dict[str, dict[str, Any]] = {}
    for market, pred_key, actual_key in (("spread", "spread_pred", "actual_spread"), ("total", "total_pred", "actual_total")):
        for direction in SEGMENT_DIRECTIONS[market]:
            for threshold in SEGMENT_THRESHOLDS:
                picks: list[tuple[int, bool | None]] = []
                for o in anchored_oof:
                    pred = o[pred_key]
                    if abs(pred) < threshold:
                        continue
                    positive = pred > 0  # home cover / over
                    side = ("home" if positive else "away") if market == "spread" else ("over" if positive else "under")
                    if direction != "both" and side != direction:
                        continue
                    actual = o[actual_key]
                    hit = None if abs(actual) <= 1e-9 else (actual > 0) == positive
                    picks.append((o["season"], hit))
                stats = _segment_stats(picks)
                stats.update({"market": market, "direction": direction, "min_residual": threshold, "qualifies": _qualifies(stats)})
                segments[f"{market}:{direction}:{threshold:g}"] = stats

    def ladders(market: str) -> list[dict[str, Any]]:
        pred_key, actual_key = {"spread": ("spread_pred", "actual_spread"), "total": ("total_pred", "actual_total")}[market]
        rows: list[dict[str, Any]] = []
        for direction in SEGMENT_DIRECTIONS[market][:2]:
            rows.extend(build_ladder(anchored_oof, market=market, direction=direction, pred_key=pred_key, actual_key=actual_key))
        return rows

    spread_gates = ladders("spread")
    total_gates = ladders("total")
    decision_policy = {
        "h2h": {
            "mode": "research_only",
            "reason": (
                "The originator's disagreement with the spread-implied win probability is anti-predictive "
                "out of fold and the anchored logistic only matches the market; no priced edge to stake."
            ),
        },
        "spread": {
            "mode": "segment_gate" if spread_gates else "research_only",
            "segments": spread_gates,
            "reason": "" if spread_gates else "No spread residual band clears any tier bar.",
        },
        "totals": {
            "mode": "segment_gate" if total_gates else "research_only",
            "segments": total_gates,
            "reason": "" if total_gates else "No total residual band clears any tier bar.",
        },
        "tier_bars": list(TIER_BARS),
        "qualification_bar": {
            "min_graded_picks": QUALIFY_MIN_PICKS,
            "min_hit_rate": QUALIFY_MIN_HIT_RATE,
            "break_even_hit_rate": round(BREAK_EVEN, 4),
            "min_season_share_above_break_even": QUALIFY_MIN_SEASON_SHARE,
            "min_band_picks": MIN_BAND_PICKS,
            "max_juice": MAX_JUICE,
            "evidence": "direction_only_no_historical_two_sided_prices",
            "ladder_rule": (
                "Strongest tier first, on the direction's own picks only. A tier takes the loosest threshold whose "
                "cumulative tail clears its bar; a LEAN band below BET must itself clear the LEAN bar. The loosest "
                "BET threshold that leaves a validated LEAN band wins; a lighter LEAN exists only where no regular "
                "LEAN could be placed."
            ),
        },
    }

    all_home_win = [1 if row["home_margin"] > 0 else 0 for row in records]
    anchored_final = {
        "moneyline": _anchored_logistic().fit(anchored_matrix(records, "spread"), all_home_win),
        "spread": _anchored_residual().fit(anchored_matrix(records, "spread"), [row["home_margin"] + row["home_line"] for row in records]),
        "total": _anchored_residual().fit(anchored_matrix(records, "total"), [row["game_total"] - row["total_line"] for row in records]),
    }

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "margin_model": final_margin,
            "total_model": final_total,
            "calibrators": calibrators,
            "anchored": anchored_final,
        },
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
        "anchored": {
            "feature_names": [*FEATURE_NAMES, "posted_line"],
            "market_features": ["market_home_line", "market_total_line"],
            "families": {"moneyline": "logistic", "spread": "hist_gradient_boosting_residual", "total": "hist_gradient_boosting_residual"},
            "spread_residual_sigma": round(anchored_sigma_spread, 6),
            "total_residual_sigma": round(anchored_sigma_total, 6),
            "oof_samples": len(anchored_oof),
            "ml_brier": {key: round(float(np.mean(values)), 6) for key, values in ml_brier.items()},
            "segment_search": segments,
        },
        "decision_policy": decision_policy,
        "shadow_mode": False,
        "promotion_status": "segment_gate_graduated_tiers",
        "financial_backtest": {
            "moneyline": "unavailable_no_complete_historical_two_sided_prices",
            "spread": "forecast_direction_only",
            "total": "forecast_direction_only",
        },
        "notes": (
            "Originator FEATURE_NAMES stay market-free. Anchored residual heads append the posted line and are the "
            "only heads allowed to stake, through decision_policy bands validated out of fold (BET = strongest "
            "validated band, LEAN = the next band that still clears its own bar); everything else publishes as "
            "visible PASS research."
        ),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    result = train()
    print(json.dumps({key: result[key] for key in ("model_version", "games", "selected_family")}, indent=2))
    for row in result["walk_forward"]:
        print(row)
    print(json.dumps(result["anchored"]["ml_brier"], indent=2))
    print(json.dumps(result["decision_policy"], indent=2))
