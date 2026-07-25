"""Train, tune, calibrate and export the tennis model.

Protocol — every number this produces is out-of-sample, and the split is fixed
before anything is fitted:

* **2000-2011  burn-in.** Replayed to warm the ratings; no rows emitted. Elo
  needs history before it says anything useful, and a player's first matches
  move their rating enormously.
* **2012-2021  development.** Walk-forward by season (train on everything
  strictly earlier, score the season) produces the out-of-fold predictions used
  to pick the Elo hyper-parameters, choose the model family, fit the isotonic
  calibrator and fit the market-combination weights.
* **2022-present  held out.** Touched once, at the end, by the same
  walk-forward loop. These are the numbers reported as the model's accuracy,
  and they are directly comparable to the published tennis benchmarks, which
  test on the same window.

The trained model deliberately never sees a betting price. A model with the
market as an input converges to the market and can only track it; keeping it
market-free means the comparison against the closing line is a real test of
whether it knows anything the market does not. The two are combined afterwards,
with weights fitted out-of-sample — which is also a formal encompassing test:
if the model's weight collapses to zero, it adds nothing.

Run offline (never in the daily cron):

    .venv/bin/python -m TennisPredictionModel.tennis_train --refresh
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .tennis_core import (
    ARTIFACT_DIR,
    FEATURE_NAMES,
    EloConfig,
    RatingEngine,
    elo_probability,
    matrix,
    mirror_vector,
    symmetrise,
)
from .tennis_backtest import BACKTEST_PATH, confidence_curve, simulate, summarise_backtest
from .tennis_data import PROCESSED_DIR, Match, build_spine, read_spine
from .tennis_infer import MODEL_PATH, TennisModel
from .tennis_model import TOURNAMENT_INDEX_PATH, build_tournament_index

# The family suffix is appended at train time from whatever walk-forward
# selection actually picked, so the version string can never claim a model that
# was not shipped.
MODEL_VERSION = "tennis_v1_welo"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

BURN_IN_THROUGH = 2011
DEV_SEASONS = range(2012, 2022)
TEST_SEASONS = range(2022, 2027)
RANDOM_SEED = 17


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def log_loss(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    total = 0.0
    for probability, label in zip(probabilities, labels):
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        total -= math.log(clipped) if label else math.log(1.0 - clipped)
    return total / max(1, len(labels))


def brier(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / max(1, len(labels))


def accuracy(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum(1 for p, y in zip(probabilities, labels) if (p >= 0.5) == bool(y)) / max(1, len(labels))


def calibration_table(probabilities: Sequence[float], labels: Sequence[int], bins: int = 10) -> list[dict[str, Any]]:
    """Reliability table: predicted vs realised win rate, by probability decile."""
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels):
        index = min(bins - 1, max(0, int(probability * bins)))
        buckets[index].append((probability, label))
    table: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        table.append({
            "bin": f"{index / bins:.1f}-{(index + 1) / bins:.1f}",
            "n": len(bucket),
            "predicted": round(sum(p for p, _ in bucket) / len(bucket), 4),
            "actual": round(sum(y for _, y in bucket) / len(bucket), 4),
        })
    return table


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], bins: int = 20) -> float:
    table = calibration_table(probabilities, labels, bins=bins)
    total = sum(row["n"] for row in table) or 1
    return sum(row["n"] * abs(row["predicted"] - row["actual"]) for row in table) / total


def summarise(probabilities: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    return {
        "n": len(labels),
        "log_loss": round(log_loss(probabilities, labels), 5),
        "brier": round(brier(probabilities, labels), 5),
        "accuracy": round(accuracy(probabilities, labels), 5),
        "ece": round(expected_calibration_error(probabilities, labels), 5),
    }


# --------------------------------------------------------------------------
# market helpers
# --------------------------------------------------------------------------


def no_vig(price_a: float, price_b: float) -> float:
    """Proportional de-vig of a two-way decimal market: fair P(side A)."""
    implied_a = 1.0 / price_a
    implied_b = 1.0 / price_b
    return implied_a / (implied_a + implied_b)


def shin_no_vig(price_a: float, price_b: float, iterations: int = 60) -> float:
    """Shin (1993) de-vig, which attributes the hold to insider trading.

    Proportional de-vig spreads the margin evenly across both sides, so it
    systematically overstates longshots in a market with a favourite-longshot
    bias — exactly the bias documented in tennis. Shin's z solves
    ``sum(sqrt(z^2 + 4(1-z) q_i^2 / S) - z) / 2 == 1`` and pulls the correction
    disproportionately onto the longer price.
    """
    implied = [1.0 / price_a, 1.0 / price_b]
    booksum = sum(implied)
    if booksum <= 1.0:
        return implied[0] / booksum
    low, high = 0.0, 1.0
    for _ in range(iterations):
        z = (low + high) / 2.0
        total = sum(
            (math.sqrt(z * z + 4.0 * (1.0 - z) * (q * q) / booksum) - z) / 2.0 for q in implied
        )
        if total > 1.0:
            low = z
        else:
            high = z
    z = (low + high) / 2.0
    fair = [
        (math.sqrt(z * z + 4.0 * (1.0 - z) * (q * q) / booksum) - z) / 2.0 for q in implied
    ]
    total = sum(fair) or 1.0
    return fair[0] / total


def market_probability(record: dict[str, Any], book: str = "ps", method: str = "proportional") -> float | None:
    """De-vigged P(player 1 wins) from the archive's closing prices."""
    odds = record.get("odds") or {}
    prices = odds.get(book) or odds.get("b365") or odds.get("avg")
    if not prices:
        return None
    winner_price, loser_price = prices
    if winner_price <= 1.0 or loser_price <= 1.0:
        return None
    solver = shin_no_vig if method == "shin" else no_vig
    winner_fair = solver(winner_price, loser_price)
    return winner_fair if record["winner_key"] == record["p1_key"] else 1.0 - winner_fair


def price_for(record: dict[str, Any], side_is_p1: bool, book: str = "ps") -> float | None:
    odds = record.get("odds") or {}
    prices = odds.get(book)
    if not prices:
        return None
    winner_price, loser_price = prices
    p1_is_winner = record["winner_key"] == record["p1_key"]
    if side_is_p1:
        return winner_price if p1_is_winner else loser_price
    return loser_price if p1_is_winner else winner_price


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def build_records(matches: list[Match], config: EloConfig, emit_from_season: int) -> list[dict[str, Any]]:
    engine = RatingEngine(config)
    return engine.replay(matches, emit_from=f"{emit_from_season}-01-01")


def _elo_only_probabilities(records: Iterable[dict[str, Any]], key: str = "blend_elo_diff") -> list[float]:
    return [elo_probability(record["features"][key], 0.0) for record in records]


def tune_elo(matches: list[Match], seasons: Iterable[int], *, verbose: bool = True) -> tuple[EloConfig, list[dict[str, Any]]]:
    """Grid-search the rating hyper-parameters on the development seasons.

    Scored on the blended-Elo log loss alone: the rating system is what is being
    tuned, and letting the downstream classifier into the loop would cost an
    order of magnitude more compute for a choice it barely moves.
    """
    season_set = set(seasons)
    first_season = min(season_set)
    grid: list[EloConfig] = []
    # The first pass of this grid selected its lower corner (scale 150, shape
    # 0.3), so both axes are extended past it — a hyper-parameter pinned to a
    # boundary means the search, not the data, chose the value.
    for blend in (0.4, 0.55, 0.7, 0.85, 1.0):
        for shape in (0.2, 0.3, 0.4, 0.5):
            for scale in (80.0, 110.0, 150.0, 200.0, 250.0, 320.0):
                for tier_weight in (1.0, 1.3):
                    grid.append(
                        EloConfig(
                            k_scale=scale,
                            k_shape=shape,
                            surface_k_scale=scale,
                            surface_k_shape=shape,
                            surface_blend=blend,
                            tier_k_weight=tier_weight,
                        )
                    )
    results: list[dict[str, Any]] = []
    for index, config in enumerate(grid, start=1):
        records = [
            record
            for record in build_records(matches, config, first_season)
            if record["season"] in season_set
        ]
        probabilities = _elo_only_probabilities(records)
        labels = [record["label"] for record in records]
        score = log_loss(probabilities, labels)
        results.append({"config": config.to_dict(), "log_loss": round(score, 5), "n": len(labels)})
        if verbose and index % 10 == 0:
            print(f"  [elo tune] {index}/{len(grid)} best so far {min(r['log_loss'] for r in results):.5f}")
    results.sort(key=lambda row: row["log_loss"])
    best = EloConfig.from_dict(results[0]["config"])
    return best, results[:12]


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def _hgb(depth: int = 5, iterations: int = 400, learning_rate: float = 0.05) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=depth,
        learning_rate=learning_rate,
        max_iter=iterations,
        min_samples_leaf=60,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")


# Deep trees overfit a signal this smooth: the first sweep had a depth-5,
# 400-round ensemble losing to plain logistic regression, so the shallow,
# heavily-shrunk variant is in the running too.
_TREE_SHAPES: dict[str, tuple[int, int, float]] = {
    "hgb": (5, 400, 0.05),
    "hgb_shallow": (3, 250, 0.04),
}


def _standardise(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = train.mean(axis=0)
    scales = train.std(axis=0)
    scales[scales < 1e-9] = 1.0
    return means, scales


def _fit(kind: str, features: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    """Fit one model on the *augmented* set (each match plus its mirror).

    Training on both orientations is what makes the learned function
    antisymmetric rather than merely symmetrised after the fact; it doubles the
    rows and removes any residual preference for the "p1" slot.
    """
    mirrored = np.array([mirror_vector(row) for row in features])
    augmented = np.vstack([features, mirrored])
    augmented_labels = np.concatenate([labels, 1 - labels])
    augmented_weights = None if weights is None else np.concatenate([weights, weights])
    if kind == "logistic":
        means, scales = _standardise(augmented)
        scaled = (augmented - means) / scales
        estimator = _logistic().fit(scaled, augmented_labels, sample_weight=augmented_weights)
        return {"kind": "linear", "estimator": estimator, "means": means, "scales": scales}
    depth, iterations, learning_rate = _TREE_SHAPES[kind]
    estimator = _hgb(depth, iterations, learning_rate).fit(augmented, augmented_labels, sample_weight=augmented_weights)
    return {"kind": "trees", "estimator": estimator, "means": None, "scales": None}


def _predict(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    estimator = model["estimator"]
    if model["kind"] == "linear":
        scaled = (features - model["means"]) / model["scales"]
        return estimator.predict_proba(scaled)[:, 1]
    return estimator.predict_proba(features)[:, 1]


def _predict_symmetric(model: dict[str, Any], features: np.ndarray) -> np.ndarray:
    forward = _predict(model, features)
    mirrored = _predict(model, np.array([mirror_vector(row) for row in features]))
    return np.array([symmetrise(f, m) for f, m in zip(forward, mirrored)])


def _sample_weights(records: Sequence[dict[str, Any]], target_season: int, half_life: float | None) -> np.ndarray | None:
    if not half_life:
        return None
    ages = np.array([max(0, target_season - record["season"]) for record in records], dtype=float)
    return np.power(0.5, ages / half_life)


def walk_forward(
    records: Sequence[dict[str, Any]],
    seasons: Iterable[int],
    kind: str,
    *,
    half_life: float | None = None,
    min_train: int = 8000,
) -> list[dict[str, Any]]:
    """Train on everything strictly before each season, score that season."""
    out: list[dict[str, Any]] = []
    by_season: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_season.setdefault(record["season"], []).append(record)
    for season in seasons:
        test = by_season.get(season) or []
        train = [record for record in records if record["season"] < season]
        if not test or len(train) < min_train:
            continue
        features = np.array(matrix(train), dtype=float)
        labels = np.array([record["label"] for record in train], dtype=int)
        weights = _sample_weights(train, season, half_life)
        model = _fit(kind, features, labels, weights)
        test_features = np.array(matrix(test), dtype=float)
        probabilities = _predict_symmetric(model, test_features)
        for record, probability in zip(test, probabilities):
            out.append({**record, "p_model": float(probability)})
    return out


# --------------------------------------------------------------------------
# calibration and market combination
# --------------------------------------------------------------------------


def fit_isotonic(predictions: Sequence[dict[str, Any]]) -> IsotonicRegression:
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    isotonic.fit(
        [record["p_model"] for record in predictions],
        [record["label"] for record in predictions],
    )
    return isotonic


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def fit_market_blend(predictions: Sequence[dict[str, Any]], *, book: str = "ps") -> dict[str, Any]:
    """Logistic forecast combination of model and market, and its encompassing test.

    ``model_weight`` is the coefficient on the model's logit once the market's
    logit is already in the regression. If it is ~0 the market encompasses the
    model and there is no edge to bet; if it is positive and the combination
    beats the market out-of-sample, the model carries information the price
    does not.
    """
    rows: list[tuple[float, float, int]] = []
    for record in predictions:
        market = market_probability(record, book=book)
        if market is None:
            continue
        rows.append((_logit(record["p_model_cal"]), _logit(market), record["label"]))
    if len(rows) < 500:
        return {"intercept": 0.0, "model_weight": 1.0, "market_weight": 0.0, "n": len(rows)}
    features = np.array([[row[0], row[1]] for row in rows], dtype=float)
    labels = np.array([row[2] for row in rows], dtype=int)
    estimator = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs").fit(features, labels)
    model_weight, market_weight = (float(value) for value in estimator.coef_[0])
    combined = estimator.predict_proba(features)[:, 1]
    market_only = np.array([1.0 / (1.0 + math.exp(-row[1])) for row in rows])
    model_only = np.array([1.0 / (1.0 + math.exp(-row[0])) for row in rows])
    return {
        "intercept": float(estimator.intercept_[0]),
        "model_weight": model_weight,
        "market_weight": market_weight,
        "n": len(rows),
        "in_sample": {
            "combined": summarise(combined, labels),
            "market_only": summarise(market_only, labels),
            "model_only": summarise(model_only, labels),
        },
    }


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------


def _export_trees(estimator: HistGradientBoostingClassifier) -> dict[str, Any]:
    trees: list[dict[str, list[Any]]] = []
    for stage in estimator._predictors:  # noqa: SLF001 - the only route to the raw nodes
        nodes = stage[0].nodes
        trees.append({
            "is_leaf": [int(value) for value in nodes["is_leaf"]],
            "left": [int(value) for value in nodes["left"]],
            "right": [int(value) for value in nodes["right"]],
            "feature": [int(value) for value in nodes["feature_idx"]],
            "threshold": [float(value) for value in nodes["num_threshold"]],
            "value": [float(value) for value in nodes["value"]],
        })
    return {
        "kind": "trees",
        "baseline": float(np.ravel(estimator._baseline_prediction)[0]),  # noqa: SLF001
        "trees": trees,
    }


def _export_linear(model: dict[str, Any]) -> dict[str, Any]:
    estimator = model["estimator"]
    return {
        "kind": "linear",
        "coefficients": [float(value) for value in estimator.coef_[0]],
        "intercept": float(estimator.intercept_[0]),
        "means": [float(value) for value in model["means"]],
        "scales": [float(value) for value in model["scales"]],
    }


def _export_isotonic(isotonic: IsotonicRegression) -> dict[str, list[float]]:
    return {
        "x": [float(value) for value in isotonic.X_thresholds_],
        "y": [float(value) for value in isotonic.y_thresholds_],
    }


def _verify_export(payload: dict[str, Any], features: np.ndarray, expected: Sequence[float]) -> float:
    """Largest probability gap between the JSON scorer and scikit-learn."""
    exported = TennisModel(payload)
    return max(
        (abs(exported.predict(vector) - float(target)) for vector, target in zip(features, expected)),
        default=0.0,
    )


# --------------------------------------------------------------------------
# training entry point
# --------------------------------------------------------------------------


def train(
    *,
    refresh: bool = False,
    tune: bool = True,
    last_season: int | None = None,
    quick: bool = False,
) -> dict[str, Any]:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    started = time.time()

    if refresh:
        print("[tennis] refreshing raw archive…")
        print(json.dumps(build_spine(download=True, last_season=last_season), indent=2))
    matches = read_spine()
    if len(matches) < 20_000:
        raise SystemExit(f"tennis spine too small ({len(matches)} matches) — run tennis_data.py first")
    print(f"[tennis] {len(matches)} matches through {max(m.date for m in matches)}")

    dev_seasons = list(DEV_SEASONS)
    test_seasons = [season for season in TEST_SEASONS if season <= (last_season or 2100)]

    config = EloConfig()
    tuning: list[dict[str, Any]] = []
    if tune:
        print("[tennis] tuning Elo hyper-parameters on the development seasons…")
        config, tuning = tune_elo(matches, dev_seasons)
        print(f"[tennis] best Elo config: {config.to_dict()}")

    records = build_records(matches, config, BURN_IN_THROUGH + 1)
    dev_records = [record for record in records if record["season"] in set(dev_seasons)]
    print(f"[tennis] {len(records)} feature rows ({len(dev_records)} in development)")

    # -- model family selection, on development seasons only ---------------
    candidates = ["logistic", "hgb", "hgb_shallow"] if not quick else ["logistic"]
    half_lives: list[float | None] = [None, 6.0] if not quick else [None]
    selection: list[dict[str, Any]] = []
    best_choice: tuple[str, float | None] | None = None
    best_score = float("inf")
    dev_predictions: dict[tuple[str, float | None], list[dict[str, Any]]] = {}
    for kind in candidates:
        for half_life in half_lives:
            predictions = walk_forward(records, dev_seasons, kind, half_life=half_life)
            if not predictions:
                continue
            probabilities = [record["p_model"] for record in predictions]
            labels = [record["label"] for record in predictions]
            score = log_loss(probabilities, labels)
            selection.append({
                "model": kind,
                "half_life": half_life,
                **summarise(probabilities, labels),
            })
            dev_predictions[(kind, half_life)] = predictions
            print(f"  [select] {kind:9s} half_life={half_life}  logloss={score:.5f}")
            if score < best_score:
                best_score = score
                best_choice = (kind, half_life)
    if best_choice is None:
        raise SystemExit("no model could be trained — check the spine")
    kind, half_life = best_choice
    print(f"[tennis] selected {kind} (half_life={half_life})")

    # -- calibration, fitted on development out-of-fold predictions --------
    development = dev_predictions[best_choice]
    isotonic = fit_isotonic(development)
    for record in development:
        record["p_model_cal"] = float(isotonic.predict([record["p_model"]])[0])
    blend = fit_market_blend(development)
    print(f"[tennis] market blend: model={blend['model_weight']:.3f} market={blend['market_weight']:.3f}")

    # -- held-out evaluation ------------------------------------------------
    print("[tennis] walk-forward over the held-out seasons…")
    test_predictions = walk_forward(records, test_seasons, kind, half_life=half_life)
    for record in test_predictions:
        record["p_model_cal"] = float(isotonic.predict([record["p_model"]])[0])
        market = market_probability(record)
        record["p_market"] = market
        record["p_final"] = (
            record["p_model_cal"]
            if market is None
            else float(
                1.0
                / (
                    1.0
                    + math.exp(
                        -(
                            blend["intercept"]
                            + blend["model_weight"] * _logit(record["p_model_cal"])
                            + blend["market_weight"] * _logit(market)
                        )
                    )
                )
            )
        )

    labels = [record["label"] for record in test_predictions]
    priced = [record for record in test_predictions if record["p_market"] is not None]
    priced_labels = [record["label"] for record in priced]
    held_out = {
        "seasons": [test_seasons[0], test_seasons[-1]] if test_seasons else [],
        "elo": summarise(_elo_only_probabilities(test_predictions), labels),
        "welo": summarise(_elo_only_probabilities(test_predictions, "welo_diff"), labels),
        "model_raw": summarise([record["p_model"] for record in test_predictions], labels),
        "model_calibrated": summarise([record["p_model_cal"] for record in test_predictions], labels),
        "market": summarise([record["p_market"] for record in priced], priced_labels),
        "model_calibrated_priced": summarise([record["p_model_cal"] for record in priced], priced_labels),
        "combined": summarise([record["p_final"] for record in priced], priced_labels),
        "calibration_curve": calibration_table([record["p_model_cal"] for record in test_predictions], labels),
    }
    print(json.dumps({key: value for key, value in held_out.items() if key != "calibration_curve"}, indent=2))

    # -- betting backtest on the same held-out predictions -------------------
    print("[tennis] backtesting against closing prices…")
    backtest = {
        "model_only": simulate(test_predictions, probability_key="p_model_cal"),
        "market_blended": simulate(test_predictions, probability_key="p_final"),
        "confidence_curve": confidence_curve(test_predictions),
    }
    BACKTEST_PATH.write_text(json.dumps(backtest, indent=2) + "\n", encoding="utf-8")
    print(summarise_backtest(backtest["model_only"]))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "heldout_predictions.json").write_text(
        json.dumps(
            [
                {key: record[key] for key in (
                    "date", "season", "tour", "surface", "tier", "best_of", "p1_key",
                    "p2_key", "winner_key", "label", "odds", "p_model", "p_model_cal", "p_market", "p_final",
                )}
                for record in test_predictions
            ],
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    # -- final fit on everything -------------------------------------------
    print("[tennis] fitting the shipped model on the full history…")
    features = np.array(matrix(records), dtype=float)
    all_labels = np.array([record["label"] for record in records], dtype=int)
    latest_season = max(record["season"] for record in records)
    weights = _sample_weights(records, latest_season, half_life)
    final_model = _fit(kind, features, all_labels, weights)

    # Refit calibration on every out-of-fold prediction we produced (dev + test),
    # never on the final model's own training predictions.
    all_out_of_fold = development + test_predictions
    final_isotonic = fit_isotonic(all_out_of_fold)
    for record in all_out_of_fold:
        record["p_model_cal"] = float(final_isotonic.predict([record["p_model"]])[0])
    final_blend = fit_market_blend(all_out_of_fold)

    payload: dict[str, Any] = {
        "schema": 1,
        "feature_names": list(FEATURE_NAMES),
        "calibration": _export_isotonic(final_isotonic),
        "market_blend": {
            "intercept": final_blend["intercept"],
            "model_weight": final_blend["model_weight"],
            "market_weight": final_blend["market_weight"],
        },
    }
    payload.update(_export_trees(final_model["estimator"]) if kind == "hgb" else _export_linear(final_model))

    sample = features[:: max(1, len(features) // 4000)]
    expected = final_isotonic.predict(_predict_symmetric(final_model, sample))
    export_error = _verify_export(payload, sample, expected)
    print(f"[tennis] JSON scorer matches scikit-learn to {export_error:.2e}")
    if export_error > 1e-6:
        raise SystemExit(f"exported scorer disagrees with scikit-learn by {export_error:.3e} — refusing to ship")

    # -- ratings snapshot + tournament index for serving --------------------
    engine = RatingEngine(config)
    engine.replay(matches, emit_from=None, emit_to=None)
    pruned = engine.prune()
    engine.save()
    TOURNAMENT_INDEX_PATH.write_text(
        json.dumps(build_tournament_index(matches), separators=(",", ":")) + "\n", encoding="utf-8"
    )
    print(f"[tennis] ratings snapshot: {pruned['players_after']}/{pruned['players_before']} players kept")

    metadata = {
        "model_version": f"{MODEL_VERSION}_{kind}",
        "model_family": kind,
        "half_life_seasons": half_life,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "elo_config": config.to_dict(),
        "matches": len(matches),
        "feature_rows": len(records),
        "ratings_through": engine.last_date,
        "export_max_abs_error": export_error,
        "training_seconds": round(time.time() - started, 1),
    }
    payload["metadata"] = metadata
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    metrics = {
        "metadata": metadata,
        "protocol": {
            "burn_in_through": BURN_IN_THROUGH,
            "development_seasons": [dev_seasons[0], dev_seasons[-1]],
            "held_out_seasons": [test_seasons[0], test_seasons[-1]] if test_seasons else [],
            "note": "Every reported figure is walk-forward out-of-sample: each season is scored by a model trained only on earlier seasons.",
        },
        "elo_tuning_top": tuning,
        "model_selection": selection,
        "market_blend": final_blend,
        "held_out": held_out,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"[tennis] wrote {MODEL_PATH.name} ({MODEL_PATH.stat().st_size / 1024:.0f} KB) and {METRICS_PATH.name}")
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the in-house tennis model.")
    parser.add_argument("--refresh", action="store_true", help="Re-download the archive before training.")
    parser.add_argument("--no-tune", action="store_true", help="Skip the Elo hyper-parameter search.")
    parser.add_argument("--quick", action="store_true", help="Logistic only; for smoke tests.")
    parser.add_argument("--last-season", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(refresh=args.refresh, tune=not args.no_tune, last_season=args.last_season, quick=args.quick)
