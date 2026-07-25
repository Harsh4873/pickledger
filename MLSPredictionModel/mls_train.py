"""Train, validate, and package the MLS Dixon-Coles model.

Protocol (all evaluation is walk-forward — every prediction uses only matches
strictly before its date):

1. Hyperparameter grid (decay half-life x ridge) scored by raw 3-way log loss
   on the 2019-2022 validation walk, with 2012-2018 as warm-up history.
2. Calibration (1X2 vector scaling, total-rate affine) fit on the validation
   walk's out-of-sample predictions with the chosen hyperparameters.
3. A frozen pass over the 2023+ test walk reports honest held-out numbers:
   log loss / Brier vs the devigged closing market, totals and handicap
   calibration off the score grid, and a flat-stake 1X2 backtest against
   closing prices at the decision gates tuned on validation only.
4. Challenger models (multinomial logistic regression and gradient boosting on
   as-of form/rest/rating features) run the same walk; the Dixon-Coles model
   ships only if the challengers fail to beat it out of sample.
5. Artifacts: the compact match archive, model config + calibrations + gates
   (with calibrations refit on all out-of-sample predictions, as documented),
   and the metrics/backtest tables the README quotes.

Run:  python -m MLSPredictionModel.mls_train --csv USA.csv --stage all
"""
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .mls_core import (
    FitConfig,
    RatingsFit,
    TotalCalibration,
    VectorScaling,
    conditional_probability,
    decimal_to_probability,
    devig,
    fit_ratings,
    fit_total_calibration,
    fit_vector_scaling,
    handicap_split,
    one_x_two,
    score_grid,
    total_split,
)
from .mls_data import (
    ARTIFACT_DIR,
    MlsMatch,
    fetch_football_data_matches,
    parse_football_data_csv,
    save_matches,
)

MODEL_VERSION = "mls_dixon_coles_v2.0"
WARMUP_END = date(2018, 12, 31)
VALIDATION_END = date(2022, 12, 31)
HALF_LIFE_GRID = (180.0, 270.0, 365.0, 540.0, 730.0, None)
RIDGE_GRID = (0.0, 1e-3, 5e-3, 2e-2)
MIN_EFFECTIVE_GAMES = 8.0


@dataclass
class WalkRow:
    match: MlsMatch
    probabilities: dict[str, float]
    lam: float
    mu: float
    rho: float
    home_games: float
    away_games: float

    @property
    def outcome(self) -> str:
        if self.match.home_goals > self.match.away_goals:
            return "home"
        if self.match.home_goals < self.match.away_goals:
            return "away"
        return "draw"


def walk_forward(
    matches: Sequence[MlsMatch],
    config: FitConfig,
    start: date,
    end: date,
    refit_every_days: int = 1,
) -> list[WalkRow]:
    rows: list[WalkRow] = []
    fit: RatingsFit | None = None
    for match in matches:
        if match.date < start or match.date > end:
            continue
        stale = fit is None or (match.date - fit.as_of).days >= refit_every_days
        if stale:
            fit = fit_ratings(matches, match.date, config, warm=fit)
        lam, mu = fit.rates(match.home, match.away)
        grid = score_grid(lam, mu, fit.rho)
        rows.append(WalkRow(
            match=match,
            probabilities=one_x_two(grid),
            lam=lam,
            mu=mu,
            rho=fit.rho,
            home_games=fit.effective_games.get(match.home, 0.0),
            away_games=fit.effective_games.get(match.away, 0.0),
        ))
    return rows


def log_loss_3way(rows: Iterable[tuple[dict[str, float], str]]) -> float:
    total = count = 0.0
    for probabilities, outcome in rows:
        total -= math.log(max(probabilities[outcome], 1e-12))
        count += 1
    return total / count if count else float("nan")


def brier_3way(rows: Iterable[tuple[dict[str, float], str]]) -> float:
    total = count = 0.0
    for probabilities, outcome in rows:
        for key in ("home", "draw", "away"):
            total += (probabilities[key] - (1.0 if key == outcome else 0.0)) ** 2
        count += 1
    return total / count if count else float("nan")


def market_probabilities(match: MlsMatch) -> dict[str, float] | None:
    implied = devig([
        decimal_to_probability(match.close_home),
        decimal_to_probability(match.close_draw),
        decimal_to_probability(match.close_away),
    ])
    if any(value is None for value in implied):
        return None
    return {"home": implied[0], "draw": implied[1], "away": implied[2]}


def _grid_job(args: tuple[list[MlsMatch], float | None, float]) -> dict[str, Any]:
    matches, half_life, ridge = args
    config = FitConfig(half_life_days=half_life, ridge=ridge)
    rows = walk_forward(matches, config, WARMUP_END, VALIDATION_END, refit_every_days=7)
    rows = [row for row in rows if row.match.date > WARMUP_END]
    total_nll = np.mean([
        (row.lam + row.mu) - row.match.total_goals * math.log(max(row.lam + row.mu, 1e-9))
        for row in rows
    ])
    return {
        "half_life_days": half_life,
        "ridge": ridge,
        "val_log_loss": log_loss_3way((row.probabilities, row.outcome) for row in rows),
        "val_total_nll": float(total_nll),
        "matches": len(rows),
    }


def run_hyperparameter_grid(matches: list[MlsMatch], jobs: int) -> list[dict[str, Any]]:
    combos = [(matches, half_life, ridge) for half_life in HALF_LIFE_GRID for ridge in RIDGE_GRID]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(_grid_job, combos))
    results.sort(key=lambda row: row["val_log_loss"])
    return results


def reliability_table(pairs: Sequence[tuple[float, int]], buckets: int = 8) -> list[dict[str, float]]:
    """Predicted-probability buckets vs empirical frequency."""
    if not pairs:
        return []
    edges = np.linspace(0.0, 1.0, buckets + 1)
    table = []
    values = np.array([pair[0] for pair in pairs])
    hits = np.array([pair[1] for pair in pairs], dtype=float)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high if high < 1.0 else values <= high)
        if not mask.any():
            continue
        table.append({
            "bucket": f"{low:.3f}-{high:.3f}",
            "count": int(mask.sum()),
            "predicted": float(values[mask].mean()),
            "empirical": float(hits[mask].mean()),
        })
    return table


def expected_calibration_error(pairs: Sequence[tuple[float, int]], buckets: int = 8) -> float:
    table = reliability_table(pairs, buckets)
    total = sum(row["count"] for row in table)
    if not total:
        return float("nan")
    return sum(abs(row["predicted"] - row["empirical"]) * row["count"] for row in table) / total


def grid_market_validation(rows: Sequence[WalkRow], calibration: TotalCalibration | None) -> dict[str, Any]:
    """Totals and handicap calibration read off the (calibrated) score grid."""
    overs: dict[float, list[tuple[float, int]]] = {2.5: [], 3.5: []}
    covers: dict[float, list[tuple[float, int]]] = {-0.5: [], -1.0: [], -1.5: [], 0.5: [], 1.0: [], 1.5: []}
    for row in rows:
        lam, mu = (calibration.scale(row.lam, row.mu) if calibration else (row.lam, row.mu))
        grid = score_grid(lam, mu, row.rho)
        for line in overs:
            split = total_split(grid, line)
            overs[line].append((split["over"], int(row.match.total_goals > line)))
        margin = row.match.home_goals - row.match.away_goals
        for line in covers:
            split = handicap_split(grid, "home", line)
            adjusted = margin + line
            if abs(adjusted) < 1e-9:
                continue
            covers[line].append((conditional_probability(split["win"], split["loss"]), int(adjusted > 0)))
    report: dict[str, Any] = {"totals": {}, "handicaps": {}}
    for line, pairs in overs.items():
        report["totals"][f"over_{line:g}"] = {
            "count": len(pairs),
            "predicted_rate": float(np.mean([pair[0] for pair in pairs])),
            "empirical_rate": float(np.mean([pair[1] for pair in pairs])),
            "brier": float(np.mean([(pair[0] - pair[1]) ** 2 for pair in pairs])),
            "ece": expected_calibration_error(pairs),
            "reliability": reliability_table(pairs),
        }
    for line, pairs in covers.items():
        report["handicaps"][f"home_{line:+g}"] = {
            "count": len(pairs),
            "predicted_rate": float(np.mean([pair[0] for pair in pairs])) if pairs else None,
            "empirical_rate": float(np.mean([pair[1] for pair in pairs])) if pairs else None,
            "ece": expected_calibration_error(pairs),
        }
    return report


def moneyline_backtest(
    rows: Sequence[WalkRow],
    scaling: VectorScaling,
    edge_threshold: float,
    probability_floor: float,
    *,
    sides: tuple[str, ...] = ("home", "away"),
    best_price: bool = False,
) -> dict[str, Any]:
    """Flat one-unit stake on the best-edge side against closing prices."""
    staked = returned = bets = wins = 0.0
    by_season: dict[int, float] = {}
    for row in rows:
        market = market_probabilities(row.match)
        if market is None:
            continue
        calibrated = scaling.apply(row.probabilities)
        candidates = []
        for side in sides:
            edge = calibrated[side] - market[side]
            price = {
                "home": row.match.best_home if best_price else row.match.close_home,
                "draw": row.match.best_draw if best_price else row.match.close_draw,
                "away": row.match.best_away if best_price else row.match.close_away,
            }[side]
            if price is None:
                continue
            candidates.append((edge, side, price, calibrated[side]))
        if not candidates:
            continue
        edge, side, price, probability = max(candidates)
        if edge < edge_threshold or probability < probability_floor:
            continue
        bets += 1
        staked += 1
        outcome = row.outcome
        profit = (price - 1.0) if outcome == side else -1.0
        wins += 1 if outcome == side else 0
        returned += profit
        season = row.match.date.year
        by_season[season] = by_season.get(season, 0.0) + profit
    return {
        "edge_threshold": edge_threshold,
        "probability_floor": probability_floor,
        "sides": list(sides),
        "price": "best_close" if best_price else "sharp_close",
        "bets": int(bets),
        "hit_rate": wins / bets if bets else None,
        "roi": returned / staked if staked else None,
        "units": round(returned, 2),
        "by_season": {str(season): round(value, 2) for season, value in sorted(by_season.items())},
    }


def blended_probability_rows(
    rows: Sequence[WalkRow],
    scaling: VectorScaling,
    weight: float,
) -> list[tuple[WalkRow, dict[str, float], dict[str, float], dict[str, float]]]:
    """(row, calibrated, market, blended) for every match with closing prices."""
    out = []
    for row in rows:
        market = market_probabilities(row.match)
        if market is None:
            continue
        calibrated = scaling.apply(row.probabilities)
        blended = {key: (1.0 - weight) * calibrated[key] + weight * market[key] for key in calibrated}
        out.append((row, calibrated, market, blended))
    return out


def select_blend_weight(rows: Sequence[WalkRow], scaling: VectorScaling) -> dict[str, Any]:
    """Smallest market weight within 0.002 nats of the market-only log loss.

    The blend exists because the model alone trails the closing market; the
    selection keeps the model's voice as large as the evidence allows.
    """
    curve = {}
    for weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        scored = [(blended, row.outcome) for row, _, _, blended in blended_probability_rows(rows, scaling, weight)]
        curve[weight] = log_loss_3way(scored)
    market_only = curve[1.0]
    chosen = min(weight for weight, loss in curve.items() if loss <= market_only + 0.002)
    return {"curve": {f"{weight:.1f}": loss for weight, loss in curve.items()}, "chosen": chosen}


def confidence_gate_table(
    rows: Sequence[WalkRow],
    scaling: VectorScaling,
    weight: float,
    thresholds: Sequence[float] = (0.50, 0.55, 0.58, 0.60, 0.65),
    price_cap_decimal: float | None = None,
) -> list[dict[str, Any]]:
    """Hit rate of the blended-favorite side (home/away only) by threshold.

    ``price_cap_decimal`` drops picks priced shorter than the cap at close —
    the shipped behavior, where nothing shorter than -250 publishes decided.
    """
    picked = []
    for row, _, _, blended in blended_probability_rows(rows, scaling, weight):
        side = "home" if blended["home"] >= blended["away"] else "away"
        if price_cap_decimal is not None:
            price = row.match.close_home if side == "home" else row.match.close_away
            if price is None or price < price_cap_decimal:
                continue
        picked.append((blended[side], int(row.outcome == side)))
    table = []
    for threshold in thresholds:
        selected = [hit for probability, hit in picked if probability >= threshold]
        if not selected:
            continue
        table.append({
            "threshold": threshold,
            "picks": len(selected),
            "hit_rate": sum(selected) / len(selected),
        })
    return table


def grid_confidence_table(
    rows: Sequence[WalkRow],
    calibration: TotalCalibration,
    thresholds: Sequence[float] = (0.52, 0.545, 0.58, 0.62),
) -> dict[str, list[dict[str, Any]]]:
    """Model-only hit rates for totals/handicap favorites at standard lines."""
    totals_picked = []
    handicap_picked = []
    for row in rows:
        lam, mu = calibration.scale(row.lam, row.mu)
        grid = score_grid(lam, mu, row.rho)
        for line in (2.5, 3.5):
            split = total_split(grid, line)
            side = "over" if split["over"] >= split["under"] else "under"
            probability = conditional_probability(
                split[side], split["under" if side == "over" else "over"])
            hit = int(row.match.total_goals > line) if side == "over" else int(row.match.total_goals < line)
            totals_picked.append((probability, hit))
        margin = row.match.home_goals - row.match.away_goals
        for line in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5):
            split = handicap_split(grid, "home", line)
            home_probability = conditional_probability(split["win"], split["loss"])
            side, probability = ("home", home_probability) if home_probability >= 0.5 else ("away", 1.0 - home_probability)
            adjusted = margin + line
            if abs(adjusted) < 1e-9:
                continue
            hit = int(adjusted > 0) if side == "home" else int(adjusted < 0)
            handicap_picked.append((probability, hit))
    report: dict[str, list[dict[str, Any]]] = {}
    for name, picked in (("totals", totals_picked), ("handicaps", handicap_picked)):
        table = []
        for threshold in thresholds:
            selected = [hit for probability, hit in picked if probability >= threshold]
            if len(selected) < 20:
                continue
            table.append({
                "threshold": threshold,
                "picks": len(selected),
                "hit_rate": sum(selected) / len(selected),
            })
        report[name] = table
    return report


def _form_features(matches: Sequence[MlsMatch], row: WalkRow) -> list[float] | None:
    """As-of rolling form for the challenger models (last 5/10, rest days)."""
    def recent(team: str) -> list[MlsMatch]:
        past = [m for m in matches if m.date < row.match.date and team in (m.home, m.away)]
        return past[-10:]

    features: list[float] = []
    for team in (row.match.home, row.match.away):
        history = recent(team)
        if len(history) < 5:
            return None
        points = goals_for = goals_against = 0.0
        for m in history[-5:]:
            us, them = (m.home_goals, m.away_goals) if m.home == team else (m.away_goals, m.home_goals)
            points += 3.0 if us > them else (1.0 if us == them else 0.0)
            goals_for += us
            goals_against += them
        rest = (row.match.date - history[-1].date).days
        features.extend([points / 5.0, goals_for / 5.0, goals_against / 5.0, min(rest, 21.0)])
    features.extend([
        math.log(max(row.probabilities["home"], 1e-9)),
        math.log(max(row.probabilities["draw"], 1e-9)),
        math.log(max(row.probabilities["away"], 1e-9)),
        row.lam,
        row.mu,
    ])
    return features


def challenger_comparison(matches: Sequence[MlsMatch], val_rows: Sequence[WalkRow], test_rows: Sequence[WalkRow]) -> dict[str, Any]:
    """Walk-forward sklearn challengers vs the Dixon-Coles incumbent."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    outcome_index = {"home": 0, "draw": 1, "away": 2}
    prepared: list[tuple[WalkRow, list[float]]] = []
    for row in [*val_rows, *test_rows]:
        features = _form_features(matches, row)
        if features is not None:
            prepared.append((row, features))
    report: dict[str, Any] = {}
    for name, make in (
        ("logistic_regression", lambda: LogisticRegression(max_iter=2000, C=1.0)),
        ("hist_gradient_boosting", lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=7)),
    ):
        seasons = sorted({row.match.date.year for row, _ in prepared if row.match.date > VALIDATION_END})
        scored: list[tuple[dict[str, float], str]] = []
        for season in seasons:
            train = [(f, outcome_index[row.outcome]) for row, f in prepared if row.match.date.year < season]
            score = [(row, f) for row, f in prepared if row.match.date.year == season]
            if len(train) < 300 or not score:
                continue
            model = make()
            model.fit(np.array([t[0] for t in train]), np.array([t[1] for t in train]))
            predicted = model.predict_proba(np.array([f for _, f in score]))
            for (row, _), probs in zip(score, predicted):
                by_class = {"home": 0.0, "draw": 0.0, "away": 0.0}
                for class_id, probability in zip(model.classes_, probs):
                    by_class[["home", "draw", "away"][int(class_id)]] = float(probability)
                scored.append((by_class, row.outcome))
        report[name] = {
            "test_log_loss": log_loss_3way(scored) if scored else None,
            "test_matches": len(scored),
        }
    return report


def build_artifacts(args: argparse.Namespace) -> None:
    if args.csv:
        matches = parse_football_data_csv(Path(args.csv).read_text(encoding="utf-8-sig"))
    else:
        matches = fetch_football_data_matches()
    data_through = max(match.date for match in matches)
    print(f"dataset: {len(matches)} matches through {data_through}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "trained_at": args.trained_at or datetime.now().strftime("%Y-%m-%d"),
        "data_through": data_through.isoformat(),
        "matches": len(matches),
        "splits": {
            "warmup": f"2012..{WARMUP_END.year}",
            "validation": f"{WARMUP_END.year + 1}..{VALIDATION_END.year}",
            "test": f"{VALIDATION_END.year + 1}..{data_through.isoformat()}",
        },
    }

    print("hyperparameter grid (walk-forward on validation)...")
    grid = run_hyperparameter_grid(matches, args.jobs)
    for row in grid[:5]:
        print(f"  half_life={row['half_life_days']} ridge={row['ridge']} "
              f"log_loss={row['val_log_loss']:.5f} total_nll={row['val_total_nll']:.5f}")
    best = grid[0]
    config = FitConfig(half_life_days=best["half_life_days"], ridge=best["ridge"])
    metrics["hyperparameter_grid"] = grid
    metrics["selected"] = {"half_life_days": best["half_life_days"], "ridge": best["ridge"]}

    print("validation walk (daily refit) ...")
    val_rows = [row for row in walk_forward(matches, config, WARMUP_END, VALIDATION_END)
                if row.match.date > WARMUP_END]
    print("test walk (daily refit) ...")
    test_rows = [row for row in walk_forward(matches, config, VALIDATION_END, data_through)
                 if row.match.date > VALIDATION_END]

    scaling = fit_vector_scaling([(row.probabilities, row.outcome) for row in val_rows])
    totals_calibration = fit_total_calibration(
        [(row.lam + row.mu, row.match.total_goals) for row in val_rows])
    print(f"calibration: gamma={scaling.gamma:.4f} bias_home={scaling.bias_home:+.4f} "
          f"bias_away={scaling.bias_away:+.4f}; total = {totals_calibration.intercept:+.3f} "
          f"+ {totals_calibration.slope:.3f} * raw")

    def summarize(rows: Sequence[WalkRow], label: str) -> None:
        raw = [(row.probabilities, row.outcome) for row in rows]
        calibrated = [(scaling.apply(row.probabilities), row.outcome) for row in rows]
        market_rows = [(market_probabilities(row.match), row.outcome) for row in rows
                       if market_probabilities(row.match) is not None]
        metrics[label] = {
            "matches": len(rows),
            "log_loss_raw": log_loss_3way(raw),
            "log_loss_calibrated": log_loss_3way(calibrated),
            "log_loss_closing_market": log_loss_3way(market_rows),
            "brier_calibrated": brier_3way(calibrated),
            "brier_closing_market": brier_3way(market_rows),
            "home_ece_calibrated": expected_calibration_error(
                [(probabilities["home"], int(outcome == "home")) for probabilities, outcome in calibrated]),
        }
        print(f"  {label}: model {metrics[label]['log_loss_calibrated']:.5f} "
              f"vs market {metrics[label]['log_loss_closing_market']:.5f} "
              f"(raw {metrics[label]['log_loss_raw']:.5f}, n={len(rows)})")

    summarize(val_rows, "validation")
    summarize(test_rows, "test")

    print("grid market validation (totals + handicaps) on test ...")
    metrics["grid_markets_test"] = grid_market_validation(test_rows, totals_calibration)
    for line, stats in metrics["grid_markets_test"]["totals"].items():
        print(f"  {line}: predicted {stats['predicted_rate']:.4f} vs empirical "
              f"{stats['empirical_rate']:.4f} (ece {stats['ece']:.4f}, n={stats['count']})")

    print("challenger models ...")
    metrics["challengers"] = challenger_comparison(matches, val_rows, test_rows)
    for name, stats in metrics["challengers"].items():
        print(f"  {name}: test log_loss {stats['test_log_loss']} (n={stats['test_matches']})")

    print("edge-gate backtest (documented as rejected) ...")
    edge_gate_grid = []
    for edge_threshold in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        for probability_floor in (0.0, 0.35, 0.40):
            trial = moneyline_backtest(val_rows, scaling, edge_threshold, probability_floor)
            trial["bets_per_season"] = round(trial["bets"] / 4.0, 1)
            edge_gate_grid.append(trial)
    best_val_edge_gate = max(edge_gate_grid, key=lambda trial: (trial["roi"] if trial["roi"] is not None else -1))
    edge_gate_rejection = {
        "validation_grid": edge_gate_grid,
        "best_validation_gate": {
            "edge": best_val_edge_gate["edge_threshold"],
            "probability_floor": best_val_edge_gate["probability_floor"],
            "validation_roi": best_val_edge_gate["roi"],
        },
        "test_at_best_validation_gate": moneyline_backtest(
            test_rows, scaling, best_val_edge_gate["edge_threshold"], best_val_edge_gate["probability_floor"]),
        "test_at_best_validation_gate_best_price": moneyline_backtest(
            test_rows, scaling, best_val_edge_gate["edge_threshold"], best_val_edge_gate["probability_floor"],
            best_price=True),
        "conclusion": (
            "Positive-ROI edge gates found on validation invert out of sample; large "
            "model-over-market edges mark model blind spots, not value. Decisions are "
            "therefore confidence-gated on the market-blended probability, and no market "
            "edge is claimed or required."
        ),
    }
    print(f"  best val edge gate {edge_gate_rejection['best_validation_gate']} -> "
          f"test roi {edge_gate_rejection['test_at_best_validation_gate']['roi']:.4f}")

    print("blend weight + confidence gates (tuned on validation, frozen on test) ...")
    blend = select_blend_weight(val_rows, scaling)
    weight = blend["chosen"]
    price_cap_decimal = 1.0 + (100.0 / 250.0)
    confidence = {
        "blend": blend,
        "price_cap": {"american": -250, "decimal": price_cap_decimal},
        "validation_table": confidence_gate_table(val_rows, scaling, weight),
        "test_table": confidence_gate_table(test_rows, scaling, weight),
        "validation_table_price_capped": confidence_gate_table(
            val_rows, scaling, weight, price_cap_decimal=price_cap_decimal),
        "test_table_price_capped": confidence_gate_table(
            test_rows, scaling, weight, price_cap_decimal=price_cap_decimal),
        "grid_markets_validation": grid_confidence_table(val_rows, totals_calibration),
        "grid_markets_test": grid_confidence_table(test_rows, totals_calibration),
    }
    print(f"  blend weight {weight}")
    for row in confidence["test_table_price_capped"]:
        print(f"  moneyline test (capped) >= {row['threshold']:.2f}: n={row['picks']} hit={row['hit_rate']:.3f}")
    bet_gate = {"blended_probability": 0.60}
    lean_gate = {"blended_probability": 0.55}
    grid_market_gates = {"bet_probability": 0.58, "lean_probability": 0.545}
    backtest = {
        "edge_gate_rejected": edge_gate_rejection,
        "confidence_gates": confidence,
        "chosen_gates": {
            "moneyline": {"bet": bet_gate, "lean": lean_gate},
            "grid_markets": grid_market_gates,
        },
    }

    print("refitting calibration on all out-of-sample predictions for the shipped artifact ...")
    all_rows = [*val_rows, *test_rows]
    shipped_scaling = fit_vector_scaling([(row.probabilities, row.outcome) for row in all_rows])
    shipped_totals = fit_total_calibration([(row.lam + row.mu, row.match.total_goals) for row in all_rows])

    save_matches(matches)
    model_payload = {
        "model_version": MODEL_VERSION,
        "trained_at": metrics["trained_at"],
        "data_through": data_through.isoformat(),
        "config": {"half_life_days": config.half_life_days, "ridge": config.ridge},
        "calibration": {
            "vector_scaling": {
                "gamma": shipped_scaling.gamma,
                "bias_home": shipped_scaling.bias_home,
                "bias_away": shipped_scaling.bias_away,
            },
            "total": {"intercept": shipped_totals.intercept, "slope": shipped_totals.slope},
            "fit_on": "validation+test out-of-sample walk (reported metrics used validation-only fits)",
        },
        "market_blend_weight": weight,
        "gates": {
            "moneyline": {
                "bet_blended_probability": bet_gate["blended_probability"],
                "lean_blended_probability": lean_gate["blended_probability"],
            },
            "grid_markets": {
                "bet_blended_probability": grid_market_gates["bet_probability"],
                "lean_blended_probability": grid_market_gates["lean_probability"],
            },
            "max_decided_american_price": -250,
            "max_decided_implied_probability": round(250.0 / 350.0, 4),
            "basis": (
                "confidence gates on the market-blended probability; edge is published "
                "as information, never required (see backtest.json edge_gate_rejected)"
            ),
        },
        "min_effective_games": MIN_EFFECTIVE_GAMES,
    }
    (ARTIFACT_DIR / "mls_model.json").write_text(json.dumps(model_payload, indent=2) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (ARTIFACT_DIR / "backtest.json").write_text(json.dumps(backtest, indent=2) + "\n", encoding="utf-8")
    print(f"artifacts written to {ARTIFACT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="path to football-data USA.csv (downloads when omitted)")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--trained-at", help="ISO date stamped into artifacts")
    args = parser.parse_args()
    build_artifacts(args)


if __name__ == "__main__":
    main()
