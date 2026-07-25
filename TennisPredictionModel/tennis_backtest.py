"""Betting backtest: does the model's edge survive contact with real prices?

Forecast quality and betting profitability are different questions. The model
can be better than Elo and better calibrated than the market in places and
still lose money, because the price it has to beat already contains a margin.
This module answers the money question on the held-out seasons only, walking
forward exactly as training did, and it reports the answer per price source:

* ``ps``   — Pinnacle's closing price, the sharpest number in the archive and
  the benchmark the market-efficiency literature uses. Beating it is the
  strong claim.
* ``b365`` — a mainstream book, wider margin, slower to move.
* ``max``  — the best price across every book in the archive at closing. This
  is what a line shopper actually gets, and it is the price at which the
  published Weighted-Elo result (+3.56% ROI on ATP, 2012-2020) was obtained.

Staking is flat by default because fractional-Kelly returns on a mis-calibrated
edge flatter the strategy; Kelly is reported alongside for reference.

One thing this data cannot answer: true closing line value. CLV compares the
price you *took* against the price at the close, and the archive stores only
closing prices — there is no bet-time number to compare with. The `avg_clv`
column is therefore the price advantage of one book's close over Pinnacle's,
which measures the value of line shopping, not model skill.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .tennis_core import ARTIFACT_DIR

BACKTEST_PATH = ARTIFACT_DIR / "backtest.json"

BOOKS = ("ps", "b365", "max")
EDGE_THRESHOLDS = (0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15)
KELLY_FRACTION = 0.25
MAX_STAKE_UNITS = 1.0


def _decimal_prices(record: dict[str, Any], book: str) -> tuple[float, float] | None:
    """Prices as (p1_price, p2_price) for the record's own orientation."""
    prices = (record.get("odds") or {}).get(book)
    if not prices:
        return None
    winner_price, loser_price = prices
    if winner_price <= 1.0 or loser_price <= 1.0:
        return None
    p1_is_winner = record["winner_key"] == record["p1_key"]
    return (winner_price, loser_price) if p1_is_winner else (loser_price, winner_price)


def kelly_stake(probability: float, price: float) -> float:
    """Fractional-Kelly stake in units, capped."""
    edge = probability * price - 1.0
    if edge <= 0:
        return 0.0
    fraction = edge / (price - 1.0)
    return min(MAX_STAKE_UNITS, max(0.0, fraction * KELLY_FRACTION))


def _segment_keys(record: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        ("tour", record["tour"]),
        ("surface", record["surface"]),
        ("tier", f"tier{record['tier']}"),
        ("best_of", f"bo{record['best_of']}"),
        ("season", str(record["season"])),
    ]


class Ledger:
    """Running P&L for one strategy."""

    def __init__(self) -> None:
        self.bets = 0
        self.staked = 0.0
        self.returned = 0.0
        self.wins = 0
        self.kelly_staked = 0.0
        self.kelly_returned = 0.0
        self.clv_sum = 0.0
        self.clv_count = 0
        self.clv_beats = 0
        # Per-unit profit moments, for the exact ROI standard error.
        self.profit_sum = 0.0
        self.profit_square_sum = 0.0

    def record(self, *, won: bool, price: float, stake: float, kelly: float, clv: float | None) -> None:
        self.bets += 1
        self.staked += stake
        self.kelly_staked += kelly
        profit = (price - 1.0) if won else -1.0
        self.profit_sum += profit
        self.profit_square_sum += profit * profit
        if won:
            self.wins += 1
            self.returned += stake * price
            self.kelly_returned += kelly * price
        if clv is not None:
            self.clv_sum += clv
            self.clv_count += 1
            self.clv_beats += 1 if clv > 0 else 0

    def summary(self) -> dict[str, Any]:
        profit = self.returned - self.staked
        kelly_profit = self.kelly_returned - self.kelly_staked
        return {
            "bets": self.bets,
            "staked_units": round(self.staked, 2),
            "profit_units": round(profit, 2),
            "roi": round(profit / self.staked, 4) if self.staked else None,
            "win_rate": round(self.wins / self.bets, 4) if self.bets else None,
            "kelly_profit_units": round(kelly_profit, 2),
            "kelly_roi": round(kelly_profit / self.kelly_staked, 4) if self.kelly_staked else None,
            "avg_clv": round(self.clv_sum / self.clv_count, 5) if self.clv_count else None,
            "clv_beat_rate": round(self.clv_beats / self.clv_count, 4) if self.clv_count else None,
        }


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    phat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denominator
    return (centre - margin, centre + margin)


def roi_significance(ledger: Ledger) -> dict[str, Any]:
    """Is the ROI distinguishable from zero, or is it noise?

    A flat-stake ROI is the mean of per-bet payouts, so its standard error comes
    straight from the sample variance of those payouts — computed exactly here
    from the accumulated moments rather than approximated from an average price,
    because payouts at 1.05 and payouts at 8.00 have wildly different spread.
    At realistic bet counts the standard error on tennis moneyline ROI is still
    a percentage point or more, which is why small ROIs belong with their
    interval rather than as a headline.
    """
    if ledger.bets < 2 or ledger.staked <= 0:
        return {"t_stat": None, "significant": False}
    roi = (ledger.returned - ledger.staked) / ledger.staked
    mean = ledger.profit_sum / ledger.bets
    variance = max(0.0, ledger.profit_square_sum / ledger.bets - mean * mean)
    standard_error = math.sqrt(variance / (ledger.bets - 1)) if variance > 0 else 0.0
    t_stat = roi / standard_error if standard_error else 0.0
    return {
        "t_stat": round(t_stat, 3),
        "significant": abs(t_stat) >= 1.96,
        "roi_stderr": round(standard_error, 4),
    }


def simulate(
    predictions: Sequence[dict[str, Any]],
    *,
    probability_key: str = "p_model_cal",
    books: Iterable[str] = BOOKS,
    thresholds: Iterable[float] = EDGE_THRESHOLDS,
    closing_book: str = "ps",
) -> dict[str, Any]:
    """Flat-stake simulation across books, thresholds and segments.

    A bet is placed on whichever side the model prefers to the price, sized at
    one unit, whenever the model's probability exceeds the book's *vigged*
    implied probability by the threshold. The comparison is deliberately
    against the vigged number: that is the real hurdle, since the bettor pays
    the margin.
    """
    results: dict[str, Any] = {}
    for book in books:
        by_threshold: dict[str, Any] = {}
        for threshold in thresholds:
            ledger = Ledger()
            segments: dict[str, dict[str, Ledger]] = defaultdict(lambda: defaultdict(Ledger))
            price_total = 0.0
            for record in predictions:
                prices = _decimal_prices(record, book)
                if prices is None:
                    continue
                probability = record.get(probability_key)
                if probability is None:
                    continue
                closing = _decimal_prices(record, closing_book)
                for side_index, side_probability in ((0, probability), (1, 1.0 - probability)):
                    price = prices[side_index]
                    implied = 1.0 / price
                    if side_probability - implied < threshold:
                        continue
                    won = (record["label"] == 1) if side_index == 0 else (record["label"] == 0)
                    stake = 1.0
                    kelly = kelly_stake(side_probability, price)
                    clv = None
                    if closing is not None and book != closing_book:
                        # Price advantage over the sharp close. This is NOT true
                        # closing line value: the archive only stores closing
                        # prices, so there is no bet-time price to compare
                        # against, and real CLV cannot be computed from it. What
                        # this measures is how the book we bet compares with
                        # Pinnacle's close — which for `max` is mechanically
                        # positive, since it is the best of every book
                        # including Pinnacle. It is reported as the value of
                        # line shopping, not as evidence of model skill. Left
                        # unset when the bet book *is* the closing book, where
                        # it would be identically zero.
                        clv = price / closing[side_index] - 1.0
                    ledger.record(won=won, price=price, stake=stake, kelly=kelly, clv=clv)
                    price_total += price
                    for name, value in _segment_keys(record):
                        segments[name][value].record(won=won, price=price, stake=stake, kelly=kelly, clv=clv)
            average_price = price_total / ledger.bets if ledger.bets else 0.0
            entry = ledger.summary()
            entry["avg_price"] = round(average_price, 3)
            entry.update(roi_significance(ledger))
            if threshold in (0.04, 0.06):
                entry["segments"] = {
                    name: {value: sub.summary() for value, sub in sorted(bucket.items())}
                    for name, bucket in segments.items()
                }
            by_threshold[f"{threshold:.2f}"] = entry
        results[book] = by_threshold
    return results


CONFIDENCE_BUCKETS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01)


def confidence_curve(
    predictions: Sequence[dict[str, Any]],
    *,
    probability_key: str = "p_model_cal",
) -> list[dict[str, Any]]:
    """Hit rate and ROI by model confidence, for the *published* side.

    This is what sets the BET/LEAN gates. The rows the site shows are unpriced,
    so the gate has to be a confidence threshold; this table says what each
    threshold has historically been worth — both as a hit rate (which is what
    the site's own ledger will grade) and as ROI at real prices (which is what
    it would have been worth to bet).
    """
    buckets: dict[int, dict[str, Any]] = {}
    for record in predictions:
        probability = record.get(probability_key)
        if probability is None:
            continue
        # Always evaluate the side the model would actually publish.
        backs_p1 = probability >= 0.5
        confidence = probability if backs_p1 else 1.0 - probability
        won = (record["label"] == 1) if backs_p1 else (record["label"] == 0)
        index = max(i for i, edge in enumerate(CONFIDENCE_BUCKETS) if confidence >= edge)
        bucket = buckets.setdefault(index, {
            "range": f"{CONFIDENCE_BUCKETS[index]:.2f}-{min(CONFIDENCE_BUCKETS[index + 1], 1.0):.2f}",
            "n": 0, "wins": 0, "confidence_sum": 0.0,
            "ledgers": {book: Ledger() for book in BOOKS},
        })
        bucket["n"] += 1
        bucket["wins"] += 1 if won else 0
        bucket["confidence_sum"] += confidence
        for book in BOOKS:
            prices = _decimal_prices(record, book)
            if prices is None:
                continue
            price = prices[0] if backs_p1 else prices[1]
            bucket["ledgers"][book].record(won=won, price=price, stake=1.0, kelly=0.0, clv=None)
    rows: list[dict[str, Any]] = []
    for index in sorted(buckets):
        bucket = buckets[index]
        low, high = _wilson_interval(bucket["wins"], bucket["n"])
        predicted = bucket["confidence_sum"] / bucket["n"]
        row = {
            "range": bucket["range"],
            "n": bucket["n"],
            "predicted": round(predicted, 4),
            "actual": round(bucket["wins"] / bucket["n"], 4),
            # 95% interval on the realised rate. A bucket is only miscalibrated
            # if the predicted rate falls outside it; otherwise the gap is
            # sample noise, which matters most in the thin high-confidence tail.
            "actual_ci": [round(low, 4), round(high, 4)],
            "calibrated": bool(low <= predicted <= high),
        }
        for book, ledger in bucket["ledgers"].items():
            summary = ledger.summary()
            row[f"roi_{book}"] = summary["roi"]
            row[f"bets_{book}"] = summary["bets"]
        rows.append(row)
    return rows


def summarise_backtest(payload: dict[str, Any]) -> str:
    lines = ["book  thr    bets   ROI      win%    CLV      t"]
    for book, thresholds in payload.items():
        for threshold, entry in thresholds.items():
            roi = entry.get("roi")
            clv = entry.get("avg_clv")
            lines.append(
                f"{book:5s} {threshold:5s} {entry['bets']:6d}  "
                f"{('%+.2f%%' % (roi * 100)) if roi is not None else '   n/a':>8s}  "
                f"{('%.1f%%' % (entry['win_rate'] * 100)) if entry.get('win_rate') else ' n/a':>6s}  "
                f"{('%+.3f%%' % (clv * 100)) if clv is not None else '  n/a':>8s}  "
                f"{entry.get('t_stat') if entry.get('t_stat') is not None else 'n/a'}"
            )
    return "\n".join(lines)


def run(predictions_path: Path, output_path: Path = BACKTEST_PATH) -> dict[str, Any]:
    """Rebuild the report from a dumped held-out prediction set.

    Produces exactly the structure `tennis_train` writes, so the artifact can
    be regenerated without a full retrain.
    """
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    report = {
        "model_only": simulate(predictions, probability_key="p_model_cal"),
        "market_blended": simulate(predictions, probability_key="p_final"),
        "confidence_curve": confidence_curve(predictions),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest the tennis model against closing prices.")
    parser.add_argument("predictions", type=Path, help="JSON produced by tennis_train --dump-predictions")
    args = parser.parse_args()
    output = run(args.predictions)
    print(summarise_backtest(output["model_only"]))
