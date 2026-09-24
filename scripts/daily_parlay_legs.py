#!/usr/bin/env python3
"""Build ReBet and Fliff 2-leg parlays from posted prices.

Daily ReBet ($1) and Fliff ($2) tickets are one 2-leg parlay per book. A leg
clears the bar only when all of the following are true:

* a posted American price is present (never assumed, never invented)
* that price is from -200 to -125 inclusive
* the model's own probability is at least 0.60
* that probability minus the posted break-even is strictly positive

Legs that miss the bar are dropped before any ticket is assembled. If a book
would need one of those legs to reach two legs, that book gets no ticket.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scripts.build_parlay_cards import american_to_decimal, model_posted_edge

# Method 9/22 bar. These numbers are the existing daily-lane bar, not the
# MLB consensus publication thresholds.
BAR_PROB_MIN = 0.60
BAR_PRICE_MIN = -200
BAR_PRICE_MAX = -125

BOOKS: tuple[tuple[str, float], ...] = (
    ("ReBet", 1.0),
    ("Fliff", 2.0),
)


@dataclass(frozen=True)
class DailyLeg:
    leg_id: str
    pick: str
    game: str
    american_odds: int | None
    model_probability: float | None


@dataclass(frozen=True)
class DailyParlay:
    book: str
    stake: float
    legs: tuple[DailyLeg, ...]
    combined_decimal: float
    hit_probability: float
    ev: float


def clears_bar(leg: DailyLeg) -> bool:
    """True when this leg can stand on a ticket by itself."""
    odds = leg.american_odds
    probability = leg.model_probability
    if isinstance(odds, bool) or isinstance(probability, bool):
        return False
    if not isinstance(odds, int):
        return False
    if odds < BAR_PRICE_MIN or odds > BAR_PRICE_MAX:
        return False
    if probability is None:
        return False
    try:
        probability_value = float(probability)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(probability_value) or probability_value < BAR_PROB_MIN:
        return False
    edge = model_posted_edge(probability_value, odds)
    return edge is not None and edge > 0


def _pair_legs(legs: list[DailyLeg]) -> list[tuple[DailyLeg, DailyLeg]]:
    """Greedy pairs from the already-filtered pool. No backfill."""
    ordered = sorted(
        legs,
        key=lambda leg: (
            -(leg.model_probability or 0.0),
            -(model_posted_edge(leg.model_probability, leg.american_odds) or 0.0),
            leg.leg_id,
        ),
    )
    used: set[str] = set()
    pairs: list[tuple[DailyLeg, DailyLeg]] = []
    for index, leg in enumerate(ordered):
        if leg.leg_id in used:
            continue
        partner = next(
            (
                other
                for other in ordered[index + 1 :]
                if other.leg_id not in used and other.game != leg.game
            ),
            None,
        )
        if partner is None:
            continue
        used.add(leg.leg_id)
        used.add(partner.leg_id)
        pairs.append((leg, partner))
        if len(pairs) >= len(BOOKS):
            break
    return pairs


def _ticket(book: str, stake: float, pair: tuple[DailyLeg, DailyLeg]) -> DailyParlay:
    combined_decimal = math.prod(american_to_decimal(leg.american_odds) for leg in pair)
    hit_probability = math.prod(float(leg.model_probability) for leg in pair)
    return DailyParlay(
        book=book,
        stake=stake,
        legs=pair,
        combined_decimal=combined_decimal,
        hit_probability=hit_probability,
        ev=hit_probability * combined_decimal - 1.0,
    )


def build_daily_parlays(legs: list[DailyLeg]) -> list[DailyParlay]:
    """One ticket per book, and only from legs that clear the bar.

    Fewer than two qualifying legs means no ticket. A third book is never
    created, and a below-bar leg is never pulled in to give Fliff a slip
    after ReBet has taken the only good pair.
    """
    eligible = [leg for leg in legs if clears_bar(leg)]
    pairs = _pair_legs(eligible)
    return [
        _ticket(book, stake, pair)
        for (book, stake), pair in zip(BOOKS, pairs)
    ]
