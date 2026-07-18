#!/usr/bin/env python3
"""Shared no-vig (devig) price helpers.

Decision gates and price-attach steps must agree on what "fair probability"
means: a single vigged side overstates edge by roughly half the hold (about
2.4 points on a standard -110/-110 market), so every gate should compare the
model against a devigged number whenever a complete market was captured. The
Profit Desk keeps its stricter, record-shape-aware derivation on top of these
primitives.
"""

from __future__ import annotations

from typing import Any, Mapping


def american_implied_probability(odds: Any) -> float | None:
    try:
        number = float(odds)
    except (TypeError, ValueError):
        return None
    if number == 0 or -100.0 < number < 100.0:
        return None
    if number > 0:
        return 100.0 / (number + 100.0)
    return abs(number) / (abs(number) + 100.0)


def two_sided_no_vig(selected_odds: Any, opposite_odds: Any) -> float | None:
    """Proportionally devig a complete two-sided market.

    Returns the fair probability of the selected side, or None when either
    side is missing — a single side must never be devigged.
    """
    selected = american_implied_probability(selected_odds)
    opposite = american_implied_probability(opposite_odds)
    if selected is None or opposite is None:
        return None
    hold = selected + opposite
    if hold <= 0:
        return None
    return selected / hold


def no_vig_selected_probability(pick: Mapping[str, Any]) -> float | None:
    """Best available no-vig probability for the pick's own side.

    Prefers an explicitly stamped value (which covers three-way markets whose
    fair price cannot be derived from a two-way pair), then derives from the
    captured selected/opposite pair.
    """
    explicit = pick.get("market_no_vig_selected_probability")
    try:
        value = float(explicit) if explicit is not None else None
    except (TypeError, ValueError):
        value = None
    if value is not None and 0.0 < value < 1.0:
        return value
    return two_sided_no_vig(pick.get("selected_odds"), pick.get("opposite_odds"))
