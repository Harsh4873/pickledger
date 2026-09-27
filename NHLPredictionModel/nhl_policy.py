"""Priced NHL research candidates. No candidate is a live stake approval."""
from __future__ import annotations

import math
from typing import Any, Mapping

from scripts.price_clock import observed_quote_timing


def _implied(odds: int) -> float:
    return 100.0 / (odds + 100.0) if odds > 0 else -odds / (100.0 - odds)


def _profit(odds: int) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / -odds


def priced_choice(
    probabilities: Mapping[str, float], prices: Mapping[str, int | None], *, push: float = 0.0,
) -> dict[str, Any] | None:
    """Choose the higher EV side of a complete two-way observed quote."""
    if len(probabilities) != 2 or set(probabilities) != set(prices):
        return None
    if not math.isfinite(push) or not 0.0 <= push < 1.0:
        return None
    if abs(sum(probabilities.values()) + push - 1.0) > 0.001:
        return None
    if any(
        not isinstance(price, int) or isinstance(price, bool) or abs(price) < 100
        for price in prices.values()
    ):
        return None
    if any(not math.isfinite(probability) or probability < 0.0 for probability in probabilities.values()):
        return None
    offered = {side: int(price) for side, price in prices.items() if price is not None}
    implied = {side: _implied(price) for side, price in offered.items()}
    vigged_sum = sum(implied.values())
    if not 0.9 <= vigged_sum <= 1.3:
        return None
    values = {
        side: probabilities[side] * _profit(price) - (1.0 - probabilities[side] - push)
        for side, price in offered.items()
    }
    side = max(values, key=lambda item: (values[item], probabilities[item]))
    opposite = next(item for item in offered if item != side)
    fair = implied[side] / vigged_sum
    return {
        "side": side,
        "odds": offered[side],
        "opposite_odds": offered[opposite],
        "probability": probabilities[side],
        "push_probability": push,
        "market_implied_probability": fair,
        "edge": probabilities[side] / (1.0 - push) - fair,
        "expected_value": values[side],
    }


def shadow_candidate(
    policy: Mapping[str, Any], *, market: str, choice: Mapping[str, Any] | None,
    evidence_ok: bool, odds_source: str | None, quote_at: str | None,
    published_at: str, start_at: str,
) -> tuple[str, float, str]:
    """Apply the frozen, unvalidated EV rule only to timely observed quotes."""
    if str(policy.get("mode")) != "research_only":
        return "PASS", 0.0, "research_only:policy_not_validated"
    rule = policy.get("shadow_candidate")
    if not isinstance(rule, Mapping) or market not in (rule.get("markets") or []):
        return "PASS", 0.0, "research_only:market_not_in_shadow_rule"
    if not evidence_ok:
        return "PASS", 0.0, "research_only:team_ratings_unavailable"
    if choice is None:
        return "PASS", 0.0, "research_only:incomplete_two_sided_quote"
    if odds_source not in {"draftkings", "observed_quote"}:
        return "PASS", 0.0, "research_only:unverified_quote_source"
    clock_reason = observed_quote_timing(
        {"market_retrieved_at": quote_at}, published_at=published_at, start_at=start_at,
    )
    if clock_reason is not None:
        return "PASS", 0.0, f"research_only:{clock_reason}"
    try:
        odds = int(choice["odds"])
        value = float(choice["expected_value"])
        max_juice = int(rule["max_juice"])
        bet_floor = float(rule["bet_min_expected_value"])
        lean_floor = float(rule["lean_min_expected_value"])
    except (KeyError, TypeError, ValueError):
        return "PASS", 0.0, "research_only:invalid_shadow_rule"
    if odds < max_juice:
        return "PASS", 0.0, "research_only:juice_above_cap"
    if value >= bet_floor:
        return "BET", float(rule.get("bet_units") or 0.5), "research_only:shadow_ev_bet"
    if value >= lean_floor:
        return "LEAN", float(rule.get("lean_units") or 0.25), "research_only:shadow_ev_lean"
    return "PASS", 0.0, "research_only:below_shadow_ev_floor"
