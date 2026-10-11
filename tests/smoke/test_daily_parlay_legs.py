"""Daily ReBet and Fliff parlays: positive model edge, no below-bar fill."""

from __future__ import annotations

from scripts.daily_parlay_legs import DailyLeg, build_daily_parlays, clears_bar
from scripts.build_parlay_cards import model_posted_edge


def _leg(leg_id: str, pick: str, game: str, odds: int | None, probability: float | None) -> DailyLeg:
    return DailyLeg(leg_id, pick, game, odds, probability)


# Posted 2026-09-23 prices. Probabilities are the 0.516 coin-flip shape.
SEP23 = [
    _leg("pirates", "Pirates ML", "STL at PIT", -121, 0.516),
    _leg("redsox", "Red Sox ML", "CLE at BOS", -136, 0.516),
    _leg("mariners", "Mariners ML", "HOU vs SEA", -149, 0.516),
    _leg("yankees", "Yankees ML", "TB vs NYY", -157, 0.516),
]

# Posted 2026-09-22 reference prices and model probabilities. These clear the bar.
SEP22 = [
    _leg("dodgers", "Dodgers ML", "SD at LAD", -125, 0.71),
    _leg("tigers", "Tigers ML", "WSH at DET", -161, 0.68),
    _leg("mariners", "Mariners ML", "HOU at SEA", -149, 0.66),
    _leg("rangers", "Rangers ML", "NYM at TEX", -149, 0.62),
]


def test_sep23_two_coin_flip_probabilities_are_not_a_recommended_parlay():
    """Two 0.516 probabilities at the 9/23 posted prices must not be a parlay."""
    rebet_shape = SEP23[:2]
    fliff_shape = SEP23[2:]
    assert build_daily_parlays(rebet_shape) == []
    assert build_daily_parlays(fliff_shape) == []
    assert build_daily_parlays(SEP23) == []
    for leg in SEP23:
        assert leg.model_probability == 0.516
        assert clears_bar(leg) is False
        edge = model_posted_edge(leg.model_probability, leg.american_odds)
        assert edge is not None and edge < 0


def test_below_bar_leg_is_not_added_to_fill_a_ticket():
    """One good pair does not recruit 9/23 coin flips so the other book has a ticket."""
    tickets = build_daily_parlays(SEP22[:2] + SEP23[:2])
    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.book == "ReBet"
    assert {leg.pick for leg in ticket.legs} == {"Dodgers ML", "Tigers ML"}
    assert all(leg.model_probability >= 0.60 for leg in ticket.legs)
    assert ticket.ev > 0
    assert all(ticket.book != "Fliff" for ticket in tickets)


def test_four_bar_clearing_legs_make_two_tickets_without_shared_legs():
    tickets = build_daily_parlays(SEP22)
    assert [ticket.book for ticket in tickets] == ["ReBet", "Fliff"]
    assert [ticket.stake for ticket in tickets] == [1.0, 2.0]
    used = [leg.leg_id for ticket in tickets for leg in ticket.legs]
    assert len(used) == len(set(used)) == 4
    for ticket in tickets:
        assert len(ticket.legs) == 2
        assert ticket.ev > 0
        assert all(clears_bar(leg) for leg in ticket.legs)


def test_missing_posted_price_is_not_invented_to_fill():
    dodgers = SEP22[0]
    missing = _leg("tigers", "Tigers ML", "WSH at DET", None, 0.68)
    assert clears_bar(missing) is False
    assert model_posted_edge(missing.model_probability, missing.american_odds) is None
    assert build_daily_parlays([dodgers, missing]) == []
