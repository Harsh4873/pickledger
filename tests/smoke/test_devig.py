import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from player_props.schema import decision_and_stake, market_fair_probability  # noqa: E402
from scripts.devig import no_vig_selected_probability, two_sided_no_vig  # noqa: E402


def test_two_sided_no_vig_symmetric_market():
    assert abs(two_sided_no_vig(-110, -110) - 0.5) < 1e-9


def test_two_sided_no_vig_refuses_single_side():
    assert two_sided_no_vig(-110, None) is None
    assert two_sided_no_vig(None, -110) is None


def test_no_vig_selected_probability_prefers_explicit_stamp():
    pick = {
        "market_no_vig_selected_probability": 0.57,
        "selected_odds": -110,
        "opposite_odds": -110,
    }
    assert no_vig_selected_probability(pick) == 0.57
    derived = no_vig_selected_probability({"selected_odds": -120, "opposite_odds": 100})
    assert abs(derived - (0.545455 / 1.045455)) < 1e-4


def test_fair_probability_only_raises_the_edge_baseline(monkeypatch):
    monkeypatch.delenv("PICKLEDGER_EDGE_BASIS", raising=False)
    # Real -110 price: implied 0.5238 exceeds the 0.50 fair, so the baseline
    # stays at breakeven and the decision is unchanged by fair data.
    with_fair = decision_and_stake(0.60, -110, fair_probability=0.50)
    without_fair = decision_and_stake(0.60, -110)
    assert with_fair == without_fair
    # Assumed -110 price on a market that really trades at 0.58 fair: the
    # captured fair probability raises the baseline and blocks phantom edge.
    blocked = decision_and_stake(0.60, -110, fair_probability=0.58)
    assert blocked[0] == "PASS"


def test_vigged_escape_hatch_ignores_fair_probability(monkeypatch):
    monkeypatch.setenv("PICKLEDGER_EDGE_BASIS", "vigged")
    legacy = decision_and_stake(0.60, -110, fair_probability=0.58)
    assert legacy == decision_and_stake(0.60, -110)


def test_market_fair_probability_from_over_under_pair():
    pick = {"market_over_odds": -120, "market_under_odds": 100}
    over = market_fair_probability(pick, "Over")
    under = market_fair_probability(pick, "Under")
    assert over is not None and under is not None
    assert abs(over + under - 1.0) < 1e-9
    assert over > 0.5
