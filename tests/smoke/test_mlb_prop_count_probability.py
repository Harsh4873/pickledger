"""Published MLB prop prices have to follow the projected count.

Classifiers were clearing juiced hit and RBI unders by a couple of points
while the projected mean sat on the other side of the breakeven price.
"""

from __future__ import annotations

import math

from player_props.schema import american_implied_probability, poisson_side_probability
from player_props.variants import _apply_consensus_publication_gate


def _qualified_under(probability: float, odds: int) -> dict:
    implied = american_implied_probability(odds)
    return {
        "required": True,
        "qualified": True,
        "reason": "qualified",
        "selection": "Under",
        "odds": odds,
        "implied_probability": implied,
        "probability": probability,
        "season_probability": probability,
        "history_probability": probability,
        "agreement": True,
        "consensus_score": probability,
        "validation_accuracy": 0.74,
        "holdout_accuracy": 0.72,
        "conservative_validation_accuracy": 0.72,
        "model_version": "player_props_consensus_v2.0.0",
        "training_fingerprint": "count-cap-test",
    }


def _pick(*, projection: float, line: float, over_odds: int, under_odds: int) -> dict:
    return {
        "id": "pp_count_cap",
        "sport": "MLB",
        "date": "2026-09-23",
        "game_id": "824301",
        "player_id": "606466",
        "player_name": "Test Hitter",
        "stat_key": "hits",
        "stat_label": "Hits",
        "line": line,
        "selection": "Under",
        "odds": under_odds,
        "projection": projection,
        "baseline_projection": projection,
        "market_over_odds": over_odds,
        "market_under_odds": under_odds,
        "market_priced": True,
        "model_variant": "all_time",
        "model_variant_label": "All Time",
        "key_factors": [],
    }


def test_poisson_under_matches_the_projected_mean():
    # Mean 1.47 on a 1.5 line is P(X <= 1), about 56.8%, not a 63% under.
    marte = poisson_side_probability(1.47, 1.5, "Under")
    assert abs(marte - (math.exp(-1.47) * (1.0 + 1.47))) < 1e-9
    assert marte < american_implied_probability(-144)

    # Mean 0.61 on a 0.5 line is P(X = 0). It does not clear -169.
    rodden = poisson_side_probability(0.61, 0.5, "Under")
    assert abs(rodden - math.exp(-0.61)) < 1e-9
    assert rodden < american_implied_probability(-169)

    # A 0.15 RBI mean is a real under, but -499 only pays if the edge clears
    # the same 3 point decision rule used everywhere else.
    heavy = poisson_side_probability(0.15, 0.5, "Under")
    assert abs(heavy - math.exp(-0.15)) < 1e-9
    assert (heavy - american_implied_probability(-499)) * 100 < 3.0


def test_classifier_cannot_publish_a_juiced_under_the_mean_does_not_support(monkeypatch):
    import player_props.variants as variants

    monkeypatch.delenv("PICKLEDGER_DISABLE_PRECISION_MODEL", raising=False)

    def gate(pick):
        return _qualified_under(float(pick["classifier"]), int(pick["market_under_odds"]))

    monkeypatch.setattr(variants, "evaluate_consensus_pick", gate)

    cases = [
        # Ketel Marte U1.5 hits, classifier 62.9% at -144, mean 1.47.
        {"projection": 1.47, "line": 1.5, "over_odds": 109, "under_odds": -144, "classifier": 0.6287},
        # Brock Rodden U0.5 hits, classifier 66.4% at -169, mean 0.61.
        {"projection": 0.61, "line": 0.5, "over_odds": 127, "under_odds": -169, "classifier": 0.6639},
        # RBI under at -499. Classifier 87.1% on a 0.15 mean is a 2.8 point edge.
        {"projection": 0.15, "line": 0.5, "over_odds": 332, "under_odds": -499, "classifier": 0.8711},
    ]
    for case in cases:
        pick = _pick(
            projection=case["projection"],
            line=case["line"],
            over_odds=case["over_odds"],
            under_odds=case["under_odds"],
        )
        pick["classifier"] = case["classifier"]
        published = _apply_consensus_publication_gate(pick)
        count_probability = poisson_side_probability(case["projection"], case["line"], "Under")
        assert published["consensus_qualified"] is True
        assert published["probability"] <= count_probability + 5e-4
        assert published["probability"] < case["classifier"]
        assert published["decision"] == "PASS"
        assert published["units"] == 0.0


def test_low_count_mean_can_still_clear_a_posted_under(monkeypatch):
    import player_props.variants as variants

    monkeypatch.delenv("PICKLEDGER_DISABLE_PRECISION_MODEL", raising=False)
    monkeypatch.setattr(
        variants,
        "evaluate_consensus_pick",
        lambda pick: _qualified_under(0.90, -200),
    )
    pick = _pick(projection=0.10, line=0.5, over_odds=170, under_odds=-200)
    published = _apply_consensus_publication_gate(pick)
    count_probability = poisson_side_probability(0.10, 0.5, "Under")
    assert published["probability"] <= count_probability + 1e-9
    assert published["decision"] in {"LEAN", "BET"}
    assert published["ml_expected_value"] > 0
