from __future__ import annotations

from scripts import backtest_parlay_cards as backtest


def test_backtest_summary_counts_whole_slips_and_leg_exposure():
    payloads = [
        (
            "parlay_cards_v3_calibrated_portfolio",
            {
                "cards": [
                    {
                        "id": "a",
                        "comboKey": "a",
                        "date": "2026-06-29",
                        "category": "three_leg_value",
                        "pickMode": "team",
                        "result": "win",
                        "profitUnits": 2.5,
                        "oddsAmerican": 250,
                        "legCount": 3,
                        "consensusLegs": 0,
                        "legs": [{"legId": "l1"}, {"legId": "l2"}, {"legId": "l3"}],
                    },
                    {
                        "id": "b",
                        "comboKey": "b",
                        "date": "2026-06-29",
                        "category": "three_leg_value",
                        "pickMode": "team",
                        "result": "pending",
                        "profitUnits": 0,
                        "oddsAmerican": 300,
                        "legCount": 3,
                        "consensusLegs": 0,
                        "legs": [{"legId": "l1"}, {"legId": "l4"}, {"legId": "l5"}],
                    },
                ]
            },
        )
    ]

    rows = backtest.summarize(payloads, settled_only=True)

    assert len(rows) == 1
    assert rows[0]["slips"] == 1
    assert rows[0]["wins"] == 1
    assert rows[0]["maxLegExposure"] == 1
    assert rows[0]["threeLegSlips"] == 1
