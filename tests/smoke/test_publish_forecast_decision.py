"""Publish path: the model's BET/LEAN is what the Today board sees.

Unpriced rows still become research PASS. Unapproved staking stays 0u with
the awaiting-holdout policy visible. True model PASS stays PASS.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.market_odds import apply_market_odds_to_payload
from scripts.merge_model_cache_payload import demote_unpriced_team_model_picks
from scripts.model_stake_policy import apply_stake_policy
from scripts.pick_calibration import apply_calibration_to_payload, load_active_calibration


ROOT = Path(__file__).resolve().parents[2]
MONDAY_CACHE = ROOT / "data" / "model_cache" / "2026-11-09.json"

IDENTITY = {
    "version": "test-publish-lock",
    "minimum_group_samples": 30,
    "global": {"intercept": 0.0, "slope": 1.0, "samples": 100},
    "groups": {},
}


def _nba_pick(**overrides):
    pick = {
        "source": "NBA New",
        "sport": "NBA",
        "date": "2026-11-09",
        "away_team": "Cavaliers",
        "home_team": "Hornets",
        "team": "Cavaliers",
        "pick": "Cavaliers ML (Cavaliers @ Hornets)",
        "decision": "BET",
        "units": 0.89,
        "probability": 0.653,
        "odds": None,
        "pricing_type": "unpriced",
        "market_priced": False,
        "decision_basis": "model_conviction_pending_price",
    }
    pick.update(overrides)
    return pick


def _publish(payload, *, book=None, approvals=None, active=IDENTITY):
    if book is not None:
        apply_market_odds_to_payload(
            payload, book, now=datetime(2026, 11, 9, 16, 0, tzinfo=timezone.utc),
        )
    apply_calibration_to_payload(payload, active)
    demote_unpriced_team_model_picks(payload)
    apply_stake_policy(
        payload,
        approvals={"approvals": [] } if approvals is None else approvals,
        model_keys={"nba"},
    )
    return payload["models"]["nba"]["picks"]


def test_priced_nba_bet_publishes_as_bet_at_zero_units():
    pick = _nba_pick(
        odds=-337,
        pricing_type="market",
        market_priced=True,
        assumed_odds_replaced=True,
        market_no_vig_selected_probability=0.740483,
        selected_odds=-337,
        opposite_odds=270,
    )
    picks = _publish({"models": {"nba": {"picks": [pick]}}})
    assert picks[0]["raw_decision"] == "BET"
    assert picks[0]["decision"] == "BET"
    assert picks[0]["units"] == 0
    assert picks[0]["staking_policy"] == "awaiting_approved_holdout"
    assert picks[0]["decision_reason"] == "awaiting_approved_holdout"
    assert picks[0]["decision_basis"] == "model_conviction"
    assert picks[0]["shadow_decision"] == "BET"


def test_model_pass_stays_pass():
    pick = _nba_pick(
        decision="PASS",
        units=0,
        probability=0.543,
        odds=-120,
        pricing_type="market",
        market_priced=True,
        assumed_odds_replaced=True,
        market_no_vig_selected_probability=0.522984,
        team="Hawks",
        away_team="Lakers",
        home_team="Hawks",
        pick="Hawks ML (Lakers @ Hawks)",
    )
    picks = _publish({"models": {"nba": {"picks": [pick]}}})
    assert picks[0]["raw_decision"] == "PASS"
    assert picks[0]["decision"] == "PASS"
    assert picks[0]["units"] == 0
    assert picks[0].get("staking_policy") != "awaiting_approved_holdout"


def test_unpriced_nba_bet_is_still_research_pass():
    pick = _nba_pick()
    picks = _publish({"models": {"nba": {"picks": [pick]}}})
    assert picks[0]["decision"] == "PASS"
    assert picks[0]["units"] == 0
    assert picks[0]["unpriced_demoted"] is True
    assert picks[0]["source_decision"] == "BET"


def test_monday_nba_new_cache_republishes_raw_forecasts():
    """Replay the 2026-11-09 NBA New rows through the publish gates.

    The committed cache hid every BET/LEAN as PASS. Recalibration plus the
    stake gate must restore the model's raw coverage at 0u without inventing
    games, odds, or an approval.
    """
    src = json.loads(MONDAY_CACHE.read_text(encoding="utf-8"))
    picks = copy.deepcopy(src["models"]["nba"]["picks"])
    assert picks, "2026-11-09 NBA New cache should be populated"
    payload = {"models": {"nba": {"picks": picks}}}
    apply_calibration_to_payload(payload, load_active_calibration() or IDENTITY)
    demote_unpriced_team_model_picks(payload)
    apply_stake_policy(payload, approvals={"approvals": []}, model_keys={"nba"})

    by_game = {(row["away_team"], row["home_team"]): row for row in picks}
    hawks = by_game[("Lakers", "Hawks")]
    assert hawks["raw_decision"] == "PASS"
    assert hawks["decision"] == "PASS"
    assert hawks["units"] == 0

    cavs = by_game[("Cavaliers", "Hornets")]
    assert cavs["raw_decision"] == "BET"
    assert cavs["decision"] == "BET"
    assert cavs["units"] == 0
    assert cavs["staking_policy"] == "awaiting_approved_holdout"
    assert cavs["market_priced"] is True
    assert cavs["decision_basis"] == "model_conviction"

    pistons = by_game[("Wizards", "Pistons")]
    assert pistons["raw_decision"] == "BET" and pistons["decision"] == "BET"

    raptors = by_game[("Grizzlies", "Raptors")]
    assert raptors["raw_decision"] == "LEAN" and raptors["decision"] == "LEAN"
    assert raptors["units"] == 0

    restored = [row for row in picks if row["raw_decision"] in {"BET", "LEAN"}]
    assert restored
    assert all(row["decision"] == row["raw_decision"] for row in restored)
    assert all(row["units"] == 0 for row in restored)
    assert all(row["staking_policy"] == "awaiting_approved_holdout" for row in restored)
    assert all(row.get("unpriced_demoted") is not True for row in restored)
