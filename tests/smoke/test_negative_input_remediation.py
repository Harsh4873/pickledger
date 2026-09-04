"""Guards for the remediation of provably-negative inputs.

Three policies are pinned here, each of which was reintroducible by a one-line
edit and each of which cost real units:

1. `beats_close` must not qualify a model group. It is measured against our own
   last pregame capture, not a true market close, and carries no information
   about whether a pick won.
2. An unpriced win must not be credited at even money, which silently
   manufactured P&L.
3. Scraped third-party tip feeds must not publish as staked BET/LEAN picks.
4. Unpriced in-house tennis must not publish as staked BET/LEAN picks.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.mlb_team_consensus import _walk_forward_performance
from scripts.merge_external_feed_cache_payload import (
    EXTERNAL_FEED_MODEL_KEYS,
    _demote_scraped_feed_picks,
)
from scripts.pick_calibration import _profit


# ---------------------------------------------------------------------------
# 1. beats_close must not qualify a group
# ---------------------------------------------------------------------------

def _ledger_record(**overrides):
    record = {
        "model_key": "mlb_new",
        "bet_type": "h2h",
        "result": "loss",
        "profit": -1.0,
        "stake_units": 1.0,
        "odds": -110.0,
        "probability": 0.52,
        "market_implied_probability": 0.524,
    }
    record.update(overrides)
    return record


def _write_ledger(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "outcome_ledger.json"
    path.write_text(json.dumps({"records": records}), encoding="utf-8")
    return path


def test_losing_group_is_not_qualified_by_beating_the_close(tmp_path):
    """A group that loses money must not qualify, however good its CLV looks.

    This is the covers_computer_mlb case: it posted strongly positive movement
    against our own capture while losing tens of units. Under the old
    `profitable or beats_close` gate it qualified anyway.
    """
    # 40 losses, each with a large positive CLV so beats_close would be True.
    records = [
        _ledger_record(closing_odds=-200.0, odds=-110.0)
        for _ in range(40)
    ]
    performance = _walk_forward_performance(_write_ledger(tmp_path, records))
    group = performance[("mlb_new", "h2h")]

    assert group["samples"] == 40
    assert group["profit"] < 0
    assert group["qualified"] is False, "a money-losing group must never qualify"


def test_profitable_group_still_qualifies(tmp_path):
    """Removing beats_close must not break the legitimate qualification path."""
    records = [_ledger_record(result="win", profit=0.91) for _ in range(40)]
    performance = _walk_forward_performance(_write_ledger(tmp_path, records))
    group = performance[("mlb_new", "h2h")]

    assert group["profit"] > 0
    assert group["qualified"] is True


def test_beats_close_is_still_recorded_for_visibility(tmp_path):
    """The metric stays observable even though it no longer gates promotion."""
    records = [_ledger_record() for _ in range(40)]
    performance = _walk_forward_performance(_write_ledger(tmp_path, records))
    assert "beats_close" in performance[("mlb_new", "h2h")]


def test_unpriced_picks_do_not_contribute_stake(tmp_path):
    """Unpriced picks count for win rate but are excluded from the money math."""
    records = [_ledger_record(result="win", profit=None, odds=None) for _ in range(10)]
    performance = _walk_forward_performance(_write_ledger(tmp_path, records))
    group = performance[("mlb_new", "h2h")]

    assert group["samples"] == 10
    assert group["wins"] == 10
    assert group["stake"] == 0.0, "no stake may accrue from picks we cannot price"
    assert group["profit"] == 0.0
    assert group["roi"] is None


# ---------------------------------------------------------------------------
# 2. An unpriced win must not be booked at even money
# ---------------------------------------------------------------------------

def test_unpriced_win_is_not_credited_at_even_money():
    assert _profit("win", 1.0, None) is None
    assert _profit("win", 1.0, 0) is None


def test_priced_and_derivable_outcomes_are_unaffected():
    # Real American odds still pay out normally.
    assert _profit("win", 1.0, 150) == 1.5
    assert _profit("win", 1.0, -200) == 0.5
    assert _profit("loss", 1.0, -110) == -1.0
    assert _profit("push", 1.0, -110) == 0.0
    # An implied probability is still enough to derive a fair payout.
    assert _profit("win", 1.0, None, 0.5) == 1.0


# ---------------------------------------------------------------------------
# 3. Scraped tip feeds must publish as untracked research rows
# ---------------------------------------------------------------------------

def _scraped_payload():
    bucket = {
        "ok": True,
        "date": "2026-09-03",
        "picks": [
            {"pick": "Cubs ML", "decision": "BET", "units": 1, "odds": -132},
            {"pick": "Over 8.5", "decision": "LEAN", "units": 1, "odds": -110},
            {"pick": "Rays ML", "decision": "PASS", "units": 0, "odds": 120},
        ],
    }
    inhouse = {
        "ok": True,
        "date": "2026-09-03",
        "picks": [{"pick": "Under 9.0", "decision": "BET", "units": 0.5, "odds": -105}],
    }
    return {
        "date": "2026-09-03",
        "models": {"scores24_mlb": bucket, "mlb_new": inhouse},
        "external_feeds": {"scores24_mlb": bucket},
        "scores24_mlb": bucket,
        "mlb_new": inhouse,
    }


def test_scraped_feed_picks_are_demoted_everywhere():
    demoted = _demote_scraped_feed_picks(_scraped_payload())

    for container in (
        demoted["models"]["scores24_mlb"],
        demoted["external_feeds"]["scores24_mlb"],
        demoted["scores24_mlb"],
    ):
        decisions = {str(p["decision"]).upper() for p in container["picks"]}
        assert decisions == {"PASS"}, "no scraped tip may publish as BET or LEAN"
        assert all(float(p["units"]) == 0 for p in container["picks"])
        assert container["scraped_tip_feed"] is True


def test_demotion_preserves_what_the_source_said():
    demoted = _demote_scraped_feed_picks(_scraped_payload())
    picks = demoted["models"]["scores24_mlb"]["picks"]
    changed = [p for p in picks if p.get("scraped_tip_demoted")]

    assert len(changed) == 2
    assert {p["source_decision"] for p in changed} == {"BET", "LEAN"}
    assert all(p["source_units"] == 1 for p in changed)


def test_inhouse_models_are_untouched():
    demoted = _demote_scraped_feed_picks(_scraped_payload())
    for container in (demoted["models"]["mlb_new"], demoted["mlb_new"]):
        pick = container["picks"][0]
        assert pick["decision"] == "BET"
        assert pick["units"] == 0.5
        assert "scraped_tip_demoted" not in pick
        assert "scraped_tip_feed" not in container


def test_every_registered_external_feed_key_is_covered():
    """The demotion is keyed off the same registry the merge pipeline uses."""
    assert "scores24_mlb" in EXTERNAL_FEED_MODEL_KEYS
    for key in EXTERNAL_FEED_MODEL_KEYS:
        payload = {
            "date": "2026-09-03",
            "models": {
                key: {
                    "ok": True,
                    "picks": [{"pick": "x", "decision": "BET", "units": 1}],
                }
            },
        }
        picks = _demote_scraped_feed_picks(payload)["models"][key]["picks"]
        assert picks[0]["decision"] == "PASS"
        assert picks[0]["units"] == 0


def test_model_cache_merge_also_demotes_scraped_feeds(tmp_path):
    """An in-house model refresh must not republish tipster rows as bets."""
    from scripts.merge_model_cache_payload import merge_payload

    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-09-03",
        "models": {
            "scores24_mlb": {
                "ok": True,
                "picks": [{"pick": "Cubs ML", "decision": "BET", "units": 1}],
            },
            "mlb_new": {
                "ok": True,
                "picks": [{"pick": "Under 9.0", "decision": "BET", "units": 0.5}],
            },
        },
        "external_feeds": {
            "scores24_mlb": {
                "ok": True,
                "picks": [{"pick": "Cubs ML", "decision": "BET", "units": 1}],
            },
        },
        "scores24_mlb": {
            "ok": True,
            "picks": [{"pick": "Cubs ML", "decision": "BET", "units": 1}],
        },
    }
    (cache_dir / "2026-09-03.json").write_text(json.dumps(current), encoding="utf-8")
    generated = {
        "date": "2026-09-03",
        "models": {
            "mlb_new": {
                "ok": True,
                "picks": [{"pick": "Under 9.0", "decision": "BET", "units": 0.5}],
            },
        },
    }

    merged = merge_payload(generated, cache_dir)
    scraped = merged["models"]["scores24_mlb"]["picks"][0]
    assert scraped["decision"] == "PASS"
    assert scraped["units"] == 0
    assert scraped["scraped_tip_demoted"] is True
    assert merged["models"]["mlb_new"]["picks"][0]["decision"] == "BET"
    assert merged["models"]["mlb_new"]["picks"][0]["units"] == 0.5


# ---------------------------------------------------------------------------
# 4. Unpriced in-house tennis must not publish as staked BET/LEAN picks
# ---------------------------------------------------------------------------

def test_unpriced_tennis_picks_are_demoted_to_pass():
    from scripts.merge_model_cache_payload import _demote_unpriced_tennis_picks

    payload = {
        "date": "2026-09-04",
        "models": {
            "tennis": {
                "ok": True,
                "picks": [
                    {"sport": "Tennis", "pick": "Alcaraz ML", "decision": "BET", "units": 1, "odds": None},
                    {"sport": "Tennis", "pick": "Paul ML", "decision": "LEAN", "units": 0.5, "odds": None},
                    {"sport": "Tennis", "pick": "Priced ML", "decision": "BET", "units": 0.5, "odds": -150},
                ],
            },
            "mlb_new": {
                "ok": True,
                "picks": [{"sport": "MLB", "pick": "Under 9.0", "decision": "BET", "units": 0.5, "odds": -105}],
            },
        },
        "tennis": {
            "ok": True,
            "picks": [{"sport": "Tennis", "pick": "Alcaraz ML", "decision": "BET", "units": 1, "odds": None}],
        },
    }

    demoted, changed = _demote_unpriced_tennis_picks(payload)
    tennis = demoted["models"]["tennis"]["picks"]
    assert changed == 3
    assert tennis[0]["decision"] == "PASS" and tennis[0]["units"] == 0
    assert tennis[0]["source_decision"] == "BET" and tennis[0]["source_units"] == 1
    assert tennis[0]["unpriced_tennis_demoted"] is True
    assert tennis[1]["decision"] == "PASS" and tennis[1]["units"] == 0
    assert tennis[1]["source_decision"] == "LEAN"
    assert tennis[2]["decision"] == "BET" and tennis[2]["units"] == 0.5
    assert demoted["models"]["mlb_new"]["picks"][0]["decision"] == "BET"
    assert demoted["tennis"]["picks"][0]["decision"] == "PASS"


def test_model_cache_merge_demotes_unpriced_tennis(tmp_path):
    from scripts.merge_model_cache_payload import merge_payload

    cache_dir = tmp_path / "data" / "model_cache"
    cache_dir.mkdir(parents=True)
    current = {
        "date": "2026-09-04",
        "models": {
            "tennis": {
                "ok": True,
                "picks": [{"sport": "Tennis", "pick": "Alcaraz ML", "decision": "BET", "units": 1, "odds": None}],
            },
            "mlb_new": {
                "ok": True,
                "picks": [{"sport": "MLB", "pick": "Under 9.0", "decision": "BET", "units": 0.5, "odds": -105}],
            },
        },
    }
    (cache_dir / "2026-09-04.json").write_text(json.dumps(current), encoding="utf-8")
    generated = {
        "date": "2026-09-04",
        "models": {
            "mlb_new": {
                "ok": True,
                "picks": [{"sport": "MLB", "pick": "Under 9.0", "decision": "BET", "units": 0.5, "odds": -105}],
            },
        },
    }

    merged = merge_payload(generated, cache_dir)
    tennis = merged["models"]["tennis"]["picks"][0]
    assert tennis["decision"] == "PASS"
    assert tennis["units"] == 0
    assert tennis["unpriced_tennis_demoted"] is True
    assert merged["models"]["mlb_new"]["picks"][0]["decision"] == "BET"
