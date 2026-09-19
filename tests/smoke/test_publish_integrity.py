"""Shared publish-integrity rules for in-house team models.

Two rules apply to every in-house bucket regardless of sport:

* a refresh that runs after kickoff must not publish or re-decide that game
  (the cache merge keeps the pre-kickoff rows);
* a BET/LEAN row with no executable price is research (PASS at 0u).
"""
from __future__ import annotations

from datetime import datetime, timezone

from scripts.merge_model_cache_payload import demote_unpriced_team_model_picks
from scripts.refresh_model_cache import freeze_started_games


def _pick(**overrides):
    pick = {
        "sport": "MLB", "date": "2026-09-19", "pick": "Team A ML (Team B @ Team A)",
        "matchup": "Team B @ Team A", "game_id": "g1", "decision": "LEAN", "units": 0.25,
        "odds": -120, "start_time": "2026-09-19T18:10:00Z", "game_start_time": "2026-09-19T18:10:00Z",
    }
    pick.update(overrides)
    return pick


def test_started_games_are_frozen_but_pregame_and_unknown_rows_survive():
    started = _pick()
    later = _pick(game_id="g2", matchup="Team D @ Team C", start_time="2026-09-19T23:10:00Z",
                  game_start_time="2026-09-19T23:10:00Z")
    unknown = _pick(game_id="g3", matchup="Team F @ Team E", start_time=None, game_start_time=None)
    naive = _pick(game_id="g4", matchup="Team H @ Team G", start_time="2026-09-19T13:00",
                  game_start_time=None)
    bucket = {"ok": True, "note": "MLB slate.", "picks": [started, later, unknown, naive]}
    payload = {"date": "2026-09-19", "models": {"mlb_new": bucket, "sportsgambler_mlb": {"picks": [_pick()]}},
               "mlb_new": bucket}

    summary = freeze_started_games(payload, now=datetime(2026, 9, 19, 19, 0, tzinfo=timezone.utc))

    assert summary == {"frozen": 1, "kept": 3, "unknown_start": 2}
    assert [pick["game_id"] for pick in bucket["picks"]] == ["g2", "g3", "g4"]
    # The alias key shares the bucket object, so the site sees the same rows.
    assert payload["mlb_new"]["picks"] is bucket["picks"]
    assert bucket["frozen_started_games"] == 1
    assert "1 row(s) for started games skipped" in bucket["note"]
    # Scraped feeds are not in scope for the freeze.
    assert len(payload["models"]["sportsgambler_mlb"]["picks"]) == 1


def test_freeze_falls_back_to_the_bucket_games_list_for_start_times():
    pick = _pick(start_time=None, game_start_time=None)
    bucket = {"picks": [pick], "games": [{"game_id": "g1", "start_time": "2026-09-19T18:10:00Z"}]}
    payload = {"models": {"wnba": bucket}}
    summary = freeze_started_games(payload, now=datetime(2026, 9, 19, 20, 0, tzinfo=timezone.utc))
    assert summary["frozen"] == 1 and bucket["picks"] == []


def test_unpriced_stakes_become_research_after_the_market_attach():
    unpriced_bet = _pick(decision="BET", units=1.13, odds=None)
    zero_odds = _pick(decision="LEAN", units=0.25, odds=0, game_id="g2")
    priced = _pick(decision="LEAN", units=0.25, odds=-115, game_id="g3", pricing_type="market",
                   odds_source="posted_market", market_priced=True)
    unpriced_pass = _pick(decision="PASS", units=0, odds=None, game_id="g4")
    scraped = _pick(decision="BET", units=1, odds=None, game_id="g5")
    payload = {"models": {
        "nba": {"picks": [unpriced_bet]},
        "mls": {"picks": [zero_odds, priced, unpriced_pass]},
        "scores24_mlb": {"picks": [scraped]},
    }}

    assert demote_unpriced_team_model_picks(payload) == 2
    assert unpriced_bet["decision"] == "PASS" and unpriced_bet["units"] == 0
    assert unpriced_bet["source_decision"] == "BET" and unpriced_bet["source_units"] == 1.13
    assert unpriced_bet["unpriced_demoted"] is True
    assert unpriced_bet["decision_reason"] == "unpriced:no_executable_price"
    assert zero_odds["decision"] == "PASS"
    assert priced["decision"] == "LEAN" and priced["units"] == 0.25
    assert unpriced_pass["decision"] == "PASS" and "unpriced_demoted" not in unpriced_pass
    assert scraped["decision"] == "BET"  # scraped feeds have their own demotion path


def test_house_assumed_prices_that_were_never_replaced_are_research():
    # mlb_inning stamps -120 on every row and no book posts that market.
    inning = _pick(decision="LEAN", units=0.25, odds=-120, assumed_odds=-120, pricing_type="user_assumed",
                   odds_source="user_assumed_no_run_inning_-120", market_priced=True)
    # A first-five total whose ladder price was replaced by a real ESPN price stays staked …
    replaced = _pick(decision="LEAN", units=0.25, odds=-105, model_assumed_odds=-170, assumed_odds_replaced=True,
                     pricing_type="market", odds_source="posted_market", market_priced=True, game_id="g2")
    # … even when the posted price happened to equal the placeholder.
    same_price = _pick(decision="BET", units=0.5, odds=-110, model_assumed_odds=-110, assumed_odds_replaced=True,
                       pricing_type="market", odds_source="posted_market", market_priced=True, game_id="g3")
    model_total = _pick(decision="LEAN", units=0.3, odds=-110, assumed_odds=-110, market_total_source="model_output",
                        game_id="g4")
    payload = {"models": {"mlb_inning": {"picks": [inning]}, "mlb_first_five": {"picks": [replaced, same_price]},
                          "mlb_new": {"picks": [model_total]}}}

    assert demote_unpriced_team_model_picks(payload) == 2
    assert inning["decision"] == "PASS" and inning["units"] == 0
    assert inning["decision_reason"] == "unpriced:assumed_price_not_replaced"
    assert inning["source_decision"] == "LEAN"
    assert replaced["decision"] == "LEAN" and same_price["decision"] == "BET"
    assert model_total["decision"] == "PASS"


def test_refresh_pipeline_freezes_then_demotes_in_order():
    source = (__import__("pathlib").Path(__file__).resolve().parents[2] / "scripts" / "refresh_model_cache.py").read_text(encoding="utf-8")
    body = source.split("def _write_json_cache")[1]
    assert body.index("freeze_started_games(payload)") < body.index("merge_payload(payload")
    assert body.index("apply_market_odds_to_payload(merged)") < body.index("demote_unpriced_team_model_picks(merged)")
