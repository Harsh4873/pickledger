"""A pick ships only with a fresh book price, and a bad feed cannot erase one."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.observed_price import fresh_observed_book_price, parse_timestamp
from scripts.refresh_external_feeds import _record_feed_attempt


START = "2026-09-23T00:00:00Z"
NOW = "2026-09-22T21:00:00Z"
FRESH_STAMP = "2026-09-22T20:00:00Z"  # 1 hour before the publish clock, 4 hours before tip
# Older than 24 hours on the publish clock. The gap to kickoff is not the test.
STALE_STAMP = "2026-09-21T20:00:00Z"
# Posted two days before kickoff, and only two hours old when the feed is read.
EARLY_NFL_STAMP = "2026-09-22T19:00:00Z"
EARLY_NFL_START = "2026-09-24T19:00:00Z"


def _wnba_prop(*, stamp: str | None, odds: int = -120, player: str = "Breanna Stewart") -> dict:
    pick = {
        "id": f"wnba-{player}",
        "sport": "WNBA",
        "date": "2026-09-22",
        "game_id": "liberty",
        "player_id": player,
        "player_name": player,
        "stat_key": "points",
        "selection": "Over",
        "line": 18.5,
        "market_priced": True,
        "pricing_type": "market",
        "odds_source": "posted_market",
        "decision": "LEAN",
        "units": 0.25,
        "odds": odds,
        "start_time": START,
        "ml_probability": 0.58,
        "probability": 0.58,
        "ml_edge": 0.06,
        "ml_expected_value": 0.08,
        "consensus_required": True,
        "consensus_qualified": True,
        "precision_required": True,
        "precision_qualified": True,
        "actionability": "consensus_qualified",
        "model_variant": "season",
    }
    if stamp is not None:
        pick["market_updated_at"] = stamp
    return pick


def _feed_pick(name: str, *, stamp: str, odds: int, date: str = "2026-09-22", start: str = START) -> dict:
    return {
        "source": "Scores24NFL",
        "pick": name,
        "date": date,
        "sport": "NFL",
        "odds": odds,
        "market_priced": True,
        "pricing_type": "market",
        "odds_source": "posted_market",
        "start_time": start,
        "market_updated_at": stamp,
        "decision": "BET",
        "units": 1,
    }


def test_missing_or_post_start_stamp_does_not_ship_and_early_nfl_line_does():
    import player_props.variants as variants

    fresh = _wnba_prop(stamp=FRESH_STAMP, odds=-120, player="Breanna Stewart")
    # 26 hours before tip, still before the start. That gap is not a reason to drop it.
    early = _wnba_prop(stamp="2026-09-21T22:00:00Z", odds=-137, player="Olivia Miles")
    missing = _wnba_prop(stamp=None, odds=-115, player="A'ja Wilson")
    after_start = _wnba_prop(stamp="2026-09-23T00:32:00Z", odds=-120, player="Breanna Stewart")
    nfl = _feed_pick("Chiefs ML", stamp=EARLY_NFL_STAMP, odds=-110, start=EARLY_NFL_START)
    nfl["date"] = "2026-09-24"

    assert fresh_observed_book_price(fresh) is True
    assert fresh_observed_book_price(early) is True
    assert (parse_timestamp(START) - parse_timestamp(early["market_updated_at"])).total_seconds() / 3600 == 26
    assert fresh_observed_book_price(missing) is False
    assert fresh_observed_book_price(after_start) is False
    assert fresh_observed_book_price(nfl) is True
    assert (parse_timestamp(EARLY_NFL_START) - parse_timestamp(EARLY_NFL_STAMP)).total_seconds() / 3600 == 48

    selected = variants._select_variant([after_start, missing, fresh], "season")
    assert [pick["player_name"] for pick in selected] == ["Breanna Stewart"]
    assert selected[0]["odds"] == -120
    assert selected[0]["units"] == 0.25
    assert selected[0]["decision"] == "LEAN"
    assert "A'ja Wilson" not in {pick["player_name"] for pick in selected}


def test_scores24_nfl_timeout_drops_stale_quote_and_keeps_fresh(tmp_path: Path):
    from scripts.scrapers import scores24_optional_publish as optional

    fresh = _feed_pick("Chiefs ML", stamp=FRESH_STAMP, odds=-110)
    early = _feed_pick(
        "Cowboys ML",
        stamp=EARLY_NFL_STAMP,
        odds=-105,
        date="2026-09-22",
        start=EARLY_NFL_START,
    )
    stale = _feed_pick("Bills ML", stamp=STALE_STAMP, odds=-137)
    cache_path = tmp_path / "2026-09-22.json"
    cache_path.write_text(
        json.dumps(
            {
                "date": "2026-09-22",
                "models": {},
                "external_feeds": {
                    "scores24_nfl": {
                        "ok": True,
                        "date": "2026-09-22",
                        "updatedAt": "2026-09-22T20:05:00Z",
                        "picks": [fresh, early, stale],
                    }
                },
            }
        )
    )

    bucket = optional.apply_optional_timeout_to_cache(
        cache_path,
        "scores24_nfl",
        "2026-09-22",
        180,
        checkpoint_dir=str(tmp_path / "empty"),
        now_iso=NOW,
    )
    by_pick = {row["pick"]: row for row in bucket["picks"]}
    assert by_pick["Chiefs ML"]["odds"] == -110
    assert by_pick["Chiefs ML"]["market_updated_at"] == FRESH_STAMP
    assert by_pick["Cowboys ML"]["odds"] == -105
    assert by_pick["Cowboys ML"]["market_updated_at"] == EARLY_NFL_STAMP
    assert by_pick["Bills ML"]["odds"] is None
    assert by_pick["Bills ML"]["market_priced"] is False
    assert by_pick["Bills ML"]["quote_withheld"] == "stale_or_missing_book_price"
    assert "market_updated_at" not in by_pick["Bills ML"]
    assert "timed out" in bucket["lastError"]


def test_timeout_does_not_keep_yesterdays_quote_on_todays_slate(tmp_path: Path):
    from scripts.scrapers import scores24_optional_publish as optional

    yesterday = _feed_pick(
        "Dolphins ML",
        stamp="2026-09-21T17:00:00Z",
        odds=-137,
        date="2026-09-21",
        start="2026-09-21T20:00:00Z",
    )
    cache_path = tmp_path / "2026-09-22.json"
    cache_path.write_text(
        json.dumps(
            {
                "date": "2026-09-22",
                "models": {},
                "external_feeds": {
                    "scores24_nfl": {
                        "ok": True,
                        "date": "2026-09-21",
                        "updatedAt": "2026-09-21T18:00:00Z",
                        "picks": [yesterday],
                    }
                },
            }
        )
    )

    bucket = optional.apply_optional_timeout_to_cache(
        cache_path,
        "scores24_nfl",
        "2026-09-22",
        180,
        checkpoint_dir=str(tmp_path / "empty"),
        now_iso=NOW,
    )
    assert bucket["date"] == "2026-09-21"
    assert bucket["picks"][0]["pick"] == "Dolphins ML"
    assert bucket["picks"][0]["odds"] is None
    assert bucket["lastAttemptDate"] == "2026-09-22"


def test_forebet_cloudflare_empty_keeps_newer_fresh_quote_and_drops_stale():
    fresh = _feed_pick("Packers ML", stamp=FRESH_STAMP, odds=-115)
    fresh["source"] = "ForebetNFL"
    stale = _feed_pick("Bears ML", stamp=STALE_STAMP, odds=-137)
    stale["source"] = "ForebetNFL"
    previous = {
        "ok": True,
        "date": "2026-09-22",
        "updatedAt": "2026-09-22T20:05:00Z",
        "picks": [fresh, stale],
    }
    blocked = {
        "ok": False,
        "date": "2026-09-22",
        "picks": [],
        "error": "ForebetNFL: listing fetch blocked by Cloudflare",
    }

    bucket = _record_feed_attempt(previous, blocked, "2026-09-22", NOW)
    by_pick = {row["pick"]: row for row in bucket["picks"]}
    assert by_pick["Packers ML"]["odds"] == -115
    assert by_pick["Bears ML"]["odds"] is None
    assert bucket["empty_feed_preserved_fresh_quotes"] is True
    assert "Cloudflare" in bucket["lastError"]

    blank = {
        "ok": True,
        "date": "2026-09-22",
        "picks": [],
        "error": "listing fetch blocked by Cloudflare",
        "meta": {"blockedUrls": 1, "officialMatchups": 2, "listedRows": 0},
    }
    kept = _record_feed_attempt(previous, blank, "2026-09-22", NOW)
    assert {row["pick"]: row["odds"] for row in kept["picks"]}["Packers ML"] == -115
    assert kept["picks"][0]["odds"] != 0


def test_empty_merge_does_not_replace_a_fresh_quote_with_nothing(tmp_path: Path):
    from scripts.merge_external_feed_cache_payload import merge_payload

    fresh = _feed_pick("Packers ML", stamp=FRESH_STAMP, odds=-115)
    fresh["source"] = "ForebetNFL"
    fresh["matchup"] = "Bears @ Packers"
    fresh["game"] = "Bears @ Packers"
    date = "2026-09-22"
    current = {
        "date": date,
        "models": {
            "forebet_nfl": {
                "ok": True,
                "date": date,
                "updatedAt": "2026-09-22T20:05:00Z",
                "picks": [fresh],
            }
        },
        "external_feeds": {
            "forebet_nfl": {
                "ok": True,
                "date": date,
                "updatedAt": "2026-09-22T20:05:00Z",
                "picks": [fresh],
            }
        },
    }
    (tmp_path / f"{date}.json").write_text(json.dumps(current))
    generated = {
        "date": date,
        "updatedAt": NOW,
        "models": {
            "forebet_nfl": {
                "ok": False,
                "date": date,
                "lastAttemptAt": NOW,
                "picks": [],
                "error": "ForebetNFL: listing fetch blocked by Cloudflare",
            }
        },
        "external_feeds": {
            "forebet_nfl": {
                "ok": False,
                "date": date,
                "lastAttemptAt": NOW,
                "picks": [],
                "error": "ForebetNFL: listing fetch blocked by Cloudflare",
            }
        },
    }

    merged = merge_payload(generated, tmp_path)
    for container in ("models", "external_feeds"):
        picks = merged[container]["forebet_nfl"]["picks"]
        assert picks[0]["odds"] == -115
        assert picks[0]["market_updated_at"] == FRESH_STAMP
        assert merged[container]["forebet_nfl"]["empty_feed_preserved_fresh_quotes"] is True
