from __future__ import annotations

from scripts.profit_desk_notify import build_notification


def artifact(date: str, picks: list[dict]) -> dict:
    return {"date": date, "portfolio": {"live": picks}}


def live_pick(pick_id: str, name: str = "Ace Over 3.5 Strikeouts") -> dict:
    return {
        "id": pick_id,
        "pick": name,
        "stakeUnits": 0.5,
        "lane": "value",
        "sport": "MLB",
        "oddsAmerican": -166,
        "startTime": "2026-07-12T18:10Z",
    }


def test_notifies_every_new_live_pick_once():
    current = artifact("2026-07-12", [live_pick("a"), live_pick("b", "Deuce Under 4.5 Ks")])
    message = build_notification(current, None)
    assert message is not None
    assert "2 new live picks" in message
    assert "Ace Over 3.5 Strikeouts — 0.5u VALUE at -166" in message
    assert "Deuce Under 4.5 Ks" in message
    assert "https://harsh.bet/pickledger/" in message

    # A rebuild of the same slate with the same picks must stay silent.
    assert build_notification(current, current) is None

    # A refresh that adds one pick announces only the addition.
    grown = artifact("2026-07-12", [live_pick("a"), live_pick("b"), live_pick("c", "Trey ML")])
    incremental = build_notification(grown, current)
    assert incremental is not None
    assert "1 new live pick" in incremental
    assert "Trey ML" in incremental
    assert "Ace Over 3.5 Strikeouts" not in incremental


def test_new_slate_date_resets_the_announced_set():
    yesterday = artifact("2026-07-11", [live_pick("a")])
    today = artifact("2026-07-12", [live_pick("a")])
    message = build_notification(today, yesterday)
    assert message is not None
    assert "2026-07-12" in message


def test_sit_out_slates_and_missing_artifacts_send_nothing():
    assert build_notification(artifact("2026-07-12", []), None) is None
    assert build_notification(None, None) is None
    assert build_notification({"portfolio": {"live": [live_pick("a")]}}, None) is None
