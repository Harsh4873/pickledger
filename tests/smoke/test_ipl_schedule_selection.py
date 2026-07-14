from __future__ import annotations

from datetime import date, datetime, timezone


def _record(match_id: str, team1: str, team2: str, day: date, hour_utc: int, status: str):
    return {
        "match_id": match_id,
        "team1": team1,
        "team2": team2,
        "venue": "Test Ground",
        "match_date": day,
        "match_start_utc": datetime(day.year, day.month, day.day, hour_utc, 0, tzinfo=timezone.utc),
        "status": status,
    }


def test_ipl_schedule_skips_completed_today_for_next_future_start():
    from ipl.data.live_feed import _select_schedule_records

    now = datetime(2026, 5, 10, 18, 0, tzinfo=timezone.utc)
    records = [
        _record("2493", "Chennai Super Kings", "Lucknow Super Giants", date(2026, 5, 10), 10, "Post"),
        _record("2494", "Royal Challengers Bengaluru", "Mumbai Indians", date(2026, 5, 10), 14, "Post"),
        _record("2495", "Punjab Kings", "Delhi Capitals", date(2026, 5, 11), 14, "UpComing"),
        _record("2496", "Gujarat Titans", "Sunrisers Hyderabad", date(2026, 5, 12), 14, "UpComing"),
    ]

    selected = _select_schedule_records(records, now)

    assert [record["match_id"] for record in selected] == ["2495"]
    assert selected[0]["team1"] == "Punjab Kings"


def test_ipl_schedule_prefers_later_today_if_not_started_yet():
    from ipl.data.live_feed import _select_schedule_records

    now = datetime(2026, 5, 10, 12, 0, tzinfo=timezone.utc)
    records = [
        _record("early", "Chennai Super Kings", "Lucknow Super Giants", date(2026, 5, 10), 10, "Post"),
        _record("late", "Royal Challengers Bengaluru", "Mumbai Indians", date(2026, 5, 10), 14, "UpComing"),
        _record("tomorrow", "Punjab Kings", "Delhi Capitals", date(2026, 5, 11), 14, "UpComing"),
    ]

    selected = _select_schedule_records(records, now)

    assert [record["match_id"] for record in selected] == ["late"]


def test_ipl_schedule_serializes_start_time_utc():
    from ipl.data.live_feed import _serialize_schedule_records

    payload = _serialize_schedule_records(
        [_record("2495", "Punjab Kings", "Delhi Capitals", date(2026, 5, 11), 14, "UpComing")]
    )

    assert payload[0]["match_start_utc"] == "2026-05-11T14:00:00+00:00"
