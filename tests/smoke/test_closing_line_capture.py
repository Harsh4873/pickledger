from datetime import datetime, timezone

from scripts.capture_closing_lines import _default_slate_date


def test_default_slate_date_stays_on_central_day_after_utc_midnight():
    # 00:30 UTC is still 19:30 on the prior Central date during daylight time.
    now = datetime(2026, 9, 6, 0, 30, tzinfo=timezone.utc)
    assert _default_slate_date(now) == "2026-09-05"
