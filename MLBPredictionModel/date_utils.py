from __future__ import annotations

import os
from datetime import date, datetime
from zoneinfo import ZoneInfo


MLB_TIMEZONE_NAME = os.environ.get("PICKLEDGER_MLB_TIMEZONE", "America/Chicago")


def get_mlb_slate_date() -> date:
    """Return the canonical MLB slate date PickLedger uses as "today"."""
    return datetime.now(ZoneInfo(MLB_TIMEZONE_NAME)).date()
