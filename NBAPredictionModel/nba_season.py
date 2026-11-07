"""NBA season labels derived from the date being modeled, so nothing pins a stale season."""
from datetime import date, datetime


def nba_season_label(as_of: str | date | None = None) -> str:
    """
    Season label such as '2026-27' for a date. A season opens in October, so August
    onward belongs to the season that ends the following year.
    """
    if isinstance(as_of, datetime):
        day = as_of.date()
    elif isinstance(as_of, date):
        day = as_of
    else:
        try:
            day = datetime.strptime(str(as_of or "")[:10], "%Y-%m-%d").date()
        except ValueError:
            day = date.today()
    start = day.year if day.month >= 8 else day.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def previous_season_label(season_label: str) -> str:
    """'2026-27' -> '2025-26'."""
    start = int(str(season_label)[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def espn_season_year(season_label: str) -> int:
    """ESPN names a season by the year it ends: '2026-27' is 2027."""
    return int(str(season_label)[:4]) + 1
