"""Decide whether an observed book price is fresh enough to ship.

A price ships when the book stamped it and that stamp is strictly before
the start. A missing stamp does not ship. A stamp at or after the start
does not ship. The 24 hour limit is the age of the stamp against the
publish clock, not the gap from the stamp to kickoff. An NFL number posted
two days before kickoff is still the book's price. Refresh clocks, capture
clocks, and empty strings are not book timestamps. This module never fills
in a replacement price.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


MAX_FRESH_OBSERVED_PRICE_HOURS = 24.0

BOOK_TIMESTAMP_FIELDS = ("book_updated_at", "market_updated_at")

# Fields that carry a number or a clock for a quote. Stripping a stale quote
# removes these and leaves odds null. It does not write a new number.
QUOTE_FIELDS = (
    "odds",
    "market_over_odds",
    "market_under_odds",
    "market_home_odds",
    "market_away_odds",
    "market_draw_odds",
    "selected_odds",
    "opposite_odds",
    "market_line",
    "market_no_vig_selected_probability",
    "market_implied_probability",
    "book_updated_at",
    "market_updated_at",
    "pricing_type",
    "odds_source",
    "price_source",
    "market_priced",
)

_NON_EXECUTABLE = ("assumed", "synthetic", "proxy", "fallback", "default", "estimated")


def parse_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _american(value: Any) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == 0 or -100.0 < number < 100.0:
        return None
    return int(round(number))


def _non_executable(pick: dict[str, Any]) -> bool:
    markers = " ".join(
        str(pick.get(key) or "").lower()
        for key in ("pricing_type", "price_source", "odds_source", "line_source", "market_source")
    )
    return any(token in markers for token in _NON_EXECUTABLE)


def book_timestamp(pick: dict[str, Any]) -> datetime | None:
    """Return the book's own stamp. Capture and publish clocks do not count."""
    for field in BOOK_TIMESTAMP_FIELDS:
        parsed = parse_timestamp(pick.get(field))
        if parsed is not None:
            return parsed
    return None


def fresh_observed_book_price(pick: dict[str, Any], *, now: datetime | None = None) -> bool:
    """True when this pick carries a pregame book price.

    The comparison against the start is only ``stamp < start``. It does not
    require the stamp to fall inside the 24 hours before kickoff. When
    ``now`` is the publish clock, the stamp also has to be at most 24 hours
    old as of that clock. A missing stamp, a post-start stamp, or a
    synthetic price is not fresh. Nothing here invents a price or a stamp.
    """
    if pick.get("market_priced") is not True:
        return False
    if _non_executable(pick):
        return False
    if _american(pick.get("odds")) is None:
        return False
    stamped = book_timestamp(pick)
    start = parse_timestamp(pick.get("start_time") or pick.get("game_start_time"))
    if stamped is None or start is None or stamped >= start:
        return False
    if now is None:
        return True
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_from_now = (now.astimezone(timezone.utc) - stamped).total_seconds() / 3600.0
    return 0.0 <= age_from_now <= MAX_FRESH_OBSERVED_PRICE_HOURS


def pick_has_quote(pick: dict[str, Any]) -> bool:
    if _american(pick.get("odds")) is not None:
        return True
    return any(
        pick.get(field) not in (None, "")
        for field in (
            "market_over_odds",
            "market_under_odds",
            "market_home_odds",
            "market_away_odds",
            "selected_odds",
            "opposite_odds",
        )
    )


def quote_is_current(
    pick: dict[str, Any],
    *,
    now: datetime,
    slate_date: str,
) -> bool:
    """A quote may stay in today's pick path only if it is still a fresh book price.

    Yesterday's row can look fresh against its own start and still be a stale
    quote on today's slate. The book stamp also has to fall inside the last
    24 hours.
    """
    if not fresh_observed_book_price(pick, now=now):
        return False
    pick_date = str(pick.get("date") or "").strip()
    if slate_date and pick_date and pick_date != slate_date:
        return False
    return True


def scrub_stale_quote(
    pick: dict[str, Any],
    *,
    now: datetime,
    slate_date: str,
    drop_untimestamped: bool,
) -> dict[str, Any]:
    """Drop a quote that must not stay in the pick path. Leave the row otherwise.

    Returns the same object when the quote can stay, so callers that compare
    untouched rows still see them. A removed quote is null, not a guessed price.
    """
    if not pick_has_quote(pick):
        return pick
    if quote_is_current(pick, now=now, slate_date=slate_date):
        return pick
    if book_timestamp(pick) is None and not drop_untimestamped:
        return pick
    revised = dict(pick)
    for field in QUOTE_FIELDS:
        if field == "odds":
            revised["odds"] = None
        elif field == "market_priced":
            revised["market_priced"] = False
        else:
            revised.pop(field, None)
    revised["quote_withheld"] = "stale_or_missing_book_price"
    return revised


def scrub_picks(
    picks: list[Any],
    *,
    now: datetime,
    slate_date: str,
    drop_untimestamped: bool,
) -> list[Any]:
    scrubbed: list[Any] = []
    for pick in picks:
        if isinstance(pick, dict):
            scrubbed.append(
                scrub_stale_quote(
                    pick,
                    now=now,
                    slate_date=slate_date,
                    drop_untimestamped=drop_untimestamped,
                )
            )
        else:
            scrubbed.append(pick)
    return scrubbed


def any_current_quote(picks: list[Any], *, now: datetime, slate_date: str) -> bool:
    return any(
        isinstance(pick, dict) and quote_is_current(pick, now=now, slate_date=slate_date)
        for pick in picks
    )


def copy_current_quote(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Copy a kept quote onto a row that arrived without one. No new numbers."""
    revised = dict(target)
    for field in QUOTE_FIELDS:
        if field in source:
            revised[field] = source[field]
    revised.pop("quote_withheld", None)
    return revised
