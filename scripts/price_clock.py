"""Time checks for an observed pregame sportsbook quote."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


QUOTE_FIELDS = ("market_updated_at", "market_retrieved_at", "odds_updated_at", "price_updated_at")


def aware_time(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def observed_quote_timing(
    row: Mapping[str, Any], *, published_at: Any, start_at: Any,
) -> str | None:
    """Return an exclusion reason, or None for a timely observed quote."""

    publication = aware_time(published_at)
    start = aware_time(start_at)
    quote = next((aware_time(row.get(key)) for key in QUOTE_FIELDS if row.get(key)), None)
    if publication is None or start is None:
        return "missing_publication_or_start_timestamp"
    if quote is None:
        return "missing_quote_timestamp"
    if quote >= start or publication >= start:
        return "post_start"
    age_seconds = (publication - quote).total_seconds()
    if age_seconds < -300:
        return "quote_after_publication"
    if age_seconds > 86400:
        return "stale_quote"
    return None
