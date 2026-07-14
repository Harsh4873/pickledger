"""Boundaries for the current machine-learned player-prop ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ML_PROBABILITY_SOURCE = "player_props_ml_v1"
# The first snapshot of the four-model season/history consensus ranking era.
ML_ERA_FIRST_SNAPSHOT_AT = "2026-06-20T20:42:53.275777Z"


def _utc_timestamp(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_ml_era_pick(pick: dict[str, Any], fallback_timestamp: Any = None) -> bool:
    """Return whether a prop belongs to the post-retraining ML ledger."""
    if str(pick.get("probability_source") or "").strip() != ML_PROBABILITY_SOURCE:
        return False
    timestamp = _utc_timestamp(
        pick.get("ranking_updated_at")
        or pick.get("generated_at")
        or pick.get("created_at")
        or fallback_timestamp
    )
    cutoff = _utc_timestamp(ML_ERA_FIRST_SNAPSHOT_AT)
    return timestamp is not None and cutoff is not None and timestamp >= cutoff
