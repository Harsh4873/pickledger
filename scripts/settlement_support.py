"""Identify forecast rows whose final score has a binary settlement."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


def settlement_exclusion_reason(record: Mapping[str, Any]) -> str | None:
    """Explain why an MLS total or handicap cannot use binary grading.

    A quarter-goal wager splits its stake across adjacent half/whole lines.
    The cache grader and frozen ROI/calibration paths only represent one
    win/loss/push result, so these rows must remain research only.
    """

    if record.get("model_key") != "mls" or record.get("market") not in {"total", "spread"}:
        return None
    snapshot = record.get("pregame_snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else record
    price = record.get("price")
    price = price if isinstance(price, Mapping) else {}
    values = [context.get("line") for context in (record, snapshot, price)
              if context.get("line") is not None]
    pick = str(snapshot.get("pick") or record.get("pick") or "").strip()
    if pick:
        pattern = (r"^(?:Over|Under)\s+(\d+(?:\.\d+)?)\b" if record.get("market") == "total"
                   else r"\s([+-]\d+(?:\.\d+)?)\s*(?:\(|$)")
        match = re.search(pattern, pick, flags=re.IGNORECASE)
        if not match:
            return "unrecognized_mls_selection"
        values.append(match.group(1))
    if not values:
        return "missing_mls_line"
    try:
        lines = [float(value) for value in values]
    except (TypeError, ValueError):
        return "invalid_mls_line"
    if not all(math.isfinite(line) for line in lines):
        return "invalid_mls_line"
    if any(abs(line - lines[0]) > 1e-9 for line in lines[1:]):
        return "conflicting_mls_line"
    return None if abs(2 * lines[0] - round(2 * lines[0])) < 1e-9 else "unsupported_fractional_settlement"


def binary_settlement_supported(record: Mapping[str, Any]) -> bool:
    return settlement_exclusion_reason(record) is None
