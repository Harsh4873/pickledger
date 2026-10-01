"""Identify forecast rows whose final score has a binary settlement."""

from __future__ import annotations

import math
from typing import Any, Mapping


def binary_settlement_supported(record: Mapping[str, Any]) -> bool:
    """MLS quarter lines need fractional returns that the grader cannot record.

    A full win/loss/push label and the current frozen ROI evaluator are valid
    for whole and half goal lines. Leave other MLS total/handicap rows pending
    until a fractional settlement contract is implemented.
    """

    if record.get("model_key") != "mls" or record.get("market") not in {"total", "spread"}:
        return True
    snapshot = record.get("pregame_snapshot")
    snapshot = snapshot if isinstance(snapshot, Mapping) else record
    price = record.get("price")
    price = price if isinstance(price, Mapping) else {}
    value = record.get("line") if "line" in record else snapshot.get("line")
    if value is None:
        value = price.get("line")
    try:
        line = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(line) and abs(2 * line - round(2 * line)) < 1e-9
