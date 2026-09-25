"""Deterministic metrics shared by in-house model evaluations."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable


def clustered_roi_interval(
    rows: Iterable[tuple[str, float, float]], *, samples: int = 2000,
) -> dict[str, float | int | None]:
    """Bootstrap unit ROI by event, keeping correlated picks together."""

    groups: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    count = 0
    for event, stake, profit in rows:
        groups[event][0] += stake
        groups[event][1] += profit
        count += 1
    keys = sorted(groups)
    if len(keys) < 2 or count < 2:
        return {"lower_95": None, "upper_95": None, "event_clusters": len(keys)}
    values = [groups[key] for key in keys]
    rng = random.Random(20260922)
    estimates: list[float] = []
    for _ in range(samples):
        drawn = [values[rng.randrange(len(values))] for _ in values]
        stake = sum(item[0] for item in drawn)
        if stake > 0:
            estimates.append(sum(item[1] for item in drawn) / stake)
    estimates.sort()
    if not estimates:
        return {"lower_95": None, "upper_95": None, "event_clusters": len(keys)}
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return {
        "lower_95": round(lower, 6),
        "upper_95": round(upper, 6),
        "event_clusters": len(keys),
    }


def binary_score(rows: Iterable[tuple[float, int]], *, bins: int = 10) -> dict[str, float | int | None]:
    items = list(rows)
    if not items:
        return {"samples": 0, "wins": 0, "hit_rate": None, "brier": None, "ece": None}
    wins = sum(outcome for _, outcome in items)
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, outcome in items:
        buckets[min(bins - 1, int(probability * bins))].append((probability, outcome))
    ece = sum(
        len(bucket) / len(items)
        * abs(sum(p for p, _ in bucket) / len(bucket) - sum(y for _, y in bucket) / len(bucket))
        for bucket in buckets if bucket
    )
    return {
        "samples": len(items),
        "wins": wins,
        "hit_rate": round(wins / len(items), 6),
        "brier": round(sum((p - y) ** 2 for p, y in items) / len(items), 6),
        "ece": round(ece, 6),
    }
