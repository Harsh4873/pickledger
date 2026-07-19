"""Park-factor and weather adjustments for the MLB Inning model.

Pre-patch the probability layer read ``venue.run_factor`` off the game
dict, but nothing ever populated it — every one of the 390 cached
production games ran with a neutral 1.0 park factor, Coors included.
Weather was fetched and displayed but never touched the probability.

This module gives the inning model the same static environment inputs
the First Five model already uses (`mlb_first_five_environment.py`),
re-expressed for scoreless probabilities instead of run deltas:

- ``park_run_factor`` — static venue_id → run-environment factor. The
  probability layer applies it as ``p ** factor`` (a Poisson scoreless
  probability scales as p^f when the run rate scales by f), so Coors at
  1.18 costs a 0.70 half-inning about 4pp instead of the ~11pp the old
  ``p / factor`` division would have charged.
- ``scoreless_weather_multiplier`` — a small multiplier on the
  half-inning scoreless probability from the live-feed weather strings.
  Wind-out and heat lower it, wind-in and cold raise it; a closed roof
  ignores wind entirely.
"""
from __future__ import annotations

import re
from typing import Any

# MLB venue_id → run-environment factor. 1.00 = league-average run
# scoring; >1.00 hitter-friendly; <1.00 pitcher-friendly. Factors from
# multi-year Statcast park factors; only listed when materially off
# neutral, everything else uses 1.00. The venue ids are verified against
# live statsapi ``gameData.venue`` payloads — the F5 model's original
# table shipped with guessed ids that mislabeled most parks (e.g. id 1
# is Angel Stadium, not Yankee Stadium).
STATIC_PARK_FACTORS: dict[int, float] = {
    19:   1.18,  # Coors Field (COL) — highest in MLB by a wide margin
    5325: 1.07,  # Globe Life Field (TEX)
    3313: 1.06,  # Yankee Stadium (NYY) — short porch in RF
    2602: 1.05,  # Great American Ball Park (CIN)
    3:    1.04,  # Fenway Park (BOS)
    17:   1.04,  # Wrigley Field (CHC) — wind-dependent
    2:    1.03,  # Oriole Park at Camden Yards (BAL)
    2681: 1.03,  # Citizens Bank Park (PHI)
    4:    1.02,  # Rate Field (CWS)
    14:   1.02,  # Rogers Centre (TOR)
    15:   1.02,  # Chase Field (ARI)
    7:    0.99,  # Kauffman Stadium (KC)
    5:    0.98,  # Progressive Field (CLE)
    22:   0.98,  # Dodger Stadium (LAD)
    31:   0.97,  # PNC Park (PIT)
    12:   0.97,  # Tropicana Field (TB)
    2889: 0.97,  # Busch Stadium (STL)
    3289: 0.97,  # Citi Field (NYM)
    4705: 0.97,  # Truist Park (ATL)
    1:    0.96,  # Angel Stadium (LAA)
    680:  0.95,  # T-Mobile Park (SEA)
    2394: 0.94,  # Comerica Park (DET)
    4169: 0.92,  # loanDepot park (MIA)
    2395: 0.92,  # Oracle Park (SF)
    2680: 0.91,  # Petco Park (SD) — most pitcher-friendly
}

_WIND_OUTWARD_RE = re.compile(r"\bout\b|\boutward\b", re.IGNORECASE)
_WIND_INWARD_RE = re.compile(r"\bin\b(?!\s*to\s+the\s+park)|\binward\b", re.IGNORECASE)
_WIND_MPH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mph", re.IGNORECASE)
_TEMP_RE = re.compile(r"(-?\d+(?:\.\d+)?)")

# Roofed parks report wind that never reaches the field.
_ROOF_CLOSED_RE = re.compile(r"roof\s*closed|dome", re.IGNORECASE)

# Bounds on the combined weather multiplier so a noisy weather string
# can never dominate a half-inning probability.
WEATHER_MULTIPLIER_FLOOR = 0.95
WEATHER_MULTIPLIER_CEIL = 1.03


def park_run_factor(venue_id: Any) -> float:
    """Return the static run-environment factor for the venue (1.0 = neutral)."""
    try:
        vid = int(venue_id)
    except (TypeError, ValueError):
        return 1.0
    return float(STATIC_PARK_FACTORS.get(vid, 1.0))


def scoreless_weather_multiplier(weather: Any) -> tuple[float, dict[str, Any]]:
    """Multiplier on a half-inning scoreless probability from game weather.

    Calibration: NRFI research puts wind-out + heat at roughly +5% on the
    probability of a first-inning run, so each individual signal here is
    kept to a 1-3% nudge and the combined multiplier is clamped to
    [0.95, 1.03]. Returns ``(multiplier, detail)`` where detail carries
    the parsed inputs for the pick payload diagnostics.
    """
    weather = weather if isinstance(weather, dict) else {}
    wind_text = str(weather.get("wind") or "")
    condition = str(weather.get("condition") or "")
    temp_match = _TEMP_RE.search(str(weather.get("temp") or ""))
    temp = float(temp_match.group(1)) if temp_match else None

    roof_closed = bool(_ROOF_CLOSED_RE.search(condition) or _ROOF_CLOSED_RE.search(wind_text))
    mph_match = _WIND_MPH_RE.search(wind_text)
    mph = float(mph_match.group(1)) if mph_match else 0.0
    direction = ""
    if not roof_closed and mph > 0:
        if _WIND_OUTWARD_RE.search(wind_text):
            direction = "out"
        elif _WIND_INWARD_RE.search(wind_text):
            direction = "in"

    multiplier = 1.0
    if direction == "out" and mph >= 6:
        # 10 mph → -1.0%; 15 mph → -2.0%; capped at -3%.
        multiplier *= 1.0 - min(0.03, 0.002 * (mph - 5))
    elif direction == "in" and mph >= 6:
        multiplier *= 1.0 + min(0.02, 0.0015 * (mph - 5))

    if temp is not None:
        if temp >= 95:
            multiplier *= 0.985
        elif temp >= 85:
            multiplier *= 0.99
        elif temp <= 55:
            multiplier *= 1.01

    multiplier = max(WEATHER_MULTIPLIER_FLOOR, min(WEATHER_MULTIPLIER_CEIL, multiplier))
    detail = {
        "wind_mph": mph,
        "wind_direction": direction or ("roof_closed" if roof_closed else "none"),
        "temp_f": temp,
        "scoreless_multiplier": round(multiplier, 4),
    }
    return multiplier, detail
