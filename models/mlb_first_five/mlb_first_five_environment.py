"""Park-factor fallback + wind/weather signal for the MLB First Five model.

The F5 model already learns a per-venue ``run_delta_per_team`` from recent
F5 totals at each park. That signal is good but has two real gaps:

1. **Thin sample fallback** — early in the season a venue may have only
   3-4 F5 games on record. The learned delta gets shrunk hard toward 0,
   even at parks with strong static reputations (Coors, Petco). Static
   park factors plug that hole as a small prior.

2. **Wind** — the F5 model has no weather input at all. Wind-out at
   Wrigley adds ~0.4 runs/team; wind-in suppresses by ~0.3. The MLB
   live-feed already includes a ``gameData.weather.wind`` field with
   strings like "10 mph, Out to RF" — we just have to parse it.

Both signals are tiny on their own but additive across an F5 slate;
together they meaningfully reduce projection bias at extreme parks
and wind days.
"""
from __future__ import annotations

import re
from typing import Optional

# MLB venue_id → run-environment factor. 1.00 = league-average run
# scoring; >1.00 hitter-friendly; <1.00 pitcher-friendly. Factors from
# multi-year Statcast park factors; only listed when materially off
# neutral, everything else uses 1.00. Venue ids verified against live
# statsapi ``gameData.venue`` payloads — the original table shipped with
# guessed ids that mislabeled most parks (e.g. id 1 is Angel Stadium,
# not Yankee Stadium; Petco is 2680, not 2889 which is Busch).
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

# Per-100-points-of-park-factor delta in F5 runs/team. A factor of 1.10
# (10% above neutral) corresponds to ≈ 0.10 extra runs/team in F5.
PARK_FACTOR_RUN_DELTA_PER_100PP = 1.0

# Below this many learned-venue games, prefer the static park-factor
# prior over the learned delta (or blend toward it).
LEARNED_THIN_SAMPLE_THRESHOLD = 20

# Wind direction patterns from MLB.com weather strings. Outward winds
# (Out to LF/CF/RF, "Outward") add scoring; inward winds suppress.
_WIND_OUTWARD_RE = re.compile(r"\bout\b|\boutward\b", re.IGNORECASE)
_WIND_INWARD_RE = re.compile(r"\bin\b(?!\s*to\s+the\s+park)|\binward\b", re.IGNORECASE)
_WIND_CROSS_RE = re.compile(r"l\s*to\s*r|r\s*to\s*l|left\s*to\s*right|right\s*to\s*left", re.IGNORECASE)
_WIND_MPH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mph", re.IGNORECASE)


def park_factor(venue_id: int | None) -> float:
    """Return the static run-environment factor for the venue (1.0 = neutral)."""
    if not venue_id:
        return 1.0
    try:
        vid = int(venue_id)
    except (TypeError, ValueError):
        return 1.0
    return float(STATIC_PARK_FACTORS.get(vid, 1.0))


def park_factor_run_delta(venue_id: int | None) -> float:
    """Static F5 runs/team delta implied by the park's static factor."""
    factor = park_factor(venue_id)
    return round((factor - 1.0) * PARK_FACTOR_RUN_DELTA_PER_100PP, 3)


def parse_wind(wind_string: str) -> dict:
    """Parse an MLB-style wind string like ``"10 mph, Out to RF"``.

    Returns ``{"mph": float, "direction": str}`` where direction is one of:
      - ``"out"`` — outward toward the field (boosts F5 runs)
      - ``"in"`` — inward from the field (suppresses)
      - ``"cross"`` — left-to-right or right-to-left (negligible effect)
      - ``"calm"`` — no measurable wind
      - ``""``    — unparseable
    """
    text = str(wind_string or "").strip()
    if not text:
        return {"mph": 0.0, "direction": ""}
    lowered = text.lower()
    if "calm" in lowered or "no wind" in lowered:
        return {"mph": 0.0, "direction": "calm"}

    mph_match = _WIND_MPH_RE.search(text)
    mph = float(mph_match.group(1)) if mph_match else 0.0

    direction = ""
    if _WIND_OUTWARD_RE.search(text):
        direction = "out"
    elif _WIND_INWARD_RE.search(text):
        direction = "in"
    elif _WIND_CROSS_RE.search(text):
        direction = "cross"

    return {"mph": mph, "direction": direction}


def wind_run_delta(wind_string: str) -> float:
    """Compute an F5 runs/team delta from a parsed wind string.

    Calibration notes:
    - Wind-out @ 15 mph at Wrigley historically adds ≈ 0.4 runs/team in F5.
    - Wind-in @ 15 mph suppresses by ≈ 0.3 runs/team.
    - Cross winds + light winds contribute ~0.

    Capped at ±0.45 runs/team so a single noisy weather string can't
    dominate the projection.
    """
    parsed = parse_wind(wind_string)
    mph = float(parsed.get("mph", 0.0))
    direction = parsed.get("direction") or ""

    if direction == "out" and mph >= 6:
        delta = 0.025 * (mph - 5)  # 10 mph → 0.125; 15 → 0.25; 20 → 0.375
    elif direction == "in" and mph >= 6:
        delta = -0.020 * (mph - 5)
    else:
        delta = 0.0

    if delta > 0.45:
        delta = 0.45
    elif delta < -0.45:
        delta = -0.45
    return round(delta, 3)


def blend_park_run_delta(
    learned_delta: float,
    learned_games: int,
    venue_id: int | None,
) -> dict:
    """Blend the learned per-venue delta with the static park-factor prior.

    When the learned sample is thin (< LEARNED_THIN_SAMPLE_THRESHOLD games),
    pull the projection toward the static prior. Returns the final delta to
    add per team to the F5 projection plus a breakdown for diagnostics.
    """
    static_delta = park_factor_run_delta(venue_id)
    if learned_games <= 0:
        return {
            "final_delta": static_delta,
            "learned_delta": 0.0,
            "static_delta": static_delta,
            "blend_weight_learned": 0.0,
            "park_factor": park_factor(venue_id),
        }

    # Linear ramp from 0% learned at 0 games to 100% learned at threshold.
    weight_learned = min(1.0, learned_games / float(LEARNED_THIN_SAMPLE_THRESHOLD))
    blended = (learned_delta * weight_learned) + (static_delta * (1.0 - weight_learned))
    return {
        "final_delta": round(blended, 3),
        "learned_delta": round(learned_delta, 3),
        "static_delta": static_delta,
        "blend_weight_learned": round(weight_learned, 3),
        "park_factor": park_factor(venue_id),
    }
