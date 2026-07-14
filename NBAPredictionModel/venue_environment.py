"""NBA venue environment lookup: timezone + altitude.

Two real signals the pre-patch NBA model didn't have:

1. **Cross-country travel** — a west-coast team flying to the east coast
   for a 7pm local tip is playing at 4pm body-clock. Empirically those
   teams shoot worse from 3 and lose ~1.5 pts/game on average vs their
   neutral-site expectation.

2. **Altitude** — Denver and Utah home games run hotter (more pace,
   higher scoring, visiting team aerobic disadvantage). Modern data
   has the Denver/Utah HCA at +1.0-1.5 pts above the league baseline.

Both are tiny per game but they bias every projection at those venues.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VenueEnv:
    timezone_offset_hours: int  # vs UTC; negative for US zones (e.g. -5 = EST)
    altitude_feet: int


# Approximate venue environment for the 30 NBA arenas. Timezone offsets are
# the venue's local standard offset (no DST adjustment — the signal we care
# about is the relative difference, which is the same DST or not).
NBA_VENUE_ENV: dict[str, VenueEnv] = {
    # Eastern Time (-5)
    "Hawks":         VenueEnv(timezone_offset_hours=-5, altitude_feet=1050),
    "Celtics":       VenueEnv(timezone_offset_hours=-5, altitude_feet=21),
    "Hornets":       VenueEnv(timezone_offset_hours=-5, altitude_feet=748),
    "Cavaliers":     VenueEnv(timezone_offset_hours=-5, altitude_feet=653),
    "Pistons":       VenueEnv(timezone_offset_hours=-5, altitude_feet=600),
    "Pacers":        VenueEnv(timezone_offset_hours=-5, altitude_feet=715),
    "Heat":          VenueEnv(timezone_offset_hours=-5, altitude_feet=6),
    "Knicks":        VenueEnv(timezone_offset_hours=-5, altitude_feet=33),
    "Nets":          VenueEnv(timezone_offset_hours=-5, altitude_feet=33),
    "Magic":         VenueEnv(timezone_offset_hours=-5, altitude_feet=82),
    "76ers":         VenueEnv(timezone_offset_hours=-5, altitude_feet=39),
    "Raptors":       VenueEnv(timezone_offset_hours=-5, altitude_feet=251),
    "Wizards":       VenueEnv(timezone_offset_hours=-5, altitude_feet=25),
    # Central Time (-6)
    "Bulls":         VenueEnv(timezone_offset_hours=-6, altitude_feet=594),
    "Bucks":         VenueEnv(timezone_offset_hours=-6, altitude_feet=617),
    "Grizzlies":     VenueEnv(timezone_offset_hours=-6, altitude_feet=337),
    "Pelicans":      VenueEnv(timezone_offset_hours=-6, altitude_feet=10),
    "Spurs":         VenueEnv(timezone_offset_hours=-6, altitude_feet=650),
    "Mavericks":     VenueEnv(timezone_offset_hours=-6, altitude_feet=430),
    "Rockets":       VenueEnv(timezone_offset_hours=-6, altitude_feet=49),
    "Timberwolves":  VenueEnv(timezone_offset_hours=-6, altitude_feet=830),
    "Thunder":       VenueEnv(timezone_offset_hours=-6, altitude_feet=1198),
    # Mountain Time (-7)
    "Nuggets":       VenueEnv(timezone_offset_hours=-7, altitude_feet=5280),  # Denver — mile high
    "Jazz":          VenueEnv(timezone_offset_hours=-7, altitude_feet=4226),  # Utah — high altitude
    # Pacific Time (-8)
    "Suns":          VenueEnv(timezone_offset_hours=-7, altitude_feet=1086),  # AZ skips DST
    "Trail Blazers": VenueEnv(timezone_offset_hours=-8, altitude_feet=50),
    "Kings":         VenueEnv(timezone_offset_hours=-8, altitude_feet=30),
    "Lakers":        VenueEnv(timezone_offset_hours=-8, altitude_feet=305),
    "Clippers":      VenueEnv(timezone_offset_hours=-8, altitude_feet=305),
    "Warriors":      VenueEnv(timezone_offset_hours=-8, altitude_feet=43),
}


# An "altitude park" gives the home team a measurable extra HCA.
ALTITUDE_HOME_BONUS_FT_THRESHOLD = 4000


def lookup_venue_env(team_nickname: str) -> Optional[VenueEnv]:
    """Best-effort lookup: tries the team nickname, falls back to None."""
    if not team_nickname:
        return None
    # Strip any leading city: callers may pass "LA Lakers" or "Trail Blazers".
    parts = str(team_nickname).strip().split()
    for end in range(len(parts), 0, -1):
        key = " ".join(parts[-end:])
        if key in NBA_VENUE_ENV:
            return NBA_VENUE_ENV[key]
    return None


def timezone_delta_hours(home_team: str, away_team: str) -> int:
    """Hours of timezone shift the away team is making.

    Positive when the away team is traveling east (worse for body clock at
    night-game start times); negative when traveling west (slightly easier).
    Returns 0 if either venue is unknown.
    """
    home = lookup_venue_env(home_team)
    away = lookup_venue_env(away_team)
    if home is None or away is None:
        return 0
    return home.timezone_offset_hours - away.timezone_offset_hours


def travel_fatigue_adjustment(home_team: str, away_team: str) -> tuple[float, str]:
    """Probability adjustment FROM THE HOME TEAM'S PERSPECTIVE for the
    travel/timezone shift the away team is suffering.

    Returns (adj, reason). Empirical magnitudes:
      - Away team going +3 hours east: +1.5% home prob
      - +2 hours east: +0.8% home prob
      - +1 hour east: +0.3% home prob
      - 0 or west travel: 0%
    Capped at +2.0% so a single signal can't dominate.
    """
    delta = timezone_delta_hours(home_team, away_team)
    if delta <= 0:
        return 0.0, ""
    # Each eastward-hour ≈ 0.5% home prob bump, capped at +2%.
    adj = min(0.020, delta * 0.005)
    return adj, f"Away east travel +{delta}h vs body clock +{adj*100:.1f}%"


def altitude_home_bonus(home_team: str) -> tuple[float, str]:
    """Home altitude advantage — Denver / Utah only.

    Returns (adj, reason). +1.5% home prob for venues above the threshold,
    0% otherwise.
    """
    venue = lookup_venue_env(home_team)
    if venue is None:
        return 0.0, ""
    if venue.altitude_feet < ALTITUDE_HOME_BONUS_FT_THRESHOLD:
        return 0.0, ""
    return 0.015, f"Altitude home court ({venue.altitude_feet:,} ft) +1.5%"
