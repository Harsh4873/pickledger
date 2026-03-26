from __future__ import annotations

# Approximate multi-year run-scoring park factors. These are meant to be stable
# priors for modeling, not exact year-specific leaderboard values. Revisit this
# table each offseason as stadium environments change.
PARK_FACTORS = {
    "Angel Stadium": 99,
    "Busch Stadium": 96,
    "Chase Field": 103,
    "Citi Field": 97,
    "Citizens Bank Park": 103,
    "Comerica Park": 97,
    "Coors Field": 118,
    "Daikin Park": 101,
    "Dodger Stadium": 100,
    "Fenway Park": 107,
    "George M. Steinbrenner Field": 101,
    "Globe Life Field": 98,
    "Great American Ball Park": 108,
    "Guaranteed Rate Field": 102,
    "Kauffman Stadium": 98,
    "loanDepot park": 94,
    "Minute Maid Park": 101,
    "Nationals Park": 101,
    "Oracle Park": 94,
    "Oriole Park at Camden Yards": 99,
    "Petco Park": 96,
    "PNC Park": 97,
    "Progressive Field": 101,
    "Rogers Centre": 103,
    "Sutter Health Park": 106,
    "Target Field": 99,
    "T-Mobile Park": 95,
    "Tropicana Field": 96,
    "Truist Park": 101,
    "Wrigley Field": 101,
    "Yankee Stadium": 105,
}


def get_park_factor(venue_name: str) -> int:
    return PARK_FACTORS.get(venue_name, 100)
