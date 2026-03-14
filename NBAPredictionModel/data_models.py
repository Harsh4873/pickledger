from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class Player:
    id: int
    name: str
    team_name: str
    position: str  # e.g., 'PG', 'SG', 'SF', 'PF', 'C'
    status: str = "Active"  # Active, Probable, Questionable, Doubtful, Out
    usage_rate: float = 0.0

@dataclass
class TeamStats:
    net_rating: float
    off_rating_10: float
    def_rating_10: float
    ts_pct: float
    reb_pct: float
    pace: float
    last_10_win_pct: float
    is_b2b_second_leg: bool
    is_3_in_4_nights: bool
    season_win_pct: float

@dataclass
class Venue:
    name: str

@dataclass
class Team:
    id: int
    name: str
    is_home: bool
    team_stats: TeamStats
    lineup: List[Player]
    key_stars_out: bool = False  # Track if any star with usage > 25% is out
    starting_center_out: bool = False
    motivation_elimination_game: bool = False
    rotation_players_out: int = 0

@dataclass
class GameContext:
    date: str
    venue: Venue
    home_team: Team
    away_team: Team
    h2h_home_win_pct_2yr: float  # H2H over last 2 years
