from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass
class Player:
    id: int
    name: str
    team_name: str
    position: str
    status: str = "Active"  # Or IL, Day-to-Day

@dataclass
class PitcherStats:
    era: float
    fip: float
    whip: float
    last_5_starts_summary: str
    days_rest: int
    home_split_era: float
    away_split_era: float
    woba_vs_l: float
    woba_vs_r: float
    pitches_per_start_avg: int

@dataclass
class TeamStats:
    ops: float
    woba: float
    wrc_plus: float
    last_10_runs_avg: float
    bullpen_pitches_yesterday: int
    travel_fatigue: bool  # Cross-country yesterday
    consecutive_games: int # 3rd game in 3 nights?
    home_win_pct: float
    away_win_pct: float
    season_win_pct: float
    last_30_days_win_pct: float

@dataclass
class Weather:
    temp_f: float
    wind_speed_mph: float
    wind_direction: str  # 'out', 'in', 'cross'
    is_dome: bool

@dataclass
class Venue:
    name: str
    park_factor_runs: int
    elevation_ft: int

@dataclass
class Team:
    id: int
    name: str
    is_home: bool
    starter: Player
    starter_stats: PitcherStats
    team_stats: TeamStats
    lineup: List[Player]

@dataclass
class GameContext:
    date: str
    venue: Venue
    weather: Weather
    home_team: Team
    away_team: Team
    h2h_home_win_pct_3yr: float  # H2H at this venue last 3 years
