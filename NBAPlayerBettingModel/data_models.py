from dataclasses import dataclass, field


@dataclass(slots=True)
class PlayerSeasonStats:
    player_id: int
    player_name: str
    team_id: int
    team_name: str
    team_abbreviation: str
    opponent_team_id: int
    opponent_team_name: str
    opponent_team_abbreviation: str
    game_id: str
    away_team_name: str
    home_team_name: str
    position: str
    position_bucket: str
    is_home: bool
    games_played: int
    mp_per_game: float
    fg_per_game: float
    fga_per_game: float
    fg_percent: float
    x3p_per_game: float
    x3pa_per_game: float
    x3p_percent: float
    x2p_per_game: float
    x2pa_per_game: float
    x2p_percent: float
    e_fg_percent: float
    ft_per_game: float
    fta_per_game: float
    ft_percent: float
    orb_per_game: float
    drb_per_game: float
    trb_per_game: float
    ast_per_game: float
    stl_per_game: float
    blk_per_game: float
    tov_per_game: float
    usage_rate: float
    points_per_game: float
    rebounds_per_game: float
    assists_per_game: float
    last10_points: float
    last10_rebounds: float
    last10_assists: float
    home_points: float | None = None
    away_points: float | None = None
    home_rebounds: float | None = None
    away_rebounds: float | None = None
    home_assists: float | None = None
    away_assists: float | None = None

    def season_average_for(self, prop_key: str) -> float:
        if prop_key == "pts":
            return self.points_per_game
        if prop_key == "reb":
            return self.rebounds_per_game
        return self.assists_per_game

    def last10_average_for(self, prop_key: str) -> float:
        if prop_key == "pts":
            return self.last10_points
        if prop_key == "reb":
            return self.last10_rebounds
        return self.last10_assists

    def split_average_for(self, prop_key: str) -> float | None:
        if prop_key == "pts":
            return self.home_points if self.is_home else self.away_points
        if prop_key == "reb":
            return self.home_rebounds if self.is_home else self.away_rebounds
        return self.home_assists if self.is_home else self.away_assists

    def matchup_label(self) -> str:
        return f"{self.away_team_name} @ {self.home_team_name}"


@dataclass(slots=True)
class OpponentDefenseStats:
    team_id: int
    team_name: str
    team_abbreviation: str
    def_rating: float
    pace: float
    pts_allowed_by_position: dict[str, float] = field(default_factory=dict)
    reb_allowed_by_position: dict[str, float] = field(default_factory=dict)
    ast_allowed_by_position: dict[str, float] = field(default_factory=dict)

    def prop_allowance(self, prop_key: str, position_bucket: str) -> float:
        if prop_key == "pts":
            lookup = self.pts_allowed_by_position
        elif prop_key == "reb":
            lookup = self.reb_allowed_by_position
        else:
            lookup = self.ast_allowed_by_position

        if position_bucket in lookup:
            return lookup[position_bucket]
        if lookup:
            return sum(lookup.values()) / len(lookup)
        return 0.0


@dataclass(slots=True)
class PropPrediction:
    player_id: int
    player_name: str
    team_abbreviation: str
    opponent_team_abbreviation: str
    opponent_team_name: str
    position: str
    game_id: str
    away_team_name: str
    home_team_name: str
    prop_key: str
    prop_label: str
    line: float
    predicted_value: float
    direction: str
    edge_pct: float
    true_prob: float
    confidence: float
    full_kelly: float
    quarter_kelly: float
    decision: str
    reason: str

    def decision_text(self) -> str:
        if self.decision == "BET":
            return f"BET {self.direction} {self.line:.1f}"
        return "PASS"

    def summary_decision(self) -> str:
        if self.decision == "BET":
            return f"BET {self.direction}"
        return "PASS"

    def matchup_label(self) -> str:
        return f"{self.away_team_name} @ {self.home_team_name}"
