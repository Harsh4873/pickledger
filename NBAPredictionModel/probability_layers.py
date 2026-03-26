import math

from data_models import Team, GameContext

_PROB_EPSILON = 1e-6


def _clamp_probability(prob: float) -> float:
    return max(_PROB_EPSILON, min(1.0 - _PROB_EPSILON, prob))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _logit(prob: float) -> float:
    clean_prob = _clamp_probability(prob)
    return math.log(clean_prob / (1.0 - clean_prob))

def calculate_layer1_base_rate(team: Team, opp_team: Team, h2h_win_pct_2yr: float) -> float:
    """
    Base rate = (Team A wins / total games) weighted as:
      - 45% season Net Rating differential vs Opponent
      - 30% last 10 games win % (recent form)
      - 25% H2H win % over last 2 years
    """
    net_rtg_weight = 0.45
    recent_weight = 0.30
    h2h_weight = 0.25
    
    # Net Rating Diff to Probability mapping
    net_rtg_diff = team.team_stats.net_rating - opp_team.team_stats.net_rating
    # A 1 net rating diff is roughly 3% win prob edge from 50%
    net_rtg_prob = 0.50 + (net_rtg_diff * 0.03)
    net_rtg_prob = max(0.05, min(0.95, net_rtg_prob))
    
    home_court_prior = 0.015 if team.is_home else -0.015

    base_rate = (
        (net_rtg_prob * net_rtg_weight) + 
        (team.team_stats.last_10_win_pct * recent_weight) + 
        (h2h_win_pct_2yr * h2h_weight) +
        home_court_prior
    )
                
    return base_rate

def calculate_layer2_situational(team: Team, opp_team: Team, game_ctx: GameContext) -> tuple[float, str]:
    """
    Apply adjustments to the base rate based on tonight's specific context.
    Cap total situational adjustment at ±15%.
    """
    adj = 0.0
    reasons = []
    
    # 1. Schedule Loss (2nd night of B2B)
    if team.team_stats.is_b2b_second_leg:
        adj -= 0.08
        reasons.append("Schedule Loss (B2B) -8.0%")
        
    # 2. Key Star Player Out
    if team.key_stars_out:
        adj -= 0.10
        reasons.append("Key Star Player Out (Usage > 25%) -10.0%")
        # "Next Man Up" (Ewing Theory): If it's JUST the star missing and the depth is healthy
        if team.rotation_players_out == 0:
            adj += 0.03
            reasons.append("Next Man Up Focus (Single Star Out) +3.0%")
            
    # Additional Rotation Players Out (Testing Depth)
    if team.rotation_players_out > 0:
        penalty = team.rotation_players_out * 0.02
        adj -= penalty
        reasons.append(f"Missing {team.rotation_players_out} Rotation/Bench Player(s) -{penalty*100:.1f}%")
        
    # 3. Starting Center Out
    if team.starting_center_out:
        adj -= 0.05
        reasons.append("Starting Center Out -5.0%")
        
    # 4. Motivation / Elimination Game
    if team.motivation_elimination_game:
        adj += 0.03
        reasons.append("Motivation/Elimination Game +3.0%")

    # 5. Home floor routine / road penalty
    if team.is_home:
        adj += 0.005
        reasons.append("Home floor routine +0.5%")
    else:
        adj -= 0.005
        reasons.append("Road travel context -0.5%")
        
    # Cap total situational adjustment at ±15% (0.15)
    adj = max(-0.15, min(0.15, adj))
    
    reason_str = ", ".join(reasons) if reasons else "No major situational adjustments"
    return adj, reason_str


def calculate_layer3_matchup_modifier(team: Team, opp_team: Team) -> tuple[float, str]:
    """
    Matchup Modifier (Pace & Efficiency)
    - Pace edge = High pace vs bottom-10 defense -> +3%
    - Rebound edge = REB% > 3% higher -> +2%
    """
    adj = 0.0
    reasons = []
    
    # Pace Edge: basic proxy is if pace > 100, offRtg > 115, and opponent defRtg > 115
    if team.team_stats.pace > 100.0 and team.team_stats.off_rating_10 > 115.0 and opp_team.team_stats.def_rating_10 > 115.0:
        adj += 0.03
        reasons.append("Pace edge vs bottom-10 def +3.0%")
        
    # Rebound Edge
    if (team.team_stats.reb_pct - opp_team.team_stats.reb_pct) > 0.03:
        adj += 0.02
        reasons.append("Rebound edge +2.0%")

    # Mild home-court shooting execution bump in favorable offensive matchups.
    if team.is_home and team.team_stats.off_rating_10 >= opp_team.team_stats.def_rating_10:
        adj += 0.005
        reasons.append("Home-court shot-making +0.5%")
        
    reason_str = ", ".join(reasons) if reasons else "No matchup modifiers"
    return adj, reason_str


def combine_home_win_probability(
    layer_probability: float,
    predicted_spread: float,
    home_team: Team,
    away_team: Team,
) -> float:
    """
    Blend the probability layers with the independent spread estimate.

    The old model derived spread from win probability in a single linear pass.
    That meant the spread carried no information of its own and probability was
    effectively double-counting the same number. Spread now feeds the win model
    as one feature among several instead of being the sole driver.
    """
    base_prob = _clamp_probability(layer_probability)
    recent_form_edge = home_team.team_stats.last_10_win_pct - away_team.team_stats.last_10_win_pct
    efficiency_edge = (
        (home_team.team_stats.off_rating_10 - away_team.team_stats.off_rating_10)
        - (home_team.team_stats.def_rating_10 - away_team.team_stats.def_rating_10)
    )
    rebound_edge = home_team.team_stats.reb_pct - away_team.team_stats.reb_pct

    blended_logit = (
        _logit(base_prob)
        + (predicted_spread / 7.5) * 0.55
        + recent_form_edge * 1.15
        + efficiency_edge * 0.03
        + rebound_edge * 8.0
    )
    return _sigmoid(blended_logit)


def legacy_extremize_probability(raw_prob: float, factor: float = 1.3) -> float:
    """
    Extremized prob = 50% + (Raw prob - 50%) * 1.3
    Cap at 95%, floor at 5%.
    """
    extremized = 0.50 + (raw_prob - 0.50) * factor
    return max(0.05, min(0.95, extremized))


def extremize_probability(raw_prob: float, factor: float = 1.3) -> float:
    """
    Extremize in log-odds space instead of hard clipping at 95%.

    The old linear `min(0.95, ...)` ceiling was wrong because very different
    game states were flattened into the same 95.0% output, which hid uncertainty
    instead of modeling it. Scaling the logit keeps the same "push away from
    50/50" intent while only approaching 0/1 asymptotically.
    """
    return _sigmoid(_logit(raw_prob) * factor)


def predict_total_points(game_ctx: GameContext) -> float:
    """
    Total: Base 225 points + (Pace Adjust) + (Defensive Rating Adjust)
    """
    base_points = 225.0
    
    h_pace = game_ctx.home_team.team_stats.pace
    a_pace = game_ctx.away_team.team_stats.pace
    pace_adj = ((h_pace - 99.0) + (a_pace - 99.0)) * 2.0
    
    h_def = game_ctx.home_team.team_stats.def_rating_10
    a_def = game_ctx.away_team.team_stats.def_rating_10
    def_adj = ((h_def - 114.0) + (a_def - 114.0)) * 0.8
    
    return base_points + pace_adj + def_adj


def legacy_predict_spread(home_prob: float) -> float:
    """
    Spread: (Extremized Prob - 0.50) * 30
    [Rule of thumb: 10% prob edge = ~3 point spread]
    """
    prob_diff = home_prob - 0.50
    return prob_diff * 30.0


def predict_spread(home_team: Team, away_team: Team) -> float:
    """
    Predict the expected home scoring margin directly from team features.

    This is intentionally independent from the probability output so spread can
    inform the win model instead of being reverse-engineered from it.
    """
    net_rating_margin = (home_team.team_stats.net_rating - away_team.team_stats.net_rating) * 0.72
    recent_form_margin = (home_team.team_stats.last_10_win_pct - away_team.team_stats.last_10_win_pct) * 8.0
    efficiency_margin = (
        (home_team.team_stats.off_rating_10 - away_team.team_stats.off_rating_10) * 0.14
        - (home_team.team_stats.def_rating_10 - away_team.team_stats.def_rating_10) * 0.12
    )
    rebound_margin = (home_team.team_stats.reb_pct - away_team.team_stats.reb_pct) * 22.0

    # Use a +3.5 point home-court prior until we have enough logged outcomes to
    # estimate the margin empirically from this model's own prediction history.
    home_court_margin = 3.5 if home_team.is_home else -3.5

    return net_rating_margin + recent_form_margin + efficiency_margin + rebound_margin + home_court_margin
