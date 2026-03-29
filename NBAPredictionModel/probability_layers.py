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
    Base rate blends season quality with recency-sensitive 5/10 game form.
    """
    net_rtg_weight = 0.28
    recent_5_weight = 0.22
    recent_10_weight = 0.18
    weighted_form_weight = 0.12
    recent_point_diff_weight = 0.10
    season_weight = 0.05
    h2h_weight = 0.05
    
    # Net Rating Diff to Probability mapping
    net_rtg_diff = team.team_stats.net_rating - opp_team.team_stats.net_rating
    # A 1 net rating diff is roughly 3% win prob edge from 50%
    net_rtg_prob = 0.50 + (net_rtg_diff * 0.03)
    net_rtg_prob = max(0.05, min(0.95, net_rtg_prob))

    recent_point_diff = team.team_stats.weighted_point_diff - opp_team.team_stats.weighted_point_diff
    recent_point_prob = max(0.05, min(0.95, 0.50 + (recent_point_diff * 0.018)))
    
    home_court_prior = 0.015 if team.is_home else -0.015

    base_rate = (
        (net_rtg_prob * net_rtg_weight) +
        (team.team_stats.recent_5_win_pct * recent_5_weight) +
        (team.team_stats.recent_10_win_pct * recent_10_weight) +
        (team.team_stats.weighted_win_pct * weighted_form_weight) +
        (recent_point_prob * recent_point_diff_weight) +
        (team.team_stats.season_win_pct * season_weight) +
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

    rest_edge = team.team_stats.rest_days - opp_team.team_stats.rest_days
    if abs(rest_edge) >= 0.5:
        rest_adj = max(-0.05, min(0.05, rest_edge * 0.015))
        adj += rest_adj
        reasons.append(f"Rest day edge {rest_adj*100:+.1f}%")

    if team.team_stats.back_to_back_flag and not team.team_stats.is_b2b_second_leg:
        adj -= 0.04
        reasons.append("Back-to-back flag -4.0%")

    if team.team_stats.is_3_in_4_nights:
        adj -= 0.03
        reasons.append("3 games in 4 nights -3.0%")
        
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

    if team.injury_flag:
        injury_adj = min(0.08, 0.015 + (team.injury_severity * 0.45))
        adj -= injury_adj
        reasons.append(f"Injury flag -{injury_adj*100:.1f}%")

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
    
    pace_diff = team.team_stats.pace - opp_team.team_stats.pace
    if team.team_stats.pace > 100.0 and team.team_stats.off_rating_10 > 115.0 and opp_team.team_stats.def_rating_10 > 115.0:
        adj += 0.03
        reasons.append("Pace edge vs bottom-10 def +3.0%")
    elif pace_diff > 2.0:
        adj += 0.02
        reasons.append("Pace differential edge +2.0%")

    # Rebound Edge
    if (team.team_stats.reb_pct - opp_team.team_stats.reb_pct) > 0.03:
        adj += 0.02
        reasons.append("Rebound edge +2.0%")

    recent_form_edge = team.team_stats.weighted_point_diff - opp_team.team_stats.weighted_point_diff
    if abs(recent_form_edge) >= 2.0:
        form_adj = max(-0.04, min(0.04, recent_form_edge * 0.004))
        adj += form_adj
        reasons.append(f"Recent point-diff edge {form_adj*100:+.1f}%")

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

    The spread input is kept intentionally light here. Probability layers own
    the team-strength, home-court, and recent-form signal; spread is a
    cross-check from the separate matchup margin path, not a second full pass
    over the same inputs.
    """
    base_prob = _clamp_probability(layer_probability)
    # Keep the layer-probability baseline from dominating the independent spread
    # cross-check; the raw layer stack is already opinionated before re-blending.
    compressed_base_prob = 0.50 + ((base_prob - 0.50) * 0.45)

    blended_logit = _logit(compressed_base_prob) + (predicted_spread / 10.0) * 0.30
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
    Total scoring model with pace, recent form, rest, B2B, and injury context.
    """
    base_points = (
        game_ctx.home_team.team_stats.recent_10_total_points
        + game_ctx.away_team.team_stats.recent_10_total_points
    ) / 2.0
    
    h_pace = game_ctx.home_team.team_stats.pace
    a_pace = game_ctx.away_team.team_stats.pace
    pace_adj = ((h_pace - 99.0) + (a_pace - 99.0)) * 2.0
    pace_diff_adj = abs(h_pace - a_pace) * 0.35
    
    h_def = game_ctx.home_team.team_stats.def_rating_10
    a_def = game_ctx.away_team.team_stats.def_rating_10
    def_adj = ((h_def - 114.0) + (a_def - 114.0)) * 0.8

    recent_total_adj = (
        (game_ctx.home_team.team_stats.recent_5_total_points - game_ctx.home_team.team_stats.recent_10_total_points)
        + (game_ctx.away_team.team_stats.recent_5_total_points - game_ctx.away_team.team_stats.recent_10_total_points)
    ) * 0.35
    rest_adj = (
        (game_ctx.home_team.team_stats.rest_days - 1.0)
        + (game_ctx.away_team.team_stats.rest_days - 1.0)
    ) * 0.6
    b2b_adj = (
        (-1.8 if game_ctx.home_team.team_stats.back_to_back_flag else 0.0)
        + (-1.8 if game_ctx.away_team.team_stats.back_to_back_flag else 0.0)
    )
    injury_adj = -2.0 * (game_ctx.home_team.injury_flag + game_ctx.away_team.injury_flag)
    injury_adj -= (game_ctx.home_team.injury_severity + game_ctx.away_team.injury_severity) * 6.0

    return base_points + pace_adj + pace_diff_adj + def_adj + recent_total_adj + rest_adj + b2b_adj + injury_adj


def legacy_predict_total_points(game_ctx: GameContext) -> float:
    """
    Legacy total model retained for side-by-side dashboard comparisons.
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
    # Spread feature ownership before -> after:
    # - Home court: additive spread prior -> probability layers only.
    # - Recent form: additive spread stack -> probability layers only.
    # - Point differential: additive spread stack -> probability layers only.
    # - Pace: additive spread term -> multiplicative spread-only adjustment.
    # - Rest/B2B: additive spread term -> multiplicative spread-only adjustment.
    # - Injuries: additive spread term -> multiplicative spread-only adjustment.
    matchup_margin = (
        (
            (home_team.team_stats.off_rating_10 - away_team.team_stats.def_rating_10)
            - (away_team.team_stats.off_rating_10 - home_team.team_stats.def_rating_10)
        ) * 0.24
        + (home_team.team_stats.ts_pct - away_team.team_stats.ts_pct) * 38.0
        + (home_team.team_stats.reb_pct - away_team.team_stats.reb_pct) * 16.0
    )
    # Map the orthogonal matchup score back into sportsbook-like point-spread units.
    base_margin = matchup_margin * 3.5

    average_pace = (home_team.team_stats.pace + away_team.team_stats.pace) / 2.0
    pace_multiplier = 1.0 + max(-0.06, min(0.06, (average_pace - 99.0) * 0.010))

    rest_load_edge = (
        (home_team.team_stats.rest_days - away_team.team_stats.rest_days)
        + (0.75 if away_team.team_stats.back_to_back_flag else 0.0)
        - (0.75 if home_team.team_stats.back_to_back_flag else 0.0)
    )
    rest_multiplier = 1.0 + max(-0.10, min(0.10, rest_load_edge * 0.035))

    injury_edge = away_team.injury_severity - home_team.injury_severity
    injury_multiplier = 1.0 + max(-0.15, min(0.15, injury_edge * 0.80))

    projected_margin = base_margin * pace_multiplier * rest_multiplier * injury_multiplier
    # NBA market spreads never exceed ~18 pts; cap the published projection.
    projected_margin = max(-18.0, min(18.0, projected_margin))
    return projected_margin
