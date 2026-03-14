from data_models import Team, GameContext

def calculate_layer1_base_rate(team: Team, home_is_team: bool, h2h_win_pct_3yr: float) -> float:
    """
    Base rate = (Team wins / total games) weighted as:
      - 40% season win %
      - 35% last 30 days win %
      - 25% H2H win % at this venue over last 3 years
    """
    season_weight = 0.40
    recent_weight = 0.35
    h2h_weight = 0.25
    
    # Simple weighted average
    base_rate = (
        (team.team_stats.season_win_pct * season_weight) + 
        (team.team_stats.last_30_days_win_pct * recent_weight) + 
        (h2h_win_pct_3yr * h2h_weight)
    )
                
    return base_rate

def calculate_layer2_situational(team: Team, opp_team: Team, game_ctx: GameContext) -> float:
    """
    Apply adjustments to the base rate based on tonight's specific context.
    Cap total situational adjustment at ±15% to avoid overconfidence.
    """
    adj = 0.0
    reasons = []
    
    # 1. Starting Pitcher ERA Advantage
    era_diff = opp_team.starter_stats.era - team.starter_stats.era
    if abs(era_diff) > 0.5:
        # e.g. 1 run ERA advantage = 5% shift
        shift = min((era_diff / 1.0) * 0.05, 0.08)
        adj += shift
        reasons.append(f"SP ERA Adv {'+' if shift>0 else ''}{shift*100:.1f}%")
        
    # 2. Weather
    wind_out = game_ctx.weather.wind_direction.lower() == 'out'
    wind_in = game_ctx.weather.wind_direction.lower() == 'in'
    
    if wind_out and game_ctx.weather.wind_speed_mph > 15:
        # This usually impacts total (Over) but can also help the better hitting team slightly
        adj += 0.01  # Minor tweak
        reasons.append(f"Wind out >15mph +1.0%")
    elif wind_in and game_ctx.weather.wind_speed_mph > 15:
        adj -= 0.01
        reasons.append(f"Wind in >15mph -1.0%")
        
    # 3. Park Factor
    if game_ctx.venue.park_factor_runs > 110:
        adj += 0.02
        reasons.append("Park Factor Extreme >110 +2.0%")
    elif game_ctx.venue.park_factor_runs < 90:
        adj -= 0.02
        reasons.append("Park Factor Extreme <90 -2.0%")
        
    # 4. Travel Fatigue (cross-country)
    if team.team_stats.travel_fatigue:
        adj -= 0.03
        reasons.append("Travel fatigue -3.0%")
        
    # 5. Bullpen Depleted
    if team.team_stats.bullpen_pitches_yesterday > 70:
        adj -= 0.03
        reasons.append("Bullpen depleted -3.0%")
        
    # 6. Rest advantage
    if team.starter_stats.days_rest >= 6: # 3+ extra days from normal 5
        adj += 0.03
        reasons.append("Rest advantage +3.0%")
        
    # Cap total situational adjustment at ±15% (0.15)
    adj = max(-0.15, min(0.15, adj))
    
    reason_str = ", ".join(reasons) if reasons else "No major situational adjustments"
    return adj, reason_str


def calculate_layer3_pitcher_modifier(team_pitcher, opp_pitcher) -> float:
    """
    Pitcher edge = abs(SP1 FIP - SP2 FIP) capped at 1.5 run difference
    Convert to probability: each 0.5 FIP diff ≈ 3% shift
    """
    fip_diff = opp_pitcher.fip - team_pitcher.fip
    
    # Cap the difference
    fip_diff = max(-1.5, min(1.5, fip_diff))
    
    modifier = (fip_diff / 0.5) * 0.03
    reason_str = f"FIP difference of {abs(fip_diff):.2f}"
    
    return modifier, reason_str


def extremize_probability(raw_prob: float, factor: float = 1.3) -> float:
    """
    Extremized prob = 50% + (Raw prob - 50%) * 1.3
    Cap at 95%, floor at 5%.
    """
    extremized = 0.50 + (raw_prob - 0.50) * factor
    return max(0.05, min(0.95, extremized))


def predict_total_runs(game_ctx: GameContext) -> float:
    """Calculates an expected run total based on pitching, park, and weather."""
    base_runs = 9.0
    
    # Pitcher adj: avg FIP is roughly 4.0
    h_fip = game_ctx.home_team.starter_stats.fip
    a_fip = game_ctx.away_team.starter_stats.fip
    pitcher_adj = (h_fip - 4.0) + (a_fip - 4.0)
    
    # Weather adj from prompt table
    weather_adj = 0.0
    wind_dir = game_ctx.weather.wind_direction.lower()
    wind_spd = game_ctx.weather.wind_speed_mph
    temp = game_ctx.weather.temp_f
    
    if wind_dir == 'out':
        if wind_spd >= 20: weather_adj += 2.0
        elif wind_spd >= 15: weather_adj += 1.5
        elif wind_spd >= 10: weather_adj += 0.75
    elif wind_dir == 'in':
        if wind_spd >= 15: weather_adj -= 1.5
        elif wind_spd >= 10: weather_adj -= 0.75
        
    if temp > 85: weather_adj += 0.4
    elif temp < 50: weather_adj -= 0.5
    
    # Park adj
    park_adj = (game_ctx.venue.park_factor_runs - 100) * 0.05
    if game_ctx.venue.elevation_ft >= 5000:
        park_adj += 1.0 # Coors field bump
        
    return base_runs + pitcher_adj + weather_adj + park_adj


def predict_spread(home_prob: float) -> float:
    """Converts a win probability into an expected run differential from the home team's perspective."""
    # Rule of thumb: every 10% prob over 50% translates to roughly 1 run diff.
    # 60% = +1.0 run edge. 40% = -1.0 run edge.
    prob_diff = home_prob - 0.50
    return prob_diff * 10.0
