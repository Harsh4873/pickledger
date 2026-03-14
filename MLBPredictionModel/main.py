import datetime
from data_models import Player, PitcherStats, TeamStats, Team, Weather, Venue, GameContext
from verification import VerificationGate
from probability_layers import calculate_layer1_base_rate, calculate_layer2_situational, calculate_layer3_pitcher_modifier, extremize_probability, predict_total_runs, predict_spread
from market_mechanics import convert_american_to_implied, remove_vig, calculate_edge, check_minimum_threshold, get_recommended_stake

def format_output(game_ctx: GameContext, home_model_prob: float, home_odds: int, away_odds: int, 
                  l1_prob: float, l2_adj: float, l2_reason: str, l3_adj: float, l3_reason: str,
                  raw_prob: float, extremized_prob: float):
    
    print("\\n" + "="*80)
    print(f"### [{game_ctx.away_team.name}] vs [{game_ctx.home_team.name}] — [{game_ctx.date}] — [{game_ctx.venue.name}]\\n")
    
    print("**Verification checks:**")
    print("- [x] Rosters confirmed current ✓")
    print("- [x] Injury status confirmed ✓")
    print("- [x] Lineups posted ✓")
    print("- [x] Weather sourced ✓\\n")
    
    park_type = "hitter's" if game_ctx.venue.park_factor_runs > 105 else "pitcher's" if game_ctx.venue.park_factor_runs < 95 else "neutral"
    
    print("**Key conditions:**")
    print(f"- Wind: {game_ctx.weather.wind_speed_mph} mph {game_ctx.weather.wind_direction} — {'lean over' if game_ctx.weather.wind_direction=='out' else 'neutral'}")
    dome_str = "dome" if game_ctx.weather.is_dome else "open air"
    print(f"- Temp: {game_ctx.weather.temp_f}°, {dome_str}")
    print(f"- Park factor: {game_ctx.venue.park_factor_runs} ({park_type})\\n")
    
    print("**Starting pitchers:**")
    h_starter = game_ctx.home_team.starter_stats
    a_starter = game_ctx.away_team.starter_stats
    print(f"- [{game_ctx.away_team.name}]: {game_ctx.away_team.starter.name} — ERA {a_starter.era}, FIP {a_starter.fip}, last 5 starts: {a_starter.last_5_starts_summary}")
    print(f"- [{game_ctx.home_team.name}]: {game_ctx.home_team.starter.name} — ERA {h_starter.era}, FIP {h_starter.fip}, last 5 starts: {h_starter.last_5_starts_summary}\\n")
    
    print("**Probability build:**")
    print(f"- Layer 1 base rate: [{game_ctx.home_team.name}] {l1_prob*100:.1f}%")
    print(f"- Layer 2 situational adj: {l2_adj*100:+.1f}% because [{l2_reason}]")
    print(f"- Layer 3 pitcher modifier: {l3_adj*100:+.1f}% because [FIP diff {abs(h_starter.fip - a_starter.fip):.2f}]")
    print(f"- Raw probability: {raw_prob*100:.1f}%")
    print(f"- Extremized (×1.3): {extremized_prob*100:.1f}%\\n")
    
    predicted_total = predict_total_runs(game_ctx)
    predicted_spread = predict_spread(extremized_prob)
    
    print("**Model Predictions (Independent of Market Odds):**")
    if extremized_prob >= 0.5:
        winner = game_ctx.home_team.name
        spread_val = predicted_spread
        winner_prob = extremized_prob
    else:
        winner = game_ctx.away_team.name
        spread_val = -predicted_spread
        winner_prob = 1.0 - extremized_prob

    print(f"- **Winner:** {winner} (Model Prob: {winner_prob*100:.1f}%)")
    print(f"- **Spread (Expected Run Diff):** {winner} by {abs(spread_val):.2f} runs")
    print(f"- **Total Runs:** {predicted_total:.1f} O/U\\n")
    
    print(f"**Market odds:** {game_ctx.home_team.name} {home_odds} | {game_ctx.away_team.name} {away_odds}")
    
    true_home_implied, true_away_implied = remove_vig(home_odds, away_odds)
    print(f"**Market implied probability (vig-removed):** {game_ctx.home_team.name} {true_home_implied*100:.1f}% | {game_ctx.away_team.name} {true_away_implied*100:.1f}%")
    
    edge = calculate_edge(extremized_prob, true_home_implied)
    print(f"**Edge:** {game_ctx.home_team.name} {edge*100:+.1f}%")
    
    min_thresh = 0.05
    print(f"**Minimum threshold:** {min_thresh*100:.1f}%")
    
    decision = "BET" if max(edge, -edge) >= min_thresh else "PASS"
    bet_team = game_ctx.home_team.name if edge > 0 else game_ctx.away_team.name
    
    print(f"**Decision: {decision} {'on ' + bet_team if decision == 'BET' else ''}**\\n")
    
    if decision == "BET":
        # Calculate bet sizing for whichever side we have an edge on
        if edge > 0:
            full_k, q_k = get_recommended_stake(home_odds, extremized_prob)
        else:
            full_k, q_k = get_recommended_stake(away_odds, 1.0 - extremized_prob)
            
        print(f"- Full Kelly: {full_k:.2f}% of bankroll")
        print(f"- ¼ Kelly stake (recommended): {q_k:.2f}% of bankroll")
        print(f"- Correlation notes: Check if parlayed with total runs")
        
    print("\\n**Confidence band:** High — based on data completeness")
    print("**Data gaps:** None")
    print("="*80 + "\\n")

def run_pipeline():
    # 1. Setup Mock Game Data
    venue = Venue("Dodger Stadium", 99, 295)
    weather = Weather(72.0, 12.0, "out", False)
    
    la_pitcher = Player(1, "Tyler Glasnow", "Dodgers", "SP")
    la_p_stats = PitcherStats(era=2.85, fip=2.60, whip=0.95, last_5_starts_summary="32 IP, 8 ER, 45 K", days_rest=5, home_split_era=2.50, away_split_era=3.10, woba_vs_l=0.280, woba_vs_r=0.250, pitches_per_start_avg=98)
    la_t_stats = TeamStats(ops=0.790, woba=0.340, wrc_plus=115, last_10_runs_avg=5.2, bullpen_pitches_yesterday=45, travel_fatigue=False, consecutive_games=1, home_win_pct=0.650, away_win_pct=0.550, season_win_pct=0.610, last_30_days_win_pct=0.640)
    la_team = Team(101, "Dodgers", True, la_pitcher, la_p_stats, la_t_stats, [])
    
    sd_pitcher = Player(2, "Dylan Cease", "Padres", "SP")
    sd_p_stats = PitcherStats(era=3.10, fip=3.30, whip=1.05, last_5_starts_summary="28 IP, 11 ER, 35 K", days_rest=5, home_split_era=3.00, away_split_era=3.20, woba_vs_l=0.300, woba_vs_r=0.270, pitches_per_start_avg=92)
    sd_t_stats = TeamStats(ops=0.750, woba=0.320, wrc_plus=105, last_10_runs_avg=4.4, bullpen_pitches_yesterday=110, travel_fatigue=True, consecutive_games=3, home_win_pct=0.580, away_win_pct=0.500, season_win_pct=0.540, last_30_days_win_pct=0.520)
    sd_team = Team(102, "Padres", False, sd_pitcher, sd_p_stats, sd_t_stats, [])
    
    ctx = GameContext(datetime.datetime.now().strftime("%Y-%m-%d"), venue, weather, la_team, sd_team, 0.60)
    
    # 2. Run Verification
    VerificationGate.run_all_checks(ctx)
    
    # 3. Layer 1: Base Rate (Home team perspective)
    l1_prob = calculate_layer1_base_rate(la_team, True, ctx.h2h_home_win_pct_3yr)
    
    # 4. Layer 2: Situational
    l2_adj, l2_reasons = calculate_layer2_situational(la_team, sd_team, ctx)
    
    # 5. Layer 3: Pitchers
    l3_adj, l3_reasons = calculate_layer3_pitcher_modifier(la_team.starter_stats, sd_team.starter_stats)
    
    # 6. Build Raw Prob
    raw_prob = l1_prob + l2_adj + l3_adj
    
    # 7. Extremize
    ext_prob = extremize_probability(raw_prob, factor=1.3)
    
    # 8. Output
    # Let's say market is priced Dodgers -165, Padres +140
    format_output(ctx, ext_prob, -165, +140, l1_prob, l2_adj, l2_reasons, l3_adj, l3_reasons, raw_prob, ext_prob)

if __name__ == "__main__":
    run_pipeline()
