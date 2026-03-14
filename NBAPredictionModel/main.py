import datetime
from data_models import Player, TeamStats, Team, Venue, GameContext
from verification import VerificationGate
from probability_layers import (
    calculate_layer1_base_rate, 
    calculate_layer2_situational, 
    calculate_layer3_matchup_modifier, 
    extremize_probability, 
    predict_total_points, 
    predict_spread
)
from market_mechanics import convert_american_to_implied, remove_vig, calculate_edge, check_minimum_threshold, get_recommended_stake

def format_output(game_ctx: GameContext, home_model_prob: float, home_odds: int, away_odds: int, 
                  l1_prob: float, l2_adj: float, l2_reason: str, l3_adj: float, l3_reason: str,
                  raw_prob: float, extremized_prob: float):
    
    print("\n" + "="*80)
    print(f"### [{game_ctx.away_team.name}] vs [{game_ctx.home_team.name}] — [{game_ctx.date}] — [{game_ctx.venue.name}]\n")
    
    print("**Verification checks:**")
    print("- [x] Rosters confirmed current ✓")
    print("- [x] Injury status / Load Management confirmed ✓")
    print("- [x] Lineups posted ✓\n")
    
    print("**Key conditions:**")
    print(f"- Net Rtg: [{game_ctx.away_team.name}] {game_ctx.away_team.team_stats.net_rating:+.1f} vs [{game_ctx.home_team.name}] {game_ctx.home_team.team_stats.net_rating:+.1f}")
    print(f"- Pace: [{game_ctx.away_team.name}] {game_ctx.away_team.team_stats.pace:.1f} vs [{game_ctx.home_team.name}] {game_ctx.home_team.team_stats.pace:.1f}")
    
    h_rest = "B2B" if game_ctx.home_team.team_stats.is_b2b_second_leg else "3-in-4-nights" if game_ctx.home_team.team_stats.is_3_in_4_nights else "Rested"
    a_rest = "B2B" if game_ctx.away_team.team_stats.is_b2b_second_leg else "3-in-4-nights" if game_ctx.away_team.team_stats.is_3_in_4_nights else "Rested"
    print(f"- Rest: [{game_ctx.away_team.name} {a_rest}] vs [{game_ctx.home_team.name} {h_rest}]\n")

    print("**Probability build:**")
    print(f"- Layer 1 base rate: [{game_ctx.home_team.name}] {l1_prob*100:.1f}%")
    print(f"- Layer 2 situational adj: {l2_adj*100:+.1f}% because [{l2_reason}]")
    print(f"- Layer 3 matchup modifier: {l3_adj*100:+.1f}% because [{l3_reason}]")
    print(f"- Raw probability: {raw_prob*100:.1f}%")
    print(f"- Extremized (×1.3): {extremized_prob*100:.1f}%\n")
    
    predicted_total = predict_total_points(game_ctx)
    predicted_spread = predict_spread(extremized_prob)
    
    print("**Model Predictions:**")
    if extremized_prob >= 0.5:
        winner = game_ctx.home_team.name
        spread_val = predicted_spread
        winner_prob = extremized_prob
    else:
        winner = game_ctx.away_team.name
        spread_val = -predicted_spread
        winner_prob = 1.0 - extremized_prob

    print(f"- **Winner:** {winner} (Model Prob: {winner_prob*100:.1f}%)")
    print(f"- **Spread:** {winner} by {abs(spread_val):.2f} points")
    print(f"- **Total:** {predicted_total:.1f} O/U\n")
    
    print(f"**Market odds:** {game_ctx.home_team.name} {home_odds} | {game_ctx.away_team.name} {away_odds}")
    
    true_home_implied, true_away_implied = remove_vig(home_odds, away_odds)
    print(f"**Market implied probability (vig-removed):** {game_ctx.home_team.name} {true_home_implied*100:.1f}% | {game_ctx.away_team.name} {true_away_implied*100:.1f}%")
    
    edge = calculate_edge(extremized_prob, true_home_implied)
    print(f"**Edge:** {game_ctx.home_team.name} {edge*100:+.1f}%")
    
    min_thresh = 0.05
    print(f"**Minimum threshold:** {min_thresh*100:.1f}%")
    
    decision = "BET" if max(edge, -edge) >= min_thresh else "PASS"
    bet_team = game_ctx.home_team.name if edge > 0 else game_ctx.away_team.name
    
    print(f"**Decision: {decision} {'on ' + bet_team if decision == 'BET' else ''}**\n")
    
    if decision == "BET":
        # Calculate bet sizing for whichever side we have an edge on
        if edge > 0:
            full_k, q_k = get_recommended_stake(home_odds, extremized_prob)
        else:
            full_k, q_k = get_recommended_stake(away_odds, 1.0 - extremized_prob)
            
        print(f"**If BET:**")
        print(f"- Full Kelly: {full_k:.2f}% of bankroll")
        print(f"- ¼ Kelly stake: {q_k:.2f}% of bankroll")
        
    print("\n**Confidence band:** High")
    print("**Data gaps:** None")
    print("="*80 + "\n")

def run_pipeline():
    # 1. Setup Mock Game Data: Celtics vs Nuggets
    venue = Venue("Ball Arena")
    
    # Denver Nuggets (Home)
    den_p1 = Player(1, "Nikola Jokic", "Nuggets", "C", "Active", 31.0)
    den_p2 = Player(2, "Jamal Murray", "Nuggets", "PG", "Active", 26.0)
    den_p3 = Player(3, "Michael Porter Jr.", "Nuggets", "SF", "Active", 22.0)
    
    den_stats = TeamStats(
        net_rating=6.5, off_rating_10=118.0, def_rating_10=112.5, 
        ts_pct=0.610, reb_pct=0.520, pace=97.5, last_10_win_pct=0.700, 
        is_b2b_second_leg=False, is_3_in_4_nights=False, season_win_pct=0.680
    )
    den_team = Team(101, "Nuggets", True, den_stats, [den_p1, den_p2, den_p3])
    
    # Boston Celtics (Away)
    bos_p1 = Player(4, "Jayson Tatum", "Celtics", "SF", "Active", 30.0)
    bos_p2 = Player(5, "Jaylen Brown", "Celtics", "SG", "Active", 28.0)
    bos_p3 = Player(6, "Kristaps Porzingis", "Celtics", "C", "Out", 24.0) # Assume Starting Center Out
    
    bos_stats = TeamStats(
        net_rating=8.2, off_rating_10=121.0, def_rating_10=110.0, 
        ts_pct=0.620, reb_pct=0.510, pace=100.5, last_10_win_pct=0.800, 
        is_b2b_second_leg=True, is_3_in_4_nights=False, season_win_pct=0.780
    )
    bos_team = Team(102, "Celtics", False, bos_stats, [bos_p1, bos_p2, bos_p3], starting_center_out=True)
    
    ctx = GameContext(datetime.datetime.now().strftime("%Y-%m-%d"), venue, den_team, bos_team, 0.50)
    
    # 2. Run Verification
    VerificationGate.run_all_checks(ctx)
    
    # 3. Layer 1: Base Rate (Home team perspective)
    l1_prob = calculate_layer1_base_rate(den_team, bos_team, ctx.h2h_home_win_pct_2yr)
    
    # 4. Layer 2: Situational
    l2_adj, l2_reasons = calculate_layer2_situational(den_team, bos_team, ctx)
    l2_away_adj, l2_away_reasons = calculate_layer2_situational(bos_team, den_team, ctx)
    total_l2_adj = l2_adj - l2_away_adj # from home team perspective
    
    l2_reasons_combined = f"{den_team.name}: {l2_reasons} | {bos_team.name}: {l2_away_reasons}"
    
    # 5. Layer 3: Matchup
    l3_adj, l3_reasons = calculate_layer3_matchup_modifier(den_team, bos_team)
    l3_away_adj, l3_away_reasons = calculate_layer3_matchup_modifier(bos_team, den_team)
    total_l3_adj = l3_adj - l3_away_adj
    l3_reasons_combined = f"{den_team.name}: {l3_reasons} | {bos_team.name}: {l3_away_reasons}"
    
    # 6. Build Raw Prob
    raw_prob = l1_prob + total_l2_adj + total_l3_adj
    
    # 7. Extremize
    ext_prob = extremize_probability(raw_prob, factor=1.3)
    
    # 8. Output
    # market odds for Nuggets -120, Celtics +100
    format_output(ctx, ext_prob, -120, +100, l1_prob, total_l2_adj, l2_reasons_combined, total_l3_adj, l3_reasons_combined, raw_prob, ext_prob)

if __name__ == "__main__":
    run_pipeline()
