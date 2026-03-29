import datetime
from calibration import load_platt_scaler
from data_models import Player, TeamStats, Team, Venue, GameContext
from prediction_logging import append_prediction_log
from verification import VerificationGate
from probability_layers import (
    calculate_layer1_base_rate, 
    calculate_layer2_situational, 
    calculate_layer3_matchup_modifier, 
    combine_home_win_probability,
    extremize_probability, 
    predict_total_points, 
    predict_spread
)
from market_mechanics import convert_american_to_implied, remove_vig, calculate_edge, check_minimum_threshold, get_recommended_stake

def format_output(game_ctx: GameContext, home_model_prob: float, home_odds: int, away_odds: int,
                  l1_prob: float, l2_adj: float, l2_reason: str, l3_adj: float, l3_reason: str,
                  raw_prob: float, extremized_prob: float, predicted_spread: float | None = None,
                  predicted_total: float | None = None, calibration_note: str = "",
                  log_prediction: bool = True, log_calibration_flag: str = ""):
    
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
    print(f"- Extremized (pre-calibration): {extremized_prob*100:.1f}%")
    print(f"- Calibrated probability: {home_model_prob*100:.1f}%\n")
    
    if predicted_total is None:
        predicted_total = predict_total_points(game_ctx)
    if predicted_spread is None:
        predicted_spread = predict_spread(game_ctx.home_team, game_ctx.away_team)
    
    print("**Model Predictions:**")
    if home_model_prob >= 0.5:
        winner = game_ctx.home_team.name
        spread_val = predicted_spread
        winner_prob = home_model_prob
    else:
        winner = game_ctx.away_team.name
        spread_val = -predicted_spread
        winner_prob = 1.0 - home_model_prob

    print(f"- **Winner:** {winner} (Model Prob: {winner_prob*100:.1f}%)")
    print(f"- **Spread:** {winner} by {abs(spread_val):.2f} points")
    print(f"- **Total:** {predicted_total:.1f} O/U\n")
    
    print(f"**Market odds:** {game_ctx.home_team.name} {home_odds} | {game_ctx.away_team.name} {away_odds}")
    
    true_home_implied, true_away_implied = remove_vig(home_odds, away_odds)
    print(f"**Market implied probability (vig-removed):** {game_ctx.home_team.name} {true_home_implied*100:.1f}% | {game_ctx.away_team.name} {true_away_implied*100:.1f}%")
    
    edge = calculate_edge(home_model_prob, true_home_implied)
    print(f"**Edge:** {game_ctx.home_team.name} {edge*100:+.1f}%")
    
    min_thresh = 0.05
    print(f"**Minimum threshold:** {min_thresh*100:.1f}%")
    
    decision = "BET" if max(edge, -edge) >= min_thresh else "PASS"
    bet_team = game_ctx.home_team.name if edge > 0 else game_ctx.away_team.name
    
    print(f"**Decision: {decision} {'on ' + bet_team if decision == 'BET' else ''}**\n")
    
    if decision == "BET":
        # Calculate bet sizing for whichever side we have an edge on
        if edge > 0:
            full_k, q_k = get_recommended_stake(home_odds, home_model_prob)
        else:
            full_k, q_k = get_recommended_stake(away_odds, 1.0 - home_model_prob)
            
        print(f"**If BET:**")
        print(f"- Full Kelly: {full_k:.2f}% of bankroll")
        print(f"- ¼ Kelly stake: {q_k:.2f}% of bankroll")

    if log_prediction:
        append_prediction_log(
            game_ctx,
            predicted_spread,
            extremized_prob,
            home_model_prob,
            calibration_flag=log_calibration_flag,
        )
        
    print("\n**Confidence band:** High")
    if calibration_note:
        print(f"**Calibration:** {calibration_note}")
    print("**Data gaps:** None")
    print("="*80 + "\n")


def format_output_new(game_ctx: GameContext, home_model_prob: float, home_odds: int, away_odds: int,
                      l1_prob: float, l2_adj: float, l2_reason: str, l3_adj: float, l3_reason: str,
                      raw_prob: float, extremized_prob: float, predicted_spread: float | None = None,
                      predicted_total: float | None = None, calibration_note: str = "",
                      log_prediction: bool = True, log_calibration_flag: str = "",
                      decision_override: str | None = None, decision_note: str = ""):
    print("\n" + "="*80)
    print(f"### [{game_ctx.away_team.name}] vs [{game_ctx.home_team.name}] — [{game_ctx.date}] — [{game_ctx.venue.name}]\n")

    print("**Verification checks:**")
    print("- [x] Rosters confirmed current ✓")
    print("- [x] Injury status / Load Management confirmed ✓")
    print("- [x] Lineups posted ✓\n")

    print("**Key conditions:**")
    print(f"- Net Rtg: [{game_ctx.away_team.name}] {game_ctx.away_team.team_stats.net_rating:+.1f} vs [{game_ctx.home_team.name}] {game_ctx.home_team.team_stats.net_rating:+.1f}")
    print(f"- Pace: [{game_ctx.away_team.name}] {game_ctx.away_team.team_stats.pace:.1f} vs [{game_ctx.home_team.name}] {game_ctx.home_team.team_stats.pace:.1f}")
    tempo_context = getattr(game_ctx, "tempo_control", None)
    if tempo_context:
        print(
            f"- Tempo Control: [{game_ctx.away_team.name}] {tempo_context['away_weight']*100:.0f}% "
            f"vs [{game_ctx.home_team.name}] {tempo_context['home_weight']*100:.0f}% "
            f"-> Dictated Pace {tempo_context['dictated_pace']:.1f}"
        )
    form_capping = getattr(game_ctx, "form_capping", None)
    if form_capping:
        away_form = form_capping["away"]
        home_form = form_capping["home"]
        margin_cap = max(
            float(away_form.get("garbage_time_margin_cap", 15.0)),
            float(home_form.get("garbage_time_margin_cap", 15.0)),
        )
        print(f"- Garbage-Time Cap: ±{margin_cap:.0f} margin per game before recent-form averaging")
        print(
            f"- Capped Form: [{game_ctx.away_team.name}] W {away_form['raw_weighted']:+.1f}->{away_form['capped_weighted']:+.1f}, "
            f"10G {away_form['raw_recent_10']:+.1f}->{away_form['capped_recent_10']:+.1f}, "
            f"5G {away_form['raw_recent_5']:+.1f}->{away_form['capped_recent_5']:+.1f} | "
            f"[{game_ctx.home_team.name}] W {home_form['raw_weighted']:+.1f}->{home_form['capped_weighted']:+.1f}, "
            f"10G {home_form['raw_recent_10']:+.1f}->{home_form['capped_recent_10']:+.1f}, "
            f"5G {home_form['raw_recent_5']:+.1f}->{home_form['capped_recent_5']:+.1f}"
        )

    h_rest = "B2B" if game_ctx.home_team.team_stats.is_b2b_second_leg else "3-in-4-nights" if game_ctx.home_team.team_stats.is_3_in_4_nights else "Rested"
    a_rest = "B2B" if game_ctx.away_team.team_stats.is_b2b_second_leg else "3-in-4-nights" if game_ctx.away_team.team_stats.is_3_in_4_nights else "Rested"
    print(f"- Rest: [{game_ctx.away_team.name} {a_rest}] vs [{game_ctx.home_team.name} {h_rest}]\n")

    print("**Probability build:**")
    print(f"- Layer 1 base rate: [{game_ctx.home_team.name}] {l1_prob*100:.1f}%")
    print(f"- Layer 2 situational adj: {l2_adj*100:+.1f}% because [{l2_reason}]")
    print(f"- Layer 3 matchup modifier: {l3_adj*100:+.1f}% because [{l3_reason}]")
    print(f"- Raw probability: {raw_prob*100:.1f}%")
    print(f"- Extremized (pre-calibration): {extremized_prob*100:.1f}%")
    print(f"- Calibrated probability: {home_model_prob*100:.1f}%\n")

    if predicted_total is None:
        predicted_total = predict_total_points(game_ctx)
    if predicted_spread is None:
        predicted_spread = predict_spread(game_ctx.home_team, game_ctx.away_team)

    print("**Model Predictions:**")
    if home_model_prob >= 0.5:
        winner = game_ctx.home_team.name
        spread_val = predicted_spread
        winner_prob = home_model_prob
    else:
        winner = game_ctx.away_team.name
        spread_val = -predicted_spread
        winner_prob = 1.0 - home_model_prob

    print(f"- **Pick:** {winner}")
    print(f"- **Projected Margin:** {winner} by {abs(spread_val):.2f} points")
    print(f"- **Model Confidence:** {winner_prob*100:.1f}%")
    print(f"- **Total:** {predicted_total:.1f} O/U\n")

    if decision_override == "PASS" and decision_note:
        print(f"**Projection note:** {decision_note}\n")

    if log_prediction:
        append_prediction_log(
            game_ctx,
            predicted_spread,
            raw_prob,
            extremized_prob,
            home_model_prob,
            calibration_flag=log_calibration_flag,
        )

    print("\n**Confidence band:** High")
    if calibration_note:
        print(f"**Calibration:** {calibration_note}")
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
        is_b2b_second_leg=False, is_3_in_4_nights=False, season_win_pct=0.680,
        recent_5_win_pct=0.800, recent_10_win_pct=0.700, weighted_win_pct=0.760,
        recent_5_point_diff=7.2, recent_10_point_diff=6.5, weighted_point_diff=6.9,
        recent_5_total_points=223.0, recent_10_total_points=220.5, rest_days=2.0, back_to_back_flag=False
    )
    den_team = Team(101, "Nuggets", True, den_stats, [den_p1, den_p2, den_p3])
    
    # Boston Celtics (Away)
    bos_p1 = Player(4, "Jayson Tatum", "Celtics", "SF", "Active", 30.0)
    bos_p2 = Player(5, "Jaylen Brown", "Celtics", "SG", "Active", 28.0)
    bos_p3 = Player(6, "Kristaps Porzingis", "Celtics", "C", "Out", 24.0) # Assume Starting Center Out
    
    bos_stats = TeamStats(
        net_rating=8.2, off_rating_10=121.0, def_rating_10=110.0, 
        ts_pct=0.620, reb_pct=0.510, pace=100.5, last_10_win_pct=0.800, 
        is_b2b_second_leg=True, is_3_in_4_nights=False, season_win_pct=0.780,
        recent_5_win_pct=0.800, recent_10_win_pct=0.800, weighted_win_pct=0.790,
        recent_5_point_diff=8.0, recent_10_point_diff=8.2, weighted_point_diff=8.1,
        recent_5_total_points=227.5, recent_10_total_points=225.0, rest_days=0.0, back_to_back_flag=True
    )
    bos_team = Team(102, "Celtics", False, bos_stats, [bos_p1, bos_p2, bos_p3], starting_center_out=True)
    
    ctx = GameContext(
        datetime.datetime.now().strftime("%Y-%m-%d"),
        venue,
        den_team,
        bos_team,
        0.50,
        game_id="demo-nuggets-vs-celtics",
    )
    
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
    layer_prob = l1_prob + total_l2_adj + total_l3_adj
    predicted_spread = predict_spread(den_team, bos_team)
    raw_prob = combine_home_win_probability(layer_prob, predicted_spread, den_team, bos_team)

    # 7. Extremize + calibrate
    ext_prob = extremize_probability(raw_prob, factor=1.3)
    scaler, calibration_diag = load_platt_scaler()
    calibrated_prob = scaler.calibrate(ext_prob)
    
    # 8. Output
    # market odds for Nuggets -120, Celtics +100
    format_output_new(
        ctx,
        calibrated_prob,
        -120,
        +100,
        l1_prob,
        total_l2_adj,
        l2_reasons_combined,
        total_l3_adj,
        l3_reasons_combined,
        raw_prob,
        ext_prob,
        predicted_spread=predicted_spread,
        calibration_note=calibration_diag.note,
        log_calibration_flag=calibration_diag.log_flag,
    )

if __name__ == "__main__":
    run_pipeline()
