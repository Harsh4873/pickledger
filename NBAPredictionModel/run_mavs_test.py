"""
Re-run the Mavericks vs Grizzlies game using REAL on/off court injury data
instead of flat percentage guesses.
"""
import datetime
import time
from calibration import load_platt_scaler
from data_models import Player, TeamStats, Team, Venue, GameContext
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
from injury_impact import calculate_injury_adjustment, print_team_impact_report
from main import format_output

def create_team(id_num, name, is_home, net_rtg, off_rtg, def_rtg, ts_pct, reb_pct, pace, win_pct):
    recent_total_proxy = 224.0 + ((pace - 99.0) * 1.4) + ((off_rtg - def_rtg) * 0.20)
    stats = TeamStats(
        net_rating=net_rtg,
        off_rating_10=off_rtg,
        def_rating_10=def_rtg,
        ts_pct=ts_pct,
        reb_pct=reb_pct,
        pace=pace,
        last_10_win_pct=win_pct,
        is_b2b_second_leg=False,
        is_3_in_4_nights=False,
        season_win_pct=win_pct,
        recent_5_win_pct=min(1.0, max(0.0, win_pct + 0.03)),
        recent_10_win_pct=win_pct,
        weighted_win_pct=win_pct,
        recent_5_point_diff=net_rtg * 1.05,
        recent_10_point_diff=net_rtg,
        weighted_point_diff=net_rtg * 1.02,
        recent_5_total_points=recent_total_proxy + 1.0,
        recent_10_total_points=recent_total_proxy,
        rest_days=1.0,
        back_to_back_flag=False,
    )
    p1 = Player(id_num*10+1, "Star 1", name, "PG", "Active", 25.0)
    p2 = Player(id_num*10+2, "Star 2", name, "SG", "Active", 25.0)
    p3 = Player(id_num*10+3, "Center", name, "C", "Active", 20.0)
    return Team(id_num, name, is_home, stats, [p1, p2, p3])

def run_with_injuries(scenario_name, mavs_out=[], griz_out=[], ou_line=238.5):
    print(f"\n{'*'*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'*'*80}")
    
    mavs_data = (-4.7, 109.5, 114.2, 0.563, 0.494, 102.34, 0.323)
    griz_data = (-2.5, 113.7, 116.3, 0.574, 0.495, 101.47, 0.359)
    
    away_team = create_team(1, "Mavericks", False, *mavs_data)
    home_team = create_team(2, "Grizzlies", True, *griz_data)
    
    venue = Venue("FedExForum")
    ctx = GameContext(
        datetime.datetime.now().strftime("%Y-%m-%d"),
        venue,
        home_team,
        away_team,
        0.50,
        game_id=f"grizzlies-vs-mavericks-{scenario_name.lower().replace(' ', '-')}",
    )
    
    # Layer 1: Base Rate
    l1_prob = calculate_layer1_base_rate(home_team, away_team, ctx.h2h_home_win_pct_2yr)
    
    # Layer 2: Standard Situational (B2B, etc.)
    l2_adj, l2_reasons = calculate_layer2_situational(home_team, away_team, ctx)
    l2_away_adj, l2_away_reasons = calculate_layer2_situational(away_team, home_team, ctx)
    total_l2_adj = l2_adj - l2_away_adj
    
    # Layer 2.5: REAL Injury Impact (from on/off court data)
    inj_adj_home = 0.0
    inj_reason_home = "No injuries"
    inj_adj_away = 0.0
    inj_reason_away = "No injuries"
    
    if griz_out:
        inj_adj_home, inj_reason_home = calculate_injury_adjustment("Grizzlies", griz_out)
        time.sleep(1)
    if mavs_out:
        inj_adj_away, inj_reason_away = calculate_injury_adjustment("Mavericks", mavs_out)
    
    # From home (Grizzlies) perspective: home injuries hurt them, away injuries help them
    total_injury_adj = inj_adj_home - inj_adj_away  # negative home inj hurts, negative away inj helps

    l2_combined = f"Situational: {l2_reasons} | Injuries: Grizzlies [{inj_reason_home}], Mavericks [{inj_reason_away}]"
    total_l2_with_inj = total_l2_adj + total_injury_adj
    total_l2_with_inj = max(-0.15, min(0.15, total_l2_with_inj))
    
    # Layer 3: Matchup
    l3_adj, l3_reasons = calculate_layer3_matchup_modifier(home_team, away_team)
    l3_away_adj, l3_away_reasons = calculate_layer3_matchup_modifier(away_team, home_team)
    total_l3_adj = l3_adj - l3_away_adj
    l3_combined = f"{home_team.name}: {l3_reasons} | {away_team.name}: {l3_away_reasons}"
    
    layer_prob = l1_prob + total_l2_with_inj + total_l3_adj
    predicted_spread = predict_spread(home_team, away_team)
    raw_prob = combine_home_win_probability(layer_prob, predicted_spread, home_team, away_team)
    ext_prob = extremize_probability(raw_prob)
    calibrator, calibration_diag = load_platt_scaler()
    calibrated_prob = calibrator.calibrate(ext_prob)
    
    format_output(
        ctx,
        calibrated_prob,
        -110,
        -110,
        l1_prob,
        total_l2_with_inj,
        l2_combined,
        total_l3_adj,
        l3_combined,
        raw_prob,
        ext_prob,
        predicted_spread=predicted_spread,
        calibration_note=calibration_diag.note,
    )
    
    predicted_total = predict_total_points(ctx)
    print(f"**Over/Under Tooling:** Predicted Total {predicted_total:.1f} vs Line {ou_line}")
    if predicted_total > ou_line + 3:
        print(f"**O/U Decision: BET OVER**\n")
    elif predicted_total < ou_line - 3:
        print(f"**O/U Decision: BET UNDER**\n")
    else:
        print(f"**O/U Decision: PASS**\n")

def main():
    # First, show the impact reports
    print("\n" + "="*80)
    print("STEP 1: FETCHING REAL ON/OFF COURT DATA")
    print("="*80)
    print_team_impact_report("Mavericks")
    time.sleep(1)
    print_team_impact_report("Grizzlies")
    time.sleep(1)
    
    print("\n" + "="*80)
    print("STEP 2: RUNNING GAME SCENARIOS WITH REAL INJURY DATA")
    print("="*80)
    
    # Scenario 1: Both teams healthy
    run_with_injuries("1. Baseline — Both Teams Healthy")
    time.sleep(1)
    
    # Scenario 2: Kyrie Irving OUT for Mavs (has been out all season with torn ACL)
    run_with_injuries("2. Kyrie Irving OUT (Torn ACL — Season-long)", mavs_out=["Kyrie Irving"])
    time.sleep(1)
    
    # Scenario 3: Ja Morant OUT for Grizzlies
    run_with_injuries("3. Ja Morant OUT for Grizzlies", griz_out=["Ja Morant"])
    time.sleep(1)
    
    # Scenario 4: AD + Lively OUT for Mavs
    run_with_injuries("4. Anthony Davis + Dereck Lively II OUT for Mavs", mavs_out=["Anthony Davis", "Dereck Lively II"])

if __name__ == "__main__":
    main()
