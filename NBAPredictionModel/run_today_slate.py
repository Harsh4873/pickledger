import datetime
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
        last_10_win_pct=win_pct, # using season win pct as proxy for recent form in this test
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
    # create default mock players for lineup
    p1 = Player(id_num*10+1, f"Star 1", name, "PG", "Active", 25.0)
    p2 = Player(id_num*10+2, f"Star 2", name, "SG", "Active", 25.0)
    p3 = Player(id_num*10+3, f"Center", name, "C", "Active", 20.0)
    
    return Team(id_num, name, is_home, stats, [p1, p2, p3], key_stars_out=False, starting_center_out=False)

def test_matchup(away_name, away_data, home_name, home_data, ou_line):
    away_team = create_team(1, away_name, False, *away_data)
    home_team = create_team(2, home_name, True, *home_data)
    
    venue = Venue(f"{home_name} Arena")
    ctx = GameContext(
        datetime.datetime.now().strftime("%Y-%m-%d"),
        venue,
        home_team,
        away_team,
        0.50,
        game_id=f"{home_name.lower()}-vs-{away_name.lower()}-demo",
    )
    
    VerificationGate.run_all_checks(ctx)
    
    # Probability layers (from home team perspective)
    l1_prob = calculate_layer1_base_rate(home_team, away_team, ctx.h2h_home_win_pct_2yr)
    
    l2_adj, l2_reasons = calculate_layer2_situational(home_team, away_team, ctx)
    l2_away_adj, l2_away_reasons = calculate_layer2_situational(away_team, home_team, ctx)
    total_l2_adj = l2_adj - l2_away_adj
    l2_reasons_combined = f"{home_team.name}: {l2_reasons} | {away_team.name}: {l2_away_reasons}"
    
    l3_adj, l3_reasons = calculate_layer3_matchup_modifier(home_team, away_team)
    l3_away_adj, l3_away_reasons = calculate_layer3_matchup_modifier(away_team, home_team)
    total_l3_adj = l3_adj - l3_away_adj
    l3_reasons_combined = f"{home_team.name}: {l3_reasons} | {away_team.name}: {l3_away_reasons}"
    
    layer_prob = l1_prob + total_l2_adj + total_l3_adj
    predicted_spread = predict_spread(home_team, away_team)
    raw_prob = combine_home_win_probability(layer_prob, predicted_spread, home_team, away_team)
    ext_prob = extremize_probability(raw_prob)
    calibrator, calibration_diag = load_platt_scaler()
    calibrated_prob = calibrator.calibrate(ext_prob)
    
    # Mock odds, pick -110/-110 as standard neutral line
    format_output(
        ctx,
        calibrated_prob,
        -110,
        -110,
        l1_prob,
        total_l2_adj,
        l2_reasons_combined,
        total_l3_adj,
        l3_reasons_combined,
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
    # Data Format: (Net Rtg, Off Rtg, Def Rtg, TS%, REB%, Pace, Win%)
    teams = {
        "76ers": (-0.6, 114.4, 115.0, 0.573, 0.489, 99.98, 0.530),
        "Pistons": (7.8, 116.6, 108.8, 0.577, 0.523, 100.14, 0.723),
        "Wizards": (-11.0, 109.5, 120.6, 0.564, 0.476, 102.32, 0.250),
        "Magic": (1.4, 114.3, 112.9, 0.575, 0.502, 100.01, 0.563),
        "Suns": (1.2, 113.8, 112.6, 0.568, 0.497, 98.24, 0.585),
        "Pacers": (-8.4, 108.6, 117.0, 0.558, 0.477, 102.03, 0.231),
        "Bucks": (-4.7, 112.5, 117.2, 0.588, 0.482, 98.39, 0.422),
        "Heat": (3.5, 114.9, 111.4, 0.575, 0.500, 104.71, 0.561),
        "Nets": (-8.8, 109.8, 118.6, 0.563, 0.487, 97.13, 0.262),
        "Hawks": (0.5, 114.2, 113.6, 0.582, 0.489, 102.85, 0.523),
        "Mavericks": (-4.7, 109.5, 114.2, 0.563, 0.494, 102.34, 0.323),
        "Grizzlies": (-2.5, 113.7, 116.3, 0.574, 0.495, 101.47, 0.359),
        "Nuggets": (4.2, 120.3, 116.1, 0.612, 0.508, 99.01, 0.606),
        "Spurs": (7.3, 117.6, 110.3, 0.593, 0.514, 100.88, 0.738),
        "Celtics": (7.9, 119.8, 111.8, 0.579, 0.521, 95.40, 0.662),
        "Thunder": (10.8, 117.0, 106.2, 0.598, 0.487, 100.60, 0.773),
        "Lakers": (0.8, 116.6, 115.7, 0.607, 0.498, 99.35, 0.615),
        "Bulls": (-4.4, 112.2, 116.6, 0.581, 0.496, 102.45, 0.415)
    }

    matchups = [
        ("Nuggets", "Spurs", 228.5),
        ("Celtics", "Thunder", 225.5),
        ("Lakers", "Bulls", 230.5)
    ]
    
    for away, home, ou_line in matchups:
        test_matchup(away, teams[away], home, teams[home], ou_line)

if __name__ == "__main__":
    main()
