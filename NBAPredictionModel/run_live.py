"""
NBA Prediction Model — Full Live Pipeline (All Today's Games)
Uses:
  - live_data.py for today's games and real team stats from NBA API
  - nbainjuries package for OFFICIAL injury reports (no API key)
  - injury_impact.py for on/off court data-driven adjustments
"""
import datetime
import time
from calibration import load_platt_scaler
from data_models import Player, TeamStats, Team, Venue, GameContext
from probability_layers import (
    calculate_layer1_base_rate,
    calculate_layer2_situational,
    calculate_layer3_matchup_modifier,
    combine_home_win_probability,
    extremize_probability,
    predict_total_points,
    predict_spread
)
from injury_impact import calculate_injury_adjustment
from injury_report import fetch_injuries, get_team_out_players
from main import format_output
from live_data import fetch_all_team_stats, fetch_todays_games, fetch_espn_total_lines

def create_team(id_num, name, is_home, stats_dict):
    stats = TeamStats(
        net_rating=stats_dict['net_rating'], 
        off_rating_10=stats_dict['off_rating'], 
        def_rating_10=stats_dict['def_rating'],
        ts_pct=stats_dict['ts_pct'], 
        reb_pct=stats_dict['reb_pct'], 
        pace=stats_dict['pace'],
        last_10_win_pct=stats_dict.get('recent_10_win_pct', stats_dict['win_pct']),
        is_b2b_second_leg=stats_dict.get('back_to_back_flag', False),
        is_3_in_4_nights=stats_dict.get('is_3_in_4_nights', False),
        season_win_pct=stats_dict['win_pct'],
        recent_5_win_pct=stats_dict.get('recent_5_win_pct', stats_dict['win_pct']),
        recent_10_win_pct=stats_dict.get('recent_10_win_pct', stats_dict['win_pct']),
        weighted_win_pct=stats_dict.get('weighted_win_pct', stats_dict['win_pct']),
        recent_5_point_diff=stats_dict.get('recent_5_point_diff', stats_dict['net_rating']),
        recent_10_point_diff=stats_dict.get('recent_10_point_diff', stats_dict['last10_net_rating']),
        weighted_point_diff=stats_dict.get('weighted_point_diff', stats_dict['net_rating']),
        recent_5_total_points=stats_dict.get('recent_5_total_points', 225.0),
        recent_10_total_points=stats_dict.get('recent_10_total_points', 225.0),
        rest_days=stats_dict.get('rest_days', 1.0),
        back_to_back_flag=stats_dict.get('back_to_back_flag', False),
    )
    # create placeholder active roster for verification display, since we handle injuries via on/off impact
    p1 = Player(id_num*10+1, "Player 1", name, "G", "Active", 25.0)
    p2 = Player(id_num*10+2, "Player 2", name, "F", "Active", 25.0)
    p3 = Player(id_num*10+3, "Player 3", name, "C", "Active", 20.0)
    return Team(id_num, name, is_home, stats, [p1, p2, p3])

def run_game(game_info, all_team_stats, injuries, calibrator, calibration_note: str, ou_line=225.0):
    away_name = game_info['away_team']
    home_name = game_info['home_team']
    
    print(f"\n{'='*80}")
    print(f"GAME: {away_name} @ {home_name} ({game_info['game_status']})")
    print(f"{'='*80}")
    
    # Check if we have stats for these teams
    if away_name not in all_team_stats or home_name not in all_team_stats:
        print(f"ERROR: Could not find team stats for {away_name} or {home_name}")
        return
        
    away_stats = all_team_stats[away_name]
    home_stats = all_team_stats[home_name]
    
    # Create teams
    away_team = create_team(1, away_name, False, away_stats)
    home_team = create_team(2, home_name, True, home_stats)
    
    venue_name = game_info.get('arena', f"{home_name} Arena")
    venue = Venue(venue_name)
    ctx = GameContext(
        datetime.datetime.now().strftime("%Y-%m-%d"),
        venue,
        home_team,
        away_team,
        0.50,
        game_id=game_info.get('game_id', ''),
    )
    
    # Layer 1: Base Rate
    l1_prob = calculate_layer1_base_rate(home_team, away_team, ctx.h2h_home_win_pct_2yr)
    
    # Layer 2: Standard Situational
    l2_adj, l2_reasons = calculate_layer2_situational(home_team, away_team, ctx)
    l2_away_adj, l2_away_reasons = calculate_layer2_situational(away_team, home_team, ctx)
    total_l2_adj = l2_adj - l2_away_adj
    
    # Layer 2.5: Injury Impact (from real on/off court data)
    away_out = get_team_out_players(injuries, away_name)
    home_out = get_team_out_players(injuries, home_name)
    
    inj_adj_home, inj_reason_home = (0.0, "No OUT players")
    inj_adj_away, inj_reason_away = (0.0, "No OUT players")
    
    if home_out:
        print(f"  🔍 Fetching on/off court data for {home_name} ({len(home_out)} OUT)...")
        inj_adj_home, inj_reason_home = calculate_injury_adjustment(home_name, home_out)
        time.sleep(1) # Rate limiting
    if away_out:
        print(f"  🔍 Fetching on/off court data for {away_name} ({len(away_out)} OUT)...")
        inj_adj_away, inj_reason_away = calculate_injury_adjustment(away_name, away_out)
        time.sleep(1) # Rate limiting
    
    total_injury_adj = inj_adj_home - inj_adj_away
    home_team.injury_flag = int(bool(injuries.get(home_name, [])))
    away_team.injury_flag = int(bool(injuries.get(away_name, [])))
    home_team.injury_severity = min(0.25, abs(inj_adj_home) + 0.01 * len(injuries.get(home_name, [])))
    away_team.injury_severity = min(0.25, abs(inj_adj_away) + 0.01 * len(injuries.get(away_name, [])))
    home_team.injury_summary = inj_reason_home
    away_team.injury_summary = inj_reason_away
    
    l2_combined = f"Sit: {l2_reasons} | Inj [{home_name}: {inj_reason_home}] | [{away_name}: {inj_reason_away}]"
    total_l2_with_inj = max(-0.25, min(0.25, total_l2_adj + total_injury_adj))
    
    # Layer 3: Matchup
    l3_adj, l3_reasons = calculate_layer3_matchup_modifier(home_team, away_team)
    l3_away_adj, l3_away_reasons = calculate_layer3_matchup_modifier(away_team, home_team)
    total_l3_adj = l3_adj - l3_away_adj
    l3_combined = f"{home_name}: {l3_reasons} | {away_name}: {l3_away_reasons}"
    
    layer_prob = l1_prob + total_l2_with_inj + total_l3_adj
    predicted_spread = predict_spread(home_team, away_team)
    raw_prob = combine_home_win_probability(layer_prob, predicted_spread, home_team, away_team)
    ext_prob = extremize_probability(raw_prob)
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
        calibration_note=calibration_note,
    )
    
    predicted_total = predict_total_points(ctx)
    print(f"**Over/Under:** Model Total {predicted_total:.1f} vs Line {ou_line}")
    if predicted_total > ou_line + 3:
        print(f"**O/U Decision: BET OVER**")
    elif predicted_total < ou_line - 3:
        print(f"**O/U Decision: BET UNDER**")
    else:
        print(f"**O/U Decision: PASS**")
    print()

def main():
    print("="*80)
    print("🏀 NBA PREDICTION MODEL — FULL SLATE")
    print(f"   Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("   Data: Official NBA Injury Reports + NBA API Stats + On/Off Court Impact")
    print("="*80)
    
    target_date = datetime.datetime.now().strftime('%Y-%m-%d')

    print("\n📡 Fetching today's games...")
    games = fetch_todays_games(target_date)
    
    if not games:
        print("No games found for today.")
        return
        
    print(f"Found {len(games)} games.")

    print("\n📡 Fetching market O/U lines from ESPN...")
    total_lines = fetch_espn_total_lines(target_date)
    if total_lines:
        print(f"Found totals for {len(total_lines)} game(s).")
    else:
        print("No market totals found. Falling back to baseline 225.0 where needed.")
    
    print("\n📡 Fetching team stats for all NBA teams...")
    all_team_stats = fetch_all_team_stats(as_of_date=target_date)
    
    print("\n📡 Fetching official NBA injury report...")
    injuries = fetch_injuries()
    calibrator, calibration_diag = load_platt_scaler()
    
    print("\n\n🎯 RUNNING PREDICTIONS FOR ALL GAMES\n")
    
    # Loop over all games today
    for game in games:
        key = (game['away_team'], game['home_team'])
        ou_line = total_lines.get(key, 225.0)
        run_game(game, all_team_stats, injuries, calibrator, calibration_diag.note, ou_line=ou_line)

if __name__ == "__main__":
    main()
