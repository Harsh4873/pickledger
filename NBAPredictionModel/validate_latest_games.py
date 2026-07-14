import datetime

from calibration import load_platt_scaler
from data_models import GameContext, Player, Team, TeamStats, Venue
from injury_impact import calculate_injury_adjustment
from injury_report import fetch_injuries, get_team_out_players
from live_data import fetch_all_team_stats, fetch_todays_games
from probability_layers import (
    calculate_layer1_base_rate,
    calculate_layer2_situational,
    calculate_layer3_matchup_modifier,
    combine_home_win_probability,
    extremize_probability,
    legacy_extremize_probability,
    predict_spread,
)


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
    lineup = [
        Player(id_num * 10 + 1, "Player 1", name, "G", "Active", 25.0),
        Player(id_num * 10 + 2, "Player 2", name, "F", "Active", 25.0),
        Player(id_num * 10 + 3, "Player 3", name, "C", "Active", 20.0),
    ]
    return Team(id_num, name, is_home, stats, lineup)


def latest_games(max_lookback_days: int = 7) -> tuple[str, list[dict]]:
    today = datetime.date.today()
    for offset in range(max_lookback_days + 1):
        target_date = today - datetime.timedelta(days=offset)
        date_str = target_date.strftime("%Y-%m-%d")
        games = fetch_todays_games(date_str)
        if games:
            return date_str, games
    raise RuntimeError(f"No NBA games found in the last {max_lookback_days + 1} days.")


def build_probabilities(game_info: dict, all_team_stats: dict, injuries: dict, calibrator) -> dict | None:
    away_name = game_info['away_team']
    home_name = game_info['home_team']
    if away_name not in all_team_stats or home_name not in all_team_stats:
        return None

    away_team = create_team(1, away_name, False, all_team_stats[away_name])
    home_team = create_team(2, home_name, True, all_team_stats[home_name])
    venue = Venue(game_info.get('arena', f"{home_name} Arena"))
    ctx = GameContext(
        datetime.datetime.now().strftime("%Y-%m-%d"),
        venue,
        home_team,
        away_team,
        0.50,
        game_id=game_info.get('game_id', ''),
    )

    l1_prob = calculate_layer1_base_rate(home_team, away_team, ctx.h2h_home_win_pct_2yr)
    l2_adj, l2_reasons = calculate_layer2_situational(home_team, away_team, ctx)
    l2_away_adj, l2_away_reasons = calculate_layer2_situational(away_team, home_team, ctx)
    total_l2_adj = l2_adj - l2_away_adj

    away_out = get_team_out_players(injuries, away_name)
    home_out = get_team_out_players(injuries, home_name)
    inj_adj_home, inj_reason_home = (0.0, "No OUT players")
    inj_adj_away, inj_reason_away = (0.0, "No OUT players")
    if home_out:
        inj_adj_home, inj_reason_home = calculate_injury_adjustment(home_name, home_out)
    if away_out:
        inj_adj_away, inj_reason_away = calculate_injury_adjustment(away_name, away_out)
    total_injury_adj = inj_adj_home - inj_adj_away

    home_team.injury_flag = int(bool(injuries.get(home_name, [])))
    away_team.injury_flag = int(bool(injuries.get(away_name, [])))
    home_team.injury_severity = min(0.25, abs(inj_adj_home) + 0.01 * len(injuries.get(home_name, [])))
    away_team.injury_severity = min(0.25, abs(inj_adj_away) + 0.01 * len(injuries.get(away_name, [])))
    home_team.injury_summary = inj_reason_home
    away_team.injury_summary = inj_reason_away

    total_l2_with_inj = max(-0.25, min(0.25, total_l2_adj + total_injury_adj))

    l3_adj, l3_reasons = calculate_layer3_matchup_modifier(home_team, away_team)
    l3_away_adj, l3_away_reasons = calculate_layer3_matchup_modifier(away_team, home_team)
    total_l3_adj = l3_adj - l3_away_adj

    layer_prob = l1_prob + total_l2_with_inj + total_l3_adj
    old_confidence = legacy_extremize_probability(layer_prob)
    predicted_spread = predict_spread(home_team, away_team)
    raw_prob = combine_home_win_probability(layer_prob, predicted_spread, home_team, away_team)
    extremized_prob = extremize_probability(raw_prob)
    calibrated_prob = calibrator.calibrate(extremized_prob)

    old_pick_prob = old_confidence if old_confidence >= 0.5 else 1.0 - old_confidence
    new_pick_prob = calibrated_prob if calibrated_prob >= 0.5 else 1.0 - calibrated_prob
    old_side = home_name if old_confidence >= 0.5 else away_name
    new_side = home_name if calibrated_prob >= 0.5 else away_name

    return {
        "matchup": f"{away_name} @ {home_name}",
        "status": game_info.get("game_status", ""),
        "old_side": old_side,
        "new_side": new_side,
        "old_confidence": old_pick_prob,
        "new_confidence": new_pick_prob,
        "predicted_spread": predicted_spread,
        "layer_prob": layer_prob,
        "extremized_prob": extremized_prob,
        "raw_prob": raw_prob,
        "l2_reasons": f"{home_name}: {l2_reasons} | {away_name}: {l2_away_reasons}",
        "l3_reasons": f"{home_name}: {l3_reasons} | {away_name}: {l3_away_reasons}",
    }


def main():
    latest_date, games = latest_games()
    print(f"Latest available games date: {latest_date}")
    all_team_stats = fetch_all_team_stats(as_of_date=latest_date)
    injuries = fetch_injuries()
    calibrator, calibration_diag = load_platt_scaler()
    print(f"Calibration status: {calibration_diag.note}")

    printed = 0
    for game in games:
        sample = build_probabilities(game, all_team_stats, injuries, calibrator)
        if not sample:
            continue
        printed += 1
        print("\n" + "=" * 80)
        print(f"Game: {sample['matchup']} ({sample['status']})")
        print(f"Old confidence: {sample['old_side']} {sample['old_confidence'] * 100:.1f}%")
        print(f"New confidence: {sample['new_side']} {sample['new_confidence'] * 100:.1f}%")
        print(f"Delta: {(sample['new_confidence'] - sample['old_confidence']) * 100:+.1f} pts")
        print(f"Predicted spread (new independent model): {sample['predicted_spread']:+.2f}")
        print(f"Layer probability before extremizing: {sample['layer_prob'] * 100:.1f}%")
        print(f"Raw blended probability before calibration: {sample['extremized_prob'] * 100:.1f}%")
        if printed >= 3:
            break

    if printed == 0:
        raise RuntimeError("Could not build any validation samples from the latest available games.")


if __name__ == "__main__":
    main()
