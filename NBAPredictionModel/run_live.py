"""
NBA Prediction Model — Full Live Pipeline (All Games on a Selected Date)
Supports:
  - NBANEW: refined probability, spread blend, and calibration wrapper
  - NBAOLD: legacy confidence path for side-by-side dashboard comparison
"""
import argparse
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
    legacy_extremize_probability,
    legacy_predict_spread,
    legacy_predict_total_points,
    predict_total_points,
    predict_spread,
)
from injury_impact import calculate_injury_adjustment
from injury_report import fetch_injuries, get_team_out_players
from main import format_output, format_output_new
from live_data import fetch_all_team_stats, fetch_todays_games, fetch_espn_total_lines


def _normalize_target_date(raw_value: str | None) -> str:
    if not raw_value:
        return datetime.date.today().isoformat()

    value = str(raw_value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return datetime.date.today().isoformat()


def _normalize_market_total_line(raw_value) -> float | None:
    if raw_value is None:
        return None
    try:
        line = float(raw_value)
    except (TypeError, ValueError):
        return None
    if line <= 0:
        return None
    return line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NBA model for a selected date and variant.")
    parser.add_argument("legacy_date", nargs="?", default="", help="Optional legacy date arg in MM/DD/YYYY or YYYY-MM-DD format.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    parser.add_argument("--variant", choices=("new", "old"), default="new", help="Model variant to run.")
    parser.add_argument("--no-log", action="store_true", help="Disable local prediction logging.")
    return parser.parse_args()


def create_team(id_num, name, is_home, stats_dict):
    stats = TeamStats(
        net_rating=stats_dict["net_rating"],
        off_rating_10=stats_dict["off_rating"],
        def_rating_10=stats_dict["def_rating"],
        ts_pct=stats_dict["ts_pct"],
        reb_pct=stats_dict["reb_pct"],
        pace=stats_dict["pace"],
        last_10_win_pct=stats_dict.get("recent_10_win_pct", stats_dict["win_pct"]),
        is_b2b_second_leg=stats_dict.get("back_to_back_flag", False),
        is_3_in_4_nights=stats_dict.get("is_3_in_4_nights", False),
        season_win_pct=stats_dict["win_pct"],
        recent_5_win_pct=stats_dict.get("recent_5_win_pct", stats_dict["win_pct"]),
        recent_10_win_pct=stats_dict.get("recent_10_win_pct", stats_dict["win_pct"]),
        weighted_win_pct=stats_dict.get("weighted_win_pct", stats_dict["win_pct"]),
        recent_5_point_diff=stats_dict.get("recent_5_point_diff", stats_dict["net_rating"]),
        recent_10_point_diff=stats_dict.get("recent_10_point_diff", stats_dict["last10_net_rating"]),
        weighted_point_diff=stats_dict.get("weighted_point_diff", stats_dict["net_rating"]),
        recent_5_total_points=stats_dict.get("recent_5_total_points", 225.0),
        recent_10_total_points=stats_dict.get("recent_10_total_points", 225.0),
        rest_days=stats_dict.get("rest_days", 1.0),
        back_to_back_flag=stats_dict.get("back_to_back_flag", False),
        efg_pct=stats_dict.get("efg_pct", stats_dict.get("ts_pct", 0.5)),
        tov_pct=stats_dict.get("tov_pct", 0.13),
    )
    # Feed the NBANEW spread model per-game scoring context directly in point units.
    pace = float(stats_dict["pace"])
    setattr(
        stats,
        "points_per_game",
        float(stats_dict.get("points_per_game", stats_dict["off_rating"] * pace / 100.0)),
    )
    setattr(
        stats,
        "opp_points_per_game",
        float(stats_dict.get("opp_points_per_game", stats_dict["def_rating"] * pace / 100.0)),
    )
    setattr(stats, "is_b2b", bool(stats_dict.get("back_to_back_flag", False)))
    p1 = Player(id_num * 10 + 1, "Player 1", name, "G", "Active", 25.0)
    p2 = Player(id_num * 10 + 2, "Player 2", name, "F", "Active", 25.0)
    p3 = Player(id_num * 10 + 3, "Player 3", name, "C", "Active", 20.0)
    return Team(id_num, name, is_home, stats, [p1, p2, p3])


def run_game(
    game_info,
    all_team_stats,
    injuries,
    variant="new",
    calibrator=None,
    calibration_note="",
    calibration_flag="",
    ou_line=None,
    game_date="",
    should_log=False,
):
    away_name = game_info["away_team"]
    home_name = game_info["home_team"]

    print(f"\n{'=' * 80}")
    print(f"GAME: {away_name} @ {home_name} ({game_info['game_status']})")
    print(f"{'=' * 80}")

    if away_name not in all_team_stats or home_name not in all_team_stats:
        print(f"ERROR: Could not find team stats for {away_name} or {home_name}")
        return

    away_stats = all_team_stats[away_name]
    home_stats = all_team_stats[home_name]

    away_team = create_team(1, away_name, False, away_stats)
    home_team = create_team(2, home_name, True, home_stats)

    venue_name = game_info.get("arena", f"{home_name} Arena")
    venue = Venue(venue_name)
    ctx = GameContext(
        game_date or datetime.date.today().isoformat(),
        venue,
        home_team,
        away_team,
        0.50,
        game_id=game_info.get("game_id", ""),
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
        print(f"  Fetching on/off court data for {home_name} ({len(home_out)} OUT)...")
        inj_adj_home, inj_reason_home = calculate_injury_adjustment(home_name, home_out)
        time.sleep(1)
    if away_out:
        print(f"  Fetching on/off court data for {away_name} ({len(away_out)} OUT)...")
        inj_adj_away, inj_reason_away = calculate_injury_adjustment(away_name, away_out)
        time.sleep(1)

    total_injury_adj = inj_adj_home - inj_adj_away
    home_team.injury_flag = int(bool(injuries.get(home_name, [])))
    away_team.injury_flag = int(bool(injuries.get(away_name, [])))
    home_team.injury_severity = min(0.25, abs(inj_adj_home) + 0.01 * len(injuries.get(home_name, [])))
    away_team.injury_severity = min(0.25, abs(inj_adj_away) + 0.01 * len(injuries.get(away_name, [])))
    home_team.injury_summary = inj_reason_home
    away_team.injury_summary = inj_reason_away

    l2_combined = f"Sit: {l2_reasons} | Inj [{home_name}: {inj_reason_home}] | [{away_name}: {inj_reason_away}]"
    total_l2_with_inj = max(-0.25, min(0.25, total_l2_adj + total_injury_adj))

    l3_adj, l3_reasons = calculate_layer3_matchup_modifier(home_team, away_team)
    l3_away_adj, l3_away_reasons = calculate_layer3_matchup_modifier(away_team, home_team)
    total_l3_adj = l3_adj - l3_away_adj
    l3_combined = f"{home_name}: {l3_reasons} | {away_name}: {l3_away_reasons}"

    layer_prob = l1_prob + total_l2_with_inj + total_l3_adj

    if variant == "old":
        raw_prob = layer_prob
        ext_prob = legacy_extremize_probability(layer_prob)
        calibrated_prob = ext_prob
        predicted_spread = legacy_predict_spread(calibrated_prob)
        predicted_total = legacy_predict_total_points(ctx)
        variant_note = "Legacy confidence output. No spread-blend layer or Platt scaling is applied."
        decision_override = None
        decision_note = ""
    else:
        predicted_spread = predict_spread(home_team, away_team)
        raw_prob = combine_home_win_probability(layer_prob, predicted_spread, home_team, away_team)
        ext_prob = extremize_probability(raw_prob)
        calibrated_prob = calibrator.calibrate(ext_prob) if calibrator else ext_prob
        predicted_total = predict_total_points(ctx)
        variant_note = calibration_note
        decision_override = None
        decision_note = ""
        # Only pass on true toss-ups.
        MIN_MARGIN_TO_BET = 1.5
        if abs(predicted_spread) < MIN_MARGIN_TO_BET:
            decision_override = "PASS"
            decision_note = f"Projected spread {predicted_spread:+.2f} is below the {MIN_MARGIN_TO_BET:.1f}-point minimum."

    formatter_kwargs = {
        "predicted_spread": predicted_spread,
        "predicted_total": predicted_total,
        "calibration_note": variant_note,
        "log_prediction": should_log and variant == "new",
        "log_calibration_flag": calibration_flag if variant == "new" else "",
    }
    if variant == "new":
        format_output_new(
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
            decision_override=decision_override,
            decision_note=decision_note,
            **formatter_kwargs,
        )
    else:
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
            **formatter_kwargs,
        )

    market_total_line = _normalize_market_total_line(ou_line)
    line_display = f"{market_total_line:.1f}" if market_total_line is not None else "N/A"
    print(f"**Over/Under:** Model Total {predicted_total:.1f} vs Line {line_display}")

    if variant == "new":
        totals_decision = "PASS"
        if market_total_line not in (None, 225.0):
            if predicted_total > market_total_line + 3.5:
                totals_decision = "BET OVER"
            elif predicted_total < market_total_line - 3.5:
                totals_decision = "BET UNDER"
        print(f"**O/U Decision: {totals_decision}**")
    else:
        if market_total_line is None:
            market_total_line = 225.0
            line_display = f"{market_total_line:.1f}"
            print(f"**Over/Under (legacy fallback):** Model Total {predicted_total:.1f} vs Line {line_display}")
        if predicted_total > market_total_line + 3:
            print("**O/U Decision: BET OVER**")
        elif predicted_total < market_total_line - 3:
            print("**O/U Decision: BET UNDER**")
        else:
            print("**O/U Decision: PASS**")
    print()


def main():
    args = parse_args()
    target_date = _normalize_target_date(args.date or args.legacy_date)
    today_iso = datetime.date.today().isoformat()
    should_log = (not args.no_log) and args.variant == "new" and target_date == today_iso
    variant_label = "NBANEW" if args.variant == "new" else "NBAOLD"

    print("=" * 80)
    print(f"🏀 NBA PREDICTION MODEL — {variant_label}")
    print(f"   Requested Date: {target_date}")
    print(f"   Run Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("   Data: Official NBA Injury Reports + NBA API Stats + On/Off Court Impact")
    if args.variant == "old":
        print("   Mode: Legacy confidence path for side-by-side comparison")
    elif not should_log:
        print("   Logging: disabled for non-today or explicit no-log runs")
    print("=" * 80)

    print(f"\nFetching NBA games for {target_date}...")
    games = fetch_todays_games(target_date)

    if not games:
        print("No games found for today.")
        return

    print(f"Found {len(games)} games.")

    print("\nFetching market O/U lines from ESPN...")
    total_lines = fetch_espn_total_lines(target_date)
    if total_lines:
        print(f"Found totals for {len(total_lines)} game(s).")
    else:
        if args.variant == "new":
            print("No market totals found. NBANEW totals will PASS without a market line.")
        else:
            print("No market totals found. Falling back to baseline 225.0 where needed.")

    print("\nFetching team stats for all NBA teams...")
    all_team_stats = fetch_all_team_stats(as_of_date=target_date)

    print("\nFetching official NBA injury report...")
    injuries = fetch_injuries()

    calibrator = None
    calibration_note = ""
    calibration_flag = ""
    if args.variant == "new":
        calibrator, calibration_diag = load_platt_scaler()
        calibration_note = calibration_diag.note
        calibration_flag = calibration_diag.log_flag

    if target_date != today_iso:
        print("\nHistorical date note: injury data is pulled from the current official report, not an archived injury snapshot.")

    print("\n\nRUNNING PREDICTIONS FOR ALL GAMES\n")

    for game in games:
        key = (game["away_team"], game["home_team"])
        if args.variant == "new":
            ou_line = total_lines.get(key)
        else:
            ou_line = total_lines.get(key, 225.0)
        run_game(
            game,
            all_team_stats,
            injuries,
            variant=args.variant,
            calibrator=calibrator,
            calibration_note=calibration_note,
            calibration_flag=calibration_flag,
            ou_line=ou_line,
            game_date=target_date,
            should_log=should_log,
        )


if __name__ == "__main__":
    main()
