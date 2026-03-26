import datetime

from calibration import fit_platt_scaler_from_log, load_platt_scaler
from data_models import GameContext, Venue
from injury_impact import calculate_injury_adjustment
from injury_report import fetch_injuries, get_team_out_players
from live_data import fetch_all_team_stats, fetch_todays_games
from prediction_logging import load_prediction_rows, write_prediction_rows
from probability_layers import (
    calculate_layer1_base_rate,
    calculate_layer2_situational,
    calculate_layer3_matchup_modifier,
    combine_home_win_probability,
    extremize_probability,
    predict_spread,
)
from run_live import create_team

TARGET_DATE = "2026-03-25"
SEED_RESULTS = [
    ("Bulls", "76ers", "76ers"),
    ("Hawks", "Pistons", "Hawks"),
    ("Lakers", "Pacers", "Pacers"),
    ("Heat", "Cavaliers", "Heat"),
    ("Thunder", "Celtics", "Celtics"),
    ("Grizzlies", "Spurs", "Spurs"),
    ("Wizards", "Jazz", "Wizards"),
    ("Rockets", "Timberwolves", "Timberwolves"),
    ("Bucks", "Trail Blazers", "Trail Blazers"),
    ("Mavericks", "Nuggets", "Nuggets"),
    ("Nets", "Warriors", "Warriors"),
    ("Raptors", "Clippers", "Clippers"),
]


def build_new_prediction_row(
    game_info: dict,
    all_team_stats: dict,
    injuries: dict,
    calibrator,
    calibration_flag: str,
) -> dict | None:
    away_name = game_info["away_team"]
    home_name = game_info["home_team"]
    if away_name not in all_team_stats or home_name not in all_team_stats:
        return None

    away_team = create_team(1, away_name, False, all_team_stats[away_name])
    home_team = create_team(2, home_name, True, all_team_stats[home_name])
    venue = Venue(game_info.get("arena", f"{home_name} Arena"))
    ctx = GameContext(
        TARGET_DATE,
        venue,
        home_team,
        away_team,
        0.50,
        game_id=game_info.get("game_id", ""),
    )

    l1_prob = calculate_layer1_base_rate(home_team, away_team, ctx.h2h_home_win_pct_2yr)

    l2_adj, _ = calculate_layer2_situational(home_team, away_team, ctx)
    l2_away_adj, _ = calculate_layer2_situational(away_team, home_team, ctx)
    total_l2_adj = l2_adj - l2_away_adj

    away_out = get_team_out_players(injuries, away_name)
    home_out = get_team_out_players(injuries, home_name)
    inj_adj_home = 0.0
    inj_adj_away = 0.0
    if home_out:
        inj_adj_home, _ = calculate_injury_adjustment(home_name, home_out)
    if away_out:
        inj_adj_away, _ = calculate_injury_adjustment(away_name, away_out)

    total_injury_adj = inj_adj_home - inj_adj_away
    home_team.injury_flag = int(bool(injuries.get(home_name, [])))
    away_team.injury_flag = int(bool(injuries.get(away_name, [])))
    home_team.injury_severity = min(0.25, abs(inj_adj_home) + 0.01 * len(injuries.get(home_name, [])))
    away_team.injury_severity = min(0.25, abs(inj_adj_away) + 0.01 * len(injuries.get(away_name, [])))

    total_l2_with_inj = max(-0.25, min(0.25, total_l2_adj + total_injury_adj))

    l3_adj, _ = calculate_layer3_matchup_modifier(home_team, away_team)
    l3_away_adj, _ = calculate_layer3_matchup_modifier(away_team, home_team)
    total_l3_adj = l3_adj - l3_away_adj

    layer_prob = l1_prob + total_l2_with_inj + total_l3_adj
    predicted_spread = predict_spread(home_team, away_team)
    raw_prob = combine_home_win_probability(layer_prob, predicted_spread, home_team, away_team)
    extremized_prob = extremize_probability(raw_prob)
    calibrated_prob = calibrator.calibrate(extremized_prob)

    return {
        "game_id": game_info.get("game_id", "") or f"{TARGET_DATE}:{away_name}@{home_name}",
        "game_date": TARGET_DATE,
        "away_team": away_name,
        "home_team": home_name,
        "predicted_spread": f"{predicted_spread:.4f}",
        "raw_probability": f"{extremized_prob:.6f}",
        "calibrated_probability": f"{calibrated_prob:.6f}",
        "calibration_flag": calibration_flag,
        "is_home": int(home_team.is_home),
        "rest_days_home": f"{home_team.team_stats.rest_days:.2f}",
        "rest_days_away": f"{away_team.team_stats.rest_days:.2f}",
        "back_to_back_home": int(home_team.team_stats.back_to_back_flag),
        "back_to_back_away": int(away_team.team_stats.back_to_back_flag),
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "actual_home_win": "",
    }


def main():
    log_path, rows = load_prediction_rows()
    row_lookup = {}
    for row in rows:
        key = (row.get("game_date", ""), row.get("away_team", ""), row.get("home_team", ""))
        row_lookup[key] = row

    print(f"Loaded {len(rows)} prediction log rows from {log_path}")

    print(f"Fetching games and team context for {TARGET_DATE}...")
    games = fetch_todays_games(TARGET_DATE)
    game_lookup = {(game["away_team"], game["home_team"]): game for game in games}
    all_team_stats = fetch_all_team_stats(as_of_date=TARGET_DATE)
    injuries = fetch_injuries()

    calibrator, calibration_diag = load_platt_scaler(log_path=str(log_path))
    print(f"Runtime calibration status: {calibration_diag.note}")

    created_rows = 0
    updated_rows = 0
    for away_name, home_name, winner_name in SEED_RESULTS:
        matchup = (away_name, home_name)
        game_info = game_lookup.get(matchup)
        resolved_away = away_name
        resolved_home = home_name
        if not game_info:
            reversed_matchup = (home_name, away_name)
            game_info = game_lookup.get(reversed_matchup)
            if game_info:
                resolved_away, resolved_home = reversed_matchup
            else:
                raise RuntimeError(f"Could not find March 25 game for {away_name} @ {home_name}")

        actual_home_win = int(winner_name == resolved_home)
        row_key = (TARGET_DATE, resolved_away, resolved_home)
        row = row_lookup.get(row_key)
        if row is None:
            row = build_new_prediction_row(
                game_info,
                all_team_stats,
                injuries,
                calibrator,
                calibration_diag.log_flag,
            )
            if row is None:
                raise RuntimeError(f"Could not build prediction row for {resolved_away} @ {resolved_home}")
            rows.append(row)
            row_lookup[row_key] = row
            created_rows += 1

        row["actual_home_win"] = str(actual_home_win)
        if calibration_diag.log_flag and not row.get("calibration_flag"):
            row["calibration_flag"] = calibration_diag.log_flag
        updated_rows += 1

    write_prediction_rows(rows, log_path=log_path)
    print(f"Seeded outcomes for {updated_rows} March 25 games ({created_rows} row(s) created).")

    partial_scaler, partial_diag = fit_platt_scaler_from_log(log_path=str(log_path), min_samples=1)
    print("Partial fit summary:")
    print(f"- samples: {partial_diag.sample_count}")
    print(f"- fitted: {partial_diag.fitted}")
    print(f"- note: {partial_diag.note}")
    print(f"- coefficients: a={partial_scaler.a:.6f}, b={partial_scaler.b:.6f}")
    if partial_diag.sample_count < 50:
        print("- runtime gate remains active until 50 realized outcomes are logged.")


if __name__ == "__main__":
    main()
