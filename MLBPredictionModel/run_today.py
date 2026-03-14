"""
MLB Prediction Model — Run Today's Full Slate
Fetches today's games from statsapi, runs each through the prediction pipeline,
and prints pipe-delimited output for the server parser.

Usage:
  python run_today.py                # today's games
  python run_today.py 03/15/2026     # specific date
"""
import sys
import datetime
import traceback
import statsapi
from data_models import Player, PitcherStats, TeamStats, Team, Weather, Venue, GameContext
from verification import VerificationGate
from probability_layers import (
    calculate_layer1_base_rate,
    calculate_layer2_situational,
    calculate_layer3_pitcher_modifier,
    extremize_probability,
    predict_total_runs,
    predict_spread,
)
from market_mechanics import (
    convert_american_to_implied,
    remove_vig,
    calculate_edge,
)


def _get_team_stats(team_name: str, win_pct: float) -> TeamStats:
    """Build TeamStats from available data. Spring training uses mock advanced stats."""
    return TeamStats(
        ops=0.750,
        woba=0.320,
        wrc_plus=100,
        last_10_runs_avg=4.5,
        bullpen_pitches_yesterday=40,
        travel_fatigue=False,
        consecutive_games=1,
        home_win_pct=win_pct,
        away_win_pct=win_pct,
        season_win_pct=win_pct,
        last_30_days_win_pct=win_pct,
    )


def _get_pitcher_stats(pitcher_name: str, era: float) -> PitcherStats:
    fip = era + 0.2
    return PitcherStats(
        era=era,
        fip=fip,
        whip=1.20,
        last_5_starts_summary="Spring Training",
        days_rest=5,
        home_split_era=era,
        away_split_era=era,
        woba_vs_l=0.300,
        woba_vs_r=0.300,
        pitches_per_start_avg=85,
    )


def run_game(game: dict) -> None:
    """Run prediction pipeline for a single game and print pipe-delimited output."""
    away_name = game["away_name"]
    home_name = game["home_name"]
    venue_name = game.get("venue_name", "Ballpark")

    venue = Venue(venue_name, 100, 0)
    weather = Weather(75.0, 10.0, "cross", False)

    home_starter_name = game.get("home_probable_pitcher", "TBD")
    away_starter_name = game.get("away_probable_pitcher", "TBD")

    import random
    random.seed(game.get("game_id", 0))
    h_era = round(random.uniform(3.0, 5.0), 2)
    a_era = round(random.uniform(3.0, 5.0), 2)

    h_pitcher = Player(1, home_starter_name, home_name, "SP")
    h_p_stats = _get_pitcher_stats(home_starter_name, h_era)
    h_t_stats = _get_team_stats(home_name, 0.520)
    h_team = Team(game.get("home_id", 0), home_name, True, h_pitcher, h_p_stats, h_t_stats, [])

    a_pitcher = Player(2, away_starter_name, away_name, "SP")
    a_p_stats = _get_pitcher_stats(away_starter_name, a_era)
    a_t_stats = _get_team_stats(away_name, 0.500)
    a_team = Team(game.get("away_id", 0), away_name, False, a_pitcher, a_p_stats, a_t_stats, [])

    ctx = GameContext(game.get("game_date", ""), venue, weather, h_team, a_team, 0.50)

    # Verification
    VerificationGate.run_all_checks(ctx)

    # Layers
    l1_prob = calculate_layer1_base_rate(h_team, True, ctx.h2h_home_win_pct_3yr)
    l2_adj, _ = calculate_layer2_situational(h_team, a_team, ctx)
    l3_adj, _ = calculate_layer3_pitcher_modifier(h_team.starter_stats, a_team.starter_stats)

    raw_prob = l1_prob + l2_adj + l3_adj
    ext_prob = extremize_probability(raw_prob, factor=1.3)

    # Determine winner and probabilities
    home_prob = ext_prob
    away_prob = 1.0 - ext_prob

    # Convert probabilities to American odds
    def prob_to_american(p):
        if p <= 0 or p >= 1:
            return -110
        if p >= 0.5:
            return int(-100 * p / (1 - p))
        else:
            return int(100 * (1 - p) / p)

    home_odds = prob_to_american(home_prob)
    away_odds = prob_to_american(away_prob)

    # Print pipe-delimited line: Away|Home|AwayOdds|HomeOdds|AwayProb|HomeProb
    print(f"---")
    print(f"{away_name}|{home_name}|{away_odds}|{home_odds}|{away_prob:.2f}|{home_prob:.2f}")

    # Over/Under prediction
    predicted_total = predict_total_runs(ctx)
    ou_line = 8.5
    if predicted_total > ou_line + 0.5:
        print(f"OU|OVER|{ou_line}|{predicted_total:.1f}")
    elif predicted_total < ou_line - 0.5:
        print(f"OU|UNDER|{ou_line}|{predicted_total:.1f}")
    else:
        print(f"OU|PASS|{ou_line}|{predicted_total:.1f}")
    print(f"---")


def main():
    # Accept optional date argument (MM/DD/YYYY format)
    if len(sys.argv) > 1:
        date_str = sys.argv[1]
    else:
        date_str = datetime.datetime.now().strftime("%m/%d/%Y")

    print(f"MLB Prediction Model - Games for {date_str}")
    print(f"{'=' * 60}")

    games = statsapi.schedule(date=date_str)
    if not games:
        print(f"No games found for {date_str}.")
        return

    print(f"Found {len(games)} games.\n")

    for game in games:
        try:
            run_game(game)
        except Exception as e:
            print(f"Error processing {game.get('away_name', '?')} vs {game.get('home_name', '?')}: {e}",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)


if __name__ == "__main__":
    main()
