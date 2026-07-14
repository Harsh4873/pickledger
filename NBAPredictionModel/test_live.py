import statsapi
import datetime
import traceback
from data_models import Player, PitcherStats, TeamStats, Team, Weather, Venue, GameContext
from verification import VerificationGate
from probability_layers import calculate_layer1_base_rate, calculate_layer2_situational, calculate_layer3_pitcher_modifier, extremize_probability
from main import format_output

def mock_get_team_stats(team_name: str, win_pct: float) -> TeamStats:
    # We will generate mock but plausible stats for testing purposes
    # Since statsapi does not easily expose advanced metrics like wRC+ in a single call.
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
        last_30_days_win_pct=win_pct
    )

def mock_get_pitcher_stats(pitcher_name: str, era: float) -> PitcherStats:
    fip = era + 0.2
    return PitcherStats(
        era=era, 
        fip=fip, 
        whip=1.20, 
        last_5_starts_summary="Mocked Spring Training Stats", 
        days_rest=5, 
        home_split_era=era, 
        away_split_era=era, 
        woba_vs_l=0.300, 
        woba_vs_r=0.300, 
        pitches_per_start_avg=85
    )


def test_game(game):
    print(f"\\n{'*'*50}")
    print(f"Processing Game: {game['away_name']} vs {game['home_name']}")
    
    venue_name = game.get('venue_name', 'Spring Training Complex')
    venue = Venue(venue_name, 100, 0)
    weather = Weather(75.0, 10.0, "cross", False)
    
    try:
        home_starter_name = game.get('home_probable_pitcher', 'TBD Home')
        away_starter_name = game.get('away_probable_pitcher', 'TBD Away')
        
        import random
        # statsapi uses 'game_id' in schedule endpoint
        random.seed(game['game_id']) 
        h_era = round(random.uniform(3.0, 5.0), 2)
        a_era = round(random.uniform(3.0, 5.0), 2)
        
        h_pitcher = Player(1, home_starter_name, game['home_name'], "SP")
        h_p_stats = mock_get_pitcher_stats(home_starter_name, h_era)
        h_t_stats = mock_get_team_stats(game['home_name'], 0.520)
        h_team = Team(game['home_id'], game['home_name'], True, h_pitcher, h_p_stats, h_t_stats, [])
        
        a_pitcher = Player(2, away_starter_name, game['away_name'], "SP")
        a_p_stats = mock_get_pitcher_stats(away_starter_name, a_era)
        a_t_stats = mock_get_team_stats(game['away_name'], 0.500)
        a_team = Team(game['away_id'], game['away_name'], False, a_pitcher, a_p_stats, a_t_stats, [])
        
        # Game stats
        ctx = GameContext(game['game_date'], venue, weather, h_team, a_team, 0.50)
        
        VerificationGate.run_all_checks(ctx)
        
        l1_prob = calculate_layer1_base_rate(h_team, True, ctx.h2h_home_win_pct_3yr)
        l2_adj, l2_reasons = calculate_layer2_situational(h_team, a_team, ctx)
        l3_adj, l3_reasons = calculate_layer3_pitcher_modifier(h_team.starter_stats, a_team.starter_stats)
        
        raw_prob = l1_prob + l2_adj + l3_adj
        ext_prob = extremize_probability(raw_prob, factor=1.3)
        
        # Format output defaults to -110 standard lines if no live edge is scraped
        format_output(ctx, ext_prob, -110, -110, l1_prob, l2_adj, l2_reasons, l3_adj, l3_reasons, raw_prob, ext_prob)
        
    except Exception as e:
        print(f"Failed to process game {game['away_name']} vs {game['home_name']}: {e}")
        traceback.print_exc()


def main():
    date_str = '03/12/2026'
    print(f"Fetching games for {date_str}...")
    
    # We will get games for 03/12/2026 using statsapi
    games = statsapi.schedule(date=date_str)
    
    if not games:
        print(f"No games found on {date_str}.")
        return

    target_matchups = [
        ("Braves", "Pirates"),
        ("Brewers", "Guardians"),
        ("White Sox", "Giants"),
        ("Reds", "Dodgers"),
        ("Royals", "Padres")
    ]
    
    target_names = [name for matchup in target_matchups for name in matchup]

    for game in games:
        # Match against our requested testing combinations
        if any(team in game['away_name'] or team in game['home_name'] for team in target_names):
            test_game(game)

if __name__ == "__main__":
    main()
