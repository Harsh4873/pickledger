import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

def get_real_stats():
    # Fetch advanced team stats for the current season
    stats = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense='Advanced',
        season='2025-26',
        per_mode_detailed='PerGame'
    )
    df = stats.get_data_frames()[0]
    
    # We need: TEAM_NAME, NET_RATING, OFF_RATING, DEF_RATING, TS_PCT, REB_PCT, PACE
    cols = ['TEAM_NAME', 'NET_RATING', 'OFF_RATING', 'DEF_RATING', 'TS_PCT', 'REB_PCT', 'PACE', 'W_PCT']
    df_filtered = df[cols]
    
    print(df_filtered.to_string())

if __name__ == "__main__":
    get_real_stats()
