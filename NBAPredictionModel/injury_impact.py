"""
Injury Impact Calculator — Uses real NBA on/off court data.

Instead of guessing flat percentages for injuries, this module pulls real
on-court vs off-court net rating differentials for every player on a team,
then calculates the actual mathematical impact of a player being out.

Formula:
  impact = (on_court_net_rtg - off_court_net_rtg) / 100
  
  A positive impact means the team is WORSE when this player sits.
  A negative impact means the team is actually BETTER without this player.
"""
from nba_api.stats.endpoints import teamplayeronoffsummary
from nba_api.stats.static import teams
import time

_nba_teams = teams.get_teams()

def get_team_id(team_name: str) -> int:
    for t in _nba_teams:
        if team_name.lower() in t['full_name'].lower() or team_name.lower() in t['nickname'].lower():
            return t['id']
    return None

def fetch_player_on_off(team_name: str, season: str = '2025-26') -> dict:
    """
    Fetches on/off court net rating for every player on a team.
    
    Returns a dict:
    {
        'Anthony Davis': {
            'on_net_rtg': -4.6,
            'off_net_rtg': -2.0,
            'impact': -2.6,  # team is 2.6 pts/100 WORSE with him on court (rare/bad fit)
            'gp': 20,
            'minutes': 626.0
        },
        'Cooper Flagg': { ... }
    }
    """
    team_id = get_team_id(team_name)
    if not team_id:
        print(f"WARNING: Could not find team ID for '{team_name}'")
        return {}
    
    # Retry logic with longer timeout for rate-limited NBA API
    for attempt in range(3):
        try:
            time.sleep(2 + attempt * 2)  # Increasing backoff: 2s, 4s, 6s
            onoff = teamplayeronoffsummary.TeamPlayerOnOffSummary(
                team_id=team_id,
                season=season,
                measure_type_detailed_defense='Advanced',
                timeout=60
            )
            dfs = onoff.get_data_frames()
            break
        except Exception as e:
            print(f"  Attempt {attempt+1}/3 failed for {team_name}: {e}")
            if attempt == 2:
                print(f"  WARNING: Could not fetch on/off data for {team_name} after 3 attempts")
                return {}
    
    
    # DataFrame 1 = ON court, DataFrame 2 = OFF court
    on_df = dfs[1]
    off_df = dfs[2]
    
    results = {}
    
    for _, row in on_df.iterrows():
        name = row['VS_PLAYER_NAME']
        # Reformat "Last, First" to "First Last"
        parts = name.split(', ')
        if len(parts) == 2:
            name = f"{parts[1]} {parts[0]}"
        
        on_net = row['NET_RATING']
        gp = row['GP']
        minutes = row['MIN']
        
        # Find their OFF court row
        off_row = off_df[off_df['VS_PLAYER_ID'] == row['VS_PLAYER_ID']]
        off_net = off_row['NET_RATING'].values[0] if len(off_row) > 0 else 0.0
        
        # Impact = how much WORSE the team is without this player
        # Positive = team is worse without them (player is valuable)
        # Negative = team is actually better without them
        impact = on_net - off_net
        mpg = minutes / gp if gp > 0 else 0.0
        
        results[name] = {
            'on_net_rtg': on_net,
            'off_net_rtg': off_net,
            'impact': impact,
            'gp': gp,
            'minutes': minutes,
            'mpg': mpg
        }
    
    return results

def calculate_injury_adjustment(team_name: str, players_out: list, season: str = '2025-26') -> tuple:
    """
    Given a team and a list of player names who are OUT,
    calculate the total probability adjustment using real on/off data.
    
    Returns: (adjustment_float, reason_string)
    """
    player_data = fetch_player_on_off(team_name, season)
    
    if not player_data:
        return 0.0, "Could not fetch on/off data"
    
    total_adj = 0.0
    reasons = []
    
    for player_name in players_out:
        # Try to find a matching player (fuzzy)
        matched = None
        for key in player_data:
            if player_name.lower() in key.lower() or key.lower() in player_name.lower():
                matched = key
                break
        
        if matched:
            data = player_data[matched]
            
            # Filter out tiny sample sizes (< 10 GP) — on/off data is noise
            if data.get('gp', 0) < 10:
                reasons.append(f"{matched} OUT (only {data.get('gp',0)} GP, skipped — too small sample)")
                continue
            
            raw_impact = data['impact']
            mpg = data.get('mpg', 20.0)
            minute_weight = min(1.0, mpg / 30.0)
            
            if raw_impact <= 0:
                # FIX A: Even if on/off says team is "better" without them,
                # high-MPG starters (25+ MPG) still cause rotation disruption.
                # Apply a minimum floor penalty of -3% for starters.
                if mpg >= 25.0:
                    prob_impact = 0.03  # Minimum starter disruption penalty
                    total_adj -= prob_impact
                    reasons.append(
                        f"{matched} OUT ({mpg:.0f} MPG, Impact: {raw_impact:+.1f}, "
                        f"starter floor → {-prob_impact*100:+.1f}%)"
                    )
                else:
                    reasons.append(
                        f"{matched} OUT (Impact: {raw_impact:+.1f}, bench player neutral → 0.0%)"
                    )
                continue
            
            # Weight by minutes per game: a 30+ MPG starter gets full weight,
            # a 12 MPG bench player only gets 40% of their impact
            prob_impact = raw_impact * 0.03 * minute_weight
            
            # Cap individual player impact at 12% max
            prob_impact = min(0.12, prob_impact)
            
            total_adj -= prob_impact  # Team loses their positive impact, so they get worse
            reasons.append(
                f"{matched} OUT ({mpg:.0f} MPG, Impact: {raw_impact:+.1f}, "
                f"Wt: {minute_weight:.0%} → {-prob_impact*100:+.1f}%)"
            )
        else:
            # FIX B: Players not found in on/off data (traded mid-season, etc.)
            # get a default -5% penalty instead of being silently ignored.
            # These are often key rotation players (VanVleet, Haliburton).
            default_penalty = 0.05
            total_adj -= default_penalty
            reasons.append(f"{player_name} OUT (not in on/off data, default → {-default_penalty*100:+.1f}%)")
    
    # FIX C: Raise cap from -15% to -25% so multi-injury games are properly penalized
    total_adj = max(-0.25, total_adj)
    
    reason_str = ", ".join(reasons) if reasons else "No injury adjustments"
    return total_adj, reason_str

def print_team_impact_report(team_name: str, season: str = '2025-26'):
    """Pretty-print the on/off impact for every player on a team, sorted by impact."""
    data = fetch_player_on_off(team_name, season)
    
    if not data:
        print(f"Could not fetch data for {team_name}")
        return
    
    # Sort by impact (most valuable player first)
    sorted_players = sorted(data.items(), key=lambda x: x[1]['impact'], reverse=True)
    
    print(f"\n{'='*80}")
    print(f"ON/OFF COURT IMPACT REPORT: {team_name.upper()}")
    print(f"{'='*80}")
    print(f"{'Player':<25} {'On NRtg':>8} {'Off NRtg':>9} {'Impact':>8} {'GP':>5} {'MIN':>7}")
    print("-"*80)
    
    for name, stats in sorted_players:
        impact_str = f"{stats['impact']:+.1f}"
        print(f"{name:<25} {stats['on_net_rtg']:>+8.1f} {stats['off_net_rtg']:>+9.1f} {impact_str:>8} {stats['gp']:>5} {stats['minutes']:>7.0f}")
    
    print("="*80)
    print("Positive Impact = Team is WORSE without this player (valuable)")
    print("Negative Impact = Team is BETTER without this player")
    print()

if __name__ == "__main__":
    # Demo: Show impact report for Mavericks
    print_team_impact_report("Mavericks")
    
    # Demo: Calculate what happens if AD is out
    adj, reason = calculate_injury_adjustment("Mavericks", ["Anthony Davis"])
    print(f"\nIf Anthony Davis is OUT:")
    print(f"  Adjustment: {adj*100:+.1f}%")
    print(f"  Reason: {reason}")
    
    time.sleep(1)
    
    # Demo: Show impact report for Grizzlies
    print_team_impact_report("Grizzlies")
