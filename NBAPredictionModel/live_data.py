"""
Live NBA Data Module
Pulls real current rosters, today's games, and injury reports.
"""
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    commonteamroster, 
    leaguedashteamstats,
    scoreboardv2
)
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

_nba_teams = teams.get_teams()

def get_team_id(team_name: str) -> int:
    """Find a team ID by partial name match."""
    for t in _nba_teams:
        if team_name.lower() in t['full_name'].lower() or team_name.lower() in t['nickname'].lower():
            return t['id']
    return None

def get_team_name(team_id: int) -> str:
    for t in _nba_teams:
        if t['id'] == team_id:
            return t['nickname']
    return str(team_id)

def fetch_roster(team_name: str, season: str = '2025-26') -> list:
    """
    Fetch the current roster for a team.
    Returns list of dicts: [{'name': 'Cooper Flagg', 'num': '32', 'position': 'F', 'age': 19}, ...]
    """
    team_id = get_team_id(team_name)
    if not team_id:
        print(f"WARNING: Could not find team '{team_name}'")
        return []
    
    time.sleep(0.6)
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    df = roster.get_data_frames()[0]
    
    players = []
    for _, row in df.iterrows():
        players.append({
            'name': row['PLAYER'],
            'num': row.get('NUM', ''),
            'position': row.get('POSITION', ''),
            'age': row.get('AGE', 0),
            'player_id': row.get('PLAYER_ID', 0)
        })
    return players

def fetch_all_team_stats(season: str = '2025-26') -> dict:
    """
    Fetch advanced stats for all teams.
    Returns dict keyed by team name.
    """
    time.sleep(0.6)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense='Advanced',
        season=season,
        per_mode_detailed='PerGame'
    )
    df = stats.get_data_frames()[0]
    
    result = {}
    for _, row in df.iterrows():
        # Normalize name (e.g. "Dallas Mavericks" -> "Mavericks")
        full_name = row['TEAM_NAME']
        if full_name == 'Portland Trail Blazers':
            short_name = 'Trail Blazers'
        else:
            short_name = full_name.split()[-1]  # Last word
        
        result[short_name] = {
            'full_name': full_name,
            'net_rating': row['NET_RATING'],
            'off_rating': row['OFF_RATING'],
            'def_rating': row['DEF_RATING'],
            'ts_pct': row['TS_PCT'],
            'reb_pct': row['REB_PCT'],
            'pace': row['PACE'],
            'win_pct': row['W_PCT']
        }
        # Also store by full name
        result[full_name] = result[short_name]
    
    return result

def fetch_todays_games(date_str: str = None) -> list:
    """
    Fetch today's NBA games from the scoreboard.
    Returns list of dicts with home/away team info.
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')
    
    # Format for scoreboard: MM/DD/YYYY
    parts = date_str.split('-')
    formatted = f"{parts[1]}/{parts[2]}/{parts[0]}"
    
    time.sleep(0.6)
    sb = scoreboardv2.ScoreboardV2(game_date=formatted)
    dfs = sb.get_data_frames()
    
    # GameHeader is the first dataframe
    header = dfs[0]
    
    games = []
    for _, row in header.iterrows():
        games.append({
            'game_id': row['GAME_ID'],
            'home_team_id': row['HOME_TEAM_ID'],
            'away_team_id': row['VISITOR_TEAM_ID'],
            'home_team': get_team_name(row['HOME_TEAM_ID']),
            'away_team': get_team_name(row['VISITOR_TEAM_ID']),
            'game_status': row.get('GAME_STATUS_TEXT', ''),
            'arena': row.get('ARENA_NAME', '')
        })
    
    return games

def scrape_injury_report() -> dict:
    """
    Scrape the NBA's official injury report from Rotowire or CBS Sports.
    Returns dict keyed by team name -> list of injured players.
    """
    url = "https://www.cbssports.com/nba/injuries/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        injuries = {}
        
        # CBS Sports injury page structure
        team_sections = soup.find_all('div', class_='TeamLogoNameLockup-pointed')
        tables = soup.find_all('table', class_='TableBase-table')
        
        if not team_sections and not tables:
            # Try alternative parsing
            # Look for table rows with injury data
            all_tables = soup.find_all('table')
            for table in all_tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        # Try to extract player name, status, injury
                        pass
        
        # If CBS doesn't work well, try a simpler approach with Rotowire
        if not injuries:
            injuries = _scrape_rotowire_injuries()
        
        return injuries
        
    except Exception as e:
        print(f"WARNING: Could not scrape injury report: {e}")
        return {}

def _scrape_rotowire_injuries() -> dict:
    """Fallback: scrape Rotowire NBA injury report."""
    url = "https://www.rotowire.com/basketball/injury-report.php"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        injuries = {}
        
        # Rotowire uses table rows for each player
        rows = soup.select('table.injury-report tr, div.injury-report__team')
        
        current_team = None
        for el in soup.find_all(['div', 'tr']):
            # Check if this is a team header
            team_header = el.find(class_=lambda x: x and 'team' in str(x).lower())
            if team_header:
                team_text = team_header.get_text(strip=True)
                if team_text and len(team_text) > 2:
                    current_team = team_text
                    if current_team not in injuries:
                        injuries[current_team] = []
            
            # Check for player injury rows
            cells = el.find_all('td')
            if cells and len(cells) >= 2 and current_team:
                player_name = cells[0].get_text(strip=True)
                status = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                injury_detail = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                
                if player_name and any(s in status.lower() for s in ['out', 'doubtful', 'questionable', 'probable', 'day-to-day']):
                    injuries[current_team].append({
                        'name': player_name,
                        'status': status,
                        'injury': injury_detail
                    })
        
        return injuries
        
    except Exception as e:
        print(f"WARNING: Rotowire scrape failed: {e}")
        return {}

def print_roster(team_name: str):
    """Pretty print a team's roster."""
    players = fetch_roster(team_name)
    print(f"\n{'='*60}")
    print(f"CURRENT ROSTER: {team_name.upper()}")
    print(f"{'='*60}")
    print(f"{'Player':<25} {'#':>4} {'Pos':<6} {'Age':>4}")
    print("-"*60)
    for p in players:
        print(f"{p['name']:<25} {str(p['num']):>4} {p['position']:<6} {p['age']:>4.0f}")
    print("="*60)

def print_todays_games():
    """Pretty print today's NBA games."""
    games = fetch_todays_games()
    print(f"\n{'='*60}")
    print(f"TODAY'S NBA GAMES ({datetime.now().strftime('%Y-%m-%d')})")
    print(f"{'='*60}")
    for g in games:
        print(f"  {g['away_team']} @ {g['home_team']} — {g['arena']} — {g['game_status']}")
    print(f"{'='*60}")
    return games

if __name__ == "__main__":
    # Show today's games
    print_todays_games()
    
    time.sleep(1)
    
    # Show rosters for today's matchup teams
    print_roster("Mavericks")
    time.sleep(1)
    print_roster("Grizzlies")
    time.sleep(1)
    print_roster("Lakers")
    
    time.sleep(1)
    
    # Try to scrape injury report
    print("\n\nATTEMPTING INJURY REPORT SCRAPE...")
    injuries = scrape_injury_report()
    if injuries:
        for team, players in injuries.items():
            if players:
                print(f"\n{team}:")
                for p in players:
                    print(f"  - {p['name']}: {p['status']} ({p.get('injury', 'N/A')})")
    else:
        print("No injury data scraped (scraper may need adjustment for current site layout)")
