"""
Live NBA Data Module
Pulls real current rosters, today's games, and injury reports.
"""
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    commonteamroster, 
    leaguegamefinder,
    leaguedashteamstats,
    scoreboardv2
)
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

_nba_teams = teams.get_teams()


def _team_key(full_name: str) -> str:
    if full_name == 'Portland Trail Blazers':
        return 'Trail Blazers'
    return full_name.split()[-1]


def _weighted_recent_metric(values: list[float], fallback: float = 0.0) -> float:
    clean_values = [float(v) for v in values if v is not None]
    if not clean_values:
        return fallback
    weights = list(range(len(clean_values), 0, -1))
    denom = sum(weights)
    if denom <= 0:
        return fallback
    return sum(value * weight for value, weight in zip(clean_values, weights)) / denom


def fetch_team_schedule_context(season: str = '2025-26', as_of_date: str | None = None) -> dict:
    """
    Build per-team schedule and recent-form features from game logs.

    Features include rest days, back-to-back flags, 3-in-4-nights stress, and
    rolling 5/10 game win-rate and point-differential windows weighted toward
    the most recent games.
    """
    time.sleep(0.6)
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable='Regular Season',
        league_id_nullable='00'
    )
    df = finder.get_data_frames()[0]
    if df.empty:
        return {}

    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    target_dt = datetime.strptime(as_of_date, '%Y-%m-%d') if as_of_date else datetime.now()
    df = df[df['GAME_DATE'] < pd.Timestamp(target_dt)]

    context = {}
    for full_name, team_games in df.groupby('TEAM_NAME'):
        ordered = team_games.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
        if ordered.empty:
            continue

        recent_5 = ordered.head(5)
        recent_10 = ordered.head(10)
        last_game_date = ordered.iloc[0]['GAME_DATE']
        rest_days = max(0.0, (target_dt.date() - last_game_date.date()).days - 1)
        days_back = (pd.Timestamp(target_dt) - ordered['GAME_DATE']).dt.days

        def _avg_total(frame: pd.DataFrame) -> float:
            if frame.empty:
                return 225.0
            points_for = frame['PTS'].astype(float)
            opp_points = points_for - frame['PLUS_MINUS'].astype(float)
            return float((points_for + opp_points).mean())

        record = {
            'rest_days': rest_days,
            'back_to_back_flag': rest_days == 0,
            'is_3_in_4_nights': int((days_back <= 3).sum()) >= 3,
            'recent_5_win_pct': float((recent_5['WL'] == 'W').mean()) if not recent_5.empty else 0.5,
            'recent_10_win_pct': float((recent_10['WL'] == 'W').mean()) if not recent_10.empty else 0.5,
            'weighted_win_pct': _weighted_recent_metric((recent_10['WL'] == 'W').astype(float).tolist(), 0.5),
            'recent_5_point_diff': float(recent_5['PLUS_MINUS'].astype(float).mean()) if not recent_5.empty else 0.0,
            'recent_10_point_diff': float(recent_10['PLUS_MINUS'].astype(float).mean()) if not recent_10.empty else 0.0,
            'weighted_point_diff': _weighted_recent_metric(recent_10['PLUS_MINUS'].astype(float).tolist(), 0.0),
            'recent_5_total_points': _avg_total(recent_5),
            'recent_10_total_points': _avg_total(recent_10),
        }

        short_name = _team_key(full_name)
        context[short_name] = record
        context[full_name] = record

    return context

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

def fetch_all_team_stats(season: str = '2025-26', as_of_date: str | None = None) -> dict:
    """
    Fetch advanced stats for all teams.
    Returns dict keyed by team name.
    Blends season-long and last-10-game Net Rating (70/30).
    """
    time.sleep(0.6)
    stats = leaguedashteamstats.LeagueDashTeamStats(
        measure_type_detailed_defense='Advanced',
        season=season,
        per_mode_detailed='PerGame'
    )
    df = stats.get_data_frames()[0]
    
    # Also fetch last-10 stats for recent form blending
    time.sleep(0.6)
    try:
        last10_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            season=season,
            per_mode_detailed='PerGame',
            last_n_games=10
        )
        last10_df = last10_stats.get_data_frames()[0]
        last10_lookup = {}
        for _, r10 in last10_df.iterrows():
            last10_lookup[r10['TEAM_NAME']] = r10['NET_RATING']
        print("  ✅ Last-10-game stats fetched for recent form blending.")
    except Exception as e:
        print(f"  ⚠️ Could not fetch last-10 stats: {e}. Using season-only.")
        last10_lookup = {}
    
    schedule_context = {}
    try:
        schedule_context = fetch_team_schedule_context(season=season, as_of_date=as_of_date)
        print("  ✅ Schedule context fetched (rest/B2B/recent form windows).")
    except Exception as exc:
        print(f"  ⚠️ Could not fetch schedule context: {exc}. Using defaults.")

    result = {}
    for _, row in df.iterrows():
        full_name = row['TEAM_NAME']
        short_name = _team_key(full_name)
        context = schedule_context.get(full_name, schedule_context.get(short_name, {}))
        
        season_nrtg = row['NET_RATING']
        last10_nrtg = last10_lookup.get(full_name, season_nrtg)
        
        # FIX E: Blend 70% season + 30% last-10 for Net Rating
        blended_nrtg = (season_nrtg * 0.70) + (last10_nrtg * 0.30)
        
        result[short_name] = {
            'full_name': full_name,
            'net_rating': blended_nrtg,
            'season_net_rating': season_nrtg,
            'last10_net_rating': last10_nrtg,
            'off_rating': row['OFF_RATING'],
            'def_rating': row['DEF_RATING'],
            'ts_pct': row['TS_PCT'],
            'reb_pct': row['REB_PCT'],
            'pace': row['PACE'],
            'win_pct': row['W_PCT'],
            'recent_5_win_pct': context.get('recent_5_win_pct', row['W_PCT']),
            'recent_10_win_pct': context.get('recent_10_win_pct', row['W_PCT']),
            'weighted_win_pct': context.get('weighted_win_pct', row['W_PCT']),
            'recent_5_point_diff': context.get('recent_5_point_diff', blended_nrtg),
            'recent_10_point_diff': context.get('recent_10_point_diff', last10_nrtg),
            'weighted_point_diff': context.get('weighted_point_diff', blended_nrtg),
            'recent_5_total_points': context.get('recent_5_total_points', 225.0),
            'recent_10_total_points': context.get('recent_10_total_points', 225.0),
            'rest_days': context.get('rest_days', 1.0),
            'back_to_back_flag': context.get('back_to_back_flag', False),
            'is_3_in_4_nights': context.get('is_3_in_4_nights', False),
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
    seen_game_ids = set()
    for _, row in header.iterrows():
        game_id = str(row['GAME_ID'])
        if game_id in seen_game_ids:
            continue
        seen_game_ids.add(game_id)

        games.append({
            'game_id': game_id,
            'home_team_id': row['HOME_TEAM_ID'],
            'away_team_id': row['VISITOR_TEAM_ID'],
            'home_team': get_team_name(row['HOME_TEAM_ID']),
            'away_team': get_team_name(row['VISITOR_TEAM_ID']),
            'game_status': row.get('GAME_STATUS_TEXT', ''),
            'arena': row.get('ARENA_NAME', '')
        })
    
    return games


def fetch_espn_total_lines(date_str: str = None) -> dict:
    """
    Fetch game total lines from ESPN scoreboard for the date.
    Returns dict keyed by (away_team, home_team) -> total_line.
    Falls back to empty dict on any fetch/parse error.
    """
    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    try:
        dt = datetime.strptime(date_str, '%Y-%m-%d')
        yyyymmdd = dt.strftime('%Y%m%d')
    except ValueError:
        return {}

    url = (
        'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
        f'?dates={yyyymmdd}'
    )

    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode('utf-8'))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return {}

    lines = {}
    for event in payload.get('events', []):
        comps = event.get('competitions', [])
        if not comps:
            continue
        comp = comps[0]
        competitors = comp.get('competitors', [])
        if len(competitors) != 2:
            continue

        away_name = ''
        home_name = ''
        for c in competitors:
            team = c.get('team', {})
            nickname = str(team.get('name', '')).strip()
            if c.get('homeAway') == 'away':
                away_name = nickname
            elif c.get('homeAway') == 'home':
                home_name = nickname

        if not away_name or not home_name:
            continue

        total_line = None
        odds_list = comp.get('odds', [])
        if odds_list:
            total_line = odds_list[0].get('overUnder')

        if total_line is None:
            continue

        try:
            lines[(away_name, home_name)] = float(total_line)
        except (ValueError, TypeError):
            continue

    return lines

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
