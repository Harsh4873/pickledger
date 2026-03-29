"""
Live NBA Data Module
Pulls real current rosters, today's games, and injury reports.
"""
import math
import json
import time
from datetime import datetime
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

import pandas as pd
import requests
from bs4 import BeautifulSoup
from nba_api.stats.static import teams
from nba_api.stats.endpoints import (
    commonteamroster, 
    leaguegamefinder,
    leaguedashteamstats,
    scoreboardv2
)

_nba_teams = teams.get_teams()
_GARBAGE_TIME_MARGIN_CAP = 15.0


def _team_key(full_name: str) -> str:
    if full_name == 'Portland Trail Blazers':
        return 'Trail Blazers'
    return full_name.split()[-1]


def _weighted_recent_metric(values: list[float], fallback: float = 0.0) -> float:
    clean_values = _clean_metric_values(values)
    if not clean_values:
        return fallback
    weights = list(range(len(clean_values), 0, -1))
    denom = sum(weights)
    if denom <= 0:
        return fallback
    return sum(value * weight for value, weight in zip(clean_values, weights)) / denom


def _clean_metric_values(values: list[float]) -> list[float]:
    clean_values: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean_values.append(number)
    return clean_values


def cap_game_margin(raw_margin: float, cap: float = _GARBAGE_TIME_MARGIN_CAP) -> float:
    try:
        margin = float(raw_margin)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(margin):
        return 0.0
    try:
        margin_cap = abs(float(cap))
    except (TypeError, ValueError):
        margin_cap = _GARBAGE_TIME_MARGIN_CAP
    if not math.isfinite(margin_cap):
        margin_cap = _GARBAGE_TIME_MARGIN_CAP
    if margin_cap <= 0:
        return margin
    return max(-margin_cap, min(margin_cap, margin))


def _average_metric(values: list[float], fallback: float = 0.0) -> float:
    clean_values = _clean_metric_values(values)
    if not clean_values:
        return fallback
    return float(sum(clean_values) / len(clean_values))


def _build_upcoming_venue_lookup(upcoming_games: list[dict] | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    if not upcoming_games:
        return lookup

    for game in upcoming_games:
        home_team = str(game.get('home_team', '')).strip()
        away_team = str(game.get('away_team', '')).strip()
        if home_team:
            lookup[home_team] = 'home'
        if away_team:
            lookup[away_team] = 'away'
    return lookup


def _matchup_site(matchup: object) -> str | None:
    if not isinstance(matchup, str):
        return None
    matchup = matchup.upper()
    if '@' in matchup:
        return 'away'
    if 'VS.' in matchup or 'VS ' in matchup:
        return 'home'
    return None


def _calculate_current_road_trip_length(team_games: pd.DataFrame, today_site: str | None) -> int:
    if today_site != 'away':
        return 0

    streak = 1  # Include tonight's away game.
    if 'MATCHUP' not in team_games.columns:
        return streak

    for matchup in team_games['MATCHUP'].tolist():
        if _matchup_site(matchup) != 'away':
            break
        streak += 1
    return streak


def fetch_team_schedule_context(
    season: str = '2025-26',
    as_of_date: str | None = None,
    upcoming_games: list[dict] | None = None,
) -> dict:
    """
    Build per-team schedule and recent-form features from game logs.

    Features include rest days, B2B stress, advanced schedule density, road-trip
    drag, and rolling 5/10-game form windows weighted toward the most recent
    games.
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
    upcoming_venue_lookup = _build_upcoming_venue_lookup(upcoming_games)

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
        short_name = _team_key(full_name)
        today_site = upcoming_venue_lookup.get(short_name)
        has_game_today = today_site in {'home', 'away'}
        games_in_last_5_days = int((days_back <= 4).sum())
        games_in_last_7_days = int((days_back <= 6).sum())

        def _avg_total(frame: pd.DataFrame) -> float:
            if frame.empty:
                return 225.0
            points_for = frame['PTS'].astype(float)
            opp_points = points_for - frame['PLUS_MINUS'].astype(float)
            return float((points_for + opp_points).mean())

        recent_5_raw_margins = recent_5['PLUS_MINUS'].astype(float).tolist() if not recent_5.empty else []
        recent_10_raw_margins = recent_10['PLUS_MINUS'].astype(float).tolist() if not recent_10.empty else []
        recent_5_capped_margins = [cap_game_margin(margin) for margin in recent_5_raw_margins]
        recent_10_capped_margins = [cap_game_margin(margin) for margin in recent_10_raw_margins]

        raw_recent_5_point_diff = _average_metric(recent_5_raw_margins, 0.0)
        raw_recent_10_point_diff = _average_metric(recent_10_raw_margins, 0.0)
        raw_weighted_point_diff = _weighted_recent_metric(recent_10_raw_margins, 0.0)
        capped_recent_5_point_diff = _average_metric(recent_5_capped_margins, 0.0)
        capped_recent_10_point_diff = _average_metric(recent_10_capped_margins, 0.0)
        capped_weighted_point_diff = _weighted_recent_metric(recent_10_capped_margins, 0.0)

        record = {
            'rest_days': rest_days,
            'back_to_back_flag': rest_days == 0,
            'is_3_in_4_nights': int((days_back <= 3).sum()) >= 3,
            'is_4_in_5_nights': has_game_today and games_in_last_5_days >= 3,
            'is_5_in_7_nights': has_game_today and games_in_last_7_days >= 4,
            'current_road_trip_length': _calculate_current_road_trip_length(ordered, today_site),
            'recent_5_win_pct': float((recent_5['WL'] == 'W').mean()) if not recent_5.empty else 0.5,
            'recent_10_win_pct': float((recent_10['WL'] == 'W').mean()) if not recent_10.empty else 0.5,
            'weighted_win_pct': _weighted_recent_metric((recent_10['WL'] == 'W').astype(float).tolist(), 0.5),
            'raw_recent_5_point_diff': raw_recent_5_point_diff,
            'raw_recent_10_point_diff': raw_recent_10_point_diff,
            'raw_weighted_point_diff': raw_weighted_point_diff,
            'capped_recent_5_point_diff': capped_recent_5_point_diff,
            'capped_recent_10_point_diff': capped_recent_10_point_diff,
            'capped_weighted_point_diff': capped_weighted_point_diff,
            'garbage_time_margin_cap': _GARBAGE_TIME_MARGIN_CAP,
            # Preserve the legacy defaults as raw values. NBANEW opts into the
            # capped fields explicitly so NBAOLD remains isolated.
            'recent_5_point_diff': raw_recent_5_point_diff,
            'recent_10_point_diff': raw_recent_10_point_diff,
            'weighted_point_diff': raw_weighted_point_diff,
            'recent_5_total_points': _avg_total(recent_5),
            'recent_10_total_points': _avg_total(recent_10),
        }

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

def fetch_all_team_stats(
    season: str = '2025-26',
    as_of_date: str | None = None,
    upcoming_games: list[dict] | None = None,
) -> dict:
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

    four_factor_lookup = {}
    try:
        time.sleep(0.6)
        four_factor_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Four Factors',
            season=season,
            per_mode_detailed='PerGame'
        )
        four_factor_df = four_factor_stats.get_data_frames()[0]
        for _, ff_row in four_factor_df.iterrows():
            full_name = ff_row['TEAM_NAME']
            short_name = _team_key(full_name)
            payload = {
                'opp_tov_pct': ff_row.get('OPP_TOV_PCT', 0.135),
                'opp_oreb_pct': ff_row.get('OPP_OREB_PCT', 0.28),
            }
            four_factor_lookup[short_name] = payload
            four_factor_lookup[full_name] = payload
        print("  ✅ Four Factors fetched for tempo-control turnover/rebound context.")
    except Exception as exc:
        print(f"  ⚠️ Could not fetch Four Factors stats: {exc}. Using defaults.")
    
    schedule_context = {}
    try:
        schedule_context = fetch_team_schedule_context(
            season=season,
            as_of_date=as_of_date,
            upcoming_games=upcoming_games,
        )
        print("  ✅ Schedule context fetched (rest/B2B/recent form + advanced fatigue windows).")
    except Exception as exc:
        print(f"  ⚠️ Could not fetch schedule context: {exc}. Using defaults.")

    result = {}
    for _, row in df.iterrows():
        full_name = row['TEAM_NAME']
        short_name = _team_key(full_name)
        context = schedule_context.get(full_name, schedule_context.get(short_name, {}))
        four_factor = four_factor_lookup.get(full_name, four_factor_lookup.get(short_name, {}))
        
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
            'efg_pct': row.get('EFG_PCT', row['TS_PCT']),
            'ts_pct': row['TS_PCT'],
            'tov_pct': row.get('TM_TOV_PCT', row.get('TOV_PCT', 0.13)),
            'reb_pct': row['REB_PCT'],
            'dreb_pct': row.get('DREB_PCT', 1.0 - float(four_factor.get('opp_oreb_pct', 0.28))),
            'opp_tov_pct': float(four_factor.get('opp_tov_pct', 0.135)),
            'opp_oreb_pct': float(four_factor.get('opp_oreb_pct', 0.28)),
            'pace': row['PACE'],
            'win_pct': row['W_PCT'],
            'recent_5_win_pct': context.get('recent_5_win_pct', row['W_PCT']),
            'recent_10_win_pct': context.get('recent_10_win_pct', row['W_PCT']),
            'weighted_win_pct': context.get('weighted_win_pct', row['W_PCT']),
            'raw_recent_5_point_diff': context.get('raw_recent_5_point_diff', blended_nrtg),
            'raw_recent_10_point_diff': context.get('raw_recent_10_point_diff', last10_nrtg),
            'raw_weighted_point_diff': context.get('raw_weighted_point_diff', blended_nrtg),
            'capped_recent_5_point_diff': context.get('capped_recent_5_point_diff', context.get('raw_recent_5_point_diff', blended_nrtg)),
            'capped_recent_10_point_diff': context.get('capped_recent_10_point_diff', context.get('raw_recent_10_point_diff', last10_nrtg)),
            'capped_weighted_point_diff': context.get('capped_weighted_point_diff', context.get('raw_weighted_point_diff', blended_nrtg)),
            'garbage_time_margin_cap': context.get('garbage_time_margin_cap', _GARBAGE_TIME_MARGIN_CAP),
            'recent_5_point_diff': context.get('recent_5_point_diff', blended_nrtg),
            'recent_10_point_diff': context.get('recent_10_point_diff', last10_nrtg),
            'weighted_point_diff': context.get('weighted_point_diff', blended_nrtg),
            'recent_5_total_points': context.get('recent_5_total_points', 225.0),
            'recent_10_total_points': context.get('recent_10_total_points', 225.0),
            'rest_days': context.get('rest_days', 1.0),
            'back_to_back_flag': context.get('back_to_back_flag', False),
            'is_3_in_4_nights': context.get('is_3_in_4_nights', False),
            'is_4_in_5_nights': context.get('is_4_in_5_nights', False),
            'is_5_in_7_nights': context.get('is_5_in_7_nights', False),
            'current_road_trip_length': context.get('current_road_trip_length', 0),
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
