"""
Live NBA Data Module
Pulls real current rosters, today's games, and injury reports.
"""
import math
import json
import os
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
REQUEST_PAUSE_SECONDS = 0.1 if os.environ.get("RENDER", "").strip().lower() == "true" else 0.6
ESPN_NBA_TEAMS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
ESPN_NBA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
ESPN_USER_AGENT = "Mozilla/5.0 PickLedgerPro NBA fallback/1.0"


def _pause() -> None:
    if REQUEST_PAUSE_SECONDS > 0:
        time.sleep(REQUEST_PAUSE_SECONDS)


def _fetch_espn_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": ESPN_USER_AGENT})
    with urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normalize_team_lookup(value: object) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _espn_team_catalog() -> list[dict]:
    payload = _fetch_espn_json(ESPN_NBA_TEAMS_URL)
    sports = payload.get("sports") if isinstance(payload.get("sports"), list) else []
    leagues = sports[0].get("leagues") if sports and isinstance(sports[0], dict) else []
    teams_payload = leagues[0].get("teams") if leagues and isinstance(leagues[0], dict) else []
    return [
        item.get("team")
        for item in teams_payload or []
        if isinstance(item, dict) and isinstance(item.get("team"), dict)
    ]


def _find_espn_team(team_name: str, catalog: list[dict]) -> dict | None:
    target = _normalize_team_lookup(team_name)
    for team in catalog:
        candidates = (
            team.get("name"),
            team.get("displayName"),
            team.get("shortDisplayName"),
            team.get("abbreviation"),
            team.get("location"),
        )
        if any(_normalize_team_lookup(value) == target for value in candidates):
            return team
    return None


def _espn_stat_lookup(payload: dict) -> dict[str, float]:
    results = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    stats_root = results.get("stats") if isinstance(results.get("stats"), dict) else {}
    categories = stats_root.get("categories") if isinstance(stats_root.get("categories"), list) else []
    values: dict[str, float] = {}
    for category in categories:
        stats = category.get("stats") if isinstance(category, dict) and isinstance(category.get("stats"), list) else []
        for stat in stats:
            if not isinstance(stat, dict):
                continue
            name = str(stat.get("name") or "").strip()
            if not name:
                continue
            try:
                values[name] = float(stat.get("value"))
            except (TypeError, ValueError):
                continue
    return values


def _espn_total_record(team_payload: dict) -> dict[str, float]:
    team = team_payload.get("team") if isinstance(team_payload.get("team"), dict) else {}
    record = team.get("record") if isinstance(team.get("record"), dict) else {}
    items = record.get("items") if isinstance(record.get("items"), list) else []
    total = next((item for item in items if isinstance(item, dict) and item.get("type") == "total"), {})
    stats = total.get("stats") if isinstance(total, dict) and isinstance(total.get("stats"), list) else []
    values: dict[str, float] = {}
    for stat in stats:
        if not isinstance(stat, dict):
            continue
        try:
            values[str(stat.get("name") or "")] = float(stat.get("value"))
        except (TypeError, ValueError):
            continue
    return values


def _clamp_metric(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _espn_fallback_team_stats(team_name: str, team: dict) -> dict:
    slug = str(team.get("abbreviation") or team.get("id") or "").strip().lower()
    stats = _espn_stat_lookup(_fetch_espn_json(f"{ESPN_NBA_TEAMS_URL}/{slug}/statistics"))
    record = _espn_total_record(_fetch_espn_json(f"{ESPN_NBA_TEAMS_URL}/{slug}"))

    points_for = float(record.get("avgPointsFor", stats.get("avgPoints", 112.0)))
    points_against = float(record.get("avgPointsAgainst", points_for))
    point_diff = float(record.get("differential", points_for - points_against))
    win_pct = _clamp_metric(float(record.get("winPercent", 0.5)), 0.0, 1.0)
    fga = max(float(stats.get("avgFieldGoalsAttempted", 88.0)), 1.0)
    fgm = float(stats.get("avgFieldGoalsMade", fga * 0.47))
    three_made = float(stats.get("avgThreePointFieldGoalsMade", 12.0))
    fta = float(stats.get("avgFreeThrowsAttempted", 22.0))
    turnovers = float(stats.get("avgTurnovers", 13.0))
    rebounds = float(stats.get("avgRebounds", 43.0))
    defensive_rebounds = float(stats.get("avgDefensiveRebounds", rebounds * 0.72))
    offensive_rebounds = float(stats.get("avgOffensiveRebounds", rebounds - defensive_rebounds))
    shooting_denom = max(2.0 * (fga + 0.44 * fta), 1.0)
    ts_pct = _clamp_metric(points_for / shooting_denom, 0.45, 0.70)
    efg_pct = _clamp_metric((fgm + (0.5 * three_made)) / fga, 0.40, 0.70)
    tov_pct = _clamp_metric(turnovers / max(fga + (0.44 * fta) + turnovers, 1.0), 0.08, 0.22)
    reb_pct = _clamp_metric(0.50 + ((rebounds - 43.0) / 200.0), 0.45, 0.55)
    dreb_pct = _clamp_metric(defensive_rebounds / max(rebounds, 1.0), 0.60, 0.85)
    opp_oreb_pct = _clamp_metric(1.0 - dreb_pct, 0.15, 0.40)

    return {
        "full_name": str(team.get("displayName") or team_name),
        "net_rating": point_diff,
        "season_net_rating": point_diff,
        "last10_net_rating": point_diff,
        "off_rating": points_for,
        "def_rating": points_against,
        "efg_pct": efg_pct,
        "ts_pct": ts_pct,
        "tov_pct": tov_pct,
        "reb_pct": reb_pct,
        "dreb_pct": dreb_pct,
        "opp_tov_pct": 0.135,
        "opp_oreb_pct": opp_oreb_pct,
        "pace": 99.0,
        "win_pct": win_pct,
        "recent_5_win_pct": win_pct,
        "recent_10_win_pct": win_pct,
        "weighted_win_pct": win_pct,
        "raw_recent_5_point_diff": point_diff,
        "raw_recent_10_point_diff": point_diff,
        "raw_weighted_point_diff": point_diff,
        "capped_recent_5_point_diff": cap_game_margin(point_diff),
        "capped_recent_10_point_diff": cap_game_margin(point_diff),
        "capped_weighted_point_diff": cap_game_margin(point_diff),
        "garbage_time_margin_cap": _GARBAGE_TIME_MARGIN_CAP,
        "recent_5_point_diff": point_diff,
        "recent_10_point_diff": point_diff,
        "weighted_point_diff": point_diff,
        "recent_5_total_points": points_for + points_against,
        "recent_10_total_points": points_for + points_against,
        "points_per_game": points_for,
        "opp_points_per_game": points_against,
        "rest_days": 1.0,
        "back_to_back_flag": False,
        "is_b2b_second_leg": False,
        "is_3_in_4_nights": False,
        "is_4_in_5_nights": False,
        "is_5_in_7_nights": False,
        "current_road_trip_length": 0,
        "stats_source": "ESPN team statistics fallback",
    }


def fetch_espn_team_stats_fallback(upcoming_games: list[dict] | None = None) -> dict:
    team_names: list[str] = []
    for game in upcoming_games or []:
        for key in ("away_team", "home_team"):
            name = str(game.get(key) or "").strip()
            if name and name not in team_names:
                team_names.append(name)
    if not team_names:
        return {}

    catalog = _espn_team_catalog()
    result: dict[str, dict] = {}
    for team_name in team_names:
        team = _find_espn_team(team_name, catalog)
        if not team:
            continue
        payload = _espn_fallback_team_stats(team_name, team)
        result[team_name] = payload
        result[str(team.get("displayName") or team_name)] = payload
        result[_team_key(str(team.get("displayName") or team_name))] = payload
    return result


def fetch_espn_roster_fallback(team_name: str) -> list:
    catalog = _espn_team_catalog()
    team = _find_espn_team(team_name, catalog)
    if not team:
        return []
    slug = str(team.get("abbreviation") or team.get("id") or "").strip().lower()
    payload = _fetch_espn_json(f"{ESPN_NBA_TEAMS_URL}/{slug}/roster")
    athletes = payload.get("athletes") if isinstance(payload.get("athletes"), list) else []
    return [
        {
            "name": str(athlete.get("displayName") or athlete.get("fullName") or "").strip(),
            "num": str(athlete.get("jersey") or "").strip(),
            "position": str((athlete.get("position") or {}).get("abbreviation") or "").strip(),
            "age": athlete.get("age") or 0,
            "player_id": athlete.get("id") or 0,
            "source": "ESPN roster fallback",
        }
        for athlete in athletes
        if isinstance(athlete, dict) and (athlete.get("displayName") or athlete.get("fullName"))
    ]


def fetch_espn_scoreboard_games(date_str: str) -> list[dict]:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return []

    payload = _fetch_espn_json(f"{ESPN_NBA_SCOREBOARD_URL}?dates={dt.strftime('%Y%m%d')}")
    games: list[dict] = []
    for event in payload.get("events", []):
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions") if isinstance(event.get("competitions"), list) else []
        competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
        competitors = competition.get("competitors") if isinstance(competition.get("competitors"), list) else []
        if len(competitors) != 2:
            continue

        teams_by_side: dict[str, dict] = {}
        for competitor in competitors:
            if not isinstance(competitor, dict):
                continue
            side = str(competitor.get("homeAway") or "").strip().lower()
            team = competitor.get("team") if isinstance(competitor.get("team"), dict) else {}
            if side in {"home", "away"}:
                teams_by_side[side] = team

        home_team = teams_by_side.get("home") or {}
        away_team = teams_by_side.get("away") or {}
        home_name = str(home_team.get("name") or home_team.get("shortDisplayName") or "").strip()
        away_name = str(away_team.get("name") or away_team.get("shortDisplayName") or "").strip()
        if not home_name or not away_name:
            continue

        venue = competition.get("venue") if isinstance(competition.get("venue"), dict) else {}
        status = competition.get("status") if isinstance(competition.get("status"), dict) else {}
        status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
        games.append({
            "game_id": str(event.get("id") or competition.get("id") or ""),
            "home_team_id": home_team.get("id") or 0,
            "away_team_id": away_team.get("id") or 0,
            "home_team": home_name,
            "away_team": away_name,
            "game_status": str(status_type.get("shortDetail") or status_type.get("description") or ""),
            "arena": str(venue.get("fullName") or venue.get("name") or ""),
            "schedule_source": "ESPN scoreboard fallback",
        })
    return games


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
    _pause()
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


def _is_valid_scoreboard_team_name(value: object) -> bool:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return False
    return not text.isdigit()

def fetch_roster(team_name: str, season: str = '2025-26') -> list:
    """
    Fetch the current roster for a team.
    Returns list of dicts: [{'name': 'Cooper Flagg', 'num': '32', 'position': 'F', 'age': 19}, ...]
    """
    team_id = get_team_id(team_name)
    if not team_id:
        print(f"WARNING: Could not find team '{team_name}'")
        return []
    
    _pause()
    try:
        roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
        df = roster.get_data_frames()[0]
    except Exception as exc:
        fallback = fetch_espn_roster_fallback(team_name)
        if fallback:
            print(f"WARNING: NBA API roster failed for {team_name} ({exc}); using ESPN roster fallback.")
            return fallback
        raise
    
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
    _pause()
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            season=season,
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]
    except Exception as exc:
        fallback = fetch_espn_team_stats_fallback(upcoming_games)
        if fallback:
            print(f"WARNING: NBA API team stats failed ({exc}); using ESPN team statistics fallback.")
            return fallback
        raise
    
    # Also fetch last-10 stats for recent form blending
    _pause()
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
        _pause()
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
            'stats_source': 'NBA API',
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
    
    _pause()
    try:
        sb = scoreboardv2.ScoreboardV2(game_date=formatted)
        dfs = sb.get_data_frames()
        header = dfs[0]
    except Exception as exc:
        fallback_games = fetch_espn_scoreboard_games(date_str)
        if fallback_games:
            print(f"WARNING: NBA API scoreboard failed ({exc}); using ESPN scoreboard fallback.")
            return fallback_games
        raise
    
    games = []
    seen_game_ids = set()
    invalid_rows_seen = False
    for _, row in header.iterrows():
        game_id = str(row['GAME_ID'])
        if game_id in seen_game_ids:
            continue
        seen_game_ids.add(game_id)
        home_team = get_team_name(row['HOME_TEAM_ID'])
        away_team = get_team_name(row['VISITOR_TEAM_ID'])
        if not (_is_valid_scoreboard_team_name(home_team) and _is_valid_scoreboard_team_name(away_team)):
            invalid_rows_seen = True
            continue

        games.append({
            'game_id': game_id,
            'home_team_id': row['HOME_TEAM_ID'],
            'away_team_id': row['VISITOR_TEAM_ID'],
            'home_team': home_team,
            'away_team': away_team,
            'game_status': row.get('GAME_STATUS_TEXT', ''),
            'arena': row.get('ARENA_NAME', '')
        })

    if invalid_rows_seen:
        fallback_games = fetch_espn_scoreboard_games(date_str)
        if fallback_games:
            print("WARNING: NBA API scoreboard had incomplete team rows; using ESPN scoreboard fallback.")
            return fallback_games
    
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
