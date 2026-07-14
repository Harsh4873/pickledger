from __future__ import annotations

import json
import math
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
API_BASE = "https://statsapi.mlb.com/api/v1"
LIVE_FEED_BASE = "https://statsapi.mlb.com/api/v1.1"
HTTP_TIMEOUT_SECONDS = 10
SCHEDULE_TTL_SECONDS = 2 * 60 * 60
STATS_TTL_SECONDS = 24 * 60 * 60

DEFAULT_BATTER = {"avg": 0.240, "obp": 0.320, "slg": 0.390}
DEFAULT_PITCHER = {
    "id": None,
    "name": "TBD",
    "era": 4.20,
    "whip": 1.30,
    "opponent_obp": 0.320,
    "opponent_slg": 0.410,
}

EARTH_RADIUS_MILES = 3958.8

# Approximate MLB venue geography for travel fatigue. Timezone offsets are
# standard-time relative values; only the difference matters for east/west
# travel direction, so DST does not change the model signal.
VENUE_TRAVEL_CONTEXT: dict[str, dict[str, float]] = {
    "angel stadium": {"lat": 33.8003, "lon": -117.8827, "tz": -8},
    "american family field": {"lat": 43.0280, "lon": -87.9712, "tz": -6},
    "busch stadium": {"lat": 38.6226, "lon": -90.1928, "tz": -6},
    "camden yards": {"lat": 39.2840, "lon": -76.6217, "tz": -5},
    "chase field": {"lat": 33.4455, "lon": -112.0667, "tz": -7},
    "citi field": {"lat": 40.7571, "lon": -73.8458, "tz": -5},
    "citizens bank park": {"lat": 39.9061, "lon": -75.1665, "tz": -5},
    "coors field": {"lat": 39.7559, "lon": -104.9942, "tz": -7},
    "comerica park": {"lat": 42.3390, "lon": -83.0485, "tz": -5},
    "daikin park": {"lat": 29.7573, "lon": -95.3555, "tz": -6},
    "dodger stadium": {"lat": 34.0739, "lon": -118.2400, "tz": -8},
    "fenway park": {"lat": 42.3467, "lon": -71.0972, "tz": -5},
    "globe life field": {"lat": 32.7473, "lon": -97.0842, "tz": -6},
    "great american ball park": {"lat": 39.0974, "lon": -84.5066, "tz": -5},
    "guaranteed rate field": {"lat": 41.8300, "lon": -87.6339, "tz": -6},
    "kauffman stadium": {"lat": 39.0517, "lon": -94.4803, "tz": -6},
    "loanDepot park": {"lat": 25.7781, "lon": -80.2197, "tz": -5},
    "loandepot park": {"lat": 25.7781, "lon": -80.2197, "tz": -5},
    "minute maid park": {"lat": 29.7573, "lon": -95.3555, "tz": -6},
    "nationals park": {"lat": 38.8730, "lon": -77.0074, "tz": -5},
    "oracle park": {"lat": 37.7786, "lon": -122.3893, "tz": -8},
    "petco park": {"lat": 32.7073, "lon": -117.1566, "tz": -8},
    "pnc park": {"lat": 40.4469, "lon": -80.0057, "tz": -5},
    "progressive field": {"lat": 41.4962, "lon": -81.6852, "tz": -5},
    "rate field": {"lat": 41.8300, "lon": -87.6339, "tz": -6},
    "rogers centre": {"lat": 43.6414, "lon": -79.3894, "tz": -5},
    "sutter health park": {"lat": 38.5803, "lon": -121.5139, "tz": -8},
    "target field": {"lat": 44.9817, "lon": -93.2776, "tz": -6},
    "t-mobile park": {"lat": 47.5914, "lon": -122.3325, "tz": -8},
    "tropicana field": {"lat": 27.7682, "lon": -82.6534, "tz": -5},
    "truist park": {"lat": 33.8907, "lon": -84.4677, "tz": -5},
    "wrigley field": {"lat": 41.9484, "lon": -87.6553, "tz": -6},
    "yankee stadium": {"lat": 40.8296, "lon": -73.9262, "tz": -5},
}


def log_warning(message: str) -> None:
    print(f"[MLBInning] Warning: {message}")


def cache_get(key: str, ttl_seconds: int = STATS_TTL_SECONDS) -> Any | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_safe_cache_key(key)}.json"
    if not path.exists():
        return None
    try:
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds > ttl_seconds:
            return None
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        log_warning(f"ignoring bad cache file {path.name}: {exc}")
        return None


def cache_set(key: str, data: Any) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_safe_cache_key(key)}.json"
    tmp_path = path.with_suffix(f".{path.suffix}.{time.time_ns()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
    tmp_path.replace(path)


def api_get_json(url: str, params: dict[str, Any] | None = None, cache_key: str | None = None, ttl_seconds: int = STATS_TTL_SECONDS) -> dict[str, Any]:
    if cache_key:
        cached = cache_get(cache_key, ttl_seconds)
        if cached is not None:
            return cached

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            if cache_key:
                cache_set(cache_key, payload)
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt == 0:
                time.sleep(0.75)

    raise RuntimeError(f"Stats API request failed for {url}: {last_exc}")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, "", "-.--", ".---"):
            return default
        parsed = float(value)
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_date(target_date: str | date | None = None) -> str:
    if target_date is None:
        return date.today().strftime("%Y-%m-%d")
    if isinstance(target_date, date):
        return target_date.strftime("%Y-%m-%d")
    raw = str(target_date).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {target_date}. Use YYYY-MM-DD or MM/DD/YYYY.")


def fetch_todays_games(target_date: str | date | None = None) -> list[dict[str, Any]]:
    game_date = normalize_date(target_date)
    try:
        schedule = _fetch_schedule_for_date(game_date)
    except RuntimeError as exc:
        log_warning(str(exc))
        return []

    # First pass: collect every team_id that needs a bullpen workload, then
    # batch-fetch them all in parallel. Cuts ~30s off the first daily run
    # for a typical 15-game slate (was 2 sequential calls per game).
    all_team_ids: list[int] = []
    schedule_games_list = list(_schedule_games(schedule))
    for schedule_game in schedule_games_list:
        if _should_skip_status(_status_state(schedule_game)):
            continue
        teams = schedule_game.get("teams") or {}
        for side in ("home", "away"):
            tid = safe_int(((teams.get(side) or {}).get("team") or {}).get("id"))
            if tid:
                all_team_ids.append(tid)

    try:
        from mlb_inning_bullpen import fetch_bullpen_workloads_parallel
    except ImportError:
        from .mlb_inning_bullpen import fetch_bullpen_workloads_parallel
    bullpen_workloads = fetch_bullpen_workloads_parallel(all_team_ids, game_date)

    games: list[dict[str, Any]] = []
    for schedule_game in schedule_games_list:
        game_id = safe_int(schedule_game.get("gamePk"))
        status = _status_state(schedule_game)
        if not game_id:
            continue
        if _should_skip_status(status):
            log_warning(f"skipped_in_progress {game_id}: {status or 'unknown status'}")
            continue

        teams = schedule_game.get("teams") or {}
        home_team_info = ((teams.get("home") or {}).get("team") or {})
        away_team_info = ((teams.get("away") or {}).get("team") or {})
        home_team_id = safe_int(home_team_info.get("id"))
        away_team_id = safe_int(away_team_info.get("id"))
        home_team_name = home_team_info.get("name") or "Home Team"
        away_team_name = away_team_info.get("name") or "Away Team"

        feed = _fetch_game_feed(game_id)
        if not feed:
            log_warning(f"skipping game {game_id}: live feed unavailable")
            continue

        game_data = feed.get("gameData") or {}
        venue_raw = game_data.get("venue") or schedule_game.get("venue") or {}
        venue_id = safe_int(venue_raw.get("id"))
        venue_name = str(venue_raw.get("name") or "")
        weather_raw = game_data.get("weather") or {}
        probable_pitchers = game_data.get("probablePitchers") or {}
        home_pitcher_raw = probable_pitchers.get("home") or ((teams.get("home") or {}).get("probablePitcher") or {})
        away_pitcher_raw = probable_pitchers.get("away") or ((teams.get("away") or {}).get("probablePitcher") or {})

        home_pitcher = _build_pitcher(home_pitcher_raw, game_date)
        away_pitcher = _build_pitcher(away_pitcher_raw, game_date)
        if not home_pitcher.get("id"):
            home_pitcher = _recent_starter(home_team_id, game_date) or home_pitcher
        if not away_pitcher.get("id"):
            away_pitcher = _recent_starter(away_team_id, game_date) or away_pitcher

        # Attach the team's bullpen-fatigue workload from the parallel batch.
        home_pitcher = {
            **home_pitcher,
            "team_bullpen": {
                **((home_pitcher or {}).get("team_bullpen") or {}),
                **(bullpen_workloads.get(home_team_id) or {}),
            },
        }
        away_pitcher = {
            **away_pitcher,
            "team_bullpen": {
                **((away_pitcher or {}).get("team_bullpen") or {}),
                **(bullpen_workloads.get(away_team_id) or {}),
            },
        }

        home_lineup = _lineup_from_feed(feed, "home")
        away_lineup = _lineup_from_feed(feed, "away")
        if len(home_lineup) < 9:
            home_lineup = _last_known_lineup(home_team_id, game_date) or _active_roster_lineup(home_team_id, game_date)
        if len(away_lineup) < 9:
            away_lineup = _last_known_lineup(away_team_id, game_date) or _active_roster_lineup(away_team_id, game_date)

        home_travel = _team_travel_context(home_team_id, game_date, venue_id, venue_name)
        away_travel = _team_travel_context(away_team_id, game_date, venue_id, venue_name)

        games.append(
            {
                "game_id": str(game_id),
                "game_date": game_date,
                "game_start_time": str(schedule_game.get("gameDate") or ""),
                "game_order": len(games),
                "status": status,
                "home_team": home_team_name,
                "away_team": away_team_name,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "venue_id": venue_id,
                "venue_name": venue_name,
                "weather": {
                    "temp": str(weather_raw.get("temp") or ""),
                    "condition": str(weather_raw.get("condition") or ""),
                    "wind": str(weather_raw.get("wind") or ""),
                },
                "travel": {
                    "home": home_travel,
                    "away": away_travel,
                },
                "home_pitcher": home_pitcher,
                "away_pitcher": away_pitcher,
                "home_lineup": home_lineup[:9],
                "away_lineup": away_lineup[:9],
            }
        )

    return games


def _fetch_schedule_for_date(game_date: str) -> dict[str, Any]:
    return api_get_json(
        f"{API_BASE}/schedule",
        params={"sportId": 1, "date": game_date, "hydrate": "probablePitcher"},
        cache_key=f"games_{game_date}",
        ttl_seconds=SCHEDULE_TTL_SECONDS,
    )


def _fetch_game_feed(game_id: int) -> dict[str, Any]:
    try:
        return api_get_json(
            f"{LIVE_FEED_BASE}/game/{game_id}/feed/live",
            cache_key=f"feed_{game_id}",
            ttl_seconds=SCHEDULE_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}


def _schedule_games(schedule_payload: dict[str, Any]) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for day in schedule_payload.get("dates") or []:
        games.extend(day.get("games") or [])
    return games


def _status_state(schedule_game: dict[str, Any]) -> str:
    status = schedule_game.get("status") or {}
    return str(status.get("detailedState") or status.get("abstractGameState") or "")


def _should_skip_status(status: str) -> bool:
    normalized = status.strip().lower()
    if not normalized:
        return False
    playable = {"scheduled", "pre-game", "pregame", "preview", "warmup"}
    if normalized in playable:
        return False
    skipped_words = (
        "in progress",
        "final",
        "game over",
        "completed",
        "delayed",
        "postponed",
        "suspended",
        "cancelled",
        "canceled",
    )
    return any(word in normalized for word in skipped_words)


def _build_pitcher(raw_pitcher: dict[str, Any] | None, game_date: str) -> dict[str, Any]:
    pitcher = dict(DEFAULT_PITCHER)
    if not raw_pitcher:
        return pitcher

    pitcher_id = safe_int(raw_pitcher.get("id"))
    if not pitcher_id:
        return pitcher

    pitcher["id"] = pitcher_id
    pitcher["name"] = raw_pitcher.get("fullName") or raw_pitcher.get("name") or "TBD"
    stat = _player_season_stat(pitcher_id, "pitching", _season_for_date(game_date))
    pitcher["era"] = safe_float(stat.get("era"), DEFAULT_PITCHER["era"])
    pitcher["whip"] = safe_float(stat.get("whip"), DEFAULT_PITCHER["whip"])
    pitcher["opponent_obp"] = safe_float(stat.get("obp"), DEFAULT_PITCHER["opponent_obp"])
    pitcher["opponent_slg"] = safe_float(stat.get("slg"), DEFAULT_PITCHER["opponent_slg"])
    return pitcher


def _lineup_from_feed(feed: dict[str, Any], side: str) -> list[dict[str, Any]]:
    box_team = (((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {}
    batting_order = [safe_int(player_id) for player_id in (box_team.get("battingOrder") or [])]
    players = box_team.get("players") or {}
    lineup: list[dict[str, Any]] = []

    for order, player_id in enumerate([pid for pid in batting_order if pid], start=1):
        player = players.get(f"ID{player_id}") or {}
        person = player.get("person") or {}
        season_batting = ((player.get("seasonStats") or {}).get("batting") or {})
        lineup.append(_batter_entry(order, player_id, person.get("fullName"), season_batting))

    return lineup


def _last_known_lineup(team_id: int, game_date: str) -> list[dict[str, Any]]:
    for game in _recent_team_games(team_id, game_date, lookback_days=14):
        if "final" not in _status_state(game).lower():
            continue
        game_id = safe_int(game.get("gamePk"))
        feed = _fetch_game_feed(game_id)
        side = _team_side(feed, team_id)
        if not side:
            continue
        lineup = _lineup_from_feed(feed, side)
        if len(lineup) >= 9:
            return lineup[:9]
    return []


def _active_roster_lineup(team_id: int, game_date: str) -> list[dict[str, Any]]:
    if not team_id:
        return []
    season = _season_for_date(game_date)
    try:
        payload = api_get_json(
            f"{API_BASE}/teams/{team_id}/roster",
            params={"rosterType": "Active", "season": season},
            cache_key=f"active_roster_{team_id}_{season}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return []

    lineup: list[dict[str, Any]] = []
    for player in payload.get("roster") or []:
        position = player.get("position") or {}
        if str(position.get("abbreviation") or "").upper() == "P":
            continue
        person = player.get("person") or {}
        player_id = safe_int(person.get("id"))
        if not player_id:
            continue
        stat = _player_season_stat(player_id, "hitting", season)
        lineup.append(_batter_entry(len(lineup) + 1, player_id, person.get("fullName"), stat))
        if len(lineup) == 9:
            break
    return lineup


def _recent_starter(team_id: int, game_date: str) -> dict[str, Any] | None:
    for game in _recent_team_games(team_id, game_date, lookback_days=21):
        if "final" not in _status_state(game).lower():
            continue
        game_id = safe_int(game.get("gamePk"))
        feed = _fetch_game_feed(game_id)
        side = _team_side(feed, team_id)
        if not side:
            continue
        box_team = (((feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}).get(side) or {}
        pitcher_ids = [safe_int(player_id) for player_id in (box_team.get("pitchers") or [])]
        if not pitcher_ids:
            continue
        players = box_team.get("players") or {}
        starter_id = pitcher_ids[0]
        starter = players.get(f"ID{starter_id}") or {}
        person = starter.get("person") or {}
        return _build_pitcher({"id": starter_id, "fullName": person.get("fullName")}, game_date)
    return None


def _recent_team_games(team_id: int, game_date: str, lookback_days: int) -> list[dict[str, Any]]:
    if not team_id:
        return []
    end_date = datetime.strptime(game_date, "%Y-%m-%d").date() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days)
    try:
        payload = api_get_json(
            f"{API_BASE}/schedule",
            params={
                "sportId": 1,
                "teamId": team_id,
                "startDate": start_date.strftime("%Y-%m-%d"),
                "endDate": end_date.strftime("%Y-%m-%d"),
            },
            cache_key=f"recent_games_{team_id}_{start_date}_{end_date}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return []

    games = _schedule_games(payload)
    games.sort(key=lambda item: (str(item.get("gameDate") or ""), safe_int(item.get("gamePk"))), reverse=True)
    return games


def _team_side(feed: dict[str, Any], team_id: int) -> str:
    teams = ((feed.get("gameData") or {}).get("teams") or {})
    if safe_int((teams.get("home") or {}).get("id")) == team_id:
        return "home"
    if safe_int((teams.get("away") or {}).get("id")) == team_id:
        return "away"
    return ""


def _player_season_stat(player_id: int, group: str, season: int) -> dict[str, Any]:
    if not player_id:
        return {}
    try:
        payload = api_get_json(
            f"{API_BASE}/people/{player_id}/stats",
            params={"stats": "season", "group": group, "season": season},
            cache_key=f"player_{player_id}_{group}_{season}",
            ttl_seconds=STATS_TTL_SECONDS,
        )
    except RuntimeError as exc:
        log_warning(str(exc))
        return {}

    splits = ((payload.get("stats") or [{}])[0].get("splits") or [])
    if not splits and season > 2000:
        try:
            prior_payload = api_get_json(
                f"{API_BASE}/people/{player_id}/stats",
                params={"stats": "season", "group": group, "season": season - 1},
                cache_key=f"player_{player_id}_{group}_{season - 1}",
                ttl_seconds=STATS_TTL_SECONDS,
            )
            splits = ((prior_payload.get("stats") or [{}])[0].get("splits") or [])
        except RuntimeError:
            splits = []
    return (splits[0].get("stat") if splits else {}) or {}


def _batter_entry(order: int, player_id: int, name: str | None, stat: dict[str, Any]) -> dict[str, Any]:
    return {
        "batting_order": order,
        "player_id": player_id,
        "name": name or f"Player {player_id}",
        "avg": safe_float(stat.get("avg"), DEFAULT_BATTER["avg"]),
        "obp": safe_float(stat.get("obp"), DEFAULT_BATTER["obp"]),
        "slg": safe_float(stat.get("slg"), DEFAULT_BATTER["slg"]),
    }


def _team_travel_context(
    team_id: int,
    game_date: str,
    current_venue_id: int,
    current_venue_name: str,
) -> dict[str, Any]:
    if not team_id:
        return {"available": False, "reason": "missing_team_id"}

    previous_game: dict[str, Any] | None = None
    for game in _recent_team_games(team_id, game_date, lookback_days=10):
        if "final" in _status_state(game).lower():
            previous_game = game
            break

    if previous_game is None:
        return {"available": False, "reason": "no_recent_final_game"}

    previous_venue = previous_game.get("venue") or {}
    previous_venue_id = safe_int(previous_venue.get("id"))
    previous_venue_name = str(previous_venue.get("name") or "")
    previous_date = _parse_game_date(previous_game.get("officialDate") or previous_game.get("gameDate"))
    target_date = _parse_game_date(game_date)
    days_since = (target_date - previous_date).days if previous_date and target_date else None

    current_geo = _venue_travel_context(current_venue_name)
    previous_geo = _venue_travel_context(previous_venue_name)
    same_venue = False
    if current_venue_id and previous_venue_id and current_venue_id == previous_venue_id:
        same_venue = True
    elif _normalize_venue_name(current_venue_name) and _normalize_venue_name(current_venue_name) == _normalize_venue_name(previous_venue_name):
        same_venue = True

    distance_miles: float | None = None
    timezone_shift: int | None = None
    travel_direction = "unknown"
    if same_venue:
        distance_miles = 0.0
        timezone_shift = 0
        travel_direction = "same"
    elif current_geo and previous_geo:
        distance_miles = _haversine_miles(
            previous_geo["lat"],
            previous_geo["lon"],
            current_geo["lat"],
            current_geo["lon"],
        )
        timezone_shift = int(current_geo["tz"] - previous_geo["tz"])
        if timezone_shift > 0:
            travel_direction = "east"
        elif timezone_shift < 0:
            travel_direction = "west"
        else:
            travel_direction = "same_timezone"

    fatigue_index = _travel_fatigue_index(days_since, distance_miles, timezone_shift, same_venue)
    run_delta = _travel_run_delta(fatigue_index)

    label_bits: list[str] = []
    if days_since is not None:
        label_bits.append(f"{days_since}d since previous game")
    if distance_miles is not None:
        label_bits.append(f"{round(distance_miles)} mi")
    if timezone_shift is not None and timezone_shift:
        label_bits.append(f"{abs(timezone_shift)}h {travel_direction}")
    if not label_bits:
        label_bits.append("travel geography unavailable")

    return {
        "available": True,
        "previous_game_date": previous_date.isoformat() if previous_date else None,
        "previous_venue_id": previous_venue_id,
        "previous_venue_name": previous_venue_name,
        "current_venue_id": current_venue_id,
        "current_venue_name": current_venue_name,
        "days_since_previous_game": days_since,
        "same_venue": same_venue,
        "distance_miles": round(distance_miles, 1) if distance_miles is not None else None,
        "timezone_shift_hours": timezone_shift,
        "travel_direction": travel_direction,
        "travel_fatigue_index": fatigue_index,
        "travel_run_delta": run_delta,
        "label": "; ".join(label_bits),
    }


def _venue_travel_context(venue_name: str) -> dict[str, float] | None:
    normalized = _normalize_venue_name(venue_name)
    if not normalized:
        return None
    if normalized in VENUE_TRAVEL_CONTEXT:
        return VENUE_TRAVEL_CONTEXT[normalized]
    for key, value in VENUE_TRAVEL_CONTEXT.items():
        if normalized in key or key in normalized:
            return value
    return None


def _normalize_venue_name(venue_name: str) -> str:
    text = str(venue_name or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    return EARTH_RADIUS_MILES * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _travel_fatigue_index(
    days_since: int | None,
    distance_miles: float | None,
    timezone_shift: int | None,
    same_venue: bool,
) -> float:
    if same_venue:
        return 0.0

    fatigue = 0.0
    if days_since is not None:
        if days_since <= 0:
            fatigue += 0.35
        elif days_since == 1:
            fatigue += 0.20
        elif days_since == 2:
            fatigue += 0.08

    if distance_miles is not None:
        if distance_miles >= 2000:
            fatigue += 0.25
        elif distance_miles >= 1200:
            fatigue += 0.16
        elif distance_miles >= 600:
            fatigue += 0.08

    if timezone_shift is not None:
        if timezone_shift >= 2:
            fatigue += 0.20
        elif timezone_shift == 1:
            fatigue += 0.08
        elif timezone_shift <= -3:
            fatigue += 0.05

    return round(min(1.0, fatigue), 3)


def _travel_run_delta(fatigue_index: float) -> float:
    return round(-0.16 * max(0.0, min(1.0, safe_float(fatigue_index, 0.0))), 3)


def _season_for_date(game_date: str) -> int:
    return datetime.strptime(game_date, "%Y-%m-%d").year


def _parse_game_date(raw_value: Any):
    raw = str(raw_value or "")[:10]
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _safe_cache_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", key).strip("_")
