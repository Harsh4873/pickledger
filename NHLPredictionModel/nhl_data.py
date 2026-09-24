"""Official NHL slate for one date.

ESPN's hockey/nhl scoreboard is the same source the other sports use. The NHL
schedule API is the fallback when that scoreboard cannot be read. Neither
path copies a live score into a pick.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.scrapers.espn_scoreboard import fetch_scoreboard_json  # noqa: E402

NHL_SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{date}"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={compact}"
)
PREGAME_STATUSES = {"STATUS_SCHEDULED", "FUT", "PRE"}
SEASON_BY_CODE = {
    "1": "PRE",
    "2": "REG",
    "3": "POST",
    "PRE": "PRE",
    "PRESEASON": "PRE",
    "REG": "REG",
    "REGULAR": "REG",
    "REGULAR SEASON": "REG",
    "POST": "POST",
    "POSTSEASON": "POST",
}


def _season_type(value: Any) -> str:
    token = str(value or "").strip().upper().replace("-", " ")
    return SEASON_BY_CODE.get(token, "")


def _espn_games(date_iso: str) -> list[dict[str, Any]]:
    compact = date_iso.replace("-", "")
    payload = fetch_scoreboard_json(ESPN_SCOREBOARD_URL.format(compact=compact))
    games: list[dict[str, Any]] = []
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        competitions = event.get("competitions") if isinstance(event.get("competitions"), list) else []
        competition = competitions[0] if competitions and isinstance(competitions[0], dict) else {}
        competitors = competition.get("competitors") if isinstance(competition.get("competitors"), list) else []
        home = away = None
        for competitor in competitors:
            if not isinstance(competitor, dict):
                continue
            team = competitor.get("team") if isinstance(competitor.get("team"), dict) else {}
            side = {
                "name": str(team.get("displayName") or "").strip(),
                "abbrev": str(team.get("abbreviation") or "").strip().upper(),
            }
            if str(competitor.get("homeAway") or "").lower() == "home":
                home = side
            elif str(competitor.get("homeAway") or "").lower() == "away":
                away = side
        if not home or not away or not home["name"] or not away["name"]:
            continue
        status = event.get("status") if isinstance(event.get("status"), dict) else {}
        status_type = status.get("type") if isinstance(status.get("type"), dict) else {}
        season = event.get("season") if isinstance(event.get("season"), dict) else {}
        season_type_value = season.get("type")
        if isinstance(season_type_value, dict):
            season_token = season_type_value.get("abbreviation") or season_type_value.get("name") or season_type_value.get("type")
        else:
            season_token = season.get("slug") or season_type_value
        games.append({
            "game_id": str(event.get("id") or competition.get("id") or ""),
            "home_team": home["name"],
            "away_team": away["name"],
            "home_abbrev": home["abbrev"],
            "away_abbrev": away["abbrev"],
            "start_time": str(event.get("date") or competition.get("date") or ""),
            "status": str(status_type.get("name") or ""),
            "season_type": _season_type(season_token),
            "slate_source": "espn_scoreboard",
        })
    return games


def _nhl_name(team: dict[str, Any]) -> str:
    place = team.get("placeName") if isinstance(team.get("placeName"), dict) else {}
    common = team.get("commonName") if isinstance(team.get("commonName"), dict) else {}
    return f"{place.get('default') or ''} {common.get('default') or ''}".strip()


def _nhl_api_games(date_iso: str) -> list[dict[str, Any]]:
    request = Request(
        NHL_SCHEDULE_URL.format(date=date_iso),
        headers={"User-Agent": "PickLedger/1.0", "Accept": "application/json"},
    )
    with urlopen(request, timeout=20) as response:
        payload = json.load(response)
    games: list[dict[str, Any]] = []
    for day in payload.get("gameWeek") or []:
        if not isinstance(day, dict) or str(day.get("date") or "") != date_iso:
            continue
        for game in day.get("games") or []:
            if not isinstance(game, dict):
                continue
            home = game.get("homeTeam") if isinstance(game.get("homeTeam"), dict) else {}
            away = game.get("awayTeam") if isinstance(game.get("awayTeam"), dict) else {}
            home_name = _nhl_name(home)
            away_name = _nhl_name(away)
            if not home_name or not away_name:
                continue
            games.append({
                "game_id": str(game.get("id") or ""),
                "home_team": home_name,
                "away_team": away_name,
                "home_abbrev": str(home.get("abbrev") or "").upper(),
                "away_abbrev": str(away.get("abbrev") or "").upper(),
                "start_time": str(game.get("startTimeUTC") or ""),
                "status": str(game.get("gameState") or ""),
                "season_type": _season_type(game.get("gameType")),
                "slate_source": "nhl_schedule_api",
            })
    return games


def load_slate(date_iso: str) -> tuple[list[dict[str, Any]], str]:
    """Return (games, source). Raises when neither official slate can be read."""
    try:
        return _espn_games(date_iso), "espn_scoreboard"
    except Exception as espn_error:
        try:
            return _nhl_api_games(date_iso), "nhl_schedule_api"
        except Exception as nhl_error:
            raise RuntimeError(
                f"NHL slate unavailable for {date_iso}: espn={espn_error}; nhl_api={nhl_error}"
            ) from nhl_error


def is_pregame(game: dict[str, Any]) -> bool:
    return str(game.get("status") or "").strip().upper() in PREGAME_STATUSES
