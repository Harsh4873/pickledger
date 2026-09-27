"""Observed NHL scoring rates.

Team means come from the completed regular-season standings already stored in
the ratings artifact. Player means come from the NHL stats summary
(counting stat divided by games played). A missing row, a too-small sample,
or an ambiguous name stays empty. Nothing here fills a league-average mean.
"""
from __future__ import annotations

import json
from typing import Any, Callable
from urllib.request import Request, urlopen

try:
    from nhl_core import canonical_abbrev
except ImportError:
    from .nhl_core import canonical_abbrev

MIN_OBSERVED_GAMES = 20
TEAM_MEAN_SOURCE = "nhl_standings_goals_for"
SKATER_MEAN_SOURCE = "nhl_skater_summary"
GOALIE_MEAN_SOURCE = "nhl_goalie_summary"
SKATER_URL = (
    "https://api.nhle.com/stats/rest/en/skater/summary"
    "?limit=100&start={start}&cayenneExp=seasonId={season}%20and%20gameTypeId=2"
)
GOALIE_URL = (
    "https://api.nhle.com/stats/rest/en/goalie/summary"
    "?limit=100&start={start}&cayenneExp=seasonId={season}%20and%20gameTypeId=2"
)

FetchJson = Callable[[str], dict[str, Any] | None]


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "á": "a", "à": "a", "ä": "a", "â": "a",
        "é": "e", "è": "e", "ë": "e", "ê": "e",
        "í": "i", "ì": "i", "ï": "i", "î": "i",
        "ó": "o", "ò": "o", "ö": "o", "ô": "o",
        "ú": "u", "ù": "u", "ü": "u", "û": "u",
        "ñ": "n", "ç": "c", "ø": "o", "å": "a",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return " ".join("".join(char if char.isalnum() else " " for char in text).split())


def stat_key_from_market(name: Any) -> str | None:
    """Map a posted market title onto a counting stat we can look up.

    Team totals, race markets, and special-teams titles are not player props.
    """
    text = " ".join(str(name or "").lower().replace(":", " ").replace("_", " ").split())
    if not text or "team total" in text or "team to" in text or "first to" in text:
        return None
    if "power play" in text or "shorthanded" in text or "short handed" in text:
        return None
    if "shots on goal" in text or "player shots" in text or text == "shots":
        return "shots"
    if "assist" in text:
        return "assists"
    if "point" in text:
        return "points"
    if "save" in text:
        return "saves"
    if "goal" in text:
        return "goals"
    return None


def _per_game(total: Any, games: Any) -> float | None:
    try:
        games_played = int(games)
        counted = float(total)
    except (TypeError, ValueError):
        return None
    if games_played < MIN_OBSERVED_GAMES or counted < 0:
        return None
    return counted / games_played


def team_observed_mean(ratings: dict[str, Any], abbrev: Any) -> dict[str, Any] | None:
    teams = ratings.get("teams") if isinstance(ratings, dict) else None
    row = teams.get(canonical_abbrev(abbrev)) if isinstance(teams, dict) else None
    if not isinstance(row, dict):
        return None
    mean = _per_game(row.get("goals_for"), row.get("games_played"))
    if mean is None:
        try:
            games_played = int(row.get("games_played"))
            mean = float(row.get("goals_for_per_game"))
        except (TypeError, ValueError):
            return None
        if games_played < MIN_OBSERVED_GAMES:
            return None
    return {
        "mean": mean,
        "games_played": int(row.get("games_played")),
        "mean_source": TEAM_MEAN_SOURCE,
        "season": ratings.get("prior_season"),
    }


def _store_player(index: dict[str, dict[str, Any]], *, name: str, games: Any, stats: dict[str, Any], source: str, season: Any) -> None:
    key = normalize_name(name)
    if not key:
        return
    rates: dict[str, Any] = {"games_played": games, "mean_source": source, "season": season, "name": name}
    for stat, total in stats.items():
        mean = _per_game(total, games)
        if mean is not None:
            rates[stat] = mean
    if len(rates) <= 4:
        return
    current = index.get(key)
    if current is None:
        index[key] = rates
        return
    if current.get("ambiguous"):
        return
    index[key] = {"ambiguous": True, "name": name}


def rates_from_summaries(
    skaters: list[dict[str, Any]] | None,
    goalies: list[dict[str, Any]] | None,
    *,
    season: Any,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in skaters or []:
        if not isinstance(row, dict):
            continue
        _store_player(
            index,
            name=str(row.get("skaterFullName") or ""),
            games=row.get("gamesPlayed"),
            stats={
                "goals": row.get("goals"),
                "assists": row.get("assists"),
                "points": row.get("points"),
                "shots": row.get("shots"),
            },
            source=SKATER_MEAN_SOURCE,
            season=season,
        )
    for row in goalies or []:
        if not isinstance(row, dict):
            continue
        _store_player(
            index,
            name=str(row.get("goalieFullName") or ""),
            games=row.get("gamesPlayed"),
            stats={"saves": row.get("saves")},
            source=GOALIE_MEAN_SOURCE,
            season=season,
        )
    return index


def lookup_player_mean(rates: dict[str, dict[str, Any]] | None, player: Any, stat: Any) -> dict[str, Any] | None:
    if not isinstance(rates, dict) or not stat:
        return None
    row = rates.get(normalize_name(player))
    if not isinstance(row, dict) or row.get("ambiguous"):
        return None
    mean = row.get(str(stat))
    if not isinstance(mean, (int, float)):
        return None
    return {
        "mean": float(mean),
        "games_played": row.get("games_played"),
        "mean_source": row.get("mean_source"),
        "season": row.get("season"),
        "stat": str(stat),
    }


def _default_fetch(url: str) -> dict[str, Any] | None:
    request = Request(url, headers={"User-Agent": "PickLedger/1.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _paged(url_template: str, season: str, fetch: FetchJson) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start = 0
    total: int | None = None
    for _ in range(20):
        payload = fetch(url_template.format(start=start, season=season))
        if not isinstance(payload, dict):
            break
        page = [row for row in payload.get("data") or [] if isinstance(row, dict)]
        if total is None:
            try:
                total = int(payload.get("total"))
            except (TypeError, ValueError):
                total = None
        if not page:
            break
        rows.extend(page)
        start += len(page)
        if total is not None and start >= total:
            break
        if len(page) < 100:
            break
    return rows


def fetch_player_rates(season: str, *, fetch_json: FetchJson | None = None) -> dict[str, dict[str, Any]]:
    fetch = fetch_json or _default_fetch
    skaters = _paged(SKATER_URL, season, fetch)
    goalies = _paged(GOALIE_URL, season, fetch)
    return rates_from_summaries(skaters, goalies, season=season)
