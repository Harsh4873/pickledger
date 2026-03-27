from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests
import statsapi

from market_mechanics import convert_american_to_implied


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
SCHEDULE_CACHE_DIR = CACHE_DIR / "schedules"
GAME_CACHE_DIR = CACHE_DIR / "games"
PLAYER_CACHE_DIR = CACHE_DIR / "players"
PLAYER_GAME_LOG_CACHE_DIR = CACHE_DIR / "player_game_logs"
SEASON_STATS_CACHE_DIR = CACHE_DIR / "season_stats"
ODDS_CACHE_FILE = CACHE_DIR / "historical_odds.json"

ODDS_ARCHIVE_URL = (
    "https://github.com/ArnavSaraogi/mlb-odds-scraper/releases/download/dataset/"
    "mlb_odds_dataset.json"
)


def ensure_data_dirs() -> None:
    for path in (
        DATA_DIR,
        CACHE_DIR,
        SCHEDULE_CACHE_DIR,
        GAME_CACHE_DIR,
        PLAYER_CACHE_DIR,
        PLAYER_GAME_LOG_CACHE_DIR,
        SEASON_STATS_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _median_int(values: list[int]) -> int | None:
    clean = sorted(v for v in values if isinstance(v, int))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2 == 1:
        return clean[mid]
    return round((clean[mid - 1] + clean[mid]) / 2)


def _parse_total_line_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
        return None

    if isinstance(value, dict):
        for key in ("line", "value", "displayValue", "overUnder", "total", "totalRuns"):
            parsed = _parse_total_line_value(value.get(key))
            if parsed is not None:
                return parsed
    return None


def _walk_total_line_candidates(node: Any) -> float | None:
    if isinstance(node, dict):
        for key, value in node.items():
            normalized_key = key.lower()
            if normalized_key in {
                "overunder",
                "over_under",
                "overunderline",
                "over_under_line",
                "totalline",
                "total_line",
                "bettingtotal",
                "sportsbooktotal",
                "totalrunsline",
            }:
                parsed = _parse_total_line_value(value)
                if parsed is not None:
                    return parsed

            if normalized_key in {"odds", "betting", "sportsbook", "lines"}:
                parsed = _walk_total_line_candidates(value)
                if parsed is not None:
                    return parsed

            if isinstance(value, (dict, list)):
                parsed = _walk_total_line_candidates(value)
                if parsed is not None:
                    return parsed

    if isinstance(node, list):
        for item in node:
            parsed = _walk_total_line_candidates(item)
            if parsed is not None:
                return parsed
    return None


def extract_total_line_from_game_feed(payload: dict[str, Any]) -> float | None:
    for path in (
        ("gameData", "odds", "overUnder"),
        ("gameData", "odds", "total"),
        ("gameData", "betting", "overUnder"),
        ("gameData", "betting", "total"),
        ("liveData", "odds", "overUnder"),
        ("liveData", "odds", "total"),
    ):
        node: Any = payload
        for part in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(part)
        parsed = _parse_total_line_value(node)
        if parsed is not None:
            return parsed

    return _walk_total_line_candidates(payload)


class StatsAPIClient:
    """Small cached wrapper around MLB StatsAPI."""

    def __init__(self, sleep_s: float = 0.0) -> None:
        ensure_data_dirs()
        self.sleep_s = sleep_s

    def _sleep(self) -> None:
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)

    def get_schedule_for_season(self, season: int) -> list[dict[str, Any]]:
        cache_file = SCHEDULE_CACHE_DIR / f"{season}.json"
        cached = _read_json(cache_file)
        if cached is not None:
            return cached

        self._sleep()
        games = statsapi.schedule(
            start_date=f"{season}-03-01",
            end_date=f"{season}-11-30",
            sportId=1,
        )
        regular_season = [
            game
            for game in games
            if str(game.get("game_type", "")).upper() == "R"
        ]
        _write_json(cache_file, regular_season)
        return regular_season

    def get_game_feed(self, game_pk: int) -> dict[str, Any]:
        cache_file = GAME_CACHE_DIR / f"{game_pk}.json"
        cached = _read_json(cache_file)
        if cached is not None:
            return cached

        return self._download_game_feed(game_pk)

    def get_game_total_line(self, game_pk: int) -> float | None:
        return extract_total_line_from_game_feed(self.get_game_feed(game_pk))

    def _download_game_feed(self, game_pk: int) -> dict[str, Any]:
        cache_file = GAME_CACHE_DIR / f"{game_pk}.json"
        cached = _read_json(cache_file)
        if cached is not None:
            return cached
        self._sleep()
        payload = statsapi.get("game", {"gamePk": game_pk})
        _write_json(cache_file, payload)
        return payload

    def prefetch_game_feeds(self, game_pks: list[int], max_workers: int = 10) -> None:
        missing = [
            int(game_pk)
            for game_pk in sorted(set(game_pks))
            if not (GAME_CACHE_DIR / f"{int(game_pk)}.json").exists()
        ]
        if not missing:
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self._download_game_feed, missing))

    def get_person(self, player_id: int) -> dict[str, Any]:
        cache_file = PLAYER_CACHE_DIR / f"{player_id}.json"
        cached = _read_json(cache_file)
        if cached is not None:
            return cached

        self._sleep()
        payload = statsapi.get("person", {"personId": player_id})
        people = payload.get("people", [])
        person = people[0] if people else {}
        _write_json(cache_file, person)
        return person

    def get_player_game_log(
        self,
        player_id: int,
        season: int,
        group: str = "pitching",
    ) -> list[dict[str, Any]]:
        cache_file = PLAYER_GAME_LOG_CACHE_DIR / f"{season}_{group}_{player_id}.json"
        cached = _read_json(cache_file)
        if cached is not None:
            return list(cached)

        self._sleep()
        payload = statsapi.get(
            "person",
            {
                "personId": player_id,
                "hydrate": f"stats(group=[{group}],type=[gameLog],season={season})",
            },
        )
        people = payload.get("people", [])
        stats = (people[0].get("stats") or []) if people else []
        splits = stats[0].get("splits", []) if stats else []
        _write_json(cache_file, splits)
        return splits

    def get_season_player_stats(
        self,
        season: int,
        group: str,
    ) -> dict[int, dict[str, Any]]:
        cache_file = SEASON_STATS_CACHE_DIR / f"{season}_{group}.json"
        cached = _read_json(cache_file)
        if cached is not None:
            return {int(key): value for key, value in cached.items()}

        self._sleep()
        payload = statsapi.get(
            "stats",
            {
                "stats": "season",
                "group": group,
                "sportIds": 1,
                "season": season,
                "limit": 5000,
            },
        )
        stats_index: dict[int, dict[str, Any]] = {}
        for split in payload.get("stats", [{}])[0].get("splits", []):
            player = split.get("player") or {}
            player_id = player.get("id")
            if player_id is None:
                continue
            stats_index[int(player_id)] = split

        _write_json(cache_file, stats_index)
        return stats_index


class HistoricalOddsArchive:
    """Optional historical MLB moneyline source for backtests and ROI."""

    def __init__(self, download_url: str = ODDS_ARCHIVE_URL) -> None:
        ensure_data_dirs()
        self.download_url = download_url
        self._index: dict[tuple[str, str, str], dict[str, Any]] | None = None

    def ensure_downloaded(self, force: bool = False) -> Path:
        if ODDS_CACHE_FILE.exists() and not force:
            return ODDS_CACHE_FILE

        response = requests.get(self.download_url, timeout=120)
        response.raise_for_status()
        ODDS_CACHE_FILE.write_bytes(response.content)
        return ODDS_CACHE_FILE

    def load(self) -> dict[str, Any]:
        self.ensure_downloaded()
        payload = _read_json(ODDS_CACHE_FILE)
        return payload or {}

    def build_index(self) -> dict[tuple[str, str, str], dict[str, Any]]:
        if self._index is not None:
            return self._index

        payload = self.load()
        index: dict[tuple[str, str, str], dict[str, Any]] = {}
        for game_date, daily_games in payload.items():
            for entry in daily_games:
                game_view = entry.get("gameView") or {}
                away_team = (game_view.get("awayTeam") or {}).get("shortName")
                home_team = (game_view.get("homeTeam") or {}).get("shortName")
                if not away_team or not home_team:
                    continue
                key = (game_date, away_team.upper(), home_team.upper())
                index[key] = entry

        self._index = index
        return index

    def lookup_moneyline(
        self,
        game_date: str,
        away_abbrev: str,
        home_abbrev: str,
    ) -> dict[str, Any]:
        index = self.build_index()
        entry = index.get((game_date, away_abbrev.upper(), home_abbrev.upper()))
        if not entry:
            return {}

        market = entry.get("odds", {}).get("moneyline") or []
        home_open: list[int] = []
        away_open: list[int] = []
        home_close: list[int] = []
        away_close: list[int] = []
        for book in market:
            opening = book.get("openingLine") or {}
            current = book.get("currentLine") or {}
            if isinstance(opening.get("homeOdds"), int):
                home_open.append(opening["homeOdds"])
            if isinstance(opening.get("awayOdds"), int):
                away_open.append(opening["awayOdds"])
            if isinstance(current.get("homeOdds"), int):
                home_close.append(current["homeOdds"])
            if isinstance(current.get("awayOdds"), int):
                away_close.append(current["awayOdds"])

        home_close_line = _median_int(home_close)
        away_close_line = _median_int(away_close)
        home_open_line = _median_int(home_open)
        away_open_line = _median_int(away_open)

        out: dict[str, Any] = {
            "home_moneyline": home_close_line,
            "away_moneyline": away_close_line,
            "home_open_moneyline": home_open_line,
            "away_open_moneyline": away_open_line,
            "moneyline_books": len(market),
        }

        if home_close_line is not None:
            out["home_implied_prob"] = convert_american_to_implied(home_close_line)
        if away_close_line is not None:
            out["away_implied_prob"] = convert_american_to_implied(away_close_line)

        return out
