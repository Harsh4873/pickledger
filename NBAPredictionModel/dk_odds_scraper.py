import json
import random
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://sportsbook.draftkings.com/",
}

DK_NBA_URL = (
    "https://sportsbook.draftkings.com/api/odds/v1/leagues/42648/"
    "categories/583/subcategories/4511"
)
DK_MLB_URL = (
    "https://sportsbook.draftkings.com/api/odds/v1/leagues/84240/"
    "categories/1074/subcategories/4511"
)


def _sleep_before_return() -> None:
    time.sleep(random.uniform(2, 4))


def _first_value(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("−", "-")
        if cleaned.upper() in {"EVEN", "PK", "PICK"}:
            return 0.0
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _coerce_int(value: Any) -> int | None:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _iter_objects(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_objects(item)


def _extract_teams_from_participants(event: dict[str, Any]) -> tuple[str | None, str | None]:
    participants = event.get("participants")
    if not isinstance(participants, list):
        return None, None

    home_team = None
    away_team = None
    fallback_names: list[str] = []

    for participant in participants:
        if not isinstance(participant, dict):
            continue

        name = _first_value(
            participant,
            ["name", "teamName", "shortName", "displayName"],
        )
        if name:
            fallback_names.append(str(name))

        role = _normalize_name(
            _first_value(
                participant,
                ["venueRole", "homeAway", "alignment", "role"],
            )
        )
        if role == "home" and name:
            home_team = str(name)
        elif role == "away" and name:
            away_team = str(name)

    if not home_team and len(fallback_names) >= 2:
        home_team = fallback_names[0]
    if not away_team and len(fallback_names) >= 2:
        away_team = fallback_names[1]

    return home_team, away_team


def _extract_events(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}

    for obj in _iter_objects(payload):
        if not isinstance(obj, dict):
            continue

        event_id = _first_value(obj, ["eventId", "id"])
        home_team = _first_value(obj, ["homeTeamName", "homeTeam", "teamName1"])
        away_team = _first_value(obj, ["awayTeamName", "awayTeam", "teamName2"])

        if not home_team or not away_team:
            parsed_home, parsed_away = _extract_teams_from_participants(obj)
            home_team = home_team or parsed_home
            away_team = away_team or parsed_away

        if not event_id or not home_team or not away_team:
            continue

        game_time = _first_value(
            obj,
            [
                "startEventDate",
                "startDate",
                "startDateTime",
                "startTime",
                "eventDate",
                "eventTime",
            ],
        )

        events[str(event_id)] = {
            "game_time": str(game_time) if game_time is not None else None,
            "home_team": str(home_team),
            "away_team": str(away_team),
            "spread_home": None,
            "spread_away": None,
            "total_line": None,
            "ml_home": None,
            "ml_away": None,
        }

    return events


def _market_name(obj: dict[str, Any]) -> str:
    parts = [
        _first_value(obj, ["label", "name", "marketType", "marketName", "criterionName"]),
        _first_value(obj, ["subcategoryName", "subcategoryLabel", "description"]),
    ]
    return " ".join(str(part) for part in parts if part).lower()


def _extract_outcomes(obj: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ["outcomes", "selections", "offers"]:
        value = obj.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _extract_market_event_id(obj: dict[str, Any], event_lookup: dict[str, dict[str, Any]]) -> str | None:
    direct = _first_value(obj, ["eventId", "event_id", "eventGroupId"])
    if direct is not None and str(direct) in event_lookup:
        return str(direct)

    event = obj.get("event")
    if isinstance(event, dict):
        nested = _first_value(event, ["eventId", "id"])
        if nested is not None and str(nested) in event_lookup:
            return str(nested)

    return None


def _extract_outcome_name(outcome: dict[str, Any]) -> str:
    return _normalize_name(
        _first_value(
            outcome,
            [
                "label",
                "name",
                "outcomeName",
                "participant",
                "participantName",
                "runnerName",
                "side",
            ],
        )
    )


def _extract_outcome_line(outcome: dict[str, Any]) -> float | None:
    for key in ["line", "points", "spread", "value", "total"]:
        parsed = _coerce_float(outcome.get(key))
        if parsed is not None:
            return parsed

    price = outcome.get("price")
    if isinstance(price, dict):
        for key in ["line", "points", "spread"]:
            parsed = _coerce_float(price.get(key))
            if parsed is not None:
                return parsed

    return None


def _extract_outcome_odds(outcome: dict[str, Any]) -> int | None:
    for key in ["americanOdds", "oddsAmerican", "displayOdds", "odds"]:
        parsed = _coerce_int(outcome.get(key))
        if parsed is not None:
            return parsed

    price = outcome.get("price")
    if isinstance(price, dict):
        for key in ["american", "americanOdds", "oddsAmerican"]:
            parsed = _coerce_int(price.get(key))
            if parsed is not None:
                return parsed

    return None


def _apply_market(game: dict[str, Any], market_name: str, outcomes: list[dict[str, Any]]) -> None:
    home_name = _normalize_name(game["home_team"])
    away_name = _normalize_name(game["away_team"])

    if "spread" in market_name or "run line" in market_name:
        for outcome in outcomes:
            outcome_name = _extract_outcome_name(outcome)
            line = _extract_outcome_line(outcome)
            if line is None:
                continue
            if home_name and home_name in outcome_name:
                game["spread_home"] = line
            elif away_name and away_name in outcome_name:
                game["spread_away"] = line
        return

    if "total" in market_name or "over/under" in market_name or "over under" in market_name:
        for outcome in outcomes:
            line = _extract_outcome_line(outcome)
            if line is not None:
                game["total_line"] = line
                return
        return

    if "moneyline" in market_name or "winner" in market_name or "to win" in market_name:
        for outcome in outcomes:
            outcome_name = _extract_outcome_name(outcome)
            odds = _extract_outcome_odds(outcome)
            if odds is None:
                continue
            if home_name and home_name in outcome_name:
                game["ml_home"] = odds
            elif away_name and away_name in outcome_name:
                game["ml_away"] = odds


def _parse_games(payload: dict[str, Any], league: str) -> list[dict[str, Any]]:
    events = _extract_events(payload)
    if not events:
        print(f"{league}: unexpected DraftKings response structure; no events found.")
        return []

    for obj in _iter_objects(payload):
        if not isinstance(obj, dict):
            continue

        outcomes = _extract_outcomes(obj)
        if not outcomes:
            continue

        market_name = _market_name(obj)
        if not market_name:
            continue

        if not any(
            token in market_name
            for token in ("spread", "run line", "total", "over/under", "over under", "moneyline", "winner", "to win")
        ):
            continue

        event_id = _extract_market_event_id(obj, events)
        if event_id is None:
            continue

        _apply_market(events[event_id], market_name, outcomes)

    rows = []
    for game in events.values():
        rows.append(
            {
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "game_time": game["game_time"],
                "spread_home": game["spread_home"],
                "spread_away": game["spread_away"],
                "total_line": game["total_line"],
                "ml_home": game["ml_home"],
                "ml_away": game["ml_away"],
            }
        )

    populated_rows = [
        row
        for row in rows
        if any(
            row[field] is not None
            for field in ("spread_home", "spread_away", "total_line", "ml_home", "ml_away")
        )
    ]

    if not populated_rows:
        print(f"{league}: events found, but no odds markets were parsed from the response.")
        return []

    return populated_rows


def fetch_dk_odds(url: str, league: str) -> list[dict[str, Any]]:
    response = None
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"{league}: request failed for {url}: {exc}")
        if response is not None:
            preview = response.text[:500]
            print(f"{league}: status={response.status_code}, response preview={preview}")
        _sleep_before_return()
        return []

    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        print(f"{league}: failed to parse JSON from {url}: {exc}")
        print(f"{league}: status={response.status_code}, response preview={response.text[:500]}")
        _sleep_before_return()
        return []

    try:
        odds_list = _parse_games(payload, league)
    except Exception as exc:
        print(f"{league}: unexpected parsing error: {exc}")
        print(f"{league}: status={response.status_code}, response preview={response.text[:500]}")
        _sleep_before_return()
        return []

    if not odds_list:
        print(f"{league}: status={response.status_code}, response preview={response.text[:500]}")

    _sleep_before_return()
    return odds_list


def save_odds_to_db(odds_list: list[dict[str, Any]], league: str) -> None:
    db_path = Path(__file__).resolve().parent.parent / "pickledger.db"
    fetched_at = datetime.now(timezone.utc).isoformat()

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS dk_odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    league TEXT,
                    fetched_at TEXT,
                    game_time TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    spread_home REAL,
                    spread_away REAL,
                    total_line REAL,
                    ml_home INTEGER,
                    ml_away INTEGER
                )
                """
            )

            cursor.executemany(
                """
                INSERT INTO dk_odds (
                    league,
                    fetched_at,
                    game_time,
                    home_team,
                    away_team,
                    spread_home,
                    spread_away,
                    total_line,
                    ml_home,
                    ml_away
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        league,
                        fetched_at,
                        row.get("game_time"),
                        row.get("home_team"),
                        row.get("away_team"),
                        row.get("spread_home"),
                        row.get("spread_away"),
                        row.get("total_line"),
                        row.get("ml_home"),
                        row.get("ml_away"),
                    )
                    for row in odds_list
                ],
            )
            conn.commit()
    except sqlite3.Error as exc:
        print(f"{league}: database write failed: {exc}")
        return

    print(f"Saved {len(odds_list)} {league} odds rows")


def _print_recent_rows() -> None:
    db_path = Path(__file__).resolve().parent.parent / "pickledger.db"
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT league, fetched_at, game_time, home_team, away_team,
                       spread_home, spread_away, total_line, ml_home, ml_away
                FROM dk_odds
                ORDER BY id DESC
                LIMIT 3
                """
            )
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        print(f"Sanity check failed while reading dk_odds: {exc}")
        return

    print("Latest dk_odds rows:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    nba_odds = fetch_dk_odds(DK_NBA_URL, "NBA")
    save_odds_to_db(nba_odds, "NBA")

    time.sleep(3)

    mlb_odds = fetch_dk_odds(DK_MLB_URL, "MLB")
    save_odds_to_db(mlb_odds, "MLB")

    _print_recent_rows()
