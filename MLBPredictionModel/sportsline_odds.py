import json
import random
import re
import sqlite3
import time
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    from MLBPredictionModel.date_utils import get_mlb_slate_date
except ModuleNotFoundError:
    from date_utils import get_mlb_slate_date


SL_NBA_SPREAD = "https://www.sportsline.com/nba/odds/picks-against-the-spread/"
SL_NBA_TOTAL = "https://www.sportsline.com/nba/odds/over-under/"
SL_NBA_ML = "https://www.sportsline.com/nba/odds/money-line/"

SL_MLB_SPREAD = "https://www.sportsline.com/mlb/odds/picks-against-the-spread/"
SL_MLB_TOTAL = "https://www.sportsline.com/mlb/odds/over-under/"
SL_MLB_ML = "https://www.sportsline.com/mlb/odds/money-line/"
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sportsline.com/",
}

SPORTSLINE_URLS = {
    "NBA": {
        "spread": SL_NBA_SPREAD,
        "total": SL_NBA_TOTAL,
        "ml": SL_NBA_ML,
    },
    "MLB": {
        "spread": SL_MLB_SPREAD,
        "total": SL_MLB_TOTAL,
        "ml": SL_MLB_ML,
    },
}

REQUIRED_FIELDS = {
    "spread": "spread_home",
    "total": "total_line",
    "ml": "ml_home",
}

MONTH_RE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")

LAST_TABLE_HTML: dict[tuple[str, str], str] = {}
LAST_MARKET_ROWS: dict[str, dict[str, list[dict[str, object | None]]]] = {}


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _strip_record_suffix(text: str) -> str:
    return re.sub(r"\s+\d+-\d+\s*$", "", text).strip()


def _strip_expert_pick_prefix(text: str) -> str:
    return re.sub(r"^\d+\s+Expert Pick[s]?\s+", "", text).strip()


def _coerce_float(value: str | None) -> float | None:
    cleaned = _clean_text(value).replace("−", "-")
    if not cleaned or cleaned == "--":
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_int(value: str | None) -> int | None:
    cleaned = _clean_text(value).replace("−", "-")
    if not cleaned or cleaned == "--":
        return None
    try:
        return int(cleaned)
    except ValueError:
        try:
            return int(float(cleaned))
        except ValueError:
            return None


def _blank_game(
    league: str,
    away_team: str | None,
    home_team: str | None,
    game_time: str | None,
) -> dict[str, object | None]:
    return {
        "league": league,
        "game_time": game_time,
        "away_team": away_team,
        "home_team": home_team,
        "spread_away": None,
        "spread_home": None,
        "spread_odds": None,
        "total_line": None,
        "total_odds": None,
        "ml_away": None,
        "ml_home": None,
    }


def _last_word(text: str | None) -> str:
    parts = _clean_text(text).lower().split()
    return parts[-1] if parts else ""


def _team_matches(left: str | None, right: str | None) -> bool:
    left_clean = _clean_text(left).lower()
    right_clean = _clean_text(right).lower()
    if not left_clean or not right_clean:
        return False
    return (
        left_clean == right_clean
        or left_clean in right_clean
        or right_clean in left_clean
        or _last_word(left_clean) == _last_word(right_clean)
    )


def _find_existing_game(
    games: list[dict[str, object | None]],
    away_team: str | None,
    home_team: str | None,
) -> dict[str, object | None] | None:
    for game in games:
        if _team_matches(game.get("away_team"), away_team) and _team_matches(game.get("home_team"), home_team):
            return game
    return None


def _extract_table_preview(soup: BeautifulSoup | None, html: str) -> str:
    if soup is not None:
        table = soup.find("table")
        if table is not None:
            return str(table)[:800]
    return html[:800]


def _looks_like_game_info(text: str) -> bool:
    cleaned = _clean_text(text)
    return bool(cleaned and MONTH_RE.search(cleaned) and ("UTC" in cleaned or " on " in cleaned))


def _market_has_values(rows: list[dict[str, object | None]], market_type: str) -> bool:
    field = REQUIRED_FIELDS[market_type]
    return any(row.get(field) is not None for row in rows)


def _parse_spread_text(text: str) -> tuple[float | None, int | None]:
    current_val = _clean_text(text).replace("−", "-").split("Open:", 1)[0].strip()
    parts = current_val.split()
    if not parts:
        return None, None

    first_part = parts[0]
    if first_part.upper() in {"PK", "PICK", "PICKEM", "EVEN"}:
        spread = 0.0
    else:
        spread = float(first_part)

    odds = int(parts[1]) if len(parts) > 1 else -110
    return spread, odds


def _parse_total_text(text: str) -> tuple[float | None, int | None]:
    current_val = _clean_text(text).replace("−", "-").split("Open:", 1)[0].strip()
    parts = current_val.split()
    if not parts:
        return None, None

    total = float(re.sub(r"^[ouOU]", "", parts[0]))
    odds = int(parts[1]) if len(parts) > 1 else -110
    return total, odds


def _parse_moneyline_text(text: str) -> int | None:
    current_val = _clean_text(text).replace("−", "-").split("Open:", 1)[0].strip()
    parts = current_val.split()
    if not parts:
        return None
    return int(parts[0])


def _parse_table_games(
    soup: BeautifulSoup,
    market_type: str,
    league: str,
) -> list[dict[str, object | None]]:
    parsed_games: list[dict[str, object | None]] = []

    for table in soup.find_all("table"):
        filtered_rows: list[list[str]] = []

        for row in table.find_all("tr"):
            try:
                cells = row.find_all(["th", "td"])
                if not cells:
                    continue

                texts = [_clean_text(cell.get_text(" ", strip=True)) for cell in cells]
                first_text = texts[0] if texts else ""
                combined = " ".join(text for text in texts if text)

                if not combined:
                    continue
                if first_text == "Matchup":
                    continue
                if "Advanced Insights" in combined:
                    continue

                if len(cells) == 1:
                    info_text = _strip_expert_pick_prefix(first_text)
                    if _looks_like_game_info(info_text):
                        filtered_rows.append([info_text])
                    continue

                if "Expert Pick" in combined:
                    continue

                filtered_rows.append(texts)
            except Exception:
                continue

        for index in range(0, len(filtered_rows) - 2, 3):
            away_row = filtered_rows[index]
            home_row = filtered_rows[index + 1]
            info_row = filtered_rows[index + 2]

            try:
                if len(away_row) < 3 or len(home_row) < 3 or not info_row:
                    continue
                if not _looks_like_game_info(info_row[0]):
                    continue

                away_team = _strip_record_suffix(away_row[0])
                home_team = _strip_record_suffix(home_row[0])
                game_time = _strip_expert_pick_prefix(info_row[0])

                if not away_team or not home_team or not game_time:
                    continue

                game = _blank_game(league, away_team, home_team, game_time)
                away_consensus = away_row[2]
                home_consensus = home_row[2]

                if market_type == "spread":
                    spread_away, spread_odds = _parse_spread_text(away_consensus)
                    game["spread_away"] = spread_away
                    game["spread_home"] = spread_away * -1 if spread_away is not None else None
                    game["spread_odds"] = spread_odds
                elif market_type == "total":
                    total_line, total_odds = _parse_total_text(away_consensus)
                    game["total_line"] = total_line
                    game["total_odds"] = total_odds
                else:
                    game["ml_away"] = _parse_moneyline_text(away_consensus)
                    game["ml_home"] = _parse_moneyline_text(home_consensus)

                if _market_has_values([game], market_type):
                    parsed_games.append(game)
            except Exception:
                continue

    return parsed_games


def _format_game_time_from_json(competition: dict) -> str | None:
    raw_start = _clean_text(competition.get("startDate") or competition.get("scheduledTime"))
    if not raw_start:
        return None

    try:
        game_dt = datetime.fromisoformat(raw_start.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None

    channels = ((competition.get("tvInfo") or {}).get("channels") or [])
    channel = _clean_text(channels[0].get("callLetters") if channels else "")
    time_text = game_dt.strftime("%I:%M%p").lstrip("0")
    game_time = f"{game_dt.strftime('%b')} {game_dt.day}, {time_text} UTC"
    if channel:
        game_time = f"{game_time} on {channel}"
    return game_time


def _parse_json_games(
    soup: BeautifulSoup,
    market_type: str,
    league: str,
) -> list[dict[str, object | None]]:
    parsed_games: list[dict[str, object | None]] = []

    try:
        next_data = soup.find("script", id="__NEXT_DATA__")
        if next_data is None or not next_data.string:
            return []

        payload = json.loads(next_data.string)
        initial_state = payload.get("props", {}).get("initialState")
        if not isinstance(initial_state, str):
            return []

        state = json.loads(initial_state)
        competition_odds = (
            (((state.get("oddsPageState") or {}).get("pageState") or {}).get("data") or {}).get("competitionOdds")
            or []
        )

        for competition in competition_odds:
            try:
                consensus = next(
                    (item for item in competition.get("odds", []) if item.get("sportsbookName") == "consensus"),
                    None,
                )
                odd = (consensus or {}).get("odd") or {}

                away_team = _clean_text((competition.get("awayTeam") or {}).get("nickName")) or _clean_text(
                    (competition.get("awayCompetitor") or {}).get("nickName")
                )
                home_team = _clean_text((competition.get("homeTeam") or {}).get("nickName")) or _clean_text(
                    (competition.get("homeCompetitor") or {}).get("nickName")
                )
                game_time = _format_game_time_from_json(competition)

                if not away_team or not home_team or not game_time:
                    continue

                game = _blank_game(league, away_team, home_team, game_time)

                if market_type == "spread":
                    point_spread = odd.get("pointSpread") or {}
                    spread_away = _coerce_float(point_spread.get("currentAwayHandicap"))
                    game["spread_away"] = spread_away
                    game["spread_home"] = spread_away * -1 if spread_away is not None else None
                    game["spread_odds"] = _coerce_int(point_spread.get("currentAwayOdds"))
                    if spread_away is not None and game["spread_odds"] is None:
                        game["spread_odds"] = -110
                elif market_type == "total":
                    over_under = odd.get("overUnder") or {}
                    game["total_line"] = _coerce_float(over_under.get("currentTotal"))
                    game["total_odds"] = _coerce_int(over_under.get("currentOverOdd"))
                    if game["total_line"] is not None and game["total_odds"] is None:
                        game["total_odds"] = -110
                else:
                    money_line = odd.get("moneyLine") or {}
                    game["ml_away"] = _coerce_int(money_line.get("currentAwayOdds"))
                    game["ml_home"] = _coerce_int(money_line.get("currentHomeOdds"))

                if _market_has_values([game], market_type):
                    parsed_games.append(game)
            except Exception:
                continue
    except Exception:
        return []

    return parsed_games


def scrape_sportsline(url: str, market_type: str, league: str) -> list[dict[str, object | None]]:
    response = None
    soup = None
    parsed_games: list[dict[str, object | None]] = []

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        LAST_TABLE_HTML[(league, market_type)] = _extract_table_preview(soup, response.text)

        parsed_games = _parse_table_games(soup, market_type, league)

        # Live SportsLine currently redirects MLB money-line to /mlb/odds/.
        # When the visible table is not usable, fall back to the same page's
        # serialized consensus odds data.
        if not _market_has_values(parsed_games, market_type):
            fallback_games = _parse_json_games(soup, market_type, league)
            if _market_has_values(fallback_games, market_type):
                parsed_games = fallback_games
    except requests.RequestException as exc:
        print(f"{league} {market_type}: request failed for {url}: {exc}")
        if response is not None:
            try:
                soup = BeautifulSoup(response.text, "html.parser")
                LAST_TABLE_HTML[(league, market_type)] = _extract_table_preview(soup, response.text)
            except Exception:
                LAST_TABLE_HTML[(league, market_type)] = response.text[:800]
    except Exception as exc:
        print(f"{league} {market_type}: unexpected parsing error: {exc}")
        if response is not None:
            try:
                soup = BeautifulSoup(response.text, "html.parser")
                LAST_TABLE_HTML[(league, market_type)] = _extract_table_preview(soup, response.text)
            except Exception:
                LAST_TABLE_HTML[(league, market_type)] = response.text[:800]
    finally:
        time.sleep(random.uniform(2, 3))

    return parsed_games


def fetch_all_odds(league: str) -> list[dict[str, object | None]]:
    league = league.upper()
    market_urls = SPORTSLINE_URLS[league]
    LAST_MARKET_ROWS[league] = {}
    merged_games: list[dict[str, object | None]] = []

    for market_type in ("spread", "total", "ml"):
        market_rows = scrape_sportsline(market_urls[market_type], market_type, league)
        LAST_MARKET_ROWS[league][market_type] = market_rows
        print(f"{league} {market_type}: {len(market_rows)} rows")

        for row in market_rows:
            game = _find_existing_game(merged_games, row.get("away_team"), row.get("home_team"))
            if game is None:
                game = _blank_game(
                    league,
                    row.get("away_team"),
                    row.get("home_team"),
                    row.get("game_time"),
                )
                merged_games.append(game)

            if not game.get("game_time") and row.get("game_time"):
                game["game_time"] = row["game_time"]

            for field in (
                "spread_away",
                "spread_home",
                "spread_odds",
                "total_line",
                "total_odds",
                "ml_away",
                "ml_home",
            ):
                if row.get(field) is not None:
                    game[field] = row[field]

    print(f"{league} merged: {len(merged_games)} games")
    return merged_games


def _schedule_date(target_date: date) -> str:
    return target_date.strftime("%Y-%m-%d")


@lru_cache(maxsize=16)
def get_mlb_schedule_games_for_date(slate_date: date) -> list[tuple[str, str]]:
    params = {
        "sportId": 1,
        "startDate": _schedule_date(slate_date),
        "endDate": _schedule_date(slate_date),
    }
    try:
        response = requests.get(MLB_SCHEDULE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        print(f"[sportsline_odds] Failed to load MLB schedule for slate_date={slate_date}: {exc}")
        return []
    except ValueError as exc:
        print(f"[sportsline_odds] Failed to decode MLB schedule for slate_date={slate_date}: {exc}")
        return []

    scheduled_games: list[tuple[str, str]] = []
    for slate in payload.get("dates", []):
        for game in slate.get("games", []):
            if str(game.get("gameType", "")).upper() != "R":
                continue
            away_team = _clean_text((((game.get("teams") or {}).get("away") or {}).get("team") or {}).get("name"))
            home_team = _clean_text((((game.get("teams") or {}).get("home") or {}).get("team") or {}).get("name"))
            if away_team and home_team:
                scheduled_games.append((away_team, home_team))
    return scheduled_games


def _filter_mlb_games_for_date(
    games: list[dict[str, object | None]],
    slate_date: date,
) -> list[dict[str, object | None]]:
    scheduled_games = get_mlb_schedule_games_for_date(slate_date)
    if not scheduled_games:
        print(
            f"[sportsline_odds] No official MLB schedule rows found for slate_date={slate_date}; "
            f"returning {len(games)} scraped games without filtering."
        )
        return games

    filtered_games = [
        row
        for row in games
        if any(
            _team_matches(row.get("away_team"), away_team)
            and _team_matches(row.get("home_team"), home_team)
            for away_team, home_team in scheduled_games
        )
    ]
    print(
        f"[sportsline_odds] Schedule filter for slate_date={slate_date}: "
        f"kept={len(filtered_games)} scraped_games={len(games)} official_games={len(scheduled_games)}"
    )
    return filtered_games


def _normalize_mlb_odds_rows(
    merged: list[dict[str, object | None]],
) -> list[dict[str, object | None]]:
    out: list[dict[str, object | None]] = []
    for row in merged or []:
        over_odds = row.get("total_odds")
        # SportsLine only supplies the over odds; mirror it for under estimate
        if over_odds is not None:
            under_odds = -over_odds if over_odds != 0 else -110
        else:
            under_odds = None

        out.append(
            {
                "away_team": row.get("away_team"),
                "home_team": row.get("home_team"),
                "game_time": row.get("game_time"),
                "ml_away": row.get("ml_away"),
                "ml_home": row.get("ml_home"),
                "total_line": row.get("total_line"),
                "total_over_odds": over_odds,
                "total_under_odds": under_odds,
                "spread_away": row.get("spread_away"),
                "spread_home": row.get("spread_home"),
                "spread_odds": row.get("spread_odds"),
            }
        )
    return out


def _get_db_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "pickledger.db",
        here / "pickledger.db",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def save_odds_to_db(odds_list: list[dict[str, object | None]], league: str) -> None:
    db_path = _get_db_path()
    fetched_at = datetime.now(timezone.utc).isoformat()

    if not odds_list:
        print(f"Saved 0 {league} rows")
        return

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO cbs_odds (
                    league,
                    fetched_at,
                    game_time,
                    away_team,
                    home_team,
                    spread_home,
                    spread_away,
                    spread_odds,
                    total_line,
                    total_odds,
                    ml_home,
                    ml_away
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        league,
                        fetched_at,
                        row.get("game_time"),
                        row.get("away_team"),
                        row.get("home_team"),
                        row.get("spread_home"),
                        row.get("spread_away"),
                        row.get("spread_odds"),
                        row.get("total_line"),
                        row.get("total_odds"),
                        row.get("ml_home"),
                        row.get("ml_away"),
                    )
                    for row in odds_list
                ],
            )
            conn.commit()
        print(f"Saved {len(odds_list)} {league} rows")
    except sqlite3.Error as exc:
        print(f"{league}: database write failed: {exc}")


def _validate_market_rows(league: str) -> bool:
    market_rows = LAST_MARKET_ROWS.get(league, {})

    for market_type, field_name in REQUIRED_FIELDS.items():
        rows = market_rows.get(market_type, [])
        if not rows or all(row.get(field_name) is None for row in rows):
            print(f"{league} {market_type} raw HTML (first 800 chars):")
            print(LAST_TABLE_HTML.get((league, market_type), ""))
            return False

    return True


def _print_recent_rows() -> None:
    db_path = _get_db_path()

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, league, fetched_at, game_time, away_team, home_team,
                       spread_home, spread_away, spread_odds,
                       total_line, total_odds, ml_home, ml_away
                FROM cbs_odds
                ORDER BY fetched_at DESC
                LIMIT 5
                """
            )
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
    except sqlite3.Error as exc:
        print(f"Failed to read cbs_odds: {exc}")
        return

    print("Columns:", ", ".join(columns))
    for row in rows:
        print(row)


def fetch_mlb_market_odds_for_date(slate_date: date) -> dict[tuple[str, str], dict]:
    """Return MLB market odds keyed by (away_last_word, home_last_word).

    Each value is a dict with keys: ml_away, ml_home, total_line, spread_away.
    Safe by construction: any scrape failure yields an empty dict rather than
    raising so callers can degrade gracefully when SportsLine is unreachable.
    """
    print(f"[sportsline_odds] Fetching MLB market odds for slate_date={slate_date}")
    try:
        merged = fetch_all_odds("MLB")
    except Exception:
        return {}

    result: dict[tuple[str, str], dict] = {}
    for row in _filter_mlb_games_for_date(merged or [], slate_date):
        away_key = _last_word(row.get("away_team"))
        home_key = _last_word(row.get("home_team"))
        if not away_key or not home_key:
            continue
        key = (away_key, home_key)
        result[key] = {
            "ml_away": row.get("ml_away"),
            "ml_home": row.get("ml_home"),
            "total_line": row.get("total_line"),
            "spread_away": row.get("spread_away"),
        }
    return result


def fetch_mlb_market_odds() -> dict[tuple[str, str], dict]:
    return fetch_mlb_market_odds_for_date(get_mlb_slate_date())


def get_mlb_odds_for_date(slate_date: date) -> list[dict[str, object | None]]:
    """Return MLB odds for *slate_date* in a flat list with uniform field names.

    Each dict contains:
        away_team, home_team, game_time,
        ml_away, ml_home,            # American odds ints (or None)
        total_line,                  # e.g. 8.5 (or None)
        total_over_odds,             # American odds for the over
        total_under_odds,            # Estimated mirror of over odds
        spread_away, spread_home, spread_odds
    """
    print(f"[sportsline_odds] Fetching MLB odds for slate_date={slate_date}")
    try:
        merged = fetch_all_odds("MLB")
    except Exception:
        return []

    filtered_games = _filter_mlb_games_for_date(merged or [], slate_date)
    return _normalize_mlb_odds_rows(filtered_games)


def get_today_mlb_odds() -> list[dict[str, object | None]]:
    return get_mlb_odds_for_date(get_mlb_slate_date())


if __name__ == "__main__":
    mlb_odds = fetch_all_odds("MLB")
    if not _validate_market_rows("MLB"):
        raise SystemExit(1)
    save_odds_to_db(mlb_odds, "MLB")
    time.sleep(3)

    _print_recent_rows()
