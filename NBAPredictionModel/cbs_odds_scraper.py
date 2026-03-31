import random
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag


CBS_NBA_URL = "https://www.cbssports.com/nba/odds/"
CBS_MLB_URL = "https://www.cbssports.com/mlb/odds/"
SL_MLB_URL = "https://www.sportsline.com/mlb/odds/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.cbssports.com/",
}

LAST_HTML_PREVIEW: dict[str, str] = {}


def _sleep_before_return() -> None:
    time.sleep(random.uniform(2, 4))


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _normalize_text(value: str | None) -> str:
    return _clean_text(value).lower()


def _coerce_float(value: str | None) -> float | None:
    if value is None:
        return None

    cleaned = _clean_text(value).replace("−", "-").rstrip(",")
    if not cleaned:
        return None
    if cleaned.upper() in {"PK", "PICK", "PICKEM", "EVEN"}:
        return 0.0

    try:
        return float(cleaned)
    except ValueError:
        return None


def _coerce_int(value: str | None) -> int | None:
    parsed = _coerce_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _first_non_empty_text(tag: Tag | None) -> str | None:
    if tag is None:
        return None
    text = _clean_text(tag.get_text(" ", strip=True))
    return text or None


def _extract_html_preview(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        preview_tag = soup.find("table") or soup.find("section") or soup.body
        if preview_tag is not None:
            return str(preview_tag)[:1000]
    except Exception:
        pass
    return html[:1000]


def _find_game_tables(soup: BeautifulSoup) -> list[Tag]:
    tables = soup.find_all('table', class_=lambda c: c and 'OddsBlock-game' in c)
    return list(tables)


def _find_header_row(rows: list[Tag]) -> tuple[int | None, int]:
    for index, row in enumerate(rows):
        try:
            cells = row.find_all(["th", "td"])
            texts = [_normalize_text(cell.get_text(" ", strip=True)) for cell in cells]
            combined = " ".join(texts)
            if (
                "open" in combined
                and "spread" in combined
                and "total" in combined
                and ("moneyline" in combined or re.search(r"\bml\b", combined))
            ):
                for open_index, text in enumerate(texts):
                    if "open" in text:
                        return index, open_index
                return index, 2
        except Exception:
            continue

    return None, 2


def _extract_game_time(table: Tag, rows: list[Tag], header_index: int | None) -> str | None:
    try:
        time_th = table.find('th', class_='OddsBlock-time')
        spans = time_th.find_all('span') if time_th else []
        game_time = ' '.join(s.get_text(strip=True) for s in spans) or None
    except Exception:
        game_time = None

    if game_time:
        return game_time

    if header_index is not None:
        for row in reversed(rows[:header_index]):
            text = _clean_text(row.get_text(" ", strip=True))
            if text:
                return text

    caption_text = _first_non_empty_text(table.find("caption"))
    if caption_text:
        return caption_text

    for sibling in table.previous_siblings:
        if not isinstance(sibling, Tag):
            continue
        text = _clean_text(sibling.get_text(" ", strip=True))
        if text:
            return text

    return None


def _extract_team_name(cells: list[Tag], open_index: int) -> str | None:
    for cell in cells[: max(open_index, 1)]:
        text = _clean_text(cell.get_text(" ", strip=True))
        if text and any(char.isalpha() for char in text):
            return text

    if cells:
        fallback = _clean_text(cells[0].get_text(" ", strip=True))
        return fallback or None

    return None


def _extract_data_rows(rows: list[Tag], header_index: int | None, open_index: int) -> list[list[Tag]]:
    if header_index is None:
        return []

    data_rows: list[list[Tag]] = []
    for row in rows[header_index + 1 :]:
        try:
            cells = row.find_all(["th", "td"])
            if len(cells) <= open_index:
                continue

            combined = _normalize_text(row.get_text(" ", strip=True))
            if (
                "open" in combined
                and "spread" in combined
                and "total" in combined
                and ("moneyline" in combined or re.search(r"\bml\b", combined))
            ):
                continue

            if not any(_clean_text(cell.get_text(" ", strip=True)) for cell in cells):
                continue

            data_rows.append(cells)
            if len(data_rows) == 2:
                break
        except Exception:
            continue

    return data_rows


def _extract_last_odds_token(text: str) -> int | None:
    tokens = _clean_text(text).replace("−", "-").split()
    for token in reversed(tokens):
        if token.startswith(("+", "-")):
            parsed = _coerce_int(token)
            if parsed is not None:
                return parsed
    return None


def _parse_total_cell(text: str) -> tuple[float | None, int | None]:
    cleaned = _clean_text(text).replace("−", "-")
    if not cleaned:
        return None, None

    tokens = cleaned.split()
    first_token = tokens[0] if tokens else ""
    if first_token[:1].lower() in {"o", "u"}:
        first_token = first_token[1:]

    total_line = _coerce_float(first_token)
    if total_line is None:
        match = re.search(r"[ou]\s*([0-9]+(?:\.[0-9]+)?)", cleaned, re.IGNORECASE)
        if match:
            total_line = _coerce_float(match.group(1))

    total_odds = _extract_last_odds_token(cleaned)
    return total_line, total_odds


def _parse_spread_cell(text: str) -> tuple[float | None, int | None]:
    cleaned = _clean_text(text).replace("−", "-")
    if not cleaned:
        return None, None

    tokens = cleaned.split()
    spread_home = _coerce_float(tokens[0] if tokens else None)
    if spread_home is None:
        match = re.search(r"([+-]?\d+(?:\.\d+)?|PK|PICK|PICKEM|EVEN)", cleaned, re.IGNORECASE)
        if match:
            spread_home = _coerce_float(match.group(1))

    spread_odds = _extract_last_odds_token(cleaned)
    return spread_home, spread_odds


def _parse_game_table(table: Tag) -> dict[str, Any] | None:
    rows = table.find_all("tr")
    header_index, open_index = _find_header_row(rows)
    game_time = _extract_game_time(table, rows, header_index)
    data_rows = _extract_data_rows(rows, header_index, open_index)
    if len(data_rows) < 2:
        return None

    away_cells = data_rows[0]
    home_cells = data_rows[1]

    away_team = _extract_team_name(away_cells, open_index)
    home_team = _extract_team_name(home_cells, open_index)

    away_open_text = _clean_text(away_cells[open_index].get_text(" ", strip=True))
    home_open_text = _clean_text(home_cells[open_index].get_text(" ", strip=True))

    total_line, total_odds = _parse_total_cell(away_open_text)
    spread_home, spread_odds = _parse_spread_cell(home_open_text)
    spread_away = spread_home * -1 if spread_home is not None else None

    if not any(
        value is not None
        for value in (game_time, away_team, home_team, total_line, total_odds, spread_home, spread_odds)
    ):
        return None

    return {
        "game_time": game_time,
        "away_team": away_team,
        "home_team": home_team,
        "total_line": total_line,
        "total_odds": total_odds,
        "spread_home": spread_home,
        "spread_odds": spread_odds,
        "spread_away": spread_away,
        "ml_home": None,
        "ml_away": None,
    }


def _parse_games(html: str, league: str) -> list[dict[str, Any]]:
    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        print(f"{league}: failed to parse HTML: {exc}")
        return []

    tables = _find_game_tables(soup)
    if not tables:
        print(f"{league}: no CBS odds tables found in HTML.")
        return []

    odds_list: list[dict[str, Any]] = []
    seen_games: set[tuple[str | None, str | None, str | None]] = set()

    for table in tables:
        try:
            parsed = _parse_game_table(table)
        except Exception as exc:
            print(f"{league}: skipped malformed game table: {exc}")
            continue

        if not parsed:
            continue

        game_key = (
            parsed.get("game_time"),
            parsed.get("away_team"),
            parsed.get("home_team"),
        )
        if game_key in seen_games:
            continue

        odds_list.append(parsed)
        seen_games.add(game_key)

    if not odds_list:
        print(f"{league}: CBS tables were found, but no games were parsed.")

    return odds_list


def fetch_cbs_odds(url: str, league: str) -> list[dict[str, Any]]:
    response = None
    odds_list: list[dict[str, Any]] = []

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        LAST_HTML_PREVIEW[league] = _extract_html_preview(response.text)
        odds_list = _parse_games(response.text, league)
    except requests.RequestException as exc:
        print(f"{league}: request failed for {url}: {exc}")
        if response is not None:
            LAST_HTML_PREVIEW[league] = _extract_html_preview(response.text)
            print(f"{league}: status={response.status_code}, response preview={response.text[:500]}")
    except Exception as exc:
        print(f"{league}: unexpected parsing error: {exc}")
        if response is not None:
            LAST_HTML_PREVIEW[league] = _extract_html_preview(response.text)
            print(f"{league}: status={response.status_code}, response preview={response.text[:500]}")
    finally:
        _sleep_before_return()

    return odds_list


def fetch_sl_mlb_odds() -> list[dict[str, Any]]:
    response = None
    odds_list: list[dict[str, Any]] = []

    try:
        headers = dict(HEADERS)
        headers["Referer"] = "https://www.sportsline.com/"

        response = requests.get(SL_MLB_URL, headers=headers, timeout=15)
        response.raise_for_status()
        LAST_HTML_PREVIEW["MLB"] = response.text

        soup = BeautifulSoup(response.text, "html.parser")
        parsed_rows: list[list[Tag]] = []

        for row in soup.find_all("tr"):
            try:
                cells = row.find_all(["th", "td"])
                if not cells:
                    continue

                first_cell_text = _clean_text(cells[0].get_text(" ", strip=True))
                if not first_cell_text:
                    continue
                if first_cell_text == "Matchup":
                    continue
                if first_cell_text == "Advanced Insights loading...":
                    continue

                if len(cells) >= 3:
                    parsed_rows.append(cells)
                    continue

                if len(cells) == 1 and re.search(
                    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},",
                    first_cell_text,
                ):
                    parsed_rows.append(cells)
            except Exception:
                continue

        for index in range(0, len(parsed_rows) - 2, 3):
            try:
                away_cells = parsed_rows[index]
                home_cells = parsed_rows[index + 1]
                time_cells = parsed_rows[index + 2]

                if len(away_cells) <= 2 or len(home_cells) <= 2 or not time_cells:
                    continue

                away_team_text = _clean_text(away_cells[0].get_text(" ", strip=True))
                home_team_text = _clean_text(home_cells[0].get_text(" ", strip=True))
                game_time = _clean_text(time_cells[0].get_text(" ", strip=True))

                away_team = re.sub(r"\s+\d+-\d+$", "", away_team_text).strip()
                home_team = re.sub(r"\s+\d+-\d+$", "", home_team_text).strip()

                away_consensus = _clean_text(away_cells[2].get_text(" ", strip=True)).replace("−", "-")
                home_consensus = _clean_text(home_cells[2].get_text(" ", strip=True)).replace("−", "-")

                spread_match = re.match(r"^([+-]?\d+\.?\d*)", away_consensus)
                spread_away = float(spread_match.group(1)) if spread_match else None

                away_odds_match = re.search(r"([+-]\d+)\s*(?:Open|$)", away_consensus)
                spread_away_odds = int(away_odds_match.group(1)) if away_odds_match else None

                home_odds_match = re.search(r"([+-]\d+)\s*(?:Open|$)", home_consensus)
                spread_home_odds = int(home_odds_match.group(1)) if home_odds_match else None

                spread_home = spread_away * -1 if spread_away is not None else None

                odds_list.append(
                    {
                        "away_team": away_team,
                        "home_team": home_team,
                        "game_time": game_time,
                        "spread_away": spread_away,
                        "spread_away_odds": spread_away_odds,
                        "spread_home": spread_home,
                        "spread_home_odds": spread_home_odds,
                        "spread_odds": spread_home_odds,
                        "total_line": None,
                        "total_odds": None,
                        "ml_home": None,
                        "ml_away": None,
                    }
                )
            except Exception:
                continue
    except requests.RequestException as exc:
        print(f"MLB: request failed for {SL_MLB_URL}: {exc}")
        if response is not None:
            LAST_HTML_PREVIEW["MLB"] = response.text
            print(f"MLB: status={response.status_code}, response preview={response.text[:500]}")
    except Exception as exc:
        print(f"MLB: unexpected parsing error: {exc}")
        if response is not None:
            LAST_HTML_PREVIEW["MLB"] = response.text
            print(f"MLB: status={response.status_code}, response preview={response.text[:500]}")
    finally:
        time.sleep(random.uniform(2, 4))

    return odds_list


def save_odds_to_db(odds_list: list[dict[str, Any]], league: str) -> None:
    db_path = Path(__file__).resolve().parent.parent / "pickledger.db"
    fetched_at = datetime.now(timezone.utc).isoformat()

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cbs_odds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    league TEXT,
                    fetched_at TEXT,
                    game_time TEXT,
                    away_team TEXT,
                    home_team TEXT,
                    spread_home REAL,
                    spread_away REAL,
                    spread_odds INTEGER,
                    total_line REAL,
                    total_odds INTEGER,
                    ml_home INTEGER,
                    ml_away INTEGER
                )
                """
            )

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
    except sqlite3.Error as exc:
        print(f"{league}: database write failed: {exc}")


def _print_first_rows() -> None:
    db_path = Path(__file__).resolve().parent.parent / "pickledger.db"

    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, league, fetched_at, game_time, away_team, home_team,
                       spread_home, spread_away, spread_odds,
                       total_line, total_odds, ml_home, ml_away
                FROM cbs_odds
                LIMIT 3
                """
            )
            rows = cursor.fetchall()
    except sqlite3.Error as exc:
        print(f"Sanity check failed while reading cbs_odds: {exc}")
        return

    print("First 3 cbs_odds rows:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    nba_odds = fetch_cbs_odds(CBS_NBA_URL, "NBA")
    save_odds_to_db(nba_odds, "NBA")
    print(f"Saved {len(nba_odds)} NBA rows")
    if not nba_odds:
        print("NBA raw HTML preview:")
        print(LAST_HTML_PREVIEW.get("NBA", "")[:1000])

    time.sleep(3)

    # MLB - SportsLine
    print("Fetching MLB odds from SportsLine...")
    mlb_odds = fetch_sl_mlb_odds()
    save_odds_to_db(mlb_odds, "MLB")
    import time as _t; _t.sleep(3)
    print(f"Saved {len(mlb_odds)} MLB rows")
    if not mlb_odds:
        print("MLB raw HTML preview:")
        print(LAST_HTML_PREVIEW.get("MLB", "")[:1000])

    _print_first_rows()
