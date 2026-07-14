"""
wnba_stats.py

Scrape and cache WNBA team stats profiles from Basketball Reference, with
optional BallDontLie season and rolling-game support.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import time

import requests
from bs4 import BeautifulSoup, Comment

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from config import BDL_API_KEY
except ImportError:
    BDL_API_KEY = None

if not BDL_API_KEY:
    BDL_API_KEY = os.getenv("BDL_API_KEY", "")

try:
    from .wnba_teams import WNBA_TEAM_MAP
except ImportError:
    from wnba_teams import WNBA_TEAM_MAP


BBREF_RATINGS_URL = "https://www.basketball-reference.com/wnba/years/{season}_ratings.html"
BBREF_SEASON_URL = "https://www.basketball-reference.com/wnba/years/{season}.html"
BBREF_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

BDL_TEAM_SEASON_STATS_URL = "https://api.balldontlie.io/wnba/v1/team_season_stats"
BDL_TEAM_STATS_URL = "https://api.balldontlie.io/wnba/v1/team_stats"

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wnba")
)
os.makedirs(DATA_DIR, exist_ok=True)
CACHE_TTL_SECONDS = 6 * 60 * 60

BBREF_NAME_MAP = {
    "Indiana Fever": "IND",
    "Las Vegas Aces": "LV",
    "Minnesota Lynx": "MIN",
    "New York Liberty": "NY",
    "Seattle Storm": "SEA",
    "Connecticut Sun": "CON",
    "Chicago Sky": "CHI",
    "Atlanta Dream": "ATL",
    "Washington Mystics": "WAS",
    "Phoenix Mercury": "PHX",
    "Dallas Wings": "DAL",
    "Golden State Valkyries": "GSV",
    "Los Angeles Sparks": "LA",
    "Portland Fire": "POR",
    "Toronto Tempo": "TOR",
}

BDL_ABBR_ALIASES = {
    "LVA": "LV",
    "CONN": "CON",
    "NYL": "NY",
    "WSH": "WAS",
    "PHO": "PHX",
    "GS": "GSV",
}

PROFILE_FIELDS = [
    "team_abbr",
    "ORtg",
    "DRtg",
    "NRtg",
    "Pace",
    "W",
    "L",
    "MOV",
    "eFG_pct",
    "TOV_pct",
    "ORB_pct",
    "FTR",
    "opp_eFG",
    "opp_TOV",
    "DRB_pct",
    "opp_FTR",
    "pts_pg",
    "opp_pts_pg",
    "rolling_pts",
    "rolling_fga",
    "rolling_fg3a",
    "rolling_fta",
    "rolling_orb",
    "rolling_drb",
    "rolling_tov",
    "rolling_opp_pts",
    "rolling_opp_fga",
    "rolling_opp_orb",
    "rolling_opp_tov",
    "rolling_games_used",
    "rolling_low_sample",
]


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _safe_div(numerator, denominator) -> float | None:
    numerator = _safe_float(numerator)
    denominator = _safe_float(denominator)
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def _normalize_team_abbr(raw: str) -> str | None:
    if not raw:
        return None
    return BDL_ABBR_ALIASES.get(raw.strip().upper(), raw.strip().upper())


def _find_table(soup: BeautifulSoup, table_ids: list[str]):
    for table_id in table_ids:
        table = soup.find("table", id=table_id)
        if table is not None:
            return table

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if not any(table_id in comment for table_id in table_ids):
            continue
        comment_soup = BeautifulSoup(comment, "html.parser")
        for table_id in table_ids:
            table = comment_soup.find("table", id=table_id)
            if table is not None:
                return table

    return None


def _extract_team_name_from_row(row) -> str:
    team_cell = row.find("td", {"data-stat": "team_name"}) or row.find("td", {"data-stat": "team"})
    if team_cell is None:
        return ""
    link = team_cell.find("a")
    if link is not None:
        return link.get_text(strip=True)
    return team_cell.get_text(strip=True)


def _extract_stat_text(row, *data_stats: str) -> str | None:
    for data_stat in data_stats:
        cell = row.find(["td", "th"], {"data-stat": data_stat})
        if cell is not None:
            return cell.get_text(strip=True)
    return None


def _extract_date(row: dict) -> str:
    if row.get("date"):
        return str(row["date"])
    game = row.get("game")
    if isinstance(game, dict) and game.get("date"):
        return str(game["date"])
    return ""


def _pick_opponent_value(row: dict, *keys: str):
    candidates = []
    for key in keys:
        candidates.extend(
            [
                f"opp_{key}",
                f"opponent_{key}",
                f"opp_{key}_total",
            ]
        )

    for candidate in candidates:
        if candidate in row:
            return row[candidate]

    for container_name in ("opponent", "opp"):
        container = row.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in keys:
            if key in container:
                return container[key]

    return None


def _cache_path(season: int) -> str:
    return os.path.join(DATA_DIR, f"wnba_team_stats_{season}.json")


def _load_cached_profiles(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, ValueError):
        return {}

    profiles = payload.get("profiles")
    if isinstance(profiles, dict):
        return profiles
    return {}


def _cache_is_fresh(path: str) -> bool:
    if not os.path.exists(path):
        return False

    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
    except (OSError, ValueError):
        return False

    last_updated = payload.get("last_updated")
    if not last_updated:
        return False

    try:
        updated_at = datetime.datetime.fromisoformat(last_updated)
    except ValueError:
        return False

    age_seconds = (datetime.datetime.now(updated_at.tzinfo) - updated_at).total_seconds()
    return 0 <= age_seconds <= CACHE_TTL_SECONDS


def _write_cache(path: str, profiles: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "last_updated": datetime.datetime.now().isoformat(),
        "profiles": profiles,
    }
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def _parse_ratings_rows(rows) -> dict:
    ratings = {}
    for row in rows:
        team_name = _extract_team_name_from_row(row).replace("*", "").strip()
        team_abbr = BBREF_NAME_MAP.get(team_name)
        if team_abbr is None:
            continue
        ratings[team_abbr] = {
            "ORtg": _safe_float(_extract_stat_text(row, "off_rtg", "ORtg")),
            "DRtg": _safe_float(_extract_stat_text(row, "def_rtg", "DRtg")),
            "NRtg": _safe_float(_extract_stat_text(row, "net_rtg", "NRtg")),
            "Pace": _safe_float(_extract_stat_text(row, "pace", "Pace")),
            "W":    _safe_int(_extract_stat_text(row, "wins", "W")),
            "L":    _safe_int(_extract_stat_text(row, "losses", "L")),
            "MOV":  _safe_float(_extract_stat_text(row, "mov", "MOV")),
        }
    return ratings


def scrape_bball_ref_ratings(season: int = 2026) -> dict:
    """Pull team ratings from BBRef, preferring the dedicated ratings page.

    Recent WNBA seasons are sometimes served only through the main season
    page's ``advanced-team`` table — the dedicated ``{season}_ratings.html``
    URL 404s for a while after the season opens. We fall back to that table
    so early-season runs still get NRtg / ORtg / DRtg / Pace.
    """
    try:
        time.sleep(2)
        response = requests.get(
            BBREF_RATINGS_URL.format(season=season),
            headers={"User-Agent": BBREF_USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = _find_table(soup, ["ratings"])
        if table is None:
            raise ValueError("ratings table not found")
        rows = table.find_all("tr", attrs={"data-row-index": True})
        ratings = _parse_ratings_rows(rows)
        if ratings:
            return ratings
        raise ValueError("ratings table empty")
    except Exception as e:
        print(f"[WNBA] BBRef ratings page unavailable ({e}); falling back to advanced-team.")

    try:
        time.sleep(2)
        response = requests.get(
            BBREF_SEASON_URL.format(season=season),
            headers={"User-Agent": BBREF_USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        table = _find_table(soup, ["advanced-team", "advanced_team"])
        if table is None:
            raise ValueError("advanced-team table not found")
        tbody = table.find("tbody") or table
        rows = [r for r in tbody.find_all("tr") if "thead" not in (r.get("class") or [])]
        return _parse_ratings_rows(rows)
    except Exception as e:
        print(f"[WNBA] BBRef advanced-team fallback failed: {e}")
        return {}


def scrape_bball_ref_four_factors(season: int = 2026) -> dict:
    try:
        time.sleep(2)
        response = requests.get(
            BBREF_SEASON_URL.format(season=season),
            headers={"User-Agent": BBREF_USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        team_table = _find_table(soup, ["team_stats", "per_game_team", "per_game-team"])
        opp_table = _find_table(soup, ["opp_stats", "per_game_opponent", "per_game-opponent"])
        if team_table is None or opp_table is None:
            raise ValueError("team or opponent stats table not found")

        def parse_table(table) -> dict:
            # The opponent per-game table prefixes every stat column with
            # "opp_" (opp_fg, opp_fga, ...), so each field checks both the
            # plain and prefixed data-stat names.
            parsed = {}
            tbody = table.find("tbody") or table
            for row in tbody.find_all("tr"):
                if "thead" in (row.get("class") or []):
                    continue

                team_name = _extract_team_name_from_row(row).replace("*", "").strip()
                team_abbr = BBREF_NAME_MAP.get(team_name)
                if team_abbr is None:
                    continue

                parsed[team_abbr] = {
                    "fg": _safe_float(_extract_stat_text(row, "fg", "opp_fg")),
                    "fga": _safe_float(_extract_stat_text(row, "fga", "opp_fga")),
                    "fg3": _safe_float(_extract_stat_text(row, "fg3", "opp_fg3")),
                    "fg3a": _safe_float(_extract_stat_text(row, "fg3a", "opp_fg3a")),
                    "ft": _safe_float(_extract_stat_text(row, "ft", "opp_ft")),
                    "fta": _safe_float(_extract_stat_text(row, "fta", "opp_fta")),
                    "orb": _safe_float(_extract_stat_text(row, "orb", "opp_orb")),
                    "drb": _safe_float(_extract_stat_text(row, "drb", "opp_drb")),
                    "tov": _safe_float(_extract_stat_text(row, "tov", "opp_tov")),
                    "pts": _safe_float(_extract_stat_text(row, "pts", "opp_pts")),
                }

            return parsed

        team_stats = parse_table(team_table)
        opp_stats = parse_table(opp_table)
        results = {}

        for team_abbr, team_row in team_stats.items():
            opp_row = opp_stats.get(team_abbr, {})

            fg = team_row.get("fg")
            fga = team_row.get("fga")
            fg3 = team_row.get("fg3")
            fta = team_row.get("fta")
            orb = team_row.get("orb")
            drb = team_row.get("drb")
            tov = team_row.get("tov")
            pts = team_row.get("pts")

            opp_fg = opp_row.get("fg")
            opp_fga = opp_row.get("fga")
            opp_fg3 = opp_row.get("fg3")
            opp_fta = opp_row.get("fta")
            opp_orb = opp_row.get("orb")
            opp_drb = opp_row.get("drb")
            opp_tov = opp_row.get("tov")
            opp_pts = opp_row.get("pts")

            results[team_abbr] = {
                "eFG_pct": _safe_div((fg + 0.5 * fg3) if fg is not None and fg3 is not None else None, fga),
                "TOV_pct": _safe_div(tov, (fga + 0.44 * fta + tov) if fga is not None and fta is not None and tov is not None else None),
                "ORB_pct": _safe_div(orb, (orb + opp_drb) if orb is not None and opp_drb is not None else None),
                "FTR": _safe_div(fta, fga),
                "opp_eFG": _safe_div((opp_fg + 0.5 * opp_fg3) if opp_fg is not None and opp_fg3 is not None else None, opp_fga),
                "opp_TOV": _safe_div(opp_tov, (opp_fga + 0.44 * opp_fta + opp_tov) if opp_fga is not None and opp_fta is not None and opp_tov is not None else None),
                "DRB_pct": _safe_div(drb, (drb + opp_orb) if drb is not None and opp_orb is not None else None),
                "opp_FTR": _safe_div(opp_fta, opp_fga),
                "pts_pg": pts,
                "opp_pts_pg": opp_pts,
            }

        return results
    except Exception as e:
        print(f"[WNBA] BBRef four factors scrape failed: {e}")
        return {}


def fetch_bdl_team_season_stats(season: int = 2026) -> dict:
    if not BDL_API_KEY:
        print("[WNBA] BDL key not set — skipping season stats.")
        return {}

    try:
        season_stats = {}
        cursor = None

        while True:
            params = {
                "season": season,
                "season_type": 2,
                "per_page": 100,
            }
            if cursor is not None:
                params["cursor"] = cursor

            response = requests.get(
                BDL_TEAM_SEASON_STATS_URL,
                headers={"Authorization": BDL_API_KEY},
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()

            for row in payload.get("data", []):
                team = row.get("team") or {}
                team_abbr = _normalize_team_abbr(team.get("abbreviation", ""))
                if team_abbr is None and team.get("full_name"):
                    team_abbr = BBREF_NAME_MAP.get(team["full_name"])
                if team_abbr is None:
                    continue

                season_stats[team_abbr] = {
                    key: value
                    for key, value in row.items()
                    if key != "team"
                }

            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor:
                break

        return season_stats
    except Exception:
        return {}


def fetch_bdl_team_game_logs(team_bdl_id: int, season: int = 2026) -> list[dict]:
    if not BDL_API_KEY or team_bdl_id is None:
        return []

    try:
        game_logs = []
        cursor = None

        while True:
            params = {
                "team_ids[]": team_bdl_id,
                "seasons[]": season,
                "per_page": 100,
            }
            if cursor is not None:
                params["cursor"] = cursor

            response = requests.get(
                BDL_TEAM_STATS_URL,
                headers={"Authorization": BDL_API_KEY},
                params=params,
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()

            game_logs.extend(payload.get("data", []))

            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor:
                break

        return sorted(game_logs, key=_extract_date)
    except Exception:
        return []


def compute_rolling_stats(game_logs: list[dict], n: int = 10) -> dict:
    if not game_logs:
        return {}

    n = max(int(n), 1)
    window = game_logs[-n:]

    fields = {
        "pts": [],
        "fga": [],
        "fg3a": [],
        "fta": [],
        "orb": [],
        "drb": [],
        "tov": [],
        "opp_pts": [],
        "opp_fga": [],
        "opp_orb": [],
        "opp_tov": [],
    }

    for row in window:
        value_map = {
            "pts": row.get("pts"),
            "fga": row.get("fga"),
            "fg3a": row.get("fg3a"),
            "fta": row.get("fta"),
            "orb": row.get("oreb", row.get("orb")),
            "drb": row.get("dreb", row.get("drb")),
            "tov": row.get("turnover", row.get("tov")),
            "opp_pts": _pick_opponent_value(row, "pts"),
            "opp_fga": _pick_opponent_value(row, "fga"),
            "opp_orb": _pick_opponent_value(row, "oreb", "orb"),
            "opp_tov": _pick_opponent_value(row, "turnover", "tov"),
        }

        for key, raw_value in value_map.items():
            value = _safe_float(raw_value)
            if value is not None:
                fields[key].append(value)

    averages = {}
    for key, values in fields.items():
        averages[key] = (sum(values) / len(values)) if values else None

    averages["games_used"] = len(window)
    averages["low_sample"] = len(window) < n
    return averages


def build_team_stats_profile(team_abbr, ratings, four_factors, bdl_season, rolling) -> dict:
    profile = {field: None for field in PROFILE_FIELDS}
    profile["team_abbr"] = team_abbr

    base = (ratings or {}).get(team_abbr, {})
    if isinstance(base, dict):
        profile.update(base)

    four_factor_row = (four_factors or {}).get(team_abbr, {})
    if isinstance(four_factor_row, dict):
        profile.update(four_factor_row)

    bdl_row = (bdl_season or {}).get(team_abbr, {})
    if isinstance(bdl_row, dict):
        profile.update(bdl_row)

    if isinstance(rolling, dict):
        for key, value in rolling.items():
            profile[f"rolling_{key}"] = value

    profile["team_abbr"] = team_abbr
    return profile


def get_all_team_stats(season: int = 2026, force_refresh: bool = False) -> dict:
    path = _cache_path(season)
    if not force_refresh and _cache_is_fresh(path):
        return _load_cached_profiles(path)

    ratings = scrape_bball_ref_ratings(season=season)
    four_factors = scrape_bball_ref_four_factors(season=season)
    bdl_season = fetch_bdl_team_season_stats(season=season)

    profiles = {}
    for team_abbr in WNBA_TEAM_MAP:
        profiles[team_abbr] = build_team_stats_profile(
            team_abbr=team_abbr,
            ratings=ratings,
            four_factors=four_factors,
            bdl_season=bdl_season,
            rolling={},
        )

    _write_cache(path, profiles)
    return profiles


def get_team_stats(team_abbr: str, season: int = 2026) -> dict:
    if not team_abbr:
        return {}
    return get_all_team_stats(season=season).get(team_abbr.strip().upper(), {})


def get_rolling_stats(team_abbr: str, n: int = 10, season: int = 2026) -> dict:
    if not team_abbr:
        return {}

    team = WNBA_TEAM_MAP.get(team_abbr.strip().upper())
    if not team:
        return {}

    game_logs = fetch_bdl_team_game_logs(team.get("bdl_id"), season=season)
    return compute_rolling_stats(game_logs, n=n)


def _opponent_bdl_id_from_log(row: dict) -> int | None:
    """Best-effort opponent team_id pull from a BDL game log row."""
    if not isinstance(row, dict):
        return None
    for key in ("opp_team_id", "opponent_team_id"):
        if key in row:
            try:
                return int(row[key])
            except (TypeError, ValueError):
                pass
    for container_name in ("opponent", "opp"):
        container = row.get(container_name)
        if isinstance(container, dict):
            for key in ("id", "team_id", "bdl_id"):
                if key in container:
                    try:
                        return int(container[key])
                    except (TypeError, ValueError):
                        pass
    game = row.get("game")
    if isinstance(game, dict):
        try:
            home_id = int((game.get("home_team") or {}).get("id"))
            away_id = int((game.get("visitor_team") or {}).get("id"))
        except (TypeError, ValueError):
            return None
        my_id = row.get("team", {}).get("id") if isinstance(row.get("team"), dict) else None
        if my_id is None:
            return None
        try:
            my_id = int(my_id)
        except (TypeError, ValueError):
            return None
        return away_id if my_id == home_id else home_id
    return None


def _location_from_log(row: dict) -> str | None:
    """Return 'home', 'away', or None for a BDL game log row."""
    if not isinstance(row, dict):
        return None
    raw = row.get("location") or row.get("home_away")
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in {"home", "away"}:
            return text
    game = row.get("game")
    if isinstance(game, dict) and isinstance(row.get("team"), dict):
        try:
            my_id = int(row["team"].get("id"))
            home_id = int((game.get("home_team") or {}).get("id"))
        except (TypeError, ValueError):
            return None
        return "home" if my_id == home_id else "away"
    return None


def get_h2h_history(
    home_abbr: str,
    away_abbr: str,
    season: int = 2026,
    as_of_date: str | None = None,
) -> list[dict]:
    """Return prior-this-season H2H games from the home team's perspective.

    Each entry: {date, is_home_for_target (bool), margin_for_target (int)}.
    Returns [] if game logs aren't available or the matchup hasn't happened
    yet — callers must treat both as 'no signal'.
    """
    if not home_abbr or not away_abbr:
        return []
    home_team = WNBA_TEAM_MAP.get(home_abbr.strip().upper())
    away_team = WNBA_TEAM_MAP.get(away_abbr.strip().upper())
    if not home_team or not away_team:
        return []
    away_id = away_team.get("bdl_id")
    if away_id is None:
        return []

    logs = fetch_bdl_team_game_logs(home_team.get("bdl_id"), season=season)
    if not logs:
        return []

    cutoff = (as_of_date or "").strip()
    games: list[dict] = []
    for row in logs:
        date = _extract_date(row)
        if cutoff and date and date >= cutoff:
            continue
        opp_id = _opponent_bdl_id_from_log(row)
        if opp_id != int(away_id):
            continue
        try:
            pts = float(row.get("pts"))
            opp_pts = float(_pick_opponent_value(row, "pts"))
        except (TypeError, ValueError):
            continue
        location = _location_from_log(row)
        is_home = location == "home"
        games.append({
            "date": date,
            "is_home_for_target": is_home,
            "margin_for_target": pts - opp_pts,
        })
    games.sort(key=lambda g: g.get("date") or "")
    return games


def _fmt(value: float | int | None) -> str:
    if value is None:
        return " n/a "
    return f"{float(value):5.1f}"


if __name__ == "__main__":
    all_team_stats = get_all_team_stats(force_refresh=True)
    ranked = sorted(
        all_team_stats.items(),
        key=lambda item: item[1].get("NRtg") if item[1].get("NRtg") is not None else float("-inf"),
        reverse=True,
    )

    print("Rank | Team | NRtg  | ORtg  | DRtg  | Pace  | W-L")
    for rank, (team_abbr, profile) in enumerate(ranked, start=1):
        wins = profile.get("W")
        losses = profile.get("L")
        wl = f"{wins}-{losses}" if wins is not None and losses is not None else "n/a"
        print(
            f"{rank:>4} | {team_abbr:<4} | {_fmt(profile.get('NRtg'))} | {_fmt(profile.get('ORtg'))} | "
            f"{_fmt(profile.get('DRtg'))} | {_fmt(profile.get('Pace'))} | {wl}"
        )

    for team_abbr, profile in all_team_stats.items():
        pace = profile.get("Pace")
        if pace is not None and not 55 <= pace <= 85:
            print(f"WARN: {team_abbr} Pace={pace} — check 40-min scaling")

    assert len(all_team_stats) == len(WNBA_TEAM_MAP)
    print(f"PASS: All {len(all_team_stats)} teams have stats profiles.")
