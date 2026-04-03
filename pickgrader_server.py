#!/usr/bin/env python3
"""
Local auto-grading service for PickLedger.

Runs a small HTTP server that accepts picks from the dashboard,
fetches completed game results from ESPN's public scoreboard endpoints,
and returns graded outcomes.

Usage:
  python3 pickgrader_server.py
Then click REFRESH in pickledger.html.
"""

from __future__ import annotations

import json
import math as _math
import os
import re
import sqlite3
import sqlite3 as _sqlite3, os as _os
import subprocess
import sys
import threading
import time
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.error import URLError, HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from ipl.ipl_model import run_ipl_model, format_ipl_output
    IPL_AVAILABLE = True
except Exception as e:
    IPL_AVAILABLE = False
    print(f"[IPL] Model not available: {e}")


def _sl_get_total(home, away, league='MLB'):
    """Get real Vegas total line and odds from cbs_odds (SportsLine data)."""
    candidates = ['pickledger.db', '../pickledger.db']
    db = next((p for p in candidates if _os.path.exists(p)), None)
    if not db:
        return None, None
    try:
        conn = _sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT total_line, total_odds FROM cbs_odds
            WHERE league=? AND total_line IS NOT NULL
            AND (home_team LIKE ? OR away_team LIKE ?)
            ORDER BY fetched_at DESC LIMIT 1
            """,
            (league, f'%{home.split()[-1]}%', f'%{away.split()[-1]}%'),
        )
        row = cur.fetchone()
        conn.close()
        return (float(row[0]), int(row[1])) if row else (None, None)
    except Exception:
        return None, None


def _sl_get_ml(home, away, league='MLB'):
    """Get real Vegas moneyline from cbs_odds (SportsLine data)."""
    candidates = ['pickledger.db', '../pickledger.db']
    db = next((p for p in candidates if _os.path.exists(p)), None)
    if not db:
        return None, None
    try:
        conn = _sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ml_home, ml_away FROM cbs_odds
            WHERE league=? AND ml_home IS NOT NULL
            AND (home_team LIKE ? OR away_team LIKE ?)
            ORDER BY fetched_at DESC LIMIT 1
            """,
            (league, f'%{home.split()[-1]}%', f'%{away.split()[-1]}%'),
        )
        row = cur.fetchone()
        conn.close()
        return (int(row[0]), int(row[1])) if row else (None, None)
    except Exception:
        return None, None


def _sl_get_spread(home, away, league='NBA'):
    """Get real Vegas spread from cbs_odds (SportsLine data).
    Returns (spread_home, spread_away, spread_odds) or (None, None, None)."""
    candidates = ['pickledger.db', '../pickledger.db']
    db = next((p for p in candidates if _os.path.exists(p)), None)
    if not db:
        return None, None, None
    try:
        conn = _sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT spread_home, spread_away, spread_odds FROM cbs_odds
            WHERE league=? AND spread_home IS NOT NULL
            AND (home_team LIKE ? OR away_team LIKE ?)
            ORDER BY fetched_at DESC LIMIT 1
            """,
            (league, f'%{home.split()[-1]}%', f'%{away.split()[-1]}%'),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None, None, None
        spread_home = float(row[0]) if row[0] is not None else None
        spread_away = float(row[1]) if row[1] is not None else None
        spread_odds = int(row[2]) if row[2] is not None else -110
        return spread_home, spread_away, spread_odds
    except Exception:
        return None, None, None


def _ou_probability(model_total: float, vegas_line: float, rmse: float) -> float:
    """
    Derive P(under) or P(over) from model point estimate using normal CDF.
    Uses the model's historical RMSE as the prediction std dev.

    P(actual < vegas_line | model_total, rmse) = Φ((vegas_line - model_total) / rmse)

    Returns probability for the DIRECTION the model is betting:
      - If model_total < vegas_line → returns P(under)  = Φ((line - model) / rmse)
      - If model_total > vegas_line → returns P(over)   = 1 - Φ((line - model) / rmse)
    """

    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + _math.erf(x / _math.sqrt(2.0)))

    if rmse <= 0:
        return 0.5

    z = (vegas_line - model_total) / rmse
    p_under = _norm_cdf(z)

    if model_total < vegas_line:
        return round(p_under, 4)
    return round(1.0 - p_under, 4)


# Model-specific RMSE constants (from backtest metadata)
_MLB_TOTALS_RMSE = 4.329383382244959
_NBA_TOTALS_RMSE = 12.5

def _load_local_env() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for filename in (".env", ".env.local"):
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError:
            continue


_load_local_env()

HOST = os.environ.get("HOST", "0.0.0.0")
try:
    PORT = int(os.environ.get("PORT", "8765"))
except ValueError:
    PORT = 8765

IS_RENDER_RUNTIME = os.environ.get("RENDER", "").strip().lower() == "true"
# Default to enabled so Render backend accepts scrape requests unless explicitly disabled.
_scores24_env = os.environ.get("ENABLE_SCORES24_REMOTE", "true").strip().lower()
ENABLE_SCORES24_REMOTE = _scores24_env not in {"0", "false", "no", "off"}
_sportytrader_env = os.environ.get("ENABLE_SPORTYTRADER_REMOTE", "true").strip().lower()
ENABLE_SPORTYTRADER_REMOTE = _sportytrader_env not in {"0", "false", "no", "off"}
PLAYWRIGHT_PROXY_CONFIGURED = bool(os.environ.get("PLAYWRIGHT_PROXY_SERVER", "").strip())

SPORT_TO_ESPNSLUG = {
    "NBA": ("basketball", "nba"),
    "NHL": ("hockey", "nhl"),
    "MLB": ("baseball", "mlb"),
    "EPL": ("soccer", "eng.1"),
    "WBC": ("baseball", "world-baseball-classic"),
}

USER_AGENT = "PickLedgerAutoGrader/1.0"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages").strip() or "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "").strip()
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01").strip() or "2023-06-01"
ANTHROPIC_MAX_TOKENS_DEFAULT = 800

TEAM_ABBREVIATION_ALIASES = {
    "WAS": {"WSH"},
    "WSH": {"WAS"},
    "NOP": {"NO"},
    "NO": {"NOP"},
    "GSW": {"GS"},
    "GS": {"GSW"},
    "PHX": {"PHO"},
    "PHO": {"PHX"},
    "SAS": {"SA"},
    "SA": {"SAS"},
    "NYK": {"NY"},
    "NY": {"NYK"},
    "BKN": {"BRK"},
    "BRK": {"BKN"},
}
_ledger_state_lock = threading.Lock()
LEDGER_DB_FILE = os.environ.get(
    "LEDGER_DB_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickledger.db"),
)
LEDGER_STATE_FILE = os.environ.get(
    "LEDGER_STATE_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "pickledger_state.json"),
)
LEDGER_STATE_KEY = "primary"


def _default_playwright_browsers_path() -> str:
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if configured:
        return configured
    darwin_cache = os.path.expanduser("~/Library/Caches/ms-playwright")
    if sys.platform == "darwin" and os.path.isdir(darwin_cache):
        return darwin_cache
    return "0"


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_MLB_ALIASES = {
    'anaheim': 'Los Angeles Angels', 'angels': 'Los Angeles Angels',
    'la angels': 'Los Angeles Angels',
    'astros': 'Houston Astros', 'houston': 'Houston Astros',
    'athletics': 'Oakland Athletics', 'oakland': 'Oakland Athletics',
    "a's": 'Oakland Athletics',
    'blue jays': 'Toronto Blue Jays', 'toronto': 'Toronto Blue Jays',
    'braves': 'Atlanta Braves', 'atlanta': 'Atlanta Braves',
    'brewers': 'Milwaukee Brewers', 'milwaukee': 'Milwaukee Brewers',
    'cardinals': 'St. Louis Cardinals', 'st. louis': 'St. Louis Cardinals',
    'st louis': 'St. Louis Cardinals', 'stl': 'St. Louis Cardinals',
    'cubs': 'Chicago Cubs', 'chicago cubs': 'Chicago Cubs',
    'dodgers': 'Los Angeles Dodgers', 'la dodgers': 'Los Angeles Dodgers',
    'giants': 'San Francisco Giants', 'san francisco': 'San Francisco Giants',
    'sf giants': 'San Francisco Giants',
    'guardians': 'Cleveland Guardians', 'gardians': 'Cleveland Guardians',
    'cleveland': 'Cleveland Guardians',
    'mariners': 'Seattle Mariners', 'seattle': 'Seattle Mariners',
    'marlins': 'Miami Marlins', 'miami': 'Miami Marlins',
    'mets': 'New York Mets', 'ny mets': 'New York Mets',
    'nationals': 'Washington Nationals', 'washington': 'Washington Nationals',
    'orioles': 'Baltimore Orioles', 'baltimore': 'Baltimore Orioles',
    'padres': 'San Diego Padres', 'san diego': 'San Diego Padres',
    'sd padres': 'San Diego Padres',
    'phillies': 'Philadelphia Phillies', 'philadelphia': 'Philadelphia Phillies',
    'pirates': 'Pittsburgh Pirates', 'pittsburgh': 'Pittsburgh Pirates',
    'rangers': 'Texas Rangers', 'texas': 'Texas Rangers',
    'rays': 'Tampa Bay Rays', 'tampa bay': 'Tampa Bay Rays',
    'tb rays': 'Tampa Bay Rays',
    'red sox': 'Boston Red Sox', 'boston': 'Boston Red Sox',
    'reds': 'Cincinnati Reds', 'cincinnati': 'Cincinnati Reds',
    'rockies': 'Colorado Rockies', 'colorado': 'Colorado Rockies',
    'royals': 'Kansas City Royals', 'kansas city': 'Kansas City Royals',
    'tigers': 'Detroit Tigers', 'detroit': 'Detroit Tigers',
    'twins': 'Minnesota Twins', 'minnesota': 'Minnesota Twins',
    'white sox': 'Chicago White Sox', 'chicago white sox': 'Chicago White Sox',
    'yankees': 'New York Yankees', 'ny yankees': 'New York Yankees',
    'diamondbacks': 'Arizona Diamondbacks', 'arizona': 'Arizona Diamondbacks',
    'd-backs': 'Arizona Diamondbacks',
}


def _norm_mlb(name: str) -> str:
    """Normalize any MLB team name/alias to canonical full name."""
    if not name:
        return ''
    key = name.strip().lower()
    if key in _MLB_ALIASES:
        return _MLB_ALIASES[key]
    for alias, canonical in _MLB_ALIASES.items():
        if key.endswith(alias) or key.startswith(alias):
            return canonical
    return name.strip()


def _default_ledger_state() -> dict[str, Any]:
    return {
        "version": 1,
        "savedAt": "",
        "addedPicks": [],
        "deletedPickIds": [],
        "results": {},
        "gameTimes": {},
    }


def _coerce_ledger_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = _default_ledger_state()
    if isinstance(payload.get("version"), int):
        state["version"] = int(payload["version"])
    if isinstance(payload.get("savedAt"), str):
        state["savedAt"] = payload["savedAt"].strip()

    added = payload.get("addedPicks")
    deleted = payload.get("deletedPickIds")
    results = payload.get("results")
    game_times = payload.get("gameTimes")

    if isinstance(added, list):
        state["addedPicks"] = added
    if isinstance(deleted, list):
        state["deletedPickIds"] = [str(v) for v in deleted]
    if isinstance(results, dict):
        state["results"] = {str(k): v for k, v in results.items()}
    if isinstance(game_times, dict):
        state["gameTimes"] = {str(k): v for k, v in game_times.items()}
    return state


def _ledger_db_connect() -> sqlite3.Connection:
    conn = sqlite3.connect(LEDGER_DB_FILE, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_ledger_state_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ledger_state (
            state_key TEXT PRIMARY KEY,
            state_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
        """
    )


def _ensure_picks_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sport TEXT NOT NULL DEFAULT 'Other',
            source TEXT NOT NULL DEFAULT '',
            pick TEXT NOT NULL DEFAULT '',
            date TEXT NOT NULL DEFAULT '',
            units INTEGER NOT NULL DEFAULT 1,
            odds INTEGER DEFAULT NULL,
            result TEXT NOT NULL DEFAULT 'pending',
            notes TEXT NOT NULL DEFAULT '',
            start_time TEXT DEFAULT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
        """
    )
    columns = {
        str(row["name"]): row
        for row in conn.execute("PRAGMA table_info(picks)").fetchall()
    }
    odds_column = columns.get("odds")
    start_time_column = columns.get("start_time")
    needs_migration = (
        odds_column is not None and bool(odds_column["notnull"])
    ) or (
        start_time_column is not None and bool(start_time_column["notnull"])
    )
    if not needs_migration:
        return

    conn.execute("ALTER TABLE picks RENAME TO picks_legacy")
    conn.execute(
        """
        CREATE TABLE picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sport TEXT NOT NULL DEFAULT 'Other',
            source TEXT NOT NULL DEFAULT '',
            pick TEXT NOT NULL DEFAULT '',
            date TEXT NOT NULL DEFAULT '',
            units INTEGER NOT NULL DEFAULT 1,
            odds INTEGER DEFAULT NULL,
            result TEXT NOT NULL DEFAULT 'pending',
            notes TEXT NOT NULL DEFAULT '',
            start_time TEXT DEFAULT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
        """
    )
    conn.execute(
        """
        INSERT INTO picks (
            id, sport, source, pick, date, units, odds, result, notes, start_time, created_at, updated_at
        )
        SELECT
            id,
            sport,
            source,
            pick,
            date,
            units,
            odds,
            result,
            notes,
            NULLIF(start_time, ''),
            created_at,
            updated_at
        FROM picks_legacy
        """
    )
    conn.execute("DROP TABLE picks_legacy")


def _ensure_nba_props_games_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nba_props_games (
            game_id TEXT PRIMARY KEY,
            game_date TEXT NOT NULL,
            home_team TEXT NOT NULL DEFAULT '',
            away_team TEXT NOT NULL DEFAULT '',
            game_time TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_nba_props_games_game_date
        ON nba_props_games (game_date)
        """
    )


def _sync_picks_table_from_state(conn: sqlite3.Connection, state: dict[str, Any]) -> None:
    _ensure_picks_table(conn)
    added = state.get("addedPicks")
    deleted = state.get("deletedPickIds")
    results = state.get("results")
    game_times = state.get("gameTimes")
    added_list = added if isinstance(added, list) else []
    deleted_ids = {str(v) for v in deleted} if isinstance(deleted, list) else set()
    result_map = results if isinstance(results, dict) else {}
    game_time_map = game_times if isinstance(game_times, dict) else {}
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    rows: list[tuple[Any, ...]] = []
    for item in added_list:
        if not isinstance(item, dict):
            continue
        pick_id_raw = item.get("id")
        pick_id_str = str(pick_id_raw)
        if pick_id_str in deleted_ids:
            continue
        try:
            pick_id = int(pick_id_raw)
        except (TypeError, ValueError):
            continue
        try:
            units = int(item.get("units", 1))
        except (TypeError, ValueError):
            units = 1
        raw_odds = item.get("odds")
        if raw_odds in {"", None}:
            odds = None
        else:
            try:
                odds = int(float(raw_odds))
            except (TypeError, ValueError):
                odds = None
        start_time_value = game_time_map.get(pick_id_str, item.get("start_time"))
        start_time = str(start_time_value).strip() if start_time_value not in {"", None} else None
        rows.append((
            pick_id,
            str(item.get("sport", "Other") or "Other"),
            str(item.get("source", "") or ""),
            str(item.get("pick", "") or ""),
            str(item.get("date", "") or ""),
            units,
            odds,
            str(result_map.get(pick_id_str, item.get("result", "pending")) or "pending"),
            str(item.get("notes", "") or ""),
            start_time,
            str(item.get("created_at", "") or now_iso),
            now_iso,
        ))

    conn.execute("DELETE FROM picks")
    if rows:
        conn.executemany(
            """
            INSERT INTO picks (
                id, sport, source, pick, date, units, odds, result, notes, start_time, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _load_ledger_state_from_sql() -> dict[str, Any] | None:
    try:
        with _ledger_db_connect() as conn:
            _ensure_ledger_state_table(conn)
            row = conn.execute(
                "SELECT state_json FROM ledger_state WHERE state_key = ? LIMIT 1",
                (LEDGER_STATE_KEY,),
            ).fetchone()
        if not row:
            return None
        payload = json.loads(str(row["state_json"] or "{}"))
        if isinstance(payload, dict):
            return _coerce_ledger_state(payload)
    except (sqlite3.Error, json.JSONDecodeError, TypeError, ValueError):
        return None
    return None


def _save_ledger_state_to_sql(state: dict[str, Any]) -> bool:
    try:
        payload = json.dumps(state, ensure_ascii=True, separators=(",", ":"))
        with _ledger_db_connect() as conn:
            _ensure_ledger_state_table(conn)
            _sync_picks_table_from_state(conn, state)
            conn.execute(
                """
                INSERT INTO ledger_state (state_key, state_json, updated_at)
                VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                ON CONFLICT(state_key) DO UPDATE SET
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (LEDGER_STATE_KEY, payload),
            )
        return True
    except sqlite3.Error:
        return False


def _load_ledger_state_from_file() -> dict[str, Any] | None:
    try:
        with open(LEDGER_STATE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return _coerce_ledger_state(payload)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return None


def _save_ledger_state_to_file(state: dict[str, Any]) -> bool:
    try:
        with open(LEDGER_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=True, indent=2)
        return True
    except OSError:
        return False


def _load_ledger_state() -> dict[str, Any]:
    with _ledger_state_lock:
        from_sql = _load_ledger_state_from_sql()
        if from_sql is not None:
            return from_sql

        # One-time compatibility fallback: hydrate SQL from prior file-backed state.
        from_file = _load_ledger_state_from_file()
        if from_file is not None:
            _save_ledger_state_to_sql(from_file)
            return from_file
    return _default_ledger_state()


def _save_ledger_state(payload: dict[str, Any]) -> bool:
    state = _coerce_ledger_state(payload)
    state["savedAt"] = datetime.utcnow().isoformat() + "Z"
    with _ledger_state_lock:
        sql_ok = _save_ledger_state_to_sql(state)
        file_ok = _save_ledger_state_to_file(state)
    return sql_ok or file_ok


def _load_ledger_state_unlocked() -> dict[str, Any]:
    from_sql = _load_ledger_state_from_sql()
    if from_sql is not None:
        return from_sql

    from_file = _load_ledger_state_from_file()
    if from_file is not None:
        _save_ledger_state_to_sql(from_file)
        return from_file
    return _default_ledger_state()


def _save_ledger_state_unlocked(payload: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    state = _coerce_ledger_state(payload)
    state["savedAt"] = datetime.utcnow().isoformat() + "Z"
    sql_ok = _save_ledger_state_to_sql(state)
    file_ok = _save_ledger_state_to_file(state)
    return sql_ok or file_ok, state


def _format_ledger_date_label(dt: datetime | None = None) -> str:
    current = dt or datetime.now()
    return current.strftime("%b %d").replace(" 0", " ")


def _normalize_pick_result(value: Any, *, allow_pending: bool = True) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"w", "win"}:
        return "win"
    if text in {"l", "loss"}:
        return "loss"
    if text in {"p", "push"}:
        return "push"
    if allow_pending and text in {"", "pending"}:
        return "pending"
    return None


def _coerce_optional_int(value: Any, default: int) -> int:
    if value in {"", None}:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value: Any) -> float | None:
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _next_ledger_pick_id(state: dict[str, Any]) -> int:
    max_id = 0
    for item in state.get("addedPicks", []):
        if not isinstance(item, dict):
            continue
        try:
            max_id = max(max_id, int(item.get("id") or 0))
        except (TypeError, ValueError):
            continue
    try:
        with _ledger_db_connect() as conn:
            _ensure_picks_table(conn)
            row = conn.execute("SELECT MAX(id) AS max_id FROM picks").fetchone()
        if row and row["max_id"] is not None:
            max_id = max(max_id, int(row["max_id"]))
    except sqlite3.Error:
        pass
    return max_id + 1


def _build_pick_log_entry(raw: dict[str, Any], pick_id: int | None = None) -> dict[str, Any]:
    now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    today_iso = datetime.now().strftime("%Y-%m-%d")
    probability = raw.get("probability")
    confidence = raw.get("confidence")
    if probability is None and confidence is not None:
        try:
            probability = max(0.0, min(float(confidence) / 100.0, 1.0))
        except (TypeError, ValueError):
            probability = None
    raw_odds = raw.get("odds")
    if raw_odds in {"", None}:
        odds = None
    else:
        try:
            odds = int(float(raw_odds))
        except (TypeError, ValueError):
            odds = None
    units = _coerce_optional_int(raw.get("units", 1), 1)
    confidence_float = _coerce_optional_float(confidence)
    confidence_int = int(round(confidence_float)) if confidence_float is not None else 0
    date_value = str(raw.get("date", "") or today_iso).strip() or today_iso
    source_value = str(raw.get("source", "") or "Manual").strip() or "Manual"
    notes_value = str(raw.get("notes", "") or "").strip()
    entry = {
        "id": pick_id,
        "sport": str(raw.get("sport", "Other") or "Other"),
        "source": source_value,
        "game": str(raw.get("game", "") or ""),
        "pick": str(raw.get("pick", "") or ""),
        "date": date_value,
        "units": units,
        "odds": odds,
        "confidence": confidence_int,
        "probability": probability,
        "result": _normalize_pick_result(raw.get("result"), allow_pending=True) or "pending",
        "notes": notes_value,
        "start_time": raw.get("start_time") if raw.get("start_time") not in {"", None} else None,
        "created_at": str(raw.get("created_at", "") or now_iso),
    }
    return entry


def _save_pick_to_ledger(raw: dict[str, Any]) -> tuple[bool, dict[str, Any] | None, str | None]:
    with _ledger_state_lock:
        state = _load_ledger_state_unlocked()
        entry = _build_pick_log_entry(raw, pick_id=_next_ledger_pick_id(state))
        if not entry["pick"]:
            return False, None, "Pick text is required"
        added = state.get("addedPicks")
        added_list = list(added) if isinstance(added, list) else []
        added_list.append(entry)
        state["addedPicks"] = added_list
        ok, _ = _save_ledger_state_unlocked(state)
    if not ok:
        return False, None, "Failed to persist pick"
    return True, entry, None


def _set_pick_result_in_ledger(pick_id: Any, result: Any) -> tuple[bool, str | None, str | None]:
    normalized = _normalize_pick_result(result, allow_pending=True)
    if normalized is None:
        return False, None, "Result must be one of W/L/P or pending"

    pick_id_str = str(pick_id or "").strip()
    if not pick_id_str:
        return False, None, "Pick id is required"

    with _ledger_state_lock:
        state = _load_ledger_state_unlocked()
        known_ids = set()
        for item in state.get("addedPicks", []):
            if isinstance(item, dict):
                known_ids.add(str(item.get("id", "")).strip())
        if pick_id_str not in known_ids:
            try:
                with _ledger_db_connect() as conn:
                    _ensure_picks_table(conn)
                    row = conn.execute(
                        "SELECT 1 FROM picks WHERE id = ? LIMIT 1",
                        (pick_id_str,),
                    ).fetchone()
                if row is None:
                    return False, None, "Pick not found"
            except sqlite3.Error:
                return False, None, "Pick lookup failed"

        results = state.get("results")
        result_map = dict(results) if isinstance(results, dict) else {}
        if normalized == "pending":
            result_map.pop(pick_id_str, None)
        else:
            result_map[pick_id_str] = normalized
        state["results"] = result_map
        ok, _ = _save_ledger_state_unlocked(state)
    if not ok:
        return False, None, "Failed to persist result"
    return True, normalized, None


def _extract_matchup_from_pick_text(pick_text: str) -> str | None:
    text = str(pick_text or "")
    matches = re.findall(r"\(([^)]+)\)", text)
    if not matches:
        return None
    for raw in matches:
        value = re.sub(r"\s+", " ", raw).strip()
        if re.search(r"\s+(vs|@)\s+", value, flags=re.IGNORECASE):
            return value
    value = re.sub(r"\s+", " ", matches[0]).strip()
    return value or None


def _parse_model_date_arg(date_str: str | None = None) -> tuple[str, str]:
    if not date_str:
        dt = datetime.now()
    else:
        dt = None
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(str(date_str).strip(), fmt)
                break
            except ValueError:
                continue
        if dt is None:
            dt = datetime.now()
    return dt.strftime("%Y-%m-%d"), dt.strftime("%m/%d/%Y")


def _fetch_nba_props_games_from_api(date_str: str | None = None) -> list[dict[str, str]]:
    date_iso, date_us = _parse_model_date_arg(date_str)
    python_candidates = [
        os.path.join(BASE_DIR, ".venv", "bin", "python"),
        os.path.join(NBA_PROPS_MODEL_DIR, "venv", "bin", "python"),
    ]
    python_bin = next((path for path in python_candidates if os.path.exists(path)), sys.executable)
    script = """
import json
import sys
from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams

date_us = sys.argv[1]
date_iso = sys.argv[2]
team_lookup = {
    int(team["id"]): str(team.get("full_name") or team.get("nickname") or team["id"])
    for team in teams.get_teams()
}
board = scoreboardv2.ScoreboardV2(game_date=date_us)
header = None
for frame in board.get_data_frames():
    columns = set(getattr(frame, "columns", []))
    if {"GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"}.issubset(columns):
        header = frame
        break
games = []
seen = set()
if header is not None:
    for _, row in header.iterrows():
        game_id = str(row.get("GAME_ID", "")).strip()
        if not game_id or game_id in seen:
            continue
        seen.add(game_id)
        home_team = team_lookup.get(int(row["HOME_TEAM_ID"]), str(row["HOME_TEAM_ID"]))
        away_team = team_lookup.get(int(row["VISITOR_TEAM_ID"]), str(row["VISITOR_TEAM_ID"]))
        games.append({
            "game_id": game_id,
            "game_date": date_iso,
            "home_team": home_team,
            "away_team": away_team,
            "game_time": str(row.get("GAME_STATUS_TEXT", "")).strip(),
        })
print(json.dumps({"ok": True, "games": games}))
"""
    result = _subprocess_run(
        [python_bin, "-c", script, date_us, date_iso],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=90,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(_compact_error_text(output))
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode NBA props slate response: {exc}") from exc
    games = payload.get("games", [])
    if not isinstance(games, list):
        raise RuntimeError("NBA props slate response was not a game list")
    cleaned: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = str(game.get("game_id", "")).strip()
        if not game_id or game_id in seen_ids:
            continue
        seen_ids.add(game_id)
        away_team = str(game.get("away_team", "")).strip()
        home_team = str(game.get("home_team", "")).strip()
        game_time = str(game.get("game_time", "")).strip()
        label = f"{away_team} @ {home_team}" if away_team and home_team else game_id
        cleaned.append({
            "game_id": game_id,
            "away_team": away_team,
            "home_team": home_team,
            "game_time": game_time,
            "label": label,
        })
    return cleaned


def _upsert_nba_props_games(games: list[dict[str, Any]], date_str: str | None = None) -> list[dict[str, str]]:
    date_iso, _ = _parse_model_date_arg(date_str)
    cleaned: list[dict[str, str]] = []
    rows: list[tuple[str, str, str, str, str]] = []
    seen_ids: set[str] = set()

    for game in games:
        if not isinstance(game, dict):
            continue
        game_id = str(game.get("game_id", "")).strip()
        if not game_id or game_id in seen_ids:
            continue
        seen_ids.add(game_id)
        away_team = str(game.get("away_team", "")).strip()
        home_team = str(game.get("home_team", "")).strip()
        game_time = str(game.get("game_time", "")).strip()
        label = f"{away_team} @ {home_team}" if away_team and home_team else game_id
        cleaned.append({
            "game_id": game_id,
            "away_team": away_team,
            "home_team": home_team,
            "game_time": game_time,
            "label": label,
        })
        rows.append((game_id, date_iso, home_team, away_team, game_time))

    with _ledger_db_connect() as conn:
        _ensure_nba_props_games_table(conn)
        if rows:
            conn.executemany(
                """
                INSERT INTO nba_props_games (
                    game_id,
                    game_date,
                    home_team,
                    away_team,
                    game_time,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
                ON CONFLICT(game_id) DO UPDATE SET
                    game_date = excluded.game_date,
                    home_team = excluded.home_team,
                    away_team = excluded.away_team,
                    game_time = excluded.game_time,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                """,
                rows,
            )
            placeholders = ", ".join("?" for _ in rows)
            conn.execute(
                f"""
                DELETE FROM nba_props_games
                WHERE game_date = ?
                  AND game_id NOT IN ({placeholders})
                """,
                [date_iso, *[row[0] for row in rows]],
            )
        else:
            conn.execute("DELETE FROM nba_props_games WHERE game_date = ?", (date_iso,))
        conn.commit()

    return cleaned


def _refresh_nba_props_games(date_str: str | None = None) -> dict[str, Any]:
    date_iso, _ = _parse_model_date_arg(date_str)
    games = _fetch_nba_props_games_from_api(date_str)
    stored_games = _upsert_nba_props_games(games, date_iso)
    return {"ok": True, "date": date_iso, "games": stored_games}


def _load_nba_games_from_sqlite(date_str: str | None = None) -> list[dict[str, str]]:
    today_iso, today_us = _parse_model_date_arg(date_str)
    labels: dict[str, str] = {}
    try:
        with _ledger_db_connect() as conn:
            _ensure_nba_props_games_table(conn)
            cached_games = conn.execute(
                """
                SELECT game_id, away_team, home_team, game_time
                FROM nba_props_games
                WHERE game_date = ?
                ORDER BY
                    CASE WHEN NULLIF(TRIM(game_time), '') IS NULL THEN 1 ELSE 0 END,
                    game_time,
                    away_team,
                    home_team
                """,
                (today_iso,),
            ).fetchall()
            if cached_games:
                games: list[dict[str, str]] = []
                for row in cached_games:
                    game_id = str(row["game_id"] or "").strip()
                    away_team = str(row["away_team"] or "").strip()
                    home_team = str(row["home_team"] or "").strip()
                    game_time = str(row["game_time"] or "").strip()
                    label = f"{away_team} @ {home_team}" if away_team and home_team else game_id
                    games.append({
                        "game_id": game_id,
                        "away_team": away_team,
                        "home_team": home_team,
                        "game_time": game_time,
                        "label": label,
                    })
                return games
            rows = conn.execute(
                """
                SELECT pick, date
                FROM picks
                WHERE UPPER(sport) = 'NBA'
                ORDER BY id DESC
                LIMIT 800
                """
            ).fetchall()
    except sqlite3.Error:
        return []

    for row in rows:
        row_date = str(row["date"] or "").strip()
        if row_date and row_date not in {today_iso, today_us}:
            continue
        matchup = _extract_matchup_from_pick_text(str(row["pick"] or ""))
        if not matchup:
            continue
        key = re.sub(r"[^a-z0-9]+", "", matchup.lower())
        if key and key not in labels:
            labels[key] = matchup

    return [{"label": label} for label in labels.values()]


def _load_nba_props_games_with_meta(date_str: str | None = None) -> dict[str, Any]:
    db_games = _load_nba_games_from_sqlite(date_str)
    if db_games:
        return {"games": db_games, "source": "db", "error": None}

    python_bin = _resolve_python_bin(os.path.join(NBA_PROPS_MODEL_DIR, "venv", "bin", "python"))
    extra_args: list[str] = ["--list-game-ids"]
    normalized_date = str(date_str or "").strip()
    if normalized_date:
        extra_args.insert(0, normalized_date)

    try:
        output = _run_script(
            python_bin,
            "run_props.py",
            NBA_PROPS_MODEL_DIR,
            timeout=90,
            extra_args=extra_args,
        )
    except subprocess.TimeoutExpired:
        return {"games": [], "source": "live", "error": "NBA props game lookup timed out"}
    except (OSError, ValueError) as exc:
        return {"games": [], "source": "live", "error": str(exc)}

    if "Traceback (most recent call last)" in output or "ModuleNotFoundError" in output:
        tail = " | ".join((output.strip().splitlines() or ["no output"])[-8:])
        return {"games": [], "source": "live", "error": f"NBA props live slate failed ({tail})"}

    error = None
    error_match = re.search(r"Error loading NBA game IDs:\s*(.+)", output)
    if error_match:
        error = error_match.group(1).strip()

    live_games = _extract_nba_props_games(output)
    if live_games:
        try:
            live_games = _upsert_nba_props_games(live_games, date_str)
        except sqlite3.Error:
            pass

    return {
        "games": live_games,
        "source": "live",
        "error": error,
    }


def _load_nba_props_games(date_str: str | None = None) -> list[dict[str, str]]:
    return _load_nba_props_games_with_meta(date_str).get("games", [])


def _normalize_person_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _person_names_match_loose(name_a: str, name_b: str) -> bool:
    a = _normalize_person_name(name_a)
    b = _normalize_person_name(name_b)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    a_parts = a.split()
    b_parts = b.split()
    if not a_parts or not b_parts:
        return False
    if a_parts[-1] != b_parts[-1]:
        return False
    return a_parts[0][0] == b_parts[0][0]


def _invoke_anthropic(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = ANTHROPIC_MAX_TOKENS_DEFAULT,
    temperature: float = 0.2,
    timeout: int = 60,
) -> dict[str, Any]:
    if not ANTHROPIC_API_KEY:
        return {
            "ok": False,
            "error": "Missing ANTHROPIC_API_KEY. Set it in your environment before calling /ask-opus.",
        }
    if not ANTHROPIC_MODEL:
        return {
            "ok": False,
            "error": "Missing ANTHROPIC_MODEL. Set it to the Opus model id before calling /ask-opus.",
        }

    clean_prompt = str(prompt or "").strip()
    if not clean_prompt:
        return {"ok": False, "error": "Prompt is required."}

    try:
        clean_max_tokens = int(max_tokens)
    except (TypeError, ValueError):
        clean_max_tokens = ANTHROPIC_MAX_TOKENS_DEFAULT
    clean_max_tokens = max(1, min(clean_max_tokens, 4096))

    try:
        clean_temperature = float(temperature)
    except (TypeError, ValueError):
        clean_temperature = 0.2
    clean_temperature = max(0.0, min(clean_temperature, 1.0))

    payload: dict[str, Any] = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": clean_max_tokens,
        "temperature": clean_temperature,
        "messages": [{"role": "user", "content": clean_prompt}],
    }
    if system is not None and str(system).strip():
        payload["system"] = str(system).strip()

    req = Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
            "User-Agent": USER_AGENT,
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            decoded = json.loads(raw)
    except HTTPError as exc:
        try:
            err_raw = exc.read().decode("utf-8")
            err_payload = json.loads(err_raw)
            err_msg = str(err_payload.get("error") or err_payload)
        except Exception:
            err_msg = str(exc)
        return {"ok": False, "error": f"Anthropic API error: {err_msg}"}
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"Anthropic request failed: {exc}"}

    text_parts: list[str] = []
    for block in decoded.get("content", []) if isinstance(decoded, dict) else []:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))

    return {
        "ok": True,
        "model": decoded.get("model", ANTHROPIC_MODEL) if isinstance(decoded, dict) else ANTHROPIC_MODEL,
        "response": "\n".join(part for part in text_parts if part).strip(),
        "raw": decoded,
    }


def parse_pick_date(date_text: str, year: int) -> str | None:
    try:
        dt = datetime.strptime(f"{date_text} {year}", "%b %d %Y")
        return dt.strftime("%Y%m%d")
    except ValueError:
        return None


def fetch_scoreboard(sport: str, league: str, yyyymmdd: str) -> dict[str, Any] | None:
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
        f"?dates={yyyymmdd}"
    )
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def fetch_event_summary(sport: str, league: str, event_id: str) -> dict[str, Any] | None:
    if not event_id:
        return None
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary"
        f"?event={event_id}"
    )
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None


def competitor_fields(comp: dict[str, Any]) -> list[str]:
    team = comp.get("team", {})
    out = [
        str(team.get("displayName", "")),
        str(team.get("shortDisplayName", "")),
        str(team.get("name", "")),
        str(team.get("abbreviation", "")),
    ]
    return [f for f in out if f]


def _team_code_aliases(value: str) -> set[str]:
    code = re.sub(r"[^A-Za-z]", "", str(value or "")).upper()
    if not code:
        return set()
    return {code, *TEAM_ABBREVIATION_ALIASES.get(code, set())}


def team_matches_competitor(team_text: str, comp: dict[str, Any]) -> bool:
    t = normalize(team_text)
    if not t:
        return False

    comp_code_aliases = _team_code_aliases(str(comp.get("team", {}).get("abbreviation", "")))
    if _team_code_aliases(team_text) & comp_code_aliases:
        return True

    for field in competitor_fields(comp):
        nf = normalize(field)
        if t == nf:
            return True
        if len(t) > 3 and t in nf:
            return True
        if nf in t and len(nf) > 2:
            return True

    # Last-token fallback for names like "Knicks", "Blues", "Senators".
    t_tokens = t.split()
    if t_tokens:
        last = t_tokens[-1]
        if len(last) >= 3:
            for field in competitor_fields(comp):
                f_tokens = normalize(field).split()
                if last in f_tokens:
                    return True

    return False


def _is_scores24_mlb_pick(pick: dict[str, Any] | None) -> bool:
    if not isinstance(pick, dict):
        return False
    return str(pick.get("source", "")).strip() == "Scores24" and str(pick.get("sport", "")).upper() == "MLB"


def _is_scores24_pick(pick: dict[str, Any] | None) -> bool:
    return isinstance(pick, dict) and str(pick.get("source", "")).strip() == "Scores24"


def _scores24_pick_matchup(pick: dict[str, Any] | None) -> tuple[str, str] | None:
    if not _is_scores24_pick(pick):
        return None
    away_team = str((pick or {}).get("away_team") or "").strip()
    home_team = str((pick or {}).get("home_team") or "").strip()
    if not away_team or not home_team:
        return None
    if _is_scores24_mlb_pick(pick):
        away_team = _norm_mlb(away_team)
        home_team = _norm_mlb(home_team)
    return away_team, home_team


def _scores24_mlb_team_matches_competitor(team_text: str, comp: dict[str, Any]) -> bool:
    team_name = _norm_mlb(team_text).lower()
    if not team_name:
        return False
    return any(_norm_mlb(field).lower() == team_name for field in competitor_fields(comp))


def _match_pick_team_to_competitor(team_text: str, comp: dict[str, Any], pick: dict[str, Any] | None = None) -> bool:
    if _is_scores24_mlb_pick(pick):
        return _scores24_mlb_team_matches_competitor(team_text, comp)
    return team_matches_competitor(team_text, comp)


def get_games(scoreboard: dict[str, Any], completed_only: bool) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for event in scoreboard.get("events", []):
        comps = event.get("competitions", [])
        if not comps:
            continue
        comp0 = comps[0]
        status = comp0.get("status", {}).get("type", {})
        if completed_only and not status.get("completed", False):
            continue

        competitors = comp0.get("competitors", [])
        if len(competitors) != 2:
            continue

        parsed = []
        valid = True
        for c in competitors:
            try:
                parsed.append({
                    "raw": c,
                    "score": int(c.get("score", "0")),
                    "homeAway": c.get("homeAway", ""),
                })
            except (ValueError, TypeError):
                valid = False
                break
        if not valid:
            continue

        start_time = str(comp0.get("date") or event.get("date") or "")
        games.append({
            "competitors": parsed,
            "startTime": start_time,
            "eventId": str(event.get("id") or ""),
        })
    return games


def parse_matchup(pick_text: str) -> tuple[str, str] | None:
    m = re.search(r"\(([^)]+)\)", pick_text)
    if not m:
        return None
    inside = m.group(1)
    parts = re.split(r"\s+(?:vs|@)\s+", inside, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    return parts[0].strip(), parts[1].strip()


def parse_nba_player_prop_pick(pick_text: str) -> dict[str, Any] | None:
    prop_m = re.search(
        r"^(.*?)\s+(points|rebounds|assists)\s+(OVER|UNDER)\s+(\d+(?:\.\d+)?)\s+vs\s+(.+?)(?:\s*\(|$)",
        str(pick_text or "").strip(),
        flags=re.IGNORECASE,
    )
    if not prop_m:
        return None
    return {
        "player_name": prop_m.group(1).strip(),
        "stat_key": prop_m.group(2).strip().lower(),
        "selection": prop_m.group(3).strip().upper(),
        "line": float(prop_m.group(4)),
        "opponent": prop_m.group(5).strip(),
    }


def find_game_for_pick(
    games: list[dict[str, Any]],
    pick_text: str,
    pick: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    matchup = _scores24_pick_matchup(pick) or parse_matchup(pick_text)
    if matchup:
        team_a, team_b = matchup
        for game in games:
            c1 = game["competitors"][0]["raw"]
            c2 = game["competitors"][1]["raw"]

            direct = _match_pick_team_to_competitor(team_a, c1, pick) and _match_pick_team_to_competitor(team_b, c2, pick)
            reverse = _match_pick_team_to_competitor(team_a, c2, pick) and _match_pick_team_to_competitor(team_b, c1, pick)
            if direct or reverse:
                return game

    prop_descriptor = parse_nba_player_prop_pick(pick_text)
    if not prop_descriptor:
        return None

    opponent = str(prop_descriptor.get("opponent") or "").strip()
    if not opponent:
        return None

    matches = [
        game for game in games
        if any(team_matches_competitor(opponent, comp["raw"]) for comp in game["competitors"])
    ]
    if len(matches) == 1:
        return matches[0]

    return None


def resolve_team_score(
    game: dict[str, Any],
    team_text: str,
    pick: dict[str, Any] | None = None,
) -> tuple[int, int] | None:
    comps = game["competitors"]
    for idx, c in enumerate(comps):
        if _match_pick_team_to_competitor(team_text, c["raw"], pick):
            opp = comps[1 - idx]
            return c["score"], opp["score"]
    return None


def parse_line(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, TypeError):
        return None


def _summary_stat_value_to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    if ":" in text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _extract_nba_player_stat(summary: dict[str, Any], player_name: str, stat_key: str) -> float | None:
    label_targets = {
        "points": {"PTS"},
        "rebounds": {"REB", "TREB", "TOTREB", "TOTAL REBOUNDS"},
        "assists": {"AST"},
    }
    targets = label_targets.get(stat_key)
    if not targets:
        return None

    boxscore = summary.get("boxscore", {}) if isinstance(summary, dict) else {}
    players = boxscore.get("players", []) if isinstance(boxscore, dict) else []
    for team_block in players if isinstance(players, list) else []:
        stat_sections = team_block.get("statistics", []) if isinstance(team_block, dict) else []
        for section in stat_sections if isinstance(stat_sections, list) else []:
            raw_labels = section.get("labels", []) if isinstance(section, dict) else []
            labels = [str(label).strip().upper() for label in raw_labels if str(label).strip()]
            stat_idx = next((idx for idx, label in enumerate(labels) if label in targets), None)
            if stat_idx is None:
                continue
            athletes = section.get("athletes", []) if isinstance(section, dict) else []
            for athlete in athletes if isinstance(athletes, list) else []:
                athlete_info = athlete.get("athlete", {}) if isinstance(athlete, dict) else {}
                display_name = str(athlete_info.get("displayName", "")).strip()
                if not _person_names_match_loose(player_name, display_name):
                    continue
                stats = athlete.get("stats", []) if isinstance(athlete, dict) else []
                if stat_idx >= len(stats):
                    return None
                return _summary_stat_value_to_float(stats[stat_idx])
    return None


def grade_nba_prop_pick(pick: dict[str, Any], game: dict[str, Any], summary: dict[str, Any] | None) -> str:
    prop = parse_nba_player_prop_pick(str(pick.get("pick", "")))
    if not prop or not summary:
        return "push"

    actual = _extract_nba_player_stat(summary, str(prop["player_name"]), str(prop["stat_key"]))
    if actual is None:
        return "push"

    line = float(prop["line"])
    selection = str(prop["selection"])
    if abs(actual - line) < 1e-9:
        return "push"
    if selection == "OVER":
        return "win" if actual > line else "loss"
    return "win" if actual < line else "loss"


def grade_pick(pick: dict[str, Any], game: dict[str, Any]) -> str:
    pick_text = str(pick.get("pick", ""))
    head = pick_text.split("(", 1)[0].strip()
    lower = head.lower()

    total_points = game["competitors"][0]["score"] + game["competitors"][1]["score"]

    # Full-game totals (Over/Under X)
    m_total = re.search(r"\b(over|under)\s+(\d+(?:\.\d+)?)\b", lower)
    # Skip if "team total" or team-prefixed TG (e.g. "senators over 3 tg"),
    # but allow game-level TG (e.g. "over 5.5 tg" where over/under is first word)
    has_team_tg = lower.endswith(" tg") and not re.match(r"^(over|under)\b", lower)
    if m_total and "team total" not in lower and not has_team_tg:
        side = m_total.group(1)
        line = float(m_total.group(2))
        if total_points == line:
            return "push"
        if side == "over":
            return "win" if total_points > line else "loss"
        return "win" if total_points < line else "loss"

    # Team total over/under, e.g. "Korea Team Total Over 9.5"
    m_team_total = re.search(r"^(.*?)\s+team total\s+(over|under)\s+(\d+(?:\.\d+)?)", lower)
    if m_team_total:
        team_label = m_team_total.group(1).strip()
        side = m_team_total.group(2)
        line = float(m_team_total.group(3))
        resolved = resolve_team_score(game, team_label, pick)
        if resolved is None:
            return "pending"
        team_score = resolved[0]
        if team_score == line:
            return "push"
        if side == "over":
            return "win" if team_score > line else "loss"
        return "win" if team_score < line else "loss"

    # Team goals shorthand, e.g. "Senators Over 3 TG"
    m_tg = re.search(r"^(.*?)\s+(over|under)\s+(\d+(?:\.\d+)?)\s*tg\b", lower)
    if m_tg:
        team_label = m_tg.group(1).strip()
        side = m_tg.group(2)
        line = float(m_tg.group(3))
        resolved = resolve_team_score(game, team_label, pick)
        if resolved is None:
            return "pending"
        team_score = resolved[0]
        if team_score == line:
            return "push"
        if side == "over":
            return "win" if team_score > line else "loss"
        return "win" if team_score < line else "loss"

    # Skip 1H / partial-game markets for now.
    if re.search(r"\b1h\b|first half|period", lower):
        return "pending"

    # Draw pick (soccer): "Draw (Team A vs Team B)"
    if re.match(r"^draw$", lower):
        c0 = game["competitors"][0]["score"]
        c1 = game["competitors"][1]["score"]
        return "win" if c0 == c1 else "loss"

    # Both Teams to Score: "BTTS Yes" or "BTTS No"
    m_btts = re.match(r"^btts\s+(yes|no)$", lower)
    if m_btts:
        c0 = game["competitors"][0]["score"]
        c1 = game["competitors"][1]["score"]
        both_scored = c0 > 0 and c1 > 0
        side = m_btts.group(1)
        if side == "yes":
            return "win" if both_scored else "loss"
        return "win" if not both_scored else "loss"

    # Spread / run line / puck line, e.g. "Knicks -11.5"
    m_spread = re.search(r"^(.*?)\s*([+-]\d+(?:\.\d+)?)\b", head)
    if m_spread:
        team_label = m_spread.group(1).strip()
        try:
            spread = float(m_spread.group(2))
        except ValueError:
            return "pending"
        resolved = resolve_team_score(game, team_label, pick)
        if resolved is None:
            return "pending"
        team_score, opp_score = resolved
        adj = team_score + spread
        if abs(adj - opp_score) < 1e-9:
            return "push"
        return "win" if adj > opp_score else "loss"

    # Moneyline explicit: "Team ML"
    m_ml = re.search(r"^(.*?)\s+ml\b", lower)
    if m_ml:
        team_label = m_ml.group(1).strip()
        resolved = resolve_team_score(game, team_label, pick)
        if resolved is None:
            return "pending"
        team_score, opp_score = resolved
        if team_score == opp_score:
            return "push"
        return "win" if team_score > opp_score else "loss"

    # Fallback: treat leading team label as winner pick.
    fallback_team = re.sub(r"\s*[+-]\d+(?:\.\d+)?\s*$", "", head, flags=re.IGNORECASE).strip()
    if fallback_team:
        resolved = resolve_team_score(game, fallback_team, pick)
        if resolved is None:
            return "pending"
        team_score, opp_score = resolved
        if team_score == opp_score:
            return "push"
        return "win" if team_score > opp_score else "loss"

    return "pending"


def auto_grade(picks: list[dict[str, Any]], existing: dict[str, str], year: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    all_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pick in picks:
        pid = str(pick.get("id"))
        if not pid:
            continue
        sport_key = str(pick.get("sport", "")).upper()
        if sport_key not in SPORT_TO_ESPNSLUG:
            continue

        d = parse_pick_date(str(pick.get("date", "")), year)
        if not d:
            continue

        all_grouped.setdefault((sport_key, d), []).append(pick)

        current = existing.get(pid, pick.get("result", "pending"))
        if current != "pending":
            continue

        grouped.setdefault((sport_key, d), []).append(pick)

    graded: dict[str, str] = {}
    start_times: dict[str, str] = {}
    attempted = 0
    board_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    summary_cache: dict[tuple[str, str], dict[str, Any] | None] = {}

    for (sport_key, d), batch in all_grouped.items():
        sport, league = SPORT_TO_ESPNSLUG[sport_key]
        key = (sport_key, d)
        if key not in board_cache:
            board_cache[key] = fetch_scoreboard(sport, league, d)
        board = board_cache[key]
        if not board:
            continue
        all_games = get_games(board, completed_only=False)

        for pick in batch:
            game = find_game_for_pick(all_games, str(pick.get("pick", "")), pick)
            if game and game.get("startTime"):
                start_times[str(pick["id"])] = str(game["startTime"])

    for (sport_key, d), batch in grouped.items():
        sport, league = SPORT_TO_ESPNSLUG[sport_key]
        key = (sport_key, d)
        if key not in board_cache:
            board_cache[key] = fetch_scoreboard(sport, league, d)
        board = board_cache[key]
        if not board:
            continue
        games = get_games(board, completed_only=True)

        for pick in batch:
            attempted += 1
            game = find_game_for_pick(games, str(pick.get("pick", "")), pick)
            if not game:
                continue
            if sport_key == "NBA" and parse_nba_player_prop_pick(str(pick.get("pick", ""))):
                event_id = str(game.get("eventId") or "").strip()
                summary_key = (sport_key, event_id)
                if summary_key not in summary_cache:
                    summary_cache[summary_key] = fetch_event_summary(sport, league, event_id)
                result = grade_nba_prop_pick(pick, game, summary_cache.get(summary_key))
            else:
                result = grade_pick(pick, game)
            if result in {"win", "loss", "push"}:
                graded[str(pick["id"])] = result

    return {
        "graded": graded,
        "startTimes": start_times,
        "summary": {
            "attempted": attempted,
            "updated": len(graded),
            "remaining": max(0, attempted - len(graded)),
        },
    }


# ─── Model Runner Helpers ──────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NBA_MODEL_DIR = os.path.join(BASE_DIR, "NBAPredictionModel")
MLB_MODEL_DIR = os.path.join(BASE_DIR, "MLBPredictionModel")
NBA_PROPS_MODEL_DIR = os.path.join(BASE_DIR, "NBAPlayerBettingModel")
SCORES24_VENV = os.path.join(BASE_DIR, ".venv", "bin", "python")
SPORTYTRADER_VENV = os.path.join(BASE_DIR, ".venv", "bin", "python")
SPORTSGAMBLER_VENV = os.path.join(BASE_DIR, ".venv", "bin", "python")

# ─── Async Job Store ──────────────────────────────────────────────────────────
# Tracks running/completed model jobs so the frontend can poll for results.
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_playwright_install_lock = threading.Lock()
_playwright_ready = False


def _subprocess_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Launch child commands with a stable stdin for launchd-hosted runtimes."""
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    return subprocess.run(*args, **kwargs)


def _run_script(python_bin: str, script: str, cwd: str, timeout: int = 300, extra_args: list[str] | None = None) -> str:
    """Run a Python script and return its stdout."""
    cmd = [python_bin, script] + (extra_args or [])
    result = _subprocess_run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr


def _resolve_python_bin(preferred_path: str) -> str:
    """Use model-specific venv if present; otherwise use current interpreter."""
    if os.path.exists(preferred_path):
        return preferred_path
    return sys.executable


def _looks_like_playwright_browser_missing(output: str) -> bool:
    text = output.lower()
    return (
        "executable doesn't exist" in text
        or "playwright was just installed or updated" in text
        or "please run the following command to download new browsers" in text
    )


def _compact_error_text(output: str, max_lines: int = 14) -> str:
    """Compress subprocess output into a readable single-line summary."""
    lines = [ln.strip() for ln in (output or "").splitlines() if ln.strip()]
    if not lines:
        return "no output"

    # Prefer the most actionable Playwright line when present.
    for ln in lines:
        if "Executable doesn't exist at" in ln:
            return ln

    tail = lines[-max_lines:]
    compact = " | ".join(tail)
    compact = re.sub(r"\s+", " ", compact)
    return compact[:1800]


def _looks_like_transient_scores24_listing_failure(output: str) -> bool:
    text = (output or "").lower()
    if "listing page status" not in text:
        return False
    transient_signals = (
        "status 408",
        "status 429",
        "status 500",
        "status 502",
        "status 503",
        "status 504",
        "just a moment",
        "attention required",
        "performing security verification",
    )
    return any(sig in text for sig in transient_signals)


def _ensure_playwright_browsers(python_bin: str, env: dict[str, str]) -> tuple[bool, str]:
    """Install Playwright Chromium browsers if missing in the current environment."""
    global _playwright_ready

    with _playwright_install_lock:
        if _playwright_ready:
            return True, "already-ready"

        try:
            install = _subprocess_run(
                [python_bin, "-m", "playwright", "install", "chromium", "chromium-headless-shell"],
                cwd=BASE_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except Exception as exc:
            return False, str(exc)

        if install.returncode == 0:
            _playwright_ready = True
            return True, "installed"

        msg = _compact_error_text((install.stdout or "") + (install.stderr or ""))
        return False, msg


def _parse_nba_output(output: str, source_label: str = "NBA Model") -> list[dict[str, Any]]:
    """Parse NBA model stdout into pick dicts."""
    picks: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()

    def _format_line_value(line_value: float | None) -> str:
        return f"{line_value:.1f}" if isinstance(line_value, (int, float)) else "N/A"

    def _format_model_total_value(total_value: float | None) -> str:
        return f"{total_value:.1f}" if isinstance(total_value, (int, float)) else "N/A"

    def _append_unique(pick: dict[str, Any]) -> None:
        key = (
            str(pick.get("source", "")),
            str(pick.get("sport", "")),
            str(pick.get("pick", "")),
        )
        if key in seen_keys:
            return
        seen_keys.add(key)
        picks.append(pick)

    # The output has GAME: headers and ### prediction blocks separated by ===
    # Strategy: scan line-by-line, tracking current game context
    lines = output.split("\n")
    current_away = ""
    current_home = ""

    for i, line in enumerate(lines):
        # Pick up game header: "GAME: Grizzlies @ Pistons (7:30 pm ET)"
        game_m = re.match(r"^GAME:\s*(.+?)\s*@\s*(.+?)(?:\s*\(|$)", line)
        if game_m:
            current_away = game_m.group(1).strip()
            current_home = game_m.group(2).strip()
            continue

        if not current_away or not current_home:
            continue

        # Extract winner from either the legacy block or the new projection-only block.
        winner = ""
        prob = None
        parsed_new_format = False

        winner_m = re.search(r"\*\*Winner:\*\*\s*(.+?)\s*\(Model Prob:\s*([\d.]+)%\)", line)
        if winner_m:
            winner = winner_m.group(1).strip()
            prob = float(winner_m.group(2)) / 100
        else:
            pick_m = re.search(r"\*\*Pick:\*\*\s*(.+)", line)
            if pick_m:
                winner = pick_m.group(1).strip()
                parsed_new_format = True

        if winner:
            # Look ahead for spread, confidence, optional edge, and decision/projection note.
            spread_val = 0.0
            edge_val: float | None = None
            decision = "BET" if parsed_new_format else "PASS"
            for j in range(i + 1, min(i + 20, len(lines))):
                sp_m = re.search(r"\*\*(?:Spread|Projected Margin):\*\*\s*.+?\s*by\s*([\d.]+)\s*points", lines[j])
                if sp_m:
                    spread_val = float(sp_m.group(1))
                conf_m = re.search(r"\*\*Model Confidence:\*\*\s*([\d.]+)%", lines[j])
                if conf_m:
                    prob = float(conf_m.group(1)) / 100
                edge_m = re.search(r"\*\*Edge:\*\*\s*\S+\s*([+-]?[\d.]+)%", lines[j])
                if edge_m:
                    edge_val = float(edge_m.group(1))
                dec_m = re.search(r"\*\*Decision:\s*(BET|PASS)", lines[j])
                if dec_m:
                    decision = dec_m.group(1)
                    break
                proj_note_m = re.search(r"\*\*Projection note:\*\*", lines[j])
                if proj_note_m:
                    decision = "PASS"
                    break

            matchup = f"{current_away} @ {current_home}"
            parsed_margin = spread_val
            # Sanity gate: never publish a NBA spread > 16 pts from the new model.
            if source_label == "NBA New" and abs(parsed_margin) > 16.0:
                decision = "PASS"
            if spread_val > 0 and (decision == "BET" or parsed_new_format):
                pick_text = f"{winner} -{spread_val:.1f} ({matchup})"
            else:
                pick_text = f"{winner} ML ({matchup})"

            # Edge is from the home team perspective; flip sign if bet is on away team
            display_edge = edge_val
            if isinstance(display_edge, float) and winner != current_home:
                display_edge = -display_edge

            pick = {
                "source": source_label,
                "pick": pick_text,
                "sport": "NBA",
                "odds": None,
                "units": 1,
                "probability": prob,
                "edge": display_edge,
                "decision": decision,
                "team": winner,
                "away_team": current_away,
                "home_team": current_home,
                "predicted_spread": spread_val,
            }
            if source_label == "NBA New":
                sl_spread_home, sl_spread_away, sl_spread_odds = _sl_get_spread(
                    current_home, current_away, 'NBA'
                )
                if sl_spread_home is not None:
                    # Pick the Vegas spread for the winning team
                    team_last = winner.split()[-1].lower()
                    home_last = current_home.split()[-1].lower()
                    vegas_spread = sl_spread_home if team_last == home_last else sl_spread_away

                    _sp_odds = sl_spread_odds if sl_spread_odds else -110
                    # Implied probability from Vegas vig
                    _implied = (abs(_sp_odds) / (abs(_sp_odds) + 100)) if _sp_odds < 0 \
                               else (100 / (_sp_odds + 100))
                    _model_prob = pick.get("probability") or 0.75
                    _edge_val = round((_model_prob - _implied) * 100, 2)

                    # Quarter-Kelly sizing (capped at 5%)
                    _b = 100.0 / abs(_sp_odds) if _sp_odds < 0 else _sp_odds / 100.0
                    _kf = round(min(max((_b * _model_prob - (1 - _model_prob)) / _b, 0.0) * 0.25, 0.05) * 100, 2)

                    pick["odds"] = _sp_odds
                    pick["market_line"] = vegas_spread
                    pick["model_prediction"] = -round(float(spread_val), 1)
                    pick["edge"] = _edge_val
                    pick["units"] = _kf
                    if _edge_val >= 5.0:
                        pick["decision"] = "BET"
                    elif _edge_val >= 3.0:
                        pick["decision"] = "LEAN"
                    else:
                        pick["decision"] = "PASS"
            _append_unique(pick)

        # Over/Under decision: "**O/U Decision: BET OVER**"
        ou_m = re.search(r"\*\*O/U Decision:\s*(BET OVER|BET UNDER|PASS)\*\*", line)
        if ou_m:
            model_total = None
            for j in range(max(0, i - 12), i):
                projected_total_m = re.search(r"-\s*\*\*Total:\*\*\s*([\d.]+)\s*O/U", lines[j])
                if projected_total_m:
                    model_total = float(projected_total_m.group(1))
                total_m = re.search(
                    r"\*\*Over/Under:\*\*\s*Model Total\s*([\d.]+)\s*vs\s*Line\s*(N/A|[\d.]+)",
                    lines[j],
                    flags=re.IGNORECASE,
                )
                if total_m:
                    model_total = float(total_m.group(1))
            if model_total is None:
                continue

            # ── O/U pick assembly ──────────────────────────────────────────
            league = "NBA"
            home_team = current_home
            away_team = current_away
            vegas_total, total_odds = _sl_get_total(home_team, away_team, league)
            if vegas_total is None:
                pass
            else:
                direction = 'Under' if model_total < vegas_total else 'Over'
                pick_label = f"{direction} {vegas_total} ({away_team} vs {home_team})"
                _odds_price = total_odds if total_odds else -110
                _prob = _ou_probability(float(model_total), float(vegas_total), _NBA_TOTALS_RMSE)
                _b = abs(_odds_price) / 100 if _odds_price > 0 else 100 / abs(_odds_price or 110)
                _implied_prob = 1 / (1 + _b)
                _edge_prob = _prob - _implied_prob
                _q = 1 - _prob
                _k = max((_b * _prob - _q) / _b, 0.0)
                _kf = round(_k * 0.25 * 100, 2)
                ou_pick = {
                    "source": source_label,
                    "pick": pick_label,
                    "sport": league,
                    "odds": _odds_price,
                    "units": 1,
                    "probability": _prob,
                    "prob": _prob,
                    "edge": round(_edge_prob * 100, 2),
                    "vegas": vegas_total,
                    "model_prediction": round(float(model_total), 1),
                    "direction": direction,
                    "kelly": _kf,
                    "decision": 'BET' if _edge_prob >= 0.05 else ('LEAN' if _edge_prob >= 0.03 else 'PASS'),
                    "market_type": "totals",
                    "selection": direction,
                    "line": vegas_total,
                    "market_line": vegas_total,
                    "away_team": away_team,
                    "home_team": home_team,
                }
                _append_unique(ou_pick)

    return picks


def _parse_nba_props_output(output: str) -> list[dict[str, Any]]:
    """Parse NBA props model stdout into pick dicts."""
    picks: list[dict[str, Any]] = []
    current_game: dict[str, str] | None = None
    current_player: dict[str, Any] | None = None
    current_prop: dict[str, Any] | None = None
    current_metrics: dict[str, Any] = {}

    prop_key_lookup = {
        "points": "points",
        "rebounds": "rebounds",
        "assists": "assists",
    }

    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        game_m = re.match(r"^GAME:\s*(.+?)\s*@\s*(.+)$", line, flags=re.IGNORECASE)
        if game_m:
            current_game = {
                "away_team": game_m.group(1).strip(),
                "home_team": game_m.group(2).strip(),
            }
            current_player = None
            current_prop = None
            current_metrics = {}
            continue

        player_m = re.match(
            r"^PLAYER:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*vs\s+(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if player_m:
            current_player = {
                "name": player_m.group(1).strip(),
                "position": player_m.group(2).strip(),
                "team": player_m.group(3).strip(),
                "opponent": player_m.group(4).strip(),
            }
            current_prop = None
            current_metrics = {}
            continue

        prop_m = re.match(
            r"^PROP:\s*(Points|Rebounds|Assists)\s*-\s*Line:\s*([0-9.]+)$",
            line,
            flags=re.IGNORECASE,
        )
        if prop_m:
            prop_label = prop_m.group(1).strip()
            current_prop = {
                "label": prop_label,
                "key": prop_key_lookup.get(prop_label.lower(), prop_label.lower()),
                "line": float(prop_m.group(2)),
            }
            current_metrics = {}
            continue

        metrics_m = re.match(
            r"^RF Predicted:\s*([0-9.]+)\s*\|\s*Direction:\s*(OVER|UNDER)\s*\|\s*Edge:\s*([0-9.]+)%$",
            line,
            flags=re.IGNORECASE,
        )
        if metrics_m:
            current_metrics["predicted"] = float(metrics_m.group(1))
            current_metrics["direction"] = metrics_m.group(2).upper()
            current_metrics["edge"] = float(metrics_m.group(3))
            continue

        kelly_m = re.match(
            r"^Confidence:\s*([0-9.]+)%\s*\|\s*Full Kelly:\s*([0-9.]+)%\s*\|\s*(?:1/4|¼)\s*Kelly:\s*([0-9.]+)% bankroll$",
            line,
            flags=re.IGNORECASE,
        )
        if kelly_m:
            current_metrics["confidence"] = float(kelly_m.group(1))
            current_metrics["full_kelly"] = float(kelly_m.group(2))
            current_metrics["quarter_kelly"] = float(kelly_m.group(3))
            continue

        decision_m = re.match(r"^\*\*Decision:\s*(BET\s+(?:OVER|UNDER)\s+[0-9.]+|PASS)\*\*$", line, flags=re.IGNORECASE)
        if decision_m and current_player and current_prop:
            edge_pct = float(current_metrics.get("edge", 0.0))
            quarter_kelly = float(current_metrics.get("quarter_kelly", 0.0))
            direction = str(current_metrics.get("direction", "OVER")).upper()
            true_prob = min(0.78, 0.5238 + (edge_pct * 0.008))
            market_type = {
                "points": "player_points",
                "rebounds": "player_rebounds",
                "assists": "player_assists",
            }.get(current_prop["key"], "")

            picks.append({
                "source": "NBA Props Model",
                "pick": (
                    f"{current_player['name']} "
                    f"{current_prop['key']} "
                    f"{direction} "
                    f"{current_prop['line']:.1f} "
                    f"vs {current_player['opponent']}"
                ),
                "sport": "NBA",
                "odds": None,
                "units": quarter_kelly,
                "probability": true_prob,
                "edge": edge_pct,
                "decision": "BET" if decision_m.group(1).upper().startswith("BET") else "PASS",
                "market_type": market_type,
                "selection": direction.title(),
                "line": current_prop["line"],
                "player_name": current_player["name"],
                "team": current_player["team"],
                "opponent": current_player["opponent"],
                "away_team": current_game["away_team"] if current_game else None,
                "home_team": current_game["home_team"] if current_game else None,
            })

    return picks


def _extract_nba_props_game_ids(output: str) -> list[str]:
    game_ids: list[str] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        game_id_m = re.match(r"^GAME_ID:\s*([0-9A-Za-z]+)(?:\s*\|.*)?$", line)
        if not game_id_m:
            continue
        game_id = game_id_m.group(1).strip()
        if not game_id or game_id in seen:
            continue
        seen.add(game_id)
        game_ids.append(game_id)
    return game_ids


def _extract_nba_props_games(output: str) -> list[dict[str, str]]:
    games: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        game_m = re.match(r"^GAME_ID:\s*([0-9A-Za-z]+)(?:\s*\|\s*(.+?)\s*@\s*(.+))?$", line)
        if not game_m:
            continue
        game_id = game_m.group(1).strip()
        if not game_id or game_id in seen:
            continue
        seen.add(game_id)
        away_team = str(game_m.group(2) or "").strip()
        home_team = str(game_m.group(3) or "").strip()
        label = f"{away_team} @ {home_team}" if away_team and home_team else game_id
        games.append({
            "game_id": game_id,
            "away_team": away_team,
            "home_team": home_team,
            "label": label,
        })
    return games


def _shorten_mlb_name(full_name: str) -> str:
    """Shorten full MLB team names to match ledger style (e.g. 'Tampa Bay Rays' -> 'Rays')."""
    # Multi-word team names that shouldn't be split
    multi_word = {"Red Sox", "White Sox", "Blue Jays"}
    for mw in multi_word:
        if full_name.endswith(mw):
            return mw
    # Default: last word
    parts = full_name.strip().split()
    return parts[-1] if parts else full_name


def _parse_mlb_output(output: str) -> list[dict[str, Any]]:
    """Parse MLB model stdout into pick dicts."""
    picks: list[dict[str, Any]] = []

    # The MLB model uses pipe-delimited output lines:
    # TeamA|TeamB|OddsA|OddsB|ProbA|ProbB
    # Also check for standard format output blocks

    # Try pipe-delimited format first (from the run_predictions.py style)
    pipe_lines = [l.strip() for l in output.split("\n") if l.count("|") >= 2 and not l.startswith("---")]

    if pipe_lines:
        # Track current game context for O/U lines
        current_team_a = ""
        current_team_b = ""

        for line in pipe_lines:
            parts = [p.strip() for p in line.split("|")]

            # O/U line: OU|OVER/UNDER/PASS|line|predicted_total
            if len(parts) >= 4 and parts[0] == "OU":
                try:
                    predicted_total = float(parts[3])
                except (ValueError, IndexError):
                    continue

                if not current_team_a or not current_team_b:
                    continue

                # ── O/U pick assembly ──────────────────────────────────────────
                league = "MLB"
                home_team = current_team_b
                away_team = current_team_a
                vegas_total, total_odds = _sl_get_total(home_team, away_team, league)
                if vegas_total is None:
                    pass
                else:
                    direction = 'Under' if predicted_total < vegas_total else 'Over'
                    pick_label = f"{direction} {vegas_total} ({away_team} vs {home_team})"
                    _odds_price = total_odds if total_odds else -110
                    _prob = _ou_probability(float(predicted_total), float(vegas_total), _MLB_TOTALS_RMSE)
                    _b = abs(_odds_price) / 100 if _odds_price > 0 else 100 / abs(_odds_price or 110)
                    _implied_prob = 1 / (1 + _b)
                    _edge_prob = _prob - _implied_prob
                    _q = 1 - _prob
                    _k = max((_b * _prob - _q) / _b, 0.0)
                    _kf = round(_k * 0.25 * 100, 2)
                    ou_pick = {
                        "source": "MLB Model",
                        "pick": pick_label,
                        "sport": league,
                        "odds": _odds_price,
                        "units": 1,
                        "probability": _prob,
                        "prob": _prob,
                        "edge": round(_edge_prob * 100, 2),
                        "vegas": vegas_total,
                        "model_prediction": round(float(predicted_total), 1),
                        "direction": direction,
                        "kelly": _kf,
                        "decision": 'BET' if _edge_prob >= 0.05 else ('LEAN' if _edge_prob >= 0.03 else 'PASS'),
                        "market_type": "totals",
                        "selection": direction,
                        "line": vegas_total,
                        "market_line": vegas_total,
                        "away_team": away_team,
                        "home_team": home_team,
                    }
                    picks.append(ou_pick)
                continue

            # Moneyline line: TeamA|TeamB|OddsA|OddsB|ProbA|ProbB
            if len(parts) < 6:
                continue
            try:
                team_a, team_b = parts[0], parts[1]
                odds_a, odds_b = int(float(parts[2])), int(float(parts[3]))
                prob_a, prob_b = float(parts[4]), float(parts[5])
            except (ValueError, IndexError):
                continue

            current_team_a = team_a
            current_team_b = team_b

            # Shorten names to match ledger style
            short_a = _shorten_mlb_name(team_a)
            short_b = _shorten_mlb_name(team_b)
            ml_home, ml_away = _sl_get_ml(team_b, team_a, "MLB")

            # Pipe format odds are model-derived (not market), so edge vs those
            # odds is always ~0.  Instead, compare model prob vs a generic 50%
            # market (flat -110 each side) — BET when model gives 55%+ to one side.
            if prob_a >= prob_b:
                bet_team = short_a
                bet_prob = prob_a
                bet_odds = odds_a
                edge = (prob_a - 0.50) * 100  # edge vs 50/50 market
            else:
                bet_team = short_b
                bet_prob = prob_b
                bet_odds = odds_b
                edge = (prob_b - 0.50) * 100

            matchup = f"{short_a} vs {short_b}"
            decision = "BET" if bet_prob >= 0.55 else "PASS"

            picks.append({
                "source": "MLB Model",
                "pick": f"{bet_team} ML ({matchup})",
                "sport": "MLB",
                "odds": ml_away if bet_team == short_a else ml_home,
                "units": 1,
                "probability": bet_prob,
                "edge": edge,
                "decision": decision,
                "market_type": "h2h",
                "team": bet_team,
                "away_team": team_a,
                "home_team": team_b,
                "model_odds": bet_odds if bet_odds != 0 else None,
            })
        return picks

    # Fallback: parse structured markdown output (from test_live.py/main.py format)
    blocks = re.split(r"={40,}", output)
    current_away = ""
    current_home = ""

    for block in blocks:
        game_m = re.search(r"###\s*\[(.+?)\]\s*vs\s*\[(.+?)\]", block)
        if game_m:
            current_away = game_m.group(1).strip()
            current_home = game_m.group(2).strip()

        if not current_away or not current_home:
            continue

        winner_m = re.search(r"\*\*Winner:\*\*\s*(.+?)\s*\(Model Prob:\s*([\d.]+)%\)", block)
        decision_m = re.search(r"\*\*Decision:\s*(BET|PASS)(?:\s+on\s+(.+?))?\*\*", block)
        edge_m = re.search(r"\*\*Edge:\*\*\s*\S+\s*([+-]?[\d.]+)%", block)
        total_m = re.search(r"\*\*Total Runs:\*\*\s*([\d.]+)", block)

        if winner_m and decision_m:
            winner = _shorten_mlb_name(winner_m.group(1).strip())
            prob = float(winner_m.group(2)) / 100
            decision = decision_m.group(1)
            edge_val = float(edge_m.group(1)) if edge_m else None
            matchup = f"{_shorten_mlb_name(current_away)} vs {_shorten_mlb_name(current_home)}"
            ml_home, ml_away = _sl_get_ml(current_home, current_away, "MLB")

            picks.append({
                "source": "MLB Model",
                "pick": f"{winner} ML ({matchup})",
                "sport": "MLB",
                "odds": ml_home if winner == _shorten_mlb_name(current_home) else ml_away,
                "units": 1,
                "probability": prob,
                "edge": edge_val,
                "decision": decision,
                "market_type": "h2h",
                "team": winner,
                "away_team": current_away,
                "home_team": current_home,
            })

            # Also emit total runs pick if available
            if total_m:
                total_val = float(total_m.group(1))
                # ── O/U pick assembly ──────────────────────────────────────────
                league = "MLB"
                home_team = current_home
                away_team = current_away
                vegas_total, total_odds = _sl_get_total(home_team, away_team, league)
                if vegas_total is None:
                    pass
                else:
                    direction = 'Under' if total_val < vegas_total else 'Over'
                    pick_label = f"{direction} {vegas_total} ({away_team} vs {home_team})"
                    _odds_price = total_odds if total_odds else -110
                    _prob = _ou_probability(float(total_val), float(vegas_total), _MLB_TOTALS_RMSE)
                    _b = abs(_odds_price) / 100 if _odds_price > 0 else 100 / abs(_odds_price or 110)
                    _implied_prob = 1 / (1 + _b)
                    _edge_prob = _prob - _implied_prob
                    _q = 1 - _prob
                    _k = max((_b * _prob - _q) / _b, 0.0)
                    _kf = round(_k * 0.25 * 100, 2)
                    ou_pick = {
                        "source": "MLB Model",
                        "pick": pick_label,
                        "sport": league,
                        "odds": _odds_price,
                        "units": 1,
                        "probability": _prob,
                        "prob": _prob,
                        "edge": round(_edge_prob * 100, 2),
                        "vegas": vegas_total,
                        "model_prediction": round(float(total_val), 1),
                        "direction": direction,
                        "kelly": _kf,
                        "decision": 'BET' if _edge_prob >= 0.05 else ('LEAN' if _edge_prob >= 0.03 else 'PASS'),
                        "market_type": "totals",
                        "selection": direction,
                        "line": vegas_total,
                        "market_line": vegas_total,
                        "away_team": away_team,
                        "home_team": home_team,
                    }
                    picks.append(ou_pick)

        if game_m:
            current_away = ""
            current_home = ""

    return picks


# ── Scores24 tip cleaning helpers ──────────────────────────────

_MULTI_WORD_NICKNAMES = {
    "trail blazers", "red sox", "white sox", "blue jays", "maple leafs",
}

_SCORES24_SPORT_ALIAS = {
    "NBA": "NBA",
    "NATIONAL BASKETBALL ASSOCIATION": "NBA",
    "NHL": "NHL",
    "NATIONAL HOCKEY LEAGUE": "NHL",
    "MLB": "MLB",
    "MAJOR LEAGUE BASEBALL": "MLB",
    "EPL": "EPL",
    "ENGLISH PREMIER LEAGUE": "EPL",
    "PREMIER LEAGUE": "EPL",
}


def _shorten_team(full_name: str) -> str:
    """'Philadelphia 76ers' → '76ers', 'Toronto Maple Leafs' → 'Maple Leafs'."""
    parts = full_name.strip().split()
    if len(parts) <= 1:
        return full_name.strip()
    last_two = " ".join(parts[-2:])
    if last_two.lower() in _MULTI_WORD_NICKNAMES:
        return last_two
    return parts[-1]


def _normalize_scores24_sport(raw_sport: str, fallback: str | None = None) -> str:
    raw = str(raw_sport or "").strip()
    norm = re.sub(r"\s+", " ", raw).strip().upper()
    mapped = _SCORES24_SPORT_ALIAS.get(norm)
    if mapped:
        return mapped
    if norm in {"OTHER", "BASKETBALL", "ICE-HOCKEY", "ICE HOCKEY", "BASEBALL", "SOCCER"} and fallback:
        return fallback
    return norm or (fallback or "Other")


def _strip_scores24_ot_qualifier(text: str) -> str:
    cleaned = str(text or "")
    # Remove bracketed OT inclusion notes such as "(inc. OT)", "(incl OT)", etc.
    cleaned = re.sub(
        r"\s*\((?=[^)]*\binc(?:l)?\.?\b)(?=[^)]*\bot\b)[^)]*\)\s*",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _scores24_matchup_candidates(matchup: str, sport: str) -> list[str]:
    teams = [part.strip() for part in re.split(r"\s+vs\s+", str(matchup or ""), maxsplit=1, flags=re.IGNORECASE) if part.strip()]
    if str(sport or "").upper() == "MLB":
        return [_norm_mlb(team) for team in teams]
    return teams


def _scores24_resolve_team_name(team_hint: str, matchup: str, sport: str) -> str:
    hint = re.sub(r"\s+", " ", str(team_hint or "")).strip()
    if not hint:
        return hint

    hint_norm = normalize(hint)
    candidates = _scores24_matchup_candidates(matchup, sport)
    if not candidates:
        return hint

    for full_team in candidates:
        full_norm = normalize(full_team)
        if not full_norm:
            continue
        if hint_norm == full_norm:
            return full_team
        if len(hint_norm) > 2 and (hint_norm in full_norm or full_norm in hint_norm):
            return full_team
        full_tokens = full_norm.split()
        hint_tokens = hint_norm.split()
        if hint_norm in full_tokens:
            return full_team
        if hint_tokens and full_tokens and hint_tokens[-1] == full_tokens[-1]:
            return full_team

    return hint


def _strip_scores24_trailing_context(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    while True:
        match = re.search(r"\s*\(([^()]*)\)\s*$", cleaned)
        if not match:
            break
        inner = re.sub(r"\s+", " ", match.group(1)).strip()
        if not re.search(r"\bvs\b|basketball|baseball|ice hockey|ice-hockey|soccer|american football|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}", inner, flags=re.IGNORECASE):
            break
        cleaned = cleaned[:match.start()].rstrip()
    return cleaned


def _is_scores24_team_hint(text: str) -> bool:
    hint = re.sub(r"\s+", " ", str(text or "")).strip()
    if not hint:
        return False
    if not re.search(r"[A-Za-z]", hint):
        return False
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", hint):
        return False
    hint_tokens = normalize(hint).split()
    generic_tokens = {"match", "game", "winner", "win", "moneyline", "ml", "home", "away", "draw", "team"}
    if hint_tokens and all(token in generic_tokens for token in hint_tokens):
        return False
    return True


def _is_generic_scores24_pick_text(text: str) -> bool:
    cleaned = normalize(str(text or ""))
    if not cleaned:
        return True
    generic_patterns = (
        r"^(?:match|game) ml$",
        r"^(?:winner|win|moneyline|ml)$",
        r"^(?:home|away|draw)(?: (?:team|win|winner|ml))?$",
        r"^both teams to score$",
        r"^handicap(?: [+-]?\d+(?:\.\d+)?)?$",
    )
    return any(re.fullmatch(pattern, cleaned) for pattern in generic_patterns)


def _clean_scores24_pick(tip: str, matchup: str, sport: str) -> str:
    """Convert raw Scores24 tip into clean format matching NBA/MLB model picks."""
    # Strip "at odds of ..." suffix from tip
    tip_clean = re.sub(r"\s*at odds of\s*[^\)]*\*?\s*$", "", tip).strip()
    tip_clean = _strip_scores24_ot_qualifier(tip_clean)
    tip_clean = re.sub(r"\s+", " ", tip_clean).strip().rstrip(".")
    tip_compact = _strip_scores24_trailing_context(tip_clean)

    # ── Pattern: "<Team> Handicap (<spread>)" ──
    m = re.match(r"^(.+?)\s+Handicap\s*\(([+-]?\d+\.?\d*)\)", tip_clean, re.IGNORECASE)
    if m and _is_scores24_team_hint(m.group(1)):
        team = _scores24_resolve_team_name(m.group(1), matchup, sport)
        spread = m.group(2)
        if not spread.startswith(("+", "-")):
            spread = "+" + spread
        return f"{team} {spread}"

    # ── Pattern: "<Team> +/-spread" (already spread-like text) ──
    m = re.match(r"^(.+?)\s+([+-]\d+\.?\d*)\b", tip_compact, re.IGNORECASE)
    if m and _is_scores24_team_hint(m.group(1)):
        team = _scores24_resolve_team_name(m.group(1), matchup, sport)
        return f"{team} {m.group(2)}"

    # ── Pattern: "<Team> Total goals/points Over/Under (<value>)" (team total) ──
    m = re.match(
        r"^(.+?)\s+Total\s+(goals|points)\s+(Over|Under)\s*\((\d+\.?\d*)\)",
        tip_clean, re.IGNORECASE,
    )
    if m:
        team = _scores24_resolve_team_name(m.group(1), matchup, sport)
        kind = m.group(2).lower()
        direction = m.group(3).title()
        value = m.group(4)
        suffix = " TG" if kind == "goals" else ""
        return f"{team} {direction} {value}{suffix}"

    # ── Pattern: "Total goals/points Over/Under (<value>)" (game total) ──
    m = re.match(
        r"^Total\s+(goals|points)\s+(Over|Under)\s*\((\d+\.?\d*)\)",
        tip_clean, re.IGNORECASE,
    )
    if m:
        kind = m.group(1).lower()
        direction = m.group(2).title()
        value = m.group(3)
        suffix = " TG" if kind == "goals" else ""
        return f"{direction} {value}{suffix}"

    # ── Pattern: "Total Over/Under (<value>)" (generic total) ──
    m = re.match(r"^Total\s+(Over|Under)\s*\((\d+\.?\d*)\)", tip_clean, re.IGNORECASE)
    if m:
        return f"{m.group(1).title()} {m.group(2)}"

    # ── Pattern: "Over/Under X" with optional trailing matchup noise ──
    m = re.match(r"^(Over|Under)\s+(\d+\.?\d*)\b", tip_compact, re.IGNORECASE)
    if m:
        return f"{m.group(1).title()} {m.group(2)}"

    # ── Pattern: "Both Teams To Score (Yes/No)" with optional period prefix ──
    m = re.match(
        r"^(?:.*?,\s*)?Both\s+Teams?\s+To\s+Score\s*\((Yes|No)\)",
        tip_clean, re.IGNORECASE,
    )
    if m:
        answer = m.group(1)
        return f"BTTS {answer}"

    # ── Pattern: "<Team> Win (...)" → moneyline ──
    m = re.match(r"^(.+?)\s+Win(?:\s*\([^)]*\))?$", tip_clean, re.IGNORECASE)
    if m and _is_scores24_team_hint(m.group(1)):
        team = _scores24_resolve_team_name(m.group(1), matchup, sport)
        return f"{team} ML"

    # ── Pattern: "<Team> to win" → moneyline ──
    m = re.match(r"^(.+?)\s+to\s+win$", tip_clean, re.IGNORECASE)
    if m and _is_scores24_team_hint(m.group(1)):
        team = _scores24_resolve_team_name(m.group(1), matchup, sport)
        return f"{team} ML"

    # ── Pattern: "<Team> ML" ──
    m = re.match(r"^(.+?)\s+ML$", tip_compact, re.IGNORECASE)
    if m and _is_scores24_team_hint(m.group(1)):
        team = _scores24_resolve_team_name(m.group(1), matchup, sport)
        return f"{team} ML"

    # ── Fallback: cleaned tip + shortened matchup ──
    return tip_compact or tip_clean


def _parse_scores24_matchup_teams(matchup: str, sport: str) -> tuple[str | None, str | None]:
    teams = re.split(r"\s+vs\s+", str(matchup or ""), maxsplit=1, flags=re.IGNORECASE)
    if len(teams) != 2:
        return None, None
    away_team = teams[0].strip()
    home_team = teams[1].strip()
    if str(sport or "").upper() == "MLB":
        away_team = _norm_mlb(away_team)
        home_team = _norm_mlb(home_team)
    return away_team or None, home_team or None


def _normalize_french_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().replace("’", "'")
    return re.sub(r"\s+", " ", normalized).strip()


_SPORTYTRADER_SPORT_ALIAS = {
    "USA - NBA": "NBA",
    "NBA": "NBA",
    "BASKETBALL": "NBA",
    "USA - MLB": "MLB",
    "MLB": "MLB",
    "BASEBALL": "MLB",
}


def _normalize_sportytrader_sport(raw_league: str, fallback: str | None = None) -> str:
    raw = re.sub(r"\s+", " ", str(raw_league or "")).strip().upper()
    mapped = _SPORTYTRADER_SPORT_ALIAS.get(raw)
    if mapped:
        return mapped
    return fallback or (raw if raw else "Other")


def _clean_sportytrader_pick(tip: str, matchup: str, sport: str = "NBA") -> str:
    """Convert SportyTrader picks into the same pick format used in the UI."""
    tip_clean = re.sub(r"\s+", " ", str(tip or "")).strip().rstrip(".")
    teams = matchup.split(" vs ")
    home = teams[0].strip() if len(teams) > 0 else ""
    away = teams[1].strip() if len(teams) > 1 else ""
    home_short = _shorten_team(home)
    away_short = _shorten_team(away)
    matchup_short = f"{home_short} vs {away_short}"

    tip_norm = _normalize_french_text(tip_clean)
    home_norm = _normalize_french_text(home)
    away_norm = _normalize_french_text(away)
    home_short_norm = _normalize_french_text(home_short)
    away_short_norm = _normalize_french_text(away_short)

    def _team_tokens(full_name_norm: str, short_name_norm: str) -> set[str]:
        tokens = set()
        for token in re.split(r"\s+", full_name_norm):
            if len(token) >= 3 and token not in {"les", "the"}:
                tokens.add(token)
        if short_name_norm:
            tokens.add(short_name_norm)
        return tokens

    home_tokens = _team_tokens(home_norm, home_short_norm)
    away_tokens = _team_tokens(away_norm, away_short_norm)

    def _resolve_team_name(team_hint: str = "") -> str:
        lookup_norm = _normalize_french_text(team_hint or tip_clean)
        if home_norm and home_norm in lookup_norm:
            return home_short
        if away_norm and away_norm in lookup_norm:
            return away_short
        if home_short_norm and home_short_norm in lookup_norm:
            return home_short
        if away_short_norm and away_short_norm in lookup_norm:
            return away_short
        home_hits = sum(1 for token in home_tokens if re.search(rf"\b{re.escape(token)}\b", lookup_norm))
        away_hits = sum(1 for token in away_tokens if re.search(rf"\b{re.escape(token)}\b", lookup_norm))
        if away_hits > home_hits:
            return away_short
        if home_hits > away_hits:
            return home_short
        return home_short

    # English totals, e.g. "Over 229.5" or "Under 7 Runs".
    m = re.match(r"^(?:The\s+)?(?:Total\s+)?(Over|Under)\s+(\d+\.?\d*)\s*(?:points?|runs?)?$", tip_clean, re.IGNORECASE)
    if m:
        return f"{m.group(1).title()} {m.group(2)} ({matchup_short})"

    # English moneyline, e.g. "Milwaukee to win" or "The Reds will win".
    m = re.match(r"^(?:The\s+)?(.+?)\s+(?:to|will)\s+win$", tip_clean, re.IGNORECASE)
    if m:
        team = _resolve_team_name(m.group(1))
        return f"{team} ML ({matchup_short})"

    # English spread/run line, e.g. "The Cubs -1.5 Runs".
    m = re.match(r"^(?:The\s+)?(.+?)\s+([+-]\d+\.?\d*)\s*(?:points?|runs?)?$", tip_clean, re.IGNORECASE)
    if m:
        team = _resolve_team_name(m.group(1))
        return f"{team} {m.group(2)} ({matchup_short})"

    # English runline phrasing, e.g. "Seattle Mariners to cover the -1.5 runline."
    m = re.match(r"^(?:The\s+)?(.+?)\s+to\s+cover\s+the\s+([+-]\d+\.?\d*)\s+runline$", tip_clean, re.IGNORECASE)
    if m:
        team = _resolve_team_name(m.group(1))
        return f"{team} {m.group(2)} ({matchup_short})"

    # "<Team> gagne (prolongations incluses)" -> Moneyline
    if re.search(r"\bgagne\b", tip_norm) and not re.search(r"\bpoints?\b", tip_norm):
        team = _resolve_team_name()
        return f"{team} ML ({matchup_short})"

    # "<Team> gagne par X points d'ecart ou plus" / "par au moins X points"
    m = re.search(r"\bgagne\b.*?\b(?:par|au moins)\s+(\d+)\s+points?", tip_norm)
    if m:
        margin = int(m.group(1))
        spread = -(margin - 0.5)
        team = _resolve_team_name()
        spread_text = f"{spread:.1f}".rstrip("0").rstrip(".")
        return f"{team} {spread_text} ({matchup_short})"

    # "<Team> ne perd pas / ne perdent pas ... X points ou moins"
    m = re.search(r"\bne perd(?:ent)? pas\b.*?(\d+)\s+points?", tip_norm)
    if m:
        margin = int(m.group(1))
        spread = margin + 0.5
        team = _resolve_team_name()
        spread_text = f"+{spread:.1f}".rstrip("0").rstrip(".")
        return f"{team} {spread_text} ({matchup_short})"

    # "<Team> par au moins X points"
    m = re.search(r"\bpar au moins\s+(\d+)\s+points?", tip_norm)
    if m:
        margin = int(m.group(1))
        spread = -(margin - 0.5)
        team = _resolve_team_name()
        spread_text = f"{spread:.1f}".rstrip("0").rstrip(".")
        return f"{team} {spread_text} ({matchup_short})"

    # Keep explicit totals readable for either sport when no team parsing matched.
    if sport.upper() in {"NBA", "MLB"}:
        m = re.match(r"^(Over|Under)\s+(\d+\.?\d*)$", tip_clean, re.IGNORECASE)
        if m:
            return f"{m.group(1).title()} {m.group(2)} ({matchup_short})"

    return f"{tip_clean} ({matchup_short})"


def _parse_scores24_output(output: str) -> list[dict[str, Any]]:
    """Parse Scores24 scraper stdout into pick dicts."""
    picks: list[dict[str, Any]] = []

    # Split by the ━━━ separators (preferred path)
    blocks = re.split(r"━{10,}", output)

    for block in blocks:
        match_m = re.search(r"Match:\s*(.+)", block)
        tip_m = re.search(r"Tip:\s*(.+)", block)
        odds_m = re.search(r"Odds:\s*(.+)", block)
        conf_m = re.search(r"Confidence:\s*(.+)", block)
        league_m = re.search(r"League:\s*(.+)", block)

        if not match_m or not tip_m:
            continue

        matchup = match_m.group(1).strip()
        tip = tip_m.group(1).strip()

        if not tip or tip == "[not found on page]":
            continue

        odds_str = odds_m.group(1).strip() if odds_m else ""
        confidence = conf_m.group(1).strip() if conf_m else ""
        league = league_m.group(1).strip() if league_m else ""
        sport = _normalize_scores24_sport(league)

        # Parse odds from Odds: field
        odds_val = None
        if odds_str and odds_str != "[not found on page]":
            try:
                odds_val = int(float(odds_str.replace("+", "").replace("*", "")))
            except ValueError:
                odds_val = None

        # Extract odds embedded in tip text (e.g. "at odds of -204*")
        tip_odds_m = re.search(r"at odds of ([+-]?\d+)\*?", tip)
        if tip_odds_m:
            try:
                odds_val = int(tip_odds_m.group(1))
            except ValueError:
                pass

        # Parse confidence to float
        conf_val = None
        if confidence and confidence != "[not found on page]":
            conf_num = re.search(r"(\d+)", confidence)
            if conf_num:
                conf_val = int(conf_num.group(1))

        # Clean the tip and build proper pick text
        pick_text = _strip_scores24_ot_qualifier(_clean_scores24_pick(tip, matchup, sport))
        if _is_generic_scores24_pick_text(pick_text):
            continue
        away_team, home_team = _parse_scores24_matchup_teams(matchup, sport)

        picks.append({
            "source": "Scores24",
            "pick": pick_text,
            "matchup": matchup,
            "tip": tip,
            "sport": sport,
            "odds": odds_val,
            "units": 1,
            "probability": conf_val / 100 if conf_val else None,
            "edge": None,
            "decision": "BET",  # All Scores24 tips are presented as BET
            "away_team": away_team,
            "home_team": home_team,
        })

    if picks:
        return picks

    # Fallback path: parse repeated field sections even when separator glyphs
    # are missing/normalized in subprocess output.
    lines = [ln.rstrip("\n") for ln in output.splitlines()]
    chunk: list[str] = []
    chunks: list[str] = []
    for ln in lines:
        if ln.strip().startswith("Match:") and chunk:
            chunks.append("\n".join(chunk))
            chunk = [ln]
            continue
        if chunk or ln.strip().startswith("Match:"):
            chunk.append(ln)
    if chunk:
        chunks.append("\n".join(chunk))

    for block in chunks:
        match_m = re.search(r"Match:\s*(.+)", block)
        tip_m = re.search(r"Tip:\s*(.+)", block)
        odds_m = re.search(r"Odds:\s*(.+)", block)
        conf_m = re.search(r"Confidence:\s*(.+)", block)
        league_m = re.search(r"League:\s*(.+)", block)

        if not match_m or not tip_m:
            continue

        matchup = match_m.group(1).strip()
        tip = tip_m.group(1).strip()
        if not tip or tip == "[not found on page]":
            continue

        odds_str = odds_m.group(1).strip() if odds_m else ""
        confidence = conf_m.group(1).strip() if conf_m else ""
        league = league_m.group(1).strip() if league_m else ""
        sport = _normalize_scores24_sport(league)

        odds_val = None
        if odds_str and odds_str != "[not found on page]":
            try:
                odds_val = int(float(odds_str.replace("+", "").replace("*", "")))
            except ValueError:
                odds_val = None

        tip_odds_m = re.search(r"at odds of ([+-]?\d+)\*?", tip)
        if tip_odds_m:
            try:
                odds_val = int(tip_odds_m.group(1))
            except ValueError:
                pass

        conf_val = None
        if confidence and confidence != "[not found on page]":
            conf_num = re.search(r"(\d+)", confidence)
            if conf_num:
                conf_val = int(conf_num.group(1))
        pick_text = _strip_scores24_ot_qualifier(_clean_scores24_pick(tip, matchup, sport))
        if _is_generic_scores24_pick_text(pick_text):
            continue
        away_team, home_team = _parse_scores24_matchup_teams(matchup, sport)

        picks.append({
            "source": "Scores24",
            "pick": pick_text,
            "matchup": matchup,
            "tip": tip,
            "sport": sport,
            "odds": odds_val,
            "units": 1,
            "probability": conf_val / 100 if conf_val else None,
            "edge": None,
            "decision": "BET",
            "away_team": away_team,
            "home_team": home_team,
        })

    return picks


def _nba_model_extra_args(date_str: str | None = None, variant: str = "new") -> list[str]:
    target_iso, _ = _parse_model_date_arg(date_str)
    args = ["--date", target_iso, "--variant", variant]
    if variant != "new" or target_iso != datetime.now().strftime("%Y-%m-%d"):
        args.append("--no-log")
    return args


def run_nba_model(date_str: str | None = None, variant: str = "new") -> dict[str, Any]:
    """Execute an NBA model variant and return parsed picks."""
    python_bin = _resolve_python_bin(os.path.join(NBA_MODEL_DIR, "venv", "bin", "python"))
    source_label = "NBA New" if variant == "new" else "NBA Model"

    try:
        output = _run_script(
            python_bin,
            "run_live.py",
            NBA_MODEL_DIR,
            timeout=300,
            extra_args=_nba_model_extra_args(date_str, variant),
        )
        if "Traceback (most recent call last)" in output or "ModuleNotFoundError" in output:
            tail = " | ".join((output.strip().splitlines() or ["no output"])[-12:])
            return {"ok": False, "error": f"{source_label} runtime failed ({tail})"}

        picks = _parse_nba_output(output, source_label=source_label)
        if not picks:
            if "No games found for today." in output:
                return {
                    "ok": True,
                    "picks": [],
                    "raw_lines": len(output.split("\n")),
                    "note": f"No NBA games found for requested date ({source_label})",
                }
            tail = " | ".join((output.strip().splitlines() or ["no output"])[-12:])
            return {
                "ok": False,
                "error": f"{source_label} parser found no predictions ({tail})",
                "raw_lines": len(output.split("\n")),
            }

        return {"ok": True, "picks": picks, "raw_lines": len(output.split("\n"))}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"{source_label} timed out (5 min limit)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_nba_props_model(
    date_str: str | None = None,
    game_id: str | None = None,
    game_label: str | None = None,
) -> dict[str, Any]:
    """Execute the NBA props model and return parsed picks."""
    python_bin = _resolve_python_bin(os.path.join(NBA_PROPS_MODEL_DIR, "venv", "bin", "python"))

    extra = []
    if date_str:
        extra = [date_str]
    selected_game_id = str(game_id or "").strip()
    selected_game_label = str(game_label or "").strip()

    if selected_game_label and not selected_game_id:
        try:
            list_output = _run_script(
                python_bin,
                "run_props.py",
                NBA_PROPS_MODEL_DIR,
                timeout=180,
                extra_args=extra + ["--list-game-ids"],
            )
            if "Traceback (most recent call last)" in list_output or "ModuleNotFoundError" in list_output:
                tail = " | ".join((list_output.strip().splitlines() or ["no output"])[-12:])
                return {"ok": False, "error": f"NBA props model runtime failed ({tail})"}
            games = _extract_nba_props_games(list_output)
            target = selected_game_label.casefold()
            for game in games:
                label = str(game.get("label") or "").strip()
                if label and label.casefold() == target:
                    selected_game_id = str(game.get("game_id") or "").strip()
                    break
            if not selected_game_id:
                return {"ok": False, "error": "Selected game is not available in today's NBA slate"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "NBA props game lookup timed out"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    if selected_game_id:
        try:
            chunk_output = _run_script(
                python_bin,
                "run_props.py",
                NBA_PROPS_MODEL_DIR,
                timeout=180,
                extra_args=extra + [f"--game-ids={selected_game_id}"],
            )
            if "Traceback (most recent call last)" in chunk_output or "ModuleNotFoundError" in chunk_output:
                tail = " | ".join((chunk_output.strip().splitlines() or ["no output"])[-12:])
                return {"ok": False, "error": f"NBA props model runtime failed ({tail})"}
            picks = _parse_nba_props_output(chunk_output)
            if not picks:
                return {
                    "ok": True,
                    "picks": [],
                    "raw_lines": len(chunk_output.split("\n")),
                    "note": "No NBA props candidates found for selected game",
                }
            return {"ok": True, "picks": picks, "raw_lines": len(chunk_output.split("\n"))}
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "NBA props model timed out (per-game run limit)"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    try:
        list_output = _run_script(
            python_bin,
            "run_props.py",
            NBA_PROPS_MODEL_DIR,
            timeout=180,
            extra_args=extra + ["--list-game-ids"],
        )
        if "Traceback (most recent call last)" in list_output or "ModuleNotFoundError" in list_output:
            tail = " | ".join((list_output.strip().splitlines() or ["no output"])[-12:])
            return {"ok": False, "error": f"NBA props model runtime failed ({tail})"}

        game_ids = _extract_nba_props_game_ids(list_output)
        if not game_ids:
            if "No NBA games found for today." in list_output:
                return {
                    "ok": True,
                    "picks": [],
                    "raw_lines": len(list_output.split("\n")),
                    "note": "No NBA props candidates found today",
                }
            output = _run_script(python_bin, "run_props.py", NBA_PROPS_MODEL_DIR, timeout=300, extra_args=extra)
            if "Traceback (most recent call last)" in output or "ModuleNotFoundError" in output:
                tail = " | ".join((output.strip().splitlines() or ["no output"])[-12:])
                return {"ok": False, "error": f"NBA props model runtime failed ({tail})"}
            picks = _parse_nba_props_output(output)
            if not picks:
                if (
                    "No NBA games found for today." in output
                    or "No qualifying player props candidates" in output
                ):
                    return {
                        "ok": True,
                        "picks": [],
                        "raw_lines": len(output.split("\n")),
                        "note": "No NBA props candidates found today",
                    }
                tail = " | ".join((output.strip().splitlines() or ["no output"])[-12:])
                return {
                    "ok": False,
                    "error": f"NBA props parser found no predictions ({tail})",
                    "raw_lines": len(output.split("\n")),
                }
            return {"ok": True, "picks": picks, "raw_lines": len(output.split("\n"))}

        all_picks: list[dict[str, Any]] = []
        raw_lines = len(list_output.split("\n"))
        for game_id in game_ids:
            chunk_output = _run_script(
                python_bin,
                "run_props.py",
                NBA_PROPS_MODEL_DIR,
                timeout=180,
                extra_args=extra + [f"--game-ids={game_id}"],
            )
            raw_lines += len(chunk_output.split("\n"))
            if "Traceback (most recent call last)" in chunk_output or "ModuleNotFoundError" in chunk_output:
                tail = " | ".join((chunk_output.strip().splitlines() or ["no output"])[-12:])
                return {"ok": False, "error": f"NBA props model runtime failed ({tail})"}
            all_picks.extend(_parse_nba_props_output(chunk_output))

        if not all_picks:
            return {
                "ok": True,
                "picks": [],
                "raw_lines": raw_lines,
                "note": "No NBA props candidates found today",
            }

        return {"ok": True, "picks": all_picks, "raw_lines": raw_lines}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "NBA props model timed out (per-game run limit)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def run_mlb_model(date_str: str | None = None) -> dict[str, Any]:
    """Execute the MLB model and return parsed picks."""
    python_bin = _resolve_python_bin(os.path.join(MLB_MODEL_DIR, "venv", "bin", "python"))

    extra = []
    if date_str:
        extra = [date_str]

    try:
        output = _run_script(python_bin, "run_today.py", MLB_MODEL_DIR, timeout=300, extra_args=extra)
        picks = _parse_mlb_output(output)
        return {"ok": True, "picks": picks, "raw_lines": len(output.split("\n"))}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "MLB model timed out (5 min limit)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _resolve_scores24_date(date_str: str | None) -> str:
    """Normalize incoming date to YYYY-MM-DD for scores24_scraper.py."""
    if date_str:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
            try:
                return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
            except ValueError:
                pass
    return datetime.now().strftime("%Y-%m-%d")


def _write_scores24_manual_feed(
    picks: list[dict[str, Any]],
    date_str: str,
    sports: list[str] | None = None,
    note: str | None = None,
) -> None:
    feed_path = os.path.join(BASE_DIR, "scores24_manual_feed.json")
    requested = {
        _normalize_scores24_sport(str(s or "").strip(), str(s or "").strip().upper())
        for s in (sports or [])
        if str(s or "").strip()
    }
    requested = {sport for sport in requested if sport}

    existing_payload: dict[str, Any] = {}
    existing_picks: list[dict[str, Any]] = []
    try:
        with open(feed_path, encoding="utf-8") as fh:
            existing_payload = json.load(fh)
        if isinstance(existing_payload.get("picks"), list):
            existing_picks = [pick for pick in existing_payload["picks"] if isinstance(pick, dict)]
    except Exception:
        existing_payload = {}
        existing_picks = []

    keep_existing = str(existing_payload.get("date") or "").strip() == date_str
    merged_picks: list[dict[str, Any]] = []
    if keep_existing and requested:
        for pick in existing_picks:
            sport = _normalize_scores24_sport(str(pick.get("sport", "")), str(pick.get("sport", "")))
            if sport in requested:
                continue
            merged_picks.append(pick)
    merged_picks.extend(picks)

    payload = {
        "updated_at": datetime.now().isoformat(),
        "date": date_str,
        "leagues": ",".join(sorted(requested)).lower() if requested else "",
        "note": note or "Synced from local backend.",
        "picks": merged_picks,
    }
    with open(feed_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def run_scores24_scraper(sports: list[str], date_str: str | None = None) -> dict[str, Any]:
    """Execute the Scores24 scraper for selected sports."""
    python_bin = _resolve_python_bin(SCORES24_VENV)

    sport_map = {
        "nba": "nba",
        "nhl": "nhl",
        "mlb": "mlb",
    }
    sport_tag_map = {
        "nba": "NBA",
        "nhl": "NHL",
        "mlb": "MLB",
    }

    selected = [str(s).lower().strip() for s in sports if str(s).strip()]
    if not selected:
        selected = ["nba"]

    all_picks: list[dict[str, Any]] = []
    errors: list[str] = []
    target_date = _resolve_scores24_date(date_str)
    scraper_path = os.path.join(BASE_DIR, "scores24_scraper.py")
    if not os.path.exists(scraper_path):
        return {"ok": False, "error": f"scores24 scraper not found at {scraper_path}"}

    def _run_one_sport(sport_code: str) -> tuple[str, list[dict[str, Any]], str | None]:
        sport_slug = sport_map.get(sport_code)
        if not sport_slug:
            return sport_code, [], f"Unsupported sport code: {sport_code}"

        timeout_s = 300 if sport_code == "mlb" else 180
        date_candidates = [target_date]
        try:
            base_date = datetime.strptime(target_date, "%Y-%m-%d")
            fallback_prev = (base_date - timedelta(days=1)).strftime("%Y-%m-%d")
            fallback_next = (base_date + timedelta(days=1)).strftime("%Y-%m-%d")
            for candidate in (fallback_prev, fallback_next):
                if candidate not in date_candidates:
                    date_candidates.append(candidate)
        except ValueError:
            pass

        def _invoke(env: dict[str, str], scrape_date: str) -> subprocess.CompletedProcess[str]:
            return _subprocess_run(
                [python_bin, scraper_path, "--sport", sport_slug, "--date", scrape_date],
                cwd=BASE_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )

        try:
            env = os.environ.copy()
            env.setdefault("PLAYWRIGHT_BROWSERS_PATH", _default_playwright_browsers_path())
            result = _invoke(env, date_candidates[0])
            output = (result.stdout or "") + (result.stderr or "")

            # Auto-heal missing browser installs in long-lived Render instances.
            if result.returncode != 0 and _looks_like_playwright_browser_missing(output):
                ok, install_msg = _ensure_playwright_browsers(python_bin, env)
                if not ok:
                    return sport_code, [], f"{sport_code}: Playwright install failed ({install_msg})"
                result = _invoke(env, date_candidates[0])
                output = (result.stdout or "") + (result.stderr or "")

            fallback_tag = sport_tag_map.get(sport_code)

            def _normalize_and_filter(picks_in: list[dict[str, Any]]) -> list[dict[str, Any]]:
                if not fallback_tag:
                    return picks_in
                kept: list[dict[str, Any]] = []
                for pick in picks_in:
                    normalized = _normalize_scores24_sport(pick.get("sport", ""), fallback_tag)
                    if normalized != fallback_tag:
                        continue
                    pick["sport"] = normalized
                    kept.append(pick)
                return kept

            picks = _normalize_and_filter(_parse_scores24_output(output))

            # Scores24 listing pages can return transient timeout/protection pages.
            # Retry the same request a couple of times before declaring no picks.
            if result.returncode == 0 and not picks and _looks_like_transient_scores24_listing_failure(output):
                for attempt in range(2):
                    time.sleep(1.5 + attempt)
                    retry_same = _invoke(env, date_candidates[0])
                    retry_same_output = (retry_same.stdout or "") + (retry_same.stderr or "")
                    retry_same_picks = _normalize_and_filter(_parse_scores24_output(retry_same_output))
                    result = retry_same
                    output = retry_same_output
                    picks = retry_same_picks
                    if retry_same.returncode == 0 and retry_same_picks:
                        break

            # Around date boundaries, listings may shift by one day depending on
            # league timezone and deployment timezone. Retry adjacent dates.
            if result.returncode == 0 and not picks and len(date_candidates) > 1:
                for retry_date in date_candidates[1:]:
                    retry = _invoke(env, retry_date)
                    retry_output = (retry.stdout or "") + (retry.stderr or "")
                    retry_picks = _normalize_and_filter(_parse_scores24_output(retry_output))
                    if retry.returncode == 0 and retry_picks:
                        result = retry
                        output = retry_output
                        picks = retry_picks
                        break

            if result.returncode != 0 and not picks:
                msg = _compact_error_text(output)
                return sport_code, [], f"{sport_code}: scraper exited {result.returncode} ({msg})"

            if not picks:
                compact = _compact_error_text(output)
                return sport_code, [], f"{sport_code}: no picks parsed ({compact})"

            return sport_code, picks, None
        except subprocess.TimeoutExpired:
            return sport_code, [], f"{sport_code}: timed out after {timeout_s}s"
        except Exception as exc:
            return sport_code, [], f"{sport_code}: {exc}"

    # Run selected sports in parallel so one slow league doesn't block all others.
    max_workers = max(1, min(len(selected), 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_one_sport, sport_code) for sport_code in selected]
        for future in as_completed(futures):
            sport_code, picks, err = future.result()
            if err:
                errors.append(err)
            if picks:
                all_picks.extend(picks)

    # If everything failed, surface why instead of silently returning 0 picks.
    if not all_picks and errors:
        return {"ok": False, "error": "; ".join(errors[:4])}

    try:
        _write_scores24_manual_feed(
            all_picks,
            target_date,
            selected,
            note="Synced from local backend.",
        )
    except Exception as exc:
        errors.append(f"feed write failed: {exc}")
    return {"ok": True, "picks": all_picks, "errors": errors}


def run_sportytrader_scraper(
    date_str: str | None = None,
    sports: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the SportyTrader scraper for NBA and/or MLB."""
    python_bin = _resolve_python_bin(SPORTYTRADER_VENV)
    target_date = _resolve_scores24_date(date_str)
    scraper_path = os.path.join(BASE_DIR, "sportytrader_scraper.py")
    if not os.path.exists(scraper_path):
        return {"ok": False, "error": f"sportytrader scraper not found at {scraper_path}"}

    timeout_s = 120
    env = os.environ.copy()
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", _default_playwright_browsers_path())

    sport_map = {
        "nba": "nba",
        "basketball": "nba",
        "mlb": "mlb",
        "baseball": "mlb",
    }
    default_sports = ["nba", "mlb"]
    selected = [sport_map.get(str(s).strip().lower(), "") for s in (sports or default_sports)]
    selected = [sport for sport in selected if sport]
    if not selected:
        selected = default_sports

    def _invoke(sport_code: str) -> subprocess.CompletedProcess[str]:
        return _subprocess_run(
            [python_bin, scraper_path, "--sport", sport_code, "--date", target_date],
            cwd=BASE_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    try:
        all_picks: list[dict[str, Any]] = []
        errors: list[str] = []

        for sport_code in selected:
            result = _invoke(sport_code)
            output = (result.stdout or "") + (result.stderr or "")
            if result.returncode != 0 and _looks_like_playwright_browser_missing(output):
                ok, install_msg = _ensure_playwright_browsers(python_bin, env)
                if not ok:
                    return {"ok": False, "error": f"sportytrader: Playwright install failed ({install_msg})"}
                result = _invoke(sport_code)
                output = (result.stdout or "") + (result.stderr or "")

            picks: list[dict[str, Any]] = []
            blocks = re.split(r"━{10,}", output)
            expected_sport = sport_code.upper()
            for block in blocks:
                match_m = re.search(r"Match:\s*(.+)", block)
                tip_m = re.search(r"Tip:\s*(.+)", block)
                odds_m = re.search(r"Odds:\s*(.+)", block)
                league_m = re.search(r"League:\s*(.+)", block)
                if not match_m or not tip_m:
                    continue

                matchup = match_m.group(1).strip()
                tip = tip_m.group(1).strip()
                if not matchup or not tip:
                    continue
                league = league_m.group(1).strip() if league_m else ""
                sport = _normalize_sportytrader_sport(league, expected_sport)
                if sport != expected_sport:
                    continue

                odds_val = None
                odds_str = odds_m.group(1).strip() if odds_m else ""
                if odds_str and odds_str != "[not found on page]":
                    try:
                        odds_val = int(float(odds_str))
                    except ValueError:
                        odds_val = None

                picks.append({
                    "source": "SportyTrader",
                    "pick": _clean_sportytrader_pick(tip, matchup, sport=sport),
                    "sport": sport,
                    "odds": odds_val,
                    "units": 1,
                    "probability": None,
                    "edge": None,
                    "decision": "BET",
                })

            if result.returncode != 0 and not picks:
                errors.append(f"{sport_code}: scraper exited {result.returncode} ({_compact_error_text(output)})")
                continue
            if not picks:
                errors.append(f"{sport_code}: no picks parsed ({_compact_error_text(output)})")
                continue

            all_picks.extend(picks)

        if not all_picks and errors:
            return {"ok": False, "error": "; ".join(errors[:4])}

        return {"ok": True, "picks": all_picks, "errors": errors}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"sportytrader: timed out after {timeout_s}s"}
    except Exception as exc:
        return {"ok": False, "error": f"sportytrader: {exc}"}


def run_sportsgambler_scraper(
    date_str: str | None = None,
    sports: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the SportsGambler scraper for NBA and/or MLB."""
    python_bin = _resolve_python_bin(SPORTSGAMBLER_VENV)
    target_date = _resolve_scores24_date(date_str)
    scraper_path = os.path.join(BASE_DIR, "sportsgambler_scraper.py")
    if not os.path.exists(scraper_path):
        return {"ok": False, "error": f"sportsgambler scraper not found at {scraper_path}"}

    timeout_s = 120
    sport_map = {
        "nba": "nba",
        "basketball": "nba",
        "mlb": "mlb",
        "baseball": "mlb",
    }
    default_sports = ["nba", "mlb"]
    selected = [sport_map.get(str(s).strip().lower(), "") for s in (sports or default_sports)]
    selected = [sport for sport in selected if sport]
    if not selected:
        selected = default_sports

    def _invoke(sport_code: str) -> subprocess.CompletedProcess[str]:
        return _subprocess_run(
            [python_bin, scraper_path, "--sport", sport_code, "--date", target_date],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    try:
        all_picks: list[dict[str, Any]] = []
        errors: list[str] = []

        for sport_code in selected:
            result = _invoke(sport_code)
            output = (result.stdout or "") + (result.stderr or "")

            picks: list[dict[str, Any]] = []
            blocks = re.split(r"━{10,}", output)
            expected_sport = sport_code.upper()
            for block in blocks:
                match_m = re.search(r"Match:\s*(.+)", block)
                tip_m = re.search(r"Tip:\s*(.+)", block)
                odds_m = re.search(r"Odds:\s*(.+)", block)
                league_m = re.search(r"League:\s*(.+)", block)
                if not match_m or not tip_m:
                    continue

                matchup = match_m.group(1).strip()
                tip = tip_m.group(1).strip()
                if not matchup or not tip:
                    continue

                league = league_m.group(1).strip() if league_m else ""
                sport = (league or expected_sport).upper()
                if sport not in {"NBA", "MLB"}:
                    sport = expected_sport

                odds_val = None
                odds_str = odds_m.group(1).strip() if odds_m else ""
                if odds_str and odds_str != "[not found on page]":
                    try:
                        odds_val = int(float(odds_str))
                    except ValueError:
                        odds_val = None

                picks.append({
                    "source": "SportsGambler",
                    "pick": f"{tip} ({matchup})",
                    "sport": sport,
                    "odds": odds_val,
                    "units": 1,
                    "probability": None,
                    "edge": None,
                    "decision": "BET",
                })

            if result.returncode != 0 and not picks:
                errors.append(f"{sport_code}: scraper exited {result.returncode} ({_compact_error_text(output)})")
                continue
            if not picks:
                errors.append(f"{sport_code}: no picks parsed ({_compact_error_text(output)})")
                continue

            all_picks.extend(picks)

        if not all_picks and errors:
            return {"ok": False, "error": "; ".join(errors[:4])}

        return {"ok": True, "picks": all_picks, "errors": errors}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"sportsgambler: timed out after {timeout_s}s"}
    except Exception as exc:
        return {"ok": False, "error": f"sportsgambler: {exc}"}


def _launch_job(target_fn, *args) -> str:
    """Launch a model run in a background thread and return a job_id."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "result": None}

    def _worker():
        result = target_fn(*args)
        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "result": result}

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return job_id


def _public_endpoints() -> list[str]:
    endpoints = [
        "/health",
        "/ledger-state",
        "/picks",
        "/grade",
        "/run-sportsline-odds",
        "/run-nba-model",
        "/run-nba-old-model",
        "/refresh-nba-props-games",
        "/run-nba-props-model",
        "/run-mlb-model",
        "/api/ipl",
        "/ask-opus",
        "/job-status?id=<id>",
        "/run-sportsgambler",
    ]
    if ENABLE_SCORES24_REMOTE:
        endpoints.append("/run-scores24")
    if ENABLE_SPORTYTRADER_REMOTE:
        endpoints.append("/run-sportytrader")
    return endpoints


def run_sportsline_odds(league: str = "NBA") -> dict[str, Any]:
    """Run the SportsLine odds scraper to refresh cbs_odds table."""
    import subprocess as _sp
    import re as _re

    scraper_path = os.path.join(BASE_DIR, "NBAPredictionModel", "cbs_odds_scraper.py")
    if not os.path.exists(scraper_path):
        return {"ok": False, "error": f"SportsLine scraper not found at {scraper_path}"}
    python_bin = _resolve_python_bin(
        os.path.join(BASE_DIR, "NBAPredictionModel", "venv", "bin", "python")
    )
    try:
        result = _sp.run(
            [python_bin, scraper_path],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=120,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        rows_saved = 0
        m = _re.search(r"Saved (\d+)", stdout)
        if m:
            rows_saved = int(m.group(1))
        if result.returncode != 0:
            return {"ok": False, "error": stderr or stdout or "scraper exited non-zero"}
        return {"ok": True, "rows_saved": rows_saved, "output": stdout}
    except _sp.TimeoutExpired:
        return {"ok": False, "error": "SportsLine scraper timed out (120s)"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


class Handler(BaseHTTPRequestHandler):
    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if extra_headers:
            for key, value in extra_headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_json(200, {"ok": True})

    def do_GET(self) -> None:  # noqa: N802
        raw_path = self.path
        parsed = urlparse(raw_path)
        path = parsed.path or raw_path
        if path == "/api":
            path = "/"
        elif path.startswith("/api/"):
            path = path[4:]

        if path == "/":
            self._send_json(200, {
                "ok": True,
                "service": "pickledger-grader",
                "status": "healthy",
                "anthropic_enabled": bool(ANTHROPIC_API_KEY),
                "anthropic_model": ANTHROPIC_MODEL,
                "scores24_remote_enabled": ENABLE_SCORES24_REMOTE,
                "playwright_proxy_configured": PLAYWRIGHT_PROXY_CONFIGURED,
                "sportytrader_remote_enabled": ENABLE_SPORTYTRADER_REMOTE,
                "endpoints": _public_endpoints(),
            })
            return

        if path == "/health":
            self._send_json(200, {
                "ok": True,
                "status": "healthy",
                "anthropic_enabled": bool(ANTHROPIC_API_KEY),
                "anthropic_model": ANTHROPIC_MODEL,
                "scores24_remote_enabled": ENABLE_SCORES24_REMOTE,
                "playwright_proxy_configured": PLAYWRIGHT_PROXY_CONFIGURED,
                "sportytrader_remote_enabled": ENABLE_SPORTYTRADER_REMOTE,
                "endpoints": _public_endpoints(),
            })
            return

        if path == "/ledger-state":
            state = _load_ledger_state()
            nba_games_meta = _load_nba_props_games_with_meta()
            self._send_json(200, {
                "ok": True,
                "state": state,
                "nba_games": nba_games_meta.get("games", []),
                "nba_games_source": nba_games_meta.get("source"),
                "nba_games_error": nba_games_meta.get("error"),
            })
            return

        if path == "/refresh-nba-props-games":
            from urllib.parse import parse_qs

            qs = parse_qs(parsed.query)
            refresh_date = (qs.get("date") or [""])[0] or None
            try:
                result = _refresh_nba_props_games(refresh_date)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, result)
            return

        if path == "/ipl":
            from urllib.parse import parse_qs

            if not IPL_AVAILABLE:
                self._send_json(503, {"error": "IPL model not loaded"})
                return

            qs = parse_qs(parsed.query)

            def _optional_query_arg(name: str) -> str | None:
                value = (qs.get(name) or [""])[0]
                text = str(value).strip()
                return text or None

            try:
                result = run_ipl_model(
                    team1=_optional_query_arg("team1"),
                    team2=_optional_query_arg("team2"),
                    venue=_optional_query_arg("venue"),
                    toss_winner=_optional_query_arg("toss_winner"),
                    toss_decision=_optional_query_arg("toss_decision"),
                    db_path=LEDGER_DB_FILE,
                )
                self._send_json(200, result)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            return

        if path in {"/scores24-feed", "/sportytrader-feed"}:
            feed_name = "scores24_manual_feed.json" if path == "/scores24-feed" else "sportytrader_manual_feed.json"
            feed_path = os.path.join(BASE_DIR, feed_name)
            if not os.path.exists(feed_path):
                self._send_json(404, {"ok": False, "error": f"{feed_name} not found"})
                return
            try:
                with open(feed_path, encoding="utf-8") as fh:
                    payload = json.load(fh)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": f"Failed to read {feed_name}: {exc}"})
                return
            self._send_json(200, payload, extra_headers={"Cache-Control": "no-store"})
            return

        # Poll job status: GET /job-status?id=<job_id>
        if path.startswith("/job-status"):
            from urllib.parse import parse_qs
            qs = parse_qs(parsed.query)
            job_id = (qs.get("id") or [""])[0]
            with _jobs_lock:
                job = _jobs.get(job_id)
            if not job:
                self._send_json(404, {"ok": False, "error": "Job not found"})
            elif job["status"] == "running":
                self._send_json(200, {"ok": True, "status": "running"})
            else:
                self._send_json(200, {"ok": True, "status": "done", **job["result"]})
                # Clean up finished job
                with _jobs_lock:
                    _jobs.pop(job_id, None)
        else:
            self._send_json(404, {"ok": False, "error": "Route not found"})

    def do_HEAD(self) -> None:  # noqa: N802
        if self.path in {
            "/",
            "/health",
            "/ledger-state",
            "/api/ledger-state",
            "/refresh-nba-props-games",
            "/api/refresh-nba-props-games",
        }:
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        path = self.path
        if path == "/api":
            path = "/"
        elif path.startswith("/api/"):
            path = path[4:]

        content_type = str(self.headers.get("Content-Type", "")).lower()
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length"})
            return

        try:
            raw = self.rfile.read(length)
            if raw:
                body = json.loads(raw.decode("utf-8"))
            else:
                body = {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            if "application/json" in content_type:
                self._send_json(400, {"error": "bad json"})
                return
            self._send_json(400, {"ok": False, "error": "Invalid JSON body"})
            return

        async_mode = body.get("async", False)
        date_str = body.get("date")  # optional MM/DD/YYYY date for the model
        game_id = body.get("game_id")
        game_label = body.get("game_label")

        if path == "/refresh-nba-props-games":
            try:
                result = _refresh_nba_props_games(date_str)
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
                return
            self._send_json(200, result)
            return

        if path == "/ledger-state":
            state = body.get("state", body)
            if not isinstance(state, dict):
                self._send_json(400, {"ok": False, "error": "Invalid ledger state payload"})
                return
            if not _save_ledger_state(state):
                self._send_json(500, {"ok": False, "error": "Failed to persist ledger state"})
                return
            self._send_json(200, {"ok": True, "state": _load_ledger_state()})
            return

        if path == "/picks":
            ok, entry, error = _save_pick_to_ledger(body if isinstance(body, dict) else {})
            if not ok or entry is None:
                self._send_json(400, {"ok": False, "error": error or "Failed to save pick"})
                return
            self._send_json(200, {"success": True, "id": entry.get("id")})
            return

        if path == "/grade" and "id" in body and "result" in body and "picks" not in body:
            ok, normalized, error = _set_pick_result_in_ledger(body.get("id"), body.get("result"))
            if not ok or normalized is None:
                self._send_json(400, {"ok": False, "error": error or "Failed to save result"})
                return
            self._send_json(200, {"ok": True, "id": body.get("id"), "result": normalized})
            return

        if path == "/grade":
            picks = body.get("picks", [])
            existing = body.get("existing", {})
            year = int(body.get("year") or datetime.now().year)

            if not isinstance(picks, list) or not isinstance(existing, dict):
                self._send_json(400, {"ok": False, "error": "Invalid payload shape"})
                return

            result = auto_grade(picks, existing, year)
            self._send_json(200, {"ok": True, **result})

        elif path == "/run-sportsline-odds":
            league = str(body.get("league", "NBA")).upper()
            if async_mode:
                job_id = _launch_job(run_sportsline_odds, league)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_sportsline_odds(league)
                self._send_json(200, result)

        elif path == "/run-nba-model":
            if async_mode:
                job_id = _launch_job(run_nba_model, date_str, "new")
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_nba_model(date_str, "new")
                self._send_json(200, result)

        elif path == "/run-nba-old-model":
            if async_mode:
                job_id = _launch_job(run_nba_model, date_str, "old")
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_nba_model(date_str, "old")
                self._send_json(200, result)

        elif path == "/run-nba-props-model":
            if async_mode:
                job_id = _launch_job(run_nba_props_model, date_str, game_id, game_label)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_nba_props_model(date_str, game_id, game_label)
                self._send_json(200, result)

        elif path == "/run-mlb-model":
            if async_mode:
                job_id = _launch_job(run_mlb_model, date_str)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_mlb_model(date_str)
                self._send_json(200, result)

        elif path == "/run-scores24":
            if IS_RENDER_RUNTIME and not ENABLE_SCORES24_REMOTE:
                self._send_json(403, {
                    "ok": False,
                    "error": "Scores24 scraping is disabled on Render. Run it locally and sync scores24_manual_feed.json.",
                })
                return

            league = str(body.get("league", "")).strip().lower()
            sports = body.get("sports")
            if not isinstance(sports, list):
                sports = [league] if league else ["nba", "nhl", "mlb"]
            scrape_date = body.get("date")
            if async_mode:
                job_id = _launch_job(run_scores24_scraper, sports, scrape_date)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_scores24_scraper(sports, scrape_date)
                self._send_json(200, result)

        elif path == "/run-sportytrader":
            if IS_RENDER_RUNTIME and not ENABLE_SPORTYTRADER_REMOTE:
                self._send_json(403, {
                    "ok": False,
                    "error": "SportyTrader scraping is disabled on Render. Run it locally and sync sportytrader_manual_feed.json.",
                })
                return

            scrape_date = body.get("date")
            league = str(body.get("league", "")).strip().lower()
            sports = body.get("sports")
            if not isinstance(sports, list):
                sports = [league] if league else ["nba", "mlb"]
            if async_mode:
                job_id = _launch_job(run_sportytrader_scraper, scrape_date, sports)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_sportytrader_scraper(scrape_date, sports)
                self._send_json(200, result)

        elif path == "/run-sportsgambler":
            scrape_date = body.get("date")
            sports = body.get("sports")
            league = str(body.get("league", "")).strip().lower()
            if not isinstance(sports, list):
                sports = [league] if league else ["nba", "mlb"]
            if async_mode:
                job_id = _launch_job(run_sportsgambler_scraper, scrape_date, sports)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_sportsgambler_scraper(scrape_date, sports)
                self._send_json(200, result)
        elif path == "/ask-opus":
            prompt = body.get("prompt")
            system = body.get("system")
            max_tokens = body.get("max_tokens", ANTHROPIC_MAX_TOKENS_DEFAULT)
            temperature = body.get("temperature", 0.2)

            result = _invoke_anthropic(
                str(prompt or ""),
                system=str(system) if system is not None else None,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            self._send_json(200 if result.get("ok") else 400, result)

        else:
            self._send_json(404, {"ok": False, "error": "Route not found"})


def main() -> None:
    # Allow concurrent requests via threading
    import socketserver
    class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
        daemon_threads = True
    server = ThreadedHTTPServer((HOST, PORT), Handler)
    print(f"Pickgrader running on http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down pickgrader...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
