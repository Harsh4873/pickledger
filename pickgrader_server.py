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
import os
import re
import sqlite3
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
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

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
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_REGION = os.environ.get("ODDS_API_REGION", "us").strip() or "us"
ODDS_API_BOOKMAKERS = os.environ.get("ODDS_API_BOOKMAKERS", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_CACHE_TTL_S = 90

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_API_URL = os.environ.get("ANTHROPIC_API_URL", "https://api.anthropic.com/v1/messages").strip() or "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "").strip()
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01").strip() or "2023-06-01"
ANTHROPIC_MAX_TOKENS_DEFAULT = 800

SPORT_TO_ODDS_API_KEY = {
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
}

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

PROP_MARKET_TO_ODDS_API_KEY = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
}

_odds_cache: dict[str, tuple[float, Any]] = {}
_odds_cache_lock = threading.Lock()
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
            odds INTEGER NOT NULL DEFAULT -110,
            result TEXT NOT NULL DEFAULT 'pending',
            notes TEXT NOT NULL DEFAULT '',
            start_time TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        )
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
        try:
            odds = int(float(item.get("odds", -110)))
        except (TypeError, ValueError):
            odds = -110
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
            str(game_time_map.get(pick_id_str, item.get("start_time", "")) or ""),
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


def _load_nba_games_from_sqlite(date_str: str | None = None) -> list[dict[str, str]]:
    today_iso, today_us = _parse_model_date_arg(date_str)
    labels: dict[str, str] = {}
    try:
        with _ledger_db_connect() as conn:
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

    return {
        "games": _extract_nba_props_games(output),
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


def _american_implied_prob(odds: int | float | None) -> float | None:
    try:
        odds_val = int(float(odds)) if odds is not None else 0
    except (ValueError, TypeError):
        return None
    if odds_val == 0:
        return None
    if odds_val > 0:
        return 100.0 / (odds_val + 100.0)
    return abs(odds_val) / (abs(odds_val) + 100.0)


def _quarter_kelly_pct(odds: int | float | None, model_prob: float | None, max_bankroll_pct: float = 0.05) -> float | None:
    try:
        odds_val = int(float(odds)) if odds is not None else 0
        prob_val = float(model_prob) if model_prob is not None else None
    except (ValueError, TypeError):
        return None
    if not prob_val or prob_val <= 0 or prob_val >= 1 or odds_val == 0:
        return None
    if odds_val > 0:
        b = odds_val / 100.0
    else:
        b = 100.0 / abs(odds_val)
    q = 1.0 - prob_val
    kelly = (b * prob_val - q) / b
    if kelly <= 0:
        return 0.0
    return min(kelly / 4.0, max_bankroll_pct) * 100.0


def _odds_cache_get(key: str) -> Any | None:
    now = time.time()
    with _odds_cache_lock:
        cached = _odds_cache.get(key)
        if not cached:
            return None
        expires_at, payload = cached
        if expires_at <= now:
            _odds_cache.pop(key, None)
            return None
        return payload


def _odds_cache_set(key: str, payload: Any) -> Any:
    with _odds_cache_lock:
        _odds_cache[key] = (time.time() + ODDS_API_CACHE_TTL_S, payload)
    return payload


def _fetch_json_url(url: str, timeout: int = 20) -> Any:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


def _should_enrich_market_odds(date_str: str | None = None) -> bool:
    if not ODDS_API_KEY:
        return False
    if not date_str:
        return True
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            target = datetime.strptime(date_str, fmt).date()
            return target >= datetime.now().date()
        except ValueError:
            continue
    return True


def _odds_api_query(path: str, params: dict[str, Any]) -> Any | None:
    clean_params = {k: v for k, v in params.items() if v not in (None, "", [])}
    clean_params["apiKey"] = ODDS_API_KEY
    query = urlencode(clean_params, doseq=True)
    url = f"{ODDS_API_BASE}{path}?{query}"
    cached = _odds_cache_get(url)
    if cached is not None:
        return cached
    try:
        payload = _fetch_json_url(url)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
        return None
    return _odds_cache_set(url, payload)


def _fetch_featured_market_odds(sport: str, markets: list[str]) -> list[dict[str, Any]]:
    sport_key = SPORT_TO_ODDS_API_KEY.get(str(sport or "").upper())
    if not sport_key or not markets:
        return []
    params: dict[str, Any] = {
        "markets": ",".join(sorted(set(markets))),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if ODDS_API_BOOKMAKERS:
        params["bookmakers"] = ODDS_API_BOOKMAKERS
    else:
        params["regions"] = ODDS_API_REGION
    payload = _odds_api_query(f"/sports/{sport_key}/odds", params)
    return payload if isinstance(payload, list) else []


def _fetch_event_market_odds(sport: str, event_id: str, markets: list[str]) -> dict[str, Any] | None:
    sport_key = SPORT_TO_ODDS_API_KEY.get(str(sport or "").upper())
    if not sport_key or not event_id or not markets:
        return None
    params: dict[str, Any] = {
        "markets": ",".join(sorted(set(markets))),
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if ODDS_API_BOOKMAKERS:
        params["bookmakers"] = ODDS_API_BOOKMAKERS
    else:
        params["regions"] = ODDS_API_REGION
    payload = _odds_api_query(f"/sports/{sport_key}/events/{event_id}/odds", params)
    return payload if isinstance(payload, dict) else None


def _team_matches_name(team_text: str, candidate_name: str) -> bool:
    t = normalize(team_text)
    c = normalize(candidate_name)
    if not t or not c:
        return False
    if t == c or t in c or (c in t and len(c) > 2):
        return True
    t_tokens = set(t.split())
    c_tokens = set(c.split())
    if t_tokens and c_tokens and (t_tokens & c_tokens):
        if t_tokens <= c_tokens or c_tokens <= t_tokens:
            return True
        if len(t_tokens & c_tokens) >= min(2, len(t_tokens), len(c_tokens)):
            return True
    t_parts = t.split()
    if t_parts:
        last = t_parts[-1]
        if len(last) >= 3 and last in c_tokens:
            return True
    return False


def _player_names_match(name_a: str, name_b: str) -> bool:
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


def _extract_pick_head(pick_text: str) -> str:
    return str(pick_text or "").split("(", 1)[0].strip()


def _find_matching_event(events: list[dict[str, Any]], team_a: str | None, team_b: str | None) -> dict[str, Any] | None:
    if not team_a or not team_b:
        return None
    for event in events:
        home = str(event.get("home_team", ""))
        away = str(event.get("away_team", ""))
        direct = _team_matches_name(team_a, away) and _team_matches_name(team_b, home)
        reverse = _team_matches_name(team_a, home) and _team_matches_name(team_b, away)
        if direct or reverse:
            return event
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _pick_market_descriptor(pick: dict[str, Any]) -> dict[str, Any] | None:
    explicit_market = str(pick.get("market_type", "")).strip()
    if explicit_market:
        descriptor = {
            "market_type": explicit_market,
            "selection": pick.get("selection"),
            "line": _float_or_none(pick.get("line")),
            "team": pick.get("team"),
            "player_name": pick.get("player_name"),
            "away_team": pick.get("away_team"),
            "home_team": pick.get("home_team"),
        }
        return descriptor

    pick_text = str(pick.get("pick", ""))
    head = _extract_pick_head(pick_text)
    matchup = parse_matchup(pick_text)
    away_team = matchup[0] if matchup else None
    home_team = matchup[1] if matchup else None

    ml_m = re.search(r"^(.*?)\s+ML\b", head, flags=re.IGNORECASE)
    if ml_m:
        return {
            "market_type": "h2h",
            "team": ml_m.group(1).strip(),
            "away_team": away_team,
            "home_team": home_team,
        }

    total_m = re.search(r"^(Over|Under)\s+(\d+(?:\.\d+)?)\b", head, flags=re.IGNORECASE)
    if total_m:
        return {
            "market_type": "totals",
            "selection": total_m.group(1).title(),
            "line": float(total_m.group(2)),
            "away_team": away_team,
            "home_team": home_team,
        }

    spread_m = re.search(r"^(.*?)\s*([+-]\d+(?:\.\d+)?)\b", head)
    if spread_m:
        return {
            "market_type": "spreads",
            "team": spread_m.group(1).strip(),
            "line": float(spread_m.group(2)),
            "away_team": away_team,
            "home_team": home_team,
        }

    prop_m = re.search(
        r"^(.*?)\s+(points|rebounds|assists)\s+(OVER|UNDER)\s+(\d+(?:\.\d+)?)\s+vs\s+(.+)$",
        pick_text,
        flags=re.IGNORECASE,
    )
    if prop_m:
        return {
            "market_type": PROP_MARKET_TO_ODDS_API_KEY.get(prop_m.group(2).lower()),
            "player_name": prop_m.group(1).strip(),
            "selection": prop_m.group(3).title(),
            "line": float(prop_m.group(4)),
            "team": pick.get("team"),
            "away_team": pick.get("away_team"),
            "home_team": pick.get("home_team"),
        }

    return None


def _best_market_price(event_payload: dict[str, Any], descriptor: dict[str, Any]) -> dict[str, Any] | None:
    target_market = str(descriptor.get("market_type", ""))
    target_line = _float_or_none(descriptor.get("line"))
    selection = str(descriptor.get("selection", "") or "").title()
    team = str(descriptor.get("team", "") or "")
    player_name = str(descriptor.get("player_name", "") or "")
    best: dict[str, Any] | None = None
    nearest: dict[str, Any] | None = None

    for bookmaker in event_payload.get("bookmakers", []) or []:
        book_title = str(bookmaker.get("title") or bookmaker.get("key") or "").strip()
        for market in bookmaker.get("markets", []) or []:
            if str(market.get("key", "")).strip() != target_market:
                continue
            for outcome in market.get("outcomes", []) or []:
                price = _int_or_none(outcome.get("price"))
                if price is None:
                    continue

                matched = False
                line_val = _float_or_none(outcome.get("point"))

                if target_market == "h2h":
                    matched = _team_matches_name(team, str(outcome.get("name", "")))
                elif target_market == "totals":
                    matched = (
                        str(outcome.get("name", "")).strip().title() == selection
                        and target_line is not None
                        and line_val is not None
                        and abs(line_val - target_line) < 0.001
                    )
                elif target_market == "spreads":
                    matched = (
                        _team_matches_name(team, str(outcome.get("name", "")))
                        and target_line is not None
                        and line_val is not None
                        and abs(line_val - target_line) < 0.001
                    )
                elif target_market in {"player_points", "player_rebounds", "player_assists"}:
                    same_player = _player_names_match(player_name, str(outcome.get("description", "")))
                    same_side = str(outcome.get("name", "")).strip().title() == selection
                    if same_player and same_side and target_line is not None and line_val is not None:
                        if abs(line_val - target_line) < 0.001:
                            matched = True
                        elif abs(line_val - target_line) <= 1.0:
                            candidate = {
                                "odds": price,
                                "bookmaker": book_title,
                                "line": line_val,
                                "line_delta": abs(line_val - target_line),
                            }
                            if nearest is None or (
                                candidate["line_delta"] < nearest["line_delta"]
                                or (
                                    abs(candidate["line_delta"] - nearest["line_delta"]) < 0.001
                                    and candidate["odds"] > nearest["odds"]
                                )
                            ):
                                nearest = candidate

                if matched:
                    candidate = {
                        "odds": price,
                        "bookmaker": book_title,
                        "line": line_val,
                    }
                    if best is None or candidate["odds"] > best["odds"]:
                        best = candidate

    return best or nearest


def _replace_pick_line(pick_text: str, new_line: float | None) -> str:
    if new_line is None:
        return pick_text
    line_text = f"{new_line:.1f}".rstrip("0").rstrip(".")
    pick_text = re.sub(
        r"(\s+(?:OVER|UNDER)\s+)(\d+(?:\.\d+)?)",
        rf"\g<1>{line_text}",
        pick_text,
        count=1,
        flags=re.IGNORECASE,
    )
    pick_text = re.sub(
        r"(\b(?:Over|Under)\s+)(\d+(?:\.\d+)?)",
        rf"\g<1>{line_text}",
        pick_text,
        count=1,
    )
    return pick_text


def _enrich_picks_with_market_odds(picks: list[dict[str, Any]], date_str: str | None = None) -> list[dict[str, Any]]:
    if not picks or not _should_enrich_market_odds(date_str):
        return picks

    sport_market_needs: dict[str, set[str]] = {}
    event_cache: dict[tuple[str, str], dict[str, Any] | None] = {}
    event_lookup_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}

    for pick in picks:
        if _int_or_none(pick.get("odds")) is not None:
            continue
        descriptor = _pick_market_descriptor(pick)
        if not descriptor:
            continue
        sport = str(pick.get("sport", "")).upper()
        market_type = str(descriptor.get("market_type", "")).strip()
        if sport in SPORT_TO_ODDS_API_KEY and market_type in {"h2h", "totals", "spreads"}:
            sport_market_needs.setdefault(sport, set()).add(market_type)

    featured_by_sport = {
        sport: _fetch_featured_market_odds(sport, list(markets))
        for sport, markets in sport_market_needs.items()
    }

    for pick in picks:
        if _int_or_none(pick.get("odds")) is not None:
            continue

        descriptor = _pick_market_descriptor(pick)
        if not descriptor:
            continue

        sport = str(pick.get("sport", "")).upper()
        away_team = str(descriptor.get("away_team") or "").strip()
        home_team = str(descriptor.get("home_team") or "").strip()
        if not away_team or not home_team:
            matchup = parse_matchup(str(pick.get("pick", "")))
            if matchup:
                away_team, home_team = matchup
        if not away_team or not home_team:
            continue

        event = _find_matching_event(featured_by_sport.get(sport, []), away_team, home_team)
        if event is None and descriptor.get("market_type") in {"player_points", "player_rebounds", "player_assists"}:
            lookup_key = (sport, away_team, home_team)
            if lookup_key not in event_lookup_cache:
                event_lookup_cache[lookup_key] = _find_matching_event(
                    _fetch_featured_market_odds(sport, ["h2h"]),
                    away_team,
                    home_team,
                )
            event = event_lookup_cache[lookup_key]
        if not event:
            continue

        payload = event
        if descriptor.get("market_type") in {"player_points", "player_rebounds", "player_assists"}:
            event_id = str(event.get("id", "")).strip()
            event_key = (sport, event_id)
            if event_key not in event_cache:
                event_cache[event_key] = _fetch_event_market_odds(
                    sport,
                    event_id,
                    [
                        "player_points",
                        "player_rebounds",
                        "player_assists",
                    ],
                )
            payload = event_cache[event_key] or {}

        market_price = _best_market_price(payload or {}, descriptor)
        if not market_price:
            continue

        pick["odds"] = market_price["odds"]
        if market_price.get("bookmaker"):
            pick["odds_bookmaker"] = market_price["bookmaker"]
        actual_line = _float_or_none(market_price.get("line"))
        if actual_line is not None:
            pick["market_line"] = actual_line
            if descriptor.get("market_type") in {"player_points", "player_rebounds", "player_assists"}:
                pick["pick"] = _replace_pick_line(str(pick.get("pick", "")), actual_line)
                pick["line"] = actual_line

        implied_prob = _american_implied_prob(pick.get("odds"))
        if implied_prob is not None:
            pick["market_implied_probability"] = round(implied_prob, 4)
            model_prob = _float_or_none(pick.get("probability"))
            if model_prob is not None:
                pick["market_edge"] = round((model_prob - implied_prob) * 100.0, 2)
                quarter_kelly = _quarter_kelly_pct(pick.get("odds"), model_prob)
                if quarter_kelly is not None:
                    pick["units"] = round(quarter_kelly, 2)

    return picks


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


def find_game_for_pick(games: list[dict[str, Any]], pick_text: str) -> dict[str, Any] | None:
    matchup = parse_matchup(pick_text)
    if matchup:
        team_a, team_b = matchup
        for game in games:
            c1 = game["competitors"][0]["raw"]
            c2 = game["competitors"][1]["raw"]

            direct = team_matches_competitor(team_a, c1) and team_matches_competitor(team_b, c2)
            reverse = team_matches_competitor(team_a, c2) and team_matches_competitor(team_b, c1)
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


def resolve_team_score(game: dict[str, Any], team_text: str) -> tuple[int, int] | None:
    comps = game["competitors"]
    for idx, c in enumerate(comps):
        if team_matches_competitor(team_text, c["raw"]):
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
                if not _player_names_match(player_name, display_name):
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
        resolved = resolve_team_score(game, team_label)
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
        resolved = resolve_team_score(game, team_label)
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
        resolved = resolve_team_score(game, team_label)
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
        resolved = resolve_team_score(game, team_label)
        if resolved is None:
            return "pending"
        team_score, opp_score = resolved
        if team_score == opp_score:
            return "push"
        return "win" if team_score > opp_score else "loss"

    # Fallback: treat leading team label as winner pick.
    fallback_team = re.sub(r"\s*[+-]\d+(?:\.\d+)?\s*$", "", head, flags=re.IGNORECASE).strip()
    if fallback_team:
        resolved = resolve_team_score(game, fallback_team)
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
            game = find_game_for_pick(all_games, str(pick.get("pick", "")))
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
            game = find_game_for_pick(games, str(pick.get("pick", "")))
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

        # Extract winner
        winner_m = re.search(r"\*\*Winner:\*\*\s*(.+?)\s*\(Model Prob:\s*([\d.]+)%\)", line)
        if winner_m:
            winner = winner_m.group(1).strip()
            prob = float(winner_m.group(2)) / 100
            # Look ahead for spread, edge, and decision
            spread_val = 0.0
            edge_val = None
            decision = "PASS"
            for j in range(i + 1, min(i + 20, len(lines))):
                sp_m = re.search(r"\*\*Spread:\*\*\s*\S+\s*by\s*([\d.]+)\s*points", lines[j])
                if sp_m:
                    spread_val = float(sp_m.group(1))
                edge_m = re.search(r"\*\*Edge:\*\*\s*\S+\s*([+-]?[\d.]+)%", lines[j])
                if edge_m:
                    edge_val = float(edge_m.group(1))
                dec_m = re.search(r"\*\*Decision:\s*(BET|PASS)", lines[j])
                if dec_m:
                    decision = dec_m.group(1)
                    break

            matchup = f"{current_away} @ {current_home}"
            if spread_val > 0 and decision == "BET":
                pick_text = f"{winner} -{spread_val:.1f} ({matchup})"
            else:
                pick_text = f"{winner} ML ({matchup})"

            # Edge is from the home team perspective; flip sign if bet is on away team
            display_edge = edge_val
            if display_edge is not None and winner != current_home:
                display_edge = -display_edge

            _append_unique({
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
            })

        # Over/Under decision: "**O/U Decision: BET OVER**"
        ou_m = re.search(r"\*\*O/U Decision:\s*(BET OVER|BET UNDER|PASS)\*\*", line)
        if ou_m:
            ou_decision_raw = ou_m.group(1)
            # Look back for total line
            model_total = None
            line_val = 225.0
            for j in range(max(0, i - 5), i):
                total_m = re.search(r"\*\*Over/Under:\*\*\s*Model Total\s*([\d.]+)\s*vs\s*Line\s*([\d.]+)", lines[j])
                if total_m:
                    model_total = float(total_m.group(1))
                    line_val = float(total_m.group(2))

            matchup = f"{current_away} @ {current_home}"
            if ou_decision_raw.startswith("BET"):
                ou_side = "Over" if "OVER" in ou_decision_raw else "Under"
                _append_unique({
                    "source": source_label,
                    "pick": f"{ou_side} {line_val:.1f} ({matchup})",
                    "sport": "NBA",
                    "odds": None,
                    "units": 1,
                    "probability": None,
                    "edge": abs(model_total - line_val) if model_total else None,
                    "decision": "BET",
                    "market_type": "totals",
                    "selection": ou_side,
                    "line": line_val,
                    "away_team": current_away,
                    "home_team": current_home,
                })
            else:
                _append_unique({
                    "source": source_label,
                    "pick": f"O/U {line_val:.1f} ({matchup})",
                    "sport": "NBA",
                    "odds": None,
                    "units": 1,
                    "probability": None,
                    "edge": None,
                    "decision": "PASS",
                    "market_type": "totals",
                    "line": line_val,
                    "away_team": current_away,
                    "home_team": current_home,
                })

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
            market_type = PROP_MARKET_TO_ODDS_API_KEY.get(current_prop["key"], "")

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
                ou_side = parts[1]  # OVER, UNDER, or PASS
                try:
                    ou_line = float(parts[2])
                    predicted_total = float(parts[3])
                except (ValueError, IndexError):
                    continue

                if not current_team_a or not current_team_b:
                    continue

                matchup = f"{_shorten_mlb_name(current_team_a)} vs {_shorten_mlb_name(current_team_b)}"
                if ou_side in ("OVER", "UNDER"):
                    picks.append({
                        "source": "MLB Model",
                        "pick": f"{'Over' if ou_side == 'OVER' else 'Under'} {ou_line:.1f} ({matchup})",
                        "sport": "MLB",
                        "odds": None,
                        "units": 1,
                        "probability": None,
                        "edge": abs(predicted_total - ou_line),
                        "decision": "BET",
                        "market_type": "totals",
                        "selection": "Over" if ou_side == "OVER" else "Under",
                        "line": ou_line,
                        "away_team": current_team_a,
                        "home_team": current_team_b,
                    })
                else:
                    picks.append({
                        "source": "MLB Model",
                        "pick": f"O/U {ou_line:.1f} ({matchup})",
                        "sport": "MLB",
                        "odds": None,
                        "units": 1,
                        "probability": None,
                        "edge": None,
                        "decision": "PASS",
                        "market_type": "totals",
                        "line": ou_line,
                        "away_team": current_team_a,
                        "home_team": current_team_b,
                    })
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
                "odds": None,
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

            picks.append({
                "source": "MLB Model",
                "pick": f"{winner} ML ({matchup})",
                "sport": "MLB",
                "odds": None,
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
                # Default line of 8.5 for MLB
                line = 8.5
                if total_val > line + 0.5:
                    picks.append({
                        "source": "MLB Model",
                        "pick": f"Over {line} ({matchup})",
                        "sport": "MLB",
                        "odds": None,
                        "units": 1,
                        "probability": None,
                        "edge": total_val - line,
                        "decision": "BET",
                        "market_type": "totals",
                        "selection": "Over",
                        "line": line,
                        "away_team": current_away,
                        "home_team": current_home,
                    })
                elif total_val < line - 0.5:
                    picks.append({
                        "source": "MLB Model",
                        "pick": f"Under {line} ({matchup})",
                        "sport": "MLB",
                        "odds": None,
                        "units": 1,
                        "probability": None,
                        "edge": line - total_val,
                        "decision": "BET",
                        "market_type": "totals",
                        "selection": "Under",
                        "line": line,
                        "away_team": current_away,
                        "home_team": current_home,
                    })

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


def _clean_scores24_pick(tip: str, matchup: str, sport: str) -> str:
    """Convert raw Scores24 tip into clean format matching NBA/MLB model picks."""
    # Strip "at odds of ..." suffix from tip
    tip_clean = re.sub(r"\s*at odds of\s*[^\)]*\*?\s*$", "", tip).strip()
    tip_clean = _strip_scores24_ot_qualifier(tip_clean)

    # Build shortened matchup
    teams = matchup.split(" vs ")
    home = teams[0].strip() if len(teams) > 0 else ""
    away = teams[1].strip() if len(teams) > 1 else ""
    home_short = _shorten_team(home)
    away_short = _shorten_team(away)
    matchup_short = f"{home_short} vs {away_short}"

    # ── Pattern: "<Team> Handicap (<spread>)" ──
    m = re.match(r"^(.+?)\s+Handicap\s*\(([+-]?\d+\.?\d*)\)", tip_clean, re.IGNORECASE)
    if m:
        team = _shorten_team(m.group(1))
        spread = m.group(2)
        if not spread.startswith(("+", "-")):
            spread = "+" + spread
        return f"{team} {spread} ({matchup_short})"

    # ── Pattern: "<Team> Total goals/points Over/Under (<value>)" (team total) ──
    m = re.match(
        r"^(.+?)\s+Total\s+(goals|points)\s+(Over|Under)\s*\((\d+\.?\d*)\)",
        tip_clean, re.IGNORECASE,
    )
    if m:
        team = _shorten_team(m.group(1))
        kind = m.group(2).lower()
        direction = m.group(3)
        value = m.group(4)
        suffix = " TG" if kind == "goals" else ""
        return f"{team} {direction} {value}{suffix} ({matchup_short})"

    # ── Pattern: "Total goals/points Over/Under (<value>)" (game total) ──
    m = re.match(
        r"^Total\s+(goals|points)\s+(Over|Under)\s*\((\d+\.?\d*)\)",
        tip_clean, re.IGNORECASE,
    )
    if m:
        kind = m.group(1).lower()
        direction = m.group(2)
        value = m.group(3)
        suffix = " TG" if kind == "goals" else ""
        return f"{direction} {value}{suffix} ({matchup_short})"

    # ── Pattern: "Both Teams To Score (Yes/No)" with optional period prefix ──
    m = re.match(
        r"^(?:.*?,\s*)?Both\s+Teams?\s+To\s+Score\s*\((Yes|No)\)",
        tip_clean, re.IGNORECASE,
    )
    if m:
        answer = m.group(1)
        return f"BTTS {answer} ({matchup_short})"

    # ── Pattern: "<Team> to win" → moneyline ──
    m = re.match(r"^(.+?)\s+to\s+win$", tip_clean, re.IGNORECASE)
    if m:
        team = _shorten_team(m.group(1))
        return f"{team} ML ({matchup_short})"

    # ── Pattern: "<Team> ML" ──
    m = re.match(r"^(.+?)\s+ML$", tip_clean, re.IGNORECASE)
    if m:
        team = _shorten_team(m.group(1))
        return f"{team} ML ({matchup_short})"

    # ── Pattern: "Over/Under (<value>)" (generic total) ──
    m = re.match(r"^(Over|Under)\s*\((\d+\.?\d*)\)", tip_clean, re.IGNORECASE)
    if m:
        direction = m.group(1)
        value = m.group(2)
        return f"{direction} {value} ({matchup_short})"

    # ── Fallback: cleaned tip + shortened matchup ──
    return f"{tip_clean} ({matchup_short})"


def _normalize_french_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text or ""))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.lower().replace("’", "'")
    return re.sub(r"\s+", " ", normalized).strip()


def _clean_sportytrader_pick(tip: str, matchup: str) -> str:
    """Convert French SportyTrader NBA tip into the same pick format used in UI."""
    tip_clean = re.sub(r"\s+", " ", str(tip or "")).strip()
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

    def _resolve_team_name() -> str:
        if home_norm and home_norm in tip_norm:
            return home_short
        if away_norm and away_norm in tip_norm:
            return away_short
        if home_short_norm and home_short_norm in tip_norm:
            return home_short
        if away_short_norm and away_short_norm in tip_norm:
            return away_short
        home_hits = sum(1 for token in home_tokens if re.search(rf"\b{re.escape(token)}\b", tip_norm))
        away_hits = sum(1 for token in away_tokens if re.search(rf"\b{re.escape(token)}\b", tip_norm))
        if away_hits > home_hits:
            return away_short
        if home_hits > away_hits:
            return home_short
        return home_short

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

        picks.append({
            "source": "Scores24",
            "pick": pick_text,
            "sport": sport,
            "odds": odds_val,
            "units": 1,
            "probability": conf_val / 100 if conf_val else None,
            "edge": None,
            "decision": "BET",  # All Scores24 tips are presented as BET
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

        picks.append({
            "source": "Scores24",
            "pick": _strip_scores24_ot_qualifier(_clean_scores24_pick(tip, matchup, sport)),
            "sport": sport,
            "odds": odds_val,
            "units": 1,
            "probability": conf_val / 100 if conf_val else None,
            "edge": None,
            "decision": "BET",
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

        picks = _enrich_picks_with_market_odds(picks, date_str)
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
            picks = _enrich_picks_with_market_odds(picks, date_str)
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
            picks = _enrich_picks_with_market_odds(picks, date_str)
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

        all_picks = _enrich_picks_with_market_odds(all_picks, date_str)
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
        picks = _enrich_picks_with_market_odds(picks, date_str)
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

        timeout_s = 120
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

    all_picks = _enrich_picks_with_market_odds(all_picks, date_str)
    return {"ok": True, "picks": all_picks, "errors": errors}


def run_sportytrader_scraper(date_str: str | None = None) -> dict[str, Any]:
    """Execute the SportyTrader scraper (NBA only)."""
    python_bin = _resolve_python_bin(SPORTYTRADER_VENV)
    target_date = _resolve_scores24_date(date_str)
    scraper_path = os.path.join(BASE_DIR, "sportytrader_scraper.py")
    if not os.path.exists(scraper_path):
        return {"ok": False, "error": f"sportytrader scraper not found at {scraper_path}"}

    timeout_s = 120
    env = os.environ.copy()
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", _default_playwright_browsers_path())

    def _invoke() -> subprocess.CompletedProcess[str]:
        return _subprocess_run(
            [python_bin, scraper_path, "--sport", "nba", "--date", target_date],
            cwd=BASE_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    try:
        result = _invoke()
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0 and _looks_like_playwright_browser_missing(output):
            ok, install_msg = _ensure_playwright_browsers(python_bin, env)
            if not ok:
                return {"ok": False, "error": f"sportytrader: Playwright install failed ({install_msg})"}
            result = _invoke()
            output = (result.stdout or "") + (result.stderr or "")

        picks: list[dict[str, Any]] = []
        blocks = re.split(r"━{10,}", output)
        for block in blocks:
            match_m = re.search(r"Match:\s*(.+)", block)
            tip_m = re.search(r"Tip:\s*(.+)", block)
            league_m = re.search(r"League:\s*(.+)", block)
            if not match_m or not tip_m:
                continue

            matchup = match_m.group(1).strip()
            tip = tip_m.group(1).strip()
            if not matchup or not tip:
                continue
            league = (league_m.group(1).strip() if league_m else "").upper()
            if "NBA" not in league:
                continue

            picks.append({
                "source": "SportyTrader",
                "pick": _clean_sportytrader_pick(tip, matchup),
                "sport": "NBA",
                "odds": None,
                "units": 1,
                "probability": None,
                "edge": None,
                "decision": "BET",
            })

        if result.returncode != 0 and not picks:
            return {"ok": False, "error": f"sportytrader: scraper exited {result.returncode} ({_compact_error_text(output)})"}
        if not picks:
            return {"ok": False, "error": f"sportytrader: no picks parsed ({_compact_error_text(output)})"}
        picks = _enrich_picks_with_market_odds(picks, date_str)
        return {"ok": True, "picks": picks}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"sportytrader: timed out after {timeout_s}s"}
    except Exception as exc:
        return {"ok": False, "error": f"sportytrader: {exc}"}


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
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
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
            endpoints = [
                "/health",
                "/ledger-state",
                "/grade",
                "/run-nba-model",
                "/run-nba-old-model",
                "/run-nba-props-model",
                "/run-mlb-model",
                "/ask-opus",
                "/job-status?id=<id>",
            ]
            if ENABLE_SCORES24_REMOTE:
                endpoints.append("/run-scores24")
            if ENABLE_SPORTYTRADER_REMOTE:
                endpoints.append("/run-sportytrader")
            self._send_json(200, {
                "ok": True,
                "service": "pickledger-grader",
                "status": "healthy",
                "odds_api_enabled": bool(ODDS_API_KEY),
                "anthropic_enabled": bool(ANTHROPIC_API_KEY),
                "anthropic_model": ANTHROPIC_MODEL,
                "scores24_remote_enabled": ENABLE_SCORES24_REMOTE,
                "playwright_proxy_configured": PLAYWRIGHT_PROXY_CONFIGURED,
                "sportytrader_remote_enabled": ENABLE_SPORTYTRADER_REMOTE,
                "endpoints": endpoints,
            })
            return

        if path == "/health":
            self._send_json(200, {
                "ok": True,
                "status": "healthy",
                "odds_api_enabled": bool(ODDS_API_KEY),
                "anthropic_enabled": bool(ANTHROPIC_API_KEY),
                "anthropic_model": ANTHROPIC_MODEL,
                "scores24_remote_enabled": ENABLE_SCORES24_REMOTE,
                "playwright_proxy_configured": PLAYWRIGHT_PROXY_CONFIGURED,
                "sportytrader_remote_enabled": ENABLE_SPORTYTRADER_REMOTE,
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
        if self.path in {"/", "/health", "/ledger-state", "/api/ledger-state"}:
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

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"ok": False, "error": "Invalid Content-Length"})
            return

        try:
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"ok": False, "error": "Invalid JSON body"})
            return

        async_mode = body.get("async", False)
        date_str = body.get("date")  # optional MM/DD/YYYY date for the model
        game_id = body.get("game_id")
        game_label = body.get("game_label")

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

        if path == "/grade":
            picks = body.get("picks", [])
            existing = body.get("existing", {})
            year = int(body.get("year") or datetime.now().year)

            if not isinstance(picks, list) or not isinstance(existing, dict):
                self._send_json(400, {"ok": False, "error": "Invalid payload shape"})
                return

            result = auto_grade(picks, existing, year)
            self._send_json(200, {"ok": True, **result})

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
            if async_mode:
                job_id = _launch_job(run_sportytrader_scraper, scrape_date)
                self._send_json(200, {"ok": True, "job_id": job_id, "status": "running"})
            else:
                result = run_sportytrader_scraper(scrape_date)
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
