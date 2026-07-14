#!/usr/bin/env python3
"""
NBA Playoffs Prediction Model.

This runner is intentionally stricter than the regular-season NBA runner:
it first verifies the date against ESPN's postseason scoreboard and only
emits machine-readable picks for official playoff games that have not started.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

BASE_DIR = Path(__file__).resolve().parents[1]
NBA_MODEL_DIR = BASE_DIR / "NBAPredictionModel"
sys.path.insert(0, str(NBA_MODEL_DIR))
sys.path.insert(1, str(BASE_DIR))

from injury_report import fetch_injuries, get_expected_injury_impact  # noqa: E402
from data_models import GameContext, Venue  # noqa: E402
from live_data import fetch_all_team_stats, fetch_roster, get_team_id  # noqa: E402
from market_mechanics import remove_vig  # noqa: E402
from probability_layers import calculate_injury_adjustment as calculate_probabilistic_injury_adjustment  # noqa: E402
from run_live import (  # noqa: E402
    IS_RENDER_RUNTIME,
    _pause_after_injury_lookup,
    _render_fast_injury_adjustment,
    create_team,
)


MODEL_SOURCE = "NBA Playoffs"
DEFAULT_SEASON = os.environ.get("NBA_MODEL_SEASON", "2025-26").strip() or "2025-26"
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
)
USER_AGENT = "Mozilla/5.0 PickLedgerPro NBAPlayoffs/1.0"
LEAGUE_AVG_RATING = 114.0
LEAGUE_AVG_PACE = 99.0
PLAYOFF_MARGIN_RMSE = 11.5

# Guardrail thresholds: real playoff edges past three games are rare; large
# moneyline "edges" on +200 dogs are almost always model overconfidence after
# the favorite has already shown what it does to the dog in Games 1 and 2.
BET_EDGE_THRESHOLD = 0.045
LEAN_EDGE_THRESHOLD = 0.025
BET_PROB_FLOOR = 0.55
LEAN_PROB_FLOOR = 0.51
MAX_TRUSTED_EDGE = 0.18
SPREAD_AGREEMENT_GAP_LEAN = 5.0
SPREAD_AGREEMENT_GAP_BET = 3.5
DOG_BET_PROB_FLOOR = 0.60
BIG_DOG_ODDS_THRESHOLD = 200
MAX_DOG_EDGE_FOR_BET = 0.12


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_target_date(raw_value: str | None) -> str:
    if not raw_value:
        return dt.date.today().isoformat()

    value = str(raw_value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return dt.date.today().isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NBA Playoffs model for a selected date.")
    parser.add_argument("legacy_date", nargs="?", default="", help="Optional date in MM/DD/YYYY or YYYY-MM-DD format.")
    parser.add_argument("--date", default="", help="Target date in YYYY-MM-DD or MM/DD/YYYY format.")
    parser.add_argument("--season", default=DEFAULT_SEASON, help="NBA API season string, e.g. 2025-26.")
    parser.add_argument("--no-log", action="store_true", help="Accepted for backend compatibility; this runner logs only to stdout.")
    return parser.parse_args()


def _request_json(url: str) -> dict[str, Any]:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _espn_date_key(date_str: str) -> str:
    return dt.date.fromisoformat(date_str).strftime("%Y%m%d")


def _parse_espn_datetime(value: str) -> dt.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _parse_american_odds(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip().replace("−", "-")
    if not text:
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _parse_line(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("−", "-")
    text = text.replace("o", "").replace("u", "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _competition_team(comp: dict[str, Any], home_away: str) -> dict[str, Any] | None:
    for competitor in comp.get("competitors", []) or []:
        if competitor.get("homeAway") == home_away:
            return competitor
    return None


def _extract_market(comp: dict[str, Any]) -> dict[str, Any]:
    odds_payload = (comp.get("odds") or [{}])[0] or {}
    moneyline = odds_payload.get("moneyline") or {}
    point_spread = odds_payload.get("pointSpread") or {}
    total = odds_payload.get("total") or {}

    return {
        "provider": ((odds_payload.get("provider") or {}).get("name") or "ESPN odds").strip(),
        "home_ml": _parse_american_odds(((moneyline.get("home") or {}).get("close") or {}).get("odds")),
        "away_ml": _parse_american_odds(((moneyline.get("away") or {}).get("close") or {}).get("odds")),
        "home_spread": _parse_line(((point_spread.get("home") or {}).get("close") or {}).get("line")),
        "away_spread": _parse_line(((point_spread.get("away") or {}).get("close") or {}).get("line")),
        "home_spread_odds": _parse_american_odds(((point_spread.get("home") or {}).get("close") or {}).get("odds")),
        "away_spread_odds": _parse_american_odds(((point_spread.get("away") or {}).get("close") or {}).get("odds")),
        "total_line": odds_payload.get("overUnder"),
        "over_odds": _parse_american_odds(((total.get("over") or {}).get("close") or {}).get("odds")),
        "under_odds": _parse_american_odds(((total.get("under") or {}).get("close") or {}).get("odds")),
        "details": str(odds_payload.get("details") or "").strip(),
    }


def _team_short_name(team: dict[str, Any]) -> str:
    name = str(team.get("name") or "").strip()
    display = str(team.get("displayName") or "").strip()
    if name:
        return name
    return display.split()[-1] if display else ""


def fetch_espn_playoff_games(date_str: str) -> list[dict[str, Any]]:
    url = f"{ESPN_SCOREBOARD_URL}?dates={_espn_date_key(date_str)}&seasontype=3"
    payload = _request_json(url)
    now_utc = dt.datetime.now(dt.timezone.utc)
    games: list[dict[str, Any]] = []

    for event in payload.get("events", []) or []:
        if ((event.get("season") or {}).get("type")) != 3:
            continue
        competitions = event.get("competitions") or []
        if not competitions:
            continue
        comp = competitions[0] or {}
        home = _competition_team(comp, "home")
        away = _competition_team(comp, "away")
        if not home or not away:
            continue

        status = comp.get("status") or {}
        status_type = status.get("type") or {}
        event_dt = _parse_espn_datetime(str(comp.get("date") or event.get("date") or ""))
        status_state = str(status_type.get("state") or "").strip().lower()
        has_started = (
            status_state != "pre"
            or bool(status_type.get("completed"))
            or (event_dt is not None and event_dt <= now_utc)
        )
        home_team = home.get("team") or {}
        away_team = away.get("team") or {}
        note = ""
        notes = comp.get("notes") or event.get("notes") or []
        if notes:
            note = str((notes[0] or {}).get("headline") or "").strip()

        games.append(
            {
                "game_id": str(event.get("id") or comp.get("id") or "").strip(),
                "slate_date": date_str,
                "date": event_dt.isoformat() if event_dt else str(event.get("date") or ""),
                "home_team": _team_short_name(home_team),
                "away_team": _team_short_name(away_team),
                "home_display": str(home_team.get("displayName") or "").strip(),
                "away_display": str(away_team.get("displayName") or "").strip(),
                "home_abbr": str(home_team.get("abbreviation") or "").strip(),
                "away_abbr": str(away_team.get("abbreviation") or "").strip(),
                "arena": ((comp.get("venue") or {}).get("fullName") or "").strip(),
                "game_status": str(status_type.get("shortDetail") or status_type.get("detail") or status_type.get("description") or "").strip(),
                "status_state": status_state,
                "has_started": has_started,
                "round": str(((comp.get("type") or {}).get("abbreviation") or "")).strip(),
                "series_status": note or "NBA Playoffs",
                "home_series_record": home.get("record"),
                "away_series_record": away.get("record"),
                "home_regular_record": next((r.get("summary") for r in home.get("records", []) if r.get("type") == "total"), ""),
                "away_regular_record": next((r.get("summary") for r in away.get("records", []) if r.get("type") == "total"), ""),
                "market": _extract_market(comp),
            }
        )

    return games


def _nba_team_key(full_name: str) -> str:
    if str(full_name) == "Portland Trail Blazers":
        return "Trail Blazers"
    return str(full_name or "").split()[-1]


def fetch_last20_context(season: str, as_of_date: str) -> dict[str, dict[str, float]]:
    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
    )
    frame = finder.get_data_frames()[0]
    if frame.empty:
        return {}
    frame = frame.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    target_dt = pd.Timestamp(dt.date.fromisoformat(as_of_date))
    frame = frame[frame["GAME_DATE"] < target_dt]

    result: dict[str, dict[str, float]] = {}
    for full_name, team_games in frame.groupby("TEAM_NAME"):
        recent = team_games.sort_values("GAME_DATE", ascending=False).head(20)
        if recent.empty:
            continue
        payload = {
            "last20_win_pct": float((recent["WL"] == "W").mean()),
            "last20_point_diff": float(recent["PLUS_MINUS"].astype(float).mean()),
        }
        result[full_name] = payload
        result[_nba_team_key(full_name)] = payload
    return result


def fetch_h2h_context(home_team: str, away_abbr: str, season: str, as_of_date: str) -> dict[str, Any]:
    team_id = get_team_id(home_team)
    if not team_id:
        return {"home_win_pct": 0.5, "games": 0, "point_diff": 0.0, "note": "H2H unavailable"}

    try:
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable="Regular Season",
            league_id_nullable="00",
        )
        frame = finder.get_data_frames()[0]
    except Exception as exc:
        return {
            "home_win_pct": 0.5,
            "games": 0,
            "point_diff": 0.0,
            "note": f"H2H unavailable because NBA API lookup failed ({exc})",
        }
    if frame.empty or "MATCHUP" not in frame.columns:
        return {"home_win_pct": 0.5, "games": 0, "point_diff": 0.0, "note": "H2H unavailable"}

    frame = frame.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    target_dt = pd.Timestamp(dt.date.fromisoformat(as_of_date))
    frame = frame[frame["GAME_DATE"] < target_dt]
    opponent = str(away_abbr or "").upper()
    h2h = frame[frame["MATCHUP"].astype(str).str.upper().str.contains(opponent, regex=False)]
    if h2h.empty:
        return {"home_win_pct": 0.5, "games": 0, "point_diff": 0.0, "note": "No current-season H2H found"}

    wins = float((h2h["WL"] == "W").mean())
    point_diff = float(h2h["PLUS_MINUS"].astype(float).mean())
    return {
        "home_win_pct": wins,
        "games": int(len(h2h)),
        "point_diff": point_diff,
        "note": f"{home_team} {int((h2h['WL'] == 'W').sum())}-{int((h2h['WL'] == 'L').sum())}, avg margin {point_diff:+.1f}",
    }


def fetch_series_history(
    home_team: str,
    away_abbr: str,
    season: str,
    as_of_date: str,
    home_abbr: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch playoff games already played in the current series.

    Returns a list of {date, is_home_for_target, margin_for_target, score_for_target,
    score_against_target} ordered earliest-first, from the perspective of
    ``home_team``. An empty list means Game 1 of the series (or that the API
    lookup failed — the caller should treat both as 'no in-series evidence').

    Tries the NBA API first; falls back to ESPN scoreboard scans for the
    18 days preceding ``as_of_date`` (long enough to cover even a Finals
    Game 1→Game 7 stretch) because the NBA API can lag a round behind
    during the playoffs.
    """
    games_via_nba = _fetch_series_history_nba(home_team, away_abbr, season, as_of_date)
    if games_via_nba:
        return games_via_nba
    if not home_abbr:
        return []
    return _fetch_series_history_espn(home_abbr, away_abbr, as_of_date)


def _fetch_series_history_nba(
    home_team: str,
    away_abbr: str,
    season: str,
    as_of_date: str,
) -> list[dict[str, Any]]:
    team_id = get_team_id(home_team)
    if not team_id:
        return []

    try:
        finder = leaguegamefinder.LeagueGameFinder(
            team_id_nullable=team_id,
            season_nullable=season,
            season_type_nullable="Playoffs",
            league_id_nullable="00",
        )
        frame = finder.get_data_frames()[0]
    except Exception:
        return []

    if frame.empty or "MATCHUP" not in frame.columns:
        return []

    frame = frame.copy()
    frame["GAME_DATE"] = pd.to_datetime(frame["GAME_DATE"])
    target_dt = pd.Timestamp(dt.date.fromisoformat(as_of_date))
    frame = frame[frame["GAME_DATE"] < target_dt]
    opponent = str(away_abbr or "").upper()
    if not opponent:
        return []
    series = frame[frame["MATCHUP"].astype(str).str.upper().str.contains(opponent, regex=False)]
    if series.empty:
        return []

    series = series.sort_values("GAME_DATE", ascending=True)
    games: list[dict[str, Any]] = []
    for _, row in series.iterrows():
        matchup_text = str(row.get("MATCHUP") or "")
        is_home = " VS " in matchup_text.upper()
        try:
            margin = float(row.get("PLUS_MINUS"))
            pts = float(row.get("PTS"))
        except (TypeError, ValueError):
            continue
        games.append({
            "date": row["GAME_DATE"].date().isoformat(),
            "is_home_for_target": is_home,
            "margin_for_target": margin,
            "score_for_target": pts,
            "score_against_target": pts - margin,
            "result": str(row.get("WL") or ""),
        })
    return games


SERIES_HISTORY_ESPN_LOOKBACK_DAYS = 18  # Finals Game 1→7 spans up to 17 days
SERIES_HISTORY_MAX_GAMES = 6  # max prior games in a best-of-7 (game 7 is the cap)


def _fetch_series_history_espn(
    home_abbr: str,
    away_abbr: str,
    as_of_date: str,
    max_lookback_days: int = SERIES_HISTORY_ESPN_LOOKBACK_DAYS,
) -> list[dict[str, Any]]:
    """Scan ESPN's postseason scoreboard for completed games between
    ``home_abbr`` and ``away_abbr`` in the prior ``max_lookback_days``.

    Default 18 days covers the worst-case Finals series (Game 1 → Game 7
    can stretch 16-17 days when ABC asks for an off-day extension between
    every game). Earlier rounds compress to ~10 days, so the loop short-
    circuits as soon as we have ``SERIES_HISTORY_MAX_GAMES`` (6) prior
    games — enough to fully describe a best-of-7 series before Game 7.
    """
    home = str(home_abbr or "").upper()
    away = str(away_abbr or "").upper()
    if not home or not away:
        return []
    try:
        target = dt.date.fromisoformat(as_of_date)
    except ValueError:
        return []

    games: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    # range(1, N+1) so offset=1 is yesterday and offset=18 is 18 days back.
    for offset in range(1, max_lookback_days + 1):
        if len(games) >= SERIES_HISTORY_MAX_GAMES:
            break
        scan_date = (target - dt.timedelta(days=offset)).strftime("%Y%m%d")
        url = f"{ESPN_SCOREBOARD_URL}?dates={scan_date}&seasontype=3"
        try:
            payload = _request_json(url)
        except Exception:
            continue
        for event in payload.get("events", []) or []:
            event_id = str(event.get("id") or "")
            if event_id in seen_ids:
                continue
            comp = (event.get("competitions") or [{}])[0] or {}
            status = ((comp.get("status") or {}).get("type") or {})
            if not status.get("completed"):
                continue
            home_side = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), {})
            away_side = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), {})
            home_e = str((home_side.get("team") or {}).get("abbreviation") or "").upper()
            away_e = str((away_side.get("team") or {}).get("abbreviation") or "").upper()
            if {home_e, away_e} != {home, away}:
                continue
            try:
                home_score = float(home_side.get("score"))
                away_score = float(away_side.get("score"))
            except (TypeError, ValueError):
                continue
            target_is_home_in_event = (home_e == home)
            if target_is_home_in_event:
                margin_for_target = home_score - away_score
                score_for_target = home_score
                score_against_target = away_score
            else:
                margin_for_target = away_score - home_score
                score_for_target = away_score
                score_against_target = home_score
            games.append({
                "date": (target - dt.timedelta(days=offset)).isoformat(),
                "is_home_for_target": target_is_home_in_event,
                "margin_for_target": margin_for_target,
                "score_for_target": score_for_target,
                "score_against_target": score_against_target,
                "result": "W" if margin_for_target > 0 else "L",
            })
            seen_ids.add(event_id)

    games.sort(key=lambda g: g["date"])
    return games


def compute_series_form_signal(
    series_results: list[dict[str, Any]],
    base_rmse: float,
) -> dict[str, Any]:
    """Derive a Bayesian-style update from prior games in the current series.

    The pre-patch model fired a BET on Lakers ML (+295) at home in Game 3 of
    the OKC/LAL semifinal because its base rate was season-stats only — it
    treated the matchup as if Games 1 and 2 had not happened. In reality
    Lakers had lost the first two games by 18 each, which is the strongest
    possible signal that the model's season-derived view was wrong for this
    specific matchup. Series form is therefore weighted as the dominant
    in-context input, with weight scaling as sqrt(games) and capped at 55%
    so a Game 7 sample doesn't crowd out everything else.
    """
    if not series_results:
        return {
            "games": 0,
            "avg_margin": 0.0,
            "max_abs_margin": 0.0,
            "implied_prob_for_home": None,
            "evidence_weight": 0.0,
            "margin_shift": 0.0,
            "rmse_inflation": 0.0,
            "note": "no prior games in this series",
        }

    margins = [float(g.get("margin_for_target") or 0.0) for g in series_results]
    games = len(margins)
    avg_margin = sum(margins) / games if games else 0.0
    max_abs_margin = max((abs(m) for m in margins), default=0.0)

    # Predictive variance for the next game = base + sampling error of the
    # mean estimate, so a 2-game sample is wider than a 5-game sample. This
    # keeps small samples from producing 5%-or-95% certainty.
    predictive_stdev = max(base_rmse * math.sqrt(1.0 + 1.0 / max(games, 1)), 1.0)
    implied_prob = _clamp(_normal_cdf(avg_margin / predictive_stdev), 0.03, 0.97)

    # Evidence weight grows with sqrt(games) — diminishing returns past 3-4
    # games. Capped so it can never crowd out the season-stats baseline.
    evidence_weight = min(0.50, 0.16 * math.sqrt(games))
    # Direct margin contribution — ~45% of the observed series margin is
    # treated as evidence about the matchup-true scoring margin.
    margin_shift = avg_margin * 0.45
    # Inflate margin RMSE if blowouts have occurred — the matchup is more
    # variable than a typical regular-season game.
    rmse_inflation = max(0.0, (max_abs_margin - 12.0) * 0.20)

    return {
        "games": games,
        "avg_margin": avg_margin,
        "max_abs_margin": max_abs_margin,
        "implied_prob_for_home": implied_prob,
        "evidence_weight": evidence_weight,
        "margin_shift": margin_shift,
        "rmse_inflation": rmse_inflation,
        "predictive_stdev": predictive_stdev,
        "note": (
            f"home avg margin {avg_margin:+.1f} over {games} game(s); "
            f"biggest |margin| {max_abs_margin:.0f}; predictive stdev {predictive_stdev:.1f}"
        ),
    }


def _rank_lookup(all_team_stats: dict[str, dict[str, Any]]) -> dict[str, int]:
    unique: dict[str, dict[str, Any]] = {}
    for key, payload in all_team_stats.items():
        full_name = str(payload.get("full_name") or key).strip()
        if full_name and full_name not in unique:
            unique[full_name] = payload

    ordered = sorted(
        unique.items(),
        key=lambda item: float(item[1].get("win_pct", 0.0) or 0.0),
        reverse=True,
    )
    ranks: dict[str, int] = {}
    for idx, (full_name, payload) in enumerate(ordered, start=1):
        ranks[full_name] = idx
        ranks[_nba_team_key(full_name)] = idx
        short = str(payload.get("nickname") or "").strip()
        if short:
            ranks[short] = idx
    return ranks


def _safe_pct(value: Any, fallback: float = 0.5) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return _clamp(number, 0.0, 1.0)


def _stat(stats: Any, attr: str, fallback: float) -> float:
    try:
        value = float(getattr(stats, attr, fallback))
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(value):
        return fallback
    return value


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _home_is_altitude(team_name: str) -> bool:
    normalized = str(team_name or "").upper()
    return any(marker in normalized for marker in ("NUGGETS", "DENVER", "JAZZ", "UTAH"))


def _record_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("displayValue", "summary", "name", "abbreviation"):
            text = str(value.get(key) or "").strip()
            if text:
                return text
    return str(value or "").strip()


def _parse_series_record(value: Any) -> tuple[int, int] | None:
    text = _record_text(value)
    match = re.search(r"(\d+)\s*[-–]\s*(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_series_context(game: dict[str, Any]) -> dict[str, Any]:
    headline = str(game.get("series_status") or "NBA Playoffs").strip()
    game_number_match = re.search(r"\bGame\s+(\d+)\b", headline, flags=re.IGNORECASE)
    game_number = int(game_number_match.group(1)) if game_number_match else 1

    home_record = _parse_series_record(game.get("home_series_record"))
    away_record = _parse_series_record(game.get("away_series_record"))
    home_wins = home_record[0] if home_record else None
    away_wins = away_record[0] if away_record else None

    if home_record and away_record is None:
        home_wins, away_wins = home_record
    elif away_record and home_record is None:
        away_wins, home_wins = away_record

    if home_wins is None or away_wins is None:
        lead_match = re.search(
            r"(.+?)\s+leads\s+series\s+(\d+)\s*[-–]\s*(\d+)",
            headline,
            flags=re.IGNORECASE,
        )
        tied_match = re.search(r"series\s+tied\s+(\d+)\s*[-–]\s*(\d+)", headline, flags=re.IGNORECASE)
        if tied_match:
            home_wins = int(tied_match.group(1))
            away_wins = int(tied_match.group(2))
        elif lead_match:
            leader = lead_match.group(1).strip().lower()
            first = int(lead_match.group(2))
            second = int(lead_match.group(3))
            home_name = str(game.get("home_team") or "").lower()
            away_name = str(game.get("away_team") or "").lower()
            if home_name and home_name in leader:
                home_wins, away_wins = first, second
            elif away_name and away_name in leader:
                away_wins, home_wins = first, second

    if home_wins is None:
        home_wins = 0
    if away_wins is None:
        away_wins = 0

    repeat_matchups = max(0, game_number - 1, home_wins + away_wins)
    return {
        "round": str(game.get("round") or "Playoffs").strip() or "Playoffs",
        "headline": headline,
        "game_number": game_number,
        "is_game_1": game_number == 1,
        "is_game_2": game_number == 2,
        "is_game_7": game_number == 7,
        "home_wins": home_wins,
        "away_wins": away_wins,
        "home_trailing": home_wins < away_wins,
        "away_trailing": away_wins < home_wins,
        "home_elimination": away_wins >= 3,
        "away_elimination": home_wins >= 3,
        "home_closeout": home_wins >= 3,
        "away_closeout": away_wins >= 3,
        "repeat_matchups": repeat_matchups,
    }


def calculate_base_rate(
    home_team: str,
    away_team: str,
    home_stats: dict[str, Any],
    last20_context: dict[str, dict[str, float]],
    ranks: dict[str, int],
    h2h: dict[str, Any],
    series_form: dict[str, Any] | None = None,
) -> tuple[float, list[str]]:
    season_win_pct = _safe_pct(home_stats.get("win_pct"))
    last20 = last20_context.get(home_team, {})
    last20_win_pct = _safe_pct(last20.get("last20_win_pct"), _safe_pct(home_stats.get("recent_10_win_pct")))

    h2h_component = _safe_pct(h2h.get("home_win_pct"), 0.5)
    home_rank = ranks.get(home_team)
    away_rank = ranks.get(away_team)
    if home_rank and away_rank:
        seeding_component = _clamp(0.50 + ((away_rank - home_rank) * 0.015), 0.35, 0.65)
        h2h_seed_component = (h2h_component * 0.60) + (seeding_component * 0.40)
        seed_note = f"rank proxy {home_team} #{home_rank} vs {away_team} #{away_rank}"
    else:
        h2h_seed_component = h2h_component
        seed_note = "rank proxy unavailable"

    base = (season_win_pct * 0.40) + (last20_win_pct * 0.30) + (h2h_seed_component * 0.30)
    notes = [
        f"season win% {season_win_pct*100:.1f}% x40%",
        f"last-20 win% {last20_win_pct*100:.1f}% x30%",
        f"H2H/seed component {h2h_seed_component*100:.1f}% x30% ({h2h.get('note')}; {seed_note})",
    ]

    # Bayesian update from in-series games. The series itself is direct
    # evidence about the matchup that season stats cannot capture.
    if series_form and series_form.get("games", 0) > 0 and series_form.get("implied_prob_for_home") is not None:
        evidence_weight = float(series_form.get("evidence_weight", 0.0) or 0.0)
        prior_base = base
        base = (prior_base * (1.0 - evidence_weight)) + (float(series_form["implied_prob_for_home"]) * evidence_weight)
        notes.append(
            f"series-form blend (weight {evidence_weight*100:.0f}%, "
            f"implied home {series_form['implied_prob_for_home']*100:.1f}%): {series_form.get('note', '')}"
        )

    return _clamp(base, 0.05, 0.95), notes


def _injury_adjustment(team_name: str, injuries: dict[str, list[dict[str, Any]]]) -> tuple[float, str, int]:
    expected = get_expected_injury_impact(injuries, team_name)
    if not expected:
        return 0.0, "No listed expected absences", 0

    if IS_RENDER_RUNTIME:
        adj, reason = _render_fast_injury_adjustment(team_name, expected)
    else:
        adj, reason = calculate_probabilistic_injury_adjustment(team_name, expected)
        _pause_after_injury_lookup()
    return adj, reason, len(expected)


def _playoff_injury_profile(
    home_name: str,
    away_name: str,
    injuries: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    home_raw, home_reason, home_count = _injury_adjustment(home_name, injuries)
    away_raw, away_reason, away_count = _injury_adjustment(away_name, injuries)

    # Playoff rotations are shorter, so an expected absence costs more than it
    # does in the regular-season NBA model. Keep the cap inside the prompt's
    # star/rotation absence range instead of letting injury news dominate.
    home_playoff = _clamp(home_raw * 1.45, -0.14, 0.04)
    away_playoff = _clamp(away_raw * 1.45, -0.14, 0.04)
    prob_delta = _clamp(home_playoff - away_playoff, -0.14, 0.14)

    return {
        "home_raw": home_raw,
        "away_raw": away_raw,
        "home_adjustment": home_playoff,
        "away_adjustment": away_playoff,
        "prob_delta": prob_delta,
        "margin_delta": _clamp(prob_delta * 34.0, -5.5, 5.5),
        "home_reason": home_reason,
        "away_reason": away_reason,
        "home_count": home_count,
        "away_count": away_count,
        "label": "1.45x playoff injury multiplier",
    }


def calculate_playoff_tempo(
    away_stats: Any,
    home_stats: Any,
    series_context: dict[str, Any],
) -> dict[str, Any]:
    home_pace = _stat(home_stats, "pace", LEAGUE_AVG_PACE)
    away_pace = _stat(away_stats, "pace", LEAGUE_AVG_PACE)
    neutral_pace = (home_pace + away_pace) / 2.0

    def control_score(team_stats: Any, opponent_stats: Any) -> float:
        pace_resistance = (LEAGUE_AVG_PACE - _stat(team_stats, "pace", LEAGUE_AVG_PACE)) * 0.08
        defense = (LEAGUE_AVG_RATING - _stat(team_stats, "def_rating_10", LEAGUE_AVG_RATING)) * 0.035
        dreb = (_stat(team_stats, "dreb_pct", 0.72) - 0.72) * 5.0
        force_turnovers = (_stat(team_stats, "opp_tov_pct", 0.135) - 0.135) * 8.0
        recent = _stat(team_stats, "recent_10_point_diff", _stat(team_stats, "net_rating", 0.0)) * 0.025
        opponent_run_game = (_stat(opponent_stats, "pace", LEAGUE_AVG_PACE) - LEAGUE_AVG_PACE) * 0.02
        return pace_resistance + defense + dreb + force_turnovers + recent - opponent_run_game

    home_control = control_score(home_stats, away_stats)
    away_control = control_score(away_stats, home_stats)
    control_gap = home_control - away_control
    if abs(control_gap) < 0.15:
        home_weight = 0.50
    else:
        home_weight = _clamp(0.50 + (control_gap * 0.07), 0.30, 0.70)
    away_weight = 1.0 - home_weight
    dictated_regular_pace = (home_pace * home_weight) + (away_pace * away_weight)

    game_number = int(series_context.get("game_number", 1) or 1)
    pace_drag = 1.6
    if game_number == 2:
        pace_drag = 2.2
    elif game_number in (3, 4):
        pace_drag = 2.8
    elif game_number >= 5:
        pace_drag = 3.5
    if series_context.get("home_elimination") or series_context.get("away_elimination"):
        pace_drag += 0.7
    if series_context.get("is_game_7"):
        pace_drag += 0.8

    playoff_pace = _clamp(dictated_regular_pace - pace_drag, 89.0, 101.5)
    halfcourt_weight = _clamp(
        0.58 + (min(game_number - 1, 5) * 0.035) + (0.05 if series_context.get("home_elimination") or series_context.get("away_elimination") else 0.0),
        0.58,
        0.82,
    )

    if home_weight > away_weight + 0.03:
        dictating_side = "home"
    elif away_weight > home_weight + 0.03:
        dictating_side = "away"
    else:
        dictating_side = "neutral"

    return {
        "dictated_pace": playoff_pace,
        "playoff_pace": playoff_pace,
        "regular_dictated_pace": dictated_regular_pace,
        "neutral_pace": neutral_pace,
        "pace_drag": pace_drag,
        "pace_factor": playoff_pace / neutral_pace if neutral_pace else 1.0,
        "halfcourt_weight": halfcourt_weight,
        "home_weight": home_weight,
        "away_weight": away_weight,
        "control_gap": control_gap,
        "dictating_side": dictating_side,
        "home_control_score": home_control,
        "away_control_score": away_control,
        "home_reason": f"slowdown/defense control {home_control:+.2f}",
        "away_reason": f"slowdown/defense control {away_control:+.2f}",
    }


def _playoff_expected_rating(team_stats: Any, opponent_stats: Any, halfcourt_weight: float) -> float:
    offense = _stat(team_stats, "off_rating_10", LEAGUE_AVG_RATING)
    opponent_defense = _stat(opponent_stats, "def_rating_10", LEAGUE_AVG_RATING)
    base_rating = (offense * 0.54) + (opponent_defense * 0.46)

    efg_edge = (_stat(team_stats, "efg_pct", 0.54) - _stat(opponent_stats, "efg_pct", 0.54)) * 34.0
    turnover_edge = (_stat(opponent_stats, "tov_pct", 0.135) - _stat(team_stats, "tov_pct", 0.135)) * 18.0
    rebounding_edge = (_stat(team_stats, "reb_pct", 0.50) - _stat(opponent_stats, "reb_pct", 0.50)) * 14.0
    halfcourt_adj = (efg_edge + turnover_edge + rebounding_edge) * halfcourt_weight

    return _clamp(base_rating + halfcourt_adj, 99.0, 128.0)


def _home_court_points(home_name: str, series_context: dict[str, Any]) -> float:
    # Modern NBA playoff home-court is ~3.0-3.5 pts per Inpredictable / 538
    # tracking; the pre-patch 4.6 inflated home favorites and home dogs alike.
    points = 3.5 if not _home_is_altitude(home_name) else 4.2
    if series_context.get("is_game_7"):
        points += 0.7
    elif series_context.get("home_elimination") or series_context.get("home_closeout"):
        points += 0.4
    if series_context.get("is_game_1"):
        points += 0.2
    return points


def predict_playoff_margin(
    home_team: Any,
    away_team: Any,
    tempo_context: dict[str, Any],
    series_context: dict[str, Any],
    h2h: dict[str, Any],
    injury_profile: dict[str, Any],
    series_form: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    home_stats = home_team.team_stats
    away_stats = away_team.team_stats
    pace = float(tempo_context["playoff_pace"])
    halfcourt_weight = float(tempo_context["halfcourt_weight"])

    home_rating = _playoff_expected_rating(home_stats, away_stats, halfcourt_weight)
    away_rating = _playoff_expected_rating(away_stats, home_stats, halfcourt_weight)
    scoring_margin = ((home_rating - away_rating) * pace / 100.0)

    home_court = _home_court_points(home_team.name, series_context)
    net_diff = _stat(home_stats, "net_rating", 0.0) - _stat(away_stats, "net_rating", 0.0)
    star_minutes = _clamp(net_diff * 0.09, -2.2, 2.2)

    # Series state bonuses are tiny tie-breakers now — the in-series margin
    # signal carries the real "what's actually happening in this series"
    # information, so we don't double-count "down 0-2 home comeback" hopes.
    series_points = 0.0
    if series_context.get("is_game_1"):
        series_points += 0.3
    if series_context.get("home_elimination"):
        series_points += 0.6
    if series_context.get("away_elimination"):
        series_points -= 0.6
    if series_context.get("home_closeout"):
        series_points += 0.3
    if series_context.get("away_closeout"):
        series_points -= 0.3

    repeat_matchups = int(series_context.get("repeat_matchups", 0) or 0)
    coaching = 0.0
    if repeat_matchups:
        if series_context.get("home_trailing"):
            coaching += min(0.4, repeat_matchups * 0.10)
        if series_context.get("away_trailing"):
            coaching -= min(0.4, repeat_matchups * 0.10)

    rest_diff = _stat(home_stats, "rest_days", 1.0) - _stat(away_stats, "rest_days", 1.0)
    rest = _clamp(rest_diff * 0.55, -1.6, 1.6)

    pace_control = 0.0
    dictating_side = str(tempo_context.get("dictating_side") or "neutral")
    if dictating_side == "home":
        pace_control = 0.7 if net_diff >= 0 else 0.25
    elif dictating_side == "away":
        pace_control = -0.7 if net_diff <= 0 else -0.25

    injury = float(injury_profile.get("margin_delta", 0.0) or 0.0)

    # Direct in-series margin evidence — the dominant signal once the series
    # has had at least one game. We use 55% of the avg series margin as the
    # best estimate of the matchup-specific advantage.
    series_margin_signal = float((series_form or {}).get("margin_shift", 0.0) or 0.0)

    margin = (
        scoring_margin
        + home_court
        + star_minutes
        + series_points
        + coaching
        + rest
        + pace_control
        + injury
        + series_margin_signal
    )
    return _clamp(margin, -26.0, 26.0), {
        "scoring_margin": scoring_margin,
        "home_court": home_court,
        "star_minutes": star_minutes,
        "series_state": series_points,
        "coaching_adjustment": coaching,
        "rest": rest,
        "pace_control": pace_control,
        "injury": injury,
        "series_form_margin": series_margin_signal,
        "home_expected_rating": home_rating,
        "away_expected_rating": away_rating,
    }


def predict_playoff_total(
    home_team: Any,
    away_team: Any,
    tempo_context: dict[str, Any],
    series_context: dict[str, Any],
    injury_profile: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    pace = float(tempo_context["playoff_pace"])
    halfcourt_weight = float(tempo_context["halfcourt_weight"])
    home_rating = _playoff_expected_rating(home_team.team_stats, away_team.team_stats, halfcourt_weight)
    away_rating = _playoff_expected_rating(away_team.team_stats, home_team.team_stats, halfcourt_weight)
    base_total = ((home_rating + away_rating) * pace / 100.0)
    physicality_drag = max(0.0, (halfcourt_weight - 0.58) * 9.0)
    if series_context.get("home_elimination") or series_context.get("away_elimination"):
        physicality_drag += 1.2
    if series_context.get("is_game_7"):
        physicality_drag += 1.8
    availability_drag = (int(injury_profile.get("home_count", 0) or 0) + int(injury_profile.get("away_count", 0) or 0)) * 0.55
    total = _clamp(base_total - physicality_drag - availability_drag, 178.0, 252.0)
    return total, {
        "base_total": base_total,
        "physicality_drag": physicality_drag,
        "availability_drag": availability_drag,
        "home_expected_rating": home_rating,
        "away_expected_rating": away_rating,
    }


def _build_adjustments(
    game: dict[str, Any],
    home_team,
    away_team,
    injuries: dict[str, list[dict[str, Any]]],
    tempo_context: dict[str, Any],
    series_context: dict[str, Any],
    h2h: dict[str, Any],
    injury_profile: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    adjustments: list[dict[str, Any]] = []

    home_name = home_team.name
    away_name = away_team.name

    hca = 0.052 if _home_is_altitude(home_name) else 0.046
    if series_context.get("is_game_7"):
        hca += 0.008
    elif series_context.get("home_elimination") or series_context.get("home_closeout"):
        hca += 0.004
    adjustments.append({"label": "Stronger playoff home court", "value": _clamp(hca, 0.035, 0.060), "reason": f"{home_name} home playoff game; game {series_context.get('game_number')}"})

    home_inj_reason = str(injury_profile.get("home_reason") or "No listed expected absences")
    away_inj_reason = str(injury_profile.get("away_reason") or "No listed expected absences")
    injury_delta = float(injury_profile.get("prob_delta", 0.0) or 0.0)
    if injury_delta:
        adjustments.append({"label": "Strict playoff injury weighting", "value": injury_delta, "reason": f"{injury_profile.get('label')}; {home_name}: {home_inj_reason}; {away_name}: {away_inj_reason}"})
    elif injury_profile.get("home_count") or injury_profile.get("away_count"):
        adjustments.append({"label": "Strict playoff injury weighting", "value": 0.0, "reason": f"{home_name}: {home_inj_reason}; {away_name}: {away_inj_reason}"})

    home_stats = home_team.team_stats
    away_stats = away_team.team_stats
    halfcourt_weight = float(tempo_context.get("halfcourt_weight", 0.62) or 0.62)
    home_attack = (_stat(home_stats, "off_rating_10", LEAGUE_AVG_RATING) - LEAGUE_AVG_RATING) + (_stat(away_stats, "def_rating_10", LEAGUE_AVG_RATING) - LEAGUE_AVG_RATING)
    away_attack = (_stat(away_stats, "off_rating_10", LEAGUE_AVG_RATING) - LEAGUE_AVG_RATING) + (_stat(home_stats, "def_rating_10", LEAGUE_AVG_RATING) - LEAGUE_AVG_RATING)
    four_factor_edge = (
        (_stat(home_stats, "efg_pct", 0.54) - _stat(away_stats, "efg_pct", 0.54)) * 1.25
        + (_stat(home_stats, "reb_pct", 0.50) - _stat(away_stats, "reb_pct", 0.50)) * 0.75
        + (_stat(away_stats, "tov_pct", 0.135) - _stat(home_stats, "tov_pct", 0.135)) * 0.80
    )
    mismatch_adj = _clamp(((home_attack - away_attack) * 0.0065) + (four_factor_edge * halfcourt_weight), -0.06, 0.06)
    if abs(mismatch_adj) >= 0.005:
        adjustments.append({
            "label": "Playoff matchup/halfcourt mismatch",
            "value": mismatch_adj,
            "reason": f"attack score {home_name} {home_attack:+.1f} vs {away_name} {away_attack:+.1f}; halfcourt weight {halfcourt_weight:.2f}",
        })

    rest_diff = _stat(home_stats, "rest_days", 1.0) - _stat(away_stats, "rest_days", 1.0)
    rest_adj = _clamp(rest_diff * 0.015, -0.03, 0.03)
    if abs(rest_adj) >= 0.005:
        adjustments.append({
            "label": "Rest/travel",
            "value": rest_adj,
            "reason": f"rest days {home_name} {_stat(home_stats, 'rest_days', 1.0):.0f} vs {away_name} {_stat(away_stats, 'rest_days', 1.0):.0f}",
        })

    dictating_side = str(tempo_context.get("dictating_side") or "neutral")
    pace_adj = 0.0
    if dictating_side == "home":
        pace_adj = 0.020 if _stat(home_stats, "net_rating", 0.0) >= _stat(away_stats, "net_rating", 0.0) else 0.008
    elif dictating_side == "away":
        pace_adj = -0.020 if _stat(away_stats, "net_rating", 0.0) >= _stat(home_stats, "net_rating", 0.0) else -0.008
    if pace_adj:
        adjustments.append({
            "label": "Playoff pace control",
            "value": pace_adj,
            "reason": f"{dictating_side} tempo control, playoff pace {tempo_context.get('playoff_pace', 0.0):.1f} after {tempo_context.get('pace_drag', 0.0):.1f}-possession drag",
        })

    net_diff = _stat(home_stats, "net_rating", 0.0) - _stat(away_stats, "net_rating", 0.0)
    star_minutes_adj = _clamp(net_diff * 0.0035, -0.030, 0.030)
    if abs(star_minutes_adj) >= 0.004:
        adjustments.append({
            "label": "Shorter rotations / star minutes",
            "value": star_minutes_adj,
            "reason": f"playoff minutes shift toward top-end quality; net rating gap {net_diff:+.1f}",
        })

    # Series-state probability adjustments here are intentionally tiny —
    # the in-series margin signal already reflects what's actually happening
    # in the matchup, so we don't double-add narrative bonuses for it.
    series_adj = 0.0
    series_notes: list[str] = []
    if series_context.get("is_game_1"):
        series_adj += 0.004
        series_notes.append("Game 1 prep edge to home team")
    if series_context.get("home_elimination"):
        series_adj += 0.008
        series_notes.append(f"{home_name} elimination urgency")
    if series_context.get("away_elimination"):
        series_adj -= 0.008
        series_notes.append(f"{away_name} elimination urgency")
    if series_context.get("home_closeout"):
        series_adj += 0.004
        series_notes.append(f"{home_name} closeout chance")
    if series_context.get("away_closeout"):
        series_adj -= 0.004
        series_notes.append(f"{away_name} closeout chance")
    if abs(series_adj) >= 0.003:
        adjustments.append({
            "label": "Series state",
            "value": _clamp(series_adj, -0.012, 0.012),
            "reason": "; ".join(series_notes),
        })

    repeated = int(series_context.get("repeat_matchups", 0) or 0)
    coaching_adj = 0.0
    coaching_notes: list[str] = []
    if repeated:
        if series_context.get("home_trailing"):
            coaching_adj += min(0.006, repeated * 0.0015)
            coaching_notes.append(f"{home_name} adjustment opportunity after repeated matchups")
        if series_context.get("away_trailing"):
            coaching_adj -= min(0.006, repeated * 0.0015)
            coaching_notes.append(f"{away_name} adjustment opportunity after repeated matchups")
    if abs(coaching_adj) >= 0.003:
        adjustments.append({
            "label": "Coaching/repeated-matchup adjustment",
            "value": _clamp(coaching_adj, -0.012, 0.012),
            "reason": "; ".join(coaching_notes),
        })

    return adjustments, home_inj_reason, away_inj_reason


def extremize_probability(raw_prob: float) -> float:
    """
    Directional, bounded version of the prompt's confidence-term extremizer.

    The literal expression base * (1 - base) * 4 + base exceeds 1.0 for
    ordinary probabilities, so this uses that term as the strength of a
    directional move away from 50%.
    """
    raw = _clamp(raw_prob, 0.01, 0.99)
    if abs(raw - 0.50) < 1e-9:
        return 0.50
    confidence_term = raw * (1.0 - raw) * 4.0
    # Damp the shift inside the noisy 50%-58% band so a small base-rate
    # advantage doesn't get amplified into a confident-looking pick.
    shift_strength = 0.55 if abs(raw - 0.50) <= 0.08 else 0.85
    directional_shift = math.copysign(
        abs(raw - 0.50) * confidence_term * shift_strength,
        raw - 0.50,
    )
    return _clamp(raw + directional_shift, 0.03, 0.97)


def _american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def _quarter_kelly_units(edge: float, american_odds: int) -> float:
    decimal_odds = _american_to_decimal(american_odds)
    b = decimal_odds - 1.0
    if b <= 0 or edge <= 0:
        return 0.0
    return min(2.0, edge / b / 4.0)


def _format_odds(odds: int | None) -> str:
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def _verify_roster(team_name: str, season: str) -> tuple[bool, str]:
    try:
        roster = fetch_roster(team_name, season=season)
    except Exception as exc:
        return False, f"NBA API roster lookup failed: {exc}"
    if not roster:
        return False, "NBA API roster lookup returned no players"
    source = str(roster[0].get("source") or "NBA API") if isinstance(roster[0], dict) else "NBA API"
    return True, f"{len(roster)} active roster entries fetched from {source}"


def _market_pick_spread(market: dict[str, Any], pick_team: str, home_name: str, away_name: str) -> float | None:
    """Return the market spread *for the picked team's perspective* if known."""
    if not market:
        return None
    if pick_team == home_name and market.get("home_spread") is not None:
        try:
            return float(market.get("home_spread"))
        except (TypeError, ValueError):
            return None
    if pick_team == away_name and market.get("away_spread") is not None:
        try:
            return float(market.get("away_spread"))
        except (TypeError, ValueError):
            return None
    if market.get("home_spread") is not None:
        try:
            home = float(market.get("home_spread"))
        except (TypeError, ValueError):
            return None
        return -home if pick_team == away_name else home
    return None


def evaluate_playoff_decision(
    pick_team: str,
    pick_prob: float,
    pick_odds: int,
    edge: float,
    predicted_spread: float,
    market: dict[str, Any],
    home_name: str,
    away_name: str,
    injuries: dict[str, list[dict[str, Any]]] | None,
    series_context: dict[str, Any],
    adjustments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify a playoff pick as BET, LEAN, or PASS with explicit reasons.

    The pre-patch model fired BETs whenever the moneyline edge exceeded 3% and
    the model spread did not strictly disagree with the pick side. That gate
    let large +200 dog edges and 0-2-down comeback narratives slip through
    even when the model's own spread layer was nowhere near the market line.
    These guardrails layer real-money sanity checks on top of that signal.
    """
    reasons: list[str] = []

    # Rule: never trust an outsized edge unconditionally — clamp the BET path.
    if edge >= MAX_TRUSTED_EDGE:
        reasons.append(f"edge {edge*100:.1f}% above {MAX_TRUSTED_EDGE*100:.0f}% trust ceiling — likely model overconfidence")

    # Rule: minimum win-probability floor so 50.x% picks can't BET on +odds.
    if pick_prob < BET_PROB_FLOOR:
        reasons.append(f"pick prob {pick_prob*100:.1f}% < {BET_PROB_FLOOR*100:.0f}% BET floor")

    # Rule: spread layer must broadly agree with the market line.
    market_pick_spread = _market_pick_spread(market, pick_team, home_name, away_name)
    spread_gap: float | None = None
    if market_pick_spread is not None:
        # Convention: predicted_spread is home margin. Convert to pick-perspective.
        pick_predicted = predicted_spread if pick_team == home_name else -predicted_spread
        # Market spread sign convention: a -4.5 favorite covers if margin >= 4.5.
        # We compare model margin vs (-market_spread) implied margin.
        market_implied_margin = -market_pick_spread
        spread_gap = pick_predicted - market_implied_margin
        if abs(spread_gap) >= SPREAD_AGREEMENT_GAP_LEAN:
            reasons.append(
                f"model margin {pick_predicted:+.1f} disagrees with market line ({market_pick_spread:+.1f}) by {spread_gap:+.1f} pts"
            )

    # Rule: dog-bet sanity — long dogs need stronger, not weaker, conviction.
    is_big_dog = pick_odds >= BIG_DOG_ODDS_THRESHOLD
    if is_big_dog:
        if pick_prob < DOG_BET_PROB_FLOOR:
            reasons.append(
                f"dog at +{pick_odds} needs >= {DOG_BET_PROB_FLOOR*100:.0f}% conviction; we have {pick_prob*100:.1f}%"
            )
        if edge > MAX_DOG_EDGE_FOR_BET:
            reasons.append(f"dog edge {edge*100:.1f}% above {MAX_DOG_EDGE_FOR_BET*100:.0f}% sanity ceiling")

    # Rule: 0-2 home comeback at Game 3 historically flips into a sweep more
    # often than not; do not let the residual context bonuses force a BET.
    in_sweep_window = (
        series_context.get("game_number") in (3, 4)
        and (
            (series_context.get("home_wins") == 0 and series_context.get("away_wins") == 2)
            or (series_context.get("away_wins") == 0 and series_context.get("home_wins") == 2)
        )
    )
    if in_sweep_window and pick_prob < 0.58:
        reasons.append("comeback-from-0-2 narrative — capped at LEAN unless model has >=58% conviction")

    # Rule: empty injury feed leaves the model running blind on availability.
    if not injuries:
        reasons.append("injury feed unavailable — capped at LEAN")

    # Rule: adjustment stack should not exceed +/-12 pp; bigger means we are
    # piling small assumptions on top of a thin baseline.
    adjustment_total = sum(float(item.get("value") or 0.0) for item in adjustments or [])
    if abs(adjustment_total) > 0.12:
        reasons.append(f"adjustment stack {adjustment_total*100:+.1f}% > 12% — too many compounding assumptions")

    # Decision tree.
    decision = "PASS"
    if not reasons and edge >= BET_EDGE_THRESHOLD and pick_prob >= BET_PROB_FLOOR:
        if (
            spread_gap is None
            or abs(spread_gap) <= SPREAD_AGREEMENT_GAP_BET
        ):
            decision = "BET"
        else:
            decision = "LEAN"
            reasons.append("spread gap inside lean band but outside bet band")
    elif edge >= LEAN_EDGE_THRESHOLD and pick_prob >= LEAN_PROB_FLOOR:
        # Allow LEAN even when reasons exist, unless the spread is wildly off
        # or we have no win-prob conviction at all.
        if (
            spread_gap is None
            or abs(spread_gap) <= SPREAD_AGREEMENT_GAP_LEAN
        ) and pick_prob >= 0.50:
            decision = "LEAN"

    if decision == "BET":
        confidence = "High" if edge >= 0.06 and pick_prob >= 0.58 else "Medium"
    elif decision == "LEAN":
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "decision": decision,
        "confidence": confidence,
        "reasons": reasons,
        "spread_gap": None if spread_gap is None else round(spread_gap, 2),
        "market_pick_spread": market_pick_spread,
        "adjustment_total": round(adjustment_total, 4),
        "is_big_dog": is_big_dog,
    }


def run_playoff_game(
    game: dict[str, Any],
    all_team_stats: dict[str, dict[str, Any]],
    last20_context: dict[str, dict[str, float]],
    ranks: dict[str, int],
    injuries: dict[str, list[dict[str, Any]]],
    season: str,
) -> dict[str, Any] | None:
    away_name = game["away_team"]
    home_name = game["home_team"]
    matchup = f"{away_name} @ {home_name}"

    print("\n" + "=" * 80)
    print(f"GAME: {matchup} ({game['series_status']})")
    print("=" * 80)

    if game.get("has_started"):
        print(f"DECLINE: {matchup} has already started or is no longer in pre-game status ({game.get('game_status')}).")
        return None

    market = game.get("market") or {}
    home_ml = market.get("home_ml")
    away_ml = market.get("away_ml")
    if home_ml is None or away_ml is None:
        print(f"DECLINE: {matchup} has no verified two-sided moneyline in the ESPN odds payload.")
        return None

    if away_name not in all_team_stats or home_name not in all_team_stats:
        print(f"DECLINE: Missing team stats for {matchup}.")
        return None

    stats_sources = {
        str(all_team_stats[name].get("stats_source") or "NBA API")
        for name in (away_name, home_name)
        if isinstance(all_team_stats.get(name), dict)
    }
    stats_source = ", ".join(sorted(stats_sources)) or "NBA API"

    home_roster_ok, home_roster_note = _verify_roster(home_name, season)
    away_roster_ok, away_roster_note = _verify_roster(away_name, season)
    if not home_roster_ok or not away_roster_ok:
        print(f"DECLINE: Roster verification failed. {home_name}: {home_roster_note}; {away_name}: {away_roster_note}.")
        return None

    home_team = create_team(2, home_name, True, all_team_stats[home_name])
    away_team = create_team(1, away_name, False, all_team_stats[away_name])
    venue = game.get("arena") or f"{home_name} Arena"
    series_context = parse_series_context(game)
    ctx = GameContext(
        game.get("slate_date") or game.get("date", "")[:10],
        Venue(venue),
        home_team,
        away_team,
        0.50,
        game_id=game.get("game_id", ""),
    )

    tempo_context = calculate_playoff_tempo(
        away_team.team_stats,
        home_team.team_stats,
        series_context,
    )

    h2h = fetch_h2h_context(home_name, game.get("away_abbr", ""), season, ctx.date)
    series_history = fetch_series_history(
        home_name,
        game.get("away_abbr", ""),
        season,
        ctx.date,
        home_abbr=game.get("home_abbr"),
    )
    series_form = compute_series_form_signal(series_history, PLAYOFF_MARGIN_RMSE)
    base_rate, base_notes = calculate_base_rate(
        home_name,
        away_name,
        all_team_stats[home_name],
        last20_context,
        ranks,
        h2h,
        series_form,
    )
    injury_profile = _playoff_injury_profile(home_name, away_name, injuries)
    adjustments, home_inj_reason, away_inj_reason = _build_adjustments(
        game,
        home_team,
        away_team,
        injuries,
        tempo_context,
        series_context,
        h2h,
        injury_profile,
    )

    adjusted_base_prob = _clamp(base_rate + sum(float(item["value"]) for item in adjustments), 0.03, 0.97)
    predicted_spread, margin_components = predict_playoff_margin(
        home_team,
        away_team,
        tempo_context,
        series_context,
        h2h,
        injury_profile,
        series_form,
    )
    effective_margin_rmse = PLAYOFF_MARGIN_RMSE + float(series_form.get("rmse_inflation", 0.0) or 0.0)
    margin_prob = _clamp(_normal_cdf(predicted_spread / max(effective_margin_rmse, 1.0)), 0.03, 0.97)
    raw_prob = _clamp((adjusted_base_prob * 0.52) + (margin_prob * 0.48), 0.03, 0.97)
    final_home_prob = extremize_probability(raw_prob)
    home_market_prob, away_market_prob = remove_vig(home_ml, away_ml)
    predicted_total, total_components = predict_playoff_total(
        home_team,
        away_team,
        tempo_context,
        series_context,
        injury_profile,
    )

    if final_home_prob >= 0.50:
        pick_team = home_name
        pick_prob = final_home_prob
        market_prob = home_market_prob
        pick_odds = home_ml
    else:
        pick_team = away_name
        pick_prob = 1.0 - final_home_prob
        market_prob = away_market_prob
        pick_odds = away_ml

    edge = pick_prob - market_prob
    spread_team = home_name if predicted_spread >= 0 else away_name
    spread_disagrees = spread_team != pick_team and abs(predicted_spread) >= 0.5

    guardrail = evaluate_playoff_decision(
        pick_team=pick_team,
        pick_prob=pick_prob,
        pick_odds=pick_odds,
        edge=edge,
        predicted_spread=predicted_spread,
        market=market,
        home_name=home_name,
        away_name=away_name,
        injuries=injuries,
        series_context=series_context,
        adjustments=adjustments,
    )
    decision = guardrail["decision"]
    if spread_disagrees and decision == "BET":
        decision = "LEAN"
        guardrail["reasons"].append(
            f"moneyline points to {pick_team} but margin layer projects {spread_team} by {abs(predicted_spread):.2f}"
        )
        guardrail["decision"] = decision

    units = _quarter_kelly_units(edge, pick_odds) if decision == "BET" else 0.0
    confidence = guardrail["confidence"]

    print("**Game Context:**")
    print(f"- {game.get('away_display') or away_name} at {game.get('home_display') or home_name}")
    print(f"- {game.get('series_status')} | Game {series_context.get('game_number')} | Venue: {venue} | Scheduled: {game.get('game_status')}")
    print(f"- Source: ESPN NBA postseason scoreboard confirms season type 3 / post-season.")

    print("\n**Verification checks:**")
    print("- [x] Official playoff game verified through ESPN scoreboard")
    print("- [x] Game has not started")
    print(f"- [x] Current rosters checked: {home_name} ({home_roster_note}); {away_name} ({away_roster_note})")
    print(f"- [x] Team efficiency and recent-form stats fetched from {stats_source}")
    print(f"- [x] Market moneyline fetched from {market.get('provider') or 'ESPN odds'}")

    print("\n**Key Factors:**")
    print(f"- Net Rating: {away_name} {away_team.team_stats.net_rating:+.1f} vs {home_name} {home_team.team_stats.net_rating:+.1f} ({stats_source})")
    print(f"- Off/Def Rating: {away_name} {away_team.team_stats.off_rating_10:.1f}/{away_team.team_stats.def_rating_10:.1f} vs {home_name} {home_team.team_stats.off_rating_10:.1f}/{home_team.team_stats.def_rating_10:.1f}")
    print(f"- Pace: {away_name} {away_team.team_stats.pace:.1f} vs {home_name} {home_team.team_stats.pace:.1f}; playoff pace {tempo_context.get('playoff_pace', 0.0):.1f} after {tempo_context.get('pace_drag', 0.0):.1f}-possession playoff drag")
    print(f"- Playoff style: halfcourt weight {tempo_context.get('halfcourt_weight', 0.0):.2f}; dictating side {tempo_context.get('dictating_side')}")
    print(f"- H2H: {h2h.get('note')}")
    print(f"- Series form: {series_form.get('note')} (evidence weight {float(series_form.get('evidence_weight') or 0.0)*100:.0f}%, RMSE inflation +{float(series_form.get('rmse_inflation') or 0.0):.1f})")
    print(f"- Injuries: {home_name}: {home_inj_reason}; {away_name}: {away_inj_reason}")
    print(f"- Rest days: {away_name} {away_team.team_stats.rest_days:.0f} vs {home_name} {home_team.team_stats.rest_days:.0f}")

    print("\n**Our Probability:**")
    print(f"- Base rate ({home_name}): {base_rate*100:.1f}%")
    for note in base_notes:
        print(f"  - {note}")
    for item in adjustments:
        print(f"- {item['label']}: {float(item['value'])*100:+.1f}% because {item['reason']}")
    print(f"- Adjustment subtotal: {sum(float(item['value']) for item in adjustments)*100:+.1f}%")
    print(f"- Adjusted base probability ({home_name}): {adjusted_base_prob*100:.1f}%")
    print(f"- Margin-implied playoff probability ({home_name}): {margin_prob*100:.1f}% from {predicted_spread:+.2f} projected home margin")
    print(f"- Blended raw probability ({home_name}): {raw_prob*100:.1f}%")
    print(f"- Extremized final probability ({home_name}): {final_home_prob*100:.1f}%")

    print("\n**Model Predictions:**")
    print(f"- **Pick:** {pick_team}")
    print(f"- **Projected Margin:** {spread_team} by {abs(predicted_spread):.2f} points")
    print(f"- **Model Confidence:** {pick_prob*100:.1f}%")
    print(f"- **Total:** {predicted_total:.1f} O/U")
    print(
        "- **Playoff Margin Components:** "
        f"scoring {margin_components['scoring_margin']:+.2f}, "
        f"home court {margin_components['home_court']:+.2f}, "
        f"star minutes {margin_components['star_minutes']:+.2f}, "
        f"series {margin_components['series_state']:+.2f}, "
        f"coaching {margin_components['coaching_adjustment']:+.2f}, "
        f"injury {margin_components['injury']:+.2f}"
    )
    print(
        "- **Playoff Total Components:** "
        f"base {total_components['base_total']:.1f}, "
        f"physicality drag -{total_components['physicality_drag']:.1f}, "
        f"availability drag -{total_components['availability_drag']:.1f}"
    )

    print("\n**Market Odds:**")
    print(f"- {home_name} {_format_odds(home_ml)} | {away_name} {_format_odds(away_ml)} ({market.get('provider') or 'ESPN odds'})")
    print(f"- Market implied probability (vig-removed): {home_name} {home_market_prob*100:.1f}% | {away_name} {away_market_prob*100:.1f}%")
    if market.get("home_spread") is not None or market.get("away_spread") is not None:
        print(f"- Spread: {home_name} {market.get('home_spread')} | {away_name} {market.get('away_spread')}")

    print("\n**Edge And Decision:**")
    print(f"**Edge:** {pick_team} {edge*100:+.1f}%")
    print(f"**BET threshold:** {BET_EDGE_THRESHOLD*100:.1f}% edge AND {BET_PROB_FLOOR*100:.0f}% pick prob (dogs need {DOG_BET_PROB_FLOOR*100:.0f}%); LEAN at {LEAN_EDGE_THRESHOLD*100:.1f}% edge.")
    if guardrail.get("market_pick_spread") is not None:
        print(
            f"**Spread cross-check:** model {('home' if pick_team == home_name else 'away')}-margin "
            f"vs market {pick_team} {guardrail['market_pick_spread']:+.1f}; gap {guardrail.get('spread_gap'):+.2f} pts."
        )
    if decision == "BET":
        print(f"**Decision: BET on {pick_team}**")
        print(f"**Stake:** {units:.2f} units (quarter Kelly, 2u cap)")
    elif decision == "LEAN":
        print(f"**Decision: LEAN {pick_team}** (no stake — context favors but guardrails block a full BET)")
    else:
        print("**Decision: PASS**")
    for reason in guardrail.get("reasons") or []:
        print(f"- Guardrail: {reason}")

    print("\n**Confidence And Honesty:**")
    print(f"- Confidence: {confidence}")
    print("- Limitations: Expected starters and final playoff rotations can change near tip; re-check official injury and lineup reports before betting.")

    pick = {
        "source": MODEL_SOURCE,
        "pick": f"{pick_team} ML ({away_name} @ {home_name})",
        "sport": "NBA",
        "league": "NBA",
        "market_type": "moneyline",
        "selection": pick_team,
        "team": pick_team,
        "away_team": away_name,
        "home_team": home_name,
        "odds": pick_odds,
        "units": round(units, 2),
        "probability": round(pick_prob, 4),
        "prob": round(pick_prob, 4),
        "edge": round(edge * 100.0, 2),
        "decision": decision,
        "market_probability": round(market_prob, 4),
        "model_prediction": round(predicted_spread, 2),
        "predicted_spread": round(predicted_spread, 2),
        "vegas": market.get("home_spread") if pick_team == home_name else market.get("away_spread"),
        "market_line": market.get("home_spread") if pick_team == home_name else market.get("away_spread"),
        "total_projection": round(predicted_total, 1),
        "base_probability": round(base_rate, 4),
        "adjusted_base_probability": round(adjusted_base_prob, 4),
        "margin_probability": round(margin_prob, 4),
        "playoff_pace": round(float(tempo_context.get("playoff_pace", 0.0) or 0.0), 2),
        "halfcourt_weight": round(float(tempo_context.get("halfcourt_weight", 0.0) or 0.0), 3),
        "series_game_number": series_context.get("game_number"),
        "margin_components": {key: round(float(value), 3) for key, value in margin_components.items()},
        "total_components": {key: round(float(value), 3) for key, value in total_components.items()},
        "series_status": game.get("series_status"),
        "game_id": game.get("game_id"),
        "guardrail_reasons": list(guardrail.get("reasons") or []),
        "guardrail_spread_gap": guardrail.get("spread_gap"),
        "guardrail_market_pick_spread": guardrail.get("market_pick_spread"),
        "guardrail_adjustment_total_pp": round(float(guardrail.get("adjustment_total") or 0.0) * 100.0, 2),
        "is_big_dog": guardrail.get("is_big_dog", False),
        "confidence": confidence,
        "stats_source": stats_source,
        "series_form": {
            "games": int(series_form.get("games", 0) or 0),
            "avg_margin": round(float(series_form.get("avg_margin", 0.0) or 0.0), 2),
            "max_abs_margin": round(float(series_form.get("max_abs_margin", 0.0) or 0.0), 2),
            "implied_prob_for_home": (
                round(float(series_form.get("implied_prob_for_home")), 4)
                if series_form.get("implied_prob_for_home") is not None
                else None
            ),
            "evidence_weight": round(float(series_form.get("evidence_weight", 0.0) or 0.0), 4),
            "margin_shift": round(float(series_form.get("margin_shift", 0.0) or 0.0), 2),
            "rmse_inflation": round(float(series_form.get("rmse_inflation", 0.0) or 0.0), 2),
        },
        "effective_margin_rmse": round(effective_margin_rmse, 2),
    }
    print(f"PICK_JSON: {json.dumps(pick, sort_keys=True)}")
    return pick


def main() -> None:
    args = _parse_args()
    target_date = _normalize_target_date(args.date or args.legacy_date)
    season = str(args.season or DEFAULT_SEASON).strip() or DEFAULT_SEASON

    print("=" * 80)
    print("NBA PLAYOFFS PREDICTION MODEL")
    print(f"Requested Date: {target_date}")
    print(f"Run Timestamp: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("Data: ESPN postseason scoreboard + NBA API stats/rosters + injury feed")
    print("=" * 80)

    try:
        playoff_games = fetch_espn_playoff_games(target_date)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"No eligible NBA playoff games: ESPN postseason scoreboard fetch failed ({exc}).")
        return

    if not playoff_games:
        print(f"No official NBA playoff games found on ESPN for {target_date}.")
        return

    eligible_games = [game for game in playoff_games if not game.get("has_started")]
    if not eligible_games:
        print("No eligible NBA playoff games: every listed playoff game has started or finished.")
        return

    print(f"Found {len(playoff_games)} playoff game(s); {len(eligible_games)} still pre-game.")

    print("\nFetching NBA API team stats and rest context...")
    all_team_stats = fetch_all_team_stats(
        season=season,
        as_of_date=target_date,
        upcoming_games=eligible_games,
    )

    print("\nFetching last-20 context from NBA API game logs...")
    try:
        last20_context = fetch_last20_context(season, target_date)
    except Exception as exc:
        print(f"WARNING: Last-20 lookup failed ({exc}); falling back to model recent-form fields.")
        last20_context = {}

    ranks = _rank_lookup(all_team_stats)

    print("\nFetching current injury report...")
    injuries = fetch_injuries()
    if not injuries:
        print("WARNING: Injury feed returned no entries; picks will be lower-confidence.")

    picks: list[dict[str, Any]] = []
    for game in eligible_games:
        try:
            pick = run_playoff_game(game, all_team_stats, last20_context, ranks, injuries, season)
            if pick:
                picks.append(pick)
        except Exception as exc:
            matchup = f"{game.get('away_team', '')} @ {game.get('home_team', '')}".strip()
            print(f"DECLINE: {matchup or 'game'} failed playoff model verification/calculation ({exc}).")

    if not picks:
        print("No eligible NBA playoff picks generated after verification gates.")


if __name__ == "__main__":
    main()
