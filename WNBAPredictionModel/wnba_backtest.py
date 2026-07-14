"""
wnba_backtest.py — Historical validation engine for the WNBA pick model.

This file replays completed WNBA seasons through the same probability stack
used for live picks, and reports model accuracy, calibration, and spread
error. No live-pick code path should be trusted until the numbers here
clear the target benchmarks at the bottom of the report.

CRITICAL RULE — NO LOOKAHEAD:
    When backtesting game N, we only use stats built from games 1..N-1.
    Season-level averages that include game N are strictly forbidden —
    every rolling stat is rebuilt "as of" the date of the game being
    simulated, so the model sees exactly what it would have seen on the
    morning of that game.
"""

from __future__ import annotations

import csv
import datetime
import json
import os
import sys

import requests

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from .wnba_probability_layers import calculate_wnba_matchup
    from .wnba_teams import WNBA_TEAM_MAP  # noqa: F401 — imported for parity with rest of module
except ImportError:
    from wnba_probability_layers import calculate_wnba_matchup
    from wnba_teams import WNBA_TEAM_MAP  # noqa: F401 — imported for parity with rest of module

try:
    from config import BDL_API_KEY
except ImportError:
    BDL_API_KEY = None

if not BDL_API_KEY:
    BDL_API_KEY = os.getenv("BDL_API_KEY", "")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BDL_GAMES_URL = "https://api.balldontlie.io/wnba/v1/games"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"

# Regular-season date windows per season for the ESPN fetcher. Generous on
# both ends — non-regular-season events are filtered by season type.
ESPN_SEASON_WINDOWS = {
    2024: ("20240501", "20241001"),
    2025: ("20250501", "20250930"),
    2026: ("20260501", "20261031"),
}

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "wnba")
)
os.makedirs(DATA_DIR, exist_ok=True)
HISTORICAL_CACHE_PATH = os.path.join(DATA_DIR, "wnba_historical_games.json")
BACKTEST_RESULTS_PATH = os.path.join(DATA_DIR, "wnba_backtest_results.csv")

HISTORICAL_CACHE_TTL_SECONDS = 24 * 60 * 60

# Mirrors the alias table used elsewhere in the codebase so every team
# collapses to a single canonical abbreviation.
_ABBR_ALIASES = {
    "LVA":  "LV",
    "CONN": "CON",
    "NYL":  "NY",
    "WSH":  "WAS",
    "PHO":  "PHX",
    "GS":   "GSV",
    "GSW":  "GSV",
}


def _normalize_abbr(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip().upper()
    return _ABBR_ALIASES.get(raw, raw)


# ---------------------------------------------------------------------------
# Section 1 — Historical Game Fetcher
# ---------------------------------------------------------------------------

def _cache_is_fresh(path: str, ttl_seconds: int) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return False
    last_updated = payload.get("last_updated")
    if not last_updated:
        return False
    try:
        updated_at = datetime.datetime.fromisoformat(last_updated)
    except ValueError:
        return False
    now = (
        datetime.datetime.now(updated_at.tzinfo)
        if updated_at.tzinfo
        else datetime.datetime.now()
    )
    age = (now - updated_at).total_seconds()
    return 0 <= age <= ttl_seconds


def _load_historical_cache() -> list[dict] | None:
    if not _cache_is_fresh(HISTORICAL_CACHE_PATH, HISTORICAL_CACHE_TTL_SECONDS):
        return None
    try:
        with open(HISTORICAL_CACHE_PATH, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        return None
    games = payload.get("games")
    if isinstance(games, list):
        return games
    return None


def _write_historical_cache(games: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    payload = {
        "last_updated": datetime.datetime.now().isoformat(),
        "games": games,
    }
    try:
        with open(HISTORICAL_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as exc:
        print(f"[WNBA Backtest] Could not write historical cache: {exc}")


def _extract_date_str(raw: str) -> str:
    """Return just the YYYY-MM-DD portion of a BDL date string."""
    if not raw:
        return ""
    return str(raw).split("T", 1)[0].strip()


def _date_chunks(start: str, end: str, days: int = 14) -> list[tuple[str, str]]:
    """Split a YYYYMMDD range into inclusive chunks of at most *days* days."""
    chunks: list[tuple[str, str]] = []
    cursor = datetime.datetime.strptime(start, "%Y%m%d").date()
    stop = datetime.datetime.strptime(end, "%Y%m%d").date()
    while cursor <= stop:
        chunk_end = min(cursor + datetime.timedelta(days=days - 1), stop)
        chunks.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cursor = chunk_end + datetime.timedelta(days=1)
    return chunks


def fetch_espn_historical_games(seasons: list[int]) -> list[dict]:
    """Fetch every Final regular-season WNBA game from the free ESPN
    scoreboard API. No API key required — this is the default historical
    source when no BallDontLie key is configured.
    """
    all_games: list[dict] = []
    seen_ids: set[str] = set()

    for season in seasons:
        window = ESPN_SEASON_WINDOWS.get(season)
        if window is None:
            window = (f"{season}0501", f"{season}1031")

        for chunk_start, chunk_end in _date_chunks(*window):
            try:
                resp = requests.get(
                    ESPN_SCOREBOARD_URL,
                    params={"dates": f"{chunk_start}-{chunk_end}", "limit": 300},
                    timeout=20,
                )
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError) as exc:
                print(f"[WNBA Backtest] ESPN fetch failed for {chunk_start}-{chunk_end}: {exc}")
                continue

            for event in payload.get("events", []) or []:
                event_id = str(event.get("id") or "")
                if not event_id or event_id in seen_ids:
                    continue
                season_type = ((event.get("season") or {}).get("type"))
                if season_type is not None and int(season_type) != 2:
                    continue

                competitions = event.get("competitions") or []
                competition = competitions[0] if competitions else {}
                status_name = (
                    ((competition.get("status") or {}).get("type") or {}).get("name") or ""
                )
                if status_name != "STATUS_FINAL":
                    continue

                home_abbr = away_abbr = ""
                home_score = away_score = None
                for comp in competition.get("competitors", []) or []:
                    team = comp.get("team") or {}
                    abbr = _normalize_abbr(team.get("abbreviation", ""))
                    try:
                        score = float(comp.get("score"))
                    except (TypeError, ValueError):
                        score = None
                    if comp.get("homeAway") == "home":
                        home_abbr, home_score = abbr, score
                    elif comp.get("homeAway") == "away":
                        away_abbr, away_score = abbr, score

                if not home_abbr or not away_abbr or home_score is None or away_score is None:
                    continue

                seen_ids.add(event_id)
                all_games.append({
                    "id": event_id,
                    "date": _extract_date_str(event.get("date", "")),
                    "home_team": home_abbr,
                    "visitor_team": away_abbr,
                    "home_team_score": home_score,
                    "visitor_team_score": away_score,
                    "status": "Final",
                })

    all_games.sort(key=lambda g: g.get("date") or "")
    return all_games


def fetch_historical_games(seasons: list[int] = [2024, 2025]) -> list[dict]:
    """Fetch every Final WNBA game for the given seasons, sorted by date ASC.

    Uses data/wnba_historical_games.json as a 24-hour cache to avoid
    hammering the sources on repeated backtest runs. Prefers BallDontLie
    when a key is configured, otherwise falls back to the free ESPN
    scoreboard API.
    """
    cached = _load_historical_cache()
    if cached is not None:
        return cached

    if not BDL_API_KEY:
        games = fetch_espn_historical_games(seasons)
        if games:
            _write_historical_cache(games)
        return games

    all_games: list[dict] = []

    for season in seasons:
        cursor: str | None = None
        while True:
            params: dict[str, str | int] = {
                "seasons[]": season,
                "season_type": 2,
                "per_page": 100,
            }
            if cursor is not None:
                params["cursor"] = cursor

            try:
                resp = requests.get(
                    BDL_GAMES_URL,
                    headers={"Authorization": BDL_API_KEY},
                    params=params,
                    timeout=20,
                )
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError) as exc:
                print(f"[WNBA Backtest] Games fetch failed for {season}: {exc}")
                break

            for row in payload.get("data", []) or []:
                status = (row.get("status") or "").strip()
                if status != "Final":
                    continue

                home_team_obj = row.get("home_team") or {}
                away_team_obj = row.get("visitor_team") or {}
                home_abbr = _normalize_abbr(home_team_obj.get("abbreviation", ""))
                away_abbr = _normalize_abbr(away_team_obj.get("abbreviation", ""))
                if not home_abbr or not away_abbr:
                    continue

                all_games.append({
                    "id": row.get("id"),
                    "date": _extract_date_str(row.get("date", "")),
                    "home_team": home_abbr,
                    "visitor_team": away_abbr,
                    "home_team_score": row.get("home_team_score"),
                    "visitor_team_score": row.get("visitor_team_score"),
                    "status": status,
                })

            cursor = (payload.get("meta") or {}).get("next_cursor")
            if not cursor:
                break

    all_games.sort(key=lambda g: g.get("date") or "")
    _write_historical_cache(all_games)
    return all_games


# ---------------------------------------------------------------------------
# Section 2 — Rolling Stats Builder (no-lookahead)
# ---------------------------------------------------------------------------

def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_rolling_stats_as_of(
    all_games: list[dict],
    team_abbr: str,
    as_of_date: str,
    n: int = 99,
) -> dict:
    """Return a rolling-stats profile built **only** from games strictly
    before *as_of_date*. The model must never peek at the game it is
    predicting, so the `<` comparison here is the single point that
    enforces the no-lookahead rule.
    """
    if not team_abbr or not as_of_date:
        return {}

    abbr = _normalize_abbr(team_abbr)
    if not abbr:
        return {}

    qualifying: list[tuple[str, dict]] = []
    for game in all_games:
        game_date = game.get("date") or ""
        if not game_date or game_date >= as_of_date:
            continue
        home = _normalize_abbr(game.get("home_team", ""))
        away = _normalize_abbr(game.get("visitor_team", ""))
        if abbr not in (home, away):
            continue
        qualifying.append((game_date, game))

    qualifying.sort(key=lambda item: item[0])
    window = qualifying[-n:]

    if len(window) < 3:
        return {}

    pts_vals: list[float] = []
    opp_pts_vals: list[float] = []
    fga_vals: list[float] = []
    fta_vals: list[float] = []
    orb_vals: list[float] = []
    tov_vals: list[float] = []

    for _date, game in window:
        home = _normalize_abbr(game.get("home_team", ""))
        home_score = _safe_float(game.get("home_team_score"))
        away_score = _safe_float(game.get("visitor_team_score"))
        if home_score is None or away_score is None:
            continue

        if abbr == home:
            own_pts = home_score
            opp_pts = away_score
            own_fga = _safe_float(game.get("home_fga"))
            own_fta = _safe_float(game.get("home_fta"))
            own_orb = _safe_float(game.get("home_orb"))
            own_tov = _safe_float(game.get("home_tov"))
        else:
            own_pts = away_score
            opp_pts = home_score
            own_fga = _safe_float(game.get("visitor_fga"))
            own_fta = _safe_float(game.get("visitor_fta"))
            own_orb = _safe_float(game.get("visitor_orb"))
            own_tov = _safe_float(game.get("visitor_tov"))

        pts_vals.append(own_pts)
        opp_pts_vals.append(opp_pts)
        if own_fga is not None:
            fga_vals.append(own_fga)
        if own_fta is not None:
            fta_vals.append(own_fta)
        if own_orb is not None:
            orb_vals.append(own_orb)
        if own_tov is not None:
            tov_vals.append(own_tov)

    if not pts_vals or not opp_pts_vals:
        return {}

    pts_pg = sum(pts_vals) / len(pts_vals)
    opp_pts_pg = sum(opp_pts_vals) / len(opp_pts_vals)

    have_box = all(vals for vals in (fga_vals, fta_vals, orb_vals, tov_vals))
    if have_box:
        fga_pg = sum(fga_vals) / len(fga_vals)
        fta_pg = sum(fta_vals) / len(fta_vals)
        orb_pg = sum(orb_vals) / len(orb_vals)
        tov_pg = sum(tov_vals) / len(tov_vals)
        possessions = 0.96 * (fga_pg - orb_pg + tov_pg + 0.44 * fta_pg)
    else:
        # Fallback: approximate possessions from points when no box-score
        # fields are available. BDL's /games feed is score-only, so in
        # practice this is the branch we hit for almost every game.
        possessions = pts_pg / 1.05

    if possessions is None or possessions <= 0:
        return {}

    nrtg = 100.0 * (pts_pg - opp_pts_pg) / possessions
    ortg = 100.0 * pts_pg / possessions
    drtg = 100.0 * opp_pts_pg / possessions

    return {
        "NRtg": nrtg,
        "ORtg": ortg,
        "DRtg": drtg,
        "Pace": possessions,
        "pts_pg": pts_pg,
        "opp_pts_pg": opp_pts_pg,
        "games_used": len(window),
        "low_sample": len(window) < n,
    }


# ---------------------------------------------------------------------------
# Section 3 — Single Game Backtest
# ---------------------------------------------------------------------------

def get_confidence_label_from_prob(win_prob: float) -> str:
    """Bucket a win probability into Low / Medium / High.

    Mirrors the thresholds in wnba_picks.get_confidence_label but is
    direction-agnostic: we fold both sides of 0.5 into a single
    "confidence in the model's pick" value so a 0.75 away-side call
    isn't mislabelled as Low. Kept local so this module can run
    standalone without importing wnba_picks.
    """
    try:
        p = float(win_prob)
    except (TypeError, ValueError):
        return "Low"
    confidence = max(p, 1.0 - p)
    if confidence >= 0.70:
        return "High"
    if confidence >= 0.60:
        return "Medium"
    return "Low"


def _rest_days_as_of(all_games: list[dict], team_abbr: str, as_of_date: str) -> int | None:
    """Days since the team's previous game strictly before *as_of_date*."""
    abbr = _normalize_abbr(team_abbr)
    last_date: str | None = None
    for game in all_games:
        game_date = game.get("date") or ""
        if not game_date or game_date >= as_of_date:
            continue
        home = _normalize_abbr(game.get("home_team", ""))
        away = _normalize_abbr(game.get("visitor_team", ""))
        if abbr not in (home, away):
            continue
        if last_date is None or game_date > last_date:
            last_date = game_date
    if last_date is None:
        return None
    try:
        delta = (
            datetime.date.fromisoformat(as_of_date)
            - datetime.date.fromisoformat(last_date)
        ).days
    except ValueError:
        return None
    return delta if delta > 0 else None


def backtest_single_game(game: dict, all_games: list[dict]) -> dict | None:
    """Run the probability stack against one historical game using only
    pre-game data. Returns None when either team lacks enough prior
    games to build a rolling profile (≥3 games required).
    """
    home_abbr = _normalize_abbr(game.get("home_team", ""))
    away_abbr = _normalize_abbr(game.get("visitor_team", ""))
    date = game.get("date") or ""
    home_score = _safe_float(game.get("home_team_score"))
    away_score = _safe_float(game.get("visitor_team_score"))

    if not home_abbr or not away_abbr or not date:
        return None
    if home_score is None or away_score is None:
        return None

    home_stats = build_rolling_stats_as_of(all_games, home_abbr, date)
    away_stats = build_rolling_stats_as_of(all_games, away_abbr, date)

    if not home_stats or not away_stats:
        return None

    # Historical injury data is not reconstructable from the games feed, so
    # injuries stay neutral — but rest days and road back-to-backs ARE
    # reconstructable from the schedule itself, so feed them in to mirror
    # what the live pipeline sees.
    home_rest = _rest_days_as_of(all_games, home_abbr, date)
    away_rest = _rest_days_as_of(all_games, away_abbr, date)
    context = {
        "home_injury_penalty": 0.0,
        "away_injury_penalty": 0.0,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "away_is_b2b": away_rest == 1,
    }

    result = calculate_wnba_matchup(
        home_abbr=home_abbr,
        away_abbr=away_abbr,
        home_stats=home_stats,
        away_stats=away_stats,
        context=context,
    )

    home_won = home_score > away_score
    model_picked_home = result["win_prob"] > 0.5
    actual_margin = home_score - away_score
    actual_total = home_score + away_score
    projected_total = result.get("projected_total")

    return {
        "date": date,
        "home": home_abbr,
        "away": away_abbr,
        "predicted_margin": result["adjusted_margin"],
        "actual_margin": actual_margin,
        "predicted_win_prob": result["win_prob"],
        "model_correct": home_won == model_picked_home,
        "margin_error": abs(result["adjusted_margin"] - actual_margin),
        "predicted_total": projected_total,
        "actual_total": actual_total,
        "total_error": (
            abs(float(projected_total) - actual_total)
            if projected_total is not None
            else None
        ),
        "confidence": get_confidence_label_from_prob(result["win_prob"]),
        "games_used_home": home_stats["games_used"],
        "games_used_away": away_stats["games_used"],
    }


# ---------------------------------------------------------------------------
# Section 4 — Full Backtest Runner
# ---------------------------------------------------------------------------

_CSV_HEADERS = [
    "date",
    "home",
    "away",
    "predicted_margin",
    "actual_margin",
    "predicted_win_prob",
    "model_correct",
    "margin_error",
    "predicted_total",
    "actual_total",
    "total_error",
    "confidence",
    "games_used_home",
    "games_used_away",
]


def _zero_metrics() -> dict:
    return {
        "overall_accuracy": 0.0,
        "low_conf_accuracy": 0.0,
        "med_conf_accuracy": 0.0,
        "high_conf_accuracy": 0.0,
        "spread_mae": 0.0,
        "spread_rmse": 0.0,
        "margin_bias": 0.0,
        "brier_score": 0.0,
        "log_loss": 0.0,
        "calibration_bins": [],
        "total_mae": 0.0,
        "total_bias": 0.0,
        "early_accuracy": 0.0,
        "mid_accuracy": 0.0,
        "late_accuracy": 0.0,
        "total_games_tested": 0,
    }


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def _write_results_csv(rows: list[dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(BACKTEST_RESULTS_PATH, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_CSV_HEADERS)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in _CSV_HEADERS})
    except OSError as exc:
        print(f"[WNBA Backtest] Could not write results CSV: {exc}")


def run_full_backtest(
    seasons: list[int] = [2024, 2025],
    games: list[dict] | None = None,
    write_csv: bool = True,
) -> dict:
    """Replay every completed game in *seasons* through the model and
    return the aggregate accuracy / calibration metrics dict.

    Pass *games* to reuse an already-fetched historical list (the
    calibration grid search does this to avoid refetching per configuration).
    """
    if games is None:
        games = fetch_historical_games(seasons)
    if not games:
        return _zero_metrics()

    results: list[dict] = []

    for game in games:
        outcome = backtest_single_game(game, games)
        if outcome is None:
            continue

        # Skip games where both teams have too little history — the model
        # can't meaningfully operate in the first few days of a season.
        if outcome["games_used_home"] < 5 and outcome["games_used_away"] < 5:
            continue

        results.append(outcome)

    total = len(results)
    if total == 0:
        metrics = _zero_metrics()
        _write_results_csv(results)
        return metrics

    correct = sum(1 for r in results if r["model_correct"])

    low = [r for r in results if r["confidence"] == "Low"]
    med = [r for r in results if r["confidence"] == "Medium"]
    high = [r for r in results if r["confidence"] == "High"]

    low_correct = sum(1 for r in low if r["model_correct"])
    med_correct = sum(1 for r in med if r["model_correct"])
    high_correct = sum(1 for r in high if r["model_correct"])

    spread_mae = sum(r["margin_error"] for r in results) / total
    spread_rmse = (sum(r["margin_error"] ** 2 for r in results) / total) ** 0.5
    # Positive bias = model overrates the home side.
    margin_bias = sum(r["predicted_margin"] - r["actual_margin"] for r in results) / total

    # Probability quality: Brier score and log loss on the home-win
    # probability (lower is better; 0.25 / 0.693 are the coin-flip marks).
    import math as _math

    brier = 0.0
    log_loss = 0.0
    for r in results:
        outcome = 1.0 if r["actual_margin"] > 0 else 0.0
        p = min(max(float(r["predicted_win_prob"]), 1e-6), 1.0 - 1e-6)
        brier += (p - outcome) ** 2
        log_loss += -(outcome * _math.log(p) + (1.0 - outcome) * _math.log(1.0 - p))
    brier /= total
    log_loss /= total

    # Calibration bins on the favorite side: predicted favorite prob vs
    # how often the favorite actually won.
    bin_edges = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)]
    calibration_bins = []
    for low_edge, high_edge in bin_edges:
        bucket = []
        for r in results:
            p = float(r["predicted_win_prob"])
            fav_prob = max(p, 1.0 - p)
            if not (low_edge <= fav_prob < high_edge):
                continue
            fav_is_home = p >= 0.5
            fav_won = (r["actual_margin"] > 0) == fav_is_home
            bucket.append((fav_prob, fav_won))
        if bucket:
            calibration_bins.append({
                "range": f"{low_edge:.2f}-{min(high_edge, 1.0):.2f}",
                "games": len(bucket),
                "predicted": sum(b[0] for b in bucket) / len(bucket),
                "actual": sum(1 for b in bucket if b[1]) / len(bucket),
            })

    totals_rows = [r for r in results if r.get("total_error") is not None]
    total_mae = (
        sum(r["total_error"] for r in totals_rows) / len(totals_rows)
        if totals_rows else 0.0
    )
    total_bias = (
        sum(float(r["predicted_total"]) - float(r["actual_total"]) for r in totals_rows)
        / len(totals_rows)
        if totals_rows else 0.0
    )

    # Season-phase buckets use the *smaller* of the two teams' games_used
    # values — the model is only as stable as its less-established team.
    early: list[dict] = []
    mid: list[dict] = []
    late: list[dict] = []
    for r in results:
        min_used = min(r["games_used_home"], r["games_used_away"])
        if min_used < 10:
            early.append(r)
        elif min_used <= 25:
            mid.append(r)
        else:
            late.append(r)

    early_correct = sum(1 for r in early if r["model_correct"])
    mid_correct = sum(1 for r in mid if r["model_correct"])
    late_correct = sum(1 for r in late if r["model_correct"])

    metrics = {
        "overall_accuracy": _pct(correct, total),
        "low_conf_accuracy": _pct(low_correct, len(low)),
        "med_conf_accuracy": _pct(med_correct, len(med)),
        "high_conf_accuracy": _pct(high_correct, len(high)),
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "margin_bias": margin_bias,
        "brier_score": brier,
        "log_loss": log_loss,
        "calibration_bins": calibration_bins,
        "total_mae": total_mae,
        "total_bias": total_bias,
        "early_accuracy": _pct(early_correct, len(early)),
        "mid_accuracy": _pct(mid_correct, len(mid)),
        "late_accuracy": _pct(late_correct, len(late)),
        "total_games_tested": total,
    }

    if write_csv:
        _write_results_csv(results)
    return metrics


# ---------------------------------------------------------------------------
# Section 5 — Calibration Report
# ---------------------------------------------------------------------------

def _badge(passed: bool) -> str:
    return "✅ PASS" if passed else "❌ FAIL"


def print_calibration_report(metrics: dict) -> None:
    """Print a boxed calibration summary plus PASS/FAIL marks for the
    three target benchmarks. Always echoes the CSV path so the caller
    can inspect per-game detail.
    """
    m = metrics or _zero_metrics()

    total_games = int(m.get("total_games_tested", 0) or 0)
    overall = float(m.get("overall_accuracy", 0.0) or 0.0)
    high = float(m.get("high_conf_accuracy", 0.0) or 0.0)
    med = float(m.get("med_conf_accuracy", 0.0) or 0.0)
    low = float(m.get("low_conf_accuracy", 0.0) or 0.0)
    mae = float(m.get("spread_mae", 0.0) or 0.0)
    rmse = float(m.get("spread_rmse", 0.0) or 0.0)
    bias = float(m.get("margin_bias", 0.0) or 0.0)
    brier = float(m.get("brier_score", 0.0) or 0.0)
    log_loss = float(m.get("log_loss", 0.0) or 0.0)
    total_mae = float(m.get("total_mae", 0.0) or 0.0)
    total_bias = float(m.get("total_bias", 0.0) or 0.0)
    early = float(m.get("early_accuracy", 0.0) or 0.0)
    mid = float(m.get("mid_accuracy", 0.0) or 0.0)
    late = float(m.get("late_accuracy", 0.0) or 0.0)

    overall_pass = overall >= 58.0
    high_pass = high >= 65.0
    # WNBA margins have a ~13.3-pt sigma, so a perfect predictor of the
    # true expected margin still shows MAE ≈ sigma·√(2/π) ≈ 10.6. The old
    # ≤9 target was mathematically unreachable.
    mae_pass = mae <= 11.0
    brier_pass = 0.0 < brier <= 0.24  # coin flip = 0.25; must beat it

    print("╔══════════════════════════════════════════╗")
    print("║ WNBA MODEL BACKTEST REPORT               ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║ Games Tested:        {total_games}")
    print(f"║ Overall Accuracy:    {overall:.1f}%")
    print(f"║ High Conf Accuracy:  {high:.1f}%")
    print(f"║ Med Conf Accuracy:   {med:.1f}%")
    print(f"║ Low Conf Accuracy:   {low:.1f}%")
    print(f"║ Spread MAE / RMSE:   {mae:.2f} / {rmse:.2f} pts")
    print(f"║ Margin Bias (home):  {bias:+.2f} pts")
    print(f"║ Brier / Log Loss:    {brier:.4f} / {log_loss:.4f}")
    print(f"║ Totals MAE / Bias:   {total_mae:.2f} / {total_bias:+.2f} pts")
    print("╠══════════════════════════════════════════╣")
    for bucket in m.get("calibration_bins", []) or []:
        print(
            f"║ Fav {bucket['range']}: predicted {bucket['predicted']:.3f} "
            f"vs actual {bucket['actual']:.3f} (n={bucket['games']})"
        )
    print("╠══════════════════════════════════════════╣")
    print(f"║ Early Season (<10g): {early:.1f}%")
    print(f"║ Mid Season (11-25g): {mid:.1f}%")
    print(f"║ Late Season (26g+):  {late:.1f}%")
    print("╠══════════════════════════════════════════╣")
    print("║ TARGET BENCHMARKS")
    print(f"║ Overall ≥ 58%:       {_badge(overall_pass)}")
    print(f"║ High Conf ≥ 65%:     {_badge(high_pass)}")
    print(f"║ Spread MAE ≤ 11 pts: {_badge(mae_pass)}")
    print(f"║ Brier ≤ 0.24:        {_badge(brier_pass)}")
    print("╚══════════════════════════════════════════╝")
    print(f"Results saved to {BACKTEST_RESULTS_PATH}")


# ---------------------------------------------------------------------------
# Section 6 — CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seasons = [2024, 2025, 2026]
    if len(sys.argv) > 1:
        seasons = [int(arg) for arg in sys.argv[1:]]
    metrics = run_full_backtest(seasons)
    print_calibration_report(metrics)

    print("PASS: Backtest engine ran without errors.")
