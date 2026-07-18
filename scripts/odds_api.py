#!/usr/bin/env python3
"""Optional sharp-book price capture via The Odds API.

Everything here is enrichment: with no ``ODDS_API_KEY`` in the environment
the public helpers return empty results and the pipeline behaves exactly as
before. When a key is present, near-close capture runs journal Pinnacle-style
sharp prices next to the DraftKings anchor rows so the ledger carries an
alternative fair-value baseline. Nothing gates publication on these rows.

Budgeted for the free tier (~500 credits/month): sharp fetches only happen
for sports with a pick starting inside ``ODDS_API_WINDOW_MINUTES`` (default
45), one request per sport per run, h2h+totals only.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

import requests

from scripts.build_profit_desk import _parse_timestamp, canonical_market_identity
from scripts.devig import american_implied_probability

API_BASE = "https://api.the-odds-api.com/v4"
REQUEST_TIMEOUT = 20

SPORT_KEYS = {
    "MLB": "baseball_mlb",
    "NBA": "basketball_nba",
    "WNBA": "basketball_wnba",
}


def _api_key() -> str:
    return os.environ.get("ODDS_API_KEY", "").strip()


def _bookmakers() -> str:
    return os.environ.get("ODDS_API_BOOKMAKERS", "pinnacle").strip() or "pinnacle"


def _sharp_window() -> timedelta:
    try:
        minutes = int(os.environ.get("ODDS_API_WINDOW_MINUTES", "45"))
    except ValueError:
        minutes = 45
    return timedelta(minutes=max(5, minutes))


def _norm(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _nickname(team_name: str) -> str:
    parts = [part for part in str(team_name or "").split() if part.strip()]
    return _norm(parts[-1]) if parts else ""


def _american_from_decimal(price: Any) -> int | None:
    try:
        decimal = float(price)
    except (TypeError, ValueError):
        return None
    if decimal <= 1.0:
        return None
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def _outcome_american(outcome: dict[str, Any]) -> int | None:
    price = outcome.get("price")
    try:
        number = float(price)
    except (TypeError, ValueError):
        return None
    # oddsFormat=american returns integers like -145; a value in (-100, 100)
    # can only be a decimal price, so convert defensively.
    if -100.0 < number < 100.0:
        return _american_from_decimal(number)
    return int(round(number))


def fetch_sharp_events(
    sport: str, *, fetch_json: Callable[[str, dict[str, Any]], Any] | None = None
) -> list[dict[str, Any]]:
    key = _api_key()
    sport_key = SPORT_KEYS.get(str(sport or "").strip().upper())
    if not key or not sport_key:
        return []
    params = {
        "apiKey": key,
        "markets": "h2h,totals",
        "oddsFormat": "american",
        "bookmakers": _bookmakers(),
    }
    url = f"{API_BASE}/sports/{sport_key}/odds"
    if fetch_json is not None:
        payload = fetch_json(url, params)
    else:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    return payload if isinstance(payload, list) else []


def _match_event(pick: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any] | None:
    matchup = _norm(
        f"{pick.get('matchup') or pick.get('game') or ''} {pick.get('away_team') or ''} "
        f"{pick.get('home_team') or ''} {pick.get('pick') or ''}"
    )
    start = _parse_timestamp(pick.get("game_start_time") or pick.get("start_time"))
    for event in events:
        home = _nickname(event.get("home_team"))
        away = _nickname(event.get("away_team"))
        if not home or not away:
            continue
        if home not in matchup or away not in matchup:
            continue
        commence = _parse_timestamp(event.get("commence_time"))
        if start is not None and commence is not None and abs(commence - start) > timedelta(hours=3):
            continue
        return event
    return None


def _pick_direction(pick: dict[str, Any]) -> str:
    text = f"{pick.get('selection') or ''} {pick.get('pick') or ''}".lower()
    if "over" in text:
        return "over"
    if "under" in text:
        return "under"
    return "side"


def _pick_line(pick: dict[str, Any]) -> float | None:
    for field in ("market_line", "line", "total_line"):
        try:
            value = float(pick.get(field))
        except (TypeError, ValueError):
            continue
        return value
    return None


def _sharp_price(pick: dict[str, Any], event: dict[str, Any]) -> tuple[int, float | None] | None:
    """Return (american_odds, no_vig_probability) for the pick's own side."""
    bookmaker = None
    for candidate in event.get("bookmakers") or []:
        if isinstance(candidate, dict):
            bookmaker = candidate
            break
    if bookmaker is None:
        return None
    markets = {
        str(market.get("key")): market
        for market in bookmaker.get("markets") or []
        if isinstance(market, dict)
    }
    direction = _pick_direction(pick)
    if direction in {"over", "under"}:
        market = markets.get("totals")
        line = _pick_line(pick)
        if market is None or line is None:
            return None
        outcomes = [row for row in market.get("outcomes") or [] if isinstance(row, dict)]
        selected = opposite = None
        for row in outcomes:
            try:
                point = float(row.get("point"))
            except (TypeError, ValueError):
                continue
            if abs(point - line) > 0.01:
                continue
            name = str(row.get("name") or "").strip().lower()
            if name == direction:
                selected = _outcome_american(row)
            elif name in {"over", "under"}:
                opposite = _outcome_american(row)
        if selected is None:
            return None
    else:
        market = markets.get("h2h")
        if market is None:
            return None
        outcomes = [row for row in market.get("outcomes") or [] if isinstance(row, dict)]
        side_tokens = _norm(f"{pick.get('team') or ''}{pick.get('selection') or ''}{pick.get('pick') or ''}")
        selected = opposite = None
        implied_sum = 0.0
        selected_implied = None
        for row in outcomes:
            odds = _outcome_american(row)
            implied = american_implied_probability(odds)
            if implied is not None:
                implied_sum += implied
            if _nickname(row.get("name")) and _nickname(row.get("name")) in side_tokens:
                selected = odds
                selected_implied = implied
            elif odds is not None and len(outcomes) == 2:
                opposite = odds
        if selected is None:
            return None
        if selected_implied is not None and implied_sum > 0 and len(outcomes) >= 2:
            return selected, selected_implied / implied_sum
    selected_implied = american_implied_probability(selected)
    opposite_implied = american_implied_probability(opposite)
    if selected_implied is not None and opposite_implied is not None:
        hold = selected_implied + opposite_implied
        if hold > 0:
            return selected, selected_implied / hold
    return selected, None


def journal_sharp_rows(
    picks: Iterable[dict[str, Any]], date_iso: str, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Sharp closing rows for picks starting inside the sharp window."""
    if not _api_key():
        return []
    now = now or datetime.now(timezone.utc)
    window = _sharp_window()
    by_sport: dict[str, list[dict[str, Any]]] = {}
    for pick in picks:
        sport = str(pick.get("sport") or "").strip().upper()
        if sport not in SPORT_KEYS:
            continue
        start = _parse_timestamp(pick.get("game_start_time") or pick.get("start_time"))
        if start is None or not (now <= start <= now + window):
            continue
        by_sport.setdefault(sport, []).append(pick)
    rows: list[dict[str, Any]] = []
    captured_at = now.isoformat().replace("+00:00", "Z")
    provider = f"the-odds-api:{_bookmakers()}"
    for sport, sport_picks in by_sport.items():
        try:
            events = fetch_sharp_events(sport)
        except Exception as exc:
            print(f"[odds-api] {sport}: fetch failed: {exc}")
            continue
        if not events:
            continue
        for pick in sport_picks:
            event = _match_event(pick, events)
            if event is None:
                continue
            priced = _sharp_price(pick, event)
            if priced is None:
                continue
            odds, no_vig = priced
            decimal = 1.0 + (odds / 100.0 if odds > 0 else 100.0 / abs(odds))
            pick["sharp_no_vig_probability"] = round(no_vig, 6) if no_vig is not None else None
            pick["sharp_book"] = _bookmakers()
            rows.append(
                {
                    "marketIdentity": canonical_market_identity(
                        pick, mode="team", sport=sport, date_iso=date_iso
                    ),
                    "sport": sport,
                    "matchup": str(pick.get("matchup") or pick.get("game") or ""),
                    "pick": str(pick.get("pick") or ""),
                    "startTime": str(pick.get("game_start_time") or pick.get("start_time") or ""),
                    "oddsAmerican": odds,
                    "decimalOdds": round(decimal, 6),
                    "noVigProbability": round(no_vig, 6) if no_vig is not None else None,
                    "capturedAt": captured_at,
                    "provider": provider,
                    "role": "sharp",
                }
            )
    if rows:
        print(f"[odds-api] journaled {len(rows)} sharp row(s) from {provider}")
    return rows
