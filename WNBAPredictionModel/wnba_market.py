"""WNBA market-odds reader, vig removal, and Kelly stake helpers.

The pre-patch WNBA model produced picks based purely on its own
probability — `odds: null`, no edge calc, flat 1u stake. This module
gives the picks pipeline a real market price to compare against by
reading the cbs_odds SQLite table that the SportsLine scraper
populates (same table that powers the MLB / NBA odds lookups).

It is intentionally small and dependency-free: just sqlite3 + math.
The picks pipeline calls `lookup_market_odds(home_abbr, away_abbr)`,
gets back `MarketOdds(...)` (or None if no row exists), and uses
`compute_edge_units(...)` to turn the model probability + market
price into a real edge and Kelly-sized stake.
"""
from __future__ import annotations

import math
import sqlite3
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

try:
    from .wnba_teams import WNBA_TEAM_MAP, get_team_by_abbr
except ImportError:
    from wnba_teams import WNBA_TEAM_MAP, get_team_by_abbr


REPO_ROOT = Path(__file__).resolve().parents[1]
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard"
DB_CANDIDATES = [
    REPO_ROOT / "pickledger.db",
    REPO_ROOT / "NBAPredictionModel" / "pickledger.db",
]
_ABBR_ALIASES = {
    "GS": "GSV",
    "LVA": "LV",
    "NYL": "NY",
    "PHO": "PHX",
    "WSH": "WAS",
}


@dataclass
class MarketOdds:
    """Compact view of one matchup's SportsLine odds."""
    home_team_nickname: str
    away_team_nickname: str
    home_ml: Optional[int]
    away_ml: Optional[int]
    spread_home: Optional[float]
    spread_away: Optional[float]
    total_line: Optional[float]
    fetched_at: Optional[str]
    source: str = "SportsLine"
    spread_odds: Optional[int] = None
    total_odds: Optional[int] = None


def _db_path() -> Optional[Path]:
    for candidate in DB_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _abbr_to_nickname_terms(abbr: str) -> list[str]:
    """Return SportsLine-style search terms for an abbreviation.

    SportsLine stores rows by team nickname (e.g. "Mystics", "Liberty").
    We always have ``full_name`` in WNBA_TEAM_MAP — its last token is
    almost always the nickname. Returns a list of LIKE patterns to try
    in priority order.
    """
    try:
        team = get_team_by_abbr(abbr)
    except (KeyError, ValueError):
        return [abbr]
    full_name = str(team.get("full_name") or "").strip()
    if not full_name:
        return [abbr]
    nickname = full_name.split()[-1]
    return [nickname, full_name]


def _normalize_abbr(value: str | None) -> str:
    raw = str(value or "").strip().upper()
    return _ABBR_ALIASES.get(raw, raw)


def _coerce_american(value: object) -> Optional[int]:
    text = str(value or "").strip().replace("+", "")
    if not text or text == "--":
        return None
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _moneyline_from_side(side: object) -> Optional[int]:
    if not isinstance(side, dict):
        return None
    close = side.get("close") if isinstance(side.get("close"), dict) else {}
    open_ = side.get("open") if isinstance(side.get("open"), dict) else {}
    return _coerce_american(close.get("odds") or open_.get("odds"))


def _coerce_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _espn_date_key(date_str: str | None) -> str:
    if not date_str:
        return datetime.date.today().strftime("%Y%m%d")
    try:
        return datetime.date.fromisoformat(str(date_str)[:10]).strftime("%Y%m%d")
    except ValueError:
        return datetime.date.today().strftime("%Y%m%d")


def _lookup_espn_market_odds(
    home_abbr: str,
    away_abbr: str,
    date_str: str | None = None,
) -> Optional[MarketOdds]:
    """Fallback to ESPN scoreboard odds when SportsLine has no WNBA row."""
    home_target = _normalize_abbr(home_abbr)
    away_target = _normalize_abbr(away_abbr)
    try:
        resp = requests.get(
            ESPN_SCOREBOARD_URL,
            params={"dates": _espn_date_key(date_str)},
            timeout=12,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    for event in data.get("events", []) or []:
        competitions = event.get("competitions") or []
        competition = competitions[0] if competitions else {}
        competitors = competition.get("competitors") or []
        event_home = ""
        event_away = ""
        for comp in competitors:
            team = comp.get("team") or {}
            abbr = _normalize_abbr(team.get("abbreviation"))
            if comp.get("homeAway") == "home":
                event_home = abbr
            elif comp.get("homeAway") == "away":
                event_away = abbr
        if event_home != home_target or event_away != away_target:
            continue

        odds_rows = competition.get("odds") or []
        odds = odds_rows[0] if odds_rows else {}
        moneyline = odds.get("moneyline") or {}
        home_team_odds = odds.get("homeTeamOdds") or {}
        away_team_odds = odds.get("awayTeamOdds") or {}
        home_ml = _moneyline_from_side(moneyline.get("home")) or _coerce_american(home_team_odds.get("moneyLine"))
        away_ml = _moneyline_from_side(moneyline.get("away")) or _coerce_american(away_team_odds.get("moneyLine"))

        spread = _coerce_float(odds.get("spread"))
        total = _coerce_float(odds.get("overUnder"))
        if home_ml is None and away_ml is None and spread is None and total is None:
            return None

        home_favorite = bool(home_team_odds.get("favorite"))
        away_favorite = bool(away_team_odds.get("favorite"))
        spread_home = None
        spread_away = None
        if spread is not None:
            spread_magnitude = abs(spread)
            if home_favorite:
                spread_home = -spread_magnitude
                spread_away = spread_magnitude
            elif away_favorite:
                spread_away = -spread_magnitude
                spread_home = spread_magnitude
            else:
                spread_home = spread
                spread_away = -spread

        spread_odds = (
            _coerce_american(home_team_odds.get("spreadOdds"))
            or _coerce_american(away_team_odds.get("spreadOdds"))
            or (-110 if spread is not None else None)
        )
        total_odds = (
            _coerce_american(odds.get("overOdds"))
            or _coerce_american(odds.get("underOdds"))
            or (-110 if total is not None else None)
        )

        provider = odds.get("provider") or {}
        provider_name = str(provider.get("displayName") or provider.get("name") or "ESPN odds").strip()
        return MarketOdds(
            home_team_nickname=home_target,
            away_team_nickname=away_target,
            home_ml=home_ml,
            away_ml=away_ml,
            spread_home=spread_home,
            spread_away=spread_away,
            total_line=total,
            fetched_at=None,
            source=f"{provider_name} via ESPN",
            spread_odds=spread_odds,
            total_odds=total_odds,
        )
    return None


def lookup_market_odds(home_abbr: str, away_abbr: str, date_str: str | None = None) -> Optional[MarketOdds]:
    """Find the most recent SportsLine WNBA row for this matchup.

    Returns None if no SportsLine row matches. Doesn't raise on DB issues.
    """
    home_terms = _abbr_to_nickname_terms(home_abbr)
    away_terms = _abbr_to_nickname_terms(away_abbr)
    db_path = _db_path()
    if not db_path:
        return _lookup_espn_market_odds(home_abbr, away_abbr, date_str)

    try:
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            for home_term in home_terms:
                for away_term in away_terms:
                    cursor.execute(
                        """
                        SELECT home_team, away_team, ml_home, ml_away,
                               spread_home, spread_away, total_line, fetched_at,
                               spread_odds, total_odds
                          FROM cbs_odds
                         WHERE league = 'WNBA'
                           AND home_team LIKE ?
                           AND away_team LIKE ?
                         ORDER BY fetched_at DESC
                         LIMIT 1
                        """,
                        (f"%{home_term}%", f"%{away_term}%"),
                    )
                    row = cursor.fetchone()
                    if row:
                        return MarketOdds(
                            home_team_nickname=str(row[0] or ""),
                            away_team_nickname=str(row[1] or ""),
                            home_ml=int(row[2]) if row[2] is not None else None,
                            away_ml=int(row[3]) if row[3] is not None else None,
                            spread_home=float(row[4]) if row[4] is not None else None,
                            spread_away=float(row[5]) if row[5] is not None else None,
                            total_line=float(row[6]) if row[6] is not None else None,
                            fetched_at=str(row[7] or ""),
                            spread_odds=int(row[8]) if row[8] is not None else None,
                            total_odds=int(row[9]) if row[9] is not None else None,
                        )
        finally:
            conn.close()
    except sqlite3.Error:
        pass
    return _lookup_espn_market_odds(home_abbr, away_abbr, date_str)


def american_to_implied(odds: int) -> float:
    """Convert an American odds price into its raw implied probability."""
    o = int(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return abs(o) / (abs(o) + 100.0)


def remove_vig(home_ml: int, away_ml: int) -> tuple[float, float]:
    """Vig-remove a two-sided moneyline — returns (home_prob, away_prob)
    that sum to 1.0. Each side is its raw implied prob normalized by the
    sum of both sides.
    """
    h = american_to_implied(home_ml)
    a = american_to_implied(away_ml)
    total = h + a
    if total <= 0:
        return 0.5, 0.5
    return h / total, a / total


def american_to_decimal(odds: int) -> float:
    o = int(odds)
    if o > 0:
        return 1.0 + (o / 100.0)
    return 1.0 + (100.0 / abs(o))


def quarter_kelly_units(edge: float, american_odds: int, cap: float = 1.75) -> float:
    """Quarter-Kelly stake for a given edge and American price, capped."""
    if edge <= 0:
        return 0.0
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    raw = edge / b / 4.0
    return round(min(cap, max(0.0, raw)), 2)


@dataclass
class EdgeAssessment:
    market_pick_odds: Optional[int]
    market_pick_prob: Optional[float]
    edge: Optional[float]
    kelly_units: Optional[float]


def compute_edge_units(
    pick_team_is_home: bool,
    model_pick_prob: float,
    market: Optional[MarketOdds],
) -> EdgeAssessment:
    """Compute model edge over vig-removed market and a Kelly stake.

    Returns ``EdgeAssessment`` with all numeric fields None when the
    market entry is missing or one side of the moneyline isn't priced.
    """
    if market is None or market.home_ml is None or market.away_ml is None:
        return EdgeAssessment(market_pick_odds=None, market_pick_prob=None, edge=None, kelly_units=None)

    home_prob, away_prob = remove_vig(market.home_ml, market.away_ml)
    if pick_team_is_home:
        market_pick_odds = market.home_ml
        market_pick_prob = home_prob
    else:
        market_pick_odds = market.away_ml
        market_pick_prob = away_prob

    edge = float(model_pick_prob) - float(market_pick_prob)
    kelly = quarter_kelly_units(edge, market_pick_odds) if edge > 0 else 0.0
    return EdgeAssessment(
        market_pick_odds=int(market_pick_odds),
        market_pick_prob=round(market_pick_prob, 4),
        edge=round(edge, 4),
        kelly_units=kelly,
    )
