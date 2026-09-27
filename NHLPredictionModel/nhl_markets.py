"""Observed NHL prices from the public DraftKings board.

ESPN's hockey scoreboard often has no odds node in preseason. This module
reads the posted DraftKings game lines and the event category list. A market
that is not on that board is left empty. Nothing here fills in a line, a
price, or a player mean.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from urllib.request import Request, urlopen

try:
    from nhl_core import canonical_abbrev
except ImportError:
    from .nhl_core import canonical_abbrev

ODDS_SOURCE = "draftkings"
PRESEASON_LEAGUE_ID = "26150"
REGULAR_LEAGUE_ID = "42133"
LEAGUE_URL = "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusva/v1/leagues/{league_id}"
EVENT_CATEGORIES_URL = (
    "https://sportsbook-nash.draftkings.com/api/sportscontent/dkusva/v1/events/{event_id}/categories"
)
PREGAME_BOOK_STATUSES = {"NOT_STARTED"}
START_MATCH_WINDOW = timedelta(hours=6)
TEAM_TOTAL_NAME = "team total"
PLAYER_MARKET_TOKENS = (
    "player",
    "goalscorer",
    "shots on goal",
    "assists",
    "points",
)


FetchJson = Callable[[str], dict[str, Any] | None]


def american_odds(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("american")
    text = str(value or "").strip().replace("\u2212", "-").replace("\u2013", "-").replace(" ", "")
    if not text:
        return None
    try:
        number = int(round(float(text)))
    except (TypeError, ValueError):
        return None
    if abs(number) < 100:
        return None
    return number


def _default_fetch(url: str) -> dict[str, Any] | None:
    request = Request(url, headers={"User-Agent": "PickLedger/1.0", "Accept": "application/json"})
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    try:
        from curl_cffi import requests as curl_requests
    except ImportError:
        return None
    try:
        response = curl_requests.get(
            url,
            headers={"Accept": "application/json", "Origin": "https://sportsbook.draftkings.com"},
            impersonate="chrome",
            timeout=20,
        )
        if int(getattr(response, "status_code", 0) or 0) != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _participant_abbrev(participant: dict[str, Any]) -> str:
    metadata = participant.get("metadata") if isinstance(participant.get("metadata"), dict) else {}
    short = str(metadata.get("shortName") or "").strip().upper()
    if short:
        return canonical_abbrev(short)
    name = str(participant.get("name") or "").strip()
    token = name.split(" ", 1)[0].upper()
    return canonical_abbrev(token)


def _side_for_selection(selection: dict[str, Any]) -> str:
    outcome = str(selection.get("outcomeType") or "").strip().lower()
    if outcome in {"home", "away", "over", "under"}:
        return outcome
    for participant in selection.get("participants") or []:
        if not isinstance(participant, dict):
            continue
        role = str(participant.get("venueRole") or "").strip().lower()
        if role in {"home", "away"}:
            return role
    return ""


def _is_player_market(market: dict[str, Any]) -> bool:
    name = str(market.get("name") or "").lower()
    market_type = market.get("marketType") if isinstance(market.get("marketType"), dict) else {}
    type_name = str(market_type.get("name") or "").lower()
    blob = f"{name} {type_name}"
    if "team total" in blob:
        return False
    return any(token in blob for token in PLAYER_MARKET_TOKENS)


def _is_team_total_market(market: dict[str, Any]) -> bool:
    name = str(market.get("name") or "").lower()
    market_type = market.get("marketType") if isinstance(market.get("marketType"), dict) else {}
    type_name = str(market_type.get("name") or "").lower()
    return TEAM_TOTAL_NAME in name or TEAM_TOTAL_NAME in type_name


def _parse_start(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    if "." in text:
        head, tail = text.split(".", 1)
        split_at = next((index for index, char in enumerate(tail) if char in "+-"), len(tail))
        fraction, rest = tail[:split_at], tail[split_at:]
        fraction = "".join(char for char in fraction if char.isdigit())[:6]
        text = f"{head}.{fraction}{rest}" if fraction else f"{head}{rest}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _empty_quote(event_id: str, status: str) -> dict[str, Any]:
    return {
        "book_event_id": event_id,
        "book_status": status,
        "odds_source": ODDS_SOURCE,
        "home_abbrev": "",
        "away_abbrev": "",
        "team_totals": {},
        "player_props": [],
        "posted_market_names": [],
    }


def _apply_game_lines(quote: dict[str, Any], markets: list[dict[str, Any]], selections: list[dict[str, Any]]) -> None:
    by_id = {str(market.get("id") or ""): market for market in markets}
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        market = by_id.get(str(selection.get("marketId") or ""))
        if market is None:
            continue
        name = str(market.get("name") or "")
        side = _side_for_selection(selection)
        price = american_odds((selection.get("displayOdds") or {}).get("american") if isinstance(selection.get("displayOdds"), dict) else None)
        if price is None:
            continue
        if name == "Moneyline" and side in {"home", "away"}:
            quote[f"{side}_moneyline"] = price
        elif name == "Puck Line" and side in {"home", "away"}:
            try:
                points = float(selection.get("points"))
            except (TypeError, ValueError):
                continue
            quote[f"{side}_spread_line"] = points
            quote[f"{side}_spread_odds"] = price
        elif name == "Total" and side in {"over", "under"}:
            try:
                points = float(selection.get("points"))
            except (TypeError, ValueError):
                continue
            quote["total_line"] = points
            quote[f"{side}_odds"] = price


def favorite_magnitude(over_odds: int, under_odds: int) -> int | None:
    """How far the favorite is from even.

    The favorite is the more negative American price. A pick'em can have both
    sides minus (for example -125/-115). Two plus prices are not a two-way.
    """
    if over_odds >= 0 and under_odds >= 0:
        return None
    favorite = over_odds if over_odds < under_odds else under_odds
    if favorite >= 0:
        return None
    return abs(favorite)


def choose_balanced_line(lines: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the posted team-total number closest to even.

    Books list several alternates on one market. The first number is often
    0.5, which is not the main line. If two lines are equally tight, none is
    chosen.
    """
    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for row in lines:
        try:
            over_odds = int(row["over_odds"])
            under_odds = int(row["under_odds"])
            line = float(row["line"])
        except (KeyError, TypeError, ValueError):
            continue
        magnitude = favorite_magnitude(over_odds, under_odds)
        if magnitude is None:
            continue
        ranked.append((magnitude, line, row))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][2]


def _apply_team_totals(quote: dict[str, Any], markets: list[dict[str, Any]], selections: list[dict[str, Any]]) -> None:
    by_id = {str(market.get("id") or ""): market for market in markets if _is_team_total_market(market)}
    if not by_id:
        return
    grouped: dict[str, dict[float, dict[str, Any]]] = {}
    names: dict[str, str] = {}
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        market = by_id.get(str(selection.get("marketId") or ""))
        if market is None:
            continue
        side = ""
        team_name = ""
        for participant in selection.get("participants") or []:
            if not isinstance(participant, dict):
                continue
            if str(participant.get("type") or "").lower() != "team":
                continue
            role = str(participant.get("venueRole") or "").strip().lower()
            if role in {"home", "away"}:
                side = role
                team_name = str(participant.get("name") or "")
        direction = str(selection.get("outcomeType") or selection.get("label") or "").strip().lower()
        if direction not in {"over", "under"} or side not in {"home", "away"}:
            continue
        price = american_odds((selection.get("displayOdds") or {}).get("american") if isinstance(selection.get("displayOdds"), dict) else None)
        try:
            line = float(selection.get("points"))
        except (TypeError, ValueError):
            continue
        if price is None:
            continue
        bucket = grouped.setdefault(side, {}).setdefault(line, {"line": line})
        bucket[f"{direction}_odds"] = price
        if team_name:
            names[side] = team_name
    published: dict[str, dict[str, Any]] = {}
    for side, by_line in grouped.items():
        complete = [
            row for row in by_line.values()
            if row.get("over_odds") is not None and row.get("under_odds") is not None
        ]
        chosen = choose_balanced_line(complete)
        if chosen is None:
            continue
        published[side] = {
            "team": names.get(side) or "",
            "line": chosen["line"],
            "over_odds": chosen["over_odds"],
            "under_odds": chosen["under_odds"],
            "alternate_lines": len(complete),
        }
    quote["team_totals"] = published


def _apply_player_props(quote: dict[str, Any], markets: list[dict[str, Any]], selections: list[dict[str, Any]]) -> None:
    """Keep posted player markets. A mean is never filled in here."""
    by_id = {str(market.get("id") or ""): market for market in markets if _is_player_market(market)}
    if not by_id:
        return
    grouped: dict[tuple[str, str, float], dict[str, Any]] = {}
    for selection in selections:
        if not isinstance(selection, dict):
            continue
        market = by_id.get(str(selection.get("marketId") or ""))
        if market is None:
            continue
        player = ""
        team = ""
        for participant in selection.get("participants") or []:
            if not isinstance(participant, dict):
                continue
            kind = str(participant.get("type") or "").lower()
            if kind == "player" and not player:
                player = str(participant.get("name") or "").strip()
            elif kind == "team":
                team = _participant_abbrev(participant)
        if not player:
            continue
        direction = str(selection.get("outcomeType") or selection.get("label") or "").strip().lower()
        if direction not in {"over", "under"}:
            continue
        price = american_odds((selection.get("displayOdds") or {}).get("american") if isinstance(selection.get("displayOdds"), dict) else None)
        try:
            line = float(selection.get("points"))
        except (TypeError, ValueError):
            continue
        if price is None:
            continue
        key = (player, str(market.get("name") or ""), line)
        row = grouped.setdefault(key, {
            "player": player,
            "team": team,
            "stat": str(market.get("name") or ""),
            "stat_label": str(market.get("name") or ""),
            "line": line,
            "mean": None,
        })
        row[f"{direction}_odds"] = price
    quote["player_props"] = [
        row
        for row in grouped.values()
        if row.get("over_odds") is not None and row.get("under_odds") is not None
    ]


def _select_quote(candidates: list[dict[str, Any]], game: dict[str, Any]) -> dict[str, Any] | None:
    """Keep the posted board whose start matches this slate game.

    The same clubs can be priced twice, once for tonight and once for a later
    regular-season date. Team names alone would attach the later price.
    """
    game_start = _parse_start(game.get("start_time"))
    if game_start is None:
        return None
    best: dict[str, Any] | None = None
    best_gap: timedelta | None = None
    for quote in candidates:
        book_start = _parse_start(quote.get("start_time"))
        if book_start is None:
            continue
        gap = abs(book_start - game_start)
        if gap > START_MATCH_WINDOW:
            continue
        if best_gap is None or gap < best_gap:
            best = quote
            best_gap = gap
    return best


def _quotes_from_payload(payload: dict[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    events = [event for event in payload.get("events") or [] if isinstance(event, dict)]
    markets = [market for market in payload.get("markets") or [] if isinstance(market, dict)]
    selections = [selection for selection in payload.get("selections") or [] if isinstance(selection, dict)]
    markets_by_event: dict[str, list[dict[str, Any]]] = {}
    for market in markets:
        markets_by_event.setdefault(str(market.get("eventId") or ""), []).append(market)
    quotes: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        status = str(event.get("status") or "")
        if status not in PREGAME_BOOK_STATUSES:
            continue
        home_abbrev = away_abbrev = ""
        for participant in event.get("participants") or []:
            if not isinstance(participant, dict):
                continue
            role = str(participant.get("venueRole") or "").lower()
            abbrev = _participant_abbrev(participant)
            if role == "home":
                home_abbrev = abbrev
            elif role == "away":
                away_abbrev = abbrev
        if not home_abbrev or not away_abbrev:
            continue
        event_id = str(event.get("id") or "")
        event_markets = markets_by_event.get(event_id, [])
        event_market_ids = {str(market.get("id") or "") for market in event_markets}
        event_selections = [
            selection for selection in selections
            if str(selection.get("marketId") or "") in event_market_ids
        ]
        quote = _empty_quote(event_id, status)
        quote["home_abbrev"] = home_abbrev
        quote["away_abbrev"] = away_abbrev
        quote["start_time"] = str(event.get("startEventDate") or "")
        quote["posted_market_names"] = sorted({str(market.get("name") or "") for market in event_markets if market.get("name")})
        _apply_game_lines(quote, event_markets, event_selections)
        home_line = quote.get("home_spread_line")
        if isinstance(home_line, (int, float)):
            quote["spread_line"] = float(home_line)
        _apply_team_totals(quote, event_markets, event_selections)
        _apply_player_props(quote, event_markets, event_selections)
        quotes.setdefault((away_abbrev, home_abbrev), []).append(quote)
    return quotes


def _merge_event_categories(quote: dict[str, Any], payload: dict[str, Any] | None) -> str:
    """Fold an event category payload into the quote. Returns the check status."""
    if not isinstance(payload, dict):
        return "quote_check_failed"
    markets = [market for market in payload.get("markets") or [] if isinstance(market, dict)]
    selections = [selection for selection in payload.get("selections") or [] if isinstance(selection, dict)]
    names = sorted({str(market.get("name") or "") for market in markets if market.get("name")})
    quote["posted_market_names"] = sorted(set(quote.get("posted_market_names") or []) | set(names))
    _apply_team_totals(quote, markets, selections)
    _apply_player_props(quote, markets, selections)
    return "checked"


def fetch_pregame_quotes(
    games: list[dict[str, Any]],
    *,
    fetch_json: FetchJson | None = None,
) -> dict[str, Any]:
    """Attach observed DraftKings prices onto pregame games, in place.

    Returns a summary of which extra markets were actually posted. Missing
    markets stay absent on the game.
    """
    fetch = fetch_json or _default_fetch
    quotes: dict[tuple[str, str], list[dict[str, Any]]] = {}
    feeds_read = 0
    for league_id in (PRESEASON_LEAGUE_ID, REGULAR_LEAGUE_ID):
        payload = fetch(LEAGUE_URL.format(league_id=league_id))
        if not isinstance(payload, dict):
            continue
        feeds_read += 1
        for key, rows in _quotes_from_payload(payload).items():
            quotes.setdefault(key, []).extend(rows)
    matched = 0
    team_total_games = 0
    player_prop_games = 0
    category_failures = 0
    for game in games:
        if not isinstance(game, dict):
            continue
        key = (canonical_abbrev(game.get("away_abbrev")), canonical_abbrev(game.get("home_abbrev")))
        quote = _select_quote(quotes.get(key) or [], game)
        if quote is None:
            continue
        matched += 1
        event_id = str(quote.get("book_event_id") or "")
        if event_id:
            status = _merge_event_categories(quote, fetch(EVENT_CATEGORIES_URL.format(event_id=event_id)))
            if status != "checked":
                category_failures += 1
                quote["category_check"] = status
            else:
                quote["category_check"] = status
        for field in (
            "home_moneyline",
            "away_moneyline",
            "spread_line",
            "home_spread_odds",
            "away_spread_odds",
            "total_line",
            "over_odds",
            "under_odds",
            "odds_source",
            "book_event_id",
        ):
            if quote.get(field) is not None:
                game[field] = quote[field]
        if quote.get("team_totals"):
            game["team_totals"] = quote["team_totals"]
            team_total_games += 1
        if quote.get("player_props"):
            game["player_props"] = quote["player_props"]
            player_prop_games += 1
        game["posted_market_names"] = list(quote.get("posted_market_names") or [])
    return {
        "feeds_read": feeds_read,
        "matched_games": matched,
        "team_total_games": team_total_games,
        "player_prop_games": player_prop_games,
        "category_failures": category_failures,
        "odds_source": ODDS_SOURCE,
    }
