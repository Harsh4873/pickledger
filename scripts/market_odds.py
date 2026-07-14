#!/usr/bin/env python3
"""Capture real two-sided market odds and attach them to cached team picks.

Every cache write may call :func:`apply_market_odds_to_payload`.  For each
pregame game on the slate it records the current DraftKings prices published
through ESPN (both moneylines, total over/under prices, spread prices, and
MLB first-5-innings markets) directly onto matching picks:

- scraped feed picks keep their own executable ``odds`` and gain the paired
  fields (``selected_odds``/``opposite_odds`` or over/under pairs) that let
  the Profit Desk verify a true no-vig baseline;
- in-house team-model picks that only carried an assumed price have that
  price replaced with the real observed price for their exact market and
  line, which is what makes their pregame-ledger rows financially
  measurable.

The attach step only ever runs for games that are still pregame, so a price
can never be captured after start.  Once a game goes live the previously
captured pregame prices are preserved by the cache merge layer.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping

import requests


FetchJson = Callable[[str, dict[str, Any]], Any]

PROVIDER_ID = "100"  # DraftKings via ESPN
REQUEST_TIMEOUT = 20

SPORT_LEAGUES: dict[str, tuple[str, str]] = {
    "MLB": ("baseball", "mlb"),
    "WNBA": ("basketball", "wnba"),
    "NBA": ("basketball", "nba"),
    "NBA SUMMER": ("basketball", "nba-summer"),
    "FIFA WC": ("soccer", "fifa.world"),
}

# In-house model buckets whose assumed prices may be replaced with a real
# observed price.  External feed buckets never have their own odds replaced.
TEAM_MODEL_BUCKET_KEYS = {
    "mlb_new",
    "mlb_inning",
    "mlb_first_five",
    "wnba",
    "nba",
    "nba_playoffs",
    "nba_summer",
    "fifa_world_cup",
}

F5_BUCKET_KEYS = {"mlb_first_five"}

_TEAM_REF_RE = re.compile(r"/teams/(\d+)")
_SIGNED_LINE_RE = re.compile(r"(?<![A-Za-z0-9])([+-]\d+(?:\.\d+)?)")
_DIRECTION_RE = re.compile(r"\b(over|under)\b", re.IGNORECASE)

_NON_EXECUTABLE_MARKERS = (
    "assumed",
    "synthetic",
    "proxy",
    "fallback",
    "default",
    "estimated",
    "derived",
    "model output",
    "model_output",
    "model price",
    "model_price",
    "in_house",
    "baseline",
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return " ".join(
        "".join(char.lower() if char.isalnum() else " " for char in _text(value)).split()
    )


def _number(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _american(value: Any) -> int | None:
    text = _text(value).replace("EVEN", "+100").replace("even", "+100")
    number = _number(text)
    if number is None or number == 0 or -100.0 < number < 100.0:
        return None
    return int(round(number))


def _implied(odds: int | None) -> float | None:
    if odds is None:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def _now_iso(now: datetime | None = None) -> str:
    stamp = now or datetime.now(timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _default_fetch(url: str, params: dict[str, Any]) -> Any:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # pragma: no cover - network resilience
        print(f"[market-odds] fetch failed for {url}: {exc}")
        return None


def _price_from_odds_node(node: Any) -> int | None:
    """Read a close-then-open american price from an ESPN odds side node."""

    if not isinstance(node, Mapping):
        return None
    for phase in ("close", "current", "open"):
        phase_node = node.get(phase)
        if isinstance(phase_node, Mapping):
            price = _american(phase_node.get("odds") or phase_node.get("american"))
            if price is not None:
                return price
    return _american(node.get("odds") or node.get("american") or node.get("moneyLine"))


def _team_names(competitor: Mapping[str, Any]) -> set[str]:
    team = competitor.get("team") if isinstance(competitor.get("team"), Mapping) else {}
    names = {
        _norm(team.get(field))
        for field in ("displayName", "shortDisplayName", "name", "abbreviation", "location")
    }
    return {name for name in names if name}


def _parse_scoreboard_event(event: Mapping[str, Any]) -> dict[str, Any] | None:
    state = _text(((event.get("status") or {}).get("type") or {}).get("state"))
    competitions = event.get("competitions") or []
    if not competitions or not isinstance(competitions[0], Mapping):
        return None
    competition = competitions[0]
    competitors = [row for row in competition.get("competitors") or [] if isinstance(row, Mapping)]
    home = next((row for row in competitors if _text(row.get("homeAway")) == "home"), None)
    away = next((row for row in competitors if _text(row.get("homeAway")) == "away"), None)
    if home is None or away is None:
        return None
    game: dict[str, Any] = {
        "eventId": _text(event.get("id")),
        "state": state,
        "startTime": _text(event.get("date") or competition.get("date")),
        "homeNames": _team_names(home),
        "awayNames": _team_names(away),
        "homeTeamId": _text((home.get("team") or {}).get("id")),
        "awayTeamId": _text((away.get("team") or {}).get("id")),
        "provider": "",
        "markets": {},
    }
    odds_rows = [row for row in competition.get("odds") or [] if isinstance(row, Mapping)]
    if not odds_rows:
        return game
    odds = odds_rows[0]
    provider = odds.get("provider") if isinstance(odds.get("provider"), Mapping) else {}
    game["provider"] = _text(provider.get("displayName") or provider.get("name")) or "unknown"
    markets: dict[str, Any] = {}

    moneyline = odds.get("moneyline") if isinstance(odds.get("moneyline"), Mapping) else {}
    home_ml = _price_from_odds_node(moneyline.get("home"))
    away_ml = _price_from_odds_node(moneyline.get("away"))
    draw_ml = _price_from_odds_node(moneyline.get("draw"))
    if home_ml is None:
        home_ml = _american((odds.get("homeTeamOdds") or {}).get("moneyLine"))
    if away_ml is None:
        away_ml = _american((odds.get("awayTeamOdds") or {}).get("moneyLine"))
    if draw_ml is None:
        draw_ml = _american((odds.get("drawOdds") or {}).get("moneyLine"))
    if home_ml is not None and away_ml is not None:
        markets["moneyline"] = {"home": home_ml, "away": away_ml, "draw": draw_ml}

    total = odds.get("total") if isinstance(odds.get("total"), Mapping) else {}
    total_line = _number(odds.get("overUnder"))
    over_price = _price_from_odds_node(total.get("over"))
    under_price = _price_from_odds_node(total.get("under"))
    if total_line is not None and over_price is not None and under_price is not None:
        markets["total"] = {"line": total_line, "over": over_price, "under": under_price}

    point_spread = odds.get("pointSpread") if isinstance(odds.get("pointSpread"), Mapping) else {}
    home_spread = _number(odds.get("spread"))
    spread_home_price = _price_from_odds_node(point_spread.get("home"))
    spread_away_price = _price_from_odds_node(point_spread.get("away"))
    if home_spread is not None and spread_home_price is not None and spread_away_price is not None:
        markets["spread"] = {
            "homeLine": home_spread,
            "home": spread_home_price,
            "away": spread_away_price,
        }

    game["markets"] = markets
    return game


def _parse_f5_prop_items(game: dict[str, Any], items: Iterable[Mapping[str, Any]]) -> None:
    """Attach MLB first-5-innings markets parsed from the propBets feed."""

    f5_moneyline: dict[str, int] = {}
    f5_run_line: dict[str, dict[str, Any]] = {}
    totals_by_line: dict[float, list[int]] = {}
    for item in items:
        type_name = _text((item.get("type") or {}).get("name"))
        if not type_name.startswith("1st 5 Innings"):
            continue
        odds_node = item.get("odds") if isinstance(item.get("odds"), Mapping) else {}
        price = _american(((odds_node.get("american") or {}).get("value")))
        if price is None:
            continue
        team_ref = _text((item.get("team") or {}).get("$ref"))
        team_match = _TEAM_REF_RE.search(team_ref)
        team_id = team_match.group(1) if team_match else ""
        side = (
            "home"
            if team_id and team_id == game.get("homeTeamId")
            else "away"
            if team_id and team_id == game.get("awayTeamId")
            else ""
        )
        line = _number((odds_node.get("total") or {}).get("value"))
        if type_name == "1st 5 Innings Moneyline" and side:
            f5_moneyline[side] = price
        elif type_name == "1st 5 Innings Run Line" and side and line is not None:
            existing = f5_run_line.setdefault(side, {})
            existing[round(line, 2)] = price
        elif type_name == "1st 5 Innings Total Runs" and line is not None:
            totals_by_line.setdefault(round(line, 2), []).append(price)

    markets = game.setdefault("markets", {})
    if "home" in f5_moneyline and "away" in f5_moneyline:
        markets["f5_moneyline"] = f5_moneyline
    if f5_run_line:
        markets["f5_run_line"] = f5_run_line
    f5_totals = {
        # The ESPN prop feed publishes each total's Over row before its Under
        # row; the repository's player-prop parser relies on the same
        # ordering convention.
        line: {"over": prices[0], "under": prices[1]}
        for line, prices in totals_by_line.items()
        if len(prices) == 2
    }
    if f5_totals:
        markets["f5_totals"] = f5_totals


def fetch_market_odds_for_date(
    date_iso: str,
    sports: Iterable[str] | None = None,
    fetch_json: FetchJson | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return pregame market odds per sport for the given slate date."""

    fetch = fetch_json or _default_fetch
    compact = date_iso.replace("-", "")
    book: dict[str, list[dict[str, Any]]] = {}
    for sport in sports if sports is not None else SPORT_LEAGUES:
        league = SPORT_LEAGUES.get(sport)
        if league is None:
            continue
        sport_path, league_path = league
        payload = fetch(
            f"https://site.api.espn.com/apis/site/v2/sports/{sport_path}/{league_path}/scoreboard",
            {"dates": compact, "limit": 100},
        )
        games: list[dict[str, Any]] = []
        for event in (payload or {}).get("events") or []:
            if not isinstance(event, Mapping):
                continue
            game = _parse_scoreboard_event(event)
            if game is None or game["state"] != "pre":
                continue
            if sport == "MLB" and game["eventId"]:
                props = fetch(
                    (
                        "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/"
                        f"events/{game['eventId']}/competitions/{game['eventId']}/odds/"
                        f"{PROVIDER_ID}/propBets"
                    ),
                    {"lang": "en", "region": "us", "limit": 1000},
                )
                items = [row for row in (props or {}).get("items") or [] if isinstance(row, Mapping)]
                if items:
                    _parse_f5_prop_items(game, items)
            games.append(game)
        if games:
            book[sport] = games
    return book


# ---------------------------------------------------------------------------
# Pick matching and attachment
# ---------------------------------------------------------------------------


def _pick_team_names(pick: Mapping[str, Any]) -> tuple[str, str]:
    return _norm(pick.get("away_team")), _norm(pick.get("home_team"))


def _names_match(pick_name: str, event_names: set[str]) -> bool:
    if not pick_name:
        return False
    if pick_name in event_names:
        return True
    return any(
        name and (pick_name.endswith(f" {name}") or name.endswith(f" {pick_name}"))
        for name in event_names
    )


def _match_game(pick: Mapping[str, Any], games: list[dict[str, Any]]) -> dict[str, Any] | None:
    away, home = _pick_team_names(pick)
    if not away or not home:
        matchup = _text(pick.get("matchup") or pick.get("game"))
        parts = re.split(r"\s+(?:@|vs\.?|v\.)\s+", matchup, flags=re.IGNORECASE)
        if len(parts) == 2:
            away = away or _norm(parts[0])
            home = home or _norm(parts[1])
    if not away or not home:
        return None
    for game in games:
        # Orientation-independent: several feeds list matchups in reverse
        # order, so each pick team just has to land on a distinct real side.
        # Market prices always stay anchored to the event's true home/away.
        def _side_of(name: str) -> str:
            if _names_match(name, game["homeNames"]):
                return "home"
            if _names_match(name, game["awayNames"]):
                return "away"
            return ""

        home_side = _side_of(home)
        away_side = _side_of(away)
        if home_side and away_side and home_side != away_side:
            return game
    return None


def _selected_side(pick: Mapping[str, Any], game: dict[str, Any]) -> str:
    selected = _norm(pick.get("team") or pick.get("side") or pick.get("selection"))
    if selected in {"draw", "tie"}:
        return "draw"
    if not selected:
        text = _text(pick.get("pick") or pick.get("tip"))
        text = re.sub(r"\([^)]*\)", " ", text)
        text = _SIGNED_LINE_RE.sub(" ", text)
        text = re.sub(r"\b(?:moneyline|ml|to win|wins?|cover|run line|spread)\b", " ", text, flags=re.IGNORECASE)
        selected = _norm(text)
    if not selected:
        return ""
    if _names_match(selected, game["homeNames"]):
        return "home"
    if _names_match(selected, game["awayNames"]):
        return "away"
    # Fall back to substring containment for pick labels with extra words.
    for side in ("home", "away"):
        if any(name and name in selected for name in game[f"{side}Names"]):
            return side
    return ""


def _pick_direction(pick: Mapping[str, Any]) -> str:
    explicit = _norm(pick.get("direction") or pick.get("selection"))
    if explicit in {"over", "under"}:
        return explicit
    match = _DIRECTION_RE.search(_text(pick.get("pick")))
    return match.group(1).lower() if match else ""


def _pick_line(pick: Mapping[str, Any]) -> float | None:
    for key in ("line", "market_line", "vegas", "market_total_line"):
        number = _number(pick.get(key))
        if number is not None:
            return number
    direction = _pick_direction(pick)
    if direction:
        direction_match = re.search(
            rf"\b{direction}\b[^0-9+-]*([+-]?\d+(?:\.\d+)?)",
            _text(pick.get("pick")),
            flags=re.IGNORECASE,
        )
        if direction_match:
            return _number(direction_match.group(1))
    return None


def _pick_spread_line(pick: Mapping[str, Any]) -> float | None:
    for key in ("line", "spread", "handicap"):
        number = _number(pick.get(key))
        if number is not None:
            return number
    text = re.sub(r"\([^)]*\)", " ", _text(pick.get("pick")))
    match = _SIGNED_LINE_RE.search(text)
    return _number(match.group(1)) if match else None


def _looks_assumed(pick: Mapping[str, Any]) -> bool:
    odds = _american(pick.get("odds"))
    if odds is None:
        return True
    # Assumed markers dominate: some model buckets stamp market_priced=True
    # on a user-assumed fallback price (e.g. user_assumed_f5_moneyline).
    assumed = _american(pick.get("assumed_odds"))
    if assumed is not None and assumed == odds:
        return True
    markers = " ".join(
        _text(pick.get(key)).lower()
        for key in ("pricing_type", "price_source", "odds_source", "market_source", "market_total_source")
    )
    if any(marker in markers for marker in _NON_EXECUTABLE_MARKERS):
        return True
    if pick.get("market_priced") is True:
        return False
    # A team-model pick with no market provenance at all is assumed-priced.
    return not any(
        marker in markers
        for marker in ("market", "sportsbook", "bookmaker", "posted", "observed", "executable")
    )


def _is_spread_like(pick: Mapping[str, Any]) -> bool:
    text = f"{_text(pick.get('market_type'))} {_text(pick.get('market'))} {_text(pick.get('pick'))}".lower()
    return bool(re.search(r"\b(?:spread|handicap|run\s*line|puck\s*line)\b", text))


def _attach_common(pick: dict[str, Any], game: dict[str, Any], captured_at: str) -> None:
    pick["market_odds_provider"] = f"espn_scoreboard:{game.get('provider') or 'unknown'}"
    pick["market_odds_captured_at"] = captured_at
    pick["market_updated_at"] = captured_at


def _replace_assumed_price(pick: dict[str, Any], real_odds: int, provider: str) -> None:
    if "model_assumed_odds" not in pick:
        pick["model_assumed_odds"] = pick.get("odds")
    pick["odds"] = real_odds
    pick["assumed_odds_replaced"] = True
    pick["pricing_type"] = "market"
    pick["odds_source"] = "posted_market"
    pick["price_source"] = provider
    pick["market_priced"] = True
    pick.pop("assumed_odds", None)


def _attach_pick(
    pick: dict[str, Any],
    game: dict[str, Any],
    *,
    bucket_key: str,
    captured_at: str,
) -> bool:
    markets = game.get("markets") or {}
    direction = _pick_direction(pick)
    # Replace assumed model prices, and keep refreshing an already replaced
    # price while the game remains pregame.
    replace = bucket_key in TEAM_MODEL_BUCKET_KEYS and (
        _looks_assumed(pick) or pick.get("assumed_odds_replaced") is True
    )
    provider = f"espn_scoreboard:{game.get('provider') or 'unknown'}"
    is_f5 = bucket_key in F5_BUCKET_KEYS

    if direction in {"over", "under"}:
        line = _pick_line(pick)
        if line is None:
            return False
        if is_f5:
            totals = (markets.get("f5_totals") or {}).get(round(line, 2))
            if not totals:
                return False
            over_price, under_price = totals["over"], totals["under"]
        else:
            total = markets.get("total")
            if not total or abs(total["line"] - line) > 0.01:
                return False
            over_price, under_price = total["over"], total["under"]
        _attach_common(pick, game, captured_at)
        pick["market_over_odds"] = over_price
        pick["market_under_odds"] = under_price
        pick["market_line"] = line
        selected = over_price if direction == "over" else under_price
        opposite = under_price if direction == "over" else over_price
        pick["selected_odds"] = selected
        pick["opposite_odds"] = opposite
        if replace:
            _replace_assumed_price(pick, selected, provider)
        return True

    side = _selected_side(pick, game)
    if side in {"home", "away"}:
        if _is_spread_like(pick):
            line = _pick_spread_line(pick)
            if line is None:
                return False
            if is_f5:
                run_line = markets.get("f5_run_line") or {}
                selected_price = (run_line.get(side) or {}).get(round(line, 2))
                opposite_price = (run_line.get("away" if side == "home" else "home") or {}).get(
                    round(-line, 2)
                )
                if selected_price is None or opposite_price is None:
                    return False
            else:
                spread = markets.get("spread")
                if not spread:
                    return False
                side_line = spread["homeLine"] if side == "home" else -spread["homeLine"]
                if abs(side_line - line) > 0.01:
                    return False
                selected_price = spread[side]
                opposite_price = spread["away" if side == "home" else "home"]
            _attach_common(pick, game, captured_at)
            pick["market_line"] = line
            pick["selected_odds"] = selected_price
            pick["opposite_odds"] = opposite_price
            if replace:
                _replace_assumed_price(pick, selected_price, provider)
            return True

        moneyline = markets.get("f5_moneyline") if is_f5 else markets.get("moneyline")
        if not moneyline:
            return False
        selected_price = moneyline.get(side)
        opposite_price = moneyline.get("away" if side == "home" else "home")
        draw_price = moneyline.get("draw")
        if selected_price is None or opposite_price is None:
            return False
        _attach_common(pick, game, captured_at)
        pick["market_home_odds"] = moneyline.get("home")
        pick["market_away_odds"] = moneyline.get("away")
        if draw_price is not None:
            # Three-way market: publish the exact no-vig for the selected
            # side so a two-way pair can never misstate fair value.
            pick["market_draw_odds"] = draw_price
            implied = [_implied(moneyline.get("home")), _implied(moneyline.get("away")), _implied(draw_price)]
            if all(value is not None for value in implied):
                hold = sum(implied)  # type: ignore[arg-type]
                selected_implied = _implied(selected_price)
                if hold and selected_implied is not None:
                    pick["market_no_vig_selected_probability"] = round(selected_implied / hold, 6)
        else:
            pick["selected_odds"] = selected_price
            pick["opposite_odds"] = opposite_price
        if replace:
            _replace_assumed_price(pick, selected_price, provider)
        return True

    return False


def apply_market_odds_to_payload(
    payload: dict[str, Any],
    book: dict[str, list[dict[str, Any]]] | None = None,
    *,
    fetch_json: FetchJson | None = None,
    now: datetime | None = None,
) -> dict[str, int]:
    """Attach pregame market odds to every matching pick in the payload."""

    date_iso = _text(payload.get("date"))
    models = payload.get("models")
    if not date_iso or not isinstance(models, dict):
        return {"attached": 0, "replacedAssumed": 0, "picksSeen": 0}
    if book is None:
        sports_present = {
            _text(pick.get("sport")).upper()
            for bucket in models.values()
            if isinstance(bucket, Mapping)
            for pick in bucket.get("picks") or []
            if isinstance(pick, Mapping)
        }
        wanted = [sport for sport in SPORT_LEAGUES if sport in sports_present]
        book = fetch_market_odds_for_date(date_iso, wanted, fetch_json) if wanted else {}

    captured_at = _now_iso(now)
    attached = replaced = seen = 0
    for bucket_key, bucket in models.items():
        if not isinstance(bucket, dict):
            continue
        for pick in bucket.get("picks") or []:
            if not isinstance(pick, dict):
                continue
            pick_date = _text(pick.get("date") or payload.get("date"))
            if pick_date != date_iso:
                continue
            sport = _text(pick.get("sport")).upper()
            games = book.get(sport) or []
            if not games:
                continue
            seen += 1
            game = _match_game(pick, games)
            if game is None:
                continue
            was_assumed = _looks_assumed(pick)
            if _attach_pick(pick, game, bucket_key=str(bucket_key), captured_at=captured_at):
                attached += 1
                if was_assumed and pick.get("assumed_odds_replaced") is True:
                    replaced += 1
    summary = {"attached": attached, "replacedAssumed": replaced, "picksSeen": seen}
    print(
        "[market-odds] "
        f"date={date_iso} picks_seen={seen} attached={attached} assumed_replaced={replaced}"
    )
    return summary
