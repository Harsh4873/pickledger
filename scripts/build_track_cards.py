#!/usr/bin/env python3
"""Publish a Track research block onto an already built parlay payload.

This is not a PickLedger model and it does not pass the v6 gates. Legs are
in-house team rows and player-prop rows that already have a posted American
price and a start time still in the future relative to the parlay rebuild
clock. The 90% screen is the vigged implied probability of that posted price,
not a win probability. Combined decimals are the product of those comparison
prices and are estimates, not an executable app parlay price. Units are 0.

Dates before TRACK_RESEARCH_CUTOVER_DATE are not this module's job. The parlay
rebuild skips the attachment so auto-grade does not inject the block into
older files.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Iterable

from scripts.build_parlay_cards import (
    _clean_text,
    _game_key,
    _game_label,
    _int_number,
    _iter_model_records,
    _market_family,
    _pick_date,
    _pick_text,
    _player_key,
    american_to_decimal,
    canonical_game_key,
    canonical_side_key,
)

TRACK_RESEARCH_CUTOVER_DATE = "2026-10-12"
NINETY_SCREEN = 0.90
FAVORITE_DECIMAL_BAND = (1.2, 1.4)
MULTIPLIER_DECIMAL_BAND = (2.0, 3.0)
# American -175 is the chalk line. Shorter prices are -175 or more negative.
STREAK_CHALK_ODDS = -175
QUOTE_BASIS = "comparison"
COMBINED_PRICE_LABEL = "estimate"

SHAPE_IDS = (
    "favorite_parlay",
    "multiplier_parlay",
    "priced_side",
    "prop_pair",
    "streak_candidate",
)

AUDIENCES = {
    "favorite_parlay": "Polymarket US and Rebet research",
    "multiplier_parlay": "Fliff research",
    "priced_side": "Novig cash research",
    "prop_pair": "Underdog research",
    "streak_candidate": "Streaks research",
}

SHAPE_NOTES = {
    "streak_candidate": "Not a daily requirement.",
}

_SCRAPED_PREFIXES = (
    "scores24_",
    "sportytrader_",
    "sportsgambler_",
    "forebet_",
    "tennistonic_",
    "covers_",
)

_PRESEASON_SEASON_TYPES = frozenset({
    "pre",
    "preseason",
    "exhibition",
    "spring_training",
    "spring training",
    "spring",
})

_ASSUMED_TOKENS = (
    "assumed",
    "synthetic",
    "placeholder",
    "unpriced",
    "model_price",
    "model price",
)

_START_FIELDS = ("start_time", "game_start_time", "event_start_time")


@dataclass(frozen=True)
class TrackLeg:
    public_id: str
    identity: str
    pick: str
    sport: str
    game: str
    game_key: str
    market: str
    player: str
    player_key: str
    odds: int
    decimal_odds: float
    implied: float
    quote_source: str
    start_time: str
    is_prop: bool


def parse_rebuild_clock(value: Any) -> datetime:
    """Return the parlay payload clock, or the current UTC time if it is blank."""
    parsed = _parse_aware(value)
    if parsed is None:
        return datetime.now(timezone.utc)
    return parsed


def build_track_research(
    date_iso: str,
    team_payload: dict[str, Any] | None,
    prop_payload: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the research block. An empty shape list is a valid publish."""
    clock = _as_utc(now) if now is not None else datetime.now(timezone.utc)
    legs = _collect_legs(date_iso, team_payload, prop_payload, clock)
    sides = [leg for leg in legs if not leg.is_prop]
    props = [leg for leg in legs if leg.is_prop]
    favorite = _select_favorite(legs)
    multiplier = _select_multiplier(legs)
    pair = _select_prop_pair(props)
    shapes = {
        "favorite_parlay": [_parlay_payload("favorite_parlay", favorite)] if favorite else [],
        "multiplier_parlay": [_parlay_payload("multiplier_parlay", multiplier)] if multiplier else [],
        "priced_side": [_leg_payload(leg) for leg in _sorted_legs(sides)],
        "prop_pair": [_parlay_payload("prop_pair", pair)] if pair else [],
        "streak_candidate": [_leg_payload(leg) for leg in _sorted_legs(_streak_props(props))],
    }
    return {
        "label": "research",
        "quoteBasis": QUOTE_BASIS,
        "units": 0,
        "notPlacedTickets": True,
        "notVerifiedEdge": True,
        "screen": (
            "The 90% screen is the vigged implied probability of the posted price, "
            "not a true win probability."
        ),
        "note": (
            "Research only. These are not placed tickets and not verified edge. "
            "Combined prices are estimates from comparison quotes, not an executable app parlay price."
        ),
        "audiences": dict(AUDIENCES),
        "notes": dict(SHAPE_NOTES),
        "shapes": shapes,
    }


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_aware(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def vigged_implied_probability(odds: int) -> float:
    """Vigged implied probability of a posted American price.

    This is the price screen, not a model win probability. -900 is exactly 90%.
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    raise ValueError("American odds cannot be zero")


def passes_ninety_screen(odds: int) -> bool:
    return vigged_implied_probability(odds) >= NINETY_SCREEN


def _in_band(product: float, band: tuple[float, float]) -> bool:
    low, high = band
    return low - 1e-9 <= product <= high + 1e-9


def _product(combo: Iterable[TrackLeg]) -> float:
    return math.prod(leg.decimal_odds for leg in combo)


def _collect_legs(
    date_iso: str,
    team_payload: dict[str, Any] | None,
    prop_payload: dict[str, Any] | None,
    clock: datetime,
) -> list[TrackLeg]:
    rows: list[tuple[str, str, dict[str, Any], bool]] = []
    if team_payload:
        rows.extend(
            (source_key, fallback_date, pick, False)
            for source_key, _source, fallback_date, pick in _iter_model_records(team_payload, player_props=False)
        )
    if prop_payload:
        rows.extend(
            (source_key, fallback_date, pick, True)
            for source_key, _source, fallback_date, pick in _iter_model_records(prop_payload, player_props=True)
        )

    legs: list[TrackLeg] = []
    for source_key, fallback_date, pick, from_prop_cache in rows:
        if _pick_date(pick, fallback_date) != date_iso:
            continue
        if _exclusion(source_key, pick) is not None:
            continue
        pick_text = _pick_text(pick)
        if not pick_text:
            continue
        start = _future_start(pick, clock)
        if start is None:
            continue
        odds = _posted_odds(pick)
        if odds is None:
            continue
        is_prop = _is_prop_row(pick, from_prop_cache)
        player = _clean_text(pick.get("player") or pick.get("player_name"))
        if is_prop and not player:
            continue
        sport = _clean_text(pick.get("sport") or pick.get("league") or "OTHER").upper()
        game = _game_label(pick)
        raw_game_key = _game_key(pick, sport, date_iso, pick_text)
        game_key = canonical_game_key(sport, game or raw_game_key, raw_game_key, date_iso)
        player_key = _player_key(pick, pick_text)
        market = _public_market(pick)
        quote_source = _quote_source(pick)
        implied = vigged_implied_probability(odds)
        decimal_odds = american_to_decimal(odds)
        if is_prop:
            identity = _prop_identity(game_key, player_key, market, pick, pick_text)
        else:
            identity = canonical_side_key(
                mode="team",
                pick_text=pick_text,
                sport=sport,
                game_label=game,
                game_key=raw_game_key,
                date_iso=date_iso,
                player_key=player_key,
                market_family=_market_family(pick),
            )
        public_id = "|".join((sport, identity, str(odds), quote_source))
        legs.append(
            TrackLeg(
                public_id=public_id,
                identity=identity,
                pick=pick_text,
                sport=sport,
                game=game,
                game_key=game_key,
                market=market,
                player=player,
                player_key=player_key,
                odds=odds,
                decimal_odds=decimal_odds,
                implied=implied,
                quote_source=quote_source,
                start_time=start,
                is_prop=is_prop,
            )
        )
    return _dedupe(legs)


def _exclusion(source_key: str, pick: dict[str, Any]) -> str | None:
    if _is_scraped(source_key, pick):
        return "scraped"
    if _is_tennis(source_key, pick):
        return "tennis"
    if _is_mlb_inning(source_key, pick):
        return "mlb_inning"
    if _is_preseason(pick):
        return "preseason"
    if _is_wnba(source_key, pick):
        return "wnba"
    return None


def _is_scraped(source_key: str, pick: dict[str, Any]) -> bool:
    key = source_key.lower()
    if any(key.startswith(prefix) for prefix in _SCRAPED_PREFIXES):
        return True
    return pick.get("scraped") is True or pick.get("scraped_tip_demoted") is True


def _is_tennis(source_key: str, pick: dict[str, Any]) -> bool:
    blob = " ".join((
        source_key,
        _clean_text(pick.get("sport")),
        _clean_text(pick.get("league")),
        _clean_text(pick.get("market")),
        _clean_text(pick.get("market_type")),
    )).lower()
    return "tennis" in blob


def _is_mlb_inning(source_key: str, pick: dict[str, Any]) -> bool:
    key = source_key.lower()
    if key == "mlb_inning" or key.startswith("mlb_inning"):
        return True
    market = _clean_text(pick.get("market") or pick.get("market_type")).lower().replace(" ", "_")
    return market in {"mlb_inning", "no_run_inning"}


def _is_preseason(pick: dict[str, Any]) -> bool:
    if pick.get("preseason") is True:
        return True
    for key in ("season_type", "seasonType", "game_type", "gameType"):
        if _clean_text(pick.get(key)).lower() in _PRESEASON_SEASON_TYPES:
            return True
    return False


def _is_wnba(source_key: str, pick: dict[str, Any]) -> bool:
    key = source_key.lower()
    if key == "wnba" or key.startswith("wnba") or "_wnba" in key:
        return True
    sport = _clean_text(pick.get("sport") or pick.get("league")).upper()
    return "WNBA" in sport


def _is_prop_row(pick: dict[str, Any], from_prop_cache: bool) -> bool:
    if from_prop_cache:
        return True
    if _clean_text(pick.get("scope")).lower() == "player":
        return True
    market = _clean_text(pick.get("market") or pick.get("market_type")).lower().replace(" ", "_")
    return market in {"player_props", "player_prop"}


def _public_market(pick: dict[str, Any]) -> str:
    for key in ("stat_label", "stat_key", "market_type", "market", "bet_type"):
        value = _clean_text(pick.get(key))
        if value and value.lower() not in {"player_props", "player_prop", "market"}:
            return value
    return _clean_text(pick.get("market") or pick.get("market_type") or "market")


def _prop_identity(game_key: str, player_key: str, market: str, pick: dict[str, Any], pick_text: str) -> str:
    line = _clean_text(pick.get("line") or pick.get("market_line"))
    direction = _clean_text(pick.get("direction") or pick.get("selection") or pick_text)
    return "|".join((game_key, player_key, market.lower(), line, direction.lower()))


def _future_start(pick: dict[str, Any], clock: datetime) -> str | None:
    for field in _START_FIELDS:
        parsed = _parse_aware(pick.get(field))
        if parsed is None:
            continue
        if parsed > clock:
            return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")
        return None
    return None


def _is_assumed_price(pick: dict[str, Any]) -> bool:
    odds_source = _clean_text(pick.get("odds_source")).lower()
    if odds_source == "posted_market":
        return False
    markers = " ".join(
        _clean_text(pick.get(key)).lower()
        for key in ("pricing_type", "odds_source", "line_source", "price_source")
    )
    if any(token in markers for token in _ASSUMED_TOKENS):
        return True
    return _clean_text(pick.get("pricing_type")).lower() in {"assumed", "unpriced", "model"}


def _has_posted_marker(pick: dict[str, Any]) -> bool:
    if pick.get("market_priced") is True:
        return True
    odds_source = _clean_text(pick.get("odds_source")).lower()
    if odds_source in {"posted_market", "draftkings", "fanduel", "betmgm", "caesars"}:
        return True
    if _quote_source(pick):
        return True
    pricing = _clean_text(pick.get("pricing_type")).lower()
    return pricing in {"market", "posted", "posted_market"}


def _quote_source(pick: dict[str, Any]) -> str:
    for key in ("market_odds_provider", "price_source", "market_source"):
        value = _clean_text(pick.get(key))
        if value:
            return value
    snapshot = pick.get("pregame_snapshot")
    if isinstance(snapshot, dict):
        for key in ("market_odds_provider", "price_source", "market_source"):
            value = _clean_text(snapshot.get(key))
            if value:
                return value
    odds_source = _clean_text(pick.get("odds_source"))
    if odds_source:
        return odds_source
    if isinstance(snapshot, dict):
        return _clean_text(snapshot.get("odds_source"))
    return ""


def _posted_odds(pick: dict[str, Any]) -> int | None:
    if pick.get("market_priced") is False:
        return None
    if _is_assumed_price(pick):
        return None
    odds: int | None = None
    for key in ("odds", "american_odds", "price"):
        if pick.get(key) in (None, ""):
            continue
        odds = _int_number(pick.get(key))
        if odds is not None:
            break
    if odds is None or odds == 0:
        return None
    if not _quote_source(pick):
        return None
    if not _has_posted_marker(pick):
        return None
    return odds


def _quote_rank(source: str) -> tuple[int, int, str]:
    text = source.lower()
    specific = 1 if any(token in text for token in ("draftkings", "espn", "fanduel", "betmgm", "caesars")) else 0
    generic = 0 if text in {"posted_market", "market", "odds"} else 1
    return (specific, generic, source)


def _dedupe(legs: list[TrackLeg]) -> list[TrackLeg]:
    chosen: dict[str, TrackLeg] = {}
    for leg in sorted(legs, key=lambda item: item.public_id):
        current = chosen.get(leg.identity)
        if current is None or _quote_rank(leg.quote_source) > _quote_rank(current.quote_source):
            chosen[leg.identity] = leg
    return list(chosen.values())


def _sorted_legs(legs: Iterable[TrackLeg]) -> list[TrackLeg]:
    return sorted(legs, key=lambda leg: (leg.start_time, leg.sport, leg.game, leg.pick, leg.public_id))


def _independent(combo: tuple[TrackLeg, ...]) -> bool:
    if len(combo) < 2:
        return False
    if len({leg.public_id for leg in combo}) != len(combo):
        return False
    if len({leg.game_key for leg in combo}) != len(combo):
        return False
    players = [leg.player_key for leg in combo if leg.player_key and not leg.player_key.startswith("no-player:")]
    return len(players) == len(set(players))


def _combos(
    pool: list[TrackLeg],
    *,
    band: tuple[float, float],
    favorite_screen: bool,
) -> list[tuple[TrackLeg, ...]]:
    if len(pool) < 2:
        return []
    found: dict[tuple[str, ...], tuple[TrackLeg, ...]] = {}
    ordered = sorted(pool, key=lambda leg: leg.public_id)
    longer_pool = ordered
    if len(ordered) > 12:
        longer_pool = sorted(ordered, key=lambda leg: (leg.implied, leg.public_id))[:12]
    max_size = min(4, len(ordered))
    for size in range(2, max_size + 1):
        source = ordered if size == 2 else longer_pool
        if len(source) < size:
            continue
        for combo in combinations(source, size):
            if favorite_screen and any(not passes_ninety_screen(leg.odds) for leg in combo):
                continue
            if not _independent(combo):
                continue
            if not _in_band(_product(combo), band):
                continue
            found[tuple(sorted(leg.public_id for leg in combo))] = combo
    return list(found.values())


def _select_favorite(legs: list[TrackLeg]) -> tuple[TrackLeg, ...] | None:
    pool = [leg for leg in legs if passes_ninety_screen(leg.odds)]
    candidates = _combos(pool, band=FAVORITE_DECIMAL_BAND, favorite_screen=True)
    if not candidates:
        return None
    candidates.sort(key=lambda combo: (
        -round(_gap(combo), 8),
        len(combo),
        tuple(sorted(leg.public_id for leg in combo)),
    ))
    return tuple(sorted(candidates[0], key=lambda leg: leg.public_id))


def _gap(combo: tuple[TrackLeg, ...]) -> float:
    implieds = [leg.implied for leg in combo]
    return max(implieds) - min(implieds)


def _select_multiplier(legs: list[TrackLeg]) -> tuple[TrackLeg, ...] | None:
    candidates = _combos(legs, band=MULTIPLIER_DECIMAL_BAND, favorite_screen=False)
    if not candidates:
        return None
    candidates.sort(key=lambda combo: (
        abs(_product(combo) - 2.5),
        len(combo),
        tuple(sorted(leg.public_id for leg in combo)),
    ))
    return tuple(sorted(candidates[0], key=lambda leg: leg.public_id))


def _select_prop_pair(props: list[TrackLeg]) -> tuple[TrackLeg, ...] | None:
    ordered = sorted(props, key=lambda leg: leg.public_id)
    for combo in combinations(ordered, 2):
        if not _independent(combo):
            continue
        if not all(leg.is_prop and leg.player for leg in combo):
            continue
        return tuple(sorted(combo, key=lambda leg: leg.public_id))
    return None


def _streak_props(props: Iterable[TrackLeg]) -> list[TrackLeg]:
    return [leg for leg in props if leg.odds > STREAK_CHALK_ODDS]


def _leg_payload(leg: TrackLeg) -> dict[str, Any]:
    # Posted comparison price only. Model probabilities are not copied.
    return {
        "pick": leg.pick,
        "sport": leg.sport,
        "game": leg.game,
        "market": leg.market,
        "player": leg.player,
        "oddsAmerican": leg.odds,
        "decimalOdds": round(leg.decimal_odds, 4),
        "viggedImpliedProbability": round(leg.implied, 4),
        "quoteSource": leg.quote_source,
        "quoteBasis": QUOTE_BASIS,
        "startTime": leg.start_time,
        "units": 0,
    }


def _parlay_payload(shape: str, legs: tuple[TrackLeg, ...]) -> dict[str, Any]:
    return {
        "id": shape,
        "shape": shape,
        "quoteBasis": QUOTE_BASIS,
        "units": 0,
        "combinedDecimalOdds": round(_product(legs), 4),
        "combinedPriceLabel": COMBINED_PRICE_LABEL,
        "executableParlayPrice": False,
        "audience": AUDIENCES[shape],
        "legs": [_leg_payload(leg) for leg in legs],
    }
