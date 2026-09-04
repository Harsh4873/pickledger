"""CFB player-props candidate model: Action Network lines + ESPN gamelog projections.

WHY this module exists
----------------------
The framework's ML artifacts (see ``player_props/ml.py``) only cover NBA/WNBA/MLB.
CFB has no captured training data and, critically, adding a CFB market family to
``MARKET_FAMILY_NAMES`` would widen ``FEATURE_NAMES`` and INVALIDATE the existing
MLB/WNBA joblib artifacts (they were fit on the current feature vector). So CFB
deliberately does NOT register in ``ml.py`` and is NOT run through
``apply_ml_to_pick`` / ML ranking. It is a market-priced candidate model only:
projections come from ESPN gamelogs, the edge is measured against the devigged
(no-vig) two-sided Action Network price, and staking/decision logic is inherited
from ``player_props.schema.build_pick``.

Line source
-----------
Action Network keyless JSON. Plain ``requests`` is blocked, so every fetch uses
``curl_cffi`` with ``impersonate='chrome124'`` and a browser User-Agent + Referer.

  scoreboard: https://api.actionnetwork.com/web/v2/scoreboard/ncaaf?period=game
  props:      https://api.actionnetwork.com/web/v2/games/{game_id}/props

Projections
-----------
ESPN college-football player gamelogs, EWMA of recent games; sigma from the
player's own game-to-game variance. The Action Network player_id is NOT an ESPN
athlete id, so ``an_espn_bridge`` joins them by (normalized name + team abbr).

Correctness rules that MUST stay (each cost a real fabricated-edge incident)
----------------------------------------------------------------------------
1. Name matching is EXACT normalized match and REFUSES ambiguous matches. A loose
   last-name fallback silently matched the wrong player off a ~100-man roster and
   produced fabricated 40-50pp edges. See ``an_espn_bridge.resolve_athlete``.
2. Non-participation games are EXCLUDED from the stat series. A game where the
   player logged no attempts/targets/receptions is almost always inactive; scoring
   it as 0 drags every projection toward zero and manufactures a slate of bogus
   "Under" picks. See ``an_espn_bridge.stat_series(drop_zero_games=True)``.
3. Sanity guards DISCARD implausible rows rather than publishing them:
   a ``MIN_PROJECTION_SANITY`` floor per stat, and a ``MAX_CREDIBLE_EDGE_PP``
   ceiling (~12pp) beyond which a "row" is evidence of a data fault, not a bet.

Public entry point
-------------------
    generate_cfb_candidate_model(client, date_iso) -> dict

Mirrors ``generate_mlb_candidate_model`` in structure and returns
``{'ok','sport','date','games','picks','errors','method'}``.
"""

from __future__ import annotations

import math
import re
import statistics
from functools import lru_cache
from typing import Any

from curl_cffi import requests as _cr

from .schema import build_pick


# --------------------------------------------------------------------------- #
# Endpoints and fetch (browser impersonation is required; plain requests 403s) #
# --------------------------------------------------------------------------- #
AN_SCOREBOARD = "https://api.actionnetwork.com/web/v2/scoreboard/ncaaf"
AN_PROPS = "https://api.actionnetwork.com/web/v2/games/{game_id}/props"
ESPN_SCOREBOARD = (
    "http://site.api.espn.com/apis/site/v2/sports/football/"
    "college-football/scoreboard"
)
ESPN_ROSTER = (
    "http://site.api.espn.com/apis/site/v2/sports/football/"
    "college-football/teams/{team_id}/roster"
)
ESPN_GAMELOG = (
    "https://site.web.api.espn.com/apis/common/v3/sports/football/"
    "college-football/athletes/{athlete_id}/gamelog"
)
_AN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.actionnetwork.com/",
    "Accept": "application/json",
}
_ESPN_HEADERS = {"User-Agent": "PickLedgerCFB/1.0", "Accept": "application/json"}


# Action Network market -> (stat_key, stat_label, ESPN gamelog stat name).
# stat_key values are intentionally NOT in MARKET_FAMILY_NAMES; CFB is excluded
# from ML ranking (see module docstring).
MARKETS: dict[str, tuple[str, str, str]] = {
    "core_bet_type_9_passing_yards": ("passing_yards", "Passing Yards", "passingYards"),
    "core_bet_type_12_rushing_yards": ("rushing_yards", "Rushing Yards", "rushingYards"),
    "core_bet_type_16_receiving_yards": ("receiving_yards", "Receiving Yards", "receivingYards"),
    "core_bet_type_15_receptions": ("receptions", "Receptions", "receptions"),
}

EWMA_DECAY = 0.75
MIN_GAMES = 3

# Sanity ceiling. A devigged market price is sharp; a genuine model edge on a
# college prop lives in the low single digits. Anything past this is evidence of
# a data fault (wrong player matched, non-participation games averaged in), not a
# betting opportunity -- so it is DISCARDED, never published.
MAX_CREDIBLE_EDGE_PP = 12.0
MIN_PROJECTION_SANITY: dict[str, float] = {
    "passing_yards": 40.0,  # a QB drawing a posted passing line clears this
    "rushing_yards": 5.0,
    "receiving_yards": 5.0,
    "receptions": 0.5,
}


def _get_json(url: str, params: dict | None = None, headers: dict | None = None):
    resp = _cr.get(
        url,
        impersonate="chrome124",
        headers=headers or _AN_HEADERS,
        params=params,
        timeout=45,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{url} -> {resp.status_code}")
    return resp.json()


# --------------------------------------------------------------------------- #
# AN -> ESPN bridge (ported from scratch_cfb/an_espn_bridge.py)               #
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    """Normalize a player name for matching: drop punctuation, suffixes, case."""
    s = re.sub(r"[^a-z ]", "", str(name or "").lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _team_abbr_from_display(display_text: str) -> str:
    """'UNC - QB' -> 'UNC'."""
    return str(display_text or "").split("-")[0].strip().upper()


@lru_cache(maxsize=8)
def _espn_team_index(date_iso: str) -> tuple[tuple[str, str], ...]:
    """(abbreviation_upper, ESPN team id) pairs for the slate's teams."""
    data = _get_json(
        ESPN_SCOREBOARD,
        {"dates": date_iso.replace("-", ""), "groups": "80", "limit": 1000},
        headers=_ESPN_HEADERS,
    )
    index: dict[str, str] = {}
    for ev in data.get("events") or []:
        for comp in ev.get("competitions") or []:
            for c in comp.get("competitors") or []:
                t = c.get("team") or {}
                abbr = str(t.get("abbreviation") or "").upper()
                tid = str(t.get("id") or "")
                if abbr and tid:
                    index[abbr] = tid
    return tuple(index.items())


@lru_cache(maxsize=400)
def _espn_roster(team_id: str) -> tuple[tuple[str, str], ...]:
    """((normalized_name, athlete_id), ...) for one team."""
    try:
        data = _get_json(ESPN_ROSTER.format(team_id=team_id), headers=_ESPN_HEADERS)
    except Exception:
        return ()
    out: list[tuple[str, str]] = []
    ath = data.get("athletes") or []
    # ESPN returns either a flat list or position-grouped buckets.
    buckets = (
        ath
        if ath and isinstance(ath[0], dict) and "items" in ath[0]
        else [{"items": ath}]
    )
    for b in buckets:
        for a in b.get("items") or []:
            nm = a.get("displayName") or a.get("fullName")
            aid = str(a.get("id") or "")
            if nm and aid:
                out.append((_norm(nm), aid))
    return tuple(out)


def _resolve_athlete(full_name: str, team_abbr: str, date_iso: str) -> str | None:
    """AN name + team abbr -> ESPN athlete id.

    EXACT normalized match only. A loose last-name+initial fallback was tried and
    removed: on a ~100-man roster it silently resolves to the wrong player (e.g. a
    backup with the same surname), which produced fabricated 40pp "edges". An
    AMBIGUOUS match (two players normalize identically) is worse than no match --
    it is refused.
    """
    idx = dict(_espn_team_index(date_iso))
    tid = idx.get((team_abbr or "").upper())
    if not tid:
        return None
    want = _norm(full_name)
    if not want:
        return None
    matches = [aid for nm, aid in _espn_roster(tid) if nm == want]
    return matches[0] if len(matches) == 1 else None


@lru_cache(maxsize=2000)
def _gamelog(athlete_id: str, season: int):
    """(stat_names, per_game_rows) for one athlete-season."""
    try:
        data = _get_json(
            ESPN_GAMELOG.format(athlete_id=athlete_id),
            {"season": season},
            headers=_ESPN_HEADERS,
        )
    except Exception:
        return (), ()
    names = tuple(str(n) for n in (data.get("names") or []))
    rows: list[tuple[float, ...]] = []
    for st in data.get("seasonTypes") or []:
        for cat in st.get("categories") or []:
            for ev in cat.get("events") or []:
                vals = []
                for v in ev.get("stats") or []:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        vals.append(float("nan"))
                if vals:
                    rows.append(tuple(vals))
    return names, tuple(rows)


def _stat_series(
    athlete_id: str,
    stat_name: str,
    season: int,
    *,
    drop_zero_games: bool = True,
) -> list[float]:
    """Per-game values for one stat.

    ``drop_zero_games`` excludes games where the player recorded nothing for the
    relevant volume stat. Those rows are usually non-participation (inactive,
    injured, or a backup who never saw the field), and averaging them in pulls
    every projection toward zero -- which manufactured a slate of bogus "Under"
    picks with ~50pp edges. Keeping this exclusion is non-negotiable.
    """
    names, rows = _gamelog(athlete_id, season)
    if not names or stat_name not in names:
        return []
    i = names.index(stat_name)
    # A snap-count proxy: no attempts/targets/receptions -> the player did not play.
    volume_keys = ("passingAttempts", "rushingAttempts", "receptions")
    vol_idx = [names.index(k) for k in volume_keys if k in names]
    out: list[float] = []
    for r in rows:
        if i >= len(r) or r[i] != r[i]:  # missing / NaN
            continue
        if drop_zero_games and vol_idx:
            played = any(v < len(r) and r[v] == r[v] and r[v] > 0 for v in vol_idx)
            if not played:
                continue
        out.append(r[i])
    return out


# --------------------------------------------------------------------------- #
# Projection                                                                  #
# --------------------------------------------------------------------------- #
def _project(vals: list[float]) -> tuple[float, float] | None:
    """EWMA mean (newest game weighted most) + game-to-game sigma."""
    if len(vals) < MIN_GAMES:
        return None
    w = 1.0
    num = den = 0.0
    for v in reversed(vals):  # newest first gets the biggest weight
        num += v * w
        den += w
        w *= EWMA_DECAY
    mu = num / den
    sd = statistics.pstdev(vals) if len(vals) > 1 else max(1.0, mu * 0.35)
    return mu, max(1.0, sd)


def _normal_over(mu: float, line: float, sigma: float) -> float:
    z = (mu - line) / (sigma * math.sqrt(2.0))
    return max(0.01, min(0.99, 0.5 * (1.0 + math.erf(z))))


def _odds_int(value: Any) -> int | None:
    try:
        o = float(value)
    except (TypeError, ValueError):
        return None
    if o == 0 or -100.0 < o < 100.0:
        return None
    return int(round(o))


# --------------------------------------------------------------------------- #
# Slate + props                                                               #
# --------------------------------------------------------------------------- #
def _slate_games(date_iso: str) -> list[dict[str, Any]]:
    data = _get_json(AN_SCOREBOARD, {"period": "game"})
    out: list[dict[str, Any]] = []
    for g in data.get("games") or []:
        if str(g.get("start_time") or "")[:10] != date_iso:
            continue
        teams = {t.get("id"): t for t in (g.get("teams") or [])}
        out.append(
            {
                "id": g.get("id"),
                "start_time": str(g.get("start_time") or ""),
                "teams": teams,
                "away_id": g.get("away_team_id"),
                "home_id": g.get("home_team_id"),
            }
        )
    return out


def _game_props(
    game: dict[str, Any],
    date_iso: str,
    season: int,
    diagnostics: dict[str, int],
) -> list[dict[str, Any]]:
    pdata = _get_json(AN_PROPS.format(game_id=game["id"]))
    players = pdata.get("players") or {}
    player_props = pdata.get("player_props") or {}
    team_name = {
        tid: (t.get("full_name") or t.get("display_name") or "")
        for tid, t in (game["teams"] or {}).items()
    }
    away = team_name.get(game["away_id"], "")
    home = team_name.get(game["home_id"], "")

    picks: list[dict[str, Any]] = []
    for market, (stat_key, label, espn_stat) in MARKETS.items():
        for entry in player_props.get(market) or []:
            lines = entry.get("lines") or {}
            # Prefer a book that quotes BOTH sides so the price can be devigged.
            chosen = None
            for book_id, side_rows in lines.items():
                rows = side_rows if isinstance(side_rows, list) else [side_rows]
                over = next((r for r in rows if str(r.get("side")) == "over"), None)
                under = next((r for r in rows if str(r.get("side")) == "under"), None)
                if over and under:
                    chosen = (book_id, over, under)
                    break
            if not chosen:
                continue
            book_id, over, under = chosen
            line = over.get("value")
            pid = str(over.get("player_id") or "")
            over_odds = _odds_int(over.get("odds"))
            under_odds = _odds_int(under.get("odds"))
            if line is None or not pid or over_odds is None or under_odds is None:
                continue
            diagnostics["markets_considered"] += 1

            info = players.get(pid) or {}
            pname = info.get("full_name") or info.get("display_text") or f"player {pid}"
            abbr = _team_abbr_from_display(info.get("display_text"))
            team = abbr or ""
            opponent = home if team and _norm(team) == _norm(away) else away

            # AN player_id is an Action Network id -- bridge to ESPN by name+team.
            athlete_id = _resolve_athlete(pname, abbr, date_iso)
            if not athlete_id:
                diagnostics["skipped_no_match"] += 1
                continue

            # Early season: prefer last season's completed sample, fall back to this one.
            series = _stat_series(athlete_id, espn_stat, season - 1)
            if len(series) < MIN_GAMES:
                series = _stat_series(athlete_id, espn_stat, season) or series
            proj = _project(series)
            if proj is None:
                diagnostics["skipped_no_projection"] += 1
                continue
            mu, sd = proj

            # Sanity floor: reject data faults, do not publish them.
            if mu < MIN_PROJECTION_SANITY.get(stat_key, 0.0):
                diagnostics["rejected_implausible"] += 1
                continue

            line = float(line)
            p_over = _normal_over(mu, line, sd)

            # Pick the side with the larger model edge vs its own market implied odds.
            def _implied(o: int) -> float:
                return 100.0 / (o + 100.0) if o > 0 else abs(o) / (abs(o) + 100.0)

            candidates = [
                ("Over", p_over, over_odds),
                ("Under", 1.0 - p_over, under_odds),
            ]
            selection, probability, odds = max(
                candidates, key=lambda c: c[1] - _implied(c[2])
            )

            # Ceiling guard on the no-vig edge. build_pick devigs the pair we pass
            # in ``extra`` (market_over_odds/market_under_odds) via schema; recompute
            # the same no-vig baseline here so an implausible row is discarded
            # BEFORE it can be emitted as a pick.
            po, pu = _implied(over_odds), _implied(under_odds)
            hold = po + pu
            fair_over = po / hold if hold > 0 else None
            if fair_over is None:
                continue
            fair_side = fair_over if selection == "Over" else 1.0 - fair_over
            edge_pp = (probability - fair_side) * 100.0
            if edge_pp > MAX_CREDIBLE_EDGE_PP:
                diagnostics["rejected_implausible"] += 1
                continue

            reason = (
                f"{pname} projects for {mu:.1f} {label.lower()} "
                f"(EWMA of {len(series)} played games, sigma {sd:.1f}) versus the "
                f"posted {line:g} line; edge measured against the no-vig fair price."
            )
            key_factors = [
                f"EWMA projection {mu:.1f} over {len(series)} played games",
                f"Game-to-game sigma {sd:.1f}",
                f"Posted {line:g} {label.lower()} line at {odds:+d}",
                f"Two-sided book {book_id}: over {over_odds:+d} / under {under_odds:+d}",
                "Non-participation games excluded; exact-name ESPN match required",
            ]
            pick = build_pick(
                sport="CFB",
                date_iso=date_iso,
                game_id=str(game["id"]),
                away_team=away,
                home_team=home,
                start_time=game["start_time"],
                player_id=pid,
                player_name=pname,
                team=team,
                opponent=opponent,
                stat_key=stat_key,
                stat_label=label,
                selection=selection,
                line=line,
                projection=mu,
                probability=probability,
                reason=reason,
                key_factors=key_factors,
                odds=odds,
                extra={
                    "game_id": str(game["id"]),
                    "player_id": pid,
                    "espn_athlete_id": athlete_id,
                    "sigma": round(sd, 2),
                    "games_used": len(series),
                    "prop_role": "cfb_player",
                    # Devig pair: schema.market_fair_probability uses these to raise
                    # the edge baseline to the no-vig fair probability.
                    "market_over_odds": over_odds,
                    "market_under_odds": under_odds,
                    "book_id": book_id,
                    "pricing_type": "market",
                    "line_source": "action_network",
                    "odds_source": "action_network",
                    "market_priced": True,
                    "actionability": "market_priced",
                },
            )
            picks.append(pick)
    return picks


# --------------------------------------------------------------------------- #
# Public entry point                                                          #
# --------------------------------------------------------------------------- #
def generate_cfb_candidate_model(client: Any, date_iso: str) -> dict[str, Any]:
    """Generate the CFB player-prop candidate pool for a slate date.

    ``client`` is accepted for signature parity with the other candidate models
    (``generate_mlb_candidate_model`` etc.) but is unused: CFB's line and stat
    sources (Action Network, ESPN) are fetched directly via ``curl_cffi`` because
    they need browser impersonation and are not part of the shared API client.

    Returns ``{'ok','sport','date','games','picks','errors','method'}``.

    NOTE: CFB is intentionally NOT run through ML ranking. Registering a CFB market
    family in ``ml.py`` would widen ``FEATURE_NAMES`` and break the existing
    MLB/WNBA joblib artifacts, so these are market-priced candidates only.
    """
    errors: list[str] = []
    diagnostics = {
        "markets_considered": 0,
        "skipped_no_match": 0,
        "skipped_no_projection": 0,
        "rejected_implausible": 0,
    }
    season = int(date_iso[:4])

    try:
        games = _slate_games(date_iso)
    except Exception as exc:
        return {
            "ok": False,
            "sport": "CFB",
            "date": date_iso,
            "games": 0,
            "picks": [],
            "errors": [f"Action Network scoreboard failed: {exc}"],
        }

    if not games:
        return {
            "ok": True,
            "sport": "CFB",
            "date": date_iso,
            "games": 0,
            "picks": [],
            "errors": [],
            "note": "No CFB games scheduled; empty slate is healthy.",
            "method": (
                "Action Network two-sided lines devigged, ESPN gamelog EWMA "
                "projection; exact-name match, non-participation games excluded"
            ),
        }

    picks: list[dict[str, Any]] = []
    for game in games:
        try:
            picks.extend(_game_props(game, date_iso, season, diagnostics))
        except Exception as exc:
            errors.append(f"game {game.get('id')}: {exc}")

    picks.sort(key=lambda p: (p.get("edge") is None, -(p.get("edge") or 0.0)))
    return {
        "ok": True,
        "sport": "CFB",
        "date": date_iso,
        "games": len(games),
        "picks": picks,
        "errors": errors,
        "diagnostics": diagnostics,
        "method": (
            "Action Network two-sided lines devigged to a no-vig fair price, "
            "ESPN college-football gamelog EWMA projection with per-player sigma; "
            "exact-name AN->ESPN match, non-participation games excluded, "
            "implausible-edge rows discarded. No ML ranking (CFB excluded from "
            "ml.py to protect MLB/WNBA artifacts)."
        ),
    }
