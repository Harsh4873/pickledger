#!/usr/bin/env python3
"""Scrape Covers.com picks by official slate matchup.

Covers publishes four pick "engines"; each becomes its own feed bucket so the
board ranks them separately and nothing is lumped together:

  experts    per-author cards on matchup detail pages. Source label is
             "Covers · <Author>" so every analyst auto-registers as an
             independently ranked source; the author name, profile URL, and
             role ride along on every pick row.
  computer   the model's chosen side per game market (moneyline/total/spread)
             from the matchup page projections board (MLB only — Covers runs
             no WNBA computer board).
  consensus  the community split on each matchup page ("62% picking St.
             Louis" + vote counts). Majority side publishes as a moneyline
             pick once it clears the vote-volume and majority thresholds.
  props      the league player-props board ("THE BAT X" for MLB; no WNBA
             props page exists). Rows are Covers-curated by EV/stars.

Scope split: player-vs-team is a property of each pick's market, not of the
engine — every row carries `scope` ("team" | "player") and player-scoped rows
also carry `external_player_feed: True`, which is what the frontend player
mode keys on. An expert can therefore appear on both sides of the toggle.

DOM notes (verified live 2026-07-20):
  - Both expert and computer picks share `div.pick-cards-expert-component`;
    a GUID card id means human expert, a numeric id means computer pick.
    `data-pick-types` = market label, `data-pick-teams` = full team name.
  - League pages attach STALE cards (prior meetings, sometimes weeks old) to
    upcoming matchup groups, so every card's `data-tracking` game datetime is
    validated against the official kickoff before a pick is accepted.
  - Projections rows `tr.game-projections-container` are emitted twice
    (desktop + mobile) — always dedupe on `data-id`. `data-market-id` 0 is a
    game market; anything else is a player-prop market.
  - Total/spread best-odds chips can quote a different line than the board
    row; odds are only taken from a book entry at the board's own line.
  - Consensus away/home percentages come from `.pick-team-away` /
    `.pick-team-home` progress values.

Rows are matched against the official ESPN slate (same identity convention as
the Scores24/Forebet scrapers) with an 8h kickoff window, publish pregame
only, and grade through the standard ESPN auto-grade path. Player props emit
the structured fields (player_name/stat_key/selection/line) that
pickgrader_server.parse_player_prop_pick consumes directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scrapers.scores24_scraper import (  # noqa: E402
    CLOUDFLARE_SIGNALS,
    _team_matches,
    fetch_daily_matchups,
)


BASE_URL = "https://www.covers.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
EASTERN = ZoneInfo("America/New_York")
MODEL_CACHE_DIR = REPO_ROOT / "data" / "model_cache"

SPORT_CONFIG = {
    "mlb": {
        "espn_sport": "baseball",
        "espn_league": "mlb",
        "league_url": f"{BASE_URL}/picks/mlb",
        "odds_url": f"{BASE_URL}/sport/baseball/mlb/odds",
        "props_url": f"{BASE_URL}/sport/baseball/mlb/player-props",
        "label": "MLB",
        "cache_keys": ("mlb_first_five", "mlb_inning", "mlb_new"),
        "has_computer_board": True,
        "has_props_board": True,
        "props_model": "BAT X",
    },
    "wnba": {
        "espn_sport": "basketball",
        "espn_league": "wnba",
        "league_url": f"{BASE_URL}/picks/wnba",
        "odds_url": f"{BASE_URL}/sport/basketball/wnba/odds",
        "props_url": "",
        "label": "WNBA",
        "cache_keys": ("wnba",),
        "has_computer_board": False,
        "has_props_board": False,
        "props_model": "",
    },
}
ENGINES = ("experts", "computer", "consensus", "props")
ENGINE_SOURCES = {
    "computer": {"mlb": "Covers Computer MLB", "wnba": "Covers Computer WNBA"},
    "consensus": {"mlb": "Covers Consensus MLB", "wnba": "Covers Consensus WNBA"},
    "props": {"mlb": "Covers Props (BAT X)"},
}
EXPERT_SOURCE_PREFIX = "Covers · "
EXPERT_SOURCE_FALLBACK = "Covers Expert"

# Consensus rows publish only on a clear, well-sampled majority.
CONSENSUS_MIN_PCT = 55
CONSENSUS_MIN_VOTES = 50
# Props board rows are already Covers-curated (~stars 3-4, EV 10%+); keep the
# high-conviction tier so daily volume stays near the in-house prop cadence.
PROPS_MIN_STARS = 4

GUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
MATCHUP_HREF_RE = re.compile(r"/sport/[a-z-]+/[a-z0-9-]+/matchup/(\d+)/picks")
SEL_ODDS_RE = re.compile(r"\(([+-]\d{3,5})\)\s*$")
SEL_LINE_RE = re.compile(r"\b([ou])(\d+(?:\.\d+)?)\b")
SEL_SPREAD_RE = re.compile(r"([+-]\d+(?:\.\d+)?)\s*(?:\([+-]\d+\)\s*)?$")
AMERICAN_ODDS_RE = re.compile(r"^[+-]\d{3,5}$")
# "STL vs LAA, Mon, Jul 20 • 10:10 PM ET" (data-tracking "text")
TRACKING_RE = re.compile(
    r"^([A-Za-z]{2,4})\s+vs\s+([A-Za-z]{2,4}),\s+\w{3},\s+(\w{3})\s+(\d{1,2})\s*•\s*(\d{1,2}):(\d{2})\s*(AM|PM)\s*ET"
)
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
KICKOFF_MATCH_WINDOW_SECONDS = 8 * 3600

# Covers market labels / prop-board slugs → the grader's canonical stat keys
# (pickgrader_server.parse_player_prop_pick stat_aliases). Markets missing
# from these maps are skipped and reported, never guessed.
PROP_STAT_BY_LABEL = {
    "total bases": "total_bases",
    "total hits": "hits",
    "total home runs": "home_runs",
    "total rbis": "rbis",
    "total runs": "runs",
    "total singles": "singles",
    "total doubles": "doubles",
    "total triples": "triples",
    "total stolen bases": "stolen_bases",
    "total walks": "batter_walks",
    "hits runs rbis": "hits_runs_rbis",
    "strikeouts thrown": "strikeouts",
    "earned runs allowed": "pitcher_earned_runs_allowed",
    "outs recorded": "pitcher_outs_recorded",
    "hits allowed": "pitcher_hits_allowed",
    "walks allowed": "pitcher_walks_allowed",
    "total points": "points",
    "total rebounds": "rebounds",
    "total assists": "assists",
    "total steals": "steals",
    "total blocks": "blocks",
    "total points and rebounds": "points_rebounds",
    "total points and assists": "points_assists",
    "total points rebounds and assists": "points_rebounds_assists",
    "total 3 point field goals": "three_pointers_made",
}
PROP_STAT_BY_SLUG = {
    "mlb_game_player_hits": "hits",
    "mlb_game_player_home_runs": "home_runs",
    "mlb_game_player_pitcher_strikeouts": "strikeouts",
    "mlb_game_player_pitcher_outs": "pitcher_outs_recorded",
    "mlb_game_player_rbis": "rbis",
    "mlb_game_player_hits_runs_rbis": "hits_runs_rbis",
    "mlb_game_player_bases": "total_bases",
}
# Display labels chosen so the pick TEXT also parses through the grader's
# text patterns (structured fields still take precedence).
STAT_DISPLAY = {
    "hits": "Hits",
    "home_runs": "Home Runs",
    "total_bases": "Total Bases",
    "rbis": "RBIs",
    "runs": "Runs",
    "singles": "Singles",
    "doubles": "Doubles",
    "triples": "Triples",
    "stolen_bases": "Stolen Bases",
    "batter_walks": "Walks",
    "hits_runs_rbis": "Hits + Runs + RBIs",
    "strikeouts": "Strikeouts",
    "pitcher_earned_runs_allowed": "Earned Runs Allowed",
    "pitcher_outs_recorded": "Outs Recorded",
    "pitcher_hits_allowed": "Hits Allowed",
    "pitcher_walks_allowed": "Walks Allowed",
    "points": "Points",
    "rebounds": "Rebounds",
    "assists": "Assists",
    "steals": "Steals",
    "blocks": "Blocks",
    "points_rebounds": "Points + Rebounds",
    "points_assists": "Points + Assists",
    "points_rebounds_assists": "Points + Rebounds + Assists",
    "three_pointers_made": "3-Point Field Goals",
}

_PAGE_CACHE: dict[str, tuple[str, str]] = {}
_SLATE_CACHE: dict[tuple[str, str], tuple[list[dict[str, str]], bool]] = {}
_FETCH_PACING_SECONDS = 0.35


def clear_caches() -> None:
    _PAGE_CACHE.clear()
    _SLATE_CACHE.clear()


def _text(node: Any) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)) if node else ""


def _fetch_html(url: str) -> tuple[str, str]:
    """Return (html, error); results are cached per process so the engines
    that share pages (experts/computer/consensus) fetch each URL once."""
    if url in _PAGE_CACHE:
        return _PAGE_CACHE[url]
    result = ("", "")
    for attempt in (1, 2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
        except requests.RequestException as exc:
            result = ("", f"fetch failed: {exc}")
            continue
        lowered = response.text.lower()
        if any(signal in lowered for signal in CLOUDFLARE_SIGNALS):
            result = ("", "fetch blocked by Cloudflare")
            break
        if response.status_code != 200:
            result = ("", f"fetch returned HTTP {response.status_code}")
            continue
        result = (response.text, "")
        break
    _PAGE_CACHE[url] = result
    time.sleep(_FETCH_PACING_SECONDS)
    return result


def _get_slate(sport_key: str, date_iso: str) -> tuple[list[dict[str, str]], bool]:
    cache_key = (sport_key, date_iso)
    if cache_key not in _SLATE_CACHE:
        _SLATE_CACHE[cache_key] = fetch_daily_matchups(
            sport_key, date_iso, config=SPORT_CONFIG[sport_key]
        )
    return _SLATE_CACHE[cache_key]


def _parse_slate_start(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        # The odds hub writes American-ordered stamps: "7-20-2026T18:40:00-04:00".
        us_match = re.match(
            r"^(\d{1,2})-(\d{1,2})-(\d{4})T(\d{2}):(\d{2}):(\d{2})([+-]\d{2}:\d{2})$", text
        )
        if not us_match:
            return None
        month, day, year, hour, minute, second, offset = us_match.groups()
        try:
            parsed = datetime.fromisoformat(
                f"{year}-{int(month):02d}-{int(day):02d}T{hour}:{minute}:{second}{offset}"
            )
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _american_odds(token: str) -> int | None:
    token = str(token or "").strip()
    return int(token) if AMERICAN_ODDS_RE.match(token) else None


def _decimal_odds(american: int) -> float:
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


def _tracking_game_start(text: str, year: int) -> datetime | None:
    match = TRACKING_RE.search(str(text or "").strip())
    if not match:
        return None
    month = MONTHS.get(match.group(3).lower())
    if month is None:
        return None
    hour = int(match.group(5)) % 12 + (12 if match.group(7) == "PM" else 0)
    try:
        eastern = datetime(year, month, int(match.group(4)), hour, int(match.group(6)), tzinfo=EASTERN)
    except ValueError:
        return None
    return eastern.astimezone(timezone.utc)


def _tracking_abbrs(text: str) -> tuple[str, str] | None:
    match = TRACKING_RE.search(str(text or "").strip())
    return (match.group(1).upper(), match.group(2).upper()) if match else None


def _card_tracking_text(card: Any) -> str:
    for node in card.select("[data-tracking]"):
        try:
            data = json.loads(node["data-tracking"])
        except (ValueError, KeyError):
            continue
        text = str(data.get("text") or "")
        if TRACKING_RE.search(text):
            return text
    return ""


def _normalize_market_label(value: str) -> str:
    lowered = re.sub(r"[+&,/-]", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _player_name_from_href(href: str) -> str:
    slug = str(href or "").rstrip("/").rsplit("/", 1)[-1]
    if not slug:
        return ""
    return " ".join(part.capitalize() if part else part for part in slug.replace("-", " ").split(" "))


def parse_league_page(html: str) -> dict[str, Any]:
    """Extract matchup groups + schema.org events from a /picks/{league} page."""
    soup = BeautifulSoup(html, "html.parser")
    events: dict[str, dict[str, Any]] = {}
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or "")
        except ValueError:
            continue
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict) or item.get("@type") != "SportsEvent":
                continue
            def _name(side: Any) -> str:
                if isinstance(side, dict):
                    return str(side.get("name") or "").strip()
                return str(side or "").strip()
            # Picks pages use a bare numeric identifier; the odds hub embeds
            # it as a "Name vs Name-369386" suffix.
            id_match = re.search(r"(\d+)\s*$", str(item.get("identifier") or "").strip())
            if id_match:
                events[id_match.group(1)] = {
                    "away": _name(item.get("awayTeam")),
                    "home": _name(item.get("homeTeam")),
                    "start_utc": _parse_slate_start(item.get("startDate")),
                }

    groups: dict[str, dict[str, Any]] = {}
    for card in soup.select("div.picks-card[id], div.pick-cards-simple-component[id]"):
        matchup_id = str(card.get("id") or "").strip()
        if not matchup_id.isdigit():
            continue
        link = card.select_one('a[href*="/matchup/"]')
        href = str(link.get("href") or "") if link else ""
        if not MATCHUP_HREF_RE.search(href):
            continue
        teams = [
            re.sub(r"\s+logo$", "", str(img.get("alt") or "").strip())
            for img in card.select("img[alt$='logo']")
        ]
        counts_text = _text(card)
        expert = re.search(r"(\d+)\s+Expert\s+Picks?", counts_text)
        computer = re.search(r"(\d+)\s+Computer\s+Picks?", counts_text)
        groups[matchup_id] = {
            "detail_url": href if href.startswith("http") else f"{BASE_URL}{href}",
            "away": teams[0] if len(teams) >= 2 else "",
            "home": teams[1] if len(teams) >= 2 else "",
            "expert_count": int(expert.group(1)) if expert else 0,
            "computer_count": int(computer.group(1)) if computer else 0,
        }
    return {"events": events, "groups": groups}


def _match_covers_matchups(
    expected: list[dict[str, str]],
    league: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Assign each official slate matchup its Covers matchup id.

    Strict home/away orientation plus an 8h kickoff window (league pages list
    future series meetings whose opponents repeat), greedy nearest-first so
    doubleheader games claim distinct Covers ids.
    """
    events, groups = league["events"], league["groups"]
    candidates: dict[str, dict[str, Any]] = {}
    for matchup_id in set(events) | set(groups):
        event = events.get(matchup_id) or {}
        group = groups.get(matchup_id) or {}
        away = event.get("away") or group.get("away") or ""
        home = event.get("home") or group.get("home") or ""
        if away and home:
            candidates[matchup_id] = {
                "matchup_id": matchup_id,
                "away": away,
                "home": home,
                "start_utc": event.get("start_utc"),
                "detail_url": group.get("detail_url", ""),
                "expert_count": group.get("expert_count", 0),
                "computer_count": group.get("computer_count", 0),
            }

    # Doubleheaders: the grader matches games by team names only, so a
    # game-2 row would grade against game 1's final score. Publish only the
    # earliest game of any repeated team pair and report the rest.
    pair_counts: dict[tuple[str, str], int] = {}
    for matchup in expected:
        pair = (matchup["away"].strip().lower(), matchup["home"].strip().lower())
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    seen_pairs: set[tuple[str, str]] = set()

    def _sort_key(matchup: dict[str, str]) -> str:
        return str(matchup.get("start_time") or "9999")

    matched: list[dict[str, Any]] = []
    unmatched: list[str] = []
    claimed: set[str] = set()
    for matchup in sorted(expected, key=_sort_key):
        pair = (matchup["away"].strip().lower(), matchup["home"].strip().lower())
        if pair_counts[pair] > 1 and pair in seen_pairs:
            unmatched.append(
                f"{matchup['away']} @ {matchup['home']} (doubleheader game 2 — grading ambiguity)"
            )
            continue
        seen_pairs.add(pair)
        slate_start = _parse_slate_start(matchup.get("start_time"))
        options = [
            candidate
            for candidate in candidates.values()
            if candidate["matchup_id"] not in claimed
            and _team_matches(matchup["away"], candidate["away"])
            and _team_matches(matchup["home"], candidate["home"])
        ]
        best = None
        if options and slate_start is not None:
            timed = [option for option in options if option["start_utc"] is not None]
            if timed:
                nearest = min(
                    timed,
                    key=lambda option: abs((option["start_utc"] - slate_start).total_seconds()),
                )
                if abs((nearest["start_utc"] - slate_start).total_seconds()) <= KICKOFF_MATCH_WINDOW_SECONDS:
                    best = nearest
            elif options:
                best = options[0]
        elif options:
            best = options[0]
        if best is None:
            unmatched.append(f"{matchup['away']} @ {matchup['home']}")
            continue
        claimed.add(best["matchup_id"])
        matched.append({**best, "matchup": matchup, "slate_start": slate_start})
    return matched, unmatched


def _detail_url_from_entry(config: dict[str, Any], entry: dict[str, Any]) -> str:
    if entry.get("detail_url"):
        return entry["detail_url"]
    # Covers detail paths mirror the ESPN sport/league slugs for MLB and WNBA.
    return f"{BASE_URL}/sport/{config['espn_sport']}/{config['espn_league']}/matchup/{entry['matchup_id']}/picks"


def parse_matchup_page(html: str) -> dict[str, Any]:
    """Extract expert cards, game-market projection rows, and consensus."""
    soup = BeautifulSoup(html, "html.parser")

    expert_cards: list[dict[str, Any]] = []
    seen_cards: set[str] = set()
    for card in soup.select("div.pick-cards-expert-component"):
        card_id = str(card.get("id") or "").strip()
        if not GUID_RE.match(card_id) or card_id in seen_cards:
            continue
        seen_cards.add(card_id)
        author_link = card.select_one('.card.profile-card a[href*="/writers/"]')
        role_node = author_link.find_parent("div").find_next_sibling("div") if author_link else None
        player_link = card.select_one("a.player-link[href]")
        expert_cards.append(
            {
                "card_id": card_id,
                "types": str(card.get("data-pick-types") or "").strip(),
                "team": str(card.get("data-pick-teams") or "").strip(),
                "selection": _text(card.select_one("div.w-100.fw-bold.small")),
                "author": _text(author_link),
                "author_url": str(author_link.get("href") or "") if author_link else "",
                "author_role": _text(role_node),
                "player_href": str(player_link.get("href") or "") if player_link else "",
                "tracking_text": _card_tracking_text(card),
            }
        )

    game_rows: list[dict[str, Any]] = []
    seen_rows: set[str] = set()
    for row in soup.select("tr.game-projections-container"):
        row_id = str(row.get("data-id") or "").strip()
        section = row.select_one("section.picks-card")
        market_id = str(row.get("data-market-id") or (section.get("data-market-id") if section else "") or "").strip()
        if not row_id or row_id in seen_rows or market_id != "0":
            continue
        seen_rows.add(row_id)
        best = row.select_one(".picks-best-odds a.deeplink")
        best_book = best.select_one("img") if best else None
        game_rows.append(
            {
                "row_id": row_id,
                "badge": _text(row.select_one("span._badge")).upper(),
                "category": _text(row.select_one("span.category")),
                "prediction": _text(row.select_one("span.prediction")),
                "ev": str(row.get("data-ev") or "").strip(),
                "best_text": _text(best),
                "best_book": str(best_book.get("alt") or "").strip() if best_book else "",
                "odds_columns": [_text(col) for col in row.select(".compare-odds-column")],
                "tracking_text": _card_tracking_text(row),
            }
        )

    consensus: dict[str, Any] = {}
    for section in soup.select("div.pick-detail-section"):
        counts_node = section.select_one("p.total-picks-count")
        away_bar = section.select_one(".pick-team-away progress")
        home_bar = section.select_one(".pick-team-home progress")
        if counts_node is None or away_bar is None or home_bar is None:
            continue
        counts = re.findall(r"([A-Za-z]{2,4})\s+(\d+)", _text(counts_node))
        try:
            consensus = {
                "away_pct": int(str(away_bar.get("value") or "")),
                "home_pct": int(str(home_bar.get("value") or "")),
                "votes": {abbr.upper(): int(count) for abbr, count in counts},
            }
        except ValueError:
            consensus = {}
        break

    abbrs = None
    for item in (*expert_cards, *game_rows):
        abbrs = _tracking_abbrs(item.get("tracking_text", ""))
        if abbrs:
            break
    return {"expert_cards": expert_cards, "game_rows": game_rows, "consensus": consensus, "abbrs": abbrs}


def _sel_odds(selection: str) -> int | None:
    match = SEL_ODDS_RE.search(selection or "")
    return int(match.group(1)) if match else None


def _line_matched_odds(row: dict[str, Any], line: float, side: str = "") -> tuple[int | None, str]:
    """Best odds among book entries quoting the board's own line AND side.

    Entries look like "u9.5 -133" (total, side marker o/u), "+1.5 -167"
    (spread, signed line), or a bare "-108" (moneyline). The best-odds chip
    can quote a different line — or the opposite side — than the board row,
    so odds are only accepted at the row's own side and line: totals require
    the matching o/u marker, spreads require the exact signed line.
    """
    def _entry(text: str) -> tuple[str, float | None, int | None]:
        match = re.match(r"^(?:([ou]))?([+-]?\d+(?:\.\d+)?)\s+([+-]\d{3,5})\s*$", str(text or "").strip())
        if not match:
            return "", None, None
        return match.group(1) or "", float(match.group(2)), int(match.group(3))

    def _accept(marker: str, entry_line: float | None) -> bool:
        if entry_line is None:
            return False
        if side:
            return marker == side and abs(entry_line - abs(line)) < 0.01
        return not marker and abs(entry_line - line) < 0.01

    best_marker, best_line, best_odds = _entry(row.get("best_text"))
    if best_odds is not None and _accept(best_marker, best_line):
        return best_odds, str(row.get("best_book") or "")
    chosen: int | None = None
    for column in row.get("odds_columns") or []:
        marker, entry_line, odds = _entry(column)
        if odds is None or not _accept(marker, entry_line):
            continue
        if chosen is None or _decimal_odds(odds) > _decimal_odds(chosen):
            chosen = odds
    return chosen, ""


def _market_identity(pick: dict[str, Any]) -> tuple[str, ...]:
    """Stable identity for carry-forward dedupe: a moved line replaces the
    old row pregame instead of duplicating it. The Covers matchup id keeps
    doubleheader games (identical matchup labels) apart."""
    return (
        str(pick.get("source") or "").strip().lower(),
        str(pick.get("covers_matchup_id") or "").strip(),
        str(pick.get("matchup") or "").strip().lower(),
        str(pick.get("market") or "").strip().lower(),
        str(pick.get("team") or "").strip().lower(),
        str(pick.get("direction") or "").strip().lower(),
        str(pick.get("player_name") or "").strip().lower(),
        str(pick.get("stat_key") or "").strip().lower(),
    )


def _base_pick(
    config: dict[str, Any],
    engine: str,
    date_iso: str,
    entry: dict[str, Any],
    *,
    source: str,
    tip: str,
    scope: str,
    market: str,
    pick_url: str,
    external_id: str,
) -> dict[str, Any]:
    matchup = entry["matchup"]
    matchup_label = f"{matchup['away']} @ {matchup['home']}"
    return {
        # A stable explicit id keeps doubleheader rows (identical pick text
        # and matchup label) from hash-colliding in the frontend pick maps.
        "id": external_id,
        "source": source,
        "source_site": "covers",
        "covers_engine": engine,
        "pick": f"{tip} ({matchup_label})",
        "tip": tip,
        "sport": config["label"],
        "date": date_iso,
        "matchup": matchup_label,
        "game": matchup_label,
        "away_team": matchup["away"],
        "home_team": matchup["home"],
        "start_time": matchup.get("start_time") or None,
        "scope": scope,
        "market": market,
        "decision": "BET",
        "units": 1,
        "odds": None,
        "probability": None,
        "edge": None,
        "source_url": pick_url,
        "pick_url": pick_url,
        "covers_matchup_id": entry["matchup_id"],
        "covers_external_id": external_id,
    }


def _expert_pick(
    config: dict[str, Any],
    date_iso: str,
    entry: dict[str, Any],
    card: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    """Normalize one expert card, or return (None, reason)."""
    matchup = entry["matchup"]
    selection = card["selection"]
    types_normalized = _normalize_market_label(card["types"])
    author = card["author"].strip()
    source = f"{EXPERT_SOURCE_PREFIX}{author}" if author else EXPERT_SOURCE_FALLBACK
    odds = _sel_odds(selection)
    pick_url = _detail_url_from_entry(config, entry)
    external_id = f"covers:expert:{config['label'].lower()}:{entry['matchup_id']}:{card['card_id']}"

    def _finish(pick: dict[str, Any]) -> tuple[dict[str, Any], str]:
        pick.update(
            {
                "odds": odds,
                "covers_author": author or None,
                "covers_author_url": card["author_url"] or None,
                "covers_author_role": card["author_role"] or None,
                "covers_pick_id": card["card_id"],
                "covers_market_label": card["types"] or None,
            }
        )
        return pick, ""

    # Player prop cards carry a player link; the stat comes from the market
    # label. Anything without a supported stat mapping is skipped, not guessed.
    if card["player_href"]:
        stat_key = PROP_STAT_BY_LABEL.get(types_normalized)
        if not stat_key:
            return None, f"unsupported prop market: {card['types'] or 'unknown'}"
        side_match = SEL_LINE_RE.search(selection)
        if not side_match:
            return None, f"unparseable prop selection: {selection}"
        direction = "Over" if side_match.group(1) == "o" else "Under"
        line = float(side_match.group(2))
        player = _player_name_from_href(card["player_href"])
        if not player:
            return None, f"missing player name: {selection}"
        tip = f"{player} {direction} {line:g} {STAT_DISPLAY[stat_key]}"
        pick = _base_pick(
            config, "experts", date_iso, entry,
            source=source, tip=tip, scope="player", market="player_prop",
            pick_url=pick_url, external_id=external_id,
        )
        pick.update(
            {
                "market_type": "external_player_prop",
                "external_player_feed": True,
                "player_name": player,
                "player": player,
                "stat_key": stat_key,
                "selection": direction.upper(),
                "direction": direction,
                "line": line,
            }
        )
        return _finish(pick)

    if "team total" in selection.lower():
        side_match = SEL_LINE_RE.search(selection)
        if not side_match or not card["team"]:
            return None, f"unparseable team total: {selection}"
        direction = "Over" if side_match.group(1) == "o" else "Under"
        line = float(side_match.group(2))
        tip = f"{card['team']} Team Total {direction} {line:g}"
        pick = _base_pick(
            config, "experts", date_iso, entry,
            source=source, tip=tip, scope="team", market="team_total",
            pick_url=pick_url, external_id=external_id,
        )
        pick.update({"team": card["team"], "direction": direction, "line": line})
        return _finish(pick)

    # A prop-market label without a player link means the card structure
    # drifted — skip it rather than let it fall through to the game-total
    # branch (every batting-prop label contains the word "total").
    if types_normalized in PROP_STAT_BY_LABEL:
        return None, f"prop card missing player link: {card['types']}"

    if "moneyline" in types_normalized:
        team = card["team"] or ""
        if not _team_matches(matchup["away"], team) and not _team_matches(matchup["home"], team):
            return None, f"moneyline team mismatch: {team or selection}"
        team_name = matchup["away"] if _team_matches(matchup["away"], team) else matchup["home"]
        tip = f"{team_name} ML"
        pick = _base_pick(
            config, "experts", date_iso, entry,
            source=source, tip=tip, scope="team", market="moneyline",
            pick_url=pick_url, external_id=external_id,
        )
        pick.update({"team": team_name})
        return _finish(pick)

    if "spread" in types_normalized or "run line" in types_normalized or "puck line" in types_normalized:
        spread_match = SEL_SPREAD_RE.search(re.sub(r"\([+-]\d+\)\s*$", "", selection).strip())
        team = card["team"] or ""
        if not spread_match or not team:
            return None, f"unparseable spread: {selection}"
        if not _team_matches(matchup["away"], team) and not _team_matches(matchup["home"], team):
            return None, f"spread team mismatch: {team}"
        team_name = matchup["away"] if _team_matches(matchup["away"], team) else matchup["home"]
        line = float(spread_match.group(1))
        tip = f"{team_name} {line:+g}"
        pick = _base_pick(
            config, "experts", date_iso, entry,
            source=source, tip=tip, scope="team", market="spread",
            pick_url=pick_url, external_id=external_id,
        )
        pick.update({"team": team_name, "line": line})
        return _finish(pick)

    if "total" in types_normalized:
        if card["team"]:
            # A team-anchored "Total" card without the literal words "team
            # total" is ambiguous — never grade it as a full-game total.
            return None, f"ambiguous team-anchored total: {selection}"
        side_match = SEL_LINE_RE.search(selection)
        if not side_match:
            return None, f"unparseable total: {selection}"
        direction = "Over" if side_match.group(1) == "o" else "Under"
        line = float(side_match.group(2))
        tip = f"{direction} {line:g}"
        pick = _base_pick(
            config, "experts", date_iso, entry,
            source=source, tip=tip, scope="team", market="total",
            pick_url=pick_url, external_id=external_id,
        )
        pick.update({"direction": direction, "line": line})
        return _finish(pick)

    return None, f"unsupported market: {card['types'] or selection}"


def _computer_pick(
    config: dict[str, Any],
    date_iso: str,
    entry: dict[str, Any],
    row: dict[str, Any],
    abbr_teams: dict[str, str],
) -> tuple[dict[str, Any] | None, str]:
    matchup = entry["matchup"]
    source = ENGINE_SOURCES["computer"][config["label"].lower()]
    pick_url = _detail_url_from_entry(config, entry)
    external_id = f"covers:computer:{config['label'].lower()}:{entry['matchup_id']}:{row['row_id']}"
    badge, category, prediction = row["badge"], row["category"], row["prediction"]

    def _team_for(abbr: str) -> str:
        if abbr in abbr_teams:
            return abbr_teams[abbr]
        for side in ("away", "home"):
            if _team_matches(matchup[side], abbr):
                return matchup[side]
        return ""

    if badge == "MONEYLINE":
        team_name = _team_for(category)
        if not team_name:
            return None, f"moneyline side unmapped: {category}"
        odds_match = re.search(r"([+-]\d{3,5})", prediction)
        odds = _american_odds(odds_match.group(1)) if odds_match else None
        pick = _base_pick(
            config, "computer", date_iso, entry,
            source=source, tip=f"{team_name} ML", scope="team", market="moneyline",
            pick_url=pick_url, external_id=external_id,
        )
        pick.update({"team": team_name, "odds": odds})
    elif badge == "TOTAL":
        direction = category.title()
        line_match = re.search(r"(\d+(?:\.\d+)?)", prediction)
        if direction not in {"Over", "Under"} or not line_match:
            return None, f"unparseable total row: {category} {prediction}"
        line = float(line_match.group(1))
        odds, book = _line_matched_odds(row, line, side="o" if direction == "Over" else "u")
        pick = _base_pick(
            config, "computer", date_iso, entry,
            source=source, tip=f"{direction} {line:g}", scope="team", market="total",
            pick_url=pick_url, external_id=external_id,
        )
        pick.update({"direction": direction, "line": line, "odds": odds})
        if book:
            pick["covers_book"] = book
    elif badge == "SPREAD":
        team_name = _team_for(category)
        line_match = re.search(r"([+-]\d+(?:\.\d+)?)", prediction)
        if not team_name or not line_match:
            return None, f"unparseable spread row: {category} {prediction}"
        line = float(line_match.group(1))
        odds, book = _line_matched_odds(row, line)
        pick = _base_pick(
            config, "computer", date_iso, entry,
            source=source, tip=f"{team_name} {line:+g}", scope="team", market="spread",
            pick_url=pick_url, external_id=external_id,
        )
        pick.update({"team": team_name, "line": line, "odds": odds})
        if book:
            pick["covers_book"] = book
    else:
        return None, f"unsupported computer market: {badge}"

    try:
        pick["covers_ev"] = float(row["ev"])
    except (TypeError, ValueError):
        pick["covers_ev"] = None
    pick["covers_pick_id"] = row["row_id"]
    return pick, ""


def _consensus_pick(
    config: dict[str, Any],
    date_iso: str,
    entry: dict[str, Any],
    consensus: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    matchup = entry["matchup"]
    away_pct, home_pct = consensus.get("away_pct"), consensus.get("home_pct")
    votes = consensus.get("votes") or {}
    total_votes = sum(votes.values())
    if away_pct is None or home_pct is None:
        return None, "consensus split unavailable"
    if total_votes < CONSENSUS_MIN_VOTES:
        return None, f"below vote floor ({total_votes} < {CONSENSUS_MIN_VOTES})"
    majority_pct = max(away_pct, home_pct)
    if majority_pct < CONSENSUS_MIN_PCT:
        return None, f"no clear majority ({majority_pct}% < {CONSENSUS_MIN_PCT}%)"
    side = "away" if away_pct >= home_pct else "home"
    team_name = matchup[side]
    source = ENGINE_SOURCES["consensus"][config["label"].lower()]
    pick = _base_pick(
        config, "consensus", date_iso, entry,
        source=source, tip=f"{team_name} ML", scope="team", market="moneyline",
        pick_url=_detail_url_from_entry(config, entry),
        external_id=f"covers:consensus:{config['label'].lower()}:{entry['matchup_id']}:{side}",
    )
    pick.update(
        {
            "team": team_name,
            "covers_consensus_pct": majority_pct,
            "covers_consensus_away_pct": away_pct,
            "covers_consensus_home_pct": home_pct,
            "covers_consensus_votes": total_votes,
        }
    )
    return pick, ""


def parse_props_rows(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in soup.select("tr.game-projections-container"):
        row_id = str(row.get("data-id") or "").strip()
        section = row.select_one("section.picks-card")
        market_id = str(row.get("data-market-id") or (section.get("data-market-id") if section else "") or "").strip()
        if not row_id or row_id in seen or market_id == "0":
            continue
        seen.add(row_id)
        player_link = row.select_one("a.player-link[href]")
        stars = 0
        for hidden in row.select("span.visually-hidden"):
            star_match = re.search(r"Star rating:\s*(\d)", hidden.get_text())
            if star_match:
                stars = int(star_match.group(1))
        best = row.select_one(".picks-best-odds a.deeplink")
        best_book = best.select_one("img") if best else None
        rows.append(
            {
                "row_id": row_id,
                "game_id": str(row.get("data-game-id") or "").strip(),
                "market_name": str(row.get("data-market-name") or "").strip(),
                "diff": str(row.get("data-diff") or "").strip(),
                "ev": str(row.get("data-ev") or "").strip(),
                "stars": stars,
                "prediction": _text(row.select_one("span.prediction")),
                "projection": _text(row.select_one(".projections-container .projections span.fs-11")),
                "player_href": str(player_link.get("href") or "") if player_link else "",
                "best_text": _text(best),
                "best_book": str(best_book.get("alt") or "").strip() if best_book else "",
            }
        )
    return rows


def _props_pick(
    config: dict[str, Any],
    date_iso: str,
    entry: dict[str, Any],
    row: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    stat_key = PROP_STAT_BY_SLUG.get(row["market_name"])
    if not stat_key:
        return None, f"unsupported prop market: {row['market_name']}"
    player = _player_name_from_href(row["player_href"])
    if not player:
        return None, f"missing player link: {row['row_id']}"
    line_match = re.search(r"(\d+(?:\.\d+)?)", row["prediction"])
    if not line_match:
        return None, f"unparseable prop line: {row['prediction']}"
    line = float(line_match.group(1))

    side_match = re.match(r"^([ou])(\d+(?:\.\d+)?)\s+([+-]\d{3,5})$", row["best_text"].strip())
    direction = None
    odds = None
    if side_match:
        direction = "Over" if side_match.group(1) == "o" else "Under"
        if abs(float(side_match.group(2)) - line) < 0.01:
            odds = int(side_match.group(3))
    if direction is None:
        try:
            diff_value = float(row["diff"].replace("+", ""))
        except ValueError:
            return None, f"prop side undeterminable: {row['row_id']}"
        if diff_value == 0:
            # Projection exactly on the line: the model has no lean.
            return None, f"prop side undeterminable (zero diff): {row['row_id']}"
        direction = "Over" if diff_value > 0 else "Under"

    source = ENGINE_SOURCES["props"][config["label"].lower()]
    tip = f"{player} {direction} {line:g} {STAT_DISPLAY[stat_key]}"
    pick = _base_pick(
        config, "props", date_iso, entry,
        source=source, tip=tip, scope="player", market="player_prop",
        pick_url=config["props_url"],
        external_id=f"covers:props:{config['label'].lower()}:{entry['matchup_id']}:{row['row_id']}",
    )
    pick.update(
        {
            "market_type": "external_player_prop",
            "external_player_feed": True,
            "player_name": player,
            "player": player,
            "stat_key": stat_key,
            "selection": direction.upper(),
            "direction": direction,
            "line": line,
            "odds": odds,
            "covers_pick_id": row["row_id"],
            "covers_stars": row["stars"],
            "covers_projection": row["projection"] or None,
            "covers_diff": row["diff"] or None,
            "covers_model": config["props_model"],
        }
    )
    if row["best_book"]:
        pick["covers_book"] = row["best_book"]
    try:
        pick["covers_ev"] = float(row["ev"])
    except (TypeError, ValueError):
        pick["covers_ev"] = None
    return pick, ""


def _previous_bucket_picks(feed_key: str, date_iso: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads((MODEL_CACHE_DIR / f"{date_iso}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    models = payload.get("models") if isinstance(payload, dict) else None
    bucket = models.get(feed_key) if isinstance(models, dict) else None
    picks = bucket.get("picks") if isinstance(bucket, dict) else None
    return [pick for pick in picks or [] if isinstance(pick, dict)]


def _apply_carry_forward(
    feed_key: str,
    date_iso: str,
    fresh_picks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Keep previously published picks that this scrape no longer sees.

    Covers pages drop cards once games go final; without this, an evening
    re-scrape would erase the morning's published (possibly settled) picks.
    A decided committed row always beats a fresh duplicate — a postponed
    game can grade push while its clock start is still in the future, and
    that settled record must never be replaced by a re-scrape.
    """
    def _slot(pick: dict[str, Any]) -> tuple[str, ...]:
        # Side/line flips and Covers card-id churn must replace the old
        # pending row, not sit next to it: one slot per source+game+market
        # (+player for props). Team totals stay per-team — both sides of a
        # game are genuinely distinct markets, not a flip.
        market = str(pick.get("market") or "").strip().lower()
        return (
            str(pick.get("source") or "").strip().lower(),
            str(pick.get("covers_matchup_id") or "").strip(),
            str(pick.get("matchup") or "").strip().lower(),
            market,
            str(pick.get("team") or "").strip().lower() if market == "team_total" else "",
            str(pick.get("player_name") or "").strip().lower(),
            str(pick.get("stat_key") or "").strip().lower(),
        )

    def _card_id(pick: dict[str, Any]) -> str:
        return str(pick.get("covers_external_id") or "").strip()

    previous = _previous_bucket_picks(feed_key, date_iso)
    decided_ids = {
        _market_identity(pick)
        for pick in previous
        if str(pick.get("result") or "").strip().lower() in {"win", "loss", "push"}
    }
    fresh = [pick for pick in fresh_picks if _market_identity(pick) not in decided_ids]
    fresh_ids = {_market_identity(pick) for pick in fresh}
    fresh_slots = {_slot(pick) for pick in fresh}
    fresh_cards = {_card_id(pick) for pick in fresh if _card_id(pick)}

    def _keep_carried(pick: dict[str, Any]) -> bool:
        if str(pick.get("result") or "").strip().lower() in {"win", "loss", "push"}:
            return _market_identity(pick) not in fresh_ids
        return (
            _market_identity(pick) not in fresh_ids
            and _slot(pick) not in fresh_slots
            and _card_id(pick) not in fresh_cards
        )

    carried = [pick for pick in previous if _keep_carried(pick)]
    return [*carried, *fresh], len(carried)


def _empty_result(config: dict[str, Any], engine: str, date_iso: str) -> dict[str, Any]:
    return {
        "ok": True,
        "date": date_iso,
        "picks": [],
        "note": f"Covers {engine} has no official {config['label']} {date_iso} matchups.",
        "meta": {
            "officialMatchups": 0,
            "expectedMatchups": 0,
            "matchedPicks": 0,
            "missingMatchups": [],
            "unpublishedMatchups": [],
            "attemptedUrls": 0,
            "blockedUrls": 0,
        },
    }


def _error_result(
    config: dict[str, Any],
    engine: str,
    date_iso: str,
    error: str,
    expected: list[dict[str, str]],
    attempted: int,
    blocked: int,
) -> dict[str, Any]:
    return {
        "ok": False,
        "date": date_iso,
        "picks": [],
        "error": f"Covers {engine} {config['label']}: {error}",
        "meta": {
            "officialMatchups": len(expected),
            "expectedMatchups": 0,
            "matchedPicks": 0,
            "missingMatchups": [f"{m['away']} @ {m['home']}" for m in expected],
            "unpublishedMatchups": [],
            "attemptedUrls": attempted,
            "blockedUrls": blocked,
        },
    }


def scrape_covers(
    sport: str,
    engine: str,
    date_iso: str,
    *,
    pages: dict[str, str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Run one Covers engine for one sport against the official slate.

    `pages` lets tests inject fetched HTML: keys "league", "props", and
    "matchup:<id>". `now` overrides the pregame-gate clock.
    """
    sport_key = sport.strip().lower()
    if sport_key not in SPORT_CONFIG:
        raise ValueError(f"unsupported Covers sport: {sport}")
    if engine not in ENGINES:
        raise ValueError(f"unsupported Covers engine: {engine}")
    config = SPORT_CONFIG[sport_key]
    feed_key = f"covers_{engine}_{sport_key}"
    now = now or datetime.now(timezone.utc)
    if engine == "computer" and not config["has_computer_board"]:
        raise ValueError(f"Covers runs no computer board for {config['label']}")
    if engine == "props" and not config["has_props_board"]:
        raise ValueError(f"Covers runs no props board for {config['label']}")

    expected, slate_resolved = _get_slate(sport_key, date_iso)
    if not slate_resolved:
        return {
            "ok": False,
            "date": date_iso,
            "picks": [],
            "error": f"Covers {engine} could not resolve an official {config['label']} {date_iso} slate",
        }
    if not expected:
        # Never let a transiently empty ESPN slate wipe already-published
        # (possibly settled) picks for the date.
        carried_picks, carried = _apply_carry_forward(feed_key, date_iso, [])
        result = _empty_result(config, engine, date_iso)
        if carried_picks:
            result["picks"] = carried_picks
            result["note"] = (
                f"Covers {engine} carried {carried} previously published "
                f"{config['label']} pick(s) through an empty {date_iso} slate."
            )
            result["meta"].update(
                {"expectedMatchups": len(carried_picks), "matchedPicks": len(carried_picks), "carriedForward": carried}
            )
        return result

    def _page(key: str, url: str) -> tuple[str, str]:
        if pages is not None:
            return pages.get(key, ""), "" if key in pages else "missing injected page"
        return _fetch_html(url)

    attempted = 1
    league_html, league_error = _page("league", config["league_url"])
    if league_error or not league_html:
        blocked = int("Cloudflare" in (league_error or ""))
        return _error_result(config, engine, date_iso, league_error or "empty league page", expected, attempted, blocked)
    league = parse_league_page(league_html)
    # The /picks page lists only matchups Covers is promoting; the odds hub
    # carries the full daily slate. Merge its events (best-effort) so the
    # computer/consensus/props engines can reach every game.
    odds_page_used = False
    if config.get("odds_url"):
        attempted += 1
        odds_html, odds_error = _page("odds", config["odds_url"])
        if odds_html and not odds_error:
            odds_events = parse_league_page(odds_html)["events"]
            league["events"] = {**odds_events, **league["events"]}
            odds_page_used = True
    entries, unmatched = _match_covers_matchups(expected, league)

    picks: list[dict[str, Any]] = []
    skipped: list[str] = []
    stale_cards = 0
    pregame_skipped = 0
    failed_urls: list[str] = []
    unpublished: dict[str, str] = {name: "not listed on Covers" for name in unmatched}

    def _is_pregame(entry: dict[str, Any]) -> bool:
        slate_start = entry.get("slate_start")
        return slate_start is None or now < slate_start

    def _matchup_label(entry: dict[str, Any]) -> str:
        return f"{entry['matchup']['away']} @ {entry['matchup']['home']}"

    if engine == "props":
        attempted += 1
        props_html, props_error = _page("props", config["props_url"])
        if props_error or not props_html:
            blocked = int("Cloudflare" in (props_error or ""))
            return _error_result(config, engine, date_iso, props_error or "empty props page", expected, attempted, blocked)
        entries_by_covers_id = {entry["matchup_id"]: entry for entry in entries}
        for row in parse_props_rows(props_html):
            entry = entries_by_covers_id.get(row["game_id"])
            if entry is None:
                continue  # future-date or off-slate board rows
            if row["stars"] < PROPS_MIN_STARS:
                skipped.append(f"below star floor ({row['stars']}): {row['row_id']}")
                continue
            if not _is_pregame(entry):
                pregame_skipped += 1
                continue
            pick, reason = _props_pick(config, date_iso, entry, row)
            if pick is None:
                skipped.append(reason)
                continue
            picks.append(pick)
    else:
        fetched_ok = 0
        for entry in entries:
            if engine == "experts" and not entry.get("expert_count"):
                unpublished.setdefault(_matchup_label(entry), "no expert picks listed")
                continue
            if not _is_pregame(entry):
                pregame_skipped += 1
                unpublished.setdefault(_matchup_label(entry), "game already started")
                continue
            attempted += 1
            detail_html, detail_error = _page(f"matchup:{entry['matchup_id']}", _detail_url_from_entry(config, entry))
            if detail_error or not detail_html:
                failed_urls.append(_detail_url_from_entry(config, entry))
                unpublished.setdefault(_matchup_label(entry), f"detail page failed: {detail_error or 'empty'}")
                continue
            fetched_ok += 1
            page = parse_matchup_page(detail_html)
            abbrs = page.get("abbrs")
            abbr_teams = (
                {abbrs[0]: entry["matchup"]["away"], abbrs[1]: entry["matchup"]["home"]} if abbrs else {}
            )
            produced = 0

            if engine == "experts":
                for card in page["expert_cards"]:
                    # League/detail pages can attach picks from earlier
                    # meetings of the same teams; the card's own game
                    # datetime must sit at this slate game's kickoff.
                    card_start = _tracking_game_start(card["tracking_text"], int(date_iso[:4]))
                    slate_start = entry.get("slate_start")
                    if card_start is not None and slate_start is not None:
                        if abs((card_start - slate_start).total_seconds()) > KICKOFF_MATCH_WINDOW_SECONDS:
                            stale_cards += 1
                            continue
                    pick, reason = _expert_pick(config, date_iso, entry, card)
                    if pick is None:
                        skipped.append(reason)
                        continue
                    picks.append(pick)
                    produced += 1
            elif engine == "computer":
                for row in page["game_rows"]:
                    pick, reason = _computer_pick(config, date_iso, entry, row, abbr_teams)
                    if pick is None:
                        skipped.append(reason)
                        continue
                    picks.append(pick)
                    produced += 1
            elif engine == "consensus":
                if not page["consensus"]:
                    unpublished.setdefault(_matchup_label(entry), "no consensus split")
                else:
                    pick, reason = _consensus_pick(config, date_iso, entry, page["consensus"])
                    if pick is None:
                        unpublished.setdefault(_matchup_label(entry), reason)
                    else:
                        picks.append(pick)
                        produced += 1

            if produced == 0:
                unpublished.setdefault(_matchup_label(entry), "no qualifying picks")

        if failed_urls and fetched_ok == 0:
            # Every attempted detail page failed: fail closed so the merge
            # keeps the previously committed bucket untouched.
            return _error_result(
                config, engine, date_iso,
                f"every matchup page failed ({len(failed_urls)})", expected, attempted, 0,
            )

    all_picks, carried = _apply_carry_forward(feed_key, date_iso, picks)
    published_matchups = {str(pick.get("matchup") or "") for pick in all_picks}
    unpublished_notes = [
        f"{label} — {reason}"
        for label, reason in sorted(unpublished.items())
        if label not in published_matchups
    ]

    authors = sorted(
        {
            str(pick.get("covers_author"))
            for pick in all_picks
            if pick.get("covers_author")
        }
    )
    result = {
        "ok": True,
        "date": date_iso,
        "picks": all_picks,
        "note": (
            f"Covers {engine} matched {len(picks)} fresh pick(s) "
            f"(+{carried} carried) against {len(expected)} official "
            f"{config['label']} {date_iso} matchup(s)."
        ),
        "meta": {
            "officialMatchups": len(expected),
            "coversMatchups": len(entries),
            "expectedMatchups": len(all_picks),
            "matchedPicks": len(all_picks),
            "freshPicks": len(picks),
            "carriedForward": carried,
            "missingMatchups": [],
            "unpublishedMatchups": unpublished_notes,
            "attemptedUrls": attempted,
            "blockedUrls": 0,
            "failedUrls": failed_urls,
            "oddsPageUsed": odds_page_used,
            "staleCards": stale_cards,
            "pregameSkipped": pregame_skipped,
            "skipped": skipped[:40],
        },
    }
    if engine == "experts":
        result["meta"]["authors"] = authors
    return result


def run_covers_experts_mlb(date_iso: str, _sports: list[str] | None = None) -> dict[str, Any]:
    return scrape_covers("mlb", "experts", date_iso)


def run_covers_experts_wnba(date_iso: str, _sports: list[str] | None = None) -> dict[str, Any]:
    return scrape_covers("wnba", "experts", date_iso)


def run_covers_computer_mlb(date_iso: str, _sports: list[str] | None = None) -> dict[str, Any]:
    return scrape_covers("mlb", "computer", date_iso)


def run_covers_consensus_mlb(date_iso: str, _sports: list[str] | None = None) -> dict[str, Any]:
    return scrape_covers("mlb", "consensus", date_iso)


def run_covers_consensus_wnba(date_iso: str, _sports: list[str] | None = None) -> dict[str, Any]:
    return scrape_covers("wnba", "consensus", date_iso)


def run_covers_props_mlb(date_iso: str, _sports: list[str] | None = None) -> dict[str, Any]:
    return scrape_covers("mlb", "props", date_iso)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Covers.com picks by official matchup.")
    parser.add_argument("--sport", default="mlb", choices=sorted(SPORT_CONFIG))
    parser.add_argument("--engine", default="experts", choices=ENGINES)
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    args = parser.parse_args()
    result = scrape_covers(args.sport, args.engine, args.date)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
