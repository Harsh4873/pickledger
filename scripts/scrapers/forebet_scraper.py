#!/usr/bin/env python3
"""Scrape Forebet 1X2 predictions by official slate matchup.

Forebet renders predictions as div-based rows (`div.rcnt`), not a table.
Each row decodes as:

  .stcn   league short code            .fprc    "44 34 22" = P(home/draw/away) %
  .tnms   teams + kickoff (schema.org) .forepr  1X2 tip: 1=home, X=draw, 2=away
  .ex_sc  predicted correct score      .avg_sc  expected total goals
  .haodd  American odds [home, draw, away] ("no"/"-" when unposted)
  .lmin_td match status (minute/FT/Postp.)  .lscr_td live/final score

Rows are matched against the official ESPN slate for the target date (same
identity-based convention as the Scores24 scraper); grading then runs through
the standard ESPN auto-grade path, so Forebet's own score cells are unused.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.scrapers.scores24_scraper import (  # noqa: E402
    CLOUDFLARE_SIGNALS,
    _soccer_market_metadata,
    _team_matches,
    fetch_daily_matchups,
)


BASE_URL = "https://www.forebet.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
SPORT_CONFIG = {
    "mls": {
        "espn_sport": "soccer",
        "espn_league": "usa.1",
        "listing_url": f"{BASE_URL}/en/football-tips-and-predictions-for-usa/mls",
        "source": "ForebetMLS",
        "label": "MLS",
        "cache_keys": ("mls",),
    },
}
SIGN_INDEX = {"1": 0, "X": 1, "2": 2}
SCORE_RE = re.compile(r"(\d{1,2})\s*-\s*(\d{1,2})")
AMERICAN_ODDS_RE = re.compile(r"^[+-]\d+$")


def _cell_text(row: Any, selector: str) -> str:
    node = row.select_one(selector)
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)) if node else ""


def _american_odds(token: str) -> int | None:
    token = str(token or "").strip()
    return int(token) if AMERICAN_ODDS_RE.match(token) else None


def parse_forebet_rows(html: str) -> list[dict[str, Any]]:
    """Extract normalized prediction rows from a Forebet listing page."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, Any]] = []
    for row in soup.select("div.rcnt"):
        home = _cell_text(row, ".homeTeam [itemprop=name]")
        away = _cell_text(row, ".awayTeam [itemprop=name]")
        sign = _cell_text(row, ".forepr span")
        if not home or not away or sign not in SIGN_INDEX:
            continue

        prob_ints = [int(tok) for tok in re.findall(r"\d{1,3}", _cell_text(row, ".fprc"))]
        probs: list[int | None] = (
            list(prob_ints[:3]) if len(prob_ints) >= 3 and 90 <= sum(prob_ints[:3]) <= 110 else [None, None, None]
        )

        odds: list[int | None] = [None, None, None]
        haodd = row.select_one(".haodd")
        if haodd:
            tokens = [span.get_text(strip=True) for span in haodd.find_all("span")][:3]
            odds = [_american_odds(token) for token in tokens] + [None] * (3 - len(tokens))

        predicted = SCORE_RE.search(_cell_text(row, ".ex_sc"))
        avg_goals = re.search(r"\d+\.\d+", _cell_text(row, ".avg_sc"))
        rows.append(
            {
                "home": home,
                "away": away,
                "sign": sign,
                "prob_home": probs[0],
                "prob_draw": probs[1],
                "prob_away": probs[2],
                "odds_home": odds[0],
                "odds_draw": odds[1],
                "odds_away": odds[2],
                "predicted_score": f"{predicted.group(1)}-{predicted.group(2)}" if predicted else None,
                "avg_goals": float(avg_goals.group()) if avg_goals else None,
                "kickoff": _cell_text(row, ".date_bah"),
            }
        )
    return rows


def _fetch_listing_html(url: str) -> tuple[str, str]:
    """Return (html, error). Forebet is server-rendered; a plain fetch suffices."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as exc:
        return "", f"listing fetch failed: {exc}"
    lowered = response.text.lower()
    if any(signal in lowered for signal in CLOUDFLARE_SIGNALS):
        return "", "listing fetch blocked by Cloudflare"
    if response.status_code != 200:
        return "", f"listing fetch returned HTTP {response.status_code}"
    return response.text, ""


def _match_row(matchup: dict[str, str], rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Strict orientation: the 1X2 sign is only meaningful relative to the home
    # side, so a swapped-orientation match must not count.
    for row in rows:
        if _team_matches(matchup["home"], row["home"]) and _team_matches(matchup["away"], row["away"]):
            return row
    return None


def _pick_payload(
    config: dict[str, Any],
    date_iso: str,
    matchup: dict[str, str],
    row: dict[str, Any],
) -> dict[str, Any]:
    sign = row["sign"]
    tip = {"1": f"{matchup['home']} ML", "X": "Draw", "2": f"{matchup['away']} ML"}[sign]
    matchup_label = f"{matchup['away']} @ {matchup['home']}"
    probability = (row["prob_home"], row["prob_draw"], row["prob_away"])[SIGN_INDEX[sign]]
    payload = {
        "source": config["source"],
        "pick": f"{tip} ({matchup_label})",
        "tip": tip,
        "sport": config["label"],
        "odds": (row["odds_home"], row["odds_draw"], row["odds_away"])[SIGN_INDEX[sign]],
        "units": 1,
        "probability": round(probability / 100.0, 4) if probability is not None else None,
        "edge": None,
        "decision": "BET",
        "date": date_iso,
        "matchup": matchup_label,
        "game": matchup_label,
        "away_team": matchup["away"],
        "home_team": matchup["home"],
        "start_time": matchup.get("start_time") or None,
        "source_url": config["listing_url"],
        "forebet_prob_home": row["prob_home"],
        "forebet_prob_draw": row["prob_draw"],
        "forebet_prob_away": row["prob_away"],
        "forebet_predicted_score": row["predicted_score"],
        "forebet_avg_goals": row["avg_goals"],
        "calibration_excluded": True,
    }
    payload.update(_soccer_market_metadata(payload["pick"]))
    return payload


def scrape_forebet(sport: str, date_iso: str, *, html: str | None = None) -> dict[str, Any]:
    sport_key = sport.strip().lower()
    if sport_key not in SPORT_CONFIG:
        raise ValueError(f"unsupported Forebet sport: {sport}")
    config = SPORT_CONFIG[sport_key]
    expected, slate_resolved = fetch_daily_matchups(sport_key, date_iso, config=config)
    if not slate_resolved:
        return {
            "ok": False,
            "date": date_iso,
            "picks": [],
            "error": f"{config['source']} could not resolve an official {date_iso} slate",
        }
    if not expected:
        return {
            "ok": True,
            "date": date_iso,
            "picks": [],
            "note": f"{config['source']} has no official {date_iso} matchups.",
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

    blocked = 0
    if html is None:
        html, fetch_error = _fetch_listing_html(config["listing_url"])
        if fetch_error:
            blocked = int("Cloudflare" in fetch_error)
            return {
                "ok": False,
                "date": date_iso,
                "picks": [],
                "error": f"{config['source']}: {fetch_error}",
                "meta": {
                    "officialMatchups": len(expected),
                    "expectedMatchups": 0,
                    "matchedPicks": 0,
                    "missingMatchups": [f"{m['away']} @ {m['home']}" for m in expected],
                    "unpublishedMatchups": [],
                    "attemptedUrls": 1,
                    "blockedUrls": blocked,
                },
            }

    rows = parse_forebet_rows(html)
    picks: list[dict[str, Any]] = []
    unpublished: list[str] = []
    for matchup in expected:
        row = _match_row(matchup, rows)
        if row is None:
            unpublished.append(f"{matchup['away']} @ {matchup['home']}")
            continue
        picks.append(_pick_payload(config, date_iso, matchup, row))

    return {
        "ok": True,
        "date": date_iso,
        "picks": picks,
        "note": (
            f"{config['source']} matched {len(picks)} published prediction(s) "
            f"against {len(expected)} official {date_iso} matchup(s)."
        ),
        "meta": {
            "officialMatchups": len(expected),
            "expectedMatchups": len(picks),
            "matchedPicks": len(picks),
            "missingMatchups": [],
            "unpublishedMatchups": unpublished,
            "attemptedUrls": 1,
            "blockedUrls": blocked,
            "listedRows": len(rows),
        },
    }


def run_forebet_mls(date_iso: str, _sports: list[str] | None = None) -> dict[str, Any]:
    return scrape_forebet("mls", date_iso)


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape Forebet 1X2 predictions by official matchup.")
    parser.add_argument("--sport", default="mls", choices=sorted(SPORT_CONFIG))
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--html-file", default="", help="Parse a saved listing page instead of fetching.")
    args = parser.parse_args()
    html = None
    if args.html_file:
        with open(args.html_file, encoding="utf-8") as handle:
            html = handle.read()
    result = scrape_forebet(args.sport, args.date, html=html)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
