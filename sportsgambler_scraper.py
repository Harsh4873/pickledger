#!/usr/bin/env python3
"""
SportsGambler Scraper
=====================
Scrapes NBA and MLB picks from SportsGambler and prints SportyTrader-style
blocks so the existing backend parser pattern can consume them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

SPORT_CONFIG = {
    "nba": {
        "aliases": {"nba", "basketball"},
        "league": "NBA",
        "title": "NBA",
        "url": "https://www.sportsgambler.com/betting-tips/basketball/nba-predictions/",
    },
    "mlb": {
        "aliases": {"mlb", "baseball"},
        "league": "MLB",
        "title": "MLB",
        "url": "https://www.sportsgambler.com/betting-tips/baseball/",
    },
}

SPORT_ALIAS_MAP = {
    alias: key
    for key, cfg in SPORT_CONFIG.items()
    for alias in cfg["aliases"]
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _normalize_line(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_sport(raw: str) -> str:
    key = _normalize_line(raw).lower()
    if key in SPORT_ALIAS_MAP:
        return SPORT_ALIAS_MAP[key]
    return ""


def _parse_target_date(raw: str | None) -> date | None:
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _fetch_html(url: str) -> str:
    resp = requests.get(url, headers=REQUEST_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def _iter_json_nodes(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_json_nodes(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_nodes(item)


def _json_ld_objects(soup: BeautifulSoup) -> list[Any]:
    parsed: list[Any] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text("\n", strip=True)
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return parsed


def _extract_iso_date(raw: str) -> date | None:
    text = _normalize_line(raw)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d").date()
    except ValueError:
        return None


def _extract_listing_matchup(item: dict[str, Any]) -> str:
    event_name = _normalize_line(item.get("name", ""))
    if event_name:
        return event_name

    competitors = item.get("competitor")
    if isinstance(competitors, list):
        teams = [
            _normalize_line(team.get("name", ""))
            for team in competitors
            if isinstance(team, dict) and _normalize_line(team.get("name", ""))
        ]
        if len(teams) >= 2:
            return f"{teams[0]} vs {teams[1]}"
    return ""


def _split_tip_and_odds(raw_prediction: str) -> tuple[str, str]:
    text = _normalize_line(raw_prediction)
    match = re.match(r"^(.*?)\s*@\s*([+-]?\d+(?:\.\d+)?)$", text)
    if not match:
        return text, ""
    return _normalize_line(match.group(1)), _normalize_line(match.group(2))


def _extract_detail_matchup(soup: BeautifulSoup) -> str:
    for obj in _json_ld_objects(soup):
        for node in _iter_json_nodes(obj):
            if node.get("@type") == "SportsEvent":
                matchup = _extract_listing_matchup(node)
                if matchup:
                    return matchup

    header = soup.select_one("h1.p_title")
    if not header:
        return ""
    text = _normalize_line(header.get_text(" ", strip=True))
    match = re.match(r"^(.*?)\s+Prediction\b", text, re.IGNORECASE)
    return _normalize_line(match.group(1) if match else text)


def _extract_nba_article_rows(target_date: date | None) -> list[dict[str, str]]:
    html = _fetch_html(SPORT_CONFIG["nba"]["url"])
    soup = BeautifulSoup(html, "html.parser")

    articles: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for obj in _json_ld_objects(soup):
        for node in _iter_json_nodes(obj):
            item = node.get("item")
            if not isinstance(item, dict) or item.get("@type") != "SportsEvent":
                continue

            url = _normalize_line(item.get("url", ""))
            matchup = _extract_listing_matchup(item)
            start_date_raw = _normalize_line(item.get("startDate", ""))
            event_date = _extract_iso_date(start_date_raw)

            if not url or not matchup or url in seen_urls:
                continue
            if target_date and event_date and event_date != target_date:
                continue

            seen_urls.add(url)
            articles.append({
                "url": url,
                "matchup": matchup,
                "datetime": start_date_raw or (event_date.isoformat() if event_date else ""),
            })

    rows: list[dict[str, str]] = []
    for article in articles:
        detail_html = _fetch_html(article["url"])
        detail_soup = BeautifulSoup(detail_html, "html.parser")

        prediction = ""
        for container in detail_soup.select("div.tpbot_container"):
            label = _normalize_line(container.select_one(".tpbot_title").get_text(" ", strip=True)) if container.select_one(".tpbot_title") else ""
            if "prediction" not in label.lower():
                continue
            spans = container.select("a.tpbot_tip span")
            if spans:
                prediction = _normalize_line(spans[-1].get_text(" ", strip=True))
                break

        if not prediction:
            continue

        matchup = _extract_detail_matchup(detail_soup) or article["matchup"]
        tip, odds = _split_tip_and_odds(prediction)
        if not matchup or not tip:
            continue

        rows.append({
            "datetime": article["datetime"],
            "league": "NBA",
            "matchup": matchup,
            "tip": tip,
            "odds": odds,
            "href": article["url"],
        })

    return rows


def _parse_listing_date(raw: str, target_date: date | None) -> date | None:
    text = _normalize_line(raw)
    if not text:
        return None

    year = (target_date or date.today()).year
    try:
        return datetime.strptime(f"{text} {year}", "%H:%M %a %d/%m %Y").date()
    except ValueError:
        return None


def _extract_mlb_rows(target_date: date | None) -> list[dict[str, str]]:
    html = _fetch_html(SPORT_CONFIG["mlb"]["url"])
    soup = BeautifulSoup(html, "html.parser")

    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in soup.select("div.tipbox_item"):
        title_spans = item.select(".tipsbox_title h3 > span")
        matchup = _normalize_line(title_spans[0].get_text(" ", strip=True)) if title_spans else ""
        meta_spans = item.select(".tipsbox_title .tipsbox_meta span")
        date_text = _normalize_line(meta_spans[0].get_text(" ", strip=True)) if meta_spans else ""
        league_text = _normalize_line(" ".join(span.get_text(" ", strip=True) for span in meta_spans[1:]))
        league_text = league_text.lstrip("-").strip()

        if league_text.upper() != "MLB":
            continue
        if target_date:
            item_date = _parse_listing_date(date_text, target_date)
            if item_date and item_date != target_date:
                continue

        tip_spans = item.select(".tipbox_tip span")
        prediction = _normalize_line(tip_spans[-1].get_text(" ", strip=True)) if tip_spans else ""
        tip, odds = _split_tip_and_odds(prediction)
        if not matchup or not tip:
            continue

        key = (matchup, tip)
        if key in seen:
            continue
        seen.add(key)

        anchor = item.get("id", "").strip()
        href = SPORT_CONFIG["mlb"]["url"] + (f"#{anchor}" if anchor else "")

        rows.append({
            "datetime": date_text,
            "league": "MLB",
            "matchup": matchup,
            "tip": tip,
            "odds": odds,
            "href": href,
        })

    return rows


def _print_pick(row: dict[str, str]) -> None:
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Match:          {row['matchup']}")
    print(f"Date/Time:      {row['datetime'] or '[not found on page]'}")
    print(f"League:         {row['league']}")
    print(f"Tip:            {row['tip']}")
    print(f"Odds:           {row['odds'] or '[not found on page]'}")
    print("Confidence:     [not found on page]")
    print("User vote:      [not found on page]")
    print(f"Source URL:     {row['href']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main() -> None:
    ap = argparse.ArgumentParser(description="SportsGambler scraper")
    ap.add_argument("--sport", "-s", default="nba", help="Supported: nba/mlb")
    ap.add_argument("--date", "-d", help="Date in YYYY-MM-DD or MM/DD/YYYY")
    args = ap.parse_args()

    sport_key = _normalize_sport(args.sport or "")
    if not sport_key:
        print("Error: SportsGambler scraper supports only NBA/basketball and MLB/baseball.")
        sys.exit(1)

    target_date = _parse_target_date(args.date)

    try:
        if sport_key == "nba":
            rows = _extract_nba_article_rows(target_date)
        else:
            rows = _extract_mlb_rows(target_date)
    except requests.RequestException as exc:
        print(f"Error loading SportsGambler {SPORT_CONFIG[sport_key]['title']} page: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"Error parsing SportsGambler {SPORT_CONFIG[sport_key]['title']} page: {exc}")
        sys.exit(1)

    if not rows:
        print(f"No SportsGambler {SPORT_CONFIG[sport_key]['title']} picks parsed.")
        return

    for row in rows:
        _print_pick(row)


if __name__ == "__main__":
    main()
