#!/usr/bin/env python3
"""SportsGambler scraper — NBA and MLB picks."""
from __future__ import annotations
import argparse, json, re, sys
from datetime import date, datetime
from typing import Any
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

NBA_URL = "https://www.sportsgambler.com/betting-tips/basketball/nba-predictions/"
MLB_URL = "https://www.sportsgambler.com/betting-tips/baseball/"

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _parse_date(raw: str) -> date | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None

def _split_tip_odds(raw: str) -> tuple[str, str]:
    m = re.match(r"^(.*?)\s*@\s*([+-]?\d+(?:\.\d+)?)$", _norm(raw))
    return (_norm(m.group(1)), _norm(m.group(2))) if m else (_norm(raw), "")

def _json_ld(soup: BeautifulSoup) -> list[Any]:
    out = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = (tag.string or tag.get_text("\n", strip=True)).strip()
        if raw:
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                pass
    return out

def _iter_nodes(v: Any):
    if isinstance(v, dict):
        yield v
        for child in v.values():
            yield from _iter_nodes(child)
    elif isinstance(v, list):
        for item in v:
            yield from _iter_nodes(item)

def _iso_date(raw: str) -> date | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", _norm(raw))
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d").date() if m else None
    except ValueError:
        return None

def _matchup_from_node(node: dict) -> str:
    name = _norm(node.get("name", ""))
    if name:
        return name
    comps = node.get("competitor")
    if isinstance(comps, list):
        teams = [_norm(t.get("name", "")) for t in comps if isinstance(t, dict) and _norm(t.get("name", ""))]
        if len(teams) >= 2:
            return f"{teams[0]} vs {teams[1]}"
    return ""

def scrape_nba(target: date | None) -> list[dict]:
    html = requests.get(NBA_URL, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    articles, seen = [], set()
    for obj in _json_ld(soup):
        for node in _iter_nodes(obj):
            item = node.get("item")
            if not isinstance(item, dict) or item.get("@type") != "SportsEvent":
                continue
            url = _norm(item.get("url", ""))
            matchup = _matchup_from_node(item)
            event_date = _iso_date(_norm(item.get("startDate", "")))
            if not url or not matchup or url in seen:
                continue
            if target and event_date and event_date != target:
                continue
            seen.add(url)
            articles.append({"url": url, "matchup": matchup, "date": item.get("startDate", "")})

    rows = []
    for art in articles:
        try:
            detail = BeautifulSoup(requests.get(art["url"], headers=HEADERS, timeout=30).text, "html.parser")
        except Exception:
            continue
        prediction = ""
        for cont in detail.select("div.tpbot_container"):
            title_el = cont.select_one(".tpbot_title")
            if title_el and "prediction" in title_el.get_text().lower():
                spans = cont.select("a.tpbot_tip span")
                if spans:
                    prediction = _norm(spans[-1].get_text(" ", strip=True))
                    break
        if not prediction:
            continue
        tip, odds = _split_tip_odds(prediction)
        if not tip:
            continue
        rows.append({"datetime": art["date"], "league": "NBA", "matchup": art["matchup"], "tip": tip, "odds": odds, "href": art["url"]})
    return rows

def scrape_mlb(target: date | None) -> list[dict]:
    html = requests.get(MLB_URL, headers=HEADERS, timeout=30).text
    soup = BeautifulSoup(html, "html.parser")
    rows, seen = [], set()
    for item in soup.select("div.tipbox_item"):
        title_spans = item.select(".tipsbox_title h3 > span")
        matchup = _norm(title_spans[0].get_text(" ", strip=True)) if title_spans else ""
        meta_spans = item.select(".tipsbox_title .tipsbox_meta span")
        date_text = _norm(meta_spans[0].get_text(" ", strip=True)) if meta_spans else ""
        league_text = _norm(" ".join(s.get_text(" ", strip=True) for s in meta_spans[1:])).lstrip("-").strip()
        if league_text.upper() != "MLB":
            continue
        tip_spans = item.select(".tipbox_tip span")
        prediction = _norm(tip_spans[-1].get_text(" ", strip=True)) if tip_spans else ""
        tip, odds = _split_tip_odds(prediction)
        if not matchup or not tip:
            continue
        key = (matchup, tip)
        if key in seen:
            continue
        seen.add(key)
        anchor = item.get("id", "").strip()
        href = MLB_URL + (f"#{anchor}" if anchor else "")
        rows.append({"datetime": date_text, "league": "MLB", "matchup": matchup, "tip": tip, "odds": odds, "href": href})
    return rows

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", "-s", default="nba")
    ap.add_argument("--date", "-d", default=None)
    args = ap.parse_args()
    sport = args.sport.strip().lower()
    target = _parse_date(args.date) if args.date else None
    try:
        rows = scrape_nba(target) if sport in ("nba", "basketball") else scrape_mlb(target)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if not rows:
        print("No picks found.")
        sys.exit(0)
    for r in rows:
        print("\n" + "━" * 32)
        print(f"Match:          {r['matchup']}")
        print(f"Date/Time:      {r['datetime'] or '[not found]'}")
        print(f"League:         {r['league']}")
        print(f"Tip:            {r['tip']}")
        print(f"Odds:           {r['odds'] or '[not found]'}")
        print(f"Source URL:     {r['href']}")
        print("━" * 32)

if __name__ == "__main__":
    main()
