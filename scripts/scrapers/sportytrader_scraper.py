#!/usr/bin/env python3
"""
SportyTrader Scraper
====================
    Scrapes NBA, NBA Summer League, WNBA, MLB, and FIFA World Cup picks from
SportyTrader and prints structured pick blocks for the backend parser.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from datetime import datetime


def _default_playwright_browsers_path() -> str:
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if configured:
        return configured
    darwin_cache = os.path.expanduser("~/Library/Caches/ms-playwright")
    if sys.platform == "darwin" and os.path.isdir(darwin_cache):
        return darwin_cache
    return ""


_playwright_browsers_path = _default_playwright_browsers_path()
if _playwright_browsers_path:
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = _playwright_browsers_path

from playwright.sync_api import TimeoutError as PwTimeout
from playwright.sync_api import sync_playwright

SPORT_CONFIG = {
    "nba": {
        "aliases": {"nba", "basketball"},
        "league": "USA - NBA",
        "league_aliases": {"USA - NBA"},
        "title": "NBA",
        "url": "https://www.sportytrader.com/us/picks/basketball/usa/nba-306/",
    },
    "nba_summer": {
        "aliases": {"nba_summer", "nba_summer_league", "summer_league"},
        "league": "USA - NBA Summer League",
        # SportyTrader can group Summer League cards on its NBA and general
        # basketball listings. The official matchup whitelist below keeps
        # ordinary NBA cards out of this bucket.
        "league_aliases": {
            "USA - NBA",
            "USA - NBA Summer League",
            "NBA Summer League",
            "Summer League",
        },
        "title": "NBA Summer League",
        "url": "https://www.sportytrader.com/us/picks/basketball/usa/nba-306/",
        "fallback_urls": ("https://www.sportytrader.com/us/picks/basketball/",),
    },
    "wnba": {
        "aliases": {"wnba"},
        "league": "USA - WNBA",
        "league_aliases": {"USA - WNBA"},
        "title": "WNBA",
        "url": "https://www.sportytrader.com/us/picks/basketball/usa/wnba-58202/",
    },
    "mlb": {
        "aliases": {"mlb", "baseball"},
        "league": "USA - MLB",
        "league_aliases": {"USA - MLB"},
        "title": "MLB",
        "url": "https://www.sportytrader.com/us/picks/baseball/usa/mlb-597/",
    },
    "fifa_world_cup": {
        "aliases": {"fifa", "fifa_world_cup", "football", "soccer", "world_cup"},
        "league": "World - World Cup",
        "league_aliases": {"World - World Cup"},
        "title": "FIFA World Cup",
        "url": "https://www.sportytrader.com/us/picks/soccer/world/world-cup-1811/",
    },
}

SPORT_ALIAS_MAP = {
    alias: key
    for key, cfg in SPORT_CONFIG.items()
    for alias in cfg["aliases"]
}

CLOUDFLARE_SIGNALS = (
    "just a moment",
    "performing security verification",
    "enable javascript and cookies to continue",
    "privacy pass",
    "cloudflare",
)

SPORTYTRADER_CARDS_JS = r"""
() => {
    const cards = Array.from(document.querySelectorAll('.pronostics-wrapper .card'));
    return cards.map((card) => {
        const text = (value) => (value || '').replace(/\s+/g, ' ').trim();
        const hrefFromCard = card.getAttribute('data-navigation-url-value') || '';
        const anchors = Array.from(card.querySelectorAll('a[href*="/us/picks/"]'));
        const pickAnchor = anchors.find((anchor) => /picks$/i.test(text(anchor.textContent))) || anchors[0] || null;
        const href = pickAnchor
            ? pickAnchor.href
            : (hrefFromCard ? new URL(hrefFromCard, window.location.origin).href : '');

        const teams = Array.from(card.querySelectorAll('span.font-semibold'))
            .map((node) => text(node.textContent))
            .filter(Boolean)
            .filter((team, index, values) => index === 0 || team !== values[index - 1]);
        const paragraphs = Array.from(card.querySelectorAll('p'))
            .map((node) => text(node.textContent))
            .filter(Boolean);
        const dateText = paragraphs.find((line) => /\b[A-Z][a-z]{2,8}\s+\d{1,2},\s+\d{4},\s+\d{1,2}:\d{2}/.test(line)) || '';
        const league = paragraphs.find((line) => /\b(?:NBA Summer League|Summer League|NBA|WNBA|MLB|World Cup)\b/i.test(line)) || '';
        const tipNode = card.querySelector('.bg-gray-100 p.font-semibold');
        const tip = text(tipNode ? tipNode.textContent : '');

        let odds = '';
        for (const node of card.querySelectorAll('.bg-gray-100 span')) {
            const value = text(node.textContent);
            if (/^[+-]?\d+(?:\.\d+)?$/.test(value)) {
                odds = value;
                break;
            }
        }

        return {
            datetime: dateText,
            league,
            home: teams[0] || '',
            away: teams[1] || '',
            tip,
            odds,
            href,
        };
    }).filter((row) => row.home && row.away && row.tip && row.href);
}
"""

SPORTYTRADER_SUMMER_LINKS_JS = r"""
() => Array.from(document.querySelectorAll('a[href]'))
    .map((anchor) => ({
        href: anchor.href || '',
        label: (anchor.textContent || '').replace(/\s+/g, ' ').trim(),
    }))
    .filter(({ href, label }) => {
        const blob = `${href} ${label}`.toLowerCase();
        return href.startsWith('https://www.sportytrader.com/us/picks/basketball/')
            && blob.includes('summer');
    })
    .map(({ href }) => href)
"""


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line or "").strip()


def _parse_target_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        return None


def _parse_english_datetime(text: str) -> datetime | None:
    compact = _normalize_line(text)
    for fmt in ("%b %d, %Y, %I:%M %p", "%B %d, %Y, %I:%M %p"):
        try:
            return datetime.strptime(compact, fmt)
        except ValueError:
            continue
    return None


def _normalize_sport(raw: str) -> str:
    key = _normalize_line(raw).lower()
    if key in SPORT_ALIAS_MAP:
        return SPORT_ALIAS_MAP[key]
    return ""


def _looks_like_cloudflare_block(title: str, body_text: str) -> bool:
    blob = f"{title}\n{body_text}".lower()
    return any(signal in blob for signal in CLOUDFLARE_SIGNALS)

def _matchup_key(raw: str) -> tuple[str, str] | None:
    teams = re.split(r"\s+(?:vs\.?|@)\s+", _normalize_line(raw), maxsplit=1, flags=re.IGNORECASE)
    if len(teams) != 2:
        return None
    aliases = {"turkiye": "turkey"}
    normalized = []
    for team in teams:
        text = unicodedata.normalize("NFKD", team)
        text = "".join(char for char in text if not unicodedata.combining(char))
        key = re.sub(r"[^a-z0-9]+", "", text.lower())
        normalized.append(aliases.get(key, key))
    normalized.sort()
    return (normalized[0], normalized[1]) if all(normalized) else None


def _launch_browser(pw):
    launch_kwargs = {
        "headless": True,
        "args": [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ],
    }
    preferred_channel = os.environ.get("SPORTYTRADER_BROWSER_CHANNEL", "chrome").strip()
    if preferred_channel:
        try:
            return pw.chromium.launch(channel=preferred_channel, **launch_kwargs)
        except Exception:
            pass
    return pw.chromium.launch(**launch_kwargs)


def _new_context(browser):
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        viewport={"width": 1365, "height": 900},
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    ctx.add_init_script(
        """
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
Object.defineProperty(navigator, 'platform', { get: () => 'MacIntel' });
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
window.chrome = window.chrome || { runtime: {} };
"""
    )
    return ctx


def _load_cards(page, url: str) -> tuple[list[dict[str, str]], str]:
    page.goto(url, timeout=45000, wait_until="domcontentloaded")
    last_title = ""
    last_text = ""
    for attempt in range(4):
        page.wait_for_timeout(3000 + (attempt * 700))
        last_title = page.title()
        last_text = page.evaluate("() => document.body?.innerText || ''")
        if not _looks_like_cloudflare_block(last_title, last_text):
            cards = page.evaluate(SPORTYTRADER_CARDS_JS)
            if isinstance(cards, list):
                return cards, last_text
        if attempt < 3:
            page.reload(timeout=45000, wait_until="domcontentloaded")
    return [], last_text


def _discover_summer_listing_urls(page) -> list[str]:
    try:
        urls = page.evaluate(SPORTYTRADER_SUMMER_LINKS_JS)
    except Exception:
        return []
    if not isinstance(urls, list):
        return []
    return list(
        dict.fromkeys(
            str(url).strip()
            for url in urls
            if str(url).startswith("https://www.sportytrader.com/us/picks/basketball/")
            and "summer" in str(url).lower()
        )
    )


def _extract_rows(
    cards: list[dict[str, str]],
    target_date: datetime | None,
    sport_key: str,
    expected_matchups: list[str] | None = None,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    sport_config = SPORT_CONFIG[sport_key]
    sport_league = sport_config["league"]
    league_aliases = sport_config.get("league_aliases", {sport_league})
    expected = {
        key: matchup
        for matchup in expected_matchups or []
        if (key := _matchup_key(matchup))
    }

    for raw in cards:
        row = {key: _normalize_line(str(raw.get(key, ""))) for key in ("datetime", "league", "home", "away", "tip", "odds", "href")}
        if row["league"] and row["league"] not in league_aliases:
            continue
        if not row["home"] or not row["away"] or not row["tip"]:
            continue
        dt = _parse_english_datetime(row["datetime"])
        matchup_key = _matchup_key(f"{row['home']} vs {row['away']}")
        if expected and matchup_key not in expected:
            continue
        if not expected and target_date and dt and dt.date() != target_date.date():
            continue
        key = (row["datetime"], row["home"], row["away"], row["tip"])
        if key in seen:
            continue
        seen.add(key)
        row["league"] = sport_league
        out.append(row)

    return out


def _print_pick(row: dict[str, str]) -> None:
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Match:          {row['home']} vs {row['away']}")
    print(f"Date/Time:      {row['datetime']}")
    print(f"League:         {row['league']}")
    print(f"Tip:            {row['tip']}")
    print(f"Odds:           {row['odds'] or '[not found on page]'}")
    print("Confidence:     [not found on page]")
    print("User vote:      [not found on page]")
    print(f"Source URL:     {row['href']}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main() -> None:
    ap = argparse.ArgumentParser(description="SportyTrader scraper")
    ap.add_argument(
        "--sport",
        "-s",
        default="nba",
        help="Supported: nba/nba_summer/wnba/mlb/fifa_world_cup",
    )
    ap.add_argument("--date", "-d", help="Date in YYYY-MM-DD")
    ap.add_argument("--expected-matchup", action="append", default=[])
    args = ap.parse_args()

    expected_matchups = [
        _normalize_line(matchup)
        for matchup in args.expected_matchup
        if _matchup_key(matchup)
    ]
    if not expected_matchups or len(expected_matchups) != len(args.expected_matchup):
        print("Error: SportyTrader requires a valid official matchup whitelist.")
        sys.exit(1)

    sport_key = _normalize_sport(args.sport or "")
    if not sport_key:
        print(
            "Error: SportyTrader scraper supports NBA/basketball, NBA Summer League, "
            "WNBA, MLB/baseball, and FIFA World Cup/soccer."
        )
        sys.exit(1)

    target_date = _parse_target_date(args.date)
    sport_config = SPORT_CONFIG[sport_key]
    target_urls = list(
        dict.fromkeys(
            [sport_config["url"], *sport_config.get("fallback_urls", ())]
        )
    )
    target_title = sport_config["title"]

    with sync_playwright() as pw:
        browser = _launch_browser(pw)
        ctx = _new_context(browser)
        page = ctx.new_page()
        cards: list[dict[str, str]] = []
        page_texts: list[str] = []
        load_errors: list[str] = []
        try:
            visited_urls: set[str] = set()
            target_index = 0
            while target_index < len(target_urls) and len(visited_urls) < 8:
                target_url = target_urls[target_index]
                target_index += 1
                if target_url in visited_urls:
                    continue
                visited_urls.add(target_url)
                try:
                    page_cards, page_text = _load_cards(page, target_url)
                except PwTimeout:
                    load_errors.append(f"timed out loading {target_url}")
                    continue
                except Exception as exc:
                    load_errors.append(f"{target_url}: {exc}")
                    continue
                cards.extend(page_cards)
                page_texts.append(page_text)
                if sport_key == "nba_summer":
                    for discovered_url in _discover_summer_listing_urls(page):
                        if discovered_url not in visited_urls:
                            target_urls.append(discovered_url)
        finally:
            page.close()
            browser.close()

    if not page_texts:
        detail = "; ".join(load_errors[:2]) or "no listing page loaded"
        print(f"Error loading SportyTrader {target_title} page: {detail}")
        sys.exit(1)
    if page_texts and all(_looks_like_cloudflare_block("", text) for text in page_texts):
        print(f"Error: SportyTrader {target_title} page hit Cloudflare verification")
        sys.exit(1)
    blocked_page_count = sum(
        1 for text in page_texts if _looks_like_cloudflare_block("", text)
    )
    if load_errors or blocked_page_count:
        detail = "; ".join(load_errors[:2])
        if blocked_page_count:
            prefix = f"{detail}; " if detail else ""
            detail = f"{prefix}{blocked_page_count} listing page(s) were blocked"
        print(
            f"Error: incomplete SportyTrader {target_title} listing coverage: {detail}"
        )
        sys.exit(1)

    try:
        rows = _extract_rows(cards, target_date, sport_key, expected_matchups)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    if not rows:
        print(f"No SportyTrader {target_title} picks parsed.")
        expected_keys = {
            key
            for matchup in expected_matchups
            if (key := _matchup_key(matchup))
        }
        official_card_count = sum(
            1
            for card in cards
            if _matchup_key(f"{card.get('home', '')} vs {card.get('away', '')}")
            in expected_keys
        )
        print(
            f"Diagnostics: listingPages={len(page_texts)} cards={len(cards)} "
            f"officialMatchupCards={official_card_count}."
        )
        return

    for row in rows:
        _print_pick(row)


if __name__ == "__main__":
    main()
