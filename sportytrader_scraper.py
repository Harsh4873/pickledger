#!/usr/bin/env python3
"""
SportyTrader NBA Scraper
========================
Scrapes NBA picks from SportyTrader and prints Scores24-style blocks so
existing backend parsers can consume them.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime

def _default_playwright_browsers_path() -> str:
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if configured:
        return configured
    darwin_cache = os.path.expanduser("~/Library/Caches/ms-playwright")
    if sys.platform == "darwin" and os.path.isdir(darwin_cache):
        return darwin_cache
    return "0"


os.environ["PLAYWRIGHT_BROWSERS_PATH"] = _default_playwright_browsers_path()

from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

SPORTYTRADER_URL = "https://www.sportytrader.com/us/picks/basketball/usa/nba-306/"

FRENCH_MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    "décembre": 12,
}

NOISE_LINES = {
    "en détails",
    "pariez maintenant !",
    "offre exclusive",
    "pronostics",
    "basket",
    "live",
    "pronostic basket",
    "detail",
    "bet now!",
}


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line or "").strip()


def _normalize_for_cmp(text: str) -> str:
    out = (text or "").lower().strip()
    out = out.replace("’", "'")
    return re.sub(r"\s+", " ", out)


def _parse_target_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        return None


def _parse_french_datetime(text: str) -> datetime | None:
    m = re.match(r"^(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4}),\s*(\d{2}):(\d{2})$", text)
    if not m:
        return None
    day = int(m.group(1))
    month_name = m.group(2).lower()
    year = int(m.group(3))
    hour = int(m.group(4))
    minute = int(m.group(5))
    month = FRENCH_MONTHS.get(month_name)
    if not month:
        return None
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def _parse_english_datetime(text: str) -> datetime | None:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    for fmt in ("%b %d, %Y, %I:%M %p", "%B %d, %Y, %I:%M %p"):
        try:
            return datetime.strptime(compact, fmt)
        except ValueError:
            continue
    return None


def _parse_sportytrader_datetime(text: str) -> datetime | None:
    return _parse_french_datetime(text) or _parse_english_datetime(text)


def _looks_like_team(line: str) -> bool:
    if not line:
        return False
    low = _normalize_for_cmp(line)
    if low in NOISE_LINES:
        return False
    if "pronostic " in low:
        return False
    if re.search(r"\d{1,2}:\d{2}", line):
        return False
    if re.search(r"\d{4}", line):
        return False
    if "nba" in low and ("etats-unis" in low or "usa" in low):
        return False
    return bool(re.search(r"[A-Za-zÀ-ÿ]", line))


def _is_noise(line: str) -> bool:
    low = _normalize_for_cmp(line)
    if not low:
        return True
    if low in NOISE_LINES:
        return True
    if low.startswith("cotes décimales") or low.startswith("18+"):
        return True
    if low.startswith("offre exclusive"):
        return True
    if low.startswith("stake"):
        return True
    if low.startswith("bonus up to"):
        return True
    if low.startswith("new customers only"):
        return True
    if low.startswith("commercial content"):
        return True
    if low.startswith("odds"):
        return True
    return False


def _extract_blocks(page_text: str, target_date: datetime | None) -> list[dict[str, str]]:
    lines = [_normalize_line(ln) for ln in (page_text or "").splitlines()]
    lines = [ln for ln in lines if ln]
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for i, line in enumerate(lines):
        dt = _parse_sportytrader_datetime(line)
        if not dt:
            continue
        if target_date and dt.date() != target_date.date():
            continue

        window = lines[i + 1 : i + 30]
        if not window:
            continue
        if not any("nba" in _normalize_for_cmp(w) for w in window[:10]):
            continue

        dash_idx = -1
        for k, candidate in enumerate(window[:15]):
            if candidate.strip() == "-":
                dash_idx = k
                break
        if dash_idx <= 0 or dash_idx >= len(window) - 1:
            continue

        home = ""
        away = ""
        for back in range(dash_idx - 1, -1, -1):
            if _looks_like_team(window[back]):
                home = window[back]
                break
        for fwd in range(dash_idx + 1, min(len(window), dash_idx + 8)):
            if _looks_like_team(window[fwd]):
                away = window[fwd]
                break
        if not home or not away:
            continue

        tip = ""
        marker_idx = -1
        for k, candidate in enumerate(window):
            cmp = _normalize_for_cmp(candidate)
            if cmp.startswith("pronostic ") or cmp.endswith(" picks") or cmp == "picks":
                marker_idx = k
                break
        if marker_idx >= 0:
            for candidate in window[marker_idx + 1 : marker_idx + 8]:
                if _is_noise(candidate):
                    continue
                tip = candidate
                break
        if not tip:
            continue

        row = {
            "datetime": line,
            "home": home,
            "away": away,
            "tip": tip,
            "league": "USA - NBA",
        }
        key = (row["datetime"], row["home"], row["away"], row["tip"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)

    return out


def _print_pick(row: dict[str, str], url: str) -> None:
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Match:          {row['home']} vs {row['away']}")
    print(f"Date/Time:      {row['datetime']}")
    print(f"League:         {row['league']}")
    print(f"Tip:            {row['tip']}")
    print("Odds:           [not found on page]")
    print("Confidence:     [not found on page]")
    print("User vote:      [not found on page]")
    print(f"Source URL:     {url}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main() -> None:
    ap = argparse.ArgumentParser(description="SportyTrader NBA scraper")
    ap.add_argument("--sport", "-s", default="basketball", help="Supported: basketball/nba only")
    ap.add_argument("--date", "-d", help="Date in YYYY-MM-DD")
    args = ap.parse_args()

    sport = (args.sport or "").strip().lower()
    if sport not in {"basketball", "nba"}:
        print("Error: SportyTrader scraper supports only NBA/basketball.")
        sys.exit(1)

    target_date = _parse_target_date(args.date)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1365, "height": 900},
        )
        page = ctx.new_page()
        try:
            page.goto(SPORTYTRADER_URL, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(4000)
            text = page.evaluate("() => document.body?.innerText || ''")
        except PwTimeout:
            print("Error: timed out loading SportyTrader page")
            browser.close()
            sys.exit(1)
        except Exception as exc:
            print(f"Error loading SportyTrader page: {exc}")
            browser.close()
            sys.exit(1)
        finally:
            page.close()
            browser.close()

    rows = _extract_blocks(text, target_date)
    if not rows:
        print("No SportyTrader NBA picks parsed.")
        return

    for row in rows:
        _print_pick(row, SPORTYTRADER_URL)


if __name__ == "__main__":
    main()
