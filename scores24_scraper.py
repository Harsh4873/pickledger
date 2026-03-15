#!/usr/bin/env python3
"""
Scores24.live Prediction Scraper (with Deep Search)
===================================================
Fetches prediction data from scores24.live using Playwright.
If a specific matchup isn't on the main index, automatically
scans sub-league directories to hunt it down.
"""

import argparse
import sys
import re
import json
import os
from datetime import datetime

# Use a deterministic browser install path that survives Render builds.
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "0")

from playwright.sync_api import sync_playwright, TimeoutError as PwTimeout

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LEAGUE_TO_SPORT = {
    "nba": "basketball", "nfl": "american-football", "mlb": "baseball",
    "nhl": "ice-hockey", "premier league": "soccer", "premier-league": "soccer",
    "la liga": "soccer", "la-liga": "soccer", "serie a": "soccer",
    "serie-a": "soccer", "bundesliga": "soccer", "ligue 1": "soccer",
    "ligue-1": "soccer", "champions league": "soccer", "champions-league": "soccer",
    "europa league": "soccer", "europa-league": "soccer",
    "fa cup": "soccer", "fa-cup": "soccer", "atp": "tennis", "wta": "tennis",
}

VALID_SPORTS = [
    "soccer", "basketball", "tennis", "ice-hockey", "volleyball", "handball",
    "baseball", "american-football", "rugby", "cricket", "mma", "boxing",
    "snooker", "futsal", "table-tennis", "waterpolo", "badminton", "darts",
    "csgo", "dota2", "lol", "horse-racing",
]

SUGGESTIVE_PHRASES = [
    "confident", "certain", "sure", "guaranteed", "safe bet", "strong pick",
    "can't lose", "well-liked", "highly recommended", "best bet", "lock",
    "must bet", "no doubt", "clearly", "obvious",
]

BASE = "https://scores24.live"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def resolve_sport(raw: str) -> str:
    key = raw.lower().strip()
    if key.replace(" ", "-") in VALID_SPORTS:
        return key.replace(" ", "-")
    if key in LEAGUE_TO_SPORT:
        return LEAGUE_TO_SPORT[key]
    if key.replace(" ", "-") in LEAGUE_TO_SPORT:
        return LEAGUE_TO_SPORT[key.replace(" ", "-")]
    return key.replace(" ", "-")


def date_variants(date_str: str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return []
    day = str(dt.day)
    return [
        f"{day} {dt.strftime('%b')}",     
        f"{day} {dt.strftime('%B')}",     
        dt.strftime("%d.%m.%y"),          
        dt.strftime("%Y-%m-%d"),          
    ]

def guess_urls(sport_slug: str, date_str: str, matchup_str: str) -> list[str]:
    if not date_str or "vs" not in matchup_str.lower(): return []
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        url_date = dt.strftime("%d-%m-%Y")
    except:
        return []

    parts = matchup_str.lower().split("vs")
    p1 = re.sub(r'[^a-z0-9]+', '-', parts[0].strip())
    p2 = re.sub(r'[^a-z0-9]+', '-', parts[1].strip())

    base_path = f"{BASE}/en/{sport_slug}/m-{url_date}"
    
    return [
        f"{base_path}-{p1}-{p2}-prediction",
        f"{base_path}-{p2}-{p1}-prediction",
        f"{base_path}-{p1}-{p2}",
        f"{base_path}-{p2}-{p1}"
    ]


def scan_suggestive(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for phrase in SUGGESTIVE_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\b"
        m = re.search(rf".{{0,40}}{pattern}.{{0,40}}", lower)
        if m:
            hits.append(m.group(0).strip())
    return hits


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BROWSER HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def make_context(pw):
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
        ],
    )
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 900},
    )
    return ctx


def load_page(ctx, url: str, wait_ms: int = 3000):
    page = ctx.new_page()
    try:
        resp = page.goto(url, timeout=25000, wait_until="domcontentloaded")
        status = resp.status if resp else 0
        if status in (403, 404):
            page.close()
            return None, status
        page.wait_for_timeout(wait_ms)
        return page, status
    except PwTimeout:
        page.close()
        return None, 408
    except Exception:
        page.close()
        return None, 500


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LISTING PAGE & DEEP SEARCH PARSERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEEP_SEARCH_JS = """
() => {
    const links = document.querySelectorAll('a[href*="/l-"]');
    const urls = new Set();
    for (const a of links) {
        const href = a.getAttribute('href');
        if (href && href.includes('/predictions')) {
            urls.add(href.startsWith('http') ? href : 'https://scores24.live' + href);
        }
    }
    return Array.from(urls);
}
"""

def extract_subleague_links(page):
    try:
        return page.evaluate(DEEP_SEARCH_JS)
    except Exception as e:
        return []


LISTING_JS = """
() => {
    const cards = document.querySelectorAll('span[data-testid="PredictionCard"]');
    return Array.from(cards).map(card => {
        const link = card.querySelector('a[itemprop="url"]');
        const href = link ? link.getAttribute('href') : '';

        const homeMeta = card.querySelector('p[itemprop="homeTeam"] meta[itemprop="name"]');
        const awayMeta = card.querySelector('p[itemprop="awayTeam"] meta[itemprop="name"]');
        const home = homeMeta ? homeMeta.content : (card.querySelector('p[itemprop="homeTeam"]')?.textContent?.trim() || '');
        const away = awayMeta ? awayMeta.content : (card.querySelector('p[itemprop="awayTeam"]')?.textContent?.trim() || '');

        const dateMeta = card.querySelector('meta[itemprop="startDate"]');
        const isoDate = dateMeta ? dateMeta.content : '';

        const dateSpans = card.querySelectorAll('span');
        let visDate = '';
        let visTime = '';
        for (const sp of dateSpans) {
            const t = sp.textContent.trim();
            if (/^\\d{1,2}\\s+[A-Za-z]{3}/.test(t)) visDate = t;
            if (/^\\d{2}:\\d{2}$/.test(t)) visTime = t;
        }

        const league = card.querySelector('span')
            ? [...card.querySelectorAll('span')].find(s => {
                  const txt = s.textContent.trim();
                  return txt.length > 1 && txt.length < 40 && !/\\d/.test(txt) && txt !== 'Prediction';
              })?.textContent?.trim() || ''
            : '';

        let confidence = '';
        const allSpans = card.querySelectorAll('span');
        for (const sp of allSpans) {
            const t = sp.textContent.trim();
            if (/^\\d{1,3}%$/.test(t) || /^[-+]\\d{3,4}$/.test(t)) {
                confidence = t;
            }
        }

        return { href, home, away, isoDate, visDate, visTime, league, confidence };
    });
}
"""

def extract_listing_cards(page):
    try:
        return page.evaluate(LISTING_JS)
    except Exception as e:
        return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# INDIVIDUAL PREDICTION PAGE PARSER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREDICTION_JS = """
() => {
    const result = {};
    result.matchTitle = document.title || '';

    const dateEl = document.querySelector('span[data-testid="MatchHeaderHeadDate"]');
    result.date = dateEl ? dateEl.textContent.trim() : '';

    const homeMeta = document.querySelector('meta[itemprop="homeTeam"]');
    const awayMeta = document.querySelector('meta[itemprop="awayTeam"]');
    result.homeTeam = homeMeta ? homeMeta.content : '';
    result.awayTeam = awayMeta ? awayMeta.content : '';

    result.tip = '';
    const allText = document.body.innerText;
    const ourChoiceMatch = allText.match(/Our choice[:\\s]*([^\\n]+)/i);
    if (ourChoiceMatch) {
        result.tip = ourChoiceMatch[1].trim();
    }
    if (!result.tip) {
        const tipPatterns = [
            /(?:prediction|tip|pick)[:\\s]+((?:over|under|total|handicap|win|draw|home|away)[^\\n]{0,60})/i,
            /(Total goals (?:Over|Under) \\([\\d.]+\\))/i,
            /((?:Home|Away|Draw)\\s+(?:Win|Team))/i,
            /(Handicap\\s+[-+]?[\\d.]+)/i,
        ];
        for (const pat of tipPatterns) {
            const m = allText.match(pat);
            if (m) { result.tip = m[1].trim(); break; }
        }
    }

    const valueSpans = document.querySelectorAll('span.value');
    result.allOdds = Array.from(valueSpans).map(s => s.textContent.trim()).filter(t => /^[-+]?\\d/.test(t));
    result.primaryOdds = result.allOdds.length > 0 ? result.allOdds[0] : '';

    result.confidence = '';
    const spans = document.querySelectorAll('span, div');
    for (const s of spans) {
        const t = s.textContent.trim();
        if (/^\\d{1,3}%$/.test(t) && parseInt(t) > 30 && parseInt(t) <= 100) {
            result.confidence = t;
            break;
        }
    }

    result.userVotes = '';
    const voteTotalEl = [...document.querySelectorAll('div')].find(d => /total votes cast/i.test(d.textContent) && d.textContent.length < 50);
    result.totalVotes = voteTotalEl ? voteTotalEl.textContent.trim() : '';
    
    const votePcts = [];
    const allDivs = document.querySelectorAll('div');
    let foundVoteSection = false;
    for (const d of allDivs) {
        const t = d.textContent.trim();
        if (/total votes cast/i.test(t)) foundVoteSection = true;
        if (foundVoteSection && /^\\d{1,3}%$/.test(t)) {
            votePcts.push(t);
            if (votePcts.length >= 2) break;
        }
    }
    if (votePcts.length < 2) {
        for (const d of allDivs) {
            const t = d.textContent.trim();
            if (/^\\d{1,3}%$/.test(t) && !votePcts.includes(t)) {
                votePcts.push(t);
                if (votePcts.length >= 2) break;
            }
        }
    }
    result.votePcts = votePcts;

    const trendEls = document.querySelectorAll('[data-testid="TrendContent"]');
    result.trends = Array.from(trendEls).map(el => el.textContent.trim());
    if (result.trends.length === 0) {
        const lis = document.querySelectorAll('li');
        result.trends = Array.from(lis)
            .map(l => l.textContent.trim())
            .filter(t => t.length > 20 && t.length < 200 && /\\d/.test(t));
    }

    result.oddsTable = [];
    const oddsLinks = document.querySelectorAll('a');
    for (const a of oddsLinks) {
        const divs = a.querySelectorAll('div');
        if (divs.length >= 2) {
            const label = divs[0].textContent.trim();
            const value = divs[divs.length - 1].textContent.trim();
            if (/^(W[12]|Draw|X|Over|Under|1|2|Handicap)/i.test(label) && /^[-+]?\\d/.test(value)) {
                result.oddsTable.push({ market: label, odds: value });
            }
        }
    }

    result.pageText = allText.substring(0, 10000);
    return result;
}
"""

def extract_prediction(page):
    try:
        return page.evaluate(PREDICTION_JS)
    except Exception as e:
        return {"error": str(e)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OUTPUT FORMATTING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fmt(label: str, value: str, fallback: str = "[not found on page]") -> str:
    return f"{label:<15}{value if value else fallback}"


def print_prediction(pred: dict, card: dict, sport_label: str, url: str):
    home = pred.get("homeTeam") or card.get("home", "")
    away = pred.get("awayTeam") or card.get("away", "")
    match_str = f"{home} vs {away}" if home and away else pred.get("matchTitle", "[unknown]")

    date_str = pred.get("date") or card.get("visDate", "")
    if card.get("visTime"):
        date_str = f"{date_str}, {card['visTime']}"

    league = card.get("league") or sport_label
    tip = pred.get("tip", "")
    odds = pred.get("primaryOdds", "")
    confidence = pred.get("confidence") or card.get("confidence", "")

    vote_parts = pred.get("votePcts", [])
    total_votes = pred.get("totalVotes", "")
    user_vote = ""
    if len(vote_parts) >= 2:
        user_vote = f"{vote_parts[0]} vs {vote_parts[1]}"
        if total_votes:
            user_vote += f" ({total_votes})"

    trends = pred.get("trends", [])
    odds_table = pred.get("oddsTable", [])
    warnings = scan_suggestive(pred.get("pageText", ""))

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(fmt("Match:", match_str))
    print(fmt("Date/Time:", date_str))
    print(fmt("League:", league))
    print(fmt("Tip:", tip))
    print(fmt("Odds:", odds))
    print(fmt("Confidence:", confidence))
    print(fmt("User vote:", user_vote))

    if trends:
        print("Stat trends:")
        for t in trends:
            print(f"  • {t}")
    else:
        print(fmt("Stat trends:", ""))

    if odds_table:
        print("Best odds:")
        for row in odds_table:
            print(f"  {row['market']:<12} {row['odds']}")

    print(fmt("Source URL:", url))
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for w in warnings:
        print(f"⚠️  Suggestive language detected: \"{w}\"")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN WORKFLOW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    ap = argparse.ArgumentParser(description="Scores24.live Prediction Scraper")
    ap.add_argument("--sport", "-s", help="Sport or league name")
    ap.add_argument("--date", "-d", help="Date in YYYY-MM-DD format")
    ap.add_argument("--matchup", "-m", help="Specific matchup string")
    ap.add_argument("--url", "-u", help="Direct URL to a specific prediction page")
    args = ap.parse_args()

    if not args.url and not args.sport:
        print("Error: provide --url OR --sport.")
        sys.exit(1)

    sport_slug = resolve_sport(args.sport) if args.sport else "Unknown"
    sport_label = (args.sport or sport_slug).upper()
    if args.sport and sport_slug not in VALID_SPORTS:
        print(f"❌ '{args.sport}' → '{sport_slug}' is not a valid sport.")
        print(f"   Available: {', '.join(VALID_SPORTS)}")
        sys.exit(1)

    with sync_playwright() as pw:
        ctx = make_context(pw)

        # ── DIRECT URL OVERRIDE ──
        if args.url and not args.url.endswith("/predictions") and not args.url.endswith("/predictions/"):
            print(f"Direct Prediction URL provided: {args.url}")
            page, status = load_page(ctx, args.url, wait_ms=4000)
            if not page:
                print(f"❌ 404 — page does not exist: {args.url}")
                return
            
            pred = extract_prediction(page)
            page.close()
            
            if "error" in pred:
                print(f"⚠️  Loaded but extraction failed: {args.url}")
                return
                
            print_prediction(pred, {}, "Unknown", args.url)
            return

        listing_url = args.url if args.url else f"{BASE}/en/predictions/{sport_slug}"
        print(f"Sport:          {sport_slug if args.sport else 'Unknown'}")
        if args.date:    print(f"Date requested: {args.date}")
        if args.matchup: print(f"Matchup:        {args.matchup}")
        print(f"Listing URL:    {listing_url}")

        listing_page, status = load_page(ctx, listing_url)
        if not listing_page:
            print(f"Listing page status: ❌ Page failed (status {status})")
            return
        print("Listing page status: ✅ Page loaded")

        cards = extract_listing_cards(listing_page)

        filtered = []
        variants = date_variants(args.date) if args.date else []
        matchup_parts = [p.strip().lower() for p in args.matchup.split("vs")] if args.matchup else []

        for c in cards:
            if variants:
                combined = f"{c.get('isoDate','')} {c.get('visDate','')}".lower()
                if not any(v.lower() in combined for v in variants): continue
            if matchup_parts:
                c_data = f"{c.get('home','')} {c.get('away','')} {c.get('href','')}".lower()
                if not all(p in c_data for p in matchup_parts): continue
            filtered.append(c)

        # ── DEEP SEARCH LOGIC ──
        if not filtered and args.matchup:
            print(f"\n🔍 '{args.matchup}' not found on main listing.")
            print("Initiating Deep Search across sub-leagues. This may take a minute...")
            
            sub_links = extract_subleague_links(listing_page)
            sub_links = list(set(sub_links))[:30] # Limit to 30 leagues to prevent hanging
            
            print(f"Found {len(sub_links)} sub-leagues to check.")

            found_in_deepSearch = False
            for idx, sl_url in enumerate(sub_links, 1):
                sys.stdout.write(f"\\rScanning sub-league {idx}/{len(sub_links)}...")
                sys.stdout.flush()
                
                sl_page, sl_status = load_page(ctx, sl_url, wait_ms=1000)
                if not sl_page: continue
                
                sl_cards = extract_listing_cards(sl_page)
                
                for c in sl_cards:
                    c_data = f"{c.get('home','')} {c.get('away','')} {c.get('href','')}".lower()
                    if all(p in c_data for p in matchup_parts):
                        filtered.append(c)
                        found_in_deepSearch = True
                
                sl_page.close()
                if found_in_deepSearch:
                    print(f"\\n✅ Matchup found in sub-league routing!")
                    break

            if not found_in_deepSearch:
                print("\\n❌ Deep search scanning complete. Matchup not found on listing pages.")
                if args.date and "vs" in args.matchup.lower():
                    print("🔮 Engaging URL Prediction Engine for hidden match...")
                    guessed = guess_urls(sport_slug, args.date, args.matchup)
                    found_guess = False
                    for gurl in guessed:
                        gpage, gstatus = load_page(ctx, gurl, wait_ms=2000)
                        if gpage and gstatus == 200:
                            gpred = extract_prediction(gpage)
                            if gpred and not "error" in gpred and (gpred.get("tip") or gpred.get("confidence")):
                                print(f"\\n✅ Prediction Engine Success! Extracted hidden page: {gurl}")
                                print_prediction(gpred, {}, "Unknown", gurl)
                                found_guess = True
                                gpage.close()
                                return
                        if gpage: gpage.close()
                    if not found_guess:
                        print("❌ Prediction Engine failed. Check your team names/date spelling.")
                
        else:
            print(f"Matches found for request on listing: {len(filtered)}")

        listing_page.close()

        if not filtered:
            return

        # ── EXTRACT DATA ──
        stats = {"loaded": 0, "404": 0, "no_data": 0}

        for card in filtered:
            href = card.get("href", "")
            if not href: continue
            full_url = href if href.startswith("http") else f"{BASE}{href}"

            pred_page, pred_status = load_page(ctx, full_url, wait_ms=4000)
            if not pred_page:
                print(f"\n❌ 404 — page does not exist: {full_url}")
                stats["404"] += 1; continue

            pred = extract_prediction(pred_page)
            pred_page.close()

            if "error" in pred:
                print(f"\n⚠️  Loaded but extraction failed on: {full_url}")
                stats["no_data"] += 1; continue

            if not pred.get("tip") and not pred.get("primaryOdds") and not pred.get("oddsTable"):
                print(f"\n⚠️  Loaded but no prediction data found on page: {full_url}")
                stats["no_data"] += 1; continue

            stats["loaded"] += 1
            print_prediction(pred, card, sport_label, full_url)

        print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print("SUMMARY")
        print(f"Total matches extracted:        {len(filtered)}")
        print(f"Individual pages loaded:        {stats['loaded']}")
        print(f"Individual pages 404'd:         {stats['404']}")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == "__main__":
    main()
