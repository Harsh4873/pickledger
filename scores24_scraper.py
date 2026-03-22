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
from zoneinfo import ZoneInfo

def _default_playwright_browsers_path() -> str:
    configured = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if configured:
        return configured
    darwin_cache = os.path.expanduser("~/Library/Caches/ms-playwright")
    if sys.platform == "darwin" and os.path.isdir(darwin_cache):
        return darwin_cache
    # Fall back to package-local browsers for environments like Render.
    return "0"


os.environ["PLAYWRIGHT_BROWSERS_PATH"] = _default_playwright_browsers_path()

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

BASE = os.environ.get("SCORES24_BASE_URL", "https://scores24.live").rstrip("/")

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


def _matches_requested_date(date_str: str, card: dict, variants: list[str]) -> bool:
    combined = f"{card.get('isoDate','')} {card.get('visDate','')}".lower()
    if any(v.lower() in combined for v in variants):
        return True
    # Some cards expose UTC startDate while listing/day filters are local (ET).
    iso = (card.get("isoDate") or "").strip()
    if not iso:
        return False
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d") == date_str
    except Exception:
        return False

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


def listing_url_candidates(sport_slug: str) -> list[str]:
    """Try multiple known Scores24 listing URL patterns."""
    candidates = [
        f"{BASE}/en/predictions/{sport_slug}",
        f"{BASE}/en/{sport_slug}/predictions",
        f"{BASE}/en/{sport_slug}",
    ]
    seen = set()
    out = []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _looks_like_cloudflare_block(text: str) -> bool:
    blob = (text or "").lower()
    signals = [
        "attention required",
        "just a moment",
        "sorry, you have been blocked",
        "performing security verification",
        "cf-error-details",
        "cloudflare",
    ]
    return any(sig in blob for sig in signals)


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
    launch_args = {
        "headless": True,
        "args": [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
        ],
    }

    proxy_server = os.environ.get("PLAYWRIGHT_PROXY_SERVER", "").strip()
    if proxy_server:
        proxy_conf = {"server": proxy_server}
        proxy_user = os.environ.get("PLAYWRIGHT_PROXY_USERNAME", "").strip()
        proxy_pass = os.environ.get("PLAYWRIGHT_PROXY_PASSWORD", "").strip()
        if proxy_user:
            proxy_conf["username"] = proxy_user
        if proxy_pass:
            proxy_conf["password"] = proxy_pass
        launch_args["proxy"] = proxy_conf

    browser = pw.chromium.launch(**launch_args)
    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        viewport={"width": 1280, "height": 900},
        locale="en-US",
        timezone_id="America/New_York",
    )
    # Reduce obvious automation fingerprints; does not bypass hard blocks by itself.
    ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
    return ctx


def load_page(ctx, url: str, wait_ms: int = 3000):
    page = ctx.new_page()
    try:
        resp = page.goto(url, timeout=25000, wait_until="domcontentloaded")
        status = resp.status if resp else 0

        # Cloudflare checks can transiently return 403 first; allow one reload pass.
        if status in (403, 429):
            page.wait_for_timeout(4500)
            try:
                resp2 = page.reload(timeout=25000, wait_until="domcontentloaded")
                if resp2:
                    status = resp2.status
            except Exception:
                pass

        if status == 404:
            page.close()
            return None, status

        # Treat Cloudflare block pages as hard failures, even if status is 200.
        title = ""
        body_head = ""
        try:
            title = page.title()
        except Exception:
            pass
        try:
            body_head = page.evaluate("() => (document.body?.innerText || '').slice(0, 1200)")
        except Exception:
            pass
        if _looks_like_cloudflare_block(f"{title}\n{body_head}"):
            page.close()
            return None, 403

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
            urls.add(href.startsWith('http') ? href : new URL(href, window.location.origin).href);
        }
    }
    return Array.from(urls);
}
"""

DIRECT_PREDICTION_LINKS_JS = r"""
() => {
    const links = document.querySelectorAll('a[href*="/m-"]');
    const urls = new Set();
    for (const a of links) {
        const href = a.getAttribute('href');
        if (!href) continue;
        const abs = href.startsWith('http') ? href : new URL(href, window.location.origin).href;
        if (/-prediction$/i.test(abs) || /\/m-\d{2}-\d{2}-\d{4}-/i.test(abs)) {
            urls.add(abs);
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


def extract_prediction_links(page):
    try:
        return page.evaluate(DIRECT_PREDICTION_LINKS_JS)
    except Exception:
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


def hydrate_listing_page(page, rounds: int = 8):
    """Scroll/click to reveal cards that load lazily on listing pages."""
    stale_rounds = 0
    for _ in range(rounds):
        try:
            before = page.evaluate("() => document.querySelectorAll('span[data-testid=\"PredictionCard\"]').length")
        except Exception:
            break
        try:
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(900)
            clicked_more = page.evaluate(
                """() => {
                    const controls = Array.from(document.querySelectorAll('button, a'));
                    const btn = controls.find(el => /show\\s+more|load\\s+more|more\\s+predictions/i.test((el.textContent || '').trim()));
                    if (!btn) return false;
                    btn.click();
                    return true;
                }"""
            )
            if clicked_more:
                page.wait_for_timeout(1200)
            after = page.evaluate("() => document.querySelectorAll('span[data-testid=\"PredictionCard\"]').length")
        except Exception:
            break
        if after <= before:
            stale_rounds += 1
            if stale_rounds >= 2:
                break
        else:
            stale_rounds = 0


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

        listing_urls = [args.url] if args.url else listing_url_candidates(sport_slug)
        listing_url = listing_urls[0]
        print(f"Sport:          {sport_slug if args.sport else 'Unknown'}")
        if args.date:    print(f"Date requested: {args.date}")
        if args.matchup: print(f"Matchup:        {args.matchup}")
        print(f"Listing URL:    {listing_url}")

        listing_page = None
        status = 0
        used_listing_url = ""
        for cand in listing_urls:
            lp, st = load_page(ctx, cand)
            if lp:
                listing_page = lp
                status = st
                used_listing_url = cand
                break
            status = st

        if not listing_page:
            if status == 403:
                print("Listing page status: ❌ Cloudflare blocked this runtime (status 403)")
                print("Hint: Configure PLAYWRIGHT_PROXY_SERVER (and optional PLAYWRIGHT_PROXY_USERNAME/PLAYWRIGHT_PROXY_PASSWORD) for Render.")
            else:
                print(f"Listing page status: ❌ Page failed (status {status})")
            return
        if used_listing_url and used_listing_url != listing_url:
            print(f"Listing URL fallback: {used_listing_url}")
        print("Listing page status: ✅ Page loaded")
        hydrate_listing_page(listing_page)

        cards = extract_listing_cards(listing_page)

        filtered = []
        variants = date_variants(args.date) if args.date else []
        matchup_parts = [p.strip().lower() for p in args.matchup.split("vs")] if args.matchup else []

        for c in cards:
            if variants:
                if not _matches_requested_date(args.date, c, variants): continue
            if matchup_parts:
                c_data = f"{c.get('home','')} {c.get('away','')} {c.get('href','')}".lower()
                if not all(p in c_data for p in matchup_parts): continue
            filtered.append(c)

        # If card extraction is blocked/empty, fall back to direct prediction links.
        if not filtered:
            direct_links = extract_prediction_links(listing_page)
            for href in direct_links:
                c = {"href": href, "home": "", "away": "", "isoDate": "", "visDate": "", "visTime": "", "league": sport_label, "confidence": ""}
                if variants:
                    blob = href.lower()
                    if not any(v.lower().replace(" ", "-") in blob for v in variants):
                        continue
                if matchup_parts:
                    blob = href.lower()
                    if not all(re.sub(r'[^a-z0-9]+', '-', p) in blob for p in matchup_parts):
                        continue
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
