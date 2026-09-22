"""Observed pregame inputs for MLB inference; no synthetic market lines."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.market_odds import _parse_scoreboard_event, _is_pregame


def fetch_mlb_market_odds_for_date(slate_date, *, fetch=None, now=None):
    clock = now or datetime.now(timezone.utc)
    url = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
    def request():
        response = requests.get(url, params={"dates": slate_date.strftime("%Y%m%d"), "limit": 100}, timeout=25)
        response.raise_for_status()
        return response.json()
    try:
        payload = fetch() if fetch else request()
        events = payload["events"]
    except Exception as exc:
        print(f"[mlb-odds] observed pregame markets unavailable: {exc}", file=sys.stderr)
        return {}
    rows = []
    pairs = []
    for event in events:
        game = _parse_scoreboard_event(event)
        if not game:
            continue
        teams = {c.get("homeAway"): c.get("team", {}) for c in event["competitions"][0]["competitors"]}
        key = tuple(str(teams[s].get("displayName", "")).split()[-1].lower() for s in ("away", "home"))
        pairs.append(key)
        if not _is_pregame(game, clock):
            continue
        markets = game["markets"]
        total, ml = markets.get("total", {}), markets.get("moneyline", {})
        rows.append((key, {"total_line": total.get("line"), "total_over_odds": total.get("over"),
            "total_under_odds": total.get("under"), "ml_home": ml.get("home"), "ml_away": ml.get("away"),
            "source": f"espn_scoreboard:{game['provider']}", "captured_at": clock.isoformat(),
            "start_time": game["startTime"], "event_id": game["eventId"]}))
    counts = Counter(pairs)
    # The legacy inference join is team-based. Never assign one doubleheader's
    # line to both games; leave ambiguous pairs unpriced until joined by ID.
    result = {key: row for key, row in rows if counts[key] == 1}
    print(f"[mlb-odds] {len(rows)} pregame games; {sum(r['total_line'] is not None for r in result.values())} observed totals; {sum(n > 1 for n in counts.values())} ambiguous pairs")
    return result
