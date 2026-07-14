"""
kelly_edge.py — Post-prediction edge and Kelly sizing layer.
Reads from cbs_odds table. Never modifies prediction output.
Called AFTER a pick is generated to append a bet recommendation.
"""

import os
import sqlite3
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────────────
KELLY_FRACTION = 0.25            # Use 1/4 Kelly (conservative, model-error robust)
MIN_EDGE = 0.05                  # 5% minimum edge threshold for a BET
LEAN_EDGE = 0.03                 # 3-5% = LEAN (track but don't size up)
DEFAULT_BANKROLL = 1000.0        # Placeholder — user sets their own bankroll


# ── DB path (mirrors existing project convention) ───────────────────────────
def _get_db_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "pickledger.db"),
        os.path.join(here, "pickledger.db"),
    ]
    for path in candidates:
        normalized = os.path.normpath(path)
        if os.path.exists(normalized):
            return normalized
    return os.path.normpath(candidates[0])


# ── Spread → implied win probability ────────────────────────────────────────
def spread_to_prob(spread: float) -> float:
    """
    Convert a point spread to an implied win probability.
    Uses the well-calibrated empirical formula from Stern (1991):
      P(win | spread s) ≈ Φ(s / 13.86)
    where Φ is the standard normal CDF and 13.86 is the empirical
    std dev of NFL/NBA final score margins (NBA uses ~13.86).
    Positive spread = favored side.
    """
    import math

    # Standard normal CDF approximation (no scipy needed)
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

    return _norm_cdf(spread / 13.86)


# ── Fetch latest Vegas line for a matchup ───────────────────────────────────
def get_vegas_line(home_team: str, away_team: str, league: str = "NBA") -> dict | None:
    """
    Pull the most recent cbs_odds row for this matchup.
    Returns dict with spread_home, spread_away, total_line or None if not found.
    Fuzzy-matches team names (last word match) to handle CBS vs model name differences.
    """
    db = _get_db_path()
    if not os.path.exists(db):
        return None

    conn = sqlite3.connect(db)
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT home_team, away_team, spread_home, spread_away, total_line, fetched_at
            FROM cbs_odds
            WHERE league = ?
            ORDER BY fetched_at DESC
            LIMIT 200
            """,
            (league,),
        )
        rows = cur.fetchall()
    except Exception:
        conn.close()
        return None
    conn.close()

    # Fuzzy match: check if last word of stored name is in the query name or vice versa
    def _match(stored: str, query: str) -> bool:
        if not stored or not query:
            return False
        stored_name = stored.strip().lower()
        query_name = query.strip().lower()
        return (
            (stored_name in query_name)
            or (query_name in stored_name)
            or (stored_name.split()[-1] == query_name.split()[-1])
        )

    for row in rows:
        db_home, db_away, spread_home, spread_away, total_line, fetched_at = row
        if _match(db_home, home_team) and _match(db_away, away_team):
            return {
                "home_team": db_home,
                "away_team": db_away,
                "spread_home": spread_home,
                "spread_away": spread_away,
                "total_line": total_line,
                "fetched_at": fetched_at,
            }
    return None


# ── Core edge + Kelly calculation ───────────────────────────────────────────
def calculate_edge(
    model_spread: float,        # Model's projected spread (home perspective, negative = home favored)
    vegas_spread: float,        # Vegas spread for home team (same sign convention)
    league: str = "NBA",
) -> dict:
    """
    Calculate edge and Kelly stake for one side of a spread bet.

    model_spread:  What your model projects (e.g. -4.5 means model says home wins by 4.5)
    vegas_spread:  What Vegas has (e.g. -3.0 means Vegas has home -3)

    Returns a dict with edge, Kelly %, verdict, and reasoning.
    """
    _ = league

    # Model's implied win probability for home team covering
    model_prob = spread_to_prob(-model_spread)    # negate: lower spread = higher P(cover)
    # Vegas implied probability (the line IS the market's probability signal)
    market_prob = spread_to_prob(-vegas_spread)

    raw_edge = model_prob - market_prob

    # Kelly formula: f* = (bp - q) / b
    # For a standard -110 spread bet: decimal odds = 100/110 ≈ 0.909
    # Net odds b = 0.909
    b = 100 / 110          # standard vig
    p = model_prob if raw_edge > 0 else (1 - model_prob)
    q = 1 - p
    kelly_full = (b * p - q) / b
    kelly_full = max(kelly_full, 0.0)   # no negative bets
    kelly_frac = kelly_full * KELLY_FRACTION

    # Verdict
    if raw_edge >= MIN_EDGE:
        verdict = "BET"
    elif raw_edge >= LEAN_EDGE:
        verdict = "LEAN"
    elif raw_edge <= -MIN_EDGE:
        verdict = "FADE"       # edge on the other side
    else:
        verdict = "PASS"

    # Which side to bet
    if raw_edge > 0:
        bet_side = "HOME covers"
    else:
        bet_side = "AWAY covers"

    edge = abs(raw_edge)

    return {
        "verdict": verdict,
        "bet_side": bet_side,
        "edge_pct": round(edge * 100, 2),
        "model_prob": round(model_prob * 100, 1),
        "market_prob": round(market_prob * 100, 1),
        "kelly_full_pct": round(kelly_full * 100, 2),
        "kelly_frac_pct": round(kelly_frac * 100, 2),
        "kelly_stake": round(kelly_frac * DEFAULT_BANKROLL, 2),
        "model_spread": model_spread,
        "vegas_spread": vegas_spread,
    }


# ── Main entry point: given a pick dict, return enriched pick ───────────────
def enrich_pick_with_edge(pick: dict, league: str = "NBA") -> dict:
    """
    Takes an existing pick dict (must have home_team, away_team, projected_spread or spread).
    Returns the same dict with a new 'kelly_edge' key containing the full edge analysis.
    Never modifies the original prediction fields.

    If no Vegas line is found in cbs_odds, kelly_edge = {'verdict': 'NO_LINE', ...}
    """
    result = dict(pick)   # copy — never mutate original

    home = pick.get("home_team") or pick.get("home") or ""
    away = pick.get("away_team") or pick.get("away") or ""
    # Accept various field names the model might use for its projected spread
    model_spread = (
        pick.get("projected_spread")
        or pick.get("model_spread")
        or pick.get("spread_projection")
        or pick.get("predicted_spread")
        or None
    )

    if model_spread is None:
        result["kelly_edge"] = {"verdict": "NO_MODEL_SPREAD", "reason": "No projected spread in pick"}
        return result

    vegas = get_vegas_line(home, away, league)
    if vegas is None:
        result["kelly_edge"] = {"verdict": "NO_LINE", "reason": f"No Vegas line found for {away} @ {home}"}
        return result

    vegas_spread = vegas.get("spread_home")
    if vegas_spread is None:
        result["kelly_edge"] = {"verdict": "NO_LINE", "reason": "Vegas spread is null in cbs_odds"}
        return result

    edge_data = calculate_edge(float(model_spread), float(vegas_spread), league)
    edge_data["vegas_total"] = vegas.get("total_line")
    edge_data["odds_fetched"] = vegas.get("fetched_at")
    result["kelly_edge"] = edge_data
    return result


# ── CLI test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _ = datetime.now(timezone.utc)
    print("=== Kelly Edge Layer — Self Test ===\n")

    # Synthetic test
    test_pick = {
        "home_team": "LA Lakers",
        "away_team": "CLE Cavaliers",
        "projected_spread": -4.5,   # Model says Lakers win by 4.5
    }
    enriched = enrich_pick_with_edge(test_pick, league="NBA")
    ke = enriched.get("kelly_edge", {})
    print(f"Matchup : {test_pick['away_team']} @ {test_pick['home_team']}")
    print(f"Model   : {ke.get('model_spread')} | Vegas: {ke.get('vegas_spread')}")
    print(f"Edge    : {ke.get('edge_pct')}%")
    print(f"Verdict : {ke.get('verdict')} — {ke.get('bet_side')}")
    print(
        f"Kelly   : {ke.get('kelly_frac_pct')}% of bankroll "
        f"(~${ke.get('kelly_stake')} on ${DEFAULT_BANKROLL:.0f} bankroll)"
    )
    print(f"Model P : {ke.get('model_prob')}%  |  Market P: {ke.get('market_prob')}%")
    print()

    # Live DB test — pull a real row from cbs_odds
    db = _get_db_path()
    if os.path.exists(db):
        conn = sqlite3.connect(db)
        rows = conn.execute(
            "SELECT home_team, away_team, spread_home FROM cbs_odds "
            "WHERE spread_home IS NOT NULL LIMIT 3"
        ).fetchall()
        conn.close()
        print("Live cbs_odds rows found:")
        for row in rows:
            home, away, vegas_home = row
            fake_pick = {"home_team": home, "away_team": away, "projected_spread": vegas_home - 1.5}
            enriched2 = enrich_pick_with_edge(fake_pick, "NBA")
            ke2 = enriched2["kelly_edge"]
            print(
                f"  {away} @ {home} | Model: {vegas_home-1.5} Vegas: {vegas_home} | "
                f"Edge: {ke2.get('edge_pct')}% | {ke2.get('verdict')}"
            )
    else:
        print(f"DB not found at {db} — run cbs_odds_scraper.py first")
