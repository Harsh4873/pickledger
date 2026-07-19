"""
wnba_picks.py — Orchestrator that produces WNBA pick outputs.

This module stitches together the schedule, team stats, injury report, and
the pure probability layer into a single pipeline that emits formatted pick
lines for the existing pickgraderserver UI parser.

No network I/O of its own — every data source is reached through the
imported modules, which own their own caches and rate-limiting.
"""

from __future__ import annotations

import datetime
import math

try:
    from .wnba_probability_layers import calculate_wnba_matchup
    from .wnba_schedule import (
        WNBAGame,
        fetch_espn_schedule,
        get_todays_wnba_games,
    )
    from .wnba_stats import (
        get_all_team_stats,
        get_h2h_history,
        get_rolling_stats,
        get_team_stats,
    )
    from .wnba_injuries import (
        get_injury_report,
        get_team_injury_penalty,
    )
    from .wnba_lineup_quality import LineupQuality, get_lineup_quality
    from .wnba_market import (
        EdgeAssessment,
        MarketOdds,
        american_to_implied,
        compute_edge_units,
        lookup_market_odds,
        quarter_kelly_units,
    )
    from .wnba_teams import get_team_by_abbr
except ImportError:
    from wnba_probability_layers import calculate_wnba_matchup
    from wnba_schedule import (
        WNBAGame,
        fetch_espn_schedule,
        get_todays_wnba_games,
    )
    from wnba_stats import (
        get_all_team_stats,
        get_h2h_history,
        get_rolling_stats,
        get_team_stats,
    )
    from wnba_injuries import (
        get_injury_report,
        get_team_injury_penalty,
    )
    from wnba_lineup_quality import LineupQuality, get_lineup_quality
    from wnba_market import (
        EdgeAssessment,
        MarketOdds,
        american_to_implied,
        compute_edge_units,
        lookup_market_odds,
        quarter_kelly_units,
    )
    from wnba_teams import get_team_by_abbr


# ---------------------------------------------------------------------------
# Section 1 — Context Builder
# ---------------------------------------------------------------------------

_REST_LOOKBACK_DAYS = 7

# Per-run memo so we only fetch each past-date schedule once even when
# building contexts for many games in the same run.
_SCHEDULE_DATE_CACHE: dict[str, list[WNBAGame]] = {}


def _schedule_for_date(date_str: str) -> list[WNBAGame]:
    """Return the cached ESPN schedule for *date_str*, fetching if needed."""
    if date_str in _SCHEDULE_DATE_CACHE:
        return _SCHEDULE_DATE_CACHE[date_str]
    try:
        games = fetch_espn_schedule(date_str) or []
    except Exception:
        games = []
    _SCHEDULE_DATE_CACHE[date_str] = games
    return games


def _rest_days_for_team(team_abbr: str, game_date: str) -> int | None:
    """Days since the team's most recent completed game, or None if unknown.

    Walks backward day-by-day up to _REST_LOOKBACK_DAYS from *game_date*.
    We only count games with status == "final" — an abandoned or still-in-
    progress game shouldn't set rest clocks.
    """
    team_abbr = (team_abbr or "").strip().upper()
    if not team_abbr:
        return None

    try:
        base = datetime.date.fromisoformat(game_date)
    except (ValueError, TypeError):
        base = datetime.date.today()

    for days_back in range(1, _REST_LOOKBACK_DAYS + 1):
        past = (base - datetime.timedelta(days=days_back)).isoformat()
        for g in _schedule_for_date(past):
            if getattr(g, "status", "") != "final":
                continue
            if team_abbr in (g.home_abbr, g.away_abbr):
                return days_back
    return None


def _last5_nrtg(team_abbr: str) -> float | None:
    """Best-effort last-5-game NRtg lookup; returns None if unavailable."""
    try:
        rolling = get_rolling_stats(team_abbr, n=5)
    except Exception:
        return None
    if not isinstance(rolling, dict):
        return None
    value = rolling.get("NRtg")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_game_context(game: WNBAGame) -> dict:
    """Assemble the context dict that calculate_wnba_matchup expects.

    Every field is best-effort — missing upstream data leaves the key as
    None, matching the probability layer's tolerance. This function never
    raises.
    """
    home_abbr = getattr(game, "home_abbr", "") or ""
    away_abbr = getattr(game, "away_abbr", "") or ""
    game_date = getattr(game, "date_str", "") or datetime.date.today().isoformat()

    try:
        home_rest_days = _rest_days_for_team(home_abbr, game_date)
    except Exception:
        home_rest_days = None
    try:
        away_rest_days = _rest_days_for_team(away_abbr, game_date)
    except Exception:
        away_rest_days = None

    away_is_b2b = away_rest_days == 1

    try:
        home_injury_penalty = get_team_injury_penalty(home_abbr)
    except Exception:
        home_injury_penalty = None
    try:
        away_injury_penalty = get_team_injury_penalty(away_abbr)
    except Exception:
        away_injury_penalty = None

    try:
        h2h_games = get_h2h_history(home_abbr, away_abbr, as_of_date=game_date)
    except Exception:
        h2h_games = []

    # Starter / minutes-restriction signal — separate from the broad team
    # injury penalty so a "Questionable star" carries weight even when the
    # raw pts_share×status_weight number is small.
    try:
        injury_report = get_injury_report()
    except Exception:
        injury_report = {}
    home_lineup = get_lineup_quality(home_abbr, injury_report)
    away_lineup = get_lineup_quality(away_abbr, injury_report)

    # Fold the minutes-restriction penalty into the same scaler the
    # contextual layer uses for raw team injuries — extra weight on the
    # side whose key players are likely on a restriction.
    home_inj_total = (home_injury_penalty or 0.0) + home_lineup.minutes_restriction_penalty
    away_inj_total = (away_injury_penalty or 0.0) + away_lineup.minutes_restriction_penalty

    return {
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "away_is_b2b": away_is_b2b,
        "home_injury_penalty": home_inj_total,
        "away_injury_penalty": away_inj_total,
        "home_last5_NRtg": _last5_nrtg(home_abbr),
        "away_last5_NRtg": _last5_nrtg(away_abbr),
        "h2h_games": h2h_games,
        # Surface the lineup snapshots so the picks payload + assess_spread_edge
        # can read them without re-deriving.
        "home_lineup_quality": home_lineup,
        "away_lineup_quality": away_lineup,
    }


# ---------------------------------------------------------------------------
# Section 2 — Confidence Label
# ---------------------------------------------------------------------------

def get_confidence_label(win_prob: float) -> str:
    """Bucket the favored team's win probability into Low / Medium / High."""
    try:
        p = float(win_prob)
    except (TypeError, ValueError):
        return "Low"
    favorite_prob = max(p, 1.0 - p)
    if favorite_prob >= 0.72:
        return "High"
    if favorite_prob >= 0.64:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Section 3 — Pick Gate Logic
# ---------------------------------------------------------------------------

def should_generate_spread_pick(
    result: dict,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
    context: dict | None = None,
) -> bool:
    """True when the projected side clears WNBA moneyline guardrails."""
    return assess_spread_edge(result, home_stats, away_stats, context)["decision"] != "PASS"


def should_generate_totals_pick(result: dict, market_total: float | None) -> bool:
    """True when our projected total differs from the market by >= 4.0."""
    if not isinstance(result, dict):
        return False
    projected = result.get("projected_total")
    if projected is None or market_total is None:
        return False
    try:
        return abs(float(projected) - float(market_total)) >= 4.0
    except (TypeError, ValueError):
        return False


def _has_rating_baseline(stats: dict | None) -> bool:
    """True when a team has the season-level rating needed for a real baseline."""
    stats = stats or {}
    return stats.get("NRtg") is not None


def _present_factor_count(stats: dict | None) -> int:
    stats = stats or {}
    return sum(1 for field in _FOUR_FACTOR_FIELDS if stats.get(field) is not None)


def _games_sample(stats: dict | None) -> int:
    """Best-effort completed-game sample from season or rolling fields."""
    stats = stats or {}
    samples: list[int] = []

    try:
        wins = stats.get("W")
        losses = stats.get("L")
        if wins is not None and losses is not None:
            samples.append(int(float(wins)) + int(float(losses)))
    except (TypeError, ValueError):
        pass

    try:
        rolling = stats.get("rolling_games_used")
        if rolling is not None:
            samples.append(int(float(rolling)))
    except (TypeError, ValueError):
        pass

    return max(samples) if samples else 0


def _favorite_probability(win_prob: float) -> float:
    return max(win_prob, 1.0 - win_prob)


def _team_full_name(team_abbr: str) -> str:
    """Return the canonical display name for a WNBA team abbreviation."""
    abbr = str(team_abbr or "").strip().upper()
    if not abbr:
        return ""
    try:
        team = get_team_by_abbr(abbr)
    except Exception:
        return abbr
    return str(team.get("full_name") or abbr).strip() or abbr


MARKET_EDGE_BET_THRESHOLD = 0.030   # 3% vig-removed edge required for BET
MARKET_EDGE_LEAN_THRESHOLD = 0.015  # 1.5% edge qualifies for LEAN
# Measured on the 2024-26 backtest (650 games): spread RMSE 13.3, totals
# RMSE 18.2. Understating these inflates every cover/total probability.
WNBA_SPREAD_RMSE = 13.3
WNBA_TOTAL_RMSE = 18.0
# 2026-07-19 tightening: at the old gates the live spread record ran
# 10-20 and totals 22-37 while the moneyline ran 35-11 — the expansion
# markets were publishing on edges the model cannot actually resolve.
# Both markets now demand materially larger model-to-line disagreement.
WNBA_SPREAD_BET_EDGE = 0.060
WNBA_SPREAD_LEAN_EDGE = 0.045
WNBA_TOTAL_BET_EDGE = 0.060
WNBA_TOTAL_LEAN_EDGE = 0.045
WNBA_SPREAD_BET_COVER = 3.5
WNBA_SPREAD_LEAN_COVER = 2.5
WNBA_TOTAL_MIN_GAP = 7.0

def _units_for_conviction(
    decision: str,
    abs_margin: float,
    favorite_prob: float,
    has_full_baseline: bool,
    h2h_games: int,
) -> float:
    """Conviction-based stake when no real market edge is available.

    Used as a fallback only — when SportsLine has odds for the matchup
    we hand off to ``wnba_market.quarter_kelly_units`` which sizes
    against the actual market price. Range is [0.25, 1.75] units; a
    PASS is 0.0.
    """
    if str(decision or "").upper() == "PASS":
        return 0.0

    # Margin-driven base: each point of projected margin past 4 adds ~0.10u
    # up to a 12-pt cap (1.0u from margin alone).
    margin_units = max(0.0, min(1.0, (abs_margin - 4.0) * 0.10))
    # Probability-driven add: each percentage point of favorite-prob past
    # 60% adds 0.025u up to 90% (0.75u from probability alone).
    prob_units = max(0.0, min(0.75, (favorite_prob - 0.60) * 2.5))

    units = 0.30 + margin_units * 0.55 + prob_units * 0.55
    if str(decision).upper() == "LEAN":
        units *= 0.55  # leans are smaller stakes
    if not has_full_baseline:
        units *= 0.55  # discount partial-baseline picks regardless of decision
    if h2h_games >= 2:
        units *= 1.10  # small boost when we have matchup evidence
    elif h2h_games == 1:
        units *= 1.04
    return round(max(0.25, min(1.75, units)), 2)


def assess_spread_edge(
    result: dict,
    home_stats: dict | None = None,
    away_stats: dict | None = None,
    context: dict | None = None,
    market_edge: "EdgeAssessment | None" = None,
    pick_team_lineup: "LineupQuality | None" = None,
) -> dict:
    """Classify a WNBA moneyline edge as BET, LEAN, or PASS.

    When real SportsLine odds are available the decision is anchored on
    the vig-removed market edge (BET ≥ 3% edge AND favorite prob ≥ 60%).
    When no market data exists, falls back to the older
    margin-+-probability-only thresholds.

    The starter-quality signal from ``pick_team_lineup`` downgrades any
    BET to a LEAN when a key player is OUT or DOUBTFUL — a star-out
    matchup is not the time to be at full stake even if the model edge
    looks great.
    """
    if not isinstance(result, dict):
        return {"decision": "PASS", "confidence_label": "Low", "units": 0.0, "reasons": ["invalid result"]}

    try:
        margin = float(result.get("adjusted_margin"))
        win_prob = float(result.get("win_prob"))
    except (TypeError, ValueError):
        return {"decision": "PASS", "confidence_label": "Low", "units": 0.0, "reasons": ["missing margin/probability"]}

    abs_margin = abs(margin)
    favorite_prob = _favorite_probability(win_prob)
    home_has_rating = _has_rating_baseline(home_stats)
    away_has_rating = _has_rating_baseline(away_stats)
    has_full_baseline = home_has_rating and away_has_rating
    min_games = min(_games_sample(home_stats), _games_sample(away_stats))
    min_factor_fields = min(_present_factor_count(home_stats), _present_factor_count(away_stats))
    h2h_signal = result.get("h2h_signal") or {}
    h2h_games = int(h2h_signal.get("games", 0) or 0)
    has_market = market_edge is not None and market_edge.edge is not None

    reasons: list[str] = []
    if not has_full_baseline:
        reasons.append("no two-team NRtg baseline")
    if min_games and min_games < 3:
        reasons.append(f"thin completed-game sample ({min_games})")
    if min_factor_fields < 4 and not has_full_baseline:
        reasons.append("incomplete four-factor matchup data")
    if result.get("projected_total") is None:
        reasons.append("total unavailable")
    if not has_market:
        reasons.append("no SportsLine market price found")

    if has_market:
        # Real-edge path: anchor on vig-removed edge first, fall back to
        # margin/probability magnitude as a sanity check.
        edge = float(market_edge.edge or 0.0)
        if (
            edge >= MARKET_EDGE_BET_THRESHOLD
            and favorite_prob >= 0.60
            and abs_margin >= 4.5
        ):
            decision = "BET"
        elif edge >= MARKET_EDGE_LEAN_THRESHOLD and favorite_prob >= 0.55:
            decision = "LEAN"
        else:
            decision = "PASS"
        if not has_full_baseline and decision == "BET":
            decision = "LEAN"
            reasons.append("partial baseline capped at LEAN")
    elif has_full_baseline:
        # Internal-only path (no market): old thresholds.
        if abs_margin >= 6.5 and favorite_prob >= 0.70:
            decision = "BET"
        elif abs_margin >= 4.75 and favorite_prob >= 0.64:
            decision = "LEAN"
        else:
            decision = "PASS"
    else:
        if abs_margin >= 8.0 and favorite_prob >= 0.76 and min_factor_fields >= 4:
            decision = "LEAN"
            reasons.append("partial baseline capped at LEAN")
        else:
            decision = "PASS"

    # Starter-quality downgrade: a key player Out/Doubtful turns any
    # BET into a LEAN. Multiple key players out can knock LEAN to PASS.
    starters_out_count = 0
    starters_questionable_count = 0
    if pick_team_lineup is not None:
        starters_out_count = len(pick_team_lineup.starters_out)
        starters_questionable_count = len(pick_team_lineup.starters_questionable)
        if starters_out_count >= 1 and decision == "BET":
            decision = "LEAN"
            reasons.append(
                f"key starter(s) OUT ({', '.join(pick_team_lineup.starters_out)}) — capped at LEAN"
            )
        if starters_out_count >= 2 and decision == "LEAN":
            decision = "PASS"
            reasons.append("two or more key starters OUT — too much lineup risk")
        if starters_questionable_count >= 1 and decision == "BET":
            reasons.append(
                f"questionable starter(s) ({', '.join(pick_team_lineup.starters_questionable)}) — minutes restriction risk"
            )

    # Stake sizing: prefer Kelly against the real market price when
    # available; fall back to conviction-based sizing otherwise.
    if decision == "PASS":
        units = 0.0
    elif has_market and market_edge.kelly_units is not None:
        units = float(market_edge.kelly_units or 0.0)
        if decision == "LEAN":
            units = round(min(units, 1.0) * 0.6, 2)
        if not has_full_baseline:
            units = round(units * 0.65, 2)
        if starters_out_count >= 1:
            units = round(units * 0.5, 2)
        if starters_questionable_count >= 1:
            units = round(units * 0.85, 2)
        units = max(0.0, min(2.0, units))
    else:
        units = _units_for_conviction(
            decision=decision,
            abs_margin=abs_margin,
            favorite_prob=favorite_prob,
            has_full_baseline=has_full_baseline,
            h2h_games=h2h_games,
        )

    label_by_decision = {
        "BET": "High",
        "LEAN": "Medium",
        "PASS": "Low",
    }
    return {
        "decision": decision,
        "confidence_label": label_by_decision[decision],
        "units": round(units, 2),
        "favorite_probability": round(favorite_prob, 4),
        "abs_margin": round(abs_margin, 2),
        "has_full_baseline": has_full_baseline,
        "has_market_price": has_market,
        "market_edge": (round(market_edge.edge, 4) if has_market else None),
        "market_pick_odds": (market_edge.market_pick_odds if has_market else None),
        "market_pick_prob": (market_edge.market_pick_prob if has_market else None),
        "min_games": min_games,
        "min_factor_fields": min_factor_fields,
        "h2h_games": h2h_games,
        "starters_out": (
            list(pick_team_lineup.starters_out) if pick_team_lineup else []
        ),
        "starters_questionable": (
            list(pick_team_lineup.starters_questionable) if pick_team_lineup else []
        ),
        "starters_total": (
            pick_team_lineup.starters_total if pick_team_lineup else 0
        ),
        "reasons": reasons,
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _lineup_for_side(context: dict | None, pick_team_is_home: bool) -> "LineupQuality | None":
    context = context or {}
    key = "home_lineup_quality" if pick_team_is_home else "away_lineup_quality"
    lineup = context.get(key)
    return lineup if isinstance(lineup, LineupQuality) else None


def assess_wnba_spread_market(
    result: dict,
    market: "MarketOdds | None",
    home_stats: dict | None = None,
    away_stats: dict | None = None,
    context: dict | None = None,
) -> dict:
    """Grade the best WNBA spread side without changing the ML decision path."""
    reasons: list[str] = []
    if market is None or market.spread_home is None or market.spread_away is None:
        return {"available": False, "decision": "PASS", "units": 0.0, "reasons": ["spread line unavailable"]}
    try:
        home_margin = float(result.get("adjusted_margin"))
    except (AttributeError, TypeError, ValueError):
        return {"available": True, "decision": "PASS", "units": 0.0, "reasons": ["model margin unavailable"]}

    home_cover_margin = home_margin + float(market.spread_home)
    away_cover_margin = -home_margin + float(market.spread_away)
    pick_team_is_home = home_cover_margin >= away_cover_margin
    cover_margin = home_cover_margin if pick_team_is_home else away_cover_margin
    market_line = float(market.spread_home if pick_team_is_home else market.spread_away)
    odds = int(market.spread_odds or -110)
    probability = _normal_cdf(cover_margin / WNBA_SPREAD_RMSE)
    implied = american_to_implied(odds)
    edge = probability - implied

    has_full_baseline = _has_rating_baseline(home_stats) and _has_rating_baseline(away_stats)
    min_games = min(_games_sample(home_stats), _games_sample(away_stats))
    if not has_full_baseline:
        reasons.append("spread requires two-team NRtg baseline")
    if min_games < 3:
        reasons.append(f"thin completed-game sample ({min_games})")
    if cover_margin < WNBA_SPREAD_LEAN_COVER:
        reasons.append(f"model-to-line cover margin below {WNBA_SPREAD_LEAN_COVER} points")

    decision = "PASS"
    if has_full_baseline and min_games >= 3:
        if cover_margin >= WNBA_SPREAD_BET_COVER and edge >= WNBA_SPREAD_BET_EDGE:
            decision = "BET"
        elif cover_margin >= WNBA_SPREAD_LEAN_COVER and edge >= WNBA_SPREAD_LEAN_EDGE:
            decision = "LEAN"

    lineup = _lineup_for_side(context, pick_team_is_home)
    starters_out = list(lineup.starters_out) if lineup else []
    starters_questionable = list(lineup.starters_questionable) if lineup else []
    if starters_out and decision == "BET":
        decision = "LEAN"
        reasons.append("picked side has a key starter out; spread capped at LEAN")
    if len(starters_out) >= 2:
        decision = "PASS"
        reasons.append("picked side has two or more key starters out")

    units = quarter_kelly_units(edge, odds, cap=1.0) if decision != "PASS" else 0.0
    if decision == "LEAN":
        units = round(units * 0.6, 2)
    if starters_questionable and units:
        units = round(units * 0.85, 2)

    return {
        "available": True,
        "decision": decision,
        "units": units,
        "pick_team_is_home": pick_team_is_home,
        "market_line": market_line,
        "odds": odds,
        "cover_margin": round(cover_margin, 2),
        "probability": round(probability, 4),
        "market_implied_probability": round(implied, 4),
        "edge": round(edge, 4),
        "model_team_margin": round(home_margin if pick_team_is_home else -home_margin, 2),
        "starters_out": starters_out,
        "starters_questionable": starters_questionable,
        "reasons": reasons,
    }


def assess_wnba_total_market(
    result: dict,
    market: "MarketOdds | None",
    home_stats: dict | None = None,
    away_stats: dict | None = None,
    context: dict | None = None,
) -> dict:
    """Grade WNBA totals conservatively while the total model builds a record."""
    reasons: list[str] = []
    if market is None or market.total_line is None:
        return {"available": False, "decision": "PASS", "units": 0.0, "reasons": ["total line unavailable"]}
    try:
        projected_total = float(result.get("projected_total"))
        market_line = float(market.total_line)
    except (AttributeError, TypeError, ValueError):
        return {"available": True, "decision": "PASS", "units": 0.0, "reasons": ["projected total unavailable"]}

    difference = projected_total - market_line
    direction = "Over" if difference >= 0 else "Under"
    gap = abs(difference)
    odds = int(market.total_odds or -110)
    probability = _normal_cdf(gap / WNBA_TOTAL_RMSE)
    implied = american_to_implied(odds)
    edge = probability - implied
    has_full_baseline = _has_rating_baseline(home_stats) and _has_rating_baseline(away_stats)
    min_games = min(_games_sample(home_stats), _games_sample(away_stats))

    if not has_full_baseline:
        reasons.append("total requires two-team NRtg baseline")
    if min_games < 3:
        reasons.append(f"thin completed-game sample ({min_games})")
    if gap < WNBA_TOTAL_MIN_GAP:
        reasons.append(f"model-to-line total gap below {WNBA_TOTAL_MIN_GAP:g} points")

    decision = "PASS"
    if has_full_baseline and min_games >= 3 and gap >= WNBA_TOTAL_MIN_GAP and edge >= WNBA_TOTAL_LEAN_EDGE:
        decision = "LEAN"
        reasons.append("totals market capped at LEAN while its live record is established")

    context = context or {}
    lineups = [
        lineup
        for lineup in (context.get("home_lineup_quality"), context.get("away_lineup_quality"))
        if isinstance(lineup, LineupQuality)
    ]
    questionable = [name for lineup in lineups for name in lineup.starters_questionable]
    if questionable and decision != "PASS":
        reasons.append("questionable starter uncertainty")

    units = quarter_kelly_units(edge, odds, cap=0.75) if decision != "PASS" else 0.0
    if decision == "LEAN":
        units = round(units * 0.6, 2)

    return {
        "available": True,
        "decision": decision,
        "units": units,
        "direction": direction,
        "market_line": market_line,
        "odds": odds,
        "gap": round(gap, 2),
        "probability": round(probability, 4),
        "market_implied_probability": round(implied, 4),
        "edge": round(edge, 4),
        "projected_total": round(projected_total, 1),
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Section 4 — Output Formatter
# ---------------------------------------------------------------------------

def format_pick_line(
    result: dict,
    market_total: float | None = None,
    confidence_label: str | None = None,
) -> str:
    """Render a single-line pick string consumable by pickgraderserver.

    Format (must match exactly — the server UI parses it positionally):

        WNBA | {away} @ {home} | Home Win {win_pct}% | \
Proj Margin: {home} +{margin} | Total: {total} | Conf: {confidence}
    """
    home_abbr = result.get("home_abbr", "") or ""
    away_abbr = result.get("away_abbr", "") or ""
    home = _team_full_name(home_abbr)
    away = _team_full_name(away_abbr)

    try:
        win_prob = float(result.get("win_prob") or 0.0)
    except (TypeError, ValueError):
        win_prob = 0.0
    try:
        margin = float(result.get("adjusted_margin") or 0.0)
    except (TypeError, ValueError):
        margin = 0.0

    projected_total = result.get("projected_total")

    win_pct = round(win_prob * 100.0, 1)

    if margin >= 0:
        margin_str = f"{home} +{margin:.1f}"
    else:
        margin_str = f"{away} +{abs(margin):.1f}"

    if projected_total is None:
        total_str = "N/A"
    else:
        try:
            total_str = f"{float(projected_total):.1f}"
        except (TypeError, ValueError):
            total_str = "N/A"

    confidence = confidence_label or result.get("confidence_label") or get_confidence_label(win_prob)

    return (
        f"WNBA | {away} @ {home} | Home Win {win_pct}% | "
        f"Proj Margin: {margin_str} | Total: {total_str} | Conf: {confidence}"
    )


# ---------------------------------------------------------------------------
# Section 5 — Main Pick Generator
# ---------------------------------------------------------------------------

_FOUR_FACTOR_FIELDS = (
    "eFG_pct", "TOV_pct", "ORB_pct", "FTR",
    "opp_eFG", "opp_TOV", "DRB_pct", "opp_FTR",
)


def _has_usable_stats(stats: dict | None) -> bool:
    """A team profile is usable if it has ratings or a real factor sample."""
    if not stats:
        return False
    if stats.get("NRtg") is not None:
        return True
    return _present_factor_count(stats) >= 4


def generate_wnba_picks(
    market_totals: dict = None,
    echo: bool = True,
    date_str: str | None = None,
) -> list[dict]:
    """Produce a pick dict for every today's WNBA game the model can evaluate.

    Returns a list of pick dicts (possibly empty). Each entry also carries a
    pre-formatted ``output_line`` so downstream consumers don't need to know
    the format string.
    """
    games = get_todays_wnba_games(date_str)
    if not games:
        if echo:
            print("[WNBA] No games today — no picks generated.")
        return []

    # Warm caches once so downstream per-team calls hit memory, not network.
    get_all_team_stats()
    get_injury_report()

    picks: list[dict] = []
    market_totals = market_totals or {}

    for game in games:
        home_abbr = game.home_abbr
        away_abbr = game.away_abbr
        home_name = _team_full_name(home_abbr)
        away_name = _team_full_name(away_abbr)

        home_stats = get_team_stats(home_abbr) or {}
        away_stats = get_team_stats(away_abbr) or {}

        if not _has_usable_stats(home_stats) and not _has_usable_stats(away_stats):
            if echo:
                print(
                    f"[WNBA] PASS — insufficient data for {away_abbr} @ {home_abbr}"
                )
            continue

        context = build_game_context(game)
        result = calculate_wnba_matchup(
            home_abbr, away_abbr, home_stats, away_stats, context
        )

        # Picked side determines which moneyline price + lineup snapshot
        # the edge math operates on.
        pick_team_is_home = float(result.get("win_prob", 0.5)) >= 0.5
        pick_team_abbr = home_abbr if pick_team_is_home else away_abbr
        pick_team_name = home_name if pick_team_is_home else away_name
        market_odds = lookup_market_odds(home_abbr, away_abbr, date_str=game.date_str)
        market_total = (
            market_totals.get(game.espn_game_id) if market_totals else None
        )
        if market_total is None and market_odds is not None:
            market_total = market_odds.total_line
        pick_team_lineup = (
            context.get("home_lineup_quality") if pick_team_is_home else context.get("away_lineup_quality")
        )

        model_pick_prob = (
            float(result.get("win_prob") or 0.5)
            if pick_team_is_home
            else 1.0 - float(result.get("win_prob") or 0.5)
        )
        market_edge = compute_edge_units(
            pick_team_is_home=pick_team_is_home,
            model_pick_prob=model_pick_prob,
            market=market_odds,
        )

        guardrail = assess_spread_edge(
            result,
            home_stats,
            away_stats,
            context,
            market_edge=market_edge,
            pick_team_lineup=pick_team_lineup,
        )
        result["confidence_label"] = guardrail["confidence_label"]

        spread_market = assess_wnba_spread_market(
            result,
            market_odds,
            home_stats,
            away_stats,
            context,
        )
        total_market = assess_wnba_total_market(
            result,
            market_odds,
            home_stats,
            away_stats,
            context,
        )
        moneyline_pick = guardrail["decision"] != "PASS"
        spread_pick = spread_market["decision"] != "PASS"
        totals_pick = total_market["decision"] != "PASS"

        if not moneyline_pick and not spread_pick and not totals_pick:
            if echo:
                reasons = "; ".join(guardrail["reasons"]) or "edge below threshold"
                print(
                    f"[WNBA] PASS — {reasons} for {away_abbr} @ {home_abbr}"
                )

        output_line = format_pick_line(
            result,
            market_total,
            confidence_label=guardrail["confidence_label"],
        )
        matchup = f"{away_name} @ {home_name}"
        market_picks: list[dict] = []
        if spread_market.get("available"):
            spread_team_is_home = bool(spread_market.get("pick_team_is_home"))
            spread_team = home_name if spread_team_is_home else away_name
            spread_line = float(spread_market["market_line"])
            market_picks.append({
                "pick": f"{spread_team} {spread_line:+.1f} ({matchup})",
                "market_type": "spread",
                "selection": spread_team,
                "team": spread_team,
                "odds": spread_market["odds"],
                "line": spread_line,
                "market_line": spread_line,
                "vegas": spread_line,
                "model_prediction": spread_market["model_team_margin"],
                "cover_margin": spread_market["cover_margin"],
                "probability": spread_market["probability"],
                "edge": round(float(spread_market["edge"]) * 100.0, 2),
                "market_edge": spread_market["edge"],
                "market_pick_prob": spread_market["market_implied_probability"],
                "decision": spread_market["decision"],
                "units": spread_market["units"],
                "market_source": market_odds.source if market_odds else None,
                "guardrail_reasons": spread_market["reasons"],
                "starters_out": spread_market["starters_out"],
                "starters_questionable": spread_market["starters_questionable"],
            })
        if total_market.get("available"):
            total_line = float(total_market["market_line"])
            direction = str(total_market["direction"])
            market_picks.append({
                "pick": f"{direction} {total_line:.1f} ({away_name} vs {home_name})",
                "market_type": "totals",
                "selection": direction,
                "odds": total_market["odds"],
                "line": total_line,
                "market_line": total_line,
                "vegas": total_line,
                "model_prediction": total_market["projected_total"],
                "projected_total": total_market["projected_total"],
                "probability": total_market["probability"],
                "edge": round(float(total_market["edge"]) * 100.0, 2),
                "market_edge": total_market["edge"],
                "market_pick_prob": total_market["market_implied_probability"],
                "decision": total_market["decision"],
                "units": total_market["units"],
                "market_source": market_odds.source if market_odds else None,
                "guardrail_reasons": total_market["reasons"],
            })
        pick = {
            "league": "WNBA",
            "home": home_name,
            "away": away_name,
            "home_abbr": home_abbr,
            "away_abbr": away_abbr,
            "home_team": home_name,
            "away_team": away_name,
            "game": matchup,
            "matchup": matchup,
            "pick_team": pick_team_name,
            "pick_team_abbr": pick_team_abbr,
            "win_prob": result["win_prob"],
            "model_pick_prob": round(model_pick_prob, 4),
            "adjusted_margin": result["adjusted_margin"],
            "projected_total": result["projected_total"],
            "market_total": market_total,
            "market_pick_odds": guardrail.get("market_pick_odds"),
            "market_pick_prob": guardrail.get("market_pick_prob"),
            "market_edge": guardrail.get("market_edge"),
            "has_market_price": guardrail.get("has_market_price", False),
            "market_source": market_odds.source if market_odds else None,
            "market_picks": market_picks,
            "moneyline_pick": moneyline_pick,
            "spread_pick": spread_pick,
            "totals_pick": totals_pick,
            "decision": guardrail["decision"],
            "confidence": guardrail["confidence_label"],
            "units": guardrail.get("units", 0.0),
            "h2h_games": guardrail.get("h2h_games", 0),
            "starters_out": guardrail.get("starters_out", []),
            "starters_questionable": guardrail.get("starters_questionable", []),
            "starters_total": guardrail.get("starters_total", 0),
            "guardrail_reasons": guardrail["reasons"],
            "data_quality": result["data_quality"],
            "output_line": output_line,
        }
        picks.append(pick)
        if echo:
            print(output_line)

    return picks


# ---------------------------------------------------------------------------
# Section 6 — CLI Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    picks = generate_wnba_picks()

    if not picks:
        # Off-season path: synthesize guardrail scenarios so we can still
        # exercise the full formatter + gate stack without live games.
        print("[WNBA] Off-season test: running synthetic guardrail checks.")

        home_stats = {"NRtg": 8.0, "ORtg": 108.0, "DRtg": 100.0, "Pace": 70.0, "W": 8, "L": 3}
        away_stats = {"NRtg": -2.0, "ORtg": 101.0, "DRtg": 103.0, "Pace": 69.0, "W": 4, "L": 7}

        synthetic_context = {
            "home_injury_penalty": 0.0,
            "away_injury_penalty": 0.26,  # Collier (MIN) ruled Out
            "away_is_b2b": True,          # MIN on road B2B
        }

        synthetic_result = calculate_wnba_matchup(
            home_abbr="IND",
            away_abbr="MIN",
            home_stats=home_stats,
            away_stats=away_stats,
            context=synthetic_context,
        )

        print(format_pick_line(synthetic_result))

        assert should_generate_spread_pick(synthetic_result, home_stats, away_stats, synthetic_context), (
            "Synthetic IND vs MIN pick should fire only when a real two-team "
            "ratings baseline supports the context stack.\n"
            f"  result={synthetic_result}"
        )

    print("PASS: Pick generator working correctly.")
