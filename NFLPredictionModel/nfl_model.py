"""NFL serving model for the daily cache.

Slate + posted lines come from the same nflverse games file used in training,
so serving features are computed by the exact code path the model trained on.
Downstream, ``scripts/market_odds.py`` overlays live DraftKings prices for
games that are still pregame.

Publication contract:

* every scheduled game that has not kicked off gets one research row per
  market (moneyline, spread, total) — the board is never silently empty on a
  slate day; games without posted lines publish as explicitly unpriced PASS;
* a game whose kickoff has passed at generation time is skipped (the cache
  merge keeps the rows published before kickoff), so a re-run can never
  re-decide a live game;
* BET/LEAN are minted only through ``metadata.decision_policy`` segments that
  ``nfl_train.py`` validated at recorded closing prices, at a real posted
  price no heavier than the segment's juice cap; everything else is PASS at
  0u with an explicit ``decision_reason``;
* prices are never invented: ``odds`` is the posted nflverse price for the
  exact side or ``None``.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from nfl_core import (
        FEATURE_NAMES,
        ensure_utc,
        features_for_date,
        kickoff_iso,
        kickoff_utc,
        load_games,
        load_team_stats,
        slate_seasons,
    )
except ImportError:
    from .nfl_core import (
        FEATURE_NAMES,
        ensure_utc,
        features_for_date,
        kickoff_iso,
        kickoff_utc,
        load_games,
        load_team_stats,
        slate_seasons,
    )

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ODDS_SOURCE = "nflverse_posted_lines"
LEAN_PROBABILITY = 0.52  # board floor shared with the viewer's isTrackedPick


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _implied(odds: float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def _no_vig(selected: float | None, opposite: float | None) -> float | None:
    a, b = _implied(selected), _implied(opposite)
    if a is None or b is None:
        return None
    return a / (a + b)


def _decimal_profit(odds: float) -> float:
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _american(value: Any) -> int | None:
    number = _num(value)
    if number is None or -100.0 < number < 100.0:
        return None
    return int(round(number))


def _load_artifacts() -> dict[str, Any] | None:
    try:
        import joblib

        metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
        return {
            "ml": joblib.load(ARTIFACT_DIR / "nfl_ml.joblib"),
            "ml_free": joblib.load(ARTIFACT_DIR / "nfl_ml_free.joblib"),
            "spread": joblib.load(ARTIFACT_DIR / "nfl_spread.joblib"),
            "total": joblib.load(ARTIFACT_DIR / "nfl_total.joblib"),
            "metadata": metadata,
        }
    except Exception:
        return None


def _vector(features: dict[str, Any], names: list[str]) -> list[list[float]]:
    return [[float(features[name]) for name in names]]


def _row_base(game: dict[str, Any], date_iso: str, model_version: str) -> dict[str, Any]:
    home = str(game.get("home_team") or "")
    away = str(game.get("away_team") or "")
    matchup = f"{away} @ {home}"
    start = kickoff_iso(game)
    return {
        "source": "NFL Model",
        "sport": "NFL",
        "league": "NFL",
        "date": date_iso,
        "game_id": str(game.get("game_id") or ""),
        "game": matchup,
        "matchup": matchup,
        "home_team": home,
        "away_team": away,
        "start_time": start,
        "game_start_time": start,
        "season_type": str(game.get("game_type") or "REG"),
        "model_version": model_version,
        "shadow_mode": False,
        "actionability": "bet_signal",
        # The model owns its probabilities (market-anchored heads validated
        # out of fold); the pooled cross-sport Platt layer must not re-shift
        # them or re-decide rows.
        "calibration_excluded": True,
        "grade_supported": True,
    }


def _price_fields(odds: int | None) -> dict[str, Any]:
    if odds is None:
        return {"odds": None, "pricing_type": "unpriced", "odds_source": None, "market_priced": False}
    return {"odds": odds, "pricing_type": "market", "odds_source": ODDS_SOURCE, "market_priced": True}


def _segment_decision(
    policy: dict[str, Any],
    *,
    direction: str,
    residual: float,
    odds: int | None,
    evidence_ok: bool,
) -> tuple[str, float, str]:
    """Return (decision, units, reason) for a spread/total row under the policy.

    Segments are graduated bands of residual magnitude (``min_residual`` up to
    an exclusive ``max_residual``); the band containing the residual decides.
    """

    mode = str(policy.get("mode") or "research_only")
    if mode != "segment_gate":
        return "PASS", 0.0, "research_only"
    if not evidence_ok:
        return "PASS", 0.0, "research_only:team_stats_unavailable"
    bands = [segment for segment in policy.get("segments") or [] if str(segment.get("direction")) == direction]
    if not bands:
        return "PASS", 0.0, f"no_segment_for_direction:{direction}"
    magnitude = abs(residual)
    floor = min(float(segment.get("min_residual") or 0.0) for segment in bands)
    for segment in bands:
        low = float(segment.get("min_residual") or 0.0)
        high = segment.get("max_residual")
        if magnitude < low or (high is not None and magnitude >= float(high)):
            continue
        if odds is None:
            return "PASS", 0.0, "unpriced"
        max_juice = float(segment.get("max_juice") or -125)
        if odds < max_juice:
            return "PASS", 0.0, f"juice_above_cap:{odds}"
        decision = str(segment.get("decision") or "LEAN").upper()
        units = float(segment.get("units") or 0.0)
        tier = str(segment.get("tier") or decision.lower())
        return decision, units, f"segment:{tier}:{direction}:{low:g}"
    return "PASS", 0.0, f"below_segment_threshold:{direction}:{floor:g}"


def _confidence_label(probability: float) -> str:
    """Display ladder for research rows: the model's own win/cover probability."""

    if probability >= 0.65:
        return "High"
    if probability >= 0.58:
        return "Medium"
    return "Low"


def generate_nfl_picks(date_iso: str, *, now: datetime | None = None) -> dict[str, Any]:
    rows = load_games()
    if not rows:
        return {"ok": False, "error": "nflverse games dataset unavailable"}
    artifacts = _load_artifacts()
    if artifacts is None:
        return {
            "ok": False,
            "date": date_iso,
            "model": "NFLModel",
            "picks": [],
            "games": [],
            "error": "NFL model artifacts are missing or unreadable; forecasts could not run.",
        }
    metadata = artifacts["metadata"]
    model_version = str(metadata.get("model_version") or "")
    head_features: dict[str, list[str]] = metadata.get("head_features") or {}
    policy: dict[str, Any] = metadata.get("decision_policy") or {}
    sigma_margin = float(metadata.get("margin_residual_sigma") or 13.0)
    sigma_total = float(metadata.get("total_residual_sigma") or 13.3)

    seasons = slate_seasons(date_iso)
    team_stats, loaded_seasons = load_team_stats(seasons, required=(seasons[-1],))
    # Week-1 states roll from the prior season, so the prior season's stats are
    # the minimum evidence for the validated segments; the current season's
    # file may legitimately not exist before its first game.
    evidence_ok = seasons[-2] in loaded_seasons or seasons[-1] in loaded_seasons

    clock = ensure_utc(now)
    slate = features_for_date(rows, date_iso, team_stats)
    coverage = {
        "official_games": len(slate),
        "pregame_games": 0,
        "started_games": 0,
        "unpriced_games": 0,
        "forecast_games": 0,
        "staked_rows": 0,
        "team_stats_seasons": loaded_seasons,
    }
    picks: list[dict[str, Any]] = []
    games_out: list[dict[str, Any]] = []
    for entry in slate:
        game = entry["game"]
        features = entry["features"]
        kickoff = kickoff_utc(game)
        if kickoff is not None and kickoff <= clock:
            coverage["started_games"] += 1
            continue
        coverage["pregame_games"] += 1
        spread_line = entry.get("spread_line")
        total_line = entry.get("total_line")
        if spread_line is None or total_line is None:
            coverage["unpriced_games"] += 1
        coverage["forecast_games"] += 1

        base = _row_base(game, date_iso, model_version)
        rounded_features = {name: round(float(features[name]), 4) for name in FEATURE_NAMES}

        # --- Moneyline (research) -------------------------------------------
        if spread_line is not None:
            p_home = float(artifacts["ml"].predict_proba(_vector(features, head_features["moneyline"]))[0][1])
            ml_probability_source = "market_anchored_logistic"
        else:
            p_home = float(artifacts["ml_free"].predict_proba(_vector(features, head_features["moneyline_free"]))[0][1])
            ml_probability_source = "market_free_logistic"
        home_ml = _american(game.get("home_moneyline"))
        away_ml = _american(game.get("away_moneyline"))
        pick_home = p_home >= 0.5
        side_prob = p_home if pick_home else 1.0 - p_home
        side_odds = home_ml if pick_home else away_ml
        opposite_odds = away_ml if pick_home else home_ml
        ml_market = _no_vig(side_odds, opposite_odds)
        team = str(game.get("home_team") if pick_home else game.get("away_team"))
        ml_ev = (side_prob * _decimal_profit(side_odds) - (1.0 - side_prob)) if side_odds is not None else None
        ml_policy = policy.get("h2h") or {"mode": "research_only"}
        ml_decision, ml_units = "PASS", 0.0
        ml_reason = (
            "research_only:market_parity"
            if str(ml_policy.get("mode") or "research_only") == "research_only"
            else "research_only:no_h2h_segment_gate"
        )
        picks.append({
            **base,
            "pick": f"{team} ML ({base['matchup']})",
            "market": "h2h",
            "market_type": "h2h",
            "team": team,
            "selection": team,
            "side": "home" if pick_home else "away",
            **_price_fields(side_odds),
            "opposite_odds": opposite_odds,
            "probability": round(side_prob, 4),
            "raw_probability": round(side_prob, 4),
            "calibrated_probability": round(side_prob, 4),
            "model_home_win_probability": round(p_home, 4),
            "probability_source": ml_probability_source,
            "market_implied_probability": round(ml_market, 4) if ml_market is not None else None,
            "market_probability": round(ml_market, 4) if ml_market is not None else None,
            "edge": round((side_prob - ml_market) * 100, 2) if ml_market is not None else None,
            "expected_value": round(ml_ev, 4) if ml_ev is not None else None,
            "decision": ml_decision,
            "source_decision": ml_decision,
            "decision_reason": ml_reason,
            "confidence_label": _confidence_label(side_prob),
            "units": ml_units,
            "features": rounded_features,
        })

        # --- Spread -------------------------------------------------------------
        if spread_line is not None:
            margin_residual = float(artifacts["spread"].predict(_vector(features, head_features["spread"]))[0])
            pick_home_spread = margin_residual > 0
            spread_team = str(game.get("home_team") if pick_home_spread else game.get("away_team"))
            team_line = -spread_line if pick_home_spread else spread_line
            cover_prob = _phi(abs(margin_residual) / sigma_margin)
            spread_odds = _american(game.get("home_spread_odds") if pick_home_spread else game.get("away_spread_odds"))
            spread_opposite = _american(game.get("away_spread_odds") if pick_home_spread else game.get("home_spread_odds"))
            spread_market = _no_vig(spread_odds, spread_opposite)
            decision, units, reason = _segment_decision(
                policy.get("spread") or {"mode": "research_only"},
                direction="home" if pick_home_spread else "away",
                residual=margin_residual,
                odds=spread_odds,
                evidence_ok=evidence_ok,
            )
            if decision != "PASS":
                coverage["staked_rows"] += 1
            picks.append({
                **base,
                "pick": f"{spread_team} {team_line:+g} ({base['matchup']})",
                "market": "spread",
                "market_type": "spread",
                "team": spread_team,
                "selection": spread_team,
                "side": "home" if pick_home_spread else "away",
                "line": team_line,
                "market_line": team_line,
                **_price_fields(spread_odds),
                "opposite_odds": spread_opposite,
                "probability": round(cover_prob, 4),
                "raw_probability": round(cover_prob, 4),
                "calibrated_probability": round(cover_prob, 4),
                "market_implied_probability": round(spread_market, 4) if spread_market is not None else None,
                "market_probability": round(spread_market, 4) if spread_market is not None else None,
                "edge": round((cover_prob - spread_market) * 100, 2) if spread_market is not None else None,
                "model_margin_residual": round(margin_residual, 2),
                "decision": decision,
                "source_decision": decision,
                "decision_reason": reason,
                "confidence_label": _confidence_label(cover_prob),
                "units": units,
                "features": rounded_features,
            })
        else:
            margin_residual = None

        # --- Total ----------------------------------------------------------------
        if total_line is not None:
            total_residual = float(artifacts["total"].predict(_vector(features, head_features["total"]))[0])
            over = total_residual > 0
            direction = "over" if over else "under"
            total_prob = _phi(abs(total_residual) / sigma_total)
            total_odds = _american(game.get("over_odds") if over else game.get("under_odds"))
            total_opposite = _american(game.get("under_odds") if over else game.get("over_odds"))
            total_market = _no_vig(total_odds, total_opposite)
            decision, units, reason = _segment_decision(
                policy.get("totals") or {"mode": "research_only"},
                direction=direction,
                residual=total_residual,
                odds=total_odds,
                evidence_ok=evidence_ok,
            )
            if decision != "PASS":
                coverage["staked_rows"] += 1
            picks.append({
                **base,
                "pick": f"{direction.title()} {total_line:g} ({base['matchup']})",
                "market": "totals",
                "market_type": "totals",
                "direction": direction,
                "selection": direction.title(),
                "line": total_line,
                "market_line": total_line,
                **_price_fields(total_odds),
                "opposite_odds": total_opposite,
                "probability": round(total_prob, 4),
                "raw_probability": round(total_prob, 4),
                "calibrated_probability": round(total_prob, 4),
                "market_implied_probability": round(total_market, 4) if total_market is not None else None,
                "market_probability": round(total_market, 4) if total_market is not None else None,
                "edge": round((total_prob - total_market) * 100, 2) if total_market is not None else None,
                "model_total_residual": round(total_residual, 2),
                "decision": decision,
                "source_decision": decision,
                "decision_reason": reason,
                "confidence_label": _confidence_label(total_prob),
                "units": units,
                "features": rounded_features,
            })
        else:
            total_residual = None

        games_out.append({
            "game_id": base["game_id"],
            "matchup": base["matchup"],
            "start_time": base["start_time"],
            "features": rounded_features,
            "priced": spread_line is not None and total_line is not None,
            "home_win_probability": round(p_home, 4),
            "margin_residual": round(margin_residual, 2) if margin_residual is not None else None,
            "total_residual": round(total_residual, 2) if total_residual is not None else None,
        })

    note = f"NFL active slate: {len(games_out)} game(s), {len(picks)} row(s)."
    if coverage["staked_rows"]:
        note += f" {coverage['staked_rows']} staked row(s) from validated segments."
    elif games_out:
        note += " No market cleared a validated segment; rows are research (PASS)."
    if coverage["started_games"]:
        note += f" {coverage['started_games']} started game(s) skipped; pre-kickoff rows are retained by the cache merge."
    if coverage["unpriced_games"]:
        note += f" {coverage['unpriced_games']} game(s) without posted lines published as unpriced research."
    if not evidence_ok:
        note += " Weekly team stats unavailable; segment gates disabled for this refresh."
    if not slate:
        note = f"NFL active slate: 0 game(s), 0 row(s). No NFL games scheduled for {date_iso}."

    return {
        "ok": True,
        "date": date_iso,
        "model": "NFLModel",
        "model_version": model_version,
        "shadow_mode": False,
        "actionability": "bet_signal",
        "coverage": coverage,
        "picks": picks,
        "games": games_out,
        "note": note,
    }
