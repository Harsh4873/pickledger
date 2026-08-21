"""NFL serving model for the daily cache.

Slate + market lines come from the same nflverse games file used in
training (the current season's schedule ships with posted spread/total/
moneylines), so serving features are computed by the exact code path the
model trained on. Downstream, market_odds attaches live DraftKings prices
for games still pregame.

Rows carry BET/LEAN/PASS decisions at posted market prices and publish to the
same site, parlay, Profit Desk, grading, and pregame-ledger surfaces as the
other active in-house team models.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from nfl_core import FEATURE_NAMES, features_for_date, load_games
except ImportError:
    from .nfl_core import FEATURE_NAMES, features_for_date, load_games

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"

# Thresholds selected from the committed walk-forward validation artifacts.
ML_BET_EDGE = 0.05
ML_LEAN_EDGE = 0.025
RESIDUAL_LEAN_POINTS = 2.5
RESIDUAL_BET_POINTS = 4.5


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _implied(odds: float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_artifacts() -> dict[str, Any] | None:
    try:
        import joblib
        metadata = json.loads((ARTIFACT_DIR / "metadata.json").read_text(encoding="utf-8"))
        return {
            "ml": joblib.load(ARTIFACT_DIR / "nfl_ml.joblib"),
            "iso": joblib.load(ARTIFACT_DIR / "nfl_ml_isotonic.joblib"),
            "spread": joblib.load(ARTIFACT_DIR / "nfl_spread.joblib"),
            "total": joblib.load(ARTIFACT_DIR / "nfl_total.joblib"),
            "metadata": metadata,
        }
    except Exception:
        return None


def _row_base(game: dict[str, Any], date_iso: str) -> dict[str, Any]:
    home = str(game.get("home_team") or "")
    away = str(game.get("away_team") or "")
    matchup = f"{away} @ {home}"
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
        "start_time": f"{game.get('gameday')}T{game.get('gametime') or '17:00'}",
        "season_type": str(game.get("game_type") or "REG"),
        "shadow_mode": False,
        "actionability": "bet_signal",
        "pricing_type": "market",
        "odds_source": "nflverse_market_lines",
        "market_priced": True,
    }


def _ml_decision(edge: float | None, probability: float) -> str:
    if edge is None:
        return "PASS"
    if edge >= ML_BET_EDGE and probability >= 0.55:
        return "BET"
    if edge >= ML_LEAN_EDGE and probability >= 0.52:
        return "LEAN"
    return "PASS"


def _residual_decision(residual: float) -> str:
    magnitude = abs(residual)
    if magnitude >= RESIDUAL_BET_POINTS:
        return "BET"
    if magnitude >= RESIDUAL_LEAN_POINTS:
        return "LEAN"
    return "PASS"


def generate_nfl_picks(date_iso: str) -> dict[str, Any]:
    rows = load_games()
    if not rows:
        return {"ok": False, "error": "nflverse games dataset unavailable"}
    slate = features_for_date(rows, date_iso)
    artifacts = _load_artifacts()
    if artifacts is None:
        return {
            "ok": True,
            "date": date_iso,
            "model": "NFLModel",
            "picks": [],
            "games": [],
            "note": "NFL artifacts not trained yet; emitting an empty slate.",
        }

    sigma_margin = float(artifacts["metadata"].get("margin_residual_sigma") or 13.2)
    sigma_total = float(artifacts["metadata"].get("total_residual_sigma") or 13.5)
    picks: list[dict[str, Any]] = []
    games_out: list[dict[str, Any]] = []
    for entry in slate:
        game = entry["game"]
        features = entry["features"]
        vector = [[float(features[name]) for name in FEATURE_NAMES]]
        raw_prob = float(artifacts["ml"].predict_proba(vector)[0][1])
        home_prob = float(artifacts["iso"].predict([raw_prob])[0])
        margin_residual = float(artifacts["spread"].predict(vector)[0])
        total_residual = float(artifacts["total"].predict(vector)[0])

        base = _row_base(game, date_iso)
        home_ml = _num(game.get("home_moneyline"))
        away_ml = _num(game.get("away_moneyline"))
        home_implied = _implied(home_ml)
        away_implied = _implied(away_ml)

        pick_home = home_prob >= 0.5
        side_prob = home_prob if pick_home else 1.0 - home_prob
        side_implied = home_implied if pick_home else away_implied
        side_odds = home_ml if pick_home else away_ml
        vig = (home_implied or 0) + (away_implied or 0)
        no_vig = (side_implied / vig) if side_implied and vig else None
        edge = (side_prob - no_vig) if no_vig is not None else None
        team = str(game.get("home_team") if pick_home else game.get("away_team"))
        ml_decision = _ml_decision(edge, side_prob)
        picks.append({
            **base,
            "pick": f"{team} ML ({base['matchup']})",
            "market": "h2h",
            "market_type": "h2h",
            "team": team,
            "odds": int(side_odds) if side_odds else None,
            "probability": round(side_prob, 4),
            "raw_probability": round(raw_prob if pick_home else 1.0 - raw_prob, 4),
            "edge": round(edge * 100, 2) if edge is not None else None,
            "market_implied_probability": round(no_vig, 4) if no_vig is not None else None,
            "decision": ml_decision,
            "units": 0.5 if ml_decision == "BET" else 0.25 if ml_decision == "LEAN" else 0.0,
        })

        spread_line = features["spread_line"]
        pick_home_spread = margin_residual > 0
        spread_team = str(game.get("home_team") if pick_home_spread else game.get("away_team"))
        team_line = -spread_line if pick_home_spread else spread_line
        cover_prob = _phi(abs(margin_residual) / sigma_margin)
        spread_decision = _residual_decision(margin_residual)
        picks.append({
            **base,
            "pick": f"{spread_team} {team_line:+g} ({base['matchup']})",
            "market": "spread",
            "market_type": "spread",
            "team": spread_team,
            "line": team_line,
            "odds": -110,
            "probability": round(cover_prob, 4),
            "edge": round((cover_prob - 110.0 / 210.0) * 100, 2),
            "model_margin_residual": round(margin_residual, 2),
            "decision": spread_decision,
            "units": 0.5 if spread_decision == "BET" else 0.25 if spread_decision == "LEAN" else 0.0,
        })

        total_line = features["total_line"]
        direction = "Over" if total_residual > 0 else "Under"
        total_prob = _phi(abs(total_residual) / sigma_total)
        total_decision = _residual_decision(total_residual)
        picks.append({
            **base,
            "pick": f"{direction} {total_line:g} ({base['matchup']})",
            "market": "totals",
            "market_type": "totals",
            "direction": direction.lower(),
            "line": total_line,
            "odds": -110,
            "probability": round(total_prob, 4),
            "edge": round((total_prob - 110.0 / 210.0) * 100, 2),
            "model_total_residual": round(total_residual, 2),
            "decision": total_decision,
            "units": 0.5 if total_decision == "BET" else 0.25 if total_decision == "LEAN" else 0.0,
        })

        games_out.append({
            "game_id": base["game_id"],
            "matchup": base["matchup"],
            "features": {name: round(float(features[name]), 4) for name in FEATURE_NAMES},
            "home_win_probability": round(home_prob, 4),
            "margin_residual": round(margin_residual, 2),
            "total_residual": round(total_residual, 2),
        })

    return {
        "ok": True,
        "date": date_iso,
        "model": "NFLModel",
        "model_version": str(artifacts["metadata"].get("model_version") or ""),
        "shadow_mode": False,
        "picks": picks,
        "games": games_out,
        "note": f"NFL active slate: {len(games_out)} game(s), {len(picks)} row(s).",
    }
