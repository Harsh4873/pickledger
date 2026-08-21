"""Daily CFB moneyline, spread, and total shadow publisher."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from cfb_core import FEATURE_NAMES, matrix, serving_rows
except ImportError:
    from .cfb_core import FEATURE_NAMES, matrix, serving_rows

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cfb_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

LEAN_EV = 0.025
BET_EV = 0.055
LEAN_PROBABILITY = 0.52
BET_PROBABILITY = 0.55
ASSUMED_TWO_WAY_ODDS = -110


def _cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _american_implied(odds: int | float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    return 100.0 / (float(odds) + 100.0) if odds > 0 else abs(float(odds)) / (abs(float(odds)) + 100.0)


def _decimal_profit(odds: int | float) -> float:
    return float(odds) / 100.0 if odds > 0 else 100.0 / abs(float(odds))


def _no_vig(selected: int, opposite: int) -> float:
    selected_implied = _american_implied(selected) or 0.5
    opposite_implied = _american_implied(opposite) or 0.5
    return selected_implied / (selected_implied + opposite_implied)


def _probabilities(
    point_prediction: float,
    threshold: float,
    sigma: float,
    *,
    push_possible: bool,
) -> tuple[float, float, float]:
    """Return win/push/loss for an integer-valued result over a threshold."""

    sigma = max(1.0, float(sigma))
    if push_possible:
        low = _cdf((threshold - 0.5 - point_prediction) / sigma)
        high = _cdf((threshold + 0.5 - point_prediction) / sigma)
        return max(0.0, 1.0 - high), max(0.0, high - low), max(0.0, low)
    loss = _cdf((threshold - point_prediction) / sigma)
    return max(0.0, 1.0 - loss), 0.0, max(0.0, loss)


def _is_integer_line(value: float) -> bool:
    return abs(value - round(value)) < 1e-9


def _calibrated_probability(calibrator: Any, raw_win: float, push: float) -> float:
    non_push = max(1e-9, 1.0 - push)
    conditional = min(0.999, max(0.001, raw_win / non_push))
    calibrated = float(calibrator.predict([conditional])[0])
    return min(non_push, max(0.0, calibrated * non_push))


def _ev(win: float, push: float, odds: int) -> float:
    loss = max(0.0, 1.0 - win - push)
    return win * _decimal_profit(odds) - loss


def _decision(ev: float, probability: float) -> str:
    if ev >= BET_EV and probability >= BET_PROBABILITY:
        return "BET"
    if ev >= LEAN_EV and probability >= LEAN_PROBABILITY:
        return "LEAN"
    return "PASS"


def _load_artifacts() -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        import joblib

        bundle = joblib.load(ARTIFACT_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return bundle, metadata


def _base(game: dict[str, Any], date_iso: str, model_version: str) -> dict[str, Any]:
    matchup = f"{game['away_team']} @ {game['home_team']}"
    return {
        "sport": "CFB",
        "league": "CFB",
        "date": date_iso,
        "game_id": game["game_id"],
        "event_id": game["game_id"],
        "espn_event_id": game["game_id"],
        "home_team_id": game["home_team_id"],
        "away_team_id": game["away_team_id"],
        "home_team": game["home_team"],
        "away_team": game["away_team"],
        "matchup": matchup,
        "game": matchup,
        "start_time": game["start_time"],
        "game_start_time": game["start_time"],
        "neutral_site": game.get("neutral_site") is True,
        "model_version": model_version,
        "shadow_mode": True,
        "actionability": "research_signal",
        "calibration_excluded": True,
        "grade_supported": True,
    }


def _row(
    base: dict[str, Any],
    *,
    source: str,
    pick: str,
    market: str,
    selection: str,
    odds: int,
    raw_probability: float,
    probability: float,
    push_probability: float,
    market_probability: float,
    features: dict[str, float],
    extra: dict[str, Any],
    price_observed: bool,
) -> dict[str, Any]:
    expected_value = _ev(probability, push_probability, odds)
    decision = _decision(expected_value, probability)
    return {
        **base,
        "source": source,
        "pick": pick,
        "market": market,
        "market_type": market,
        "selection": selection,
        "odds": odds,
        "raw_probability": round(raw_probability, 6),
        "probability": round(probability, 6),
        "calibrated_probability": round(probability, 6),
        "push_probability": round(push_probability, 6),
        "market_probability": round(market_probability, 6),
        "market_implied_probability": round(market_probability, 6),
        "edge": round((probability - market_probability) * 100.0, 3),
        "expected_value": round(expected_value, 6),
        "decision": decision,
        "units": 0.5 if decision == "BET" else 0.25 if decision == "LEAN" else 0.0,
        "pricing_type": "market" if price_observed else "assumed",
        "odds_source": base.get("odds_source") if price_observed else "model_assumed_two_way_price",
        "market_priced": price_observed,
        "features": {name: round(float(features[name]), 6) for name in FEATURE_NAMES},
        **extra,
    }


def generate_cfb_picks(date_iso: str) -> dict[str, Any]:
    artifacts = _load_artifacts()
    if artifacts is None:
        return {
            "ok": True,
            "date": date_iso,
            "model": "CFBShadow",
            "shadow_mode": True,
            "games": [],
            "picks": [],
            "note": "CFB artifacts not trained yet; emitting an empty shadow slate.",
        }
    bundle, metadata = artifacts
    slate = serving_rows(date_iso)
    if not slate:
        return {
            "ok": True,
            "date": date_iso,
            "model": "CFBShadow",
            "model_version": metadata["model_version"],
            "shadow_mode": True,
            "games": [],
            "picks": [],
            "note": "No fully priced FBS games on the CFB slate.",
        }

    vectors = matrix(slate)
    margin_predictions = bundle["margin_model"].predict(vectors)
    total_predictions = bundle["total_model"].predict(vectors)
    calibrators = bundle["calibrators"]
    sigma_margin = float(metadata["residual_distribution"]["margin_sigma"])
    sigma_total = float(metadata["residual_distribution"]["total_sigma"])
    model_version = str(metadata["model_version"])

    games: list[dict[str, Any]] = []
    picks: list[dict[str, Any]] = []
    for entry, model_margin_raw, model_total_raw in zip(slate, margin_predictions, total_predictions):
        game = entry["game"]
        features = entry["features"]
        model_margin = float(model_margin_raw)
        model_total = float(model_total_raw)
        base = _base(game, date_iso, model_version)
        base["odds_source"] = game["odds_source"]

        raw_home, _, raw_away = _probabilities(model_margin, 0.0, sigma_margin, push_possible=False)
        home_probability = _calibrated_probability(calibrators["moneyline"], raw_home, 0.0)
        away_probability = 1.0 - home_probability
        home_ml, away_ml = int(game["home_moneyline"]), int(game["away_moneyline"])
        ml_candidates = [
            ("home", game["home_team"], home_ml, raw_home, home_probability, _no_vig(home_ml, away_ml)),
            ("away", game["away_team"], away_ml, raw_away, away_probability, _no_vig(away_ml, home_ml)),
        ]
        ml_side, ml_team, ml_odds, ml_raw, ml_probability, ml_market = max(
            ml_candidates, key=lambda row: _ev(row[4], 0.0, row[2])
        )
        picks.append(
            _row(
                base,
                source="CFB ML",
                pick=f"{ml_team} ML ({base['matchup']})",
                market="h2h",
                selection=ml_team,
                odds=ml_odds,
                raw_probability=ml_raw,
                probability=ml_probability,
                push_probability=0.0,
                market_probability=ml_market,
                features=features,
                extra={"team": ml_team, "side": ml_side, "model_home_win_probability": round(home_probability, 6)},
                price_observed=True,
            )
        )

        home_line = float(game["home_line"])
        home_win, spread_push, home_loss = _probabilities(
            model_margin,
            -home_line,
            sigma_margin,
            push_possible=_is_integer_line(home_line),
        )
        calibrated_home_cover = _calibrated_probability(calibrators["spread"], home_win, spread_push)
        calibrated_away_cover = max(0.0, 1.0 - spread_push - calibrated_home_cover)
        spread_market = _american_implied(ASSUMED_TWO_WAY_ODDS) or 0.52381
        spread_candidates = [
            ("home", game["home_team"], home_line, home_win, calibrated_home_cover),
            ("away", game["away_team"], -home_line, home_loss, calibrated_away_cover),
        ]
        spread_side, spread_team, spread_line, spread_raw, spread_probability = max(
            spread_candidates,
            key=lambda row: _ev(row[4], spread_push, ASSUMED_TWO_WAY_ODDS),
        )
        picks.append(
            _row(
                base,
                source="CFB Spread",
                pick=f"{spread_team} {spread_line:+g} ({base['matchup']})",
                market="spread",
                selection=spread_team,
                odds=ASSUMED_TWO_WAY_ODDS,
                raw_probability=spread_raw,
                probability=spread_probability,
                push_probability=spread_push,
                market_probability=spread_market,
                features=features,
                extra={
                    "team": spread_team,
                    "side": spread_side,
                    "line": spread_line,
                    "market_line": spread_line,
                    "model_margin": round(model_margin, 3),
                },
                price_observed=False,
            )
        )

        total_line = float(game["total_line"])
        raw_over, total_push, raw_under = _probabilities(
            model_total,
            total_line,
            sigma_total,
            push_possible=_is_integer_line(total_line),
        )
        calibrated_over = _calibrated_probability(calibrators["total"], raw_over, total_push)
        calibrated_under = max(0.0, 1.0 - total_push - calibrated_over)
        total_candidates = [
            ("over", "Over", raw_over, calibrated_over),
            ("under", "Under", raw_under, calibrated_under),
        ]
        direction, direction_label, total_raw, total_probability = max(
            total_candidates,
            key=lambda row: _ev(row[3], total_push, ASSUMED_TWO_WAY_ODDS),
        )
        picks.append(
            _row(
                base,
                source="CFB Total",
                pick=f"{direction_label} {total_line:g} ({base['matchup']})",
                market="totals",
                selection=direction_label,
                odds=ASSUMED_TWO_WAY_ODDS,
                raw_probability=total_raw,
                probability=total_probability,
                push_probability=total_push,
                market_probability=spread_market,
                features=features,
                extra={
                    "direction": direction,
                    "line": total_line,
                    "market_line": total_line,
                    "model_total": round(model_total, 3),
                },
                price_observed=False,
            )
        )

        games.append(
            {
                "game_id": game["game_id"],
                "event_id": game["game_id"],
                "home_team_id": game["home_team_id"],
                "away_team_id": game["away_team_id"],
                "matchup": base["matchup"],
                "start_time": game["start_time"],
                "features": {name: round(float(features[name]), 6) for name in FEATURE_NAMES},
                "model_margin": round(model_margin, 3),
                "model_total": round(model_total, 3),
            }
        )

    return {
        "ok": True,
        "date": date_iso,
        "model": "CFBShadow",
        "model_version": model_version,
        "shadow_mode": True,
        "actionability": "research_signal",
        "games": games,
        "picks": picks,
        "note": f"CFB shadow slate: {len(games)} game(s), {len(picks)} market row(s).",
    }
