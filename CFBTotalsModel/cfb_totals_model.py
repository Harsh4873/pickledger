"""Daily CFB publisher for the market-residual model: totals, spread, moneyline.

One row per market per priced FBS game. The three markets do NOT carry equal
weight, and the payload says so explicitly:

  * TOTALS is certified -- a threshold cleared the -110 break-even at p<0.05 and
    held in most individual seasons -- so it can emit BET.
  * SPREAD is uncertified: every candidate threshold came in BELOW break-even
    with negative ROI. Rows are published for visibility only, as PASS/0 units.
  * MONEYLINE is derived from the same margin model as the spread, so it
    inherits that lack of evidence and is likewise PASS-only.

Publishing an uncertified row is a deliberate transparency choice, not a
recommendation: `market_status` and `evidence` on every pick record which bucket
it falls in so an uncertified row can never be mistaken for a validated one.

Prices: ESPN publishes real juice on the two-way markets, so the observed price
is used when present and the assumed -110 is only a fallback.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from CFBTotalsModel.cfb_totals_core import (
    FEATURE_NAMES,
    features_for_slate,
    known_fbs_ids,
    load_history,
    matrix,
)
from CFBPredictionModel.cfb_core import load_live_slate

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cfb_totals_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

ASSUMED_TWO_WAY_ODDS = -110


def _american_implied(odds: Any) -> float | None:
    try:
        value = float(odds)
    except (TypeError, ValueError):
        return None
    if value == 0 or -100 < value < 100:
        return None
    return 100.0 / (value + 100.0) if value > 0 else abs(value) / (abs(value) + 100.0)


def _decimal_profit(odds: Any) -> float:
    value = float(odds)
    return value / 100.0 if value > 0 else 100.0 / abs(value)


def _no_vig(selected: Any, opposite: Any) -> float | None:
    a, b = _american_implied(selected), _american_implied(opposite)
    if a is None or b is None:
        return None
    total = a + b
    return a / total if total > 0 else None


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _load_artifacts() -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        import joblib

        bundle = joblib.load(ARTIFACT_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None
    return bundle, metadata


def _empty(date_iso: str, note: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "ok": True,
        "date": date_iso,
        "model": "CFBMarketResidual",
        "model_version": (metadata or {}).get("model_version"),
        "games": [],
        "picks": [],
        "note": note,
    }


def _base_row(game: dict[str, Any], date_iso: str, model_version: str) -> dict[str, Any]:
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
        "model_class": "market_residual",
        "shadow_mode": False,
        "grade_supported": True,
    }


def _finish(row: dict[str, Any], *, probability: float, market_probability: float,
            odds: Any, actionable: bool, certified: bool, market_status: str,
            evidence: str) -> dict[str, Any]:
    expected_value = probability * _decimal_profit(odds) - (1.0 - probability)
    decision = "BET" if actionable else "PASS"
    row.update({
        "odds": odds,
        "probability": round(probability, 6),
        "calibrated_probability": round(probability, 6),
        "market_probability": round(market_probability, 6),
        "market_implied_probability": round(market_probability, 6),
        "edge": round((probability - market_probability) * 100.0, 3),
        "expected_value": round(expected_value, 6),
        "decision": decision,
        "units": 0.5 if decision == "BET" else 0.0,
        "market_status": market_status,
        "certified": certified,
        "actionability": "validated_edge" if actionable else "research_signal",
        "evidence": evidence,
    })
    return row


def generate_cfb_totals_picks(date_iso: str) -> dict[str, Any]:
    artifacts = _load_artifacts()
    if artifacts is None:
        return _empty(date_iso, "CFB artifacts not trained yet; emitting an empty slate.")
    bundle, metadata = artifacts

    totals_cert = metadata.get("threshold_certification") or {}
    spread_cert = metadata.get("spread_certification") or {}
    totals_threshold = totals_cert.get("selected_threshold")
    spread_threshold = spread_cert.get("selected_threshold")
    totals_evidence = totals_cert.get("selected_evidence") or {}
    statuses = metadata.get("market_status") or {}
    margin_sigma = max(1.0, float(metadata.get("margin_residual_sigma") or 16.0))

    try:
        # Only the two-way markets are required; a game whose moneyline the book
        # has pulled is still usable for totals and spread.
        slate = load_live_slate(date_iso, require_moneyline=False)
    except Exception as exc:
        return {**_empty(date_iso, f"CFB slate fetch failed: {exc}", metadata),
                "ok": False, "error": str(exc)}
    if not slate:
        return _empty(date_iso, "No priced FBS games on the CFB slate.", metadata)

    season = int(date_iso[:4])
    history = load_history(season)
    fbs = known_fbs_ids(history)
    slate = [g for g in slate if g["home_team_id"] in fbs and g["away_team_id"] in fbs]
    if not slate:
        return _empty(date_iso, "No FBS-vs-FBS games on the CFB slate.", metadata)

    priced = {str(g["game_id"]): g for g in slate}
    history = [g for g in history if str(g.get("game_id") or "") not in priced]
    entries = features_for_slate(history, slate)
    records = [
        {"features": e["features"], "game": priced.get(str(e["game"].get("game_id")), e["game"])}
        for e in entries
    ]
    records = [r for r in records if r["game"].get("total_line") is not None]
    if not records:
        return _empty(date_iso, "No CFB games with a posted total.", metadata)

    design = matrix(records)
    total_predictions = bundle["total_residual_model"].predict(design)
    margin_predictions = (
        bundle["margin_residual_model"].predict(design)
        if "margin_residual_model" in bundle else [0.0] * len(records)
    )
    model_version = str(metadata["model_version"])

    games: list[dict[str, Any]] = []
    picks: list[dict[str, Any]] = []
    for record, total_dev_raw, margin_dev_raw in zip(records, total_predictions, margin_predictions):
        game = record["game"]
        total_dev = float(total_dev_raw)
        margin_dev = float(margin_dev_raw)
        total_line = float(game["total_line"])
        home_line = float(game.get("home_line") or 0.0)
        matchup = f"{game['away_team']} @ {game['home_team']}"

        # ---------------- TOTALS (certified) ----------------
        over_odds = game.get("total_odds_over") or ASSUMED_TWO_WAY_ODDS
        under_odds = game.get("total_odds_under") or ASSUMED_TWO_WAY_ODDS
        direction = "Over" if total_dev > 0 else "Under"
        selected_odds = over_odds if direction == "Over" else under_odds
        opposite_odds = under_odds if direction == "Over" else over_odds
        total_priced = bool(game.get("total_odds_over") and game.get("total_odds_under"))
        total_market_prob = (_no_vig(selected_odds, opposite_odds)
                             or _american_implied(ASSUMED_TWO_WAY_ODDS) or 0.5238)
        totals_actionable = (
            totals_threshold is not None and abs(total_dev) >= float(totals_threshold)
        )
        # The certified band's measured hit rate IS the probability claim; below it
        # the model has demonstrated nothing, so it asserts only the market price.
        total_prob = (float(totals_evidence.get("rate", total_market_prob))
                      if totals_actionable else total_market_prob)
        row = _base_row(game, date_iso, model_version)
        row.update({
            "source": "CFB Total",
            "pick": f"{direction} {total_line:g} ({matchup})",
            "market": "totals", "market_type": "totals",
            "selection": direction, "direction": direction.lower(),
            "line": total_line, "market_line": total_line,
            "selected_odds": selected_odds, "opposite_odds": opposite_odds,
            "model_total": round(total_line + total_dev, 2),
            "predicted_deviation": round(total_dev, 3),
            "certified_threshold": totals_threshold,
            "pricing_type": "market" if total_priced else "assumed",
            "market_priced": total_priced,
            "odds_source": game.get("odds_source") if total_priced else "model_assumed_two_way_price",
        })
        picks.append(_finish(
            row, probability=total_prob, market_probability=total_market_prob,
            odds=selected_odds, actionable=totals_actionable,
            certified=totals_threshold is not None,
            market_status=str(statuses.get("totals", "unknown")),
            evidence=(
                f"certified |dev|>={totals_threshold}: {totals_evidence.get('rate')} over "
                f"n={totals_evidence.get('n')} (p={totals_evidence.get('p_value_vs_break_even')}, "
                f"{totals_evidence.get('seasons_beating_break_even')}/5 seasons)"
                if totals_actionable else "below certified confidence threshold"
            ),
        ))

        # ---------------- SPREAD (uncertified) ----------------
        spread_home_odds = game.get("spread_odds_home") or ASSUMED_TWO_WAY_ODDS
        spread_away_odds = game.get("spread_odds_away") or ASSUMED_TWO_WAY_ODDS
        home_covers = margin_dev > 0
        spread_team = game["home_team"] if home_covers else game["away_team"]
        spread_line = home_line if home_covers else -home_line
        spread_odds = spread_home_odds if home_covers else spread_away_odds
        spread_opposite = spread_away_odds if home_covers else spread_home_odds
        spread_priced = bool(game.get("spread_odds_home") and game.get("spread_odds_away"))
        spread_market_prob = (_no_vig(spread_odds, spread_opposite)
                              or _american_implied(ASSUMED_TWO_WAY_ODDS) or 0.5238)
        spread_actionable = (
            spread_threshold is not None and abs(margin_dev) >= float(spread_threshold)
        )
        spread_prob = (_normal_cdf(abs(margin_dev) / margin_sigma)
                       if spread_actionable else spread_market_prob)
        row = _base_row(game, date_iso, model_version)
        row.update({
            "source": "CFB Spread",
            "pick": f"{spread_team} {spread_line:+g} ({matchup})",
            "market": "spread", "market_type": "spread",
            "selection": spread_team, "team": spread_team,
            "side": "home" if home_covers else "away",
            "line": spread_line, "market_line": spread_line,
            "selected_odds": spread_odds, "opposite_odds": spread_opposite,
            "model_margin": round(-home_line + margin_dev, 2),
            "predicted_deviation": round(margin_dev, 3),
            "certified_threshold": spread_threshold,
            "pricing_type": "market" if spread_priced else "assumed",
            "market_priced": spread_priced,
            "odds_source": game.get("odds_source") if spread_priced else "model_assumed_two_way_price",
        })
        picks.append(_finish(
            row, probability=spread_prob, market_probability=spread_market_prob,
            odds=spread_odds, actionable=spread_actionable,
            certified=spread_threshold is not None,
            market_status=str(statuses.get("spread", "uncertified")),
            evidence=(
                "UNCERTIFIED: no spread threshold cleared the -110 break-even in "
                "walk-forward testing (all bands below 50.7% with negative ROI). "
                "Published for visibility only -- not a validated edge."
            ),
        ))

        # ---------------- MONEYLINE (uncertified, derived) ----------------
        home_ml, away_ml = game.get("home_moneyline"), game.get("away_moneyline")
        if home_ml is not None and away_ml is not None:
            # model margin = -home_line + predicted deviation; home wins if > 0
            model_margin = -home_line + margin_dev
            home_win_prob = _normal_cdf(model_margin / margin_sigma)
            pick_home = home_win_prob >= 0.5
            ml_team = game["home_team"] if pick_home else game["away_team"]
            ml_odds = home_ml if pick_home else away_ml
            ml_opposite = away_ml if pick_home else home_ml
            ml_market_prob = _no_vig(ml_odds, ml_opposite) or 0.5
            row = _base_row(game, date_iso, model_version)
            row.update({
                "source": "CFB ML",
                "pick": f"{ml_team} ML ({matchup})",
                "market": "h2h", "market_type": "h2h",
                "selection": ml_team, "team": ml_team,
                "side": "home" if pick_home else "away",
                "selected_odds": ml_odds, "opposite_odds": ml_opposite,
                "model_home_win_probability": round(home_win_prob, 6),
                "predicted_deviation": round(margin_dev, 3),
                "certified_threshold": None,
                "pricing_type": "market",
                "market_priced": True,
                "odds_source": game.get("odds_source"),
            })
            picks.append(_finish(
                row,
                probability=(home_win_prob if pick_home else 1.0 - home_win_prob),
                market_probability=ml_market_prob, odds=ml_odds,
                actionable=False, certified=False,
                market_status=str(statuses.get("moneyline", "uncertified_derived_from_margin")),
                evidence=(
                    "UNCERTIFIED: derived from the same margin model as the spread, "
                    "which showed no edge in walk-forward testing. Published for "
                    "visibility only -- not a validated edge."
                ),
            ))

        games.append({
            "game_id": game["game_id"],
            "matchup": matchup,
            "start_time": game["start_time"],
            "total_line": total_line,
            "home_line": home_line,
            "model_total": round(total_line + total_dev, 2),
            "total_deviation": round(total_dev, 3),
            "margin_deviation": round(margin_dev, 3),
            "features": {n: round(float(record["features"][n]), 6) for n in FEATURE_NAMES},
        })

    picks.sort(key=lambda p: (p["decision"] != "BET", -abs(p.get("predicted_deviation") or 0)))
    actionable_count = sum(1 for p in picks if p["decision"] == "BET")
    return {
        "ok": True,
        "date": date_iso,
        "model": "CFBMarketResidual",
        "model_version": model_version,
        "shadow_mode": False,
        "certified_threshold": totals_threshold,
        "market_status": statuses,
        "games": games,
        "picks": picks,
        "note": (
            f"CFB market-residual: {actionable_count} actionable (totals only) of "
            f"{len(picks)} row(s) across {len(games)} game(s); spread and moneyline "
            f"are published uncertified for visibility."
        ),
    }
