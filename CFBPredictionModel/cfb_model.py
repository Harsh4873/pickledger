"""Daily CFB moneyline, spread, and total publisher.

Publication contract (see ``cfb_train.py`` for the walk-forward evidence):

* every eligible pregame FBS game gets one research row per posted market;
* the market-free originator supplies the displayed margin/total forecast and
  the ML/spread research cards;
* stakes are minted only through ``metadata.decision_policy`` — segments of
  the market-anchored residual heads validated out of fold (currently the
  total residual gate) — at an observed two-sided price no heavier than the
  segment's juice cap.  Moneyline and spread publish as PASS research because
  the originator's disagreement with the market is anti-predictive out of
  fold, which is exactly what the 2026-09 "PASS flood plus a few overconfident
  ML BETs" board was showing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

try:
    from cfb_core import FEATURE_NAMES, anchored_vector, matrix, serving_rows
except ImportError:
    from .cfb_core import FEATURE_NAMES, anchored_vector, matrix, serving_rows

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "cfb_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

LEAN_EV = 0.025
BET_EV = 0.055
LEAN_PROBABILITY = 0.52
BET_PROBABILITY = 0.55
# Serving-time fallbacks when an artifact predates the anchored heads.
DEFAULT_POLICY: dict[str, Any] = {
    "h2h": {"mode": "research_only", "reason": "legacy_artifact"},
    "spread": {"mode": "research_only", "reason": "legacy_artifact"},
    "totals": {"mode": "research_only", "reason": "legacy_artifact"},
}


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


# Early-season isotonic calibrators can collapse to a flat plateau across the
# whole mid-range (the CFB spread calibrator returns ~0.5 for every conditional
# input from ~0.35 to ~0.65). When that happens the published probability is a
# meaningless constant that erases the model's actual lean. We detect the
# plateau by probing the calibrator's own output at the conditional input plus a
# margin on each side; if it does not move, we surface the raw model
# probability instead.
#
# Complementary construction means a raw 0.401 home cover becomes a 0.599 away
# cover, which can clear BET/LEAN gates. Spread *selection* ignores that
# complement (`_select_spread_candidate` stays on the win-aligned favorite).
# Fallback is still display-only: `_row(..., uncalibrated=True)` forces PASS
# so an uninformative calibrator cannot mint a stake.
_FLAT_PLATEAU_EPS = 1e-6
_FLAT_PROBE_DELTA = 0.05


def _calibrator_probes(calibrator: Any, raw_win: float, push: float) -> tuple[float, list[float]]:
    non_push = max(1e-9, 1.0 - push)
    conditional = min(0.999, max(0.001, raw_win / non_push))
    low = max(0.001, conditional - _FLAT_PROBE_DELTA)
    high = min(0.999, conditional + _FLAT_PROBE_DELTA)
    probes = [float(value) for value in calibrator.predict([low, conditional, high])]
    return non_push, probes


def _published_probability(calibrator: Any, raw_win: float, push: float) -> tuple[float, bool]:
    """Return (win probability, used_raw_fallback)."""

    non_push, probes = _calibrator_probes(calibrator, raw_win, push)
    if max(probes) - min(probes) <= _FLAT_PLATEAU_EPS:
        return min(non_push, max(0.0, raw_win)), True
    return min(non_push, max(0.0, probes[1] * non_push)), False


def _calibrated_probability_or_raw(calibrator: Any, raw_win: float, push: float) -> float:
    probability, _fallback = _published_probability(calibrator, raw_win, push)
    return probability


def _ev(win: float, push: float, odds: int) -> float:
    loss = max(0.0, 1.0 - win - push)
    return win * _decimal_profit(odds) - loss


def _decision(ev: float, probability: float) -> str:
    if ev >= BET_EV and probability >= BET_PROBABILITY:
        return "BET"
    if ev >= LEAN_EV and probability >= LEAN_PROBABILITY:
        return "LEAN"
    return "PASS"


def _board_eligible(row: dict[str, Any]) -> bool:
    """Canonical public-board rule for in-house CFB PASS cards.

    BET/LEAN always belong on the board. Moneyline (and total) PASS belongs only
    when selected probability clears the LEAN floor (0.52), so +500 dog junk stays
    off the board. The one published spread card per game still belongs on the
    board even when the favorite's cover probability is under 0.5 — that card is
    the model's side, not a longshot ML. The serving payload still includes
    ineligible PASS rows so cache merge can replace prior market cards and the
    CFB forecast-audit ledger can score them. The viewer applies the same rule
    in `isTrackedPick`.
    """

    decision = str(row.get("decision") or "").upper()
    if decision in {"BET", "LEAN"}:
        return True
    if decision != "PASS":
        return False
    market = str(row.get("market") or row.get("market_type") or "").strip().lower()
    if market == "spread":
        return True
    try:
        probability = float(row.get("probability"))
    except (TypeError, ValueError):
        return False
    return probability >= LEAN_PROBABILITY


def _selection_rank(ev: float, probability: float, *, priced: bool) -> tuple[int, float, float]:
    """Rank two market sides so the displayed pick matches the model's read.

    EV-max alone surfaces longshot underdogs (e.g. a +500 dog the model gives
    24.7%) as the board "pick", producing PASS cards whose selection contradicts
    the model. A side that cannot clear the LEAN probability floor can never be a
    BET/LEAN, so among two such sides we show the model's more probable side
    rather than the higher-EV longshot. Only sides that could actually be staked
    are ranked by EV. Unpriced markets fall back to raw probability, unchanged.

    Do not use this for CFB spreads: complementary cover% will pick the dog.
    `_select_spread_candidate` keeps those cards on the win-aligned favorite.
    """

    if not priced:
        return (0, probability, probability)
    actionable = 1 if probability >= LEAN_PROBABILITY else 0
    # For actionable sides prefer EV; for non-actionable sides prefer the model's
    # favored (higher-probability) side. Probability breaks ties in both tiers.
    return (actionable, ev if actionable else probability, probability)


def _model_aligned_spread_side(model_margin: float) -> str:
    """Spread side of the team the model likes to win the game, not necessarily cover."""

    return "home" if model_margin >= 0.0 else "away"


def _select_spread_candidate(
    candidates: list[tuple[Any, ...]],
    *,
    model_margin: float,
) -> tuple[Any, ...]:
    """Publish the spread side of the team the model likes to win.

    Cover probability and EV-max will flip a PASS card to the complementary
    dog whenever the favorite is not expected to cover (home raw cover 0.401
    → away 0.599). For CFB spreads we ignore that complement and keep the
    win-aligned favorite: home -spread when model_margin > 0, away +spread
    when the model likes the visitor.
    """

    aligned = _model_aligned_spread_side(model_margin)
    return next(row for row in candidates if row[0] == aligned)


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
        "shadow_mode": False,
        "actionability": "bet_signal",
        "calibration_excluded": True,
        "grade_supported": True,
    }


def _segment_decision(
    policy: dict[str, Any],
    *,
    direction: str,
    residual: float | None,
    odds: int | None,
) -> tuple[str, float, str]:
    """Return (decision, units, reason) for a market under its validated policy.

    ``research_only`` markets always PASS.  A ``segment_gate`` market stakes only
    when the anchored residual clears the validated threshold for that side and
    the observed price is no heavier than the segment's juice cap.
    """

    mode = str(policy.get("mode") or "research_only")
    if mode != "segment_gate":
        return "PASS", 0.0, "research_only"
    if residual is None:
        return "PASS", 0.0, "research_only:anchored_head_unavailable"
    for segment in policy.get("segments") or []:
        if str(segment.get("direction")) != direction:
            continue
        threshold = float(segment.get("min_residual") or 0.0)
        if abs(residual) < threshold:
            return "PASS", 0.0, f"below_segment_threshold:{direction}:{threshold:g}"
        if odds is None:
            return "PASS", 0.0, "unpriced"
        max_juice = float(segment.get("max_juice") or -125)
        if odds < max_juice:
            return "PASS", 0.0, f"juice_above_cap:{odds}"
        decision = str(segment.get("decision") or "LEAN").upper()
        units = float(segment.get("units") or 0.0)
        return decision, units, f"segment:{direction}:{threshold:g}"
    return "PASS", 0.0, f"no_segment_for_direction:{direction}"


def _row(
    base: dict[str, Any],
    *,
    source: str,
    pick: str,
    market: str,
    selection: str,
    odds: int | None,
    raw_probability: float,
    probability: float,
    push_probability: float,
    market_probability: float | None,
    features: dict[str, float],
    extra: dict[str, Any],
    price_observed: bool,
    uncalibrated: bool = False,
    policy_decision: tuple[str, float, str] | None = None,
) -> dict[str, Any]:
    expected_value = _ev(probability, push_probability, odds) if odds is not None else None
    decision = _decision(expected_value, probability) if expected_value is not None else "PASS"
    if uncalibrated:
        decision = "PASS"
    units = 0.5 if decision == "BET" else 0.25 if decision == "LEAN" else 0.0
    model_decision = decision
    decision_reason = "ev_gate"
    if policy_decision is not None:
        # The validated decision policy owns the published decision; the raw
        # EV gate stays visible as model_decision for the forecast audit.
        decision, units, decision_reason = policy_decision
        if odds is None and decision != "PASS":
            decision, units, decision_reason = "PASS", 0.0, "unpriced"
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
        "market_probability": round(market_probability, 6) if market_probability is not None else None,
        "market_implied_probability": round(market_probability, 6) if market_probability is not None else None,
        "edge": round((probability - market_probability) * 100.0, 3) if market_probability is not None else None,
        "expected_value": round(expected_value, 6) if expected_value is not None else None,
        "source_decision": decision,
        "decision": decision,
        "model_decision": model_decision,
        "decision_reason": decision_reason,
        "units": units,
        "pricing_type": "market" if price_observed else "unpriced",
        "odds_source": base.get("odds_source") if price_observed else None,
        "market_priced": price_observed,
        "features": {
            **{name: round(float(features[name]), 6) for name in FEATURE_NAMES},
            **{name: round(float(features[name]), 6) for name in ("market_home_line", "market_total_line") if features.get(name) is not None},
        },
        **extra,
    }


def generate_cfb_picks(date_iso: str) -> dict[str, Any]:
    artifacts = _load_artifacts()
    if artifacts is None:
        return {
            "ok": False,
            "date": date_iso,
            "model": "CFB Model",
            "shadow_mode": False,
            "games": [],
            "picks": [],
            "error": "CFB model artifacts are missing or unreadable; forecasts could not run.",
        }
    bundle, metadata = artifacts
    coverage: dict[str, int] = {}
    slate = serving_rows(date_iso, coverage=coverage)
    if not slate:
        return {
            "ok": True,
            "date": date_iso,
            "model": "CFB Model",
            "model_version": metadata["model_version"],
            "shadow_mode": False,
            "games": [],
            "picks": [],
            "coverage": coverage,
            "note": (
                "No FBS games on the official CFB scoreboard."
                if coverage.get("official_games") == 0 else
                "No eligible pregame FBS-vs-FBS games; started games and unsupported opponents are excluded."
            ),
        }

    vectors = matrix(slate)
    margin_predictions = bundle["margin_model"].predict(vectors)
    total_predictions = bundle["total_model"].predict(vectors)
    calibrators = bundle["calibrators"]
    anchored = bundle.get("anchored") if isinstance(bundle.get("anchored"), dict) else {}
    anchored_meta = metadata.get("anchored") if isinstance(metadata.get("anchored"), dict) else {}
    policy = metadata.get("decision_policy") if isinstance(metadata.get("decision_policy"), dict) else DEFAULT_POLICY
    sigma_margin = float(metadata["residual_distribution"]["margin_sigma"])
    sigma_total = float(metadata["residual_distribution"]["total_sigma"])
    sigma_spread_residual = float(anchored_meta.get("spread_residual_sigma") or sigma_margin)
    sigma_total_residual = float(anchored_meta.get("total_residual_sigma") or sigma_total)
    model_version = str(metadata["model_version"])

    games: list[dict[str, Any]] = []
    picks: list[dict[str, Any]] = []
    staked_rows = 0
    for entry, model_margin_raw, model_total_raw in zip(slate, margin_predictions, total_predictions):
        game = entry["game"]
        features = dict(entry["features"])
        model_margin = float(model_margin_raw)
        model_total = float(model_total_raw)
        base = _base(game, date_iso, model_version)
        base["odds_source"] = game.get("odds_source")
        home_line_value = game.get("home_line")
        total_line_value = game.get("total_line")
        if home_line_value is not None:
            features["market_home_line"] = float(home_line_value)
        if total_line_value is not None:
            features["market_total_line"] = float(total_line_value)

        # Anchored residual heads (posted line appended to the originator vector).
        anchored_spread_residual: float | None = None
        anchored_total_residual: float | None = None
        anchored_home_probability: float | None = None
        if home_line_value is not None and anchored.get("spread") is not None:
            anchored_spread_residual = float(anchored["spread"].predict([anchored_vector(features, float(home_line_value))])[0])
        if home_line_value is not None and anchored.get("moneyline") is not None:
            anchored_home_probability = float(anchored["moneyline"].predict_proba([anchored_vector(features, float(home_line_value))])[0][1])
        if total_line_value is not None and anchored.get("total") is not None:
            anchored_total_residual = float(anchored["total"].predict([anchored_vector(features, float(total_line_value))])[0])

        raw_home, _, raw_away = _probabilities(model_margin, 0.0, sigma_margin, push_possible=False)
        if anchored_home_probability is not None:
            # The anchored logistic sits at market parity out of fold (Brier
            # 0.179 vs 0.178); the originator alone is 0.190 and its plateau
            # calibrator is what forced the old PASS flood.
            home_probability, ml_uncalibrated = min(0.999, max(0.001, anchored_home_probability)), False
        else:
            home_probability, ml_uncalibrated = _published_probability(calibrators["moneyline"], raw_home, 0.0)
        away_probability = 1.0 - home_probability
        home_ml, away_ml = game.get("home_moneyline"), game.get("away_moneyline")
        ml_priced = home_ml is not None and away_ml is not None
        ml_candidates = [
            ("home", game["home_team"], home_ml if ml_priced else None, raw_home, home_probability, _no_vig(home_ml, away_ml) if ml_priced else None),
            ("away", game["away_team"], away_ml if ml_priced else None, raw_away, away_probability, _no_vig(away_ml, home_ml) if ml_priced else None),
        ]
        ml_side, ml_team, ml_odds, ml_raw, ml_probability, ml_market = max(
            ml_candidates,
            key=lambda row: _selection_rank(
                _ev(row[4], 0.0, row[2]) if ml_priced else 0.0, row[4], priced=ml_priced
            ),
        )
        ml_policy = policy.get("h2h") or DEFAULT_POLICY["h2h"]
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
                extra={
                    "team": ml_team,
                    "side": ml_side,
                    "model_home_win_probability": round(home_probability, 6),
                    # raw_probability keeps the market-free originator's view of
                    # the selected side for the forecast audit; the published
                    # probability is the anchored head when a line is posted.
                    "originator_probability": round(ml_raw, 6),
                    "probability_source": "market_anchored_logistic" if anchored_home_probability is not None else "originator_calibrated",
                },
                price_observed=ml_priced,
                uncalibrated=ml_uncalibrated,
                policy_decision=_segment_decision(ml_policy, direction=ml_side, residual=None, odds=ml_odds),
            )
        )

        if home_line_value is not None:
            home_line = float(home_line_value)
            home_win, spread_push, home_loss = _probabilities(
                model_margin,
                -home_line,
                sigma_margin,
                push_possible=_is_integer_line(home_line),
            )
            if anchored_spread_residual is not None:
                # Cover probability of the anchored residual head for each side
                # (push mass from the integer-line rule); the win-aligned
                # selection rule below still decides which side is the card.
                anchored_home_cover, spread_push, anchored_away_cover = _probabilities(
                    anchored_spread_residual, 0.0, sigma_spread_residual, push_possible=_is_integer_line(home_line)
                )
                calibrated_home_cover, calibrated_away_cover = anchored_home_cover, anchored_away_cover
                spread_uncalibrated = False
            else:
                calibrated_home_cover, spread_uncalibrated = _published_probability(
                    calibrators["spread"], home_win, spread_push
                )
                calibrated_away_cover = max(0.0, 1.0 - spread_push - calibrated_home_cover)
            home_price, away_price = game.get("home_spread_odds"), game.get("away_spread_odds")
            spread_priced = home_price is not None and away_price is not None
            spread_candidates = [
                ("home", game["home_team"], home_line, home_win, calibrated_home_cover, home_price, away_price),
                ("away", game["away_team"], -home_line, home_loss, calibrated_away_cover, away_price, home_price),
            ]
            spread_side, spread_team, spread_line, spread_raw, spread_probability, spread_odds, opposite_odds = (
                _select_spread_candidate(spread_candidates, model_margin=model_margin)
            )
            side_residual = (
                None if anchored_spread_residual is None
                else anchored_spread_residual if spread_side == "home" else -anchored_spread_residual
            )
            spread_decision = _segment_decision(
                policy.get("spread") or DEFAULT_POLICY["spread"],
                direction=spread_side,
                residual=side_residual,
                odds=spread_odds if spread_priced else None,
            )
            if spread_decision[0] != "PASS":
                staked_rows += 1
            picks.append(
                _row(
                    base,
                    source="CFB Spread",
                    pick=f"{spread_team} {spread_line:+g} ({base['matchup']})",
                    market="spread",
                    selection=spread_team,
                    odds=spread_odds if spread_priced else None,
                    raw_probability=spread_raw,
                    probability=spread_probability,
                    push_probability=spread_push,
                    market_probability=_no_vig(spread_odds, opposite_odds) if spread_priced else None,
                    features=features,
                    extra={
                        "team": spread_team,
                        "side": spread_side,
                        "line": spread_line,
                        "market_line": spread_line,
                        "model_margin": round(model_margin, 3),
                        "model_spread_residual": round(side_residual, 3) if side_residual is not None else None,
                    },
                    price_observed=spread_priced,
                    uncalibrated=spread_uncalibrated,
                    policy_decision=spread_decision,
                )
            )

        if total_line_value is not None:
            total_line = float(total_line_value)
            raw_over, total_push, raw_under = _probabilities(
                model_total,
                total_line,
                sigma_total,
                push_possible=_is_integer_line(total_line),
            )
            over_odds, under_odds = game.get("over_odds"), game.get("under_odds")
            total_priced = over_odds is not None and under_odds is not None
            if anchored_total_residual is not None:
                # Direction and probability come from the anchored residual head,
                # the only CFB head with a validated priced segment.
                anchored_over, total_push, anchored_under = _probabilities(
                    anchored_total_residual, 0.0, sigma_total_residual, push_possible=_is_integer_line(total_line)
                )
                direction = "over" if anchored_total_residual > 0 else "under"
                direction_label = "Over" if direction == "over" else "Under"
                total_raw = raw_over if direction == "over" else raw_under
                total_probability = anchored_over if direction == "over" else anchored_under
                total_odds, opposite_odds = (over_odds, under_odds) if direction == "over" else (under_odds, over_odds)
                total_uncalibrated = False
            else:
                calibrated_over, total_uncalibrated = _published_probability(
                    calibrators["total"], raw_over, total_push
                )
                calibrated_under = max(0.0, 1.0 - total_push - calibrated_over)
                total_candidates = [
                    ("over", "Over", raw_over, calibrated_over, over_odds, under_odds),
                    ("under", "Under", raw_under, calibrated_under, under_odds, over_odds),
                ]
                direction, direction_label, total_raw, total_probability, total_odds, opposite_odds = max(
                    total_candidates,
                    key=lambda row: _selection_rank(
                        _ev(row[3], total_push, row[4]) if total_priced else 0.0, row[3], priced=total_priced
                    ),
                )
            total_decision = _segment_decision(
                policy.get("totals") or DEFAULT_POLICY["totals"],
                direction=direction,
                residual=anchored_total_residual,
                odds=total_odds if total_priced else None,
            )
            if total_decision[0] != "PASS":
                staked_rows += 1
            picks.append(
                _row(
                    base,
                    source="CFB Total",
                    pick=f"{direction_label} {total_line:g} ({base['matchup']})",
                    market="totals",
                    selection=direction_label,
                    odds=total_odds if total_priced else None,
                    raw_probability=total_raw,
                    probability=total_probability,
                    push_probability=total_push,
                    market_probability=_no_vig(total_odds, opposite_odds) if total_priced else None,
                    features=features,
                    extra={
                        "direction": direction,
                        "line": total_line,
                        "market_line": total_line,
                        "model_total": round(model_total, 3),
                        "model_total_residual": round(anchored_total_residual, 3) if anchored_total_residual is not None else None,
                    },
                    price_observed=total_priced,
                    uncalibrated=total_uncalibrated,
                    policy_decision=total_decision,
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
                "model_spread_residual": round(anchored_spread_residual, 3) if anchored_spread_residual is not None else None,
                "model_total_residual": round(anchored_total_residual, 3) if anchored_total_residual is not None else None,
            }
        )

    coverage["staked_rows"] = staked_rows
    note = f"CFB active slate: {len(games)} game(s), {len(picks)} row(s)."
    if staked_rows:
        note += f" {staked_rows} staked row(s) from validated segments."
    elif games:
        note += " No market cleared a validated segment; rows are research (PASS)."
    if coverage.get("unpriced_games"):
        note += f" {coverage['unpriced_games']} game(s) missing a posted market published as unpriced research."
    return {
        "ok": True,
        "date": date_iso,
        "model": "CFB Model",
        "model_version": model_version,
        "shadow_mode": False,
        "actionability": "bet_signal",
        "coverage": coverage,
        "games": games,
        "picks": picks,
        "note": note,
    }
