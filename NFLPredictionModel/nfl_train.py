"""Train the NFL heads with a strict walk-forward-by-season protocol.

Four heads, every one anchored to the posted market line:

  1. moneyline      — logistic regression on [spread_line, Elo, EPA, CPOE, QB
                      change, rest].  Walk-forward Brier lands on the market's
                      (0.2115 vs 0.2111 for a spread-only logistic), so the
                      published probability is honest rather than the 0.220
                      the Phase-1 gradient-boosted head produced.
  2. moneyline_free — the same logistic without market inputs; only used to
                      describe games whose lines are not posted yet (research
                      rows, never staked).
  3. spread         — Ridge on (margin - spread_line).
  4. total          — Ridge on (total - total_line).

The walk-forward report (train <= season N-1, score season N, 2012-2025)
also measures what each candidate decision segment would have earned at the
recorded closing prices, and only segments that clear the qualification bar
are written into ``metadata.decision_policy`` for the serving path.  Nothing
in ``nfl_model.py`` stakes a unit that this file did not validate.

Run locally or via a manual workflow — never in the daily cron.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from nfl_core import FEATURE_NAMES, FIRST_TEAM_STATS_SEASON, build_dataset, load_games, load_team_stats, matrix
except ImportError:
    from .nfl_core import FEATURE_NAMES, FIRST_TEAM_STATS_SEASON, build_dataset, load_games, load_team_stats, matrix

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_VERSION = "nfl_v1_epa_elo_market_anchored"
FIRST_TRAIN_SEASON = 2007
WALK_FORWARD_SEASONS = range(2012, 2026)

HEAD_FEATURES: dict[str, list[str]] = {
    "moneyline": ["spread_line", "elo_diff", "epa_net_diff", "cpoe_diff", "home_qb_change", "away_qb_change", "rest_diff"],
    "moneyline_free": ["elo_diff", "epa_net_diff", "cpoe_diff", "net_rating_diff", "home_qb_change", "away_qb_change", "rest_diff"],
    "spread": [
        "elo_minus_spread", "epa_net_diff", "epa_off_diff", "epa_def_diff", "pass_matchup_home", "pass_matchup_away",
        "rush_matchup_home", "rush_matchup_away", "cpoe_diff", "home_qb_change", "away_qb_change", "rest_diff",
        "div_game", "week", "spread_line",
    ],
    "total": [
        "total_line", "epa_total_env", "pass_matchup_home", "pass_matchup_away", "rush_matchup_home",
        "rush_matchup_away", "roof_dome", "week", "home_qb_change", "away_qb_change",
    ],
}

# Candidate decision segments.  The search space is deliberately small so a
# qualifying segment is a structural finding, not a lucky bucket.
SEGMENT_THRESHOLDS = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
SEGMENT_DIRECTIONS = {"spread": ("home", "away", "both"), "total": ("over", "under", "both")}
# Graduated confidence ladder, strongest tier first.  Every tier is a band of
# residual magnitude whose OWN picks (not the cumulative tail) must clear the
# tier's bar at recorded prices: enough picks, flat ROI, and a share of
# seasons finishing positive.  BET is the strongest validated band, LEAN the
# next band that still beats break-even comfortably, and a lighter LEAN (fewer
# units) is allowed for a market whose only evidence is weaker but still above
# break-even.  Everything else publishes as PASS research.
TIER_BARS = (
    {"tier": "bet", "decision": "BET", "units": 0.5, "min_picks": 100, "min_roi": 0.08, "min_season_share": 0.75},
    {"tier": "lean", "decision": "LEAN", "units": 0.25, "min_picks": 100, "min_roi": 0.03, "min_season_share": 0.70},
    {"tier": "lean_light", "decision": "LEAN", "units": 0.15, "min_picks": 150, "min_roi": 0.02, "min_season_share": 0.50},
)
# A LEAN band below a BET band must itself clear this many picks so a thin
# slice cannot ride on the tail above it.
MIN_BAND_PICKS = 60
MAX_JUICE = -125  # never stake a validated segment at heavier juice than this


def _logistic() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=50.0))


def _brier(truth: list[int], probs: list[float]) -> float:
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, truth)])) if truth else float("nan")


def fit_signed_side_probability(
    rows: list[dict[str, Any]],
    *,
    pred_key: str,
    actual_key: str,
) -> dict[str, Any] | None:
    """Logistic P(actual residual > 0) with separate slopes for each sign.

    Walk-forward NFL totals are not a symmetric normal: a residual of -1.5
    (under) wins far more often than a residual of +1.5 (over). One slope on
    the signed residual cannot say both of those things. Pushes are left out
    of the fit; the probability is conditional on a non-push.
    """

    usable = [row for row in rows if row.get(actual_key) not in (None, 0)]
    if len(usable) < 200:
        return None
    design = [[min(float(row[pred_key]), 0.0), max(float(row[pred_key]), 0.0)] for row in usable]
    target = [1 if float(row[actual_key]) > 0 else 0 for row in usable]
    if len(set(target)) < 2:
        return None
    model = LogisticRegression(C=1.0, max_iter=1000).fit(design, target)
    return {
        "family": "signed_residual_logistic",
        "intercept": round(float(model.intercept_[0]), 6),
        "negative_slope": round(float(model.coef_[0][0]), 6),
        "positive_slope": round(float(model.coef_[0][1]), 6),
        "fit_rows": len(usable),
        "fit": "walk_forward_oof_excluding_pushes",
    }


def _implied(odds: float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    return 100.0 / (odds + 100.0) if odds > 0 else abs(odds) / (abs(odds) + 100.0)


def _profit(hit: bool | None, odds: float) -> float:
    if hit is None:
        return 0.0
    if not hit:
        return -1.0
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def _segment_stats(picks: list[tuple[int, float, bool | None]]) -> dict[str, Any]:
    if not picks:
        return {"picks": 0}
    by_season: dict[int, float] = defaultdict(float)
    for season, profit, _ in picks:
        by_season[season] += profit
    hits = [hit for _, _, hit in picks if hit is not None]
    seasons_positive = sum(1 for value in by_season.values() if value > 0)
    return {
        "picks": len(picks),
        "hit_rate": round(sum(hits) / len(hits), 4) if hits else None,
        "flat_roi": round(sum(profit for _, profit, _ in picks) / len(picks), 4),
        "seasons": len(by_season),
        "seasons_positive": seasons_positive,
        "season_roi": {str(season): round(value / sum(1 for s, _, _ in picks if s == season), 4) for season, value in sorted(by_season.items())},
    }


def _clears(stats: dict[str, Any], bar: dict[str, Any], *, min_picks: int | None = None) -> bool:
    picks = int(stats.get("picks") or 0)
    seasons = int(stats.get("seasons") or 0)
    return (
        picks >= (min_picks if min_picks is not None else int(bar["min_picks"]))
        and float(stats.get("flat_roi") if stats.get("flat_roi") is not None else -1.0) >= float(bar["min_roi"])
        and seasons > 0
        and int(stats.get("seasons_positive") or 0) / seasons >= float(bar["min_season_share"])
    )


def _qualifies(stats: dict[str, Any]) -> bool:
    """Loosest live gate: the LEAN bar (kept for the segment-search report)."""

    return _clears(stats, TIER_BARS[1])


def _segment_picks(
    oof: list[dict[str, Any]],
    *,
    pred_key: str,
    actual_key: str,
    odds_keys: tuple[str, str],
    direction: str,
    market_key: str,
    low: float,
    high: float | None,
) -> list[tuple[int, float, bool | None]]:
    """Graded picks whose residual magnitude lies in [low, high) on ``direction``."""

    picks: list[tuple[int, float, bool | None]] = []
    for o in oof:
        pred = o[pred_key]
        magnitude = abs(pred)
        if magnitude < low or (high is not None and magnitude >= high):
            continue
        positive_side = pred > 0  # home cover / over
        side_name = ("home" if positive_side else "away") if market_key == "spread" else ("over" if positive_side else "under")
        if direction != "both" and side_name != direction:
            continue
        odds = o[odds_keys[0]] if positive_side else o[odds_keys[1]]
        if odds is None or odds < MAX_JUICE:
            continue
        actual = o[actual_key]
        hit = None if actual == 0 else (actual > 0) == positive_side
        picks.append((o["season"], _profit(hit, odds), hit))
    return picks


def build_ladder(
    oof: list[dict[str, Any]],
    *,
    market_key: str,
    direction: str,
    pred_key: str,
    actual_key: str,
    odds_keys: tuple[str, str],
) -> list[dict[str, Any]]:
    """Graduated tiers for one market direction from band-level evidence.

    Walk the tiers strongest first.  Each tier takes the loosest threshold
    whose cumulative tail clears the tier's bar (using the direction's own
    picks, or the pooled both-direction picks for LEAN tiers) and whose band
    below the stronger tier — measured on the direction's own picks — still
    clears the tier's ROI and season bars.  A tier that cannot be placed is
    simply absent; the market then publishes PASS research there.
    """

    def stats(low: float, high: float | None, use_direction: str) -> dict[str, Any]:
        return _segment_stats(_segment_picks(
            oof, pred_key=pred_key, actual_key=actual_key, odds_keys=odds_keys,
            direction=use_direction, market_key=market_key, low=low, high=high,
        ))

    def placement(bar: dict[str, Any], threshold: float, ceiling: float | None) -> dict[str, Any] | None:
        """Tier row for [threshold, ceiling) if that band clears ``bar``."""

        # Only the direction's own picks count: pooling both directions let a
        # losing side (over ≥1.5, −0.4%) ride on the other side's tail.
        own_tail = stats(threshold, None, direction)
        if not _clears(own_tail, bar):
            return None
        band = stats(threshold, ceiling, direction)
        if ceiling is not None and not _clears(band, bar, min_picks=MIN_BAND_PICKS):
            return None
        return {
            "direction": direction,
            "min_residual": threshold,
            "max_residual": ceiling,
            "decision": bar["decision"],
            "tier": bar["tier"],
            "units": bar["units"],
            "max_juice": MAX_JUICE,
            "walk_forward": {
                "band": {k: band[k] for k in ("picks", "hit_rate", "flat_roi", "seasons", "seasons_positive", "season_roi") if k in band},
                "cumulative": {k: own_tail[k] for k in ("picks", "hit_rate", "flat_roi", "seasons", "seasons_positive") if k in own_tail},
            },
        }

    bet_bar, lean_bar, light_bar = TIER_BARS
    thresholds = list(SEGMENT_THRESHOLDS)  # loosest first

    def lean_below(ceiling: float | None, bar: dict[str, Any]) -> dict[str, Any] | None:
        for threshold in thresholds:
            if ceiling is not None and threshold >= ceiling:
                break
            placed = placement(bar, threshold, ceiling)
            if placed is not None:
                return placed
        return None

    # Prefer a two-tier ladder: the loosest BET threshold that still leaves a
    # validated LEAN band underneath it.  Otherwise the loosest BET alone.
    bet_candidates = [t for t in thresholds if placement(bet_bar, t, None) is not None]
    for bet_threshold in bet_candidates:
        lean = lean_below(bet_threshold, lean_bar)
        if lean is not None:
            return [placement(bet_bar, bet_threshold, None), lean]
    if bet_candidates:
        bet = placement(bet_bar, bet_candidates[0], None)
        light = lean_below(bet_candidates[0], light_bar)
        return [bet] + ([light] if light else [])
    lean = lean_below(None, lean_bar)
    if lean is not None:
        return [lean]
    light = lean_below(None, light_bar)
    return [light] if light else []


def train(first_season: int = FIRST_TRAIN_SEASON, last_season: int = 2025) -> dict[str, Any]:
    games = load_games()
    team_stats, stats_seasons = load_team_stats(range(FIRST_TEAM_STATS_SEASON, last_season + 1), required=(last_season,))
    if len(stats_seasons) < (last_season - FIRST_TEAM_STATS_SEASON):
        raise SystemExit(f"weekly team stats incomplete ({stats_seasons}); refusing to train")
    records = build_dataset(games, first_season=first_season, last_season=last_season, team_stats=team_stats)
    if len(records) < 500:
        raise SystemExit(f"dataset too small ({len(records)} games) — refusing to train")

    oof: list[dict[str, Any]] = []
    walk_forward: list[dict[str, Any]] = []
    for season in WALK_FORWARD_SEASONS:
        if season > last_season:
            continue
        train_recs = [r for r in records if r["season"] < season]
        test_recs = [r for r in records if r["season"] == season]
        if len(train_recs) < 500 or not test_recs:
            continue
        ml = _logistic().fit(matrix(train_recs, HEAD_FEATURES["moneyline"]), [r["home_win"] for r in train_recs])
        ml_free = _logistic().fit(matrix(train_recs, HEAD_FEATURES["moneyline_free"]), [r["home_win"] for r in train_recs])
        market = LogisticRegression(C=10.0).fit([[r["features"]["spread_line"]] for r in train_recs], [r["home_win"] for r in train_recs])
        spread = _ridge().fit(matrix(train_recs, HEAD_FEATURES["spread"]), [r["margin_residual"] for r in train_recs])
        total = _ridge().fit(matrix(train_recs, HEAD_FEATURES["total"]), [r["total_residual"] for r in train_recs])

        p_home = ml.predict_proba(matrix(test_recs, HEAD_FEATURES["moneyline"]))[:, 1]
        p_free = ml_free.predict_proba(matrix(test_recs, HEAD_FEATURES["moneyline_free"]))[:, 1]
        p_market = market.predict_proba([[r["features"]["spread_line"]] for r in test_recs])[:, 1]
        spread_pred = spread.predict(matrix(test_recs, HEAD_FEATURES["spread"]))
        total_pred = total.predict(matrix(test_recs, HEAD_FEATURES["total"]))
        truth = [r["home_win"] for r in test_recs]
        for rec, ph, pf, pm, sp, tp in zip(test_recs, p_home, p_free, p_market, spread_pred, total_pred):
            oof.append({**rec, "season": season, "p_home": float(ph), "p_free": float(pf), "p_market": float(pm),
                        "spread_pred": float(sp), "total_pred": float(tp)})
        cover_graded = [(sp > 0) == (r["margin_residual"] > 0) for r, sp in zip(test_recs, spread_pred) if r["margin_residual"] != 0]
        total_graded = [(tp > 0) == (r["total_residual"] > 0) for r, tp in zip(test_recs, total_pred) if r["total_residual"] != 0]
        walk_forward.append({
            "season": season,
            "games": len(test_recs),
            "ml_brier": round(_brier(truth, [float(p) for p in p_home]), 5),
            "ml_free_brier": round(_brier(truth, [float(p) for p in p_free]), 5),
            "market_spread_logit_brier": round(_brier(truth, [float(p) for p in p_market]), 5),
            "spread_direction_rate": round(sum(cover_graded) / len(cover_graded), 4) if cover_graded else None,
            "total_direction_rate": round(sum(total_graded) / len(total_graded), 4) if total_graded else None,
        })

    # Residual scale of each head, from out-of-fold predictions only.
    sigma_margin = float(np.std([o["margin_residual"] - o["spread_pred"] for o in oof], ddof=1))
    sigma_total = float(np.std([o["total_residual"] - o["total_pred"] for o in oof], ddof=1))

    # Market benchmark for the moneyline head.
    market_ml_brier = _brier(
        [o["home_win"] for o in oof if o["home_moneyline"] and o["away_moneyline"]],
        [
            _implied(o["home_moneyline"]) / (_implied(o["home_moneyline"]) + _implied(o["away_moneyline"]))
            for o in oof if o["home_moneyline"] and o["away_moneyline"]
        ],
    )

    # Segment search at recorded prices.
    segments: dict[str, dict[str, Any]] = {}
    for market_key, pred_key, actual_key, odds_keys in (
        ("spread", "spread_pred", "margin_residual", ("home_spread_odds", "away_spread_odds")),
        ("total", "total_pred", "total_residual", ("over_odds", "under_odds")),
    ):
        for direction in SEGMENT_DIRECTIONS[market_key]:
            for threshold in SEGMENT_THRESHOLDS:
                picks: list[tuple[int, float, bool | None]] = []
                for o in oof:
                    pred = o[pred_key]
                    if abs(pred) < threshold:
                        continue
                    positive_side = pred > 0  # home cover / over
                    side_name = ("home" if positive_side else "away") if market_key == "spread" else ("over" if positive_side else "under")
                    if direction != "both" and side_name != direction:
                        continue
                    odds = o[odds_keys[0]] if positive_side else o[odds_keys[1]]
                    if odds is None or odds < MAX_JUICE:
                        continue
                    actual = o[actual_key]
                    hit = None if actual == 0 else (actual > 0) == positive_side
                    picks.append((o["season"], _profit(hit, odds), hit))
                stats = _segment_stats(picks)
                stats.update({"market": market_key, "direction": direction, "min_residual": threshold, "qualifies": _qualifies(stats)})
                segments[f"{market_key}:{direction}:{threshold:g}"] = stats

    # Moneyline: two-sided edge buckets against the no-vig closing moneyline.
    ml_buckets: dict[str, dict[str, Any]] = {}
    for lo, hi in ((0.0, 0.02), (0.02, 0.04), (0.04, 0.07), (0.07, 1.0)):
        picks = []
        for o in oof:
            if not (o["home_moneyline"] and o["away_moneyline"]):
                continue
            ih, ia = _implied(o["home_moneyline"]), _implied(o["away_moneyline"])
            no_vig_home = ih / (ih + ia)
            edge_home = o["p_home"] - no_vig_home
            edge_away = (1.0 - o["p_home"]) - (1.0 - no_vig_home)
            pick_home = edge_home >= edge_away
            edge = edge_home if pick_home else edge_away
            if not (lo <= edge < hi):
                continue
            odds = o["home_moneyline"] if pick_home else o["away_moneyline"]
            picks.append((o["season"], _profit(bool(o["home_win"]) == pick_home, odds), bool(o["home_win"]) == pick_home))
        stats = _segment_stats(picks)
        stats.update({"edge_lower": lo, "edge_upper": hi, "qualifies": _qualifies(stats)})
        ml_buckets[f"edge:{lo:g}-{hi:g}"] = stats

    def ladders(market_key: str) -> list[dict[str, Any]]:
        pred_key, actual_key, odds_keys = {
            "spread": ("spread_pred", "margin_residual", ("home_spread_odds", "away_spread_odds")),
            "total": ("total_pred", "total_residual", ("over_odds", "under_odds")),
        }[market_key]
        rows: list[dict[str, Any]] = []
        for direction in SEGMENT_DIRECTIONS[market_key][:2]:
            rows.extend(build_ladder(
                oof, market_key=market_key, direction=direction,
                pred_key=pred_key, actual_key=actual_key, odds_keys=odds_keys,
            ))
        return rows

    spread_ladder = ladders("spread")
    total_ladder = ladders("total")
    decision_policy = {
        "h2h": {
            "mode": "research_only",
            "reason": (
                "Walk-forward moneyline probabilities match the closing market (Brier parity); "
                "no two-sided edge bucket clears the qualification bar."
            ),
            "buckets": ml_buckets,
        },
        "spread": {
            "mode": "segment_gate" if spread_ladder else "research_only",
            "segments": spread_ladder,
            "reason": "" if spread_ladder else "No spread residual band clears any tier bar at recorded prices.",
        },
        "totals": {
            "mode": "segment_gate" if total_ladder else "research_only",
            "segments": total_ladder,
            "reason": "" if total_ladder else "No total residual band clears any tier bar at recorded prices.",
        },
        "tier_bars": list(TIER_BARS),
        "qualification_bar": {
            "min_band_picks": MIN_BAND_PICKS,
            "max_juice": MAX_JUICE,
            "evidence": "flat_roi_at_recorded_closing_prices",
            "ladder_rule": (
                "Strongest tier first, on the direction's own picks only. A tier takes the loosest threshold whose "
                "cumulative tail clears its bar; a LEAN band below BET must itself clear the LEAN bar. The loosest "
                "BET threshold that leaves a validated LEAN band wins; a lighter LEAN exists only where no regular "
                "LEAN could be placed."
            ),
        },
    }

    final_ml = _logistic().fit(matrix(records, HEAD_FEATURES["moneyline"]), [r["home_win"] for r in records])
    final_ml_free = _logistic().fit(matrix(records, HEAD_FEATURES["moneyline_free"]), [r["home_win"] for r in records])
    final_spread = _ridge().fit(matrix(records, HEAD_FEATURES["spread"]), [r["margin_residual"] for r in records])
    final_total = _ridge().fit(matrix(records, HEAD_FEATURES["total"]), [r["total_residual"] for r in records])

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_ml, ARTIFACT_DIR / "nfl_ml.joblib")
    joblib.dump(final_ml_free, ARTIFACT_DIR / "nfl_ml_free.joblib")
    joblib.dump(final_spread, ARTIFACT_DIR / "nfl_spread.joblib")
    joblib.dump(final_total, ARTIFACT_DIR / "nfl_total.joblib")
    legacy_iso = ARTIFACT_DIR / "nfl_ml_isotonic.joblib"
    if legacy_iso.exists():
        legacy_iso.unlink()

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "train_window": [first_season, last_season],
        "games": len(records),
        "team_stats_seasons": stats_seasons,
        "feature_names": FEATURE_NAMES,
        "market_features": ["spread_line", "total_line", "elo_minus_spread"],
        "head_features": HEAD_FEATURES,
        "head_families": {"moneyline": "logistic", "moneyline_free": "logistic", "spread": "ridge", "total": "ridge"},
        "margin_residual_sigma": round(sigma_margin, 4),
        "total_residual_sigma": round(sigma_total, 4),
        "total_side_probability": fit_signed_side_probability(oof, pred_key="total_pred", actual_key="total_residual"),
        "walk_forward": walk_forward,
        "oof_ml_brier": round(_brier([o["home_win"] for o in oof], [o["p_home"] for o in oof]), 5),
        "oof_ml_free_brier": round(_brier([o["home_win"] for o in oof], [o["p_free"] for o in oof]), 5),
        "market_reference_brier": round(_brier([o["home_win"] for o in oof], [o["p_market"] for o in oof]), 5),
        "market_moneyline_brier": round(market_ml_brier, 5) if not math.isnan(market_ml_brier) else None,
        "segment_search": segments,
        "decision_policy": decision_policy,
        "promotion_status": "segment_gate_graduated_tiers",
        "notes": (
            "Every head is anchored to the posted line. BET/LEAN are minted only through decision_policy "
            "bands validated at recorded closing prices (BET = strongest validated band, LEAN = the next band "
            "that still clears its own bar); everything else publishes as visible PASS research."
        ),
    }
    (ARTIFACT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    meta = train()
    print(json.dumps({k: meta[k] for k in ("model_version", "games", "oof_ml_brier", "market_reference_brier", "margin_residual_sigma", "total_residual_sigma")}, indent=2))
    for row in meta["walk_forward"]:
        print(row)
    print(json.dumps(meta["decision_policy"], indent=2)[:4000])
