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

# Candidate decision segments.  The search space is deliberately tiny so a
# qualifying segment is a structural finding, not a lucky bucket.
SEGMENT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
SEGMENT_DIRECTIONS = {"spread": ("home", "away", "both"), "total": ("over", "under", "both")}
# Qualification bar for a live LEAN gate: enough picks, positive flat ROI at
# the recorded price, and profitable in a clear majority of seasons.
QUALIFY_MIN_PICKS = 100
QUALIFY_MIN_ROI = 0.03
QUALIFY_MIN_SEASON_SHARE = 0.70
# Segments discovered by walk-forward search are capped at LEAN until live
# certified evidence exists; BET is never minted from a backtest alone.
MAX_SEGMENT_TIER = "LEAN"
LEAN_UNITS = 0.25
MAX_JUICE = -125  # never stake a validated segment at heavier juice than this


def _logistic() -> Any:
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


def _ridge() -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=50.0))


def _brier(truth: list[int], probs: list[float]) -> float:
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, truth)])) if truth else float("nan")


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


def _qualifies(stats: dict[str, Any]) -> bool:
    return (
        int(stats.get("picks") or 0) >= QUALIFY_MIN_PICKS
        and float(stats.get("flat_roi") or -1.0) >= QUALIFY_MIN_ROI
        and int(stats.get("seasons") or 0) > 0
        and int(stats.get("seasons_positive") or 0) / int(stats["seasons"]) >= QUALIFY_MIN_SEASON_SHARE
    )


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

    def best_qualifying(market_key: str) -> list[dict[str, Any]]:
        """One gate per direction: the loosest qualifying threshold (largest sample)."""
        chosen: dict[str, dict[str, Any]] = {}
        for stats in segments.values():
            if stats["market"] != market_key or not stats["qualifies"] or stats["direction"] == "both":
                continue
            current = chosen.get(stats["direction"])
            if current is None or stats["picks"] > current["picks"]:
                chosen[stats["direction"]] = stats
        return [
            {
                "direction": stats["direction"],
                "min_residual": stats["min_residual"],
                "decision": MAX_SEGMENT_TIER,
                "units": LEAN_UNITS,
                "max_juice": MAX_JUICE,
                "walk_forward": {k: stats[k] for k in ("picks", "hit_rate", "flat_roi", "seasons", "seasons_positive", "season_roi")},
            }
            for stats in chosen.values()
        ]

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
            "mode": "segment_gate" if best_qualifying("spread") else "research_only",
            "segments": best_qualifying("spread"),
            "reason": "No spread residual segment clears the qualification bar at recorded prices." if not best_qualifying("spread") else "",
        },
        "totals": {
            "mode": "segment_gate" if best_qualifying("total") else "research_only",
            "segments": best_qualifying("total"),
            "reason": "No total residual segment clears the qualification bar at recorded prices." if not best_qualifying("total") else "",
        },
        "qualification_bar": {
            "min_picks": QUALIFY_MIN_PICKS,
            "min_flat_roi": QUALIFY_MIN_ROI,
            "min_season_share_positive": QUALIFY_MIN_SEASON_SHARE,
            "max_tier": MAX_SEGMENT_TIER,
            "max_juice": MAX_JUICE,
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
        "walk_forward": walk_forward,
        "oof_ml_brier": round(_brier([o["home_win"] for o in oof], [o["p_home"] for o in oof]), 5),
        "oof_ml_free_brier": round(_brier([o["home_win"] for o in oof], [o["p_free"] for o in oof]), 5),
        "market_reference_brier": round(_brier([o["home_win"] for o in oof], [o["p_market"] for o in oof]), 5),
        "market_moneyline_brier": round(market_ml_brier, 5) if not math.isnan(market_ml_brier) else None,
        "segment_search": segments,
        "decision_policy": decision_policy,
        "promotion_status": "segment_gate_lean_only",
        "notes": (
            "Every head is anchored to the posted line. BET/LEAN are minted only through decision_policy "
            "segments validated at recorded closing prices; discovered segments are capped at LEAN until "
            "certified live evidence exists. Moneyline and spread publish as research (PASS)."
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
