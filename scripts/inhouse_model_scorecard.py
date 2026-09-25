#!/usr/bin/env python3
"""Score every owned model from frozen pregame publications and real prices.

Team rows come from the certified pregame ledger. Player props come from the
archived publication files, with only their final result joined from the
outcome ledger. Neither mutable daily cache nor assumed odds can create a bet.
This report is evidence for review; it never promotes a model automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.model_scorecard_stats import binary_score, clustered_roi_interval  # noqa: E402
from scripts.team_prop_model_evaluator import (  # noqa: E402
    SUPPORTED_MODEL_KEYS,
    evaluate_team_prop_ledger,
)


SCHEMA_VERSION = 1
FORWARD_SINCE = "2026-09-22T00:00:00Z"
MIN_PRICED_SETTLED = 100
PROP_MODEL_KEYS = (
    "mlb_player_props", "wnba_player_props", "nfl_player_props",
    "cfb_player_props", "nba_player_props",
)
BAD_PRICE_MARKERS = ("assumed", "proxy", "synthetic", "default", "unpriced", "model_output")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def _first(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _prop_version(pick: Mapping[str, Any], bucket: Mapping[str, Any]) -> str:
    version = str(_first(pick, "ml_model_version", "model_version", "prediction_model_version")
                  or _first(bucket, "model_version", "prediction_model_version") or "unversioned")
    fingerprint = str(_first(pick, "ml_training_fingerprint", "training_fingerprint") or "")
    return f"{version}:{fingerprint[:12]}" if fingerprint else version


def _prop_market(pick: Mapping[str, Any]) -> str:
    return str(_first(pick, "stat_key", "market", "market_type") or "unknown").strip().lower()


def _slot_id(pick: Mapping[str, Any]) -> str:
    supplied = str(pick.get("id") or "").strip()
    if supplied:
        return supplied
    values = [_first(pick, key) for key in (
        "date", "game_id", "player_id", "stat_key", "selection", "line",
    )]
    return hashlib.sha256(json.dumps(values, default=str).encode()).hexdigest()[:24]


def _prop_price(
    pick: Mapping[str, Any], published: datetime, start: datetime,
) -> tuple[float | None, str | None]:
    if pick.get("market_priced") is not True:
        return None, "not_market_priced"
    provenance = " ".join(str(pick.get(key) or "").lower() for key in (
        "pricing_type", "odds_source", "line_source", "market_source",
    ))
    if any(marker in provenance for marker in BAD_PRICE_MARKERS):
        return None, "assumed_or_proxy_price"
    odds = _number(pick.get("odds"))
    if odds is None or abs(odds) < 100:
        return None, "missing_observed_odds"
    quote_at = _timestamp(_first(pick, "market_updated_at", "market_retrieved_at",
                                 "odds_updated_at", "price_updated_at"))
    if quote_at is None:
        return None, "missing_quote_timestamp"
    if quote_at >= start or published >= start:
        return None, "post_start"
    age_hours = (published - quote_at).total_seconds() / 3600.0
    if age_hours < -5.0 / 60.0:
        return None, "quote_after_publication"
    if age_hours > 24.0:
        return None, "stale_quote"
    return odds, None


def _prop_outcomes(outcome_ledger: Mapping[str, Any]) -> dict[str, str]:
    results: dict[str, str] = {}
    for record in outcome_ledger.get("records") or []:
        if not isinstance(record, dict) or record.get("cache_type") != "player_props_cache":
            continue
        snapshot = record.get("pregame_snapshot")
        if not isinstance(snapshot, dict):
            continue
        pick_id = str(snapshot.get("id") or "").strip()
        result = str(record.get("result") or "").lower()
        if pick_id and result in {"win", "loss", "push"}:
            results[pick_id] = result
    return results


def _prop_slots(snapshot_dir: Path) -> tuple[dict[tuple[str, str, str, str, str], dict[str, Any]], Counter]:
    slots: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    exclusions: Counter = Counter()
    for path in sorted(snapshot_dir.glob("20??-??-??/*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            exclusions["unreadable_snapshot"] += 1
            continue
        if not isinstance(payload, dict):
            exclusions["invalid_snapshot"] += 1
            continue
        published = _timestamp(_first(payload, "publishedAt", "generatedAt", "updatedAt"))
        if published is None:
            exclusions["missing_publication_timestamp"] += 1
            continue
        models = payload.get("models") if isinstance(payload.get("models"), dict) else {}
        for model_key in PROP_MODEL_KEYS:
            bucket = models.get(model_key)
            if not isinstance(bucket, dict):
                continue
            for pick in bucket.get("picks") or []:
                if not isinstance(pick, dict):
                    continue
                start = _timestamp(_first(pick, "start_time", "game_start_time", "event_start_time"))
                if start is None or published >= start:
                    exclusions["missing_or_post_start_game_clock"] += 1
                    continue
                version = _prop_version(pick, bucket)
                market = _prop_market(pick)
                variant = str(pick.get("model_variant") or "base")
                key = (model_key, version, market, variant, _slot_id(pick))
                item = {"pick": pick, "published": published, "start": start}
                slot = slots.setdefault(key, {"forecast": None, "action": None})
                if slot["forecast"] is None or published < slot["forecast"]["published"]:
                    slot["forecast"] = item
                if str(pick.get("decision") or "").upper() in {"BET", "LEAN"}:
                    if slot["action"] is None or published < slot["action"]["published"]:
                        slot["action"] = item
    return slots, exclusions


def _prop_group_report(slots: list[dict[str, Any]], outcomes: Mapping[str, str]) -> dict[str, Any]:
    model_binary: list[tuple[float, int]] = []
    paired_model: list[tuple[float, int]] = []
    paired_market: list[tuple[float, int]] = []
    returns: list[tuple[str, float, float]] = []
    shadow_returns: list[tuple[str, float, float]] = []
    results = Counter()
    excluded = Counter()
    for slot in slots:
        forecast = slot["forecast"]
        if forecast is None:
            continue
        pick = forecast["pick"]
        result = outcomes.get(_slot_id(pick), str(pick.get("result") or "pending").lower())
        results[result] += 1
        probability = _number(pick.get("probability"))
        if result in {"win", "loss"} and probability is not None and 0 <= probability <= 1:
            outcome = int(result == "win")
            model_binary.append((probability, outcome))
            odds, _ = _prop_price(pick, forecast["published"], forecast["start"])
            fair = _number(pick.get("market_no_vig_selected_probability"))
            if odds is not None and fair is not None and 0 < fair < 1:
                paired_model.append((probability, outcome))
                paired_market.append((fair, outcome))
        action = slot["action"]
        is_shadow = action is None and str(pick.get("shadow_decision") or "").upper() in {"BET", "LEAN"}
        if action is None and not is_shadow:
            continue
        action = forecast if is_shadow else action
        bet = action["pick"]
        bet_result = outcomes.get(_slot_id(bet), str(bet.get("result") or "pending").lower())
        odds, reason = _prop_price(bet, action["published"], action["start"])
        if reason:
            excluded[reason] += 1
            continue
        stake = _number(bet.get("shadow_units") if is_shadow else bet.get("units"))
        if stake is None or stake <= 0:
            excluded["missing_positive_stake"] += 1
            continue
        if bet_result not in {"win", "loss", "push"}:
            excluded["unsettled"] += 1
            continue
        assert odds is not None
        profit = (stake * (odds / 100 if odds > 0 else 100 / abs(odds))) if bet_result == "win" else (-stake if bet_result == "loss" else 0.0)
        event = str(_first(bet, "game_id", "event_id") or "") or str(_first(bet, "date", "matchup") or _slot_id(bet))
        (shadow_returns if is_shadow else returns).append((event, stake, profit))
    stake_total = sum(stake for _, stake, _ in returns)
    profit_total = sum(profit for _, _, profit in returns)
    model_score = binary_score(model_binary)
    paired_score = binary_score(paired_model)
    market_score = binary_score(paired_market)
    return {
        "forecasts": len(slots),
        "result_counts": dict(sorted(results.items())),
        "model": model_score,
        "market_comparison": {"paired_model": paired_score, "observed_no_vig": market_score},
        "priced_settled_bets": len(returns),
        "stake_units": round(stake_total, 6),
        "profit_units": round(profit_total, 6),
        "roi": round(profit_total / stake_total, 6) if stake_total else None,
        "roi_interval": clustered_roi_interval(returns),
        "shadow_priced_settled": len(shadow_returns),
        "shadow_stake_units": round(sum(stake for _, stake, _ in shadow_returns), 6),
        "shadow_profit_units": round(sum(profit for _, _, profit in shadow_returns), 6),
        "shadow_roi": round(sum(profit for _, _, profit in shadow_returns) / sum(stake for _, stake, _ in shadow_returns), 6) if shadow_returns else None,
        "shadow_roi_interval": clustered_roi_interval(shadow_returns),
        "exclusions": dict(sorted(excluded.items())),
    }


def _team_card(segment: Mapping[str, Any]) -> dict[str, Any]:
    roi = segment["real_price_roi"]
    model = segment["model_metrics"]
    return {
        "source": "certified_team_ledger",
        "model_key": segment["model_key"],
        "model_version": segment["model_version"],
        "market": segment["market"],
        "variant": "base",
        "forecasts": segment["records"],
        "result_counts": segment["result_counts"],
        "model": {
            "samples": model["settled_records"], "wins": model["wins"],
            "hit_rate": model["hit_rate"], "brier": model["brier_score"],
            "ece": segment["calibration"]["expected_calibration_error"],
        },
        "market_comparison": {
            "paired_model_brier": segment["model_on_same_priced_sample"]["brier_score"],
            "observed_market_brier": segment["market_benchmark"]["brier_score"],
            "paired_samples": segment["market_benchmark"]["settled_records"],
            "probability_sources": segment["market_benchmark"]["probability_sources"],
        },
        "priced_settled_bets": roi["priced_settled_actionable_records"],
        "stake_units": roi["stake_units"],
        "profit_units": roi["profit_units"],
        "roi": roi["roi"],
        "roi_interval": roi["roi_interval"],
        "shadow_real_price_roi": segment["shadow_real_price_roi"],
        "exclusions": roi["excluded"],
    }


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def build_scorecard(
    team_ledger: Mapping[str, Any], outcome_ledger: Mapping[str, Any],
    snapshot_dir: Path, *, forward_since: str = FORWARD_SINCE,
) -> dict[str, Any]:
    boundary = _timestamp(forward_since)
    if boundary is None:
        raise ValueError("forward_since must be an aware ISO timestamp")
    team = evaluate_team_prop_ledger(team_ledger, forward_since=forward_since)
    cards = [_team_card(segment) for segment in team["segments"]]
    forward_cards = [_team_card(segment) for segment in team["forward_holdout"]]
    slots, prop_exclusions = _prop_slots(snapshot_dir)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    forward_grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (model_key, version, market, variant, _), slot in slots.items():
        group = (model_key, version, market, variant)
        grouped[group].append(slot)
        first = slot["forecast"]
        if first is not None and first["published"] >= boundary:
            forward_grouped[group].append(slot)
    outcomes = _prop_outcomes(outcome_ledger)
    for groups, target in ((grouped, cards), (forward_grouped, forward_cards)):
        for (model_key, version, market, variant), members in sorted(groups.items()):
            target.append({
                "source": "immutable_player_prop_snapshots",
                "model_key": model_key, "model_version": version,
                "market": market, "variant": variant,
                **_prop_group_report(members, outcomes),
            })
    observed = {card["model_key"] for card in cards}
    for model_key in sorted((set(SUPPORTED_MODEL_KEYS) | set(PROP_MODEL_KEYS)) - observed):
        if model_key not in observed:
            cards.append({"model_key": model_key, "status": "no_certified_evidence", "forecasts": 0})

    def status(card: dict[str, Any]) -> None:
        if "status" in card:
            return
        count = int(card["priced_settled_bets"])
        lower = card["roi_interval"]["lower_95"]
        card["status"] = (
            "insufficient_priced_sample" if count < MIN_PRICED_SETTLED
            else "unproven_return" if lower is None or lower <= 0
            else "requires_untouched_holdout_review"
        )
        card["promotion_approved"] = False

    for card in cards + forward_cards:
        status(card)
    cards.sort(key=lambda c: (c["model_key"], c.get("model_version", ""), c.get("market", ""), c.get("variant", "")))
    forward_cards.sort(key=lambda c: (c["model_key"], c.get("model_version", ""), c.get("market", ""), c.get("variant", "")))
    return {
        "schema_version": SCHEMA_VERSION,
        "forward_since": forward_since,
        "minimum_priced_settled_for_review": MIN_PRICED_SETTLED,
        "promotion_note": "A positive historical scorecard is not an untouched holdout or automatic staking approval.",
        "sources": {
            "team": "data/calibration/team_prop_pregame_ledger.json",
            "player_props": "data/player_props_snapshots/YYYY-MM-DD/*.json",
            "prop_outcomes": "data/calibration/outcome_ledger.json (result join only)",
        },
        "team_record_quality": team["record_quality"],
        "player_prop_snapshot_exclusions": dict(sorted(prop_exclusions.items())),
        "scorecards": cards,
        "forward_scorecards": forward_cards,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, help="Write JSON here; stdout when omitted.")
    args = parser.parse_args()
    root = args.repo_root
    report = build_scorecard(
        _read(root / "data/calibration/team_prop_pregame_ledger.json"),
        _read(root / "data/calibration/outcome_ledger.json"),
        root / "data/player_props_snapshots",
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
