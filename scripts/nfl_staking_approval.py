#!/usr/bin/env python3
"""Earn (never invent) an NFL staking approval from the frozen holdout.

The publication gate in ``scripts/model_stake_policy.py`` cannot infer
approval from a pretty scorecard. This script is the only writer for an
``nfl`` row in ``data/calibration/staking_approvals.json``, and it writes
only when every ``_approved()`` field is actually true on an unused
chronological holdout of the frozen rule.

Live BET/LEAN is demoted to PASS @ 0u until that happens. Shadow
decisions still grade so the holdout sample can grow.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.inhouse_model_scorecard import MIN_PRICED_SETTLED  # noqa: E402
from scripts.model_scorecard_stats import binary_score, clustered_roi_interval  # noqa: E402
from scripts.model_stake_policy import POLICY_PATH, _approved  # noqa: E402
from scripts.price_clock import borrow_missing_quote_clocks, observed_quote_timing  # noqa: E402
from scripts.team_prop_pregame_ledger import (  # noqa: E402
    backfill_team_prop_pregame_from_cache,
    load_team_prop_pregame_ledger,
)
from scripts.team_prop_model_evaluator import (  # noqa: E402
    certification_status,
    _american_odds,
    _financial_eligible,
    _price_provenance_is_disallowed,
    _result_label,
    _value_from_contexts,
)


FREEZE_PATH = REPO_ROOT / "data" / "calibration" / "nfl_staking_freeze.json"
MATERIAL_BRIER_REGRESSION = 0.01


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    freeze = _read_json(path)
    if freeze.get("model_key") != "nfl" or not isinstance(freeze.get("holdout"), dict):
        raise ValueError("invalid NFL staking freeze")
    return freeze


def _candidate_action(record: Mapping[str, Any]) -> tuple[str, float | None]:
    """Return the frozen-rule action, preferring shadow after the 0u gate."""

    shadow = str(_value_from_contexts(record, "shadow_decision") or "").upper()
    shadow_units = _number(_value_from_contexts(record, "shadow_units"))
    if shadow in {"BET", "LEAN"}:
        return shadow, shadow_units
    source = str(_value_from_contexts(record, "source_decision", "raw_decision") or "").upper()
    source_units = _number(_value_from_contexts(record, "source_units", "raw_stake", "units", "stake"))
    if source in {"BET", "LEAN"}:
        return source, source_units
    decision = str(_value_from_contexts(record, "decision") or "").upper()
    stake = _number(_value_from_contexts(record, "units", "stake"))
    return decision, stake


def _holdout_price_context(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Use the captured ledger price; old records fall back to their snapshot.

    Odds and provenance stay on the ledger price. Quote clocks that were stored
    only on the pregame image (or the top-level row) are filled in when the
    price blob itself has none.
    """

    price = record.get("price")
    if isinstance(price, Mapping) and price:
        return borrow_missing_quote_clocks(price, record.get("pregame_snapshot"), record)
    snapshot = record.get("pregame_snapshot")
    return snapshot if isinstance(snapshot, Mapping) else record


def _holdout_price_provenance_is_disallowed(record: Mapping[str, Any]) -> bool:
    # A model's earlier pregame image may still contain an assumed price after
    # an observed sportsbook quote was attached to the immutable ledger row.
    return _price_provenance_is_disallowed({"price": _holdout_price_context(record)})


def _holdout_american_odds(record: Mapping[str, Any]) -> float | None:
    price = record.get("price")
    if isinstance(price, Mapping) and price:
        odds = _number(price.get("odds"))
        captured = _number(record.get("observed_american_odds"))
        if (odds is None or odds == 0 or abs(odds) < 100
                or (captured is not None and captured != odds)):
            return None
        return odds
    return _american_odds(record)


def _holdout_market_probability(record: Mapping[str, Any]) -> float | None:
    price = record.get("price")
    if isinstance(price, Mapping) and price:
        for field in ("market_no_vig_selected_probability", "market_pick_prob",
                      "market_probability", "market_implied_probability"):
            probability = _number(price.get(field))
            if probability is not None:
                return probability
        return None
    return _number(_value_from_contexts(
        record, "market_no_vig_selected_probability", "market_probability",
        "market_implied_probability",
    ))


def _in_holdout(record: Mapping[str, Any], starts_at: datetime) -> bool:
    published = _timestamp(record.get("published_at") or record.get("data_as_of"))
    return published is not None and published >= starts_at


def _version_matches(record: Mapping[str, Any], fitted: str) -> bool:
    version = str(record.get("model_version") or "")
    if version == fitted:
        return True
    snapshot = record.get("pregame_snapshot") if isinstance(record.get("pregame_snapshot"), dict) else {}
    return str(snapshot.get("model_version") or "") == fitted


def evaluate_frozen_holdout(
    ledger: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    holdout = freeze["holdout"]
    starts_at = _timestamp(holdout.get("starts_at"))
    if starts_at is None:
        raise ValueError("freeze holdout.starts_at must be an aware ISO timestamp")
    fitted = str(freeze.get("fitted_version") or "")
    market = str(freeze.get("market") or "totals").lower()
    model_key = str(freeze.get("model_key") or "nfl")

    exclusions: dict[str, int] = {}
    returns: list[tuple[str, float, float]] = []
    model_binary: list[tuple[float, int]] = []
    paired_model: list[tuple[float, int]] = []
    paired_market: list[tuple[float, int]] = []
    certified = pending = 0
    for raw in ledger.get("records") or []:
        if not isinstance(raw, dict) or str(raw.get("model_key") or "") != model_key:
            continue
        if str(raw.get("market") or "").lower() != market:
            continue
        if not _version_matches(raw, fitted):
            exclusions["version_mismatch"] = exclusions.get("version_mismatch", 0) + 1
            continue
        if not _in_holdout(raw, starts_at):
            exclusions["before_holdout_start"] = exclusions.get("before_holdout_start", 0) + 1
            continue
        ok, reason = certification_status(raw, ledger)
        if not ok:
            exclusions[f"uncertified:{reason}"] = exclusions.get(f"uncertified:{reason}", 0) + 1
            continue
        certified += 1
        probability = _number(
            _value_from_contexts(raw, "raw_probability", "displayed_probability", "probability")
        )
        # The grader attaches the outcome to the ledger row after publication;
        # a retained pregame image can still carry a pending result.
        result = str(raw.get("result") or _result_label(raw)).strip().lower()
        binary_outcome = (
            int(result == "win") if result in {"win", "loss"}
            and probability is not None and 0 <= probability <= 1 else None
        )
        if binary_outcome is not None:
            model_binary.append((probability, binary_outcome))
        action, stake = _candidate_action(raw)
        if action not in {"BET", "LEAN"}:
            exclusions["not_actionable"] = exclusions.get("not_actionable", 0) + 1
            continue
        if not _financial_eligible(raw) or _holdout_price_provenance_is_disallowed(raw):
            exclusions[str(raw.get("financial_eligibility_reason") or "not_financial")] = (
                exclusions.get(str(raw.get("financial_eligibility_reason") or "not_financial"), 0) + 1
            )
            continue
        price_clock = observed_quote_timing(
            _holdout_price_context(raw), published_at=raw.get("published_at"),
            start_at=raw.get("game_start_time"),
        )
        if price_clock:
            exclusions[price_clock] = exclusions.get(price_clock, 0) + 1
            continue
        odds = _holdout_american_odds(raw)
        if odds is None:
            exclusions["missing_verified_american_price"] = exclusions.get("missing_verified_american_price", 0) + 1
            continue
        if stake is None or stake <= 0:
            exclusions["missing_positive_stake"] = exclusions.get("missing_positive_stake", 0) + 1
            continue
        # A paired market score needs the same observed, timely, verified
        # stake and quote as ROI. Forecast calibration can use an unpriced row.
        if binary_outcome is not None:
            market_p = _holdout_market_probability(raw)
            if market_p is not None and 0 < market_p < 1:
                paired_model.append((probability, binary_outcome))
                paired_market.append((market_p, binary_outcome))
        if result not in {"win", "loss", "push"}:
            pending += 1
            exclusions["unsettled"] = exclusions.get("unsettled", 0) + 1
            continue
        profit = 0.0 if result == "push" else (stake * (odds / 100.0 if odds > 0 else 100.0 / abs(odds)))
        if result == "loss":
            profit = -stake
        event = str(_value_from_contexts(raw, "game_id", "event_id") or "").strip() or str(
            _value_from_contexts(raw, "slate_date", "matchup") or raw.get("id")
        )
        returns.append((event, float(stake), float(profit)))

    stake_total = sum(item[1] for item in returns)
    profit_total = sum(item[2] for item in returns)
    roi = (profit_total / stake_total) if stake_total else None
    interval = clustered_roi_interval(returns)
    model_score = binary_score(model_binary)
    paired_score = binary_score(paired_model)
    market_score = binary_score(paired_market)
    paired_n = int(paired_score.get("samples") or 0)
    model_brier = paired_score.get("brier")
    market_brier = market_score.get("brier")
    if paired_n >= 20 and model_brier is not None and market_brier is not None:
        calibration_ok = float(model_brier) <= float(market_brier) + MATERIAL_BRIER_REGRESSION
        calibration_reason = "paired_brier_within_tolerance" if calibration_ok else "material_brier_regression"
    else:
        calibration_ok = False
        calibration_reason = "insufficient_paired_market_sample"

    n = len(returns)
    lower = interval.get("lower_95")
    clears = (
        holdout.get("unused_during_selection") is True
        and n >= MIN_PRICED_SETTLED
        and roi is not None and roi > 0
        and lower is not None and float(lower) > 0
        and calibration_ok is True
    )
    return {
        "model_key": model_key,
        "fitted_version": fitted,
        "market": market,
        "variant": str(freeze.get("variant") or "base"),
        "frozen_rule": freeze.get("frozen_rule"),
        "frozen_at": freeze.get("frozen_at"),
        "holdout_starts_at": holdout.get("starts_at"),
        "unused_during_selection": holdout.get("unused_during_selection") is True,
        "certified_holdout_forecasts": certified,
        "independently_priced_settled": n,
        "pending_actionable": pending,
        "stake_units": round(stake_total, 6),
        "profit_units": round(profit_total, 6),
        "roi": round(roi, 6) if roi is not None else None,
        "clustered_lower_95": lower,
        "clustered_upper_95": interval.get("upper_95"),
        "event_clusters": interval.get("event_clusters"),
        "model": model_score,
        "paired_model": paired_score,
        "observed_market": market_score,
        "calibration_no_material_regression": calibration_ok,
        "calibration_reason": calibration_reason,
        "exclusions": dict(sorted(exclusions.items())),
        "clears_gate": clears,
        "approval_written": False,
        "blocker": (
            None if clears
            else "holdout_not_unused" if holdout.get("unused_during_selection") is not True
            else "insufficient_priced_settled" if n < MIN_PRICED_SETTLED
            else "nonpositive_roi" if roi is None or roi <= 0
            else "nonpositive_clustered_lower_95" if lower is None or float(lower) <= 0
            else "calibration_regression_or_unproven"
        ),
    }


def build_approval_entry(freeze: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_key": freeze["model_key"],
        "model_version": freeze["fitted_version"],
        "market": freeze["market"],
        "variant": freeze.get("variant") or "base",
        "approved": True,
        "frozen_rule": freeze["frozen_rule"],
        "holdout": {
            "unused_during_selection": True,
            "starts_at": freeze["holdout"]["starts_at"],
            "independently_priced_settled": int(report["independently_priced_settled"]),
            "roi": float(report["roi"]),
            "clustered_lower_95": float(report["clustered_lower_95"]),
            "calibration_no_material_regression": True,
            "calibration_reason": report.get("calibration_reason"),
        },
    }


def maybe_write_approval(
    freeze: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    policy_path: Path = POLICY_PATH,
    write: bool = False,
) -> bool:
    """Write a valid approval only when the holdout actually clears."""

    if not report.get("clears_gate"):
        return False
    entry = build_approval_entry(freeze, report)
    if not _approved(entry):
        return False
    if not write:
        return False
    policy = _read_json(policy_path)
    if policy.get("schema_version") != 1 or not isinstance(policy.get("approvals"), list):
        raise ValueError("invalid staking approval policy")
    key = (entry["model_key"], entry["model_version"], entry["market"], entry["variant"])
    kept = [
        row for row in policy["approvals"]
        if not isinstance(row, dict) or (
            str(row.get("model_key")), str(row.get("model_version")),
            str(row.get("market")), str(row.get("variant") or "base"),
        ) != key
    ]
    policy["approvals"] = kept + [entry]
    policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def diagnose_ledger(ledger: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    """Explain why NFL cannot yet clear a real staking approval."""

    records = [row for row in (ledger.get("records") or []) if isinstance(row, dict) and row.get("model_key") == "nfl"]
    versions: dict[str, int] = {}
    cert: dict[str, int] = {}
    markets: dict[str, int] = {}
    for row in records:
        versions[str(row.get("model_version") or "")] = versions.get(str(row.get("model_version") or ""), 0) + 1
        status = str((row.get("certification") or {}).get("reason") or (row.get("certification") or {}).get("status") or "")
        cert[status] = cert.get(status, 0) + 1
        markets[str(row.get("market") or "")] = markets.get(str(row.get("market") or ""), 0) + 1
    return {
        "nfl_records": len(records),
        "versions": dict(sorted(versions.items())),
        "certification_reasons": dict(sorted(cert.items())),
        "markets": dict(sorted(markets.items())),
        "fitted_target": freeze.get("fitted_version"),
        "holdout_starts_at": freeze.get("holdout", {}).get("starts_at"),
        "notes": [
            "Historical 2012-2025 residual-band search is mined and is not a holdout.",
            "Sunday 1pm ET slates were often first written after kickoff when a slow refresh restamped publication clocks.",
            "Serving-hash prediction_model_version values fragment the scorecard; new captures key on the fitted nfl_v1 label.",
            "Unapproved BET/LEAN publish as PASS @ 0u with shadow_decision so the unused holdout can accumulate.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--backfill", action="store_true", help="Capture missing NFL rows from dated model-cache files.")
    parser.add_argument("--write", action="store_true", help="Write staking_approvals.json only if the holdout clears.")
    parser.add_argument("--output", type=Path, help="Optional JSON status path.")
    args = parser.parse_args()
    root = args.repo_root
    freeze = load_freeze(root / "data" / "calibration" / "nfl_staking_freeze.json")
    backfill = None
    if args.backfill:
        backfill = backfill_team_prop_pregame_from_cache(
            root / "data" / "model_cache", repo_root=root, model_keys={"nfl"},
        )
    ledger = load_team_prop_pregame_ledger(root)
    report = evaluate_frozen_holdout(ledger, freeze)
    written = maybe_write_approval(
        freeze, report, policy_path=root / "data" / "calibration" / "staking_approvals.json", write=args.write,
    )
    report["approval_written"] = written
    payload = {
        "freeze": {
            "fitted_version": freeze.get("fitted_version"),
            "frozen_rule": freeze.get("frozen_rule"),
            "frozen_at": freeze.get("frozen_at"),
            "holdout_starts_at": freeze.get("holdout", {}).get("starts_at"),
        },
        "backfill": backfill,
        "diagnosis": diagnose_ledger(ledger, freeze),
        "holdout": report,
        "approval_written": written,
        "approvals_remain_empty": not written,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
