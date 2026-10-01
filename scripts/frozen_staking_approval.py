#!/usr/bin/env python3
"""Evaluate a frozen NHL/MLS rule on certified prospective ledger evidence.

--write is explicit evidence review, never part of the scheduled status job.
NFL's evaluator/statistics are reused without changing its behavior.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.frozen_staking_candidate import load_freeze, candidate_fingerprint
from scripts.nfl_staking_approval import (
    evaluate_frozen_holdout as evaluate_market, maybe_write_approval,
    _candidate_action, _timestamp,
)
from scripts.team_prop_model_evaluator import _financial_eligible, _price_provenance_is_disallowed, certification_status
from scripts.price_clock import observed_quote_timing, aware_time
from scripts.team_prop_pregame_ledger import load_team_prop_pregame_ledger, backfill_team_prop_pregame_from_cache
from scripts.model_stake_policy import POLICY_PATH


def evaluate(ledger: dict, freeze: dict) -> dict:
    start = _timestamp(freeze["holdout"]["starts_at"])
    frozen = _timestamp(freeze["frozen_at"])
    if aware_time(freeze["holdout"]["starts_at"]) is None or aware_time(freeze["frozen_at"]) is None or start <= frozen:
        raise ValueError("holdout must start strictly after freeze")
    records = []
    excluded = {market: {} for market in freeze["markets"]}
    seen = set()
    # First eligible publication per event/market, independent of later outcomes.
    # Sort by instant, not ISO text: offsets need not sort chronologically.
    for raw in sorted(
        (row for row in ledger.get("records", []) if isinstance(row, dict)),
        key=lambda row: aware_time(row.get("published_at")) or start,
    ):
        if raw.get("model_key") != freeze["model_key"]:
            continue
        market = str(raw.get("market") or "").lower()
        if market not in excluded:
            continue
        snapshot = raw.get("pregame_snapshot") or {}
        reason = None
        published = aware_time(raw.get("published_at"))
        if published is None or published < start:
            reason = "before_holdout_start"
        elif raw.get("model_version") != freeze["fitted_version"]:
            reason = "fitted_version_mismatch"
        elif not certification_status(raw, ledger)[0]:
            reason = "uncertified"
        elif observed_quote_timing(snapshot, published_at=raw.get("published_at"), start_at=raw.get("game_start_time")):
            reason = "invalid_quote_clock"
        if reason is None:
            if snapshot.get("staking_candidate_fingerprint") != freeze["candidate_fingerprint"]:
                reason = "candidate_fingerprint_mismatch"
            elif not _financial_eligible(raw) or _price_provenance_is_disallowed(raw):
                reason = "not_independently_priced"
            elif _candidate_action(raw)[0] not in {"BET", "LEAN"}:
                reason = "not_candidate_action"
        event = raw.get("game_id") or snapshot.get("game_id")
        if not event:
            reason = "missing_event_identity"
        key = (event, raw.get("market"))
        if reason is None and key in seen:
            reason = "duplicate_event_market"
        if reason:
            counts = excluded[market]
            counts[reason] = counts.get(reason, 0) + 1
            continue
        seen.add(key)
        row = copy.deepcopy(raw)
        # MLS grid probabilities are conditional on no push already. NHL
        # probabilities are unconditional; binary scores omit pushes.
        if freeze["model_key"] == "nhl":
            push = float(snapshot.get("push_probability") or 0)
            p = row.get("raw_probability")
            if p is not None and 0 <= push < 1:
                row["raw_probability"] = float(p) / (1 - push)
        records.append(row)
    reports = []
    for market in freeze["markets"]:
        market_freeze = {**freeze, "market": market}
        report = evaluate_market({**ledger, "records": records}, market_freeze)
        report["candidate_exclusions"] = dict(sorted(excluded[market].items()))
        report["status"] = (
            "gate_clear_review_required" if report["clears_gate"]
            else "awaiting_holdout_evidence" if report["independently_priced_settled"] == 0
            else "accruing_holdout_evidence"
        )
        reports.append(report)
    return {"model_key": freeze["model_key"], "markets": reports}


def write_reviewed(freeze: dict, report: dict, policy_path: Path = POLICY_PATH) -> bool:
    if candidate_fingerprint(freeze) != freeze["candidate_fingerprint"]:
        raise ValueError("candidate changed: freeze a new version and unused window")
    changed = False
    for result in report["markets"]:
        # Bind the existing gate's frozen-rule identity to the exact content.
        market_freeze = {**freeze, "market": result["market"]}
        result["approval_written"] = maybe_write_approval(market_freeze, result, policy_path=policy_path, write=True)
        changed |= result["approval_written"]
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["nhl", "mls"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--backfill", action="store_true", help="Capture real missing cache publications without inventing timestamps")
    args = parser.parse_args()
    freeze = load_freeze(args.model)
    if args.backfill:
        backfill_team_prop_pregame_from_cache(repo_root=ROOT, model_keys={args.model})
    report = evaluate(load_team_prop_pregame_ledger(repo_root=ROOT), freeze)
    report["candidate_unchanged"] = candidate_fingerprint(freeze) == freeze["candidate_fingerprint"]
    if not report["candidate_unchanged"]:
        for market in report["markets"]:
            market.update(clears_gate=False, blocker="candidate_changed_refreeze_required",
                          status="candidate_changed_refreeze_required")
    report["approval_written"] = write_reviewed(freeze, report) if args.write else False
    for market in report["markets"]:
        if market.get("approval_written"):
            market["status"] = "approval_written"
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
