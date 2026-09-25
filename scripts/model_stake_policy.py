"""Publication gate for owned model stakes.

Approvals are exact fitted-version, market, and variant matches. An approval
must name a frozen rule and a reviewed, unused chronological holdout; this
module deliberately cannot infer approval from a favorable scorecard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from scripts.price_clock import aware_time


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "data" / "calibration" / "staking_approvals.json"
MIN_PRICED_SETTLED = 100


def load_approvals(path: Path = POLICY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("approvals"), list):
        raise ValueError("invalid staking approval policy")
    return payload


def _approved(approval: Mapping[str, Any]) -> bool:
    holdout = approval.get("holdout")
    return (
        approval.get("approved") is True
        and isinstance(approval.get("frozen_rule"), str)
        and bool(approval["frozen_rule"].strip())
        and isinstance(holdout, Mapping)
        and holdout.get("unused_during_selection") is True
        and int(holdout.get("independently_priced_settled") or 0) >= MIN_PRICED_SETTLED
        and float(holdout.get("roi") or 0) > 0
        and float(holdout.get("clustered_lower_95") or 0) > 0
        and holdout.get("calibration_no_material_regression") is True
    )


def _start_time(pick: Mapping[str, Any], bucket: Mapping[str, Any]):
    for field in ("game_start_time", "start_time", "event_start_time", "scheduled_start_time"):
        found = aware_time(pick.get(field))
        if found is not None:
            return found
    game_id = str(pick.get("game_id") or pick.get("gamePk") or pick.get("event_id") or "")
    matchup = str(pick.get("matchup") or pick.get("game") or "").lower()
    for game in bucket.get("games") or []:
        if not isinstance(game, Mapping):
            continue
        if not ((game_id and game_id == str(game.get("game_id") or game.get("gamePk") or game.get("event_id") or ""))
                or (matchup and matchup == str(game.get("matchup") or game.get("game") or "").lower())):
            continue
        for field in ("game_start_time", "start_time", "event_start_time", "scheduled_start_time"):
            found = aware_time(game.get(field))
            if found is not None:
                return found
    return None


def apply_stake_policy(
    payload: dict[str, Any], *, approvals: Mapping[str, Any] | None = None,
    model_keys: set[str], prop: bool = False,
) -> int:
    """Demote unapproved model wagers to visible, zero-stake shadow picks."""

    policy = approvals if approvals is not None else load_approvals()
    entries = policy.get("approvals")
    if not isinstance(entries, list):
        raise ValueError("staking approvals must be a list")
    allowed = {
        (str(a.get("model_key")), str(a.get("model_version")),
         str(a.get("market")), str(a.get("variant", "base")))
        for a in entries if isinstance(a, Mapping) and _approved(a)
    }
    models = payload.get("models")
    if not isinstance(models, dict):
        return 0
    publication = aware_time(payload.get("publishedAt"))
    demoted = 0
    for model_key, bucket in models.items():
        if model_key not in model_keys or not isinstance(bucket, dict):
            continue
        for pick in bucket.get("picks") or []:
            if not isinstance(pick, dict):
                continue
            decision = str(pick.get("decision") or "").upper()
            if decision not in {"BET", "LEAN"}:
                continue
            # Team cache merges retain earlier pregame publications after
            # kickoff. Their original wagers are history, not new decisions.
            # A newly generated post-start row has no earlier trusted clock
            # and still goes through the gate.
            if not prop and publication is not None:
                start = _start_time(pick, bucket)
                timing = pick.get("certification_timing")
                prior = aware_time(timing.get("published_at")) if isinstance(timing, Mapping) else None
                if start is not None and start <= publication and prior is not None and prior < start:
                    continue
            version = str(
                pick.get("ml_model_version") or pick.get("prediction_model_version")
                or pick.get("model_version") or bucket.get("prediction_model_version")
                or bucket.get("model_version") or "unversioned"
            )
            if prop:
                fingerprint = str(pick.get("ml_training_fingerprint") or pick.get("training_fingerprint") or "")
                if fingerprint:
                    version += ":" + fingerprint[:12]
            market = str(
                (pick.get("stat_key") if prop else None)
                or pick.get("market") or pick.get("market_type") or "unknown"
            ).strip().lower()
            variant = str(pick.get("model_variant") or "base") if prop else "base"
            if (model_key, version, market, variant) in allowed:
                continue
            pick.setdefault("source_decision", decision)
            pick.setdefault("source_units", pick.get("units"))
            pick["shadow_decision"] = decision
            pick["shadow_units"] = pick.get("units")
            pick["decision"] = "PASS"
            pick["units"] = 0
            pick["staking_policy"] = "awaiting_approved_holdout"
            demoted += 1
    return demoted
