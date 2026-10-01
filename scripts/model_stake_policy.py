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
from scripts.settlement_support import binary_settlement_supported


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


def _policy_version(pick: Mapping[str, Any], bucket: Mapping[str, Any]) -> str:
    fitted = str(pick.get("model_version") or bucket.get("model_version") or "").strip()
    if fitted.startswith("nfl_v") or fitted.startswith("nhl_poisson_v") or fitted.startswith("mls_dixon_coles_v"):
        return fitted
    return str(
        pick.get("ml_model_version") or pick.get("prediction_model_version")
        or pick.get("model_version") or bucket.get("prediction_model_version")
        or bucket.get("model_version") or "unversioned"
    )


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
            frozen_candidate = model_key in {"nhl", "mls"}
            if frozen_candidate:
                from scripts.frozen_staking_candidate import candidate_matches, load_freeze
                freeze = load_freeze(model_key)
                version = _policy_version(pick, bucket)
                market = str(pick.get("market") or pick.get("market_type") or "").lower()
                approved_candidate = candidate_matches(pick, model_key) and any(
                    isinstance(a, Mapping) and _approved(a)
                    and (a.get("model_key"), a.get("model_version"), a.get("market"), a.get("variant", "base"))
                    == (model_key, version, market, "base")
                    and a.get("frozen_rule") == freeze["frozen_rule"]
                    for a in entries
                )
                if approved_candidate and model_key == "mls":
                    approved_candidate = binary_settlement_supported({**pick, "model_key": model_key, "market": market})
                if approved_candidate:
                    from scripts.merge_model_cache_payload import _still_assumed_price
                    from scripts.price_clock import observed_quote_timing
                    from datetime import datetime, timezone
                    timing = pick.get("certification_timing") or {}
                    price_clock = observed_quote_timing(
                        pick, published_at=payload.get("publishedAt") or timing.get("published_at") or datetime.now(timezone.utc).isoformat(),
                        start_at=pick.get("game_start_time") or pick.get("start_time"),
                    )
                    approved_candidate = not _still_assumed_price(pick) and price_clock is None
                pick["calibration_excluded"] = not approved_candidate
                # Restore only actions deliberately shadowed by this gate.
                # Unpriced/model PASS rows never become actions here.
                if decision == "PASS" and pick.get("staking_policy") == "awaiting_approved_holdout":
                    decision = str(pick.get("shadow_decision") or "PASS")
                    if approved_candidate and decision in {"BET", "LEAN"}:
                        pick["decision"] = decision
                        pick["units"] = pick.get("shadow_units", 0)
                if approved_candidate:
                    pick["staking_policy"] = "approved_holdout"
                    if pick.get("decision") in {"BET", "LEAN"}:
                        pick.setdefault("source_decision_reason", pick.get("decision_reason"))
                        pick["decision_reason"] = "approved_holdout"
                        pick["actionability"] = "approved"
                    continue
            if decision not in {"BET", "LEAN"}:
                continue
            version = _policy_version(pick, bucket)
            if prop:
                fingerprint = str(pick.get("ml_training_fingerprint") or pick.get("training_fingerprint") or "")
                if fingerprint:
                    version += ":" + fingerprint[:12]
            market = str(
                (pick.get("stat_key") if prop else None)
                or pick.get("market") or pick.get("market_type") or "unknown"
            ).strip().lower()
            variant = str(pick.get("model_variant") or "base") if prop else "base"
            if not frozen_candidate and (model_key, version, market, variant) in allowed:
                continue
            pick.setdefault("source_decision", decision)
            pick.setdefault("source_units", pick.get("units"))
            pick["shadow_decision"] = decision
            pick["shadow_units"] = pick.get("shadow_units") if pick.get("decision") == "PASS" else pick.get("units")
            pick["decision"] = "PASS"
            pick["units"] = 0
            pick["staking_policy"] = "awaiting_approved_holdout"
            if frozen_candidate:
                pick.setdefault("source_decision_reason", pick.get("decision_reason"))
                pick["decision_reason"] = "awaiting_approved_holdout"
                pick["actionability"] = "research"
            demoted += 1
    return demoted
