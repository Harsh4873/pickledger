"""Fit one newer probability version per in-house model on graded rows.

The fit is a positive-slope Platt map of the model's own raw probability.
Training uses only settled rows before the holdout date. The holdout is
scored once and never used to choose the slope. A version is shippable only
when its holdout Brier is lower than the current probability and the slope
stays positive, so the fit cannot flip a side.

This does not read prices, write stakes, or change a consensus threshold.
The rejected MLB First Five feature model (holdout Brier 0.322 vs 0.250)
is recorded and is not a ship candidate.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HOLDOUT_START = "2026-09-22"
MIN_TRAIN = 30
MIN_HOLDOUT = 15
BRIER_MARGIN = 0.001
# Recorded earlier. That feature fit lost and must not be published.
FIRST_FIVE_FEATURE_V2 = {
    "name": "mlb_first_five_v2_2026-10-09",
    "holdout_brier_old": 0.25036243887640447,
    "holdout_brier_new": 0.32238619798280943,
    "ship": False,
    "reason": "feature logistic lost the untouched holdout",
}

SERVING_VERSION = {
    "mlb_new": "mlb_new",
    "mlb_inning": "mlb_inning_v2_2026-08-25",
    "mlb_first_five": "mlb_first_five",
    "mlb_team_total": "mlb_team_total",
    "wnba": "wnba_total_v2_2026-07-26",
    "nba": "nba",
    "nba_playoffs": "nba_playoffs",
    "nba_summer": "nba_summer",
    "mls": "mls_dixon_coles_v2.0",
    "nfl": "nfl_v1_epa_elo_market_anchored",
    "cfb": "cfb_v2_market_anchored_totals",
    "nhl": "nhl_poisson_v2_20252026_shadow_ev_v1",
    "tennis": "tennis_v1_welo",
    "fifa_world_cup": "fifa_world_cup",
    "ipl": "ipl",
}


def new_version_name(model_key: str, trained_on: str = "2026-10-09") -> str:
    return f"{model_key}_platt_{trained_on}"


# Frozen from the 2026-10-09 graded ledger. Holdout dates are on or after
# 2026-09-09 (189 rows). Brier 0.2550 -> 0.2511. Do not refit this on later
# rows inside the publisher; that would score the same games it was trained on.
SHIPPED_FITS = {
    "mlb_inning": {
        "model_version": "mlb_inning_platt_2026-10-09",
        "slope": 0.7920966361300178,
        "intercept": -0.03583113927649127,
    },
}


def _clip(probability: float) -> float:
    return min(max(probability, 1e-6), 1.0 - 1e-6)


def _logit(probability: float) -> float:
    clipped = _clip(probability)
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    term = math.exp(value)
    return term / (1.0 + term)


def _probability(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if 1.0 < number <= 100.0:
        number /= 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return _clip(number)


def _outcome(record: Mapping[str, Any]) -> int | None:
    text = str(record.get("result") or "").strip().lower()
    if text == "win":
        return 1
    if text == "loss":
        return 0
    return None


def _revision(record: Mapping[str, Any]) -> int:
    try:
        return int(record.get("revision") or 0)
    except (TypeError, ValueError):
        return 0


def _row_key(record: Mapping[str, Any]) -> str:
    stable = str(record.get("stable_id") or "").strip()
    if stable:
        return stable
    return "|".join(
        str(record.get(field) or "")
        for field in ("game_id", "market", "selection", "pick")
    )


def graded_rows(records: Iterable[Mapping[str, Any]], model_key: str) -> list[dict[str, Any]]:
    """Earliest settled revision of each pick. Later reprints are not new games."""
    chosen: dict[str, dict[str, Any]] = {}
    for record in records:
        if str(record.get("model_key") or "") != model_key:
            continue
        outcome = _outcome(record)
        probability = _probability(record.get("raw_probability"))
        slate = str(record.get("slate_date") or "")[:10]
        if outcome is None or probability is None or len(slate) != 10:
            continue
        key = _row_key(record)
        if not key.strip("|"):
            continue
        revision = _revision(record)
        previous = chosen.get(key)
        if previous is not None and previous["revision"] <= revision:
            continue
        chosen[key] = {
            "date": slate,
            "probability": probability,
            "outcome": outcome,
            "revision": revision,
            "model_version": str(record.get("model_version") or SERVING_VERSION.get(model_key) or model_key),
        }
    return list(chosen.values())


def fit_platt(rows: Sequence[Mapping[str, Any]], *, steps: int = 500, learning_rate: float = 0.08, l2: float = 0.25) -> dict[str, float]:
    """Map logit(probability) with slope a and intercept b. a starts at 1."""
    if not rows:
        raise ValueError("empty training rows")
    slope = 1.0
    intercept = 0.0
    rate = learning_rate
    count = float(len(rows))
    for _step in range(steps):
        grad_slope = l2 * (slope - 1.0)
        grad_intercept = l2 * intercept
        for row in rows:
            logit = _logit(float(row["probability"]))
            prediction = _sigmoid(slope * logit + intercept)
            error = prediction - int(row["outcome"])
            grad_slope += error * logit
            grad_intercept += error
        slope -= rate * grad_slope / count
        intercept -= rate * grad_intercept / count
        rate *= 0.998
    return {"slope": slope, "intercept": intercept}


def apply_platt(probability: float, fit: Mapping[str, float]) -> float:
    return _clip(_sigmoid(float(fit["slope"]) * _logit(probability) + float(fit["intercept"])))


def _brier(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum((float(row[key]) - int(row["outcome"])) ** 2 for row in rows) / len(rows)


def _modal_version(rows: Sequence[Mapping[str, Any]]) -> str:
    versions = Counter(str(row.get("model_version") or "") for row in rows)
    versions.pop("", None)
    if not versions:
        return "unversioned"
    return versions.most_common(1)[0][0]


def chronological_split(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Train on earlier slates. The latest 30% of dates is the holdout."""
    dates = sorted({str(row["date"]) for row in rows})
    if len(dates) < 2:
        return list(rows), [], ""
    cut_index = min(len(dates) - 1, max(1, int(len(dates) * 0.7)))
    cut = dates[cut_index]
    train = [row for row in rows if str(row["date"]) < cut]
    holdout = [row for row in rows if str(row["date"]) >= cut]
    return train, holdout, cut


def score_model(records: Iterable[Mapping[str, Any]], model_key: str) -> dict[str, Any]:
    rows = graded_rows(records, model_key)
    train, holdout, holdout_start = chronological_split(rows)
    ledger_version = _modal_version(holdout or train)
    serving = SERVING_VERSION.get(model_key, model_key)
    old_version = serving if ledger_version in {"", "unversioned", serving} else f"{serving} ({ledger_version})"
    base = {
        "model": model_key,
        "old_version": old_version,
        "serving_version": serving,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "holdout_start": holdout_start,
        "new_version": "not shipped",
        "ship": False,
    }
    if len(train) < 8 or len(holdout) < 1:
        base["reason"] = f"need at least 8 train rows and 1 holdout row, have {len(train)} and {len(holdout)}"
        base["holdout_brier_old"] = None
        base["holdout_brier_new"] = None
        if model_key == "mlb_first_five":
            base["feature_v2"] = FIRST_FIVE_FEATURE_V2
            base["reason"] = FIRST_FIVE_FEATURE_V2["reason"]
        return base
    fit = fit_platt(train)
    scored = []
    for row in holdout:
        scored.append({**row, "refit": apply_platt(float(row["probability"]), fit)})
    old_brier = _brier(scored, "probability")
    new_brier = _brier(scored, "refit")
    positive_slope = fit["slope"] > 0
    better = new_brier + BRIER_MARGIN < old_brier
    enough = len(train) >= MIN_TRAIN and len(holdout) >= MIN_HOLDOUT
    ship = bool(positive_slope and better and enough)
    if ship:
        reason = "holdout Brier improved"
    elif not positive_slope:
        reason = "negative slope would flip sides"
    elif better and not enough:
        reason = f"holdout improved but sample is below {MIN_TRAIN} train and {MIN_HOLDOUT} holdout"
    else:
        reason = "holdout Brier did not improve"
    if model_key == "mlb_first_five":
        reason = FIRST_FIVE_FEATURE_V2["reason"] + "; Platt: " + reason
    return {
        **base,
        "new_version": new_version_name(model_key) if ship else "not shipped",
        "ship": ship,
        "reason": reason,
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "holdout_brier_old": old_brier,
        "holdout_brier_new": new_brier,
        "feature_v2": FIRST_FIVE_FEATURE_V2 if model_key == "mlb_first_five" else None,
    }


def score_models(records: Iterable[Mapping[str, Any]], model_keys: Sequence[str] | None = None) -> list[dict[str, Any]]:
    keys = list(model_keys) if model_keys is not None else sorted(SERVING_VERSION)
    return [score_model(records, key) for key in keys]


def shipped_artifact(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    shipped = {}
    for result in results:
        if not result.get("ship"):
            continue
        shipped[str(result["model"])] = {
            "model_version": result["new_version"],
            "replaces": result["old_version"],
            "slope": result["slope"],
            "intercept": result["intercept"],
            "holdout_start": result["holdout_start"],
            "holdout_brier_old": result["holdout_brier_old"],
            "holdout_brier_new": result["holdout_brier_new"],
            "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
    return {
        "schema": "inhouse_platt_holdout_v1",
        "holdout_start": HOLDOUT_START,
        "note": "Applied to the model probability only. Stake thresholds and the consensus gate are unchanged.",
        "models": shipped,
    }


def format_table(results: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| model | old version | new version | holdout Brier old vs new | shipped |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        old_brier = result.get("holdout_brier_old")
        new_brier = result.get("holdout_brier_new")
        if old_brier is None or new_brier is None:
            metric = result.get("reason") or "not scored"
        else:
            metric = f"{old_brier:.4f} vs {new_brier:.4f}"
            if not result.get("ship") and result.get("reason"):
                metric += f" ({result['reason']})"
        if result.get("feature_v2"):
            feature = result["feature_v2"]
            metric += (
                f"; feature v2 {feature['holdout_brier_old']:.3f} vs "
                f"{feature['holdout_brier_new']:.3f} not shipped"
            )
        lines.append(
            f"| {result['model']} | {result['old_version']} | {result['new_version']} | {metric} | "
            f"{'yes' if result.get('ship') else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    from scripts.team_prop_pregame_ledger import TEAM_PROP_MODEL_KEYS, load_team_prop_pregame_ledger

    root = Path(__file__).resolve().parents[1]
    ledger = load_team_prop_pregame_ledger(root)
    results = score_models(ledger.get("records") or [], sorted(TEAM_PROP_MODEL_KEYS))
    rendered = format_table(results)
    print(rendered)
    print(json.dumps(shipped_artifact(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
