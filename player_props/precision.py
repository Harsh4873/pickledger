"""Season-trained, abstaining player-prop precision model.

The model predicts the probability of the over from immutable historical
DraftKings markets and prior player outcomes.  Publication is intentionally
stricter than prediction: only the market family and price band that cleared
70% on both chronological validation and a later holdout may be released.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .schema import (
    kelly,
    safe_float,
    stable_id,
)


PRECISION_MODEL_VERSION = "player_props_precision_v1.0.0"
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "mlb_player_props_precision.joblib"
METADATA_PATH = ARTIFACT_DIR / "mlb_player_props_precision_metadata.json"

NUMERIC_FEATURES = [
    "line",
    "over_implied",
    "under_implied",
    "no_vig_over",
    "history_count",
    "mean3",
    "mean5",
    "mean10",
    "mean20",
    "season_mean",
    "std5",
    "std10",
    "season_std",
    "last_actual",
    "over_rate",
    "over_rate5",
    "over_rate10",
    "delta3",
    "delta5",
    "delta10",
    "delta_season",
    "z5",
    "z10",
]
CATEGORICAL_FEATURES = ["stat_key", "market_format"]

DEFAULT_POLICY = {
    "sport": "MLB",
    "stat_key": "batter_walks",
    "selection": "Under",
    "minimum_history": 5,
    "minimum_under_history_rate": 0.60,
    "minimum_under_last5_rate": 0.70,
    "minimum_under_odds": -350,
    "maximum_under_odds": -251,
    "require_market_favorite": True,
    "require_mean10_below_line": True,
    "require_model_under": True,
    "minimum_model_edge": 0.0,
    "maximum_picks_per_game": 1,
    "stake_units": 0.25,
}

_BUNDLE: dict[str, Any] | None | bool = False


def _mean(values: list[float], count: int | None = None) -> float:
    selected = values[-count:] if count else values
    return statistics.fmean(selected) if selected else 0.0


def _std(values: list[float], count: int | None = None) -> float:
    selected = values[-count:] if count else values
    return statistics.pstdev(selected) if len(selected) > 1 else 0.0


def history_features(
    values: list[float],
    *,
    line: float,
    over_implied: float,
    under_implied: float | None,
    stat_key: str,
    market_format: str,
) -> dict[str, Any] | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 3:
        return None
    mean3 = _mean(clean, 3)
    mean5 = _mean(clean, 5)
    mean10 = _mean(clean, 10)
    mean20 = _mean(clean, 20)
    season_mean = _mean(clean)
    std5 = _std(clean, 5)
    std10 = _std(clean, 10)
    season_std = _std(clean)
    over_rate = sum(value > line for value in clean) / len(clean)
    last5 = clean[-5:]
    last10 = clean[-10:]
    over_rate5 = sum(value > line for value in last5) / len(last5)
    over_rate10 = sum(value > line for value in last10) / len(last10)
    under_value = under_implied if under_implied is not None else 0.5
    no_vig_over = (
        over_implied / (over_implied + under_value)
        if under_implied is not None and over_implied + under_value > 0
        else over_implied
    )
    return {
        "line": line,
        "over_implied": over_implied,
        "under_implied": under_implied,
        "no_vig_over": no_vig_over,
        "history_count": len(clean),
        "mean3": mean3,
        "mean5": mean5,
        "mean10": mean10,
        "mean20": mean20,
        "season_mean": season_mean,
        "std5": std5,
        "std10": std10,
        "season_std": season_std,
        "last_actual": clean[-1],
        "over_rate": over_rate,
        "over_rate5": over_rate5,
        "over_rate10": over_rate10,
        "delta3": mean3 - line,
        "delta5": mean5 - line,
        "delta10": mean10 - line,
        "delta_season": season_mean - line,
        "z5": (mean5 - line) / (std5 + 0.5),
        "z10": (mean10 - line) / (std10 + 0.5),
        "stat_key": stat_key,
        "market_format": market_format,
    }


def build_training_features(
    market_rows: list[dict[str, Any]],
    prior_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, list[float]]]:
    """Build chronological features without allowing same-game outcomes to leak."""
    events: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in market_rows:
        events[(str(row.get("sport") or ""), str(row.get("event_id") or ""))].append(row)
    history: dict[str, list[float]] = defaultdict(list)
    for row in sorted(
        prior_rows or [],
        key=lambda item: (
            str(item.get("date") or ""),
            str(item.get("event_id") or ""),
            str(item.get("athlete_id") or ""),
            str(item.get("stat_key") or ""),
        ),
    ):
        actual = safe_float(row.get("actual"), float("nan"))
        if math.isfinite(actual):
            history[_profile_key(row.get("athlete_id"), row.get("stat_key"))].append(actual)
    features: list[dict[str, Any]] = []
    ordered_events = sorted(
        events.items(),
        key=lambda item: (
            str(item[1][0].get("start_time") or "") if item[1] else "",
            item[0],
        ),
    )
    for _, rows in ordered_events:
        seen: set[str] = set()
        for row in rows:
            key = _profile_key(row.get("athlete_id"), row.get("stat_key"))
            values = history[key]
            feature = history_features(
                values,
                line=safe_float(row.get("line")),
                over_implied=safe_float(row.get("over_implied"), 0.5),
                under_implied=(
                    safe_float(row.get("under_implied"))
                    if row.get("under_implied") is not None
                    else None
                ),
                stat_key=str(row.get("stat_key") or ""),
                market_format=str(row.get("market_format") or "total"),
            )
            if feature:
                features.append({**row, **feature})
        for row in rows:
            key = _profile_key(row.get("athlete_id"), row.get("stat_key"))
            if key in seen:
                continue
            actual = safe_float(row.get("actual"), float("nan"))
            if math.isfinite(actual):
                history[key].append(actual)
                seen.add(key)
    return features, dict(history)


def _profile_key(athlete_id: Any, stat_key: Any) -> str:
    return f"{str(athlete_id or '').strip()}|{str(stat_key or '').strip()}"


# The v2 four-model consensus supersedes the original single-market precision
# artifact. These late-bound wrappers preserve the public API used by the
# generators while allowing the old feature builder to remain reusable.
def load_precision_bundle() -> dict[str, Any] | None:  # type: ignore[no-redef]
    from .consensus import load_consensus_bundle

    return load_consensus_bundle()


def precision_model_active(sport: str | None = None) -> bool:  # type: ignore[no-redef]
    from .consensus import consensus_active

    return consensus_active(sport)


def precision_model_required() -> bool:  # type: ignore[no-redef]
    return load_precision_bundle() is not None


def evaluate_precision_pick(pick: dict[str, Any]) -> dict[str, Any]:  # type: ignore[no-redef]
    from .consensus import evaluate_consensus_pick

    return evaluate_consensus_pick(pick)


def apply_precision_to_pick(pick: dict[str, Any]) -> dict[str, Any]:  # type: ignore[no-redef]
    result = evaluate_precision_pick(pick)
    if not result.get("required"):
        return pick
    pick["precision_required"] = True
    pick["precision_evaluated"] = True
    pick["precision_qualified"] = bool(result.get("qualified"))
    pick["precision_reason"] = result.get("reason")
    if not result.get("qualified"):
        pick["decision"] = "PASS"
        pick["units"] = 0.0
        pick["full_kelly"] = 0.0
        pick["quarter_kelly"] = 0.0
        pick["actionability"] = "research_signal"
        return pick

    probability = safe_float(result.get("probability"))
    implied = safe_float(result.get("implied_probability"))
    odds = int(result["odds"])
    edge = probability - implied
    full_kelly, quarter_kelly = kelly(probability, odds)
    profit_multiple = 100.0 / abs(odds) if odds < 0 else odds / 100.0
    expected_value = probability * profit_multiple - (1.0 - probability)
    stat_label = str(pick.get("stat_label") or pick.get("market_type") or pick.get("stat_key") or "")
    selection = str(result["selection"])
    line = safe_float(pick.get("line"))
    validation_accuracy = safe_float(result.get("validation_accuracy"))
    holdout_accuracy = safe_float(result.get("holdout_accuracy"))
    conservative_accuracy = safe_float(result.get("conservative_validation_accuracy"))
    sport = str(pick.get("sport") or "").upper()
    bundle = load_precision_bundle() or {}
    metadata = bundle.get("metadata") if isinstance(bundle.get("metadata"), dict) else {}
    model_map = metadata.get("models") if isinstance(metadata.get("models"), dict) else {}
    consensus_models = [f"{name}: {description}" for name, description in sorted(model_map.items())]
    sport_model_prefix = f"{sport.lower()}_"
    consensus_applicable_models = [
        f"{name}: {description}"
        for name, description in sorted(model_map.items())
        if name.startswith(sport_model_prefix)
    ]
    consensus_model_names = ", ".join(sorted(model_map)) or "season/history consensus"
    pick.update(
        {
            "id": stable_id(
                str(pick.get("sport") or ""),
                str(pick.get("date") or ""),
                str(pick.get("game_id") or ""),
                str(pick.get("player_id") or ""),
                str(pick.get("stat_key") or ""),
                selection,
                line,
            ),
            "selection": selection,
            "pick": f"{pick.get('player_name')} {selection} {line:.1f} {stat_label}",
            "odds": odds,
            "market_implied_probability": round(implied, 4),
            "probability": round(probability, 4),
            "ml_probability": round(probability, 4),
            "ml_raw_probability": round(probability, 4),
            "ml_edge": round(edge, 6),
            "ml_expected_value": round(expected_value, 6),
            "ml_model_active": True,
            "ml_model_version": result["model_version"],
            "ml_probability_mode": "four_model_consensus_gate",
            "ml_training_fingerprint": result["training_fingerprint"],
            "decision": "LEAN",
            "confidence": "High",
            "edge": round(edge * 100.0, 2),
            "full_kelly": full_kelly,
            "quarter_kelly": quarter_kelly,
            "units": 0.25,
            "actionability": "precision_qualified",
            "precision_probability": round(probability, 4),
            "precision_validation_accuracy": validation_accuracy,
            "precision_holdout_accuracy": holdout_accuracy,
            "precision_conservative_validation_accuracy": conservative_accuracy,
            "consensus_season_probability": result.get("season_probability"),
            "consensus_history_probability": result.get("history_probability"),
            "consensus_season_projection": result.get("season_projection"),
            "consensus_history_projection": result.get("history_projection"),
            "consensus_model_agreement": result.get("agreement"),
            "consensus_score": safe_float(result.get("consensus_score")),
            "consensus_model_count": len(consensus_models),
            "consensus_models": consensus_models,
            "consensus_applicable_models": consensus_applicable_models or consensus_models,
            "consensus_record_models": consensus_applicable_models or consensus_models,
            "reason": (
                f"The active four-model consensus suite qualifies this market through the "
                f"{sport} season and roster-aware history voters; "
                f"chronological validation/holdout accuracy is "
                f"{validation_accuracy:.1%}/{holdout_accuracy:.1%}."
            ),
        }
    )
    history_window = "2022-26" if sport == "MLB" else "2024-26"
    pick.setdefault("key_factors", []).extend(
        [
            f"Four-model consensus suite active: {consensus_model_names}",
            "2026 season model evaluated",
            f"Roster-aware {history_window} history model evaluated",
            f"Consensus pick-level probability {probability:.1%}",
            f"Conservative validation floor {conservative_accuracy:.1%}",
        ]
    )
    fingerprint = str(result["training_fingerprint"] or "unfingerprinted")[:16]
    rank_epoch = f"{str(pick.get('sport') or '').upper()}:{result['model_version']}:{fingerprint}"
    pick["ml_rank_epoch"] = rank_epoch
    pick["ranking_epoch"] = rank_epoch
    pick["model_epoch"] = rank_epoch
    return pick
