#!/usr/bin/env python3
"""Train four roster-aware prop models and validate their publication consensus."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from player_props.consensus import (  # noqa: E402
    ARTIFACT_DIR,
    CONSENSUS_METADATA_PATH,
    CONSENSUS_VERSION,
    MODEL_PATHS,
    OUTCOME_FEATURES,
    OUTCOME_MARKET_FEATURES,
    TARGET_STATS,
    build_outcome_training_features,
    outcome_features,
    outcome_profile_key,
)
from player_props.precision import NUMERIC_FEATURES, build_training_features  # noqa: E402
from player_props.schema import safe_float  # noqa: E402


DEFAULT_MARKETS = REPO_ROOT / "data" / "player_props_training" / "market_history_2026.jsonl"
DEFAULT_OUTCOMES = REPO_ROOT / "data" / "player_props_training" / "outcome_history_2022_2026.jsonl.gz"
TARGET_ACCURACY = 0.70
COUNT_GATE_FEATURES = [
    "season_margin",
    "history_margin",
    "season_mean_margin",
    "usage_trend",
    "over_implied",
    "line",
    "season_count",
    "rest_days",
    "season_std",
    "last_actual",
    "last_usage",
]

POLICIES: dict[str, dict[str, dict[str, Any]]] = {
    "MLB": {
        "hits_runs_rbis": {
            "line": 1.5,
            "minimum_season_probability": 0.625,
            "minimum_history_probability": 0.70,
            "minimum_season_rate": 0.50,
            "minimum_history_rate": 0.0,
            "minimum_implied": 0.55,
            "require_classifier_agreement": True,
            "minimum_validation_samples": 8,
            "minimum_holdout_samples": 5,
        },
        "hits": {
            "minimum_season_probability": 0.50,
            "minimum_history_probability": 0.50,
            "minimum_season_rate": 0.50,
            "minimum_history_rate": 0.0,
            "minimum_implied": 0.62,
            "require_classifier_agreement": True,
            "minimum_validation_samples": 15,
            "minimum_holdout_samples": 8,
        },
        "strikeouts": {
            "minimum_season_probability": 0.575,
            "minimum_history_probability": 0.625,
            "minimum_season_rate": 0.60,
            "minimum_history_rate": 0.0,
            "minimum_implied": 0.58,
            "require_classifier_agreement": True,
            "minimum_validation_samples": 20,
            "minimum_holdout_samples": 10,
        },
        "pitcher_walks_allowed": {
            "minimum_season_probability": 0.60,
            "minimum_history_probability": 0.65,
            "minimum_season_rate": 0.55,
            "minimum_history_rate": 0.0,
            "minimum_implied": 0.60,
            "require_classifier_agreement": True,
            "minimum_validation_samples": 10,
            "minimum_holdout_samples": 5,
        },
    },
    "WNBA": {
        "points": {
            "selection": "Over",
            "minimum_season_probability": 0.55,
            "minimum_history_probability": 0.50,
            "minimum_season_rate": 0.55,
            "minimum_history_rate": 0.0,
            "minimum_implied": 0.55,
            "require_classifier_agreement": True,
            "minimum_validation_samples": 4,
            "minimum_holdout_samples": 2,
        },
        "totalRebounds": {
            "selection": "Over",
            "minimum_history_margin": -2.0,
            "maximum_history_margin": -1.0,
            "minimum_season_mean_margin": -2.0,
            "minimum_usage_trend": 2.0,
            "minimum_implied": 0.0,
            "minimum_validation_samples": 5,
            "minimum_holdout_samples": 3,
            "history_only_publication": True,
        },
        "assists": {
            "selection": "Over",
            "minimum_implied": 0.0,
            "meta_gate_threshold": 0.60,
            "minimum_validation_samples": 5,
            "minimum_holdout_samples": 3,
        },
        "three_pointers_made": {
            "selection": "Over",
            "minimum_implied": 0.0,
            "meta_gate_threshold": 0.60,
            "minimum_validation_samples": 5,
            "minimum_holdout_samples": 3,
        },
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _classifier(*, min_child_weight: int = 15) -> Any:
    from xgboost import XGBClassifier  # type: ignore

    return XGBClassifier(
        n_estimators=220 if min_child_weight < 20 else 250,
        max_depth=2,
        learning_rate=0.035 if min_child_weight < 20 else 0.03,
        min_child_weight=min_child_weight,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=8 if min_child_weight < 20 else 10,
        reg_alpha=0.4 if min_child_weight < 20 else 0.5,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )


def _history_classifier(*, min_child_weight: int = 10) -> Any:
    from xgboost import XGBClassifier  # type: ignore

    return XGBClassifier(
        n_estimators=260,
        max_depth=2,
        learning_rate=0.028,
        min_child_weight=min_child_weight,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=10,
        reg_alpha=0.5,
        n_jobs=-1,
        random_state=42,
        eval_metric="logloss",
    )


def _regressor() -> Any:
    from xgboost import XGBRegressor  # type: ignore

    return XGBRegressor(
        n_estimators=350,
        max_depth=2,
        learning_rate=0.025,
        min_child_weight=10,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=10,
        reg_alpha=0.5,
        n_jobs=-1,
        random_state=42,
        objective="reg:squarederror",
    )


def _fit_market(frame: Any, stat_key: str, cutoff: str | None = None, *, min_child_weight: int = 15) -> Any:
    rows = frame[frame["stat_key"].eq(stat_key)]
    if cutoff:
        rows = rows[rows["date"].le(cutoff)]
    model = _classifier(min_child_weight=min_child_weight)
    model.fit(rows[NUMERIC_FEATURES], rows["over_outcome"].astype(int))
    return model


def _fit_paired_history(frame: Any, stat_key: str, cutoff: str | None = None) -> Any:
    rows = frame[frame["stat_key"].eq(stat_key)]
    if cutoff:
        rows = rows[rows["date"].le(cutoff)]
    columns = [
        name if name in {"line", "over_implied", "under_implied"} else f"history_{name}"
        for name in NUMERIC_FEATURES
    ]
    training = rows[columns].copy()
    training.columns = NUMERIC_FEATURES
    model = _classifier()
    model.fit(training, rows["over_outcome"].astype(int))
    return model


def _fit_hrr_history(outcomes: Any, cutoff: str | None = None) -> Any:
    rows = outcomes[outcomes["sport"].eq("MLB") & outcomes["stat_key"].eq("hits_runs_rbis")]
    if cutoff:
        rows = rows[rows["date"].le(cutoff)]
    model = _classifier(min_child_weight=20)
    model.fit(rows[OUTCOME_FEATURES], (rows["actual"] > 1.5).astype(int))
    return model


def _fit_mlb_outcome_market_history(frame: Any, stat_key: str, cutoff: str | None = None) -> Any:
    rows = frame[frame["sport"].eq("MLB") & frame["stat_key"].eq(stat_key)]
    if cutoff:
        rows = rows[rows["date"].le(cutoff)]
    min_child_weight = 20 if stat_key == "hits" else 8 if stat_key in {"strikeouts", "pitcher_walks_allowed"} else 10
    model = _history_classifier(min_child_weight=min_child_weight)
    model.fit(rows[OUTCOME_MARKET_FEATURES], rows["over_outcome"].astype(int))
    return model


def _fit_wnba_count(outcomes: Any, stat_key: str, cutoff: str | None, *, season_only: bool) -> Any:
    rows = outcomes[outcomes["sport"].eq("WNBA") & outcomes["stat_key"].eq(stat_key)]
    if season_only:
        rows = rows[rows["season"].eq(2026)]
        columns = [name for name in OUTCOME_FEATURES if not name.startswith("all_")]
    else:
        columns = OUTCOME_FEATURES
    if cutoff:
        rows = rows[rows["date"].le(cutoff)]
    model = _regressor()
    model.fit(rows[columns], rows["actual"])
    return model


def _market_probability(model: Any, frame: Any) -> Any:
    return model.predict_proba(frame[NUMERIC_FEATURES])[:, 1]


def _outcome_frame_for_markets(rows: Any, profiles: dict[str, list[dict[str, Any]]]) -> Any:
    import pandas as pd  # type: ignore

    output: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        profile = profiles.get(outcome_profile_key(row.get("sport"), row.get("athlete_id"), row.get("stat_key")))
        built = outcome_features(profile or [], target_date=str(row.get("date") or ""))
        if built:
            output.append({**row, **built})
    return pd.DataFrame(output)


def _synthetic_count_line(value: Any) -> float:
    number = safe_float(value, 0.0)
    if not math.isfinite(number):
        number = 0.0
    return max(0.5, math.floor(max(0.0, number)) + 0.5)


def _synthetic_count_rows(outcome_frame: Any, sport: str, stat_key: str) -> Any:
    rows = outcome_frame[
        outcome_frame["sport"].eq(sport)
        & outcome_frame["stat_key"].eq(stat_key)
    ].copy()
    if rows.empty:
        return rows
    rows["line"] = rows["season_mean"].map(_synthetic_count_line)
    rows["over_odds"] = -110
    rows["under_odds"] = -110
    rows["over_implied"] = 0.5238
    rows["under_implied"] = 0.5238
    rows["market_format"] = "synthetic_total"
    rows["over_outcome"] = (rows["actual"] > rows["line"]).astype(int)
    rows["market_priced"] = False
    rows["pricing_type"] = "synthetic"
    rows["line_source"] = "in_house_3pm_model" if stat_key == "three_pointers_made" else "in_house_count_model"
    return rows


def _count_market_rows(
    outcome_market: Any,
    outcome_frame: Any,
    sport: str,
    stat_key: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> Any:
    rows = outcome_market[
        outcome_market["sport"].eq(sport)
        & outcome_market["stat_key"].eq(stat_key)
    ]
    if rows.empty and sport == "WNBA" and stat_key == "three_pointers_made":
        rows = _synthetic_count_rows(outcome_frame, sport, stat_key)
    if start and end:
        rows = rows[rows["date"].between(start, end)]
    return rows


def _one_per_event(rows: Any, *, score: str) -> Any:
    if rows.empty:
        return rows
    return rows.sort_values(["event_id", score], ascending=[True, False]).drop_duplicates("event_id")


def _metrics(rows: Any) -> dict[str, Any]:
    samples = len(rows)
    wins = int(rows["selected_outcome"].sum()) if samples else 0
    return {
        "samples": samples,
        "wins": wins,
        "losses": samples - wins,
        "accuracy": wins / samples if samples else None,
    }


def _evaluate_classifier_policy(
    rows: Any,
    *,
    season_model: Any,
    history_model: Any,
    policy: dict[str, Any],
    hrr_history: bool = False,
    outcome_market_history: bool = False,
) -> dict[str, Any]:
    import numpy as np  # type: ignore

    frame = rows.copy()
    frame["season_over"] = _market_probability(season_model, frame)
    if hrr_history:
        frame["history_over"] = history_model.predict_proba(frame[OUTCOME_FEATURES])[:, 1]
        frame["history_rate_value"] = 0.5
    elif outcome_market_history:
        history_input = frame[OUTCOME_MARKET_FEATURES].copy()
        frame["history_over"] = history_model.predict_proba(history_input)[:, 1]
        frame["history_rate_value"] = frame["over_rate"]
    else:
        history_input = frame[
            [name if name in {"line", "over_implied", "under_implied"} else f"history_{name}" for name in NUMERIC_FEATURES]
        ].copy()
        history_input.columns = NUMERIC_FEATURES
        frame["history_over"] = history_model.predict_proba(history_input)[:, 1]
        frame["history_rate_value"] = frame["history_over_rate"]
    fixed = str(policy.get("selection") or "")
    if fixed == "Over":
        frame["selection"] = "Over"
        frame["selected_outcome"] = frame["over_outcome"]
        frame["season_probability"] = frame["season_over"]
        frame["history_probability"] = frame["history_over"]
        frame["selected_implied"] = frame["over_implied"]
        frame["season_rate_value"] = frame["over_rate"]
        classifier_agreement = (frame["season_over"] >= 0.5) & (frame["history_over"] >= 0.5)
    else:
        frame["is_over"] = frame["season_over"] >= 0.5
        frame["selection"] = np.where(frame["is_over"], "Over", "Under")
        frame["selected_outcome"] = np.where(frame["is_over"], frame["over_outcome"], 1 - frame["over_outcome"])
        frame["season_probability"] = np.where(frame["is_over"], frame["season_over"], 1 - frame["season_over"])
        frame["history_probability"] = np.where(frame["is_over"], frame["history_over"], 1 - frame["history_over"])
        frame["selected_implied"] = np.where(frame["is_over"], frame["over_implied"], frame["under_implied"])
        frame["season_rate_value"] = np.where(frame["is_over"], frame["over_rate"], 1 - frame["over_rate"])
        frame["history_rate_value"] = np.where(frame["is_over"], frame["history_rate_value"], 1 - frame["history_rate_value"])
        classifier_agreement = (frame["season_over"] >= 0.5) == (frame["history_over"] >= 0.5)
    qualified = (
        (frame["season_probability"] >= safe_float(policy.get("minimum_season_probability")))
        & (frame["history_probability"] >= safe_float(policy.get("minimum_history_probability")))
        & (frame["season_rate_value"] >= safe_float(policy.get("minimum_season_rate")))
        & (frame["history_rate_value"] >= safe_float(policy.get("minimum_history_rate")))
        & (frame["selected_implied"] >= safe_float(policy.get("minimum_implied")))
    )
    if policy.get("require_classifier_agreement"):
        qualified &= classifier_agreement
    selected = frame[qualified].copy()
    selected["consensus_score"] = (
        selected["season_probability"] + selected["history_probability"] + selected["selected_implied"]
    ) / 3
    return _metrics(_one_per_event(selected, score="consensus_score"))


def _evaluate_count_policy(
    rows: Any,
    *,
    season_model: Any,
    history_model: Any,
    policy: dict[str, Any],
) -> dict[str, Any]:
    frame = _count_prediction_frame(
        rows,
        season_model=season_model,
        history_model=history_model,
    )
    qualified = (
        (frame["over_implied"] >= safe_float(policy.get("minimum_implied")))
        & (frame["season_margin"] >= safe_float(policy.get("minimum_season_margin"), -99))
        & (frame["season_margin"] <= safe_float(policy.get("maximum_season_margin"), 99))
        & (frame["history_margin"] >= safe_float(policy.get("minimum_history_margin"), -99))
        & (frame["history_margin"] <= safe_float(policy.get("maximum_history_margin"), 99))
        & (frame["season_mean_margin"] >= safe_float(policy.get("minimum_season_mean_margin"), -99))
        & (frame["usage_trend"] >= safe_float(policy.get("minimum_usage_trend"), -99))
        & (frame["season_count"] >= 5)
    )
    selected = frame[qualified].copy()
    selected["consensus_score"] = selected["over_implied"]
    return _metrics(_one_per_event(selected, score="consensus_score"))


def _count_prediction_frame(rows: Any, *, season_model: Any, history_model: Any) -> Any:
    season_columns = [name for name in OUTCOME_FEATURES if not name.startswith("all_")]
    frame = rows.copy()
    frame["season_projection"] = season_model.predict(frame[season_columns])
    frame["history_projection"] = history_model.predict(frame[OUTCOME_FEATURES])
    frame["season_margin"] = frame["season_projection"] - frame["line"]
    frame["history_margin"] = frame["history_projection"] - frame["line"]
    frame["season_mean_margin"] = frame["season_mean"] - frame["line"]
    frame["usage_trend"] = frame["usage_mean3"] - frame["usage_mean10"]
    frame["selected_outcome"] = frame["over_outcome"]
    return frame


def _fit_count_gate(frame: Any) -> Any:
    from sklearn.ensemble import ExtraTreesClassifier  # type: ignore
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.pipeline import Pipeline  # type: ignore

    gate = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=10,
                max_features=0.8,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ])
    gate.fit(frame[COUNT_GATE_FEATURES], frame["over_outcome"].astype(int))
    return gate


def _evaluate_count_gate(frame: Any, gate: Any, threshold: float) -> dict[str, Any]:
    selected = frame.copy()
    selected["consensus_score"] = gate.predict_proba(selected[COUNT_GATE_FEATURES])[:, 1]
    selected = selected[(selected["consensus_score"] >= threshold) & (selected["season_count"] >= 5)]
    return _metrics(_one_per_event(selected, score="consensus_score"))


def _windows(sport: str) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    if sport == "MLB":
        return ("2026-05-20", "2026-05-21", "2026-06-10"), ("2026-06-10", "2026-06-11", "2026-06-19")
    return ("2026-05-23", "2026-05-24", "2026-06-10"), ("2026-06-10", "2026-06-11", "2026-06-19")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", type=Path, default=DEFAULT_MARKETS)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    args = parser.parse_args()
    market_rows = _read_jsonl(args.markets.resolve())
    outcome_rows = [
        row for row in _read_jsonl(args.outcomes.resolve())
        if str(row.get("stat_key") or "") in TARGET_STATS.get(str(row.get("sport") or "").upper(), set())
        and (
            (
                str(row.get("sport") or "").upper() == "MLB"
                and int(row.get("season") or 0) in {2022, 2023, 2024, 2025, 2026}
            )
            or (
                str(row.get("sport") or "").upper() == "WNBA"
                and int(row.get("season") or 0) in {2024, 2025, 2026}
            )
        )
    ]
    prior_rows = [row for row in outcome_rows if int(row.get("season") or 0) in {2024, 2025}]
    season_features, season_profiles = build_training_features(market_rows)
    history_features, history_profiles = build_training_features(market_rows, prior_rows)
    outcome_feature_rows, outcome_profiles = build_outcome_training_features(outcome_rows)

    import joblib  # type: ignore
    import pandas as pd  # type: ignore

    season_frame = pd.DataFrame(season_features)
    merge_keys = [
        "sport", "date", "event_id", "athlete_id", "stat_key", "line", "market_format",
        "over_outcome", "over_odds", "under_odds", "over_implied", "under_implied",
    ]
    history_frame = pd.DataFrame(history_features)
    history_prefixed = history_frame.rename(
        columns={name: f"history_{name}" for name in NUMERIC_FEATURES if name not in merge_keys}
    )
    paired = season_frame.merge(
        history_prefixed[merge_keys + [f"history_{name}" for name in NUMERIC_FEATURES if name not in merge_keys]],
        on=merge_keys,
    )
    outcome_frame = pd.DataFrame(outcome_feature_rows)
    outcome_market = _outcome_frame_for_markets(season_frame, outcome_profiles)
    outcome_for_pair = outcome_market[merge_keys + OUTCOME_FEATURES].rename(
        columns={name: f"outcome_{name}" for name in OUTCOME_FEATURES}
    )
    paired_outcome = paired.merge(
        outcome_for_pair,
        on=merge_keys,
        how="inner",
    )

    validation_results: dict[str, dict[str, dict[str, Any]]] = {"MLB": {}, "WNBA": {}}
    count_gate_models: dict[str, Any] = {}
    for sport, sport_policies in POLICIES.items():
        validation_window, holdout_window = _windows(sport)
        for stat_key, policy in sport_policies.items():
            if sport == "WNBA" and stat_key in {"assists", "three_pointers_made"}:
                prediction_frames: list[Any] = []
                for cutoff, start, end in (validation_window, holdout_window):
                    season_model = _fit_wnba_count(outcome_frame, stat_key, cutoff, season_only=True)
                    history_model = _fit_wnba_count(outcome_frame, stat_key, cutoff, season_only=False)
                    rows = _count_market_rows(
                        outcome_market,
                        outcome_frame,
                        sport,
                        stat_key,
                        start=start,
                        end=end,
                    )
                    prediction_frames.append(
                        _count_prediction_frame(
                            rows,
                            season_model=season_model,
                            history_model=history_model,
                        )
                    )
                gate = _fit_count_gate(prediction_frames[0])
                threshold = safe_float(policy.get("meta_gate_threshold"), 0.60)
                validation = _evaluate_count_gate(prediction_frames[0], gate, threshold)
                holdout = _evaluate_count_gate(prediction_frames[1], gate, threshold)
                count_gate_models[stat_key] = gate
                active = bool(
                    validation["samples"] >= int(policy["minimum_validation_samples"])
                    and holdout["samples"] >= int(policy["minimum_holdout_samples"])
                    and safe_float(validation["accuracy"]) >= TARGET_ACCURACY
                    and safe_float(holdout["accuracy"]) >= TARGET_ACCURACY
                )
                validation_results[sport][stat_key] = {
                    **policy,
                    "active": active,
                    "validation": validation,
                    "holdout": holdout,
                }
                continue
            evaluations: list[dict[str, Any]] = []
            for cutoff, start, end in (validation_window, holdout_window):
                if sport == "MLB":
                    season_model = _fit_market(
                        outcome_market[outcome_market["sport"].eq(sport)],
                        stat_key,
                        cutoff,
                        min_child_weight=20 if stat_key == "hits_runs_rbis" else 15,
                    )
                    if stat_key == "hits_runs_rbis":
                        season_model = _fit_market(
                            paired_outcome[
                                paired_outcome["sport"].eq(sport)
                                & paired_outcome["line"].eq(1.5)
                            ],
                            stat_key,
                            cutoff,
                            min_child_weight=20,
                        )
                        history_model = _fit_hrr_history(outcome_frame, cutoff)
                        rows = paired_outcome[
                            paired_outcome["sport"].eq(sport)
                            & paired_outcome["stat_key"].eq(stat_key)
                            & paired_outcome["line"].eq(1.5)
                            & paired_outcome["date"].between(start, end)
                        ].copy()
                        for feature_name in OUTCOME_FEATURES:
                            rows[feature_name] = rows[f"outcome_{feature_name}"]
                        metrics = _evaluate_classifier_policy(
                            rows,
                            season_model=season_model,
                            history_model=history_model,
                            policy=policy,
                            hrr_history=True,
                        )
                    else:
                        history_model = _fit_mlb_outcome_market_history(outcome_market, stat_key, cutoff)
                        rows = outcome_market[
                            outcome_market["sport"].eq(sport)
                            & outcome_market["stat_key"].eq(stat_key)
                            & outcome_market["date"].between(start, end)
                        ]
                        metrics = _evaluate_classifier_policy(
                            rows,
                            season_model=season_model,
                            history_model=history_model,
                            policy=policy,
                            outcome_market_history=True,
                        )
                elif stat_key == "points":
                    season_model = _fit_market(paired[paired["sport"].eq(sport)], stat_key, cutoff)
                    history_model = _fit_paired_history(paired[paired["sport"].eq(sport)], stat_key, cutoff)
                    rows = paired[
                        paired["sport"].eq(sport)
                        & paired["stat_key"].eq(stat_key)
                        & paired["date"].between(start, end)
                    ]
                    metrics = _evaluate_classifier_policy(
                        rows,
                        season_model=season_model,
                        history_model=history_model,
                        policy=policy,
                    )
                else:
                    season_model = _fit_wnba_count(outcome_frame, stat_key, cutoff, season_only=True)
                    history_model = _fit_wnba_count(outcome_frame, stat_key, cutoff, season_only=False)
                    rows = _count_market_rows(
                        outcome_market,
                        outcome_frame,
                        sport,
                        stat_key,
                        start=start,
                        end=end,
                    )
                    metrics = _evaluate_count_policy(
                        rows,
                        season_model=season_model,
                        history_model=history_model,
                        policy=policy,
                    )
                evaluations.append(metrics)
            validation, holdout = evaluations
            active = bool(
                validation["samples"] >= int(policy["minimum_validation_samples"])
                and holdout["samples"] >= int(policy["minimum_holdout_samples"])
                and safe_float(validation["accuracy"]) >= TARGET_ACCURACY
                and safe_float(holdout["accuracy"]) >= TARGET_ACCURACY
            )
            validation_results[sport][stat_key] = {
                **policy,
                "active": active,
                "validation": validation,
                "holdout": holdout,
            }

    final_artifacts: dict[tuple[str, str], dict[str, Any]] = {}
    for sport in ("MLB", "WNBA"):
        season_models: dict[str, Any] = {}
        history_models: dict[str, Any] = {}
        season_kinds: dict[str, str] = {}
        history_kinds: dict[str, str] = {}
        season_model_features: dict[str, list[str]] = {}
        history_model_features: dict[str, list[str]] = {}
        for stat_key in POLICIES[sport]:
            if sport == "WNBA" and stat_key in {"totalRebounds", "assists", "three_pointers_made"}:
                season_models[stat_key] = _fit_wnba_count(outcome_frame, stat_key, None, season_only=True)
                history_models[stat_key] = _fit_wnba_count(outcome_frame, stat_key, None, season_only=False)
                season_kinds[stat_key] = "regressor"
                history_kinds[stat_key] = "regressor"
                season_model_features[stat_key] = [
                    name for name in OUTCOME_FEATURES if not name.startswith("all_")
                ]
                history_model_features[stat_key] = OUTCOME_FEATURES
            else:
                season_models[stat_key] = _fit_market(
                    (
                        paired_outcome[
                            paired_outcome["sport"].eq(sport)
                            & paired_outcome["line"].eq(1.5)
                        ]
                        if stat_key == "hits_runs_rbis"
                        else outcome_market[outcome_market["sport"].eq(sport)]
                        if sport == "MLB"
                        else paired[paired["sport"].eq(sport)]
                    ),
                    stat_key,
                    min_child_weight=20 if stat_key == "hits_runs_rbis" else 15,
                )
                season_kinds[stat_key] = "market_classifier"
                season_model_features[stat_key] = NUMERIC_FEATURES
                if stat_key == "hits_runs_rbis":
                    history_models[stat_key] = _fit_hrr_history(outcome_frame)
                    history_kinds[stat_key] = "classifier"
                    history_model_features[stat_key] = OUTCOME_FEATURES
                elif sport == "MLB":
                    history_models[stat_key] = _fit_mlb_outcome_market_history(outcome_market, stat_key)
                    history_kinds[stat_key] = "outcome_market_classifier"
                    history_model_features[stat_key] = OUTCOME_MARKET_FEATURES
                else:
                    history_models[stat_key] = _fit_paired_history(
                        paired[paired["sport"].eq(sport)], stat_key
                    )
                    history_kinds[stat_key] = "market_classifier"
                    history_model_features[stat_key] = NUMERIC_FEATURES
        final_artifacts[(sport, "season")] = {
            "version": CONSENSUS_VERSION,
            "sport": sport,
            "role": "season",
            "models": season_models,
            "kinds": season_kinds,
            "model_features": season_model_features,
            "market_profiles": season_profiles,
            "outcome_profiles": outcome_profiles,
            "numeric_features": NUMERIC_FEATURES,
            "outcome_features": OUTCOME_FEATURES,
        }
        final_artifacts[(sport, "history")] = {
            "version": CONSENSUS_VERSION,
            "sport": sport,
            "role": "history",
            "models": history_models,
            "kinds": history_kinds,
            "model_features": history_model_features,
            "market_profiles": history_profiles,
            "outcome_profiles": outcome_profiles,
            "numeric_features": NUMERIC_FEATURES,
            "outcome_features": OUTCOME_FEATURES,
            "gate_models": count_gate_models if sport == "WNBA" else {},
            "gate_features": COUNT_GATE_FEATURES,
        }

    fingerprint = hashlib.sha256(args.markets.read_bytes() + args.outcomes.read_bytes()).hexdigest()
    sports_metadata: dict[str, Any] = {}
    for sport, policies in validation_results.items():
        active_policies = {key: value for key, value in policies.items() if value.get("active") is True}
        combined_samples = sum(
            int((policy.get(window) or {}).get("samples") or 0)
            for policy in active_policies.values()
            for window in ("validation", "holdout")
        )
        combined_wins = sum(
            int((policy.get(window) or {}).get("wins") or 0)
            for policy in active_policies.values()
            for window in ("validation", "holdout")
        )
        sports_metadata[sport] = {
            "active": bool(
                active_policies
                and combined_samples
                and combined_wins / combined_samples >= TARGET_ACCURACY
            ),
            "models": [
                "season_2026",
                "history_2022_2026" if sport == "MLB" else "history_2024_2026",
            ],
            "policies": active_policies,
            "failed_policies": {
                key: value for key, value in policies.items() if value.get("active") is not True
            },
            "combined_out_of_sample": {
                "samples": combined_samples,
                "wins": combined_wins,
                "losses": combined_samples - combined_wins,
                "accuracy": combined_wins / combined_samples if combined_samples else None,
            },
        }
    metadata = {
        "version": CONSENSUS_VERSION,
        "active": all(item.get("active") is True for item in sports_metadata.values()),
        "target_accuracy": TARGET_ACCURACY,
        "seasons": {"MLB": [2022, 2023, 2024, 2025, 2026], "WNBA": [2024, 2025, 2026]},
        "history_years": {"MLB": 5, "WNBA": 3},
        "history_years_by_market": {
            "MLB": {
                "hits_runs_rbis": 5,
                "hits": 5,
                "strikeouts": 5,
                "pitcher_walks_allowed": 5,
            },
            "WNBA": {"points": 3, "totalRebounds": 3, "assists": 3, "three_pointers_made": 3},
        },
        "roster_aware": True,
        "roster_policy": "Current player IDs only; season features reset annually; recent 3/5/10-game workload dominates older priors.",
        "models": {
            "mlb_season": "2026 market classifier",
            "mlb_history": "2022-26 five-year roster-aware outcome-history classifiers",
            "wnba_season": "2026 market/count models",
            "wnba_history": "2024-26 workload-aware market/count models",
        },
        "sports": sports_metadata,
        "training_fingerprint": fingerprint,
        "market_rows": len(market_rows),
        "outcome_rows": len(outcome_rows),
        "activation_requirements": {
            "minimum_accuracy": TARGET_ACCURACY,
            "chronological_validation_and_later_holdout": True,
            "unqualified_markets_must_abstain": True,
            "sport_specific_three_or_five_year_window_selected_by_holdout": True,
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for key, artifact in final_artifacts.items():
        joblib.dump(artifact, MODEL_PATHS[key], compress=3)
    existing_metadata: dict[str, Any] | None = None
    if CONSENSUS_METADATA_PATH.exists():
        try:
            existing_metadata = json.loads(CONSENSUS_METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_metadata = None
    if metadata.get("active") is not True:
        if isinstance(existing_metadata, dict) and existing_metadata.get("active") is True:
            print(
                "[player-prop-consensus] activation gate failed; preserving existing active metadata",
                file=sys.stderr,
            )
            print(json.dumps(existing_metadata, indent=2, sort_keys=True))
            return 2
        CONSENSUS_METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return 2
    CONSENSUS_METADATA_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
