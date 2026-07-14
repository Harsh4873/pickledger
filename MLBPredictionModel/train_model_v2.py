"""
Train the MLB v2 stack (the "new" variant).

This script is safe to run locally once you have a refreshed
`data/mlb_historical_dataset.csv` (produced by `build_historical_dataset.py`).
It is also invoked by the `mlb-train` GitHub Actions workflow so we can rebuild
the model on demand across multiple full seasons.

Outputs:
  artifacts/mlb_moneyline_model_new.joblib (+ metadata JSON)
  artifacts/mlb_totals_model_new.joblib (+ metadata JSON)
  artifacts/mlb_probability_calibration_new.joblib (+ metadata JSON)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from features_v2 import (
    build_feature_frame,
    select_feature_matrix,
    select_training_rows_v2,
)
from historical_data import DATASET_PATH
from model_v2 import (
    chronological_split,
    fit_calibration,
    save_training_artifacts,
    train_moneyline_v2,
    train_totals_v2,
)


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"Historical dataset not found at {path}. "
            "Run `python build_historical_dataset.py --seasons 2023 2024 2025 2026` first."
        )
    return pd.read_csv(path, parse_dates=["game_date"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLB model v2 (the new variant).")
    parser.add_argument(
        "--dataset",
        default=str(DATASET_PATH),
        help="Path to the historical dataset CSV (default: data/mlb_historical_dataset.csv).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of the timeline reserved for out-of-sample validation.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_path = Path(args.dataset)
    frame = _load_dataset(dataset_path)
    if frame.empty:
        print(f"Dataset at {dataset_path} is empty. Nothing to train on.", file=sys.stderr)
        return 1

    print(f"Loaded {len(frame):,} historical rows from {dataset_path}")
    print(
        "Season counts:",
        dict(sorted(frame["game_date"].dt.year.value_counts().items())),
    )

    moneyline_result = train_moneyline_v2(frame, validation_fraction=args.validation_fraction)
    totals_result = train_totals_v2(frame, validation_fraction=args.validation_fraction)

    # Fit calibration on the same chronological validation fold that
    # train_moneyline_v2 used. Re-derive the fold here to avoid coupling to
    # internal state.
    prepared = build_feature_frame(frame)
    prepared = select_training_rows_v2(prepared)
    _, validation_frame = chronological_split(prepared, args.validation_fraction)
    validation_x = select_feature_matrix(validation_frame)
    validation_y = validation_frame["home_win"].astype(int).to_numpy()
    calibration_artifact = fit_calibration(
        moneyline_result.artifact["pipeline"], validation_x, validation_y
    )

    save_training_artifacts(moneyline_result, totals_result, calibration_artifact)

    print("Moneyline metrics:")
    print("  raw model:       ", moneyline_result.metadata["raw_model_metrics"])
    print("  market baseline: ", moneyline_result.metadata["market_baseline_metrics"])
    print("Totals metrics:")
    print("  residual model:  ", totals_result.metadata["model_metrics"])
    print("  market baseline: ", totals_result.metadata["market_baseline_metrics"])
    print("Calibration:")
    print("  ", calibration_artifact.get("metadata", {}))
    print("Artifacts saved under artifacts/*_new.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
