from pathlib import Path

import joblib
import pandas as pd

from totals_model import MODEL_PATH
from totals_model import train_totals_model


def _current_validation_mae() -> float | None:
    if not MODEL_PATH.exists():
        return None
    artifact = joblib.load(MODEL_PATH)
    return artifact.get("metadata", {}).get("model_metrics", {}).get("mae")


if __name__ == "__main__":
    # Use the canonical build_historical_dataset.py output (data/mlb_historical_dataset.csv).
    # The previous frozen 2023-2024 snapshot is retained as a fallback only if the
    # refreshed dataset is missing.
    base_dir = Path(__file__).resolve().parent
    primary_path = base_dir / "data" / "mlb_historical_dataset.csv"
    legacy_path = base_dir / "data" / "mlb_historical_dataset_2023_2024.csv"
    dataset_path = primary_path if primary_path.exists() else legacy_path
    dataset = pd.read_csv(dataset_path, parse_dates=["game_date"])

    min_date = dataset["game_date"].min().date()
    max_date = dataset["game_date"].max().date()
    season_rows = dataset["game_date"].dt.year.value_counts().sort_index()
    season_2024 = dataset.loc[dataset["game_date"].dt.year == 2024, "game_date"]
    includes_full_2024 = (
        not season_2024.empty
        and season_2024.min() <= pd.Timestamp("2024-03-20")
        and season_2024.max() >= pd.Timestamp("2024-09-30")
    )

    old_mae = _current_validation_mae()
    result = train_totals_model(dataset_path)
    new_mae = float(result["metadata"]["model_metrics"]["mae"])

    print(f"Training data covers {min_date} to {max_date}")
    print(
        "Season row counts: "
        + ", ".join(f"{int(year)}={int(count)}" for year, count in season_rows.items())
    )
    print(f"2024 fully included: {'yes' if includes_full_2024 else 'no'}")
    if old_mae is None:
        print("Old validation MAE: unavailable")
    else:
        print(f"Old validation MAE: {old_mae:.2f}")
    print(f"New validation MAE: {new_mae:.2f}")
    print(result["metadata"])
