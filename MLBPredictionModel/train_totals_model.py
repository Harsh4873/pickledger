from pathlib import Path

from totals_model import train_totals_model


if __name__ == "__main__":
    result = train_totals_model(Path("data/mlb_historical_dataset_2023_2024.csv"))
    print(result["metadata"])
