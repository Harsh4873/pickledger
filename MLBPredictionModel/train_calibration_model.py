from pathlib import Path

from calibration import train_moneyline_calibration


if __name__ == "__main__":
    result = train_moneyline_calibration(Path("data/mlb_historical_dataset_2023_2024.csv"))
    print(result["metadata"])
