from pathlib import Path

from calibration import train_moneyline_calibration


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    primary_path = base_dir / "data" / "mlb_historical_dataset.csv"
    legacy_path = base_dir / "data" / "mlb_historical_dataset_2023_2024.csv"
    dataset_path = primary_path if primary_path.exists() else legacy_path
    result = train_moneyline_calibration(dataset_path)
    print(result["metadata"])
