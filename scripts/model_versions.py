"""Fingerprint the prediction implementation and fitted artifacts at publication."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORIES = {
    "mlb_new": "MLBPredictionModel", "mlb_inning": "models/mlb_inning",
    "mlb_first_five": "models/mlb_first_five", "mlb_team_total": "models/mlb_team_total",
    "nfl": "NFLPredictionModel", "cfb": "CFBPredictionModel", "wnba": "WNBAPredictionModel",
    "nba": "NBAPredictionModel", "nba_playoffs": "NBAPlayoffsPredictionModel",
    "nba_summer": "NBASummerPredictionModel", "mls": "MLSPredictionModel",
    "tennis": "TennisPredictionModel", "fifa_world_cup": "FIFAWorldCupPredictionModel", "ipl": "ipl",
}


def stamp_prediction_versions(payload, root=ROOT):
    for key, bucket in payload.get("models", {}).items():
        if key not in MODEL_DIRECTORIES or not isinstance(bucket, dict):
            continue
        directory = root / MODEL_DIRECTORIES[key]
        paths = list(directory.rglob("*.py")) + list((directory / "artifacts").rglob("*"))
        paths += [root / "pickgrader_server.py", root / "scripts/pick_calibration.py", root / "scripts/mlb_team_consensus.py",
                  root / "data/calibration/active.json"]
        digest = hashlib.sha256()
        for path in sorted(p for p in paths if p.is_file()):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        version = f"{key}:{digest.hexdigest()[:16]}"
        bucket["prediction_model_version"] = version
        for pick in bucket.get("picks") or []:
            pick["prediction_model_version"] = version
