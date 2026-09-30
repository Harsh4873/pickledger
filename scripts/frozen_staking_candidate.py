"""Content-bound prospective NHL/MLS rules; no historical approval inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_freeze(model_key: str, root: Path = ROOT) -> dict:
    if model_key not in {"nhl", "mls"}:
        raise ValueError("unsupported frozen candidate")
    return json.loads((root / "data/calibration" / f"{model_key}_staking_freeze.json").read_text())


def candidate_fingerprint(freeze: dict, root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for name in sorted(freeze["candidate_files"]):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def stamp_candidate(picks: list[dict], model_key: str) -> None:
    freeze = load_freeze(model_key)
    fingerprint = candidate_fingerprint(freeze)
    for pick in picks:
        pick["staking_candidate_fingerprint"] = fingerprint
        if model_key == "mls":
            # The frozen forecast includes its own calibration and market blend.
            pick["raw_probability"] = pick.get("probability")


def candidate_matches(pick: dict, model_key: str) -> bool:
    freeze = load_freeze(model_key)
    return (pick.get("model_version") == freeze["fitted_version"]
            and pick.get("staking_candidate_fingerprint") == freeze["candidate_fingerprint"]
            and candidate_fingerprint(freeze) == freeze["candidate_fingerprint"])
