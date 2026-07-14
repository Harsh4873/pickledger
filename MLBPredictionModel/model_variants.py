from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MLB_CANONICAL_ARTIFACTS = {
    "moneyline": ARTIFACT_DIR / "mlb_moneyline_model.joblib",
    "moneyline_metadata": ARTIFACT_DIR / "mlb_moneyline_model_metadata.json",
    "totals": ARTIFACT_DIR / "mlb_totals_model.joblib",
    "totals_metadata": ARTIFACT_DIR / "mlb_totals_model_metadata.json",
    "calibration": ARTIFACT_DIR / "mlb_probability_calibration.joblib",
    "calibration_metadata": ARTIFACT_DIR / "mlb_probability_calibration_metadata.json",
}

MLB_MODEL_VARIANTS = {
    "old": {
        "moneyline": ARTIFACT_DIR / "mlb_moneyline_model_old.joblib",
        "moneyline_metadata": ARTIFACT_DIR / "mlb_moneyline_model_old_metadata.json",
        "totals": ARTIFACT_DIR / "mlb_totals_model_old.joblib",
        "totals_metadata": ARTIFACT_DIR / "mlb_totals_model_old_metadata.json",
        "calibration": ARTIFACT_DIR / "mlb_probability_calibration_old.joblib",
        "calibration_metadata": ARTIFACT_DIR / "mlb_probability_calibration_old_metadata.json",
    },
    "new": {
        "moneyline": ARTIFACT_DIR / "mlb_moneyline_model_new.joblib",
        "moneyline_metadata": ARTIFACT_DIR / "mlb_moneyline_model_new_metadata.json",
        "totals": ARTIFACT_DIR / "mlb_totals_model_new.joblib",
        "totals_metadata": ARTIFACT_DIR / "mlb_totals_model_new_metadata.json",
        "calibration": ARTIFACT_DIR / "mlb_probability_calibration_new.joblib",
        "calibration_metadata": ARTIFACT_DIR / "mlb_probability_calibration_new_metadata.json",
    },
}

VALID_MLB_MODEL_VARIANTS = tuple(MLB_MODEL_VARIANTS.keys())
DEFAULT_MLB_MODEL_VARIANT = "old"


def normalize_mlb_model_variant(variant: str | None) -> str | None:
    if variant is None:
        return None
    normalized = str(variant).strip().lower()
    if not normalized:
        return None
    if normalized not in MLB_MODEL_VARIANTS:
        valid = ", ".join(VALID_MLB_MODEL_VARIANTS)
        raise ValueError(f"Unsupported MLB model variant: {variant!r}. Expected one of: {valid}.")
    return normalized


def get_mlb_model_artifacts(variant: str | None = None) -> dict[str, Path]:
    normalized = normalize_mlb_model_variant(variant)
    if normalized is None:
        return MLB_CANONICAL_ARTIFACTS
    return MLB_MODEL_VARIANTS[normalized]


def get_mlb_cache_label(variant: str | None = None) -> str:
    return "MLB Model"


def get_mlb_log_label(variant: str | None = None) -> str:
    return "MLB NEW" if normalize_mlb_model_variant(variant) == "new" else "MLB OLD"


def load_mlb_models(variant: str = DEFAULT_MLB_MODEL_VARIANT) -> dict[str, Any]:
    normalized = normalize_mlb_model_variant(variant) or DEFAULT_MLB_MODEL_VARIANT

    base_dir_str = str(BASE_DIR)
    if base_dir_str not in sys.path:
        sys.path.insert(0, base_dir_str)

    from calibration import load_calibration_artifact
    from moneyline_model import load_moneyline_model
    from totals_model import load_totals_model

    return {
        "variant": normalized,
        "moneyline": load_moneyline_model(normalized),
        "totals": load_totals_model(normalized),
        "calibration": load_calibration_artifact(normalized),
    }
