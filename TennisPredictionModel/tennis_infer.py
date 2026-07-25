"""Dependency-free scorer for the exported tennis model.

Training runs on scikit-learn; serving does not. The trained ensemble is
exported to a plain JSON document — decision-tree nodes, the isotonic
calibration knots, and the market-combination coefficients — and evaluated
here with nothing but the standard library.

That is a deliberate operational choice, not a stylistic one. The daily cache
refresh runs unattended in GitHub Actions; a pickled estimator silently couples
it to one scikit-learn build, and a version bump in ``requirements.txt`` would
turn into a wrong-or-missing tennis slate at 6am. The export is verified against
scikit-learn at train time (``max |Δp|`` is recorded in the artifact metadata),
so this path is exact rather than approximate.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

from .tennis_core import FEATURE_NAMES, mirror_vector, symmetrise

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "tennis_model.json"


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _logit(probability: float, floor: float = 1e-6) -> float:
    clipped = min(max(probability, floor), 1.0 - floor)
    return math.log(clipped / (1.0 - clipped))


class TennisModel:
    """A trained artifact: trees (or coefficients), calibration and blending."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.kind = str(payload.get("kind") or "trees")
        self.feature_names: list[str] = list(payload.get("feature_names") or FEATURE_NAMES)
        self.baseline = float(payload.get("baseline", 0.0))
        self.trees: list[dict[str, list[Any]]] = payload.get("trees") or []
        self.coefficients: list[float] = [float(v) for v in (payload.get("coefficients") or [])]
        self.intercept = float(payload.get("intercept", 0.0))
        self.means: list[float] = [float(v) for v in (payload.get("means") or [])]
        self.scales: list[float] = [float(v) for v in (payload.get("scales") or [])]
        calibration = payload.get("calibration") or {}
        self.calibration_x: list[float] = [float(v) for v in (calibration.get("x") or [])]
        self.calibration_y: list[float] = [float(v) for v in (calibration.get("y") or [])]
        blend = payload.get("market_blend") or {}
        self.blend_intercept = float(blend.get("intercept", 0.0))
        self.blend_model = float(blend.get("model_weight", 1.0))
        self.blend_market = float(blend.get("market_weight", 0.0))
        self.metadata: dict[str, Any] = payload.get("metadata") or {}

    # -- raw model ---------------------------------------------------------

    def _raw(self, vector: Sequence[float]) -> float:
        if self.kind == "linear":
            total = self.intercept
            for index, weight in enumerate(self.coefficients):
                value = float(vector[index])
                if self.scales and self.scales[index]:
                    value = (value - self.means[index]) / self.scales[index]
                total += weight * value
            return total
        total = self.baseline
        for tree in self.trees:
            is_leaf = tree["is_leaf"]
            left = tree["left"]
            right = tree["right"]
            feature = tree["feature"]
            threshold = tree["threshold"]
            value = tree["value"]
            node = 0
            while not is_leaf[node]:
                node = left[node] if float(vector[feature[node]]) <= threshold[node] else right[node]
            total += value[node]
        return total

    def _uncalibrated(self, vector: Sequence[float]) -> float:
        return _sigmoid(self._raw(vector))

    # -- calibration -------------------------------------------------------

    def calibrate(self, probability: float) -> float:
        """Piecewise-linear interpolation over the exported isotonic knots."""
        knots_x = self.calibration_x
        knots_y = self.calibration_y
        if not knots_x:
            return probability
        if probability <= knots_x[0]:
            return knots_y[0]
        if probability >= knots_x[-1]:
            return knots_y[-1]
        low, high = 0, len(knots_x) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if knots_x[middle] <= probability:
                low = middle
            else:
                high = middle
        span = knots_x[high] - knots_x[low]
        if span <= 0:
            return knots_y[low]
        weight = (probability - knots_x[low]) / span
        return knots_y[low] + weight * (knots_y[high] - knots_y[low])

    # -- public API --------------------------------------------------------

    def predict(self, vector: Sequence[float]) -> float:
        """Calibrated, symmetrised P(player 1 wins), market-free."""
        forward = self._uncalibrated(vector)
        mirrored = self._uncalibrated(mirror_vector(vector))
        return min(max(self.calibrate(symmetrise(forward, mirrored)), 1e-4), 1.0 - 1e-4)

    def blend_with_market(self, model_probability: float, market_probability: float | None) -> float:
        """Combine the model with the de-vigged market price.

        The weights come from a logistic forecast-combination fitted on
        out-of-sample predictions, so they encode how much the model actually
        adds once the market is known — which, for tennis, is not much. Without
        a price the model stands alone.
        """
        if market_probability is None or not (0.0 < market_probability < 1.0):
            return model_probability
        combined = (
            self.blend_intercept
            + self.blend_model * _logit(model_probability)
            + self.blend_market * _logit(market_probability)
        )
        return min(max(_sigmoid(combined), 1e-4), 1.0 - 1e-4)

    @property
    def version(self) -> str:
        return str(self.metadata.get("model_version") or "unknown")


def load_model(path: Path = MODEL_PATH) -> TennisModel | None:
    """Load the artifact, refusing one whose feature contract has drifted.

    The scorer indexes the feature vector positionally, so an artifact trained
    against a different feature list does not fail — it silently reads the wrong
    column and returns confident nonsense. If the code has moved on from the
    committed artifact, returning None is the right answer: the caller emits an
    empty slate and the missing picks are visible, where wrong picks would not
    be.
    """
    if not path.exists():
        return None
    try:
        model = TennisModel(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return None
    if model.feature_names != list(FEATURE_NAMES):
        print(
            "[tennis] artifact feature contract does not match the code "
            f"({len(model.feature_names)} vs {len(FEATURE_NAMES)} features); retrain required"
        )
        return None
    return model
