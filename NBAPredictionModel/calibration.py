import csv
import math
from dataclasses import dataclass
from pathlib import Path

_PROB_EPSILON = 1e-6
MIN_PLATT_SAMPLES = 50


def _clamp_probability(probability: float) -> float:
    return max(_PROB_EPSILON, min(1.0 - _PROB_EPSILON, float(probability)))


def _logit(probability: float) -> float:
    clean_probability = _clamp_probability(probability)
    return math.log(clean_probability / (1.0 - clean_probability))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


@dataclass
class CalibrationDiagnostics:
    fitted: bool
    sample_count: int
    note: str


class PlattScaler:
    def __init__(self, a: float = 1.0, b: float = 0.0, fitted: bool = False):
        self.a = a
        self.b = b
        self.fitted = fitted

    def fit(self, raw_probabilities: list[float], outcomes: list[int], iterations: int = 600, learning_rate: float = 0.35):
        if not raw_probabilities or len(raw_probabilities) != len(outcomes):
            raise ValueError("PlattScaler.fit requires matching probability and outcome arrays.")

        xs = [_logit(prob) for prob in raw_probabilities]
        ys = [1 if int(outcome) else 0 for outcome in outcomes]

        a = 1.0
        b = 0.0
        sample_count = len(xs)
        for _ in range(iterations):
            grad_a = 0.0
            grad_b = 0.0
            for x_value, outcome in zip(xs, ys):
                prediction = _sigmoid((a * x_value) + b)
                error = prediction - outcome
                grad_a += error * x_value
                grad_b += error

            a -= learning_rate * (grad_a / sample_count)
            b -= learning_rate * (grad_b / sample_count)

        self.a = a
        self.b = b
        self.fitted = True
        return self

    def calibrate(self, raw_probability: float) -> float:
        score = (self.a * _logit(raw_probability)) + self.b
        return _sigmoid(score)


def _default_prediction_log_path() -> Path:
    return Path(__file__).resolve().parent / "logs" / "predictions.csv"


def load_platt_scaler(log_path: str | None = None) -> tuple[PlattScaler, CalibrationDiagnostics]:
    prediction_log = Path(log_path) if log_path else _default_prediction_log_path()
    scaler = PlattScaler()

    if not prediction_log.exists():
        return scaler, CalibrationDiagnostics(
            fitted=False,
            sample_count=0,
            note="Calibration wrapper is active, but there are no logged outcomes yet. Fit begins once 50+ predictions have actual_home_win recorded.",
        )

    raw_probabilities: list[float] = []
    outcomes: list[int] = []
    with prediction_log.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_value = str(row.get("raw_probability", "")).strip()
            outcome_value = str(row.get("actual_home_win", "")).strip()
            if not raw_value or outcome_value == "":
                continue
            try:
                raw_probabilities.append(float(raw_value))
                outcomes.append(int(float(outcome_value)))
            except ValueError:
                continue

    sample_count = len(raw_probabilities)
    if sample_count < MIN_PLATT_SAMPLES:
        return scaler, CalibrationDiagnostics(
            fitted=False,
            sample_count=sample_count,
            note="Calibration wrapper is active, but fewer than 50 logged predictions have realized outcomes. Raw outputs remain uncalibrated until the sample is large enough.",
        )

    scaler.fit(raw_probabilities, outcomes)
    return scaler, CalibrationDiagnostics(
        fitted=True,
        sample_count=sample_count,
        note=f"Platt scaling fitted from {sample_count} logged predictions with realized outcomes.",
    )
