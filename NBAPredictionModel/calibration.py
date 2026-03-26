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
    class_distribution: str = ""
    balanced_distribution: str = ""


class PlattScaler:
    def __init__(self, a: float = 1.0, b: float = 0.0, fitted: bool = False):
        self.a = a
        self.b = b
        self.fitted = fitted

    def fit(
        self,
        raw_probabilities: list[float],
        outcomes: list[int],
        sample_weights: list[float] | None = None,
        iterations: int = 600,
        learning_rate: float = 0.35,
    ):
        if not raw_probabilities or len(raw_probabilities) != len(outcomes):
            raise ValueError("PlattScaler.fit requires matching probability and outcome arrays.")

        xs = [_logit(prob) for prob in raw_probabilities]
        ys = [1 if int(outcome) else 0 for outcome in outcomes]
        weights = sample_weights or [1.0] * len(xs)

        a = 1.0
        b = 0.0
        total_weight = sum(weights)
        if total_weight <= 0:
            total_weight = float(len(xs))
        for _ in range(iterations):
            grad_a = 0.0
            grad_b = 0.0
            for x_value, outcome, weight in zip(xs, ys, weights):
                prediction = _sigmoid((a * x_value) + b)
                error = prediction - outcome
                grad_a += weight * error * x_value
                grad_b += weight * error

            a -= learning_rate * (grad_a / total_weight)
            b -= learning_rate * (grad_b / total_weight)

        self.a = a
        self.b = b
        self.fitted = True
        return self

    def calibrate(self, raw_probability: float) -> float:
        score = (self.a * _logit(raw_probability)) + self.b
        return _sigmoid(score)


def _default_prediction_log_path() -> Path:
    return Path(__file__).resolve().parent / "logs" / "predictions.csv"


def _class_balance_summary(outcomes: list[int]) -> tuple[str, str, list[float], bool]:
    sample_count = len(outcomes)
    if sample_count == 0:
        return "", "", [], False

    home_wins = sum(outcomes)
    away_wins = sample_count - home_wins
    before = f"before={home_wins}/{sample_count} home wins ({(home_wins / sample_count) * 100:.1f}%)"

    if home_wins == 0 or away_wins == 0:
        return before, "after=single-class sample, no balancing possible", [1.0] * sample_count, False

    home_weight = sample_count / (2.0 * home_wins)
    away_weight = sample_count / (2.0 * away_wins)
    sample_weights = [home_weight if outcome == 1 else away_weight for outcome in outcomes]
    weighted_home = sum(weight for weight, outcome in zip(sample_weights, outcomes) if outcome == 1)
    weighted_away = sum(weight for weight, outcome in zip(sample_weights, outcomes) if outcome == 0)
    after = f"after={weighted_home:.1f}/{(weighted_home + weighted_away):.1f} effective home-win weight ({(weighted_home / (weighted_home + weighted_away)) * 100:.1f}%)"
    is_imbalanced = abs((home_wins / sample_count) - 0.5) >= 0.03
    return before, after, sample_weights, is_imbalanced


def load_platt_scaler(log_path: str | None = None) -> tuple[PlattScaler, CalibrationDiagnostics]:
    prediction_log = Path(log_path) if log_path else _default_prediction_log_path()
    scaler = PlattScaler()

    if not prediction_log.exists():
        return scaler, CalibrationDiagnostics(
            fitted=False,
            sample_count=0,
            note="Calibration wrapper is active, but there are no logged outcomes yet. Fit begins once 50+ predictions have actual_home_win recorded.",
            class_distribution="before=0 samples",
            balanced_distribution="after=not available",
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
    class_distribution, balanced_distribution, sample_weights, is_imbalanced = _class_balance_summary(outcomes)
    if sample_count < MIN_PLATT_SAMPLES:
        return scaler, CalibrationDiagnostics(
            fitted=False,
            sample_count=sample_count,
            note="Calibration wrapper is active, but fewer than 50 logged predictions have realized outcomes. Raw outputs remain uncalibrated until the sample is large enough.",
            class_distribution=class_distribution,
            balanced_distribution=balanced_distribution or "after=not available",
        )

    if is_imbalanced:
        scaler.fit(raw_probabilities, outcomes, sample_weights=sample_weights)
        note = (
            f"Platt scaling fitted from {sample_count} logged predictions with realized outcomes "
            f"using balanced class weights ({class_distribution}; {balanced_distribution})."
        )
    else:
        scaler.fit(raw_probabilities, outcomes)
        note = (
            f"Platt scaling fitted from {sample_count} logged predictions with realized outcomes. "
            f"Class balance was close to even ({class_distribution})."
        )

    return scaler, CalibrationDiagnostics(
        fitted=True,
        sample_count=sample_count,
        note=note,
        class_distribution=class_distribution,
        balanced_distribution=balanced_distribution,
    )
