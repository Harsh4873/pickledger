import csv
import math
from dataclasses import dataclass
from pathlib import Path

_PROB_EPSILON = 1e-6
MIN_PLATT_SAMPLES = 50
UNCALIBRATED_CONFIDENCE_CAP = 0.75


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
    log_flag: str = ""


def cap_probability_confidence(probability: float, cap: float = UNCALIBRATED_CONFIDENCE_CAP) -> float:
    """
    Cap certainty on either side of 50/50 while preserving the predicted side.
    """
    floor = 1.0 - cap
    return max(floor, min(cap, _clamp_probability(probability)))


class PlattScaler:
    def __init__(
        self,
        a: float = 1.0,
        b: float = 0.0,
        fitted: bool = False,
        confidence_cap: float | None = None,
        log_flag: str = "",
    ):
        self.a = a
        self.b = b
        self.fitted = fitted
        self.confidence_cap = confidence_cap
        self.log_flag = log_flag

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
        if not self.fitted and self.confidence_cap is not None:
            return cap_probability_confidence(raw_probability, self.confidence_cap)
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


def _extract_logged_raw_probability(row: dict[str, str]) -> float | None:
    # Fit Platt scaling from the pre-extremization signal only.
    raw_value = str(row.get("raw_probability", "")).strip()
    if not raw_value:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def _load_logged_outcomes(prediction_log: Path) -> tuple[list[float], list[int]]:
    raw_probabilities: list[float] = []
    outcomes: list[int] = []
    with prediction_log.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            outcome_value = str(row.get("actual_home_win", "")).strip()
            raw_value = _extract_logged_raw_probability(row)
            if raw_value is None or outcome_value == "":
                continue
            try:
                raw_probabilities.append(raw_value)
                outcomes.append(int(float(outcome_value)))
            except ValueError:
                continue
    return raw_probabilities, outcomes


def _fit_scaler_from_samples(
    raw_probabilities: list[float],
    outcomes: list[int],
    min_samples: int,
) -> tuple[PlattScaler, CalibrationDiagnostics]:
    sample_count = len(raw_probabilities)
    class_distribution, balanced_distribution, sample_weights, is_imbalanced = _class_balance_summary(outcomes)
    if sample_count < min_samples:
        scaler = PlattScaler(
            confidence_cap=UNCALIBRATED_CONFIDENCE_CAP,
            log_flag="[UNCALIBRATED]",
        )
        note = (
            "Calibration wrapper is active, but "
            f"fewer than {MIN_PLATT_SAMPLES} logged predictions have realized outcomes. "
            f"Confidence is temporarily capped at {UNCALIBRATED_CONFIDENCE_CAP * 100:.0f}% "
            "until the sample is large enough."
        )
        if sample_count == 0:
            note = (
                "Calibration wrapper is active, but there are no logged outcomes yet. "
                f"Confidence is temporarily capped at {UNCALIBRATED_CONFIDENCE_CAP * 100:.0f}% "
                f"until {MIN_PLATT_SAMPLES}+ predictions have actual_home_win recorded."
            )
        return scaler, CalibrationDiagnostics(
            fitted=False,
            sample_count=sample_count,
            note=note,
            class_distribution=class_distribution or "before=0 samples",
            balanced_distribution=balanced_distribution or "after=not available",
            log_flag="[UNCALIBRATED]",
        )

    scaler = PlattScaler()

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


def fit_platt_scaler_from_log(
    log_path: str | None = None,
    min_samples: int = MIN_PLATT_SAMPLES,
) -> tuple[PlattScaler, CalibrationDiagnostics]:
    prediction_log = Path(log_path) if log_path else _default_prediction_log_path()
    if not prediction_log.exists():
        return _fit_scaler_from_samples([], [], min_samples=min_samples)

    raw_probabilities, outcomes = _load_logged_outcomes(prediction_log)
    return _fit_scaler_from_samples(raw_probabilities, outcomes, min_samples=min_samples)


def load_platt_scaler(log_path: str | None = None) -> tuple[PlattScaler, CalibrationDiagnostics]:
    return fit_platt_scaler_from_log(log_path=log_path, min_samples=MIN_PLATT_SAMPLES)
