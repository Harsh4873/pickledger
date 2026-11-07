import csv
import math
from dataclasses import dataclass
from pathlib import Path

_PROB_EPSILON = 1e-6
MIN_PLATT_SAMPLES = 50
UNCALIBRATED_CONFIDENCE_CAP = 0.75
# Logged rows before this date came from the model whose layer sum overflowed to
# 99.9999% and whose early-season inputs were a handful of games. They describe a
# different model, and fitting on them flattened every probability toward 51%.
PLATT_FIT_SINCE = "2026-11-08"
# Priors on the Platt fit, centred on no change. A season of graded games is needed
# before the data can move the slope far from 1.
PLATT_SLOPE_PRIOR_SD = 0.15
PLATT_INTERCEPT_PRIOR_SD = 0.15


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
        iterations: int = 100,
    ):
        """
        Maximum a posteriori Platt fit. Gaussian priors centred on no change
        (slope 1, intercept 0) keep a small or lopsided sample from flattening
        every probability, which is what an unregularised fit on 80 games did.
        """
        if not raw_probabilities or len(raw_probabilities) != len(outcomes):
            raise ValueError("PlattScaler.fit requires matching probability and outcome arrays.")

        xs = [_logit(prob) for prob in raw_probabilities]
        ys = [1 if int(outcome) else 0 for outcome in outcomes]
        weights = sample_weights or [1.0] * len(xs)
        slope_precision = 1.0 / (PLATT_SLOPE_PRIOR_SD ** 2)
        intercept_precision = 1.0 / (PLATT_INTERCEPT_PRIOR_SD ** 2)

        def objective(slope: float, intercept: float) -> float:
            total = 0.5 * ((slope_precision * (slope - 1.0) ** 2) + (intercept_precision * intercept ** 2))
            for x_value, outcome, weight in zip(xs, ys, weights):
                score = (slope * x_value) + intercept
                # -log(sigmoid(score)) for a home win, -log(1 - sigmoid(score)) otherwise.
                margin = -score if outcome else score
                total += weight * (max(margin, 0.0) + math.log1p(math.exp(-abs(margin))))
            return total

        a = 1.0
        b = 0.0
        current = objective(a, b)
        for _ in range(iterations):
            grad_a = slope_precision * (a - 1.0)
            grad_b = intercept_precision * b
            hess_aa = slope_precision
            hess_ab = 0.0
            hess_bb = intercept_precision
            for x_value, outcome, weight in zip(xs, ys, weights):
                prediction = _sigmoid((a * x_value) + b)
                error = prediction - outcome
                variance = prediction * (1.0 - prediction)
                grad_a += weight * error * x_value
                grad_b += weight * error
                hess_aa += weight * variance * x_value * x_value
                hess_ab += weight * variance * x_value
                hess_bb += weight * variance
            determinant = (hess_aa * hess_bb) - (hess_ab * hess_ab)
            step_a = ((hess_bb * grad_a) - (hess_ab * grad_b)) / determinant
            step_b = ((hess_aa * grad_b) - (hess_ab * grad_a)) / determinant
            # Saturated scores (a logged 99.9999%) make full Newton steps overshoot,
            # so halve the step until the objective improves.
            scale = 1.0
            while scale > 1e-6:
                candidate = objective(a - (scale * step_a), b - (scale * step_b))
                if candidate <= current:
                    break
                scale *= 0.5
            a -= scale * step_a
            b -= scale * step_b
            improvement = current - candidate
            current = candidate
            if max(abs(scale * step_a), abs(scale * step_b)) < 1e-10 or improvement < 1e-12:
                break

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
    try:
        from prediction_logging import default_prediction_log_path
    except ImportError:
        return Path(__file__).resolve().parent.parent / "data" / "nba" / "predictions.csv"
    return default_prediction_log_path()


def _class_balance_summary(outcomes: list[int]) -> str:
    sample_count = len(outcomes)
    if sample_count == 0:
        return ""
    home_wins = sum(outcomes)
    return f"{home_wins}/{sample_count} home wins ({(home_wins / sample_count) * 100:.1f}%)"


def _extract_logged_raw_probability(row: dict[str, str]) -> float | None:
    # Fit Platt scaling from the pre-extremization signal only.
    raw_value = str(row.get("raw_probability", "")).strip()
    if not raw_value:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def _load_logged_outcomes(prediction_log: Path, since: str = "") -> tuple[list[float], list[int]]:
    raw_probabilities: list[float] = []
    outcomes: list[int] = []
    with prediction_log.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if since and str(row.get("game_date", "")).strip() < since:
                continue
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
    since: str = "",
) -> tuple[PlattScaler, CalibrationDiagnostics]:
    sample_count = len(raw_probabilities)
    class_distribution = _class_balance_summary(outcomes)
    window = f" since {since}" if since else ""
    if sample_count < min_samples:
        # Below the minimum the model's own probability stands. The old 75% cap
        # turned every heavy favourite into a fake edge against the market.
        scaler = PlattScaler(log_flag="[UNCALIBRATED]")
        note = (
            f"{sample_count} graded prediction(s){window}; Platt scaling starts at {min_samples}. "
            "Probabilities are the model's raw output, which a point-in-time backtest found close to calibrated."
        )
        return scaler, CalibrationDiagnostics(
            fitted=False,
            sample_count=sample_count,
            note=note,
            class_distribution=class_distribution or "0 samples",
            log_flag="[UNCALIBRATED]",
        )

    # Home wins are not reweighted: the model already prices home court, and
    # balancing the classes would push every home team toward 50%.
    scaler = PlattScaler().fit(raw_probabilities, outcomes)
    note = (
        f"Platt scaling fitted from {sample_count} graded predictions{window} ({class_distribution}), "
        f"held near no change by its priors: slope {scaler.a:.2f}, intercept {scaler.b:+.2f}."
    )
    return scaler, CalibrationDiagnostics(
        fitted=True,
        sample_count=sample_count,
        note=note,
        class_distribution=class_distribution,
    )


def fit_platt_scaler_from_log(
    log_path: str | None = None,
    min_samples: int = MIN_PLATT_SAMPLES,
    since: str = PLATT_FIT_SINCE,
) -> tuple[PlattScaler, CalibrationDiagnostics]:
    prediction_log = Path(log_path) if log_path else _default_prediction_log_path()
    if not prediction_log.exists():
        return _fit_scaler_from_samples([], [], min_samples=min_samples, since=since)

    raw_probabilities, outcomes = _load_logged_outcomes(prediction_log, since=since)
    return _fit_scaler_from_samples(raw_probabilities, outcomes, min_samples=min_samples, since=since)


def load_platt_scaler(log_path: str | None = None) -> tuple[PlattScaler, CalibrationDiagnostics]:
    return fit_platt_scaler_from_log(log_path=log_path, min_samples=MIN_PLATT_SAMPLES)
