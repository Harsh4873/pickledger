"""Dixon-Coles engine behind the MLS model.

Everything the model publishes — 1X2, totals, Asian-handicap spreads — is read
off one bivariate score grid: time-decayed weighted-MLE attack/defense ratings
per club with a fitted home advantage and the Dixon-Coles low-score dependence
correction, then a trained total-rate calibration and a trained vector-scaling
recalibration of the 1X2 probabilities. Hyperparameters (decay half-life,
ridge) are chosen by walk-forward validation in ``mls_train.py``, never here.

The fit is deliberately re-run at prediction time from the committed match
archive plus the roll-forward — the exact procedure walk-forward validation
scored, rather than a drifting incremental approximation of it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from .mls_data import MlsMatch

MAX_GOALS = 10
RHO_BOUNDS = (-0.25, 0.25)
# Soft identifiability pin: attack and defense ratings each sum to ~0 so the
# intercept carries the league scoring level.
SUM_PENALTY = 10.0


@dataclass(frozen=True)
class FitConfig:
    half_life_days: float | None = 365.0
    ridge: float = 5e-3
    max_goals: int = MAX_GOALS


@dataclass
class RatingsFit:
    as_of: date
    config: FitConfig
    teams: list[str]
    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home_advantage: float
    rho: float
    effective_games: dict[str, float]
    matches_used: int = 0
    _params: np.ndarray | None = field(default=None, repr=False)

    def rates(self, home: str, away: str) -> tuple[float, float]:
        """Expected 90-minute goals (home, away); unknown teams rate league-average."""
        home_attack = self.attack.get(home, 0.0)
        home_defense = self.defense.get(home, 0.0)
        away_attack = self.attack.get(away, 0.0)
        away_defense = self.defense.get(away, 0.0)
        lam = math.exp(self.intercept + self.home_advantage + home_attack - away_defense)
        mu = math.exp(self.intercept + away_attack - home_defense)
        return lam, mu

    def known(self, team: str) -> bool:
        return team in self.attack


def decay_weights(match_dates: Sequence[date], as_of: date, half_life_days: float | None) -> np.ndarray:
    ages = np.array([(as_of - when).days for when in match_dates], dtype=float)
    ages = np.maximum(ages, 0.0)
    if not half_life_days or not math.isfinite(half_life_days):
        return np.ones_like(ages)
    return np.power(0.5, ages / float(half_life_days))


def _poisson_nll_grad(params: np.ndarray, hi: np.ndarray, ai: np.ndarray,
                      x: np.ndarray, y: np.ndarray, w: np.ndarray,
                      team_count: int, ridge: float) -> tuple[float, np.ndarray]:
    intercept, home_adv = params[0], params[1]
    attack = params[2:2 + team_count]
    defense = params[2 + team_count:]
    log_lam = intercept + home_adv + attack[hi] - defense[ai]
    log_mu = intercept + attack[ai] - defense[hi]
    lam = np.exp(log_lam)
    mu = np.exp(log_mu)
    nll = float(np.sum(w * ((lam - x * log_lam) + (mu - y * log_mu))))
    nll += ridge * float(np.sum(attack ** 2) + np.sum(defense ** 2))
    nll += SUM_PENALTY * (float(np.sum(attack)) ** 2 + float(np.sum(defense)) ** 2)

    home_residual = w * (lam - x)
    away_residual = w * (mu - y)
    grad = np.zeros_like(params)
    grad[0] = float(np.sum(home_residual + away_residual))
    grad[1] = float(np.sum(home_residual))
    grad_attack = np.zeros(team_count)
    grad_defense = np.zeros(team_count)
    np.add.at(grad_attack, hi, home_residual)
    np.add.at(grad_attack, ai, away_residual)
    np.add.at(grad_defense, ai, -home_residual)
    np.add.at(grad_defense, hi, -away_residual)
    grad_attack += 2.0 * ridge * attack + 2.0 * SUM_PENALTY * float(np.sum(attack))
    grad_defense += 2.0 * ridge * defense + 2.0 * SUM_PENALTY * float(np.sum(defense))
    grad[2:2 + team_count] = grad_attack
    grad[2 + team_count:] = grad_defense
    return nll, grad


def _dc_tau_nll(rho: float, lam: np.ndarray, mu: np.ndarray, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    tau = np.ones_like(lam)
    both_zero = (x == 0) & (y == 0)
    home_zero = (x == 0) & (y == 1)
    away_zero = (x == 1) & (y == 0)
    both_one = (x == 1) & (y == 1)
    tau[both_zero] = 1.0 - lam[both_zero] * mu[both_zero] * rho
    tau[home_zero] = 1.0 + lam[home_zero] * rho
    tau[away_zero] = 1.0 + mu[away_zero] * rho
    tau[both_one] = 1.0 - rho
    return -float(np.sum(w * np.log(np.maximum(tau, 1e-9))))


def fit_ratings(
    matches: Iterable[MlsMatch],
    as_of: date,
    config: FitConfig,
    warm: RatingsFit | None = None,
) -> RatingsFit:
    """Weighted-MLE Dixon-Coles fit on matches strictly before ``as_of``."""
    past = [match for match in matches if match.date < as_of]
    teams = sorted({match.home for match in past} | {match.away for match in past})
    index = {team: position for position, team in enumerate(teams)}
    team_count = len(teams)
    if team_count < 2 or len(past) < 20:
        raise ValueError(f"not enough MLS history before {as_of} ({len(past)} matches)")

    hi = np.array([index[match.home] for match in past], dtype=int)
    ai = np.array([index[match.away] for match in past], dtype=int)
    x = np.array([match.home_goals for match in past], dtype=float)
    y = np.array([match.away_goals for match in past], dtype=float)
    w = decay_weights([match.date for match in past], as_of, config.half_life_days)

    start = np.zeros(2 + 2 * team_count)
    start[0] = math.log(max(1.05, float(np.average((x + y) / 2.0, weights=w))))
    start[1] = 0.25
    if warm is not None and warm._params is not None:
        start[0] = warm.intercept
        start[1] = warm.home_advantage
        for team, position in index.items():
            if team in warm.attack:
                start[2 + position] = warm.attack[team]
                start[2 + team_count + position] = warm.defense[team]

    result = minimize(
        _poisson_nll_grad,
        start,
        args=(hi, ai, x, y, w, team_count, config.ridge),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    params = result.x
    attack = params[2:2 + team_count]
    defense = params[2 + team_count:]
    lam = np.exp(params[0] + params[1] + attack[hi] - defense[ai])
    mu = np.exp(params[0] + attack[ai] - defense[hi])
    rho_fit = minimize_scalar(
        _dc_tau_nll,
        bounds=RHO_BOUNDS,
        args=(lam, mu, x, y, w),
        method="bounded",
    )

    effective = np.zeros(team_count)
    np.add.at(effective, hi, w)
    np.add.at(effective, ai, w)
    return RatingsFit(
        as_of=as_of,
        config=config,
        teams=teams,
        attack={team: float(attack[index[team]]) for team in teams},
        defense={team: float(defense[index[team]]) for team in teams},
        intercept=float(params[0]),
        home_advantage=float(params[1]),
        rho=float(rho_fit.x),
        effective_games={team: float(effective[index[team]]) for team in teams},
        matches_used=len(past),
        _params=params,
    )


def score_grid(lam: float, mu: float, rho: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Dixon-Coles score matrix, rows = home goals, columns = away goals."""
    goals = np.arange(max_goals + 1)
    log_fact = np.array([math.lgamma(g + 1) for g in goals])
    home = np.exp(-lam + goals * math.log(max(lam, 1e-9)) - log_fact)
    away = np.exp(-mu + goals * math.log(max(mu, 1e-9)) - log_fact)
    grid = np.outer(home, away)
    grid[0, 0] *= max(1e-9, 1.0 - lam * mu * rho)
    grid[0, 1] *= max(1e-9, 1.0 + lam * rho)
    grid[1, 0] *= max(1e-9, 1.0 + mu * rho)
    grid[1, 1] *= max(1e-9, 1.0 - rho)
    total = grid.sum()
    return grid / total if total > 0 else grid


def one_x_two(grid: np.ndarray) -> dict[str, float]:
    home = float(np.sum(np.tril(grid, -1)))
    away = float(np.sum(np.triu(grid, 1)))
    draw = float(np.trace(grid))
    return {"home": home, "draw": draw, "away": away}


def total_split(grid: np.ndarray, line: float) -> dict[str, float]:
    """Over/under/push mass at a goals line; push only at integer lines."""
    size = grid.shape[0]
    totals = np.add.outer(np.arange(size), np.arange(size))
    over = float(grid[totals > line + 1e-9].sum())
    under = float(grid[totals < line - 1e-9].sum())
    push = max(0.0, 1.0 - over - under)
    return {"over": over, "under": under, "push": push}


def _handicap_component(grid: np.ndarray, side: str, line: float) -> dict[str, float]:
    size = grid.shape[0]
    margin = np.subtract.outer(np.arange(size), np.arange(size))
    if side == "away":
        margin = -margin
    adjusted = margin + line
    win = float(grid[adjusted > 1e-9].sum())
    loss = float(grid[adjusted < -1e-9].sum())
    push = max(0.0, 1.0 - win - loss)
    return {"win": win, "push": push, "loss": loss}


def handicap_split(grid: np.ndarray, side: str, line: float) -> dict[str, float]:
    """Asian-handicap settlement mass for ``side`` at ``line``.

    Quarter lines settle as two half-stakes; the returned masses are
    stake-weighted so ``win`` minus ``loss`` remains the expected net result
    at even prices.
    """
    quarter = abs((line * 4) - round(line * 4)) < 1e-9 and abs((line * 2) - round(line * 2)) > 1e-9
    if not quarter:
        return _handicap_component(grid, side, line)
    lower = _handicap_component(grid, side, line - 0.25)
    upper = _handicap_component(grid, side, line + 0.25)
    return {
        "win": (lower["win"] + upper["win"]) / 2.0,
        "push": (lower["push"] + upper["push"]) / 2.0,
        "loss": (lower["loss"] + upper["loss"]) / 2.0,
    }


def conditional_probability(win: float, loss: float) -> float:
    """Cover probability given no push — the number a two-way price implies."""
    settled = win + loss
    return win / settled if settled > 0 else 0.5


def american_to_probability(odds: float | None) -> float | None:
    if odds is None or odds == 0:
        return None
    value = float(odds)
    return 100.0 / (value + 100.0) if value > 0 else abs(value) / (abs(value) + 100.0)


def decimal_to_probability(price: float | None) -> float | None:
    if price is None or price <= 1.0:
        return None
    return 1.0 / float(price)


def devig(implied: Sequence[float | None]) -> list[float | None]:
    """Multiplicative devig; returns None-preserving normalized probabilities."""
    known = [value for value in implied if value is not None]
    total = sum(known)
    if total <= 0 or len(known) < 2:
        return list(implied)
    return [value / total if value is not None else None for value in implied]


@dataclass(frozen=True)
class VectorScaling:
    """1X2 recalibration: softmax(gamma * log p + bias), draw as reference."""

    gamma: float = 1.0
    bias_home: float = 0.0
    bias_away: float = 0.0

    def apply(self, probabilities: dict[str, float]) -> dict[str, float]:
        eps = 1e-9
        z_home = self.gamma * math.log(max(probabilities["home"], eps)) + self.bias_home
        z_draw = self.gamma * math.log(max(probabilities["draw"], eps))
        z_away = self.gamma * math.log(max(probabilities["away"], eps)) + self.bias_away
        peak = max(z_home, z_draw, z_away)
        e_home, e_draw, e_away = (math.exp(z - peak) for z in (z_home, z_draw, z_away))
        norm = e_home + e_draw + e_away
        return {"home": e_home / norm, "draw": e_draw / norm, "away": e_away / norm}


def fit_vector_scaling(rows: Sequence[tuple[dict[str, float], str]]) -> VectorScaling:
    """Fit on out-of-sample (probabilities, outcome) pairs; outcome in H/D/A."""
    outcomes = {"home": 0, "draw": 1, "away": 2}
    logp = np.log(np.clip(np.array([
        [row[0]["home"], row[0]["draw"], row[0]["away"]] for row in rows
    ]), 1e-9, 1.0))
    target = np.array([outcomes[row[1]] for row in rows])

    def nll(theta: np.ndarray) -> float:
        gamma, bias_home, bias_away = theta
        z = gamma * logp + np.array([bias_home, 0.0, bias_away])
        z -= z.max(axis=1, keepdims=True)
        log_softmax = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -float(log_softmax[np.arange(len(target)), target].sum())

    best = minimize(nll, np.array([1.0, 0.0, 0.0]), method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 2000})
    gamma, bias_home, bias_away = best.x
    return VectorScaling(gamma=float(gamma), bias_home=float(bias_home), bias_away=float(bias_away))


@dataclass(frozen=True)
class TotalCalibration:
    """Affine correction of the fitted total goal rate: total' = a + b * total."""

    intercept: float = 0.0
    slope: float = 1.0

    def scale(self, lam: float, mu: float) -> tuple[float, float]:
        total = lam + mu
        adjusted = max(0.3, self.intercept + self.slope * total)
        factor = adjusted / total if total > 0 else 1.0
        return lam * factor, mu * factor


def fit_total_calibration(rows: Sequence[tuple[float, int]]) -> TotalCalibration:
    """Fit on out-of-sample (predicted total rate, observed total goals)."""
    predicted = np.array([row[0] for row in rows], dtype=float)
    observed = np.array([row[1] for row in rows], dtype=float)

    def nll(theta: np.ndarray) -> float:
        adjusted = np.maximum(0.3, theta[0] + theta[1] * predicted)
        return float(np.sum(adjusted - observed * np.log(adjusted)))

    best = minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead",
                    options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 2000})
    return TotalCalibration(intercept=float(best.x[0]), slope=float(best.x[1]))
