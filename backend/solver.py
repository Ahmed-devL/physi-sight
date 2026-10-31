"""Sparse proposals followed by robust trajectory shooting on raw observations.

Scalar coordinates must have consistent units; angles are radians and time is
seconds. Cartesian y is positive upward. Gravity-derived calibration does not
constitute an independent gravity measurement. Fits never use held-out samples.
"""
from dataclasses import dataclass

import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter


TERMS = {
    "constant": lambda x, v: 1.0,
    "velocity": lambda x, v: v,
    "quadratic_drag": lambda x, v: v * abs(v),
    "position": lambda x, v: x,
    "sin_position": lambda x, v: np.sin(x),
}
CANDIDATES = {
    "pendulum": {
        "nonlinear_pendulum": ("sin_position", "velocity"),
        "linear_pendulum": ("position", "velocity"),
    },
    "drag": {
        "linear_drag": ("constant", "velocity"),
        "quadratic_drag": ("constant", "quadratic_drag"),
    },
    "spring": {"damped_spring": ("position", "velocity")},
}


@dataclass(frozen=True)
class FitResult:
    name: str
    coefficients: dict[str, float]
    initial_state: tuple[float, float]
    prediction: np.ndarray
    train_size: int
    bic: float
    heldout_r2: float
    heldout_rmse: float
    sparse_proposal: dict[str, float]

    @property
    def gamma(self) -> float:
        return -self.coefficients["velocity"]

    @property
    def omega0_squared(self) -> float:
        return -self.coefficients["sin_position"]


def _validate(times, observations):
    t, y = np.asarray(times, dtype=float), np.asarray(observations, dtype=float)
    if t.ndim != 1 or y.shape != t.shape or len(t) < 20:
        raise ValueError("Expected at least 20 scalar observations and matching timestamps")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(np.diff(t) <= 0):
        raise ValueError("Observations must be finite and timestamps strictly increasing")
    return t - t[0], y


def integrate(times, terms, coefficients, initial_state):
    """Forward integrate from the first timestamp, using RK45."""
    t = np.asarray(times, dtype=float)
    def rhs(_t, state):
        x, v = state
        return [v, sum(c * TERMS[term](x, v) for term, c in zip(terms, coefficients))]
    result = solve_ivp(rhs, (t[0], t[-1]), initial_state, t_eval=t,
                       method="RK45", rtol=1e-8, atol=1e-10)
    if not result.success or result.y.shape[1] != len(t) or not np.all(np.isfinite(result.y)):
        raise ValueError("Candidate trajectory could not be integrated")
    return result.y[0]


def _proposal(t, y, terms):
    # Resample ONLY for initialization so irregular presentation times are valid.
    grid = np.linspace(t[0], t[-1], len(t))
    values = np.interp(grid, t, y)
    window = min(15, len(t) if len(t) % 2 else len(t) - 1)
    dt = grid[1] - grid[0]
    smooth = savgol_filter(values, window, 3)
    velocity = savgol_filter(values, window, 3, deriv=1, delta=dt)
    acceleration = savgol_filter(values, window, 3, deriv=2, delta=dt)
    design = np.column_stack([
        np.broadcast_to(TERMS[term](smooth, velocity), smooth.shape) for term in terms
    ])
    # PySINDy's sparse optimizer proposes/prunes acceleration terms. These
    # differentiated data never enter the shooting residual or model score.
    optimizer = ps.STLSQ(threshold=0.1, alpha=1e-5, normalize_columns=True)
    trim = slice(window // 2, -window // 2)
    optimizer.fit(design[trim], acceleration[trim, None])
    coefficients = optimizer.coef_.ravel()
    return dict(zip(terms, coefficients)), float(smooth[0]), float(velocity[0])


def _score(name, coefficients, initial, prediction, y, split, count, proposal):
    rss = max(float(np.sum((prediction[:split] - y[:split]) ** 2)), np.finfo(float).tiny)
    bic = split * np.log(rss / split) + count * np.log(split)
    error = prediction[split:] - y[split:]
    variance = np.sum((y[split:] - y[split:].mean()) ** 2)
    r2 = float(1 - np.sum(error**2) / variance) if variance > 0 else float("nan")
    return FitResult(name, coefficients, initial, prediction, split, float(bic),
                     r2, float(np.sqrt(np.mean(error**2))), proposal)


def fit_trajectory(times, observations, *, regime="pendulum", noise_scale=0.015):
    """Return candidates sorted by training BIC, including a degree-5 baseline.

    Fit coefficients AND initial position/velocity on the first 70%. Pendulum
    and drag coefficients are constrained to restoring/dissipative signs.
    Coefficients multiply the named terms directly (e.g. velocity = -gamma).
    """
    t, y = _validate(times, observations)
    if regime not in CANDIDATES or not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("Unknown regime or invalid noise_scale")
    split = int(0.7 * len(t))
    candidates = dict(CANDIDATES[regime])
    terms = tuple(dict.fromkeys(term for terms in candidates.values() for term in terms))
    proposal, x0, v0 = _proposal(t[:split], y[:split], terms)
    sparse = tuple(term for term, value in proposal.items() if abs(value) > 1e-10)
    if sparse and sparse not in candidates.values():
        candidates["sparse_" + regime] = sparse
    results = []
    for name, active in candidates.items():
        # A pruned term may still occur in a physical candidate: give it a
        # nonzero seed and allow shooting to determine its coefficient.
        initial = [min(proposal[term], -1e-3) for term in active] + [x0, v0]
        def residual(parameters):
            try:
                predicted = integrate(t[:split], active, parameters[:-2], parameters[-2:])
                return predicted - y[:split]
            except (ValueError, FloatingPointError):
                return np.full(split, 1e10)
        fit = least_squares(
            residual, initial, loss="soft_l1", f_scale=noise_scale,
            bounds=([-np.inf] * (len(active) + 2), [0.0] * len(active) + [np.inf, np.inf]),
            max_nfev=500,
        )
        if not fit.success:
            raise RuntimeError(f"Shooting failed for {name}: {fit.message}")
        prediction = integrate(t, active, fit.x[:-2], fit.x[-2:])
        results.append(_score(name, dict(zip(active, fit.x[:-2])), tuple(fit.x[-2:]),
                              prediction, y, split, len(fit.x), proposal))
    polynomial = np.polynomial.Polynomial.fit(t[:split], y[:split], deg=5)
    results.append(_score("polynomial_degree_5", {}, (float(polynomial(0)), float(polynomial.deriv()(0))),
                          polynomial(t), y, split, 6, proposal))
    return sorted(results, key=lambda result: result.bic)


def simulate_pendulum(*, length=0.50, gamma=0.25, gravity=9.81, frames=120,
                      fps=30.0, theta0=1.0, omega0=0.0, noise_sigma=0.015, seed=42):
    """Return timestamps, clean angles, and reproducible noisy raw angles."""
    if (not np.all(np.isfinite([length, gamma, gravity, fps, theta0, omega0, noise_sigma]))
            or length <= 0 or gamma < 0 or gravity <= 0 or frames < 2
            or fps <= 0 or noise_sigma < 0):
        raise ValueError("Invalid pendulum simulation parameters")
    t = np.arange(frames) / fps
    clean = integrate(t, ("sin_position", "velocity"), (-gravity / length, -gamma), (theta0, omega0))
    return t, clean, clean + np.random.default_rng(seed).normal(0, noise_sigma, frames)
