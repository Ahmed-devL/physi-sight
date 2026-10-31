"""Standalone synthetic preparation, without a dependency on backend code.

Run: .venv/bin/python tests/test_synthetic_solver.py
Or:  .venv/bin/python -m pytest -s tests/test_synthetic_solver.py

Uses the requested gamma=0.15 and pixel noise, rather than the full Gate 1
benchmark's gamma=0.25 and angle noise. Gamma is the coefficient in
theta'' + gamma*theta' + (g/L)*sin(theta) = 0 (units 1/s).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


G = 9.81
TRUE_LENGTH = 0.50
TRUE_GAMMA = 0.15
FPS = 30
FRAMES = 120
NOISE_PX = 1.5
RADIUS_PX = 150.0
PIVOT_PX = np.array([320.0, 120.0])


def simulate(times: np.ndarray, length: float, gamma: float,
             theta0: float, omega0: float) -> np.ndarray:
    def rhs(_time: float, state: np.ndarray) -> list[float]:
        theta, omega = state
        return [omega, -gamma * omega - (G / length) * np.sin(theta)]

    result = solve_ivp(
        rhs, (0.0, float(times[-1])), [theta0, omega0], t_eval=times,
        method="RK45", rtol=1e-9, atol=1e-11,
    )
    if not result.success:
        raise RuntimeError(result.message)
    theta = result.y[0]
    # Screen coordinates: y grows downward. Pivot and radius are known here;
    # real-video circle fitting and calibration belong to the later pipeline.
    return PIVOT_PX + RADIUS_PX * np.column_stack((np.sin(theta), np.cos(theta)))


def test_synthetic_pendulum_shooting() -> None:
    times = np.arange(FRAMES, dtype=float) / FPS
    clean = simulate(times, TRUE_LENGTH, TRUE_GAMMA, theta0=0.9, omega0=0.0)
    observations = clean + np.random.default_rng(42).normal(
        0.0, NOISE_PX, size=clean.shape,
    )
    split = int(0.70 * FRAMES)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        prediction = simulate(times[:split], *parameters)
        return (prediction - observations[:split]).ravel()

    # Refine length, damping, and both initial conditions against raw pixels.
    # No differentiated observations or held-out samples enter optimization.
    fit = least_squares(
        residuals, x0=[0.55, 0.20, 0.85, 0.05],
        bounds=([0.1, 0.0, 0.1, -3.0], [2.0, 2.0, 1.5, 3.0]),
        loss="soft_l1", f_scale=NOISE_PX, max_nfev=200,
    )
    assert fit.success, fit.message
    length, gamma, theta0, omega0 = fit.x
    forecast = simulate(times, length, gamma, theta0, omega0)
    held_out = observations[split:]
    errors = forecast[split:] - held_out
    rmse_px = float(np.sqrt(np.mean(errors**2)))
    total_variance = np.sum((held_out - held_out.mean(axis=0)) ** 2)
    r_squared = float(1.0 - np.sum(errors**2) / total_variance)
    print(
        f"Synthetic pendulum: {FRAMES} frames at {FPS} FPS, noise={NOISE_PX}px\n"
        f"L={length:.5f}m (true {TRUE_LENGTH}), "
        f"gamma={gamma:.5f}/s (true {TRUE_GAMMA})\n"
        f"Train/holdout={split}/{FRAMES - split}, "
        f"held-out R²={r_squared:.6f}, RMSE={rmse_px:.3f}px"
    )
    assert observations.shape == (120, 2)
    assert abs(length / TRUE_LENGTH - 1.0) < 0.03
    # Pixel-noise preparation tolerance; not the full Gate 1 3% damping check.
    assert abs(gamma - TRUE_GAMMA) < 0.04
    assert r_squared > 0.95
    assert rmse_px < 3.0 * NOISE_PX


if __name__ == "__main__":
    test_synthetic_pendulum_shooting()
