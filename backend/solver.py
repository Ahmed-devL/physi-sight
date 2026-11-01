"""Sparse proposals followed by robust trajectory shooting on raw observations.

Scalar coordinates must have consistent units; angles are radians and time is
seconds. Cartesian y is positive upward. Gravity-derived calibration does not
constitute an independent gravity measurement. Fits never use held-out samples.
"""
from dataclasses import dataclass, replace

import numpy as np
import pysindy as ps
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, find_peaks


TERMS = {
    "constant": lambda x, v: 1.0,
    "velocity": lambda x, v: v,
    "quadratic_drag": lambda x, v: v * abs(v),
    "position": lambda x, v: x,
    "sin_position": lambda x, v: np.sin(x),
}
CANDIDATES = {
    "pendulum": {
        "simple_harmonic": ("position",),
        "nonlinear_pendulum": ("sin_position", "velocity"),
        "linear_pendulum": ("position", "velocity"),
    },
    "drag": {
        "free_fall": ("constant",),
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
    release_index: int = 0

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


def integrate(times, terms, coefficients, initial_state, theta_eq=0.0):
    """Forward integrate from the first timestamp, using RK45."""
    t = np.asarray(times, dtype=float)
    def rhs(_t, state):
        x, v = state
        return [v, sum(c * TERMS[term](x - theta_eq, v) for term, c in zip(terms, coefficients))]
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


def fit_trajectory(times, observations, *, regime="pendulum", noise_scale=0.015,
                   release_index=None):
    """Fit the free-motion window; preserve pre-release observed positions.

    train_size counts active training samples; release_index anchors the fitted
    initial state in the original observations. Pre-release points are unscored.
    """
    from backend.kinematics import detect_release_index

    t, y = _validate(times, observations)
    if not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("Invalid noise_scale")
    release = (detect_release_index(t, y, regime=regime, noise_scale=noise_scale)
               if release_index is None else release_index)
    if not isinstance(release, (int, np.integer)) or release < 0 or len(t) - release < 20:
        raise ValueError("Release must leave at least 20 free-motion observations")
    results = _fit_active(t[release:] - t[release], y[release:], regime=regime,
                          noise_scale=noise_scale)
    return [replace(r, prediction=np.concatenate((y[:release], r.prediction)),
                    release_index=int(release)) for r in results]


def _fit_active(times, observations, *, regime="pendulum", noise_scale=0.015):
    """Return candidates sorted by training BIC, including a degree-5 baseline.

    Fit coefficients AND initial position/velocity on the first 70%. Pendulum
    and drag coefficients are constrained to restoring/dissipative signs.
    Coefficients multiply the named terms directly (e.g. velocity = -gamma).
    """
    t, y = _validate(times, observations)
    if regime == "bouncing_ball":
        return _fit_bounces(t, y, noise_scale)
    regime = {"drag_fall": "drag", "vertical_projectile": "drag"}.get(regime, regime)
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
        offset = regime == "pendulum"
        # Correlated position/sine terms in the joint sparse proposal can
        # cancel, leaving a physical candidate near zero frequency. Seed each
        # candidate's shooting problem from its own training-only regression.
        seed, _, _ = _proposal(t[:split], y[:split], active)
        initial = [min(seed[term], -1e-3) for term in active] + [x0, v0]
        if offset:
            initial.append(0.0)
        def unpack(parameters):
            return parameters[:len(active)], parameters[len(active):len(active)+2], (parameters[-1] if offset else 0.0)
        def residual(parameters):
            try:
                coefficients, state, eq = unpack(parameters)
                predicted = integrate(t[:split], active, coefficients, state, eq)
                return predicted - y[:split]
            except (ValueError, FloatingPointError):
                return np.full(split, 1e10)
        lower = [-np.inf] * (len(active) + 2) + ([-0.2] if offset else [])
        upper = [0.0] * len(active) + [np.inf, np.inf] + ([0.2] if offset else [])
        fit = least_squares(
            residual, initial, loss="soft_l1", f_scale=noise_scale,
            bounds=(lower, upper), max_nfev=500,
        )
        if not fit.success:
            raise RuntimeError(f"Shooting failed for {name}: {fit.message}")
        coefficients, state, eq = unpack(fit.x)
        prediction = integrate(t, active, coefficients, state, eq)
        fitted = dict(zip(active, coefficients))
        if offset:
            fitted["theta_eq"] = float(eq)
        results.append(_score(name, fitted, tuple(state), prediction, y, split, len(fit.x), proposal))
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


def _bounce_states(times, gravity, restitution, floor, initial_state, *, events=True):
    """Ballistic flights; shooting uses terminal downward solve_ivp events.

    The analytic variant has identical impact resets and serves interactive
    counterfactuals without numerical integration overhead.
    """
    t = np.asarray(times, dtype=float)
    out = np.empty((len(t), 2))
    start, end = float(t[0]), float(t[-1])
    height, velocity = map(float, initial_state)
    if height < floor - 1e-7:
        raise ValueError("Initial height is below the floor")
    height = max(height, floor)
    if height == floor and velocity < 0:
        velocity = -restitution * velocity
    for _ in range(1000):
        speed = np.sqrt(max(0, velocity**2 + 2 * gravity * (height - floor)))
        flight = (velocity + speed) / gravity
        if flight < 1e-8:
            out[t >= start] = (floor, 0)
            break
        impact = start + flight
        stop = min(impact, end)
        selected = (t >= start) & (t <= stop + 1e-12)
        dt = t[selected] - start
        if events and stop > start:
            def crossing(_time, state):
                return state[0] - floor
            crossing.terminal = True
            crossing.direction = -1
            solution = solve_ivp(lambda _time, s: [s[1], -gravity],
                                 (start, min(end, impact + 1e-7)), [height, velocity],
                                 events=crossing, dense_output=True, method="RK45",
                                 rtol=1e-9, atol=1e-9)
            if not solution.success:
                raise ValueError("Ballistic integration failed")
            if np.any(selected):
                out[selected] = solution.sol(t[selected]).T
            if solution.t_events[0].size:
                impact = float(solution.t_events[0][0])
                speed = -float(solution.y_events[0][0, 1])
        else:
            out[selected, 0] = height + velocity * dt - 0.5 * gravity * dt**2
            out[selected, 1] = velocity - gravity * dt
        if impact >= end:
            break
        start, height, velocity = impact, floor, restitution * speed
    else:
        raise ValueError("Too many impacts")
    return out


def _fit_bounces(t, y, noise_scale):
    if not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("Invalid noise_scale")
    split = int(0.7 * len(t))
    tt, yy = t[:split], y[:split]
    minima, _ = find_peaks(-yy, prominence=max(np.ptp(yy) * 0.04, noise_scale * 2), distance=3)
    if not len(minima):
        raise ValueError("Training observations must contain a bounce to identify restitution")
    # Local parabolas between minima seed gravity and impact velocity ratios.
    segments = []
    edges = [0, *minima, split - 1]
    for left, right in zip(edges[:-1], edges[1:]):
        indices = np.arange(left + (left != 0), right)
        if len(indices) >= 4:
            segments.append((left, right, np.polyfit(tt[indices], yy[indices], 2)))
    if not segments:
        raise ValueError("Insufficient ballistic samples")
    gravity = max(1e-3, float(np.median([-2 * p[2][0] for p in segments])))
    first = segments[0][2]
    floor = float(yy[minima].min() - gravity * np.median(np.diff(tt))**2)
    ratios = []
    for before, after in zip(segments[:-1], segments[1:]):
        impact_t = tt[before[1]]
        pre = np.polyval(np.polyder(before[2]), impact_t)
        post = np.polyval(np.polyder(after[2]), impact_t)
        if pre < 0 < post:
            ratios.append(-post / pre)
    e0 = float(np.clip(np.median(ratios) if ratios else 0.8, 0.1, 0.99))
    results = []
    for name, fixed_e in (("elastic_bounces", 1.0), ("inelastic_bounces", None)):
        def unpack(p):
            g, f, offset, v = p[:4]
            return g, (p[4] if fixed_e is None else fixed_e), f, (f + offset, v)
        def residual(p):
            g, e, f, state = unpack(p)
            return _bounce_states(tt, g, e, f, state)[:, 0] - yy
        seed = [gravity, floor, max(1e-3, np.polyval(first, 0) - floor), first[1]]
        lower, upper = [1e-6, -np.inf, 0, -np.inf], [np.inf] * 4
        if fixed_e is None:
            seed.append(e0)
            lower.append(0.01)
            upper.append(1.0)
        fit = least_squares(residual, seed, bounds=(lower, upper), loss="soft_l1",
                            f_scale=noise_scale, x_scale="jac", max_nfev=500)
        if not fit.success:
            raise RuntimeError(f"Shooting failed for {name}")
        g, e, f, state = unpack(fit.x)
        prediction = _bounce_states(t, g, e, f, state)[:, 0]
        results.append(_score(name, {"g_pixels": g, "e": e, "floor": f},
                              tuple(state), prediction, y, split, len(fit.x), {}))
    return sorted(results, key=lambda result: result.bic)


def simulate_counterfactual(regime, params, initial_state, times):
    """Return Nx2 (position, velocity) in radians or Cartesian pixels.

    g_scale multiplies the fitted acceleration. Pendulum omega0_squared is
    g/L; drag uses g_pixels (or g) and c1/c2. Bounces use a Cartesian floor.
    Fixed-step RK4 is for fast slider previews; fitting still uses RK45.
    """
    t = np.asarray(times, dtype=float)
    state = np.asarray(initial_state, dtype=float)
    if (t.ndim != 1 or not len(t) or not np.all(np.isfinite(t))
            or np.any(np.diff(t) <= 0) or state.shape != (2,)
            or not np.all(np.isfinite(state))
            or not all(np.isfinite(v) for v in params.values())):
        raise ValueError("Invalid simulation input")
    scale = params.get("g_scale", 1.0)
    if scale <= 0:
        raise ValueError("g_scale must be positive")
    if regime == "bouncing_ball":
        gravity = params.get("g_pixels", params.get("g", 9.81)) * scale
        e = params.get("e", 0.8)
        if gravity <= 0 or not 0 <= e <= 1:
            raise ValueError("Invalid gravity or restitution")
        return _bounce_states(t, gravity, e, params.get("floor", 0), state, events=False)
    if regime == "pendulum":
        if params.get("L", 0.5) <= 0:
            raise ValueError("Pendulum length must be positive")
        frequency = params.get("omega0_squared", params.get("g", 9.81) / params.get("L", 0.5)) * scale
        gamma = params.get("gamma", -params.get("velocity", 0))
        if frequency <= 0 or gamma < 0:
            raise ValueError("Invalid pendulum parameters")
        linear = params.get("linear", 0)
        def acceleration(x, v):
            displacement = x - params.get("theta_eq", 0.0)
            return -frequency * (displacement if linear else np.sin(displacement)) - gamma * v
    elif regime in ("drag", "drag_fall", "vertical_projectile"):
        gravity = params.get("g_pixels", params.get("g", -params.get("constant", -9.81))) * scale
        c1 = params.get("c1", -params.get("velocity", 0))
        c2 = params.get("c2", -params.get("quadratic_drag", 0))
        if gravity <= 0 or c1 < 0 or c2 < 0:
            raise ValueError("Invalid drag parameters")
        def acceleration(x, v):
            return -gravity - c1 * v - c2 * abs(v) * v
    else:
        raise ValueError("Unknown regime")
    out = np.empty((len(t), 2))
    out[0] = state
    x, v = state
    for i, interval in enumerate(np.diff(t), 1):
        steps = max(1, int(np.ceil(interval * 120)))
        h = interval / steps
        for _ in range(steps):
            a = acceleration(x, v)
            vb = v + h * a / 2
            b = acceleration(x + h * v / 2, vb)
            vc = v + h * b / 2
            c = acceleration(x + h * vb / 2, vc)
            vd = v + h * c
            d = acceleration(x + h * vc, vd)
            x += h * (v + 2 * vb + 2 * vc + vd) / 6
            v += h * (a + 2 * b + 2 * c + d) / 6
        out[i] = x, v
    if not np.all(np.isfinite(out)):
        raise ValueError("Simulation diverged")
    return out


def calibrate_physical_metrics(analysis, known_object_width_cm=None, known_length_cm=None):
    """Constant-time scale conversion; never fit or invoke a model here.

    ODE length uses reference gravity for the independent ruler comparison.
    Width or a supplied string length can instead independently set SI scale.
    """
    for value in (known_object_width_cm, known_length_cm):
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError("Calibration dimensions must be positive and finite")
    regime = analysis["regime"]
    params, geometry = analysis["discovered_params"], analysis["pivot_or_bounds"]
    # Preserve only pixel measurements; discard earlier SI calibration results.
    stored = analysis.get("inferred_physical_metrics", {})
    keys = ("median_box_width_px", "median_box_height_px", "median_area_px2", "equiv_diameter_px")
    metrics = {k: stored[k] for k in keys if k in stored}
    scale = None
    if known_object_width_cm is not None:
        width = metrics.get("median_box_width_px", 0)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("Analysis lacks mask box width; re-analyze the clip")
        scale = known_object_width_cm / 100 / width
        metrics["known_object_width_cm"] = known_object_width_cm
    if regime == "pendulum":
        omega2 = params["omega0_squared"]
        if omega2 <= 0 or geometry["radius"] <= 0:
            raise ValueError("Pendulum frequency and arc radius must be positive")
        inferred = 9.81 / omega2
        metrics.update(L_meters=inferred, L_ode_inferred_m=inferred)
        if known_length_cm is not None:
            length = known_length_cm / 100
            metrics.update(known_length_cm=known_length_cm,
                           ruler_length_error_percent=100 * abs(inferred - length) / length)
            if scale is None:
                scale = length / geometry["radius"]
        if scale is not None:
            measured = geometry["radius"] * scale
            metrics.update(L_measured_m=measured, L_meters=measured,
                           g_measured_m_s2=omega2 * measured)
    elif regime in ("drag_fall", "bouncing_ball"):
        if params["g_pixels"] <= 0:
            raise ValueError("Pixel gravity must be positive")
        if scale is not None:
            metrics["g_measured_m_s2"] = params["g_pixels"] * scale
        elif regime == "drag_fall":
            scale = 9.81 / params["g_pixels"]
        if regime == "bouncing_ball":
            metrics.update(g_pixels=params["g_pixels"], restitution_e=params["e"])
    else:
        raise ValueError("Unknown regime")
    if scale is not None:
        metrics.update(meters_per_pixel=scale, pixels_per_meter=1 / scale)
        if regime == "drag_fall":
            metrics.update(c1_per_s=params.get("c1", 0), c2_per_m=params.get("c2", 0) / scale)
        if regime == "bouncing_ball":
            metrics["floor_m"] = params["floor"] * scale
    if "g_measured_m_s2" in metrics:
        metrics["g_error_percent"] = 100 * abs(metrics["g_measured_m_s2"] - 9.81) / 9.81
    elif regime != "bouncing_ball":
        metrics["calibration_gravity_m_s2"] = 9.81
    return metrics
