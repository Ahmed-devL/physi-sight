"""Sparse proposals followed by robust trajectory shooting on raw observations.

Scalar coordinates must have consistent units; angles are radians and time is
seconds. Cartesian y is positive upward. Gravity-derived calibration does not
constitute an independent gravity measurement. Fits never use held-out samples.
"""
from dataclasses import dataclass, field, replace

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
# Optional hypotheses never change the original deterministic candidate library.
OPTIONAL_CANDIDATES = {"drag": {
    "mixed_drag": ("constant", "velocity", "quadratic_drag"),
}}


def candidate_library(regime):
    regime = {"drag_fall": "drag", "vertical_projectile": "drag"}.get(regime, regime)
    if regime == "bouncing_ball":
        return {"elastic_bounces": (), "inelastic_bounces": ()}
    return {**CANDIDATES.get(regime, {}), **OPTIONAL_CANDIDATES.get(regime, {})}


def term_bounds(regime, term):
    """Bounds are acceleration-term specific, in upward Cartesian coordinates."""
    if term == "constant" and regime == "drag":
        return -np.inf, -1e-9  # strictly downward gravity
    if term in ("velocity", "quadratic_drag"):
        return -np.inf, 0.0  # dissipative; physical c1/c2 are their negatives
    if term in ("position", "sin_position") and regime in ("pendulum", "spring"):
        return -np.inf, -1e-9  # restoring
    raise ValueError("Term is not physical in this regime")


def valid_sparse_structure(regime, proposal, t, y):
    """Accept only identifiable subsets of the regime's physical library.

    In particular a pendulum cannot combine position and sin(position), and
    drag must include downward gravity. Derivatives only screen structures.
    """
    active = tuple(k for k, v in proposal.items() if abs(v) > 1e-10)
    if not active or not all(np.isfinite(v) for v in proposal.values()):
        return ()
    if not any(set(active) == set(v) for v in candidate_library(regime).values()):
        return ()
    for term in active:
        lo, hi = term_bounds(regime, term)
        if not lo <= proposal[term] <= hi:
            return ()
    velocity = np.gradient(y, t)
    design = np.column_stack([np.broadcast_to(TERMS[k](y, velocity), y.shape) for k in active])
    norms = np.linalg.norm(design, axis=0)
    if np.any(norms == 0) or np.linalg.cond(design / norms) > 1e6:
        return ()
    return active


@dataclass(frozen=True)
class FitResult:
    name: str
    coefficients: dict[str, float]
    initial_state: tuple[float, float]
    prediction: np.ndarray
    train_size: int
    bic: float
    heldout_r2: float | None
    heldout_rmse: float | None
    sparse_proposal: dict[str, float]
    release_index: int = 0
    locally_identifiable: bool = True
    uncertainty: dict = field(default_factory=dict)

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


def local_uncertainty(fit, names, noise_scale):
    """Approximate soft_l1 sandwich covariance, conditional on the chosen law.

    SciPy returns a robust-loss-adjusted Jacobian. Undo that row weighting
    for the score variance, retain it for the curvature, and include ALL
    fitted initial conditions/offsets as nuisance parameters. Never use a
    pseudoinverse to manufacture intervals for a deficient problem.
    This local approximation excludes geometry, calibration, correlated
    tracking error, and model-selection uncertainty; it is not a 95% CI.
    """
    jac = np.asarray(fit.jac, dtype=float)
    n, p = jac.shape
    norms = np.linalg.norm(jac, axis=0)
    valid = np.all(np.isfinite(jac)) and np.all(norms > 0)
    scaled = jac / norms if valid else np.zeros_like(jac)
    rank = int(np.linalg.matrix_rank(scaled)) if valid else 0
    condition = float(np.linalg.cond(scaled)) if valid else np.inf
    at_bound = np.asarray(fit.active_mask) != 0
    status = ("rank_deficient" if rank < p or n <= p else
              "ill_conditioned" if condition > 1e6 else
              "boundary_limited" if np.any(at_bound) else "approximate_local")
    if status == "approximate_local" and np.max(np.abs(fit.fun)) <= 1e-12:
        status = "insufficient_residual_variation"
    errors = None
    if status == "approximate_local":
        z = (fit.fun / noise_scale)**2
        derivative = (1 + z)**-1.5
        if np.min(derivative) < 1e-12:
            status = "robust_loss_saturated"
        else:
            # Bread: robust curvature. Meat: empirical variance of scores.
            raw = scaled / np.sqrt(derivative[:, None])
            psi = fit.fun / np.sqrt(1 + z)
            bread = scaled.T @ scaled
            meat = (raw * psi[:, None]).T @ (raw * psi[:, None])
            try:
                left = np.linalg.solve(bread, meat)
                covariance = np.linalg.solve(bread, left.T).T
                covariance *= n / (n - p)
                diagonal = np.diag(covariance) / norms**2
                if not np.all(np.isfinite(diagonal)) or np.any(diagonal < 0):
                    status = "unstable_covariance"
                else:
                    errors = np.sqrt(diagonal)
            except np.linalg.LinAlgError:
                status = "rank_deficient"
    parameters = {}
    for i, name in enumerate(names):
        error = float(errors[i]) if errors is not None else None
        # A near-zero coefficient with an appreciable error is unresolved.
        unresolved = error is not None and error > .5 * max(abs(float(fit.x[i])), 1e-12)
        parameters[name] = {"estimate": float(fit.x[i]),
                            "standard_error": None if unresolved else error,
                            "status": "weakly_identified" if unresolved else status,
                            "at_bound": bool(at_bound[i]),
                            "nuisance": name in ("initial_position", "initial_velocity", "initial_height_above_floor", "theta_eq")}
    return {"method": "approximate_local_soft_l1_sandwich", "confidence_interval": None,
            "rank": rank, "parameter_count_including_nuisance": p,
            "scaled_jacobian_condition": condition if np.isfinite(condition) else None,
            "status": status, "parameters": parameters,
            "limitations": "Conditional on tracking geometry, calibration and selected law; assumes independent residuals. Not a guaranteed 95% confidence interval."}


def _score(name, coefficients, initial, prediction, y, split, count, proposal, *, evaluate=True):
    rss = max(float(np.sum((prediction[:split] - y[:split]) ** 2)), np.finfo(float).tiny)
    bic = split * np.log(rss / split) + count * np.log(split)
    result = FitResult(name, coefficients, initial, prediction, split, float(bic),
                       None, None, proposal)
    return evaluate_fit(result, y) if evaluate else result


def evaluate_fit(result, observations):
    """Post-selection forecast evaluation; never used by fitting or filtering."""
    split = result.release_index + result.train_size
    y = np.asarray(observations)
    error = result.prediction[split:] - y[split:]
    variance = np.sum((y[split:] - y[split:].mean()) ** 2)
    # Finite constant-target convention: perfect forecast = 1, otherwise 0.
    # Keep genuinely negative R² values intact for alternative-model comparisons.
    squared_error = float(np.sum(error**2))
    r2 = float(1 - squared_error / variance) if variance > 0 else float(squared_error == 0)
    return replace(result, heldout_r2=r2, heldout_rmse=float(np.sqrt(np.mean(error**2))))


def fit_trajectory(times, observations, *, regime="pendulum", noise_scale=0.015,
                   release_index=None, candidate=None, evaluate_heldout=True, include_sparse=True):
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
                          noise_scale=noise_scale, candidate=candidate, include_sparse=include_sparse)
    results = [replace(r, prediction=np.concatenate((y[:release], r.prediction)),
                       release_index=int(release)) for r in results]
    # BIC order is fixed before compatibility diagnostics evaluate forecasts.
    return [evaluate_fit(r, y) for r in results] if evaluate_heldout else results


def _fit_active(times, observations, *, regime="pendulum", noise_scale=0.015, candidate=None,
                include_sparse=True):
    """Return candidates sorted by training BIC, including a degree-5 baseline.

    Fit coefficients AND initial position/velocity on the first 70%. Pendulum
    and drag coefficients are constrained to restoring/dissipative signs.
    Coefficients multiply the named terms directly (e.g. velocity = -gamma).
    """
    t, y = _validate(times, observations)
    if regime == "bouncing_ball":
        if candidate is not None and candidate not in candidate_library(regime):
            raise ValueError("Invalid candidate")
        return _fit_bounces(t, y, noise_scale, candidate=candidate)
    regime = {"drag_fall": "drag", "vertical_projectile": "drag"}.get(regime, regime)
    if regime not in CANDIDATES or not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("Unknown regime or invalid noise_scale")
    split = int(0.7 * len(t))
    candidates = dict(CANDIDATES[regime])
    if candidate is not None:
        library = candidate_library(regime)
        if candidate not in library:
            raise ValueError("Invalid candidate")
        candidates = {candidate: library[candidate]}
    terms = tuple(dict.fromkeys(term for terms in candidates.values() for term in terms))
    proposal, x0, v0 = _proposal(t[:split], y[:split], terms)
    sparse = valid_sparse_structure(regime, proposal, t[:split], y[:split])
    if include_sparse and candidate is None and sparse and sparse not in candidates.values():
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
        bounds = [term_bounds(regime, term) for term in active]
        initial = [np.clip(seed[term], lo if np.isfinite(lo) else -1e10, hi - 1e-3)
                   for term, (lo, hi) in zip(active, bounds)] + [x0, v0]
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
        lower = [b[0] for b in bounds] + [-np.inf] * 2 + ([-0.2] if offset else [])
        upper = [b[1] for b in bounds] + [np.inf, np.inf] + ([0.2] if offset else [])
        fit = least_squares(
            residual, initial, loss="soft_l1", f_scale=noise_scale,
            bounds=(lower, upper), max_nfev=500,
        )
        if not fit.success:
            if name.startswith("sparse_"):
                continue
            raise RuntimeError(f"Shooting failed for {name}: {fit.message}")
        norms = np.linalg.norm(fit.jac, axis=0)
        identifiable = bool(np.all(norms > 0) and np.linalg.cond(fit.jac / norms) <= 1e6)
        if name.startswith("sparse_") and not identifiable:
            continue
        coefficients, state, eq = unpack(fit.x)
        prediction = integrate(t, active, coefficients, state, eq)
        fitted = dict(zip(active, coefficients))
        if offset:
            fitted["theta_eq"] = float(eq)
        result = _score(name, fitted, tuple(state), prediction, y, split, len(fit.x), proposal, evaluate=False)
        uncertainty = local_uncertainty(fit, [*active, "initial_position", "initial_velocity",
                                            *(["theta_eq"] if offset else [])], noise_scale)
        results.append(replace(result, locally_identifiable=identifiable, uncertainty=uncertainty))
    if candidate is None:
        polynomial = np.polynomial.Polynomial.fit(t[:split], y[:split], deg=5)
        results.append(_score("polynomial_degree_5", {}, (float(polynomial(0)), float(polynomial.deriv()(0))),
                              polynomial(t), y, split, 6, proposal, evaluate=False))
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


def _fit_bounces(t, y, noise_scale, *, candidate=None):
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
        if candidate is not None and candidate != name:
            continue
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
        result = _score(name, {"g_pixels": g, "e": e, "floor": f},
                        tuple(state), prediction, y, split, len(fit.x), {}, evaluate=False)
        uncertainty = local_uncertainty(fit, ["g_pixels", "floor", "initial_height_above_floor",
                                            "initial_velocity", *(["e"] if fixed_e is None else [])], noise_scale)
        results.append(replace(result, uncertainty=uncertainty,
                               locally_identifiable=uncertainty["status"] not in ("rank_deficient", "ill_conditioned")))
    return sorted(results, key=lambda result: result.bic)


def simulate_counterfactual(regime, params, initial_state, times, *, accurate=False):
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
        return _bounce_states(t, gravity, e, params.get("floor", 0), state, events=accurate)
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
    if accurate and len(t) > 1:
        solution = solve_ivp(lambda _t, s: [s[1], acceleration(*s)],
                             (t[0], t[-1]), state, t_eval=t, method="RK45",
                             rtol=1e-10, atol=1e-12)
        if not solution.success or solution.y.shape[1] != len(t) or not np.all(np.isfinite(solution.y)):
            raise ValueError("Simulation diverged")
        return solution.y.T
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


def quantity_provenance(analysis, known_object_width_cm=None, known_length_cm=None):
    """Classify reported quantities without confusing calibration with a fit."""
    result = {key: {"kind": "fitted", "source": "raw training observations; trajectory shooting"}
              for key in analysis["discovered_params"]}
    if "linear" in result:
        result["linear"] = {"kind": "selected_structure", "source": "training BIC"}
    if analysis.get("selected_candidate") == "elastic_bounces":
        result["e"] = {"kind": "fixed_by_candidate", "source": "elastic hypothesis"}
    independent = known_object_width_cm is not None or (
        analysis["regime"] == "pendulum" and known_length_cm is not None)
    for key in analysis.get("inferred_physical_metrics", {}):
        if key in ("median_box_width_px", "median_box_height_px", "median_area_px2", "equiv_diameter_px"):
            kind, source = "segmentation_measured", "observed SAM masks; source pixels"
        elif key.startswith("known_"):
            kind, source = "independent_input", "user supplied dimension; accuracy not verified"
        elif key == "calibration_gravity_m_s2":
            kind, source = "reference_assumption", "g = 9.81 m/s² sets physical scale"
        elif key in ("g_pixels", "restitution_e", "c1_per_s"):
            kind, source = "fitted", "training trajectory; no independent length scale needed"
        elif key in ("L_ode_inferred_m", "ruler_length_error_percent") or (key == "L_meters" and not independent):
            kind, source = "gravity_derived", "L = 9.81 / omega0_squared; gravity assumed"
        elif independent:
            kind = "independently_calibrated"
            source = ("user supplied object width and observed mask width" if known_object_width_cm is not None
                      else "user supplied pendulum length and fitted arc radius")
        elif analysis["regime"] == "drag_fall":
            kind, source = "gravity_derived", "known g = 9.81 m/s² calibrates pixel scale"
        else:
            kind, source = "derived", "fitted quantities; physical scale may be ambiguous"
        result[key] = {"kind": kind, "source": source}
    return result


def scale_warnings(analysis, warnings=()):
    result = [w for w in warnings if w["code"] != "ambiguous_physical_scale"]
    metrics = analysis.get("inferred_physical_metrics", {})
    if "g_measured_m_s2" not in metrics:
        result.append({"code": "ambiguous_physical_scale", "message": (
            "Gravity is fitted in pixels; no independent SI length scale is available."
            if analysis["regime"] == "bouncing_ball" else
            "SI scale uses assumed gravity (9.81 m/s²). Gravity and length are not independently identified.")})
    return result


def mechanical_energy_diagnostics(regime, params, times, states, *, metrics=None,
                                  release_index=0, empirical_states=None):
    """Mass-specific energy of the selected law, with a fixed potential datum.

    SI conversion scales *both* energy terms by length squared. Impact loss is
    impulsive, so bounce power is zero between impacts, not a finite rate.
    Linear pendulums use their matching harmonic (small-angle) potential.
    """
    metrics = metrics or {}
    t = np.asarray(times, dtype=float)
    states = np.asarray(states, dtype=float)
    if regime == "pendulum":
        length = metrics.get("L_meters", params.get("L"))
        factor = length**2 if length is not None else 1.0
        units = "J/kg" if length is not None else "rad²/s²"
        frequency = params.get("omega0_squared", params.get("g", 9.81) / params.get("L", .5)) * params.get("g_scale", 1)
        datum = params.get("theta_eq", 0)
    else:
        scale = metrics.get("meters_per_pixel", 1.0)
        factor = scale**2
        units = "J/kg" if "meters_per_pixel" in metrics else "px²/s²"
        gravity = params.get("g_pixels", params.get("g", -params.get("constant", -9.81))) * params.get("g_scale", 1)
        datum = params.get("floor", 0) if regime == "bouncing_ball" else float(states[release_index:, 0].min())

    def series(values):
        x, v = values.T
        kinetic = .5 * v**2 * factor
        if regime == "pendulum":
            displacement = x - datum
            potential = frequency * (0.5 * displacement**2 if params.get("linear", 0) else 1 - np.cos(displacement)) * factor
            power = params.get("gamma", -params.get("velocity", 0)) * v**2 * factor
        else:
            height = np.maximum(0, x - datum) if regime == "bouncing_ball" else x - datum
            potential = gravity * height * factor
            power = (np.zeros_like(v) if regime == "bouncing_ball" else
                     (params.get("c1", -params.get("velocity", 0)) * v**2 +
                      params.get("c2", -params.get("quadratic_drag", 0)) * np.abs(v)**3) * factor)
        return [{"t": float(time), "kinetic": float(k), "potential": float(p),
                 "total": float(k + p), "dissipation_power": float(d)}
                for time, k, p, d in zip(t, kinetic, potential, power)]

    fitted = series(states)
    initial, final = fitted[release_index]["total"], fitted[-1]["total"]
    retained = 100 * final / initial if initial > np.finfo(float).eps else 100.0
    return {"series": fitted, "empirical_series": series(np.asarray(empirical_states)) if empirical_states is not None else [],
            "units": units, "energy_scale": float(factor), "power_units": "W/kg" if units == "J/kg" else f"{units}/s",
            "t_release": float(t[release_index]), "t_end": float(t[-1]),
            "energy_retained_pct": float(retained), "total_energy_dissipated_pct": float(100 - retained),
            "potential_model": "harmonic" if regime == "pendulum" and params.get("linear", 0) else "mechanical",
            "impact_loss_fraction": float(1 - params.get("e", 1)**2) if regime == "bouncing_ball" else None}


def backfill_energy_diagnostics(analysis):
    """Reconstruct smooth fitted states locally, including legacy cached fits.

    Empirical derivatives use a uniform SG grid, then map back to exact PTS.
    Pre-release observations are held; summary metrics cover free motion only.
    """
    points = analysis.get("trajectory_points", [])
    if not points:
        return analysis
    times = np.array([p["t"] for p in points])
    regime, params = analysis["regime"], analysis["discovered_params"]
    geometry = analysis["pivot_or_bounds"]
    release_time = analysis.get("release_time", geometry.get("release_time", times[0]))
    release = int(np.searchsorted(times, release_time))
    active_times = np.unique(np.r_[release_time, times[release:]])
    fitted = simulate_counterfactual(regime, params, analysis["initial_state"], active_times, accurate=True)
    states = np.tile(analysis["initial_state"], (len(times), 1)).astype(float)
    states[:release, 1] = 0
    states[release:] = fitted[np.searchsorted(active_times, times[release:])]
    y = geometry["frame_height"] - np.array([p["y_px"] for p in points])
    if regime == "pendulum":
        x = np.array([p["x_px"] for p in points])
        y = np.unwrap(np.arctan2(x - geometry.get("x0", 0), geometry.get("y0", 0) - y))
    empirical = np.column_stack((y, np.zeros(len(y))))
    # Do not smooth the release transient into free-motion derivatives.
    for start, stop in ((0, release), (release, len(times))):
        n = stop - start
        if n < 5:
            continue
        grid = np.linspace(times[start], times[stop - 1], n)
        values = np.interp(grid, times[start:stop], y[start:stop])
        window = min(15, n if n % 2 else n - 1)
        empirical[start:stop, 0] = np.interp(times[start:stop], grid, savgol_filter(values, window, 3))
        empirical[start:stop, 1] = np.interp(times[start:stop], grid, savgol_filter(values, window, 3, deriv=1, delta=grid[1] - grid[0]))
    analysis["energy_diagnostics"] = mechanical_energy_diagnostics(
        regime, params, times, states, metrics=analysis.get("inferred_physical_metrics"),
        release_index=release, empirical_states=empirical)
    return analysis
