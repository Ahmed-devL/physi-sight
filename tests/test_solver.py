import numpy as np
import pytest

from backend.solver import fit_trajectory, integrate, simulate_pendulum


def test_gate1_nonlinear_pendulum():
    times, clean, observed = simulate_pendulum()
    assert len(times) == 120
    assert np.std(observed - clean) == pytest.approx(0.015, rel=0.25)
    results = fit_trajectory(times, observed)
    best = results[0]
    assert best.name == "nonlinear_pendulum"
    assert abs(best.omega0_squared / 19.62 - 1) < 0.03
    assert abs(best.gamma / 0.25 - 1) < 0.03
    assert best.heldout_r2 > 0.95
    assert best.train_size == 84
    assert best.bic < next(r.bic for r in results if r.name == "polynomial_degree_5")
    assert best.sparse_proposal


def test_holdout_never_changes_fit():
    t, _, observed = simulate_pendulum()
    original = {r.name: r for r in fit_trajectory(t, observed)}
    observed[84:] += 100
    changed = {r.name: r for r in fit_trajectory(t, observed)}
    for name, before in original.items():
        after = changed[name]
        assert before.coefficients == after.coefficients
        assert before.bic == after.bic
        np.testing.assert_array_equal(before.prediction, after.prediction)
        assert after.heldout_rmse == pytest.approx(
            np.sqrt(np.mean((before.prediction[84:] - observed[84:]) ** 2))
        )
        assert after.heldout_rmse != before.heldout_rmse


@pytest.mark.parametrize("term,name", [("velocity", "linear_drag"), ("quadratic_drag", "quadratic_drag")])
def test_drag_recovery(term, name):
    t = np.arange(120) / 30
    clean = integrate(t, ("constant", term), (-9.81, -0.25), (12, 0))
    y = clean + np.random.default_rng(42).normal(0, 0.005, len(t))
    best = fit_trajectory(t, y, regime="drag", noise_scale=0.005)[0]
    assert best.name == name
    assert best.coefficients["constant"] == pytest.approx(-9.81, rel=0.03)
    assert best.coefficients[term] == pytest.approx(-0.25, rel=0.03)
    assert best.heldout_r2 > 0.95


def test_irregular_times_and_invalid_input():
    t = np.cumsum(np.random.default_rng(1).uniform(0.025, 0.04, 120))
    t -= t[0]
    y = integrate(t, ("position", "velocity"), (-4, -0.2), (1, 0))
    best = fit_trajectory(t, y, regime="spring")[0]
    assert best.coefficients["position"] == pytest.approx(-4, rel=0.01)
    with pytest.raises(ValueError):
        fit_trajectory(t[::-1], y)
    with pytest.raises(ValueError):
        fit_trajectory(t, np.full(120, np.nan))
