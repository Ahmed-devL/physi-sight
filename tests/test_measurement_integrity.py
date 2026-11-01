import numpy as np
import pytest
from types import SimpleNamespace
from scipy.optimize import least_squares

from backend.solver import (fit_trajectory, simulate_pendulum, local_uncertainty,
                            calibrate_physical_metrics, quantity_provenance, scale_warnings)
from backend.agent import fit_observations
from backend.kinematics import MaskMoments


def test_shooting_uncertainty_accounts_for_nuisance_and_robust_loss():
    t, _, y = simulate_pendulum()
    best = fit_trajectory(t, y)[0]
    diagnostic = best.uncertainty
    assert diagnostic['method'] == 'approximate_local_soft_l1_sandwich'
    assert diagnostic['parameter_count_including_nuisance'] == 5
    assert diagnostic['rank'] == 5
    assert diagnostic['confidence_interval'] is None
    assert diagnostic['parameters']['initial_velocity']['nuisance']
    assert diagnostic['parameters']['theta_eq']['nuisance']
    assert diagnostic['parameters']['sin_position']['standard_error'] > 0
    assert diagnostic['parameters']['velocity']['standard_error'] > 0
    assert 'Not a guaranteed 95%' in diagnostic['limitations']
    # More observational noise should not imply more precise coefficients.
    _, _, noisy = simulate_pendulum(noise_sigma=.05)
    noisier = next(r for r in fit_trajectory(t, noisy) if r.name == best.name)
    assert noisier.uncertainty['parameters']['sin_position']['standard_error'] > diagnostic['parameters']['sin_position']['standard_error']


def test_rank_deficiency_never_manufactures_intervals():
    x = np.linspace(0, 1, 40)
    fit = least_squares(lambda p: p[0] + p[1] - x, [0., 0.], loss='soft_l1', f_scale=.1)
    result = local_uncertainty(fit, ['coefficient', 'initial_position'], .1)
    assert result['status'] == 'rank_deficient'
    assert all(p['standard_error'] is None for p in result['parameters'].values())


def test_active_physical_bound_suppresses_symmetric_uncertainty():
    x = np.linspace(0, 1, 40)
    fit = least_squares(lambda p: p[0]*x + p[1] - (-2*x+1), [.1, 0],
                        bounds=([0, -np.inf], [np.inf, np.inf]), loss='soft_l1', f_scale=.1)
    result = local_uncertainty(fit, ['drag', 'initial_position'], .1)
    assert result['status'] == 'boundary_limited'
    assert result['parameters']['drag']['at_bound']
    assert all(p['standard_error'] is None for p in result['parameters'].values())


def test_ill_conditioned_parameters_are_unavailable():
    x = np.linspace(0, 1, 40)
    fit = SimpleNamespace(jac=np.column_stack((np.ones(40), 1 + 1e-8*x)),
                          x=np.array([1., 1.]), active_mask=np.zeros(2), fun=.01*np.sin(x))
    result = local_uncertainty(fit, ['coefficient', 'initial_position'], .1)
    assert result['status'] == 'ill_conditioned'
    assert result['parameters']['coefficient']['standard_error'] is None


def test_exact_fit_does_not_claim_zero_measurement_uncertainty():
    x = np.linspace(0, 1, 40)
    fit = SimpleNamespace(jac=np.column_stack((np.ones(40), x)), x=np.array([1., 1.]),
                          active_mask=np.zeros(2), fun=np.zeros(40))
    result = local_uncertainty(fit, ['coefficient', 'initial_position'], .1)
    assert result['status'] == 'insufficient_residual_variation'
    assert all(p['standard_error'] is None for p in result['parameters'].values())


def test_short_arc_and_sparse_tracking_are_explicit():
    t, clean, _ = simulate_pendulum(theta0=.12)
    keep = np.arange(0, 120, 2)
    masks = [MaskMoments('bob', int(i), (0, 0, 0, 0), np.ones((1, 1), np.uint8),
                         1, float(320+180*np.sin(clean[i])), float(100+180*np.cos(clean[i])),
                         None, 0, 0, 0) for i in keep]
    data = fit_observations(masks, {'fps': 30, 'frames': 120, 'height': 720,
                                   'timestamps': t.tolist()}, 'pendulum')
    codes = {w['code'] for w in data['scientific_warnings']}
    assert 'short_pendulum_arc' in codes
    assert 'insufficient_tracking_coverage' in codes
    assert 'ambiguous_physical_scale' in codes
    assert data['quantity_provenance']['omega0_squared']['kind'] == 'fitted'
    assert data['quantity_provenance']['L_meters']['kind'] == 'gravity_derived'


@pytest.mark.parametrize('regime', ['pendulum', 'drag_fall', 'bouncing_ball'])
def test_independent_calibration_provenance_and_scale_warning(regime):
    params = {'pendulum': {'omega0_squared': 19.62, 'gamma': .25},
              'drag_fall': {'g_pixels': 981, 'c1': .2, 'c2': .003},
              'bouncing_ball': {'g_pixels': 981, 'e': .8, 'floor': 10}}[regime]
    data = {'regime': regime, 'discovered_params': params, 'pivot_or_bounds': {'radius': 50},
            'inferred_physical_metrics': {'median_box_width_px': 20}}
    data['inferred_physical_metrics'] = calibrate_physical_metrics(data)
    assert any(w['code'] == 'ambiguous_physical_scale' for w in scale_warnings(data))
    data['inferred_physical_metrics'] = calibrate_physical_metrics(data, known_object_width_cm=20)
    provenance = quantity_provenance(data, known_object_width_cm=20)
    assert provenance['g_measured_m_s2']['kind'] == 'independently_calibrated'
    assert provenance['median_box_width_px']['kind'] == 'segmentation_measured'
    assert not scale_warnings(data)
    if regime == 'pendulum':
        assert provenance['L_ode_inferred_m']['kind'] == 'gravity_derived'
