"""Gate 5: independent calibration, phone normalization and portable exports."""
from dataclasses import replace
import csv
import io
import json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend.app import app, calibrate, CalibrationRequest
from backend.kinematics import (MaskMoments, interpolate_trajectory, select_primary_track,
                                mask_size_metrics, normalize_video, is_normalized_video)
from backend.solver import calibrate_physical_metrics, simulate_counterfactual


def analysis(regime):
    params = {
        'pendulum': {'omega0_squared': 19.62, 'gamma': .25, 'linear': 0},
        'drag_fall': {'g_pixels': 981, 'c1': .2, 'c2': .003},
        'bouncing_ball': {'g_pixels': 981, 'e': .8, 'floor': 10},
    }[regime]
    state = [.9, 0] if regime == 'pendulum' else [210, 0]
    times = np.arange(90) / 30
    positions = simulate_counterfactual(regime, params, state, times)[:, 0]
    return {'regime': regime, 'discovered_params': params,
            'initial_state': state, 'pivot_or_bounds': {'radius': 50, 'frame_height': 720},
            'inferred_physical_metrics': {'median_box_width_px': 20, 'median_box_height_px': 18,
                                          'median_area_px2': 300, 'equiv_diameter_px': 2*np.sqrt(300/np.pi)},
            'trajectory_points': [dict(frame_idx=i, t=float(t), x_px=10.25, y_px=float(y),
                                       pred_x_px=10.25, pred_y_px=float(y), is_heldout=i >= 63)
                                  for i, (t, y) in enumerate(zip(times, positions))]}


def test_spike_rejection_track_selection_and_mask_sizes():
    mask = MaskMoments('moving', 0, (10, 20, 29, 37), np.ones((18, 20), np.uint8),
                       300, 0., 50., None, 0, 0, 0)
    masks = [replace(mask, frame_index=i, xc=float(i), yc=50.) for i in range(30)]
    masks[12] = replace(masks[12], xc=800., yc=600., box=(0, 0, 999, 999), area=1000000)
    masks += [replace(mask, object_id='stationary', frame_index=i, xc=100.) for i in range(30)]
    tracks = interpolate_trajectory(masks, 30, 720)
    track = select_primary_track(tracks)
    assert track.object_id == 'moving'
    assert not track.observed[12]
    np.testing.assert_allclose(track.x, np.arange(30), atol=1e-10)
    metrics = mask_size_metrics(masks, track)
    assert metrics['median_box_width_px'] == 20
    assert metrics['median_box_height_px'] == 18
    assert metrics['equiv_diameter_px'] == pytest.approx(2*np.sqrt(300/np.pi))


@pytest.mark.parametrize('regime', ['pendulum', 'drag_fall', 'bouncing_ball'])
def test_independent_gravity_math_and_scale_reset(regime):
    data = analysis(regime)
    result = calibrate_physical_metrics(data, known_object_width_cm=20, known_length_cm=50)
    assert result['meters_per_pixel'] == .01
    assert result['g_measured_m_s2'] == pytest.approx(9.81)
    assert result['g_error_percent'] == pytest.approx(0, abs=1e-12)
    assert 'calibration_gravity_m_s2' not in result
    if regime == 'pendulum':
        assert result['L_measured_m'] == .5
        assert result['ruler_length_error_percent'] == 0
        doubled = calibrate_physical_metrics(data, 40, 100)
        assert doubled['g_measured_m_s2'] == pytest.approx(19.62)
        assert doubled['ruler_length_error_percent'] == pytest.approx(50)
    elif regime == 'drag_fall':
        assert result['c2_per_m'] == pytest.approx(.3)
        assert result['c1_per_s'] == .2
    else:
        assert result['floor_m'] == .1
        assert result['restitution_e'] == .8
    reset = calibrate_physical_metrics({**data, 'inferred_physical_metrics': result})
    assert 'g_measured_m_s2' not in reset
    assert 'known_object_width_cm' not in reset


def test_ruler_only_independent_length_and_width_precedence():
    data = analysis('pendulum')
    result = calibrate_physical_metrics(data, known_length_cm=100)
    assert result['g_measured_m_s2'] == 19.62
    assert result['L_measured_m'] == 1
    result = calibrate_physical_metrics(data, 20, 100)
    assert result['L_measured_m'] == .5
    assert result['ruler_length_error_percent'] == 50


@pytest.mark.parametrize('regime', ['pendulum', 'drag_fall', 'bouncing_ball'])
def test_calibration_endpoint_without_models_and_fast_hot_path(regime):
    client = TestClient(app)
    body = {'analysis': analysis(regime), 'known_object_width_cm': 20}
    response = client.post('/api/calibrate', json=body)
    assert response.status_code == 200, response.text
    assert response.json()['inferred_physical_metrics']['g_measured_m_s2'] == pytest.approx(9.81)
    # Time only the endpoint calculation; HTTP scheduling/network is not <5ms guaranteed.
    parsed = CalibrationRequest.model_validate(body)
    durations = []
    for _ in range(20):
        start = perf_counter()
        calibrate(parsed)
        durations.append(perf_counter() - start)
    assert np.median(durations) < .005
    for bad in (0, -1, 'nan', 'inf'):
        assert client.post('/api/calibrate', json={**body, 'known_object_width_cm': bad}).status_code == 422
    incomplete = {**body, 'analysis': {**body['analysis'], 'inferred_physical_metrics': {}}}
    assert client.post('/api/calibrate', json=incomplete).status_code == 422


@pytest.mark.parametrize('regime', ['pendulum', 'drag_fall', 'bouncing_ball'])
def test_exports_are_runnable_and_preserve_discovered_dynamics(regime):
    client = TestClient(app)
    data = analysis(regime)
    data['inferred_physical_metrics'] = calibrate_physical_metrics(data, 20)
    response = client.post('/api/export/python', json=data)
    assert response.status_code == 200, response.text
    assert f'physisight_sim_{regime}.py' in response.headers['content-disposition']
    scope = {'__name__': 'export_test'}
    exec(compile(response.text, 'exported_sim.py', 'exec'), scope)
    exported = scope['simulate']()
    times = [p['t'] for p in data['trajectory_points']]
    reference = simulate_counterfactual(regime, data['discovered_params'], data['initial_state'], times)
    if regime != 'pendulum':
        reference *= .01
    np.testing.assert_allclose(exported, reference, atol=2e-5)
    response = client.post('/api/export/mujoco', json=data)
    assert response.status_code == 200, response.text
    root = ET.fromstring(response.text)
    assert root.tag == 'mujoco'
    assert root.find('option').get('gravity') == '0 0 -9.81'
    joint = root.find('.//joint')
    assert float(joint.get('damping')) == pytest.approx(.25*.5**2 if regime == 'pendulum' else data['discovered_params'].get('c1', 0))
    response = client.post('/api/export/csv', json=data)
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert list(rows[0]) == ['frame_idx', 't_sec', 'x_px', 'y_px', 'pred_x_px', 'pred_y_px', 'is_heldout']
    assert len(rows) == 90
    assert float(rows[0]['x_px']) == 10.25
    assert sum(row['is_heldout'] == 'True' for row in rows) == 27


def test_normalization_command_and_error_handling(tmp_path, monkeypatch):
    import backend.kinematics as kinematics
    calls = []
    monkeypatch.setattr(kinematics.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    source, target = tmp_path/'phone.mp4', tmp_path/'normalized.mp4'
    assert normalize_video(source, target) == target
    cmd = calls[0]
    assert cmd[cmd.index('-vf')+1] == 'fps=30,scale=-2:720'
    assert cmd[cmd.index('-t')+1] == '5.0'
    assert cmd[cmd.index('-c:v')+1] == 'libx264'
    assert '-an' in cmd
    with pytest.raises(ValueError):
        normalize_video(source, source)


def test_real_ffmpeg_normalization(tmp_path):
    # Exercise the actual ffmpeg/ffprobe contract on an existing clip.
    source = Path(__file__).resolve().parents[1] / 'backend/demo_clips/damped_pendulum.mp4'
    target = normalize_video(source, tmp_path / 'phone_normalized.mp4')
    assert is_normalized_video(target)


def test_analysis_routes_forward_optional_dimensions(monkeypatch):
    from unittest.mock import AsyncMock
    import backend.app as api
    mocked = AsyncMock(return_value={'regime': 'pendulum'})
    monkeypatch.setattr(api, 'analyze_video', mocked)
    client = TestClient(app)
    query = '?known_object_width_cm=4&known_length_cm=50'
    assert client.post('/api/analyze/demo/damped_pendulum.mp4'+query).status_code == 200
    assert mocked.call_args.kwargs['known_object_width_cm'] == 4
    assert mocked.call_args.kwargs['known_length_cm'] == 50
    assert client.post('/api/analyze/upload'+query,
                       files={'file': ('phone.mp4', b'placeholder', 'video/mp4')}).status_code == 200
    assert mocked.call_args.kwargs['known_object_width_cm'] == 4
    assert mocked.call_args.kwargs['known_length_cm'] == 50
    for value in ('0', '-2', 'nan', 'inf'):
        assert client.post('/api/analyze/demo/damped_pendulum.mp4?known_object_width_cm='+value).status_code == 422


def test_agent_normalizes_phone_video_before_analysis_and_keeps_playback(tmp_path, monkeypatch):
    import asyncio
    import shutil
    from unittest.mock import AsyncMock
    import backend.agent as agent
    from backend.cache import DiskCache
    source = Path(__file__).resolve().parents[1] / 'backend/demo_clips/damped_pendulum.mp4'
    phone = tmp_path/'phone.mp4'
    shutil.copyfile(source, phone)
    paths = []
    async def fake_analysis(path, **kwargs):
        paths.append(path)
        metadata, _ = agent._video_info(path)
        assert metadata['fps'] == 30
        assert metadata['height'] == 720
        assert metadata['frames'] <= 150
        return {**analysis('pendulum'), 'video_metadata': metadata}
    mocked = AsyncMock(side_effect=fake_analysis)
    monkeypatch.setattr(agent, '_analyze_video', mocked)
    cache = DiskCache(tmp_path/'cache')
    async def run():
        first = await agent.analyze_video(phone, cache=cache, known_object_width_cm=20)
        second = await agent.analyze_video(phone, cache=cache)
        assert first['video_url'].startswith('/analysis_videos/')
        assert first['video_url'] == second['video_url']
        assert first['inferred_physical_metrics']['g_measured_m_s2'] == pytest.approx(9.81)
        assert 'g_measured_m_s2' not in second['inferred_physical_metrics']
        assert paths[0] == paths[1] and paths[0].exists()
    asyncio.run(run())
