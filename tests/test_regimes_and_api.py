"""Offline regime recovery and API integration; no model/network calls."""
import asyncio
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
import numpy as np
import pytest

from backend import agent
from backend import app as api
from backend.cache import DiskCache, video_sha256
from backend.kinematics import MaskMoments
from backend.solver import fit_trajectory, integrate, simulate_counterfactual


@pytest.mark.parametrize("regime,terms,coefficients,name", [
    ("pendulum", ("position",), (-19.62,), "simple_harmonic"),
    ("pendulum", ("sin_position", "velocity"), (-19.62, -.25), "nonlinear_pendulum"),
    ("drag_fall", ("constant",), (-9.81,), "free_fall"),
    ("drag_fall", ("constant", "velocity"), (-9.81, -.5), "linear_drag"),
    ("drag_fall", ("constant", "quadratic_drag"), (-9.81, -.25), "quadratic_drag"),
])
def test_physical_candidate_selection(regime, terms, coefficients, name):
    t = np.arange(120) / 30
    initial = (.9, 0) if regime == "pendulum" else (12, 0)
    clean = integrate(t, terms, coefficients, initial)
    observed = clean + np.random.default_rng(4).normal(0, .0005, len(t))
    results = fit_trajectory(t, observed, regime=regime, noise_scale=.0005)
    best = results[0]
    assert best.name == name
    for term, coefficient in zip(terms, coefficients):
        assert best.coefficients[term] == pytest.approx(coefficient, rel=.03)
    assert best.heldout_r2 > .99
    assert best.train_size == 84
    # Changing the held-out observations must never alter model selection.
    observed[84:] += 100
    changed = fit_trajectory(t, observed, regime=regime, noise_scale=.0005)
    assert [(r.name, r.coefficients, r.bic) for r in changed] == [
        (r.name, r.coefficients, r.bic) for r in results]


def _independent_ballistic(times, g=1569.6, e=.82, floor=59., y0=443.):
    """Independent scalar impact simulator for checking the event solver."""
    output = []
    for t in times:
        y, v, remaining = y0, 0., float(t)
        while remaining > 0:
            flight = (v + np.sqrt(v*v + 2*g*(y-floor))) / g
            if remaining <= flight:
                y += v * remaining - .5*g*remaining**2
                break
            remaining -= flight
            v = e * np.sqrt(v*v + 2*g*(y-floor))
            y = floor
        output.append(y)
    return np.array(output)


def test_bounce_recovery_and_training_only():
    t = np.arange(120) / 30
    raw = _independent_ballistic(t) + np.random.default_rng(42).normal(0, .05, len(t))
    best = fit_trajectory(t, raw, regime="bouncing_ball", noise_scale=.05)[0]
    assert best.name == "inelastic_bounces"
    assert best.coefficients["g_pixels"] == pytest.approx(1569.6, rel=.01)
    assert best.coefficients["e"] == pytest.approx(.82, rel=.01)
    assert best.heldout_r2 > .999
    changed = raw.copy()
    changed[84:] += 100
    after = fit_trajectory(t, changed, regime="bouncing_ball", noise_scale=.05)[0]
    assert best.coefficients == after.coefficients
    np.testing.assert_array_equal(best.prediction, after.prediction)
    with pytest.raises(ValueError, match="bounce"):
        fit_trajectory(t, 10000 - t*t, regime="bouncing_ball")


@pytest.mark.parametrize("regime,params,state", [
    ("pendulum", {"omega0_squared": 19.62, "gamma": .25}, [.9, 0]),
    ("drag_fall", {"g_pixels": 255.06, "c2": .025}, [467, 0]),
    ("bouncing_ball", {"g_pixels": 1569.6, "e": .82, "floor": 59}, [443, 0]),
])
def test_counterfactual_accuracy_and_speed(regime, params, state):
    times = np.arange(120) / 30
    simulation = simulate_counterfactual(regime, params, state, times)
    if regime == "pendulum":
        reference = integrate(times, ("sin_position", "velocity"), (-19.62, -.25), state)
    elif regime == "drag_fall":
        reference = integrate(times, ("constant", "quadratic_drag"), (-255.06, -.025), state)
    else:
        reference = _independent_ballistic(times)
    np.testing.assert_allclose(simulation[:, 0], reference, atol=1e-4)
    durations = []
    for _ in range(5):
        start = perf_counter()
        simulate_counterfactual(regime, params, state, times)
        durations.append(perf_counter() - start)
    assert np.median(durations) < .010
    modified = simulate_counterfactual(regime, {**params, "g_scale": .5}, state, times)
    assert np.max(np.abs(simulation[:, 0] - modified[:, 0])) > .1


def test_disk_cache_content_addressing_and_atomic_sections(tmp_path):
    cache = DiskCache(tmp_path / "cache")
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"same-video")
    digest = video_sha256(source)
    cache.update(digest, masks=[{"mask": "bits"}], analyses={"meta": {"score": 1}})
    cache.update(digest, analyses={"baseten": {"score": 2}})
    result = cache.load(digest)
    assert result["masks"] == [{"mask": "bits"}]
    assert set(result["analyses"]) == {"meta", "baseten"}
    with pytest.raises(ValueError):
        cache.load("../escape")
    (cache.directory / f"{digest}.json").write_text("partial write")
    assert cache.load(digest) == {}


def _fixture_masks(regime):
    t = np.arange(120) / 30
    if regime == "pendulum":
        theta = integrate(t, ("sin_position", "velocity"), (-19.62, -.15), (.9, 0))
        x = 256 + 280 * np.sin(theta)
        screen_y = 48 + 280 * np.cos(theta)
    elif regime == "drag_fall":
        y = integrate(t, ("constant", "quadratic_drag"), (-255.06, -.025), (467, 0))
        x, screen_y = np.full(120, 256), 512 - y
    else:
        x, screen_y = np.full(120, 256), 512 - _independent_ballistic(t)
    return [MaskMoments("1", i, (0, 0, 0, 0), np.ones((1, 1), dtype=np.uint8),
                        1, float(x[i]), float(screen_y[i]), None, 0, 0, 0)
            for i in range(120)]


@pytest.mark.parametrize("regime,clip", [
    ("pendulum", "damped_pendulum.mp4"),
    ("drag_fall", "falling_drag_sphere.mp4"),
    ("bouncing_ball", "bouncing_ball.mp4"),
])
def test_agent_pipeline_cache_force_and_provider(tmp_path, monkeypatch, regime, clip):
    cache = DiskCache(tmp_path)
    ground = AsyncMock(return_value={"regime": regime, "object_prompt": "ball"})
    track = AsyncMock(return_value=_fixture_masks(regime))
    hypotheses = AsyncMock(return_value={"candidate": {
        "pendulum": "nonlinear_pendulum", "drag_fall": "quadratic_drag",
        "bouncing_ball": "inelastic_bounces"}[regime], "rationale": "Test physical dissipation"})
    monkeypatch.setattr(agent, "ground_scene", ground)
    monkeypatch.setattr(agent, "track_video", track)
    monkeypatch.setattr(agent, "hypothesis_loop", hypotheses)
    path = api.DEMO_DIR / clip

    async def run():
        result = await agent.analyze_video(path, cache=cache)
        assert result["heldout_r2"] > .999
        assert result["train_r2"] > .999
        assert len(result["reasoning_steps"]) == 2
        assert result["background_image"].startswith("data:image/png;base64,")
        assert len(result["trajectory_points"]) == 120
        assert sum(p["is_heldout"] for p in result["trajectory_points"]) == 36
        assert result["cache_hit"] is False
        cached = await agent.analyze_video(path, cache=cache)
        assert cached["cache_hit"] is True
        assert cached["model_api_cost_usd"] == 0
        assert cached["discovered_params"] == result["discovered_params"]
        assert (ground.await_count, track.await_count, hypotheses.await_count) == (1, 1, 1)
        await agent.analyze_video(path, cache=cache, model_provider="baseten")
        assert (ground.await_count, track.await_count, hypotheses.await_count) == (1, 1, 2)
        await agent.analyze_video(path, cache=cache, force_live=True)
        assert (ground.await_count, track.await_count, hypotheses.await_count) == (2, 2, 3)
        # The public counterfactual API round-trips the inferred geometry/units.
        response = TestClient(api.app).post("/api/counterfactual", json={
            "regime": regime, "params": result["discovered_params"],
            "initial_state": result["initial_state"], "pivot_or_bounds": result["pivot_or_bounds"],
            "frames": 120, "fps": 30})
        assert response.status_code == 200, response.text
        predicted = response.json()["trajectory_points"]
        np.testing.assert_allclose([p["pred_y_px"] for p in predicted],
                                   [p["pred_y_px"] for p in result["trajectory_points"]], atol=.001)
    asyncio.run(run())


def test_demo_static_analysis_upload_and_errors(tmp_path, monkeypatch):
    client = TestClient(api.app)
    listing = client.get("/api/demos")
    assert listing.status_code == 200
    assert {"damped_pendulum.mp4", "falling_drag_sphere.mp4", "bouncing_ball.mp4"} <= {
        clip["clip_name"] for clip in listing.json()["clips"]}
    assert client.get(listing.json()["clips"][0]["url"]).status_code == 200
    captured = []
    async def fake_analyze(path, **kwargs):
        assert Path(path).exists()
        captured.append((Path(path), kwargs))
        return {"regime": "pendulum", "heldout_r2": .999996}
    mocked = AsyncMock(side_effect=fake_analyze)
    monkeypatch.setattr(api, "analyze_video", mocked)
    response = client.post("/api/analyze/demo/damped_pendulum.mp4?force_live=true&model_provider=baseten")
    assert response.status_code == 200
    assert captured[-1][1] == {"force_live": True, "model_provider": "baseten"}
    assert client.post("/api/analyze/demo/missing.mp4").status_code == 404
    assert client.post("/api/analyze/demo/damped_pendulum.mp4?model_provider=bad").status_code == 422
    response = client.post("/api/analyze/upload", files={"file": ("test.mp4", b"test", "video/mp4")})
    assert response.status_code == 200
    assert not captured[-1][0].exists()
    assert client.post("/api/analyze/upload", files={"file": ("test.txt", b"x")}).status_code == 415
    assert client.post("/api/analyze/upload", files={"file": ("test.mp4", b"")}).status_code == 422
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 1)
    assert client.post("/api/analyze/upload", files={"file": ("test.mp4", b"xx")}).status_code == 413
    mocked.side_effect = ValueError("Cannot decode MP4 video")
    assert client.post("/api/analyze/demo/damped_pendulum.mp4").status_code == 422
    mocked.side_effect = RuntimeError("missing API key")
    assert client.post("/api/analyze/demo/damped_pendulum.mp4").status_code == 503
    cors = client.options("/api/counterfactual", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert cors.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_counterfactual_validation():
    client = TestClient(api.app)
    body = {"regime": "pendulum", "params": {"omega0_squared": 19.62},
            "initial_state": [.9, 0], "pivot_or_bounds": {"x0": 256, "y0": 464,
            "radius": 280, "frame_height": 512}, "frames": 120, "fps": 30}
    response = client.post("/api/counterfactual", json=body)
    assert response.status_code == 200
    assert len(response.json()["phase_space"]) == 120
    assert response.json()["trajectory_points"][0]["pred_y_px"] == pytest.approx(48 + 280*np.cos(.9))
    for change in ({"frames": 0}, {"fps": 0}, {"regime": "bad"}, {"initial_state": [0]},
                   {"params": {"gamma": -1}}, {"pivot_or_bounds": {}}):
        assert client.post("/api/counterfactual", json={**body, **change}).status_code == 422


def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_one_hypothesis_uses_training_scores(monkeypatch):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.chat.completions.create = AsyncMock(return_value=_response(
        '{"candidate":"mixed_drag","rationale":"Systematic training residuals"}'))
    monkeypatch.setattr(agent, "_hypothesis_client", lambda p: (client, "test-model"))
    evidence = {"regime": "drag_fall", "baseline_fit": {"training_bic": 10},
                "candidate_library": agent.candidate_library("drag_fall")}
    result = asyncio.run(agent.hypothesis_loop(evidence, "meta"))
    assert result["candidate"] == "mixed_drag"
    assert client.chat.completions.create.await_count == 1
    prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
    assert '"training_bic": 10' in prompt
    assert "heldout" not in prompt


def test_grounding_fallback_and_sam_wire_contract(monkeypatch):
    import httpx
    from openai import NotFoundError
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.chat.completions.create = AsyncMock(side_effect=[
        NotFoundError("model unavailable", response=httpx.Response(404, request=httpx.Request("POST", "https://example.test")), body=None),
        _response('{"regime":"bouncing_ball","object_prompt":"ball"}'),
    ])
    stream = MagicMock()
    stream.__aenter__ = AsyncMock(return_value=stream)
    stream.__aexit__ = AsyncMock(return_value=None)
    client.responses.create = AsyncMock(return_value=stream)
    monkeypatch.setattr(agent, "_meta_client", lambda: client)
    parser = AsyncMock(return_value=[])
    monkeypatch.setattr(agent, "parse_sam_video", parser)
    async def run():
        result = await agent.ground_scene("unused", ["frame0", "frame15"])
        assert result == {"regime": "bouncing_ball", "object_prompt": "ball"}
        assert [call.kwargs["model"] for call in client.chat.completions.create.call_args_list] == [
            "muse-spark-1.3-contributor", "muse-spark-1.3"]
        await agent.track_video(api.DEMO_DIR / "bouncing_ball.mp4", "ball")
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["model"] == "sam-3.1"
        assert kwargs["stream"] is True
        assert kwargs["metadata"] == {"mask_encoding": "one_bit"}
        assert kwargs["input"][0]["content"][0]["text"] == "ball"
        parser.assert_awaited_once_with(stream)
        client.chat.completions.create.side_effect = None
        client.chat.completions.create.return_value = _response(
            '{"regime":"drag_fall","object_prompt":"segment ball and floor"}')
        with pytest.raises(ValueError, match="noun phrase"):
            await agent.ground_scene("unused", ["frame0", "frame15"])
    asyncio.run(run())


def test_baseten_configuration(monkeypatch):
    monkeypatch.setenv("BASETEN_API_KEY", "unit-test-placeholder")
    monkeypatch.setenv("BASETEN_BASE_URL", "https://example.test/v1")
    constructor = MagicMock()
    monkeypatch.setattr(agent, "AsyncOpenAI", constructor)
    client, model = agent._hypothesis_client("baseten")
    assert model == "meta-models/muse-spark-open-1026"
    assert client is constructor.return_value
    assert constructor.call_args.kwargs["base_url"] == "https://example.test/v1"


def test_pendulum_release_window_and_camera_tilt():
    times = np.arange(138) / 30
    active = times[18:] - times[18]
    clean = integrate(active, ("sin_position", "velocity"), (-19.62, -.25),
                      (.95, .025), theta_eq=.05)
    raw = np.r_[np.full(18, clean[0]), clean]
    raw += np.random.default_rng(42).normal(0, .003, len(raw))
    best = next(r for r in fit_trajectory(times, raw, noise_scale=.003)
                if not r.name.startswith(("sparse_", "polynomial")))
    assert 16 <= best.release_index <= 24
    assert best.name == "nonlinear_pendulum"
    assert best.heldout_r2 > .98
    assert best.omega0_squared == pytest.approx(19.62, rel=.03)
    assert best.gamma == pytest.approx(.25, rel=.03)
    assert best.coefficients["theta_eq"] == pytest.approx(.05, abs=.005)
    assert best.train_size == int(.7 * (len(raw) - best.release_index))
    assert len(best.prediction) == len(raw)
    np.testing.assert_array_equal(best.prediction[:best.release_index], raw[:best.release_index])
    changed = raw.copy()
    changed[best.release_index + best.train_size:] += 10
    after = next(r for r in fit_trajectory(times, changed, noise_scale=.003)
                 if r.name == best.name)
    assert best.release_index == after.release_index
    assert best.coefficients == after.coefficients
    np.testing.assert_array_equal(best.prediction[best.release_index:], after.prediction[after.release_index:])


@pytest.mark.parametrize("regime", ["drag_fall", "bouncing_ball"])
def test_stationary_hold_before_fall(regime):
    t = np.arange(138) / 30
    active = t[18:] - t[18]
    clean = (_independent_ballistic(active) if regime == "bouncing_ball" else
             integrate(active, ("constant", "quadratic_drag"), (-255.06, -.025), (467, 0)))
    raw = np.r_[np.full(18, clean[0]), clean]
    best = fit_trajectory(t, raw, regime=regime, noise_scale=.05)[0]
    assert 16 <= best.release_index <= 24
    assert best.heldout_r2 > .99


def test_upload_returns_playable_normalized_video(monkeypatch):
    async def fake_pipeline(path, **kwargs):
        assert Path(path).exists()
        return {"regime": "pendulum", "discovered_params": {"omega0_squared": 19.62},
                "pivot_or_bounds": {"radius": 280}, "inferred_physical_metrics": {}}
    monkeypatch.setattr(agent, "_analyze_video", fake_pipeline)
    client = TestClient(api.app)
    with (api.DEMO_DIR / "damped_pendulum.mp4").open("rb") as source:
        response = client.post("/api/analyze/upload", files={"file": ("custom.mp4", source, "video/mp4")})
    assert response.status_code == 200, response.text
    url = response.json()["video_url"]
    assert url.startswith("/analysis_videos/")
    assert client.get(url).status_code == 200
    assert response.json()["inferred_physical_metrics"]["L_meters"] == pytest.approx(.5)


def test_agent_release_marks_full_track_and_anchors_counterfactual():
    times = np.arange(138) / 30
    clean = integrate(times[18:] - times[18], ("sin_position", "velocity"),
                      (-19.62, -.25), (.95, .025), theta_eq=.05)
    angles = np.r_[np.full(18, clean[0]), clean]
    masks = [MaskMoments("1", i, (0, 0, 0, 0), np.ones((1, 1), dtype=np.uint8),
                         1, float(256 + 180*np.sin(theta)),
                         float(48 + 180*np.cos(theta)), None, 0, 0, 0)
             for i, theta in enumerate(angles)]
    result = agent.fit_observations(masks, {"fps": 30, "frames": 138, "height": 512,
                                           "timestamps": times.tolist()}, "pendulum")
    release = result["release_frame_idx"]
    points = result["trajectory_points"]
    assert 16 <= release <= 24
    assert len(points) == 138
    assert sum(p["is_pre_release"] for p in points) == release
    assert sum(p["is_heldout"] for p in points) == 138 - release - int(.7 * (138 - release))
    assert result["heldout_r2"] > .98
    response = TestClient(api.app).post("/api/counterfactual", json={
        "regime": "pendulum", "params": result["discovered_params"],
        "initial_state": result["initial_state"], "pivot_or_bounds": result["pivot_or_bounds"],
        "frames": 138, "fps": 30})
    assert response.status_code == 200, response.text
    predicted = response.json()["trajectory_points"]
    np.testing.assert_allclose([p["pred_y_px"] for p in predicted[release:]],
                               [p["pred_y_px"] for p in points[release:]], atol=.001)
