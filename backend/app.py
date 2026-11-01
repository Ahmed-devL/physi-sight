"""Local development API: .venv/bin/uvicorn backend.app:app --reload."""
import json
import csv
import io
import math
from pathlib import Path
import tempfile
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from openai import APIError
from pydantic import BaseModel, ConfigDict, Field

from backend.agent import analyze_video
from backend.solver import simulate_counterfactual, calibrate_physical_metrics
from backend.cache import CACHE_DIR

DEMO_DIR = Path(__file__).resolve().parent / "demo_clips"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
app = FastAPI(title="PhysiSight", description="Built with Muse Spark Open")
app.add_middleware(CORSMiddleware, allow_origins=[
    "http://localhost:3000", "http://localhost:5173",
    "http://127.0.0.1:3000", "http://127.0.0.1:5173",
], allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/demo_clips", StaticFiles(directory=DEMO_DIR), name="demo_clips")
(CACHE_DIR / "videos").mkdir(parents=True, exist_ok=True)
app.mount("/analysis_videos", StaticFiles(directory=CACHE_DIR / "videos"), name="analysis_videos")


def _manifest():
    return json.loads((DEMO_DIR / "manifest.json").read_text())


@app.get("/api/demos")
def demos():
    manifest = _manifest()
    shared = {key: value for key, value in manifest.items() if key != "clips"}
    regimes = {"damped_pendulum.mp4": "pendulum", "falling_drag_sphere.mp4": "drag_fall",
               "bouncing_ball.mp4": "bouncing_ball"}
    return {"clips": [{"clip_name": name, "url": f"/demo_clips/{name}",
                       "regime": metadata.get("regime", regimes.get(name, "pendulum")), "metadata": {**shared, **metadata}}
                      for name, metadata in manifest["clips"].items()]}


async def _analyze(path, force_live, model_provider, known_object_width_cm=None, known_length_cm=None):
    try:
        options = {"force_live": force_live, "model_provider": model_provider}
        if path.parent == DEMO_DIR:
            registered = _manifest()["clips"].get(path.name, {})
            if "regime" in registered and "object_prompt" in registered:
                options.update(regime=registered["regime"], object_prompt=registered["object_prompt"])
            if "trim_window_seconds" in registered:
                options["trim_window_seconds"] = registered["trim_window_seconds"]
        if known_object_width_cm is not None:
            options["known_object_width_cm"] = known_object_width_cm
        if known_length_cm is not None:
            options["known_length_cm"] = known_length_cm
        return await analyze_video(path, **options)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except APIError as error:
        # Do not expose provider response bodies, headers or credentials.
        raise HTTPException(502, "Model provider request failed") from error
    except RuntimeError as error:
        raise HTTPException(503, "Analysis unavailable; check model configuration and server logs") from error


@app.post("/api/analyze/demo/{clip_name}")
async def analyze_demo(clip_name: str, force_live: bool = False,
                       model_provider: Literal["meta", "baseten"] = "meta",
                       known_object_width_cm: float | None = Query(default=None, gt=0, allow_inf_nan=False),
                       known_length_cm: float | None = Query(default=None, gt=0, allow_inf_nan=False)):
    if clip_name not in _manifest()["clips"]:
        raise HTTPException(404, "Unknown demo clip")
    return await _analyze(DEMO_DIR / clip_name, force_live, model_provider, known_object_width_cm, known_length_cm)


@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...), force_live: bool = False,
                         model_provider: Literal["meta", "baseten"] = "meta",
                       known_object_width_cm: float | None = Query(default=None, gt=0, allow_inf_nan=False),
                       known_length_cm: float | None = Query(default=None, gt=0, allow_inf_nan=False)):
    temporary = None
    try:
        if not file.filename or not file.filename.lower().endswith(".mp4"):
            raise HTTPException(415, "Upload an MP4 file")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as target:
            temporary = Path(target.name)
            size = 0
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "MP4 exceeds 100 MB upload limit")
                target.write(chunk)
        if size == 0:
            raise HTTPException(422, "MP4 is empty")
        return await _analyze(temporary, force_live, model_provider, known_object_width_cm, known_length_cm)
    finally:
        await file.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class CounterfactualRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    regime: Literal["pendulum", "drag_fall", "drag", "vertical_projectile", "bouncing_ball"]
    params: dict[str, float]
    initial_state: tuple[float, float]
    pivot_or_bounds: dict[str, float]
    fps: float = Field(default=30, gt=0, le=240)
    frames: int = Field(default=120, ge=2, le=3600)


@app.post("/api/counterfactual")
def counterfactual(request: CounterfactualRequest):
    geometry = request.pivot_or_bounds
    try:
        frame_height = geometry["frame_height"]
        if frame_height <= 0:
            raise ValueError("frame_height must be positive")
        times = np.arange(request.frames) / request.fps
        params = dict(request.params)
        if request.regime == "bouncing_ball" and "floor" not in params:
            params["floor"] = geometry.get("floor", 0)
        release_time = geometry.get("release_time", 0.0)
        if release_time < 0 or release_time >= times[-1]:
            raise ValueError("Invalid release time")
        active = times >= release_time
        # Include the exact release instant even when it falls between grid samples.
        active_times = np.unique(np.r_[release_time, times[active]])
        fitted = simulate_counterfactual(request.regime, params, request.initial_state, active_times)
        states = np.tile(request.initial_state, (len(times), 1)).astype(float)
        states[~active, 1] = 0.0
        states[active] = fitted[np.searchsorted(active_times, times[active])]
        if request.regime == "pendulum":
            radius = geometry["radius"]
            if radius <= 0:
                raise ValueError("radius must be positive")
            x = geometry["x0"] + radius * np.sin(states[:, 0])
            y = frame_height - geometry["y0"] + radius * np.cos(states[:, 0])
        else:
            x = geometry["x0"] + geometry.get("vx", 0) * times
            y = frame_height - states[:, 0]
    except (ValueError, KeyError, ZeroDivisionError, OverflowError) as error:
        raise HTTPException(422, "Invalid counterfactual parameters or pivot/bounds") from error
    return {"trajectory_points": [{"frame_idx": i, "t": float(times[i]),
                                    "pred_x_px": float(x[i]), "pred_y_px": float(y[i])}
                                   for i in range(request.frames)],
            "phase_space": [{"position": float(p), "velocity": float(v)} for p, v in states],
            "phase_units": "radians" if request.regime == "pendulum" else "pixels"}


class StoredAnalysis(BaseModel):
    # Accept the existing analysis response, while validating the export inputs.
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    regime: Literal["pendulum", "drag_fall", "bouncing_ball"]
    discovered_params: dict[str, float]
    pivot_or_bounds: dict[str, float]
    inferred_physical_metrics: dict[str, float] = Field(default_factory=dict)
    initial_state: tuple[float, float] = (0, 0)
    trajectory_points: list[dict[str, float | bool]] = Field(default_factory=list)


class CalibrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    analysis: StoredAnalysis
    known_object_width_cm: float | None = Field(default=None, gt=0)
    known_length_cm: float | None = Field(default=None, gt=0)


@app.post("/api/calibrate")
def calibrate(request: CalibrationRequest):
    # No disk access, large arrays, SAM calls or integrations in this hot path.
    try:
        metrics = calibrate_physical_metrics(request.analysis.model_dump(exclude={"trajectory_points"}),
                                             request.known_object_width_cm, request.known_length_cm)
    except (ValueError, KeyError, ZeroDivisionError) as error:
        raise HTTPException(422, str(error)) from error
    return {"inferred_physical_metrics": metrics}


def simulator_source(analysis):
    regime = analysis["regime"]
    params = dict(analysis["discovered_params"])
    metrics = analysis["inferred_physical_metrics"]
    initial = list(analysis["initial_state"])
    scale = metrics.get("meters_per_pixel")
    units = "radians" if regime == "pendulum" else "pixels"
    if scale is not None and regime != "pendulum":
        initial = [v * scale for v in initial]
        params["g_pixels"] *= scale
        if "floor" in params:
            params["floor"] *= scale
        if "c2" in params:
            params["c2"] /= scale
        units = "meters"
    points = analysis["trajectory_points"]
    times = [p["t"] for p in points if not p.get("is_pre_release", False)] or [i / 30 for i in range(150)]
    payload = json.dumps({"regime": regime, "params": params, "initial": initial,
                          "times": times, "units": units}, allow_nan=False)
    return '\n'.join([
        '#!/usr/bin/env python3',
        '"""PhysiSight discovered law. Install: pip install numpy scipy matplotlib.',
        'Coordinates are Cartesian (positive upward); pendulum angles are radians.',
        'Scale provenance is recorded below. Point-mass dynamics, no mass estimate.',
        '"""',
        f'# Calibration: {json.dumps(metrics, sort_keys=True)}',
        'import json', 'import numpy as np', 'from scipy.integrate import solve_ivp',
        f'data = json.loads({payload!r})',
        'params = data["params"]', 'times = np.array(data["times"])',
        'initial = np.array(data["initial"], dtype=float)', '',
        'def rhs(t, state):', '    x, v = state',
        '    if data["regime"] == "pendulum":',
        '        displacement = x - params.get("theta_eq", 0)',
        '        restoring = displacement if params.get("linear", 0) else np.sin(displacement)',
        '        acceleration = -params["omega0_squared"] * restoring - params["gamma"] * v',
        '    else:',
        '        acceleration = -params["g_pixels"] - params.get("c1", 0) * v - params.get("c2", 0) * abs(v) * v',
        '    return [v, acceleration]', '',
        'def simulate():',
        '    if data["regime"] != "bouncing_ball":',
        '        solution = solve_ivp(rhs, (times[0], times[-1]), initial, t_eval=times, method="RK45", rtol=1e-9, atol=1e-10)',
        '        if not solution.success: raise RuntimeError(solution.message)',
        '        return solution.y.T',
        '    def floor_event(t, state):',
        '        return state[0] - params["floor"]',
        '    floor_event.terminal = True', '    floor_event.direction = -1',
        '    output = np.empty((len(times), 2))',
        '    start, state = float(times[0]), initial.copy()',
        '    if state[0] <= params["floor"] and state[1] < 0:',
        '        state[0], state[1] = params["floor"], -params["e"] * state[1]',
        '    for _ in range(1000):',
        '        if state[0] <= params["floor"] + 1e-10 and abs(state[1]) < 1e-5:',
        '            output[times >= start] = [params["floor"], 0]', '            return output',
        '        solution = solve_ivp(rhs, (start, times[-1]), state, events=floor_event, dense_output=True, method="RK45", rtol=1e-9, atol=1e-10)',
        '        if not solution.success: raise RuntimeError(solution.message)',
        '        selected = (times >= start) & (times <= solution.t[-1])',
        '        output[selected] = solution.sol(times[selected]).T',
        '        if not len(solution.t_events[0]): return output',
        '        start = float(solution.t_events[0][0])',
        '        state = solution.y_events[0][0].copy()',
        '        state[0], state[1] = params["floor"], -params["e"] * state[1]',
        '        if start >= times[-1]: return output',
        '    raise RuntimeError("Too many impacts")', '',
        'if __name__ == "__main__":', '    import matplotlib.pyplot as plt', '    states = simulate()',
        '    plt.plot(times, states[:, 0], label="Discovered ODE")',
        '    plt.xlabel("Time (s)")', '    plt.ylabel(data["units"])',
        '    plt.legend()', '    plt.tight_layout()', '    plt.show()', '',
    ])


def mujoco_source(analysis):
    params, metrics = analysis["discovered_params"], analysis["inferred_physical_metrics"]
    regime = analysis["regime"]
    gravity = metrics.get("g_measured_m_s2", 9.81)
    scale = metrics.get("meters_per_pixel")
    if scale is None and regime != "pendulum":
        scale = gravity / params["g_pixels"]
    provenance = "Independently calibrated gravity" if "g_measured_m_s2" in metrics else "Reference gravity assumed; SI scale is not independently measured"
    parts = [f'<mujoco model="physisight_{regime}">',
             f'  <!-- {provenance}. Unit mass = 1 kg; video does not identify mass. -->',
             '  <compiler angle="radian"/>',
             f'  <option gravity="0 0 -{gravity:.12g}" timestep="0.001"/>']
    if regime == "pendulum":
        length = metrics.get("L_meters", gravity / params["omega0_squared"])
        damping = params.get("gamma", 0) * length**2
        parts += ['  <!-- Point mass with inertia m*L^2; damping = gamma*m*L^2. -->',
                  '  <!-- MuJoCo uses sin(theta); a selected linear law is its small-angle approximation. -->',
                  '  <worldbody><body name="pendulum" pos="0 0 2">',
                  f'    <inertial pos="0 0 -{length:.12g}" mass="1" diaginertia="1e-8 1e-8 1e-8"/>',
                  f'    <joint name="hinge" type="hinge" axis="0 -1 0" damping="{damping:.12g}"/>',
                  f'    <geom type="sphere" pos="0 0 -{length:.12g}" size="0.02" contype="0" conaffinity="0" mass="0"/>',
                  '  </body></worldbody>',
                  f'  <keyframe><key qpos="{analysis["initial_state"][0]:.12g}" qvel="{analysis["initial_state"][1]:.12g}"/></keyframe>']
    else:
        radius = metrics.get("equiv_diameter_px", 10) * scale / 2
        floor = params.get("floor", 0) * scale
        height, velocity = [v * scale for v in analysis["initial_state"]]
        damping = params.get("c1", 0)
        e = params.get("e", 0)
        zeta = 1.0 if e <= 0 else max(0.001, -math.log(e) / math.sqrt(math.pi**2 + math.log(e)**2))
        parts += ['  <!-- Contact solref is an approximation to restitution, not an exact impact reset.',
                  '       Time constant 0.02 s is chosen, not identified. Validate against your geometry. -->',
                  f'  <custom><numeric name="video_restitution" data="{e:.12g}"/></custom>',
                  '  <worldbody>',
                  *([f'    <geom name="floor" type="plane" size="10 10 0.1" pos="0 0 {floor-radius:.12g}" solref="0.02 {zeta:.12g}"/>'] if regime == "bouncing_ball" else []),
                  '    <body name="ball">',
                  f'      <joint name="vertical" type="slide" axis="0 0 1" damping="{damping:.12g}"/>',
                  f'      <geom type="sphere" size="{radius:.12g}" mass="1" solref="0.02 {zeta:.12g}"/>',
                  '    </body>', '  </worldbody>',
                  f'  <keyframe><key qpos="{height:.12g}" qvel="{velocity:.12g}"/></keyframe>']
        if regime == "drag_fall":
            parts += [f'  <!-- Quadratic drag c2 = {params.get("c2", 0)/scale:.12g} 1/m.',
                      '       Apply -m*c2*abs(v)*v as qfrc_applied in your controller. -->']
    return '\n'.join(parts + ['</mujoco>', ''])


@app.post("/api/export/{kind}")
def export_analysis(kind: Literal["python", "mujoco", "csv"], analysis: StoredAnalysis):
    data = analysis.model_dump()
    try:
        if kind == "csv":
            output = io.StringIO(newline="")
            writer = csv.writer(output)
            writer.writerow(["frame_idx", "t_sec", "x_px", "y_px", "pred_x_px", "pred_y_px", "is_heldout"])
            for p in data["trajectory_points"]:
                writer.writerow([int(p["frame_idx"]), p["t"], p.get("x_px", ""), p.get("y_px", ""),
                                 p["pred_x_px"], p["pred_y_px"], bool(p.get("is_heldout", False))])
            content, name, media = output.getvalue(), "physisight_kinematics.csv", "text/csv"
        elif kind == "python":
            content, name, media = simulator_source(data), f"physisight_sim_{data['regime']}.py", "text/x-python"
        else:
            content, name, media = mujoco_source(data), f"physisight_mujoco_{data['regime']}.xml", "application/xml"
    except (ValueError, KeyError, ZeroDivisionError, TypeError) as error:
        raise HTTPException(422, "Analysis lacks valid export parameters") from error
    return Response(content, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})
