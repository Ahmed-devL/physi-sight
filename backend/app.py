"""Local development API: .venv/bin/uvicorn backend.app:app --reload."""
import json
from pathlib import Path
import tempfile
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
from openai import APIError
from pydantic import BaseModel, ConfigDict, Field

from backend.agent import analyze_video
from backend.solver import simulate_counterfactual

DEMO_DIR = Path(__file__).resolve().parent / "demo_clips"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
app = FastAPI(title="PhysiSight", description="Built with Muse Spark Open")
app.add_middleware(CORSMiddleware, allow_origins=[
    "http://localhost:3000", "http://localhost:5173",
    "http://127.0.0.1:3000", "http://127.0.0.1:5173",
], allow_methods=["GET", "POST"], allow_headers=["*"])
app.mount("/demo_clips", StaticFiles(directory=DEMO_DIR), name="demo_clips")


def _manifest():
    return json.loads((DEMO_DIR / "manifest.json").read_text())


@app.get("/api/demos")
def demos():
    manifest = _manifest()
    shared = {key: value for key, value in manifest.items() if key != "clips"}
    regimes = {"damped_pendulum.mp4": "pendulum", "falling_drag_sphere.mp4": "drag_fall",
               "bouncing_ball.mp4": "bouncing_ball"}
    return {"clips": [{"clip_name": name, "url": f"/demo_clips/{name}",
                       "regime": regimes[name], "metadata": {**shared, **metadata}}
                      for name, metadata in manifest["clips"].items()]}


async def _analyze(path, force_live, model_provider):
    try:
        return await analyze_video(path, force_live=force_live, model_provider=model_provider)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except APIError as error:
        # Do not expose provider response bodies, headers or credentials.
        raise HTTPException(502, "Model provider request failed") from error
    except RuntimeError as error:
        raise HTTPException(503, "Analysis unavailable; check model configuration and server logs") from error


@app.post("/api/analyze/demo/{clip_name}")
async def analyze_demo(clip_name: str, force_live: bool = False,
                       model_provider: Literal["meta", "baseten"] = "meta"):
    if clip_name not in _manifest()["clips"]:
        raise HTTPException(404, "Unknown demo clip")
    return await _analyze(DEMO_DIR / clip_name, force_live, model_provider)


@app.post("/api/analyze/upload")
async def analyze_upload(file: UploadFile = File(...), force_live: bool = False,
                         model_provider: Literal["meta", "baseten"] = "meta"):
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
        return await _analyze(temporary, force_live, model_provider)
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
        states = simulate_counterfactual(request.regime, params, request.initial_state, times)
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
