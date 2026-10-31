"""Manual live Gate 2 test; never run automatically by pytest.

Generate clips first with scripts/generate_demo_videos.py, then run this file
with .venv/bin/python. Requires MODEL_API_KEY in .env. Sends one video to SAM.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import numpy as np
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.kinematics import (
    fit_pendulum_pivot, infer_pendulum_length, interpolate_trajectory, parse_sam_video,
)
from backend.solver import fit_trajectory


async def main():
    directory = ROOT / "backend" / "demo_clips"
    manifest = json.loads((directory / "manifest.json").read_text())
    truth = manifest["clips"]["damped_pendulum.mp4"]
    load_dotenv(ROOT / ".env")
    if not os.environ.get("MODEL_API_KEY"):
        raise RuntimeError("Set MODEL_API_KEY in .env before running this live test")
    video_url = "data:video/mp4;base64," + base64.b64encode(
        (directory / "damped_pendulum.mp4").read_bytes()
    ).decode("ascii")
    async with AsyncOpenAI(api_key=os.environ["MODEL_API_KEY"],
                           base_url=os.environ.get("META_BASE_URL", "https://api.meta.ai/v1")) as client:
        stream = await client.responses.create(
            model="sam-3.1",
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": "circle"},
                {"type": "input_video", "video_url": video_url},
            ]}],
            metadata={"mask_encoding": "one_bit"}, stream=True,
        )
        async with stream:
            masks = await parse_sam_video(stream)
    tracks = interpolate_trajectory(masks, manifest["fps"], manifest["height"])
    if not tracks:
        raise RuntimeError("SAM returned no nonempty tracks")
    # Select using training detections only; require the complete demo time span.
    split = int(0.7 * manifest["frames"])
    track = max(tracks.values(), key=lambda t: np.sum(t.observed & (t.frame_indices < split)))
    if track.frame_indices[0] != 0 or track.frame_indices[-1] != manifest["frames"] - 1:
        raise RuntimeError("Track does not cover the demo's first and last frames")
    x, y = track.x[track.observed], track.y_phys[track.observed]
    times = track.times[track.observed]
    # Solver splits the chronological raw observations 70/30 and supports
    # irregular timestamps. Interpolated positions never become fit targets.
    split = int(0.7 * len(times))
    pivot = fit_pendulum_pivot(x[:split], y[:split])
    angles = pivot.angles(x, y)
    results = fit_trajectory(times, angles, regime="pendulum")
    best = results[0]
    if best.name != "nonlinear_pendulum":
        raise RuntimeError(f"Training BIC selected {best.name}, expected nonlinear_pendulum")
    # Score only raw held-out detections, excluding interpolated samples.
    heldout = np.arange(len(times)) >= best.train_size
    residual = best.prediction[heldout] - angles[heldout]
    variance = np.sum((angles[heldout] - angles[heldout].mean())**2)
    r2 = float(1 - np.sum(residual**2)/variance) if variance > 0 else float("nan")
    length = infer_pendulum_length(best.omega0_squared, truth["g"])
    print(f"g/L={best.omega0_squared:.6f} s^-2 (true {truth['omega0_squared']})")
    print(f"gamma={best.gamma:.6f} s^-1 (true {truth['gamma']})")
    print(f"L_meters={length:.6f} (true {truth['L_meters']}; calibrated using g={truth['g']})")
    print(f"Held-out R^2={r2:.6f}; RMSE={np.sqrt(np.mean(residual**2)):.6f} rad")
    if not np.isfinite(r2) or r2 <= 0.95:
        raise RuntimeError("Gate 2 failed: held-out R^2 must exceed 0.95")
    print("PASS: Gate 2 video forecast exceeds R^2 > 0.95")


if __name__ == "__main__":
    asyncio.run(main())
