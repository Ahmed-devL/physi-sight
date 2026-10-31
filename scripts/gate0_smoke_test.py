"""Run manually: .venv/bin/python scripts/gate0_smoke_test.py.

Makes two live API calls; requires MODEL_API_KEY in the repository's .env.
This is the requested two-call smoke test, not the entire Gate 0 matrix.
"""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
import sys

import cv2
from dotenv import load_dotenv
import numpy as np
from openai import AsyncOpenAI

# Keep this script runnable directly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.kinematics import parse_sam_frame


def circle_data_url() -> str:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.circle(image, (32, 32), 16, (255, 255, 255), thickness=-1)
    success, png = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not encode the synthetic PNG")
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


async def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("MODEL_API_KEY"):
        raise RuntimeError("Set MODEL_API_KEY in .env before running this script")
    async with AsyncOpenAI(api_key=os.environ["MODEL_API_KEY"], base_url="https://api.meta.ai/v1") as client:
        print("Test A: muse-spark-1.3 text response", flush=True)
        response = await client.chat.completions.create(
            model="muse-spark-1.3",
            messages=[{"role": "user", "content": "Describe a damped pendulum in one sentence."}],
        )
        answer = response.choices[0].message.content
        if not answer:
            raise RuntimeError("Text model returned no response text")
        print(answer, flush=True)

        print("Test B: sam-3.1 synthetic white circle", flush=True)
        stream = await client.responses.create(
            model="sam-3.1",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "white circle"},
                    {"type": "input_image", "image_url": circle_data_url()},
                ],
            }],
            metadata={"mask_encoding": "one_bit"},
            stream=True,
        )
        async with stream:
            masks = await parse_sam_frame(stream)
        if not masks or any(mask.area == 0 for mask in masks):
            raise RuntimeError("SAM returned no nonempty white-circle mask")
        for mask in masks:
            print(
                f"Frame {mask.frame_index}, track {mask.object_id}: "
                f"box={mask.box}, area={mask.area}, "
                f"global centroid=({mask.xc:.2f}, {mask.yc:.2f}), "
                f"orientation={mask.theta} radians"
            )
        print(f"PASS: both smoke tests completed; decoded {len(masks)} mask(s).")

if __name__ == "__main__":
    asyncio.run(main())
