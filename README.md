# PhysiSight

Built with Muse Spark Open

Run the local API with `.venv/bin/uvicorn backend.app:app --reload`.
Interactive API documentation is available at `http://localhost:8000/docs`.
Install dependencies with `.venv/bin/pip install -r backend/requirements.txt`.

`GET /api/demos` lists clips and static MP4 URLs. Analyze a demo with
`POST /api/analyze/demo/damped_pendulum.mp4`, or send an MP4 as the multipart
`file` field to `POST /api/analyze/upload`. Both accept `model_provider=meta`
(default) or `model_provider=baseten`, and `force_live=true` to bypass caches.
Live grounding and SAM require `MODEL_API_KEY`; Baseten hypotheses additionally
require `BASETEN_API_KEY`. Configure credentials only in the ignored `.env`.

Decoded masks and full analyses are cached in `backend/cache/<video_sha256>.json`.
Repeated analysis with the same provider makes no model calls. A different
provider reuses masks and runs its own two-step explanation. Live API cost is
reported as unknown; cache hits report $0.00. No demo results are fabricated or
prepopulated: the first analysis must run live to populate the cache.

For `POST /api/counterfactual`, use `regime`, `discovered_params` (as `params`),
`initial_state`, and `pivot_or_bounds` from an analysis, plus `fps` and `frames`.
Coordinates use Cartesian pixels (positive upward) internally and screen pixels
in the response. Pendulum geometry includes `x0`, `y0`, `radius`, `frame_height`;
other regimes use `x0`, optional `vx`, `frame_height`, and a Cartesian `floor`
for bounces. Slider parameters include `g_scale`, `gamma`, and `e`.
The response contains `trajectory_points` and `phase_space` (position, velocity).

Run offline verification with `.venv/bin/python -m pytest`.

The React scientific workbench lives in `frontend/`. Start it with
`cd frontend`, `npm install`, then `npm run dev` and open http://localhost:5173
while the API is running. Select a demo or upload an MP4 and click
**Analyze motion**. See [frontend/README.md](frontend/README.md) for overlay,
phase-space, counterfactual, and verification details. `npm run build` performs
the frontend TypeScript check and production build.
