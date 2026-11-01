# PhysiSight

Built with Muse Spark Open — model license attribution. This notice does not imply an inference call.

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
provider reuses masks and requests one constrained physical hypothesis. Live API cost is
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

Scientific model criticism uses Muse Spark 1.3 Contributor for scene grounding
(with a Muse Spark 1.3 fallback), SAM 3.1 for segmentation/tracking, and Muse
Wonder 1.0 for a constrained hypothesis proposal. The optional Baseten backend
invokes Muse Spark Open only when explicitly selected and configured. Requested
model names and identifiers actually returned by the provider are recorded
separately; an omitted identifier is displayed as “Not returned by provider.”

Before proposing, Wonder sees one predefined diagnostic fit (linear damped
pendulum, gravity-only fall, or elastic bounces), its coefficients/initial state,
training BIC and training residual summaries, plus the allowlisted candidate/term
library. It sees no expanded ranking or held-out scores. After the proposal, the
established deterministic candidate library is fitted and retained as a fallback
and ablation comparator. A valid, locally identifiable proposal can add a
candidate (currently mixed drag); a proposal already in that library adds no
candidate and cannot change the selected law. No separate sparse expansion is
attributed to Wonder. PySINDy supplies numerical initialization within the solver.
Training BIC selects among eligible physical candidates. Forecast metrics on the
chronological final 30% are evaluated after selection, never used for selection.
This is constrained model criticism and numerical validation, not unrestricted
physical-law discovery. Mock ablations demonstrate possible effects, not actual
Wonder performance. Fitted parameters, gravity-derived scales and independently
calibrated measurements are labeled separately; gravity used to set scale is
not an independently measured result.

Offline replay requires an existing local cache for the exact video bytes.
Disabling hypothesis proposals (`wonder_enabled=false`) alone does **not** disable
live grounding or SAM on a cache miss. To verify a previously recorded Meta
analysis with model clients and outbound socket connections blocked, run:

```bash
.venv/bin/python scripts/verify_offline_replay.py backend/demo_clips/damped_pendulum.mp4
```

The verifier requires matching video and stored-analysis SHA-256 fingerprints,
then exercises `analyze_video`, checks cached replay provenance, zero new model
calls, original model-call records and unchanged stored physical results/provenance.
Current derived-quantity labels may be added and are reported separately. It fails on
missing/unverified caches and attempted inference. Existing cache records retain
their original proposal architecture; replay does not rerun the revised loop.
The returned stored-analysis digest identifies the original record, not a hash
of the replay response with its new request metadata.

A clean checkout includes demo videos and the synthetic generation manifest,
but **does not include analysis caches or SAM observations**. Therefore the
recorded offline product demo is not reproducible from checkout alone: it needs
the matching `backend/cache/<video_sha256>.json` from a previously recorded run
(or a separately approved redistributable fixture). No raw cache or new fixture
is shipped here. Tests use synthetic observations and mocked providers and are
independently reproducible after installing Python/frontend dependencies.

Mocked drag ablation (`tests/test_hypothesis_selection.py`): for mixed-drag
synthetic data, proposing `mixed_drag` expands the evaluated set and changes the
winner from quadratic to mixed drag (training ΔBIC −746.63; post-selection
held-out RMSE 0.011891 → 0.000109). For quadratic-drag synthetic data, the same
proposal expands the set but loses (proposal ΔBIC +4.43), leaving selection and
held-out RMSE unchanged (0.000113). A gravity-only proposal already in the
library changes neither the set nor the winner. Failure/invalid/unidentifiable
proposal tests retain the deterministic winner. These are mocked mechanism
checks, not paid model evaluations or evidence that Wonder reliably improves
scientific inference.
