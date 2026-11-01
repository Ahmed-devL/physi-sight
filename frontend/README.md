# PhysiSight scientific workbench

Built with Muse Spark Open

Start the FastAPI backend from the repository root:

```bash
.venv/bin/uvicorn backend.app:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Select a demo or upload an MP4 (up to 100 MB),
choose the model provider, and click **Analyze motion**. Analysis requests use
backend caches unless **Force Live API Re-run** is enabled. Switching the clip
or provider clears the displayed analysis; re-run explicitly to fetch results.
No results are fabricated. Baseten requires backend configuration.

The default API is http://localhost:8000. Override with `VITE_API_BASE_URL` in
a local `frontend/.env.local` if needed; the backend must allow your UI origin.
API keys belong only in the root backend `.env`, never in a `VITE_` variable.

`npm run build` checks TypeScript and builds the production app.
`npm run preview` serves that build (configure backend CORS if using a port
other than 5173). No chart, animation, or video framework is included.

## Playback and counterfactuals

- Video and overlay share the exact source aspect ratio. Canvas backing stores
  scale to device pixel ratio. Presented video frame timestamps synchronize the
  tracked playhead; a requestAnimationFrame loop draws video and phase overlays.
- Source presentation timestamps align analysis observations, including tracks
  that start after frame zero. Previous/next frame controls use those timestamps
  when analysis is available, with a 30 FPS fallback before analysis.
- Cream marks training centroids, sage marks the held-out position forecast,
  and dashed gold marks the counterfactual. The median background replaces the
  visible video while its playback clock continues driving the overlays.
- The phase portrait comes from a baseline `/api/counterfactual` integration
  using the discovered parameters; its held-out segment is sage. It is a
  simulated phase trajectory, not differentiated measured data.
- Sliders send at most one request per display frame. Superseded requests are
  aborted and stale responses ignored. The previous curve remains visible
  while integrating. The UI displays measured end-to-end request latency;
  the <15 ms target depends on browser, backend, clip length, and hardware and
  is not a guaranteed browser-to-server response time.
- Gravity scales fitted acceleration by `g / 9.81`. Damping changes `gamma`
  for pendulums or linear drag `c1` for falling spheres; restitution changes
  `e` for bounces. Irrelevant controls are disabled. Quadratic drag stays at
  its fitted value. Reset restores all fitted conditions.
- The counterfactual API accepts a uniform grid of at most 3600 frames; the
  client covers the observation duration with that grid and interpolates
  returned states at the video time. Trajectories hide their playhead outside
  that observation window. Baseline phase motion may differ slightly from the
  fitting solver because the endpoint uses RK4 for interactive previews.

## Manual verification

After installing dependencies, run `npm run build`, then use all three demos:

1. Analyze a demo; verify the equation, actual BIC selection, units, score
   badges, reasoning, cache status, and selected provider.
2. Play, pause, scrub, and step across the training/held-out boundary. Resize
   the browser and check centroid/ghost alignment with the source object.
3. Switch each layer; enable the median background and verify the clock,
   overlays, and phase playhead keep moving.
4. Choose every gravity preset, drag the relevant sliders rapidly, and reset.
   Confirm only the newest response controls the ghost and phase curve; check
   the displayed RTT locally against the desired latency target.
5. Switch clips/providers during an analysis. Confirm old results do not
   appear on the new video. Upload an H.264 MP4 and analyze it.
6. Stop the backend to inspect errors and retry. Check a narrow/mobile layout
   and keyboard access to playback, toggles, controls, and reasoning.
