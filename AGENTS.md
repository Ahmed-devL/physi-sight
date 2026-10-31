# PhysiSight Workspace Instructions

## Source of truth and development rules
- Read `.specs/p-engineering.md` (master engineering plan) and `.specs/p-architecture.md` for the full design. Documentation lives in `.specs/`, formerly `specs/`.
- Use bash and Python 3.11+ in `.venv`. Run pytest after modifying backend code or tests.
- Keep credentials in the ignored `.env`; never commit keys or print them. `.env.example` contains placeholders only.
- The UI header and README must visibly display "Built with Muse Spark Open" when implementing the planned Muse Spark Open integration.
- Keep modules flat: FastAPI backend with SSE progress and deterministic caches; React workbench with synchronized mask/ghost overlays, equations, phase portrait, and counterfactual sliders.
- Scope: damped pendulum, aerodynamic drag fall, and inelastic bouncing sphere. Interactive grab-and-flick is a stretch goal after the core workbench works on all three clips.

## Meta Model API contract from the engineering plan
- Base URL: `https://api.meta.ai/v1`; configuration variable: `META_BASE_URL`.
- Use the `openai` Python SDK with `base_url="https://api.meta.ai/v1"` and `api_key=os.environ["MODEL_API_KEY"]`. Baseten credentials use `BASETEN_API_KEY`.
- `muse-spark-1.3` grounds the scene from frames 0 and 15, identifying a moving-object noun phrase and physical regime.
- `muse-wonder-1.0` and Baseten Muse Spark Open propose scientific hypotheses and inspect model-selection residuals. Reasoning/vision calls use chat completions or responses.
- These contracts are taken from the local specs; Gate 0 must verify them with live smoke calls before pipeline implementation.

## SAM 3.1 wire format and coordinates
- Call `client.responses.create(model="sam-3.1", input=[...], metadata={"mask_encoding": "one_bit"}, stream=True)`.
- Supply ONE concrete noun phrase in `input_text`, such as `"pendulum bob"`. No lists, conjunctions, or instructions like `"segment the ball"`.
- Parse `response.output_text.delta` streams, buffering partial lines as needed. The exact wire grammar is:

```text
<Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=..;h=..|><|mask;x=0;y=0;data=H,W,ENCpayload|>
```

- `N` is the explicit zero-based frame index; `id` is a stable integer track ID. Empty frames emit no line, so never infer frame numbers from line order. Timestamp is `N / FPS`.
- Box coordinates are inclusive source-frame pixels; `w` and `h` are full frame dimensions. Mask dimensions are box-local: `H = y2 - y1 + 1` rows, `W = x2 - x1 + 1` columns.
- `ENC` is `~` (lossless) or `!` (one_bit), followed by a base85 payload.
- Use `meta_sam_parser` helpers `decode_mask_to_raster`, `decode_mask_to_rle`, and `decode_mask_to_svg_path`; retain a small regex + `base64.b85decode` fallback in `sam_tracker.py` as specified by the plan.
- Reshape the decoded raster as `M = raster.reshape((H, W))`. For nonempty masks, compute global centroids as `X_px = x1 + sum(c * M[r,c]) / sum(M)` and `Y_px = y1 + sum(r * M[r,c]) / sum(M)`. ALWAYS add the bounding-box offset `(x1, y1)` to local `(column_mean, row_mean)`.
- Interpolate short gaps of 1–3 missing frames with `scipy.interpolate.CubicSpline` on a uniform time grid.

## Kinematics
- Convert screen pixels to Cartesian coordinates: `x_cart = X_px`, `y_cart = H_frame - Y_px`.
- Fit a circle to the pendulum arc to infer pivot `(x0, y0)` and radius `R_px`; compute `theta = arctan2(x_cart - x0, y0 - y_cart)`.
- Infer pendulum length with `L = g / omega0_squared` and scale `L / R_px`. For drag clips, use known gravity or an optional known object width for calibration; do not claim independently measured gravity when gravity sets the scale.
- Normalize video to 30 FPS / 720p with ffmpeg, or use exact presentation timestamps for variable-frame-rate input.

## Trajectory Shooting solver (`backend/ode_solver.py`)
- NEVER fit ODE coefficients exclusively by twice differentiating noisy pixel positions.
- Use `scipy.signal.savgol_filter` smoothing only for initial guesses via linear regression / PySINDy; PySINDy proposes and prunes candidate structures.
- Refine coefficients and initial conditions using Trajectory Shooting: `scipy.integrate.solve_ivp(..., method="RK45")` forward-integrates each candidate; `scipy.optimize.least_squares(..., loss="soft_l1")` minimizes simulated-position residuals against RAW training observations.
- Fit the first chronological 70% of frames only. Forecast the remaining 30% without refitting; report held-out R² and RMSE, and use BIC to favor parsimonious physical laws.
- Candidates include nonlinear vs. linear damped pendulum, linear vs. quadratic drag, and event-driven bouncing sphere.
- For bounces, use a terminal downward floor-crossing event in `solve_ivp`, reset velocity with `v_after = -e * v_before`, and restart integration.

## Sequential verification gates
1. Gate 0: live smoke tests for `muse-wonder-1.0`, `muse-spark-1.3`, and `sam-3.1` with parser decoding; verify Baseten Muse Spark Open when configured.
2. Gate 1: synthetic noisy pendulum, 4 seconds at 30 FPS, `omega0_squared = 19.62`, `gamma = 0.25`, angle noise sigma `0.015` radians. Recover parameters within 3% and select the damped nonlinear pendulum over a polynomial via BIC. The master plan governs these values where the architecture document differs.
3. Gate 2: real-video pipeline through masks, circle-fit angle extraction, ODE fitting, and held-out forecast R² > 0.95.
4. Gate 3: complete the React workbench only after Gates 1 and 2 pass in terminal scripts.
