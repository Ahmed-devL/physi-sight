export type Regime = 'pendulum' | 'drag_fall' | 'bouncing_ball';
export type Provider = 'meta' | 'baseten';
export interface Demo { clip_name: string; url: string; regime: Regime }
export interface Point {
  frame_idx: number; t: number; pred_x_px: number; pred_y_px: number;
  x_px?: number; y_px?: number; is_heldout?: boolean; is_pre_release?: boolean;
}
export interface Phase { position: number; velocity: number }
export interface EnergyPoint { t: number; kinetic: number; potential: number; total: number; dissipation_power: number }
export interface EnergyDiagnostics {
  series: EnergyPoint[]; empirical_series: EnergyPoint[]; units: string; energy_scale: number;
  energy_retained_pct: number; total_energy_dissipated_pct: number;
  t_release: number; t_end: number; potential_model: string; impact_loss_fraction: number | null;
}
export interface Simulation { energy_diagnostics: EnergyDiagnostics; trajectory_points: Point[]; phase_space: Phase[]; phase_units: string }
export interface Analysis {
  energy_diagnostics: EnergyDiagnostics;
  regime: Regime;
  video_sha256: string;
  video_url?: string;
  release_frame_idx?: number;
  release_time?: number;
  latex_equation: string;
  plain_english_explanation: string;
  discovered_params: Record<string, number>;
  inferred_physical_metrics: Record<string, number>;
  train_r2: number | null; heldout_r2: number | null; heldout_rmse: number | null;
  metric_units: string;
  candidate_comparison_table: { name: string; bic: number | null; heldout_r2: number; heldout_rmse: number; selected: boolean }[];
  trajectory_points: Point[];
  initial_state: [number, number];
  pivot_or_bounds: Record<string, number>;
  selected_candidate: string;
  reasoning_steps: { step: number; hypothesis: string; evidence: string; conclusion: string }[];
  object_prompt: string;
  video_metadata: { fps: number; frames: number; width: number; height: number; timestamps: number[]; trim_window_seconds?: [number, number] };
  background_image: string;
  model_provider: Provider;
  cache_hit: boolean;
  model_api_cost_usd: number | null;
}
export const API = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, '');
export const assetURL = (url: string) => new URL(url, `${API}/`).href;
export const clipLabel = (regime: Regime) => ({
  pendulum: 'Damped Pendulum', drag_fall: 'Aerodynamic Drag Sphere', bouncing_ball: 'Inelastic Bouncing Ball',
})[regime];

export async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') message = body.detail;
    } catch { /* Preserve status when the server returns a non-JSON error. */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
export interface Controls { gravity: number; damping: number; restitution: number }
export const defaults = (analysis: Analysis): Controls => ({
  gravity: analysis.inferred_physical_metrics.g_measured_m_s2 ?? 9.81,
  damping: analysis.discovered_params.gamma ?? analysis.discovered_params.c1 ?? 0,
  restitution: analysis.discovered_params.e ?? 0.8,
});
export function simulate(analysis: Analysis, controls: Controls, signal: AbortSignal) {
  const params: Record<string, number> = { ...analysis.discovered_params, g_scale: controls.gravity / defaults(analysis).gravity };
  if (analysis.regime === 'pendulum') params.gamma = controls.damping;
  if (analysis.regime === 'drag_fall') params.c1 = controls.damping;
  if (analysis.regime === 'bouncing_ball') params.e = controls.restitution;
  // The server accepts a uniform grid only. Cover the exact PTS duration and
  // interpolate that grid during playback, including variable-frame-rate clips.
  const observations = analysis.trajectory_points;
  const duration = observations.at(-1)!.t;
  const frames = Math.min(3600, Math.max(2, analysis.video_metadata.frames));
  const fps = Math.min(240, (frames - 1) / duration);
  return request<Simulation>('/api/counterfactual', {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ regime: analysis.regime, params, initial_state: analysis.initial_state,
      pivot_or_bounds: analysis.pivot_or_bounds, inferred_physical_metrics: analysis.inferred_physical_metrics, frames, fps }),
  });
}
export function observationStart(analysis: Analysis) {
  const first = analysis.trajectory_points[0];
  return analysis.video_metadata.timestamps[first.frame_idx] ?? first.frame_idx / analysis.video_metadata.fps;
}
export function bracket(points: Point[], time: number): [number, number, number] {
  if (!points.length || time <= points[0].t) return [0, 0, 0];
  if (time >= points[points.length - 1].t) return [points.length - 1, points.length - 1, 0];
  let low = 0, high = points.length - 1;
  while (high - low > 1) {
    const mid = (low + high) >>> 1;
    if (points[mid].t <= time) low = mid; else high = mid;
  }
  return [low, high, (time - points[low].t) / (points[high].t - points[low].t)];
}
export const format = (value: number | null | undefined, digits = 3) =>
  value == null || !Number.isFinite(value) ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });
