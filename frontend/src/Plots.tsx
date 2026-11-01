import { useEffect, useMemo, useRef } from 'react';
import type { RefObject } from 'react';
import { bracket, observationStart } from './api';
import type { Analysis, Point, Simulation } from './api';

const CREAM = '#D79921', SAGE = '#687D31', GOLD = '#D65D0E';
export interface Layers { tracked: boolean; forecast: boolean; ghost: boolean; background: boolean }

function useCanvas(draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;
  useEffect(() => {
    const canvas = ref.current!;
    const context = canvas.getContext('2d');
    if (!context) return;
    let width = 0, height = 0, raf = 0;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width; height = rect.height;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas); resize();
    const frame = () => {
      context.clearRect(0, 0, width, height);
      if (width && height) drawRef.current(context, width, height);
      raf = requestAnimationFrame(frame);
    };
    frame();
    return () => { cancelAnimationFrame(raf); observer.disconnect(); };
  }, []);
  return ref;
}

function path(points: Point[], x: (p: Point) => number, y: (p: Point) => number) {
  const result = new Path2D();
  points.forEach((p, i) => { if (i) result.lineTo(x(p), y(p)); else result.moveTo(x(p), y(p)); });
  return result;
}
function dot(ctx: CanvasRenderingContext2D, x: number, y: number, color: string, r = 4) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = color; ctx.fill();
  ctx.beginPath(); ctx.arc(x, y, r + 5, 0, Math.PI * 2);
  ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke();
}

export function Overlay({ analysis, simulation, layers, videoRef }: {
  analysis: Analysis | null; simulation: Simulation | null; layers: Layers;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  const paths = useMemo(() => {
    const points = analysis?.trajectory_points ?? [];
    return {
      train: path(points.filter(p => !p.is_heldout), p => p.x_px!, p => p.y_px!),
      forecast: path(points.filter(p => p.is_heldout), p => p.pred_x_px, p => p.pred_y_px),
      ghost: path(simulation?.trajectory_points ?? [], p => p.pred_x_px, p => p.pred_y_px),
    };
  }, [analysis, simulation]);
  // requestVideoFrameCallback supplies the presented media timestamp; paused
  // seeks use currentTime immediately rather than waiting for another frame.
  const presented = useRef(0);
  useEffect(() => {
    const video = videoRef.current;
    if (!video?.requestVideoFrameCallback) return;
    let id = 0;
    const frame: VideoFrameRequestCallback = (_, metadata) => {
      presented.current = metadata.mediaTime;
      id = video.requestVideoFrameCallback(frame);
    };
    id = video.requestVideoFrameCallback(frame);
    return () => video.cancelVideoFrameCallback(id);
  }, [videoRef, analysis]);
  const ref = useCanvas((ctx, width, height) => {
    if (!analysis) return;
    const { width: sourceW, height: sourceH } = analysis.video_metadata;
    const sx = width / sourceW, sy = height / sourceH;
    const video = videoRef.current;
    const time = (video?.paused || !video?.requestVideoFrameCallback ? video?.currentTime ?? 0 : presented.current) - observationStart(analysis);
    ctx.save(); ctx.scale(sx, sy); ctx.lineWidth = 3 / sx;
    if (layers.tracked) {
      ctx.strokeStyle = CREAM; ctx.globalAlpha = 0.6; ctx.stroke(paths.train); ctx.globalAlpha = 1;
      for (const p of analysis.trajectory_points) {
        if (!p.is_heldout && p.t <= time) {
          ctx.beginPath(); ctx.arc(p.x_px!, p.y_px!, 1.8 / sx, 0, Math.PI * 2); ctx.fillStyle = CREAM; ctx.fill();
        }
      }
    }
    if (layers.forecast) {
      ctx.strokeStyle = SAGE; ctx.shadowColor = SAGE; ctx.shadowBlur = 7; ctx.stroke(paths.forecast); ctx.shadowBlur = 0;
    }
    if (layers.ghost && simulation) {
      ctx.strokeStyle = GOLD; ctx.setLineDash([7 / sx, 6 / sx]); ctx.stroke(paths.ghost); ctx.setLineDash([]);
      if (time >= 0 && time <= simulation.trajectory_points.at(-1)!.t) {
        const [a, b, f] = bracket(simulation.trajectory_points, time);
        const p = simulation.trajectory_points[a], q = simulation.trajectory_points[b];
        const x = p.pred_x_px + (q.pred_x_px - p.pred_x_px) * f;
        const y = p.pred_y_px + (q.pred_y_px - p.pred_y_px) * f;
        if (analysis.regime === 'pendulum') {
          const g = analysis.pivot_or_bounds;
          ctx.beginPath(); ctx.moveTo(g.x0, sourceH - g.y0); ctx.lineTo(x, y);
          ctx.strokeStyle = '#a26b53'; ctx.globalAlpha = 0.6; ctx.stroke(); ctx.globalAlpha = 1;
        }
        ctx.beginPath(); ctx.arc(x, y, 10 / sx, 0, Math.PI * 2);
        ctx.fillStyle = '#D65D0E44'; ctx.fill();
        ctx.strokeStyle = GOLD; ctx.lineWidth = 1.5 / sx; ctx.stroke();
        dot(ctx, x, y, GOLD, 2 / sx);
      }
    }
    const observations = analysis.trajectory_points;
    if (time >= 0 && time <= observations.at(-1)!.t) {
      const [a, b, f] = bracket(observations, time), p = observations[a], q = observations[b];
      if (p.is_heldout ? layers.forecast : layers.tracked) {
        const xKey = p.is_heldout ? 'pred_x_px' : 'x_px';
        const yKey = p.is_heldout ? 'pred_y_px' : 'y_px';
        dot(ctx, p[xKey]! + (q[xKey]! - p[xKey]!) * f,
          p[yKey]! + (q[yKey]! - p[yKey]!) * f, p.is_heldout ? SAGE : CREAM, 3 / sx);
      }
    }
    ctx.restore();
  });
  return <canvas ref={ref} className="video-overlay" aria-label="Tracked centroids, held-out forecast, and counterfactual trajectory overlay" />;
}

export function PhasePortrait({ analysis, baseline, simulation, videoRef, showGhost }: {
  analysis: Analysis | null; baseline: Simulation | null; simulation: Simulation | null;
  videoRef: RefObject<HTMLVideoElement | null>; showGhost: boolean;
}) {
  const geometry = useMemo(() => {
    const base = baseline?.phase_space ?? [];
    const ghost = showGhost ? simulation?.phase_space ?? [] : [];
    const all = [...base, ...ghost].filter(p => Number.isFinite(p.position) && Number.isFinite(p.velocity));
    let xmin = -1, xmax = 1, ymin = -1, ymax = 1;
    if (all.length) {
      xmin = Math.min(...all.map(p => p.position)); xmax = Math.max(...all.map(p => p.position));
      ymin = Math.min(...all.map(p => p.velocity)); ymax = Math.max(...all.map(p => p.velocity));
      const dx = (xmax - xmin || 1) * 0.12, dy = (ymax - ymin || 1) * 0.15;
      xmin -= dx; xmax += dx; ymin -= dy; ymax += dy;
    }
    return { xmin, xmax, ymin, ymax };
  }, [baseline, simulation, showGhost]);
  const cached = useRef<{ geometry: typeof geometry; width: number; height: number;
    train: Path2D; heldout: Path2D; ghost: Path2D } | null>(null);
  const ref = useCanvas((ctx, width, height) => {
    const left = 76, top = 30, w = width - left - 30, h = height - top - 68;
    const { xmin, xmax, ymin, ymax } = geometry;
    const x = (p: number) => left + (p - xmin) / (xmax - xmin) * w;
    const y = (v: number) => top + h - (v - ymin) / (ymax - ymin) * h;
    ctx.font = '600 15px "JetBrains Mono", "Roboto Mono", monospace';
    for (let i = 0; i <= 4; i++) {
      const xx = left + w * i / 4, yy = top + h * i / 4;
      ctx.strokeStyle = '#D8C7B4'; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(xx, top); ctx.lineTo(xx, top + h); ctx.moveTo(left, yy); ctx.lineTo(left + w, yy); ctx.stroke();
      ctx.fillStyle = '#4A4238'; ctx.textAlign = 'center';
      ctx.fillText((xmin + (xmax - xmin) * i / 4).toFixed(1), xx, top + h + 25);
      ctx.textAlign = 'right'; ctx.fillText((ymax - (ymax - ymin) * i / 4).toFixed(1), left - 9, yy + 3);
    }
    ctx.fillStyle = '#4A4238'; ctx.textAlign = 'center';
    const units = analysis?.regime === 'pendulum' ? 'rad' : 'px';
    ctx.fillText(`${analysis?.regime === 'pendulum' ? 'θ' : 'y'} [${units}]`, left + w / 2, height - 12);
    ctx.save(); ctx.translate(18, top + h / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(`velocity [${units}/s]`, 0, 0); ctx.restore();
    if (!analysis || !baseline) return;
    const time = (videoRef.current?.currentTime ?? 0) - observationStart(analysis);
    const heldoutTime = analysis.trajectory_points.find(p => p.is_heldout)?.t ?? Infinity;
    // Build projected paths only when the data or plot dimensions change.
    // Playback redraws three cached paths and interpolates two dots.
    if (!cached.current || cached.current.geometry !== geometry || cached.current.width !== width || cached.current.height !== height) {
      const train = new Path2D(), heldout = new Path2D(), ghost = new Path2D();
      for (let i = 1; i < baseline.phase_space.length; i++) {
        const p = baseline.phase_space[i - 1], q = baseline.phase_space[i];
        const target = baseline.trajectory_points[i].t >= heldoutTime ? heldout : train;
        target.moveTo(x(p.position), y(p.velocity)); target.lineTo(x(q.position), y(q.velocity));
      }
      if (showGhost && simulation) simulation.phase_space.forEach((p, i) => {
        if (i) ghost.lineTo(x(p.position), y(p.velocity)); else ghost.moveTo(x(p.position), y(p.velocity));
      });
      cached.current = { geometry, width, height, train, heldout, ghost };
    }
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, w, h); ctx.clip();
    ctx.lineWidth = 3; ctx.strokeStyle = CREAM; ctx.stroke(cached.current.train);
    ctx.strokeStyle = SAGE; ctx.stroke(cached.current.heldout);
    if (showGhost && simulation) {
      ctx.setLineDash([5, 4]); ctx.lineWidth = 3; ctx.strokeStyle = GOLD; ctx.stroke(cached.current.ghost); ctx.setLineDash([]);
    }
    function playhead(sim: Simulation, isGhost: boolean) {
      if (time < 0 || time > sim.trajectory_points.at(-1)!.t) return;
      const [a, b, f] = bracket(sim.trajectory_points, time), p = sim.phase_space[a], q = sim.phase_space[b];
      dot(ctx, x(p.position + (q.position - p.position) * f), y(p.velocity + (q.velocity - p.velocity) * f),
        isGhost ? GOLD : time >= heldoutTime ? SAGE : CREAM, isGhost ? 3 : 4);
    }
    playhead(baseline, false);
    if (showGhost && simulation) playhead(simulation, true);
    ctx.restore();
  });
  return <canvas ref={ref} className="phase-canvas" role="img" aria-label="Position versus velocity phase portrait. The playhead follows the video; the dashed curve shows the counterfactual." />;
}

export function EnergyChart({ analysis, simulation, showGhost, videoRef }: {
  analysis: Analysis | null; simulation: Simulation | null; showGhost: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
}) {
  const diagnostics = analysis?.energy_diagnostics;
  const metrics = analysis?.inferred_physical_metrics;
  // Calibration changes only the common squared-length multiplier; no refit.
  const calibrated = analysis?.regime === 'pendulum' ? metrics?.L_meters != null : metrics?.meters_per_pixel != null;
  const scale = analysis?.regime === 'pendulum' ? (metrics?.L_meters ?? 1) ** 2 : (metrics?.meters_per_pixel ?? 1) ** 2;
  const ratio = scale / (diagnostics?.energy_scale ?? 1);
  const ref = useCanvas((ctx, width, height) => {
    if (!analysis || !diagnostics?.series.length) return;
    const points = diagnostics.series;
    const left = 76, top = 38, w = width - left - 26, h = height - top - 64;
    const first = points[0].t, last = points.at(-1)!.t;
    const ghost = showGhost ? simulation?.energy_diagnostics : undefined;
    const ghostRatio = scale / (ghost?.energy_scale ?? 1);
    const maximum = Math.max(1e-12, ...points.map(p => p.total * ratio), ...(ghost?.series.map(p => p.total * ghostRatio) ?? [])) * 1.08;
    const x = (t: number) => left + (t - first) / (last - first || 1) * w;
    const y = (e: number) => top + h - e / maximum * h;
    ctx.font = '12px "JetBrains Mono", monospace';
    for (let i = 0; i <= 4; i++) {
      const yy = top + h * i / 4, xx = left + w * i / 4;
      ctx.strokeStyle = '#D8C7B4'; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(left, yy); ctx.lineTo(left + w, yy); ctx.stroke();
      ctx.fillStyle = '#4A4238'; ctx.textAlign = 'right';
      ctx.fillText((maximum * (1 - i / 4)).toPrecision(3), left - 8, yy + 4);
      ctx.textAlign = 'center'; ctx.fillText((first + (last - first) * i / 4).toFixed(2), xx, top + h + 23);
    }
    ctx.textAlign = 'left'; ctx.fillText(`Specific energy [${calibrated ? 'J/kg' : analysis.regime === 'pendulum' ? 'rad²/s²' : 'px²/s²'}]`, left, 18);
    ctx.textAlign = 'center'; ctx.fillText('Time [s]', left + w / 2, height - 10);
    ctx.save(); ctx.beginPath(); ctx.rect(left, top, w, h); ctx.clip();
    const split = analysis.trajectory_points.find(p => p.is_heldout)?.t;
    if (split != null) {
      ctx.fillStyle = '#687D3110'; ctx.fillRect(x(split), top, x(last) - x(split), h);
      ctx.strokeStyle = '#807264'; ctx.setLineDash([5, 4]); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x(split), top); ctx.lineTo(x(split), top + h); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#4A4238'; ctx.textAlign = 'right'; ctx.fillText('70% | 30%', x(split) - 5, top + 15);
    }
    const colors = { kinetic: '#BC870C', potential: '#548778', total: '#912F46' };
    for (const key of ['kinetic', 'potential', 'total'] as const) {
      ctx.beginPath(); ctx.strokeStyle = colors[key]; ctx.lineWidth = key === 'total' ? 3 : 1.8;
      points.forEach((p, i) => {
        if (i) {
          if (analysis.regime === 'bouncing_ball' && key === 'total') ctx.lineTo(x(p.t), y(points[i - 1].total * ratio));
          ctx.lineTo(x(p.t), y(p[key] * ratio));
        } else ctx.moveTo(x(p.t), y(p[key] * ratio));
      });
      ctx.stroke();
    }
    if (ghost) {
      ctx.beginPath(); ctx.strokeStyle = GOLD; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ghost.series.forEach((p, i) => { if (i) ctx.lineTo(x(p.t), y(p.total * ghostRatio)); else ctx.moveTo(x(p.t), y(p.total * ghostRatio)); });
      ctx.stroke(); ctx.setLineDash([]);
    }
    const time = (videoRef.current?.currentTime ?? 0) - observationStart(analysis);
    if (time >= first && time <= last) {
      ctx.strokeStyle = '#4A423899'; ctx.lineWidth = 1; ctx.beginPath();
      ctx.moveTo(x(time), top); ctx.lineTo(x(time), top + h); ctx.stroke();
      let a = 0, b = points.length - 1;
      while (b - a > 1) { const mid = (a + b) >>> 1; if (points[mid].t <= time) a = mid; else b = mid; }
      const f = (time - points[a].t) / (points[b].t - points[a].t || 1);
      for (const key of ['kinetic', 'potential', 'total'] as const) {
        // Bounce energy is discontinuous; avoid interpolating across an impact.
        const value = analysis.regime === 'bouncing_ball' && key === 'total' ? points[a][key]
          : points[a][key] + f * (points[b][key] - points[a][key]);
        dot(ctx, x(time), y(value * ratio), colors[key], 3);
      }
    }
    ctx.restore();
  });
  const mechanism = analysis?.regime === 'pendulum' ? 'Damping' : analysis?.regime === 'drag_fall' ? 'Drag' : 'Impact';
  return <div className="energy-diagnostics">
    <h3>Mechanical Energy &amp; Hamiltonian Decay <span>E(t) = T + V</span></h3>
    <div className="energy-pills">
      <span>Energy Retained <strong>{diagnostics ? diagnostics.energy_retained_pct.toFixed(2) : '—'}%</strong></span>
      <span>Dissipated by {mechanism} <strong>{diagnostics ? diagnostics.total_energy_dissipated_pct.toFixed(2) : '—'}%</strong></span>
    </div>
    <div className="phase-legend"><span style={{color: '#BC870C'}}>● Kinetic T(t)</span><span style={{color: '#548778'}}>● Potential V(t)</span><span style={{color: '#912F46'}}>● Total E(t)</span>{showGhost && simulation && <span>┄ Counterfactual E(t)</span>}</div>
    <div className="phase-stage energy-stage"><canvas ref={ref} className="phase-canvas" role="img" aria-label="Kinetic, potential and total mechanical energy against time, with held-out split and video-synchronized playhead" />{!diagnostics && <div className="plot-empty">Energy curves appear after analysis</div>}</div>
    {analysis && diagnostics && <p className="calibration-note">{analysis.regime === 'bouncing_ball'
      ? `Each impact loses ${(100 * (diagnostics.impact_loss_fraction ?? 0)).toFixed(1)}% of energy (1 − e²); energy is constant during flight. Impact loss is impulsive.`
      : analysis.regime === 'pendulum'
        ? `dE/dt = −γ(Lω)². The small-angle, cycle-averaged envelope follows exp(−γt), γ = ${analysis.discovered_params.gamma.toFixed(3)} s⁻¹.${diagnostics.potential_model === 'harmonic' ? ' The selected linear law uses harmonic potential.' : ''}`
        : 'Aerodynamic drag does negative work: dE/dt = −(c₁v² + c₂|v|³). Height is referenced to the lowest fitted trajectory point.'} Percentages cover release to end. Empirical SG energy estimates are included in the analysis data.</p>}
  </div>;
}
