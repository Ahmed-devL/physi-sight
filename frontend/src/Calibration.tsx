import { useEffect, useRef, useState } from 'react';
import { API, format, request } from './api';
import type { Analysis } from './api';

export function Calibration({ analysis, onMetrics }: {
  analysis: Analysis; onMetrics: (metrics: Record<string, number>) => void;
}) {
  const [width, setWidth] = useState('');
  const [length, setLength] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const current = useRef({ analysis, onMetrics });
  current.current = { analysis, onMetrics };
  useEffect(() => {
    const controller = new AbortController();
    const invalid = [width, length].some(value => value !== '' && (!Number.isFinite(Number(value)) || Number(value) <= 0));
    if (invalid) { setError('Enter positive dimensions in centimeters.'); setBusy(false); return; }
    setError(''); setBusy(true);
    const timer = window.setTimeout(() => {
      const a = current.current.analysis;
      request<{ inferred_physical_metrics: Record<string, number> }>('/api/calibrate', {
        method: 'POST', signal: controller.signal, headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis: { regime: a.regime, discovered_params: a.discovered_params,
          pivot_or_bounds: a.pivot_or_bounds, inferred_physical_metrics: a.inferred_physical_metrics },
          known_object_width_cm: width === '' ? null : Number(width),
          known_length_cm: length === '' ? null : Number(length) }),
      }).then(result => { if (!controller.signal.aborted) current.current.onMetrics(result.inferred_physical_metrics); })
        .catch(e => { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : 'Calibration failed'); })
        .finally(() => { if (!controller.signal.aborted) setBusy(false); });
    }, 250);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [width, length]);
  const metrics = analysis.inferred_physical_metrics;
  return <div className="calibration-panel">
    <h3>Physical Scale Calibration (Measure g &amp; Ruler Check)</h3>
    <p>Use the object’s visible width in the plane of motion. Keep the camera still and avoid perspective changes.</p>
    <div className="calibration-inputs">
      <label>Known Object Width (cm)<input type="number" min="0.001" step="any" value={width} placeholder="e.g. 4.0" onChange={e => setWidth(e.target.value)} /></label>
      {analysis.regime === 'pendulum' && <label>Ruler String Length (cm)<input type="number" min="0.001" step="any" value={length} placeholder="e.g. 50" onChange={e => setLength(e.target.value)} /></label>}
    </div>
    <div className="physical-metrics" aria-live="polite">
      <div><span>Measured g (m/s²)</span><strong>{format(metrics.g_measured_m_s2)}</strong></div>
      <div><span>Error vs 9.81 (%)</span><strong>{format(metrics.g_error_percent)}</strong></div>
      {analysis.regime === 'pendulum' && <div><span>ODE length vs ruler error (%)</span><strong>{format(metrics.ruler_length_error_percent)}</strong></div>}
    </div>
    <p role="status">{busy ? 'Updating calibration…' : metrics.g_measured_m_s2 != null ? 'Independent length scale applied.' : 'Enter a known width to measure gravity.'}{analysis.regime === 'pendulum' && ' Ruler check compares g/ω² (assuming 9.81) with your ruler length.'}</p>
    {error && <p className="inline-error" role="alert">{error}</p>}
  </div>;
}

export function Exports({ analysis }: { analysis: Analysis }) {
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  async function download(kind: 'python' | 'mujoco' | 'csv') {
    setError(''); setBusy(true);
    try {
      const response = await fetch(`${API}/api/export/${kind}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(analysis),
      });
      if (!response.ok) throw new Error(`Export failed (${response.status}). Check calibration and try again.`);
      const filename = kind === 'csv' ? 'physisight_kinematics.csv' : kind === 'python'
        ? `physisight_sim_${analysis.regime}.py` : `physisight_mujoco_${analysis.regime}.xml`;
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a'); link.href = url; link.download = filename;
      document.body.appendChild(link); link.click(); link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (e) { setError(e instanceof Error ? e.message : 'Export failed'); }
    finally { setBusy(false); }
  }
  return <div className="export-panel"><h3>Sim-to-Real &amp; Lab Exports</h3>
    <div className="export-buttons">
      <button disabled={busy} onClick={() => void download('python')}>Standalone Python Simulator</button>
      <button disabled={busy} onClick={() => void download('mujoco')}>MuJoCo XML Calibration Snippet</button>
      <button disabled={busy} onClick={() => void download('csv')}>Tracked Kinematics CSV</button>
    </div>
    <p>Python reproduces the fitted law. MuJoCo assumes unit mass; contact restitution is approximate and quadratic drag requires a controller force.</p>
    {error && <p className="inline-error" role="alert">{error}</p>}
  </div>;
}
