import { useEffect, useRef, useState } from 'react';
import type { ChangeEvent, ReactNode } from 'react';
import { Activity, ArrowRight, Check, ChevronDown, CircleDot, FlaskConical, Layers3,
  LoaderCircle, Pause, Play, RotateCcw, ScanLine, SkipBack, SkipForward, Upload, Workflow, Zap } from 'lucide-react';
import katex from 'katex';
import { API, assetURL, clipLabel, defaults, format, observationStart, request, simulate } from './api';
import type { Analysis, Controls, Demo, Provider, Simulation } from './api';
import { Overlay, PhasePortrait, EnergyChart } from './Plots';
import { Calibration, Exports } from './Calibration';
import type { Layers } from './Plots';

type Source = { url: string; label: string; clip?: string; file?: File };
const initialControls: Controls = { gravity: 9.81, damping: 0.25, restitution: 0.8 };
const planets = [['Earth', 9.81], ['Moon', 1.62], ['Mars', 3.72], ['Jupiter', 24.79]] as const;
const chapters = [
  ['observation', '01', 'Observation & Counterfactual Studio'],
  ['law', '02', 'Discovered Law'],
  ['hypotheses', '03', 'Hypothesis Arena'],
  ['phase', '04', 'Phase-Space & Diagnostics'],
  ['reasoning', '05', 'Agent Reasoning Trace'],
] as const;
const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'An unexpected error occurred';
const pretty = (name: string) => name.replaceAll('_', ' ');

function Equation({ latex }: { latex: string }) {
  return <div className="equation" dangerouslySetInnerHTML={{ __html: katex.renderToString(latex, {
    displayMode: true, throwOnError: false, trust: false, output: 'htmlAndMathml',
  }) }} />;
}
function Toggle({ active, onClick, children, disabled = false }: {
  active: boolean; onClick: () => void; children: ReactNode; disabled?: boolean;
}) {
  return <button className={`layer-toggle ${active ? 'selected' : ''}`} aria-pressed={active} disabled={disabled} onClick={onClick}>
    <span className="check-box">{active && <Check size={11} />}</span>{children}
  </button>;
}
function Slider({ label, value, min = 0, max, step, unit, disabled, onChange }: {
  label: string; value: number; min?: number; max: number; step: number; unit: string;
  disabled?: boolean; onChange: (value: number) => void;
}) {
  return <label className={`slider-field ${disabled ? 'disabled' : ''}`}>
    <span><span>{label}</span><output>{format(value, 2)} <small>{unit}</small></output></span>
    <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
      onChange={e => onChange(Number(e.target.value))} />
    <span className="slider-limits"><span>{min}</span><span>{disabled ? 'Not applicable to this regime' : unit}</span><span>{format(max, 0)}</span></span>
  </label>;
}

export default function App() {
  const [demos, setDemos] = useState<Demo[]>([]);
  const [source, setSource] = useState<Source | null>(null);
  const [analyzedVideoURL, setAnalyzedVideoURL] = useState<string | null>(null);
  const [provider, setProvider] = useState<Provider>('meta');
  const [forceLive, setForceLive] = useState(false);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [baseline, setBaseline] = useState<Simulation | null>(null);
  const [simulation, setSimulation] = useState<Simulation | null>(null);
  const [controls, setControls] = useState<Controls>(initialControls);
  const [layers, setLayers] = useState<Layers>({ tracked: true, forecast: true, ghost: true, background: false });
  const [loading, setLoading] = useState(false);
  const [fetchingDemos, setFetchingDemos] = useState(true);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState('');
  const [simulationError, setSimulationError] = useState('');
  const [videoError, setVideoError] = useState('');
  const [latency, setLatency] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [videoSize, setVideoSize] = useState({ width: 16, height: 10 });
  const [activeChapter, setActiveChapter] = useState('observation');
  const headerRef = useRef<HTMLElement>(null);
  const [demoRetry, setDemoRetry] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const uploadRef = useRef<HTMLInputElement>(null);
  const analysisRequest = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setFetchingDemos(true); setError('');
    request<{ clips: Demo[] }>('/api/demos', { signal: controller.signal })
      .then(({ clips }) => {
        setDemos(clips);
        if (clips.length) setSource(current => current ?? { url: assetURL(clips[0].url), label: clipLabel(clips[0].regime), clip: clips[0].clip_name });
        else setError('The API has no demo clips. Upload an MP4 to begin.');
      })
      .catch(e => { if (!controller.signal.aborted) setError(`Cannot load demo clips. Start the API at ${API}. ${errorMessage(e)}`); })
      .finally(() => { if (!controller.signal.aborted) setFetchingDemos(false); });
    return () => controller.abort();
  }, [demoRetry]);

  useEffect(() => {
    analysisRequest.current?.abort();
    setAnalysis(null); setAnalyzedVideoURL(null); setBaseline(null); setSimulation(null); setLoading(false);
    setError(''); setSimulationError(''); setLatency(null);
    return () => analysisRequest.current?.abort();
  }, [source, provider]);
  useEffect(() => {
    setTime(0); setDuration(0); setPlaying(false); setVideoError(''); setVideoSize({ width: 16, height: 10 });
    return () => { if (source?.file) URL.revokeObjectURL(source.url); };
  }, [source]);

  useEffect(() => {
    if (!analysis) return;
    const controller = new AbortController();
    const original = defaults(analysis);
    setControls(original); setSimulating(true); setSimulationError('');
    simulate(analysis, original, controller.signal)
      .then(result => { if (!controller.signal.aborted) { setBaseline(result); setSimulation(result); } })
      .catch(e => { if (!controller.signal.aborted) setSimulationError(`Phase simulation failed: ${errorMessage(e)}. Run analysis again to retry.`); })
      .finally(() => { if (!controller.signal.aborted) setSimulating(false); });
    return () => controller.abort();
  }, [analysis]);

  useEffect(() => {
    if (!analysis || !baseline) return;
    const original = defaults(analysis);
    if (controls.gravity === original.gravity && controls.damping === original.damping && controls.restitution === original.restitution) {
      setSimulation(baseline); setLatency(null); setSimulating(false); setSimulationError(''); return;
    }
    const controller = new AbortController();
    setSimulating(true); setSimulationError('');
    // Coalesce rapid inputs to one request per display frame; never queue old
    // slider states. Aborting plus the signal check prevents stale responses.
    const raf = requestAnimationFrame(() => {
      const started = performance.now();
      simulate(analysis, controls, controller.signal)
        .then(result => {
          if (controller.signal.aborted) return;
          setSimulation(result); setLatency(performance.now() - started);
        })
        .catch(e => { if (!controller.signal.aborted) { setSimulation(null); setSimulationError(errorMessage(e)); } })
        .finally(() => { if (!controller.signal.aborted) setSimulating(false); });
    });
    return () => { cancelAnimationFrame(raf); controller.abort(); };
  }, [analysis, baseline, controls]);


  useEffect(() => {
    const header = headerRef.current;
    const resize = new ResizeObserver(() => {
      document.documentElement.style.setProperty('--nav-height', `${header?.getBoundingClientRect().height ?? 160}px`);
    });
    if (header) resize.observe(header);
    const sections = [...document.querySelectorAll<HTMLElement>('[data-chapter]')];
    const reveal = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) { entry.target.classList.add('revealed'); reveal.unobserve(entry.target); }
      });
    }, { threshold: 0, rootMargin: '0px 0px -40px 0px' });
    sections.forEach(section => { section.classList.add('reveal-ready'); reveal.observe(section); });
    let raf = 0;
    const updateChapter = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const edge = (header?.getBoundingClientRect().height ?? 160) + 100;
        const current = [...sections].reverse().find(section => section.getBoundingClientRect().top <= edge);
        setActiveChapter(current?.id ?? 'observation');
      });
    };
    window.addEventListener('scroll', updateChapter, { passive: true });
    updateChapter();
    return () => {
      resize.disconnect(); reveal.disconnect(); cancelAnimationFrame(raf);
      window.removeEventListener('scroll', updateChapter);
      document.documentElement.style.removeProperty('--nav-height');
    };
  }, []);

  async function analyze() {
    if (!source) return;
    analysisRequest.current?.abort();
    const controller = new AbortController(); analysisRequest.current = controller;
    setLoading(true); setError(''); setAnalysis(null); setBaseline(null); setSimulation(null); setLatency(null);
    setSimulationError(''); setSimulating(false);
    try {
      const query = `?force_live=${forceLive}&model_provider=${provider}`;
      const body = source.file ? new FormData() : undefined;
      if (body && source.file) body.append('file', source.file);
      const endpoint = source.file ? '/api/analyze/upload' : `/api/analyze/demo/${encodeURIComponent(source.clip!)}`;
      const result = await request<Analysis>(endpoint + query, { method: 'POST', body, signal: controller.signal });
      if (!controller.signal.aborted) {
        setAnalyzedVideoURL(result.video_url ? assetURL(result.video_url) : source.url);
        setTime(0); setPlaying(false); setVideoError('');
        setAnalysis(result);
      }
    } catch (e) { if (!controller.signal.aborted) setError(errorMessage(e)); }
    finally { if (!controller.signal.aborted) setLoading(false); }
  }
  function chooseDemo(name: string) {
    const demo = demos.find(d => d.clip_name === name);
    if (demo) setSource({ url: assetURL(demo.url), label: clipLabel(demo.regime), clip: name });
  }
  function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]; event.target.value = '';
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.mp4')) { setError('Choose an MP4 video.'); return; }
    if (!file.size || file.size > 100 * 1024 * 1024) { setError('Choose a nonempty MP4 no larger than 100 MB.'); return; }
    setSource({ url: URL.createObjectURL(file), label: file.name, file });
  }
  function seek(next: number) {
    if (videoRef.current) videoRef.current.currentTime = Math.min(duration, Math.max(0, next));
    setTime(Math.min(duration, Math.max(0, next)));
  }
  function step(direction: number) {
    const stamps = analysis?.video_metadata.timestamps;
    const current = videoRef.current?.currentTime ?? time;
    videoRef.current?.pause();
    if (stamps?.length) {
      let index = stamps.findIndex(t => t > current + 0.0001);
      if (index < 0) index = stamps.length;
      seek(stamps[Math.max(0, Math.min(stamps.length - 1, direction > 0 ? index : index - 2))]);
    } else seek(current + direction / 30);
  }
  async function togglePlayback() {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      try { await video.play(); } catch (e) { setVideoError(`Playback failed: ${errorMessage(e)}`); }
    } else video.pause();
  }
  const toggleLayer = (key: keyof Layers) => setLayers(current => ({ ...current, [key]: !current[key] }));
  const update = (key: keyof Controls, value: number) => setControls(current => ({ ...current, [key]: value }));
  const splitPoint = analysis?.trajectory_points.find(p => p.is_heldout);
  const splitTime = analysis && splitPoint ? observationStart(analysis) + splitPoint.t : null;
  const splitPercent = duration && splitTime != null ? Math.min(100, splitTime / duration * 100) : 70;
  const metadata = analysis?.video_metadata;
  const frameIndex = metadata ? Math.max(0, metadata.timestamps.filter(t => t <= time + 0.0001).length - 1) : null;
  const ghostChanged = analysis && (controls.gravity !== defaults(analysis).gravity || controls.damping !== defaults(analysis).damping || controls.restitution !== defaults(analysis).restitution);

  return <div className="app-shell">
    <header ref={headerRef} className="topbar"><div className="topbar-inner">
      <div className="nav-main"><a href="#" className="wordmark" aria-label="PhysiSight home"><span className="brand-icon"><Activity size={28} strokeWidth={1.5} /></span><span>PhysiSight<small>A field guide to physical laws</small></span></a>
      <div className="experiment-bar" aria-label="Experiment configuration">
        <div className="clip-select"><label htmlFor="demo">EXPERIMENT</label><div><FlaskConical size={16} /><select id="demo" value={source?.clip ?? ''} onChange={e => chooseDemo(e.target.value)} disabled={!demos.length}>
          {!source?.clip && <option value="">{source?.file ? source.label : fetchingDemos ? 'Connecting to API…' : 'No demo selected'}</option>}
          {demos.map(d => <option key={d.clip_name} value={d.clip_name}>{clipLabel(d.regime)}</option>)}
        </select><ChevronDown size={13} /></div></div>
        <button className="upload-button" onClick={() => uploadRef.current?.click()}><Upload size={14} />Upload MP4</button>
        <input ref={uploadRef} className="sr-only" type="file" accept="video/mp4,.mp4" onChange={upload} aria-label="Upload custom MP4" />
        <button className="analyze-button" disabled={!source || loading} onClick={() => void analyze()}>{loading ? <LoaderCircle size={15} className="spin" /> : <Zap size={15} />}{loading ? 'Analyzing…' : analysis ? 'Re-analyze' : 'Analyze motion'}<ArrowRight size={14} /></button>
      </div>
</div><div className="nav-secondary"><span className="built-with">Built with Muse Spark Open</span><span className="nav-caption">OBSERVE · DISCOVER · REIMAGINE</span></div>
      <nav className="chapter-nav" aria-label="Discovery chapters">{chapters.map(([id, number, label]) => <a key={id} href={`#${id}`} className={activeChapter === id ? 'active' : ''} aria-current={activeChapter === id ? 'location' : undefined}><span>{number}</span>{label}</a>)}</nav>
    </div></header>
    <main>
      <div className="page-heading"><div className="eyebrow"><span className="status-dot" /> THE COMPUTATIONAL FIELD NOTES / VOL. 01</div><h1>Every motion<br />tells <em>a story.</em></h1><p>Find the physical law hiding in a few seconds of video.<br className="desktop-break" /> Then change the world, and see what happens next.</p><span className="heading-index">FROM PIXELS TO PRINCIPLES <ArrowRight size={20} /></span></div>
      <div className="analysis-settings" aria-label="Analysis settings">        <div className="provider-field"><label htmlFor="provider">MODEL PROVIDER</label><select id="provider" value={provider} onChange={e => setProvider(e.target.value as Provider)}><option value="meta">Meta Muse Wonder 1.0</option></select><span>SAM 3.1 / Muse Spark 1.3</span></div>
        <label className="live-toggle"><input type="checkbox" checked={forceLive} onChange={e => setForceLive(e.target.checked)} /><span className="switch" /><span>Force Live API Re-run</span></label>
</div>
      {error && <div role="alert" className="error-banner">{error}<button onClick={() => source ? void analyze() : setDemoRetry(n => n + 1)}>Retry</button></div>}
      {loading && <div role="status" className="progress-banner"><LoaderCircle size={15} className="spin" /><span>Analyzing video with SAM 3.1 &amp; ODE Solver... {source?.label} {forceLive ? 'Live API re-run requested.' : 'Checking the analysis cache first.'}</span><button onClick={() => { analysisRequest.current?.abort(); setLoading(false); }}>Cancel</button></div>}

      <div className="workbench">
        <section id="observation" data-chapter className="panel chapter studio-section"><div className="panel-header"><div><span className="section-number">01</span><h2>Observation &amp; counterfactual studio</h2></div><span className="micro-label">ONE EXPERIMENT. OTHER WORLDS.</span></div><p className="section-intro">Watch what is. Wonder what could be. Change gravity or damping and follow the ghost into a different world.</p><div className="studio-grid"><div className="observation-column">          <div className="observation-panel">
            <div className="panel-header"><div><h3>Observation field</h3></div><span className="micro-label"><span className={`status-dot ${analysis ? '' : 'idle'}`} />{analysis ? 'TRACK REGISTERED' : 'AWAITING ANALYSIS'}</span></div>
            <div className="video-metadata"><span><ScanLine size={12} />{source?.label ?? 'No video selected'}</span><span>{metadata ? `${metadata.width} × ${metadata.height} / ${format(metadata.fps, 2)} FPS` : 'SOURCE VIDEO'}</span></div>
            <div className="video-viewport"><div className="video-stage" style={{ width: `min(100%, ${540 * (metadata ? metadata.width / metadata.height : videoSize.width / videoSize.height)}px)`, aspectRatio: metadata ? `${metadata.width} / ${metadata.height}` : `${videoSize.width} / ${videoSize.height}` }}>
              {source ? <video key={analyzedVideoURL ?? source.url} ref={videoRef} src={analyzedVideoURL ?? source.url} playsInline preload="metadata"
                className={layers.background && analysis?.background_image ? 'video-hidden' : ''}
                onLoadedMetadata={e => { setDuration(e.currentTarget.duration); setVideoSize({ width: e.currentTarget.videoWidth || 16, height: e.currentTarget.videoHeight || 10 }); }} onDurationChange={e => { if (Number.isFinite(e.currentTarget.duration)) setDuration(e.currentTarget.duration); }}
                onTimeUpdate={e => setTime(e.currentTarget.currentTime)} onSeeked={e => setTime(e.currentTarget.currentTime)}
                onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)}
                onError={() => setVideoError('Unable to decode this MP4. Use a browser-compatible H.264 MP4 and verify the API is running.')} aria-label={source.label} />
                : <div className="video-placeholder"><ScanLine size={36} strokeWidth={1} /><span>{fetchingDemos ? 'Connecting to the observatory…' : 'Select a demo or upload an MP4'}</span></div>}
              {layers.background && analysis?.background_image && <img className="median-background" src={analysis.background_image} alt="Clean temporal median background" />}
              <div className="field-grid" aria-hidden="true" />
              <Overlay analysis={analysis} simulation={simulation} layers={layers} videoRef={videoRef} />
              <span className="field-corner top-left" /><span className="field-corner bottom-right" />
              <div className="field-label">{layers.background && analysis ? 'MEDIAN BACKGROUND' : 'CAM 01 / SOURCE'}<span>{analysis ? analysis.object_prompt.toUpperCase() : 'UNPROCESSED'}</span></div>
              {analysis && <span className={`field-state ${time >= (splitTime ?? Infinity) ? 'heldout' : ''}`}>{time < observationStart(analysis) + (analysis.release_time ?? 0) ? 'PRE-RELEASE / OBSERVED' : time >= (splitTime ?? Infinity) ? 'HELD-OUT / FORECAST' : 'TRAINING / FREE MOTION'}</span>}
            </div>
            </div>
            {metadata?.trim_window_seconds && <p className="calibration-note">Analyzed source window: {format(metadata.trim_window_seconds[0], 2)}–{format(metadata.trim_window_seconds[1], 2)} s.</p>}
            {videoError && <p role="alert" className="inline-error">{videoError}</p>}
            <div className="transport"><button className="play-button" aria-label={playing ? 'Pause video' : 'Play video'} disabled={!source} onClick={() => void togglePlayback()}>{playing ? <Pause size={17} /> : <Play size={17} />}</button><button title="Previous frame" aria-label="Previous frame" disabled={!source} onClick={() => step(-1)}><SkipBack size={14} /></button><button title="Next frame" aria-label="Next frame" disabled={!source} onClick={() => step(1)}><SkipForward size={14} /></button><span className="timecode">{format(time, 2)} <span>/ {format(duration, 2)} s</span></span><span className="frame-label">FRAME {frameIndex == null ? '—' : String(frameIndex).padStart(4, '0')}</span></div>
            <div className="timeline-wrap"><div className="timeline-colors" style={{ background: `linear-gradient(to right, #D79921 0%, #D79921 ${splitPercent}%, #687D31 ${splitPercent}%, #687D31 100%)` }} /><input aria-label="Video playhead in seconds" type="range" min={0} max={duration || 1} step={0.001} value={Math.min(time, duration || 1)} disabled={!duration} onChange={e => seek(Number(e.target.value))} /><span className="split-marker" style={{ left: `${splitPercent}%` }} /></div>
            <div className="timeline-labels"><span><i className="legend-dot cream" />70% FREE-MOTION TRAINING</span><span>CHRONOLOGICAL SPLIT</span><span><i className="legend-dot sage" />30% HELD-OUT FORECAST</span></div>
            <div className="layer-controls"><div className="micro-label"><Layers3 size={13} /> OVERLAY CHANNELS</div><div className="layer-grid"><Toggle active={layers.tracked} onClick={() => toggleLayer('tracked')}>Show Tracked Centroids</Toggle><Toggle active={layers.forecast} onClick={() => toggleLayer('forecast')}>Show 30% Held-Out Forecast</Toggle><Toggle active={layers.ghost} onClick={() => toggleLayer('ghost')}>Show Counterfactual Ghost</Toggle><Toggle active={layers.background} disabled={!analysis?.background_image} onClick={() => toggleLayer('background')}>Show Clean Median Background</Toggle></div></div>
          </div>
          <div className="observation-note"><CircleDot size={16} /><p>{analysis ? <>The model starts at the detected release; forecasts follow the free-motion training window. <strong>The final 30% is never used for fitting.</strong> Amber points are SAM centroids; olive is the held-out model forecast.</> : <>The video is the experiment. <strong>Analyze motion to extract its governing law.</strong> Only backend measurements and model results appear here.</>}</p></div>

</div>          <section className="panel counterfactual-panel"><div className="panel-header"><div><span className="world-icon">✳</span><h3>A different world</h3></div><button className="reset-button" disabled={!analysis || !baseline} onClick={() => analysis && setControls(defaults(analysis))}><RotateCcw size={12} />Reset</button></div><p className="counterfactual-intro">Change the physics. Watch the ghost diverge.</p><div className="planet-presets">{planets.map(([name, gravity]) => <button key={name} disabled={!baseline} aria-pressed={Math.abs(controls.gravity - gravity) < 0.001} className={Math.abs(controls.gravity - gravity) < 0.001 ? 'active' : ''} onClick={() => update('gravity', gravity)}><span>{name}</span><small>{gravity.toFixed(2)}</small></button>)}</div>
            <Slider label="Gravity g" value={controls.gravity} min={0.1} max={30} step={0.01} unit="m/s²" disabled={!baseline} onChange={value => update('gravity', value)} />
            <Slider label={analysis?.regime === 'drag_fall' ? 'Linear drag c₁' : 'Damping γ'} value={controls.damping} max={Math.max(3, Math.ceil((analysis?.discovered_params.gamma ?? analysis?.discovered_params.c1 ?? 0) * 2))} step={0.01} unit="s⁻¹" disabled={!baseline || analysis?.regime === 'bouncing_ball'} onChange={value => update('damping', value)} />
            <Slider label="Restitution e" value={controls.restitution} max={1} step={0.01} unit="dimensionless" disabled={!baseline || analysis?.regime !== 'bouncing_ball'} onChange={value => update('restitution', value)} />
            {simulationError && <p className="inline-error" role="alert">{simulationError}</p>}
            <div className="simulation-status" role="status"><span>{simulating ? <LoaderCircle size={12} className="spin" /> : <span className={`status-dot ${baseline ? '' : 'idle'}`} />}{simulating ? 'Integrating latest parameters…' : !baseline ? 'Awaiting fitted system' : ghostChanged ? 'Counterfactual trajectory active' : 'Ghost matches the fitted system'}</span><span>{latency == null ? 'SIMULATION READY' : `${format(latency, 1)} ms RTT`}</span></div>
          </section>
</div></section>
          <section id="law" data-chapter className="panel chapter equation-panel"><div className="panel-header"><div><span className="section-number">02</span><h2>Discovered governing law</h2></div><span className={`result-status ${analysis ? 'verified' : ''}`}>{analysis ? <><Check size={12} />FITTED</> : 'PENDING'}</span></div>
            {analysis ? <><div className="law-name"><span>{pretty(analysis.selected_candidate)}</span><span>{analysis.cache_hit ? 'CACHED ANALYSIS' : 'LIVE ANALYSIS'}</span></div><Equation latex={analysis.latex_equation} /><p className="interpretation">{analysis.plain_english_explanation}</p><div className="physical-metrics">{Object.entries(analysis.inferred_physical_metrics).map(([key, value]) => <div key={key}><span>{({ L_meters: 'STRING LENGTH L', calibration_gravity_m_s2: 'CALIBRATION g (m/s²)', pixels_per_meter: 'SCALE (px/m)', restitution_e: 'RESTITUTION e', g_pixels: 'GRAVITY (px/s²)' } as Record<string, string>)[key] ?? pretty(key)}</span><strong>{format(value)}{key === 'L_meters' ? ' m' : ''}</strong></div>)}{analysis.regime === 'pendulum' && <div><span>DAMPING RATIO ζ</span><strong>{format(analysis.discovered_params.gamma / (2 * Math.sqrt(analysis.discovered_params.omega0_squared)))}</strong></div>}</div><div className="score-grid"><div><span>TRAIN R²</span><strong>{format(analysis.train_r2, 4)}</strong><small>First 70%</small></div><div className="sage-score"><span>HELD-OUT R²</span><strong>{format(analysis.heldout_r2, 4)}</strong><small>Final 30% · forecast</small></div><div className="sage-score"><span>HELD-OUT RMSE</span><strong>{format(analysis.heldout_rmse, 4)}</strong><small>{analysis.metric_units}</small></div><div><span>{analysis.regime === 'pendulum' ? 'INFERRED LENGTH' : analysis.regime === 'bouncing_ball' ? 'RESTITUTION e' : analysis.inferred_physical_metrics.g_measured_m_s2 != null ? 'MEASURED GRAVITY' : 'CALIBRATION GRAVITY'}</span><strong>{format(analysis.regime === 'pendulum' ? analysis.inferred_physical_metrics.L_meters : analysis.regime === 'bouncing_ball' ? analysis.discovered_params.e : analysis.inferred_physical_metrics.g_measured_m_s2 ?? 9.81, 3)}</strong><small>{analysis.regime === 'pendulum' ? analysis.inferred_physical_metrics.g_measured_m_s2 != null ? 'meters · independent scale' : 'meters · Earth gravity assumed' : analysis.regime === 'bouncing_ball' ? 'Velocity retained at impact' : analysis.inferred_physical_metrics.g_measured_m_s2 != null ? 'm/s² · independent scale' : 'm/s² · assumed for scale'}</small></div></div><p className="calibration-note">{analysis.inferred_physical_metrics.g_measured_m_s2 != null ? 'Gravity measured using an independently supplied length scale.' : analysis.regime === 'pendulum' ? 'Length inferred assuming Earth gravity (9.81 m/s²).' : analysis.regime === 'drag_fall' ? 'Known Earth gravity sets the pixel-to-meter scale.' : 'Gravity is fitted in pixel units; SI scale is not independently measured.'}</p><Calibration key={analysis.video_sha256} analysis={analysis} onMetrics={metrics => setAnalysis(previous => previous ? { ...previous, inferred_physical_metrics: metrics } : previous)} /><Exports analysis={analysis} /></>
              : <div className="empty-law"><Workflow size={30} strokeWidth={1} /><span>{loading ? 'Searching for a physical explanation…' : 'A physical law, waiting to be discovered.'}</span><p>Run an analysis to inspect the equation, inferred parameters, and forecast accuracy.</p></div>}
          </section>
<section id="hypotheses" data-chapter className="panel chapter hypotheses-panel">
        <div className="panel-header"><div><span className="section-number">03</span><h2>Competing hypotheses &amp; BIC selection</h2></div><span className="micro-label">LOWER BIC IS BETTER</span></div>
        <p className="section-intro">A convincing law earns its place. Compare the candidates on training evidence, then see how well they predict the unseen final 30%.</p>
        <div className="table-scroll"><table><caption className="sr-only">Candidate physical laws compared by training BIC and held-out forecast metrics.</caption><thead><tr><th>Candidate model</th><th>Training BIC ↓</th><th>Held-out R²</th><th>Held-out RMSE</th><th>Selection</th></tr></thead><tbody>{analysis ? analysis.candidate_comparison_table.map(candidate => <tr key={candidate.name} className={candidate.selected ? 'chosen' : ''}><td>{pretty(candidate.name)}</td><td>{format(candidate.bic, 1)}</td><td title={`R² = ${candidate.heldout_r2}`}>{candidate.heldout_r2 < -9.99 ? '<-9.99' : format(candidate.heldout_r2, 4)}</td><td>{format(candidate.heldout_rmse, 4)}</td><td>{candidate.selected ? <span className="selected-badge"><Check size={16} />Selected</span> : 'Alternative'}</td></tr>) : <tr><td colSpan={5} className="table-empty">Analyze an experiment to compare its candidate laws.</td></tr>}</tbody></table></div>
        {analysis && <div className="bic-comparison"><h3>Distance from the strongest training evidence</h3><p>ΔBIC relative to the lowest score. A shorter bar means stronger support.</p>{analysis.candidate_comparison_table.map(candidate => {
          const scores = analysis.candidate_comparison_table.map(c => c.bic).filter((v): v is number => v != null && Number.isFinite(v));
          const best = scores.length ? Math.min(...scores) : 0;
          const spread = scores.length ? Math.max(...scores) - best : 0;
          const delta = candidate.bic == null || !Number.isFinite(candidate.bic) ? null : Math.max(0, candidate.bic - best);
          return <div className={`bic-row ${candidate.selected ? 'selected' : ''}`} key={candidate.name}><span>{pretty(candidate.name)}</span><div className="bic-track"><div style={{ width: delta == null ? '0%' : `${spread ? delta / spread * 100 : 0}%` }} /></div><strong>{delta == null ? '—' : `+${format(delta, 1)}`}</strong></div>;
        })}</div>}
        <div className="panel-footnote">Training BIC selects the law. Held-out observations test its forecast. Alternative R² is unavailable from this analysis.</div>
      </section>          <section id="phase" data-chapter className="panel chapter phase-panel"><div className="panel-header"><div><span className="section-number">04</span><h2>Phase-Space &amp; Diagnostics</h2></div><span className="micro-label">{analysis?.regime === 'pendulum' ? 'θ × dθ/dt' : 'y × dy/dt'}</span></div><p className="section-intro">Follow the phase-space portrait and mechanical energy through time. Both playheads stay in step with your video.</p><div className="diagnostics-grid"><div><div className="phase-legend"><span><i className="legend-dot cream" />Fitted system</span><span><i className="legend-dot sage" />Held-out</span><span><i className="legend-line" />Counterfactual</span></div><div className="phase-stage"><PhasePortrait analysis={analysis} baseline={baseline} simulation={simulation} videoRef={videoRef} showGhost={layers.ghost} />{!baseline && <div className="plot-empty">{simulating ? 'Integrating phase trajectory…' : analysis ? 'Phase simulation unavailable' : 'Phase trajectory appears after analysis'}</div>}</div><div className="panel-footnote">POSITION ↔ VELOCITY<span>● PLAYHEAD SYNCED TO VIDEO</span></div></div><EnergyChart analysis={analysis} simulation={simulation} showGhost={layers.ghost} videoRef={videoRef} /></div></section>
<section id="reasoning" data-chapter className="panel chapter reasoning-panel"><div className="panel-header"><div><span className="section-number">05</span><h2>Multi-model agent reasoning trace</h2></div><span className="micro-label">THREE STAGES OF DISCOVERY</span></div><p className="section-intro">From a moving object to a testable explanation. Follow the evidence at each step.</p><div id="reasoning-content" className="reasoning-content"><article><span className="trace-index">01</span><div><h3>Muse Spark 1.3 <span>Grounding</span></h3><p>{analysis ? `Frames 0 and 15 identify “${analysis.object_prompt}” in the ${pretty(analysis.regime)} regime.` : 'Scene grounding identifies the object and physical regime from frames 0 and 15.'}</p></div></article><article><span className="trace-index">02</span><div><h3>SAM 3.1 <span>Sub-Pixel Kinematics</span></h3><p>{analysis ? `${analysis.trajectory_points.length} observed centroids. Training observations calibrate geometry; shooting fits raw positions and forecasts held-out motion.` : 'Segmentation masks become sub-pixel centroids and calibrated trajectories.'}</p></div></article><article><span className="trace-index">03</span><div><h3>{analysis?.model_provider === 'baseten' ? 'Muse Spark Open' : 'Muse Wonder 1.0'} <span>Symbolic Regression</span></h3>{analysis ? analysis.reasoning_steps.map(step => <div className="reasoning-step" key={step.step}><strong>{step.hypothesis}</strong><p>{step.evidence}</p><p>{step.conclusion}</p></div>) : <p>Candidate laws are fit by trajectory shooting, ranked by BIC, and reviewed against held-out forecast evidence.</p>}</div></article></div></section></div>
      <footer><span><Activity size={12} />PhysiSight / A laboratory for observable laws</span><span>{analysis ? `MODEL CALL COST: ${analysis.model_api_cost_usd == null ? 'UNKNOWN' : `$${format(analysis.model_api_cost_usd, 2)}`}` : 'NO ANALYSIS LOADED'}<span className="footer-divider">/</span>CANVAS + SVG · LOCAL SIMULATION</span></footer>
    </main>
  </div>;
}
