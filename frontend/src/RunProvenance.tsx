import { format } from './api';
import type { Analysis } from './api';

const readable = (value: string) => value.replaceAll('_', ' ');
export function runMode(analysis: Analysis) {
  const mode = analysis.run_provenance?.execution_mode;
  return mode ? readable(mode) : analysis.cache_hit ? 'Cached analysis · provenance unavailable' : 'Execution provenance unavailable';
}

export function RunProvenance({ analysis }: { analysis: Analysis }) {
  const provenance = analysis.run_provenance;
  const calls = provenance?.model_calls ?? [];
  return <div className="calibration-panel"><h3>Run provenance</h3>
    <p><strong>{runMode(analysis)}</strong>{provenance?.original_execution_mode && ` · original mode: ${readable(provenance.original_execution_mode)}`}</p>
    <p>{provenance?.cache_note ?? (provenance?.execution_mode === 'offline_demonstration' ? 'Offline numerical analysis of existing observations; no model calls were made.' : '')}</p>
    <p>Grounding: {readable(provenance?.grounding_source ?? 'not recorded')}. Tracking: {readable(provenance?.tracking_source ?? 'not recorded')}.</p>
    <p>Analysis video SHA-256: <code>{analysis.video_sha256}</code></p>
    <p>Analysis record SHA-256: <code>{analysis.analysis_integrity_sha256 ?? 'Not recorded'}</code></p>
    <p>Real SAM coverage: {analysis.segmentation_overlays?.frames.length ?? 'not recorded'} frames
      {analysis.segmentation_overlays?.frames.length ? ` · first frame ${Math.min(...analysis.segmentation_overlays.frames.map(f => f.frame_idx))} · last frame ${Math.max(...analysis.segmentation_overlays.frames.map(f => f.frame_idx))}` : ''}.</p>
    <p>API calls for this request: {provenance?.api_calls_this_request ?? 'not recorded'}.</p>
    <div className="table-scroll"><table><caption>Model calls recorded for the original run</caption><thead><tr><th>Stage</th><th>Model requested</th><th>Model reported by API</th><th>Result</th><th>API elapsed</th></tr></thead>
      <tbody>{calls.length ? calls.map((call, i) => <tr key={i}><td>{readable(call.stage)}</td><td>{call.requested_model}</td><td>{call.response_model ?? 'Not returned by provider'}</td><td>{call.status}{call.error_type && ` (${call.error_type})`}</td><td>{format(call.api_elapsed_seconds, 2)} s</td></tr>)
        : <tr><td colSpan={5}>{provenance?.execution_mode === 'unverified_cached_replay' || !provenance ? 'Original model execution was not recorded.' : 'No model calls in the recorded run.'}</td></tr>}</tbody></table></div>
    {analysis.timings && <p title={`${analysis.timings.solver_scope} ${analysis.timings.api_scope}`}>
      Original run: numerical solver {format(analysis.timings.solver_execution_seconds, 2)} s · API/stream elapsed {format(analysis.timings.api_elapsed_seconds, 2)} s.
      API elapsed includes retries, transfer and mask decoding; server inference and network latency cannot be separated.</p>}
    <p>Model license attribution: Built with Muse Spark Open.</p>
  </div>;
}

export function HypothesisValidation({ analysis }: { analysis: Analysis }) {
  const proposal = analysis.wonder_proposal;
  const verdict = analysis.bic_verdict;
  return <div className="calibration-panel"><h3>Hypothesis Proposal &amp; Numerical Validation</h3>
    {analysis.baseline_fit ? <>
      {analysis.proposal_evidence_fit && <p>Predefined fit shown to proposer: {readable(analysis.proposal_evidence_fit.name)}. Only training evidence was supplied.</p>}
      <p>Original deterministic winner: <strong>{readable(analysis.baseline_fit.name)}</strong> · training BIC {format(analysis.baseline_fit.training_bic, 1)}.</p>
      <p>Proposal status: {readable(proposal?.status ?? 'not recorded')}{proposal?.candidate && ` · ${readable(proposal.candidate)}`}. {proposal?.rationale && `Model rationale (unvalidated): ${proposal.rationale}`}</p>
      <p>{analysis.fallback_used ? 'Proposal processing failed; deterministic selection was retained.' : proposal?.status === 'disabled' ? 'Hypothesis inference was disabled; deterministic analysis is active.' : 'Local trajectory shooting and training BIC validate the proposed structure.'}</p>
      {analysis.proposal_fit && <p>Proposed candidate training BIC: {format(analysis.proposal_fit.training_bic, 1)}.</p>}
      {verdict && <p>Selected: {readable(verdict.selected)}. Change in training BIC: {format(verdict.delta_bic_vs_baseline, 1)}. Winning equation changed: {verdict.winning_equation_changed ? 'yes' : 'no'}.</p>}
      {analysis.heldout_evaluation && <p>Post-selection held-out RMSE: original pipeline {format(analysis.heldout_evaluation.baseline.rmse, 4)}; selected pipeline {format(analysis.heldout_evaluation.selected.rmse, 4)} {analysis.metric_units}.</p>}
      <p>This per-run comparison does not establish that the proposing model improves discovery.</p>
    </> : <p>No proposal ablation was recorded for this legacy analysis. Rerun analysis to test a bounded hypothesis.</p>}
  </div>;
}
