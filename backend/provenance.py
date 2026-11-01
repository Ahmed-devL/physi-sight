"""Per-run API execution records and content-integrity checks; no credentials."""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
from time import perf_counter

_CALLS = ContextVar('model_calls', default=None)


@contextmanager
def capture_calls():
    calls = []
    token = _CALLS.set(calls)
    try:
        yield calls
    finally:
        _CALLS.reset(token)


@contextmanager
def api_event(stage, model):
    event = {'stage': stage, 'requested_model': model, 'response_model': None, 'status': 'failed'}
    started = perf_counter()
    try:
        yield event
    except BaseException as error:
        event['error_type'] = type(error).__name__
        raise
    else:
        event['status'] = 'completed'
    finally:
        event['api_elapsed_seconds'] = perf_counter() - started
        calls = _CALLS.get()
        if calls is not None:
            calls.append(event)


async def chat_completion(client, stage, model, messages):
    with api_event(stage, model) as event:
        response = await client.chat.completions.create(model=model, messages=messages)
        returned_model = getattr(response, 'model', None)
        if isinstance(returned_model, str):
            event['response_model'] = returned_model
        return response


def execution_mode(calls):
    if any(call['status'] == 'completed' for call in calls):
        return 'live_inference'
    return 'live_api_failure_with_deterministic_fallback' if calls else 'offline_demonstration'


def analysis_fingerprint(result):
    payload = {key: value for key, value in result.items() if key != 'analysis_integrity_sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def replay_provenance(stored, video_digest):
    prior = stored.get('run_provenance', {})
    fingerprint = stored.get('analysis_integrity_sha256')
    verified = bool(fingerprint and fingerprint == analysis_fingerprint(stored)
                    and stored.get('video_sha256') == video_digest
                    and prior.get('video_sha256') == video_digest)
    return {**prior,
            'execution_mode': 'verified_cached_replay' if verified else 'unverified_cached_replay',
            'original_execution_mode': prior.get('execution_mode', 'not_recorded'),
            'cache_integrity_verified': verified, 'api_calls_this_request': 0,
            'verification_scope': 'Stored analysis content and source video SHA256; model outputs are not independently verified.',
            'cache_note': ('Recorded run replayed without API calls.' if verified else
                           'Legacy or modified cache: original model execution cannot be verified. Force a live rerun to record provenance.')}
