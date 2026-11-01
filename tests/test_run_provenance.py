import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import NotFoundError

import backend.agent as agent
from backend.provenance import capture_calls, chat_completion, analysis_fingerprint, replay_provenance, execution_mode
from backend.cache import DiskCache, video_sha256


def response(model):
    return SimpleNamespace(model=model, choices=[SimpleNamespace(message=SimpleNamespace(
        content='{"regime":"pendulum","object_prompt":"ball"}'))])


def test_execution_records_actual_response_model_and_failed_fallback(monkeypatch):
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.chat.completions.create = AsyncMock(side_effect=[
        NotFoundError('unavailable', response=httpx.Response(404, request=httpx.Request('POST', 'https://example.test')), body=None),
        response('muse-spark-1.3')])
    monkeypatch.setattr(agent, '_meta_client', lambda: client)
    with capture_calls() as calls:
        grounding = asyncio.run(agent.ground_scene('unused', ['frame0', 'frame15']))
    assert grounding == {'regime': 'pendulum', 'object_prompt': 'ball'}
    assert [c['requested_model'] for c in calls] == ['muse-spark-1.3-contributor', 'muse-spark-1.3']
    assert [c['status'] for c in calls] == ['failed', 'completed']
    assert calls[1]['response_model'] == 'muse-spark-1.3'
    assert all(c['api_elapsed_seconds'] >= 0 for c in calls)
    assert 'unavailable' not in str(calls)


def test_parallel_runs_keep_call_provenance_separate():
    async def run(model):
        client = MagicMock()
        async def complete(**kwargs):
            await asyncio.sleep(0)
            return response(model)
        client.chat.completions.create = complete
        with capture_calls() as calls:
            await chat_completion(client, 'hypothesis_proposal', model, [])
        return calls
    async def both():
        return await asyncio.gather(run('muse-wonder-1.0'), run('meta-models/muse-spark-open-1026'))
    left, right = asyncio.run(both())
    assert [c['requested_model'] for c in left] == ['muse-wonder-1.0']
    assert [c['requested_model'] for c in right] == ['meta-models/muse-spark-open-1026']


def test_verified_replay_requires_content_integrity_and_matching_video():
    stored = {'video_sha256': 'a'*64, 'selected_candidate': 'mixed_drag',
              'run_provenance': {'video_sha256': 'a'*64, 'execution_mode': 'live_inference',
                                 'model_calls': [{'requested_model': 'muse-wonder-1.0'}]}}
    stored['analysis_integrity_sha256'] = analysis_fingerprint(stored)
    replay = replay_provenance(stored, 'a'*64)
    assert replay['execution_mode'] == 'verified_cached_replay'
    assert replay['api_calls_this_request'] == 0
    assert replay['original_execution_mode'] == 'live_inference'
    assert replay_provenance(stored, 'b'*64)['execution_mode'] == 'unverified_cached_replay'
    stored['selected_candidate'] = 'free_fall'
    assert replay_provenance(stored, 'a'*64)['execution_mode'] == 'unverified_cached_replay'


def test_legacy_cache_does_not_invent_model_execution():
    replay = replay_provenance({'model_provider': 'meta'}, 'a'*64)
    assert replay['execution_mode'] == 'unverified_cached_replay'
    assert not replay['cache_integrity_verified']
    assert 'model_calls' not in replay


def test_failed_api_attempt_does_not_claim_successful_live_inference():
    assert execution_mode([{'status': 'failed'}]) == 'live_api_failure_with_deterministic_fallback'
    assert execution_mode([]) == 'offline_demonstration'
    assert execution_mode([{'status': 'failed'}, {'status': 'completed'}]) == 'live_inference'


def test_offline_replay_remains_identified_as_offline():
    stored = {'video_sha256': 'a'*64, 'run_provenance': {
        'video_sha256': 'a'*64, 'execution_mode': 'offline_demonstration', 'model_calls': []}}
    stored['analysis_integrity_sha256'] = analysis_fingerprint(stored)
    replay = replay_provenance(stored, 'a'*64)
    assert replay['original_execution_mode'] == 'offline_demonstration'
    assert replay['model_calls'] == []


def test_pipeline_offline_switch_replay_and_separate_timings(tmp_path, monkeypatch):
    from test_regimes_and_api import _fixture_masks
    import backend.app as api
    path = api.DEMO_DIR / 'damped_pendulum.mp4'
    metadata, _ = agent._video_info(path)
    cache = DiskCache(tmp_path)
    cache.update(video_sha256(path), grounding={'regime': 'pendulum', 'object_prompt': 'ball'},
                 masks=agent._encode_masks(_fixture_masks('pendulum')),
                 video_metadata=metadata, background_image='fixture')
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        model='muse-wonder-1.0', choices=[SimpleNamespace(message=SimpleNamespace(
            content='{"candidate":"nonlinear_pendulum","rationale":"Test damping"}'))]))
    monkeypatch.setattr(agent, '_hypothesis_client', lambda p: (client, 'muse-wonder-1.0'))
    async def run():
        live = await agent._analyze_video(path, cache=cache)
        assert live['run_provenance']['execution_mode'] == 'live_inference'
        assert live['run_provenance']['grounding_source'] == 'cached_grounding'
        assert live['run_provenance']['model_calls'][0]['response_model'] == 'muse-wonder-1.0'
        assert live['timings']['solver_execution_seconds'] > 0
        assert live['timings']['api_elapsed_seconds'] >= 0
        replay = await agent._analyze_video(path, cache=cache)
        assert replay['run_provenance']['execution_mode'] == 'verified_cached_replay'
        assert replay['run_provenance']['api_calls_this_request'] == 0
        offline = await agent._analyze_video(path, cache=cache, wonder_enabled=False)
        assert offline['run_provenance']['execution_mode'] == 'offline_demonstration'
        assert offline['run_provenance']['model_calls'] == []
        assert offline['selected_candidate'] == live['baseline_fit']['name']
        assert offline['discovered_params'] == live['discovered_params']
        offline_replay = await agent._analyze_video(path, cache=cache, wonder_enabled=False)
        assert offline_replay['run_provenance']['original_execution_mode'] == 'offline_demonstration'
        assert offline_replay['run_provenance']['execution_mode'] == 'verified_cached_replay'
        assert client.chat.completions.create.await_count == 1
    asyncio.run(run())
