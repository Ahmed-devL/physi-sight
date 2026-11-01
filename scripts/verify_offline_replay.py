"""Verify an existing analysis through the public pipeline with inference blocked.

Requires a previously populated local cache; never creates a live analysis.
"""
import argparse
import asyncio
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.agent import analyze_video
from backend.cache import DiskCache, video_sha256
from backend.provenance import replay_provenance


def verify(path, cache):
    digest = video_sha256(path)
    stored = cache.load(digest).get('analyses', {}).get('meta')
    if not stored or not replay_provenance(stored, digest)['cache_integrity_verified']:
        raise ValueError('Requires an existing verified Meta analysis for this exact video')
    def blocked(*args, **kwargs):
        raise AssertionError('Network or model inference attempted during offline replay')
    with patch('backend.agent._meta_client', blocked), patch('backend.agent._hypothesis_client', blocked), \
            patch.object(socket.socket, 'connect', blocked), patch.object(socket.socket, 'connect_ex', blocked):
        result = asyncio.run(analyze_video(path, cache=cache))
    provenance = result['run_provenance']
    assert result['cache_hit']
    assert provenance['execution_mode'] == 'verified_cached_replay'
    assert provenance['api_calls_this_request'] == 0
    assert provenance['model_calls'] == stored['run_provenance']['model_calls']
    for key in ('selected_candidate', 'discovered_params', 'initial_state',
                'heldout_r2', 'heldout_rmse', 'parameter_uncertainty'):
        assert result.get(key) == stored.get(key), key
    for key, value in stored.get('quantity_provenance', {}).items():
        assert result['quantity_provenance'][key] == value, key
    for key, value in stored.get('inferred_physical_metrics', {}).items():
        assert result['inferred_physical_metrics'][key] == value, key
    additions = sorted(set(result.get('quantity_provenance', {})) -
                       set(stored.get('quantity_provenance', {})))
    return {'video_sha256': digest, 'stored_analysis_sha256': stored['analysis_integrity_sha256'],
            'execution_mode': provenance['execution_mode'], 'api_calls_this_request': 0,
            'selected_candidate': result['selected_candidate'],
            'physical_results_and_original_model_calls_preserved': True,
            'additional_derived_quantity_labels': additions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=Path('backend/cache'))
    args = parser.parse_args()
    print(json.dumps(verify(args.video, DiskCache(args.cache_dir)), indent=2))
