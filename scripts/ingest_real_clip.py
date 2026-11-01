"""Normalize/register a phone video and warm its live analysis cache."""
import argparse
import asyncio
import json
import math
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.agent import analyze_video, _video_info
from backend.kinematics import normalize_video, find_active_motion_window


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--regime', required=True, choices=['pendulum', 'drag_fall', 'bouncing_ball'])
    parser.add_argument('--prompt', required=True, help='One concrete moving-object noun phrase')
    parser.add_argument('--known-object-width-cm', type=float)
    parser.add_argument('--known-length-cm', type=float)
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name.lower().endswith('.mp4'):
        parser.error('--name must be a plain MP4 filename')
    if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9 -]{0,79}', args.prompt)
            or re.search(r'\b(and|or|segment|track|identify|please)\b', args.prompt, re.I)):
        parser.error('--prompt must be one concrete noun phrase')
    for value in (args.known_object_width_cm, args.known_length_cm):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error('Calibration dimensions must be positive and finite')
    directory = ROOT / 'backend' / 'demo_clips'
    target = directory / args.name
    if target.exists():
        parser.error(f'Clip already exists: {args.name}; choose a new name')
    trim_window = find_active_motion_window(args.path)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        normalized = normalize_video(args.path, Path(temporary) / 'normalized.mp4',
                                     trim_window_seconds=trim_window)
        metadata, _ = _video_info(normalized)
        normalized.replace(target)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['clips'][args.name] = {
        'regime': args.regime, 'object_prompt': args.prompt, 'normalized': True,
        **{k: metadata[k] for k in ('fps', 'frames', 'width', 'height')},
        'duration_seconds': metadata['frames'] / metadata['fps'],
        'trim_window_seconds': list(trim_window),
    }
    with tempfile.NamedTemporaryFile(mode='w', dir=directory, delete=False) as output:
        json.dump(manifest, output, indent=2)
        output.write('\n')
    Path(output.name).replace(manifest_path)
    result = await analyze_video(target, force_live=True, regime=args.regime, object_prompt=args.prompt,
                                 known_object_width_cm=args.known_object_width_cm,
                                 known_length_cm=args.known_length_cm, trim_window_seconds=trim_window)
    print(json.dumps({key: result[key] for key in (
        'latex_equation', 'discovered_params', 'heldout_r2', 'inferred_physical_metrics',
        'video_metadata')}, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
