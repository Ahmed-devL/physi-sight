"""Verify decoded real-video mask alignment; write a contact sheet, never caches.

Usage: .venv/bin/python scripts/verify_mask_contours.py CACHE.json VIDEO.mp4 --output /tmp/contours.png
The video hash must match the content-addressed cache. No API calls are made.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.agent import _decode_masks
from backend.cache import video_sha256
from backend.kinematics import mask_contours


def verify(cache_path, video_path, output):
    assert video_sha256(video_path) == cache_path.stem, 'Video/cache hash mismatch'
    data = json.loads(cache_path.read_text())
    masks = _decode_masks(data['masks'])
    capture = cv2.VideoCapture(str(video_path))
    scores, tiles = [], []
    examples = set(np.linspace(0, len(masks)-1, min(6, len(masks)), dtype=int))
    try:
        for i, mask in enumerate(masks):
            result = mask_contours(mask)
            assert result['status'] == 'observed_sam_mask', result['status']
            x1, y1, x2, y2 = map(int, mask.box)
            offset = np.array([x1, y1])
            reconstructed = np.zeros_like(mask.raster)
            for polygon in result['polygons']:
                points = (np.asarray(polygon['points']) - offset).astype(np.int32)
                cv2.fillPoly(reconstructed, [points], 0 if polygon['hole'] else 1)
            truth = mask.raster != 0
            overlap = np.count_nonzero(truth & (reconstructed != 0)) / np.count_nonzero(truth | (reconstructed != 0))
            scores.append(overlap)
            capture.set(cv2.CAP_PROP_POS_FRAMES, mask.frame_index)
            ok, frame = capture.read()
            assert ok, 'Cannot decode source frame'
            h, w = frame.shape[:2]
            assert (w, h) == (data['video_metadata']['width'], data['video_metadata']['height'])
            assert 0 <= x1 <= x2 < w and 0 <= y1 <= y2 < h
            if i in examples:
                for polygon in result['polygons']:
                    cv2.polylines(frame, [np.asarray(polygon['points'], np.int32)], True, (0, 255, 0), 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 180, 0), 1)
                # Context around the actual source position, enlarged for inspection.
                crop = frame[max(0, y1-30):min(h, y2+31), max(0, x1-30):min(w, x2+31)]
                tile = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_NEAREST)
                cv2.putText(tile, f'Frame {mask.frame_index}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1)
                tiles.append(tile)
    finally:
        capture.release()
    assert scores and min(scores) > .85, 'Simplified contours do not adequately reproduce decoded masks'
    sheet = np.hstack(tiles)
    assert cv2.imwrite(str(output), sheet)
    report = {'masks_checked': len(scores), 'minimum_mask_iou': min(scores),
              'mean_mask_iou': float(np.mean(scores)), 'video_hash_verified': True,
              'note': 'Contour fidelity to decoded masks, not segmentation accuracy or a solver benchmark.'}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.cache, args.video, args.output)
