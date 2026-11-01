from dataclasses import replace

import cv2
import numpy as np

from backend.kinematics import MaskMoments, mask_contours, segmentation_overlays


def mask(raster, *, frame=3):
    h, w = raster.shape
    rows, columns = np.nonzero(raster)
    return MaskMoments('ball', frame, (101, 203, 100+w, 202+h), raster,
                       len(rows), float(101+columns.mean()), float(203+rows.mean()), None, 0, 0, 0)


def test_contours_apply_box_offsets_in_original_frame_coordinates():
    raster = np.zeros((9, 12), np.uint8)
    raster[2:7, 3:10] = 1
    result = mask_contours(mask(raster))
    assert result['status'] == 'observed_sam_mask'
    assert len(result['polygons']) == 1
    assert sorted(result['polygons'][0]['points']) == sorted([[104, 205], [110, 205], [110, 209], [104, 209]])


def test_disconnected_components_and_holes_never_become_one_polygon():
    raster = np.zeros((40, 70), np.uint8)
    raster[2:28, 2:28] = 1
    raster[9:20, 9:20] = 0
    raster[6:22, 50:65] = 1
    result = mask_contours(mask(raster))
    assert len(result['polygons']) == 3
    assert sum(p['hole'] for p in result['polygons']) == 1
    # The empty gap between disconnected components stays outside every ring.
    for polygon in result['polygons']:
        points = np.array(polygon['points'], np.float32)
        assert cv2.pointPolygonTest(points, (141., 218.), False) < 0


def test_budget_omission_is_explicit_and_keeps_real_boxes():
    raster = np.zeros((20, 30), np.uint8)
    raster[2:8, 2:8] = 1
    raster[10:18, 20:28] = 1
    source = mask(raster)
    assert mask_contours(source, max_rings=1) == {'polygons': [], 'status': 'contour_budget_exceeded'}
    result = segmentation_overlays([source], 'ball', [3], max_total_vertices=0)
    assert result['omitted_contours'] == 1
    assert result['frames'][0]['box'] == list(source.box)
    assert not result['frames'][0]['polygons']


def test_no_interpolated_or_rejected_observation_gets_a_mask():
    source = mask(np.ones((8, 10), np.uint8), frame=0)
    masks = [source, replace(source, frame_index=2), replace(source, frame_index=5)]
    result = segmentation_overlays(masks, 'ball', [0, 5])
    assert [f['frame_idx'] for f in result['frames']] == [0, 5]
    assert all(f['source'] == 'sam' for f in result['frames'])
    capped = segmentation_overlays(masks, 'ball', [0, 2, 5], max_frames=1)
    assert len(capped['frames']) == 1
    assert capped['omitted_frames'] == 2


def test_inconsistent_box_dimensions_are_reported():
    source = mask(np.ones((8, 10), np.uint8))
    result = mask_contours(replace(source, box=(101, 203, 104, 206)))
    assert result['status'] == 'invalid_box_local_dimensions'
    assert not result['polygons']
