import asyncio
from dataclasses import replace

import cv2
import numpy as np
import pytest

from backend.kinematics import (
    extract_median_background,
    fit_pendulum_pivot,
    infer_pendulum_length,
    interpolate_trajectory,
    parse_sam_video,
)
from test_kinematics import events, wire


def test_video_parser_explicit_frames_tracks_and_revisions():
    text = "\n".join([
        wire(frame=2), wire(frame=5), wire(frame=2),
        wire(frame=5).replace("f>7", "f>9"),
    ])
    masks = asyncio.run(parse_sam_video(events(text)))
    assert {(m.frame_index, m.object_id) for m in masks} == {(2, "7"), (5, "7"), (5, "9")}
    assert len(masks) == 3
    assert all((m.xc, m.yc) == (11, 20.5) for m in masks)


def test_video_parser_empty_and_incomplete():
    assert asyncio.run(parse_sam_video(events(""))) == []
    with pytest.raises(ValueError, match="Incomplete"):
        asyncio.run(parse_sam_video(events(wire(frame=5), completed=False)))
    with pytest.raises(ValueError):
        asyncio.run(parse_sam_video(events(wire("!invalid", frame=5))))


def test_interpolate_tracks_cartesian_offsets_and_skipped_frames():
    mask, = asyncio.run(parse_sam_video(events(wire(frame=2))))
    # Four detections exactly determine a cubic; the gap includes >3 frames.
    frames = np.array([2, 4, 9, 11])
    masks = [replace(mask, frame_index=int(i), xc=float(i**3), yc=float(20+i**2))
             for i in frames]
    masks += [replace(mask, object_id="other", frame_index=6, xc=45, yc=8),
              replace(mask, object_id="empty", area=0, xc=None, yc=None)]
    tracks = interpolate_trajectory(masks[::-1], fps=30, frame_height=64)
    track = tracks["7"]
    grid = np.arange(2, 12)
    np.testing.assert_array_equal(track.frame_indices, grid)
    np.testing.assert_allclose(track.times, grid/30)
    np.testing.assert_allclose(track.x, grid**3)
    np.testing.assert_allclose(track.y_phys, 64-20-grid**2)
    np.testing.assert_array_equal(track.observed, np.isin(grid, frames))
    assert set(tracks) == {"7", "other"}
    np.testing.assert_array_equal(tracks["other"].y_phys, [56])
    assert interpolate_trajectory([], 30, 64) == {}


def test_interpolate_two_points_and_invalid_inputs():
    mask, = asyncio.run(parse_sam_video(events(wire(frame=2))))
    last = replace(mask, frame_index=4, xc=15, yc=24.5)
    track = interpolate_trajectory([mask, last], 30, 64)["7"]
    np.testing.assert_allclose(track.x, [11, 13, 15])
    np.testing.assert_allclose(track.y_phys, [43.5, 41.5, 39.5])
    with pytest.raises(ValueError, match="Duplicate"):
        interpolate_trajectory([mask, mask], 30, 64)
    for fps, height in [(0, 64), (np.nan, 64), (30, -1)]:
        with pytest.raises(ValueError):
            interpolate_trajectory([], fps, height)


def test_circle_fit_unseen_pivot_angles_and_length():
    theta = np.linspace(-0.9, 0.9, 120)
    x0, y0, radius = 256, 464, 280
    x, y = x0+radius*np.sin(theta), y0-radius*np.cos(theta)
    fit = fit_pendulum_pivot(x, y)
    assert (fit.x0, fit.y0, fit.L_pixels) == pytest.approx((x0, y0, radius))
    np.testing.assert_allclose(fit.theta, theta, atol=1e-12)
    np.testing.assert_allclose(fit.angles(x, y), theta, atol=1e-12)
    assert infer_pendulum_length(19.62) == pytest.approx(0.50)
    assert infer_pendulum_length(8, g=4) == pytest.approx(0.50)


def test_circle_fit_noisy_arc():
    rng = np.random.default_rng(73)
    theta = np.linspace(-1, 1, 120)
    x = 1000+180*np.sin(theta)+rng.normal(0, 0.3, len(theta))
    y = 2000-180*np.cos(theta)+rng.normal(0, 0.3, len(theta))
    fit = fit_pendulum_pivot(x, y)
    assert fit.x0 == pytest.approx(1000, abs=0.3)
    assert fit.y0 == pytest.approx(2000, abs=0.5)
    assert fit.L_pixels == pytest.approx(180, abs=0.5)
    assert np.sqrt(np.mean((fit.theta-theta)**2)) < 0.004


@pytest.mark.parametrize("x,y", [([1, 2], [1, 2]), ([1, 2, 3], [2, 4, 6]),
                                     ([1, 1, 1], [2, 2, 2]), ([1, 2, 3], [1, 2, np.nan])])
def test_circle_fit_rejects_unidentifiable_data(x, y):
    with pytest.raises(ValueError):
        fit_pendulum_pivot(x, y)


@pytest.mark.parametrize("frequency,g", [(0, 9.81), (-1, 9.81), (np.inf, 9.81), (1, 0)])
def test_length_invalid(frequency, g):
    with pytest.raises(ValueError):
        infer_pendulum_length(frequency, g)


@pytest.mark.parametrize("frame_count", [5, 40])
def test_background_rgb_and_temporal_median(tmp_path, frame_count):
    path = tmp_path / "moving.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened()
    try:
        for i in range(frame_count):
            image = np.full((48, 64, 3), (40, 130, 210), dtype=np.uint8)
            # A transient occlusion of each patch leaves the underlying studio.
            x = (i % 5)*12
            image[12:24, x:x+8] = (230, 20, 20)
            writer.write(image)
    finally:
        writer.release()
    background = extract_median_background(path)
    assert background.shape == (48, 64, 3)
    assert background.dtype == np.uint8
    np.testing.assert_allclose(background, np.broadcast_to([210, 130, 40], background.shape), atol=7)


def test_background_missing_video(tmp_path):
    with pytest.raises(ValueError, match="Cannot open video"):
        extract_median_background(tmp_path / "missing.mp4")
