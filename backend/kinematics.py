"""SAM mask moments, tracked Cartesian trajectories, and pendulum calibration."""

from dataclasses import dataclass
from collections.abc import AsyncIterable

import meta_sam_parser as sam
import numpy as np
from scipy.interpolate import CubicSpline
import cv2


@dataclass(frozen=True)
class MaskMoments:
    object_id: str
    frame_index: int
    box: tuple[float, float, float, float]  # inclusive x1, y1, x2, y2
    raster: np.ndarray
    area: int
    xc: float | None
    yc: float | None
    theta: float | None  # principal axis, radians modulo pi; None if isotropic
    mu_11: float
    mu_20: float
    mu_02: float


def mask_moments(record: sam.SegmentationMaskRecord) -> MaskMoments:
    """Decode a box-local mask; central moments are unnormalized pixel sums.

    The installed parser exposes half-open left/top/right/bottom bounds,
    whereas SAM's wire box and the returned box use inclusive endpoints.
    Orientation describes the mask shape, not the pendulum swing angle.
    """
    mask, bounds = record.mask, record.bounds
    if (bounds.right - bounds.left, bounds.bottom - bounds.top) != (mask.width, mask.height):
        raise ValueError("Mask dimensions do not match box-local bounds")
    raster = np.frombuffer(sam.decode_mask_to_raster(mask), dtype=np.uint8).reshape(
        mask.height, mask.width
    )
    rows, cols = np.nonzero(raster)
    area = len(rows)
    xc = yc = theta = None
    mu_11 = mu_20 = mu_02 = 0.0
    if area:
        x, y = cols - cols.mean(), rows - rows.mean()
        xc, yc = float(bounds.left + cols.mean()), float(bounds.top + rows.mean())
        mu_11, mu_20, mu_02 = float(x @ y), float(x @ x), float(y @ y)
        if not np.isclose(np.hypot(mu_20 - mu_02, 2 * mu_11), 0):
            theta = float(0.5 * np.arctan2(2 * mu_11, mu_20 - mu_02))
    return MaskMoments(
        record.object_id, record.frame.frame_index if record.frame else 0,
        (bounds.left, bounds.top, bounds.right - 1, bounds.bottom - 1),
        raster, area, xc, yc, theta, mu_11, mu_20, mu_02,
    )


async def parse_sam_frame(stream_events: AsyncIterable[sam.ResponsesEventLike]) -> list[MaskMoments]:
    """Consume an async Responses stream for ONE image, retaining latest revisions.

    Use an AsyncOpenAI stream. Chunk buffering, lane validation, and decoding
    belong to meta_sam_parser. Empty images return []; incomplete or malformed
    responses raise instead of silently producing partial measurements.
    """
    return await _parse_masks(stream_events, sam.image_segmentation_format(), "image")


async def parse_sam_video(stream_events: AsyncIterable[sam.ResponsesEventLike]) -> list[MaskMoments]:
    """Decode video masks with explicit frame indices and stable object IDs."""
    return await _parse_masks(stream_events, sam.video_segmentation_format(), "video")


async def _parse_masks(stream_events, response_format, kind):
    async with sam.parse_responses_stream(stream_events, response_format) as parsed:
        result = await parsed.final_result()
    if result.outcome.status != "completed":
        raise ValueError(f"Incomplete SAM {kind} response")
    if result.diagnostics:
        raise ValueError("SAM parser diagnostics: " + ", ".join(d.code for d in result.diagnostics))
    latest = {}
    for record in result.records:
        if isinstance(record, sam.SegmentationMaskRecord):
            latest[record.identity] = record
        elif isinstance(record, sam.SegmentationTextRecord) and record.text.strip():
            raise ValueError("Unexpected text in SAM segmentation response")
    return [mask_moments(record) for record in latest.values()]


@dataclass(frozen=True)
class Trajectory:
    object_id: str
    frame_indices: np.ndarray
    times: np.ndarray
    x: np.ndarray
    y_phys: np.ndarray  # Cartesian pixels, not meters
    observed: np.ndarray  # False for interpolated samples


def interpolate_trajectory(masks, fps, frame_height) -> dict[str, Trajectory]:
    """Group nonempty masks and fill internal gaps on each track's frame grid.

    No extrapolation beyond a track's first/last detection. Explicit frame
    indices determine timestamps; input order and missing lines do not.
    All internal gaps are filled, with provenance retained in ``observed``.
    """
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(frame_height) or frame_height <= 0:
        raise ValueError("fps and frame_height must be positive and finite")
    tracks = {}
    for mask in masks:
        if not mask.area or mask.xc is None or mask.yc is None:
            continue
        if mask.frame_index < 0 or not np.all(np.isfinite([mask.xc, mask.yc])):
            raise ValueError("Invalid mask centroid or frame index")
        track = tracks.setdefault(mask.object_id, {})
        if mask.frame_index in track:
            raise ValueError("Duplicate object/frame observation")
        track[mask.frame_index] = (mask.xc, frame_height - mask.yc)
    result = {}
    for object_id, track in tracks.items():
        frames = np.array(sorted(track), dtype=int)
        grid = np.arange(frames[0], frames[-1] + 1)
        values = np.array([track[frame] for frame in frames])
        coordinates = (CubicSpline(frames / fps, values, axis=0)(grid / fps)
                       if len(frames) > 1 else values)
        result[object_id] = Trajectory(
            object_id, grid, grid / fps, coordinates[:, 0], coordinates[:, 1],
            np.isin(grid, frames),
        )
    return result


@dataclass(frozen=True)
class PendulumPivot:
    x0: float
    y0: float
    L_pixels: float
    theta: np.ndarray

    def angles(self, x, y_phys):
        return np.arctan2(np.asarray(x) - self.x0, self.y0 - np.asarray(y_phys))


def fit_pendulum_pivot(x, y_phys) -> PendulumPivot:
    """Kasa least-squares circle fit in Cartesian pixels; theta is radians.

    Centering improves conditioning for large source-image coordinates.
    Collinear/constant data cannot identify a pivot and are rejected.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y_phys, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 3 or not np.all(np.isfinite([x, y])):
        raise ValueError("Expected at least three matching finite coordinates")
    mx, my = x.mean(), y.mean()
    u, v = x - mx, y - my
    design = np.column_stack((2 * u, 2 * v, np.ones(len(x))))
    solution, _, rank, _ = np.linalg.lstsq(design, u*u + v*v, rcond=None)
    if rank < 3:
        raise ValueError("A circle requires noncollinear observations")
    cx, cy, c = solution
    radius = np.sqrt(c + cx*cx + cy*cy)
    x0, y0 = float(mx + cx), float(my + cy)
    return PendulumPivot(x0, y0, float(radius), np.arctan2(x - x0, y0 - y))


def infer_pendulum_length(omega0_squared, g=9.81) -> float:
    """Infer meters from known gravity; this does not independently measure g."""
    if not np.all(np.isfinite([omega0_squared, g])) or omega0_squared <= 0 or g <= 0:
        raise ValueError("Frequency squared and gravity must be positive and finite")
    return float(g / omega0_squared)


def extract_median_background(video_path) -> np.ndarray:
    """Return an RGB uint8 temporal median from up to 30 uniform samples.

    A moving object must leave each pixel in most sampled frames for that
    pixel to reveal the background. Stationary scene elements remain.
    """
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            raise ValueError("Video has no frames")
        frames = []
        for index in np.linspace(0, count - 1, min(count, 30), dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            success, frame = capture.read()
            if not success:
                raise ValueError(f"Cannot decode video frame {index}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.median(frames, axis=0).astype(np.uint8)
    finally:
        capture.release()
