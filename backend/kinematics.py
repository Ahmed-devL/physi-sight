"""SAM mask moments, tracked Cartesian trajectories, and pendulum calibration."""

from dataclasses import dataclass
from collections.abc import AsyncIterable

import meta_sam_parser as sam
import numpy as np
from scipy.interpolate import CubicSpline
import cv2
import json
from pathlib import Path
import subprocess


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
        if len(frames) >= 5:
            # A spike has two large adjacent jumps and returns to the local path.
            # Divide by frame gaps so missing detections do not inflate the step.
            steps = np.linalg.norm(np.diff(values, axis=0), axis=1) / np.diff(frames)
            median = float(np.median(steps))
            threshold = 3.5 * max(median, 1e-6)
            reject = []
            for i in range(1, len(frames) - 1):
                if frames[i] - frames[i-1] != 1 or frames[i+1] - frames[i] != 1:
                    continue
                bridge = np.linalg.norm(values[i+1] - values[i-1]) / 2
                if steps[i-1] > threshold and steps[i] > threshold and bridge <= threshold:
                    reject.append(i)
            if reject:
                frames = np.delete(frames, reject)
                values = np.delete(values, reject, axis=0)
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


def select_primary_track(tracks):
    """Prefer valid coverage, then moving rather than stationary detections."""
    if not tracks:
        raise ValueError("SAM returned no nonempty tracks")
    # Exclude stationary background detections when a resolved moving track exists.
    moving = [tr for tr in tracks.values() if np.var(tr.x[tr.observed])
              + np.var(tr.y_phys[tr.observed]) > 1.0]
    return max(moving or tracks.values(), key=lambda tr: (
        int(np.sum(tr.observed)),
        float(np.var(tr.x[tr.observed]) + np.var(tr.y_phys[tr.observed])),
        str(tr.object_id),
    ))


def mask_size_metrics(masks, track):
    valid_frames = set(track.frame_indices[track.observed].tolist())
    selected = [m for m in masks if m.object_id == track.object_id
                and m.frame_index in valid_frames and m.area > 0]
    return {
        "median_box_width_px": float(np.median([m.box[2] - m.box[0] + 1 for m in selected])),
        "median_box_height_px": float(np.median([m.box[3] - m.box[1] + 1 for m in selected])),
        "median_area_px2": float(np.median([m.area for m in selected])),
        "equiv_diameter_px": float(2 * np.sqrt(np.median([m.area for m in selected]) / np.pi)),
    }


def normalize_video(input_path, output_path):
    """Normalize VFR/phone video, including rotation, to at most five seconds."""
    source, target = Path(input_path), Path(output_path)
    if source.resolve() == target.resolve():
        raise ValueError("Normalization requires a separate output path")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(source), "-t", "5.0",
            "-vf", "fps=30,scale=-2:720", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-an", str(target),
        ], check=True, capture_output=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required for video normalization") from error
    except subprocess.CalledProcessError as error:
        raise ValueError("Video normalization failed; upload a decodable MP4") from error
    return target


def is_normalized_video(path):
    """Check timestamps as well as nominal rate; average FPS alone misses VFR."""
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-read_intervals", "%+6", "-select_streams", "v:0", "-show_streams",
            "-show_frames", "-show_entries",
            "stream=codec_name,pix_fmt,height:frame=best_effort_timestamp_time",
            "-of", "json", str(path),
        ], check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        stamps = np.array([float(f["best_effort_timestamp_time"]) for f in data["frames"]])
        return (stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
                and stream["height"] == 720 and 20 <= len(stamps) <= 150
                and abs(stamps[0]) < 1e-4 and np.allclose(np.diff(stamps), 1/30, atol=2e-5))
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, IndexError):
        return False


def detect_release_index(times, observations, *, regime, noise_scale=0.015):
    """Find an initial hold or hand-driven approach to a release extremum.

    Inspect only the first 60% (strictly inside any subsequent training
    window). Require sustained displacement beyond tracking noise. A clip
    already moving freely keeps its first sample. Return an observation index,
    not a video frame number; callers map through the explicit frame indices.
    """
    from scipy.signal import savgol_filter, find_peaks

    y = np.asarray(observations, dtype=float)
    if regime not in ("pendulum", "drag", "drag_fall", "bouncing_ball"):
        return 0
    t = np.asarray(times, dtype=float)
    prefix = y[:int(len(y) * 0.6)]
    if len(prefix) < 10:
        return 0
    span = float(np.ptp(prefix))
    threshold = max(4 * noise_scale, .005 * span)
    baseline = float(np.median(prefix[:8]))
    release = 0
    if np.ptp(prefix[:8]) < max(6 * noise_scale, .002 * span):
        moving = np.abs(prefix - baseline) > threshold
        for i in range(8, len(prefix) - 2):
            if np.all(moving[i:i+3]):
                # Retain the transition shoulder; shooting refines its state.
                release = max(0, i - 2)
                break
    if regime == "pendulum":
        smooth = savgol_filter(prefix, 5, 2)
        # A significant initial rise/fall followed by a reversal indicates
        # hand positioning before release. Do not skip an ordinary first
        # half-cycle when the clip already starts at its release extremum.
        peaks = sorted([*find_peaks(smooth)[0], *find_peaks(-smooth)[0]])
        for peak in peaks:
            if peak < 2 or peak > len(prefix) // 2 or t[peak] - t[release] > 0.2:
                continue
            approach = smooth[peak] - smooth[0]
            reversal = smooth[peak] - smooth[min(len(prefix)-1, peak+4)]
            if abs(approach) > max(2 * noise_scale, .025 * span) and approach * reversal > 0 and abs(reversal) > threshold:
                return int(peak)
            break
    return release
