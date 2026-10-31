"""Image SAM masks and moments in source-image coordinates (positive y down)."""

from dataclasses import dataclass
from collections.abc import AsyncIterable

import meta_sam_parser as sam
import numpy as np


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
    async with sam.parse_responses_stream(stream_events, sam.image_segmentation_format()) as parsed:
        result = await parsed.final_result()
    if result.outcome.status != "completed":
        raise ValueError("Incomplete SAM image response")
    if result.diagnostics:
        raise ValueError("SAM parser diagnostics: " + ", ".join(d.code for d in result.diagnostics))
    latest = {}
    for record in result.records:
        if isinstance(record, sam.SegmentationMaskRecord):
            latest[record.identity] = record
        elif isinstance(record, sam.SegmentationTextRecord) and record.text.strip():
            raise ValueError("Unexpected text in SAM segmentation response")
    return [mask_moments(record) for record in latest.values()]
