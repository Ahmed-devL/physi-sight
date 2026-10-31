import asyncio

import numpy as np
import pytest

from backend.kinematics import parse_sam_frame


async def events(text, *, completed=True):
    lane = dict(item_id="mask", output_index=0, content_index=0)
    # Split inside both the header and the base85 payload.
    for start in range(0, len(text), 7):
        yield dict(type="response.output_text.delta", delta=text[start:start + 7], **lane)
    yield dict(type="response.output_text.done", text=text, **lane)
    if completed:
        yield dict(type="response.completed")


def wire(payload="!!!!!&w{Dcs", *, height=2, frame=0):
    return (f"<{frame}f>7<|box;x1=10;y1=20;x2=12;y2={19 + height};w=64;h=64|>"
            f"<|mask;x=0;y=0;data={height},3,{payload}|>")


def test_chunked_mask_offsets_and_moments():
    mask, = asyncio.run(parse_sam_frame(events(wire())))
    assert mask.box == (10, 20, 12, 21)
    assert (mask.xc, mask.yc, mask.area) == (11, 20.5, 6)
    assert (mask.mu_20, mask.mu_02, mask.mu_11) == (4, 1.5, 0)
    assert mask.theta == 0
    assert mask.object_id == "7"
    assert mask.frame_index == 0


def test_diagonal_orientation():
    mask, = asyncio.run(parse_sam_frame(events(wire("!!!!!'Qq$N(!!", height=3))))
    assert mask.theta == pytest.approx(np.pi / 4)
    assert (mask.xc, mask.yc) == (11, 21)


def test_empty_mask_and_empty_frame():
    mask, = asyncio.run(parse_sam_frame(events(wire("!!!!!&!!!!!"))))
    assert mask.area == 0
    assert mask.xc is mask.yc is mask.theta is None
    assert asyncio.run(parse_sam_frame(events(""))) == []


@pytest.mark.parametrize("text", [wire(frame=3), wire("!invalid"), "unstructured response"])
def test_reject_malformed_or_wrong_frame(text):
    with pytest.raises(ValueError):
        asyncio.run(parse_sam_frame(events(text)))


def test_reject_incomplete():
    with pytest.raises(ValueError, match="Incomplete"):
        asyncio.run(parse_sam_frame(events(wire(), completed=False)))


def test_latest_revision_only():
    masks = asyncio.run(parse_sam_frame(events(wire() + "\n" + wire("!!!!!&!!!!!"))))
    assert len(masks) == 1
    assert masks[0].area == 0
