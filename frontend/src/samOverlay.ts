import type { Analysis } from './api';

type SAMFrame = NonNullable<Analysis['segmentation_overlays']>['frames'][number];
export function prepareSAMFrames(frames: SAMFrame[]) {
  return new Map(frames.map(frame => {
    const contour = new Path2D();
    frame.polygons.forEach(polygon => {
      polygon.points.forEach(([x, y], i) => { if (i) contour.lineTo(x, y); else contour.moveTo(x, y); });
      contour.closePath();
    });
    return [frame.frame_idx, { ...frame, contour }] as const;
  }));
}

export function mediaTime(video: Pick<HTMLVideoElement, 'paused' | 'currentTime' | 'requestVideoFrameCallback'> | null, presented: number) {
  return video?.paused || !video?.requestVideoFrameCallback ? video?.currentTime ?? 0 : presented;
}

export function frameAtMediaTime(stamps: number[], time: number) {
  // Metadata PTS and contour indices both refer to the analyzed playback asset.
  // Original-source trim offsets and fitted-trajectory/release offsets do not apply.
  let low = 0, high = stamps.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (stamps[middle] <= time + .00001) low = middle + 1; else high = middle;
  }
  return low - 1;
}

export function renderSAMFrame(ctx: CanvasRenderingContext2D,
  frame: (SAMFrame & { contour: Path2D }) | undefined,
  masks: boolean, boxes: boolean, sx: number) {
  if (frame?.source !== 'sam') return;
  if (masks && frame.status === 'observed_sam_mask') {
    ctx.fillStyle = '#24b5a533'; ctx.fill(frame.contour, 'evenodd');
    ctx.strokeStyle = '#168d80'; ctx.lineWidth = 1.5 / sx; ctx.stroke(frame.contour);
  }
  if (boxes) {
    const [x1, y1, x2, y2] = frame.box;
    ctx.strokeStyle = '#168d80'; ctx.lineWidth = 1 / sx;
    ctx.strokeRect(x1, y1, x2-x1+1, y2-y1+1);
  }
}
