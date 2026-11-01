import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import vm from 'node:vm';
import { buildSync } from 'esbuild';

class Path {
  moveTo() {} lineTo() {} closePath() {}
}
const bundle = buildSync({ entryPoints: [new URL('../src/samOverlay.ts', import.meta.url).pathname], bundle: true,
  platform: 'node', format: 'cjs', write: false }).outputFiles[0].text;
const module = { exports: {} };
vm.runInNewContext(bundle, { module, exports: module.exports, Path2D: Path });
const { prepareSAMFrames, frameAtMediaTime, mediaTime, renderSAMFrame } = module.exports;
const times = [0, .5, 1, 1.5, 2, 2.49, 2.5, 3];
function render(frames, stamps, video, presented = 0) {
  let fills = 0, boxes = 0;
  const ctx = { fill() { fills++; }, stroke() {}, strokeRect() { boxes++; } };
  const index = frameAtMediaTime(stamps, mediaTime(video, presented));
  renderSAMFrame(ctx, frames.get(index), true, true, 1);
  return { index, fills, boxes };
}
function fixture(count, start = 0) {
  return { stamps: Array.from({ length: count }, (_, i) => i/30),
    frames: prepareSAMFrames(Array.from({ length: count-start }, (_, i) => ({
      frame_idx: start+i, source: 'sam', status: 'observed_sam_mask', box: [100, 200, 110, 210],
      polygons: [{ points: [[100, 200], [110, 200], [110, 210]], hole: false }],
    }))) };
}
const video = time => ({ paused: true, currentTime: time, requestVideoFrameCallback() {} });
test('paused seeks render real masks from frame zero, independently of release or tracking start', () => {
  const { frames, stamps } = fixture(120);
  for (const time of times) {
    const result = render(frames, stamps, video(time), 3);
    assert.equal(result.index, Math.floor(time*30 + .00001));
    assert.equal(result.fills, 1, `contour at ${time}s`);
    assert.equal(result.boxes, 1);
  }
  assert.equal(render(frames, stamps, video(0), 3).fills, 1, 'seek back to zero ignores stale presented time');
});
test('playback from the beginning follows presented frame timestamps', () => {
  const { frames, stamps } = fixture(120);
  for (const time of times) {
    assert.equal(render(frames, stamps, { ...video(3), paused: false }, time).fills, 1);
    assert.equal(render(frames, stamps, { ...video(time), paused: false, requestVideoFrameCallback: undefined }).fills, 1);
  }
});
test('frame-75 coverage reproduces the threshold without inventing interpolated masks', () => {
  const { frames, stamps } = fixture(135, 75);
  for (const time of times) assert.equal(render(frames, stamps, video(time)).fills, time < 2.5 ? 0 : 1, `${time}s`);
});
for (const [hash, clip, first, count] of [
  ['676505835354255c2b40dd278207d2014d58713152213401fc0056c905f0747c', 'damped_pendulum.mp4', 0, 120],
  ['b57c8a7cf6ed09347e63dc51af7387d55f0c7a20437466fd7d5662e4335ace08', 'youtube_pendulum.mp4', 75, 135],
]) {
  const cache = new URL(`../../backend/cache/${hash}.json`, import.meta.url);
  test(`local artifact ${clip}: hash and paused-seek reproduction`, { skip: !existsSync(cache) }, () => {
    const data = JSON.parse(readFileSync(cache, 'utf8'));
    const analysis = data.analyses.meta;
    assert.equal(analysis.video_sha256, hash);
    assert.equal(createHash('sha256').update(readFileSync(new URL(`../../backend/demo_clips/${clip}`, import.meta.url))).digest('hex'), hash);
    assert.equal(data.video_metadata.frames, count);
    const frames = prepareSAMFrames(analysis.segmentation_overlays.frames);
    for (const time of times) assert.equal(render(frames, data.video_metadata.timestamps, video(time)).fills, time*30 < first ? 0 : 1, `${clip} ${time}s`);
  });
}
