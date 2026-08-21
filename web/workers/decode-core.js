/* ---------------------------------------------------------------------------
   WebCodecs decode, as a function that runs in a Worker or on the main thread.

   The <video> seek loop it replaces is one `currentTime =` and one `seeked`
   event PER FRAME: the browser re-primes the decoder, walks back to a
   keyframe and re-decodes the gap every time, which is why 150 frames cost
   ~5 s and 2,700 cost ~5.5 minutes. A demuxer plus VideoDecoder decodes the
   stream once, in order, which is what the codec was designed for.

   Frame identity is preserved on purpose. The clip's frame `i` is still the
   picture on screen at t0 + (i + 0.5)/fps -- the last sample whose
   presentation time is at or before that instant -- so both paths index the
   same picture as frame 42 and the server's ffmpeg `-r 30` grid still lines up.

   Timestamps are the join: an EncodedVideoChunk's timestamp comes back on the
   VideoFrame it produced, so output order does not matter and B-frame
   reordering needs no buffer here.
--------------------------------------------------------------------------- */
'use strict';

import { demuxMp4, looksLikeMp4 } from './demux-mp4.js';
import { demuxWebM, looksLikeWebM } from './demux-webm.js';

export function demux(bytes) {
  if (looksLikeMp4(bytes)) return demuxMp4(bytes);
  if (looksLikeWebM(bytes)) return demuxWebM(bytes);
  throw new Error('not an MP4 or a WebM');
}

/** The samples in PRESENTATION order. A demuxer hands them over in decode
 *  order, which is the order the decoder must be fed in and not the order the
 *  pictures appear in. */
export function presentationOrder(samples) {
  return samples.map((s, i) => i)
    .sort((a, b) => samples[a].pts - samples[b].pts || a - b);
}

/** Which picture each target instant lands on, as a position in `pts` (an
 *  ascending list): the last one at or before it, and the first picture for an
 *  instant that precedes it. Both lists ascend, so this is one walk. */
export function pickSamples(pts, targets) {
  const out = new Int32Array(targets.length);
  let s = 0;
  for (let i = 0; i < targets.length; i++) {
    while (s + 1 < pts.length && pts[s + 1] <= targets[i]) s++;
    out[i] = s;
  }
  return out;
}

/* Holding VideoFrames is the way to stall a decoder: the hardware decode path
 * has a small pool of picture buffers and stops producing output while they
 * are all checked out. So a frame is drawn onto a canvas and closed in the
 * same turn it arrives, and the feed loop is gated on how many are still out
 * rather than on how far ahead the decode queue is. */
const OUTSTANDING = 4;         // decoded frames not yet drawn (and not closed)
const QUEUED = 8;              // chunks handed to the decoder but not decoded
const CANVASES = 6;            // concurrent JPEG encodes
const STALL_MS = 20000;        // no progress for this long is a dead decoder

/* A zero-delay macrotask. `setTimeout(0)` is clamped to 4 ms once timers nest,
 * and this loop's every wait is scheduled from inside the previous one, so a
 * timer-based yield would cap the feed at 250 frames a second. A MessagePort
 * message is a task with no clamp. */
const CHAN = typeof MessageChannel !== 'undefined' ? new MessageChannel() : null;
const TICKS = [];
if (CHAN) {
  CHAN.port1.onmessage = () => { const f = TICKS.shift(); if (f) f(); };
  CHAN.port1.start();
}
const tick = () => new Promise((ok) => {
  if (!CHAN) return setTimeout(ok, 0);
  TICKS.push(ok); CHAN.port2.postMessage(0);
});

const now = () => (globalThis.performance || Date).now();

/**
 * Decode `targets` (seconds) out of a demuxed clip into JPEG blobs at w x h.
 *
 *   emit(i, blob)   called once per target, in whatever order they finish
 *   onProgress(n)   how many targets are done
 *
 * Returns { ms, decoded, samples, wanted }.
 */
export async function decodeToJpeg({ bytes, clip, targets, w, h, quality = 0.92,
                                     hardware = 'prefer-hardware',
                                     emit, onProgress, signal }) {
  const t0 = now();
  const order = presentationOrder(clip.samples);
  const pick = pickSamples(order.map((k) => clip.samples[k].pts), targets);
  const wanted = new Map();                       // sample index -> [target...]
  for (let i = 0; i < pick.length; i++) {
    const s = order[pick[i]];
    const a = wanted.get(s);
    if (a) a.push(i); else wanted.set(s, [i]);
  }
  // the span to feed, in DECODE order, from the sync sample that opens it
  let first = Infinity, last = -1;
  for (const s of wanted.keys()) { if (s < first) first = s; if (s > last) last = s; }
  let start = first;
  while (start > 0 && !clip.samples[start].key) start--;

  // timestamp (microseconds) -> sample, and the sorted list of them, so a
  // decoder that rounds can still be matched to the sample it came from
  const us = new Float64Array(clip.samples.length);
  const byTs = new Map();
  for (let s = 0; s < clip.samples.length; s++) {
    us[s] = Math.round(clip.samples[s].pts * 1e6);
    byTs.set(us[s], s);
  }
  const sortedUs = order.map((k) => us[k]);
  const nearest = (t) => {
    let lo = 0, hi = sortedUs.length - 1, best = -1, bd = Infinity;
    while (lo <= hi) {
      const m = (lo + hi) >> 1, d = Math.abs(sortedUs[m] - t);
      if (d < bd) { bd = d; best = order[m]; }
      if (sortedUs[m] < t) lo = m + 1; else hi = m - 1;
    }
    return bd <= 2000 ? best : -1;                 // within 2 ms, or not ours
  };

  // a small ring of canvases: convertToBlob is async, so the canvas it read
  // must not be redrawn until it resolves
  const free = [];
  for (let i = 0; i < CANVASES; i++) {
    const cv = new OffscreenCanvas(w, h);
    free.push({ cv, g: cv.getContext('2d', { alpha: false, willReadFrequently: false }) });
  }

  const queue = [];              // [sampleIdx, VideoFrame] waiting for a canvas
  const jobs = new Set();
  let done = 0, decoded = 0, fatal = null, moved = now();
  let drawMs = 0, jpegMs = 0, firstOut = 0;
  let wake = null;
  const bump = () => { moved = now(); if (wake) { const f = wake; wake = null; f(); } };

  /** Draw everything that has a canvas waiting for it, closing frames as it
   *  goes. Synchronous up to the point the pixels are safe. */
  function pump() {
    while (queue.length && free.length) {
      const [s, frame] = queue.shift();
      const c = free.pop();
      const tDraw = now();
      try { c.g.drawImage(frame, 0, 0, w, h); } finally { frame.close(); }
      drawMs += now() - tDraw;
      const idxs = wanted.get(s);
      const tEnc = now();
      const job = c.cv.convertToBlob({ type: 'image/jpeg', quality })
        .then((blob) => {
          jpegMs += now() - tEnc;
          // a Blob is immutable, so a picture that two grid instants land on
          // (a 24 fps clip resampled to 30) is handed out twice, not copied
          for (const i of idxs) { emit(i, blob); done++; }
          if (onProgress) onProgress(done);
        })
        .catch((e) => { fatal = fatal || e; })
        .finally(() => { free.push(c); jobs.delete(job); bump(); pump(); });
      jobs.add(job);
    }
  }

  const decoder = new VideoDecoder({
    output: (frame) => {
      decoded++;
      if (!firstOut) firstOut = now();
      let s = byTs.has(frame.timestamp) ? byTs.get(frame.timestamp)
        : nearest(frame.timestamp);
      if (s < 0 || !wanted.has(s)) { frame.close(); bump(); return; }
      queue.push([s, frame]);
      bump();
      pump();
    },
    error: (e) => { fatal = fatal || e; bump(); },
  });

  /* `prefer-hardware` is not a hint to Chrome: isConfigSupported answers false
   * when no hardware decoder exists for the codec, which is exactly how a
   * machine with no VP8 block in silicon reports itself. Asking twice is
   * therefore both the way to get a decoder AND an honest answer to "was this
   * hardware?", which nothing else in the API will tell you. */
  const cfg = { codec: clip.codec, optimizeForLatency: false };
  // codedWidth/codedHeight are left out on purpose: tkhd carries the DISPLAY
  // size, which is not the coded size when the pixels are not square, and a
  // wrong hint is a configure() failure rather than a correction.
  if (clip.description) cfg.description = clip.description;
  let accel = '';
  for (const a of (hardware === 'prefer-hardware'
    ? ['prefer-hardware', 'no-preference'] : [hardware || 'no-preference'])) {
    const probe = Object.assign({ hardwareAcceleration: a }, cfg);
    let ok = false;
    try { ok = !!(await VideoDecoder.isConfigSupported(probe)).supported; }
    catch (e) { ok = false; }
    if (ok) { cfg.hardwareAcceleration = a; accel = a === 'prefer-hardware' ? 'hardware' : 'software'; break; }
  }
  if (!accel) throw new Error('no decoder for ' + clip.codec);
  decoder.configure(cfg);

  const stalled = () => now() - moved > STALL_MS;
  const waitRoom = async () => {
    while (queue.length > OUTSTANDING || jobs.size >= CANVASES
           || decoder.decodeQueueSize > QUEUED) {
      if (fatal) throw fatal;
      if (stalled()) throw new Error('the decoder stopped producing frames');
      await Promise.race([tick(), new Promise((ok) => { wake = ok; })]);
    }
  };

  try {
    for (let s = start; s <= last; s++) {
      if (fatal) throw fatal;
      if (signal && signal.aborted) throw new Error('decode cancelled');
      await waitRoom();
      const smp = clip.samples[s];
      decoder.decode(new EncodedVideoChunk({
        type: smp.key ? 'key' : 'delta',
        timestamp: us[s],
        duration: Math.max(1, Math.round((smp.dur || 0) * 1e6)),
        data: bytes.subarray(smp.offset, smp.offset + smp.size),
      }));
    }
    await decoder.flush();
    while (queue.length || jobs.size) {
      if (fatal) throw fatal;
      if (stalled()) throw new Error('the JPEG encoder stopped');
      pump();
      await Promise.race([tick(), new Promise((ok) => { wake = ok; })]);
    }
    if (fatal) throw fatal;
    if (done !== targets.length) {
      throw new Error(`decoded ${done} of ${targets.length} frames`);
    }
  } finally {
    try { decoder.close(); } catch (e) { /* already closed by the error path */ }
    for (const [, frame] of queue) frame.close();
    queue.length = 0;
  }
  return { ms: now() - t0, decoded, accel, samples: last - start + 1,
           wanted: wanted.size,
           drawMs: Math.round(drawMs), jpegMs: Math.round(jpegMs),
           firstFrameMs: firstOut ? Math.round(firstOut - t0) : 0 };
}
