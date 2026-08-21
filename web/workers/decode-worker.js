/* ---------------------------------------------------------------------------
   The decode worker: a module Worker whose whole job is to keep the demuxer,
   VideoDecoder and the JPEG encoder off the main thread.

   The main thread hands over the file's bytes (transferred, so nothing is
   copied) and the grid of instants it wants; blobs come back one at a time as
   they are made, so the progress line moves and the UI stays interactive for
   the whole decode. Blobs cross a postMessage by reference — the pixels are
   not serialised.
--------------------------------------------------------------------------- */
'use strict';

/* When this worker's own module graph finished loading. The main thread cannot
 * see it any other way, and "how long did the worker take to boot" is a real
 * part of a one-second decode. */
const BOOT = performance.now();

import { demux, decodeToJpeg } from './decode-core.js';

self.onmessage = async (ev) => {
  const m = ev.data || {};
  if (m.type !== 'decode') return;
  try {
    const t0 = performance.now();
    const bytes = new Uint8Array(m.buffer);
    const clip = demux(bytes);
    const demuxMs = performance.now() - t0;
    const stats = await decodeToJpeg({
      bytes, clip, targets: m.targets, w: m.w, h: m.h,
      quality: m.quality, hardware: m.hardware,
      emit: (i, blob) => { self.postMessage({ type: 'frame', i, blob }); },
      onProgress: (n) => { self.postMessage({ type: 'progress', done: n }); },
    });
    self.postMessage({ type: 'done', stats, bootMs: BOOT, demuxMs,
                       accel: stats.accel,
                       codec: clip.codec, container: clip.container,
                       samples: clip.samples.length });
  } catch (e) {
    self.postMessage({ type: 'error', message: (e && e.message) || String(e) });
  }
};
