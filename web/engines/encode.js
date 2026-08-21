/* ---------------------------------------------------------------------------
   ENCODING A CLIP IN THE TAB — WebCodecs, with the timestamps written down.

   THE BUG THIS FILE EXISTS TO KILL. The old export was `MediaRecorder` over
   `canvas.captureStream()`: paint a frame, hand it over, paint the next. A
   recorder stamps a frame with the moment it ARRIVES, so the file's time base
   was the wall clock of the render. A 3.0 s clip whose dots took 84 s to draw
   came out as an 84 s file — every frame there, every frame in the wrong
   place. The page even said so, which made it a documented defect rather than
   a hidden one.

   `VideoEncoder` takes the timestamp as an argument. Frame i is stamped
   `i * 1e6 / fps` microseconds and carries `1e6 / fps` of duration, so the
   output is exactly `nFrames / fps` seconds long whether the render took two
   seconds or two minutes. Nothing about the encode is paced; it runs as fast
   as the machine will go.

   WHAT COMES OUT. The encoder is asked what it can do (`isConfigSupported`)
   and the first working codec for the container wins:

     webm   VP9 -> VP8 -> AV1        muxed by vendor/webm-muxer.js
     mp4    H.264 -> HEVC -> AV1     muxed by vendor/mp4-muxer.js

   Safari has no VP9/VP8 encoder and Chrome may have no H.264 one; that is why
   both containers are probed and why a container that cannot be written says
   so in the format list instead of failing at the export button.

   WHAT IS NOT HERE. Alpha. A transparent WebM carries its alpha plane as a
   Matroska BlockAdditional holding a second, separately-coded bitstream, and
   `VideoEncoder` will not produce one — `alpha: 'keep'` is Chrome-only and
   does not hand back anything a muxer can use. The alpha export therefore
   stays on `MediaRecorder`, and its timing is fixed the other way: render
   every frame first, then REPLAY the finished frames to the recorder at the
   clip's own rate. See `replayToRecorder` in engines/browser.js.

   MUXERS. web/vendor/webm-muxer.js and web/vendor/mp4-muxer.js are Vanilagy's
   packages, MIT, vendored verbatim from npm (see NOTICE). Nothing is fetched
   from a CDN; they are imported lazily so a page that only dithers stills
   never pays for them.
--------------------------------------------------------------------------- */
'use strict';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* The candidates, best first, per container.
 *
 *   `codec`  goes to VideoEncoder.configure
 *   `mux`    is what the muxer calls the same thing: Matroska codec IDs for
 *            WebM, mp4-muxer's short names for MP4
 *   `label`  is what the UI shows once one has been picked
 *
 * H.264 is listed at several levels because `isConfigSupported` answers for a
 * SIZE: avc1.42001f is level 3.1, which tops out around 720p, and a 1080x1920
 * export needs 4.0 or better. Asking in descending order and taking the first
 * yes is cheaper than working the level tables out here.
 */
const CANDIDATES = {
  webm: [
    { codec: 'vp09.00.10.08', mux: 'V_VP9', label: 'VP9' },
    { codec: 'vp8', mux: 'V_VP8', label: 'VP8' },
    { codec: 'av01.0.04M.08', mux: 'V_AV1', label: 'AV1' },
  ],
  mp4: [
    { codec: 'avc1.640034', mux: 'avc', label: 'H.264' },   // High 5.2
    { codec: 'avc1.640028', mux: 'avc', label: 'H.264' },   // High 4.0
    { codec: 'avc1.4d0028', mux: 'avc', label: 'H.264' },   // Main 4.0
    { codec: 'avc1.42e01f', mux: 'avc', label: 'H.264' },   // Baseline 3.1
    { codec: 'hvc1.1.6.L123.B0', mux: 'hevc', label: 'HEVC' },
    { codec: 'hvc1.1.6.L93.B0', mux: 'hevc', label: 'HEVC' },
    { codec: 'av01.0.04M.08', mux: 'av1', label: 'AV1' },
  ],
};

export const CONTAINERS = {
  webm: { ext: 'webm', mime: 'video/webm' },
  mp4: { ext: 'mp4', mime: 'video/mp4' },
};

export function hasVideoEncoder() {
  return typeof VideoEncoder !== 'undefined'
    && typeof VideoEncoder.isConfigSupported === 'function'
    && typeof VideoFrame !== 'undefined';
}

/** Bits per second for a w x h clip at `fps`. The same curve the recorder path
 *  used, so a WebM does not suddenly change size because the encoder did. */
export function bitrateFor(w, h, fps) {
  return Math.round(Math.min(24e6, Math.max(4e6, w * h * Math.max(1, fps) * 0.15)));
}

/* Codecs are probed per (container, size) and the answer cached: a probe is a
 * round trip into the platform encoder and the format list asks for one every
 * time it is painted. */
const PROBED = new Map();

function encoderConfig(cand, w, h, fps) {
  const cfg = {
    codec: cand.codec,
    width: w,
    height: h,
    bitrate: bitrateFor(w, h, fps),
    framerate: Math.max(1, fps),
  };
  // AVCC / HVCC, not Annex B: both muxers want the parameter sets in a
  // `description`, which is what the 'avc'/'hevc' formats produce.
  if (cand.mux === 'avc') cfg.avc = { format: 'avc' };
  if (cand.mux === 'hevc') cfg.hevc = { format: 'hevc' };
  return cfg;
}

/** The first codec this browser will actually encode `w x h` with, or null.
 *  Dimensions matter — an encoder can support H.264 and refuse 1080x1920 at
 *  the level asked for — so the probe is done at the size that will be used. */
export async function pickCodec(container, w, h, fps) {
  if (!hasVideoEncoder()) return null;
  // even dimensions: every codec here is 4:2:0 and rejects an odd side
  const cw = Math.max(2, w + (w & 1)), ch = Math.max(2, h + (h & 1));
  const key = `${container}|${cw}x${ch}|${Math.round(fps)}`;
  if (PROBED.has(key)) return PROBED.get(key);
  let hit = null;
  for (const cand of CANDIDATES[container] || []) {
    try {
      const r = await VideoEncoder.isConfigSupported(
        encoderConfig(cand, cw, ch, fps));
      if (r && r.supported) { hit = cand; break; }
    } catch (e) { /* a codec string this build has never heard of */ }
  }
  PROBED.set(key, hit);
  return hit;
}

/** What both containers can do at a nominal size, for the format list. 1080p
 *  is the probe size because it is the level boundary that actually decides
 *  the answer, and because the list is painted before any clip is open. */
export async function encoderSupport(w, h, fps) {
  const W = w || 1920, H = h || 1080, F = fps || 30;
  const [webm, mp4] = await Promise.all([
    pickCodec('webm', W, H, F), pickCodec('mp4', W, H, F),
  ]);
  return { webm, mp4, available: hasVideoEncoder() };
}

/* A canvas the frames are painted on before they become VideoFrames.
 * `new VideoFrame(canvas, ...)` is supported everywhere VideoEncoder is;
 * constructing one straight out of an RGBA buffer is not, and the copy is
 * cheap next to the encode. */
function makeCanvas(w, h) {
  if (typeof OffscreenCanvas === 'function') {
    const c = new OffscreenCanvas(w, h);
    // an OffscreenCanvas with no 2d context yet is not a valid VideoFrame source
    return { canvas: c, ctx: c.getContext('2d', { alpha: false, willReadFrequently: false }) };
  }
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return { canvas: c, ctx: c.getContext('2d', { alpha: false }) };
}

/**
 * Render `nFrames` frames and encode them into one file.
 *
 * @param {object} o
 *   o.w, o.h, o.fps      the output grid
 *   o.from, o.nFrames    the window of the clip
 *   o.container          'webm' | 'mp4'
 *   o.renderFrame(i,w,h) -> ImageData, awaited
 *   o.onProgress({done,total,text})
 *   o.slowMs             test hook: sleep this long after every render, to
 *                        make a render that is slower than real time on
 *                        purpose. Nothing about the OUTPUT may change.
 * @returns {Promise<{blob, ext, mime, codec, label, frames, durationS}>}
 */
export async function encodeClip(o) {
  const { w, h, fps, from, nFrames, container, renderFrame, onProgress } = o;
  const cand = await pickCodec(container, w, h, fps);
  if (!cand) {
    throw new Error(`this browser has no ${container === 'mp4' ? 'H.264/HEVC' : 'VP9/VP8/AV1'}`
      + ' encoder in WebCodecs');
  }
  const mod = container === 'mp4'
    ? await import('../vendor/mp4-muxer.js')
    : await import('../vendor/webm-muxer.js');

  // 4:2:0 wants even sides. The dither grid is whatever the canvas preset says,
  // so pad by one rather than refuse the export; the extra column/row is the
  // edge pixel repeated and no player shows it as anything else.
  const ew = Math.max(2, w + (w & 1)), eh = Math.max(2, h + (h & 1));

  const target = new mod.ArrayBufferTarget();
  const muxer = new mod.Muxer(container === 'mp4'
    ? { target,
        video: { codec: cand.mux, width: ew, height: eh, frameRate: fps },
        // metadata at the front: a blob: URL in a <video> cannot seek to the
        // end of the file to find a moov that is written last
        fastStart: 'in-memory' }
    : { target,
        video: { codec: cand.mux, width: ew, height: eh, frameRate: fps },
        type: 'webm' });

  let failure = null;
  const enc = new VideoEncoder({
    output: (chunk, meta) => {
      try { muxer.addVideoChunk(chunk, meta); }
      catch (e) { failure = failure || e; }
    },
    error: (e) => { failure = failure || e; },
  });
  enc.configure(encoderConfig(cand, ew, eh, fps));

  const { canvas, ctx } = makeCanvas(ew, eh);
  // one keyframe every two seconds, and always the first
  const gop = Math.max(1, Math.round(Math.max(1, fps) * 2));
  const usPerFrame = 1e6 / Math.max(1e-6, fps);
  const dur = Math.round(usPerFrame);
  const slowMs = Math.max(0, +o.slowMs || 0);
  const t0 = (typeof performance !== 'undefined' ? performance : Date).now();

  try {
    for (let i = 0; i < nFrames; i++) {
      const img = await renderFrame(from + i, w, h);
      if (slowMs) await sleep(slowMs);
      ctx.putImageData(img, 0, 0);
      // the padded column first (from real pixels), then the padded row across
      // the whole width, so the corner comes from a column that is already there
      if (ew !== w) ctx.drawImage(canvas, w - 1, 0, 1, h, w, 0, 1, h);
      if (eh !== h) ctx.drawImage(canvas, 0, h - 1, ew, 1, 0, h, ew, 1);
      /* THE WHOLE POINT. The timestamp is computed from the frame's index and
       * the clip's rate — never from a clock — so the file is nFrames/fps long
       * however long this loop took. */
      const frame = new VideoFrame(canvas, {
        timestamp: Math.round(i * usPerFrame), duration: dur,
      });
      enc.encode(frame, { keyFrame: i % gop === 0 });
      frame.close();
      if (failure) throw failure;
      // let the encoder drain rather than queueing a whole clip of raw frames
      while (enc.encodeQueueSize > 6 && !failure) await sleep(2);
      if (onProgress) {
        onProgress({ done: i + 1, total: nFrames, text: `${i + 1}/${nFrames}` });
      }
    }
    if (onProgress) {
      onProgress({ done: nFrames, total: nFrames, text: 'finishing the file…' });
    }
    await enc.flush();
    if (failure) throw failure;
    muxer.finalize();
  } finally {
    try { if (enc.state !== 'closed') enc.close(); } catch (e) { /* already gone */ }
  }

  const C = CONTAINERS[container];
  const blob = new Blob([target.buffer], { type: C.mime });
  return {
    blob, ext: C.ext, mime: C.mime,
    codec: cand.codec, label: cand.label,
    frames: nFrames,
    durationS: nFrames / Math.max(1e-6, fps),
    padded: ew !== w || eh !== h,
    elapsedS: ((typeof performance !== 'undefined' ? performance : Date).now() - t0) / 1000,
  };
}
