/* ---------------------------------------------------------------------------
   DECODING A CLIP IN THE TAB — three paths, one frame grid.

   What every path produces is identical and that is the point: `n` JPEG blobs
   where frame `i` is the picture on screen at t0 + (i + 0.5)/fps, `n` and the
   decode size taken from the same `<video>` metadata they always were. The
   server's ffmpeg `-r 30` grid, the mask indices, the trim windows and every
   frame number the suites assert on are unchanged; only how the pixels are
   obtained differs.

     webcodecs-worker   a module Worker demuxes the file, runs VideoDecoder
                        once over the stream in order, and encodes the JPEGs
                        on four OffscreenCanvases. Nothing touches the main
                        thread but the finished blobs.
     webcodecs-main     the same code with no Worker, for a browser that has
                        VideoDecoder but not OffscreenCanvas in workers.
     video-seek         the original: `currentTime =` and wait for `seeked`,
                        once per frame, on the main thread. Every browser has
                        it; it is ~8x slower and it blocks.

   The chain is feature-detected and then verified by trying: a container this
   demuxer does not understand, or a codec `VideoDecoder.isConfigSupported`
   turns down, falls through to the seek path with the reason recorded in
   `stats.note` rather than failing the open.

   `globalThis.DV_DECODE_PATH` forces one of them. That exists so the
   verification suite can decode the same clip both ways and compare the frames
   it gets; a forced path does not fall back, because a silent fallback in a
   measurement is a lie.
--------------------------------------------------------------------------- */
'use strict';

export const DECODE_LABEL = {
  'webcodecs-worker': 'WebCodecs · worker',
  'webcodecs-main': 'WebCodecs · main thread',
  'video-seek': '<video> seek',
};

/** What this browser could do, before any file is looked at. */
export function decodeSupport() {
  const webcodecs = typeof VideoDecoder !== 'undefined'
    && typeof EncodedVideoChunk !== 'undefined'
    && typeof VideoDecoder.isConfigSupported === 'function';
  const offscreen = typeof OffscreenCanvas !== 'undefined'
    && typeof OffscreenCanvas.prototype.convertToBlob === 'function';
  const worker = typeof Worker !== 'undefined' && offscreen;
  return {
    webcodecs, offscreen, worker: webcodecs && worker,
    best: !webcodecs ? 'video-seek' : worker ? 'webcodecs-worker' : 'webcodecs-main',
  };
}

/* ---------------------------------------------------------------- helpers */

/** A stream whose header has no duration: seek past the end and read it back. */
function probeDuration(v) {
  return new Promise((ok) => {
    const done = () => {
      v.removeEventListener('durationchange', done);
      ok(isFinite(v.duration) && v.duration > 0 ? v.duration : 0);
    };
    v.addEventListener('durationchange', done);
    setTimeout(done, 3000);
    try { v.currentTime = 1e6; } catch (e) { done(); }
  });
}

function seek(v, t) {
  return new Promise((ok, no) => {
    const done = () => { v.removeEventListener('seeked', done); ok(); };
    v.addEventListener('seeked', done);
    const bail = setTimeout(() => { v.removeEventListener('seeked', done);
      no(new Error('seek stalled at ' + t.toFixed(3) + 's')); }, 20000);
    v.addEventListener('seeked', () => clearTimeout(bail), { once: true });
    v.currentTime = Math.max(0, t);
  });
}

const say = (onProgress, i, n) => {
  if (onProgress && (i % 5 === 0 || i === n - 1)) {
    onProgress({ done: i + 1, total: n, phase: 'decode',
                 text: `decoding ${i + 1}/${n} frames…` });
  }
};

/* ------------------------------------------------------- the seek loop --- */

async function viaSeek(v, { targets, w, h, onProgress }) {
  const cv = new OffscreenCanvas(w, h);
  const g = cv.getContext('2d', { willReadFrequently: false });
  const frames = new Array(targets.length);
  const t0 = performance.now();
  for (let i = 0; i < targets.length; i++) {
    await seek(v, targets[i]);
    g.drawImage(v, 0, 0, w, h);
    frames[i] = await cv.convertToBlob({ type: 'image/jpeg', quality: 0.92 });
    say(onProgress, i, targets.length);
  }
  return { frames, ms: performance.now() - t0, note: '' };
}

/* ------------------------------------------------------------ WebCodecs --- */

async function viaWorker(file, { targets, w, h, onProgress }) {
  const t0 = performance.now();
  const bytes = new Uint8Array(await file.arrayBuffer());
  const tRead = performance.now();
  const worker = new Worker(new URL('../workers/decode-worker.js', import.meta.url),
                            { type: 'module' });
  const frames = new Array(targets.length);
  try {
    const out = await new Promise((ok, no) => {
      worker.onerror = (e) => no(new Error(e.message || 'the decode worker failed'));
      worker.onmessageerror = () => no(new Error('the decode worker sent something unreadable'));
      worker.onmessage = (ev) => {
        const m = ev.data;
        if (m.type === 'frame') frames[m.i] = m.blob;
        else if (m.type === 'progress') say(onProgress, m.done - 1, targets.length);
        else if (m.type === 'done') ok(m);
        else if (m.type === 'error') no(new Error(m.message));
      };
      worker.postMessage({ type: 'decode', buffer: bytes.buffer,
                           targets: Array.from(targets), w, h,
                           quality: 0.92, hardware: 'prefer-hardware' },
                         [bytes.buffer]);
    });
    return { frames, ms: performance.now() - t0, accel: out.accel,
             note: `${out.container} · ${out.codec}`
               + (out.accel ? ' · ' + out.accel : ''),
             split: { readMs: Math.round(tRead - t0),
                      bootMs: Math.round(out.bootMs || 0),
                      demuxMs: Math.round(out.demuxMs || 0),
                      coreMs: Math.round(out.stats.ms),
                      drawMs: out.stats.drawMs, jpegMs: out.stats.jpegMs,
                      firstFrameMs: out.stats.firstFrameMs } };
  } finally { worker.terminate(); }
}

async function viaMain(file, { targets, w, h, onProgress }) {
  const { demux, decodeToJpeg } = await import('../workers/decode-core.js');
  const bytes = new Uint8Array(await file.arrayBuffer());
  const clip = demux(bytes);
  const frames = new Array(targets.length);
  const t0 = performance.now();
  const st = await decodeToJpeg({
    bytes, clip, targets: Array.from(targets), w, h,
    quality: 0.92, hardware: 'prefer-hardware',
    emit: (i, blob) => { frames[i] = blob; },
    onProgress: (n) => say(onProgress, n - 1, targets.length),
  });
  return { frames, ms: performance.now() - t0, accel: st.accel,
           note: `${clip.container} · ${clip.codec}`
             + (st.accel ? ' · ' + st.accel : '') };
}

/* ================================================================ entry === */

export async function decodeClip(file, { fps = 30, maxHeight = 720,
                                         trimStart = 0, trimEnd = null,
                                         onProgress, path } = {}) {
  const url = URL.createObjectURL(file);
  const v = document.createElement('video');
  v.preload = 'auto'; v.muted = true; v.playsInline = true; v.src = url;
  try {
    await new Promise((ok, no) => {
      v.onloadedmetadata = ok;
      v.onerror = () => no(new Error('this browser cannot decode that file'));
      setTimeout(() => no(new Error('timed out reading the video header')), 30000);
    });
    // MediaRecorder WebM (a camera recording) carries no duration until it has
    // been seeked past its end; do that before deciding how many frames there are
    let full = v.duration;
    if (!isFinite(full) || full <= 0) full = await probeDuration(v);
    const t0 = Math.max(0, trimStart || 0);
    const avail = Math.max(0, (full || 0) - t0);
    const want = trimEnd ? Math.max(0, trimEnd - t0) : avail;
    // no cap: the whole clip, or exactly the trim range
    const dur = Math.min(want || avail, avail);
    if (!isFinite(dur) || dur <= 0) throw new Error('the clip has no duration');
    const n = Math.max(1, Math.floor(dur * fps));
    const scale = Math.min(1, maxHeight / (v.videoHeight || maxHeight));
    const w = Math.max(2, Math.round(v.videoWidth * scale / 2) * 2);
    const h = Math.max(2, Math.round(v.videoHeight * scale / 2) * 2);
    // + half a frame: land in the middle of frame i's display interval so the
    // instant never falls on a boundary and picks up i-1
    const targets = new Float64Array(n);
    for (let i = 0; i < n; i++) targets[i] = t0 + (i + 0.5) / fps;

    const sup = decodeSupport();
    const asked = path || globalThis.DV_DECODE_PATH || 'auto';
    const forced = asked !== 'auto';
    const order = forced ? [asked]
      : sup.best === 'webcodecs-worker' ? ['webcodecs-worker', 'video-seek']
        : sup.best === 'webcodecs-main' ? ['webcodecs-main', 'video-seek']
          : ['video-seek'];

    const args = { targets, w, h, onProgress };
    let out = null, used = '', fell = [];
    for (const p of order) {
      try {
        if (p === 'video-seek') { out = await viaSeek(v, args); }
        else if (p === 'webcodecs-worker') {
          if (!forced && !sup.worker) throw new Error('no WebCodecs worker here');
          out = await viaWorker(file, args);
        } else if (p === 'webcodecs-main') {
          if (!forced && !sup.webcodecs) throw new Error('no WebCodecs here');
          out = await viaMain(file, args);
        } else throw new Error('unknown decode path ' + p);
        used = p;
        break;
      } catch (e) {
        if (forced) throw e;
        fell.push(`${DECODE_LABEL[p] || p}: ${e.message}`);
      }
    }
    if (!out) throw new Error(fell.join(' · ') || 'no decode path worked');
    if (out.frames.some((f) => !f)) throw new Error('the decode dropped a frame');

    const stats = {
      path: used, label: DECODE_LABEL[used] || used,
      ms: Math.round(out.ms), frames: n, split: out.split || null,
      accel: out.accel || '',
      fps: +(n / Math.max(out.ms / 1000, 1e-6)).toFixed(1),
      note: out.note, fellBack: fell,
      support: sup,
    };
    stats.line = `decoded ${n} frames in ${(out.ms / 1000).toFixed(2)} s · `
      + stats.label + (out.note ? ' · ' + out.note : '')
      + (fell.length ? ' · after ' + fell.join(' · ') : '');
    if (onProgress) {
      onProgress({ done: n, total: n, phase: 'decode', text: stats.line });
    }
    return { frames: out.frames, w, h, fps, nFrames: out.frames.length,
             trimStart: t0, seconds: dur, decode: stats };
  } finally {
    v.src = ''; v.load?.();
    URL.revokeObjectURL(url);
  }
}
