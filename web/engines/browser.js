/* ---------------------------------------------------------------------------
   BROWSER ENGINE — the whole tool in the tab, no server at all.

     decode      <video> + canvas seek loop -> one JPEG blob per frame
     track       EdgeTAM as four ONNX graphs on onnxruntime-web (web/track.js)
     dither      web/dither.js, the same engine the server mirrors
     export      MediaRecorder over a canvas capture stream -> WebM

   Nothing is uploaded and nothing is fetched from a CDN. The cost is speed
   (12.4 fps tracking on an M4 Pro against the local server's 20.9) and the
   container: MediaRecorder gives WebM, not the H.264 MP4 the server writes.
   Both trade-offs are stated in the UI rather than hidden.

   MEMORY. A 150-frame 720p clip is kept as ~15 MB of JPEG blobs plus, per
   subject, 150 x 192x192 float32 mask logits (~22 MB). Full-resolution RGBA is
   never retained — frames are decoded on demand and the LRU in app.js holds
   forty of them.
--------------------------------------------------------------------------- */
'use strict';

import { WebTracker } from '../track.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const NO_OBJ = -1024;                    // EdgeTAM's "this object is not here"

/* ===================================================== blue noise, in JS ===
 * The port of render.blue_noise: iterated high-pass + histogram remap. The
 * server does the low-pass as a gaussian multiply in the frequency domain;
 * a wrapped separable gaussian blur is the same operator, so the tiles have
 * the same spectrum but not the same bits (numpy's RNG is not portable). The
 * default seed-7 tile is shipped as web/bluenoise.json precisely so the two
 * engines start from an identical field; this generates the rest. */
function gaussianKernel(sigma) {
  const r = Math.max(1, Math.ceil(sigma * 3));
  const k = new Float32Array(2 * r + 1);
  let s = 0;
  for (let i = -r; i <= r; i++) { k[i + r] = Math.exp(-(i * i) / (2 * sigma * sigma)); s += k[i + r]; }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return { k, r };
}
function blurWrap(v, n, sigma) {
  const { k, r } = gaussianKernel(sigma);
  const t = new Float32Array(n * n), o = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++) a += k[i + r] * v[y * n + ((x + i + n * 8) % n)];
    t[y * n + x] = a;
  }
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    let a = 0;
    for (let i = -r; i <= r; i++) a += k[i + r] * t[((y + i + n * 8) % n) * n + x];
    o[y * n + x] = a;
  }
  return o;
}
export function blueNoiseTile(n = 64, seed = 7, iters = 40, sigma = 1.6) {
  const N = n * n;
  const v = new Float32Array(N);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) v[i * n + j] = Dither.hash01(i, j, 17, seed);
  const idx = new Int32Array(N);
  for (let it = 0; it < iters; it++) {
    const lp = blurWrap(v, n, sigma);
    for (let q = 0; q < N; q++) v[q] -= lp[q];
    for (let q = 0; q < N; q++) idx[q] = q;
    const arr = Array.from(idx).sort((a, b) => (v[a] - v[b]) || (a - b));
    for (let r = 0; r < N; r++) v[arr[r]] = (r + 0.5) / N;
  }
  return v;
}

/* ================================================ decoding a clip in-tab ===
 * `requestVideoFrameCallback` is the fast path but it only ever gives you the
 * frames the compositor chose to show, which is not a stable 30 fps grid. The
 * seek loop is slower and exactly reproducible, and it is what the server's
 * ffmpeg `-r 30` produces, so both engines index the same picture as frame 42.
 */
async function decodeClip(file, { fps = 30, maxSeconds = 10, maxFrames = 300,
                                  maxHeight = 720, trimStart = 0, trimEnd = null,
                                  onProgress } = {}) {
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
    const avail = Math.max(0, (full || maxSeconds) - t0);
    const want = trimEnd ? Math.max(0, trimEnd - t0) : avail;
    const dur = Math.min(want, avail, maxSeconds);
    if (!isFinite(dur) || dur <= 0) throw new Error('the clip has no duration');
    const n = Math.max(1, Math.min(maxFrames, Math.floor(dur * fps)));
    const scale = Math.min(1, maxHeight / (v.videoHeight || maxHeight));
    const w = Math.max(2, Math.round(v.videoWidth * scale / 2) * 2);
    const h = Math.max(2, Math.round(v.videoHeight * scale / 2) * 2);
    const cv = new OffscreenCanvas(w, h);
    const g = cv.getContext('2d', { willReadFrequently: false });

    const frames = [];
    for (let i = 0; i < n; i++) {
      // + half a frame: land in the middle of frame i's display interval so a
      // seek never lands on the boundary and returns i-1
      await seek(v, t0 + (i + 0.5) / fps);
      g.drawImage(v, 0, 0, w, h);
      frames.push(await cv.convertToBlob({ type: 'image/jpeg', quality: 0.92 }));
      if (onProgress && (i % 5 === 0 || i === n - 1)) {
        onProgress({ done: i + 1, total: n, phase: 'decode',
                     text: `decoding ${i + 1}/${n} frames…` });
      }
    }
    return { frames, w, h, fps, nFrames: frames.length, trimStart: t0, seconds: dur };
  } finally {
    v.src = ''; v.load?.();
    URL.revokeObjectURL(url);
  }
}

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
    const wrap = () => clearTimeout(bail);
    v.addEventListener('seeked', wrap, { once: true });
    v.currentTime = Math.max(0, t);
  });
}

/* ====================================================== export formats ===
 * What the tab can actually write, decided at load: MediaRecorder's codec
 * support is a runtime fact, and mp4/ProRes are a flat no. `available: false`
 * entries stay in the list on purpose — the UI shows them greyed with the
 * reason, which is more honest than pretending the format does not exist.
 */
function canRecord(t) {
  return typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(t);
}

export function browserFormats() {
  const webm = canRecord('video/webm;codecs=vp9') || canRecord('video/webm');
  const alpha = canRecord('video/webm;codecs=vp8') || canRecord('video/webm');
  return [
    { id: 'webm', label: 'WebM · VP9', ext: 'webm', mime: 'video/webm',
      alpha: false, available: webm,
      note: webm ? '' : 'this browser has no MediaRecorder WebM encoder' },
    { id: 'gif', label: 'GIF · looping', ext: 'gif', mime: 'image/gif',
      alpha: false, available: true,
      note: 'encoded in the tab; a 300-frame 720p GIF needs ~280 MB of scratch memory' },
    { id: 'webm-alpha', label: 'WebM · VP8 + alpha', ext: 'webm', mime: 'video/webm',
      alpha: true, available: alpha,
      note: alpha ? 'alpha only survives in players that read WebM alpha (Chrome, Firefox)'
        : 'this browser has no MediaRecorder WebM encoder' },
    { id: 'mp4', label: 'MP4 · H.264', ext: 'mp4', mime: 'video/mp4',
      alpha: false, available: false,
      note: 'needs the local server — writing H.264 in the tab means vendoring a ~32 MB encoder' },
    { id: 'prores', label: 'ProRes 4444 · alpha', ext: 'mov', mime: 'video/quicktime',
      alpha: true, available: false, note: 'needs the local server' },
  ];
}

/** The colour table a GIF export needs: background first, then every subject
 *  colour, de-duplicated. Mirrors what the renderer actually puts on screen. */
function gifPalette(params) {
  const out = [];
  const push = (c) => { if (c && out.indexOf(c) < 0 && out.length < 256) out.push(c); };
  if (params.compose === 'overlay') (params.palette || []).forEach(push);
  else push(params.bg);
  for (const s of (params.subjects || [])) (s.palette || []).forEach(push);
  (params.palette || []).forEach(push);
  if (out.length < 2) push('#000000');
  return out;
}

/* ============================================================= the engine === */
/* The tracker's output is 192x192 logits. The server upsamples logits to the
 * clip's resolution and THEN takes the sigmoid; doing it the other way round
 * widens the soft edge by a pixel or two, so this does it in the same order,
 * carrying the logit through the canvas resampler as a clamped +/-20 ramp.
 * `cache` is any object to hang the two scratch canvases off. */
function upsampleMask(low, w, h, cache) {
  const P = 192;
  const small = cache._msk || (cache._msk = new OffscreenCanvas(P, P));
  const sg = small.getContext('2d', { willReadFrequently: true });
  const id = sg.createImageData(P, P);
  for (let q = 0; q < P * P; q++) {
    const v = low ? low[q] : NO_OBJ;
    const t = Math.max(0, Math.min(255, Math.round((v + 20) * (255 / 40))));
    id.data[q * 4] = t; id.data[q * 4 + 3] = 255;
  }
  sg.putImageData(id, 0, 0);
  const big = cache._mskBig || (cache._mskBig = new OffscreenCanvas(w, h));
  if (big.width !== w || big.height !== h) { big.width = w; big.height = h; }
  const bg = big.getContext('2d', { willReadFrequently: true });
  bg.clearRect(0, 0, w, h);
  bg.drawImage(small, 0, 0, w, h);
  const px = bg.getImageData(0, 0, w, h);
  const d = px.data;
  for (let q = 0, p = 0; q < w * h; q++, p += 4) {
    const logit = d[p] * (40 / 255) - 20;
    const a = Math.round(255 / (1 + Math.exp(-logit)));
    d[p] = a; d[p + 1] = a; d[p + 2] = a; d[p + 3] = 255;
  }
  bg.putImageData(px, 0, 0);
  return createImageBitmap(big);
}

/** What `BrowserEngine.snapshot()` hands out: frames and masks, no tracker,
 *  no lifecycle. The frame blobs are shared with the engine (they are
 *  immutable), the mask logits are a shallow copy of the map, so loading
 *  another clip cannot take either away. */
export class LocalClipSource {
  constructor(clip, masks) {
    this.id = 'browser';
    this.w = clip.w; this.h = clip.h;
    this.nFrames = clip.nFrames; this.fps = clip.fps;
    this.frames = clip.frames;
    this.masks = masks;
  }

  async frame(i) {
    return createImageBitmap(
      this.frames[Math.max(0, Math.min(this.nFrames - 1, i))]);
  }

  async mask(objId, i) {
    return upsampleMask((this.masks.get(String(objId)) || [])[i],
                        this.w, this.h, this);
  }
}

export class BrowserEngine {
  constructor(opts = {}) {
    this.id = 'browser';
    this.label = 'Browser';
    // the chip is narrow; the full claim lives in its title and the popover
    this.sublabel = 'free';
    this.dir = opts.dir || './models/';
    this.ortDir = opts.ortDir || './ort/';
    this.fp16 = opts.fp16 !== false;
    this.ep = opts.ep || 'webgpu';
    this.clip = null;
    this.tracker = null;
    this.masks = new Map();              // objId -> Float32Array[](192*192 logits)
    this.promptFrames = new Map();       // objId -> the frame it was prompted on
    this.supports = {
      maskPrompt: false,                 // set from the manifest in init()
      perObjectPromptFrames: true,
      backward: true,
      multiObject: 'sequential',         // one full pass per subject
      exportMime: 'video/webm',
      exportExt: 'webm',
      exportPlayable: true,
      stillSubjects: true,          // single-image segmentation, no propagation
      formats: browserFormats(),
    };
  }

  /* ---------------------------------------------------------- model set
   * Two steps, deliberately separate. manifest.json is small and committed, so
   * it is what tells the page what the model set can do; the weights are ~130 MB
   * and are not, so a page can perfectly well render stills and whole-frame
   * clips without them. `meta()` needs only the first, `init()` needs both. */
  async manifestOf() {
    if (this.manifest) return this.manifest;
    let man;
    try {
      const r = await fetch(this.dir + 'manifest.json');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      man = await r.json();
    } catch (e) {
      throw new modelsMissing('models/manifest.json is not there (' + e.message + ')');
    }
    this.manifest = man;
    this.supports.maskPrompt = !!man.has_mask_prompt;
    return man;
  }

  async init() {
    if (this.ready) return this;
    await this.manifestOf();
    // A HEAD that 404s prints a line in the console. That is unavoidable and,
    // in the only case it happens, correct: the file really is not there. The
    // alternative is finding out when the user presses Track.
    const probe = this.fp16 ? 'encoder.fp16.onnx' : 'encoder.onnx';
    const head = await fetch(this.dir + probe, { method: 'HEAD' }).catch(() => null);
    if (!head || !head.ok) {
      throw new modelsMissing(`${probe} is not there — the ONNX weights are not `
        + 'committed to the repo. See web/README.md: download the release bundle '
        + 'into web/models/, or run ./setup.sh to export them yourself.');
    }
    if (this.ep === 'webgpu' && !navigator.gpu) {
      this.ep = 'wasm';
      this.epNote = 'no WebGPU in this browser — running on WASM, ~6x slower';
    }
    this.ready = true;
    return this;
  }

  async loadTracker(log) {
    if (this.tracker) return this.tracker;
    await this.init();
    // `fetch('./x')` resolves against the DOCUMENT, but `import('./x')` resolves
    // against this module — which lives one directory deeper. Resolve both the
    // same way the rest of the engine does, or the runtime 404s under engines/.
    const ortBase = new URL(this.ortDir, document.baseURI).href;
    const ort = await import(ortBase + 'ort.all.bundle.min.mjs');
    ort.env.wasm.wasmPaths = ortBase;
    ort.env.wasm.numThreads = Math.min(8, navigator.hardwareConcurrency || 4);
    ort.env.logLevel = 'error';
    this.ort = ort;
    const t = new WebTracker(ort, { ep: this.ep, fp16: this.fp16,
                                    chain: this.ep === 'webgpu', dir: this.dir });
    await t.load(log || (() => {}));
    this.tracker = t;
    return t;
  }

  /* ------------------------------------------------------------ metadata */
  async meta() {
    // The manifest, not init(): missing weights must not stop the palette and
    // mode tables from loading. checkModels() in app.js reports that separately.
    let S = 768;
    try { S = (await this.manifestOf()).image_size; } catch (e) { /* reported later */ }
    return {
      palettes: Dither.PALETTES,
      modes: Dither.MODES,
      stable: Dither.STABLE,
      kernels: Object.entries(Dither.KERNELS).map(([id, v]) => ({ id, name: v.name })),
      subject_colors: ['#b0413e', '#2f4f4a', '#7a6a4f', '#3c5a7a', '#8a5a8a', '#4a7a4a'],
      device: this.ep === 'webgpu' ? 'webgpu' : 'wasm',
      backend: this.fp16 ? 'fp16' : 'fp32',
      max_objects: 6,
      // One exported resolution, so one chip. Exporting 512 and 1024 as well
      // would triple the download for a knob that mostly matters when you are
      // waiting on a server.
      track_sizes: [{ size: S, id: 'balanced', label: 'balanced', fps: 12.4 }],
      default_track_size: S,
      engine: 'browser',
    };
  }

  async blueNoise(n, seed) {
    if (n === 64 && seed === 7) {
      try {
        const j = await (await fetch('./bluenoise.json')).json();
        return Float32Array.from(j.tile);
      } catch (e) { /* generate it instead */ }
    }
    return blueNoiseTile(n, seed);
  }

  /* --------------------------------------------------------------- clip */
  async open(file, opts = {}) {
    const c = await decodeClip(file, opts);
    this.clip = c;
    this.masks.clear(); this.promptFrames.clear();
    return { job: 'local', nFrames: c.nFrames, w: c.w, h: c.h, fps: c.fps,
             trimStart: c.trimStart, seconds: c.seconds };
  }

  /* --------------------------------------------------------------- still
   * A photograph is a clip of one frame. Saying it that way costs nothing and
   * buys everything: trackerInput, promptTensor, maskBitmap and the mask cache
   * all keep working, and single-image segmentation is the conditioning step
   * the clip flow already runs on frame 0 -- with nothing after it. */
  async openStill(blob, { w = 0, h = 0 } = {}) {
    let W = w, H = h;
    if (!W || !H) {
      const b = await createImageBitmap(blob);
      W = b.width; H = b.height; b.close?.();
    }
    this.clip = { frames: [blob], nFrames: 1, w: W, h: H, fps: 1, still: true };
    this.masks.clear(); this.promptFrames.clear();
    return { job: 'local', kind: 'image', nFrames: 1, w: W, h: H, fps: 1 };
  }

  /** One prompt, one answer, no propagation: encoder + heads_prompt (or
   *  heads_mask), exactly the frame-0 preview path. The soft masks are kept
   *  under this.masks so `mask(id, 0)` hands back the same full-resolution
   *  upsample the tracked flow uses -- the still and the clip compose through
   *  identical pixels. */
  async segmentImage({ objects }, onLog) {
    if (!this.clip || !this.clip.still) throw new Error('no still is open');
    const t = await this.loadTracker(onLog);
    const S = t.man.image_size;
    const t0 = performance.now();
    const rgba = await this.trackerInput(0, S);
    const keep = new Set(objects.map((o) => String(o.id)));
    for (const k of [...this.masks.keys()]) if (!keep.has(k)) this.masks.delete(k);
    const out = [], notes = [];
    for (const o of objects) {
      const r = await this.resolvePrompt(o);
      if (r.degraded) notes.push('#' + o.id + ': shape approximated as a box + a click');
      t.reset();
      const step = await this.stepPrompt(t, rgba, r);
      this.masks.set(String(o.id), [step.low]);
      out.push({ id: String(o.id), image: await this.mask(o.id, 0), area: step.area });
    }
    return { objects: out, elapsedS: (performance.now() - t0) / 1000,
             imageSize: S, backend: this.ep, frameIdx: 0, note: notes.join(' · ') };
  }

  async frame(i) {
    const b = this.clip.frames[Math.max(0, Math.min(this.clip.nFrames - 1, i))];
    return createImageBitmap(b);
  }

  /** RGBA of frame `i` resized into the tracker's square. */
  async trackerInput(i, S) {
    const bmp = await this.frame(i);
    const c = this._sq || (this._sq = new OffscreenCanvas(S, S));
    if (c.width !== S) { c.width = S; c.height = S; }
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(bmp, 0, 0, S, S);
    bmp.close?.();
    return g.getImageData(0, 0, S, S).data;
  }

  /* -------------------------------------------------------------- masks
   * The tracker's output is 192x192 logits. The server upsamples logits to the
   * clip's resolution and THEN takes the sigmoid; doing it the other way round
   * widens the soft edge by a pixel or two, so this does it in the same order,
   * carrying the logit through the canvas resampler as a clamped +/-20 ramp. */
  maskBitmap(objId, i) {
    const { w, h } = this.clip;
    return upsampleMask((this.masks.get(String(objId)) || [])[i], w, h, this);
  }

  async mask(objId, i) { return this.maskBitmap(objId, i); }

  /** A detached handle on the clip that is open RIGHT NOW: the same frame
   *  blobs and the same mask logits, in an object that does not care what the
   *  engine loads next. A sequence item keeps one of these so it can redraw
   *  its dots at any look long after its clip has left the studio. */
  snapshot() {
    const c = this.clip;
    if (!c) return null;
    return new LocalClipSource(c, new Map(this.masks));
  }

  async frameURL(i) {
    const b = this.clip.frames[Math.max(0, Math.min(this.clip.nFrames - 1, i))];
    return { url: URL.createObjectURL(b), revoke: true };
  }

  /* ------------------------------------------------------------ prompts
   * Coordinates arrive in clip pixels and the graphs want the tracker square.
   * Labels follow SAM's convention: 1 keep, 0 drop, 2/3 the two box corners. */
  promptTensor(o, S) {
    const { w, h } = this.clip, sx = S / w, sy = S / h;
    const co = [], la = [];
    if (o.box) {
      co.push(o.box[0] * sx, o.box[1] * sy, o.box[2] * sx, o.box[3] * sy);
      la.push(2, 3);
    }
    for (const p of (o.points || [])) { co.push(p[0] * sx, p[1] * sy); la.push(p[2] ? 1 : 0); }
    if (!la.length) throw new Error('subject #' + o.id + ' has no prompt');
    return { coords: Float32Array.from(co), labels: Float32Array.from(la) };
  }

  /** A drawn shape, rasterised into the tracker's square as 0/1. */
  async maskTensor(dataUrl, S) {
    const im = new Image();
    await new Promise((ok, no) => { im.onload = ok; im.onerror = no; im.src = dataUrl; });
    const c = new OffscreenCanvas(S, S);
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(im, 0, 0, S, S);
    const d = g.getImageData(0, 0, S, S).data;
    const out = new Float32Array(S * S);
    let any = 0;
    for (let q = 0, p = 0; q < out.length; q++, p += 4) { out[q] = d[p] > 127 ? 1 : 0; any += out[q]; }
    if (!any) throw new Error('that shape is empty');
    return out;
  }

  /** A drawn shape reduced to a box + a centroid click.
   *  Used only when the model set predates the mask-prompt graph. */
  async shapeToPoints(dataUrl) {
    const { w, h } = this.clip;
    const im = new Image();
    await new Promise((ok, no) => { im.onload = ok; im.onerror = no; im.src = dataUrl; });
    const c = new OffscreenCanvas(w, h);
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(im, 0, 0, w, h);
    const d = g.getImageData(0, 0, w, h).data;
    let x0 = w, y0 = h, x1 = -1, y1 = -1, sx = 0, sy = 0, n = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      if (d[(y * w + x) * 4] > 127) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
        sx += x; sy += y; n++;
      }
    }
    if (!n) throw new Error('that shape is empty');
    return { box: [x0, y0, x1, y1],
             points: [[Math.round(sx / n), Math.round(sy / n), 1]] };
  }

  /** Normalise one subject into what the tracker can actually consume. */
  async resolvePrompt(o) {
    if (!o.mask) return { kind: 'points', obj: o };
    if (this.supports.maskPrompt) return { kind: 'mask', obj: o };
    const approx = await this.shapeToPoints(o.mask);
    return { kind: 'points', degraded: true,
             obj: Object.assign({}, o, { mask: null, box: approx.box,
                                         points: approx.points }) };
  }

  /* ------------------------------------------------- one-frame preview */
  async previewFrame({ frameIdx, objects }, onLog) {
    const t = await this.loadTracker(onLog);
    const S = t.man.image_size;
    const t0 = performance.now();
    const rgba = await this.trackerInput(frameIdx, S);
    const out = [];
    const notes = [];
    for (const o of objects) {
      const r = await this.resolvePrompt(o);
      if (r.degraded) notes.push('#' + o.id + ': shape approximated as a box + a click');
      t.reset();
      const step = await this.stepPrompt(t, rgba, r);
      out.push({ id: String(o.id), image: await this.maskImage(step.low),
                 area: step.area });
    }
    return { objects: out, elapsedS: (performance.now() - t0) / 1000,
             imageSize: S, backend: this.ep, frameIdx,
             note: notes.join(' · ') };
  }

  /** A conditioning frame: heads_prompt (points) or heads_mask (a drawn shape). */
  async stepPrompt(t, rgba, resolved) {
    const o = resolved.obj;
    const S = t.man.image_size;
    let r;
    r = resolved.kind === 'mask'
      ? await t.step(rgba, { mask: await this.maskTensor(o.mask, S) })
      : await t.step(rgba, this.promptTensor(o, S));
    let area = 0;
    for (let q = 0; q < r.low.length; q++) if (r.low[q] > 0) area++;
    // the count is in 192-space; report it in clip pixels like the server does
    const k = (this.clip.w * this.clip.h) / (r.size * r.size);
    return { low: r.low, area: Math.round(area * k) };
  }

  /** 192x192 logits -> an <img>-shaped thing the overlay can paint. */
  async maskImage(low) {
    const P = 192;
    const c = new OffscreenCanvas(P, P);
    const g = c.getContext('2d');
    const id = g.createImageData(P, P);
    for (let q = 0; q < P * P; q++) {
      const a = Math.round(255 / (1 + Math.exp(-low[q])));
      id.data[q * 4] = a; id.data[q * 4 + 3] = 255;
    }
    g.putImageData(id, 0, 0);
    return createImageBitmap(c);
  }

  /* ----------------------------------------------------------- tracking
   * One WebTracker instance is a single-object memory bank, so subjects are
   * tracked one after another rather than as a batch the way the server does
   * it. Each subject is walked out from ITS OWN prompt frame, both ways:
   *
   *   forward   own prompt frame -> last frame
   *   backward  own prompt frame -> 0
   *
   * which is N steps per subject however late the prompt is. The server reaches
   * the same answer a cheaper way: one inference state, but a memory bank per
   * subject inside it, and the image encoder shared across subjects on a frame
   * (server/edgetam_util.py, `propagate_per_object`). Either way a subject that
   * is not in the shot yet comes back empty, because EdgeTAM's object score
   * goes negative and NO_OBJ_SCORE logits follow.
   *
   * The bank being per subject is load-bearing, not incidental. SAM2's own
   * batched loop shares ONE bank across every object and consolidates every
   * prompt frame before tracking starts, so a frame prompted for subject B
   * lands in subject A's memory as a CONDITIONING frame holding the NO_OBJ
   * placeholder — and conditioning frames are attended to forever, so A's track
   * dies from there on. This loop never had that bug and the server no longer
   * does; batching subjects together here would reintroduce it.
   */
  async track({ objects, imageSize }, onProgress) {
    const t = await this.loadTracker((s) => onProgress
      && onProgress({ done: 0, total: 1, text: s }));
    const S = t.man.image_size;
    const N = this.clip.nFrames;
    const t0 = performance.now();
    const plans = [];
    for (const o of objects) {
      const fp = Math.max(0, Math.min(N - 1, o.frameIdx | 0));
      plans.push({ o, fp, back: fp > 0, steps: N });
    }
    const total = plans.reduce((a, p) => a + p.steps, 0);
    let done = 0;
    const notes = [];

    for (const p of plans) {
      const id = String(p.o.id);
      const seq = new Array(N).fill(null);
      const r = await this.resolvePrompt(p.o);
      if (r.degraded) notes.push('#' + id + ': drawn shape approximated as a box + a click');
      const tick = async () => {
        done++;
        if (onProgress && done % 3 === 0) {
          const el = (performance.now() - t0) / 1000;
          onProgress({ done, total,
                       text: `${done}/${total} · ${(done / Math.max(el, 1e-6)).toFixed(1)} fps` });
        }
        await sleep(0);          // let the progress bar actually paint
      };

      // --- forward, from this subject's own prompt frame
      t.reset();
      let rgba = await this.trackerInput(p.fp, S);
      let step = await this.stepPrompt(t, rgba, r);
      seq[p.fp] = step.low;
      await tick();
      for (let i = p.fp + 1; i < N; i++) {
        rgba = await this.trackerInput(i, S);
        const out = await t.step(rgba, null);
        seq[i] = out.low;
        await tick();
      }

      // --- backward, same prompt, frames walked in reverse
      if (p.back) {
        t.reset();
        rgba = await this.trackerInput(p.fp, S);
        await this.stepPrompt(t, rgba, r);
        for (let i = p.fp - 1; i >= 0; i--) {
          rgba = await this.trackerInput(i, S);
          const out = await t.step(rgba, null);
          seq[i] = out.low;
          await tick();
        }
      }
      this.masks.set(id, seq);
      this.promptFrames.set(id, p.fp);
    }

    const el = (performance.now() - t0) / 1000;
    return {
      frames: N, elapsedS: +el.toFixed(2),
      fps: +(total / Math.max(el, 1e-6)).toFixed(2),
      device: this.ep.toUpperCase(), backend: this.fp16 ? 'fp16' : 'fp32',
      imageSize: S, steps: total,
      note: [objects.length > 1 ? `${objects.length} subjects, one pass each` : '',
             this.epNote || '', notes.join(' · ')].filter(Boolean).join(' · '),
    };
  }

  /* ------------------------------------------------------------- export
   * Four containers in the tab, one missing:
   *
   *   webm         MediaRecorder, VP9 (VP8 on older builds)
   *   webm-alpha   the same recorder over an alpha canvas — VP8, because that
   *                is the codec Chrome carries an alpha plane in
   *   gif          web/vendor/gifenc.js, our own LZW encoder; the look is 2-4
   *                flat colours, which is what a 256-entry palette is for
   *   mp4          NOT here. Writing H.264 in the tab means shipping an
   *                encoder; ffmpeg.wasm is ~32 MB and would have to be vendored
   *                to keep the no-CDN rule, which is a bigger download than the
   *                tracker. The server engine writes it.
   *   prores       NOT here, same reason and then some.
   */
  /**
   * params.source, when present, overrides the open clip's {w, h, nFrames, fps}
   * — that is how the sequence view gets its GIF and its alpha WebM out of the
   * tab: a sequence is not the clip that happens to be loaded (it may be longer,
   * a different size, or the only thing in the session), but it is exactly the
   * same job of "call renderFrame(i) and feed an encoder".
   */
  async exportClip(params, onProgress, renderFrame) {
    const fmt = params.format || 'webm';
    const f = (this.supports.formats || []).find((x) => x.id === fmt);
    if (!f || !f.available) {
      throw new Error(`the browser engine cannot write ${fmt}`
        + (f && f.note ? ` — ${f.note}` : ' — use the local server'));
    }
    if (fmt === 'gif') return this.exportGIF(params, onProgress, renderFrame);
    return this.exportWebM(params, onProgress, renderFrame, fmt === 'webm-alpha');
  }

  /** Frames -> palette indices -> one GIF. The whole animation has to be in
   *  memory at once (one byte per pixel per frame: ~0.9 MB a frame at 720p),
   *  which is the reason for the size note in the UI. */
  async exportGIF(params, onProgress, renderFrame) {
    await import('../vendor/gifenc.js');
    const G = globalThis.GifEnc;
    const { w, h, nFrames, fps } = params.source || this.clip;
    const gfps = Math.max(1, Math.min(50, params.gif_fps || 15));
    // keep every k-th frame so the GIF runs at the asked-for rate in real time
    const step = Math.max(1, Math.round(fps / gfps));
    const palette = gifPalette(params);
    const cache = new Map();
    const frames = [];
    const t0 = performance.now();
    for (let i = 0; i < nFrames; i += step) {
      const img = await renderFrame(i, w, h);
      frames.push(G.indexFrame(img.data, w, h, palette, cache));
      if (onProgress) {
        onProgress({ done: i + 1, total: nFrames,
                     text: `${frames.length} frames · dithering` });
      }
      await sleep(0);
    }
    if (onProgress) onProgress({ done: nFrames, total: nFrames, text: 'encoding GIF…' });
    await sleep(0);
    const bytes = G.encode({ width: w, height: h, fps: fps / step, frames,
                             palette, loop: 0 });
    const blob = new Blob([bytes], { type: 'image/gif' });
    const el = (performance.now() - t0) / 1000;
    return {
      url: URL.createObjectURL(blob), mime: 'image/gif', ext: 'gif',
      playable: false, image: true, frames: frames.length,
      elapsedS: +el.toFixed(2), fps: +(frames.length / Math.max(el, 1e-6)).toFixed(2),
      bytes: blob.size,
      note: `${frames.length} frames at ${(fps / step).toFixed(0)} fps · `
        + `${palette.length} colours · loops forever`,
    };
  }

  /* MediaRecorder over `captureStream(0)` + `requestFrame()`: the recorder
   * timestamps each frame when we hand it over, so pacing the loop to the
   * clip's own frame interval gives a file that plays at the right speed. If a
   * frame takes longer than that interval to dither, the export runs behind and
   * the result is slower than real time — the returned `fps` says whether it
   * did.
   *
   * The container is WebM (VP9, VP8 on older builds). Writing H.264 MP4 in the
   * tab would mean shipping an encoder; ffmpeg.wasm is ~32 MB and would have to
   * be vendored to keep the no-CDN rule, which is a bigger download than the
   * tracker itself. The server engine writes MP4.
   */
  async exportWebM(params, onProgress, renderFrame, alpha) {
    const { w, h, nFrames, fps } = params.source || this.clip;
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    const g = cv.getContext('2d', { alpha: !!alpha });
    // VP8 first for alpha: Chrome's WebM writer carries an alpha plane for VP8
    // (alpha_mode=1) and not for VP9. Opaque exports prefer VP9 for the bitrate.
    const types = alpha
      ? ['video/webm;codecs=vp8', 'video/webm']
      : ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const mime = types.find((t) => window.MediaRecorder
      && MediaRecorder.isTypeSupported(t));
    if (!mime) throw new Error('this browser has no MediaRecorder WebM encoder');
    const stream = cv.captureStream(0);
    const vtrack = stream.getVideoTracks()[0];
    const chunks = [];
    const rec = new MediaRecorder(stream, {
      mimeType: mime,
      videoBitsPerSecond: Math.min(24e6, Math.max(4e6, w * h * fps * 0.15)),
    });
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    const stopped = new Promise((ok) => { rec.onstop = ok; });
    const t0 = performance.now();
    rec.start();
    const dt = 1000 / fps;
    for (let i = 0; i < nFrames; i++) {
      const fs = performance.now();
      const img = await renderFrame(i, w, h);     // ImageData from app.js
      if (alpha) g.clearRect(0, 0, w, h);
      g.putImageData(img, 0, 0);
      vtrack.requestFrame();
      if (onProgress) onProgress({ done: i + 1, total: nFrames,
                                   text: `${i + 1}/${nFrames}` });
      const left = dt - (performance.now() - fs);
      await sleep(Math.max(0, left));
    }
    // give the encoder the last frame's full interval before cutting
    await sleep(dt * 2);
    rec.stop();
    await stopped;
    vtrack.stop();
    const blob = new Blob(chunks, { type: mime });
    const el = (performance.now() - t0) / 1000;
    const real = nFrames / fps;
    const slow = el > real * 1.25
      ? `rendered slower than real time (${el.toFixed(1)} s of work for a `
        + `${real.toFixed(1)} s clip) — the WebM is paced to wall clock, so it `
        + 'plays slow; pick a lighter look or use the local server'
      : '';
    return {
      url: URL.createObjectURL(blob), mime, ext: 'webm', playable: true,
      alpha: !!alpha,
      frames: nFrames, elapsedS: +el.toFixed(2),
      fps: +(nFrames / Math.max(el, 1e-6)).toFixed(2),
      bytes: blob.size,
      note: slow || (alpha
        ? `${mime} with an alpha channel — the server writes ProRes 4444 too`
        : `${mime} — the server engine writes H.264 MP4`),
    };
  }

  dispose() {
    this.clip = null;
    this.masks.clear();
    this.promptFrames.clear();
  }
}

/* A distinguishable error so the UI can show the "download the models" panel
 * instead of a red toast that reads like a crash. */
export class modelsMissing extends Error {
  constructor(msg) { super(msg); this.name = 'ModelsMissing'; this.modelsMissing = true; }
}
