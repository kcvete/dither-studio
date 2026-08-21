/* ---------------------------------------------------------------------------
   BROWSER ENGINE — the whole tool in the tab, no server at all.

     decode      demux + VideoDecoder in a module Worker -> one JPEG blob per
                 frame; the <video> seek loop is the fallback (engines/decode.js)
     track       EdgeTAM as four ONNX graphs on onnxruntime-web (web/track.js)
     dither      web/dither.js, the same engine the server mirrors
     export      MediaRecorder over a canvas capture stream -> WebM

   Nothing is uploaded and nothing is fetched from a CDN. The cost is speed
   (12.4 fps tracking on an M4 Pro against the local server's 20.9) and the
   container: MediaRecorder gives WebM, not the H.264 MP4 the server writes.
   Both trade-offs are stated in the UI rather than hidden.

   MEMORY. A 150-frame 720p clip is kept as ~15 MB of JPEG blobs plus, per
   subject, 150 x 192x192 float32 mask logits (~22 MB). Full-resolution RGBA is
   never retained — frames are decoded on demand, this engine keeps a sixteen
   frame bitmap LRU in front of the blobs and the LRU in app.js holds forty
   more. Both are bounded by bytes, not by frames, so a 1080p clip holds fewer.
--------------------------------------------------------------------------- */
'use strict';

import { WebTracker } from '../track.js';
import { decodeClip, decodeSupport } from './decode.js';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
/* One compositor frame. Only the capture path that has no requestFrame needs
 * it, and a document that is hidden never fires one — hence the timeout. */
const raf = () => new Promise((r) => {
  if (typeof requestAnimationFrame !== 'function') return setTimeout(r, 16);
  let done = false;
  const go = () => { if (!done) { done = true; r(); } };
  requestAnimationFrame(go);
  setTimeout(go, 100);
});

/** How many video frames a recorded WebM actually contains. Uses the demuxer
 *  the decode path already ships; a file it cannot read reports 0, which the
 *  caller treats as "no count", never as "no frames". */
async function countWebMFrames(blob) {
  try {
    const { demuxWebM, looksLikeWebM } = await import('../workers/demux-webm.js');
    const bytes = new Uint8Array(await blob.arrayBuffer());
    if (!looksLikeWebM(bytes)) return 0;
    return demuxWebM(bytes).samples.length;
  } catch (e) { return 0; }
}
const NO_OBJ = -1024;                    // EdgeTAM's "this object is not here"

/* Said in three places (the probe in init(), and both retries in
 * loadTracker), so it is written once. */
const NO_FP32_NOTE = 'this GPU has no shader-f16 and the fp32 graphs are not '
  + 'in this deployment \u2014 running fp16 on WASM instead, which is much '
  + 'slower. See web/README.md for the fp32 bundle';

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
 * The three decode paths, the frame grid they all reproduce and why the grid
 * is what it is live in engines/decode.js. What is left here is what a decoded
 * clip costs the tab.
 */
export const BITMAP_BUDGET = 48e6;       // bytes of RGBA the engine caches

export function clipMemoryEstimate(nFrames, w, h, subjects = 1,
                                   trackSize = 768) {
  // What a decoded clip actually costs this tab:
  //   the JPEG blobs it keeps           ~90 KB a frame at 1280x720
  //   this engine's bitmap LRU           48 MB, or fewer frames' worth
  //   the 40-frame bitmap LRU in app.js  w*h*4 each
  //   one float32 mask logit grid per frame per subject, once tracked, at
  //   grid*4 squared: 128 at the 512 px tracker, 192 at 768, 256 at 1024
  const px = Math.max(1, w * h);
  const jpeg = nFrames * 90e3 * px / (1280 * 720);
  const bitmaps = Math.min(BITMAP_BUDGET, nFrames * px * 4);
  const lru = 40 * px * 4;
  const P = Math.max(64, Math.round((trackSize || 768) / 4));
  const masks = nFrames * P * P * 4 * Math.max(1, subjects);
  return { jpeg, bitmaps, lru, masks, total: jpeg + bitmaps + lru + masks };
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

/* The three tracker squares, named as the server names them so the quality
 * chips in app.js need to know nothing about which engine is running.
 *
 * `fps` is measured END TO END through the UI — Chrome on an M4 Pro / 24 GB,
 * 150 frames of 1280x720, one subject, WebGPU fp16, with the streaming preview
 * painting and the progress bar moving, which is what a person actually waits
 * through. It is what the estimate under the Track button quotes, so it is a
 * median on the reference clip and not a promise. The graphs run faster than
 * this in isolation; see README, Performance. */
export const TRACK_TIERS = {
  512: { id: 'fast', label: 'fast · prototyping', fps: 15.4 },
  768: { id: 'balanced', label: 'balanced · default', fps: 8.9 },
  1024: { id: 'best', label: 'best · production', fps: 5.4 },
};

/* ============================================================= the engine === */
/* The tracker's output is 192x192 logits. The server upsamples logits to the
 * clip's resolution and THEN takes the sigmoid; doing it the other way round
 * widens the soft edge by a pixel or two, so this does it in the same order,
 * carrying the logit through the canvas resampler as a clamped +/-20 ramp.
 * `cache` is any object to hang the two scratch canvases off. */
function upsampleMask(low, w, h, cache, size) {
  // grid*4: 128 at the 512 px tracker, 192 at 768, 256 at 1024. Taking it from
  // the logits themselves means a sequence item drawn long after its clip can
  // carry a different tracker resolution than whatever is loaded now.
  const P = size || (low ? Math.round(Math.sqrt(low.length)) : 192);
  const small = cache._msk && cache._msk.width === P ? cache._msk
    : (cache._msk = new OffscreenCanvas(P, P));
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
    this.manifests = new Map();          // image size -> manifest.json
    this.trackerSize = 0;                // which one the loaded tracker is
    this.fp16 = opts.fp16 !== false;
    this.ep = opts.ep || 'webgpu';
    this.clip = null;
    this.tracker = null;
    this.bmp = new Map();                // frame index -> ImageBitmap (LRU)
    this.bmpMax = 0;                     // set from the clip's size on open
    this.lastDecode = null;              // which decode path ran, and how fast
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
      reextract: true,              // reopen(): the File handle is still here
      extractProgress: true,        // decode reports frame by frame
      uncapped: true,               // whole clip, however long it is
      original: true,               // the matched cut, straight off the decoded frames
      // one subject at a time. The loop below was always one memory bank per
      // subject, so "track only #2" is the same loop with a shorter list --
      // #1's logits are simply never written over.
      incrementalTrack: true,
      // what this browser could decode with, before any file is opened;
      // `decode` is filled in per clip by open() with what actually ran
      decodePaths: decodeSupport(),
      decode: null,
      formats: browserFormats(),
    };
  }

  /* ---------------------------------------------------------- model set
   * Two steps, deliberately separate. manifest.json is small and committed, so
   * it is what tells the page what the model set can do; the weights are ~130 MB
   * and are not, so a page can perfectly well render stills and whole-frame
   * clips without them. `meta()` needs only the first, `init()` needs both.
   *
   * `size` picks a tracker square. Omitted is the default set at the root of
   * models/ -- the layout the release and the Pages build already have, and
   * the only one most visitors ever download; the other squares live in a
   * subdirectory named after themselves. */
  sizeDir(size) {
    return !size || size === this.baseSize ? this.dir : `${this.dir}${size}/`;
  }

  async manifestOf(size) {
    // the default set names itself, and is what says which others exist, so it
    // is always read first
    if (size && !this.baseSize) await this.manifestOf();
    const key = size && size !== this.baseSize ? size : 0;
    if (this.manifests.has(key)) return this.manifests.get(key);
    let man;
    try {
      const r = await fetch(this.sizeDir(size) + 'manifest.json');
      if (!r.ok) throw new Error('HTTP ' + r.status);
      man = await r.json();
    } catch (e) {
      throw new modelsMissing(`models/${size && size !== this.baseSize
        ? size + '/' : ''}manifest.json is not there (${e.message})`);
    }
    this.manifests.set(key, man);
    if (!key) {
      this.manifest = man;
      this.baseSize = man.image_size;
      this.manifests.set(man.image_size, man);
      this.supports.maskPrompt = !!man.has_mask_prompt;
      /* Which resolutions this deployment actually carries. It is written into
       * the default manifest at export time rather than probed, because probing
       * means a 404 per absent tier in the console of every page that only has
       * the default set — which is most of them. A manifest with no `tiers` is
       * the one-resolution build this used to be. */
      this.tiers = Array.isArray(man.tiers) && man.tiers.length
        ? man.tiers.slice().sort((a, b) => a - b) : [man.image_size];
    }
    return man;
  }

  /* --------------------------------------------------- backend, for real
   * Which execution provider and which precision, decided by asking the
   * hardware rather than by assuming Chrome. Three answers:
   *
   *   webgpu + fp16   the adapter reports `shader-f16`
   *   webgpu + fp32   it does not. The fp32 graphs are exported beside the
   *                   fp16 ones and are ~20% slower, which is a far better
   *                   answer than emulating half floats in a shader or
   *                   dropping to WASM. They are a separate download and a
   *                   deployment without them falls to fp16 on WASM, saying so
   *   wasm            no navigator.gpu, or no adapter at all
   *
   * Idempotent and cheap, so `meta()` can call it before any weights exist —
   * the chip has to be honest from the first paint, not from the first track.
   */
  async pickBackend() {
    if (this._picked) return this;
    this._picked = true;
    if (this.ep !== 'webgpu') return this;
    if (!navigator.gpu) {
      this.ep = 'wasm';
      this.epNote = 'no WebGPU in this browser — running on WASM, ~6x slower';
      return this;
    }
    let adapter = null;
    try { adapter = await navigator.gpu.requestAdapter(); } catch (e) { adapter = null; }
    if (!adapter) {
      // navigator.gpu EXISTS and requestAdapter still answered null. That is
      // not a browser without WebGPU, it is a browser that has switched it off:
      // Brave does it by default through Shields' fingerprinting protection,
      // and Chrome does it on blocklisted GPUs. Naming the cause is the whole
      // difference between a dead end and a setting to change.
      this.ep = 'wasm';
      this.adapterBlocked = true;
      this.epNote = 'WebGPU is present but this browser refused an adapter — '
        + 'in Brave that is Shields\u2019 fingerprinting protection (set it to '
        + 'Standard at brave://settings/shields, or enable '
        + 'brave://flags/#enable-unsafe-webgpu). Running on WASM instead, '
        + '~6x slower';
      return this;
    }
    if (this.fp16 && !(adapter.features && adapter.features.has('shader-f16'))) {
      this.fp16 = false;
      this.epNote = 'this GPU has no shader-f16 — running the fp32 graphs '
        + '(bigger download, ~20% slower, same masks)';
    }
    return this;
  }

  async init() {
    if (this.ready) return this;
    await this.manifestOf();
    // the precision has to be settled BEFORE the probe, or a machine without
    // shader-f16 gets told the fp16 weights are missing
    await this.pickBackend();
    // A HEAD that 404s prints a line in the console. That is unavoidable and,
    // in the only case it happens, correct: the file really is not there. The
    // alternative is finding out when the user presses Track.
    let probe = this.fp16 ? 'encoder.fp16.onnx' : 'encoder.onnx';
    let head = await fetch(this.dir + probe, { method: 'HEAD' }).catch(() => null);
    /* fp32 was asked for because this GPU has no shader-f16, and the fp32
     * graphs are a separate download that a fp16-only deployment (Pages, and
     * the default release tarball) does not carry. loadTracker has always had
     * the answer to that — fp16 on WASM — but it only ran if the SESSION
     * failed, and this probe threw first, so the machine that needed the
     * fallback got "the ONNX weights are not committed to the repo" instead.
     * Decide it here, before anything is downloaded. */
    if ((!head || !head.ok) && !this.fp16) {
      const alt = await fetch(this.dir + 'encoder.fp16.onnx', { method: 'HEAD' })
        .catch(() => null);
      if (alt && alt.ok) {
        this.ep = 'wasm'; this.fp16 = true;
        this.epNote = NO_FP32_NOTE;
        probe = 'encoder.fp16.onnx'; head = alt;
      }
    }
    if (!head || !head.ok) {
      throw new modelsMissing(`${probe} is not there — the ONNX weights are not `
        + 'committed to the repo. See web/README.md: download the release bundle '
        + 'into web/models/, or run ./setup.sh to export them yourself.');
    }
    this.ready = true;
    return this;
  }

  /** The tracker for one resolution. One is kept alive at a time on purpose:
   *  a model set is ~50 MB of fp16 weights plus its GPU buffers, and holding
   *  three of them so that switching a chip is instant is the wrong trade in a
   *  tab. Switching costs the reload, which is 0.4-1.0 s. */
  async loadTracker(log, size, carryNote) {
    await this.manifestOf();                 // settles baseSize and tiers
    let want = size && this.tiers.includes(+size) ? +size : this.baseSize;
    /* `tiers` says what the deployment SHOULD carry; this is the check that it
     * does. Someone who installed only the default tarball has a manifest
     * listing three squares and the files for one, and the answer to that is a
     * track at the default square with a sentence saying why — not a 404 in
     * the middle of a job. */
    if (want !== this.baseSize) {
      try { await this.manifestOf(want); } catch (e) {
        this.tierNote = `the ${want} px graphs are not in this deployment — `
          + `tracking at ${this.baseSize} px instead (see web/README.md)`;
        want = this.baseSize;
      }
    }
    if (want === this.baseSize && this.tierNote && size === this.baseSize) {
      this.tierNote = '';
    }
    // a retry that already knows why it is at the default square keeps its
    // sentence: the clear above is for a chip the visitor pressed, not for this
    if (carryNote) this.tierNote = carryNote;
    if (this.tracker && this.trackerSize === want) return this.tracker;
    if (this.tracker) { await this.releaseTracker(); }
    await this.init();                       // settles ep and fp16 first
    // `fetch('./x')` resolves against the DOCUMENT, but `import('./x')` resolves
    // against this module — which lives one directory deeper. Resolve both the
    // same way the rest of the engine does, or the runtime 404s under engines/.
    const ortBase = new URL(this.ortDir, document.baseURI).href;
    const ort = await import(ortBase + 'ort.all.bundle.min.mjs');
    ort.env.wasm.wasmPaths = ortBase;
    /* Threads need SharedArrayBuffer, and SharedArrayBuffer needs the page to
     * be cross-origin isolated (COOP + COEP), which a plain static host does
     * not do and GitHub Pages cannot. Asking for eight threads there is not a
     * degraded fast path, it is a failed init — so ask for what is actually
     * available. One thread is slow; slow and working beats "no available
     * backend found". */
    const isolated = typeof SharedArrayBuffer !== 'undefined'
      && (globalThis.crossOriginIsolated !== false);
    this.threads = isolated ? Math.min(8, navigator.hardwareConcurrency || 4) : 1;
    ort.env.wasm.numThreads = this.threads;
    ort.env.logLevel = 'error';
    this.ort = ort;
    const t = new WebTracker(ort, { ep: this.ep, fp16: this.fp16,
                                    chain: this.ep === 'webgpu',
                                    dir: this.sizeDir(want) });
    try {
      await t.load(log || (() => {}));
    } catch (e) {
      /* A file that did not come back as model bytes (web/track.js checks every
       * one) is a DEPLOYMENT fault, not a backend fault, and it says which file.
       * It used to arrive here as onnxruntime's "protobuf parsing failed", which
       * named nothing and left the visitor at a dead end. Two recoveries, in
       * order, then a sentence that names the file. */
      if (e && e.modelFetch) {
        // (a) a tier whose graphs are not all in this deployment — or whose 404
        //     an edge cache is still serving in the minutes after they shipped.
        //     The default square is always there, so go back to it and say so.
        if (want !== this.baseSize) {
          return this.loadTracker(log, this.baseSize,
            `the ${want} px graphs are not complete in this deployment `
            + `(${e.file}: ${e.why}) — tracking at ${this.baseSize} px instead`);
        }
        // (b) this GPU has no shader-f16, so the fp32 graphs were asked for and
        //     this deployment does not carry them. WASM does have fp16.
        if (this.ep === 'webgpu' && !this.fp16) {
          this.ep = 'wasm'; this.fp16 = true;
          this.epNote = NO_FP32_NOTE;
          return this.loadTracker(log, want, carryNote);
        }
        throw new modelsMissing(`${e.file} could not be loaded from `
          + `${e.url} — ${e.why}. See web/README.md: download the release `
          + 'bundle into web/models/, or run ./setup.sh to export them yourself.');
      }
      /* One retry, for one specific case: this GPU has no shader-f16, so the
       * fp32 graphs were asked for, and this deployment does not carry them —
       * the release ships fp16 by default and the fp32 set is a separate
       * download. WASM does have fp16, so that is where this goes rather than
       * into a 404 the visitor cannot act on. */
      if (this.ep === 'webgpu' && !this.fp16) {
        this.ep = 'wasm'; this.fp16 = true;
        this.epNote = NO_FP32_NOTE;
        return this.loadTracker(log, want, carryNote);
      }
      // ORT's own message for this is "no available backend found", which tells
      // a visitor nothing. Say which backend was tried and what to do.
      throw new Error(`${this.backendLine()} at ${want} px could not start — `
        + e.message + (this.epNote ? ' · ' + this.epNote : ''));
    }
    this.tracker = t;
    this.trackerSize = want;
    return t;
  }

  /** Give the sessions back before loading another resolution's. */
  async releaseTracker() {
    const t = this.tracker;
    this.tracker = null; this.trackerSize = 0;
    if (t && t.release) { try { await t.release(); } catch (e) { /* gone anyway */ } }
  }

  /** The backend, in the words the stats line uses. */
  backendLine() {
    const px = this.trackerSize ? ` \u00b7 ${this.trackerSize} px` : '';
    return this.ep === 'webgpu'
      ? `WebGPU ${this.fp16 ? 'fp16' : 'fp32'}${px}`
      : `WASM ${this.fp16 ? 'fp16' : 'fp32'}${px}`
        + (this.threads ? ` \u00b7 ${this.threads} thread`
          + (this.threads > 1 ? 's' : '') : '')
        + ' \u00b7 slow';
  }

  /* ------------------------------------------------------------ metadata */
  async meta() {
    // The manifest, not init(): missing weights must not stop the palette and
    // mode tables from loading. checkModels() in app.js reports that separately.
    let S = 768;
    try { S = (await this.manifestOf()).image_size; } catch (e) { /* reported later */ }
    // ask the GPU what it can do before claiming a backend on the chip
    await this.pickBackend();
    const tiers = (this.tiers || [S]).map((size) => Object.assign(
      { size }, TRACK_TIERS[size]
        || { id: String(size), label: `${size} px`, fps: 0 }));
    return {
      palettes: Dither.PALETTES,
      modes: Dither.MODES,
      stable: Dither.STABLE,
      kernels: Object.entries(Dither.KERNELS).map(([id, v]) => ({ id, name: v.name })),
      subject_colors: ['#b0413e', '#2f4f4a', '#7a6a4f', '#3c5a7a', '#8a5a8a', '#4a7a4a'],
      device: this.ep === 'webgpu' ? 'webgpu' : 'wasm',
      backend: this.fp16 ? 'fp16' : 'fp32',
      max_objects: 6,
      /* The same three tiers the server offers, with the same ids, so the
       * quality chips in app.js need to know nothing about which engine is
       * running. Only the ones this deployment actually carries are listed:
       * `tiers` comes out of the default manifest, so a page with just the
       * 768 px set advertises one chip and is telling the truth.
       *
       * 768 stays the default and the only set the first visit downloads. The
       * others load when their chip is picked and replace it, one at a time. */
      track_sizes: tiers,
      default_track_size: S,
      engine: 'browser',
    };
  }

  /** How the frames this track just walked were decoded — the short form, for
   *  the stats line. Empty for a still, which was never decoded. */
  decodeNote() {
    const d = this.lastDecode;
    if (!d) return '';
    return d.path === 'video-seek' ? 'frames from the <video> seek path'
      : `frames via WebCodecs${d.accel ? ' (' + d.accel + ')' : ''}`;
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
    // the File itself, so a different trim range re-decodes without a re-pick
    this.srcFile = file;
    this.srcOpts = opts;
    this.masks.clear(); this.promptFrames.clear();
    this.dropBitmaps(c.w, c.h);
    // Which path ran is a fact about this decode, not about the browser, so it
    // is reported per clip. `supports.decode` is the same object, which is how
    // window.DV_engine() and the verification suite see it without app.js
    // having to carry it.
    this.lastDecode = c.decode || null;
    this.supports.decode = this.lastDecode;
    if (c.decode) console.info('[dither] ' + c.decode.line);
    return { job: 'local', nFrames: c.nFrames, w: c.w, h: c.h, fps: c.fps,
             trimStart: c.trimStart, seconds: c.seconds, decode: c.decode };
  }

  /** A different range out of the same file. The tab still holds the File
   *  handle, so this is a re-decode and nothing is read off disk twice. */
  async reopen(opts = {}) {
    if (!this.srcFile) throw new Error('no clip is open');
    return this.open(this.srcFile, Object.assign({}, this.srcOpts, opts));
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
    this.dropBitmaps(W, H);
    return { job: 'local', kind: 'image', nFrames: 1, w: W, h: H, fps: 1 };
  }

  /** One prompt, one answer, no propagation: encoder + heads_prompt (or
   *  heads_mask), exactly the frame-0 preview path. The soft masks are kept
   *  under this.masks so `mask(id, 0)` hands back the same full-resolution
   *  upsample the tracked flow uses -- the still and the clip compose through
   *  identical pixels. */
  async segmentImage({ objects, imageSize }, onLog) {
    if (!this.clip || !this.clip.still) throw new Error('no still is open');
    const t = await this.loadTracker(onLog, imageSize);
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
             imageSize: S, backend: this.ep, backendLine: this.backendLine(),
             frameIdx: 0,
             note: [notes.join(' · '), this.tierNote || '',
                    this.epNote || ''].filter(Boolean).join(' · ') };
  }

  /* ------------------------------------------------------ the bitmap LRU
   * Decoding a JPEG costs 3-5 ms at 720p and the same frames are asked for
   * again and again: the prompt stage, the hover scrub, the polish window,
   * every export loop. The cache is bounded in BYTES rather than frames, so a
   * 1080p clip holds a third as many as a 720p one and the ceiling is the same
   * 48 MB either way.
   *
   * `frame()` hands out a COPY. app.js closes the bitmaps it evicts from its
   * own LRU, and a cache whose entries can be closed from outside is not a
   * cache. The copy is a GPU-side blit, an order of magnitude under the JPEG
   * decode it replaces. */
  dropBitmaps(w, h) {
    this.bmp.forEach((b) => b.close?.());
    this.bmp.clear();
    this.bmpMax = Math.max(4, Math.floor(BITMAP_BUDGET / Math.max(1, w * h * 4)));
  }

  /** The cached bitmap for frame `i` — the engine's own, do not close it. */
  async bitmapAt(i) {
    const k = Math.max(0, Math.min(this.clip.nFrames - 1, i | 0));
    const hit = this.bmp.get(k);
    if (hit) { this.bmp.delete(k); this.bmp.set(k, hit); return hit; }
    const made = await createImageBitmap(this.clip.frames[k]);
    this.bmp.set(k, made);
    while (this.bmp.size > this.bmpMax) {
      const old = this.bmp.keys().next().value;
      this.bmp.get(old).close?.();
      this.bmp.delete(old);
    }
    return made;
  }

  async frame(i) {
    return createImageBitmap(await this.bitmapAt(i));
  }

  /** RGBA of frame `i` resized into the tracker's square. */
  async trackerInput(i, S) {
    const bmp = await this.bitmapAt(i);
    const c = this._sq || (this._sq = new OffscreenCanvas(S, S));
    if (c.width !== S) { c.width = S; c.height = S; }
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(bmp, 0, 0, S, S);
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
  async previewFrame({ frameIdx, objects, imageSize }, onLog) {
    const t = await this.loadTracker(onLog, imageSize);
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
             imageSize: S, backend: this.ep, backendLine: this.backendLine(),
             frameIdx,
             note: [notes.join(' · '), this.tierNote || '',
                    this.epNote || ''].filter(Boolean).join(' · ') };
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
    const P = Math.round(Math.sqrt(low.length));
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
  async track({ objects, imageSize, only }, onProgress) {
    const t = await this.loadTracker((s) => onProgress
      && onProgress({ done: 0, total: 1, text: s }), imageSize);
    const S = t.man.image_size;
    const N = this.clip.nFrames;
    const t0 = performance.now();
    const plans = [];
    // INCREMENTAL: `only` narrows the run to a subset of the cast. Nothing
    // else has to change -- each subject's logits live under its own key in
    // this.masks and a subject that is not walked is not written, so the ones
    // tracked in an earlier run stay exactly as they were.
    const run = only ? objects.filter((o) => only.includes(o.id)) : objects;
    for (const o of run) {
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
      backendLine: this.backendLine(),
      imageSize: S, steps: total,
      tracked: [...this.masks.keys()], ran: run.map((o) => String(o.id)),
      note: [run.length > 1 ? `${run.length} subjects, one pass each` : '',
             this.decodeNote(), this.tierNote || '', this.epNote || '',
             notes.join(' · ')].filter(Boolean).join(' · '),
    };
  }

  /** Forget one subject: its logits and its prompt frame. The tab is the
   *  storage here, so this IS the delete -- there is nothing else to free. */
  async forget(objId) {
    const id = String(objId);
    const had = this.masks.has(id);
    this.masks.delete(id);
    this.promptFrames.delete(id);
    return { removed: id, existed: had, tracked: [...this.masks.keys()] };
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
  /** The frames one export covers: [first, count].
   *
   *  `params.frame_in` / `params.frame_out` are INCLUSIVE indices into the
   *  clip that is open -- the same window server/render.frame_range() slices.
   *  A trim after the tracking narrows this and nothing else: the decoded
   *  frames and the mask logits stay exactly where they are, and renderFrame
   *  is still called with absolute indices. Omitted = the whole clip, which is
   *  what the sequence exports (which bring their own `source`) always want.
   */
  static window(params, src) {
    const n = src.nFrames;
    const a = Math.max(0, Math.min(n - 1, params.frame_in | 0));
    const b = params.frame_out === undefined || params.frame_out === null
      ? n - 1 : Math.max(a, Math.min(n - 1, params.frame_out | 0));
    return [a, b - a + 1];
  }

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

  /* ------------------------------------------------------- original cut
   * The clip as it came in, cut to exactly the frames the render just used.
   *
   * Same recorder, same canvas, same loop, same `this.clip` frame count — the
   * only difference is that `renderFrame` hands back the decoded frame instead
   * of a dithered one. WebM whatever the render's container was: this tab has
   * no H.264 encoder (see exportWebM), and the format the pair has to agree on
   * is the frame grid, not the codec.
   */
  async exportOriginal(params, onProgress, renderFrame) {
    const r = await this.exportWebM(Object.assign({}, params, { format: 'webm' }),
                                    onProgress, renderFrame, false);
    const src = params.source || this.clip;
    const { w, h, fps } = src;
    const [, nFrames] = BrowserEngine.window(params, src);
    return Object.assign(r, {
      w, h, fps, format: 'webm', matched: (params.format || 'webm') === 'webm',
      note: `${nFrames} frames, undithered — the tab writes WebM only`,
    });
  }

  /** Frames -> palette indices -> one GIF. The whole animation has to be in
   *  memory at once (one byte per pixel per frame: ~0.9 MB a frame at 720p),
   *  which is the reason for the size note in the UI. */
  async exportGIF(params, onProgress, renderFrame) {
    await import('../vendor/gifenc.js');
    const G = globalThis.GifEnc;
    const src = params.source || this.clip;
    const { w, h, fps } = src;
    const [from, nFrames] = BrowserEngine.window(params, src);
    const gfps = Math.max(1, Math.min(50, params.gif_fps || 15));
    // keep every k-th frame so the GIF runs at the asked-for rate in real time
    const step = Math.max(1, Math.round(fps / gfps));
    const palette = gifPalette(params);
    const cache = new Map();
    const frames = [];
    const t0 = performance.now();
    for (let i = 0; i < nFrames; i += step) {
      const img = await renderFrame(from + i, w, h);
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

  /* MediaRecorder over a canvas capture stream. Two ways to drive it, because
   * `requestFrame` is not everywhere:
   *
   *   captureStream(0) + requestFrame()   one frame per call, exactly when we
   *                                       say. The recorder timestamps it on
   *                                       arrival, so pacing the loop to the
   *                                       clip's frame interval gives a file
   *                                       that plays at the right speed.
   *   captureStream(fps)                  the compositor samples the canvas
   *                                       instead. No frame is guaranteed and
   *                                       none is exactly placed.
   *
   * The first is the standard `CanvasCaptureMediaStreamTrack.requestFrame()`;
   * Firefox has the older `CanvasCaptureMediaStream.requestFrame()` on the
   * STREAM instead, which is why both are looked for. Safari has neither, and
   * calling the missing one is what "vtrack.requestFrame is not a function"
   * was. When neither exists the stream is rebuilt at `fps` and each painted
   * frame is given a compositor frame plus its interval to be picked up —
   * and the result is COUNTED rather than assumed: on that path only, the
   * finished WebM is demuxed and the number of frames in it reported, with a
   * note when it is not the number that went in. (Only on that path: counting
   * reads the whole export back into memory, and on the requestFrame path it
   * would be hundreds of megabytes spent confirming a tautology.)
   *
   * If a frame takes longer than the interval to dither, the export runs
   * behind and the result is slower than real time — the returned `fps` says
   * whether it did.
   *
   * The container is WebM (VP9, VP8 on older builds). Writing H.264 MP4 in the
   * tab would mean shipping an encoder; ffmpeg.wasm is ~32 MB and would have to
   * be vendored to keep the no-CDN rule, which is a bigger download than the
   * tracker itself. The server engine writes MP4.
   */
  async exportWebM(params, onProgress, renderFrame, alpha) {
    const src = params.source || this.clip;
    const { w, h, fps } = src;
    const [from, nFrames] = BrowserEngine.window(params, src);
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
    let stream = cv.captureStream(0);
    let vtrack = stream.getVideoTracks()[0];
    /* Which of the three, and it matters beyond "does it throw". Only the
     * track-level call captures the canvas SYNCHRONOUSLY; Firefox's
     * stream-level one queues the grab for the next paint, so two calls inside
     * one paint interval collapse into one frame and the file comes out short.
     * Both of the async kinds therefore get a compositor frame each. */
    const kind = typeof vtrack.requestFrame === 'function' ? 'track'
      : typeof stream.requestFrame === 'function' ? 'stream' : 'compositor';
    let push = kind === 'track' ? () => vtrack.requestFrame()
      : kind === 'stream' ? () => stream.requestFrame() : null;
    if (!push) {
      // nothing can hand this stream a frame, so let the compositor take them
      vtrack.stop();
      stream = cv.captureStream(fps);
      vtrack = stream.getVideoTracks()[0];
    }
    const chunks = [];
    const rec = new MediaRecorder(stream, {
      mimeType: mime,
      videoBitsPerSecond: Math.min(24e6, Math.max(4e6, w * h * fps * 0.15)),
    });
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    const stopped = new Promise((ok) => { rec.onstop = ok; });
    const t0 = performance.now();
    rec.start();
    // `pace_ms` is the matched cut asking to be handed over at the rate the
    // DITHERED pass actually managed. A recorder timestamps a frame when it
    // gets it, so a render that ran slower than real time writes a file whose
    // rate is that slower one; an original paced to 1/fps beside it would be a
    // shorter clip with the same frames in it. Same pacing, same duration.
    const dt = Math.max(1, +params.pace_ms || 1000 / fps);
    for (let i = 0; i < nFrames; i++) {
      const fs = performance.now();
      const img = await renderFrame(from + i, w, h);   // ImageData from app.js
      if (alpha) g.clearRect(0, 0, w, h);
      g.putImageData(img, 0, 0);
      if (push) push();
      if (kind !== 'track') await raf();   // let the compositor take it
      if (onProgress) onProgress({ done: i + 1, total: nFrames,
                                   text: `${i + 1}/${nFrames}` });
      const left = dt - (performance.now() - fs);
      await sleep(Math.max(0, left));
    }
    // Give the encoder the last frame's full interval before cutting — plus a
    // fixed tail, because on the two asynchronous paths the final grab has not
    // necessarily happened yet when the loop ends. Measured: without it Firefox
    // writes 59 of 60 frames.
    await raf();
    await sleep(dt * 2 + (kind === 'track' ? 0 : 250));
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
    // Count what actually landed in the file — but only on the compositor
    // path, where it is genuinely unknown. Counting means reading the whole
    // export back into an ArrayBuffer, and on the requestFrame path that is a
    // few hundred megabytes spent confirming what the loop already guaranteed.
    const written = kind === 'track' ? 0 : await countWebMFrames(blob);
    const short = written && written !== nFrames
      ? `${written} of ${nFrames} frames reached the file — this browser hands `
        + 'the recorder frames asynchronously, so the cut is approximate. Use '
        + 'Chrome or Safari, or the local server, for an exact one'
      : '';
    return {
      url: URL.createObjectURL(blob), mime, ext: 'webm', playable: true,
      alpha: !!alpha,
      frames: nFrames, framesWritten: written || nFrames,
      paced: kind,
      elapsedS: +el.toFixed(2),
      fps: +(nFrames / Math.max(el, 1e-6)).toFixed(2),
      bytes: blob.size,
      note: short || slow || (alpha
        ? `${mime} with an alpha channel — the server writes ProRes 4444 too`
        : `${mime} — the server engine writes H.264 MP4`),
    };
  }

  dispose() {
    this.clip = null;
    this.releaseTracker();
    this.bmp.forEach((b) => b.close?.());
    this.bmp.clear();
    this.masks.clear();
    this.promptFrames.clear();
  }
}

/* A distinguishable error so the UI can show the "download the models" panel
 * instead of a red toast that reads like a crash. */
export class modelsMissing extends Error {
  constructor(msg) { super(msg); this.name = 'ModelsMissing'; this.modelsMissing = true; }
}
