/* ---------------------------------------------------------------------------
   DITHER PLAYER — plays a .dots.gz file on a <canvas>. No dependencies, no
   network, no build step, ~1 file.

   A .dots.gz is not a video. It is the dot positions themselves: for every
   frame, for every subject, the integer centre of every dot the renderer lit.
   Playing it back is a replay, not a re-dither — the player draws exactly the
   squares Dither Studio drew, at exactly the same pixels, and can do it at any
   background colour, dot colour or dot size you like afterwards.

   LOADING. A classic <script> cannot contain `export`, so this file publishes
   itself on `globalThis.DitherPlayer` (and on `module.exports` under Node).
   `dither-player.mjs` next to it re-exports those as real ES module bindings.

       <script src="dither-player.js"></script>
       const p = new DitherPlayer.Player(canvas, { loop: true });
       await p.load('clip.dots.gz'); p.play();

       import { Player } from './dither-player.mjs';

   ============================================================ FILE FORMAT ===
   gzip( body ), extension .dots.gz. Everything little-endian.

     off  size            field
     0    4               magic "DOTS"
     4    1               version = 1
     5    1               flags (0)
     6    2  uint16       width
     8    2  uint16       height
     10   2  uint16       n_frames
     12   1  uint8        fps
     13   1  uint8        dotpx        dot square, in pixels
     14   1  uint8        n_palette    1..255
     15   1  uint8        n_subjects   1..255
     16   1  uint8        bg_index     palette entry the background uses
     17   1  uint8        reserved = 0
     18   n_palette*3     palette, RGB bytes
     ..   n_subjects      one uint8 palette index per subject (its dot colour)
     ..   frames          for each frame, for each subject:
                            varint  count
                            count x  zigzag-varint dx, zigzag-varint dy
                          dx/dy are deltas from the previous dot of the same
                          subject in the same frame (first dot from 0,0). Dots
                          come out of the renderer in cell-scan order, so the
                          deltas are small and the whole thing gzips well.

   A .dots.json variant carries the same fields as readable JSON, for debugging.
--------------------------------------------------------------------------- */
'use strict';

const DitherPlayer = (() => {

const MAGIC = 0x53544f44;                      // 'DOTS' little-endian
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

/* ------------------------------------------------------------ byte sink --- */
function Sink(cap) { this.b = new Uint8Array(cap || 1 << 16); this.n = 0; }
Sink.prototype.need = function (k) {
  if (this.n + k <= this.b.length) return;
  let c = this.b.length * 2;
  while (c < this.n + k) c *= 2;
  const nb = new Uint8Array(c); nb.set(this.b.subarray(0, this.n)); this.b = nb;
};
Sink.prototype.u8 = function (v) { this.need(1); this.b[this.n++] = v & 0xff; };
Sink.prototype.u16 = function (v) {
  this.need(2); this.b[this.n++] = v & 0xff; this.b[this.n++] = (v >> 8) & 0xff;
};
Sink.prototype.varint = function (v) {
  this.need(5);
  v >>>= 0;
  while (v >= 0x80) { this.b[this.n++] = (v & 0x7f) | 0x80; v >>>= 7; }
  this.b[this.n++] = v;
};
/* zigzag: small negatives cost one byte, not five */
Sink.prototype.svarint = function (v) { this.varint((v << 1) ^ (v >> 31)); };
Sink.prototype.done = function () { return this.b.slice(0, this.n); };

function Src(b) { this.b = b; this.n = 0; }
Src.prototype.u8 = function () { return this.b[this.n++]; };
Src.prototype.u16 = function () { const v = this.b[this.n] | (this.b[this.n + 1] << 8); this.n += 2; return v; };
Src.prototype.varint = function () {
  let v = 0, s = 0, c;
  do { c = this.b[this.n++]; v |= (c & 0x7f) << s; s += 7; } while (c & 0x80);
  return v >>> 0;
};
Src.prototype.svarint = function () { const v = this.varint(); return (v >>> 1) ^ -(v & 1); };

/* -------------------------------------------------------------- palette --- */
function hexRGB(h) {
  h = String(h).replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  return [parseInt(h.slice(0, 2), 16) | 0, parseInt(h.slice(2, 4), 16) | 0,
          parseInt(h.slice(4, 6), 16) | 0];
}
const hx = (v) => (v & 0xff).toString(16).padStart(2, '0');
const rgbHex = (r, g, b) => '#' + hx(r) + hx(g) + hx(b);

/* =============================================================== codec ===
 * A doc is:
 *   { w, h, fps, dotpx, palette:['#rrggbb',…], bgIndex, subjects:[{color}],
 *     frames: [ [Uint16Array(x,y,x,y,…) per subject] per frame ] }
 */
function encode(doc) {
  const pal = doc.palette.slice(0, 255);
  const subs = doc.subjects.length ? doc.subjects : [{ color: pal[pal.length - 1] }];
  const idxOf = (c) => {
    const i = pal.indexOf(c);
    if (i >= 0) return i;
    if (pal.length < 255) { pal.push(c); return pal.length - 1; }
    return pal.length - 1;
  };
  const subIdx = subs.map((s) => idxOf(s.color));
  const bgIndex = doc.bgIndex === undefined ? idxOf(doc.bg || pal[0]) : doc.bgIndex;

  const o = new Sink(1 << 20);
  o.u8(0x44); o.u8(0x4f); o.u8(0x54); o.u8(0x53);       // "DOTS"
  o.u8(1); o.u8(0);
  o.u16(doc.w); o.u16(doc.h); o.u16(doc.frames.length);
  o.u8(doc.fps || 30); o.u8(doc.dotpx || 3);
  o.u8(pal.length); o.u8(subs.length); o.u8(bgIndex); o.u8(0);
  for (const c of pal) { const [r, g, b] = hexRGB(c); o.u8(r); o.u8(g); o.u8(b); }
  for (const i of subIdx) o.u8(i);
  for (const frame of doc.frames) {
    for (let k = 0; k < subs.length; k++) {
      const xy = frame[k] || EMPTY;
      const n = xy.length >> 1;
      o.varint(n);
      let px = 0, py = 0;
      for (let i = 0; i < n; i++) {
        o.svarint(xy[i * 2] - px); o.svarint(xy[i * 2 + 1] - py);
        px = xy[i * 2]; py = xy[i * 2 + 1];
      }
    }
  }
  return o.done();
}
const EMPTY = new Uint16Array(0);

function decode(bytes) {
  const b = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const dv = new DataView(b.buffer, b.byteOffset, 4);
  if (dv.getUint32(0, true) !== MAGIC) throw new Error('not a .dots file');
  const s = new Src(b);
  s.n = 4;
  const version = s.u8(); s.u8();
  if (version !== 1) throw new Error('unsupported .dots version ' + version);
  const w = s.u16(), h = s.u16(), nFrames = s.u16();
  const fps = s.u8(), dotpx = s.u8(), nPal = s.u8(), nSub = s.u8();
  const bgIndex = s.u8(); s.u8();
  const palette = [];
  for (let i = 0; i < nPal; i++) palette.push(rgbHex(s.u8(), s.u8(), s.u8()));
  const subjects = [];
  for (let i = 0; i < nSub; i++) { const k = s.u8(); subjects.push({ color: palette[k], index: k }); }
  const frames = [];
  for (let f = 0; f < nFrames; f++) {
    const per = [];
    for (let k = 0; k < nSub; k++) {
      const n = s.varint();
      const xy = new Uint16Array(n * 2);
      let px = 0, py = 0;
      for (let i = 0; i < n; i++) {
        px += s.svarint(); py += s.svarint();
        xy[i * 2] = px; xy[i * 2 + 1] = py;
      }
      per.push(xy);
    }
    frames.push(per);
  }
  return { w, h, fps, dotpx, palette, bgIndex, bg: palette[bgIndex], subjects, frames };
}

/* --------------------------------------------------------------- gzip ----
 * CompressionStream in a browser, node:zlib under Node. Both are built in;
 * neither is a dependency. */
async function gzip(bytes) {
  if (typeof CompressionStream !== 'undefined') {
    const cs = new CompressionStream('gzip');
    const w = cs.writable.getWriter();
    w.write(bytes); w.close();
    return new Uint8Array(await new Response(cs.readable).arrayBuffer());
  }
  const zlib = require('node:zlib');
  return new Uint8Array(zlib.gzipSync(bytes, { level: 9 }));
}
async function gunzip(bytes) {
  if (typeof DecompressionStream !== 'undefined') {
    const ds = new DecompressionStream('gzip');
    const w = ds.writable.getWriter();
    w.write(bytes); w.close();
    return new Uint8Array(await new Response(ds.readable).arrayBuffer());
  }
  const zlib = require('node:zlib');
  return new Uint8Array(zlib.gunzipSync(bytes));
}

const pack = async (doc) => gzip(encode(doc));
const unpack = async (bytes) => decode(await gunzip(
  bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)));

/* JSON variant — same numbers, readable, ~8x the bytes. For debugging. */
function toJSON(doc) {
  return {
    format: 'dither-studio/dots', version: 1,
    w: doc.w, h: doc.h, fps: doc.fps, dotpx: doc.dotpx,
    palette: doc.palette, bgIndex: doc.bgIndex,
    subjects: doc.subjects.map((s) => ({ color: s.color })),
    frames: doc.frames.map((f) => f.map((xy) => Array.from(xy))),
  };
}
function fromJSON(j) {
  return {
    w: j.w, h: j.h, fps: j.fps, dotpx: j.dotpx, palette: j.palette,
    bgIndex: j.bgIndex || 0, bg: j.palette[j.bgIndex || 0],
    subjects: j.subjects, frames: j.frames.map((f) => f.map((a) => Uint16Array.from(a))),
  };
}

/* ============================================================ rasteriser ===
 * The one rule this file exists to keep: a dot lands on exactly the pixels
 * app.js::renderDots and server/render.py::_frame_dots put it on. Both draw a
 * dotpx square centred on the integer dot position with `half = dotpx >> 1`,
 * and CLAMP each pixel to the frame instead of clipping it — a dot near the
 * edge smears against it rather than being cut off. fillRect would clip, so
 * this writes pixels itself. */
function paintFrame(out, w, h, doc, frame, opts) {
  const dp = Math.max(1, (opts.dotpx || doc.dotpx) | 0), half = dp >> 1;
  const bg = opts.bg === null ? null : hexRGB(opts.bg || doc.bg || '#000000');
  const alpha = opts.transparent ? 0 : 255;
  if (bg) {
    for (let p = 0, n = w * h * 4; p < n; p += 4) {
      out[p] = bg[0]; out[p + 1] = bg[1]; out[p + 2] = bg[2]; out[p + 3] = alpha;
    }
  } else {
    out.fill(0);
  }
  let lit = 0;
  for (let k = 0; k < frame.length; k++) {
    const xy = frame[k];
    const col = hexRGB((opts.colors && opts.colors[k])
      || (doc.subjects[k] && doc.subjects[k].color) || '#ffffff');
    for (let i = 0; i < xy.length; i += 2) {
      const xc = xy[i], yc = xy[i + 1];
      lit++;
      for (let dy = 0; dy < dp; dy++) {
        const yy = clamp(yc + dy - half, 0, h - 1);
        for (let dx = 0; dx < dp; dx++) {
          const xx = clamp(xc + dx - half, 0, w - 1);
          const p = (yy * w + xx) * 4;
          out[p] = col[0]; out[p + 1] = col[1]; out[p + 2] = col[2]; out[p + 3] = 255;
        }
      }
    }
  }
  return lit;
}

/* ================================================================ morph ===
 * Port of the image demo's transition, in clip pixels instead of unit space:
 *
 *   MATCH    both dot clouds are sorted by Hilbert-curve index and paired rank
 *            for rank, so neighbours stay neighbours and the swarm flows
 *            instead of scrambling.
 *   COUNTS   the two clouds are never the same size. The overlap is paired
 *            1:1; the surplus on whichever side is bigger POPS — a dot with no
 *            partner appears at its target (or vanishes from its source) at its
 *            own staggered moment. Dots here are binary: there is no fade to
 *            fade with, so the stagger is what makes it read as a dissolve.
 *   STAGGER  per-dot delay = (0.55*noise + 0.45*spatial wave) * 0.38 * duration
 *   FLIGHT   ease-in-out cubic, plus a perpendicular curl that peaks mid-flight
 */
const TAU = Math.PI * 2;
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

const HN = 1024;
function hilbert(x, y) {
  let rx, ry, d = 0;
  for (let s = HN >> 1; s > 0; s >>= 1) {
    rx = (x & s) > 0 ? 1 : 0;
    ry = (y & s) > 0 ? 1 : 0;
    d += s * s * ((3 * rx) ^ ry);
    if (ry === 0) {
      if (rx === 1) { x = s - 1 - x; y = s - 1 - y; }
      const t = x; x = y; y = t;
    }
  }
  return d;
}
/** Ranks of a dot cloud (a flat x,y Uint16Array) along the Hilbert curve. */
function hilbertOrder(xy, w, h) {
  const n = xy.length >> 1;
  const keys = new Float64Array(n), ord = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const gx = clamp(Math.round(xy[i * 2] / Math.max(1, w - 1) * (HN - 1)), 0, HN - 1);
    const gy = clamp(Math.round(xy[i * 2 + 1] / Math.max(1, h - 1) * (HN - 1)), 0, HN - 1);
    keys[i] = hilbert(gx, gy);
    ord[i] = i;
  }
  const arr = Array.from(ord);
  arr.sort((a, b) => keys[a] - keys[b] || a - b);
  return Int32Array.from(arr);
}

/**
 * Tween one dot cloud into another.
 *   a, b     Uint16Array of x,y pairs (the last frame of A, the first of B)
 *   opts     { w, h, fps, durationMs = 900, seed = 11, curl = 0.42 }
 * Returns an array of Uint16Array frames, NOT including a or b themselves.
 */
function buildMorph(a, b, opts = {}) {
  const w = opts.w || 1280, h = opts.h || 720, fps = opts.fps || 30;
  const dur = opts.durationMs || 900;
  const stagger = dur * 0.38;
  const curlAmp = opts.curl === undefined ? 0.42 : opts.curl;
  const rnd = mulberry32(opts.seed || 11);
  const nA = a.length >> 1, nB = b.length >> 1;
  const oA = hilbertOrder(a, w, h), oB = hilbertOrder(b, w, h);
  const n = Math.max(nA, nB);
  const m = Math.min(nA, nB);

  // one record per moving dot: source, target, delay, curl, and whether it has
  // a partner at either end
  const sx = new Float32Array(n), sy = new Float32Array(n);
  const tx = new Float32Array(n), ty = new Float32Array(n);
  const dl = new Float32Array(n), cu = new Float32Array(n);
  const pop = new Float32Array(n);      // when a partnerless dot appears/vanishes
  const born = new Uint8Array(n), dies = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    // rank k of the shorter cloud is paired; beyond it, the surplus pops
    const ia = k < nA ? oA[k] : -1;
    const ib = k < nB ? oB[k] : -1;
    // when one side is longer, spread its surplus over the other's ranks so the
    // extras start (or end) near where they belong instead of all in one corner
    const fa = ia >= 0 ? ia : oA[Math.floor(k * nA / Math.max(1, n)) % Math.max(1, nA)];
    const fb = ib >= 0 ? ib : oB[Math.floor(k * nB / Math.max(1, n)) % Math.max(1, nB)];
    sx[k] = a[fa * 2]; sy[k] = a[fa * 2 + 1];
    tx[k] = b[fb * 2]; ty[k] = b[fb * 2 + 1];
    born[k] = ia < 0 ? 1 : 0;              // no source: pops in at the target
    dies[k] = ib < 0 ? 1 : 0;              // no target: pops out at the source
    const wave = 0.5 + 0.5 * Math.sin(sx[k] / w * 5.1 + sy[k] / h * 4.3);
    dl[k] = (0.55 * rnd() + 0.45 * wave) * stagger;
    cu[k] = (rnd() - 0.5) * curlAmp;
    // a dot with no partner pops at its own moment, spread across the WHOLE
    // flight rather than the stagger window — otherwise every surplus dot
    // arrives inside the same three frames and the dissolve reads as a cut
    pop[k] = dl[k] + dur * rnd();
  }
  void m;

  const total = Math.max(1, Math.round((stagger + dur) / 1000 * fps));
  const frames = [];
  for (let f = 1; f <= total; f++) {
    const t = f / fps * 1000;
    const xy = [];
    for (let k = 0; k < n; k++) {
      const p = (t - dl[k]) / dur;
      if (born[k]) {                        // spawns at its own moment
        if (t < pop[k]) continue;
        xy.push(tx[k] | 0, ty[k] | 0);
        continue;
      }
      if (dies[k]) {                        // and the surplus at the other end
        if (t >= pop[k]) continue;
        xy.push(sx[k] | 0, sy[k] | 0);
        continue;
      }
      let x, y;
      if (p <= 0) { x = sx[k]; y = sy[k]; } else if (p >= 1) { x = tx[k]; y = ty[k]; } else {
        const e = easeInOut(p), s = Math.sin(Math.PI * e);
        const dx = tx[k] - sx[k], dy = ty[k] - sy[k];
        x = sx[k] + dx * e - dy * cu[k] * s;
        y = sy[k] + dy * e + dx * cu[k] * s;
      }
      xy.push(clamp(Math.round(x), 0, w - 1), clamp(Math.round(y), 0, h - 1));
    }
    frames.push(Uint16Array.from(xy));
  }
  return frames;
}

/**
 * Stitch segments and transitions into one document.
 *   segments  [{ frames: [Uint16Array], color }] — each already one subject
 *   opts      { w, h, fps, dotpx, bg, durationMs, seed }
 * Every segment is flattened onto ONE subject track with ONE palette: a
 * sequence is a single swarm changing shape, not a multi-track timeline.
 */
function buildSequence(segments, opts = {}) {
  const w = opts.w || 1280, h = opts.h || 720, fps = opts.fps || 30;
  const bg = opts.bg || '#c9d4c5';
  const color = opts.color || (segments[0] && segments[0].color) || '#b0413e';
  const frames = [];
  const marks = [];
  segments.forEach((seg, i) => {
    if (i > 0) {
      const prev = frames[frames.length - 1] || new Uint16Array(0);
      const next = seg.frames[0] || new Uint16Array(0);
      const dur = (seg.transitionMs === undefined ? opts.durationMs : seg.transitionMs) || 900;
      const tween = buildMorph(prev, next, { w, h, fps, durationMs: dur,
                                             seed: (opts.seed || 11) + i * 7919 });
      marks.push({ kind: 'morph', start: frames.length, frames: tween.length, ms: dur });
      for (const f of tween) frames.push(f);
    }
    marks.push({ kind: 'segment', name: seg.name || ('#' + (i + 1)),
                 start: frames.length, frames: seg.frames.length });
    for (const f of seg.frames) frames.push(f);
  });
  return {
    w, h, fps, dotpx: opts.dotpx || 3,
    palette: [bg, color], bgIndex: 0, bg,
    subjects: [{ color }],
    frames: frames.map((f) => [f]),
    marks,
  };
}

/* =============================================================== player === */
class Player {
  /** canvas, { loop, bg, colors, dotpx, transparent, autoplay, onFrame } */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: !!opts.transparent });
    this.opts = Object.assign({ loop: true, autoplay: false }, opts);
    this.doc = null;
    this.frame = 0;
    this.playing = false;
    this._raf = 0;
    this._t0 = 0;
    this._f0 = 0;
    this._paint = 0;              // ms spent in the last paint
    this._painted = 0;            // frames painted since play() — for fps
  }

  get nFrames() { return this.doc ? this.doc.frames.length : 0; }
  get fps() { return this.doc ? this.doc.fps : 30; }
  get duration() { return this.nFrames / Math.max(1, this.fps); }

  /** url of a .dots.gz (or .dots / .dots.json), or an already-decoded doc */
  async load(src) {
    if (src && src.frames && !(src instanceof Uint8Array)) { this.setDoc(src); return this.doc; }
    const r = await fetch(src, { cache: 'no-store' });
    if (!r.ok) throw new Error(`${src}: HTTP ${r.status}`);
    const buf = new Uint8Array(await r.arrayBuffer());
    let doc;
    if (buf[0] === 0x1f && buf[1] === 0x8b) doc = await unpack(buf);
    else if (buf[0] === 0x44 && buf[1] === 0x4f) doc = decode(buf);
    else doc = fromJSON(JSON.parse(new TextDecoder().decode(buf)));
    this.setDoc(doc);
    return doc;
  }

  setDoc(doc) {
    this.doc = doc;
    this.canvas.width = doc.w; this.canvas.height = doc.h;
    this._img = this.ctx.createImageData(doc.w, doc.h);
    this.frame = 0;
    this.draw(0);
    if (this.opts.autoplay) this.play();
  }

  draw(i) {
    if (!this.doc) return 0;
    const n = this.nFrames;
    const f = ((i % n) + n) % n;
    this.frame = f;
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const lit = paintFrame(this._img.data, this.doc.w, this.doc.h, this.doc,
                           this.doc.frames[f], this.opts);
    this.ctx.putImageData(this._img, 0, 0);
    this._paint = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
    this._painted++;
    if (this.opts.onFrame) this.opts.onFrame(f, lit, this._paint);
    return lit;
  }

  seek(i) { this.pause(); this.draw(i); }

  play() {
    if (this.playing || !this.doc) return;
    this.playing = true;
    this._t0 = performance.now();
    this._f0 = this.frame;
    this._painted = 0;
    const tick = () => {
      if (!this.playing) return;
      const el = (performance.now() - this._t0) / 1000;
      let want = this._f0 + Math.floor(el * this.fps);
      if (!this.opts.loop && want >= this.nFrames) { this.draw(this.nFrames - 1); this.pause(); return; }
      if (want !== this.frame) this.draw(want);
      this._raf = requestAnimationFrame(tick);
    };
    this._raf = requestAnimationFrame(tick);
  }

  pause() {
    this.playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this._raf = 0;
  }

  toggle() { return this.playing ? this.pause() : this.play(); }

  /** live look changes — bg, colors, dotpx, transparent */
  set(opts) { Object.assign(this.opts, opts); this.draw(this.frame); }

  /** wall-clock frames per second since play() started */
  measuredFps() {
    const el = (performance.now() - this._t0) / 1000;
    return el > 0 ? this._painted / el : 0;
  }

  /** ms of CPU the last frame's rasterisation took */
  paintMs() { return this._paint; }
}

return { Player, encode, decode, gzip, gunzip, pack, unpack, toJSON, fromJSON,
         paintFrame, buildMorph, buildSequence, hilbertOrder, hexRGB,
         easeInOut, mulberry32, version: 1 };
})();

globalThis.DitherPlayer = DitherPlayer;
if (typeof module !== 'undefined') module.exports = DitherPlayer;
