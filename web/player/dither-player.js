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

  if (doc.frames.length > 65535) {
    throw new Error('.dots carries n_frames as a uint16: ' + doc.frames.length
      + ' frames is over the 65,535 limit (36 minutes at 30 fps)');
  }
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

/* ========================================================== transitions ===
 * Four ways for one dot cloud to become another. All of them return the same
 * thing -- a list of frames, each frame a pair of clouds {a, b}: the dots that
 * still belong to the OUTGOING item and the dots that already belong to the
 * INCOMING one. Colour is carried by that split, which is what lets a sequence
 * keep a per-item colour through a transition without a per-dot palette.
 *
 *   morph     the dot flight below: matched by Hilbert rank, staggered, curled
 *   scatter   A disperses on noise and gravity while B assembles out of the
 *             same mess -- two overlapping clouds, no pairing at all
 *   cut       nothing between the two items
 *   density   A re-grids to huge cells, coarsens once more, becomes B at that
 *             coarseness, then refines back down. Five cheap morphs on
 *             progressively smaller dot sets rather than one expensive one, so
 *             it reads as the picture dissolving into its own resolution.
 *
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
 * Returns an array of { a, b } frames, NOT including a or b themselves: `a`
 * holds the dots still reading as the outgoing cloud, `b` the ones that have
 * become the incoming one. A dot with a partner changes hands at the halfway
 * point of its OWN flight, so the colour change is staggered exactly the way
 * the movement is; a dot with no partner belongs to the side it exists on.
 */
function morphPairs(a, b, opts = {}) {
  const w = opts.w || 1280, h = opts.h || 720, fps = opts.fps || 30;
  const dur = opts.durationMs || 900;
  const stagger = dur * (opts.stagger === undefined ? 0.38 : opts.stagger);
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
    // one side can be empty (an item with no dots on its first frame, a cut
    // down to nothing); `|| 0` keeps the arithmetic below finite, and the
    // born/dies flags mean the bogus end is never actually drawn
    sx[k] = a[fa * 2] || 0; sy[k] = a[fa * 2 + 1] || 0;
    tx[k] = b[fb * 2] || 0; ty[k] = b[fb * 2 + 1] || 0;
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
    const A = [], B = [];
    for (let k = 0; k < n; k++) {
      const p = (t - dl[k]) / dur;
      if (born[k]) {                        // spawns at its own moment
        if (t < pop[k]) continue;
        B.push(tx[k] | 0, ty[k] | 0);
        continue;
      }
      if (dies[k]) {                        // and the surplus at the other end
        if (t >= pop[k]) continue;
        A.push(sx[k] | 0, sy[k] | 0);
        continue;
      }
      let x, y;
      if (p <= 0) { x = sx[k]; y = sy[k]; } else if (p >= 1) { x = tx[k]; y = ty[k]; } else {
        const e = easeInOut(p), s = Math.sin(Math.PI * e);
        const dx = tx[k] - sx[k], dy = ty[k] - sy[k];
        x = sx[k] + dx * e - dy * cu[k] * s;
        y = sy[k] + dy * e + dx * cu[k] * s;
      }
      (p < 0.5 ? A : B).push(clamp(Math.round(x), 0, w - 1),
                             clamp(Math.round(y), 0, h - 1));
    }
    frames.push({ a: Uint16Array.from(A), b: Uint16Array.from(B) });
  }
  return frames;
}

/** The old single-cloud signature, kept because `demo.html` and anything that
 *  imported it treat a morph as one swarm with one colour. */
function buildMorph(a, b, opts = {}) {
  return morphPairs(a, b, opts).map((f) => {
    const xy = new Uint16Array(f.a.length + f.b.length);
    xy.set(f.a, 0); xy.set(f.b, f.a.length);
    return xy;
  });
}

/* ------------------------------------------------------------- scatter ---
 * No pairing, no matching: A's dots are thrown outwards on a random heading
 * with gravity under them and vanish one by one, while B's dots come in from
 * scattered positions of their own and settle. The two overlap for most of the
 * transition, so the swarm reads as a cloud of debris that reassembles rather
 * than a flight from one shape to the other. Dots that leave the frame are
 * dropped rather than clamped -- a clamped dot piles up on the edge and the
 * edge lights up like a bar. */
function scatterPairs(a, b, opts = {}) {
  const w = opts.w || 1280, h = opts.h || 720, fps = opts.fps || 30;
  const dur = opts.durationMs || 900;
  const rnd = mulberry32(opts.seed || 11);
  const spread = (opts.spread || 0.55) * Math.min(w, h);
  const grav = (opts.gravity === undefined ? 1.0 : opts.gravity) * Math.min(w, h);
  const nA = a.length >> 1, nB = b.length >> 1;
  const av = new Float32Array(nA * 2), ago = new Float32Array(nA);
  for (let i = 0; i < nA; i++) {
    const ang = rnd() * TAU, sp = (0.35 + 0.9 * rnd()) * spread;
    av[i * 2] = Math.cos(ang) * sp;
    av[i * 2 + 1] = Math.sin(ang) * sp - 0.35 * spread;   // a little upward bias
    ago[i] = 0.35 + 0.6 * rnd();                          // when it gives up
  }
  const bs = new Float32Array(nB * 2), bin = new Float32Array(nB);
  for (let i = 0; i < nB; i++) {
    const ang = rnd() * TAU, sp = (0.4 + 0.9 * rnd()) * spread;
    bs[i * 2] = b[i * 2] + Math.cos(ang) * sp;
    bs[i * 2 + 1] = b[i * 2 + 1] + Math.sin(ang) * sp;
    bin[i] = 0.12 + 0.42 * rnd();                         // when it appears
  }
  const total = Math.max(1, Math.round(dur / 1000 * fps));
  const frames = [];
  for (let f = 1; f <= total; f++) {
    const t = f / total;
    const A = [], B = [];
    for (let i = 0; i < nA; i++) {
      if (t >= ago[i]) continue;
      const x = a[i * 2] + av[i * 2] * t;
      const y = a[i * 2 + 1] + av[i * 2 + 1] * t + 0.5 * grav * t * t;
      const xi = Math.round(x), yi = Math.round(y);
      if (xi < 0 || yi < 0 || xi >= w || yi >= h) continue;
      A.push(xi, yi);
    }
    for (let i = 0; i < nB; i++) {
      if (t < bin[i]) continue;
      const p = easeInOut(clamp((t - bin[i]) / Math.max(1e-3, 1 - bin[i]), 0, 1));
      const x = bs[i * 2] + (b[i * 2] - bs[i * 2]) * p;
      const y = bs[i * 2 + 1] + (b[i * 2 + 1] - bs[i * 2 + 1]) * p;
      const xi = Math.round(x), yi = Math.round(y);
      if (xi < 0 || yi < 0 || xi >= w || yi >= h) continue;
      B.push(xi, yi);
    }
    frames.push({ a: Uint16Array.from(A), b: Uint16Array.from(B) });
  }
  return frames;
}

/* ------------------------------------------------------- density fade ---
 * Snap a cloud to a coarser grid and the duplicates collapse: a 6,000-dot
 * subject at cell 16 is a few hundred dots that still read as the same shape.
 * The transition walks A down that ladder, swaps to B at the coarsest rung,
 * and walks back up -- five short morphs over small clouds instead of one long
 * one over big ones, which is both cheaper and a different effect: the picture
 * dissolves into its own resolution rather than flying across the frame. */
function regrid(xy, cell) {
  if (cell <= 1) return xy;
  const seen = new Set(), out = [];
  for (let i = 0; i < xy.length; i += 2) {
    const gx = Math.floor(xy[i] / cell), gy = Math.floor(xy[i + 1] / cell);
    const key = gy * 65536 + gx;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(Math.round(gx * cell + cell / 2), Math.round(gy * cell + cell / 2));
  }
  return Uint16Array.from(out);
}

function densityPairs(a, b, opts = {}) {
  const cell = Math.max(2, opts.cell || 4);
  const rungs = [2, 4].map((k) => k * cell);
  const seed = opts.seed || 11;
  // A -> A coarse -> A coarsest -> B coarsest -> B coarse -> B
  const ladder = [
    { xy: a, side: 'a' },
    { xy: regrid(a, rungs[0]), side: 'a' },
    { xy: regrid(a, rungs[1]), side: 'a' },
    { xy: regrid(b, rungs[1]), side: 'b' },
    { xy: regrid(b, rungs[0]), side: 'b' },
    { xy: b, side: 'b' },
  ];
  const hops = ladder.length - 1;
  const each = Math.max(60, (opts.durationMs || 900) / hops);
  const frames = [];
  for (let i = 0; i < hops; i++) {
    const from = ladder[i], to = ladder[i + 1];
    const hop = morphPairs(from.xy, to.xy, {
      w: opts.w, h: opts.h, fps: opts.fps, durationMs: each,
      seed: seed + i * 131, curl: 0.06, stagger: 0.12,
    });
    for (const f of hop) {
      // inside A's half of the ladder every dot is still A's, and vice versa;
      // only the middle hop actually changes hands
      if (from.side === to.side) {
        const one = new Uint16Array(f.a.length + f.b.length);
        one.set(f.a, 0); one.set(f.b, f.a.length);
        frames.push(from.side === 'a' ? { a: one, b: EMPTY } : { a: EMPTY, b: one });
      } else {
        frames.push(f);
      }
    }
  }
  return frames;
}

/* ------------------------------------------------- thinning, for the flight
 * A dot cloud made by one of the pixel dither modes is one lit PIXEL per dot:
 * tens of thousands of them a frame. Flying that dot for dot is neither
 * affordable nor legible, and it is not what a morph is for. So a join thins
 * whichever side is over the cap and flies the survivors — the cloud visibly
 * loosens into particles, which is the effect — while the dots left behind are
 * shed in place over the first fifth of the flight and the incoming ones spawn
 * in place over the last fifth. Frame 0 of a transition is therefore the
 * outgoing item's last frame and the final frame the incoming item's first, at
 * full density and to within a dot or two of the same count, with the thinning
 * only in between.
 *
 * The thinning is density-weighted rather than uniform: the frame is bucketed
 * into a coarse grid, every occupied bucket keeps at least one dot, and the
 * rest of the budget is shared out in proportion to how many dots a bucket
 * holds. A thin limb or an outline survives that; taking every Nth dot in scan
 * order does not. The seed comes from the join, so a sequence always flies the
 * same particles.
 */
const PARTICLE_CAP = 8000;

/** -> { keep, rest }: `keep` is at most `cap` dots, `rest` is everything else
 *  in a deterministic shuffle so it can be dissolved a slice at a time. */
function thinCloud(xy, cap, seed, bucket) {
  const n = xy.length >> 1;
  if (n <= cap) return { keep: xy, rest: EMPTY };
  const g = Math.max(4, bucket || 24);
  const rnd = mulberry32((seed | 0) || 1);
  const cells = new Map();
  for (let i = 0; i < n; i++) {
    const q = (((xy[i * 2 + 1] / g) | 0) * 4096) + ((xy[i * 2] / g) | 0);
    let a = cells.get(q);
    if (!a) { a = []; cells.set(q, a); }
    a.push(i);
  }
  // a deterministic shuffle inside every bucket, so the quota is not a
  // scan-order slice
  for (const a of cells.values()) {
    for (let i = a.length - 1; i > 0; i--) {
      const j = (rnd() * (i + 1)) | 0;
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
  }
  const B = cells.size;
  const budget = Math.max(0, cap - B);
  const keep = [], rest = [];
  for (const a of cells.values()) {
    const quota = Math.min(a.length, 1 + Math.floor(budget * (a.length / n)));
    for (let i = 0; i < a.length; i++) {
      (i < quota ? keep : rest).push(a[i]);
    }
  }
  // interleave the leftovers across buckets, so a partial spawn fills the whole
  // shape rather than one corner of it
  for (let i = rest.length - 1; i > 0; i--) {
    const j = (rnd() * (i + 1)) | 0;
    const t = rest[i]; rest[i] = rest[j]; rest[j] = t;
  }
  const pick = (idx) => {
    const out = new Uint16Array(idx.length * 2);
    for (let i = 0; i < idx.length; i++) {
      out[i * 2] = xy[idx[i] * 2]; out[i * 2 + 1] = xy[idx[i] * 2 + 1];
    }
    return out;
  };
  return { keep: pick(keep), rest: pick(rest) };
}

/**
 * One transition, whichever kind. `kind` is 'morph' | 'scatter' | 'cut' |
 * 'density'; anything else is a morph.
 */
function buildTransition(a, b, opts = {}) {
  const kind = opts.kind || 'morph';
  if (kind === 'cut') return [];
  if (kind === 'scatter') return scatterPairs(a, b, opts);
  if (kind === 'density') return densityPairs(a, b, opts);
  return morphPairs(a, b, opts);
}

const TRANSITIONS = [
  { id: 'morph', name: 'morph',
    note: 'dots fly from one shape to the other, matched neighbour to neighbour' },
  { id: 'scatter', name: 'scatter',
    note: 'the first shape blows apart while the second assembles out of it' },
  { id: 'cut', name: 'cut', note: 'no transition at all — one frame to the next' },
  { id: 'density', name: 'density fade',
    note: 'the shape coarsens to huge cells, becomes the next one there, and refines back' },
];

/**
 * Stitch items and transitions into one document.
 *
 *   items   [{ name, frames:[Uint16Array], color }]                 one track
 *           [{ name, tracks:[{ frames:[Uint16Array], color }] }]    several
 *           each may carry { transition: { kind, ms } } — the join BEFORE it —
 *           and { cell }, the dot grid it was made on, which a density fade
 *           coarsens from.
 *   opts    { w, h, fps, dotpx, bg, durationMs, kind, seed, cell, cap }
 *
 * `cap` is the most dots a single transition will actually fly (8,000 by
 * default, 0 to disable): a pixel-mode item is one dot per lit pixel, and the
 * flight is thinned to the cap and handed back at full density on arrival. See
 * `thinCloud`.
 *
 * Colour is per item (and per track inside an item), not per document: the
 * output has one .dots subject track per DISTINCT COLOUR, and every item's dots
 * land in the track its colour owns. A transition splits its dots between the
 * outgoing colour and the incoming one, dot by dot, so a red swarm becoming a
 * green one changes colour the way it changes shape instead of all at once.
 */
function buildSequence(items, opts = {}) {
  const w = opts.w || 1280, h = opts.h || 720, fps = opts.fps || 30;
  const bg = opts.bg || '#c9d4c5';
  const fallback = opts.color || '#b0413e';
  const norm = items.map((it, i) => ({
    name: it.name || ('#' + (i + 1)),
    transition: it.transition || null,
    // the dot cell this item was made at — a density fade coarsens from it, so
    // an item with its own cell gets its own ladder rather than the strip's
    cell: it.cell || 0,
    tracks: (it.tracks && it.tracks.length ? it.tracks
      : [{ frames: it.frames || [], color: it.color || fallback }])
      .map((t) => ({ frames: t.frames || [], color: t.color || fallback })),
  }));

  const colours = [];
  const colourOf = (c) => {
    let i = colours.indexOf(c);
    if (i < 0) { colours.push(c); i = colours.length - 1; }
    return i;
  };
  norm.forEach((it) => it.tracks.forEach((t) => colourOf(t.color)));
  if (!colours.length) colours.push(fallback);

  const frames = [];        // [ [Uint16Array per colour] per frame ]
  const marks = [];
  const blank = () => colours.map(() => []);
  const push = (buckets) => frames.push(buckets.map((b) => Uint16Array.from(b)));
  const add = (buckets, idx, xy) => {
    const b = buckets[idx];
    for (let i = 0; i < xy.length; i++) b.push(xy[i]);
  };

  norm.forEach((it, i) => {
    if (i > 0) {
      const prev = norm[i - 1];
      const kind = (it.transition && it.transition.kind) || opts.kind || 'morph';
      const ms = (it.transition && it.transition.ms) || opts.durationMs || 900;
      const pairs = [];
      const n = Math.max(prev.tracks.length, it.tracks.length);
      const cap = opts.cap === undefined ? PARTICLE_CAP : opts.cap;
      let len = 0, thinned = 0;
      for (let k = 0; k < n; k++) {
        const pt = prev.tracks[k], nt = it.tracks[k];
        const fromAll = (pt && pt.frames[pt.frames.length - 1]) || EMPTY;
        const toAll = (nt && nt.frames[0]) || EMPTY;
        const seed = (opts.seed || 11) + i * 7919 + k * 101;
        // a pixel-mode item is one dot per lit pixel: cap what actually flies
        const A = cap ? thinCloud(fromAll, cap, seed, opts.bucket)
                      : { keep: fromAll, rest: EMPTY };
        const B = cap ? thinCloud(toAll, cap, seed + 1, opts.bucket)
                      : { keep: toAll, rest: EMPTY };
        thinned += (A.rest.length + B.rest.length) >> 1;
        const tween = buildTransition(A.keep, B.keep, {
          w, h, fps, durationMs: ms, kind,
          cell: it.cell || prev.cell || opts.cell,
          seed,
        });
        pairs.push({ tween,
                     a: colourOf((pt || it.tracks[0]).color),
                     b: colourOf((nt || prev.tracks[0]).color),
                     shed: A.rest, grow: B.rest });
        len = Math.max(len, tween.length);
      }
      if (len) {
        marks.push({ kind, start: frames.length, frames: len, ms, thinned,
                     name: prev.name + ' → ' + it.name });
        // the dots the thinning left behind do not fly: A's dissolve in place
        // over the first fifth, B's spawn in place over the last fifth, so the
        // two ends of the flight are the items themselves at full density
        const HOLD = 0.2;
        const slice = (buckets, idx, xy, frac) => {
          const m = Math.round(Math.min(1, Math.max(0, frac)) * (xy.length >> 1));
          const b = buckets[idx];
          for (let q = 0; q < m * 2; q++) b.push(xy[q]);
        };
        for (let f = 0; f < len; f++) {
          const buckets = blank();
          const t = len > 1 ? f / (len - 1) : 1;
          for (const p of pairs) {
            const fr = p.tween[f];
            if (fr) { add(buckets, p.a, fr.a); add(buckets, p.b, fr.b); }
            if (p.shed.length) slice(buckets, p.a, p.shed, 1 - t / HOLD);
            if (p.grow.length) slice(buckets, p.b, p.grow, (t - (1 - HOLD)) / HOLD);
          }
          push(buckets);
        }
      } else {
        marks.push({ kind: 'cut', start: frames.length, frames: 0, ms: 0,
                     name: prev.name + ' → ' + it.name });
      }
    }
    const nf = it.tracks.reduce((a, t) => Math.max(a, t.frames.length), 0);
    marks.push({ kind: 'item', name: it.name, start: frames.length, frames: nf });
    for (let f = 0; f < nf; f++) {
      const buckets = blank();
      it.tracks.forEach((t) => {
        const xy = t.frames[f];
        if (xy && xy.length) add(buckets, colourOf(t.color), xy);
      });
      push(buckets);
    }
  });

  return {
    w, h, fps, dotpx: opts.dotpx || 3,
    palette: [bg].concat(colours), bgIndex: 0, bg,
    subjects: colours.map((color) => ({ color })),
    frames,
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
         paintFrame, buildMorph, morphPairs, scatterPairs, densityPairs, regrid,
         buildTransition, TRANSITIONS, buildSequence, hilbertOrder, hexRGB,
         thinCloud, PARTICLE_CAP,
         easeInOut, mulberry32, version: 1 };
})();

globalThis.DitherPlayer = DitherPlayer;
if (typeof module !== 'undefined') module.exports = DitherPlayer;
