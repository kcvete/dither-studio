/* ---------------------------------------------------------------------------
   DITHER ENGINE — one implementation, used for
     * the live image preview and the client-side PNG export
     * the live video preview
   `dither.py` mirrors this file 1:1 so the server-side MP4 export matches what
   you see. If you change a rule here, change it there.

   Modes
     dots        flicker-free particle look (blue-noise threshold over a cell
                 grid + fixed jitter) — the original Dither Video renderer
     ordered     Bayer / clustered-dot threshold matrix
     bluenoise   64x64 void-and-cluster tile
     whitenoise  hashed random threshold
     errordiff   14 error-diffusion kernels, optional serpentine
     riemersma   Hilbert-curve error diffusion (space-filling, no directional bias)
--------------------------------------------------------------------------- */
'use strict';

const D = (() => {

/* ------------------------------------------------------------ palettes ---- */
const PALETTES = [
  { id: 'bw',        name: 'Black & White',  colors: ['#000000', '#ffffff'] },
  { id: 'sage',      name: 'Sage',           colors: ['#c9d4c5', '#b0413e'] },
  { id: 'forest',    name: 'Forest',         colors: ['#0f1f18', '#d7e3d5'] },
  { id: 'ember',     name: 'Ember',          colors: ['#e8804a', '#f6ece2'] },
  { id: 'mist',      name: 'Mist',           colors: ['#7d8f80', '#c2cfbd', '#ffffff'] },
  { id: 'gameboy',   name: 'Game Boy DMG',   colors: ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'] },
  { id: 'red',       name: 'Red Mono',       colors: ['#1a0000', '#ff2d2d'] },
  { id: 'green',     name: 'Green Mono',     colors: ['#001a05', '#2dff6a'] },
  { id: 'blue',      name: 'Blue Mono',      colors: ['#00081a', '#3d8bff'] },
  { id: 'amber',     name: 'Amber Mono',     colors: ['#1a1000', '#ffb000'] },
  { id: 'cmyk',      name: 'CMYK',           colors: ['#000000', '#00ffff', '#ff00ff', '#ffff00', '#ffffff'] },
  { id: 'rgby',      name: 'RGBY',           colors: ['#000000', '#ff0000', '#00ff00', '#0000ff', '#ffff00', '#ffffff'] },
  { id: 'bwr',       name: 'Black White Red',colors: ['#000000', '#ffffff', '#d02f26'] },
  { id: 'purpgreen', name: 'Purple & Green', colors: ['#2b1b46', '#7b5ea7', '#8fd694', '#f2f0e6'] },
  { id: 'blueyellow',name: 'Blue & Yellow',  colors: ['#10214b', '#3b5bbf', '#f4c542', '#fdf6e3'] },
  { id: 'c64',       name: 'Commodore 64',   colors: ['#000000', '#626262', '#898989', '#adadad', '#ffffff',
                                                      '#9f4e44', '#cb7e75', '#6d5412', '#a1683c', '#c9d487',
                                                      '#9ae29b', '#5cab5e', '#6abfc6', '#887ecb', '#50459b', '#a057a3'] },
  { id: 'grey4',     name: '4 Greys',        colors: ['#000000', '#555555', '#aaaaaa', '#ffffff'] },
  { id: 'grey8',     name: '8 Greys',        colors: ['#000000', '#242424', '#484848', '#6d6d6d',
                                                      '#919191', '#b6b6b6', '#dadada', '#ffffff'] },
];

/* ----------------------------------------------- error-diffusion kernels --- */
/* [dx, dy, weight]; dy>=0, and dx<0 only for dy>0 (nothing behind on the row) */
const KERNELS = {
  'floyd-steinberg':  { name: 'Floyd–Steinberg',      div: 16,  k: [[1,0,7],[-1,1,3],[0,1,5],[1,1,1]] },
  'false-fs':         { name: 'False Floyd–Steinberg',div: 8,   k: [[1,0,3],[0,1,3],[1,1,2]] },
  'jarvis':           { name: 'Jarvis–Judice–Ninke',  div: 48,  k: [[1,0,7],[2,0,5],
                                                                   [-2,1,3],[-1,1,5],[0,1,7],[1,1,5],[2,1,3],
                                                                   [-2,2,1],[-1,2,3],[0,2,5],[1,2,3],[2,2,1]] },
  'stucki':           { name: 'Stucki',               div: 42,  k: [[1,0,8],[2,0,4],
                                                                   [-2,1,2],[-1,1,4],[0,1,8],[1,1,4],[2,1,2],
                                                                   [-2,2,1],[-1,2,2],[0,2,4],[1,2,2],[2,2,1]] },
  'atkinson':         { name: 'Atkinson',             div: 8,   k: [[1,0,1],[2,0,1],
                                                                   [-1,1,1],[0,1,1],[1,1,1],[0,2,1]] },
  'burkes':           { name: 'Burkes',               div: 32,  k: [[1,0,8],[2,0,4],
                                                                   [-2,1,2],[-1,1,4],[0,1,8],[1,1,4],[2,1,2]] },
  'sierra3':          { name: 'Sierra 3',             div: 32,  k: [[1,0,5],[2,0,3],
                                                                   [-2,1,2],[-1,1,4],[0,1,5],[1,1,4],[2,1,2],
                                                                   [-1,2,2],[0,2,3],[1,2,2]] },
  'sierra2':          { name: 'Sierra 2',             div: 16,  k: [[1,0,4],[2,0,3],
                                                                   [-2,1,1],[-1,1,2],[0,1,3],[1,1,2],[2,1,1]] },
  'sierra-lite':      { name: 'Sierra 2-4A (lite)',   div: 4,   k: [[1,0,2],[-1,1,1],[0,1,1]] },
  'fan93':            { name: 'Fan 93',               div: 16,  k: [[1,0,7],[-2,1,1],[-1,1,3],[0,1,5]] },
  'shiau-fan':        { name: 'Shiau–Fan',            div: 8,   k: [[1,0,4],[-2,1,1],[-1,1,1],[0,1,2]] },
  'shiau-fan2':       { name: 'Shiau–Fan 2',          div: 16,  k: [[1,0,8],[-3,1,1],[-2,1,1],[-1,1,2],[0,1,4]] },
  'stevenson-arce':   { name: 'Stevenson–Arce',       div: 200, k: [[2,0,32],
                                                                   [-3,1,12],[-1,1,26],[1,1,30],[3,1,16],
                                                                   [-2,2,12],[0,2,26],[2,2,12],
                                                                   [-3,3,5],[-1,3,12],[1,3,12],[3,3,5]] },
  'simple-2d':        { name: 'Simple 2D',            div: 2,   k: [[1,0,1],[0,1,1]] },
};

/* ------------------------------------------------- threshold matrices ----- */
function bayer(n) {                      // n = 2 | 4 | 8 | 16
  let m = [[0, 2], [3, 1]];
  while (m.length < n) {
    const s = m.length, o = [];
    for (let y = 0; y < s * 2; y++) o.push(new Array(s * 2));
    for (let y = 0; y < s; y++) for (let x = 0; x < s; x++) {
      const v = m[y][x] * 4;
      o[y][x] = v; o[y][x + s] = v + 2; o[y + s][x] = v + 3; o[y + s][x + s] = v + 1;
    }
    m = o;
  }
  const N = n * n, out = new Float32Array(N);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) out[y * n + x] = (m[y][x] + 0.5) / N;
  return out;
}

/* clustered-dot (halftone) matrix: threshold grows with distance from a screen
   dot centre, so "on" pixels clump into growing circles instead of scattering */
function clustered(n) {
  const out = new Float32Array(n * n), vals = [];
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const cx = (x + 0.5) / n - 0.5, cy = (y + 0.5) / n - 0.5;
    // 45-degree rotated screen reads as a classic newsprint halftone
    const u = (cx + cy), v = (cx - cy);
    vals.push([Math.sqrt(u * u + v * v), y * n + x]);
  }
  vals.sort((a, b) => a[0] - b[0]);
  vals.forEach(([, i], r) => { out[i] = (r + 0.5) / (n * n); });
  return out;
}

const MATRIX_CACHE = new Map();
function matrix(kind, n) {
  const key = kind + n;
  if (!MATRIX_CACHE.has(key)) MATRIX_CACHE.set(key, kind === 'halftone' ? clustered(n) : bayer(n));
  return MATRIX_CACHE.get(key);
}

/* portable per-pixel hash — identical in dither.py (see hash01 there) */
function hash01(i, j, salt, seed) {
  let x = ((Math.imul(i, 73856093) >>> 0) ^ (Math.imul(j, 19349663) >>> 0)
        ^ (Math.imul(salt, 83492791) >>> 0) ^ (Math.imul(seed, 0x9E3779B1) >>> 0)) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0; x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0; x = Math.imul(x, 0x846ca68b) >>> 0;
  return ((x ^ (x >>> 16)) >>> 0) / 4294967296;
}

/* --------------------------------------------------------- blue noise ----- */
/* generated server-side (FFT) and injected once via setBlueNoise() */
let BLUE = null;
function setBlueNoise(tile) { BLUE = tile; }
function blueNoise() { return BLUE; }

/* ------------------------------------------------------------ helpers ----- */
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);
function hexRGB(h) {
  h = h.replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function paletteBytes(colors) {
  const p = new Uint8Array(colors.length * 3);
  colors.forEach((c, i) => { const [r, g, b] = hexRGB(c); p[i * 3] = r; p[i * 3 + 1] = g; p[i * 3 + 2] = b; });
  return p;
}

/* nearest palette entry by squared euclidean distance in RGB (0..255 floats) */
function nearest(pal, n, r, g, b) {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < n; i++) {
    const dr = r - pal[i * 3], dg = g - pal[i * 3 + 1], db = b - pal[i * 3 + 2];
    const d = dr * dr + dg * dg + db * db;
    if (d < bd) { bd = d; bi = i; }
  }
  return bi;
}

/* median-cut palette extraction, used by the "from image" preset */
function extractPalette(rgba, w, h, k) {
  const px = [];
  const step = Math.max(1, Math.floor(Math.sqrt((w * h) / 20000)));
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    const p = (y * w + x) * 4;
    px.push([rgba[p], rgba[p + 1], rgba[p + 2]]);
  }
  let boxes = [px];
  while (boxes.length < k) {
    let bi = -1, br = -1;
    boxes.forEach((b, i) => {
      if (b.length < 2) return;
      for (let c = 0; c < 3; c++) {
        let lo = 255, hi = 0;
        for (const p of b) { if (p[c] < lo) lo = p[c]; if (p[c] > hi) hi = p[c]; }
        if (hi - lo > br) { br = hi - lo; bi = i; }
      }
    });
    if (bi < 0) break;
    const b = boxes[bi];
    let ch = 0, best = -1;
    for (let c = 0; c < 3; c++) {
      let lo = 255, hi = 0;
      for (const p of b) { if (p[c] < lo) lo = p[c]; if (p[c] > hi) hi = p[c]; }
      if (hi - lo > best) { best = hi - lo; ch = c; }
    }
    b.sort((p, q) => p[ch] - q[ch]);
    const mid = b.length >> 1;
    boxes.splice(bi, 1, b.slice(0, mid), b.slice(mid));
  }
  return boxes.filter((b) => b.length).map((b) => {
    let r = 0, g = 0, bl = 0;
    for (const p of b) { r += p[0]; g += p[1]; bl += p[2]; }
    const n = b.length;
    const hx = (v) => Math.round(v / n).toString(16).padStart(2, '0');
    return '#' + hx(r) + hx(g) + hx(bl);
  });
}

/* --------------------------------------------------------- tone shaping --- */
/* returns a 256-entry LUT per channel for brightness / contrast / gamma / invert */
function toneLUT(p) {
  const lut = new Float32Array(256);
  const br = p.brightness || 0, co = (p.contrast === undefined ? 1 : p.contrast), ga = p.gamma || 1;
  for (let i = 0; i < 256; i++) {
    let v = i / 255;
    v = (v - 0.5) * co + 0.5 + br;
    v = clamp01(v);
    if (ga !== 1) v = Math.pow(v, ga);
    if (p.invert) v = 1 - v;
    lut[i] = v * 255;
  }
  return lut;
}

/* --------------------------------------------------------------- core ----- */
/* Dither `src` (RGBA bytes, w x h) into `dst` (RGBA bytes, same size).
   `gate` is an optional Float32Array(w*h) of 0..1 coverage: where gate is 0 the
   pixel is left untouched (that is how per-subject video masks are applied).
   Returns the number of pixels written. */
function ditherRGBA(src, dst, w, h, p, gate) {
  const pal = paletteBytes(p.palette), np = p.palette.length;
  const lut = toneLUT(p);
  const seed = p.seed === undefined ? 7 : p.seed;
  const strength = p.strength === undefined ? 1 : p.strength;
  const N = w * h;

  // spread: with a 2-colour palette we want the full range; with many colours the
  // perturbation must shrink or the image turns to noise
  const spread = 255 * strength / Math.max(1, np - 1);

  if (p.mode === 'errordiff' || p.mode === 'riemersma') {
    // float working buffer so quantisation error can be carried forward
    const buf = new Float32Array(N * 3);
    for (let i = 0, q = 0; i < N; i++, q += 3) {
      const s = i * 4;
      buf[q] = lut[src[s]]; buf[q + 1] = lut[src[s + 1]]; buf[q + 2] = lut[src[s + 2]];
    }
    if (p.mode === 'errordiff') errorDiffuse(buf, w, h, pal, np, p, strength, gate);
    else riemersma(buf, w, h, pal, np, p, strength, gate);
    let n = 0;
    for (let i = 0, q = 0; i < N; i++, q += 3) {
      if (gate && gate[i] <= 0) continue;
      const s = i * 4;
      dst[s] = buf[q]; dst[s + 1] = buf[q + 1]; dst[s + 2] = buf[q + 2]; dst[s + 3] = 255;
      n++;
    }
    return n;
  }

  // ---- threshold modes: fully parallel, temporally stable
  let thr = null, mn = 0;
  if (p.mode === 'ordered' || p.mode === 'halftone') {
    mn = p.matrix || 4;
    thr = matrix(p.mode === 'halftone' ? 'halftone' : 'bayer', mn);
  } else if (p.mode === 'bluenoise') {
    mn = 64; thr = BLUE;
  }
  let n = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (gate && gate[i] <= 0) continue;
      const s = i * 4;
      let t;
      if (thr) t = thr[(y % mn) * mn + (x % mn)];
      else t = hash01(y, x, 11, seed);            // whitenoise
      const off = (t - 0.5) * spread;
      const k = nearest(pal, np,
        clamp255(lut[src[s]] + off),
        clamp255(lut[src[s + 1]] + off),
        clamp255(lut[src[s + 2]] + off));
      dst[s] = pal[k * 3]; dst[s + 1] = pal[k * 3 + 1]; dst[s + 2] = pal[k * 3 + 2]; dst[s + 3] = 255;
      n++;
    }
  }
  return n;
}

function errorDiffuse(buf, w, h, pal, np, p, strength, gate) {
  const kd = KERNELS[p.algo] || KERNELS['floyd-steinberg'];
  const kern = kd.k, div = kd.div, kn = kern.length;
  const serp = !!p.serpentine;
  for (let y = 0; y < h; y++) {
    const rev = serp && (y & 1);
    for (let xi = 0; xi < w; xi++) {
      const x = rev ? w - 1 - xi : xi;
      const i = y * w + x, q = i * 3;
      const r = buf[q], g = buf[q + 1], b = buf[q + 2];
      const k = nearest(pal, np, r, g, b);
      const nr = pal[k * 3], ng = pal[k * 3 + 1], nb = pal[k * 3 + 2];
      buf[q] = nr; buf[q + 1] = ng; buf[q + 2] = nb;
      if (gate && gate[i] <= 0) continue;      // outside the subject: don't spread
      const er = (r - nr) * strength / div, eg = (g - ng) * strength / div,
            eb = (b - nb) * strength / div;
      for (let t = 0; t < kn; t++) {
        const dx = rev ? -kern[t][0] : kern[t][0], dy = kern[t][1], wt = kern[t][2];
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= w || ny >= h) continue;
        const nq = (ny * w + nx) * 3;
        buf[nq] += er * wt; buf[nq + 1] += eg * wt; buf[nq + 2] += eb * wt;
      }
    }
  }
}

/* Hilbert d2xy for a 2^order lattice */
function hilbertXY(order, d) {
  let rx, ry, t = d, x = 0, y = 0;
  for (let s = 1; s < order; s *= 2) {
    rx = 1 & (t / 2); ry = 1 & (t ^ rx);
    if (ry === 0) { if (rx === 1) { x = s - 1 - x; y = s - 1 - y; } const tmp = x; x = y; y = tmp; }
    x += s * rx; y += s * ry; t = Math.floor(t / 4);
  }
  return [x, y];
}

function riemersma(buf, w, h, pal, np, p, strength, gate) {
  const QL = p.queue || 16, RATIO = p.ratio || 16;
  let order = 1; while (order < w || order < h) order *= 2;
  const wq = new Float32Array(QL);
  for (let i = 0; i < QL; i++) wq[i] = Math.pow(RATIO, (i + 1) / QL - 1) * strength;
  const er = new Float32Array(QL), eg = new Float32Array(QL), eb = new Float32Array(QL);
  let head = 0;
  const total = order * order;
  for (let d = 0; d < total; d++) {
    const [x, y] = hilbertXY(order, d);
    if (x >= w || y >= h) continue;
    const i = y * w + x, q = i * 3;
    let ar = 0, ag = 0, ab = 0;
    for (let t = 0; t < QL; t++) {
      const j = (head + t) % QL;
      ar += er[j] * wq[t]; ag += eg[j] * wq[t]; ab += eb[j] * wq[t];
    }
    const r = buf[q] + ar / QL, g = buf[q + 1] + ag / QL, b = buf[q + 2] + ab / QL;
    const k = nearest(pal, np, r, g, b);
    const nr = pal[k * 3], ng = pal[k * 3 + 1], nb = pal[k * 3 + 2];
    buf[q] = nr; buf[q + 1] = ng; buf[q + 2] = nb;
    er[head] = gate && gate[i] <= 0 ? 0 : r - nr;
    eg[head] = gate && gate[i] <= 0 ? 0 : g - ng;
    eb[head] = gate && gate[i] <= 0 ? 0 : b - nb;
    head = (head + 1) % QL;
  }
}


/* ------------------------------------------------ pixel scale + compose --- */
/* Box-mean downscale by an integer factor, rounded half-up so it matches
   render.py's downscale() (np.floor(x+0.5)). */
function pixDownRGBA(src, w, h, P) {
  if (P <= 1) return { data: src, w, h };
  const w2 = Math.floor(w / P), h2 = Math.floor(h / P);
  if (w2 < 1 || h2 < 1) return { data: src, w, h };
  const out = new Uint8ClampedArray(w2 * h2 * 4);
  const inv = 1 / (P * P);
  for (let y = 0; y < h2; y++) {
    for (let x = 0; x < w2; x++) {
      let r = 0, g = 0, b = 0;
      for (let dy = 0; dy < P; dy++) {
        let p = ((y * P + dy) * w + x * P) * 4;
        for (let dx = 0; dx < P; dx++, p += 4) { r += src[p]; g += src[p + 1]; b += src[p + 2]; }
      }
      const q = (y * w2 + x) * 4;
      out[q] = Math.floor(r * inv + 0.5);
      out[q + 1] = Math.floor(g * inv + 0.5);
      out[q + 2] = Math.floor(b * inv + 0.5);
      out[q + 3] = 255;
    }
  }
  return { data: out, w: w2, h: h2 };
}

function pixDownGate(g, w, h, P) {
  if (P <= 1) return g;
  const w2 = Math.floor(w / P), h2 = Math.floor(h / P);
  if (w2 < 1 || h2 < 1) return g;
  const out = new Float32Array(w2 * h2), inv = 1 / (P * P);
  for (let y = 0; y < h2; y++) for (let x = 0; x < w2; x++) {
    let a = 0;
    for (let dy = 0; dy < P; dy++) {
      let p = (y * P + dy) * w + x * P;
      for (let dx = 0; dx < P; dx++, p++) a += g[p];
    }
    out[y * w2 + x] = a * inv;
  }
  return out;
}

function pixUpRGBA(src, w2, h2, P, w, h) {
  if (P <= 1) return src;
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(Math.floor(y / P), h2 - 1);
    for (let x = 0; x < w; x++) {
      const sx = Math.min(Math.floor(x / P), w2 - 1);
      const s = (sy * w2 + sx) * 4, d = (y * w + x) * 4;
      out[d] = src[s]; out[d + 1] = src[s + 1]; out[d + 2] = src[s + 2];
      out[d + 3] = src[s + 3];
    }
  }
  return out;
}

/* Build one finished frame. Mirrors render.py::_frame_pixels exactly.
     srcRGBA  full-resolution source pixels
     masks    array of Float32Array(w*h) coverage, 0..1 (empty = whole frame)
     palettes [backgroundPalette, subject1Palette, ...]
   With p.alpha the flat background is left transparent (alpha 0) and only the
   dithered subjects are opaque — the transparent exports. `overlay` compose and
   whole-frame dithers have no background to remove and stay opaque, exactly as
   render.py::_frame_pixels does it.
   Returns Uint8ClampedArray(w*h*4). */
function composeFrame(srcRGBA, w, h, masks, p, palettes, bg) {
  const P = Math.max(1, p.pixel | 0);
  const d = pixDownRGBA(srcRGBA, w, h, P);
  const w2 = d.w, h2 = d.h, N2 = w2 * h2;

  if (!masks.length) {
    const out = new Uint8ClampedArray(d.data);
    ditherRGBA(d.data, out, w2, h2, Object.assign({}, p, { palette: palettes[0] }), null);
    for (let q = 3, n = N2 * 4; q < n; q += 4) out[q] = 255;
    return pixUpRGBA(out, w2, h2, P, w, h);
  }

  const ms = masks.map((m) => pixDownGate(m, w, h, P));
  const owner = new Int8Array(N2), inside = new Uint8Array(N2);
  for (let q = 0; q < N2; q++) {
    let best = 0, bv = -1;
    for (let k = 0; k < ms.length; k++) if (ms[k][q] > bv) { bv = ms[k][q]; best = k; }
    owner[q] = best; inside[q] = bv >= 0.5 ? 1 : 0;
  }

  const out = new Uint8ClampedArray(N2 * 4);
  if (p.compose === 'overlay') {
    const tmp = new Uint8ClampedArray(d.data);
    ditherRGBA(d.data, tmp, w2, h2, Object.assign({}, p, { palette: palettes[0] }), null);
    out.set(tmp);
  } else {
    const c = hexRGB(bg), a0 = p.alpha ? 0 : 255;
    for (let q = 0, n = N2 * 4; q < n; q += 4) {
      out[q] = c[0]; out[q + 1] = c[1]; out[q + 2] = c[2]; out[q + 3] = a0;
    }
  }

  for (let k = 0; k < ms.length; k++) {
    const gate = new Float32Array(N2);
    let any = false;
    for (let q = 0; q < N2; q++) if (inside[q] && owner[q] === k) { gate[q] = 1; any = true; }
    if (!any) continue;
    const pal = palettes[k + 1] || palettes[0];
    ditherRGBA(d.data, out, w2, h2, Object.assign({}, p, { palette: pal }), gate);
  }
  return pixUpRGBA(out, w2, h2, P, w, h);
}

/* which modes hold still frame to frame */
const STABLE = { dots: true, ordered: true, halftone: true, bluenoise: true, whitenoise: true,
                 errordiff: false, riemersma: false };

const MODES = [
  { id: 'dots',       name: 'Dots',        note: 'particle swarm, flicker-free' },
  { id: 'bluenoise',  name: 'Blue noise',  note: 'organic grain, flicker-free' },
  { id: 'ordered',    name: 'Bayer',       note: 'crosshatch, flicker-free' },
  { id: 'halftone',   name: 'Halftone',    note: 'newsprint screen, flicker-free' },
  { id: 'whitenoise', name: 'White noise', note: 'raw grain, flicker-free' },
  { id: 'errordiff',  name: 'Error diffusion', note: 'sharpest detail — flickers on video' },
  { id: 'riemersma',  name: 'Riemersma',   note: 'Hilbert curve — flickers on video' },
];

return { PALETTES, KERNELS, MODES, STABLE, bayer, clustered, matrix, hash01,
         pixDownRGBA, pixDownGate, pixUpRGBA, composeFrame,
         setBlueNoise, blueNoise, hexRGB, paletteBytes, nearest, extractPalette,
         toneLUT, ditherRGBA, hilbertXY };
})();

const Dither = D;
if (typeof module !== 'undefined') module.exports = D;
