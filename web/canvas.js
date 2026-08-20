/* ---------------------------------------------------------------------------
   THE CANVAS — one aspect ratio, two ways of getting there.

   Everything this tool renders has, until now, been the size of what came in:
   a still at its own pixels, a clip at the 720p-normalised frames the tracker
   and the dither already agreed on. A canvas is the other half of that
   sentence — 9:16 for TikTok, 1:1 for a grid, 4:5 for a feed — and it is a
   pure geometry problem, so it lives here, on its own, with no DOM and no
   engine in sight.

   The whole thing is ONE affine map from source pixels to canvas pixels:

       X = (x - cx) * k + tw/2        k = cover scale x zoom
       Y = (y - cy) * k + th/2        (cx, cy) = where the crop is centred

   which `place()` hands back as {k, x0, y0} so a caller can either
   `drawImage(src, x0, y0, sw*k, sh*k)` (the frame) or invert it per pixel
   (`warpMask`, the masks). server/render.py applies the same three numbers to
   the same source, which is why an export and its matched original cut and its
   .dots.gz all land on the same grid.

   The two looks want different things out of it:

     overlay / whole-frame   real footage is visible, so the crop window has to
                             STAY INSIDE the source (clamp: true) or it would
                             show bars of nothing. No zoom by default: the crop
                             is the largest rectangle of the target aspect that
                             fits, and it is scaled up to the target's pixels.
     cutout                  the background is flat, so there is nothing to
                             fall off the edge of. The crop is not clamped and
                             the subject can sit dead centre wherever it is in
                             frame; the dots are re-measured on the canvas
                             itself, so they come out at the canvas's own
                             resolution rather than a scaled-up 720p.

   `smoothPath` is the third piece: a mask centroid per frame is jittery, and a
   crop that follows it frame by frame is unwatchable. A gaussian over +/-15
   frames is what turns it into a camera move.
--------------------------------------------------------------------------- */
'use strict';

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const even = (v) => Math.max(2, Math.round(v / 2) * 2);

/* The presets. Pixel sizes are the ones the platforms actually want, and every
 * one of them is even in both axes because H.264 in yuv420p cannot encode an
 * odd dimension. `ar` is what a custom size is measured against. */
export const PRESETS = [
  { id: 'source', label: 'source', note: 'whatever came in' },
  { id: '16:9', label: '16:9', w: 1920, h: 1080, note: 'landscape · 1920×1080' },
  { id: '9:16', label: '9:16', w: 1080, h: 1920, note: 'TikTok · Reels · Shorts · 1080×1920' },
  { id: '1:1', label: '1:1', w: 1080, h: 1080, note: 'square · 1080×1080' },
  { id: '4:5', label: '4:5', w: 1080, h: 1350, note: 'feed portrait · 1080×1350' },
  { id: 'custom', label: 'custom', note: 'any width × height' },
];

export const presetOf = (id) => PRESETS.find((p) => p.id === id) || PRESETS[0];

/** The output size a canvas setting asks for, or null for "the source's own".
 *  `cv` is {preset, w, h}; a custom preset carries its own w/h. */
export function targetSize(cv, sw, sh) {
  if (!cv || !cv.preset || cv.preset === 'source') return null;
  if (cv.preset === 'custom') {
    const w = even(cv.w || sw || 1280), h = even(cv.h || sh || 720);
    return (w === sw && h === sh) ? null : { w, h };
  }
  const p = presetOf(cv.preset);
  if (!p.w) return null;
  return { w: p.w, h: p.h };
}

/** The scale at which the largest rectangle of the TARGET's aspect that fits
 *  inside the source fills the target exactly. Equivalently: cover. */
export function baseScale(sw, sh, tw, th) {
  return Math.max(tw / sw, th / sh);
}

/** The crop rectangle, in SOURCE pixels, that one plan is showing. */
export function cropRect(sw, sh, tw, th, zoom) {
  const k = baseScale(sw, sh, tw, th) * (zoom || 1);
  return { w: tw / k, h: th / k, k };
}

/**
 * The affine map, as the three numbers everything downstream needs.
 *
 *   {k, x0, y0}   X = x * k + x0,  Y = y * k + y0
 *   {cx, cy}      where the crop actually ended up (after clamping)
 *   {clamped}     whether the clamp moved it — the UI says so
 */
export function place({ sw, sh, tw, th, cx, cy, zoom = 1, clamp: doClamp = false }) {
  const k = baseScale(sw, sh, tw, th) * (zoom || 1);
  const halfW = tw / (2 * k), halfH = th / (2 * k);
  let X = cx === undefined || cx === null ? sw / 2 : cx;
  let Y = cy === undefined || cy === null ? sh / 2 : cy;
  let moved = false;
  if (doClamp) {
    const nx = halfW * 2 <= sw ? clamp(X, halfW, sw - halfW) : sw / 2;
    const ny = halfH * 2 <= sh ? clamp(Y, halfH, sh - halfH) : sh / 2;
    moved = Math.abs(nx - X) > 0.5 || Math.abs(ny - Y) > 0.5;
    X = nx; Y = ny;
  }
  return { k, x0: tw / 2 - X * k, y0: th / 2 - Y * k, cx: X, cy: Y, clamped: moved };
}

/** How much a plan is enlarging the source. > 1 means real footage is being
 *  upscaled, which is a thing the UI has to say out loud. */
export const upscaleOf = (plan) => plan.k;

/* ------------------------------------------------------------- the path */
/**
 * A jittery per-frame centre -> a camera move.
 *
 *   pts     [{x, y, ok}] — `ok` false where the subject was not on that frame
 *   radius  half-window, in frames (15 => +/-15, a full second at 30 fps)
 *
 * Gaps are filled from the nearest frame that had a subject BEFORE the smooth
 * runs, so a subject that leaves for ten frames does not drag the crop to the
 * origin and back. A series with nothing in it at all comes back untouched and
 * the caller falls back to the frame's centre.
 */
export function smoothPath(pts, radius = 15) {
  const n = pts.length;
  if (!n) return [];
  const have = pts.map((p) => !!(p && p.ok));
  if (!have.some(Boolean)) return pts.map((p) => ({ x: p ? p.x : 0, y: p ? p.y : 0 }));
  // fill forwards, then backwards
  const fx = new Float64Array(n), fy = new Float64Array(n);
  let lx = 0, ly = 0, seen = false;
  for (let i = 0; i < n; i++) {
    if (have[i]) { lx = pts[i].x; ly = pts[i].y; seen = true; }
    fx[i] = lx; fy[i] = ly;
    if (!seen) { fx[i] = NaN; fy[i] = NaN; }
  }
  for (let i = n - 1, bx = NaN, by = NaN; i >= 0; i--) {
    if (!Number.isNaN(fx[i])) { bx = fx[i]; by = fy[i]; } else { fx[i] = bx; fy[i] = by; }
  }
  const sigma = Math.max(1, radius / 2.5);
  const w = [];
  for (let d = -radius; d <= radius; d++) w.push(Math.exp(-(d * d) / (2 * sigma * sigma)));
  const out = [];
  for (let i = 0; i < n; i++) {
    let sx = 0, sy = 0, sw = 0;
    for (let d = -radius; d <= radius; d++) {
      const j = clamp(i + d, 0, n - 1), g = w[d + radius];
      sx += fx[j] * g; sy += fy[j] * g; sw += g;
    }
    out.push({ x: sx / sw, y: sy / sw });
  }
  return out;
}

/** Does the whole clip's subject fit inside one fixed crop? That is the
 *  question "follow or hold still" actually asks: a subject that never leaves
 *  a static frame should not make the frame move. */
export function fitsStatic(union, crop, margin = 0.04) {
  if (!union) return true;
  const mw = crop.w * (1 - margin * 2), mh = crop.h * (1 - margin * 2);
  return (union.x1 - union.x0) <= mw && (union.y1 - union.y0) <= mh;
}

/* --------------------------------------------------------------- masks */
/** The box outside which a float coverage mask is exactly zero, or null. */
export function maskBox(m, w, h) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0, q = 0; y < h; y++) {
    for (let x = 0; x < w; x++, q++) {
      if (m[q] <= 0) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  return x1 < 0 ? null : { x0, y0, x1, y1 };
}

/** Area-weighted centroid + box of a float coverage mask, at >= 0.5. */
export function maskCentroid(m, w, h) {
  let n = 0, sx = 0, sy = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0, q = 0; y < h; y++) {
    for (let x = 0; x < w; x++, q++) {
      if (m[q] < 0.5) continue;
      n++; sx += x; sy += y;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  return n ? { ok: true, area: n, x: sx / n, y: sy / n, box: { x0, y0, x1, y1 } }
           : { ok: false, area: 0, x: w / 2, y: h / 2, box: null };
}

/**
 * One mask, mapped onto the canvas. Bilinear, inverse-mapped, and only over
 * the rectangle the source's own non-zero box lands in — a subject is a small
 * part of a frame and a 1080x1920 canvas is two million pixels, so touching
 * only what can possibly be non-zero is the difference between a preview that
 * scrubs and one that does not.
 */
export function warpMask(src, sw, sh, out, tw, th, plan, box) {
  out.fill(0);
  const { k, x0, y0 } = plan;
  const b = box || { x0: 0, y0: 0, x1: sw - 1, y1: sh - 1 };
  const X0 = clamp(Math.floor(b.x0 * k + x0) - 1, 0, tw - 1);
  const X1 = clamp(Math.ceil((b.x1 + 1) * k + x0) + 1, 0, tw - 1);
  const Y0 = clamp(Math.floor(b.y0 * k + y0) - 1, 0, th - 1);
  const Y1 = clamp(Math.ceil((b.y1 + 1) * k + y0) + 1, 0, th - 1);
  if (X1 < X0 || Y1 < Y0) return out;
  const ik = 1 / k;
  for (let Y = Y0; Y <= Y1; Y++) {
    const sy = (Y + 0.5 - y0) * ik - 0.5;
    const iy = Math.floor(sy), fy = sy - iy;
    const y1i = clamp(iy + 1, 0, sh - 1), y0i = clamp(iy, 0, sh - 1);
    if (iy < -1 || iy > sh - 1) continue;
    const rowA = y0i * sw, rowB = y1i * sw;
    for (let X = X0; X <= X1; X++) {
      const sx = (X + 0.5 - x0) * ik - 0.5;
      const ix = Math.floor(sx), fx = sx - ix;
      if (ix < -1 || ix > sw - 1) continue;
      const x0i = clamp(ix, 0, sw - 1), x1i = clamp(ix + 1, 0, sw - 1);
      const a = src[rowA + x0i] * (1 - fx) + src[rowA + x1i] * fx;
      const c = src[rowB + x0i] * (1 - fx) + src[rowB + x1i] * fx;
      out[Y * tw + X] = a * (1 - fy) + c * fy;
    }
  }
  return out;
}
