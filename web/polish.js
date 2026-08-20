/* ---------------------------------------------------------------------------
   MASK POLISH — the same soft mask, steadier.

   EdgeTAM's per-frame masks are good outlines and bad neighbours: the edge
   wobbles by a pixel or two between frames, small holes open and close, and a
   subject that is standing still still shimmers. Averaging a few frames of the
   soft mask together fixes all of that at once... and destroys anything that
   moves fast, because averaging a ball across five frames IS a streak. The
   naive version of this smeared the free kick into a comet.

   So the window is not fixed. Per frame, per subject, this measures how far the
   mask's centroid moved relative to how big the mask is -- displacement over
   sqrt(area) -- and closes the temporal window as that ratio grows. A player
   drifting two pixels a frame against a 140 px body gets the whole window; a
   ball crossing thirty pixels a frame at 36 px across gets none of it, frame by
   frame, with no threshold to tune and nothing to label by hand.

   Three stages, all scaled by one 0-100 strength:

     TEMPORAL   weighted mean of the soft masks in the window. Weight is
                triangular in distance and multiplied by a motion gate that
                falls to zero once the cumulative centroid walk reaches
                ALPHA x sqrt(area). Fast things keep their own frame.
     MORPH      grayscale close then open, square structuring element. Closes
                the pinholes the tracker leaves inside a body, then removes the
                specks it leaves outside one.
     BLUR       one or two passes of a separable [1 2 1]/4. Softens the stair
                stepping the 192x192 mask upsample leaves behind.

   PARITY. server/polish.py is a numpy transcription of this file, operation for
   operation and rounding for rounding, and server/parity.py gates the two
   against each other on a synthetic sequence. Keeping them identical is what
   lets the preview in the tab and the render on the server show the same
   subject: the browser polishes mask bitmaps, the server polishes mask PNGs,
   and the arithmetic in between is the same arithmetic.

   LOADING. Like web/dither.js this is a classic script: it publishes itself on
   globalThis.MaskPolish and, under Node, on module.exports.
--------------------------------------------------------------------------- */
'use strict';

const MaskPolish = (() => {

/* How far the centroid may walk, as a fraction of the subject's own size,
 * before a neighbouring frame counts for nothing. 0.35 of sqrt(area) is about
 * a third of a body width — generous for a walking figure, hopeless for a
 * struck ball, which is the entire point. */
const ALPHA = 0.35;

const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
/* floor(v + 0.5): JS Math.round's rule, written out so Python can copy it
 * rather than inherit numpy's round-half-to-even. */
const round0 = (v) => Math.floor(v + 0.5);

/**
 * What a strength means. Everything the algorithm does is one of these three
 * integers, so "strength 70" is a reproducible thing rather than a feeling.
 *   radius  frames each side considered   0..3
 *   morph   close/open radius in pixels   0..2
 *   blur    [1 2 1]/4 passes              0..2
 */
function params(strength) {
  const u = clamp((+strength || 0) / 100, 0, 1);
  return { u, radius: round0(u * 3), morph: round0(u * 2), blur: round0(u * 2),
           alpha: ALPHA };
}

/** area (soft >= 0.5) and its centroid, in pixels. */
function stats(m, w, h) {
  let n = 0, sx = 0, sy = 0;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      if (m[row + x] >= 0.5) { n++; sx += x; sy += y; }
    }
  }
  return n ? { area: n, cx: sx / n, cy: sy / n } : { area: 0, cx: 0, cy: 0 };
}

/**
 * The per-frame weights the temporal stage would use.
 *   win     stats() for each frame of the window, in frame order
 *   c       index of the frame being polished
 *   radius  how many frames each side to consider
 * Returns a Float64Array the same length as `win`. Exposed because it is what
 * the motion-awareness claim is actually about, and a verifier should be able
 * to read it rather than infer it from pixels.
 */
function weights(win, c, radius) {
  const w = new Float64Array(win.length);
  w[c] = 1;
  const size = Math.sqrt(win[c].area);
  if (radius <= 0 || size < 1) return w;
  const tol = ALPHA * size;
  for (let dir = -1; dir <= 1; dir += 2) {
    let disp = 0;
    for (let d = 1; d <= radius; d++) {
      const j = c + dir * d, prev = j - dir;
      if (j < 0 || j >= win.length) break;
      if (win[j].area <= 0 || win[prev].area <= 0) break;
      const dx = win[j].cx - win[prev].cx, dy = win[j].cy - win[prev].cy;
      disp += Math.sqrt(dx * dx + dy * dy);
      const tri = (radius + 1 - d) / (radius + 1);
      const gate = clamp(1 - disp / tol, 0, 1);
      const wj = tri * gate;
      if (wj <= 0) break;           // and everything past it is further still
      w[j] = wj;
    }
  }
  return w;
}

/* ------------------------------------------------------- spatial helpers
 * Separable min/max over a (2r+1)-square, edges clamped. Written the naive way
 * on purpose: r is at most 2, and a running-extremum version would be a second
 * implementation for Python to have to match exactly. */
function _extreme(src, w, h, r, max) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = src[row + x];
      for (let k = -r; k <= r; k++) {
        const xx = clamp(x + k, 0, w - 1), s = src[row + xx];
        if (max ? s > v : s < v) v = s;
      }
      tmp[row + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = tmp[y * w + x];
      for (let k = -r; k <= r; k++) {
        const yy = clamp(y + k, 0, h - 1), s = tmp[yy * w + x];
        if (max ? s > v : s < v) v = s;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}
const dilate = (a, w, h, r) => (r > 0 ? _extreme(a, w, h, r, true) : a);
const erode = (a, w, h, r) => (r > 0 ? _extreme(a, w, h, r, false) : a);
const close = (a, w, h, r) => erode(dilate(a, w, h, r), w, h, r);
const open = (a, w, h, r) => dilate(erode(a, w, h, r), w, h, r);

/** One pass of a separable [1 2 1]/4, edges clamped. */
function blur121(src, w, h) {
  const tmp = new Float32Array(w * h), out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const a = src[row + clamp(x - 1, 0, w - 1)];
      const b = src[row + x];
      const c = src[row + clamp(x + 1, 0, w - 1)];
      tmp[row + x] = (a + 2 * b + c) / 4;
    }
  }
  for (let y = 0; y < h; y++) {
    const up = clamp(y - 1, 0, h - 1) * w, dn = clamp(y + 1, 0, h - 1) * w;
    for (let x = 0; x < w; x++) {
      out[y * w + x] = (tmp[up + x] + 2 * tmp[y * w + x] + tmp[dn + x]) / 4;
    }
  }
  return out;
}

/**
 * Polish one frame's mask.
 *   win       contiguous soft masks (Float32Array, w*h, 0..1) in frame order
 *   c         index within `win` of the frame being polished
 *   strength  0..100; 0 hands the frame straight back
 *   st        optional pre-computed stats() per window frame
 * Returns a new Float32Array.
 */
function polishFrame(win, c, w, h, strength, st) {
  const p = params(strength);
  const mid = win[c];
  if (p.u <= 0) return Float32Array.from(mid);
  let cur;
  if (p.radius > 0 && win.length > 1) {
    const S = st || win.map((m) => stats(m, w, h));
    const wt = weights(S, c, p.radius);
    let sum = 0;
    for (let j = 0; j < wt.length; j++) sum += wt[j];
    if (sum > 0 && wt.some((v, j) => j !== c && v > 0)) {
      const acc = new Float32Array(w * h);
      for (let j = 0; j < wt.length; j++) {
        const wj = wt[j];
        if (wj <= 0) continue;
        const m = win[j];
        for (let q = 0; q < acc.length; q++) acc[q] += wj * m[q];
      }
      for (let q = 0; q < acc.length; q++) acc[q] = acc[q] / sum;
      cur = acc;
    } else {
      cur = Float32Array.from(mid);
    }
  } else {
    cur = Float32Array.from(mid);
  }
  if (p.morph > 0) cur = open(close(cur, w, h, p.morph), w, h, p.morph);
  for (let i = 0; i < p.blur; i++) cur = blur121(cur, w, h);
  return cur;
}

/**
 * Polish a whole sequence. The browser does not use this — it polishes the one
 * frame it is drawing — but the parity gate and any batch job do.
 */
function polishSequence(masks, w, h, strength) {
  const p = params(strength);
  if (p.u <= 0) return masks.map((m) => Float32Array.from(m));
  const st = masks.map((m) => stats(m, w, h));
  const out = [];
  for (let i = 0; i < masks.length; i++) {
    const lo = Math.max(0, i - p.radius), hi = Math.min(masks.length - 1, i + p.radius);
    out.push(polishFrame(masks.slice(lo, hi + 1), i - lo, w, h, strength,
                         st.slice(lo, hi + 1)));
  }
  return out;
}

/** 0..1 floats -> the 0..255 the server's polished PNG holds, so the two
 *  engines quantise at the same place as well as computing the same numbers. */
function quantise(m) {
  const out = new Uint8Array(m.length);
  for (let q = 0; q < m.length; q++) {
    out[q] = clamp(round0(m[q] * 255), 0, 255);
  }
  return out;
}

return { params, stats, weights, polishFrame, polishSequence, quantise,
         dilate, erode, close, open, blur121, ALPHA, version: 1 };
})();

globalThis.MaskPolish = MaskPolish;
if (typeof module !== 'undefined') module.exports = MaskPolish;
