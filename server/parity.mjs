import fs from 'node:fs';
const D = (await import('../web/dither.js')).default || (await import('node:module'))
  .createRequire(import.meta.url)('../web/dither.js');
const j = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
D.setBlueNoise(Float32Array.from(j.blue));
const { w, h } = j;
const src = Uint8Array.from(Buffer.from(j.src, 'base64'));
let gate = null;
if (j.gate) { const gb = Buffer.from(j.gate, 'base64');
  gate = new Float32Array(gb.buffer, gb.byteOffset, gb.length / 4); }
const out = {};
for (const c of j.cases) {
  const dst = new Uint8ClampedArray(src);
  D.ditherRGBA(src, dst, w, h, c, gate);
  out[c._name] = Buffer.from(dst.buffer, dst.byteOffset, dst.length).toString('base64');
}

/* The compose gate: whole frames, and the masked still the cutout/overlay PNG
 * export actually goes through — several subjects, a palette each, a flat or
 * kept background, chunky pixels, and the transparent variant. dither.js's
 * composeFrame against render.py's _frame_pixels, byte for byte. */
const masks = (j.masks || []).map((b64) => {
  const mb = Buffer.from(b64, 'base64');
  return new Float32Array(mb.buffer, mb.byteOffset, mb.length / 4);
});
for (const c of (j.compose || [])) {
  const ms = (c._masks || []).map((i) => masks[i]);
  const dst = D.composeFrame(src, w, h, ms, c, c._palettes, c._bg);
  out['C:' + c._name] = Buffer.from(dst.buffer, dst.byteOffset, dst.length)
    .toString('base64');
}
/* The polish gate: web/polish.js against server/polish.py on a synthetic mask
 * sequence — a body drifting a couple of pixels a frame and a ball crossing
 * thirty, so both ends of the motion gate are exercised. Floats, not pixels:
 * these masks feed the renderer, so a difference here is a difference in every
 * dot downstream.
 *
 * The second half checks the browser's own shortcut — polishing the padded
 * bounding box instead of the whole frame — against the whole frame, because
 * that optimisation only exists in the tab and the server would never catch it.
 */
const MP = (await import('node:module')).createRequire(import.meta.url)('../web/polish.js');
if (j.polish) {
  const { w: pw, h: ph, masks: pmasks, strengths } = j.polish;
  const seq = pmasks.map((b64) => {
    const b = Buffer.from(b64, 'base64');
    return new Float32Array(b.buffer, b.byteOffset, b.length / 4);
  });
  const pol = {};
  for (const st of strengths) {
    const done = MP.polishSequence(seq, pw, ph, st);
    done.forEach((m, i) => {
      pol[st + ':' + i] = Buffer.from(m.buffer, m.byteOffset, m.byteLength)
        .toString('base64');
    });
  }
  // whole frame vs padded crop, same frame, same strength
  const st = 100, P = MP.params(st);
  const pad = 2 * P.morph + P.blur + 2;
  let worst = 0;
  for (let i = 0; i < seq.length; i++) {
    const lo = Math.max(0, i - P.radius), hi = Math.min(seq.length - 1, i + P.radius);
    const win = seq.slice(lo, hi + 1);
    const full = MP.polishFrame(win, i - lo, pw, ph, st);
    // the union box of everything non-zero in the window, padded
    let x0 = pw, y0 = ph, x1 = -1, y1 = -1;
    for (const m of win) {
      for (let y = 0; y < ph; y++) {
        for (let x = 0; x < pw; x++) {
          if (!m[y * pw + x]) continue;
          if (x < x0) x0 = x; if (x > x1) x1 = x;
          if (y < y0) y0 = y; if (y > y1) y1 = y;
        }
      }
    }
    if (x1 < 0) continue;
    x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad);
    x1 = Math.min(pw - 1, x1 + pad); y1 = Math.min(ph - 1, y1 + pad);
    const cw = x1 - x0 + 1, ch = y1 - y0 + 1;
    const cwin = win.map((m) => {
      const a = new Float32Array(cw * ch);
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) a[y * cw + x] = m[(y0 + y) * pw + x0 + x];
      }
      return a;
    });
    const st2 = win.map((m) => MP.stats(m, pw, ph));
    const cropped = MP.polishFrame(cwin, i - lo, cw, ch, st, st2);
    for (let q = 0; q < full.length; q++) {
      const y = (q / pw) | 0, x = q % pw;
      const inside = x >= x0 && x <= x1 && y >= y0 && y <= y1;
      const v = inside ? cropped[(y - y0) * cw + (x - x0)] : 0;
      const d = Math.abs(full[q] - v);
      if (d > worst) worst = d;
    }
  }
  out._polish = pol;
  out._polishCropMaxDiff = worst;
  out._polishParams = strengths.map((v) => Object.assign({ strength: v },
                                                         MP.params(v)));
}

fs.writeFileSync(process.argv[3], JSON.stringify(out));
