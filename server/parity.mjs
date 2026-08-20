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
fs.writeFileSync(process.argv[3], JSON.stringify(out));
