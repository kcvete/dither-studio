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
fs.writeFileSync(process.argv[3], JSON.stringify(out));
