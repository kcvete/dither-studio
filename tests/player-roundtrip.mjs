/* ---------------------------------------------------------------------------
   PLAYER ROUND TRIP — the .dots.gz codec, out and back, byte for byte.

   A .dots.gz is not a video: it is the dot positions themselves, and the whole
   promise of the format is that what comes back out is what went in. Every dot
   the renderer lit, at the integer pixel it lit, frame by frame and subject by
   subject. If a varint, a zigzag sign or a delta base ever drifts, the file
   still decodes — it just decodes into a slightly different picture, silently,
   forever. That is the failure this test exists to catch.

   It runs the doc through both paths the player offers:

       encode -> gzip -> gunzip -> decode        (the pieces)
       pack   ->               -> unpack        (the two-call convenience)

   and asserts the two agree with each other and with the doc that went in.

   No canvas, no DOM: paintFrame is the one export that needs a real
   ImageData and it is deliberately not called here. Everything below is pure
   arithmetic, so it runs anywhere node does, in well under a second.
--------------------------------------------------------------------------- */
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const P = require(path.join(ROOT, 'web/player/dither-player.js'));

const { encode, decode, gzip, gunzip, pack, unpack, toJSON, fromJSON, mulberry32 } = P;

let checks = 0;
const problems = [];
function ok(cond, what) {
  checks++;
  if (!cond) problems.push(what);
}
function eq(got, want, what) {
  checks++;
  if (got !== want) problems.push(`${what}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

/* ------------------------------------------------------------ fixtures --- */

/* A dot cloud in the order the renderer emits it: cell-scan, top row first.
 * Deterministic, so a failure is reproducible from the seed alone. */
function cloud(seed, n, w, h) {
  const rnd = mulberry32(seed);
  const pts = [];
  for (let i = 0; i < n; i++) {
    pts.push([Math.floor(rnd() * w), Math.floor(rnd() * h)]);
  }
  pts.sort((a, b) => (a[1] - b[1]) || (a[0] - b[0]));
  const xy = new Uint16Array(n * 2);
  for (let i = 0; i < n; i++) { xy[i * 2] = pts[i][0]; xy[i * 2 + 1] = pts[i][1]; }
  return xy;
}

function sameFrames(a, b, label) {
  eq(a.length, b.length, `${label}: frame count`);
  const n = Math.min(a.length, b.length);
  for (let f = 0; f < n; f++) {
    eq(a[f].length, b[f].length, `${label}: frame ${f} subject count`);
    const k = Math.min(a[f].length, b[f].length);
    for (let s = 0; s < k; s++) {
      const x = a[f][s], y = b[f][s];
      if (x.length !== y.length) { checks++; problems.push(`${label}: frame ${f} subject ${s} dot count: got ${y.length >> 1}, want ${x.length >> 1}`); continue; }
      for (let i = 0; i < x.length; i++) {
        if (x[i] !== y[i]) {
          checks++;
          problems.push(`${label}: frame ${f} subject ${s} coord ${i}: got ${y[i]}, want ${x[i]}`);
          break;                                   // one report per subject is plenty
        }
      }
      checks++;
    }
  }
}

/* ============================================================ case one ===
 * Twelve frames, three subjects, counts that move around, one subject that
 * blinks out entirely mid-clip. The ordinary case. */
const W = 320, H = 240, NF = 12;
const palette = ['#101014', '#ff3366', '#33ddaa', '#ffcc00'];
const subjects = [{ color: '#ff3366' }, { color: '#33ddaa' }, { color: '#ffcc00' }];
const frames = [];
for (let f = 0; f < NF; f++) {
  frames.push([
    cloud(1000 + f, 400 + f * 17, W, H),
    cloud(2000 + f, f === 5 ? 0 : 250 - f * 3, W, H),   // blinks out on frame 5
    cloud(3000 + f, 60, W, H),
  ]);
}
const doc = { w: W, h: H, fps: 24, dotpx: 3, palette, bgIndex: 0, subjects, frames };

const bytes = encode(doc);
eq(bytes[0], 0x44, 'magic byte 0');                     // 'D'
eq(bytes[1], 0x4f, 'magic byte 1');                     // 'O'
eq(bytes[2], 0x54, 'magic byte 2');                     // 'T'
eq(bytes[3], 0x53, 'magic byte 3');                     // 'S'
eq(bytes[4], 1, 'format version byte');
eq(P.version, 1, 'exported version');

const gz = await gzip(bytes);
eq(gz[0], 0x1f, 'gzip magic 0');
eq(gz[1], 0x8b, 'gzip magic 1');
ok(gz.length < bytes.length, `gzip shrinks the body (${bytes.length} -> ${gz.length})`);

const back = decode(await gunzip(gz));
eq(back.w, W, 'width');
eq(back.h, H, 'height');
eq(back.fps, 24, 'fps');
eq(back.dotpx, 3, 'dotpx');
eq(back.bgIndex, 0, 'bgIndex');
eq(back.bg, palette[0], 'bg colour');
eq(back.palette.join(','), palette.join(','), 'palette');
eq(back.subjects.map((s) => s.color).join(','), subjects.map((s) => s.color).join(','), 'subject colours');
sameFrames(frames, back.frames, 'encode/gzip/gunzip/decode');

/* the two-call path has to land on exactly the same doc */
const viaPack = await unpack(await pack(doc));
sameFrames(frames, viaPack.frames, 'pack/unpack');
eq(viaPack.palette.join(','), back.palette.join(','), 'pack/unpack palette matches piecewise path');

/* pack is gzip(encode(...)) — same input, same bytes */
eq(Buffer.from(await pack(doc)).toString('base64'), Buffer.from(gz).toString('base64'),
   'pack() equals gzip(encode())');

/* the JSON debug variant carries the same numbers */
const viaJSON = fromJSON(JSON.parse(JSON.stringify(toJSON(doc))));
sameFrames(frames, viaJSON.frames, 'toJSON/fromJSON');
eq(viaJSON.palette.join(','), palette.join(','), 'toJSON/fromJSON palette');

/* ============================================================ case two ===
 * Empty everything, and the coordinate extremes. A frame with no dots at all,
 * a subject the caller left off the frame array entirely, dots at 0,0 and at
 * the uint16 ceiling, and a cloud walked backwards so every delta is negative
 * and the zigzag sign bit actually gets exercised. */
const edgeSubjects = [{ color: '#ffffff' }, { color: '#000000' }];
const edgeFrames = [
  [new Uint16Array(0), new Uint16Array(0)],                       // nothing lit
  [new Uint16Array([0, 0, 65535, 65535, 0, 65535, 65535, 0])],    // extremes; subject 1 omitted
  [new Uint16Array([900, 900, 500, 500, 100, 100, 0, 0]),         // strictly decreasing
   new Uint16Array([7, 7])],
];
const edgeDoc = {
  w: 1024, h: 1024, fps: 1, dotpx: 1,
  palette: ['#ffffff', '#000000'], bgIndex: 1,
  subjects: edgeSubjects, frames: edgeFrames,
};
const edgeBack = await unpack(await pack(edgeDoc));
eq(edgeBack.frames.length, 3, 'edge: frame count');
eq(edgeBack.frames[0][0].length, 0, 'edge: empty frame subject 0 is empty');
eq(edgeBack.frames[0][1].length, 0, 'edge: empty frame subject 1 is empty');
eq(edgeBack.frames[1][1].length, 0, 'edge: omitted subject decodes as no dots');
eq(edgeBack.frames[1][0].join(','), '0,0,65535,65535,0,65535,65535,0', 'edge: uint16 extremes survive');
eq(edgeBack.frames[2][0].join(','), '900,900,500,500,100,100,0,0', 'edge: negative deltas survive');
eq(edgeBack.bgIndex, 1, 'edge: bgIndex');
eq(edgeBack.fps, 1, 'edge: fps');
eq(edgeBack.dotpx, 1, 'edge: dotpx');

/* ========================================================== case three ===
 * The documented ceiling. n_frames is a uint16, so 65,535 frames is the last
 * legal clip and 65,536 has to be refused loudly rather than wrap to zero. */
const LIMIT = 65535;
const blank = new Uint16Array(0);
const longFrames = new Array(LIMIT);
for (let i = 0; i < LIMIT; i++) longFrames[i] = [blank];
const longDoc = { w: 16, h: 16, fps: 30, dotpx: 1, palette: ['#000000'], bgIndex: 0,
                  subjects: [{ color: '#000000' }], frames: longFrames };
const longBack = await unpack(await pack(longDoc));
eq(longBack.frames.length, LIMIT, 'limit: 65,535 frames round-trip');
eq(longBack.frames[LIMIT - 1][0].length, 0, 'limit: last frame decodes');

let threw = null;
try { encode({ ...longDoc, frames: longFrames.concat([[blank]]) }); }
catch (e) { threw = e; }
ok(threw !== null, 'limit: 65,536 frames must throw');
ok(threw && /65,?535/.test(threw.message), `limit: the error names the ceiling (got ${threw && threw.message})`);

/* ---------------------------------------------------------------- verdict */
if (problems.length) {
  console.error('player round trip FAILED:');
  for (const p of problems.slice(0, 25)) console.error('  - ' + p);
  if (problems.length > 25) console.error(`  ... and ${problems.length - 25} more`);
  process.exit(1);
}
const dots = frames.reduce((a, f) => a + f.reduce((b, s) => b + (s.length >> 1), 0), 0);
console.log(`player round trip ok — ${checks} assertions; ${NF} frames / ${subjects.length} subjects / ${dots} dots exact through encode+gzip+gunzip+decode and pack/unpack; empty, uint16-extreme, negative-delta and 65,535-frame cases clean`);
