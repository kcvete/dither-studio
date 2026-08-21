/* End-to-end check against a real running server, a real EdgeTAM run and real
 * ffmpeg. No mocks.
 *
 *   node verify.mjs [baseURL] [clip.mp4] [still.jpg]
 *
 * Covers the flows the app offers:
 *   A  still   -> every algorithm -> client-side PNG download
 *   A2 still   -> whole-image dots -> PNG + a one-frame .dots.gz
 *   A3 still   -> a clicked subject, segmented on the SERVER in one frame
 *                 (/api/upload_image + /preview, no propagation) -> cutout,
 *                 overlay, per-subject palette -> PNG with alpha
 *   B  clip    -> whole-frame dither -> MP4
 *   C  clip    -> two tracked subjects -> dots + a pixel mode -> MP4
 *   D  clip    -> one tracked subject at "fast" tracking quality (512 px)
 *   E  clip    -> frame-0 preview, then a polygon mask prompt -> tracked
 *   O  clip    -> a dithered render AND the original cut to the same frames,
 *                 compared frame for frame against jobs/<id>/frames/
 *   R  clip    -> tracked, THEN trimmed: a narrower range renders, exports,
 *                 cuts its original and writes its .dots.gz out of the frames
 *                 and masks already on disk, with no second track; a wider one
 *                 is offered rather than silently re-cut
 *   X  clip    -> the CANVAS: 9:16 out of a 16:9 source, as a cutout (dots
 *                 re-measured at 1080x1920) and as an auto-reframed overlay
 *                 whose crop follows the tracked subject, with the matched
 *                 original cut on the identical path; the .dots.gz and the
 *                 sequence at the same shape; a still at 1:1
 *   P  clip    -> a tracked subject with MASK POLISH on: the motion gate, the
 *                 tab's polished mask against the server's byte for byte, the
 *                 wipe, and preview-vs-export
 *   S  clip    -> PER-SUBJECT INCREMENTAL TRACKING: track subject #1 alone,
 *                 add #2 and track only the new one (with #1's mask PNGs
 *                 checked byte for byte across the run), edit #2's prompt so
 *                 it goes stale, re-track #2 alone from its chip menu, then
 *                 remove #1 and render what is left. Every POST /track body
 *                 is logged and asserted on: `only` must name exactly the
 *                 subjects the UI said it would walk
 * Writes screenshots to docs/ and a JSON report to docs/verify-report.json.
 */
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.argv[2] || 'http://127.0.0.1:8765';
const CLIP = process.argv[3] || path.join(HERE, 'sample.mp4');
const STILL = process.argv[4] || CLIP.replace(/\.\w+$/, '.jpg');
const DOCS = path.join(HERE, 'docs');
fs.mkdirSync(DOCS, { recursive: true });
/* the after-evidence directory the UX screenshots live in */
const AFTER = path.join(DOCS, 'ux-after');
fs.mkdirSync(AFTER, { recursive: true });

const SUBJECT_A = { box: [435, 95, 625, 360], point: [545, 205] };   // parkour athlete
const SUBJECT_B = { box: [1005, 5, 1279, 470], point: [1150, 160] }; // tree, right edge

const R = { base: BASE, clip: CLIP, still: STILL, consoleErrors: [], pageErrors: [],
            requestFailures: [], runs: {} };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function ffprobe(f) {
  const out = execFileSync('ffprobe', ['-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate', '-of', 'json', f]);
  return JSON.parse(out).streams[0];
}

/* codec + pixel format + the alpha_mode tag WebM carries an alpha plane under */
function ffprobeFull(f) {
  const out = execFileSync('ffprobe', ['-v', 'error', '-count_frames',
    '-select_streams', 'v:0', '-show_entries',
    'stream=codec_name,pix_fmt,width,height,nb_read_frames,r_frame_rate:stream_tags=alpha_mode',
    '-of', 'json', f]);
  return JSON.parse(out).streams[0];
}

/* One decoded frame as RGBA, straight out of ffmpeg. */
function decodeFrameRGBA(file, n, w, h, extraIn) {
  const args = ['-v', 'error'].concat(extraIn || [],
    ['-i', file, '-vf', `select=eq(n\\,${n})`, '-vframes', '1',
     '-pix_fmt', 'rgba', '-f', 'rawvideo', '-']);
  const buf = execFileSync('ffmpeg', args, { maxBuffer: 1 << 28 });
  return { data: buf, w, h };
}

/* what an exported PNG's alpha channel actually says */
function pngAlphaCensus(file) {
  const data = execFileSync('ffmpeg', ['-v', 'error', '-i', file, '-pix_fmt', 'rgba',
    '-f', 'rawvideo', '-'], { maxBuffer: 1 << 28 });
  let zero = 0, full = 0;
  for (let p = 3; p < data.length; p += 4) {
    if (data[p] < 16) zero++; else if (data[p] > 200) full++;
  }
  const px = data.length / 4;
  return { pixels: px, transparentPct: +(100 * zero / px).toFixed(1),
           opaquePct: +(100 * full / px).toFixed(1) };
}

/* colour census of the live preview canvas */
const census = (page) => page.evaluate(() => {
  const c = document.querySelector('#vcv');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const m = new Map();
  for (let i = 0; i < d.length; i += 4) {
    const k = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
    m.set(k, (m.get(k) || 0) + 1);
  }
  const top = [...m.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([k, n]) => ['#' + k.toString(16).padStart(6, '0'), n]);
  return { w: c.width, h: c.height, distinctColours: m.size, top };
});

async function waitText(page, sel, re, timeout) {
  const t0 = Date.now();
  for (;;) {
    const t = await page.textContent(sel).catch(() => '');
    if (t && re.test(t)) return t;
    if (Date.now() - t0 > timeout) throw new Error(`timeout on ${sel} (last: ${t})`);
    await sleep(500);
  }
}

async function stageXY(page, sel, x, y) {
  return page.evaluate(([s, fx, fy]) => {
    const el = document.querySelector(s), r = el.getBoundingClientRect();
    return [r.left + (fx / el.width) * r.width, r.top + (fy / el.height) * r.height];
  }, [sel, x, y]);
}

async function prompt(page, s) {
  const [x0, y0] = await stageXY(page, '#pov', s.box[0], s.box[1]);
  const [x1, y1] = await stageXY(page, '#pov', s.box[2], s.box[3]);
  await page.mouse.move(x0, y0); await page.mouse.down();
  await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 6 });
  await page.mouse.move(x1, y1, { steps: 6 }); await page.mouse.up();
  const [px, py] = await stageXY(page, '#pov', s.point[0], s.point[1]);
  await page.mouse.click(px, py);
}

const setMode = async (page, m) => {
  await openStep(page, 'st3');
  await page.click(`#modes .chip[data-mode="${m}"]`);
  await sleep(700);
};
/* the panel headers toggle, so only click when the section is actually closed */
const openStep = async (page, id) => {
  const open = await page.getAttribute(`#${id}`, 'data-open');
  if (open !== '1') { await page.click(`#${id} .sh`); await sleep(200); }
};

/* ------------------------------------------------------------ A: a still */
async function runStill(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.screenshot({ path: path.join(DOCS, 'a-landing.png') });

  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 20000 });
  await sleep(800);
  r.source = await page.textContent('#upstat');

  r.modes = {};
  for (const m of ['bluenoise', 'ordered', 'halftone', 'whitenoise', 'errordiff', 'riemersma']) {
    await setMode(page, m);
    const c = await census(page);
    r.modes[m] = { colours: c.distinctColours, w: c.w, h: c.h };
  }
  // every error-diffusion kernel must produce a distinct, 2-colour image
  await setMode(page, 'errordiff');
  const kernels = await page.$$eval('#sAlgo option', (o) => o.map((x) => x.value));
  r.kernels = {};
  for (const k of kernels) {
    await page.selectOption('#sAlgo', k);
    await sleep(320);
    const sig = await page.evaluate(() => {
      const c = document.querySelector('#vcv');
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      let on = 0, h = 0;
      for (let i = 0; i < d.length; i += 4) { if (d[i] > 127) { on++; h = (h * 31 + i) >>> 0; } }
      return { on, h };
    });
    r.kernels[k] = sig;
  }
  r.kernelsDistinct = new Set(Object.values(r.kernels).map((v) => v.h)).size;
  r.kernelCount = kernels.length;

  await page.selectOption('#sAlgo', 'atkinson');
  await sleep(300);
  await openStep(page, 'st4');                      // palette
  await page.click('#pals .chip:nth-child(6)');     // Game Boy DMG
  await sleep(700);
  r.gameboy = await census(page);
  await page.click('#bFromImg'); await sleep(700);
  r.fromImage = await census(page);
  await page.click('#pals .chip:nth-child(1)'); await sleep(600);   // back to B&W

  // pixel scale
  await openStep(page, 'st3');   // look
  await page.locator('#sPx').fill('4');
  await page.locator('#sPx').dispatchEvent('input');
  await sleep(700);
  r.pixel4 = await census(page);
  await page.locator('#sPx').fill('1');
  await page.locator('#sPx').dispatchEvent('input');
  await sleep(600);

  await page.click('#bCmp'); await sleep(500);
  await page.screenshot({ path: path.join(DOCS, 'a-still.png') });
  await page.click('#bCmp'); await sleep(300);

  await openStep(page, 'st5');
  await page.click('#bExport');
  r.export = await waitText(page, '#rinfo', /PNG|failed/, 60000);
  if (/failed/.test(r.export)) throw new Error(r.export);
  const dl = await page.waitForEvent('download', { timeout: 30000 }).catch(() => null);
  const clicked = page.click('#dl');
  const d = await Promise.race([page.waitForEvent('download', { timeout: 30000 }), clicked.then(() => null)]);
  if (d) {
    const p = path.join(DOCS, 'a-exported.png');
    await d.saveAs(p);
    r.exportedBytes = fs.statSync(p).size;
    r.exportedProbe = execFileSync('ffprobe', ['-v', 'error', '-show_entries',
      'stream=width,height', '-of', 'csv=p=0', p]).toString().trim();
  }
  await page.screenshot({ path: path.join(DOCS, 'a-export.png') });
  return r;
}

/* ------------------------------------ A2: a still, whole-image dots =====
 * The look this tool started as, applied to a photograph with nothing
 * selected: one mask covering everything, density from luminance. */
async function runStillDots(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 20000 });
  await sleep(700);

  const off = await page.getAttribute('#modes .chip[data-mode="dots"]', 'class');
  if (/\boff\b/.test(off || '')) throw new Error('dots is still gated off for a still');
  await setMode(page, 'dots');
  await sleep(600);
  r.fps = await page.textContent('#fps');
  r.dots = +(/· (\d+) dots/.exec(r.fps) || [])[1];
  if (!(r.dots > 200)) throw new Error('whole-image dots produced ' + r.fps);
  r.census = await census(page);
  if (r.census.distinctColours !== 2) {
    throw new Error('whole-image dots is not two colours: ' + JSON.stringify(r.census));
  }
  // the flat-background picker and the dot-data export both belong to a still
  r.bgUiShown = await page.getAttribute('#bgui', 'hidden') === null;
  await openStep(page, 'st5');
  r.dotsExportShown = await page.getAttribute('#dotsexp', 'hidden') === null;
  if (!r.dotsExportShown) throw new Error('no .dots.gz export offered for a dotted still');
  await page.screenshot({ path: path.join(DOCS, 'a2-still-dots.png') });

  // the dot-data buttons live under the "for developers" disclosure now
  await page.evaluate(() => { const d = document.querySelector('#devexp'); if (d) d.open = true; });
  const [d] = await Promise.all([
    page.waitForEvent('download', { timeout: 60000 }).catch(() => null),
    page.click('#bDots'),
  ]);
  if (!d) throw new Error('no .dots.gz download');
  const gz = path.join(DOCS, 'a2-still.dots.gz');
  await d.saveAs(gz);
  const bytes = fs.readFileSync(gz);
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) throw new Error('.dots.gz is not gzip');
  r.dotsBytes = bytes.length;
  r.dotsInfo = await page.textContent('#rinfo');
  // decode it with the player's own codec and check it is one frame of dots
  const P = await import(path.join(HERE, 'web', 'player', 'dither-player.mjs'));
  const doc = await P.unpack(new Uint8Array(bytes));
  r.doc = { w: doc.w, h: doc.h, frames: doc.frames.length,
            dots: doc.frames[0].reduce((a, x) => a + (x.length >> 1), 0),
            palette: doc.palette, bg: doc.bg };
  if (doc.frames.length !== 1) throw new Error('a still is not one frame: ' + doc.frames.length);
  if (Math.abs(r.doc.dots - r.dots) > 0) {
    throw new Error(`the file has ${r.doc.dots} dots, the preview showed ${r.dots}`);
  }

  await page.click('#bExport');
  r.export = await waitText(page, '#rinfo', /PNG|failed/, 60000);
  if (/failed/.test(r.export)) throw new Error(r.export);
  return r;
}

/* --------------------- A3: a still, a subject, cut out on the server =====
 * /api/upload_image once, then one /preview per click. No propagation, no
 * Track button: the mask is re-cut live and the picture follows it. */
async function runStillSubject(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 20000 });
  await sleep(700);
  await openStep(page, 'st2');
  r.scopeLabels = await page.$$eval('#scope .chip', (n) => n.map((e) => e.textContent));
  await page.click('#scope .chip[data-scope="track"]');
  await sleep(600);
  r.promptFrameHidden = await page.getAttribute('#pfui', 'hidden') !== null;
  r.trackButton = await page.textContent('#bTrack');

  const t0 = Date.now();
  await prompt(page, SUBJECT_A);                      // box, then a click
  r.info = await waitText(page, '#pvinfo', /subject|failed/, 60000);
  if (/failed/.test(r.info)) throw new Error(r.info);
  r.liveSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  r.segmentSeconds = +(/in ([\d.]+) s/.exec(r.info) || [])[1];
  r.areas = await page.evaluate(() => window.DV_still.areas());
  const area = r.areas[Object.keys(r.areas)[0]];
  if (!(area > 5000 && area < 60000)) {
    throw new Error('the subject mask is ' + area + ' px, which is not a person');
  }
  await page.screenshot({ path: path.join(DOCS, 'a3-still-prompt.png') });

  await page.click('#bTrack');                        // "use this selection"
  await sleep(800);
  r.targets = await page.$$eval('#target .chip', (n) => n.map((e) => e.textContent));
  if (r.targets.length !== 2) throw new Error('no per-subject palette: ' + r.targets);

  // cutout, a pixel mode
  await setMode(page, 'bluenoise');
  await sleep(500);
  r.cutout = await census(page);
  await page.screenshot({ path: path.join(DOCS, 'a3-still-cutout.png') });
  // overlay: the photograph is kept and the subject is dithered into it
  await openStep(page, 'st3');
  await page.click('#compose .chip[data-compose="overlay"]');
  await sleep(600);
  r.overlay = await census(page);
  if (r.overlay.distinctColours <= r.cutout.distinctColours) {
    throw new Error('overlay is not keeping the scene: '
      + JSON.stringify([r.cutout, r.overlay]));
  }
  await page.screenshot({ path: path.join(DOCS, 'a3-still-overlay.png') });
  await page.click('#compose .chip[data-compose="cutout"]');
  await sleep(500);
  // dots on the subject alone
  await setMode(page, 'dots');
  await sleep(700);
  r.dotsFps = await page.textContent('#fps');
  r.subjectDots = +(/· (\d+) dots/.exec(r.dotsFps) || [])[1];
  if (!(r.subjectDots > 50)) throw new Error('no dots on the subject: ' + r.dotsFps);

  // the transparent PNG
  await openStep(page, 'st5');
  r.alphaUiShown = await page.getAttribute('#pngalpha', 'hidden') === null;
  if (!r.alphaUiShown) throw new Error('no transparent-background switch offered');
  await page.check('#cAlpha');
  await page.click('#bExport');
  r.export = await waitText(page, '#rinfo', /PNG|failed/, 60000);
  if (/failed/.test(r.export)) throw new Error(r.export);
  const [d] = await Promise.all([
    page.waitForEvent('download', { timeout: 30000 }).catch(() => null),
    page.click('#dl'),
  ]);
  if (!d) throw new Error('no PNG download');
  const out = path.join(DOCS, 'a3-still-cutout-alpha.png');
  await d.saveAs(out);
  r.pngBytes = fs.statSync(out).size;
  r.pngProbe = execFileSync('ffprobe', ['-v', 'error', '-show_entries',
    'stream=width,height,pix_fmt', '-of', 'csv=p=0', out]).toString().trim();
  r.alpha = pngAlphaCensus(out);
  if (!/rgba/.test(r.pngProbe)) throw new Error('the PNG has no alpha: ' + r.pngProbe);
  if (r.alpha.transparentPct < 80) {
    throw new Error('the cutout PNG is not transparent: ' + JSON.stringify(r.alpha));
  }
  if (r.alpha.opaquePct < 0.1) {
    throw new Error('the cutout PNG has nothing in it: ' + JSON.stringify(r.alpha));
  }
  await page.screenshot({ path: path.join(DOCS, 'a3-still-export.png') });
  return r;
}

/* ------------------------------------------------- B: a clip, whole frame */
async function runWhole(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.setInputFiles('#file', CLIP);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
  r.source = await page.textContent('#upstat');
  r.job = await page.evaluate(() => window.DV.job);
  await sleep(900);

  await setMode(page, 'ordered');
  await page.click('#mxui .chip[data-mx="8"]'); await sleep(600);
  await openStep(page, 'st4');
  await page.click('#pals .chip:nth-child(6)');   // Game Boy DMG
  await sleep(800);
  r.preview = await census(page);

  await page.click('#bPlay'); await sleep(2200); await page.click('#bPlay');
  r.previewFps = await page.textContent('#fps');
  await page.evaluate(() => window.DV_draw(20)); await sleep(500);
  r.previewFrame20 = await census(page);
  await page.screenshot({ path: path.join(DOCS, 'b-whole.png') });

  await openStep(page, 'st5');
  await page.click('#bExport');
  r.render = await waitText(page, '#rinfo', /rendered|failed/, 300000);
  if (/failed/.test(r.render)) throw new Error(r.render);
  await sleep(700);
  r.ffprobe = ffprobe(path.join(HERE, 'jobs', r.job, 'out.mp4'));
  await page.screenshot({ path: path.join(DOCS, 'b-export.png') });
  return r;
}

/* -------------------------------------------- C: a clip, tracked subjects */
async function runTracked(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.setInputFiles('#file', CLIP);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
  r.job = await page.evaluate(() => window.DV.job);

  await page.click('#scope .chip[data-scope="track"]');
  await sleep(700);
  await prompt(page, SUBJECT_A);
  await page.click('#bAdd');
  await prompt(page, SUBJECT_B);
  r.subjectChips = await page.$$eval('#subs .chip', (n) => n.map((e) => e.textContent));

  // tracking quality: the chips must exist, carry measured fps, and default to best
  r.qualityChips = await page.$$eval('#tq .chip', (n) => n.map((e) => ({
    size: +e.dataset.size, text: e.textContent.trim(),
    on: e.getAttribute('aria-pressed') === 'true' })));
  r.qualityDefault = (r.qualityChips.find((c) => c.on) || {}).size;
  await page.screenshot({ path: path.join(DOCS, 'c-prompts.png') });

  const t0 = Date.now();
  await page.click('#bTrack');
  r.trackInfo = await waitText(page, '#tinfo', /tracked|failed/, 300000);
  r.trackWallSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  if (/failed/.test(r.trackInfo)) throw new Error(r.trackInfo);
  r.status = await (await page.request.get(`${BASE}/api/jobs/${r.job}/status`)).json();
  r.backend = r.status.backend;
  r.trackImageSize = r.status.image_size;

  // --- dots look
  await setMode(page, 'dots');
  await page.evaluate(() => window.DV_draw(20)); await sleep(600);
  r.dotsPreview = await census(page);
  r.dotsFps = await page.textContent('#fps');
  await page.screenshot({ path: path.join(DOCS, 'c-dots.png') });

  await openStep(page, 'st5');
  await page.click('#bExport');
  r.dotsRender = await waitText(page, '#rinfo', /rendered|failed/, 300000);
  if (/failed/.test(r.dotsRender)) throw new Error(r.dotsRender);
  r.dotsProbe = ffprobe(path.join(HERE, 'jobs', r.job, 'out.mp4'));
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', path.join(HERE, 'jobs', r.job, 'out.mp4'),
    '-vf', 'select=eq(n\\,20)', '-vframes', '1', path.join(DOCS, 'c-dots-frame20.png')]);

  // --- a per-pixel look through the same masks, each subject its own palette
  await openStep(page, 'st3');
  await setMode(page, 'errordiff');
  await page.selectOption('#sAlgo', 'atkinson'); await sleep(300);
  await page.click('#composeui .chip[data-compose="overlay"]'); await sleep(300);
  await openStep(page, 'st4');
  const targets = await page.$$eval('#target .chip', (n) => n.map((e) => e.textContent));
  r.paletteTargets = targets;
  await page.click('#target .chip:nth-child(1)');            // background / scene
  await page.click('#pals .chip:nth-child(18)'); await sleep(400);   // 8 Greys
  await page.click('#target .chip:nth-child(2)');            // subject 1
  await page.click('#pals .chip:nth-child(7)'); await sleep(400);    // Red Mono
  await page.click('#target .chip:nth-child(3)');            // subject 2
  await page.click('#pals .chip:nth-child(8)'); await sleep(400);    // Green Mono
  await page.evaluate(() => window.DV_draw(20)); await sleep(700);
  r.mixedPreview = await census(page);
  await page.click('#bCmp'); await sleep(400);
  await page.screenshot({ path: path.join(DOCS, 'c-mixed.png') });
  await page.click('#bCmp'); await sleep(200);

  await openStep(page, 'st5');
  await page.click('#bExport');
  r.mixedRender = await waitText(page, '#rinfo', /rendered|failed/, 300000);
  if (/failed/.test(r.mixedRender)) throw new Error(r.mixedRender);
  r.mixedProbe = ffprobe(path.join(HERE, 'jobs', r.job, 'out.mp4'));
  const mixPng = path.join(DOCS, 'c-mixed-frame20.png');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', path.join(HERE, 'jobs', r.job, 'out.mp4'),
    '-vf', 'select=eq(n\\,20)', '-vframes', '1', mixPng]);

  // preview vs export agreement on the same frame (JPEG decode + h264 differ,
  // so this is a similarity measure, not an equality assertion)
  const raw = execFileSync('ffmpeg', ['-v', 'error', '-i', mixPng, '-f', 'rawvideo',
    '-pix_fmt', 'rgb24', '-'], { maxBuffer: 1 << 28 });
  const prev = await page.evaluate(() => {
    const c = document.querySelector('#vcv');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    return Array.from(d);
  });
  let near = 0, n = 0;
  for (let p = 0, q = 0; p < raw.length; p += 3, q += 4) {
    const dr = raw[p] - prev[q], dg = raw[p + 1] - prev[q + 1], db = raw[p + 2] - prev[q + 2];
    if (dr * dr + dg * dg + db * db < 900) near++;
    n++;
  }
  r.previewVsExport = { pixels: n, within30: near, pct: +(100 * near / n).toFixed(2) };
  await page.screenshot({ path: path.join(DOCS, 'c-export.png') });
  return r;
}

/* --------- E: lasso prompt + first-frame preview, one subject -------------- */
async function runLasso(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.setInputFiles('#file', CLIP);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
  r.job = await page.evaluate(() => window.DV.job);
  await page.click('#scope .chip[data-scope="track"]');
  await sleep(700);

  // --- box prompt first, and preview it: this is the "is my click enough?" step
  await prompt(page, SUBJECT_A);
  let t0 = Date.now();
  await page.click('#bPrev');
  r.boxPreviewText = await waitText(page, '#pvinfo', /subject|failed/, 120000);
  r.boxPreviewWallSeconds = +((Date.now() - t0) / 1000).toFixed(2);
  if (/failed/.test(r.boxPreviewText)) throw new Error(r.boxPreviewText);
  r.boxPreviewOverlay = await page.evaluate(() => Object.keys(window.DV.previewMasks || {}).length);
  await page.screenshot({ path: path.join(DOCS, 'e-preview-box.png') });

  // --- now a polygon around the same subject, drawn on the prompt canvas
  await page.click('#ptool .chip[data-tool="poly"]');
  await page.click('#bClr');
  const poly = await page.evaluate(() => {
    const c = document.querySelector('#pov'), b = c.getBoundingClientRect();
    // a 27-point trace of the athlete at frame 0 — what a careful lasso looks
    // like (IoU 0.944 against the box-prompted mask on its own)
    const P = [[597,111],[580,114],[573,140],[568,144],[527,145],[473,159],[447,158],
               [444,162],[447,175],[470,167],[524,164],[493,218],[494,236],[507,263],
               [526,279],[520,296],[520,350],[568,357],[544,338],[564,271],[552,238],
               [552,216],[560,198],[585,172],[584,158],[607,135],[609,120]];
    return { rect: { x: b.x, y: b.y, w: b.width, h: b.height }, pts: P, W: c.width, H: c.height };
  });
  for (const [x, y] of poly.pts) {
    await page.mouse.click(poly.rect.x + (x / poly.W) * poly.rect.w,
                           poly.rect.y + (y / poly.H) * poly.rect.h);
  }
  await page.keyboard.press('Enter');
  r.shapes = await page.evaluate(() => window.DV.subjects[0].paths.map((p) => p.pts.length));
  r.maskDataURLBytes = await page.evaluate(() => {
    const u = window.DV_maskURL(window.DV.subjects[0]); return u ? u.length : 0;
  });

  t0 = Date.now();
  await page.click('#bPrev');
  r.lassoPreviewText = await waitText(page, '#pvinfo', /subject|failed/, 120000);
  r.lassoPreviewWallSeconds = +((Date.now() - t0) / 1000).toFixed(2);
  if (/failed/.test(r.lassoPreviewText)) throw new Error(r.lassoPreviewText);
  await page.screenshot({ path: path.join(DOCS, 'e-preview-lasso.png') });

  // --- and it tracks the whole clip from that mask alone
  t0 = Date.now();
  await page.click('#bTrack');
  r.trackInfo = await waitText(page, '#tinfo', /tracked|failed/, 300000);
  r.trackWallSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  if (/failed/.test(r.trackInfo)) throw new Error(r.trackInfo);
  const st = await (await page.request.get(`${BASE}/api/jobs/${r.job}/status`)).json();
  r.doneFrames = st.done_frames; r.fps = st.fps; r.backend = st.backend;
  const meta = await (await page.request.get(`${BASE}/api/jobs/${r.job}/meta`)).json();
  r.promptRecorded = meta.prompts;
  await setMode(page, 'dots');
  await page.evaluate(() => window.DV_draw(20)); await sleep(600);
  r.dotsPreview = await census(page);
  await page.screenshot({ path: path.join(DOCS, 'e-lasso-tracked.png') });

  // --- and the mask prompt has to agree with the box prompt over the clip
  const FR = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140];
  const grab = (want) => page.evaluate(async (frames) => {
    const out = {};
    for (const n of frames) {
      const im = new Image();
      await new Promise((res) => { im.onload = im.onerror = res;
        im.src = `/api/jobs/${window.DV.job}/mask/1/${n}?t=` + Date.now(); });
      const c = document.createElement('canvas');
      c.width = im.width; c.height = im.height;
      const g = c.getContext('2d'); g.drawImage(im, 0, 0);
      const d = g.getImageData(0, 0, c.width, c.height).data;
      const bits = new Uint8Array(c.width * c.height);
      for (let i = 0, p = 0; i < bits.length; i++, p += 4) bits[i] = d[p] > 127 ? 1 : 0;
      out[n] = Array.from(bits);
    }
    return out;
  }, want);
  const lassoMasks = await grab(FR);

  await openStep(page, 'st2');                       // tracking collapses it
  await page.click('#bClr');
  await page.click('#ptool .chip[data-tool="point"]');
  await prompt(page, SUBJECT_A);
  await page.click('#bTrack');
  // #tinfo still holds the first run's text; track() clears it on click, so wait
  // for that before waiting for the new one, or we would read the old masks
  await page.waitForFunction(() => document.querySelector('#tinfo').hidden === true,
                             { timeout: 10000 });
  await waitText(page, '#tinfo', /tracked|failed/, 300000);
  const boxMasks = await grab(FR);

  const ious = FR.map((n) => {
    const a = lassoMasks[n], b = boxMasks[n];
    let i = 0, u = 0;
    for (let k = 0; k < a.length; k++) { if (a[k] & b[k]) i++; if (a[k] | b[k]) u++; }
    return u ? i / u : 1;
  });
  r.maskVsBoxIoU = {
    frames: FR, mean: +(ious.reduce((x, y) => x + y, 0) / ious.length).toFixed(4),
    min: +Math.min(...ious).toFixed(4),
  };
  if (r.maskVsBoxIoU.mean < 0.95) {
    throw new Error('mask-prompt IoU vs box prompt ' + r.maskVsBoxIoU.mean + ' < 0.95');
  }
  return r;
}

/* ------------------------- D: one subject at a non-default tracking quality */
async function runTrackedFast(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.setInputFiles('#file', CLIP);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
  r.job = await page.evaluate(() => window.DV.job);

  await page.click('#scope .chip[data-scope="track"]');
  await sleep(700);
  await prompt(page, SUBJECT_A);
  await page.click('#tq .chip[data-size="512"]');
  r.selected = await page.evaluate(() => window.DV.trackSize);

  const t0 = Date.now();
  await page.click('#bTrack');
  r.trackInfo = await waitText(page, '#tinfo', /tracked|failed/, 300000);
  r.trackWallSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  if (/failed/.test(r.trackInfo)) throw new Error(r.trackInfo);
  const st = await (await page.request.get(`${BASE}/api/jobs/${r.job}/status`)).json();
  r.imageSize = st.image_size;
  r.backend = st.backend;
  r.doneFrames = st.done_frames;
  r.fps = st.fps;
  if (st.image_size !== 512) throw new Error('server tracked at ' + st.image_size + ', wanted 512');

  // the masks must still come back at the clip's own resolution
  const meta = await (await page.request.get(`${BASE}/api/jobs/${r.job}/meta`)).json();
  const png = await (await page.request.get(`${BASE}/api/jobs/${r.job}/mask/1/10`)).body();
  r.maskBytes = png.length;
  r.maskSize = [png.readUInt32BE(16), png.readUInt32BE(20)];   // IHDR w,h
  r.clipSize = [meta.w, meta.h];
  if (r.maskSize[0] !== meta.w || r.maskSize[1] !== meta.h) {
    throw new Error('mask ' + r.maskSize + ' != clip ' + r.clipSize);
  }
  await setMode(page, 'dots');
  await page.evaluate(() => window.DV_draw(20)); await sleep(600);
  r.dotsPreview = await census(page);
  await page.screenshot({ path: path.join(DOCS, 'd-fast-quality.png') });
  return r;
}

/* ---------- F: every export format the server can write --------------------
 * Reuses whatever clip is already tracked on the page, so this costs five
 * renders and no tracking. Each file is ffprobed, and both alpha formats have
 * a frame decoded to prove the background really is transparent.
 */
async function runFormats(page) {
  const r = { formats: {} };
  const job = await page.evaluate(() => window.DV.job);
  r.job = job;
  r.offered = await page.evaluate(() => window.DV_formats());
  if (!r.offered.length) throw new Error('the server engine offered no formats');
  await setMode(page, 'dots');
  await openStep(page, 'st5');
  const files = { mp4: 'out.mp4', webm: 'out.webm', gif: 'out.gif',
                  'webm-alpha': 'out.alpha.webm', prores: 'out.mov' };
  for (const id of ['mp4', 'webm', 'gif', 'webm-alpha', 'prores']) {
    const t0 = Date.now();
    const sel = await page.evaluate((x) => window.DV_setFormat(x), id);
    if (sel.id !== id) throw new Error(`could not select ${id}`);
    await page.click('#bExport');
    const txt = await waitText(page, '#rinfo', /rendered|failed/, 300000);
    if (/failed/.test(txt)) throw new Error(`${id}: ${txt}`);
    const file = path.join(HERE, 'jobs', job, files[id]);
    const st = fs.statSync(file);
    const probe = ffprobeFull(file);
    r.formats[id] = { seconds: +((Date.now() - t0) / 1000).toFixed(1), bytes: st.size,
                      probe, info: txt.trim() };
  }
  const F = r.formats;
  const nFrames = +F.mp4.probe.nb_read_frames;
  if (F.mp4.probe.codec_name !== 'h264' || F.mp4.probe.pix_fmt !== 'yuv420p') {
    throw new Error('mp4: ' + JSON.stringify(F.mp4.probe));
  }
  if (F.webm.probe.codec_name !== 'vp9') throw new Error('webm: ' + JSON.stringify(F.webm.probe));
  if (F.gif.probe.codec_name !== 'gif') throw new Error('gif: ' + JSON.stringify(F.gif.probe));
  // the GIF is asked for at 15 fps against a 30 fps clip: half the frames
  const gifFrames = +F.gif.probe.nb_read_frames;
  if (Math.abs(gifFrames - nFrames / 2) > 1) {
    throw new Error(`gif has ${gifFrames} frames, expected ~${nFrames / 2}`);
  }
  if (F['webm-alpha'].probe['tags:alpha_mode'] !== '1'
      && F['webm-alpha'].probe.tags?.alpha_mode !== '1') {
    throw new Error('webm-alpha carries no alpha_mode tag: '
      + JSON.stringify(F['webm-alpha'].probe));
  }
  if (!/^yuva/.test(F.prores.probe.pix_fmt)) {
    throw new Error('prores has no alpha plane: ' + F.prores.probe.pix_fmt);
  }
  // decode one frame out of each and count what the alpha channel says
  const w = +F.mp4.probe.width, h = +F.mp4.probe.height;
  const alphaCensus = (file, extraIn) => {
    const { data } = decodeFrameRGBA(file, 10, w, h, extraIn);
    // lossy codecs round the alpha plane, so this is a threshold, not equality
    let zero = 0, full = 0;
    for (let p = 3; p < data.length; p += 4) {
      if (data[p] < 16) zero++; else if (data[p] > 200) full++;
    }
    const n = data.length / 4;
    return { pixels: n, transparentPct: +(100 * zero / n).toFixed(1),
             opaquePct: +(100 * full / n).toFixed(1) };
  };
  r.alpha = {
    'webm-alpha': alphaCensus(path.join(HERE, 'jobs', job, files['webm-alpha']),
                              ['-c:v', 'libvpx-vp9']),
    prores: alphaCensus(path.join(HERE, 'jobs', job, files.prores)),
  };
  for (const [k, v] of Object.entries(r.alpha)) {
    if (v.transparentPct < 50) {
      throw new Error(`${k}: only ${v.transparentPct}% of the frame is transparent`);
    }
    if (v.opaquePct < 0.2) throw new Error(`${k}: nothing is opaque — no dots?`);
  }
  // and the GIF decodes to the flat colours it was given
  const gifPng = path.join(DOCS, 'f-gif-frame.png');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i',
    path.join(HERE, 'jobs', job, files.gif), '-vf', 'select=eq(n\\,5)',
    '-vframes', '1', gifPng]);
  const raw = execFileSync('ffmpeg', ['-v', 'error', '-i', gifPng, '-f', 'rawvideo',
    '-pix_fmt', 'rgb24', '-'], { maxBuffer: 1 << 28 });
  const seen = new Set();
  for (let p = 0; p < raw.length; p += 3) {
    seen.add((raw[p] << 16) | (raw[p + 1] << 8) | raw[p + 2]);
  }
  r.gifColours = seen.size;
  if (seen.size < 2 || seen.size > 8) {
    throw new Error('the GIF frame has ' + seen.size + ' colours, expected the flat few');
  }
  await page.screenshot({ path: path.join(DOCS, 'f-formats.png') });
  return r;
}

/* ---------- O: the matched cut ---------------------------------------------
 * "Also save the original" is only worth anything if the second file is frame
 * for frame the first one. So: the same trim (a 2 s window out of a 5 s clip,
 * not the whole file), the same count, rate and size out of ffprobe, and three
 * sampled frames decoded and compared against the very JPEGs the render read.
 * Plus the two things that keep it honest -- the server refuses a frame count
 * that does not match, and a GIF export pairs with an MP4 rather than a
 * pointless second GIF.
 */
async function runOriginal(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.evaluate(() => window.DV_limit(2));      // 2 s of a 5 s clip
  await page.setInputFiles('#file', CLIP);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
  r.job = await page.evaluate(() => window.DV.job);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  if (r.nFrames !== 60) throw new Error('the 2 s window gave ' + r.nFrames + ' frames');

  await setMode(page, 'ordered');
  await openStep(page, 'st5');
  await page.evaluate((x) => window.DV_setFormat(x), 'mp4');
  r.checkboxOffered = !(await page.locator('#origui').isHidden());
  if (!r.checkboxOffered) throw new Error('no "also save the original" checkbox');
  await page.check('#cOrig');
  await page.click('#bExport');
  r.info = await waitText(page, '#rinfo', /original cut|failed/, 300000);
  if (/failed/.test(r.info)) throw new Error(r.info);
  await sleep(600);
  // both links in the frame: the screenshot IS the "two files" claim
  await page.locator('#dlorig').scrollIntoViewIfNeeded();
  await sleep(300);
  await page.screenshot({ path: path.join(DOCS, 'o-original.png') });

  const dith = path.join(HERE, 'jobs', r.job, 'out.mp4');
  const orig = path.join(HERE, 'jobs', r.job, 'out.original.mp4');
  r.dithered = ffprobeFull(dith);
  r.original = ffprobeFull(orig);
  for (const k of ['nb_read_frames', 'width', 'height', 'r_frame_rate']) {
    if (String(r.dithered[k]) !== String(r.original[k])) {
      throw new Error(`the pair disagrees on ${k}: ${r.dithered[k]} vs ${r.original[k]}`);
    }
  }
  if (+r.original.nb_read_frames !== r.nFrames) {
    throw new Error(`the original cut has ${r.original.nb_read_frames} frames, `
      + `the clip has ${r.nFrames}`);
  }

  /* frame N of the original IS jobs/<id>/frames/<N>.jpg -- the fps-normalised,
   * trimmed ground truth both files were made from. h264 is lossy, so this is
   * a mean absolute difference and not equality. */
  const rawRGB = (args) => execFileSync('ffmpeg', ['-v', 'error'].concat(args),
                                        { maxBuffer: 1 << 28 });
  const meanAbs = (a, b) => {
    if (a.length !== b.length) throw new Error('frame sizes differ: ' + a.length + ' vs ' + b.length);
    let d = 0;
    for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
    return +(d / a.length).toFixed(3);
  };
  r.frameMeanAbsDiff = {};
  for (const n of [0, r.nFrames >> 1, r.nFrames - 1]) {
    const fromFile = rawRGB(['-i', orig, '-vf', `select=eq(n\\,${n})`, '-vframes', '1',
                             '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']);
    const fromDisk = rawRGB(['-i', frameFile(r.job, n), '-pix_fmt', 'rgb24',
                             '-f', 'rawvideo', '-']);
    const d = meanAbs(fromFile, fromDisk);
    r.frameMeanAbsDiff[n] = d;
    if (d >= 2) {
      throw new Error(`original frame ${n} is not the clip's frame ${n} (mean abs diff ${d})`);
    }
  }

  // two links, two files: the second one downloads and is named for the pair
  const [d1] = await Promise.all([
    page.waitForEvent('download', { timeout: 30000 }),
    page.click('#dlorig'),
  ]);
  r.downloadName = d1.suggestedFilename();
  const saved = path.join(DOCS, 'o-original-cut.mp4');
  await d1.saveAs(saved);
  r.downloadBytes = fs.statSync(saved).size;
  if (!/\.original\.mp4$/.test(r.downloadName)) {
    throw new Error('the second download is called ' + r.downloadName);
  }

  // the server refuses a count that would not line up
  const bad = await page.request.post(`${BASE}/api/jobs/${r.job}/original`,
    { data: { format: 'mp4', expect_frames: r.nFrames - 1 }, timeout: 60000 });
  r.mismatchStatus = bad.status();
  if (bad.status() !== 409) throw new Error('a wrong frame count got ' + bad.status());

  // a GIF pairs with an MP4: a GIF of the original would be decimated to
  // gif_fps, and pairing a GIF with a GIF is pointless
  await page.evaluate((x) => window.DV_setFormat(x), 'gif');
  await page.click('#bExport');
  r.gifInfo = await waitText(page, '#rinfo', /original cut|failed/, 300000);
  if (/failed/.test(r.gifInfo)) throw new Error(r.gifInfo);
  r.gifPairName = await page.getAttribute('#dlorig', 'download');
  if (!/\.original\.mp4$/.test(r.gifPairName)) {
    throw new Error('the GIF paired with ' + r.gifPairName);
  }
  r.gifOriginal = ffprobeFull(orig);
  if (+r.gifOriginal.nb_read_frames !== r.nFrames) {
    throw new Error('the GIF pair lost frames: ' + r.gifOriginal.nb_read_frames);
  }

  // the checkbox is remembered for the session, and it is a video-only question
  await page.reload({ waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  r.rememberedAcrossReload = await page.evaluate(() => window.DV.saveOriginal);
  if (!r.rememberedAcrossReload) throw new Error('the checkbox forgot itself');
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 30000 });
  await openStep(page, 'st5');
  r.hiddenForStills = await page.locator('#origui').isHidden();
  if (!r.hiddenForStills) throw new Error('a still was offered a matched cut');
  return r;
}

/* ---------- T: trim and length, at the API level ---------------------------
 * The UI half of this is verify-web.mjs's camera run (record, drag the handles,
 * re-open). This is the server's own arithmetic, and the thing that replaced
 * the old 10 s / 300 frame ceiling: -ss/-t cut exactly the window asked for,
 * a clip well past the old cap arrives whole, the legacy `max_seconds` /
 * `max_frames` fields are accepted and ignored, and a SECOND trim of a clip
 * the server already holds costs one ffmpeg run and no upload.
 */
async function runTrim(page) {
  const r = {};
  const post = async (file, fields) => {
    const res = await page.request.post(`${BASE}/api/upload`, {
      multipart: Object.assign({
        file: { name: path.basename(file), mimeType: 'video/mp4',
                buffer: fs.readFileSync(file) },
      }, fields),
      timeout: 600000,
    });
    if (!res.ok()) throw new Error('upload failed: ' + res.status() + ' ' + await res.text());
    return res.json();
  };
  r.whole = await post(CLIP, {});
  r.middle = await post(CLIP, { trim_start: '2.0', trim_end: '4.0' });
  // the fields the old capped API took. A page from before this change still
  // sends them; the answer to both is now "no cap", so they change nothing.
  r.legacyFieldsIgnored = await post(CLIP, { max_seconds: '1', max_frames: '30' });
  if (r.whole.n_frames !== 150) throw new Error('whole clip: ' + r.whole.n_frames + ' frames');
  if (r.middle.n_frames !== 60) throw new Error('2 s trim: ' + r.middle.n_frames + ' frames');
  if (r.legacyFieldsIgnored.n_frames !== 150) {
    throw new Error('max_seconds=1 still capped the clip to '
      + r.legacyFieldsIgnored.n_frames + ' frames');
  }

  // and the trimmed clip really starts where it was told to
  const ref = path.join(DOCS, 't-trim-ref.jpg');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-ss', '2.0', '-i', CLIP,
    '-frames:v', '1', '-vf', 'scale=-2:720', '-q:v', '3', ref]);
  const raw = (f) => execFileSync('ffmpeg', ['-v', 'error', '-i', f, '-f', 'rawvideo',
    '-pix_fmt', 'rgb24', '-'], { maxBuffer: 1 << 28 });
  const meanAbs = (fa, fb) => {
    const a = raw(fa), b = raw(fb);
    let diff = 0;
    for (let i = 0; i < a.length; i++) diff += Math.abs(a[i] - b[i]);
    return +(diff / a.length).toFixed(3);
  };
  r.firstFrameMeanAbsDiff = meanAbs(ref, frameFile(r.middle.job, 0));
  if (r.firstFrameMeanAbsDiff > 0.5) {
    throw new Error('the trimmed clip does not start at 2 s (mean abs diff '
      + r.firstFrameMeanAbsDiff + ')');
  }

  /* No cap. The old ceiling was 300 frames / 10 s, so the proof is a clip
   * comfortably past it: the 5 s sample looped to 30 s, which has to arrive as
   * all 900 frames. Built here rather than committed — it is the same pixels. */
  const long = path.join(os.tmpdir(), 'dv-verify-long-30s.mp4');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-stream_loop', '5', '-i', CLIP,
    '-c', 'copy', long]);
  const t0 = Date.now();
  r.uncapped = await post(long, {});
  r.uncappedExtractS = +((Date.now() - t0) / 1000).toFixed(1);
  if (r.uncapped.n_frames !== 900) {
    throw new Error('a 30 s clip came back as ' + r.uncapped.n_frames
      + ' frames, not 900 — something is still capping');
  }

  /* Re-trim without re-upload: the source clip stayed in the job directory,
   * so a different range is one ffmpeg run against bytes that are already
   * here. It lands in a NEW job — the old one's masks belong to the old
   * range — and it starts exactly where the new -ss says. */
  const re = await page.request.post(
    `${BASE}/api/jobs/${r.whole.job}/reextract`,
    { data: { trim_start: 1.0, trim_end: 2.5 }, timeout: 300000 });
  if (!re.ok()) throw new Error('reextract failed: ' + re.status() + ' ' + await re.text());
  r.reextract = await re.json();
  if (r.reextract.job === r.whole.job) throw new Error('re-extract reused the old job');
  if (r.reextract.n_frames !== 45) {
    throw new Error('a 1.5 s re-cut gave ' + r.reextract.n_frames + ' frames');
  }
  const ref1 = path.join(DOCS, 't-recut-ref.jpg');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-ss', '1.0', '-i', CLIP,
    '-frames:v', '1', '-vf', 'scale=-2:720', '-q:v', '3', ref1]);
  r.recutFirstFrameMeanAbsDiff = meanAbs(ref1, frameFile(r.reextract.job, 0));
  if (r.recutFirstFrameMeanAbsDiff > 0.5) {
    throw new Error('the re-cut clip does not start at 1 s (mean abs diff '
      + r.recutFirstFrameMeanAbsDiff + ')');
  }
  // no second copy of the source: the re-cut job hard-links it
  const ino = (j) => fs.statSync(path.join(HERE, 'jobs', j, 'source.mp4')).ino;
  r.sourceShared = ino(r.whole.job) === ino(r.reextract.job);
  if (!r.sourceShared) throw new Error('the re-cut copied the source instead of linking it');

  // a range that runs off the end is clamped to what is there, not refused
  r.pastEnd = await post(CLIP, { trim_start: '4.0', trim_end: '99.0' });
  if (r.pastEnd.n_frames !== 30) {
    throw new Error('a trim past the end gave ' + r.pastEnd.n_frames + ' frames');
  }
  return r;
}

/* jobs/<id>/frames/<n>.jpg — the filename widens past ~9,000 frames so that
 * sorted() stays in order, so ask the job how wide its names are. */
function frameFile(job, n) {
  const d = path.join(HERE, 'jobs', job, 'frames');
  const pad = (fs.readdirSync(d).find((f) => f.endsWith('.jpg')) || '0000.jpg')
    .replace(/\.jpg$/, '').length;
  return path.join(d, String(n).padStart(pad, '0') + '.jpg');
}

/* ---------- R: the range, AFTER the track ---------------------------------
 * The bug this closes: dragging the trim after tracking used to re-extract the
 * clip into a new job, which threw the masks away and forced a second track.
 *
 * The frames and the per-frame masks are on disk under one job id and neither
 * moves, so a narrower range is a WINDOW on them. What this proves:
 *
 *   - narrowing fires no /track, no /reextract and no /upload at all, and the
 *     job id and frame count on the page do not change
 *   - the render, the matched original cut and the .dots.gz all come out at
 *     the window's length, and frame 0 of the original IS jobs/<id>/frames/
 *     <in>.jpg -- the frame-exactness the matched cut promises, for a trim
 *   - the .dots.gz of the window is byte-for-byte the same slice of the dot
 *     positions the whole clip gives, which is the mask offset being right
 *   - a range that runs PAST what was extracted is not silently re-cut: the
 *     page names the frames that are not tracked yet and waits, and only a
 *     click on the offer costs a track
 */
async function runRange(page) {
  const r = {};
  const seen = [];
  const tap = (rq) => seen.push(rq.method() + ' ' + rq.url().replace(BASE, ''));
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  // 2 s of the 5 s clip, so there is room to widen later as well as narrow
  await page.evaluate(() => window.DV_limit(2));
  await page.setInputFiles('#file', CLIP);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
  r.job = await page.evaluate(() => window.DV.job);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  if (r.nFrames !== 60) throw new Error('the 2 s window gave ' + r.nFrames + ' frames');

  await page.click('#scope .chip[data-scope="track"]');
  await sleep(700);
  await prompt(page, SUBJECT_A);
  await page.click('#bTrack');
  r.track = await waitText(page, '#tinfo', /tracked|failed/, 600000);
  if (/failed/.test(r.track)) throw new Error(r.track);
  r.maskFiles = fs.readdirSync(path.join(HERE, 'jobs', r.job, 'masks', '1')).length;
  if (r.maskFiles !== r.nFrames) {
    throw new Error(`${r.maskFiles} mask files for ${r.nFrames} frames`);
  }
  r.rangeBefore = await page.evaluate(() => window.DV_range.label());

  /* --- narrow. Nothing may go over the wire but frames and masks. --- */
  page.on('request', tap);
  await page.evaluate(() => window.DV_range.seconds(0.5, 1.5));
  await sleep(400);
  r.range = await page.evaluate(() => window.DV_range.get());
  r.rangeLabel = await page.evaluate(() => window.DV_range.label());
  r.resetOffered = await page.evaluate(() => window.DV_range.resetShown());
  r.narrowRequests = seen.slice();
  r.retracked = seen.filter((x) => /\/(track|reextract|upload)/.test(x));
  if (r.range.in !== 15 || r.range.out !== 44 || r.range.n !== 30) {
    throw new Error('0.5–1.5 s of a 30 fps clip gave ' + JSON.stringify(r.range));
  }
  if (r.retracked.length) {
    throw new Error('narrowing the trim fired ' + r.retracked.join(', '));
  }
  if ((await page.evaluate(() => window.DV.job)) !== r.job) {
    throw new Error('narrowing the trim moved the clip to another job');
  }
  if ((await page.evaluate(() => window.DV.nFrames)) !== r.nFrames) {
    throw new Error('narrowing the trim changed the frame count on disk');
  }
  if (!/15–44 of 60/.test(r.rangeLabel)) throw new Error('range label: ' + r.rangeLabel);
  if (!r.resetOffered) throw new Error('no way back to the whole clip');
  await page.screenshot({ path: path.join(DOCS, 'r-range-narrowed.png') });

  /* --- render + matched cut, both cut to the window --- */
  await setMode(page, 'dots');
  await openStep(page, 'st5');
  await page.evaluate((x) => window.DV_setFormat(x), 'mp4');
  await page.check('#cOrig');
  await page.click('#bExport');
  r.export = await waitText(page, '#rinfo', /original cut|failed/, 300000);
  if (/failed/.test(r.export)) throw new Error(r.export);
  const dith = path.join(HERE, 'jobs', r.job, 'out.mp4');
  const orig = path.join(HERE, 'jobs', r.job, 'out.original.mp4');
  r.dithered = ffprobeFull(dith);
  r.original = ffprobeFull(orig);
  for (const k of ['nb_read_frames', 'width', 'height', 'r_frame_rate']) {
    if (String(r.dithered[k]) !== String(r.original[k])) {
      throw new Error(`the pair disagrees on ${k}: ${r.dithered[k]} vs ${r.original[k]}`);
    }
  }
  if (+r.dithered.nb_read_frames !== r.range.n) {
    throw new Error(`the render has ${r.dithered.nb_read_frames} frames for a `
      + `${r.range.n}-frame range`);
  }

  /* frame k of the matched cut IS jobs/<id>/frames/<in + k>.jpg */
  const rawRGB = (args) => execFileSync('ffmpeg', ['-v', 'error'].concat(args),
                                        { maxBuffer: 1 << 28 });
  const meanAbs = (a, b) => {
    if (a.length !== b.length) throw new Error('frame sizes differ');
    let d = 0;
    for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
    return +(d / a.length).toFixed(3);
  };
  const fromCut = (k) => rawRGB(['-i', orig, '-vf', `select=eq(n\\,${k})`,
    '-vframes', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']);
  const fromDisk = (n) => rawRGB(['-i', frameFile(r.job, n), '-pix_fmt', 'rgb24',
    '-f', 'rawvideo', '-']);
  r.frameMeanAbsDiff = {};
  for (const k of [0, r.range.n >> 1, r.range.n - 1]) {
    const d = meanAbs(fromCut(k), fromDisk(r.range.in + k));
    r.frameMeanAbsDiff[`${k} vs frames/${r.range.in + k}`] = d;
    if (d >= 2) {
      throw new Error(`cut frame ${k} is not the clip's frame ${r.range.in + k} `
        + `(mean abs diff ${d})`);
    }
  }
  // the control: it must NOT be frame 0 of the clip, or the check proves nothing
  r.controlMeanAbsDiff = meanAbs(fromCut(0), fromDisk(0));
  if (r.controlMeanAbsDiff < 2) {
    throw new Error('the cut starts at frame 0 — the window was ignored');
  }

  /* --- the dots, as data: the window is the slice --- */
  const dots = async (body) => {
    const res = await page.request.post(`${BASE}/api/jobs/${r.job}/dots`,
      { data: Object.assign({ subjects: [{ id: 1 }], json: true }, body),
        timeout: 300000 });
    if (!res.ok()) throw new Error('dots: ' + res.status() + ' ' + await res.text());
    const j = await res.json();
    return { stats: j,
             doc: JSON.parse(fs.readFileSync(path.join(HERE, 'jobs', r.job,
                                                       'out.dots.json'), 'utf8')) };
  };
  const win = await dots({ frame_in: r.range.in, frame_out: r.range.out });
  const all = await dots({});
  r.dotsWindow = { frames: win.stats.frames, bytes: win.stats.bytes,
                   frame_in: win.stats.frame_in, frame_out: win.stats.frame_out };
  r.dotsWhole = { frames: all.stats.frames, bytes: all.stats.bytes };
  if (win.stats.frames !== r.range.n) {
    throw new Error(`the .dots.gz has ${win.stats.frames} frames for a `
      + `${r.range.n}-frame range`);
  }
  r.dotsIsTheSlice = win.doc.frames.every(
    (f, i) => JSON.stringify(f) === JSON.stringify(all.doc.frames[r.range.in + i]));
  if (!r.dotsIsTheSlice) {
    throw new Error('the windowed dot positions are not the whole clip\'s slice '
      + '— the masks are being read at the wrong offset');
  }

  /* --- back to the whole clip, in one click --- */
  await page.evaluate(() => window.DV_range.full());
  await sleep(300);
  r.afterReset = await page.evaluate(() => window.DV_range.get());
  if (!r.afterReset.whole || r.afterReset.n !== r.nFrames) {
    throw new Error('"full clip" did not restore the whole range');
  }

  /* --- wider than what is on disk: an offer, not a silent re-cut --- */
  seen.length = 0;
  await page.evaluate(() => window.DV_range.seconds(0, 4));
  await sleep(400);
  r.offer = await page.evaluate(() => window.DV_range.offer());
  r.offerVisible = await page.locator('#trimoffer').isVisible();
  r.offerRequests = seen.filter((x) => /\/(track|reextract|upload)/.test(x));
  if (!r.offer) throw new Error('a range past the end did not raise the offer');
  if (!r.offerVisible) throw new Error('the offer is in the DOM but not on screen');
  if (r.offerRequests.length) {
    throw new Error('the offer alone fired ' + r.offerRequests.join(', '));
  }
  if (JSON.stringify(r.offer.missing) !== JSON.stringify([[60, 119]])) {
    throw new Error('the offer names ' + JSON.stringify(r.offer.missing)
      + ', not frames 60–119');
  }
  if ((await page.evaluate(() => window.DV.job)) !== r.job) {
    throw new Error('the offer moved the clip before anyone accepted it');
  }
  await page.locator('#trimoffer').scrollIntoViewIfNeeded();
  await sleep(200);
  await page.screenshot({ path: path.join(DOCS, 'r-range-offer.png') });

  /* --- and only now does it cost a track --- */
  await page.click('#bExtend');
  await page.waitForFunction((old) => window.DV.job && window.DV.job !== old,
                             r.job, { timeout: 300000 });
  r.extendTrack = await waitText(page, '#tinfo', /tracked 120|failed/, 900000);
  if (/failed/.test(r.extendTrack)) throw new Error(r.extendTrack);
  r.afterExtend = await page.evaluate(() => ({
    job: window.DV.job, nFrames: window.DV.nFrames, tracked: window.DV.tracked,
    promptFrames: window.DV.subjects.map((s) => s.promptFrame),
    range: window.DV_range.get(),
  }));
  r.extendRequests = seen.filter((x) => /\/(track|reextract)/.test(x));
  if (r.afterExtend.nFrames !== 120) {
    throw new Error('the wider range came back as ' + r.afterExtend.nFrames
      + ' frames, not 120');
  }
  if (!r.afterExtend.tracked) throw new Error('the wider range is not tracked');
  if (!r.afterExtend.range.whole) throw new Error('the new clip opened trimmed');
  r.extendMasks = fs.readdirSync(
    path.join(HERE, 'jobs', r.afterExtend.job, 'masks', '1')).length;
  if (r.extendMasks !== 120) {
    throw new Error(`${r.extendMasks} masks for 120 frames after the extend`);
  }
  page.removeListener('request', tap);
  return r;
}

/* ---------- G: dot data and sequences, server side ------------------------
 * The dots as positions rather than pixels, the .dots.gz they pack into, and
 * the route that turns a finished sequence back into a video. The identity
 * that matters is the last one: replaying a document has to land on exactly
 * the pixels the renderer painted.
 */
async function runDotsServer(page) {
  const r = {};
  const job = await page.evaluate(() => window.DV.job);
  r.job = job;
  const res = await page.request.post(`${BASE}/api/jobs/${job}/dots`, {
    data: { subjects: [{ id: 1, palette: ['#c9d4c5', '#b0413e'] }], json: true },
    timeout: 300000,
  });
  if (!res.ok()) throw new Error('/dots failed: ' + res.status() + ' ' + await res.text());
  r.stats = await res.json();
  const gz = path.join(HERE, 'jobs', job, 'out.dots.gz');
  const bytes = fs.readFileSync(gz);
  if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) throw new Error('.dots.gz is not gzip');
  r.mp4Bytes = fs.existsSync(path.join(HERE, 'jobs', job, 'out.mp4'))
    ? fs.statSync(path.join(HERE, 'jobs', job, 'out.mp4')).size : null;
  r.sizes = { gz: bytes.length, raw: r.stats.raw_bytes, json: r.stats.json_bytes,
              mp4: r.mp4Bytes };
  if (r.stats.frames < 10) throw new Error('dot data has ' + r.stats.frames + ' frames');
  if (r.stats.dots_mean < 50) throw new Error('dot data is nearly empty: '
    + JSON.stringify(r.stats));

  // replaying the document must paint exactly what the renderer painted
  const py = path.join(HERE, 'env', 'venv', 'bin', 'python');
  const code = `
import sys, json, numpy as np
sys.path.insert(0, ${JSON.stringify(path.join(HERE, 'server'))})
import render as R, dots as D
from PIL import Image
job = ${JSON.stringify(path.join(HERE, 'jobs', job))}
doc = D.unpack(open(job + '/out.dots.gz','rb').read())
a = R._params(dict(mode='dots', fps=doc['fps'], dotpx=doc['dotpx']))
blue = R.blue_noise(64, int(a['seed'])).astype(np.float32)
rgb = np.asarray(Image.open(job + '/frames/0010.jpg').convert('RGB'))
H, W = rgb.shape[:2]
F = R.dots_fields(H, W, a, blue)
m = np.asarray(Image.open(job + '/masks/1/0010.png').convert('L'), np.float32) / 255.0
painted = R._frame_dots(rgb, [m], a, F, [doc['subjects'][0]['color']], doc['bg'])
replay = D.paint(doc, 10)
print(json.dumps({'identical': bool(np.array_equal(painted, replay)),
                  'differing_px': int((painted != replay).any(-1).sum()),
                  'shape': list(painted.shape)}))
`;
  r.replay = JSON.parse(execFileSync(py, ['-c', code], { encoding: 'utf8' }).trim());
  if (!r.replay.identical) {
    throw new Error('replaying the dot data does not match the render: '
      + JSON.stringify(r.replay));
  }

  // and the sequence route: dot positions in, video out
  const up = await page.request.post(`${BASE}/api/sequence`, {
    multipart: { file: { name: 'seq.dots.gz', mimeType: 'application/octet-stream',
                         buffer: bytes }, format: 'mp4' },
    timeout: 300000,
  });
  if (!up.ok()) throw new Error('/api/sequence failed: ' + up.status());
  r.sequence = await up.json();
  const out = path.join(HERE, 'jobs', r.sequence.sequence, 'out.mp4');
  r.sequenceProbe = ffprobe(out);
  if (+r.sequenceProbe.nb_read_frames !== r.stats.frames) {
    throw new Error(`sequence mp4 has ${r.sequenceProbe.nb_read_frames} frames, `
      + `dot data has ${r.stats.frames}`);
  }
  return r;
}

/* --------- P: mask polish — motion-aware smoothing, both engines ----------
 * The claim under test is not "the mask looks nicer". It is:
 *   1. the strength maps to a documented set of filter sizes;
 *   2. the temporal window closes itself as a subject moves fast for its size,
 *      which is what keeps a struck ball out of the smear;
 *   3. the mask the tab computes for a frame is BYTE FOR BYTE the mask the
 *      server renders that frame with -- otherwise the preview is a lie;
 *   4. the render path uses it, and the before/after wipe still works.
 */
async function runPolish(page) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });

  // (1) and (2) are pure functions — check them before any pixels exist
  r.params = await page.evaluate(() => [0, 30, 70, 100].map((s) =>
    Object.assign({ strength: s }, window.MaskPolish.params(s))));
  if (r.params[2].radius !== 2 || r.params[3].radius !== 3 || r.params[0].radius !== 0) {
    throw new Error('polish strengths do not map to the documented radii: '
      + JSON.stringify(r.params));
  }
  r.gate = await page.evaluate(() => {
    const MP = window.MaskPolish;
    // a body: 140 px across, drifting 3 px a frame
    const slow = [-2, -1, 0, 1, 2].map((d) => ({ area: 20000, cx: 100 + 3 * d, cy: 100 }));
    // a ball: 36 px across, crossing 30 px a frame
    const fast = [-2, -1, 0, 1, 2].map((d) => ({ area: 1300, cx: 100 + 30 * d, cy: 100 }));
    const sum = (w) => Array.from(w).reduce((a, b) => a + b, 0);
    return { slow: Array.from(MP.weights(slow, 2, 2)).map((v) => +v.toFixed(3)),
             fast: Array.from(MP.weights(fast, 2, 2)).map((v) => +v.toFixed(3)),
             slowTotal: +sum(MP.weights(slow, 2, 2)).toFixed(3),
             fastTotal: +sum(MP.weights(fast, 2, 2)).toFixed(3) };
  });
  if (r.gate.fastTotal !== 1) {
    throw new Error('a fast small subject still got temporal averaging: '
      + JSON.stringify(r.gate));
  }
  if (r.gate.slowTotal < 2) {
    throw new Error('a slow large subject did not get its window: '
      + JSON.stringify(r.gate));
  }

  await page.setInputFiles('#file', CLIP);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
  r.job = await page.evaluate(() => window.DV.job);
  await page.click('#scope .chip[data-scope="track"]');
  await sleep(700);
  await prompt(page, SUBJECT_A);
  await page.click('#bTrack');
  r.trackInfo = await waitText(page, '#tinfo', /tracked|failed/, 300000);
  if (/failed/.test(r.trackInfo)) throw new Error(r.trackInfo);

  await setMode(page, 'dots');
  await page.evaluate(() => window.DV_draw(20)); await sleep(600);
  r.dotsOff = await page.textContent('#fps');
  await page.screenshot({ path: path.join(DOCS, 'p-polish-off.png') });

  // the UI: one row per subject, default off
  r.rowsBefore = await page.$$eval('#pollist .mini', (n) => n.map((e) => e.textContent));
  r.before = await page.evaluate(() => window.DV_polish.get());
  if (r.before[0].polish !== 0) throw new Error('polish is not off by default');
  await openStep(page, 'st3');
  await page.click('#pollist .chip.pol');
  await sleep(1500);
  r.after = await page.evaluate(() => window.DV_polish.get());
  if (r.after[0].polish !== 70) {
    throw new Error('the toggle did not turn polish on: ' + JSON.stringify(r.after));
  }
  await page.evaluate(() => window.DV_draw(20)); await sleep(1200);
  r.dotsOn = await page.textContent('#fps');
  await page.screenshot({ path: path.join(DOCS, 'p-polish-on.png') });

  // (3) the tab's polished mask against the server's polished PNG, byte for byte
  r.maskParity = await page.evaluate(async ({ job, id, frame }) => {
    const mine = await window.DV_polish.mask(id, frame);
    const im = new Image();
    await new Promise((ok, no) => {
      im.onload = ok; im.onerror = () => no(new Error('mask fetch failed'));
      im.src = `/api/jobs/${job}/mask/${id}/${frame}?polish=70&t=` + Date.now();
    });
    const c = document.createElement('canvas');
    c.width = im.naturalWidth; c.height = im.naturalHeight;
    c.getContext('2d').drawImage(im, 0, 0);
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let diff = 0, worst = 0, lit = 0;
    for (let q = 0; q < mine.length; q++) {
      const a = mine[q], b = d[q * 4];
      if (a) lit++;
      if (a !== b) { diff++; worst = Math.max(worst, Math.abs(a - b)); }
    }
    return { pixels: mine.length, lit, differing: diff, worst,
             w: c.width, h: c.height };
  }, { job: r.job, id: r.after[0].id, frame: 20 });
  if (r.maskParity.differing !== 0) {
    throw new Error('the tab and the server polish differently: '
      + JSON.stringify(r.maskParity));
  }

  // (4) the wipe still works with polish on, and the render uses the same masks
  await page.click('#bCmp'); await sleep(500);
  r.wipeVisible = await page.isVisible('#wipe');
  await page.screenshot({ path: path.join(DOCS, 'p-polish-wipe.png') });
  r.wipe = await page.evaluate(() => {
    const c = document.querySelector('#vcv');
    const g = c.getContext('2d');
    const l = g.getImageData(10, (c.height / 2) | 0, 60, 1).data;
    const rr = g.getImageData(c.width - 70, (c.height / 2) | 0, 60, 1).data;
    let dl = new Set(), dr = new Set();
    for (let p = 0; p < l.length; p += 4) dl.add(l[p] + ',' + l[p + 1] + ',' + l[p + 2]);
    for (let p = 0; p < rr.length; p += 4) dr.add(rr[p] + ',' + rr[p + 1] + ',' + rr[p + 2]);
    return { leftColours: dl.size, rightColours: dr.size };
  });
  if (r.wipe.leftColours <= r.wipe.rightColours) {
    throw new Error('the before/after wipe is not showing the original on the left: '
      + JSON.stringify(r.wipe));
  }
  await page.click('#bCmp'); await sleep(300);

  await openStep(page, 'st5');
  const t0 = Date.now();
  await page.click('#bExport');
  r.render = await waitText(page, '#rinfo', /rendered|failed/, 300000);
  if (/failed/.test(r.render)) throw new Error(r.render);
  r.renderWallSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  r.probe = ffprobe(path.join(HERE, 'jobs', r.job, 'out.mp4'));
  r.cached = fs.existsSync(path.join(HERE, 'jobs', r.job, 'polish',
                                     String(r.after[0].id), '70'));
  if (!r.cached) throw new Error('the render did not build the polished masks');

  // preview against export, on the same frame, with polish on: JPEG decode and
  // h264 both move pixels, so this is a similarity measure like run C's
  const png = path.join(DOCS, 'p-polish-frame20.png');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', path.join(HERE, 'jobs', r.job, 'out.mp4'),
    '-vf', 'select=eq(n\\,20)', '-vframes', '1', png]);
  const raw = execFileSync('ffmpeg', ['-v', 'error', '-i', png, '-f', 'rawvideo',
    '-pix_fmt', 'rgb24', '-'], { maxBuffer: 1 << 28 });
  await page.evaluate(() => window.DV_draw(20)); await sleep(1200);
  const prev = await page.evaluate(() => {
    const c = document.querySelector('#vcv');
    return Array.from(c.getContext('2d').getImageData(0, 0, c.width, c.height).data);
  });
  let near = 0, n = 0;
  for (let p = 0, q = 0; p < raw.length; p += 3, q += 4) {
    const dr = raw[p] - prev[q], dg = raw[p + 1] - prev[q + 1], db = raw[p + 2] - prev[q + 2];
    if (dr * dr + dg * dg + db * db < 900) near++;
    n++;
  }
  r.previewVsExport = { pixels: n, within30: near, pct: +(100 * near / n).toFixed(2) };
  if (r.previewVsExport.pct < 97) {
    throw new Error('preview and export disagree with polish on: '
      + JSON.stringify(r.previewVsExport));
  }
  await page.screenshot({ path: path.join(DOCS, 'p-polish-export.png') });
  return r;
}

/* ---------- G: the jobs/ janitor, against the real jobs directory ---------
 * jobs/ grew to 5.4 GB in two days before this existed. server/jobsgc.py has
 * the policy and server/jobsgc_check.py checks it against fabricated trees in
 * a temp directory; this checks the same rules through the live HTTP API, on
 * the real jobs/ -- including the one thing a unit test cannot show: that a
 * sweep firing in the middle of a track -> render does not eat the clip.
 */
function fakeJob(id, { ageDays = 0, filename = 'fake.mp4', source = 'source.webm',
                       frames = 3, masks = true, render = true } = {}) {
  const d = path.join(HERE, 'jobs', id);
  fs.mkdirSync(path.join(d, 'frames'), { recursive: true });
  for (let i = 0; i < frames; i++) {
    fs.writeFileSync(path.join(d, 'frames', String(i).padStart(4, '0') + '.jpg'),
                     Buffer.alloc(64 << 10));
  }
  if (masks) {
    fs.mkdirSync(path.join(d, 'masks', '1'), { recursive: true });
    fs.writeFileSync(path.join(d, 'masks', '1', '0000.png'), Buffer.alloc(4096));
  }
  if (source) fs.writeFileSync(path.join(d, source), Buffer.alloc(256 << 10));
  if (render) fs.writeFileSync(path.join(d, 'out.mp4'), Buffer.alloc(256 << 10));
  const meta = { job: id, n_frames: frames, w: 16, h: 16, fps: 30, filename };
  if (source) meta.source = source;
  fs.writeFileSync(path.join(d, 'meta.json'), JSON.stringify(meta));
  const t = new Date(Date.now() - ageDays * 86400e3);
  for (const sub of ['frames', 'masks/1', 'masks', '']) {
    const q = path.join(d, sub);
    if (fs.existsSync(q)) fs.utimesSync(q, t, t);
  }
  return d;
}

async function runGC(page) {
  const r = {};
  const has = (p) => fs.existsSync(p);
  const ids = { old: 'gcfake-old', fresh: 'gcfake-fresh', cam: 'gcfake-cam',
                photo: 'gcfake-photo', seq: 'seq-gcfake0000' };
  const wipe = () => Object.values(ids).forEach((id) => fs.rmSync(
    path.join(HERE, 'jobs', id), { recursive: true, force: true }));
  wipe();
  try {
    await page.goto(BASE + '/', { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });

    r.status = await (await page.request.get(`${BASE}/api/gc/status`)).json();
    for (const k of ['budget_mb', 'max_age_days', 'keep_hours', 'usage_mb', 'jobs']) {
      if (typeof r.status[k] !== 'number') throw new Error('gc status has no ' + k);
    }
    r.line = await page.textContent('#gcuse');
    r.barVisible = await page.isVisible('#gcbar');
    if (!r.barVisible || !/^storage: /.test(r.line || '')) {
      throw new Error('the storage line is not showing: ' + JSON.stringify(r.line));
    }

    // fabricate: one stale normal job, one fresh one, one stale camera
    // recording, one stale photo, one stale seq-* rasterise directory
    const age = r.status.max_age_days + 2;
    const d = {};
    d.old = fakeJob(ids.old, { ageDays: age });
    d.fresh = fakeJob(ids.fresh, { ageDays: 0 });
    d.cam = fakeJob(ids.cam, { ageDays: age, filename: 'camera-101500.webm' });
    d.photo = fakeJob(ids.photo, { ageDays: age, filename: 'photo-101500.png',
                                   source: null, frames: 1 });
    d.seq = fakeJob(ids.seq, { ageDays: age, filename: null, source: null,
                               masks: false });

    r.run = await (await page.request.post(`${BASE}/api/gc/run`)).json();
    const ran = r.run.ran;
    r.freedBytes = ran.freed_bytes;

    if (has(d.old)) throw new Error('a job past the age limit survived the sweep');
    if (!has(path.join(d.fresh, 'frames'))) throw new Error('a fresh job was deleted');
    if (!has(path.join(d.cam, 'source.webm')) || !has(path.join(d.cam, 'meta.json'))) {
      throw new Error('the camera recording lost its original');
    }
    if (has(path.join(d.cam, 'frames')) || has(path.join(d.cam, 'masks'))
        || has(path.join(d.cam, 'out.mp4'))) {
      throw new Error('the camera job kept rebuildable data it should have shed');
    }
    if (!has(path.join(d.photo, 'frames', '0000.jpg'))) {
      throw new Error('the photo lost the only copy of itself');
    }
    if (has(d.seq)) throw new Error('an old seq-* directory survived');
    if (!ran.trimmed.includes(ids.cam) || !ran.deleted.includes(ids.old)) {
      throw new Error('the report disagrees with the disk: ' + JSON.stringify(ran));
    }
    const mine = (x) => Object.values(ids).includes(x);
    r.deletedFakes = ran.deleted.filter(mine);
    r.trimmedFakes = ran.trimmed.filter(mine);
    wipe();

    /* --- the race: a sweep every 1.5 s through a whole track -> render.
     * The 48 h keep window is what makes this safe, and this is the proof. */
    await page.goto(BASE + '/', { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
    await page.setInputFiles('#file', CLIP);
    await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
    r.job = await page.evaluate(() => window.DV.job);

    let sweeps = 0, stop = false, sweepErr = null;
    const hammer = (async () => {
      while (!stop) {
        try {
          const q = await page.request.post(`${BASE}/api/gc/run`);
          if (!q.ok()) sweepErr = 'gc/run ' + q.status();
          sweeps++;
        } catch (e) { sweepErr = String(e); }
        await sleep(1500);
      }
    })();

    await page.click('#scope .chip[data-scope="track"]');
    await sleep(700);
    await prompt(page, SUBJECT_A);
    r.trackInfo = await (async () => {
      await page.click('#bTrack');
      return waitText(page, '#tinfo', /tracked|failed/, 300000);
    })();
    await openStep(page, 'st5');
    await page.click('#bExport');
    r.render = await waitText(page, '#rinfo', /rendered|failed/, 300000);
    stop = true; await hammer;
    r.sweepsDuringFlow = sweeps;
    if (sweepErr) throw new Error('a sweep failed mid-flow: ' + sweepErr);
    if (/failed/.test(r.trackInfo)) throw new Error(r.trackInfo);
    if (/failed/.test(r.render)) throw new Error(r.render);
    if (sweeps < 2) throw new Error('no sweep actually fired during the flow');

    const jd = path.join(HERE, 'jobs', r.job);
    for (const want of ['frames', 'masks', 'meta.json', 'out.mp4']) {
      if (!has(path.join(jd, want))) {
        throw new Error(`the live job lost ${want} to a sweep it was using`);
      }
    }
    r.stamped = has(path.join(jd, '.access'));
    if (!r.stamped) throw new Error('no .access stamp was written for a job in use');
    r.probe = ffprobe(path.join(jd, 'out.mp4'));

    // and the button in the panel does the same thing without throwing
    await page.click('#bGC');
    await page.waitForFunction(() => !document.querySelector('#bGC').disabled,
                               { timeout: 60000 });
    r.lineAfter = await page.textContent('#gcuse');
    r.after = await (await page.request.get(`${BASE}/api/gc/status`)).json();
    await page.screenshot({ path: path.join(DOCS, 'g-storage.png') });
    return r;
  } finally {
    wipe();
  }
}


/* ---------- X: the canvas — one clip, four shapes ---------------------------
 * The aspect-ratio control end to end on the server engine: a 9:16 CUTOUT
 * (the dots re-measured on the canvas, so 1080x1920 of real dots), a 9:16
 * OVERLAY (a crop window of that aspect following the tracked subject, with
 * the matched original cut following the identical path), the .dots.gz that
 * comes out carrying the new frame, and the sequence at 9:16.
 *
 * The claim being tested is not "the file is 1080x1920" -- that is one line of
 * ffprobe -- but that the SUBJECT IS STILL IN IT: every assertion below is
 * about where the dots or the crop centre landed relative to the mask the
 * tracker produced.
 */
function bgCensus(file, n, bg) {
  const data = execFileSync('ffmpeg', ['-v', 'error', '-i', file,
    '-vf', `select=eq(n\\,${n})`, '-vframes', '1', '-pix_fmt', 'rgb24',
    '-f', 'rawvideo', '-'], { maxBuffer: 1 << 28 });
  const probe = ffprobe(file);
  const w = +probe.width, h = +probe.height;
  const [br, bgc, bb] = [1, 3, 5].map((i) => parseInt(bg.slice(i, i + 2), 16));
  let n0 = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0, p = 0; y < h; y++) {
    for (let x = 0; x < w; x++, p += 3) {
      const d = Math.abs(data[p] - br) + Math.abs(data[p + 1] - bgc)
        + Math.abs(data[p + 2] - bb);
      if (d < 60) continue;
      n0++;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  return { w, h, lit: n0, pct: +(100 * n0 / (w * h)).toFixed(2),
           box: x1 < 0 ? null : { x0, y0, x1, y1 },
           cx: x1 < 0 ? 0 : (x0 + x1) / 2, cy: y1 < 0 ? 0 : (y0 + y1) / 2 };
}

function contactSheet(src, frames, out, cols) {
  const sel = frames.map((n) => `eq(n\\,${n})`).join('+');
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', src,
    '-vf', `select='${sel}',scale=270:-1,tile=${cols || frames.length}x1`,
    '-frames:v', '1', out]);
  return out;
}

async function runCanvas(page) {
  const r = { steps: [] };
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.evaluate(() => window.DV_limit(2));
  await page.setInputFiles('#file', CLIP);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
  r.job = await page.evaluate(() => window.DV.job);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  await page.click('#scope .chip[data-scope="track"]');
  await sleep(700);
  await prompt(page, SUBJECT_A);
  await page.click('#bTrack');
  r.trackInfo = await waitText(page, '#tinfo', /tracked|failed/, 300000);
  if (/failed/.test(r.trackInfo)) throw new Error(r.trackInfo);

  // the control itself: six shapes, source by default, nothing enabled
  r.presets = await page.evaluate(() => window.DV_canvas.presets());
  r.before = await page.evaluate(() => window.DV_canvas.get());
  if (r.before.target) throw new Error('a fresh clip did not default to its own shape');
  if (!r.presets.some((p) => p.id === '9:16' && p.w === 1080 && p.h === 1920)) {
    throw new Error('no 9:16 preset at 1080×1920');
  }

  /* ---- 9:16, cutout: the dots are measured on the canvas ---------------- */
  await setMode(page, 'dots');
  r.cutout = await page.evaluate(() => window.DV_canvas.set('9:16'));
  if (r.cutout.target.w !== 1080 || r.cutout.target.h !== 1920) {
    throw new Error('9:16 did not give 1080×1920');
  }
  if (r.cutout.clamps) throw new Error('a cutout crop should not be clamped to the source');
  r.path = await page.evaluate(async () => {
    const p = await window.DV_canvas.path();
    return { n: p.n, mode: p.mode, union: p.union };
  });
  r.previewSize = await page.evaluate(() => {
    const c = document.querySelector('#vcv'); return [c.width, c.height];
  });
  if (r.previewSize[0] !== 1080 || r.previewSize[1] !== 1920) {
    throw new Error('the preview is ' + r.previewSize.join('×') + ', not the canvas');
  }
  await page.screenshot({ path: path.join(DOCS, 'x-canvas-916-preview.png') });

  await openStep(page, 'st5');
  await page.evaluate((x) => window.DV_setFormat(x), 'mp4');
  await page.click('#bExport');
  r.cutoutRender = await waitText(page, '#rinfo', /rendered|failed/, 300000);
  if (/failed/.test(r.cutoutRender)) throw new Error(r.cutoutRender);
  const cutFile = path.join(HERE, 'jobs', r.job, 'out.mp4');
  r.cutoutProbe = ffprobe(cutFile);
  if (+r.cutoutProbe.width !== 1080 || +r.cutoutProbe.height !== 1920) {
    throw new Error(`the 9:16 export is ${r.cutoutProbe.width}×${r.cutoutProbe.height}`);
  }
  if (+r.cutoutProbe.nb_read_frames !== r.nFrames) {
    throw new Error(`the 9:16 export has ${r.cutoutProbe.nb_read_frames} frames`);
  }
  // the subject is IN the frame on every sampled frame, and not against an edge
  const bg = await page.evaluate(() => window.DV.bg);
  r.cutoutFrames = {};
  for (const n of [0, r.nFrames >> 1, r.nFrames - 1]) {
    const c = bgCensus(cutFile, n, bg);
    r.cutoutFrames[n] = { lit: c.lit, pct: c.pct, box: c.box,
                          cx: +c.cx.toFixed(0), cy: +c.cy.toFixed(0) };
    if (c.pct < 1) throw new Error(`frame ${n} of the 9:16 cutout is ${c.pct}% dots`);
    if (c.box.x0 <= 0 || c.box.x1 >= c.w - 1) {
      throw new Error(`frame ${n} of the 9:16 cutout runs off the side`);
    }
    if (Math.abs(c.cx - c.w / 2) > c.w * 0.25) {
      throw new Error(`frame ${n} of the 9:16 cutout is off centre by `
        + `${Math.abs(c.cx - c.w / 2).toFixed(0)} px`);
    }
  }
  r.cutoutSheet = contactSheet(cutFile, [0, r.nFrames >> 1, r.nFrames - 1],
                               path.join(DOCS, 'x-canvas-916-cutout.png'));

  /* ---- the dot data carries the canvas --------------------------------- */
  r.dots = await page.evaluate(async () => {
    const { doc } = await window.DV_dots.doc();
    // beyond the frame, not on its last pixel: a cell centre on the last
    // column rounds to exactly w, which both renderers clamp when they draw
    let out = 0, max = 0;
    doc.frames.forEach((f) => f.forEach((xy) => {
      for (let i = 0; i < xy.length; i += 2) {
        if (xy[i] > doc.w || xy[i + 1] > doc.h) out++;
        max = Math.max(max, xy[i + 1]);
      }
    }));
    return { w: doc.w, h: doc.h, frames: doc.frames.length, outside: out, maxY: max };
  });
  if (r.dots.w !== 1080 || r.dots.h !== 1920) {
    throw new Error(`the .dots doc is ${r.dots.w}×${r.dots.h}`);
  }
  if (r.dots.outside) throw new Error(r.dots.outside + ' dots outside the canvas');

  /* ---- 9:16 overlay: the crop follows the subject, and so does the cut -- */
  await openStep(page, 'st3');
  await page.click('#composeui .chip[data-compose="overlay"]');
  await sleep(400);
  await setMode(page, 'ordered');
  r.overlay = await page.evaluate(() => window.DV_canvas.get());
  if (!r.overlay.clamps) throw new Error('an overlay crop must stay inside the source');
  r.overlayNote = await page.evaluate(() => window.DV_canvas.note());
  if (!/crop/.test(r.overlayNote)) throw new Error('the note does not mention the crop');

  /* AUTO decides against the crop that is set: this clip is two seconds long
   * and the athlete never leaves a 9:16 window, so it holds still. Forcing
   * `follow` is what exercises the moving crop — and the assertion below is
   * that it really does move, and moves TO THE SUBJECT. */
  r.autoFraming = r.overlay.framing;
  r.followMode = await page.evaluate(() => window.DV_canvas.framing('follow'));
  if (r.followMode !== 'follow') throw new Error('the framing switch did not take');

  // ground truth: the tracker's own mask centroids, from the server
  const cen = await (await page.request.get(
    `${BASE}/api/jobs/${r.job}/centroids`)).json();
  r.centroidFrames = cen.frames.length;
  if (cen.frames.length !== r.nFrames) {
    throw new Error(`/centroids gave ${cen.frames.length} frames for ${r.nFrames}`);
  }
  r.follow = [];
  for (let n = 0; n < r.nFrames; n += 5) {
    const at = await page.evaluate((i) => window.DV_canvas.at(i), n);
    const c = cen.frames[n];
    if (!c.ok) continue;
    const cx = c.x * at.sw, cy = c.y * at.sh;
    const cropW = at.tw / at.k, cropH = at.th / at.k;
    const inside = Math.abs(cx - at.cx) <= cropW / 2 && Math.abs(cy - at.cy) <= cropH / 2;
    r.follow.push({ n, dx: +(cx - at.cx).toFixed(1), dy: +(cy - at.cy).toFixed(1),
                    cx: +at.cx.toFixed(1), subjX: +cx.toFixed(1),
                    cropW: +cropW.toFixed(0), inside });
    if (!inside) {
      throw new Error(`frame ${n}: the crop lost the subject `
        + `(centroid ${cx.toFixed(0)},${cy.toFixed(0)} vs crop centre `
        + `${at.cx.toFixed(0)},${at.cy.toFixed(0)}, window ${cropW.toFixed(0)}×${cropH.toFixed(0)})`);
    }
  }
  r.followMaxDx = Math.max(...r.follow.map((f) => Math.abs(f.dx)));
  r.followMaxDy = Math.max(...r.follow.map((f) => Math.abs(f.dy)));
  const cxs = r.follow.map((f) => f.cx);
  r.cropTravelPx = +(Math.max(...cxs) - Math.min(...cxs)).toFixed(1);
  const subjTravel = Math.max(...r.follow.map((f) => f.subjX))
    - Math.min(...r.follow.map((f) => f.subjX));
  r.subjectTravelPx = +subjTravel.toFixed(1);
  if (r.cropTravelPx < 20) {
    throw new Error('the following crop never moved (' + r.cropTravelPx + ' px)');
  }
  // smoothed, so it lags — but it is the same journey, not a different one
  const cropW = r.follow[0].cropW;
  if (r.followMaxDx > cropW * 0.4) {
    throw new Error(`the crop is ${r.followMaxDx} px off the subject `
      + `(more than 40% of a ${cropW} px window)`);
  }

  await openStep(page, 'st5');
  await page.check('#cOrig');
  await page.click('#bExport');
  r.overlayRender = await waitText(page, '#rinfo', /original cut|failed/, 300000);
  if (/failed/.test(r.overlayRender)) throw new Error(r.overlayRender);
  const dith = path.join(HERE, 'jobs', r.job, 'out.mp4');
  const orig = path.join(HERE, 'jobs', r.job, 'out.original.mp4');
  r.overlayProbe = ffprobeFull(dith);
  r.originalProbe = ffprobeFull(orig);
  for (const k of ['nb_read_frames', 'width', 'height', 'r_frame_rate']) {
    if (String(r.overlayProbe[k]) !== String(r.originalProbe[k])) {
      throw new Error(`the 9:16 pair disagrees on ${k}: `
        + `${r.overlayProbe[k]} vs ${r.originalProbe[k]}`);
    }
  }
  if (+r.originalProbe.width !== 1080 || +r.originalProbe.height !== 1920) {
    throw new Error(`the matched cut is ${r.originalProbe.width}×${r.originalProbe.height}`);
  }
  r.originalSheet = contactSheet(orig, [0, r.nFrames >> 1, r.nFrames - 1],
                                 path.join(DOCS, 'x-canvas-916-original.png'));
  r.overlaySheet = contactSheet(dith, [0, r.nFrames >> 1, r.nFrames - 1],
                                path.join(DOCS, 'x-canvas-916-overlay.png'));
  await page.uncheck('#cOrig');

  /* ---- the camera ANCHORED to one subject ------------------------------ *
   * A second subject, and the crop told to keep just ONE of them in frame.
   * On this engine the per-frame boxes come from the server's own numpy pass
   * over the masks (GET /centroids?objs=2 is the anchor alone rather than the
   * union), so the assertion below is the tab's geometry checked against the
   * server's measurement, not against itself. */
  await openStep(page, 'st2');
  await page.click('#bAdd');
  await sleep(400);
  await prompt(page, SUBJECT_B);
  r.anchorPlan = await page.evaluate(() => window.DV_subjects.plan());
  if (JSON.stringify(r.anchorPlan.ids) !== '[2]') {
    throw new Error('the second subject should be the plan, is '
                    + JSON.stringify(r.anchorPlan.ids));
  }
  {
    // #tinfo still carries the first run's sentence: wait for it to CHANGE
    const was = (await page.textContent('#tinfo')) || '';
    await page.click('#bTrack');
    for (const t0 = Date.now(); ;) {
      const now = (await page.textContent('#tinfo')) || '';
      if (now !== was && /tracked|failed/.test(now)) { r.trackTwo = now; break; }
      if (Date.now() - t0 > 300000) throw new Error('the second track never finished');
      await sleep(500);
    }
  }
  if (/failed/.test(r.trackTwo)) throw new Error(r.trackTwo);
  r.anchorActive = await page.evaluate(() => window.DV_subjects.active());
  if (JSON.stringify(r.anchorActive) !== '[1,2]') {
    throw new Error('two subjects should be in the picture, are '
                    + JSON.stringify(r.anchorActive));
  }
  await page.evaluate(() => window.DV_canvas.framing('follow'));
  r.anchorAllPath = await page.evaluate(async () =>
    (await window.DV_canvas.path()).centers);
  r.anchorSet = await page.evaluate(() => window.DV_canvas.follow(1));
  if (r.anchorSet !== 1) throw new Error('the anchor did not take: ' + r.anchorSet);
  r.anchorGet = await page.evaluate(() => window.DV_canvas.get());
  if (!r.anchorGet.pickable || r.anchorGet.picker.length !== 3) {
    throw new Error('the picker should offer all + two subjects, offers '
                    + JSON.stringify(r.anchorGet.picker));
  }
  r.anchorPath = await page.evaluate(async () =>
    (await window.DV_canvas.path()).centers);
  if (JSON.stringify(r.anchorPath) === JSON.stringify(r.anchorAllPath)) {
    throw new Error('anchoring the camera to #1 did not change the path');
  }
  await page.screenshot({ path: path.join(DOCS, 'x-canvas-916-follow.png') });

  // the server's own measurement of the anchor, asked for by id
  const one = await (await page.request.get(
    `${BASE}/api/jobs/${r.job}/centroids?objs=1`)).json();
  if (JSON.stringify(one.subjects) !== '["1"]') {
    throw new Error('/centroids?objs=1 answered for ' + JSON.stringify(one.subjects));
  }
  r.anchorFrames = { total: 0, inside: 0, clampedAtEdge: 0, gaps: 0, escaped: 0,
                     worstPx: 0 };
  for (let n = 0; n < r.nFrames; n++) {
    const c = one.frames[n];
    if (!c || !c.ok) { r.anchorFrames.gaps++; continue; }
    const at = await page.evaluate((i) => window.DV_canvas.at(i), n);
    const cw = at.tw / at.k, ch = at.th / at.k;
    if (at.cx - cw / 2 < -0.5 || at.cx + cw / 2 > at.sw + 0.5
        || at.cy - ch / 2 < -0.5 || at.cy + ch / 2 > at.sh + 0.5) {
      throw new Error(`frame ${n}: the crop window left the source`);
    }
    r.anchorFrames.total++;
    const x0 = c.x0 * at.sw, x1 = c.x1 * at.sw;
    const y0 = c.y0 * at.sh, y1 = c.y1 * at.sh;
    const inX = x0 >= at.cx - cw / 2 - 1 && x1 <= at.cx + cw / 2 + 1;
    const inY = y0 >= at.cy - ch / 2 - 1 && y1 <= at.cy + ch / 2 + 1;
    if (inX && inY) { r.anchorFrames.inside++; continue; }
    /* the camera stops at the limits of the picture: a box that cannot be
       held by ANY legal window position is the clamp doing its job */
    const okX = x1 - x0 <= cw
      && Math.max(x1 - cw / 2, cw / 2) <= Math.min(x0 + cw / 2, at.sw - cw / 2) + 1;
    const okY = y1 - y0 <= ch
      && Math.max(y1 - ch / 2, ch / 2) <= Math.min(y0 + ch / 2, at.sh - ch / 2) + 1;
    if ((inX || !okX) && (inY || !okY)) { r.anchorFrames.clampedAtEdge++; continue; }
    r.anchorFrames.escaped++;
    const miss = Math.max(inX ? 0 : Math.max(at.cx - cw / 2 - x0, x1 - at.cx - cw / 2),
                          inY ? 0 : Math.max(at.cy - ch / 2 - y0, y1 - at.cy - ch / 2));
    r.anchorFrames.worstPx = Math.max(r.anchorFrames.worstPx, +miss.toFixed(1));
    throw new Error(`frame ${n}: the anchored crop lost #1 by ${miss.toFixed(1)} px `
      + `(box ${x0.toFixed(0)}..${x1.toFixed(0)} x ${y0.toFixed(0)}..${y1.toFixed(0)}, `
      + `window ${cw.toFixed(0)}x${ch.toFixed(0)} centred ${at.cx.toFixed(0)},${at.cy.toFixed(0)})`);
  }

  /* the map that crosses the wire is the map the tab is drawing */
  r.anchorPayload = await page.evaluate(() => window.DV_canvas.payload());
  const at0 = await page.evaluate(() => window.DV_canvas.at(0));
  if (r.anchorPayload.place.length !== r.nFrames) {
    throw new Error(`the payload carries ${r.anchorPayload.place.length} placements `
                    + `for ${r.nFrames} frames`);
  }
  if (Math.abs(r.anchorPayload.place[0][0] - at0.x0) > 0.01
      || Math.abs(r.anchorPayload.place[0][1] - at0.y0) > 0.01) {
    throw new Error('the payload disagrees with the tab about frame 0');
  }
  /* and the server renders it: the anchored place[] all the way to a file */
  await openStep(page, 'st5');
  await page.click('#bExport');
  r.anchorRender = await waitText(page, '#rinfo', /rendered|failed/, 300000);
  if (/failed/.test(r.anchorRender)) throw new Error(r.anchorRender);
  r.anchorProbe = ffprobeFull(path.join(HERE, 'jobs', r.job, 'out.mp4'));
  if (+r.anchorProbe.width !== 1080 || +r.anchorProbe.height !== 1920) {
    throw new Error(`the anchored render is ${r.anchorProbe.width}×${r.anchorProbe.height}`);
  }

  await page.evaluate(() => window.DV_canvas.follow(null));
  r.anchorBack = await page.evaluate(async () =>
    (await window.DV_canvas.path()).centers);
  if (JSON.stringify(r.anchorBack) !== JSON.stringify(r.anchorAllPath)) {
    throw new Error('“all” did not reproduce the path it had before the anchor');
  }
  await page.evaluate(() => window.DV_subjects.remove(2));
  await sleep(900);
  await page.evaluate(() => window.DV_canvas.framing('auto'));

  /* ---- the sequence has a frame size of its own ------------------------- */
  await page.evaluate(() => window.DV_seq.view('studio'));
  r.seqAdded = await page.evaluate(async () => {
    const cands = window.DV_seq.candidates();
    await window.DV_seq.add(cands[0].id, cands[0].arg);
    return window.DV_seq.strip().length;
  });
  r.seqSource = await page.evaluate(() => window.DV_seq.canvas());
  r.seqCanvas = await page.evaluate(() => window.DV_seq.canvas('9:16'));
  if (r.seqCanvas.w !== 1080 || r.seqCanvas.h !== 1920) {
    throw new Error('the sequence did not take 9:16');
  }
  r.seqDoc = await page.evaluate(async () => {
    const doc = await window.DV_seq.build();
    let out = 0, lit = 0;
    doc.frames.forEach((f) => f.forEach((xy) => {
      lit += xy.length >> 1;
      for (let i = 0; i < xy.length; i += 2) {
        if (xy[i] > doc.w || xy[i + 1] > doc.h) out++;
      }
    }));
    return { w: doc.w, h: doc.h, frames: doc.frames.length, outside: out, dots: lit };
  });
  if (r.seqDoc.w !== 1080 || r.seqDoc.h !== 1920) {
    throw new Error(`the sequence document is ${r.seqDoc.w}×${r.seqDoc.h}`);
  }
  if (r.seqDoc.outside) throw new Error(r.seqDoc.outside + ' sequence dots outside the frame');
  if (!r.seqDoc.dots) throw new Error('the 9:16 sequence has no dots in it');
  await page.evaluate(() => window.DV_seq.view('sequence'));
  await sleep(1500);
  await page.screenshot({ path: path.join(DOCS, 'x-canvas-seq-916.png') });
  await page.evaluate(() => window.DV_seq.view('studio'));

  /* ---- a still, square ------------------------------------------------- */
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 30000 });
  await setMode(page, 'dots');
  r.still = await page.evaluate(() => window.DV_canvas.set('1:1'));
  await sleep(600);
  await openStep(page, 'st5');
  await page.click('#bExport');
  r.stillInfo = await waitText(page, '#rinfo', /PNG|failed/, 120000);
  if (/failed/.test(r.stillInfo)) throw new Error(r.stillInfo);
  const [dl] = await Promise.all([
    page.waitForEvent('download', { timeout: 30000 }),
    page.click('#dl'),
  ]);
  r.stillName = dl.suggestedFilename();
  const png = path.join(DOCS, 'x-canvas-still-1x1.png');
  await dl.saveAs(png);
  r.stillProbe = ffprobe(png);
  if (+r.stillProbe.width !== 1080 || +r.stillProbe.height !== 1080) {
    throw new Error(`the 1:1 still is ${r.stillProbe.width}×${r.stillProbe.height}`);
  }
  if (!/1x1/.test(r.stillName)) throw new Error('the file is called ' + r.stillName);
  await page.screenshot({ path: path.join(DOCS, 'x-canvas-still.png') });

  // and the server refuses a canvas that could not be encoded
  const bad = await page.request.post(`${BASE}/api/jobs/${r.job}/render`, {
    data: { subjects: [], mode: 'ordered',
            canvas: { w: 1081, h: 1920, k: 2, place: [[0, 0]] } },
    timeout: 60000 });
  r.oddStatus = bad.status();
  r.oddDetail = (await bad.json()).detail || '';
  if (bad.status() !== 400 || !/even/.test(r.oddDetail)) {
    throw new Error('an odd canvas size got ' + bad.status() + ' ' + r.oddDetail);
  }
  return r;
}

/* ---- S: per-subject incremental tracking ---------------------------------
 *
 * The claim: one Track run walks the subjects it says it is going to walk and
 * nobody else's masks move. That is checked twice over -- from the outside, by
 * hashing masks/<obj>/ on disk before and after, and from the inside, by
 * logging every POST /track body the page sends.
 */
function maskDirHash(job, obj) {
  const d = path.join(HERE, 'jobs', job, 'masks', String(obj));
  if (!fs.existsSync(d)) return null;
  const names = fs.readdirSync(d).sort();
  const h = crypto.createHash('sha256');
  for (const n of names) { h.update(n); h.update(fs.readFileSync(path.join(d, n))); }
  return { files: names.length, hash: h.digest('hex').slice(0, 16) };
}

async function runSubjects(page) {
  const r = { trackPosts: [] };
  const log = (rq) => {
    if (!/\/track$/.test(rq.url()) || rq.method() !== 'POST') return;
    let b = null;
    try { b = JSON.parse(rq.postData() || 'null'); } catch (e) { b = null; }
    if (!b) return;
    r.trackPosts.push({ only: b.only === undefined ? null : b.only,
                        objects: (b.objects || []).map((o) => o.id) });
  };
  page.on('request', log);
  try {
    await page.goto(BASE + '/', { waitUntil: 'networkidle' });
    await page.waitForFunction(() => window.DV_ready === true, { timeout: 20000 });
    await page.setInputFiles('#file', CLIP);
    await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 90000 });
    const job = await page.evaluate(() => window.DV.job);
    r.job = job;
    await page.click('#scope .chip[data-scope="track"]');
    await sleep(700);

    /* --- 1. one subject, tracked alone -------------------------------- */
    await prompt(page, SUBJECT_A);
    r.ctaFirst = (await page.textContent('#bTrack')).trim();
    const t1 = Date.now();
    await page.click('#bTrack');
    r.trackOne = await waitText(page, '#tinfo', /tracked|failed/, 300000);
    if (/failed/.test(r.trackOne)) throw new Error(r.trackOne);
    r.secondsOne = +((Date.now() - t1) / 1000).toFixed(1);
    r.afterOne = { m1: maskDirHash(job, 1), m2: maskDirHash(job, 2) };
    if (!r.afterOne.m1) throw new Error('subject #1 has no masks on disk');
    if (r.afterOne.m2) throw new Error('subject #2 was tracked and should not have been');
    r.statusOne = await (await page.request.get(`${BASE}/api/jobs/${job}/status`)).json();
    if (JSON.stringify(r.statusOne.tracked) !== '["1"]') {
      throw new Error('status.tracked is ' + JSON.stringify(r.statusOne.tracked));
    }
    r.stateOne = await page.evaluate(() => window.DV_subjects.list());
    await page.screenshot({ path: path.join(AFTER, 'subjects-1-tracked.png') });

    /* --- 2. add a second subject; the button offers only the new one --- */
    await openStep(page, 'st2');
    await page.click('#bAdd');
    await sleep(400);
    await prompt(page, SUBJECT_B);
    r.planNew = await page.evaluate(() => window.DV_subjects.plan());
    if (JSON.stringify(r.planNew.ids) !== '[2]') {
      throw new Error('the plan should be [2], is ' + JSON.stringify(r.planNew.ids));
    }
    if (!/1 new subject/.test(r.planNew.cta)) {
      throw new Error('the CTA should offer the new subject, says ' + r.planNew.cta);
    }
    r.chipsNew = await page.$$eval('#subs .chip', (n) => n.map((e) => ({
      text: e.textContent.replace(/\s+/g, ' ').trim(), state: e.dataset.state })));
    await page.screenshot({ path: path.join(AFTER, 'subjects-2-new.png') });

    const t2 = Date.now();
    await page.click('#bTrack');
    r.trackTwo = await waitText(page, '#tinfo', /tracked|failed/, 300000);
    if (/failed/.test(r.trackTwo)) throw new Error(r.trackTwo);
    r.secondsTwo = +((Date.now() - t2) / 1000).toFixed(1);
    r.afterTwo = { m1: maskDirHash(job, 1), m2: maskDirHash(job, 2) };
    if (!r.afterTwo.m2) throw new Error('subject #2 still has no masks');
    if (r.afterTwo.m1.hash !== r.afterOne.m1.hash) {
      throw new Error(`#1's masks moved: ${r.afterOne.m1.hash} -> ${r.afterTwo.m1.hash}`);
    }
    r.keptNoted = /kept from an earlier run/.test(r.trackTwo);
    if (!r.keptNoted) throw new Error('the run did not say #1 was kept: ' + r.trackTwo);
    await page.screenshot({ path: path.join(AFTER, 'subjects-3-both.png') });

    /* --- 3. edit #2's prompt: it goes stale, #1 does not --------------- */
    await openStep(page, 'st2');
    await page.evaluate(() => { window.DV_subjects.prompt(); window.DV.active = 1; });
    await sleep(400);
    const [bx, by] = await stageXY(page, '#pov', SUBJECT_B.point[0] - 40,
                                   SUBJECT_B.point[1] + 120);
    await page.mouse.click(bx, by);
    await sleep(400);
    r.stateStale = await page.evaluate(() => window.DV_subjects.list());
    const st2 = (r.stateStale.find((x) => x.id === 2) || {}).state;
    const st1 = (r.stateStale.find((x) => x.id === 1) || {}).state;
    if (st2 !== 'stale') throw new Error('#2 should be stale, is ' + st2);
    if (st1 !== 'tracked') throw new Error('#1 should be untouched, is ' + st1);
    r.planStale = await page.evaluate(() => window.DV_subjects.plan());
    if (!/Re-track #2/.test(r.planStale.cta)) {
      throw new Error('the CTA should offer to re-track #2, says ' + r.planStale.cta);
    }
    await page.screenshot({ path: path.join(AFTER, 'subjects-4-stale.png') });

    /* --- 4. the chip menu, and a re-track of one subject --------------- */
    r.menu = await page.evaluate(() => window.DV_subjects.menu(1));
    await page.screenshot({ path: path.join(AFTER, 'subjects-5-menu.png') });
    await page.evaluate(() => window.DV_subjects.closeMenu());
    const t3 = Date.now();
    await page.evaluate(() => window.DV_subjects.track([2]));
    r.trackAgain = await waitText(page, '#tinfo', /tracked|failed/, 300000);
    if (/failed/.test(r.trackAgain)) throw new Error(r.trackAgain);
    r.secondsAgain = +((Date.now() - t3) / 1000).toFixed(1);
    r.afterAgain = { m1: maskDirHash(job, 1), m2: maskDirHash(job, 2) };
    if (r.afterAgain.m1.hash !== r.afterOne.m1.hash) {
      throw new Error('#1 moved on a #2-only re-track');
    }
    if (r.afterAgain.m2.hash === r.afterTwo.m2.hash) {
      throw new Error('#2 was re-tracked from a different prompt and did not change');
    }

    /* --- 5. hide, then remove -- the render follows immediately -------- */
    await page.evaluate(() => window.DV_draw(20)); await sleep(600);
    r.bothActive = await page.evaluate(() => window.DV_subjects.active());
    r.censusBoth = await census(page);
    await page.evaluate(() => window.DV_subjects.hide(1, true));
    await sleep(500);
    await page.evaluate(() => window.DV_draw(20)); await sleep(600);
    r.hiddenActive = await page.evaluate(() => window.DV_subjects.active());
    if (JSON.stringify(r.hiddenActive) !== '[2]') {
      throw new Error('hiding #1 left ' + JSON.stringify(r.hiddenActive));
    }
    r.hiddenMasksKept = !!maskDirHash(job, 1);
    if (!r.hiddenMasksKept) throw new Error('hiding threw the masks away');
    await page.screenshot({ path: path.join(AFTER, 'subjects-6-hidden.png') });
    await page.evaluate(() => window.DV_subjects.hide(1, false));
    await sleep(400);

    await page.evaluate(() => window.DV_subjects.remove(1));
    await sleep(1200);
    r.afterRemove = { m1: maskDirHash(job, 1), m2: maskDirHash(job, 2) };
    if (r.afterRemove.m1) throw new Error('#1 was removed and its masks are still there');
    if (!r.afterRemove.m2) throw new Error('removing #1 took #2 with it');
    r.leftActive = await page.evaluate(() => window.DV_subjects.active());
    r.leftChips = await page.$$eval('#subs .chip', (n) => n.map(
      (e) => e.textContent.replace(/\s+/g, ' ').trim()));
    await page.evaluate(() => window.DV_draw(20)); await sleep(700);
    r.censusLeft = await census(page);
    await page.screenshot({ path: path.join(AFTER, 'subjects-7-removed.png') });

    /* --- 6. and it still exports, with the subject that is left -------- */
    await openStep(page, 'st5');
    await page.click('#bExport');
    r.render = await waitText(page, '#rinfo', /rendered|failed/, 300000);
    if (/failed/.test(r.render)) throw new Error(r.render);
    r.probe = ffprobe(path.join(HERE, 'jobs', job, 'out.mp4'));
    await page.screenshot({ path: path.join(AFTER, 'subjects-8-export.png') });

    /* --- 7. what actually went over the wire --------------------------- */
    r.onlySent = r.trackPosts.map((p) => p.only);
    const want = JSON.stringify([[1], [2], [2]]);
    if (JSON.stringify(r.onlySent.slice(0, 3)) !== want) {
      throw new Error('POST /track only: ' + JSON.stringify(r.onlySent)
        + ' — expected ' + want);
    }

    /* --- 8. what incremental actually costs ---------------------------- *
     * Two subjects walked in ONE run against the same clip at the same
     * quality, timed the same way, so the price of splitting a run is a
     * measurement in the report and not a claim in a comment. */
    await openStep(page, 'st2');
    await page.evaluate(() => { window.DV_subjects.prompt(); });
    await sleep(300);
    await page.click('#bAdd'); await sleep(400);
    await prompt(page, SUBJECT_A);
    const t4 = Date.now();
    await page.evaluate(() => window.DV_subjects.track(
      window.DV.subjects.map((x) => x.id)));
    r.trackBoth = await waitText(page, '#tinfo', /tracked|failed/, 300000);
    if (/failed/.test(r.trackBoth)) throw new Error(r.trackBoth);
    r.secondsBoth = +((Date.now() - t4) / 1000).toFixed(1);
    r.onlySent = r.trackPosts.map((p) => p.only);
    if (r.onlySent.length !== 4 || r.onlySent[3].length !== 2) {
      throw new Error('the joint run sent ' + JSON.stringify(r.onlySent));
    }
    r.runs = await page.evaluate(() => window.DV_subjects.runs());
    r.cost = {
      separately: +(r.runs.filter((x) => x.n === 1)
        .slice(-2).reduce((a, x) => a + x.seconds, 0)).toFixed(2),
      together: (r.runs.find((x) => x.n === 2) || {}).seconds,
    };
    r.costNote = (await page.textContent('#tracknote') || '').replace(/\s+/g, ' ').trim();
  } finally {
    page.off('request', log);
  }
  return r;
}

/* ------------------------------------------------------------------ main */
const browser = await chromium.launch({ headless: true, channel: process.env.DV_CHANNEL || undefined });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2,
                                       acceptDownloads: true });
const page = await ctx.newPage();
page.on('console', (m) => { if (m.type() === 'error') R.consoleErrors.push(m.text()); });
page.on('pageerror', (e) => R.pageErrors.push(String(e)));
page.on('requestfailed', (rq) => {
  const why = (rq.failure() || {}).errorText || '';
  R.requestFailures.push({ url: rq.url(), why });
  if (why !== 'net::ERR_ABORTED') R.consoleErrors.push('requestfailed ' + rq.url() + ' ' + why);
});

/* DV_ONLY=polish,tracked runs a subset — the suite is minutes long and the
 * thing you just changed is usually one of its ten flows. Unset runs all of
 * them, which is what CI and the README numbers mean by "the suite". */
const ONLY = (process.env.DV_ONLY || '').split(',').map((x) => x.trim()).filter(Boolean);
const want = (name) => !ONLY.length || ONLY.includes(name);
R.only = ONLY;
const run = async (name, fn) => { if (want(name)) R.runs[name] = await fn(page); };

try {
  await run('still', runStill);
  await run('stillDots', runStillDots);
  await run('stillSubject', runStillSubject);
  await run('whole', runWhole);
  await run('tracked', runTracked);
  await run('trackedFast', runTrackedFast);
  await run('formats', runFormats);
  await run('trim', runTrim);
  await run('dots', runDotsServer);
  // Everything above this line shares one page: runFormats and runDotsServer
  // read the job the PAGE still has open, so a run that navigates -- this one,
  // and the three below it -- has to come after them.
  await run('original', runOriginal);
  await run('canvas', runCanvas);
  await run('range', runRange);
  await run('lasso', runLasso);
  await run('subjects', runSubjects);
  await run('polish', runPolish);
  await run('gc', runGC);
} catch (e) {
  R.fatal = String(e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e);
} finally {
  await browser.close();
}
console.log(JSON.stringify(R, null, 1));
// The report is committed as evidence, so it must not carry this machine's
// home directory around in it: every absolute path under the checkout is
// rewritten to a repo-relative one on the way out.
/* A DV_ONLY run is a subset, so it writes a subset report. It used to write
 * over docs/verify-report.json, which is committed evidence for the WHOLE
 * suite -- one narrow re-run and the evidence was gone. (verify-web.mjs has
 * had this guard for a while; this is the other half of it.) */
fs.writeFileSync(path.join(DOCS, ONLY.length
  ? `verify-report.${ONLY.join('-')}.json` : 'verify-report.json'),
  JSON.stringify(R, null, 1).split('file://' + HERE + '/').join('')
                             .split(HERE + '/').join(''));
process.exit(R.fatal || R.consoleErrors.length || R.pageErrors.length ? 1 : 0);
