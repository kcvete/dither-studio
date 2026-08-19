/* End-to-end check against a real running server, a real EdgeTAM run and real
 * ffmpeg. No mocks.
 *
 *   node verify.mjs [baseURL] [clip.mp4] [still.jpg]
 *
 * Covers the three flows the app offers:
 *   A  still   -> every algorithm -> client-side PNG download
 *   B  clip    -> whole-frame dither -> MP4
 *   C  clip    -> two tracked subjects -> dots + a pixel mode -> MP4
 *   D  clip    -> one tracked subject at "fast" tracking quality (512 px)
 * Writes screenshots to docs/ and a JSON report to docs/verify-report.json.
 */
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.argv[2] || 'http://127.0.0.1:8765';
const CLIP = process.argv[3] || path.join(HERE, 'sample.mp4');
const STILL = process.argv[4] || CLIP.replace(/\.\w+$/, '.jpg');
const DOCS = path.join(HERE, 'docs');
fs.mkdirSync(DOCS, { recursive: true });

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

try {
  R.runs.still = await runStill(page);
  R.runs.whole = await runWhole(page);
  R.runs.tracked = await runTracked(page);
  R.runs.trackedFast = await runTrackedFast(page);
} catch (e) {
  R.fatal = String(e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e);
} finally {
  await browser.close();
}
console.log(JSON.stringify(R, null, 1));
fs.writeFileSync(path.join(DOCS, 'verify-report.json'), JSON.stringify(R, null, 1));
process.exit(R.fatal || R.consoleErrors.length || R.pageErrors.length ? 1 : 0);
