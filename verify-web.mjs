/* Headless end-to-end check of the BROWSER engine — the free, no-server tier —
 * plus the things that only exist because there are two engines: the auto
 * probe, the manual switch, per-subject prompt frames on both, and the optional
 * bearer-token gate a paid backend would use.
 *
 *   node verify-web.mjs [baseURL] [clip.mp4] [entry-clip.mp4] [still.jpg]
 *
 * The page is the same page either way; `baseURL` only has to serve web/. It
 * happens to be the FastAPI server here, which is also what makes the
 * remote-engine half of this file possible.
 *
 * Flows:
 *   W0  the engine chip: what auto picked, and switching by hand
 *   W1  browser · a still through several algorithms -> PNG download
 *   W2  browser · a clip, whole frame -> WebM
 *   W3  browser · a clip, one tracked subject, frame-0 preview -> dots -> WebM
 *   W4  browser · a polygon mask prompt (the heads_mask graph) -> tracked
 *   W5  browser · two subjects prompted on DIFFERENT frames, mask-area census
 *   R5  server  · the same two-frame prompt, so the feature is checked on both
 *   K   an optional DV_API_KEY server: 401 without the header, 200 with it
 *
 * Screenshots go to docs/, the report to docs/verify-web-report.json.
 * Exits non-zero on any console error, page error, or a failed assertion.
 */
import { chromium } from 'playwright';
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.argv[2] || 'http://127.0.0.1:8765';
const CLIP = process.argv[3] || path.join(HERE, 'sample.mp4');
const ENTRY = process.argv[4] || path.join(HERE, 'docs', 'entry-clip.mp4');
const STILL = process.argv[5] || CLIP.replace(/\.\w+$/, '.jpg');
/* A second clip, for the morph that goes from a subject in one clip to a
 * subject in another. Defaults to the tennis job's source; if it is not there
 * the sequence run still does clip -> static shape. */
const CLIP2 = process.argv[6]
  || path.join(HERE, 'jobs', '0e22cc3753bc', 'source.mp4');
const DOCS = path.join(HERE, 'docs');
fs.mkdirSync(DOCS, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* the parkour athlete, in the reference clip's own 1280x720 pixels */
const SUBJECT_A = { box: [435, 95, 625, 360], point: [545, 205] };
/* docs/entry-clip.mp4 (Mixkit): a static park shot a jogger runs into.
 * The tree is there from frame 0; she is not in the shot until frame 38. */
const TREE = { box: [150, 1, 206, 430], point: [178, 220], frame: 0 };
const JOGGER = { box: [10, 256, 98, 452], point: [52, 318], frame: 48 };
const JOGGER_ENTERS = 38;
/* the tennis player, in the frame of jobs/0e22cc3753bc/source.mp4 — the box is
 * the one that job's own meta.json recorded */
const TENNIS = { box: [406, 152, 831, 719], point: [600, 400], frame: 0 };

const R = {
  base: BASE, clip: CLIP, entryClip: ENTRY, still: STILL,
  consoleErrors: [], pageErrors: [], runs: {}, checks: [],
};

function check(name, ok, detail) {
  R.checks.push({ name, ok: !!ok, detail });
  if (!ok) throw new Error(`${name}: ${detail}`);
}

/** A port nothing is listening on — for the 'unreachable server' path. */
async function freePort() {
  return new Promise((ok) => {
    const s = net.createServer();
    s.listen(0, '127.0.0.1', () => { const p = s.address().port; s.close(() => ok(p)); });
  });
}

/* --------------------------------------------------------------- browser */
const GPU_ARGS = ['--enable-unsafe-webgpu', '--enable-features=Vulkan',
  '--use-angle=metal', '--ignore-gpu-blocklist', '--enable-gpu',
  '--autoplay-policy=no-user-gesture-required'];

/** Headless Chromium first; a real Chrome if it has no WebGPU adapter; the
 *  multi-threaded WASM backend if neither does. The last one is ~6x slower, so
 *  it also shortens the clip — and says so in the report. */
async function pickBrowser() {
  const tries = [
    { how: 'chromium (headless, WebGPU flags)', opts: { headless: true, args: GPU_ARGS } },
    { how: "chrome (channel:'chrome')", opts: { headless: true, channel: 'chrome', args: GPU_ARGS } },
  ];
  for (const t of tries) {
    let b = null;
    try {
      b = await chromium.launch(t.opts);
      const p = await b.newPage();
      // about:blank is not a secure context in every build; ask on the real
      // origin the run will use, or a working adapter reads as missing
      await p.goto(BASE + '/', { waitUntil: 'domcontentloaded' }).catch(() => {});
      const gpu = await p.evaluate(async () => {
        if (!navigator.gpu) return 'no navigator.gpu';
        try { return (await navigator.gpu.requestAdapter()) ? 'ok' : 'no adapter'; }
        catch (e) { return 'error: ' + e.message; }
      });
      await p.close();
      if (gpu === 'ok') return { browser: b, how: t.how, ep: 'webgpu', gpu, seconds: 5 };
      await b.close();
    } catch (e) {
      if (b) await b.close().catch(() => {});
    }
  }
  const b = await chromium.launch({ headless: true, args: GPU_ARGS });
  return { browser: b, how: 'chromium, WASM backend (no WebGPU adapter anywhere)',
           ep: 'wasm', gpu: 'none', seconds: 2 };
}

let BR;   // {browser, how, ep, seconds}

/* One test deliberately points the page at a port nothing is listening on, to
 * prove the fallback. The connection failure that produces is the assertion,
 * not a defect, so it is excused by URL — and only that URL. */
let EXPECTED = null;
const expected = (t) => EXPECTED && t.includes(EXPECTED);

async function newPage(pref, opts = {}) {
  const br = opts.browser || BR.browser;
  const ctx = await br.newContext({
    viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2,
    acceptDownloads: true,
    ...(opts.permissions ? { permissions: opts.permissions } : {}),
  });
  await ctx.addInitScript((p) => {
    try { localStorage.setItem('dither-studio.engine', JSON.stringify(p)); }
    catch (e) { /* storage blocked */ }
  }, pref);
  const page = await ctx.newPage();
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const t = m.text();
    // onnxruntime routes its own WARNING lines through console.error; sessions
    // are opened with logSeverityLevel 3 so these should not appear at all, but
    // do not let a future ORT build fail the run over a warning.
    if (/\[W:onnxruntime/.test(t)) return;
    if (expected(t) || /ERR_CONNECTION_REFUSED/.test(t) && EXPECTED) return;
    R.consoleErrors.push(t);
  });
  page.on('pageerror', (e) => R.pageErrors.push(String(e)));
  page.on('requestfailed', (rq) => {
    const why = (rq.failure() || {}).errorText || '';
    if (why !== 'net::ERR_ABORTED' && !expected(rq.url())) {
      R.consoleErrors.push('requestfailed ' + rq.url() + ' ' + why);
    }
  });
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => window.DV_ready === true, { timeout: 60000 });
  return { ctx, page };
}

const browserPref = () => ({ mode: 'browser', ep: BR.ep, url: '', key: '' });

/* ------------------------------------------------------------- helpers */
const openStep = async (page, id) => {
  if (await page.getAttribute(`#${id}`, 'data-open') !== '1') {
    await page.click(`#${id} .sh`); await sleep(250);
  }
};
const setMode = async (page, m) => {
  await openStep(page, 'st3');
  await page.click(`#modes .chip[data-mode="${m}"]`);
  await sleep(700);
};
async function waitText(page, sel, re, timeout) {
  const t0 = Date.now();
  for (;;) {
    const t = await page.textContent(sel).catch(() => '');
    if (t && re.test(t)) return t;
    if (Date.now() - t0 > timeout) throw new Error(`timeout on ${sel} (last: ${t})`);
    await sleep(500);
  }
}
const stageXY = (page, x, y) => page.evaluate(([fx, fy]) => {
  const el = document.querySelector('#pov'), r = el.getBoundingClientRect();
  return [r.left + (fx / el.width) * r.width, r.top + (fy / el.height) * r.height];
}, [x, y]);

async function promptBoxPoint(page, s) {
  const [x0, y0] = await stageXY(page, s.box[0], s.box[1]);
  const [x1, y1] = await stageXY(page, s.box[2], s.box[3]);
  await page.mouse.move(x0, y0); await page.mouse.down();
  await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 6 });
  await page.mouse.move(x1, y1, { steps: 6 }); await page.mouse.up();
  const [px, py] = await stageXY(page, s.point[0], s.point[1]);
  await page.mouse.click(px, py);
}

const census = (page) => page.evaluate(() => {
  const c = document.querySelector('#vcv');
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  const m = new Set();
  for (let i = 0; i < d.length; i += 4) m.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
  return { w: c.width, h: c.height, distinctColours: m.size };
});

/** Per-subject binary mask area on every frame, read straight off the engine. */
const maskAreas = (page) => page.evaluate(async () => {
  const out = {};
  const cv = document.createElement('canvas');
  const g = cv.getContext('2d', { willReadFrequently: true });
  for (const s of window.DV.subjects) {
    const a = [];
    for (let n = 0; n < window.DV.nFrames; n++) {
      const bmp = await window.DV.engine.mask(s.id, n);
      cv.width = bmp.width; cv.height = bmp.height;
      g.clearRect(0, 0, cv.width, cv.height); g.drawImage(bmp, 0, 0);
      const d = g.getImageData(0, 0, cv.width, cv.height).data;
      let n2 = 0;
      for (let i = 0; i < d.length; i += 4) if (d[i] > 127) n2++;
      a.push(n2);
      if (bmp.close) bmp.close();
    }
    out[s.id] = a;
  }
  return out;
});

async function loadClip(page, file, seconds) {
  await openStep(page, 'st1');
  if (seconds) {
    // #sSec lives inside #vidopts, which take() only unhides once it knows the
    // file is a video — so it is legitimately hidden at this point
    await page.$eval('#sSec', (el, v) => {
      el.value = String(v);
      el.dispatchEvent(new Event('input', { bubbles: true }));
    }, seconds);
  }
  await page.setInputFiles('#file', file);
  await page.waitForFunction(() => window.DV.kind === 'video', { timeout: 300000 });
  await sleep(500);
}

async function saveDownload(page, to) {
  const [d] = await Promise.all([
    page.waitForEvent('download', { timeout: 30000 }).catch(() => null),
    page.click('#dl'),
  ]);
  if (!d) return null;
  await d.saveAs(to);
  return fs.statSync(to).size;
}

function probe(file) {
  const out = execFileSync('ffprobe', ['-v', 'error', '-count_frames',
    '-select_streams', 'v:0', '-show_entries',
    'stream=nb_read_frames,width,height,r_frame_rate,codec_name,pix_fmt:stream_tags=alpha_mode',
    '-of', 'json', file]);
  return JSON.parse(out).streams[0];
}

/* alpha census of one decoded frame — the only honest way to ask whether a
 * container really kept the transparency it claims */
function alphaCensus(file, n, extraIn) {
  const args = ['-v', 'error'].concat(extraIn || [],
    ['-i', file, '-vf', `select=eq(n\\,${n})`, '-vframes', '1',
     '-pix_fmt', 'rgba', '-f', 'rawvideo', '-']);
  const data = execFileSync('ffmpeg', args, { maxBuffer: 1 << 28 });
  // lossy codecs round the alpha plane, so this is a threshold, not equality
  let zero = 0, full = 0;
  for (let p = 3; p < data.length; p += 4) {
    if (data[p] < 16) zero++; else if (data[p] > 200) full++;
  }
  const px = data.length / 4;
  return { pixels: px, transparentPct: +(100 * zero / px).toFixed(1),
           opaquePct: +(100 * full / px).toFixed(1) };
}

/* ============================================ W0: the engine chip and switch */
async function runEngineChip() {
  const r = {};
  const { ctx, page } = await newPage({ mode: 'auto', url: '', key: '' });
  r.autoChip = (await page.textContent('#engName')).trim();
  r.auto = await page.evaluate(() => window.DV_engine());
  check('auto probe finds the local server', r.auto.id === 'remote',
        'auto picked ' + r.auto.id);
  await page.screenshot({ path: path.join(DOCS, 'w-engine-auto.png') });

  await page.click('#engine'); await sleep(300);
  r.popoverVisible = await page.isVisible('#engpop');
  await page.screenshot({ path: path.join(DOCS, 'w-engine-popover.png') });

  await page.click('#engpop .opt[data-eng="browser"]');
  await page.waitForFunction(() => window.DV_engine().id === 'browser', { timeout: 30000 });
  r.afterSwitch = await page.evaluate(() => window.DV_engine());
  r.browserChip = (await page.textContent('#engName')).trim();
  check('manual switch to the browser engine', r.afterSwitch.id === 'browser');
  check('browser engine reports the mask-prompt graph',
        r.afterSwitch.supports.maskPrompt === true,
        'has_mask_prompt is false — re-run ./setup.sh');
  check('browser engine exports WebM',
        r.afterSwitch.supports.exportExt === 'webm');
  check('the chip fits without truncating',
        await page.evaluate(() => {
          const el = document.querySelector('#engName');
          return el.scrollWidth <= el.clientWidth + 1;
        }), r.browserChip);

  // a URL that is not a Dither Studio server must fall back, loudly
  await page.click('#engine'); await sleep(200);
  await page.click('#engpop .opt[data-eng="custom"]');
  const dead = await freePort();          // nothing is listening there
  r.deadPort = dead;
  EXPECTED = `127.0.0.1:${dead}`;
  await page.fill('#engUrl', `http://127.0.0.1:${dead}`);
  await page.click('#engGo');
  r.badCustom = await waitText(page, '#engstat', /could not reach|unreachable|no answer/, 20000);
  await page.screenshot({ path: path.join(DOCS, 'w-engine-custom.png') });
  r.afterBadCustom = await page.evaluate(() => window.DV_engine());
  check('an unreachable custom URL falls back to the browser',
        r.afterBadCustom.id === 'browser');
  await ctx.close();
  EXPECTED = null;
  return r;
}

/* ================================================= W1: a still, browser only */
async function runStill() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  check('browser engine is live for the still',
        (await page.evaluate(() => window.DV_engine())).id === 'browser');
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 30000 });
  await sleep(700);
  r.source = await page.textContent('#upstat');
  r.modes = {};
  for (const m of ['bluenoise', 'ordered', 'halftone', 'whitenoise', 'errordiff', 'riemersma']) {
    await setMode(page, m);
    r.modes[m] = await census(page);
    check(`still · ${m} renders`, r.modes[m].distinctColours >= 2,
          JSON.stringify(r.modes[m]));
  }
  await setMode(page, 'bluenoise');
  await openStep(page, 'st4');
  await page.click('#pals .chip:nth-child(6)');        // Game Boy DMG
  await sleep(600);
  r.gameboy = await census(page);
  await page.screenshot({ path: path.join(DOCS, 'w-still.png') });

  await openStep(page, 'st5');
  await page.click('#bExport');
  r.export = await waitText(page, '#rinfo', /PNG|failed/, 60000);
  check('still exports a PNG', !/failed/.test(r.export), r.export);
  r.bytes = await saveDownload(page, path.join(DOCS, 'w-still-export.png'));
  r.probe = probe(path.join(DOCS, 'w-still-export.png'));
  await ctx.close();
  return r;
}

/* ============================================ W2: a clip, whole frame, in-tab */
async function runWhole() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  const t0 = Date.now();
  await loadClip(page, CLIP, BR.seconds);
  r.decodeSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  r.source = await page.textContent('#upstat');
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  check('the clip decoded in the tab', r.nFrames > 10, 'nFrames=' + r.nFrames);

  await setMode(page, 'ordered');
  await page.click('#mxui .chip[data-mx="8"]'); await sleep(500);
  await openStep(page, 'st4');
  await page.click('#pals .chip:nth-child(6)'); await sleep(600);
  r.preview = await census(page);
  await page.evaluate(() => window.DV_draw(10)); await sleep(400);
  r.previewFrame10 = await census(page);
  await page.screenshot({ path: path.join(DOCS, 'w-whole.png') });

  await openStep(page, 'st5');
  const t1 = Date.now();
  await page.click('#bExport');
  r.render = await waitText(page, '#rinfo', /rendered|failed/, 600000);
  check('whole-frame clip exports', !/failed/.test(r.render), r.render);
  r.renderSeconds = +((Date.now() - t1) / 1000).toFixed(1);
  const out = path.join(DOCS, 'w-whole-export.webm');
  r.bytes = await saveDownload(page, out);
  r.probe = probe(out);
  check('the WebM has every frame',
        +r.probe.nb_read_frames === r.nFrames,
        `${r.probe.nb_read_frames} != ${r.nFrames}`);
  check('the WebM keeps the clip resolution', r.probe.width === 1280);
  await ctx.close();
  return r;
}

/* ======================== W3: a clip, one tracked subject, preview then track */
async function runTracked() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  await loadClip(page, CLIP, BR.seconds);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
  await promptBoxPoint(page, SUBJECT_A);
  r.chips = await page.$$eval('#subs .chip', (n) => n.map((e) => e.textContent));
  check('the subject chip records its prompt frame', /@ 0/.test(r.chips[0]), r.chips[0]);
  r.qualityChips = await page.$$eval('#tq .chip', (n) => n.map((e) => e.textContent.trim()));
  check('the browser engine offers the resolution it shipped models for',
        r.qualityChips.length === 1, JSON.stringify(r.qualityChips));

  let t0 = Date.now();
  await page.click('#bPrev');
  r.previewText = await waitText(page, '#pvinfo', /subject|failed/, 240000);
  r.previewSeconds = +((Date.now() - t0) / 1000).toFixed(2);
  check('frame-0 preview runs in the tab', !/failed/.test(r.previewText), r.previewText);
  await page.screenshot({ path: path.join(DOCS, 'w-preview.png') });

  t0 = Date.now();
  await page.click('#bTrack');
  r.trackText = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  r.trackSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  check('the clip tracks in the tab', !/failed/.test(r.trackText), r.trackText);
  r.trackFps = +(r.trackText.match(/\(([\d.]+) fps\)/) || [])[1];

  await setMode(page, 'dots');
  await page.evaluate(() => window.DV_draw(10)); await sleep(600);
  r.dots = await census(page);
  r.dotsFps = await page.textContent('#fps');
  check('the dots look draws dots', /\d+ dots/.test(r.dotsFps), r.dotsFps);
  await page.screenshot({ path: path.join(DOCS, 'w-dots.png') });

  await openStep(page, 'st5');
  t0 = Date.now();
  await page.click('#bExport');
  r.render = await waitText(page, '#rinfo', /rendered|failed/, 600000);
  check('the tracked clip exports', !/failed/.test(r.render), r.render);
  r.renderSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  const out = path.join(DOCS, 'w-dots-export.webm');
  r.bytes = await saveDownload(page, out);
  r.probe = probe(out);
  await page.screenshot({ path: path.join(DOCS, 'w-export.png') });

  // --- the other containers the tab can write, and the two it cannot
  r.offered = await page.evaluate(() => window.DV_formats());
  r.unavailable = r.offered.filter((f) => !f.available).map((f) => f.id);
  check('the browser engine says plainly that MP4 and ProRes need the server',
        r.unavailable.includes('mp4') && r.unavailable.includes('prores'),
        JSON.stringify(r.unavailable));
  check('every unavailable format carries a reason',
        r.offered.filter((f) => !f.available).every((f) => f.note && f.note.length > 10),
        JSON.stringify(r.offered));
  r.formats = {};
  for (const [id, ext] of [['gif', 'gif'], ['webm-alpha', 'webm']]) {
    const t = Date.now();
    await page.evaluate((x) => window.DV_setFormat(x), id);
    await page.click('#bExport');
    const txt = await waitText(page, '#rinfo', /rendered|failed/, 900000);
    check(`browser · ${id} exports`, !/failed/.test(txt), txt);
    const file = path.join(DOCS, `w-${id}-export.${ext}`);
    const bytes = await saveDownload(page, file);
    r.formats[id] = { seconds: +((Date.now() - t) / 1000).toFixed(1), bytes,
                      probe: probe(file), info: txt.trim() };
  }
  check('browser · the GIF is a GIF and loops',
        r.formats.gif.probe.codec_name === 'gif',
        JSON.stringify(r.formats.gif.probe));
  const gifLoop = fs.readFileSync(path.join(DOCS, 'w-gif-export.gif'))
    .includes(Buffer.from('NETSCAPE2.0'));
  check('browser · the GIF carries the loop-forever extension', gifLoop);
  const wa = r.formats['webm-alpha'].probe;
  r.formats['webm-alpha'].alphaTag = (wa.tags || {}).alpha_mode || wa['tags:alpha_mode'] || null;
  check('browser · the alpha WebM declares an alpha channel',
        r.formats['webm-alpha'].alphaTag === '1', JSON.stringify(wa));
  // the alpha plane only decodes through the matching libvpx decoder — VP8 and
  // VP9 are different decoders and each rejects the other's bitstream
  r.formats['webm-alpha'].census = alphaCensus(
    path.join(DOCS, 'w-webm-alpha-export.webm'), 10,
    ['-c:v', wa.codec_name === 'vp9' ? 'libvpx-vp9' : 'libvpx']);
  check('browser · the alpha WebM really is mostly transparent',
        r.formats['webm-alpha'].census.transparentPct > 50,
        JSON.stringify(r.formats['webm-alpha'].census));
  await ctx.close();
  return r;
}

/* ======================= W4: a polygon mask prompt through the heads_mask graph */
async function runLasso() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  await loadClip(page, CLIP, BR.seconds);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
  await page.click('#ptool .chip[data-tool="poly"]');
  const poly = await page.evaluate(() => {
    const c = document.querySelector('#pov'), b = c.getBoundingClientRect();
    const P = [[597, 111], [580, 114], [573, 140], [568, 144], [527, 145], [473, 159],
      [447, 158], [444, 162], [447, 175], [470, 167], [524, 164], [493, 218],
      [494, 236], [507, 263], [526, 279], [520, 296], [520, 350], [568, 357],
      [544, 338], [564, 271], [552, 238], [552, 216], [560, 198], [585, 172],
      [584, 158], [607, 135], [609, 120]];
    return { rect: { x: b.x, y: b.y, w: b.width, h: b.height }, pts: P, W: c.width, H: c.height };
  });
  for (const [x, y] of poly.pts) {
    await page.mouse.click(poly.rect.x + (x / poly.W) * poly.rect.w,
                           poly.rect.y + (y / poly.H) * poly.rect.h);
  }
  await page.keyboard.press('Enter');
  r.shapes = await page.evaluate(() => window.DV.subjects[0].paths.map((p) => p.pts.length));
  check('the polygon was captured', r.shapes[0] === 27, JSON.stringify(r.shapes));

  let t0 = Date.now();
  await page.click('#bPrev');
  r.previewText = await waitText(page, '#pvinfo', /subject|failed/, 240000);
  r.previewSeconds = +((Date.now() - t0) / 1000).toFixed(2);
  check('a mask prompt previews in the tab', !/failed/.test(r.previewText), r.previewText);
  check('the mask prompt was NOT downgraded to a box',
        !/approximated/.test(r.previewText), r.previewText);
  await page.screenshot({ path: path.join(DOCS, 'w-lasso-preview.png') });

  t0 = Date.now();
  await page.click('#bTrack');
  r.trackText = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  r.trackSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  check('a mask prompt tracks the clip', !/failed/.test(r.trackText), r.trackText);
  r.areas = (await maskAreas(page))[1];
  const nonEmpty = r.areas.filter((a) => a > 500).length;
  r.nonEmptyFrames = nonEmpty;
  check('the mask-prompted subject survives the clip',
        nonEmpty >= r.areas.length * 0.9,
        `${nonEmpty}/${r.areas.length} frames non-empty`);
  await setMode(page, 'dots');
  await page.evaluate(() => window.DV_draw(10)); await sleep(600);
  await page.screenshot({ path: path.join(DOCS, 'w-lasso-tracked.png') });
  await ctx.close();
  return r;
}

/* ============ W5 / R5: two subjects prompted on two different frames ========= */
async function runEntry(engineId) {
  const r = { engine: engineId };
  const pref = engineId === 'browser' ? browserPref()
    : { mode: 'local', url: '', key: '' };
  const { ctx, page } = await newPage(pref);
  check(`${engineId} · engine is live`,
        (await page.evaluate(() => window.DV_engine())).id
          === (engineId === 'browser' ? 'browser' : 'remote'));
  await loadClip(page, ENTRY, 5);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);

  // subject 1: the tree, which is in the shot from the first frame
  await promptBoxPoint(page, TREE);
  // subject 2: the jogger, who is not. Scrub to where she is, then prompt.
  await page.click('#bAdd');
  await page.locator('#sPF').fill(String(JOGGER.frame));
  await page.locator('#sPF').dispatchEvent('input');
  await sleep(900);
  r.scrubbedTo = await page.evaluate(() => window.DV.promptFrame);
  await promptBoxPoint(page, JOGGER);

  r.chips = await page.$$eval('#subs .chip', (n) => n.map((e) => e.textContent.trim()));
  check(`${engineId} · chips carry two different prompt frames`,
        /@ 0/.test(r.chips[0]) && new RegExp('@ ' + JOGGER.frame).test(r.chips[1]),
        JSON.stringify(r.chips));
  r.offFrameHint = (await page.textContent('#offframe') || '').trim();
  check(`${engineId} · the off-frame subject is announced`,
        /#1 prompted @ 0/.test(r.offFrameHint), r.offFrameHint);
  r.subject1MarksHidden = await page.evaluate(() =>
    // subject #1's marks belong to frame 0 and must not paint on frame 48
    window.DV.subjects[0].promptFrame === 0 && window.DV.promptFrame === 48);
  check(`${engineId} · marks are scoped to their own frame`, r.subject1MarksHidden);
  await page.screenshot({ path: path.join(DOCS, `w-entry-prompts-${engineId}.png`) });

  // the preview on frame 48 must cover the jogger only
  await page.click('#bPrev');
  r.previewText = await waitText(page, '#pvinfo', /subject|failed/, 240000);
  check(`${engineId} · preview covers only this frame's subject`,
        /1 subject/.test(r.previewText), r.previewText);
  await page.screenshot({ path: path.join(DOCS, `w-entry-preview-${engineId}.png`) });

  const t0 = Date.now();
  await page.click('#bTrack');
  r.trackText = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  r.trackSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  check(`${engineId} · both subjects track`, !/failed/.test(r.trackText), r.trackText);
  check(`${engineId} · the two prompt frames are reported`,
        /2 different frames/.test(r.trackText), r.trackText);

  const areas = await maskAreas(page);
  const A = (id) => areas[id] || [];
  const firstOn = (a) => a.findIndex((v) => v > 500);
  const emptyCount = (a) => a.filter((v) => v <= 500).length;
  r.tree = { firstOn: firstOn(A(1)), empty: emptyCount(A(1)), n: A(1).length };
  r.jogger = { firstOn: firstOn(A(2)), empty: emptyCount(A(2)), n: A(2).length };
  r.samples = {
    tree: [0, 10, 30, 48, 100, 140].map((i) => A(1)[i]),
    jogger: [0, 10, 30, 37, 38, 40, 48, 100, 140].map((i) => A(2)[i]),
  };

  check(`${engineId} · the subject present from the start is never empty`,
        r.tree.empty === 0, JSON.stringify(r.tree));
  check(`${engineId} · the subject that enters mid-clip is empty before it does`,
        r.jogger.firstOn >= JOGGER_ENTERS - 2 && r.jogger.firstOn <= JOGGER_ENTERS + 3,
        `first non-empty frame ${r.jogger.firstOn}, she enters at ${JOGGER_ENTERS}`);
  check(`${engineId} · and present afterwards`,
        emptyCount(A(2).slice(JOGGER_ENTERS + 5)) <= 3,
        JSON.stringify(r.samples.jogger));

  // the renderer must show nothing for an absent subject and dots once it is there
  await setMode(page, 'dots');
  await page.evaluate(() => window.DV_draw(10)); await sleep(700);
  r.dotsBefore = await page.textContent('#fps');
  await page.screenshot({ path: path.join(DOCS, `w-entry-f10-${engineId}.png`) });
  await page.evaluate(() => window.DV_draw(100)); await sleep(700);
  r.dotsAfter = await page.textContent('#fps');
  await page.screenshot({ path: path.join(DOCS, `w-entry-f100-${engineId}.png`) });
  const n = (s) => +((s.match(/(\d+) dots/) || [])[1] || 0);
  r.dotsBeforeN = n(r.dotsBefore); r.dotsAfterN = n(r.dotsAfter);
  check(`${engineId} · dots pop in with the subject, not before`,
        r.dotsAfterN > r.dotsBeforeN,
        `frame 10: ${r.dotsBeforeN} dots, frame 100: ${r.dotsAfterN}`);
  await ctx.close();
  return r;
}

/* ============ W6: the camera, the trim bar, and both engines behind them =====
 * Chromium's fake capture device stands in for a webcam: --use-fake-device
 * generates a moving test pattern, --use-fake-ui answers the permission prompt.
 * The recording that comes out is a real MediaRecorder WebM and goes down
 * exactly the path a dropped file does, which is the point of the test.
 */
async function runCamera(engineId, deep) {
  const r = { engine: engineId };
  // channel:'chromium' is the new headless mode. The old headless shell has no
  // media-capture stack at all -- getUserMedia there answers NotSupportedError
  // however many fake-device flags you pass it.
  const br = await chromium.launch({
    headless: true, channel: 'chromium',
    args: GPU_ARGS.concat(['--use-fake-device-for-media-capture',
                           '--use-fake-ui-for-media-capture']),
  });
  try {
    const pref = engineId === 'browser' ? browserPref() : { mode: 'local', url: '', key: '' };
    const { ctx, page } = await newPage(pref, { browser: br, permissions: ['camera'] });
    r.engineId = (await page.evaluate(() => window.DV_engine())).id;
    check(`${engineId} · engine is live for the camera run`,
          r.engineId === (engineId === 'browser' ? 'browser' : 'remote'), r.engineId);

    await page.click('#bCam');
    await page.waitForFunction(() => window.DV_camera.state().live, null, { timeout: 20000 });
    r.cameraTrack = await page.evaluate(() => window.DV_camera.state().track);
    check(`${engineId} · the camera is live at the resolution it asked for`,
          r.cameraTrack && r.cameraTrack.width >= 640,
          JSON.stringify(r.cameraTrack));
    // --- a photo first: the same feed, straight into the still flow
    await page.click('#bSnap');
    await page.waitForFunction(() => window.DV.kind === 'image', null, { timeout: 60000 });
    await sleep(600);
    r.photo = await page.evaluate(() => ({
      photo: window.DV_camera.state().photo, natW: window.DV.natW,
      natH: window.DV.natH, source: document.querySelector('#upstat').textContent,
    }));
    check(`${engineId} · a photo comes back at the camera's resolution`,
          r.photo.natW >= 640 && r.photo.natW === (r.photo.photo || {}).w,
          JSON.stringify(r.photo));
    check(`${engineId} · the photo closed the camera`,
          !(await page.evaluate(() => window.DV_camera.state().live)));
    await setMode(page, 'bluenoise');
    await openStep(page, 'st4');
    await page.click('#pals .chip:nth-child(2)');            // Sage
    await sleep(600);
    r.photoCensus = await census(page);
    check(`${engineId} · the photo dithers`, r.photoCensus.distinctColours === 2,
          JSON.stringify(r.photoCensus));
    await page.screenshot({ path: path.join(DOCS, `w-camera-photo-${engineId}.png`) });
    await openStep(page, 'st5');
    await page.click('#bExport');
    r.photoExport = await waitText(page, '#rinfo', /PNG|failed/, 60000);
    check(`${engineId} · the photo exports a PNG`, !/failed/.test(r.photoExport),
          r.photoExport);
    const png = path.join(DOCS, `w-camera-photo-${engineId}-export.png`);
    r.photoBytes = await saveDownload(page, png);
    r.photoProbe = probe(png);
    check(`${engineId} · the exported PNG is the camera's own resolution`,
          r.photoProbe.width === r.photo.natW && r.photoProbe.height === r.photo.natH,
          JSON.stringify(r.photoProbe));

    // --- and now the recording
    await openStep(page, 'st1');
    await page.click('#bCam');
    await page.waitForFunction(() => window.DV_camera.state().live, null, { timeout: 20000 });
    await page.click('#bRec');
    await page.waitForFunction(() => window.DV_camera.state().recording, null, { timeout: 15000 });
    await sleep(600);
    await page.screenshot({ path: path.join(DOCS, `w-camera-live-${engineId}.png`) });
    await sleep(2600);                                  // ~3.2 s of recording
    await page.click('#bRec');
    await page.waitForFunction(() => window.DV.kind === 'video', null, { timeout: 180000 });
    await sleep(400);
    r.recordedS = await page.evaluate(() => window.DV.recordedS);
    r.wholeFrames = await page.evaluate(() => window.DV.nFrames);
    check(`${engineId} · about three seconds were recorded`,
          r.recordedS >= 2.5 && r.recordedS <= 4.5, String(r.recordedS));
    check(`${engineId} · the recording became a clip`, r.wholeFrames > 50,
          String(r.wholeFrames));

    // the filmstrip has to have actually drawn something
    await page.waitForFunction(() => window.DV.srcDuration > 0, null, { timeout: 60000 });
    r.srcDuration = await page.evaluate(() => window.DV.srcDuration);
    await sleep(2500);
    r.strip = await page.evaluate(() => {
      const c = document.querySelector('#stripcv');
      const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
      const seen = new Set();
      for (let i = 0; i < d.length; i += 4) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
      return { colours: seen.size, w: c.width, h: c.height };
    });
    check(`${engineId} · the filmstrip has thumbnails on it`, r.strip.colours > 8,
          JSON.stringify(r.strip));
    await page.screenshot({ path: path.join(DOCS, `w-camera-trim-${engineId}.png`) });

    // trim to the middle two seconds and re-open. Loading a clip hands the
    // panel to step 2, so the trim bar is behind its own header again.
    await openStep(page, 'st1');
    const mid = r.srcDuration / 2;
    r.range = await page.evaluate(([a, b]) => window.DV_trim(a, b),
                                  [Math.max(0, mid - 1), mid + 1]);
    await page.click('#bTrim');
    await page.waitForFunction((n) => window.DV.nFrames !== n && window.DV.kind === 'video',
                               r.wholeFrames, { timeout: 180000 });
    await sleep(400);
    r.trimmedFrames = await page.evaluate(() => window.DV.nFrames);
    r.upstat = (await page.textContent('#upstat')).trim();
    check(`${engineId} · the trim really shortened the clip`,
          r.trimmedFrames >= 50 && r.trimmedFrames <= 66,
          `${r.trimmedFrames} frames for a 2 s range (whole clip was ${r.wholeFrames})`);
    check(`${engineId} · the trim is stated in the source line`,
          /trimmed/.test(r.upstat), r.upstat);

    if (deep) {
      // one subject on the test pattern, tracked and rendered end to end
      await page.click('#scope .chip[data-scope="track"]'); await sleep(700);
      const box = await page.evaluate(() => {
        const c = document.querySelector('#pov');
        return { w: c.width, h: c.height };
      });
      await promptBoxPoint(page, {
        box: [box.w * 0.3, box.h * 0.3, box.w * 0.7, box.h * 0.75],
        point: [box.w * 0.5, box.h * 0.5],
      });
      const t0 = Date.now();
      await page.click('#bTrack');
      r.trackText = await waitText(page, '#tinfo', /tracked|failed/, 900000);
      r.trackSeconds = +((Date.now() - t0) / 1000).toFixed(1);
      check(`${engineId} · a camera clip tracks`, !/failed/.test(r.trackText), r.trackText);
      await setMode(page, 'dots');
      await page.evaluate(() => window.DV_draw(10)); await sleep(500);
      r.dots = await page.textContent('#fps');
      await openStep(page, 'st5');
      await page.click('#bExport');
      r.render = await waitText(page, '#rinfo', /rendered|failed/, 600000);
      check(`${engineId} · a camera clip renders`, !/failed/.test(r.render), r.render);
      const ext = engineId === 'browser' ? 'webm' : 'mp4';
      const out = path.join(DOCS, `w-camera-export.${ext}`);
      r.bytes = await saveDownload(page, out);
      r.probe = probe(out);
      check(`${engineId} · the render has the trimmed frame count`,
            Math.abs(+r.probe.nb_read_frames - r.trimmedFrames) <= 1,
            `${r.probe.nb_read_frames} vs ${r.trimmedFrames}`);
      await page.screenshot({ path: path.join(DOCS, `w-camera-render-${engineId}.png`) });
    }
    await ctx.close();
  } finally {
    await br.close();
  }
  return r;
}

/* ===== W7: the dots, as data — and the replay that has to match the render ===
 * The claim a .dots.gz makes is strong: it is not a compression of the picture,
 * it IS the picture's dots, and playing it back must land on the same pixels.
 * So this compares the app's own rendered frame with the player's replay of the
 * exported positions, byte for byte, on the same frames.
 */
async function runDots(engineId) {
  const r = { engine: engineId };
  const pref = engineId === 'browser' ? browserPref() : { mode: 'local', url: '', key: '' };
  const { ctx, page } = await newPage(pref);
  await loadClip(page, CLIP, BR.seconds);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
  await promptBoxPoint(page, SUBJECT_A);
  await page.click('#bTrack');
  r.trackText = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check(`${engineId} · dots run tracks`, !/failed/.test(r.trackText), r.trackText);
  await setMode(page, 'dots');

  const t0 = Date.now();
  r.stats = await page.evaluate(async () => {
    const { doc, bytes } = await window.DV_dots.doc();
    const counts = doc.frames.map((f) => f.reduce((a, x) => a + (x.length >> 1), 0));
    return { frames: doc.frames.length, w: doc.w, h: doc.h, fps: doc.fps,
             dotpx: doc.dotpx, palette: doc.palette, bytes: bytes.length,
             dotsMean: Math.round(counts.reduce((a, b) => a + b, 0) / counts.length),
             dotsMax: Math.max(...counts) };
  });
  r.seconds = +((Date.now() - t0) / 1000).toFixed(1);
  check(`${engineId} · the dot data covers every frame`,
        r.stats.frames === r.nFrames, `${r.stats.frames} vs ${r.nFrames}`);
  check(`${engineId} · there are dots in it`, r.stats.dotsMean > 100,
        JSON.stringify(r.stats));

  // The replay test: the app's own rendered frame against the player's replay
  // of the exported positions, byte for byte.
  //
  // On the browser engine both sides are the same JS, so the answer has to be
  // zero. On the server engine the dots are decided in numpy and the preview
  // paints them in JS, and a cell whose weight sits within a float rounding
  // error of the blue-noise threshold can land on the other side of it: a
  // couple of dots a frame, 9 bytes each (a 3x3 square). That is a real
  // difference and it is measured rather than hidden — it is also why the
  // export always comes from the same engine that rendered the preview.
  r.replay = await page.evaluate(async (frames) => {
    const P = await window.DV_dots.lib();
    const { doc } = await window.DV_dots.doc();
    const out = [];
    for (const i of frames) {
      const img = await window.DV_composeAt(i);
      const buf = new Uint8ClampedArray(doc.w * doc.h * 4);
      P.paintFrame(buf, doc.w, doc.h, doc, doc.frames[i], { bg: doc.bg });
      let bad = 0;
      for (let q = 0; q < buf.length; q++) if (buf[q] !== img.data[q]) bad++;
      out.push({ frame: i, bytes: buf.length, differing: bad });
    }
    return out;
  }, [0, 12, Math.min(40, r.nFrames - 1)]);
  for (const f of r.replay) {
    if (engineId === 'browser') {
      check(`${engineId} · frame ${f.frame} replays byte-identical`,
            f.differing === 0, `${f.differing} of ${f.bytes} bytes differ`);
    } else {
      check(`${engineId} · frame ${f.frame} replays within a dot or two`,
            f.differing / f.bytes < 5e-5,
            `${f.differing} of ${f.bytes} bytes differ `
            + `(${(f.differing / 9).toFixed(1)} dots)`);
    }
  }
  r.replayWorstDots = Math.max(...r.replay.map((f) => f.differing)) / 9;

  // and the JSON variant is the same numbers
  r.json = await page.evaluate(async () => {
    const P = await window.DV_dots.lib();
    const { doc } = await window.DV_dots.doc();
    const back = P.fromJSON(JSON.parse(JSON.stringify(P.toJSON(doc))));
    let same = back.frames.length === doc.frames.length;
    for (let i = 0; same && i < doc.frames.length; i++) {
      for (let k = 0; k < doc.frames[i].length; k++) {
        const a = doc.frames[i][k], b = back.frames[i][k];
        if (a.length !== b.length) { same = false; break; }
        for (let q = 0; q < a.length; q++) if (a[q] !== b[q]) { same = false; break; }
      }
    }
    return { identical: same, bytes: JSON.stringify(P.toJSON(doc)).length };
  });
  check(`${engineId} · the JSON variant round-trips`, r.json.identical);
  r.jsonBytes = r.json.bytes;
  await page.screenshot({ path: path.join(DOCS, `w-dots-data-${engineId}.png`) });
  await ctx.close();
  return r;
}

/* ============ W8: the flagship — subject to subject, and subject to shape ====
 * Clip A's subject morphs into clip B's subject, then into a ring rasterised
 * through the same dither pipeline. Preview in the player, export the dot data,
 * render the whole thing to MP4 on the server, and measure the player's own
 * frame rate on the finished sequence.
 */
async function runSequence() {
  const r = { twoClips: fs.existsSync(CLIP2) };
  const { ctx, page } = await newPage({ mode: 'local', url: '', key: '' });

  const trackAndCapture = async (file, box, at, len, seconds) => {
    await loadClip(page, file, seconds);
    await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
    await promptBoxPoint(page, box);
    await page.click('#bTrack');
    const t = await waitText(page, '#tinfo', /tracked|failed/, 900000);
    check('sequence · ' + path.basename(file) + ' tracks', !/failed/.test(t), t);
    await setMode(page, 'dots');
    await page.evaluate((i) => window.DV_draw(i), at); await sleep(500);
    await openStep(page, 'st6');
    await page.$eval('#seqLen', (el, v) => {
      el.value = String(v); el.dispatchEvent(new Event('input', { bubbles: true }));
    }, len);
    await page.click('#bCap');
    return waitText(page, '#seqinfo', /captured|failed/, 300000);
  };

  r.captureA = await trackAndCapture(CLIP, SUBJECT_A, 100, 45, BR.seconds);
  check('sequence · segment A captured', /captured/.test(r.captureA), r.captureA);
  if (r.twoClips) {
    // the tennis player, prompted with the box the job's own meta.json records
    r.captureB = await trackAndCapture(CLIP2, TENNIS, 0, 45, 3);
    check('sequence · segment B captured from the second clip',
          /captured/.test(r.captureB), r.captureB);
    r.libraryKept = await page.evaluate(() => window.DV_dots.library().length);
    check('sequence · the library survived loading another clip',
          r.libraryKept === 2, String(r.libraryKept));
  }
  await openStep(page, 'st6');
  await page.click('#shapes .chip[data-shape="ring"]');
  await sleep(2000);
  r.library = await page.evaluate(() => window.DV_dots.library()
    .map((x) => ({ name: x.name, kind: x.kind, frames: x.frames.length, dots: x.dots })));
  check('sequence · the ring rasterised into dots',
        (r.library[r.library.length - 1] || {}).dots > 500, JSON.stringify(r.library));

  await page.click('#bSeqPrev');
  r.preview = await waitText(page, '#seqinfo', /frames|failed/, 180000);
  check('sequence · previews in the player', !/failed/.test(r.preview), r.preview);
  r.doc = await page.evaluate(() => {
    const d = window.DV.seqDoc;
    const counts = d.frames.map((f) => f[0].length >> 1);
    return { frames: d.frames.length, fps: d.fps, marks: d.marks,
             counts: { min: Math.min(...counts), max: Math.max(...counts) } };
  });
  const morphs = r.doc.marks.filter((m) => m.kind === 'morph');
  check('sequence · one morph per join',
        morphs.length === r.library.length - 1,
        `${morphs.length} morphs for ${r.library.length} items`);
  check('sequence · a 900 ms morph is ~37 frames at 30 fps',
        morphs.every((m) => Math.abs(m.frames - 37) <= 2),
        JSON.stringify(morphs));
  await sleep(1200);
  await page.screenshot({ path: path.join(DOCS, 'w-sequence.png') });

  // the dot data for the whole sequence
  const dots = path.join(DOCS, 'w-sequence-export.dots.gz');
  const [dl] = await Promise.all([
    page.waitForEvent('download', { timeout: 60000 }),
    page.click('#bSeqDots'),
  ]);
  await dl.saveAs(dots);
  r.dotsBytes = fs.statSync(dots).size;
  const gz = fs.readFileSync(dots);
  check('sequence · the export really is gzip', gz[0] === 0x1f && gz[1] === 0x8b);

  // hand that same file to the server: dots in, MP4 out
  const up = await page.request.post(`${BASE}/api/sequence`, {
    multipart: { file: { name: 'sequence.dots.gz', mimeType: 'application/octet-stream',
                         buffer: gz },
                 format: 'mp4' },
    timeout: 180000,
  });
  check('sequence · the server rasterises the dot data', up.ok(), String(up.status()));
  r.server = await up.json();
  const mp4 = path.join(DOCS, 'w-sequence-export.mp4');
  const vid = await page.request.get(`${BASE}${r.server.url}`);
  fs.writeFileSync(mp4, await vid.body());
  r.mp4 = { bytes: fs.statSync(mp4).size, probe: probe(mp4) };
  check('sequence · the MP4 has every frame of the sequence',
        +r.mp4.probe.nb_read_frames === r.doc.frames,
        `${r.mp4.probe.nb_read_frames} vs ${r.doc.frames}`);
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', mp4,
    '-vf', `select='not(mod(n\,${Math.max(1, Math.floor(r.doc.frames / 15))}))',`
      + 'scale=320:-1,tile=5x3', '-frames:v', '1', '-update', '1',
    path.join(DOCS, 'seq-morph-sheet.png')]);

  // and the player's own frame rate on the finished sequence
  const demo = await ctx.newPage();
  demo.on('console', (m) => { if (m.type() === 'error') R.consoleErrors.push('demo: ' + m.text()); });
  demo.on('pageerror', (e) => R.pageErrors.push('demo: ' + String(e)));
  await demo.goto(`${BASE}/player/demo.html?src=/api/sequence/${r.server.sequence}/dots.gz`);
  await demo.waitForFunction(() => window.DP && window.DP.doc, null, { timeout: 60000 });
  r.player = await demo.evaluate(async () => {
    const P = window.DP;
    P.pause();
    let n = 0, worst = 0, sum = 0;
    const t0 = performance.now();
    await new Promise((ok) => {
      const step = () => {
        const a = performance.now();
        P.draw(n % P.nFrames);
        const dt = performance.now() - a;
        sum += dt; if (dt > worst) worst = dt;
        n++;
        if (performance.now() - t0 > 3000) return ok();
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
    const el = (performance.now() - t0) / 1000;
    return { frames: P.nFrames, painted: n, seconds: +el.toFixed(2),
             fps: +(n / el).toFixed(1), meanPaintMs: +(sum / n).toFixed(2),
             worstPaintMs: +worst.toFixed(2) };
  });
  check('sequence · the player holds 60 fps', r.player.fps >= 60,
        JSON.stringify(r.player));
  check('sequence · a frame costs well under a 60 fps budget',
        r.player.meanPaintMs < 16.6, JSON.stringify(r.player));
  await demo.screenshot({ path: path.join(DOCS, 'w-player-demo.png') });
  await demo.close();
  await ctx.close();
  return r;
}

/* ================================== K: the optional bearer-token gate ======== */
async function runApiKey() {
  const r = {};
  const port = await freePort();
  const key = 'verify-' + Math.random().toString(36).slice(2, 10);
  const py = path.join(HERE, 'env', 'venv', 'bin', 'python');
  if (!fs.existsSync(py)) return { skipped: 'no env/venv — run ./setup.sh' };
  const srv = spawn(py, [path.join(HERE, 'server', 'server.py')], {
    env: { ...process.env, DV_PORT: String(port), DV_API_KEY: key },
    stdio: 'ignore', detached: true,
  });
  const url = `http://127.0.0.1:${port}`;
  try {
    for (let i = 0; i < 80; i++) {
      const ok = await fetch(url + '/api/meta', {
        headers: { Authorization: 'Bearer ' + key },
      }).then((x) => x.ok).catch(() => false);
      if (ok) break;
      await sleep(500);
    }
    r.port = port;
    const bare = await fetch(url + '/api/meta').catch(() => null);
    r.withoutKey = bare ? bare.status : 'no answer';
    check('DV_API_KEY: /api/meta is 401 without a bearer token',
          r.withoutKey === 401, String(r.withoutKey));
    const wrong = await fetch(url + '/api/meta',
      { headers: { Authorization: 'Bearer nope' } }).catch(() => null);
    r.wrongKey = wrong ? wrong.status : 'no answer';
    check('DV_API_KEY: a wrong token is 401', r.wrongKey === 401, String(r.wrongKey));
    const good = await fetch(url + '/api/meta',
      { headers: { Authorization: 'Bearer ' + key } });
    r.withKey = good.status;
    r.meta = await good.json();
    check('DV_API_KEY: the right token is 200', r.withKey === 200);
    check('DV_API_KEY: /api/meta advertises the gate', r.meta.auth === 'bearer',
          JSON.stringify(r.meta));
    const pageOK = await fetch(url + '/').then((x) => x.status);
    r.pageWithoutKey = pageOK;
    check('DV_API_KEY: the page itself stays reachable', pageOK === 200,
          'a browser cannot put a header on the request that loads the HTML');

    // and the engine, given the key, drives it end to end
    const { ctx, page } = await newPage({ mode: 'custom', url, key });
    r.engine = await page.evaluate(() => window.DV_engine());
    check('DV_API_KEY: the remote engine connects with the key',
          r.engine.id === 'remote' && r.engine.baseUrl === url,
          JSON.stringify(r.engine));
    r.chip = (await page.textContent('#engName')).trim();
    await page.screenshot({ path: path.join(DOCS, 'w-engine-keyed.png') });
    await ctx.close();
  } finally {
    try { process.kill(-srv.pid); } catch (e) { try { srv.kill('SIGKILL'); } catch (e2) { /* gone */ } }
  }
  return r;
}

/* --------------------------------------------------------------------- main */
BR = await pickBrowser();
R.browser = { how: BR.how, executionProvider: BR.ep, webgpu: BR.gpu,
              clipSeconds: BR.seconds };
if (BR.ep === 'wasm') {
  R.browser.note = 'no WebGPU adapter was available, so the tracker ran on the '
    + 'multi-threaded WASM backend (~6x slower) over a shorter clip. The numbers '
    + 'below are correctness, not performance.';
}
console.error('[verify-web] ' + BR.how + ' · EP ' + BR.ep);

try {
  R.runs.engineChip = await runEngineChip();
  R.runs.still = await runStill();
  R.runs.whole = await runWhole();
  R.runs.tracked = await runTracked();
  R.runs.lasso = await runLasso();
  R.runs.cameraRemote = await runCamera('remote', true);
  R.runs.cameraBrowser = await runCamera('browser', false);
  R.runs.dotsRemote = await runDots('remote');
  R.runs.dotsBrowser = await runDots('browser');
  R.runs.sequence = await runSequence();
  R.runs.entryBrowser = await runEntry('browser');
  R.runs.entryRemote = await runEntry('remote');
  R.runs.apiKey = await runApiKey();
} catch (e) {
  R.fatal = String(e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e);
} finally {
  await BR.browser.close();
}

R.checksPassed = R.checks.filter((c) => c.ok).length;
R.checksTotal = R.checks.length;
console.log(JSON.stringify(R, null, 1));
fs.writeFileSync(path.join(DOCS, 'verify-web-report.json'), JSON.stringify(R, null, 1));
process.exit(R.fatal || R.consoleErrors.length || R.pageErrors.length ? 1 : 0);
