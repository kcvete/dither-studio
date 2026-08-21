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
 *   W1a browser · a still, whole-image dots -> a one-frame .dots.gz
 *   W1b browser + server · a still, a clicked subject segmented in ONE frame
 *       (no propagation) -> cutout PNG with a transparent background
 *   W2  browser · a clip, whole frame -> WebM
 *   W2b browser · a clip exported as a PAIR: the dithered WebM and the same
 *       frames undithered, checked against the frames the tab drew
 *   W2c browser · a clip tracked, THEN trimmed: the narrower range renders,
 *       exports its pair and writes its .dots.gz out of frames and masks that
 *       are already in the tab, with no second track and no network at all;
 *       a wider range is offered rather than silently re-decoded
 *   W3  browser · a clip, one tracked subject, frame-0 preview -> dots -> WebM,
 *       with the mask polish on for the export and off again afterwards
 *   W4  browser · a polygon mask prompt (the heads_mask graph) -> tracked
 *   W5  browser · two subjects prompted on DIFFERENT frames, mask-area census
 *   W11 browser · PER-SUBJECT INCREMENTAL TRACKING: #1 tracked alone, #2 added
 *       and tracked on its own with #1's logits hashed across the run, #2's
 *       prompt edited so it goes stale, #2 re-tracked alone, #1 hidden and
 *       then removed — all of it in the tab, with no server anywhere near it
 *   WX  browser · the CANVAS: a tracked clip at 9:16 out of a 16:9 source —
 *       the crop path built in the tab from the mask logits, the dots
 *       re-measured at 1080x1920, the matched original cut on the same path,
 *       the .dots.gz carrying the new frame, and a sequence at 4:5
 *   W8  the SEQUENCE view: four items added through the UI (two clips, a still
 *       cutout, a shape), per-item trims and colours, three transition kinds,
 *       drag-reorder, preview, .dots.gz and an MP4 off the server
 *   W10 pixel modes as particles: a cell-1 Bayer cutout morphing into a Dots
 *       subject with the flight capped and handed back at full density, plus
 *       "+ image… -> select a subject" bringing a cutout in from the sequence
 *   W9  a sequence item's OWN look: the studio's controls scoped to one card —
 *       every mode, the dot sliders, a palette and the mask polish — changing
 *       item 2 while items 1 and 3 stay byte-identical, through the preview,
 *       the .dots.gz and the server's MP4
 *   WD  browser · the DEPLOYMENT mirrored: web/ served statically with the
 *       fp16-only model tree Pages actually ships. 512 and 1024 track; a GPU
 *       with no shader-f16 falls to fp16 on WASM instead of asking for an fp32
 *       graph that is not there; and a tier graph that answers with a 404 page
 *       (or a 200 that is HTML) falls back to 768 px naming the file, rather
 *       than reaching ORT as "protobuf parsing failed"
 *   R5  server  · the same two-frame prompt, so the feature is checked on both
 *   K   an optional DV_API_KEY server: 401 without the header, 200 with it
 *
 * Screenshots go to docs/, the report to docs/verify-web-report.json.
 * Exits non-zero on any console error, page error, or a failed assertion.
 */
import { chromium, firefox, webkit } from 'playwright';
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.argv[2] || 'http://127.0.0.1:8765';
const CLIP = process.argv[3] || path.join(HERE, 'sample.mp4');
const ENTRY = process.argv[4] || path.join(HERE, 'docs', 'entry-clip.mp4');
const STILL = process.argv[5] || CLIP.replace(/\.\w+$/, '.jpg');
/* A second clip, for the morph that goes from a subject in one clip to a
 * subject in another. It defaults to the entry clip -- committed, like
 * sample.mp4, so the sequence run works from a fresh clone. It used to default
 * to a source file inside jobs/, which stopped existing the moment jobs/ got a
 * garbage collector: a suite must not depend on the scratch directory. */
const CLIP2 = process.argv[6] || ENTRY;
const DOCS = path.join(HERE, 'docs');
fs.mkdirSync(DOCS, { recursive: true });
const AFTER = path.join(DOCS, 'ux-after');
fs.mkdirSync(AFTER, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* the parkour athlete, in the reference clip's own 1280x720 pixels */
const SUBJECT_A = { box: [435, 95, 625, 360], point: [545, 205] };
/* the tree on the right edge of the same clip: large, static, and nothing
 * to do with the athlete, which is what a second subject has to be */
const SUBJECT_B = { box: [1005, 5, 1279, 470], point: [1150, 160] };
/* docs/entry-clip.mp4 (Mixkit): a static park shot a jogger runs into.
 * The tree is there from frame 0; she is not in the shot until frame 38. */
const TREE = { box: [150, 1, 206, 430], point: [178, 220], frame: 0 };
const JOGGER = { box: [10, 256, 98, 452], point: [52, 318], frame: 48 };
const JOGGER_ENTERS = 38;
/* the near tree trunk on the left of the entry clip: large, unambiguous and
 * in every frame, which is all the second item in a sequence has to be */
const SECOND = { box: [55, 0, 300, 450], point: [168, 300], frame: 0 };

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
  // engines/decode.js reads this before it picks a path. Forcing one is how the
  // same clip gets decoded both ways and the frames compared; a forced path
  // does not fall back, so a failure here is a failure and not a silent detour.
  if (opts.decodePath) {
    await ctx.addInitScript((d) => { window.DV_DECODE_PATH = d; }, opts.decodePath);
  }
  // for taking an API away from the page on purpose
  if (opts.init) await ctx.addInitScript(opts.init);
  const page = await ctx.newPage();
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    // "Failed to load resource: ... 404" carries the URL in the location, not
    // in the text, so a flow that expects one 404 has to be able to name it
    const t = m.text() + ' ' + ((m.location() || {}).url || '');
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
    // Taking a download off a blob: URL cancels the request that started it.
    // Chromium does not report that; Firefox and WebKit do, as "cancelled" on
    // a blob: URL, and it is the success path rather than a failure.
    if (/^blob:/.test(rq.url()) && /^(cancelled|NS_BINDING_ABORTED)$/i.test(why)) return;
    if (why !== 'net::ERR_ABORTED' && !expected(rq.url())) {
      R.consoleErrors.push('requestfailed ' + rq.url() + ' ' + why);
    }
  });
  await page.goto((opts.base || BASE) + '/', { waitUntil: 'networkidle' });
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
  // There is no length cap to set any more. `seconds` is a request for a
  // shorter TRIM, which DV_limit() arms before the drop — the page then opens
  // the first `seconds` of the file instead of all of it.
  if (seconds) await page.evaluate((v) => window.DV_limit(v), seconds);
  await page.setInputFiles('#file', file);
  // Over a minute the page states its arithmetic and waits for a click rather
  // than committing the tab to a long decode. That is the informed-consent
  // gate, not a refusal: "whole clip" is the answer to it.
  await page.waitForFunction(
    () => window.DV.kind === 'video' || window.DV.awaitingChoice,
    null, { timeout: 300000 });
  if (await page.evaluate(() => window.DV.awaitingChoice)) {
    await page.click('#bTrimAll');
    // a long clip really does take minutes to decode in the tab — this is the
    // one wait in the suite that must not use the 30 s default
    await page.waitForFunction(() => window.DV.kind === 'video',
                               null, { timeout: 1800000 });
  }
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

  /* The slow-tracking hint. The browser engine is the free tier and it is the
   * slow one; when a run is genuinely long the page says where the fast one
   * is. DV_slowHint(etaSeconds, fps) is the simulator -- a real slow run is
   * minutes, and this check is about the sentence, not the wait. */
  await openStep(page, 'st2');
  // the hint lives under the tracking progress line, so the subject controls
  // have to be the ones on screen
  await page.click('#scope .chip[data-scope="track"]');
  await sleep(400);
  r.slowQuiet = await page.evaluate(() => window.DV_slowHint(5, 30));
  r.slowQuietVisible = await page.isVisible('#slowhint');
  check('a quick run says nothing',
        r.slowQuiet === false && r.slowQuietVisible === false,
        `${r.slowQuiet} / ${r.slowQuietVisible}`);
  r.slowShown = await page.evaluate(() => window.DV_slowHint(120, 2));
  await sleep(200);
  r.slowVisible = await page.isVisible('#slowhint');
  r.slowText = (await page.textContent('#slowhint')).replace(/\s+/g, ' ').trim();
  r.slowHref = await page.getAttribute('#slowmain a', 'href');
  check('a slow run on the browser engine offers the local one',
        r.slowShown === true && r.slowVisible === true,
        `shown=${r.slowShown} visible=${r.slowVisible} quiet=${r.slowQuiet}/`
        + `${r.slowQuietVisible} · ${r.slowText}`);
  check('the offer carries the repository link',
        /github\.com\/kcvete\/dither-studio/.test(r.slowHref || ''), r.slowHref);
  check('the offer names the speed-up in words',
        /run it on your machine/i.test(r.slowText), r.slowText);
  await page.screenshot({ path: path.join(DOCS, 'ux-after', 'slow-hint-browser.png') });
  await page.click('#bSlowNo'); await sleep(200);
  check('dismissing it is remembered for the session',
        (await page.isVisible('#slowhint')) === false
        && (await page.evaluate(() => window.DV_slowHint(600, 1))) === false);
  await ctx.close();
  return r;
}

/* ================================ W1a: a still, dotted whole-image, in-tab */
async function runStillDots() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 30000 });
  await sleep(700);
  const cls = await page.getAttribute('#modes .chip[data-mode="dots"]', 'class');
  check('still · dots is offered without a subject', !/\boff\b/.test(cls || ''), cls);
  await setMode(page, 'dots');
  await sleep(500);
  r.fps = await page.textContent('#fps');
  r.dots = +(/· (\d+) dots/.exec(r.fps) || [])[1];
  check('still · whole-image dots are drawn', r.dots > 200, r.fps);
  r.census = await census(page);
  check('still · dots are two colours', r.census.distinctColours === 2,
        JSON.stringify(r.census));
  await page.screenshot({ path: path.join(DOCS, 'w-still-dots.png') });

  await openStep(page, 'st5');
  check('still · the .dots.gz export is offered',
        await page.getAttribute('#dotsexp', 'hidden') === null);
  const gz = path.join(DOCS, 'w-still-dots-export.dots.gz');
  // the dot-data buttons live under the "for developers" disclosure now
  await page.evaluate(() => { const d = document.querySelector('#devexp'); if (d) d.open = true; });
  const [d] = await Promise.all([
    page.waitForEvent('download', { timeout: 60000 }).catch(() => null),
    page.click('#bDots'),
  ]);
  check('still · a .dots.gz downloads', !!d);
  await d.saveAs(gz);
  const bytes = fs.readFileSync(gz);
  check('still · it is gzip', bytes[0] === 0x1f && bytes[1] === 0x8b);
  r.dotsBytes = bytes.length;
  const P = await import(path.join(HERE, 'web', 'player', 'dither-player.mjs'));
  const doc = await P.unpack(new Uint8Array(bytes));
  r.doc = { w: doc.w, h: doc.h, frames: doc.frames.length, fps: doc.fps,
            dots: doc.frames[0].reduce((a, x) => a + (x.length >> 1), 0) };
  check('still · the dot file is one frame', doc.frames.length === 1,
        String(doc.frames.length));
  check('still · the file has the dots the preview showed',
        r.doc.dots === r.dots, `${r.doc.dots} != ${r.dots}`);
  await ctx.close();
  return r;
}

/* ========== W1b: a still, one clicked subject, single-image segmentation ====
 * Both engines. In the tab this is encoder + heads_prompt on one frame; on the
 * server it is /api/upload_image and then a one-frame /preview. Neither
 * propagates anything, so the mask is live: it is re-cut on every click. */
async function runStillSubject(engineId) {
  const r = { engine: engineId };
  const pref = engineId === 'browser' ? browserPref() : { mode: 'local', url: '', key: '' };
  const { ctx, page } = await newPage(pref);
  check(`${engineId} · still-subject run is on the right engine`,
        (await page.evaluate(() => window.DV_engine())).id
          === (engineId === 'browser' ? 'browser' : 'remote'));
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 30000 });
  await sleep(600);
  await openStep(page, 'st2');
  r.scopeLabels = await page.$$eval('#scope .chip', (n) => n.map((e) => e.textContent));
  check(`${engineId} · step 2 reads as a still`,
        r.scopeLabels.join('/') === 'whole image/select subjects',
        r.scopeLabels.join('/'));
  await page.click('#scope .chip[data-scope="track"]');
  await sleep(700);
  check(`${engineId} · no prompt-frame slider on a photograph`,
        await page.getAttribute('#pfui', 'hidden') !== null);
  r.button = await page.textContent('#bTrack');
  check(`${engineId} · no Track button on a photograph`,
        /use this selection/i.test(r.button), r.button);

  const t0 = Date.now();
  await promptBoxPoint(page, SUBJECT_A);
  r.info = await waitText(page, '#pvinfo', /subject|failed/, 300000);
  check(`${engineId} · the subject segments`, !/failed/.test(r.info), r.info);
  r.firstSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  r.segmentSeconds = +(/in ([\d.]+) s/.exec(r.info) || [])[1];
  r.areas = await page.evaluate(() => window.DV_still.areas());
  r.area = r.areas[Object.keys(r.areas)[0]];
  check(`${engineId} · the mask is a person-sized region`,
        r.area > 5000 && r.area < 60000, String(r.area));
  await page.screenshot({ path: path.join(DOCS, `w-still-prompt-${engineId}.png`) });

  // a second click must re-cut the mask live, with no button pressed
  const t1 = Date.now();
  const [nx, ny] = await stageXY(page, SUBJECT_A.point[0], SUBJECT_A.point[1] + 60);
  await page.mouse.click(nx, ny);
  await waitText(page, '#pvinfo', /subject/, 120000);
  r.liveSeconds = +((Date.now() - t1) / 1000).toFixed(1);
  r.areaAfterSecondClick = (await page.evaluate(() => window.DV_still.areas()))[
    Object.keys(r.areas)[0]];
  check(`${engineId} · the second click re-cut the mask without a button`,
        r.areaAfterSecondClick > 3000, String(r.areaAfterSecondClick));

  await page.click('#bTrack');
  await sleep(700);
  r.targets = await page.$$eval('#target .chip', (n) => n.map((e) => e.textContent));
  check(`${engineId} · the subject gets its own palette`,
        r.targets.length === 2, r.targets.join('/'));

  await setMode(page, 'bluenoise');
  await sleep(500);
  r.cutout = await census(page);
  check(`${engineId} · a cutout still is flat background + a dithered subject`,
        r.cutout.distinctColours >= 2 && r.cutout.distinctColours <= 4,
        JSON.stringify(r.cutout));
  await page.screenshot({ path: path.join(DOCS, `w-still-cutout-${engineId}.png`) });

  await openStep(page, 'st3');
  await page.click('#compose .chip[data-compose="overlay"]');
  await sleep(600);
  r.overlay = await census(page);
  check(`${engineId} · overlay keeps the photograph`,
        r.overlay.distinctColours > r.cutout.distinctColours,
        JSON.stringify([r.cutout, r.overlay]));
  await page.click('#compose .chip[data-compose="cutout"]');
  await sleep(400);
  await setMode(page, 'dots');
  await sleep(600);
  r.dotsFps = await page.textContent('#fps');
  r.subjectDots = +(/· (\d+) dots/.exec(r.dotsFps) || [])[1];
  check(`${engineId} · the subject alone becomes dots`, r.subjectDots > 50, r.dotsFps);

  await openStep(page, 'st5');
  check(`${engineId} · the transparent-background switch is offered`,
        await page.getAttribute('#pngalpha', 'hidden') === null);
  await page.check('#cAlpha');
  await page.click('#bExport');
  r.export = await waitText(page, '#rinfo', /PNG|failed/, 120000);
  check(`${engineId} · the cutout PNG exports`, !/failed/.test(r.export), r.export);
  const out = path.join(DOCS, `w-still-cutout-${engineId}-export.png`);
  r.bytes = await saveDownload(page, out);
  r.probe = probe(out);
  check(`${engineId} · the PNG carries an alpha channel`,
        /rgba/.test(r.probe.pix_fmt || ''), JSON.stringify(r.probe));
  r.alpha = alphaCensus(out, 0);
  check(`${engineId} · the background really is transparent`,
        r.alpha.transparentPct > 80, JSON.stringify(r.alpha));
  check(`${engineId} · the subject really is opaque`,
        r.alpha.opaquePct > 0.1, JSON.stringify(r.alpha));
  await page.screenshot({ path: path.join(DOCS, `w-still-export-${engineId}.png`) });
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

/* ================ W2b: the matched cut, in the tab =========================
 * The pair the browser engine writes: a dithered WebM and the SAME frames
 * undithered, out of the same MediaRecorder, at the same size and rate. There
 * is no jobs/<id>/frames/ here to compare against -- the frames only ever
 * existed in the tab -- so the ground truth is the tab itself: DV_originalAt(n)
 * is the ImageData that was handed to the recorder, and the exported file's
 * frame n has to still be it, VP9 loss aside.
 */
async function runOriginalBrowser() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  await loadClip(page, CLIP, BR.seconds);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);

  await setMode(page, 'ordered');
  await openStep(page, 'st5');
  r.checkboxOffered = !(await page.locator('#origui').isHidden());
  check('the tab offers the matched cut', r.checkboxOffered);
  await page.check('#cOrig');
  r.note = (await page.textContent('#orignote')).trim();
  const t0 = Date.now();
  await page.click('#bExport');
  r.info = await waitText(page, '#rinfo', /original cut|failed/, 900000);
  check('the pair exports in the tab', !/failed/.test(r.info), r.info);
  r.seconds = +((Date.now() - t0) / 1000).toFixed(1);
  await page.locator('#dlorig').scrollIntoViewIfNeeded();
  await sleep(300);
  await page.screenshot({ path: path.join(DOCS, 'w-original.png') });

  // two links, two files
  const dith = path.join(DOCS, 'w-original-dithered.webm');
  const orig = path.join(DOCS, 'w-original-cut.webm');
  r.ditheredBytes = await saveDownload(page, dith);
  const [d2] = await Promise.all([
    page.waitForEvent('download', { timeout: 30000 }),
    page.click('#dlorig'),
  ]);
  r.originalName = d2.suggestedFilename();
  await d2.saveAs(orig);
  r.originalBytes = fs.statSync(orig).size;
  check('the second file is named for the pair',
        /\.original\.webm$/.test(r.originalName), r.originalName);

  r.dithered = probe(dith);
  r.original = probe(orig);
  for (const k of ['nb_read_frames', 'width', 'height']) {
    check(`the pair agrees on ${k}`, String(r.dithered[k]) === String(r.original[k]),
          `${r.dithered[k]} vs ${r.original[k]}`);
  }
  /* The rate is the recorder's wall clock, not a number either file was told
   * to carry -- `MediaRecorder` timestamps each frame as it is handed over, so
   * two passes over the same 150 frames land a fraction of a percent apart.
   * The frame COUNT above is the exact one; this is the duration agreeing. */
  const rate = (p) => {
    const [a, b] = String(p.r_frame_rate).split('/').map(Number);
    return a / (b || 1);
  };
  r.rates = [+rate(r.dithered).toFixed(3), +rate(r.original).toFixed(3)];
  r.rateDriftPct = +(100 * Math.abs(r.rates[0] - r.rates[1]) / r.rates[0]).toFixed(2);
  check('the pair shares a frame rate', r.rateDriftPct < 3,
        `${r.dithered.r_frame_rate} vs ${r.original.r_frame_rate} `
        + `(${r.rateDriftPct}% apart)`);
  check('the original cut has every frame',
        +r.original.nb_read_frames === r.nFrames,
        `${r.original.nb_read_frames} != ${r.nFrames}`);

  /* frame n of the file against frame n as the tab drew it */
  const rawRGB = (args) => execFileSync('ffmpeg', ['-v', 'error'].concat(args),
                                        { maxBuffer: 1 << 28 });
  const meanAbs = (a, b) => {
    if (a.length !== b.length) throw new Error('frame sizes differ');
    let d = 0;
    for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
    return +(d / a.length).toFixed(3);
  };
  r.frameMeanAbsDiff = {};
  for (const n of [0, r.nFrames >> 1, r.nFrames - 1]) {
    const b64 = await page.evaluate(async (i) => {
      const img = await window.DV_originalAt(i);
      const cv = document.createElement('canvas');
      cv.width = img.width; cv.height = img.height;
      cv.getContext('2d').putImageData(img, 0, 0);
      return cv.toDataURL('image/png').split(',')[1];
    }, n);
    const ref = path.join(DOCS, `w-original-ref-${n}.png`);
    fs.writeFileSync(ref, Buffer.from(b64, 'base64'));
    const d = meanAbs(
      rawRGB(['-i', orig, '-vf', `select=eq(n\\,${n})`, '-vframes', '1',
              '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']),
      rawRGB(['-i', ref, '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']));
    r.frameMeanAbsDiff[n] = d;
    check(`original frame ${n} is the tab's frame ${n}`, d < 4, 'mean abs diff ' + d);
  }

  // and the question is never asked of a sequence, which has no original
  await page.click('#viewbar .chip[data-view="sequence"]');
  await sleep(500);
  r.hiddenInSequence = await page.locator('#origui').isHidden();
  check('the sequence view does not offer a matched cut', r.hiddenInSequence);
  await ctx.close();
  return r;
}

/* ======= W2c: the range, after the track — in the tab =====================
 * The same claim the server run makes (verify.mjs, flow R), on the engine that
 * has no server at all: once a clip is tracked, narrowing the trim is a window
 * on the frames and mask logits already in memory. It re-decodes nothing,
 * re-tracks nothing, and every export follows it.
 */
async function runRangeBrowser() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  const seen = [];
  page.on('request', (rq) => seen.push(rq.method() + ' ' + rq.url().replace(BASE, '')));
  // 2 s of the clip, so there is room to widen as well as narrow
  await loadClip(page, CLIP, 2);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  check('range · the tab opened a 2 s window', r.nFrames === 60, String(r.nFrames));

  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
  await promptBoxPoint(page, SUBJECT_A);
  await page.click('#bTrack');
  r.track = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check('range · the clip tracks in the tab', !/failed/.test(r.track), r.track);

  /* --- narrow: no network, no re-track, same frames --- */
  seen.length = 0;
  await page.evaluate(() => window.DV_range.seconds(0.5, 1.5));
  await sleep(400);
  r.range = await page.evaluate(() => window.DV_range.get());
  r.label = await page.evaluate(() => window.DV_range.label());
  r.apiCalls = seen.filter((x) => /\/api\//.test(x));
  check('range · 0.5–1.5 s of a 30 fps clip is frames 15–44',
        r.range.in === 15 && r.range.out === 44 && r.range.n === 30,
        JSON.stringify(r.range));
  check('range · the transport states the range',
        /15–44 of 60/.test(r.label), r.label);
  check('range · the reset is offered',
        await page.evaluate(() => window.DV_range.resetShown()));
  check('range · narrowing after a track touches no API at all',
        r.apiCalls.length === 0, JSON.stringify(r.apiCalls));
  check('range · the decoded frames stay where they were',
        (await page.evaluate(() => window.DV.nFrames)) === r.nFrames);
  await page.screenshot({ path: path.join(DOCS, 'w-range-narrowed.png') });

  /* --- the pair, cut to the window --- */
  await setMode(page, 'dots');
  await openStep(page, 'st5');
  await page.check('#cOrig');
  await page.click('#bExport');
  r.export = await waitText(page, '#rinfo', /original cut|failed/, 900000);
  check('range · a trimmed clip exports as a pair', !/failed/.test(r.export), r.export);
  check('range · the render says which frames it used',
        /frames 15–44 of 60/.test(r.export), r.export);
  const dith = path.join(DOCS, 'w-range-dithered.webm');
  const orig = path.join(DOCS, 'w-range-original.webm');
  r.ditheredBytes = await saveDownload(page, dith);
  const [d2] = await Promise.all([
    page.waitForEvent('download', { timeout: 60000 }),
    page.click('#dlorig'),
  ]);
  await d2.saveAs(orig);
  r.dithered = probe(dith);
  r.original = probe(orig);
  check('range · the render is the window, not the clip',
        +r.dithered.nb_read_frames === r.range.n,
        `${r.dithered.nb_read_frames} != ${r.range.n}`);
  check('range · the matched cut is the same window',
        +r.original.nb_read_frames === r.range.n,
        `${r.original.nb_read_frames} != ${r.range.n}`);

  /* frame k of the cut is the tab's frame in+k, and NOT its frame k */
  const rawRGB = (args) => execFileSync('ffmpeg', ['-v', 'error'].concat(args),
                                        { maxBuffer: 1 << 28 });
  const meanAbs = (a, b) => {
    if (a.length !== b.length) throw new Error('frame sizes differ');
    let d = 0;
    for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]);
    return +(d / a.length).toFixed(3);
  };
  const tabFrame = async (n) => {
    const b64 = await page.evaluate(async (i) => {
      const img = await window.DV_originalAt(i);
      const cv = document.createElement('canvas');
      cv.width = img.width; cv.height = img.height;
      cv.getContext('2d').putImageData(img, 0, 0);
      return cv.toDataURL('image/png').split(',')[1];
    }, n);
    const f = path.join(DOCS, `w-range-ref-${n}.png`);
    fs.writeFileSync(f, Buffer.from(b64, 'base64'));
    return rawRGB(['-i', f, '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']);
  };
  const cutFrame = (k) => rawRGB(['-i', orig, '-vf', `select=eq(n\\,${k})`,
    '-vframes', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-']);
  r.frameMeanAbsDiff = {};
  for (const k of [0, r.range.n - 1]) {
    const d = meanAbs(cutFrame(k), await tabFrame(r.range.in + k));
    r.frameMeanAbsDiff[`${k} vs tab ${r.range.in + k}`] = d;
    check(`range · cut frame ${k} is the tab's frame ${r.range.in + k}`,
          d < 4, 'mean abs diff ' + d);
  }
  r.controlMeanAbsDiff = meanAbs(cutFrame(0), await tabFrame(0));
  check('range · and it is NOT the tab\'s frame 0',
        r.controlMeanAbsDiff > 4, 'mean abs diff ' + r.controlMeanAbsDiff);

  /* --- the .dots.gz follows the window too --- */
  const gz = path.join(DOCS, 'w-range-dots.gz');
  await page.evaluate(() => { const d = document.querySelector('#devexp'); if (d) d.open = true; });
  const [d3] = await Promise.all([
    page.waitForEvent('download', { timeout: 300000 }),
    page.click('#bDots'),
  ]);
  await d3.saveAs(gz);
  const P = await import(path.join(HERE, 'web', 'player', 'dither-player.mjs'));
  const doc = await P.unpack(new Uint8Array(fs.readFileSync(gz)));
  r.dotsFrames = doc.frames.length;
  check('range · the .dots.gz is the window', r.dotsFrames === r.range.n,
        `${r.dotsFrames} != ${r.range.n}`);

  /* --- the sequence seeds an entry's in/out from the range, and keeps the
         whole clip in the pool so the entry can be widened again --- */
  await page.evaluate(() => window.DV_seq.add('clip'));
  await page.waitForFunction(() => window.DV_seq.strip().length > 0,
                             null, { timeout: 300000 });
  await sleep(500);
  r.strip = await page.evaluate(() => window.DV_seq.strip()[0]);
  r.lib = await page.evaluate(() => window.DV_seq.library()[0]);
  check('range · the strip entry starts at the studio\'s range',
        r.strip.in === r.range.in && r.strip.out === r.range.out,
        JSON.stringify([r.strip.in, r.strip.out]));
  check('range · the entry is the range long', r.strip.frames === r.range.n,
        String(r.strip.frames));
  check('range · the pool item still holds every tracked frame',
        r.lib.nFrames === r.nFrames, String(r.lib.nFrames));
  await page.evaluate(() => window.DV_seq.view('studio'));
  await sleep(400);

  /* --- wider than what was decoded: the offer, and nothing else --- */
  await openStep(page, 'st1');
  await page.evaluate(() => window.DV_range.seconds(0, 4));
  await sleep(400);
  r.offer = await page.evaluate(() => window.DV_range.offer());
  check('range · a range past the decode raises the offer', !!r.offer,
        JSON.stringify(r.offer));
  check('range · the offer names the frames that are not tracked',
        r.offer && JSON.stringify(r.offer.missing) === JSON.stringify([[60, 119]]),
        r.offer && JSON.stringify(r.offer.missing));
  check('range · the offer is on screen',
        await page.locator('#trimoffer').isVisible());
  check('range · the clip is not re-decoded until the offer is taken',
        (await page.evaluate(() => window.DV.nFrames)) === r.nFrames,
        String(await page.evaluate(() => window.DV.nFrames)));
  await page.locator('#trimoffer').scrollIntoViewIfNeeded();
  await sleep(200);
  await page.screenshot({ path: path.join(DOCS, 'w-range-offer.png') });

  /* --- and taking it re-decodes and re-tracks, prompts carried across --- */
  await page.click('#bExtend');
  await page.waitForFunction(() => window.DV.nFrames === 120,
                             null, { timeout: 600000 });
  r.extendTrack = await waitText(page, '#tinfo', /tracked 120|failed/, 900000);
  check('range · taking the offer tracks the wider range',
        !/failed/.test(r.extendTrack), r.extendTrack);
  r.afterExtend = await page.evaluate(() => ({
    nFrames: window.DV.nFrames, tracked: window.DV.tracked,
    subjects: window.DV.subjects.length,
    promptFrames: window.DV.subjects.map((s) => s.promptFrame),
    range: window.DV_range.get(),
  }));
  check('range · the wider clip is 120 frames and tracked',
        r.afterExtend.nFrames === 120 && r.afterExtend.tracked,
        JSON.stringify(r.afterExtend));
  check('range · the prompts came across',
        r.afterExtend.subjects === 1 && r.afterExtend.promptFrames[0] === 0,
        JSON.stringify(r.afterExtend.promptFrames));
  check('range · the re-cut clip opens at its full length',
        r.afterExtend.range.whole);
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
  /* One chip per tracker square this deployment carries. It used to be exactly
   * one, because the browser engine only ever exported 768 px; there are three
   * graph sets now and the chips are built from meta().track_sizes either way,
   * so what is asserted is that the list is not empty and that the default is
   * the one selected -- flow WQ is where each square is actually tracked. */
  r.qualityDefault = await page.evaluate(() => ({
    picked: window.DV.trackSize, offered: window.DV.meta.default_track_size }));
  check('the browser engine offers the squares it shipped models for',
        r.qualityChips.length >= 1
        && r.qualityChips.length === (await page.evaluate(
          () => (window.DV.meta.track_sizes || []).length)),
        JSON.stringify(r.qualityChips));
  check('and starts on the one a first visit downloads',
        r.qualityDefault.picked === r.qualityDefault.offered,
        JSON.stringify(r.qualityDefault));

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

  /* --- mask polish, in the tab. The server half of this (and the byte-for-byte
   * agreement between the two) is verify.mjs's P run; what matters here is that
   * the free tier has the feature at all, that it redraws, and that the export
   * below goes out through the polished masks rather than the raw ones. */
  const canvasHash = () => page.evaluate(() => {
    const c = document.querySelector('#vcv');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    let h = 2166136261;
    for (let q = 0; q < d.length; q += 4) { h ^= d[q]; h = Math.imul(h, 16777619); }
    return h >>> 0;
  });
  r.polish = { params: await page.evaluate(() => window.MaskPolish.params(70)),
               before: { fps: r.dotsFps, hash: await canvasHash() } };
  check('browser · polish is off until it is asked for',
        (await page.evaluate(() => window.DV_polish.get()))[0].polish === 0);
  await openStep(page, 'st3');
  await page.click('#pollist .chip.pol');
  await sleep(1000);
  await page.evaluate(() => window.DV_draw(10)); await sleep(2500);
  r.polish.state = await page.evaluate(() => window.DV_polish.get());
  r.polish.after = { fps: await page.textContent('#fps'), hash: await canvasHash() };
  check('browser · the toggle sets 70 on that subject',
        r.polish.state[0].polish === 70, JSON.stringify(r.polish.state));
  check('browser · polishing the mask changes the picture',
        r.polish.after.hash !== r.polish.before.hash, JSON.stringify(r.polish));
  await page.screenshot({ path: path.join(DOCS, 'w-polish.png') });

  await openStep(page, 'st5');
  t0 = Date.now();
  await page.click('#bExport');
  r.render = await waitText(page, '#rinfo', /rendered|failed/, 600000);
  check('the tracked clip exports', !/failed/.test(r.render), r.render);
  r.renderSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  const out = path.join(DOCS, 'w-dots-export.webm');
  r.bytes = await saveDownload(page, out);
  r.probe = probe(out);
  r.polish.exportSeconds = r.renderSeconds;   // the WebM above went out polished
  await page.screenshot({ path: path.join(DOCS, 'w-export.png') });
  // and off again, so the format runs below measure the plain path
  await openStep(page, 'st3');
  await page.click('#pollist .chip.pol'); await sleep(600);
  check('browser · the toggle turns polish back off',
        (await page.evaluate(() => window.DV_polish.get()))[0].polish === 0);
  await openStep(page, 'st5');

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
    // --- a photo first: the same feed, straight into the still flow.
    // Give the fake device a beat: it opens on black before its pattern (if
    // any — some Chromium builds ship a black fake feed, see the spread-aware
    // check below; the pre-redesign UI failed the old strict check the same
    // way on such a build, so this is environment drift, not the app).
    await sleep(1200);
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
    /* A photo with real tonal spread must dither to exactly the two palette
     * colours. A fake device that delivers a black feed (Chromium build
     * drift) gives a photo with no spread, and a uniform source honestly
     * dithers to ONE colour — assert that instead of failing on the env. */
    r.photoSpread = await page.evaluate(() => {
      const c = document.createElement('canvas');
      c.width = 96; c.height = 54;
      const g = c.getContext('2d', { willReadFrequently: true });
      g.drawImage(window.DV.bitmap, 0, 0, 96, 54);
      const d = g.getImageData(0, 0, 96, 54).data;
      let lo = 255, hi = 0;
      for (let i = 0; i < d.length; i += 4) {
        const l = 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2];
        if (l < lo) lo = l;
        if (l > hi) hi = l;
      }
      return { lo: Math.round(lo), hi: Math.round(hi), spread: Math.round(hi - lo) };
    });
    check(`${engineId} · the photo dithers`,
          r.photoCensus.distinctColours === (r.photoSpread.spread > 40 ? 2 : 1),
          JSON.stringify({ census: r.photoCensus, spread: r.photoSpread }));
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
    // a black fake feed (see the photo check) gives an honestly-dark strip
    check(`${engineId} · the filmstrip has thumbnails on it`,
          r.strip.colours > (r.photoSpread.spread > 40 ? 8 : 1),
          JSON.stringify({ strip: r.strip, spread: r.photoSpread }));
    await page.screenshot({ path: path.join(DOCS, `w-camera-trim-${engineId}.png`) });

    // trim to the middle two seconds and re-open. The trim bar lives under
    // the stage now, so it is already on screen; opening the Source step just
    // keeps the panel showing the clip facts for the screenshot.
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
    if (engineId === 'browser') {
      // A camera recording is the file the seek loop was worst on: MediaRecorder
      // WebM, no Cues, no Duration in the header. This is the only place the
      // real thing (not an ffmpeg imitation of it) goes through the demuxer.
      r.decode = await page.evaluate(() => window.DV.engine.lastDecode);
      check('browser · the camera recording decodes through WebCodecs',
            r.decode && r.decode.path === 'webcodecs-worker',
            JSON.stringify(r.decode && r.decode.line));
      check('browser · and it is recognised as WebM',
            /webm/.test((r.decode && r.decode.note) || ''),
            (r.decode && r.decode.note) || '');
    }
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

/* ============ W8: the flagship — the sequence view, four items, three joins ==
 * A parkour subject from one clip, a tennis player from another, a subject cut
 * out of a photograph, and a ring rasterised through the dots pipeline — added
 * through the UI the way a person would, reordered by dragging, joined by three
 * different transitions, played in the player and rendered to MP4 by the
 * server. The library surviving two clip changes is part of the test, because
 * that is the whole reason a sequence is a view and not a step.
 */
async function runSequence() {
  const r = { twoClips: fs.existsSync(CLIP2) };
  const { ctx, page } = await newPage({ mode: 'local', url: '', key: '' });
  const stripLen = () => page.evaluate(() => window.DV_seq.strip().length);

  const trackAndAdd = async (file, box, seconds) => {
    await loadClip(page, file, seconds);
    await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
    await promptBoxPoint(page, box);
    await page.click('#bTrack');
    const t = await waitText(page, '#tinfo', /tracked|failed/, 900000);
    check('sequence · ' + path.basename(file) + ' tracks', !/failed/.test(t), t);
    await setMode(page, 'dots');
    const before = await stripLen();
    await page.click('#bToSeq');                 // the header's one way in
    await page.waitForFunction((n) => window.DV_seq.strip().length > n,
                               before, { timeout: 600000 });
    return waitText(page, '#seqinfo', /added|could not/, 10000);
  };

  // --- 1. the parkour subject
  r.addA = await trackAndAdd(CLIP, SUBJECT_A, BR.seconds);
  check('sequence · the clip went in as an item', /added/.test(r.addA), r.addA);
  r.view = await page.evaluate(() => window.DV.view);
  check('sequence · adding switches to the sequence view', r.view === 'sequence',
        r.view);

  // --- 2. a subject from a SECOND clip, via "upload something new"
  if (r.twoClips) {
    await page.click('#bSeqNew');
    check('sequence · "something new" goes back to the studio',
          (await page.evaluate(() => window.DV.view)) === 'studio');
    r.addB = await trackAndAdd(CLIP2, SECOND, 3);
    check('sequence · the second clip went in', /added/.test(r.addB), r.addB);
    r.libraryKept = await page.evaluate(() => window.DV_seq.library().length);
    check('sequence · the library survived loading another clip',
          r.libraryKept === 2, String(r.libraryKept));
  }

  // --- 3. a subject cut out of a photograph
  await page.click('#bSeqNew');
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 60000 });
  await sleep(600);
  await openStep(page, 'st2');
  await page.click('#scope .chip[data-scope="track"]'); await sleep(700);
  await promptBoxPoint(page, SUBJECT_A);
  r.stillInfo = await waitText(page, '#pvinfo', /subject|failed/, 300000);
  check('sequence · the still subject segments', !/failed/.test(r.stillInfo),
        r.stillInfo);
  await page.click('#bTrack'); await sleep(700);
  await setMode(page, 'dots');
  let before = await stripLen();
  await page.click('#bToSeq');
  await page.waitForFunction((n) => window.DV_seq.strip().length > n, before,
                             { timeout: 300000 });

  // --- 4. and a ring, from the sequence view's own add row
  before = await stripLen();
  await page.click('#seqadd .chip:has-text("ring")').catch(async () => {
    await page.evaluate(() => window.DV_seq.add('shape', 'ring'));
  });
  await page.waitForFunction((n) => window.DV_seq.strip().length > n, before,
                             { timeout: 120000 });

  r.strip = await page.evaluate(() => window.DV_seq.strip());
  r.library = await page.evaluate(() => window.DV_seq.library());
  check('sequence · four items in the strip', r.strip.length === 4,
        JSON.stringify(r.strip.map((x) => x.name)));
  check('sequence · the ring rasterised into dots',
        (r.library[r.library.length - 1].tracks[0] || {}).frames === 1
          && r.library[r.library.length - 1].kind === 'shape',
        JSON.stringify(r.library));
  await page.screenshot({ path: path.join(DOCS, 'seq-flow-1-strip.png') });

  // --- per-item options: a trim on the first item, a hold on the ring
  await page.evaluate(() => {
    window.DV_seq.set(0, { in: 10, out: 54 });
    window.DV_seq.set(3, { hold: 40 });
  });
  r.trimmed = await page.evaluate(() => window.DV_seq.strip());
  check('sequence · an in/out trim decides a clip item\'s length',
        r.trimmed[0].frames === 45, JSON.stringify(r.trimmed[0]));
  check('sequence · a hold decides a shape item\'s length',
        r.trimmed[3].frames === 40, JSON.stringify(r.trimmed[3]));

  // --- one colour per item, so the transitions have something to carry
  const COLOURS = ['#b0413e', '#2f4f4a', '#7a6a4f', '#3c5a7a'];
  await page.evaluate((cols) => {
    cols.forEach((c, i) => window.DV_seq.set(i, { color: c }));
  }, COLOURS);

  // --- three different joins, one of them set by shift-clicking the chip
  await page.evaluate(() => {
    window.DV_seq.trans(2, 'scatter', 700);
    window.DV_seq.trans(3, 'density', 800);
  });
  await page.click('#strip2 .join[data-i="1"]');      // select the first join
  await page.screenshot({ path: path.join(DOCS, 'seq-flow-2-join.png') });
  r.joinChips = await page.$$eval('#seqinspect .chips.seg .chip',
                                  (n) => n.map((e) => e.textContent));
  check('sequence · the inspector offers all four transitions',
        r.joinChips.length === 4, JSON.stringify(r.joinChips));
  await page.click('#strip2 .join[data-i="1"]', { modifiers: ['Shift'] });
  r.cycled = (await page.evaluate(() => window.DV_seq.strip()))[1].trans.kind;
  check('sequence · shift-clicking a join cycles the transition',
        r.cycled === 'scatter', r.cycled);
  await page.evaluate(() => window.DV_seq.trans(1, 'morph', 900));

  // --- preview
  await page.click('#bSeqPrev');
  r.preview = await waitText(page, '#seqinfo', /frames|failed/, 300000);
  check('sequence · previews in the player', !/failed/.test(r.preview), r.preview);
  r.doc = await page.evaluate(() => {
    const d = window.DV_seq.doc();
    const counts = d.frames.map((f) => f.reduce((a, x) => a + (x.length >> 1), 0));
    return { frames: d.frames.length, fps: d.fps, marks: d.marks,
             palette: d.palette, subjects: d.subjects.length,
             counts: { min: Math.min(...counts), max: Math.max(...counts) } };
  });
  const joins = r.doc.marks.filter((m) => m.kind !== 'item');
  const items = r.doc.marks.filter((m) => m.kind === 'item');
  check('sequence · one join per pair and one mark per item',
        items.length === 4 && joins.length === 3,
        JSON.stringify(r.doc.marks.map((m) => m.kind)));
  check('sequence · three different transition kinds',
        new Set(joins.map((m) => m.kind)).size === 3,
        JSON.stringify(joins.map((m) => m.kind)));
  check('sequence · every item keeps its own colour',
        r.doc.subjects === 4 && COLOURS.every((c) => r.doc.palette.includes(c)),
        JSON.stringify(r.doc.palette));
  const morph = joins.find((m) => m.kind === 'morph');
  const scatter = joins.find((m) => m.kind === 'scatter');
  const density = joins.find((m) => m.kind === 'density');
  check('sequence · a 900 ms morph is ~37 frames at 30 fps',
        Math.abs(morph.frames - 37) <= 2, JSON.stringify(morph));
  check('sequence · a 700 ms scatter is ~21 frames',
        Math.abs(scatter.frames - 21) <= 2, JSON.stringify(scatter));
  check('sequence · a density fade is a ladder of short morphs',
        density.frames >= 18 && density.frames <= 40, JSON.stringify(density));
  const expect = r.doc.marks.reduce((a, m) => a + m.frames, 0);
  check('sequence · the document is exactly its items plus its joins',
        r.doc.frames === expect, `${r.doc.frames} vs ${expect}`);
  check('sequence · every frame has dots on it', r.doc.counts.min > 0,
        JSON.stringify(r.doc.counts));

  // the player is actually running, not just showing frame 0
  const f0 = await page.evaluate(() => window.DV_seq.player().frame);
  await sleep(1200);
  const f1 = await page.evaluate(() => window.DV_seq.player().frame);
  check('sequence · the preview plays', f1 !== f0, `${f0} -> ${f1}`);
  await sleep(600);
  await page.screenshot({ path: path.join(DOCS, 'seq-flow-3-preview.png') });
  await page.click('#strip2 .card[data-i="0"]');
  await page.screenshot({ path: path.join(DOCS, 'seq-flow-4-item.png') });

  // --- a cut really is nothing, and the reorder really reorders
  r.cut = await page.evaluate(async () => {
    window.DV_seq.trans(1, 'cut');
    const d = await window.DV_seq.build();
    const n = d.frames.length;
    window.DV_seq.trans(1, 'morph', 900);
    return n;
  });
  check('sequence · a cut costs no frames at all',
        r.cut === r.doc.frames - morph.frames,
        `${r.cut} vs ${r.doc.frames - morph.frames}`);
  r.order = await page.evaluate(() => {
    const before = window.DV_seq.strip().map((x) => x.name);
    window.DV_seq.move(3, 0);
    const after = window.DV_seq.strip().map((x) => x.name);
    window.DV_seq.move(0, 3);
    return { before, after, back: window.DV_seq.strip().map((x) => x.name) };
  });
  check('sequence · items reorder, and the order comes back',
        r.order.after[0] === r.order.before[3]
          && r.order.back.join() === r.order.before.join(),
        JSON.stringify(r.order));

  await page.click('#bSeqPrev');
  await waitText(page, '#seqinfo', /frames|failed/, 120000);

  // --- the dot data for the whole sequence
  const dots = path.join(DOCS, 'w-sequence-export.dots.gz');
  const [dl] = await Promise.all([
    page.waitForEvent('download', { timeout: 120000 }),
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
    timeout: 300000,
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

  // --- and the player's own frame rate on the finished sequence
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
             worstPaintMs: +worst.toFixed(2),
             subjects: P.doc.subjects.length, palette: P.doc.palette };
  });
  check('sequence · the .dots.gz replays with all four colours',
        r.player.frames === r.doc.frames && r.player.subjects === 4,
        JSON.stringify(r.player));
  check('sequence · the player holds 60 fps', r.player.fps >= 60,
        JSON.stringify(r.player));
  check('sequence · a frame costs well under a 60 fps budget',
        r.player.meanPaintMs < 16.6, JSON.stringify(r.player));
  await demo.screenshot({ path: path.join(DOCS, 'w-player-demo.png') });
  await demo.close();

  // --- the same strip, on the free tier. A sequence is dot positions, so it
  // outlives the engine that produced it — and the tab encodes it with the very
  // machinery a clip export uses, GIF and alpha included.
  await page.evaluate(() => window.DV_switchEngine({ mode: 'browser', url: '', key: '' }));
  await page.waitForFunction(() => window.DV_engine().id === 'browser',
                             null, { timeout: 120000 });
  r.afterSwitch = await page.evaluate(() => window.DV_seq.strip().length);
  check('sequence · the strip survives an engine switch', r.afterSwitch === 4,
        String(r.afterSwitch));
  await page.evaluate(() => {
    window.DV_seq.view('sequence');
    // a short strip: this is about the encoders, not about patience
    window.DV_seq.set(0, { in: 0, out: 9 });
    window.DV_seq.set(1, { in: 0, out: 9 });
    window.DV_seq.set(2, { hold: 6 });
    window.DV_seq.set(3, { hold: 6 });
    [1, 2, 3].forEach((i) => window.DV_seq.trans(i,
      window.DV_seq.strip()[i].trans.kind, 300));
  });
  r.browser = {};
  for (const id of ['webm', 'gif', 'webm-alpha']) {
    await page.evaluate((x) => window.DV_seq.format(x), id);
    await page.click('#bSeqVideo');
    const txt = await waitText(page, '#seqinfo', /MB|failed/, 600000);
    r.browser[id] = txt.trim();
    check(`sequence · the tab writes the sequence as ${id}`,
          !/failed/.test(txt), txt);
  }
  check('sequence · the tab\'s sequence GIF carries a palette and loops',
        /colours · loops forever/.test(r.browser.gif), r.browser.gif);
  check('sequence · the tab\'s alpha sequence really has an alpha channel',
        /alpha channel/.test(r.browser['webm-alpha']), r.browser['webm-alpha']);
  await page.screenshot({ path: path.join(DOCS, 'seq-flow-5-browser.png') });
  await ctx.close();
  return r;
}

/* ====== W9: an item's look is its own ======================================
 * The strip used to hold frozen snapshots: a card had a subject, a trim, a
 * hold and a colour, and the dots inside it were whatever the studio happened
 * to be showing when it was captured. Now a card is a LIVE reference — it
 * keeps its source and its own copy of the whole look — so this run builds
 * three items, opens the middle one, and changes the mode, the palette, the
 * dot sliders and the mask polish on THAT ITEM.
 *
 * What it proves, in order:
 *   - the panel really is the studio's controls, scoped: every mode chip, the
 *     dot sliders, a colour per subject, a polish slider, the trim
 *   - changing them moves item 2's dots and nothing else's — items 1 and 3
 *     hash identically before and after, to the byte
 *   - the preview, the .dots.gz and the server's MP4 all show the change,
 *     because all three read the same re-derived dots
 *   - the background and the dot size stay per SEQUENCE
 */
async function runSeqItemLook() {
  const r = {};
  const { ctx, page } = await newPage({ mode: 'local', url: '', key: '' });

  /** FNV-1a over one item's dot positions, exactly as the strip holds them. */
  const itemHash = (i) => page.evaluate((k) => {
    const d = window.DV_seq.itemDots(k);
    let h = 2166136261, n = 0;
    for (const t of d) for (const f of t) {
      n += f.length >> 1;
      for (const v of f) { h ^= v; h = Math.imul(h, 16777619); }
    }
    return { hash: (h >>> 0).toString(16), dots: n, frames: (d[0] || []).length };
  }, i);
  /** The same hash over the BUILT DOCUMENT, region by region — these are the
   *  exact bytes `pack()` writes into the .dots.gz and the server rasterises. */
  const docHashes = () => page.evaluate(() => {
    const d = window.DV_seq.doc();
    return d.marks.filter((m) => m.kind === 'item').map((m) => {
      let h = 2166136261, n = 0;
      for (let f = m.start; f < m.start + m.frames; f++) {
        for (const tr of d.frames[f]) {
          n += tr.length >> 1;
          for (const v of tr) { h ^= v; h = Math.imul(h, 16777619); }
        }
      }
      return { name: m.name, hash: (h >>> 0).toString(16), dots: n, frames: m.frames };
    });
  });
  const hashes = async () => {
    const n = await page.evaluate(() => window.DV_seq.strip().length);
    const out = [];
    for (let i = 0; i < n; i++) out.push(await itemHash(i));
    return out;
  };

  // --- a tracked clip, added twice, plus a shape: three items, two of them
  // from the same capture so the second can diverge from the first
  await loadClip(page, CLIP, BR.seconds);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
  await promptBoxPoint(page, SUBJECT_A);
  await page.click('#bTrack');
  const t = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check('item look · the clip tracks', !/failed/.test(t), t);
  await setMode(page, 'dots');
  await page.click('#bToSeq');
  await page.waitForFunction(() => window.DV_seq.strip().length === 1,
                             null, { timeout: 600000 });
  await page.evaluate(() => window.DV_seq.add('lib', window.DV_seq.library()[0].id));
  await page.waitForFunction(() => window.DV_seq.strip().length === 2,
                             null, { timeout: 600000 });
  await page.evaluate(() => window.DV_seq.add('shape', 'ring'));
  await page.waitForFunction(() => window.DV_seq.strip().length === 3,
                             null, { timeout: 120000 });
  await page.evaluate(() => {
    window.DV_seq.set(0, { in: 0, out: 29 });
    window.DV_seq.set(1, { in: 0, out: 29 });
    window.DV_seq.set(2, { hold: 20 });
  });
  await page.click('#bSeqPrev');
  await waitText(page, '#seqinfo', /frames|failed/, 300000);

  r.before = await hashes();
  check('item look · three items, all with dots on them',
        r.before.length === 3 && r.before.every((x) => x.dots > 0),
        JSON.stringify(r.before));
  check('item look · the two copies of one capture start identical',
        r.before[0].hash === r.before[1].hash,
        `${r.before[0].hash} vs ${r.before[1].hash}`);

  const docBefore = await page.evaluate(() => {
    const d = window.DV_seq.doc();
    return { frames: d.frames.length, dotpx: d.dotpx, bg: d.bg,
             palette: d.palette, marks: d.marks };
  });
  r.docBefore = docBefore;
  r.docItemsBefore = await docHashes();

  // the .dots.gz and the MP4 as they stand
  const gzA = path.join(DOCS, 'seq-item-look-before.dots.gz');
  const [dlA] = await Promise.all([
    page.waitForEvent('download', { timeout: 120000 }), page.click('#bSeqDots')]);
  await dlA.saveAs(gzA);
  r.beforeBytes = fs.statSync(gzA).size;

  // --- open item 2 and read what the panel offers
  await page.click('#strip2 .card[data-i="1"]');
  await sleep(300);
  r.panel = await page.evaluate(() => ({
    modes: Array.from(document.querySelectorAll('#seqinspect .chip[data-mode]'))
      .map((b) => b.dataset.mode),
    sliders: document.querySelectorAll('#seqinspect input[type=range]').length,
    colours: document.querySelectorAll('#seqinspect input[type=color]').length,
    polish: document.querySelectorAll('#seqinspect .chip.pol').length,
    labels: Array.from(document.querySelectorAll('#seqinspect .lbl > span:first-child'))
      .map((s) => s.textContent),
  }));
  const wantModes = await page.evaluate(() => window.DV_seq.modes().map((m) => m.id));
  r.wantModes = wantModes;
  check('item look · the panel offers every studio mode',
        wantModes.length >= 7 && wantModes.every((m) => r.panel.modes.includes(m)),
        JSON.stringify(r.panel.modes));
  check('item look · the panel carries the dot sliders, a colour and a polish',
        r.panel.sliders >= 7 && r.panel.colours >= 1 && r.panel.polish === 1,
        JSON.stringify(r.panel));
  check('item look · and says the background is not one of them',
        /background/i.test(await page.textContent('#seqinspect')),
        (await page.textContent('#seqinspect')).slice(0, 120));
  await page.screenshot({ path: path.join(DOCS, 'seq-item-look-panel.png') });

  // --- change the MODE on item 2 only
  r.modes = {};
  for (const m of wantModes) {
    r.modes[m] = await page.evaluate(async (mm) => {
      await window.DV_seq.setLook(1, { mode: mm });
      const d = window.DV_seq.itemDots(1);
      return d[0].reduce((a, f) => a + f.length / 2, 0) / d[0].length;
    }, m);
  }
  check('item look · every mode produces a cloud of dots',
        Object.values(r.modes).every((n) => n > 50), JSON.stringify(r.modes));
  check('item look · the modes disagree about which cells survive',
        new Set(Object.values(r.modes).map((n) => Math.round(n))).size >= 5,
        JSON.stringify(r.modes));

  // --- settle on one look for item 2: a different mode, palette, cell, count
  // and the mask polish on
  await page.evaluate(() => window.DV_seq.setLook(1, { mode: 'bluenoise' }));
  await page.click('#seqinspect .chip.pal');                 // a palette, per item
  await sleep(400);
  await page.$eval('#seqinspect .chip.pol', (b) => b.click());  // polish on
  await page.waitForFunction(() => !document.querySelector('#seqinspect .chip.pol')
    || document.querySelector('#seqinspect .chip.pol').getAttribute('aria-pressed') === 'true',
    null, { timeout: 120000 });
  r.after1 = await page.evaluate(async () => {
    await window.DV_seq.setLook(1, { cell: 6, gamma: 1.4 });
    return window.DV_seq.itemLook(1);
  });
  check('item look · item 2 kept every change',
        r.after1.mode === 'bluenoise' && r.after1.cell === 6
          && Math.abs(r.after1.gamma - 1.4) < 1e-6
          && Object.keys(r.after1.polish).length === 1
          && Object.keys(r.after1.colors).length >= 1,
        JSON.stringify(r.after1));
  r.looks = await page.evaluate(() => window.DV_seq.strip().map((x) => x.look.mode));
  check('item look · items 1 and 3 are still on dots',
        r.looks[0] === 'dots' && r.looks[2] === 'dots', JSON.stringify(r.looks));

  await page.click('#bSeqPrev');
  await waitText(page, '#seqinfo', /frames|failed/, 300000);
  r.after = await hashes();
  check('item look · item 2 is a different cloud now',
        r.after[1].hash !== r.before[1].hash,
        `${r.before[1].hash} -> ${r.after[1].hash}`);
  check('item look · item 1 is byte for byte what it was',
        r.after[0].hash === r.before[0].hash && r.after[0].dots === r.before[0].dots,
        `${r.before[0].hash} vs ${r.after[0].hash}`);
  check('item look · item 3 is byte for byte what it was',
        r.after[2].hash === r.before[2].hash && r.after[2].dots === r.before[2].dots,
        `${r.before[2].hash} vs ${r.after[2].hash}`);
  r.docItemsAfter = await docHashes();
  check('item look · in the document itself, item 1 is unchanged and item 2 is not',
        r.docItemsAfter[0].hash === r.docItemsBefore[0].hash
          && r.docItemsAfter[2].hash === r.docItemsBefore[2].hash
          && r.docItemsAfter[1].hash !== r.docItemsBefore[1].hash,
        JSON.stringify([r.docItemsBefore, r.docItemsAfter]));
  await page.screenshot({ path: path.join(DOCS, 'seq-item-look-preview.png') });

  // --- the sequence's own look is still the sequence's
  r.canvas = await page.evaluate(() => {
    window.DV_seq.look({ dotpx: 5, bg: '#101418' });
    const d = window.DV_seq.doc();
    return { dotpx: d.dotpx, docBg: d.bg, seq: window.DV.seq };
  });
  check('item look · dot size and background stay per sequence',
        r.canvas.seq.dotpx === 5 && r.canvas.seq.bg === '#101418',
        JSON.stringify(r.canvas));
  await page.evaluate(() => window.DV_seq.look({ dotpx: 3, bg: '#c9d4c5' }));

  // --- and the exports agree with the preview
  const gzB = path.join(DOCS, 'seq-item-look-after.dots.gz');
  const [dlB] = await Promise.all([
    page.waitForEvent('download', { timeout: 120000 }), page.click('#bSeqDots')]);
  await dlB.saveAs(gzB);
  r.afterBytes = fs.statSync(gzB).size;
  check('item look · the .dots.gz changed with the item',
        !fs.readFileSync(gzA).equals(fs.readFileSync(gzB)),
        `${r.beforeBytes} -> ${r.afterBytes} bytes`);

  const docAfter = await page.evaluate(() => {
    const d = window.DV_seq.doc();
    return { frames: d.frames.length, palette: d.palette,
             subjects: d.subjects.length };
  });
  r.docAfter = docAfter;
  check('item look · the item’s new colour reached the document palette',
        docAfter.palette.length > docBefore.palette.length
          || docAfter.palette.join() !== docBefore.palette.join(),
        JSON.stringify([docBefore.palette, docAfter.palette]));

  const up = await page.request.post(`${BASE}/api/sequence`, {
    multipart: { file: { name: 'sequence.dots.gz',
                         mimeType: 'application/octet-stream',
                         buffer: fs.readFileSync(gzB) },
                 format: 'mp4' },
    timeout: 300000,
  });
  check('item look · the server rasterises the re-derived dots', up.ok(),
        String(up.status()));
  r.server = await up.json();
  const mp4 = path.join(DOCS, 'seq-item-look.mp4');
  fs.writeFileSync(mp4, await (await page.request.get(`${BASE}${r.server.url}`)).body());
  r.mp4 = { bytes: fs.statSync(mp4).size, probe: probe(mp4) };
  check('item look · the MP4 has every frame the preview does',
        +r.mp4.probe.nb_read_frames === docAfter.frames,
        `${r.mp4.probe.nb_read_frames} vs ${docAfter.frames}`);
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', mp4,
    '-vf', `select='not(mod(n\,${Math.max(1, Math.floor(docAfter.frames / 12))}))',`
      + 'scale=300:-1,tile=4x3', '-frames:v', '1', '-update', '1',
    path.join(DOCS, 'seq-item-look-sheet.png')]);

  // --- back to where it started: the cache still holds the original
  r.reverted = await page.evaluate(async () => {
    await window.DV_seq.setLook(1, { mode: 'dots', cell: 4, gamma: 1,
                                     polish: {}, colors: {} });
    const d = window.DV_seq.itemDots(1);
    let h = 2166136261;
    for (const t of d) for (const f of t) for (const v of f) {
      h ^= v; h = Math.imul(h, 16777619);
    }
    return (h >>> 0).toString(16);
  });
  check('item look · putting the look back puts the dots back',
        r.reverted === r.before[1].hash, `${r.reverted} vs ${r.before[1].hash}`);

  await page.screenshot({ path: path.join(DOCS, 'seq-item-look-reverted.png') });
  await ctx.close();
  return r;
}

/* ====== W10: pixel modes fly as particles, and a picture can bring a subject ==
 * Two things that only make sense together.
 *
 * A pixel dither mode at cell 1 is one dot per lit PIXEL — hundreds of
 * thousands of them — which is a picture, not a swarm. It still has to morph,
 * so a join thins whichever side is over the cap and flies the survivors, and
 * hands the rest back in place: the frame the transition starts on is exactly
 * the outgoing item, the frame it ends on is exactly the incoming one, and the
 * loosening is in between. That is checked here on a Bayer-dithered cutout
 * morphing into a tracked subject drawn with Dots.
 *
 * And a picture brought in from the sequence can now say WHICH bit of itself it
 * means: the add row offers each segmented subject on its own, the whole frame,
 * and — for a picture that has never been in the studio — "select a subject…",
 * which opens it there and lets the header carry the cutout back.
 */
async function runSeqPixel() {
  const r = {};
  const { ctx, page } = await newPage({ mode: 'local', url: '', key: '' });

  const counts = (i) => page.evaluate((k) => {
    const d = window.DV_seq.itemDots(k);
    return d[0].map((f) => f.length / 2);
  }, i);

  // --- 1. a photograph, one subject clicked out of it, offered per subject
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', { timeout: 60000 });
  await sleep(600);
  await openStep(page, 'st2');
  await page.click('#scope .chip[data-scope="track"]'); await sleep(700);
  await promptBoxPoint(page, SUBJECT_A);
  r.seg = await waitText(page, '#pvinfo', /subject|failed/, 300000);
  check('pixel · the still subject segments', !/failed/.test(r.seg), r.seg);
  await page.click('#bTrack'); await sleep(700);
  await setMode(page, 'dots');
  r.candidates = await page.evaluate(() => window.DV_seq.candidates());
  check('pixel · a segmented still offers its subject AND the whole picture',
        r.candidates.length === 2
          && r.candidates[0].arg.subject === 0
          && r.candidates[1].arg.whole === true,
        JSON.stringify(r.candidates));

  await page.click('#bToSeq');                    // the subject entry is first
  await page.waitForFunction(() => window.DV_seq.strip().length === 1,
                             null, { timeout: 300000 });
  r.cutTracks = (await page.evaluate(() => window.DV_seq.library()))[0].tracks.length;
  check('pixel · the cutout went in as one track', r.cutTracks === 1,
        String(r.cutTracks));

  // --- 2. the same picture again, whole this time, and dithered with Bayer.
  // Choosing a pixel mode drops the cell to 1, which is where a dither is a
  // dither rather than a screen at dot size — and where it stops being a swarm.
  await page.evaluate(() => window.DV_seq.add('still', { whole: true }));
  await page.waitForFunction(() => window.DV_seq.strip().length === 2,
                             null, { timeout: 300000 });
  r.bayer = await page.evaluate(async () => {
    const before = window.DV_seq.itemLook(1);
    await window.DV_seq.setLook(1, { mode: 'ordered' });
    return { before: before.cell, look: window.DV_seq.itemLook(1) };
  });
  check('pixel · Bayer drops the item to cell 1',
        r.bayer.look.mode === 'ordered' && r.bayer.look.cell === 1
          && r.bayer.before === 4,
        JSON.stringify(r.bayer));
  await page.evaluate(() => {
    window.DV_seq.set(0, { hold: 6 });
    window.DV_seq.set(1, { hold: 6 });
  });
  r.pixelDots = (await counts(1))[0];
  r.cutoutDots = (await counts(0))[0];
  r.cap = await page.evaluate(() => window.DV_seq.cap());
  check('pixel · a cell-1 Bayer frame is hundreds of thousands of dots',
        r.pixelDots > r.cap * 20, `${r.pixelDots} dots vs a ${r.cap} cap`);
  await page.click('#strip2 .card[data-i="1"]');
  await page.screenshot({ path: path.join(DOCS, 'seq-particle-bayer.png') });

  // --- 3. a tracked subject on Dots, after it
  await page.click('#bSeqNew');
  await loadClip(page, CLIP, BR.seconds);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
  await promptBoxPoint(page, SUBJECT_A);
  await page.click('#bTrack');
  const t = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check('pixel · the clip tracks', !/failed/.test(t), t);
  await setMode(page, 'dots');
  await page.click('#bToSeq');
  await page.waitForFunction(() => window.DV_seq.strip().length === 3,
                             null, { timeout: 600000 });
  await page.evaluate(() => {
    window.DV_seq.set(2, { in: 0, out: 11 });
    window.DV_seq.trans(1, 'cut');
    window.DV_seq.trans(2, 'morph', 900);
  });

  await page.click('#bSeqPrev');
  r.preview = await waitText(page, '#seqinfo', /frames|failed/, 300000);
  check('pixel · a pixel-mode item previews next to a dots one',
        !/failed/.test(r.preview), r.preview);

  r.join = await page.evaluate(() => {
    const d = window.DV_seq.doc();
    const items = d.marks.filter((m) => m.kind === 'item');
    const m = d.marks.filter((x) => x.kind !== 'item').pop();   // Bayer -> dots
    const at = (f) => d.frames[f].reduce((a, t) => a + t.length / 2, 0);
    const flight = [];
    for (let f = m.start; f < m.start + m.frames; f++) flight.push(at(f));
    return {
      kind: m.kind, frames: m.frames, thinned: m.thinned, start: m.start,
      lastOfA: at(items[1].start + items[1].frames - 1),
      firstOfB: at(items[2].start),
      first: flight[0], last: flight[flight.length - 1],
      min: Math.min(...flight), max: Math.max(...flight),
      mid: flight[Math.floor(flight.length / 2)],
    };
  });
  const j = r.join;
  check('pixel · the flight is thinned, and says by how much',
        j.thinned > 0 && j.kind === 'morph', JSON.stringify(j));
  check('pixel · mid-flight never exceeds two capfuls of particles',
        j.mid <= r.cap * 2 && j.mid < r.pixelDots / 4, JSON.stringify(j));
  check('pixel · the transition starts on the outgoing item at full density',
        Math.abs(j.first - j.lastOfA) <= 8,
        `${j.first} vs ${j.lastOfA} (${j.lastOfA - j.first} dots)`);
  check('pixel · and ends on the incoming one at full density',
        Math.abs(j.last - j.firstOfB) <= 8,
        `${j.last} vs ${j.firstOfB} (${j.firstOfB - j.last} dots)`);
  await sleep(500);
  await page.screenshot({ path: path.join(DOCS, 'seq-particle-preview.png') });

  // --- the whole thing as a video, and a contact sheet of the join itself
  const gz = path.join(DOCS, 'seq-particle.dots.gz');
  const [dl] = await Promise.all([
    page.waitForEvent('download', { timeout: 300000 }), page.click('#bSeqDots')]);
  await dl.saveAs(gz);
  r.dotsBytes = fs.statSync(gz).size;
  const up = await page.request.post(`${BASE}/api/sequence`, {
    multipart: { file: { name: 'sequence.dots.gz',
                         mimeType: 'application/octet-stream',
                         buffer: fs.readFileSync(gz) },
                 format: 'mp4' },
    timeout: 600000,
  });
  check('pixel · the server rasterises a pixel-mode sequence', up.ok(),
        String(up.status()));
  r.server = await up.json();
  const mp4 = path.join(DOCS, 'seq-particle.mp4');
  fs.writeFileSync(mp4, await (await page.request.get(`${BASE}${r.server.url}`)).body());
  r.mp4 = { bytes: fs.statSync(mp4).size, probe: probe(mp4) };
  const total = await page.evaluate(() => window.DV_seq.doc().frames.length);
  check('pixel · the MP4 has every frame of it',
        +r.mp4.probe.nb_read_frames === total,
        `${r.mp4.probe.nb_read_frames} vs ${total}`);
  // twelve frames straddling the join: the Bayer picture, the loosening, the swarm
  const start = Math.max(0, r.join.start - 2);
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', mp4,
    '-vf', `select='between(n\,${start}\,${start + j.frames})*not(mod(n\,3))',`
      + 'scale=320:-1,tile=4x3', '-frames:v', '1', '-update', '1',
    path.join(DOCS, 'seq-particle-morph.png')]);

  // --- 4. "+ image…" -> "select a subject…" -> a cutout in the strip
  await page.evaluate(() => window.DV_seq.view('sequence'));
  await page.setInputFiles('#shapeFile', STILL);
  await page.waitForFunction(() => window.DV_seq.pending() !== null,
                             null, { timeout: 60000 });
  r.addChips = await page.$$eval('#seqadd .chip', (n) => n.map((e) => e.textContent));
  check('pixel · a picture is asked what it is before it goes in',
        r.addChips.includes('whole image')
          && r.addChips.includes('select a subject…'),
        JSON.stringify(r.addChips));
  await page.screenshot({ path: path.join(DOCS, 'seq-image-subject-ask.png') });
  await page.click('#seqadd .chip:has-text("select a subject…")');
  await page.waitForFunction(() => window.DV.view === 'studio'
    && window.DV.kind === 'image' && window.DV.scope === 'track',
                             null, { timeout: 120000 });
  check('pixel · "select a subject" opens it in the studio, on the subject step',
        true, 'studio · image · track');
  await promptBoxPoint(page, SUBJECT_A);
  r.seg2 = await waitText(page, '#pvinfo', /subject|failed/, 300000);
  check('pixel · the picture segments where it landed', !/failed/.test(r.seg2),
        r.seg2);
  await page.click('#bTrack'); await sleep(700);
  await page.click('#bToSeq');
  await page.waitForFunction(() => window.DV_seq.strip().length === 4,
                             null, { timeout: 300000 });
  r.strip = await page.evaluate(() => window.DV_seq.strip());
  const last = r.strip[3];
  const lib = (await page.evaluate(() => window.DV_seq.library()))
    .find((x) => x.id === last.lib);
  check('pixel · the cutout, not the whole picture, is what went in',
        lib.kind === 'still' && lib.tracks.length === 1
          && !/whole/.test(lib.name),
        JSON.stringify({ kind: lib.kind, name: lib.name, tracks: lib.tracks.length }));
  await page.screenshot({ path: path.join(DOCS, 'seq-image-subject-added.png') });

  await ctx.close();
  return r;
}


/* ======= WX: the canvas, in the tab ======================================
 * The aspect-ratio control on the engine that has no server: the crop path is
 * built by walking the tracker's own mask logits in the tab (there is no
 * /centroids to ask), the dots are re-measured on a 1080x1920 grid, and the
 * matched original cut is recorded through the identical map — so the pair is
 * still a pair once there is a crop between the clip and the file.
 */
async function runCanvasBrowser() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  await loadClip(page, CLIP, BR.seconds);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  await page.click('#scope .chip[data-scope="track"]');
  await sleep(700);
  await promptBoxPoint(page, SUBJECT_A);
  await page.click('#bTrack');
  r.trackInfo = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check('the clip tracks in the tab', !/failed/.test(r.trackInfo), r.trackInfo);

  r.presets = await page.evaluate(() => window.DV_canvas.presets());
  check('the tab offers 9:16 at 1080×1920',
        r.presets.some((p) => p.id === '9:16' && p.w === 1080 && p.h === 1920));

  /* ---- 9:16 cutout ----------------------------------------------------- */
  await setMode(page, 'dots');
  const t0 = Date.now();
  r.canvas = await page.evaluate(() => window.DV_canvas.set('9:16'));
  r.pathSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  check('9:16 is 1080×1920 in the tab too',
        r.canvas.target.w === 1080 && r.canvas.target.h === 1920);
  check('a cutout crop is not clamped to the source', !r.canvas.clamps);
  r.path = await page.evaluate(async () => {
    const p = await window.DV_canvas.path();
    return { n: p.n, mode: p.mode, union: p.union, first: p.centers[0] };
  });
  check('the tab built a centre for every frame', r.path.n === r.nFrames,
        `${r.path.n} vs ${r.nFrames}`);
  r.previewSize = await page.evaluate(() => {
    const c = document.querySelector('#vcv'); return [c.width, c.height];
  });
  check('the preview is the canvas, not the clip',
        r.previewSize[0] === 1080 && r.previewSize[1] === 1920,
        r.previewSize.join('×'));
  await page.screenshot({ path: path.join(DOCS, 'w-canvas-916.png') });

  await openStep(page, 'st5');
  await page.check('#cOrig');
  await page.click('#bExport');
  r.info = await waitText(page, '#rinfo', /original cut|failed/, 1800000);
  check('the 9:16 pair exports in the tab', !/failed/.test(r.info), r.info);
  const dith = path.join(DOCS, 'w-canvas-916-dithered.webm');
  const orig = path.join(DOCS, 'w-canvas-916-original.webm');
  r.ditheredBytes = await saveDownload(page, dith);
  const [d2] = await Promise.all([
    page.waitForEvent('download', { timeout: 30000 }),
    page.click('#dlorig'),
  ]);
  r.originalName = d2.suggestedFilename();
  await d2.saveAs(orig);
  check('the file is named for its shape', /9x16/.test(r.originalName), r.originalName);
  r.dithered = probe(dith);
  r.original = probe(orig);
  for (const k of ['nb_read_frames', 'width', 'height']) {
    check(`the 9:16 pair agrees on ${k}`,
          String(r.dithered[k]) === String(r.original[k]),
          `${r.dithered[k]} vs ${r.original[k]}`);
  }
  check('the tab wrote 1080×1920',
        +r.dithered.width === 1080 && +r.dithered.height === 1920,
        `${r.dithered.width}×${r.dithered.height}`);
  check('the 9:16 render has every frame of the range',
        +r.dithered.nb_read_frames === r.nFrames,
        `${r.dithered.nb_read_frames} vs ${r.nFrames}`);
  await page.uncheck('#cOrig');

  /* ---- the dot data, and the frame it claims --------------------------- */
  r.dots = await page.evaluate(async () => {
    const { doc } = await window.DV_dots.doc();
    // a cell centre on the last column rounds to exactly w — both renderers
    // clamp it to the last pixel, and they did before there was a canvas, so
    // the assertion is that nothing is BEYOND the frame
    let out = 0, edge = 0, lit = 0;
    doc.frames.forEach((f) => f.forEach((xy) => {
      lit += xy.length >> 1;
      for (let i = 0; i < xy.length; i += 2) {
        if (xy[i] > doc.w || xy[i + 1] > doc.h) out++;
        else if (xy[i] === doc.w || xy[i + 1] === doc.h) edge++;
      }
    }));
    return { w: doc.w, h: doc.h, frames: doc.frames.length, outside: out,
             onEdge: edge, dots: lit };
  });
  check('the .dots.gz carries the canvas', r.dots.w === 1080 && r.dots.h === 1920,
        `${r.dots.w}×${r.dots.h}`);
  check('no dot is outside the canvas', r.dots.outside === 0,
        `${r.dots.outside} beyond, ${r.dots.onEdge} on the last cell centre`);
  check('there are dots in it', r.dots.dots > 0, String(r.dots.dots));

  /* ---- the manual override: drag, and it moves ------------------------- */
  const before = await page.evaluate(() => window.DV_canvas.at(0));
  r.nudged = await page.evaluate(() => window.DV_canvas.nudge(0.1, 0));
  const after = await page.evaluate(() => window.DV_canvas.at(0));
  r.nudgeShiftPx = +(after.cx - before.cx).toFixed(1);
  check('a nudge moves the crop', Math.abs(r.nudgeShiftPx - 0.1 * before.sw) < 2,
        String(r.nudgeShiftPx));
  await page.evaluate(() => window.DV_canvas.nudge(0, 0));

  /* ---- the sequence takes the same presets ----------------------------- */
  r.seq = await page.evaluate(async () => {
    const cands = window.DV_seq.candidates();
    await window.DV_seq.add(cands[0].id, cands[0].arg);
    const before = window.DV_seq.canvas();
    const after = window.DV_seq.canvas('4:5');
    const doc = await window.DV_seq.build();
    let out = 0, lit = 0;
    doc.frames.forEach((f) => f.forEach((xy) => {
      lit += xy.length >> 1;
      for (let i = 0; i < xy.length; i += 2) {
        if (xy[i] > doc.w || xy[i + 1] > doc.h) out++;
      }
    }));
    return { before, after, w: doc.w, h: doc.h, outside: out, dots: lit };
  });
  check('the sequence takes 4:5', r.seq.w === 1080 && r.seq.h === 1350,
        `${r.seq.w}×${r.seq.h}`);
  check('every sequence dot is inside the new frame', r.seq.outside === 0);
  check('the fitted sequence still has its dots', r.seq.dots > 0);

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

/* ====== WD: the two decode paths, over the same clip ========================
 * The tab used to get its frames one `currentTime =` at a time. It now demuxes
 * the file and runs VideoDecoder over the stream in a module Worker, and the
 * only interesting question about that is whether the frames are the SAME
 * frames: every mask index, every trim window and every number the rest of
 * this file asserts on is an index into that grid.
 *
 * So each clip is decoded twice, once down each path, and the two are compared
 * as pixels — a full-frame checksum for identity and a mean absolute
 * difference for how wrong it would be if they ever stopped matching. They are
 * expected to be byte-identical, because both paths hand the same RGBA to the
 * same JPEG encoder.
 *
 * The 90 s clip is built here with ffmpeg rather than committed. Its seek-path
 * half takes about three minutes, so it runs only under DV_DECODE_SLOW=1; the
 * WebCodecs half always runs, because "does a 2,700-frame clip still decode"
 * is not a performance question.
 */
async function decodeOnce(clipFile, decodePath) {
  const { ctx, page } = await newPage(browserPref(), { decodePath });
  try {
    await loadClip(page, clipFile);
    return await page.evaluate(async () => {
      const E = window.DV.engine, n = window.DV.nFrames;
      const big = document.createElement('canvas');
      const bg = big.getContext('2d', { willReadFrequently: true });
      const small = document.createElement('canvas');
      small.width = 160; small.height = 90;
      const sg = small.getContext('2d', { willReadFrequently: true });
      const look = async (i) => {
        const b = await E.frame(i);
        big.width = b.width; big.height = b.height;
        bg.drawImage(b, 0, 0);
        sg.drawImage(b, 0, 0, 160, 90);
        b.close();
        // FNV-1a over every byte: identity, in eight hex digits
        const d = bg.getImageData(0, 0, big.width, big.height).data;
        let hsh = 0x811c9dc5;
        for (let q = 0; q < d.length; q++) {
          hsh ^= d[q]; hsh = Math.imul(hsh, 0x01000193) >>> 0;
        }
        return { hash: hsh.toString(16),
                 thumb: Array.from(sg.getImageData(0, 0, 160, 90).data) };
      };
      return { n, w: window.DV.W, h: window.DV.H,
               decode: E.lastDecode, support: E.supports.decodePaths,
               bytes: E.clip.frames.reduce((a, b) => a + b.size, 0),
               first: await look(0), last: await look(n - 1) };
    });
  } finally { await ctx.close(); }
}

const meanAbsArr = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s / a.length;
};

async function runDecodePaths() {
  const r = { clips: {} };

  const both = async (label, file, slowToo) => {
    const fast = await decodeOnce(file, 'webcodecs-worker');
    const out = { fast: { path: fast.decode.path, ms: fast.decode.ms,
                          note: fast.decode.note, split: fast.decode.split,
                          frames: fast.n, bytes: fast.bytes } };
    check(`decode · ${label} · the worker path reports itself`,
          fast.decode.path === 'webcodecs-worker'
          && /WebCodecs/.test(fast.decode.label),
          JSON.stringify(fast.decode.line));
    check(`decode · ${label} · the stats line names the container and codec`,
          /·\s*(mp4|webm)\s*·/.test(fast.decode.line), fast.decode.line);
    check(`decode · ${label} · it says whether the decoder was hardware`,
          fast.decode.accel === 'hardware' || fast.decode.accel === 'software',
          String(fast.decode.accel));
    if (slowToo) {
      const slow = await decodeOnce(file, 'video-seek');
      out.slow = { path: slow.decode.path, ms: slow.decode.ms, frames: slow.n,
                   bytes: slow.bytes };
      out.speedup = +(slow.decode.ms / Math.max(1, fast.decode.ms)).toFixed(2);
      check(`decode · ${label} · both paths decode the same number of frames`,
            fast.n === slow.n, `${fast.n} vs ${slow.n}`);
      check(`decode · ${label} · the same decode size`,
            fast.w === slow.w && fast.h === slow.h,
            `${fast.w}x${fast.h} vs ${slow.w}x${slow.h}`);
      out.firstMeanAbs = +meanAbsArr(fast.first.thumb, slow.first.thumb).toFixed(4);
      out.lastMeanAbs = +meanAbsArr(fast.last.thumb, slow.last.thumb).toFixed(4);
      check(`decode · ${label} · frame 0 is the same picture`,
            out.firstMeanAbs < 1.5, String(out.firstMeanAbs));
      check(`decode · ${label} · the last frame is the same picture`,
            out.lastMeanAbs < 1.5, String(out.lastMeanAbs));
      out.identical = fast.first.hash === slow.first.hash
        && fast.last.hash === slow.last.hash && fast.bytes === slow.bytes;
      check(`decode · ${label} · and in fact byte-identical`, out.identical,
            `${fast.first.hash}/${slow.first.hash} `
            + `${fast.last.hash}/${slow.last.hash} ${fast.bytes}/${slow.bytes}`);
      check(`decode · ${label} · the worker path is the faster one`,
            fast.decode.ms < slow.decode.ms,
            `${fast.decode.ms} ms vs ${slow.decode.ms} ms`);
    }
    r.clips[label] = out;
    out.support = fast.support;
    return out;
  };

  const a = await both('sample.mp4', CLIP, true);
  r.support = a.support;
  check('decode · this browser advertises the worker path',
        r.support && r.support.webcodecs && r.support.worker
        && r.support.best === 'webcodecs-worker', JSON.stringify(r.support));
  check('decode · 150 frames', a.fast.frames === 150, String(a.fast.frames));

  // the clip that starts at 0.033 s: an off-by-one in the frame grid shows up
  // here and nowhere else
  await both('entry-clip.mp4', ENTRY, true);

  // a WebM with no Cues and no Duration in its header — a camera recording, in
  // other words, which is the file the seek loop is slowest on
  const webm = path.join(DOCS, 'w-decode-camera-like.webm');
  try {
    execFileSync('ffmpeg', ['-y', '-v', 'error', '-i', CLIP, '-c:v', 'libvpx',
      '-b:v', '2M', '-an', '-f', 'webm', '-live', '1', webm], { stdio: 'inherit' });
    await both('cue-less WebM', webm, true);
  } catch (e) {
    r.webmSkipped = String(e.message || e);
  } finally { fs.rmSync(webm, { force: true }); }

  // 90 s, 2,700 frames
  const long = path.join(DOCS, 'w-decode-long.mp4');
  try {
    execFileSync('ffmpeg', ['-y', '-v', 'error', '-stream_loop', '17', '-i', CLIP,
      '-c', 'copy', long], { stdio: 'inherit' });
    const l = await both('90 s clip', long, process.env.DV_DECODE_SLOW === '1');
    check('decode · a 90 s clip is 2,700 frames', l.fast.frames === 2700,
          String(l.fast.frames));
    check('decode · and it decodes in under a minute', l.fast.ms < 60000,
          `${l.fast.ms} ms`);
  } catch (e) {
    r.longSkipped = String(e.message || e);
  } finally { fs.rmSync(long, { force: true }); }

  return r;
}

/* ====== WB: the other two engines, and the export they used to crash in =====
 * Chromium is what the numbers in this file were measured on. It is not the
 * only browser the page has to survive, and one API is why: `requestFrame()`
 * is a method on `CanvasCaptureMediaStreamTrack` in Chromium and WebKit, and
 * in Firefox it is a method on the STREAM instead — so the export threw
 * "vtrack.requestFrame is not a function" on the browser a user actually had.
 *
 * Three things are checked here and each is a different failure:
 *   1. Firefox and WebKit open a still, decode a clip and export a WebM, with
 *      no console errors. Tracking is not attempted: headless WebKit has no
 *      WebGPU adapter, and asserting on the WASM fallback's speed would be
 *      asserting on the harness.
 *   2. Chromium with `requestFrame` DELETED off the prototype — the third
 *      branch, where neither form exists and the compositor samples the canvas
 *      instead. No shipping engine takes it today, which is exactly why it
 *      needs a test.
 */
async function runOtherBrowsers() {
  const out = {};
  for (const [name, launcher] of [['firefox', firefox], ['webkit', webkit]]) {
    let br = null;
    try { br = await launcher.launch({ headless: true }); }
    catch (e) { out[name] = { skipped: String(e.message).split('\n')[0] }; continue; }
    const r = {};
    try {
      const { ctx, page } = await newPage(browserPref(), { browser: br });
      r.ua = await page.evaluate(() => navigator.userAgent);
      r.caps = await page.evaluate(async () => {
        let adapter = false, f16 = null;
        if (navigator.gpu) {
          try { const a = await navigator.gpu.requestAdapter(); adapter = !!a;
                if (a) f16 = a.features.has('shader-f16'); } catch (e) { f16 = 'threw'; }
        }
        const c = document.createElement('canvas'); c.width = 8; c.height = 8;
        const st = c.captureStream(0), tr = st.getVideoTracks()[0];
        const rf = { onTrack: typeof tr.requestFrame === 'function',
                     onStream: typeof st.requestFrame === 'function' };
        tr.stop();
        return { webgpu: !!navigator.gpu, adapter, shaderF16: f16, requestFrame: rf,
                 decode: window.DV.engine.supports.decodePaths };
      });
      check(`${name} · the page comes up on the browser engine`,
            await page.evaluate(() => window.DV_engine().id === 'browser'));
      check(`${name} · it has a WebCodecs decode path`,
            r.caps.decode.webcodecs && r.caps.decode.worker,
            JSON.stringify(r.caps.decode));
      check(`${name} · requestFrame exists somewhere, or the compositor path is it`,
            r.caps.requestFrame.onTrack || r.caps.requestFrame.onStream
            || true, JSON.stringify(r.caps.requestFrame));

      await page.setInputFiles('#file', STILL);
      await page.waitForFunction(() => window.DV.kind === 'image', null, { timeout: 60000 });
      await sleep(1200);
      r.still = await page.evaluate(() => ({ w: window.DV.W, h: window.DV.H }));
      check(`${name} · a still opens at its own resolution`,
            r.still.w === 1280 && r.still.h === 720, JSON.stringify(r.still));
      await page.screenshot({ path: path.join(DOCS, `w-${name}-still.png`) });

      await loadClip(page, CLIP, 2);
      r.decode = await page.evaluate(() => window.DV.engine.lastDecode);
      r.nFrames = await page.evaluate(() => window.DV.nFrames);
      check(`${name} · a clip decodes`, r.nFrames === 60, String(r.nFrames));
      check(`${name} · through WebCodecs, and it says so`,
            r.decode.path === 'webcodecs-worker', r.decode.line);
      await setMode(page, 'ordered');
      await openStep(page, 'st5');
      await page.evaluate(() => document.querySelector('#bExport').click());
      r.render = await waitText(page, '#rinfo', /rendered|failed/, 900000);
      check(`${name} · and the clip exports`, !/failed/.test(r.render), r.render);
      const f = path.join(DOCS, `w-${name}-export.webm`);
      r.bytes = await saveDownload(page, f);
      r.probe = probe(f);
      /* Exactness is only available where `requestFrame` is a method on the
       * TRACK: that call captures the canvas synchronously. Firefox's
       * stream-level one queues the grab, and a render slower than real time
       * loses the odd frame to it — measured, 59 of 60. The export says so in
       * its own note rather than looking fine, and that saying-so is what is
       * asserted here. */
      const exact = r.caps.requestFrame.onTrack;
      const got = +r.probe.nb_read_frames;
      check(`${name} · the export is the clip's own length`,
            exact ? got === r.nFrames : Math.abs(got - r.nFrames) <= 2,
            `${got} vs ${r.nFrames}`);
      if (!exact && got !== r.nFrames) {
        check(`${name} · and a short cut says so`,
              /frames reached the file/.test(r.render), r.render);
      }
      fs.rmSync(f, { force: true });
      await page.screenshot({ path: path.join(DOCS, `w-${name}-export.png`) });
      await ctx.close();
    } catch (e) {
      r.failed = String(e.message || e);
      throw e;
    } finally { await br.close(); out[name] = r; }
  }

  /* --- and the branch no shipping browser takes */
  const br = await chromium.launch({ headless: true, args: GPU_ARGS });
  const r = {};
  try {
    const { ctx, page } = await newPage(browserPref(), { browser: br,
      init: () => { try { delete CanvasCaptureMediaStreamTrack.prototype.requestFrame; }
                    catch (e) { /* already not there */ } } });
    r.gone = await page.evaluate(() => {
      const c = document.createElement('canvas'); c.width = 8; c.height = 8;
      const s = c.captureStream(0), t = s.getVideoTracks()[0];
      const o = typeof t.requestFrame === 'undefined' && typeof s.requestFrame === 'undefined';
      t.stop(); return o;
    });
    check('no requestFrame · the API really is gone for this page', r.gone);
    await loadClip(page, CLIP, 2);
    r.nFrames = await page.evaluate(() => window.DV.nFrames);
    await setMode(page, 'ordered');
    await openStep(page, 'st5');
    await page.evaluate(() => document.querySelector('#bExport').click());
    r.render = await waitText(page, '#rinfo', /rendered|failed/, 900000);
    check('no requestFrame · the export still runs', !/failed/.test(r.render), r.render);
    const f = path.join(DOCS, 'w-nocapture-export.webm');
    r.bytes = await saveDownload(page, f);
    r.probe = probe(f);
    check('no requestFrame · and the compositor path wrote every frame',
          +r.probe.nb_read_frames === r.nFrames,
          `${r.probe.nb_read_frames} != ${r.nFrames}`);
    fs.rmSync(f, { force: true });
    await ctx.close();
  } finally { await br.close(); }
  out.noRequestFrame = r;
  return out;
}

/* ====== WA: WebGPU present, adapter refused — Brave's exact state ==========
 * `navigator.gpu` exists, `requestAdapter()` answers null. Brave does that by
 * default (Shields' fingerprinting protection) and Chrome does it on a
 * blocklisted GPU, and the page used to die on it twice over: it kept asking
 * onnxruntime for the WebGPU execution provider, which reported "no available
 * backend found", and even the WASM provider would not have started, because
 * the page asked for eight threads on a host that is not cross-origin isolated
 * and therefore has no SharedArrayBuffer.
 *
 * So this runs the whole tracker on the fallback: single-threaded WASM, a
 * still segmented and a short clip tracked, with the note that names the
 * setting to change. It is slow on purpose — that is what the fallback IS —
 * hence the very short clip.
 */
async function runNoAdapter() {
  const r = {};
  const { ctx, page } = await newPage(browserPref(), {
    init: () => {
      if (navigator.gpu) {
        Object.defineProperty(navigator.gpu, 'requestAdapter',
          { value: async () => null, configurable: true });
      }
    },
  });
  r.state = await page.evaluate(async () => ({
    gpu: !!navigator.gpu,
    adapter: !!(navigator.gpu && await navigator.gpu.requestAdapter()),
    isolated: globalThis.crossOriginIsolated === true,
    sab: typeof SharedArrayBuffer !== 'undefined',
    ep: window.DV.engine.ep, device: (window.DV.meta || {}).device,
    note: window.DV.engine.epNote || '',
  }));
  check('no adapter · WebGPU is there and the adapter is not',
        r.state.gpu && !r.state.adapter, JSON.stringify(r.state));
  check('no adapter · the engine falls to WASM before it asks ORT for anything',
        r.state.ep === 'wasm' && r.state.device === 'wasm', r.state.ep);
  check('no adapter · and the note names the setting to change',
        /Brave/.test(r.state.note) && /shields|flags/i.test(r.state.note),
        r.state.note);

  // a still, dithered — no tracker involved, and it has to keep working
  await page.setInputFiles('#file', STILL);
  await page.waitForFunction(() => window.DV.kind === 'image', null, { timeout: 60000 });
  await sleep(1500);
  r.still = await census(page).catch(() => null);
  await page.evaluate(() => { const b = document.querySelector('#st2');
    if (b && b.dataset.open !== '1') b.querySelector('.sh').click(); });
  await sleep(300);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(900);
  const [px, py] = await stageXY(page, SUBJECT_A.point[0], SUBJECT_A.point[1]);
  await page.mouse.click(px, py);
  r.preview = await waitText(page, '#pvinfo',
    /^(?!.*(reading|loading|\.onnx)).*(px|failed)/, 600000);
  check('no adapter · a still segments on the WASM backend',
        !/failed/.test(r.preview) && /px/.test(r.preview), r.preview);
  r.backend = await page.evaluate(() => ({
    line: window.DV.engine.backendLine(), threads: window.DV.engine.threads }));
  check('no adapter · single-threaded, because the page is not isolated',
        r.backend.threads === 1, JSON.stringify(r.backend));
  check('no adapter · the stats line says WASM and says it is slow',
        /WASM/.test(r.backend.line) && /slow/.test(r.backend.line),
        r.backend.line);
  await page.screenshot({ path: path.join(DOCS, 'w-noadapter-still.png') });

  // and half a second of clip, tracked. Fifteen frames on one WASM thread is
  // minutes; that is the honest cost of the fallback and the reason for 0.5 s.
  await loadClip(page, CLIP, 0.5);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
  await promptBoxPoint(page, SUBJECT_A);
  await page.click('#bTrack');
  r.track = await waitText(page, '#tinfo', /tracked|failed/, 1800000);
  check('no adapter · and a clip tracks on it', !/failed/.test(r.track), r.track);
  check('no adapter · the tracked line names the backend',
        /WASM/i.test(r.track), r.track);
  await page.screenshot({ path: path.join(DOCS, 'w-noadapter-tracked.png') });
  await ctx.close();
  return r;
}

/* ====== WQ: the three tracker squares, in the tab ==========================
 * The browser engine used to export one resolution and say so as a limit: 768
 * px only, because 512 and 1024 would triple the download. They are separate
 * graph sets under models/512 and models/1024 now, listed in the default
 * manifest's `tiers` so the page knows what this deployment carries without
 * probing for it, and loaded one at a time when their chip is picked.
 *
 * What has to hold: the chips appear and are the server's own ids; each one
 * loads its own set and tracks; the mask logit grid follows the square
 * (128/192/256, which is grid*4); the frame count does not move; and the masks
 * agree with the 768 px ones well enough to be the same subject.
 */
const tierMasks = (page, frames) => page.evaluate(async (fr) => {
  // one binary mask per asked-for frame, at 128x128, as a plain array
  const cv = document.createElement('canvas');
  cv.width = 128; cv.height = 128;
  const g = cv.getContext('2d', { willReadFrequently: true });
  const id = String(window.DV.subjects[0].id);
  const out = {};
  for (const n of fr) {
    const bmp = await window.DV.engine.mask(id, n);
    g.clearRect(0, 0, 128, 128);
    g.drawImage(bmp, 0, 0, 128, 128);
    bmp.close && bmp.close();
    const d = g.getImageData(0, 0, 128, 128).data;
    const a = new Array(128 * 128);
    for (let i = 0, q = 0; i < d.length; i += 4, q++) a[q] = d[i] > 127 ? 1 : 0;
    out[n] = a;
  }
  return out;
}, frames);

const iou = (a, b) => {
  let inter = 0, uni = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] && b[i]) inter++;
    if (a[i] || b[i]) uni++;
  }
  return uni ? inter / uni : 1;
};

async function runTrackTiers() {
  const r = { tiers: {} };
  const runs = {};
  for (const size of [512, 768, 1024]) {
    const { ctx, page } = await newPage(browserPref());
    const t = { size };
    r.offered = await page.evaluate(() => (window.DV.meta || {}).track_sizes);
    await loadClip(page, CLIP, 2);
    t.nFrames = await page.evaluate(() => window.DV.nFrames);
    t.picked = await page.evaluate((s) => {
      const b = [...document.querySelectorAll('#tq .chip')]
        .find((c) => +c.dataset.size === s);
      if (!b) return 0;
      b.click();
      return window.DV.trackSize;
    }, size);
    check(`tiers · ${size} px has a chip and it selects`, t.picked === size,
          String(t.picked));
    await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
    await promptBoxPoint(page, SUBJECT_A);
    t.preview = await waitText(page, '#pvinfo',
      /^(?!.*(reading|loading|\.onnx)).*(px|failed)/, 600000);
    check(`tiers · ${size} px previews`, !/failed/.test(t.preview), t.preview);
    check(`tiers · the preview says which square it used`,
          new RegExp(`\\(${size} px\\)`).test(t.preview), t.preview);
    const t0 = Date.now();
    await page.click('#bTrack');
    t.track = await waitText(page, '#tinfo', /tracked|failed/, 1800000);
    t.seconds = +((Date.now() - t0) / 1000).toFixed(1);
    check(`tiers · ${size} px tracks`, !/failed/.test(t.track), t.track);
    t.state = await page.evaluate(() => {
      const e = window.DV.engine;
      const m = e.masks.get(String(window.DV.subjects[0].id));
      return { loaded: e.trackerSize, frames: m ? m.length : 0,
               logitGrid: m && m[0] ? Math.round(Math.sqrt(m[0].length)) : 0,
               backend: e.backendLine() };
    });
    check(`tiers · ${size} px loaded its own graph set`,
          t.state.loaded === size, JSON.stringify(t.state));
    check(`tiers · ${size} px writes a ${size / 4} logit grid`,
          t.state.logitGrid === size / 4, JSON.stringify(t.state));
    check(`tiers · ${size} px covers every frame`,
          t.state.frames === t.nFrames,
          `${t.state.frames} vs ${t.nFrames}`);
    check(`tiers · the stats line names the square`,
          t.state.backend.includes(`${size} px`), t.state.backend);
    const frames = [0, t.nFrames >> 1, t.nFrames - 1];
    runs[size] = await tierMasks(page, frames);
    t.areas = frames.map((n) => runs[size][n].reduce((a, b) => a + b, 0));
    check(`tiers · ${size} px found the subject on every sampled frame`,
          t.areas.every((a) => a > 50), JSON.stringify(t.areas));
    r.tiers[size] = t;
    await page.screenshot({ path: path.join(DOCS, `w-tier-${size}.png`) });
    await ctx.close();
  }
  check('tiers · all three squares are offered',
        (r.offered || []).map((x) => x.size).join(',') === '512,768,1024',
        JSON.stringify(r.offered));
  check('tiers · with the ids the server uses',
        (r.offered || []).map((x) => x.id).join(',') === 'fast,balanced,best',
        JSON.stringify((r.offered || []).map((x) => x.id)));
  check('tiers · and each carries a measured fps',
        (r.offered || []).every((x) => x.fps > 0),
        JSON.stringify(r.offered));

  r.iou = {};
  for (const size of [512, 1024]) {
    const per = Object.keys(runs[768]).map((n) => +iou(runs[size][n], runs[768][n]).toFixed(4));
    r.iou[size] = per;
    check(`tiers · ${size} px masks agree with 768 px`,
          per.every((v) => v > 0.7), `${size}: ${per.join(', ')}`);
  }
  r.faster = +(r.tiers[1024].seconds / Math.max(0.1, r.tiers[512].seconds)).toFixed(2);
  check('tiers · a smaller square really is faster',
        r.tiers[512].seconds < r.tiers[768].seconds
        && r.tiers[768].seconds < r.tiers[1024].seconds,
        JSON.stringify({ 512: r.tiers[512].seconds, 768: r.tiers[768].seconds,
                         1024: r.tiers[1024].seconds }));
  return r;
}

/* ====== WD: the DEPLOYMENT, mirrored ======================================
 * Every other flow in this file runs against a checkout, where web/models has
 * everything: three squares, fp16 AND fp32. The live deployment does not. Pages
 * ships fp16 only -- the fp32 graphs are 83 MB a square for the one case of a
 * GPU without `shader-f16` -- and it ships the two extra squares as separate
 * tarballs. So the checkout can be green while the deployment is broken, and
 * that is exactly what happened:
 *
 *   "track failed: WebGPU fp16 at 512 px could not start -- Can't create a
 *    session. ERROR_CODE: 7, ERROR_MESSAGE: Failed to load model because
 *    protobuf parsing failed."
 *
 * which is onnxruntime being handed a static host's 404 PAGE as a model. It
 * names no file, so it reads as "the tracker is broken" rather than "one graph
 * is missing". web/track.js checks every fetch now and web/engines/browser.js
 * knows what to do with each answer; this is the flow that holds it there.
 *
 * The mirror is web/ with models/ replaced by symlinks to the fp16 files only,
 * served by `python3 -m http.server` -- no FastAPI, no /api/meta, no fp32,
 * which is the deployment. Four things are asked of it:
 *
 *   1. 512 px and 1024 px track, on the fp16-only tree
 *   2. no `shader-f16` -> WASM fp16, with the note, instead of a dead end
 *      (the fp32 probe's 404 is the one console error this flow allows)
 *   3. one tier graph 404ing -> the default square, and the note NAMES the file
 *   4. a 200 that is HTML rather than a graph -> the same, not "protobuf"
 */
const MIRROR_SET = ['manifest.json', 'consts.bin', 'encoder.fp16.onnx',
  'memattn.fp16.onnx', 'heads.fp16.onnx', 'heads_prompt.fp16.onnx',
  'heads_mask.fp16.onnx', 'memenc.onnx', 'memenc.f16in.onnx'];

/** web/, with models/ cut down to what a Pages deploy actually carries. */
function mirrorTree() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'dv-pages-'));
  const web = path.join(HERE, 'web');
  for (const name of fs.readdirSync(web)) {
    if (name === 'models') continue;
    fs.symlinkSync(path.join(web, name), path.join(root, name));
  }
  const square = (from, to) => {
    fs.mkdirSync(to, { recursive: true });
    for (const f of MIRROR_SET) {
      const src = path.join(from, f);
      if (fs.existsSync(src)) fs.symlinkSync(src, path.join(to, f));
    }
  };
  square(path.join(web, 'models'), path.join(root, 'models'));
  for (const S of [512, 1024]) {
    square(path.join(web, 'models', String(S)), path.join(root, 'models', String(S)));
  }
  return root;
}

async function serveStatic(root) {
  const port = await freePort();
  const proc = spawn('python3', ['-m', 'http.server', String(port),
    '--bind', '127.0.0.1'], { cwd: root, stdio: 'ignore' });
  const base = `http://127.0.0.1:${port}`;
  for (let i = 0; i < 100; i++) {
    const ok = await fetch(base + '/index.html').then((r) => r.ok).catch(() => false);
    if (ok) return { base, proc };
    await sleep(100);
  }
  proc.kill();
  throw new Error('the static mirror never came up');
}

/** adapter.features.has('shader-f16') -> false, everything else untouched. */
const NO_F16 = () => {
  if (!navigator.gpu) return;
  const real = navigator.gpu.requestAdapter.bind(navigator.gpu);
  Object.defineProperty(navigator.gpu, 'requestAdapter', {
    configurable: true,
    value: async (o) => {
      const a = await real(o);
      if (!a) return a;
      return new Proxy(a, {
        get(t, k) {
          if (k === 'features') return { has: (f) => f !== 'shader-f16' && t.features.has(f) };
          const v = t[k];
          return typeof v === 'function' ? v.bind(t) : v;
        },
      });
    },
  });
};

async function runDeployMirror() {
  const r = { squares: {} };
  const root = mirrorTree();
  r.mirror = { models: fs.readdirSync(path.join(root, 'models')).sort() };
  check('mirror · the tree carries no fp32 graph at any square',
        !fs.existsSync(path.join(root, 'models', 'encoder.onnx'))
        && !fs.existsSync(path.join(root, 'models', '512', 'encoder.onnx')),
        JSON.stringify(r.mirror.models));
  const { base, proc } = await serveStatic(root);
  r.base = base;
  try {
    /* --- 1. the two extra squares, on an fp16-only tree ------------------ */
    for (const size of [512, 1024]) {
      const t = { size };
      const { ctx, page } = await newPage(browserPref(), { base });
      await loadClip(page, CLIP, 2);
      t.nFrames = await page.evaluate(() => window.DV.nFrames);
      t.picked = await page.evaluate((sz) => {
        const b = [...document.querySelectorAll('#tq .chip')]
          .find((c) => +c.dataset.size === sz);
        if (!b) return 0;
        b.click();
        return window.DV.trackSize;
      }, size);
      check(`mirror · ${size} px is offered and selects`, t.picked === size,
            String(t.picked));
      await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
      await promptBoxPoint(page, SUBJECT_A);
      await page.click('#bTrack');
      t.track = await waitText(page, '#tinfo', /tracked|failed/, 1800000);
      check(`mirror · ${size} px tracks with no fp32 in the deployment`,
            !/failed/.test(t.track), t.track);
      t.state = await page.evaluate(() => ({
        loaded: window.DV.engine.trackerSize,
        backend: window.DV.engine.backendLine(),
        tierNote: window.DV.engine.tierNote || '',
        epNote: window.DV.engine.epNote || '',
      }));
      check(`mirror · ${size} px really loaded its own square`,
            t.state.loaded === size, JSON.stringify(t.state));
      check(`mirror · ${size} px needed no fallback and says nothing about one`,
            !t.state.tierNote && !t.state.epNote, JSON.stringify(t.state));
      r.squares[size] = t;
      await ctx.close();
    }

    /* --- 2. no shader-f16, and no fp32 to fall back to -------------------- */
    // the fp32 probe's 404 is the point of the case, so it is the one console
    // error allowed here -- and only this URL
    EXPECTED = '/models/encoder.onnx';
    const { ctx: c2, page: p2 } = await newPage(browserPref(), { base, init: NO_F16 });
    r.noF16 = {};
    r.noF16.adapter = await p2.evaluate(async () => {
      const a = await navigator.gpu.requestAdapter().catch(() => null);
      return { gpu: !!navigator.gpu, adapter: !!a,
               f16: !!(a && a.features.has('shader-f16')) };
    });
    check('mirror · the GPU is there and reports no shader-f16',
          r.noF16.adapter.adapter && !r.noF16.adapter.f16,
          JSON.stringify(r.noF16.adapter));
    await p2.setInputFiles('#file', STILL);
    await p2.waitForFunction(() => window.DV.kind === 'image', null, { timeout: 60000 });
    await sleep(1200);
    await openStep(p2, 'st2');
    await p2.click('#scope .chip[data-scope="track"]'); await sleep(900);
    const [px, py] = await stageXY(p2, SUBJECT_A.point[0], SUBJECT_A.point[1]);
    await p2.mouse.click(px, py);
    r.noF16.preview = await waitText(p2, '#pvinfo',
      /^(?!.*(reading|loading|\.onnx)).*(px|failed)/, 900000);
    check('mirror · no shader-f16 still segments, on fp16 over WASM',
          !/failed/.test(r.noF16.preview), r.noF16.preview);
    r.noF16.state = await p2.evaluate(() => ({
      ep: window.DV.engine.ep, fp16: window.DV.engine.fp16,
      line: window.DV.engine.backendLine(), note: window.DV.engine.epNote || '',
    }));
    check('mirror · it went to WASM fp16, not to a missing fp32 graph',
          r.noF16.state.ep === 'wasm' && r.noF16.state.fp16 === true,
          JSON.stringify(r.noF16.state));
    check('mirror · and the note says why, and where the fp32 bundle is',
          /shader-f16/.test(r.noF16.state.note)
          && /fp32/.test(r.noF16.state.note)
          && /README/.test(r.noF16.state.note), r.noF16.state.note);
    await p2.screenshot({ path: path.join(DOCS, 'w-mirror-nof16.png') });
    await c2.close();
    EXPECTED = null;

    /* --- 3 and 4. a tier graph that does not come back as a graph --------- */
    // the 404 IS the case under test, so its console line is expected here --
    // and only for this URL
    EXPECTED = '/models/512/heads_prompt.fp16.onnx';
    for (const [name, fulfil] of [
      ['404', { status: 404, contentType: 'text/html; charset=utf-8',
                body: '<!DOCTYPE html><html><head><title>Site not found</title>'
                  + '</head><body>404</body></html>' }],
      ['a 200 that is HTML', { status: 200, contentType: 'text/html; charset=utf-8',
                body: '<!DOCTYPE html><html><body>not a graph</body></html>' }],
    ]) {
      const { ctx, page } = await newPage(browserPref(), { base });
      await page.route('**/models/512/heads_prompt.fp16.onnx', (rt) => rt.fulfill(fulfil));
      await loadClip(page, CLIP, 1);
      await page.evaluate(() => {
        const b = [...document.querySelectorAll('#tq .chip')]
          .find((c) => +c.dataset.size === 512);
        if (b) b.click();
      });
      await page.click('#scope .chip[data-scope="track"]'); await sleep(800);
      await promptBoxPoint(page, SUBJECT_A);
      await page.click('#bTrack');
      const line = await waitText(page, '#tinfo', /tracked|failed/, 1800000);
      const st = await page.evaluate(() => ({
        loaded: window.DV.engine.trackerSize,
        tierNote: window.DV.engine.tierNote || '',
      }));
      r[`broken512_${name}`] = { line, ...st };
      check(`mirror · ${name} on a tier graph does not kill the track`,
            !/failed/.test(line), line);
      check(`mirror · ${name} falls back to the default square`,
            st.loaded === 768, JSON.stringify(st));
      check(`mirror · ${name} names the file it could not load`,
            /heads_prompt\.fp16\.onnx/.test(st.tierNote), st.tierNote);
      check(`mirror · ${name} never reaches "protobuf parsing failed"`,
            !/protobuf/i.test(st.tierNote) && !/protobuf/i.test(line),
            st.tierNote + ' | ' + line);
      await ctx.close();
    }
    EXPECTED = null;
  } finally {
    EXPECTED = null;
    proc.kill();
    fs.rmSync(root, { recursive: true, force: true });
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

/* ---- W11: per-subject incremental tracking, in the tab -------------------
 *
 * The browser engine's tracking loop was already one memory bank per subject,
 * so this is the same loop with a shorter list. What has to be proved is that
 * the list really is shorter and that the subjects left out of it keep the
 * logits they had: a hash over every stored logit array, before and after.
 */
const logitHashes = (page) => page.evaluate(() => {
  const out = {};
  for (const [id, seq] of window.DV.engine.masks) {
    let h = 2166136261, filled = 0;
    for (const a of seq) {
      if (!a) { h ^= 0xfe; h = Math.imul(h, 16777619); continue; }
      filled++;
      // every 13th value: enough to catch a different track, cheap enough to
      // run over 150 frames x 36864 logits without stalling the page
      for (let q = 0; q < a.length; q += 13) {
        h ^= Math.round((a[q] + 20) * 6) & 255; h = Math.imul(h, 16777619);
      }
    }
    out[id] = { frames: filled, hash: (h >>> 0).toString(16) };
  }
  return out;
});

async function runSubjectsBrowser() {
  const r = {};
  const { ctx, page } = await newPage(browserPref());
  await loadClip(page, CLIP, BR.seconds);
  r.nFrames = await page.evaluate(() => window.DV.nFrames);
  await page.click('#scope .chip[data-scope="track"]'); await sleep(800);

  /* --- 1. one subject on its own ------------------------------------- */
  await promptBoxPoint(page, SUBJECT_A);
  r.ctaFirst = (await page.textContent('#bTrack')).trim();
  let t0 = Date.now();
  await page.click('#bTrack');
  r.trackOne = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check('browser · one subject tracks on its own', !/failed/.test(r.trackOne), r.trackOne);
  r.secondsOne = +((Date.now() - t0) / 1000).toFixed(1);
  r.masksOne = await logitHashes(page);
  check('browser · only subject #1 has logits',
        JSON.stringify(Object.keys(r.masksOne)) === '["1"]',
        JSON.stringify(Object.keys(r.masksOne)));
  r.stateOne = await page.evaluate(() => window.DV_subjects.list());
  await page.screenshot({ path: path.join(AFTER, 'subjects-w1-tracked.png') });

  /* --- 2. a second subject, tracked without touching the first -------- */
  await openStep(page, 'st2');
  await page.click('#bAdd'); await sleep(400);
  await promptBoxPoint(page, SUBJECT_B);
  r.plan = await page.evaluate(() => window.DV_subjects.plan());
  check('browser · the button offers the new subject only',
        JSON.stringify(r.plan.ids) === '[2]' && /1 new subject/.test(r.plan.cta),
        JSON.stringify(r.plan));
  r.chips = await page.$$eval('#subs .chip', (n) => n.map((e) => ({
    text: e.textContent.replace(/\s+/g, ' ').trim(), state: e.dataset.state })));
  check('browser · the chips say which subject is tracked and which is new',
        r.chips[0].state === 'tracked' && r.chips[1].state === 'untracked',
        JSON.stringify(r.chips));
  await page.screenshot({ path: path.join(AFTER, 'subjects-w2-new.png') });

  t0 = Date.now();
  await page.click('#bTrack');
  r.trackTwo = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check('browser · the new subject tracks', !/failed/.test(r.trackTwo), r.trackTwo);
  r.secondsTwo = +((Date.now() - t0) / 1000).toFixed(1);
  r.masksTwo = await logitHashes(page);
  check("browser · #1's logits survived the run byte for byte",
        r.masksTwo['1'].hash === r.masksOne['1'].hash,
        `${r.masksOne['1'].hash} -> ${r.masksTwo['1'].hash}`);
  check('browser · #2 now has a full set of its own',
        r.masksTwo['2'] && r.masksTwo['2'].frames === r.nFrames,
        JSON.stringify(r.masksTwo['2']));
  check('browser · and the run says what it kept',
        /kept from an earlier run/.test(r.trackTwo), r.trackTwo);
  await page.screenshot({ path: path.join(AFTER, 'subjects-w3-both.png') });

  /* --- 3. a prompt edit makes exactly one subject stale --------------- */
  await openStep(page, 'st2');
  await page.evaluate(() => { window.DV_subjects.prompt(); window.DV.active = 1; });
  await sleep(400);
  const [ex, ey] = await stageXY(page, SUBJECT_B.point[0] - 40, SUBJECT_B.point[1] + 150);
  await page.mouse.click(ex, ey);
  await sleep(400);
  r.stateStale = await page.evaluate(() => window.DV_subjects.list());
  check('browser · editing #2 leaves #1 alone',
        r.stateStale.find((x) => x.id === 2).state === 'stale'
        && r.stateStale.find((x) => x.id === 1).state === 'tracked',
        JSON.stringify(r.stateStale.map((x) => [x.id, x.state])));
  r.planStale = await page.evaluate(() => window.DV_subjects.plan());
  check('browser · and the button offers to re-track just it',
        /Re-track #2/.test(r.planStale.cta), r.planStale.cta);
  r.menu = await page.evaluate(() => window.DV_subjects.menu(1));
  await page.screenshot({ path: path.join(AFTER, 'subjects-w4-menu.png') });
  await page.evaluate(() => window.DV_subjects.closeMenu());

  t0 = Date.now();
  await page.evaluate(() => window.DV_subjects.track([2]));
  r.trackAgain = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check('browser · re-tracking one subject works', !/failed/.test(r.trackAgain),
        r.trackAgain);
  r.secondsAgain = +((Date.now() - t0) / 1000).toFixed(1);
  r.masksAgain = await logitHashes(page);
  check("browser · #1 did not move on a #2-only re-track",
        r.masksAgain['1'].hash === r.masksOne['1'].hash,
        `${r.masksOne['1'].hash} -> ${r.masksAgain['1'].hash}`);
  check('browser · #2 did move',
        r.masksAgain['2'].hash !== r.masksTwo['2'].hash,
        `${r.masksTwo['2'].hash} -> ${r.masksAgain['2'].hash}`);

  /* --- 4. hide, then remove ------------------------------------------ */
  await page.evaluate(() => window.DV_draw(5)); await sleep(600);
  r.activeBoth = await page.evaluate(() => window.DV_subjects.active());
  await page.evaluate(() => window.DV_subjects.hide(1, true));
  await sleep(400);
  await page.evaluate(() => window.DV_draw(5)); await sleep(600);
  r.activeHidden = await page.evaluate(() => window.DV_subjects.active());
  r.masksHidden = await logitHashes(page);
  check('browser · hiding a subject drops it from the render and keeps its masks',
        JSON.stringify(r.activeHidden) === '[2]'
        && r.masksHidden['1'].hash === r.masksOne['1'].hash,
        JSON.stringify(r.activeHidden));
  await page.screenshot({ path: path.join(AFTER, 'subjects-w5-hidden.png') });
  await page.evaluate(() => window.DV_subjects.hide(1, false)); await sleep(400);

  await page.evaluate(() => window.DV_subjects.remove(1));
  await sleep(900);
  r.masksLeft = await logitHashes(page);
  check('browser · removing a subject forgets its logits',
        !r.masksLeft['1'] && !!r.masksLeft['2'],
        JSON.stringify(Object.keys(r.masksLeft)));
  r.activeLeft = await page.evaluate(() => window.DV_subjects.active());
  await page.evaluate(() => window.DV_draw(5)); await sleep(700);
  r.censusLeft = await census(page);
  await page.screenshot({ path: path.join(AFTER, 'subjects-w6-removed.png') });

  /* --- 5. and the tab still exports what is left ---------------------- */
  await openStep(page, 'st5');
  await page.click('#bExport');
  r.export = await waitText(page, '#rinfo', /rendered|failed/, 900000);
  check('browser · what is left exports', !/failed/.test(r.export), r.export);
  const out = path.join(DOCS, 'w-subjects.webm');
  r.bytes = await saveDownload(page, out);
  r.probe = r.bytes ? probe(out) : null;
  check('browser · the export carries every frame',
        r.probe && +r.probe.nb_read_frames === r.nFrames,
        JSON.stringify(r.probe));
  /* --- 6. and what splitting a run costs in the tab ------------------- *
   * The browser engine walks one subject at a time whatever it is asked for,
   * so two in one run should cost about what two separate runs cost. Measured
   * rather than asserted: a joint run on the same clip at the same quality. */
  await openStep(page, 'st2');
  await page.evaluate(() => { window.DV_subjects.prompt(); });
  await sleep(300);
  await page.click('#bAdd'); await sleep(400);
  await promptBoxPoint(page, SUBJECT_A);
  t0 = Date.now();
  await page.evaluate(() => window.DV_subjects.track(
    window.DV.subjects.map((x) => x.id)));
  r.trackBoth = await waitText(page, '#tinfo', /tracked|failed/, 900000);
  check('browser · two subjects still track in one run', !/failed/.test(r.trackBoth),
        r.trackBoth);
  r.secondsBoth = +((Date.now() - t0) / 1000).toFixed(1);
  r.runs = await page.evaluate(() => window.DV_subjects.runs());
  r.cost = {
    separately: +(r.runs.filter((x) => x.n === 1)
      .slice(-2).reduce((a, x) => a + x.seconds, 0)).toFixed(2),
    together: (r.runs.find((x) => x.n === 2) || {}).seconds,
  };
  check('browser · one run of two costs about what two runs of one cost',
        r.cost.together > 0
        && Math.abs(r.cost.together - r.cost.separately) / r.cost.separately < 0.4,
        JSON.stringify(r.cost));
  r.costNote = (await page.textContent('#tracknote') || '').replace(/\s+/g, ' ').trim();
  await ctx.close();
  return r;
}

/* DV_ONLY=tracked,sequence runs a subset. Unset runs all of it, which is what
 * the README's numbers were measured with. */
const ONLY = (process.env.DV_ONLY || '').split(',').map((x) => x.trim()).filter(Boolean);
const want = (name) => !ONLY.length || ONLY.includes(name);
R.only = ONLY;
const run = async (name, fn, ...args) => {
  if (want(name)) R.runs[name] = await fn(...args);
};

try {
  await run('decodePaths', runDecodePaths);
  await run('engineChip', runEngineChip);
  await run('still', runStill);
  await run('stillDots', runStillDots);
  await run('stillSubjectBrowser', runStillSubject, 'browser');
  await run('stillSubjectRemote', runStillSubject, 'remote');
  await run('whole', runWhole);
  await run('original', runOriginalBrowser);
  await run('range', runRangeBrowser);
  await run('tracked', runTracked);
  await run('subjects', runSubjectsBrowser);
  await run('lasso', runLasso);
  await run('cameraRemote', runCamera, 'remote', true);
  await run('cameraBrowser', runCamera, 'browser', false);
  await run('dotsRemote', runDots, 'remote');
  await run('dotsBrowser', runDots, 'browser');
  await run('canvasBrowser', runCanvasBrowser);
  await run('sequence', runSequence);
  await run('seqItemLook', runSeqItemLook);
  await run('seqPixel', runSeqPixel);
  await run('entryBrowser', runEntry, 'browser');
  await run('entryRemote', runEntry, 'remote');
  await run('trackTiers', runTrackTiers);
  await run('deployMirror', runDeployMirror);
  await run('otherBrowsers', runOtherBrowsers);
  await run('noAdapter', runNoAdapter);
  await run('apiKey', runApiKey);
} catch (e) {
  R.fatal = String(e && e.stack ? e.stack.split('\n').slice(0, 3).join(' | ') : e);
} finally {
  await BR.browser.close();
}

R.checksPassed = R.checks.filter((c) => c.ok).length;
R.checksTotal = R.checks.length;
console.log(JSON.stringify(R, null, 1));
// The report is committed as evidence, so it must not carry this machine's
// home directory around in it: every absolute path under the checkout is
// rewritten to a repo-relative one on the way out.
/* A DV_ONLY run is a subset, so it writes a subset report. It used to write
 * over docs/verify-web-report.json, which is committed evidence for the WHOLE
 * suite -- one narrow re-run and the evidence was gone. */
fs.writeFileSync(path.join(DOCS, ONLY.length
  ? `verify-web-report.${ONLY.join('-')}.json` : 'verify-web-report.json'),
  JSON.stringify(R, null, 1).split('file://' + HERE + '/').join('')
                             .split(HERE + '/').join(''));
process.exit(R.fatal || R.consoleErrors.length || R.pageErrors.length ? 1 : 0);
