/* End-to-end check against a real running server + real EdgeTAM.
 *   NODE_PATH=~/node_modules node verify.mjs [baseURL] [clip.mp4]
 * Writes screenshots to docs/ and prints a JSON report.
 */
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.argv[2] || 'http://127.0.0.1:8765';
const CLIP = process.argv[3] || path.join(HERE, 'sample.mp4');
const DOCS = path.join(HERE, 'docs');
fs.mkdirSync(DOCS, { recursive: true });

const SUBJECT_A = { box: [435, 95, 625, 360], point: [545, 205] };   // parkour athlete
const SUBJECT_B = { box: [1005, 5, 1279, 470], point: [1150, 160] }; // tree, right edge

const report = { base: BASE, clip: CLIP, consoleErrors: [], pageErrors: [],
                 requestFailures: [], runs: {} };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function ffprobe(f) {
  const out = execFileSync('ffprobe', ['-v', 'error', '-count_frames', '-select_streams', 'v:0',
    '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate', '-of', 'json', f]);
  return JSON.parse(out).streams[0];
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

async function stageXY(page, sel, x, y) {
  return page.evaluate(([s, fx, fy]) => {
    const el = document.querySelector(s);
    const r = el.getBoundingClientRect();
    const W = el.width, H = el.height;
    return [r.left + (fx / W) * r.width, r.top + (fy / H) * r.height];
  }, [sel, x, y]);
}

async function prompt(page, s) {
  const [x0, y0] = await stageXY(page, '#pov', s.box[0], s.box[1]);
  const [x1, y1] = await stageXY(page, '#pov', s.box[2], s.box[3]);
  await page.mouse.move(x0, y0);
  await page.mouse.down();
  await page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, { steps: 6 });
  await page.mouse.move(x1, y1, { steps: 6 });
  await page.mouse.up();
  const [px, py] = await stageXY(page, '#pov', s.point[0], s.point[1]);
  await page.mouse.click(px, py);
}

async function waitText(page, sel, re, timeout) {
  const t0 = Date.now();
  for (;;) {
    const t = await page.textContent(sel).catch(() => '');
    if (t && re.test(t)) return t;
    if (Date.now() - t0 > timeout) throw new Error(`timeout on ${sel} (last: ${t})`);
    await sleep(500);
  }
}

/* count canvas pixels that exactly match each dot colour + the bg */
async function canvasStats(page, dots, bg) {
  return page.evaluate(([dots, bg]) => {
    const c = document.querySelector('#vcv');
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    const hx = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
    const cols = dots.map(hx), b = hx(bg);
    const hits = cols.map(() => 0);
    let bgN = 0, other = 0;
    for (let p = 0; p < d.length; p += 4) {
      let m = -1;
      for (let k = 0; k < cols.length; k++) {
        if (d[p] === cols[k][0] && d[p + 1] === cols[k][1] && d[p + 2] === cols[k][2]) { m = k; break; }
      }
      if (m >= 0) hits[m]++;
      else if (d[p] === b[0] && d[p + 1] === b[1] && d[p + 2] === b[2]) bgN++;
      else other++;
    }
    return { w: c.width, h: c.height, hits, bg: bgN, other };
  }, [dots, bg]);
}

async function run(page, name, subjects) {
  const r = {};
  await page.goto(BASE + '/', { waitUntil: 'networkidle' });
  await page.setInputFiles('#file', CLIP);
  const up = await waitText(page, '#upstat', /frames/, 60000);
  r.upload = up;
  r.nFrames = parseInt(up.match(/^(\d+) frames/)[1], 10);

  await prompt(page, subjects[0]);
  for (let i = 1; i < subjects.length; i++) {
    await page.click('#bAdd');
    await prompt(page, subjects[i]);
  }
  r.subjectChips = await page.$$eval('#subs .chip', (n) => n.map((e) => e.textContent));
  await page.screenshot({ path: path.join(DOCS, `${name}-step2-prompts.png`), fullPage: false });

  const t0 = Date.now();
  await page.click('#bTrack');
  r.trackInfo = await waitText(page, '#tinfo', /tracked|failed/, 300000);
  r.trackWallSeconds = +((Date.now() - t0) / 1000).toFixed(1);
  if (/failed/.test(r.trackInfo)) throw new Error(r.trackInfo);

  const st = await (await page.request.get(`${BASE}/api/jobs/${await page.evaluate(() => window.DV.job)}/status`)).json();
  r.status = { state: st.state, done_frames: st.done_frames, n_frames: st.n_frames,
               elapsed_s: st.elapsed_s, fps: st.fps, device: st.device };

  await page.waitForSelector('#vwrap:not([hidden])');
  await sleep(600);
  const dots = await page.evaluate(() => window.DV.subjects.map((s) => s.dot));
  const bg = await page.evaluate(() => window.DV.P.bg);
  r.dots = dots; r.bg = bg;
  r.previewFrame0 = await canvasStats(page, dots, bg);

  // play a little, then land on a frame where both subjects are on screen
  await page.click('#bPlay');
  await sleep(2500);
  await page.click('#bPlay');
  r.previewFps = await page.textContent('#fps');
  await page.evaluate(() => window.DV_draw(20));
  await sleep(400);
  r.previewFrame20 = await canvasStats(page, dots, bg);
  await page.screenshot({ path: path.join(DOCS, `${name}-step3-preview.png`) });
  await page.locator('#vcv').screenshot({ path: path.join(DOCS, `${name}-preview-frame20.png`) });

  const t1 = Date.now();
  await page.click('#st4 .sh');
  await page.click('#bRender');
  r.renderInfo = await waitText(page, '#rinfo', /rendered|failed/, 300000);
  r.renderWallSeconds = +((Date.now() - t1) / 1000).toFixed(1);
  if (/failed/.test(r.renderInfo)) throw new Error(r.renderInfo);
  r.downloadHref = await page.getAttribute('#dl', 'href');
  await sleep(800);
  await page.screenshot({ path: path.join(DOCS, `${name}-step4-export.png`) });

  const job = await page.evaluate(() => window.DV.job);
  const mp4 = path.join(HERE, 'jobs', job, 'out.mp4');
  r.job = job;
  r.ffprobe = ffprobe(mp4);
  const png = path.join(DOCS, `${name}-exported-frame20.png`);
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', mp4, '-vf', 'select=eq(n\\,20)', '-vframes', '1', png]);
  r.exportedFrame = png;

  // colour census on the exported frame (jpeg/h264 shifts colours, so nearest-match)
  const raw = execFileSync('ffmpeg', ['-v', 'error', '-i', png, '-f', 'rawvideo',
    '-pix_fmt', 'rgb24', '-'], { maxBuffer: 1 << 28 });
  const cols = dots.map(hex), b = hex(bg);
  const near = cols.map(() => 0);
  let bgN = 0;
  for (let p = 0; p < raw.length; p += 3) {
    const px = [raw[p], raw[p + 1], raw[p + 2]];
    let best = -1, bd = 60;
    cols.forEach((c, k) => { const d2 = dist(px, c); if (d2 < bd) { bd = d2; best = k; } });
    if (best >= 0) near[best]++; else if (dist(px, b) < 30) bgN++;
  }
  r.exportedNearDot = near;
  r.exportedNearBg = bgN;
  return r;
}

const browser = await chromium.launch({ headless: true, channel: process.env.DV_CHANNEL || undefined });
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
page.on('console', (m) => { if (m.type() === 'error') report.consoleErrors.push(m.text()); });
page.on('pageerror', (e) => report.pageErrors.push(String(e)));
// Chromium aborts a media stream once it has buffered enough; that shows up as
// ERR_ABORTED on out.mp4 and is not an application error.
page.on('requestfailed', (r2) => {
  const why = (r2.failure() || {}).errorText || '';
  report.requestFailures.push({ url: r2.url(), why });
  if (why !== 'net::ERR_ABORTED') report.consoleErrors.push('requestfailed ' + r2.url() + ' ' + why);
});

try {
  report.runs.single = await run(page, 'single', [SUBJECT_A]);
  report.runs.dual = await run(page, 'dual', [SUBJECT_A, SUBJECT_B]);
} catch (e) {
  report.fatal = String(e);
} finally {
  await browser.close();
}
console.log(JSON.stringify(report, null, 2));
fs.writeFileSync(path.join(DOCS, 'verify-report.json'), JSON.stringify(report, null, 2));
process.exit(report.fatal || report.consoleErrors.length || report.pageErrors.length ? 1 : 0);
