/* ---------------------------------------------------------------------------
   DITHER VIDEO — track subjects through a clip, then dither them into dots.

   The browser preview reimplements render.py's blue-noise threshold dither
   1:1 (same 64x64 tile fetched from /api/bluenoise, same portable per-cell
   hash for jitter/stray, same gain search, same cell ownership rule), so what
   plays here is what the MP4 export writes.
--------------------------------------------------------------------------- */
'use strict';

const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

/* ---------------------------------------------------------------- state */
const S = {
  job: null, meta: null, nFrames: 0, W: 0, H: 0, fps: 30,
  subjects: [], active: 0, nextId: 1,
  promptFrame: 0,
  tracked: false, trackInfo: null,
  playing: false, cur: 0,
  tile: null, seed: 7,
  compare: false, split: 0.5,
  P: { mode: 'cutout', bg: '#c9d4c5', n: 8000, cell: 4, dotpx: 3, gamma: 1.0,
       fill: 0.7, stray: 0.02, band: 9, invert: false },
  palettes: [], pal: 0,
};

const DOTS_FALLBACK = ['#b0413e', '#2f4f4a', '#7a6a4f', '#3c5a7a', '#8a5a8a', '#4a7a4a'];
const MAX_SUBJECTS = 6;

function toast(msg, err) {
  const t = $('#toast');
  t.textContent = msg; t.hidden = false; t.classList.toggle('err', !!err);
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, err ? 8000 : 3500);
}

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let d = r.statusText;
    try { d = (await r.json()).detail || d; } catch (e) { /* non-json body */ }
    throw new Error(d);
  }
  return r.json();
}

function openStep(n) {
  $$('.step').forEach((el) => el.setAttribute('data-open', el.id === 'st' + n ? '1' : '0'));
}
$$('.step .sh').forEach((h) => h.addEventListener('click', () => {
  const st = h.parentElement;
  st.setAttribute('data-open', st.getAttribute('data-open') === '1' ? '0' : '1');
}));

/* =========================================================== step 1: upload */
const drop = $('#drop'), fileIn = $('#file');
drop.addEventListener('click', () => fileIn.click());
fileIn.addEventListener('change', () => { if (fileIn.files[0]) upload(fileIn.files[0]); });
['dragenter', 'dragover'].forEach((e) => document.addEventListener(e, (ev) => {
  ev.preventDefault(); document.body.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((e) => document.addEventListener(e, (ev) => {
  ev.preventDefault();
  if (e === 'drop' && ev.dataTransfer.files[0]) upload(ev.dataTransfer.files[0]);
  document.body.classList.remove('dragging');
}));
$('#sSec').addEventListener('input', (e) => { $('#vSec').textContent = e.target.value + ' s'; });

async function upload(f) {
  const box = $('#upstat');
  box.hidden = false; box.classList.remove('err'); box.textContent = 'uploading ' + f.name + '…';
  const fd = new FormData();
  fd.append('file', f);
  fd.append('max_seconds', $('#sSec').value);
  try {
    const j = await api('/api/upload', { method: 'POST', body: fd });
    S.job = j.job; S.nFrames = j.n_frames; S.W = j.w; S.H = j.h; S.fps = j.fps;
    S.tracked = false; S.trackInfo = null; S.subjects = []; S.nextId = 1; S.cur = 0;
    S.promptFrame = 0;
    addSubject();
    box.textContent = `${j.n_frames} frames · ${j.w}×${j.h} · ${j.fps} fps · job ${j.job}`;
    $('#s1sum').textContent = `${j.n_frames}f`;
    $('#sPF').max = j.n_frames - 1; $('#sPF').value = 0; $('#vPF').textContent = '0';
    $('#sFrame').max = j.n_frames - 1;
    $('#empty').hidden = true; $('#pwrap').hidden = false; $('#vwrap').hidden = true;
    $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
    $('#tinfo').hidden = true;
    showPromptFrame(0);
    openStep(2);
  } catch (err) {
    box.classList.add('err'); box.textContent = 'upload failed: ' + err.message;
  }
}

/* ================================================== step 2: prompt / track */
const pimg = $('#pimg'), pov = $('#pov'), pctx = pov.getContext('2d');

function palDots() {
  return (S.palettes[S.pal] && S.palettes[S.pal].dots) || DOTS_FALLBACK;
}

function addSubject() {
  if (S.subjects.length >= MAX_SUBJECTS) return;
  const i = S.subjects.length;
  S.subjects.push({ id: S.nextId++, dot: palDots()[i % 6], points: [], box: null });
  S.active = S.subjects.length - 1;
  renderSubjects();
}

function renderSubjects() {
  const wrap = $('#subs');
  wrap.textContent = '';
  S.subjects.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'chip sub';
    b.setAttribute('aria-pressed', i === S.active ? 'true' : 'false');
    const sw = document.createElement('span');
    sw.className = 'sw'; sw.style.background = s.dot;
    const nm = document.createElement('span');
    const np = s.points.length, hasBox = s.box ? 1 : 0;
    nm.textContent = `subject ${s.id}` + (np || hasBox ? ` · ${np}pt${hasBox ? '+box' : ''}` : '');
    b.append(sw, nm);
    b.addEventListener('click', () => { S.active = i; renderSubjects(); drawOverlay(); });
    if (S.subjects.length > 1) {
      const x = document.createElement('span');
      x.className = 'x'; x.textContent = '✕';
      x.addEventListener('click', (e) => {
        e.stopPropagation();
        S.subjects.splice(i, 1);
        S.active = Math.min(S.active, S.subjects.length - 1);
        renderSubjects(); drawOverlay(); renderDotCols();
      });
      b.append(x);
    }
    wrap.append(b);
  });
  $('#vSubs').textContent = `${S.subjects.length} / ${MAX_SUBJECTS}`;
  $('#s2sum').textContent = S.tracked ? 'tracked' : `${S.subjects.length} subj`;
  renderDotCols();
}

$('#bAdd').addEventListener('click', addSubject);
$('#bClr').addEventListener('click', () => {
  S.subjects.forEach((s) => { s.points = []; s.box = null; });
  renderSubjects(); drawOverlay();
});

function showPromptFrame(n) {
  S.promptFrame = n;
  pimg.src = `/api/jobs/${S.job}/frame/${n}`;
  pov.width = S.W; pov.height = S.H;
  drawOverlay();
}
$('#sPF').addEventListener('input', (e) => {
  $('#vPF').textContent = e.target.value;
  showPromptFrame(+e.target.value);
});

function drawOverlay() {
  if (!S.W) return;
  pov.width = S.W; pov.height = S.H;
  pctx.clearRect(0, 0, S.W, S.H);
  S.subjects.forEach((s, i) => {
    const on = i === S.active;
    pctx.globalAlpha = on ? 1 : 0.45;
    if (s.box) {
      pctx.strokeStyle = s.dot; pctx.lineWidth = on ? 3 : 2;
      pctx.setLineDash(on ? [] : [7, 5]);
      pctx.strokeRect(s.box[0], s.box[1], s.box[2] - s.box[0], s.box[3] - s.box[1]);
      pctx.setLineDash([]);
    }
    s.points.forEach((p) => {
      pctx.beginPath(); pctx.arc(p[0], p[1], 7, 0, Math.PI * 2);
      pctx.fillStyle = p[2] ? s.dot : '#0f1f18';
      pctx.fill();
      pctx.lineWidth = 2.5; pctx.strokeStyle = p[2] ? '#ffffffcc' : s.dot; pctx.stroke();
    });
    pctx.globalAlpha = 1;
  });
  if (S.dragBox) {
    const b = S.dragBox;
    pctx.strokeStyle = S.subjects[S.active].dot; pctx.lineWidth = 2;
    pctx.setLineDash([6, 4]);
    pctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
    pctx.setLineDash([]);
  }
}

function povXY(e) {
  const r = pov.getBoundingClientRect();
  return [clamp((e.clientX - r.left) / r.width * S.W, 0, S.W - 1),
          clamp((e.clientY - r.top) / r.height * S.H, 0, S.H - 1)];
}

let down = null;
pov.addEventListener('pointerdown', (e) => {
  if (!S.subjects.length) return;
  pov.setPointerCapture(e.pointerId);
  down = { xy: povXY(e), moved: false, neg: e.shiftKey || e.altKey };
});
pov.addEventListener('pointermove', (e) => {
  if (!down) return;
  const p = povXY(e);
  if (Math.abs(p[0] - down.xy[0]) > 5 || Math.abs(p[1] - down.xy[1]) > 5) down.moved = true;
  if (down.moved) {
    S.dragBox = [Math.min(down.xy[0], p[0]), Math.min(down.xy[1], p[1]),
                 Math.max(down.xy[0], p[0]), Math.max(down.xy[1], p[1])];
    drawOverlay();
  }
});
pov.addEventListener('pointerup', (e) => {
  if (!down) return;
  const p = povXY(e), s = S.subjects[S.active];
  if (down.moved && S.dragBox) s.box = S.dragBox.map((v) => Math.round(v));
  else s.points.push([Math.round(p[0]), Math.round(p[1]), down.neg ? 0 : 1]);
  down = null; S.dragBox = null;
  renderSubjects(); drawOverlay();
});

$('#bTrack').addEventListener('click', track);

async function track() {
  const bad = S.subjects.filter((s) => !s.points.length && !s.box);
  if (bad.length) { toast('subject ' + bad[0].id + ' has no prompt yet', true); return; }
  const btn = $('#bTrack'); btn.disabled = true;
  $('#tinfo').hidden = true;
  const prog = $('#prog'); prog.hidden = false;
  $('.bar i', prog).style.width = '0%';
  $('span', prog).textContent = 'loading model…';
  try {
    await api(`/api/jobs/${S.job}/track`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame_idx: S.promptFrame,
        objects: S.subjects.map((s) => ({ id: s.id, points: s.points, box: s.box })),
      }),
    });
    await pollTrack();
  } catch (err) {
    prog.hidden = true;
    const box = $('#tinfo'); box.hidden = false; box.classList.add('err');
    box.textContent = 'track failed: ' + err.message;
  }
  btn.disabled = false;
}

async function pollTrack() {
  const prog = $('#prog'), bar = $('.bar i', prog), lab = $('span', prog);
  for (;;) {
    const st = await api(`/api/jobs/${S.job}/status`);
    const pct = st.n_frames ? (st.done_frames / st.n_frames) * 100 : 0;
    bar.style.width = pct.toFixed(1) + '%';
    if (st.state === 'loading') lab.textContent = 'loading frames…';
    else lab.textContent = `${st.done_frames}/${st.n_frames} · ${st.fps.toFixed(1)} fps`;
    if (st.state === 'done') {
      prog.hidden = true;
      S.tracked = true; S.trackInfo = st;
      const box = $('#tinfo'); box.hidden = false; box.classList.remove('err');
      box.textContent = `tracked ${st.done_frames} frames in ${st.elapsed_s.toFixed(1)} s `
        + `(${st.fps.toFixed(1)} fps) on ${st.device.toUpperCase()} · `
        + `${S.subjects.length} subject${S.subjects.length > 1 ? 's' : ''}`;
      $('#s2sum').textContent = `${st.done_frames}f · ${st.fps.toFixed(1)} fps`;
      await startPreview();
      return;
    }
    if (st.state === 'error') {
      prog.hidden = true;
      const box = $('#tinfo'); box.hidden = false; box.classList.add('err');
      box.textContent = 'track failed: ' + st.error;
      return;
    }
    await new Promise((r) => setTimeout(r, 350));
  }
}

/* ============================================ portable per-cell hash (= py) */
function hash01(i, j, salt, seed) {
  let x = ((Math.imul(i, 73856093) >>> 0) ^ (Math.imul(j, 19349663) >>> 0)
        ^ (Math.imul(salt, 83492791) >>> 0) ^ (Math.imul(seed, 0x9E3779B1) >>> 0)) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x = (x ^ (x >>> 16)) >>> 0;
  return x / 4294967296;
}

/* ============================================================ dither engine */
const E = {
  fkey: null, F: null,          // cell-grid fields
  fc: null, mc: null,           // offscreen contexts
  out: null, outKey: null,
};

function fields(W, H, cell, seed) {
  const key = [W, H, cell, seed].join('|');
  if (E.fkey === key) return E.F;
  const gw = (W / cell) | 0, gh = (H / cell) | 0, N = gw * gh;
  const thr = new Float32Array(N), cx = new Float32Array(N), cy = new Float32Array(N),
        strayR = new Float32Array(N);
  for (let i = 0; i < gh; i++) {
    for (let j = 0; j < gw; j++) {
      const q = i * gw + j;
      thr[q] = S.tile[(i % 64) * 64 + (j % 64)];
      cx[q] = j * cell + cell / 2 + (hash01(i, j, 1, seed) - 0.5) * cell * 0.8;
      cy[q] = i * cell + cell / 2 + (hash01(i, j, 2, seed) - 0.5) * cell * 0.8;
      strayR[q] = hash01(i, j, 3, seed);
    }
  }
  E.fkey = key;
  E.F = { gw, gh, N, thr, cx, cy, strayR, cell };
  return E.F;
}

/* cross-shaped max dilation, edges clamp (mirrors render.py's dilate) */
function dilateCross(a, gh, gw, r) {
  let out = Float32Array.from(a);
  for (let d = 1; d <= r; d++) {
    const nx = Float32Array.from(out);
    for (let i = 0; i < gh; i++) {
      for (let j = 0; j < gw; j++) {
        const q = i * gw + j;
        let v = nx[q];
        if (i - d >= 0 && a[(i - d) * gw + j] > v) v = a[(i - d) * gw + j];
        if (i + d < gh && a[(i + d) * gw + j] > v) v = a[(i + d) * gw + j];
        if (j - d >= 0 && a[i * gw + (j - d)] > v) v = a[i * gw + (j - d)];
        if (j + d < gw && a[i * gw + (j + d)] > v) v = a[i * gw + (j + d)];
        nx[q] = v;
      }
    }
    out = nx;
  }
  return out;
}

function gainForCount(w, thr, target, N) {
  let lo = 1e-3, hi = 1e3;
  for (let it = 0; it < 24; it++) {
    const mid = Math.sqrt(lo * hi);
    let c = 0;
    for (let q = 0; q < N; q++) if (Math.min(w[q] * mid, 1) > thr[q]) c++;
    if (c < target) lo = mid; else hi = mid;
  }
  return Math.sqrt(lo * hi);
}

function ctx2d(w, h, slot) {
  let c = E[slot];
  if (!c || c.canvas.width !== w || c.canvas.height !== h) {
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    c = E[slot] = cv.getContext('2d', { willReadFrequently: true });
  }
  return c;
}

const hexRGB = (s) => {
  const h = s.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
};

/* Draw one dithered frame into `vcv`. bmp = {frame, masks:[ImageBitmap,...]}. */
function ditherFrame(bmp, dots) {
  const P = S.P, W = S.W, H = S.H, cell = P.cell | 0;
  const F = fields(W, H, cell, S.seed);
  const { gw, gh, N, thr, cx, cy, strayR } = F;
  const K = bmp.masks.length;

  const fc = ctx2d(W, H, 'fc');
  fc.drawImage(bmp.frame, 0, 0);
  const fd = fc.getImageData(0, 0, W, H).data;

  const mc = ctx2d(W, H, 'mc');
  const md = [];
  for (let k = 0; k < K; k++) {
    mc.clearRect(0, 0, W, H);
    mc.drawImage(bmp.masks[k], 0, 0);
    md.push(mc.getImageData(0, 0, W, H).data);
  }


  const wgt = [], mg = [];
  for (let k = 0; k < K; k++) { wgt.push(new Float32Array(N)); mg.push(new Float32Array(N)); }
  const useH = gh * cell, useW = gw * cell, inv = P.invert, g1 = P.gamma === 1;
  for (let y = 0; y < useH; y++) {
    const row = (y / cell | 0) * gw, base = y * W * 4;
    for (let x = 0; x < useW; x++) {
      const p = base + x * 4;
      const lum = (0.2126 * fd[p] + 0.7152 * fd[p + 1] + 0.0722 * fd[p + 2]) / 255;
      const t = inv ? lum : 1 - lum;
      const tone = g1 ? t : Math.pow(t < 0 ? 0 : t > 1 ? 1 : t, P.gamma);
      const q = row + (x / cell | 0);
      for (let k = 0; k < K; k++) {
        const m = md[k][p] / 255;
        if (m > 0) { wgt[k][q] += m * tone; mg[k][q] += m; }
      }
    }
  }
  const cc = cell * cell;
  for (let k = 0; k < K; k++) {
    for (let q = 0; q < N; q++) { wgt[k][q] /= cc; mg[k][q] /= cc; }
  }

  // one cell belongs to exactly one subject (the one covering it most)
  const owner = new Int8Array(N), anyMg = new Float32Array(N);
  for (let q = 0; q < N; q++) {
    let best = -1, bv = -1;
    for (let k = 0; k < K; k++) if (mg[k][q] > bv) { bv = mg[k][q]; best = k; }
    owner[q] = bv > 0 ? best : -1;
    anyMg[q] = bv;
  }

  // ---- canvas
  if (!E.out || E.outKey !== W + 'x' + H) {
    E.out = new ImageData(W, H); E.outKey = W + 'x' + H;
  }
  const o = E.out.data, bg = hexRGB(P.bg);
  if (P.mode === 'overlay') {
    for (let p = 0, n = W * H * 4; p < n; p += 4) {
      const lum = (0.2126 * fd[p] + 0.7152 * fd[p + 1] + 0.0722 * fd[p + 2]) / 255;
      const g = (lum * 0.55 + 0.22) * 1.15;
      o[p] = g * bg[0]; o[p + 1] = g * bg[1]; o[p + 2] = g * bg[2]; o[p + 3] = 255;
    }
  } else {
    for (let p = 0, n = W * H * 4; p < n; p += 4) {
      o[p] = bg[0]; o[p + 1] = bg[1]; o[p + 2] = bg[2]; o[p + 3] = 255;
    }
  }

  const dp = P.dotpx | 0, half = dp >> 1;
  let lit = 0;
  for (let k = 0; k < K; k++) {
    const w = new Float32Array(N);
    let cover = 0;
    for (let q = 0; q < N; q++) {
      if (owner[q] === k) { w[q] = wgt[k][q]; if (w[q] > 0) cover++; }
    }
    if (P.n) {
      const tgt = Math.min(P.n, Math.max(1, (P.fill * cover) | 0));
      const g = gainForCount(w, thr, tgt, N);
      for (let q = 0; q < N; q++) w[q] = Math.min(w[q] * g, 1);
    }
    const on = new Uint8Array(N);
    let onN = 0;
    for (let q = 0; q < N; q++) if (w[q] > thr[q]) { on[q] = 1; onN++; }
    if (P.stray > 0 && P.band > 0) {
      const dl = dilateCross(mg[k], gh, gw, P.band | 0);
      let nb = 0;
      for (let q = 0; q < N; q++) if (dl[q] > 0.15 && anyMg[q] <= 0.15) nb++;
      if (nb) {
        const pr = Math.min(1, P.stray * Math.max(onN, 1) / nb);
        for (let q = 0; q < N; q++) {
          if (!on[q] && dl[q] > 0.15 && anyMg[q] <= 0.15 && strayR[q] < pr) on[q] = 1;
        }
      }
    }
    const col = hexRGB(dots[k] || DOTS_FALLBACK[k % 6]);
    for (let q = 0; q < N; q++) {
      if (!on[q]) continue;
      lit++;
      const yc = Math.round(cy[q]), xc = Math.round(cx[q]);
      for (let dy = 0; dy < dp; dy++) {
        const yy = clamp(yc + dy - half, 0, H - 1);
        for (let dx = 0; dx < dp; dx++) {
          const xx = clamp(xc + dx - half, 0, W - 1);
          const p = (yy * W + xx) * 4;
          o[p] = col[0]; o[p + 1] = col[1]; o[p + 2] = col[2]; o[p + 3] = 255;
        }
      }
    }
  }

  const cv = $('#vcv');
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const g = cv.getContext('2d');
  g.putImageData(E.out, 0, 0);
  if (S.compare) {
    // reveal the untouched frame to the left of the divider; export is unaffected
    const x = Math.round(clamp(S.split, 0, 1) * W);
    if (x > 0) {
      g.save();
      g.beginPath(); g.rect(0, 0, x, H); g.clip();
      g.drawImage(bmp.frame, 0, 0);
      g.restore();
    }
  }
  return lit;
}

/* ---------------------------------------------------- frame / mask bitmaps */
const CACHE = new Map();
const CACHE_MAX = 48;

async function bitmapAt(i) {
  const hit = CACHE.get(i);
  if (hit) { CACHE.delete(i); CACHE.set(i, hit); return hit; }
  const ids = S.subjects.map((s) => s.id);
  const [frame, ...masks] = await Promise.all([
    fetch(`/api/jobs/${S.job}/frame/${i}`).then((r) => r.blob()).then(createImageBitmap),
    ...ids.map((id) => fetch(`/api/jobs/${S.job}/mask/${id}/${i}`)
      .then((r) => r.blob()).then(createImageBitmap)),
  ]);
  const rec = { frame, masks };
  CACHE.set(i, rec);
  while (CACHE.size > CACHE_MAX) {
    const k = CACHE.keys().next().value;
    const v = CACHE.get(k); CACHE.delete(k);
    v.frame.close(); v.masks.forEach((m) => m.close());
  }
  return rec;
}

function dropCache() {
  CACHE.forEach((v) => { v.frame.close(); v.masks.forEach((m) => m.close()); });
  CACHE.clear();
}

let drawSeq = 0;
async function drawAt(i) {
  const seq = ++drawSeq;
  const bmp = await bitmapAt(i);
  if (seq !== drawSeq) return 0;
  const t0 = performance.now();
  const lit = ditherFrame(bmp, S.subjects.map((s) => s.dot));
  const ms = performance.now() - t0;
  S.cur = i;
  $('#fcount').textContent = `${i} / ${S.nFrames - 1}`;
  $('#sFrame').value = i;
  $('#fps').textContent = `${(1000 / Math.max(ms, 0.01)).toFixed(1)} fps · ${lit} dots`;
  return lit;
}

async function startPreview() {
  if (!S.tile) S.tile = Float32Array.from((await api('/api/bluenoise?n=64&seed=7')).tile);
  dropCache();
  $('#pwrap').hidden = true; $('#vwrap').hidden = false;
  $('#sFrame').max = S.nFrames - 1;
  openStep(3);
  await drawAt(0);
  $('#s3sum').textContent = S.P.mode;
}

/* --------------------------------------------------------------- transport */
const wipe = $('#wipe');

function setSplit(v) {
  S.split = clamp(v, 0, 1);
  wipe.style.setProperty('--x', (S.split * 100).toFixed(2) + '%');
  if (S.tracked && !S.playing) drawAt(S.cur);   // while playing the loop repaints
}

function setCompare(on) {
  S.compare = on;
  wipe.hidden = !on;
  $('#bCmp').setAttribute('aria-pressed', String(on));
  if (on) setSplit(S.split);
  else if (S.tracked && !S.playing) drawAt(S.cur);
}
$('#bCmp').addEventListener('click', () => setCompare(!S.compare));

let wiping = false;
const wipeAt = (e) => {
  const r = wipe.getBoundingClientRect();
  setSplit((e.clientX - r.left) / r.width);
};
wipe.addEventListener('pointerdown', (e) => {
  wiping = true; wipe.setPointerCapture(e.pointerId); wipeAt(e); e.preventDefault();
});
wipe.addEventListener('pointermove', (e) => { if (wiping) wipeAt(e); });
wipe.addEventListener('pointerup', () => { wiping = false; });
wipe.addEventListener('pointercancel', () => { wiping = false; });
setSplit(0.5);

$('#bPlay').addEventListener('click', () => { S.playing ? stop() : play(); });
$('#sFrame').addEventListener('input', (e) => { stop(); drawAt(+e.target.value); });

function stop() {
  S.playing = false;
  $('#bPlay').setAttribute('aria-pressed', 'false');
  $('#bPlay').textContent = 'play';
}
function play() {
  if (!S.tracked) return;
  S.playing = true;
  $('#bPlay').setAttribute('aria-pressed', 'true');
  $('#bPlay').textContent = 'pause';
  loop();
}
async function loop() {
  while (S.playing) {
    const t0 = performance.now();
    const next = (S.cur + 1) % S.nFrames;
    // keep a few frames warm ahead of playback
    for (let k = 1; k <= 4; k++) bitmapAt((next + k) % S.nFrames);
    await drawAt(next);
    const wait = Math.max(0, 1000 / S.fps - (performance.now() - t0));
    await new Promise((r) => setTimeout(r, wait));
  }
}

/* ------------------------------------------------------------ style panel */
function bindSlider(id, out, key, fmt, int) {
  const el = $(id);
  el.addEventListener('input', () => {
    const v = int ? parseInt(el.value, 10) : parseFloat(el.value);
    S.P[key] = v;
    $(out).textContent = fmt(v);
    if (S.tracked) drawAt(S.cur);
  });
}
bindSlider('#sN', '#vN', 'n', (v) => String(v), true);
bindSlider('#sCell', '#vCell', 'cell', (v) => v + ' px', true);
bindSlider('#sDot', '#vDot', 'dotpx', (v) => v + ' px', true);
bindSlider('#sGam', '#vGam', 'gamma', (v) => v.toFixed(2));
bindSlider('#sFill', '#vFill', 'fill', (v) => v.toFixed(2));
bindSlider('#sStray', '#vStray', 'stray', (v) => v.toFixed(3));
bindSlider('#sBand', '#vBand', 'band', (v) => String(v), true);

$$('[data-mode]').forEach((b) => b.addEventListener('click', () => {
  S.P.mode = b.dataset.mode;
  $$('[data-mode]').forEach((o) => o.setAttribute('aria-pressed', o === b ? 'true' : 'false'));
  $('#s3sum').textContent = S.P.mode;
  if (S.tracked) drawAt(S.cur);
}));
$('#tInv').addEventListener('click', () => {
  S.P.invert = !S.P.invert;
  $('#tInv').setAttribute('aria-pressed', String(S.P.invert));
  if (S.tracked) drawAt(S.cur);
});

function renderPalettes() {
  const wrap = $('#pals'); wrap.textContent = '';
  S.palettes.forEach((p, i) => {
    const b = document.createElement('button');
    b.className = 'chip pal';
    b.setAttribute('aria-pressed', i === S.pal ? 'true' : 'false');
    const pv = document.createElement('span'); pv.className = 'pv';
    [p.bg, p.dots[0], p.dots[1]].forEach((c) => {
      const s = document.createElement('b'); s.style.background = c; pv.append(s);
    });
    const nm = document.createElement('span'); nm.textContent = p.name;
    b.append(pv, nm);
    b.addEventListener('click', () => {
      S.pal = i; S.P.bg = p.bg;
      S.subjects.forEach((s, k) => { s.dot = p.dots[k % p.dots.length]; });
      renderPalettes(); renderSubjects();
      if (S.tracked) drawAt(S.cur);
    });
    wrap.append(b);
  });
}

function renderDotCols() {
  const wrap = $('#dotcols'); if (!wrap) return;
  wrap.textContent = '';
  S.subjects.forEach((s) => {
    const l = document.createElement('label');
    l.className = 'chip'; l.style.cursor = 'pointer';
    const inp = document.createElement('input');
    inp.type = 'color'; inp.value = s.dot;
    inp.addEventListener('input', () => {
      s.dot = inp.value; renderSubjects(); drawOverlay();
      if (S.tracked) drawAt(S.cur);
    });
    const t = document.createElement('span'); t.textContent = ' #' + s.id;
    l.append(inp, t);
    wrap.append(l);
  });
}

/* ============================================================= step 4: mp4 */
$('#bRender').addEventListener('click', doRender);

async function doRender() {
  if (!S.tracked) { toast('track first', true); return; }
  const btn = $('#bRender'); btn.disabled = true;
  $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
  const prog = $('#rprog'); prog.hidden = false;
  $('.bar i', prog).style.width = '0%'; $('span', prog).textContent = 'starting…';
  try {
    await api(`/api/jobs/${S.job}/render`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({}, S.P, {
        subjects: S.subjects.map((s) => ({ id: s.id, dot: s.dot })), seed: S.seed,
      })),
    });
    for (;;) {
      const st = (await api(`/api/jobs/${S.job}/status`)).render;
      const pct = st.n_frames ? (st.done_frames / st.n_frames) * 100 : 0;
      $('.bar i', prog).style.width = pct.toFixed(1) + '%';
      $('span', prog).textContent = `${st.done_frames}/${st.n_frames}`;
      if (st.state === 'done') {
        prog.hidden = true;
        const url = `/api/jobs/${S.job}/out.mp4?t=${Date.now()}`;
        const dl = $('#dl'); dl.href = url; dl.hidden = false;
        const v = $('#outvid'); v.src = url; v.hidden = false;
        const box = $('#rinfo'); box.hidden = false; box.classList.remove('err');
        box.textContent = `rendered ${st.done_frames} frames in ${st.elapsed_s.toFixed(1)} s `
          + `(${st.fps.toFixed(1)} fps)`;
        $('#s4sum').textContent = 'ready';
        break;
      }
      if (st.state === 'error') {
        prog.hidden = true;
        const box = $('#rinfo'); box.hidden = false; box.classList.add('err');
        box.textContent = 'render failed: ' + st.error;
        break;
      }
      await new Promise((r) => setTimeout(r, 300));
    }
  } catch (err) {
    prog.hidden = true;
    const box = $('#rinfo'); box.hidden = false; box.classList.add('err');
    box.textContent = 'render failed: ' + err.message;
  }
  btn.disabled = false;
}

/* ------------------------------------------------------------------ boot */
(async function boot() {
  try {
    const j = await api('/api/palettes');
    S.palettes = j.palettes;
    S.P.bg = j.palettes[0].bg;
    $('#dev').textContent = j.device;
    renderPalettes();
  } catch (e) {
    S.palettes = [{ name: 'sage', bg: '#c9d4c5', dots: DOTS_FALLBACK }];
    renderPalettes();
  }
  api('/api/bluenoise?n=64&seed=7').then((b) => { S.tile = Float32Array.from(b.tile); })
    .catch(() => {});
  window.DV = S;          // handy for the playwright checks
  window.DV_draw = drawAt;
})();
