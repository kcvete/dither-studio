/* ---------------------------------------------------------------------------
   DITHER STUDIO — one flow for three jobs.

     still           drop an image, dither it, download a PNG (never leaves the tab)
     clip            drop a video, every frame gets dithered, export an MP4
     clip + subject  point at something, EdgeTAM tracks it through the clip, and
                     only that gets dithered

   The preview is not an approximation: it runs static/dither.js, which dither.py
   mirrors pixel for pixel (parity.py is the gate). What plays here is what the
   MP4 contains.
--------------------------------------------------------------------------- */
'use strict';

const $ = (s, r) => (r || document).querySelector(s);
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const PREVIEW_MAX = 1600;      // longest edge used for the live still preview
const MAX_SUBJECTS = 6;
const CACHE_MAX = 40;

const S = {
  kind: 'none',                // none | image | video
  // still
  bitmap: null, natW: 0, natH: 0, fileName: '',
  // clip
  job: null, nFrames: 0, W: 0, H: 0, fps: 30,
  scope: 'whole', subjects: [], active: 0, nextId: 1, promptFrame: 0,
  trackSize: 1024, tracked: false, playing: false, cur: 0,
  // look
  P: {
    mode: 'bluenoise', algo: 'floyd-steinberg', matrix: 4, serpentine: false,
    strength: 1, brightness: 0, contrast: 1, gamma: 1, invert: false, pixel: 1,
    compose: 'cutout', seed: 7,
    n: 8000, cell: 4, dotpx: 3, fill: 0.7, stray: 0.02, band: 9,
  },
  palette: ['#000000', '#ffffff'],   // background / whole-frame palette
  bg: '#c9d4c5',
  target: 'bg',                      // which palette the editor is editing: 'bg' | subject id
  meta: null,
  compare: false, split: 0.5,
};

/* ------------------------------------------------------------------ chrome */
function toast(msg, err) {
  const t = $('#toast');
  t.textContent = msg; t.hidden = false; t.classList.toggle('err', !!err);
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, err ? 7000 : 3000);
}
const busy = (on) => { $('#busy').hidden = !on; };

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let d = r.statusText;
    try { d = (await r.json()).detail || d; } catch (e) { /* not json */ }
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

/* ======================================================== step 1: source */
$('#drop').addEventListener('click', () => $('#file').click());
$('#bPick').addEventListener('click', () => $('#file').click());
$('#file').addEventListener('change', (e) => { if (e.target.files[0]) take(e.target.files[0]); });
['dragenter', 'dragover'].forEach((e) => document.addEventListener(e, (ev) => {
  ev.preventDefault(); document.body.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((e) => document.addEventListener(e, (ev) => {
  ev.preventDefault();
  if (e === 'drop' && ev.dataTransfer.files[0]) take(ev.dataTransfer.files[0]);
  document.body.classList.remove('dragging');
}));
document.addEventListener('paste', (e) => {
  const it = Array.from(e.clipboardData.files)[0];
  if (it) take(it);
});
$('#sSec').addEventListener('input', (e) => { $('#vSec').textContent = e.target.value + ' s'; });

function take(f) {
  const isVid = /^video\//.test(f.type) || /\.(mp4|mov|m4v)$/i.test(f.name);
  $('#vidopts').hidden = !isVid;
  return isVid ? uploadClip(f) : loadStill(f);
}

function showSteps(kind) {
  $('#st2').hidden = kind !== 'video';
  $('#st3').hidden = $('#st4').hidden = $('#st5').hidden = kind === 'none';
  $('#empty').hidden = kind !== 'none';
  $$('.step .sh i').forEach((el, i) => { el.textContent = i + 1; });
  // renumber visible steps so the rail always reads 1,2,3…
  let n = 0;
  $$('.step').forEach((st) => { if (!st.hidden) $('.sh i', st).textContent = ++n; });
}

async function loadStill(f) {
  const box = $('#upstat'); box.hidden = false; box.classList.remove('err');
  box.textContent = 'reading ' + f.name + '…';
  try {
    const bmp = await createImageBitmap(f);
    S.kind = 'image'; S.bitmap = bmp; S.natW = bmp.width; S.natH = bmp.height;
    S.fileName = f.name.replace(/\.[^.]+$/, '');
    S.tracked = false; S.subjects = []; S.scope = 'whole';
    if (S.P.mode === 'dots') setMode('bluenoise');
    box.textContent = `${bmp.width} × ${bmp.height} · ${(f.size / 1024).toFixed(0)} KB · stays in this tab`;
    $('#s1sum').textContent = `${bmp.width}×${bmp.height}`;
    showSteps('image');
    $('#pwrap').hidden = true; $('#vwrap').hidden = false;
    $('#bPlay').hidden = $('#sFrame').hidden = $('#fcount').hidden = true;
    $('#composeui').hidden = true; $('#bgui').hidden = true;
    $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
    $('#s5sum').textContent = ''; $('#bExport').textContent = 'Download PNG';
    buildTargets(); renderModes(); openStep(3);
    await draw();
  } catch (err) {
    box.classList.add('err'); box.textContent = 'could not read that image: ' + err.message;
  }
}

async function uploadClip(f) {
  const box = $('#upstat'); box.hidden = false; box.classList.remove('err');
  box.textContent = 'uploading ' + f.name + '…';
  busy(true);
  const fd = new FormData();
  fd.append('file', f);
  fd.append('max_seconds', $('#sSec').value);
  try {
    const j = await api('/api/upload', { method: 'POST', body: fd });
    S.kind = 'video'; S.job = j.job; S.nFrames = j.n_frames; S.W = j.w; S.H = j.h; S.fps = j.fps;
    S.fileName = f.name.replace(/\.[^.]+$/, '');
    S.tracked = false; S.subjects = []; S.nextId = 1; S.cur = 0; S.promptFrame = 0;
    S.scope = 'whole';
    dropCache();
    box.textContent = `${j.n_frames} frames · ${j.w}×${j.h} · ${j.fps} fps`;
    $('#s1sum').textContent = `${j.n_frames}f`;
    $('#sPF').max = j.n_frames - 1; $('#sPF').value = 0; $('#vPF').textContent = '0';
    $('#sFrame').max = j.n_frames - 1;
    $('#bPlay').hidden = $('#sFrame').hidden = $('#fcount').hidden = false;
    $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
    $('#tinfo').hidden = true; $('#s5sum').textContent = '';
    $('#bExport').textContent = 'Render MP4';
    if (S.P.mode === 'dots') setMode('bluenoise');
    showSteps('video'); setScope('whole');
    buildTargets(); renderModes(); openStep(2);
    await draw();
  } catch (err) {
    box.classList.add('err'); box.textContent = 'upload failed: ' + err.message;
  }
  busy(false);
}

/* ==================================================== step 2: subjects */
$$('[data-scope]').forEach((b) => b.addEventListener('click', () => setScope(b.dataset.scope)));

function setScope(v) {
  S.scope = v;
  $$('[data-scope]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.scope === v)));
  $('#trackui').hidden = v !== 'track';
  $('#wholenote').hidden = v === 'track';
  $('#composeui').hidden = !(v === 'track' && S.tracked);
  $('#bgui').hidden = !(v === 'track' && S.tracked && S.P.compose === 'cutout');
  if (v === 'track') {
    if (!S.subjects.length) addSubject();
    $('#pwrap').hidden = S.tracked; $('#vwrap').hidden = !S.tracked;
    if (!S.tracked) showPromptFrame(S.promptFrame);
  } else {
    $('#pwrap').hidden = true; $('#vwrap').hidden = false;
    draw();
  }
  buildTargets(); renderModes();
  $('#s2sum').textContent = v === 'track'
    ? (S.tracked ? `${S.subjects.length} tracked` : `${S.subjects.length} subj`) : 'whole clip';
}

function subjectColor(i) { return (S.meta.subject_colors || ['#b0413e'])[i % 6]; }

function addSubject() {
  if (S.subjects.length >= MAX_SUBJECTS) return;
  const i = S.subjects.length;
  S.subjects.push({ id: S.nextId++, palette: [S.bg, subjectColor(i)], points: [], box: null });
  S.active = S.subjects.length - 1;
  renderSubjects(); buildTargets();
}
$('#bAdd').addEventListener('click', addSubject);
$('#bClr').addEventListener('click', () => {
  S.subjects.forEach((s) => { s.points = []; s.box = null; });
  renderSubjects(); drawOverlay();
});

function renderSubjects() {
  const wrap = $('#subs'); wrap.textContent = '';
  S.subjects.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'chip sub';
    b.setAttribute('aria-pressed', String(i === S.active));
    const sw = document.createElement('span');
    sw.className = 'sw'; sw.style.background = s.palette[s.palette.length - 1];
    const nm = document.createElement('span');
    const np = s.points.length;
    nm.textContent = `#${s.id}` + (np || s.box ? ` · ${np}pt${s.box ? '+box' : ''}` : '');
    b.append(sw, nm);
    b.addEventListener('click', () => { S.active = i; renderSubjects(); drawOverlay(); });
    if (S.subjects.length > 1) {
      const x = document.createElement('span');
      x.className = 'x'; x.textContent = '✕';
      x.addEventListener('click', (e) => {
        e.stopPropagation();
        S.subjects.splice(i, 1);
        S.active = Math.min(S.active, S.subjects.length - 1);
        renderSubjects(); drawOverlay(); buildTargets();
      });
      b.append(x);
    }
    wrap.append(b);
  });
  $('#vSubs').textContent = `${S.subjects.length} / ${MAX_SUBJECTS}`;
}

const pimg = $('#pimg'), pov = $('#pov'), pctx = pov.getContext('2d');

function showPromptFrame(n) {
  S.promptFrame = n;
  pimg.src = `/api/jobs/${S.job}/frame/${n}`;
  pov.width = S.W; pov.height = S.H;
  drawOverlay();
}
$('#sPF').addEventListener('input', (e) => {
  $('#vPF').textContent = e.target.value; showPromptFrame(+e.target.value);
});

function drawOverlay() {
  if (!S.W || S.kind !== 'video') return;
  pov.width = S.W; pov.height = S.H;
  pctx.clearRect(0, 0, S.W, S.H);
  S.subjects.forEach((s, i) => {
    const col = s.palette[s.palette.length - 1], on = i === S.active;
    pctx.globalAlpha = on ? 1 : 0.4;
    if (s.box) {
      pctx.strokeStyle = col; pctx.lineWidth = on ? 3 : 2;
      pctx.setLineDash(on ? [] : [7, 5]);
      pctx.strokeRect(s.box[0], s.box[1], s.box[2] - s.box[0], s.box[3] - s.box[1]);
      pctx.setLineDash([]);
    }
    s.points.forEach((p) => {
      pctx.beginPath(); pctx.arc(p[0], p[1], 7, 0, Math.PI * 2);
      pctx.fillStyle = p[2] ? col : '#0f1f18'; pctx.fill();
      pctx.lineWidth = 2.5; pctx.strokeStyle = p[2] ? '#ffffffcc' : col; pctx.stroke();
    });
    pctx.globalAlpha = 1;
  });
  if (S.dragBox && S.subjects[S.active]) {
    const b = S.dragBox, s = S.subjects[S.active];
    pctx.strokeStyle = s.palette[s.palette.length - 1]; pctx.lineWidth = 2;
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
  if (down.moved && S.dragBox) s.box = S.dragBox.map(Math.round);
  else s.points.push([Math.round(p[0]), Math.round(p[1]), down.neg ? 0 : 1]);
  down = null; S.dragBox = null;
  renderSubjects(); drawOverlay();
});

/* ---- tracking quality: the square the tracker resizes every frame to ----
 * The clip is untouched; masks come back at the source resolution whatever
 * this says. Lower = a coarser outline, and a lot more frames per second. */
function buildTrackSizes() {
  const wrap = $('#tq');
  const sizes = S.meta.track_sizes || [{ size: 1024, id: 'best', label: 'best', fps: 0 }];
  S.trackSize = S.meta.default_track_size || sizes[sizes.length - 1].size;
  wrap.innerHTML = '';
  sizes.forEach((t) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.dataset.size = t.size;
    b.setAttribute('aria-pressed', String(t.size === S.trackSize));
    b.innerHTML = `${t.label} · ${t.size} px`
      + (t.fps ? ` <em class="fl">${t.fps.toFixed(0)} fps</em>` : '');
    b.addEventListener('click', () => {
      S.trackSize = t.size;
      $$('#tq .chip').forEach((c) => c.setAttribute('aria-pressed',
        String(+c.dataset.size === S.trackSize)));
      paintTrackSize();
    });
    wrap.appendChild(b);
  });
  paintTrackSize();
}

function paintTrackSize() {
  const t = (S.meta.track_sizes || []).find((x) => x.size === S.trackSize);
  $('#vTQ').textContent = t ? `${t.label} · ${t.size} px` : `${S.trackSize} px`;
}

$('#bTrack').addEventListener('click', track);

async function track() {
  const bad = S.subjects.filter((s) => !s.points.length && !s.box);
  if (bad.length) { toast('subject #' + bad[0].id + ' has no prompt yet', true); return; }
  const btn = $('#bTrack'); btn.disabled = true;
  $('#tinfo').hidden = true; $('#tinfo').textContent = '';
  const prog = $('#prog'); prog.hidden = false;
  $('.bar i', prog).style.width = '0%';
  $('span', prog).textContent = 'loading model…';
  try {
    await api(`/api/jobs/${S.job}/track`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame_idx: S.promptFrame,
        image_size: S.trackSize,
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
    bar.style.width = (st.n_frames ? (st.done_frames / st.n_frames) * 100 : 0).toFixed(1) + '%';
    lab.textContent = st.state === 'loading' ? 'loading frames…'
      : `${st.done_frames}/${st.n_frames} · ${st.fps.toFixed(1)} fps`;
    if (st.state === 'done') {
      prog.hidden = true; S.tracked = true;
      const box = $('#tinfo'); box.hidden = false; box.classList.remove('err');
      box.textContent = `tracked ${st.done_frames} frames in ${st.elapsed_s.toFixed(1)} s `
        + `(${st.fps.toFixed(1)} fps) on ${st.device.toUpperCase()} ${st.backend || st.precision || ''} · `
        + `${S.subjects.length} subject${S.subjects.length > 1 ? 's' : ''}`;
      $('#s2sum').textContent = `${st.done_frames}f · ${st.fps.toFixed(1)} fps`;
      $('#pwrap').hidden = true; $('#vwrap').hidden = false;
      $('#composeui').hidden = false;
      $('#bgui').hidden = S.P.compose !== 'cutout';
      dropCache(); buildTargets(); renderModes(); openStep(3);
      await draw();
      return;
    }
    if (st.state === 'error') {
      prog.hidden = true;
      const box = $('#tinfo'); box.hidden = false; box.classList.add('err');
      box.textContent = 'track failed: ' + st.error;
      return;
    }
    await sleep(350);
  }
}

/* ===================================================== frames + masks cache */
const CACHE = new Map();

async function frameAt(i) {
  const hit = CACHE.get(i);
  if (hit) { CACHE.delete(i); CACHE.set(i, hit); return hit; }
  const ids = usingSubjects() ? S.subjects.map((s) => s.id) : [];
  const [frame, ...masks] = await Promise.all([
    fetch(`/api/jobs/${S.job}/frame/${i}`).then((r) => r.blob()).then(createImageBitmap),
    ...ids.map((id) => fetch(`/api/jobs/${S.job}/mask/${id}/${i}`)
      .then((r) => r.blob()).then(createImageBitmap)),
  ]);
  const rec = { frame, masks };
  CACHE.set(i, rec);
  while (CACHE.size > CACHE_MAX) {
    const k = CACHE.keys().next().value, v = CACHE.get(k);
    CACHE.delete(k); v.frame.close(); v.masks.forEach((m) => m.close());
  }
  return rec;
}
function dropCache() {
  CACHE.forEach((v) => { v.frame.close(); v.masks.forEach((m) => m.close()); });
  CACHE.clear();
}
const usingSubjects = () => S.kind === 'video' && S.scope === 'track' && S.tracked
  && S.subjects.length > 0;

/* ------------------------------------------------------ offscreen contexts */
const CTX = {};
function ctx2d(w, h, slot) {
  let c = CTX[slot];
  if (!c || c.canvas.width !== w || c.canvas.height !== h) {
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    c = CTX[slot] = cv.getContext('2d', { willReadFrequently: true });
  }
  return c;
}
function bitmapAlpha(bmp, w, h, slot) {
  const c = ctx2d(w, h, slot);
  c.clearRect(0, 0, w, h);
  c.drawImage(bmp, 0, 0, w, h);
  const d = c.getImageData(0, 0, w, h).data;
  const out = new Float32Array(w * h);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) out[i] = d[p] / 255;
  return out;
}

/* =============================================== the "dots" particle look */
const DOTS = { key: null, F: null };
function dotFields(W, H, cell, seed, tile) {
  const key = [W, H, cell, seed].join('|');
  if (DOTS.key === key) return DOTS.F;
  const gw = (W / cell) | 0, gh = (H / cell) | 0, N = gw * gh;
  const thr = new Float32Array(N), cx = new Float32Array(N), cy = new Float32Array(N),
        strayR = new Float32Array(N);
  for (let i = 0; i < gh; i++) for (let j = 0; j < gw; j++) {
    const q = i * gw + j;
    thr[q] = tile[(i % 64) * 64 + (j % 64)];
    cx[q] = j * cell + cell / 2 + (Dither.hash01(i, j, 1, seed) - 0.5) * cell * 0.8;
    cy[q] = i * cell + cell / 2 + (Dither.hash01(i, j, 2, seed) - 0.5) * cell * 0.8;
    strayR[q] = Dither.hash01(i, j, 3, seed);
  }
  DOTS.key = key; DOTS.F = { gw, gh, N, thr, cx, cy, strayR, cell };
  return DOTS.F;
}
function dilateCross(a, gh, gw, r) {
  let out = Float32Array.from(a);
  for (let d = 1; d <= r; d++) {
    const nx = Float32Array.from(out);
    for (let i = 0; i < gh; i++) for (let j = 0; j < gw; j++) {
      const q = i * gw + j; let v = nx[q];
      if (i - d >= 0 && a[(i - d) * gw + j] > v) v = a[(i - d) * gw + j];
      if (i + d < gh && a[(i + d) * gw + j] > v) v = a[(i + d) * gw + j];
      if (j - d >= 0 && a[i * gw + (j - d)] > v) v = a[i * gw + (j - d)];
      if (j + d < gw && a[i * gw + (j + d)] > v) v = a[i * gw + (j + d)];
      nx[q] = v;
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

function renderDots(srcData, W, H, masks, P, palettes, bg, tile) {
  const cell = P.cell | 0;
  const F = dotFields(W, H, cell, P.seed, tile);
  const { gw, gh, N, thr, cx, cy, strayR } = F;
  const K = masks.length;
  const wgt = [], mg = [];
  for (let k = 0; k < K; k++) { wgt.push(new Float32Array(N)); mg.push(new Float32Array(N)); }
  const useH = gh * cell, useW = gw * cell, inv = P.invert, g1 = P.gamma === 1;
  for (let y = 0; y < useH; y++) {
    const row = ((y / cell) | 0) * gw, base = y * W * 4;
    for (let x = 0; x < useW; x++) {
      const p = base + x * 4;
      const lum = (0.2126 * srcData[p] + 0.7152 * srcData[p + 1] + 0.0722 * srcData[p + 2]) / 255;
      const t = inv ? lum : 1 - lum;
      const tone = g1 ? t : Math.pow(clamp(t, 0, 1), P.gamma);
      const q = row + ((x / cell) | 0);
      for (let k = 0; k < K; k++) {
        const m = masks[k][y * W + x];
        if (m > 0) { wgt[k][q] += m * tone; mg[k][q] += m; }
      }
    }
  }
  const cc = cell * cell;
  for (let k = 0; k < K; k++) for (let q = 0; q < N; q++) { wgt[k][q] /= cc; mg[k][q] /= cc; }

  const owner = new Int8Array(N), anyMg = new Float32Array(N);
  for (let q = 0; q < N; q++) {
    let best = -1, bv = -1;
    for (let k = 0; k < K; k++) if (mg[k][q] > bv) { bv = mg[k][q]; best = k; }
    owner[q] = bv > 0 ? best : -1; anyMg[q] = bv;
  }

  const out = new Uint8ClampedArray(W * H * 4), bgc = Dither.hexRGB(bg);
  if (P.compose === 'overlay') {
    for (let p = 0, n = W * H * 4; p < n; p += 4) {
      const lum = (0.2126 * srcData[p] + 0.7152 * srcData[p + 1] + 0.0722 * srcData[p + 2]) / 255;
      const g = (lum * 0.55 + 0.22) * 1.15;
      out[p] = g * bgc[0]; out[p + 1] = g * bgc[1]; out[p + 2] = g * bgc[2]; out[p + 3] = 255;
    }
  } else {
    for (let p = 0, n = W * H * 4; p < n; p += 4) {
      out[p] = bgc[0]; out[p + 1] = bgc[1]; out[p + 2] = bgc[2]; out[p + 3] = 255;
    }
  }

  const dp = P.dotpx | 0, half = dp >> 1;
  let lit = 0;
  for (let k = 0; k < K; k++) {
    const w = new Float32Array(N);
    let cover = 0;
    for (let q = 0; q < N; q++) if (owner[q] === k) { w[q] = wgt[k][q]; if (w[q] > 0) cover++; }
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
    const pal = palettes[k + 1] || palettes[0];
    const col = Dither.hexRGB(pal[pal.length - 1]);
    for (let q = 0; q < N; q++) {
      if (!on[q]) continue;
      lit++;
      const yc = Math.round(cy[q]), xc = Math.round(cx[q]);
      for (let dy = 0; dy < dp; dy++) {
        const yy = clamp(yc + dy - half, 0, H - 1);
        for (let dx = 0; dx < dp; dx++) {
          const xx = clamp(xc + dx - half, 0, W - 1);
          const p = (yy * W + xx) * 4;
          out[p] = col[0]; out[p + 1] = col[1]; out[p + 2] = col[2]; out[p + 3] = 255;
        }
      }
    }
  }
  return { out, lit };
}

/* ========================================================== the main draw */
let BLUE = null;
let drawSeq = 0;

function palettesForRender() {
  return [S.palette].concat(S.subjects.map((s) => s.palette));
}

/* fit the source into the preview budget; stills only, clips are already 720p */
function previewSize() {
  if (S.kind !== 'image') return [S.W, S.H];
  const m = Math.max(S.natW, S.natH);
  if (m <= PREVIEW_MAX) return [S.natW, S.natH];
  const k = PREVIEW_MAX / m;
  return [Math.max(1, Math.round(S.natW * k)), Math.max(1, Math.round(S.natH * k))];
}

/* Render one image/frame into `cv`. Returns {lit, ms}. */
function paint(cv, srcData, W, H, masks, opts) {
  const t0 = performance.now();
  const pal = palettesForRender();
  let out, lit = 0;
  if (S.P.mode === 'dots' && masks.length) {
    const r = renderDots(srcData, W, H, masks, S.P, pal, S.bg, BLUE);
    out = r.out; lit = r.lit;
  } else {
    out = Dither.composeFrame(srcData, W, H, masks, S.P, pal, S.bg);
  }
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  const g = cv.getContext('2d');
  g.putImageData(new ImageData(out, W, H), 0, 0);
  if (S.compare && opts && opts.original) {
    const x = Math.round(clamp(S.split, 0, 1) * W);
    if (x > 0) {
      g.save(); g.beginPath(); g.rect(0, 0, x, H); g.clip();
      g.drawImage(opts.original, 0, 0, W, H); g.restore();
    }
  }
  return { lit, ms: performance.now() - t0 };
}

async function draw(i) {
  const seq = ++drawSeq;
  if (S.kind === 'image') {
    const [w, h] = previewSize();
    const c = ctx2d(w, h, 'src');
    c.drawImage(S.bitmap, 0, 0, w, h);
    const src = c.getImageData(0, 0, w, h).data;
    if (seq !== drawSeq) return;
    const r = paint($('#vcv'), src, w, h, [], { original: S.bitmap });
    $('#fps').textContent = `${w}×${h} · ${r.ms.toFixed(0)} ms`;
    return;
  }
  if (S.kind !== 'video') return;
  const idx = i === undefined ? S.cur : i;
  const bmp = await frameAt(idx);
  if (seq !== drawSeq) return;
  const c = ctx2d(S.W, S.H, 'src');
  c.drawImage(bmp.frame, 0, 0);
  const src = c.getImageData(0, 0, S.W, S.H).data;
  const masks = usingSubjects()
    ? bmp.masks.map((m, k) => bitmapAlpha(m, S.W, S.H, 'm' + k)) : [];
  const r = paint($('#vcv'), src, S.W, S.H, masks, { original: bmp.frame });
  S.cur = idx;
  $('#fcount').textContent = `${idx} / ${S.nFrames - 1}`;
  $('#sFrame').value = idx;
  $('#fps').textContent = `${(1000 / Math.max(r.ms, 0.01)).toFixed(1)} fps`
    + (S.P.mode === 'dots' ? ` · ${r.lit} dots` : '');
  return r.lit;
}

/* ------------------------------------------------------------- transport */
const wipe = $('#wipe');
function setSplit(v) {
  S.split = clamp(v, 0, 1);
  wipe.style.setProperty('--x', (S.split * 100).toFixed(2) + '%');
  if (!S.playing) draw();
}
function setCompare(on) {
  S.compare = on; wipe.hidden = !on;
  $('#bCmp').setAttribute('aria-pressed', String(on));
  if (on) setSplit(S.split); else draw();
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

$('#bPlay').addEventListener('click', () => (S.playing ? stop() : play()));
$('#sFrame').addEventListener('input', (e) => { stop(); draw(+e.target.value); });
function stop() {
  S.playing = false;
  $('#bPlay').setAttribute('aria-pressed', 'false'); $('#bPlay').textContent = 'play';
}
function play() {
  if (S.kind !== 'video') return;
  S.playing = true;
  $('#bPlay').setAttribute('aria-pressed', 'true'); $('#bPlay').textContent = 'pause';
  loop();
}
async function loop() {
  while (S.playing) {
    const t0 = performance.now();
    const next = (S.cur + 1) % S.nFrames;
    for (let k = 1; k <= 3; k++) frameAt((next + k) % S.nFrames);
    await draw(next);
    await sleep(Math.max(0, 1000 / S.fps - (performance.now() - t0)));
  }
}

/* ========================================================= step 3: look */
function setMode(id) {
  S.P.mode = id;
  const ed = id === 'errordiff';
  $('#edui').hidden = !ed;
  $('#mxui').hidden = !(id === 'ordered' || id === 'halftone');
  $('#dotsui').hidden = id !== 'dots';
  $('#pxui').hidden = id === 'dots';
  const m = (S.meta.modes || []).find((x) => x.id === id);
  const risky = S.kind === 'video' && S.meta.stable && S.meta.stable[id] === false;
  $('#modenote').textContent = m ? m.note : '';
  $('#modenote').classList.toggle('warn', !!risky);
  $('#s3sum').textContent = m ? m.name : id;
  renderModes();
}

function renderModes() {
  const wrap = $('#modes'); if (!S.meta) return;
  wrap.textContent = '';
  S.meta.modes.forEach((m) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.dataset.mode = m.id;
    b.setAttribute('aria-pressed', String(S.P.mode === m.id));
    b.textContent = m.name;
    const needsSubjects = m.id === 'dots';
    const ok = !needsSubjects || usingSubjects();
    if (!ok) { b.classList.add('off'); b.title = 'track a subject first'; }
    if (S.kind === 'video' && S.meta.stable[m.id] === false) {
      const w = document.createElement('i'); w.className = 'fl'; w.textContent = '≈';
      w.title = 'flickers frame to frame'; b.append(w);
    }
    b.addEventListener('click', () => {
      if (!ok) { toast('the dots look needs a tracked subject', true); return; }
      setMode(m.id); draw();
    });
    wrap.append(b);
  });
}

function bindSlider(id, out, key, fmt, int) {
  const el = $(id);
  el.addEventListener('input', () => {
    S.P[key] = int ? parseInt(el.value, 10) : parseFloat(el.value);
    $(out).textContent = fmt(S.P[key]);
    draw();
  });
}
bindSlider('#sN', '#vN', 'n', String, true);
bindSlider('#sCell', '#vCell', 'cell', (v) => v + ' px', true);
bindSlider('#sDot', '#vDot', 'dotpx', (v) => v + ' px', true);
bindSlider('#sFill', '#vFill', 'fill', (v) => v.toFixed(2));
bindSlider('#sStray', '#vStray', 'stray', (v) => v.toFixed(3));
bindSlider('#sBand', '#vBand', 'band', String, true);
bindSlider('#sStr', '#vStr', 'strength', (v) => v.toFixed(2));
bindSlider('#sPx', '#vPx', 'pixel', (v) => v + '×', true);
bindSlider('#sBri', '#vBri', 'brightness', (v) => v.toFixed(2));
bindSlider('#sCon', '#vCon', 'contrast', (v) => v.toFixed(2));
bindSlider('#sGam', '#vGam', 'gamma', (v) => v.toFixed(2));

$('#bTone').addEventListener('click', () => {
  S.P.brightness = 0; S.P.contrast = 1; S.P.gamma = 1;
  $('#sBri').value = 0; $('#vBri').textContent = '0.00';
  $('#sCon').value = 1; $('#vCon').textContent = '1.00';
  $('#sGam').value = 1; $('#vGam').textContent = '1.00';
  draw();
});
$('#tInv').addEventListener('click', () => {
  S.P.invert = !S.P.invert;
  $('#tInv').setAttribute('aria-pressed', String(S.P.invert));
  draw();
});
$('#tSerp').addEventListener('click', () => {
  S.P.serpentine = !S.P.serpentine;
  $('#tSerp').setAttribute('aria-pressed', String(S.P.serpentine));
  draw();
});
$('#sAlgo').addEventListener('change', (e) => { S.P.algo = e.target.value; draw(); });
$$('[data-mx]').forEach((b) => b.addEventListener('click', () => {
  S.P.matrix = +b.dataset.mx;
  $$('[data-mx]').forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
  draw();
}));
$$('[data-compose]').forEach((b) => b.addEventListener('click', () => {
  S.P.compose = b.dataset.compose;
  $$('[data-compose]').forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
  $('#bgui').hidden = S.P.compose !== 'cutout';
  draw();
}));
$('#bSeed').addEventListener('click', async () => {
  S.P.seed = 1 + Math.floor(Math.random() * 100000);
  BLUE = Float32Array.from((await api('/api/bluenoise?n=64&seed=' + S.P.seed)).tile);
  Dither.setBlueNoise(BLUE);
  DOTS.key = null;
  draw();
});

/* ====================================================== step 4: palette */
function currentPalette() {
  if (S.target === 'bg') return S.palette;
  const s = S.subjects.find((x) => String(x.id) === String(S.target));
  return s ? s.palette : S.palette;
}
function setPalette(list) {
  if (S.target === 'bg') S.palette = list;
  else {
    const s = S.subjects.find((x) => String(x.id) === String(S.target));
    if (s) s.palette = list;
  }
  renderSwatches(); renderSubjects(); drawOverlay(); draw();
}

function buildTargets() {
  const wrap = $('#target'); if (!wrap) return;
  wrap.textContent = '';
  const withSubs = usingSubjects();
  if (!withSubs) { S.target = 'bg'; wrap.hidden = true; renderSwatches(); return; }
  wrap.hidden = false;
  const mk = (id, label) => {
    const b = document.createElement('button');
    b.className = 'chip'; b.textContent = label;
    b.setAttribute('aria-pressed', String(String(S.target) === String(id)));
    b.addEventListener('click', () => { S.target = id; buildTargets(); });
    wrap.append(b);
  };
  mk('bg', S.P.compose === 'overlay' ? 'scene' : 'background');
  S.subjects.forEach((s) => mk(s.id, '#' + s.id));
  renderSwatches();
}

function renderPalettes() {
  const wrap = $('#pals'); wrap.textContent = '';
  S.meta.palettes.forEach((p) => {
    const b = document.createElement('button');
    b.className = 'chip pal';
    const pv = document.createElement('span'); pv.className = 'pv';
    p.colors.slice(0, 5).forEach((c) => {
      const s = document.createElement('b'); s.style.background = c; pv.append(s);
    });
    const nm = document.createElement('span'); nm.textContent = p.name;
    b.append(pv, nm);
    b.addEventListener('click', () => setPalette(p.colors.slice()));
    wrap.append(b);
  });
}

function renderSwatches() {
  const wrap = $('#swatches'); wrap.textContent = '';
  const list = currentPalette();
  $('#vNc').textContent = list.length + ' colours';
  $('#s4sum').textContent = list.length + 'c';
  list.forEach((c, i) => {
    const l = document.createElement('label');
    l.className = 'chip sw1';
    const inp = document.createElement('input');
    inp.type = 'color'; inp.value = c;
    inp.addEventListener('input', () => {
      const l2 = currentPalette().slice(); l2[i] = inp.value; setPalette(l2);
    });
    l.append(inp);
    if (list.length > 2) {
      const x = document.createElement('span');
      x.className = 'x'; x.textContent = '✕';
      x.addEventListener('click', (e) => {
        e.preventDefault(); e.stopPropagation();
        const l2 = currentPalette().slice(); l2.splice(i, 1); setPalette(l2);
      });
      l.append(x);
    }
    wrap.append(l);
  });
}
$('#bAddCol').addEventListener('click', () => {
  const l = currentPalette().slice();
  if (l.length >= 32) return;
  l.push('#888888'); setPalette(l);
});
$('#bFromImg').addEventListener('click', async () => {
  const n = currentPalette().length;
  let data, w, h;
  if (S.kind === 'image') {
    [w, h] = previewSize();
    const c = ctx2d(w, h, 'src'); c.drawImage(S.bitmap, 0, 0, w, h);
    data = c.getImageData(0, 0, w, h).data;
  } else if (S.kind === 'video') {
    const bmp = await frameAt(S.cur);
    w = S.W; h = S.H;
    const c = ctx2d(w, h, 'src'); c.drawImage(bmp.frame, 0, 0);
    data = c.getImageData(0, 0, w, h).data;
  } else return;
  setPalette(Dither.extractPalette(data, w, h, Math.max(2, n)));
  toast(`palette pulled from the ${S.kind === 'image' ? 'image' : 'current frame'}`);
});
$('#cBg').addEventListener('input', (e) => { S.bg = e.target.value; draw(); });

/* ======================================================= step 5: export */
$('#bExport').addEventListener('click', () => (S.kind === 'image' ? exportPNG() : exportMP4()));

async function exportPNG() {
  const info = $('#rinfo'); info.hidden = true; info.textContent = '';
  busy(true);
  await sleep(16);
  try {
    // re-render at the source's native resolution, not the preview's
    const c = ctx2d(S.natW, S.natH, 'exp');
    c.drawImage(S.bitmap, 0, 0, S.natW, S.natH);
    const src = c.getImageData(0, 0, S.natW, S.natH).data;
    const out = Dither.composeFrame(src, S.natW, S.natH, [], S.P, palettesForRender(), S.bg);
    const cv = document.createElement('canvas');
    cv.width = S.natW; cv.height = S.natH;
    cv.getContext('2d').putImageData(new ImageData(out, S.natW, S.natH), 0, 0);
    const blob = await new Promise((r) => cv.toBlob(r, 'image/png'));
    const dl = $('#dl');
    if (dl.dataset.url) URL.revokeObjectURL(dl.dataset.url);
    const url = URL.createObjectURL(blob);
    dl.dataset.url = url; dl.href = url;
    dl.download = `${S.fileName || 'dither'}-${S.P.mode}.png`;
    dl.hidden = false;
    const box = $('#rinfo'); box.hidden = false; box.classList.remove('err');
    box.textContent = `${S.natW}×${S.natH} PNG · ${(blob.size / 1024).toFixed(0)} KB`;
    $('#s5sum').textContent = 'ready';
  } catch (err) {
    const box = $('#rinfo'); box.hidden = false; box.classList.add('err');
    box.textContent = 'export failed: ' + err.message;
  }
  busy(false);
}

async function exportMP4() {
  const btn = $('#bExport'); btn.disabled = true;
  $('#dl').hidden = true; $('#outvid').hidden = true;
  const info = $('#rinfo'); info.hidden = true; info.textContent = '';
  const prog = $('#rprog'); prog.hidden = false;
  $('.bar i', prog).style.width = '0%'; $('span', prog).textContent = 'starting…';
  try {
    const body = Object.assign({}, S.P, {
      bg: S.bg, palette: S.palette,
      subjects: usingSubjects() ? S.subjects.map((s) => ({ id: s.id, palette: s.palette })) : [],
    });
    await api(`/api/jobs/${S.job}/render`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    for (;;) {
      const st = (await api(`/api/jobs/${S.job}/status`)).render;
      $('.bar i', prog).style.width =
        (st.n_frames ? (st.done_frames / st.n_frames) * 100 : 0).toFixed(1) + '%';
      $('span', prog).textContent = `${st.done_frames}/${st.n_frames}`;
      if (st.state === 'done') {
        prog.hidden = true;
        const url = `/api/jobs/${S.job}/out.mp4?t=${Date.now()}`;
        const dl = $('#dl');
        dl.href = url; dl.download = `${S.fileName || 'dither'}-${S.P.mode}.mp4`; dl.hidden = false;
        const v = $('#outvid'); v.src = url; v.hidden = false;
        const box = $('#rinfo'); box.hidden = false; box.classList.remove('err');
        box.textContent = `rendered ${st.done_frames} frames in ${st.elapsed_s.toFixed(1)} s `
          + `(${st.fps.toFixed(1)} fps)`;
        $('#s5sum').textContent = 'ready';
        break;
      }
      if (st.state === 'error') {
        prog.hidden = true;
        const box = $('#rinfo'); box.hidden = false; box.classList.add('err');
        box.textContent = 'render failed: ' + st.error;
        break;
      }
      await sleep(300);
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
    S.meta = await api('/api/palettes');
  } catch (e) {
    S.meta = { palettes: Dither.PALETTES, modes: Dither.MODES, stable: Dither.STABLE,
               kernels: [], subject_colors: ['#b0413e'], device: '?' };
  }
  $('#dev').textContent = (S.meta.device || '')
    + (S.meta.backend && S.meta.backend !== 'auto' ? ' ' + S.meta.backend
       : S.meta.precision ? ' ' + S.meta.precision : '');
  buildTrackSizes();
  const sel = $('#sAlgo');
  (S.meta.kernels.length ? S.meta.kernels
    : Object.entries(Dither.KERNELS).map(([id, v]) => ({ id, name: v.name })))
    .forEach((k) => {
      const o = document.createElement('option');
      o.value = k.id; o.textContent = k.name; sel.append(o);
    });
  sel.value = S.P.algo;
  renderPalettes();
  try {
    BLUE = Float32Array.from((await api('/api/bluenoise?n=64&seed=7')).tile);
  } catch (e) {
    BLUE = new Float32Array(4096).map((_, i) => Dither.hash01(i >> 6, i & 63, 5, 7));
  }
  Dither.setBlueNoise(BLUE);
  setMode(S.P.mode);
  buildTargets();
  showSteps('none');
  window.DV = S;
  window.DV_draw = draw;
  window.DV_ready = true;
})();
