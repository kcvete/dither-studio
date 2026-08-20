/* ---------------------------------------------------------------------------
   DITHER STUDIO — one flow for three jobs.

     still           drop an image, dither it, download a PNG (never leaves the tab)
     clip            drop a video, every frame gets dithered, export an MP4
     clip + subject  point at something, EdgeTAM tracks it through the clip, and
                     only that gets dithered

   The preview is not an approximation: it runs web/dither.js, which
   server/dither.py mirrors pixel for pixel (server/parity.py is the gate).
   What plays here is what the export contains.

   Nothing below knows whether the tracking happened in this tab or on a
   server -- every call that could need one goes through `S.engine`, which is
   one of web/engines/{browser,remote}.js. That is the whole of the free /
   local / paid split.
--------------------------------------------------------------------------- */
'use strict';

import { chooseEngine, probeRemote, loadPref, savePref,
         BrowserEngine, RemoteEngine } from './engines/index.js';

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
  // `promptFrame` is where the SCRUBBER is. Each subject remembers the frame it
  // was actually prompted on -- a ball that flies in at frame 80 does not exist
  // on frame 0, so one prompt frame per clip was never enough.
  scope: 'whole', subjects: [], active: 0, nextId: 1, promptFrame: 0,
  tool: 'point', curPath: null, hoverXY: null, previewMasks: null,
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
  // engine
  engine: null, enginePref: null, modelsMissing: null,
  exportURL: null, frameURL: null,
};
const E = () => S.engine;

/* ------------------------------------------------------------------ chrome */
function toast(msg, err) {
  const t = $('#toast');
  t.textContent = msg; t.hidden = false; t.classList.toggle('err', !!err);
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, err ? 7000 : 3000);
}
const busy = (on) => { $('#busy').hidden = !on; };

/* Nothing here talks to a URL any more; this is only for the odd bit of code
 * that wants a readable message out of an engine failure. */
const why = (err) => (err && err.message) || String(err);

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
    $('#offframe').hidden = true;
    buildTargets(); renderModes(); openStep(3);
    await draw();
  } catch (err) {
    box.classList.add('err'); box.textContent = 'could not read that image: ' + err.message;
  }
}

async function uploadClip(f) {
  const box = $('#upstat'); box.hidden = false; box.classList.remove('err');
  box.textContent = (E().id === 'browser' ? 'decoding ' : 'uploading ') + f.name + '…';
  busy(true);
  try {
    const j = await E().open(f, {
      maxSeconds: +$('#sSec').value,
      onProgress: (p) => { if (p.text) box.textContent = p.text; },
    });
    S.kind = 'video'; S.job = j.job; S.nFrames = j.nFrames; S.W = j.w; S.H = j.h; S.fps = j.fps;
    S.fileName = f.name.replace(/\.[^.]+$/, '');
    S.tracked = false; S.subjects = []; S.nextId = 1; S.cur = 0; S.promptFrame = 0;
    S.scope = 'whole';
    dropCache();
    box.textContent = `${j.nFrames} frames · ${j.w}×${j.h} · ${j.fps} fps`
      + (E().id === 'browser' ? ' · stays in this tab' : '');
    $('#s1sum').textContent = `${j.nFrames}f`;
    $('#sPF').max = j.nFrames - 1; $('#sPF').value = 0; $('#vPF').textContent = '0';
    $('#sFrame').max = j.nFrames - 1;
    $('#bPlay').hidden = $('#sFrame').hidden = $('#fcount').hidden = false;
    $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
    $('#tinfo').hidden = true; $('#s5sum').textContent = '';
    $('#bExport').textContent = 'Render ' + E().supports.exportExt.toUpperCase();
    if (S.P.mode === 'dots') setMode('bluenoise');
    showSteps('video'); setScope('whole');
    buildTargets(); renderModes(); openStep(2);
    await draw();
  } catch (err) {
    box.classList.add('err');
    box.textContent = (E().id === 'browser' ? 'could not read that clip: '
      : 'upload failed: ') + why(err);
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
  // promptFrame stays null until the subject actually gets a prompt, so it
  // adopts whatever frame the user was looking at when they drew it
  S.subjects.push({ id: S.nextId++, palette: [S.bg, subjectColor(i)],
                    points: [], box: null, paths: [], promptFrame: null });
  S.active = S.subjects.length - 1;
  renderSubjects(); buildTargets();
}
/* Once a clip is tracked the transport takes over the canvas, so anything that
 * edits a prompt has to hand the prompt canvas back — otherwise there is no way
 * to correct a bad track short of reloading. The masks stay until you re-track. */
function backToPrompt() {
  if (S.kind !== 'video' || S.scope !== 'track') return;
  $('#pwrap').hidden = false; $('#vwrap').hidden = true;
  S.playing = false;
  showPromptFrame(S.promptFrame);
}
$('#bAdd').addEventListener('click', () => { addSubject(); backToPrompt(); });
$('#bClr').addEventListener('click', () => {
  S.subjects.forEach((s) => { s.points = []; s.box = null; s.paths = []; s.promptFrame = null; });
  S.curPath = null; S.previewMasks = null; $('#pvinfo').hidden = true;
  renderSubjects(); backToPrompt(); drawOverlay();
});

/* A subject's prompt frame is decided by the first mark placed on it, and never
 * moves afterwards unless the subject is cleared. */
const hasPrompt = (s) => !!(s.points.length || s.box || (s.paths || []).length);
const frameOf = (s) => (s.promptFrame === null ? S.promptFrame : s.promptFrame);
const onThisFrame = (s) => frameOf(s) === S.promptFrame;

function claimFrame(s) {
  if (s.promptFrame === null) { s.promptFrame = S.promptFrame; paintOffFrame(); }
}

/* Subjects prompted on some other frame: named, with a way back to them. */
function paintOffFrame() {
  const box = $('#offframe');
  const away = S.subjects.filter((s) => hasPrompt(s) && !onThisFrame(s));
  if (!away.length || S.tracked) { box.hidden = true; box.textContent = ''; return; }
  box.hidden = false; box.textContent = '';
  away.forEach((s, i) => {
    if (i) box.append(document.createTextNode(' · '));
    const a = document.createElement('button');
    a.className = 'lnk';
    a.textContent = `#${s.id} prompted @ ${s.promptFrame} — jump`;
    a.addEventListener('click', () => {
      $('#sPF').value = s.promptFrame; $('#vPF').textContent = String(s.promptFrame);
      S.active = S.subjects.indexOf(s);
      showPromptFrame(s.promptFrame); renderSubjects();
    });
    box.append(a);
  });
}

function renderSubjects() {
  const wrap = $('#subs'); wrap.textContent = '';
  S.subjects.forEach((s, i) => {
    const b = document.createElement('button');
    b.className = 'chip sub';
    b.setAttribute('aria-pressed', String(i === S.active));
    const sw = document.createElement('span');
    sw.className = 'sw'; sw.style.background = s.palette[s.palette.length - 1];
    const nm = document.createElement('span');
    const np = s.points.length, nl = (s.paths || []).length;
    const bits = [];
    if (nl) bits.push(`${nl} shape${nl > 1 ? 's' : ''}`);
    else if (np || s.box) bits.push(`${np}pt${s.box ? '+box' : ''}`);
    if (s.promptFrame !== null) bits.push('@ ' + s.promptFrame);
    nm.textContent = `#${s.id}` + (bits.length ? ' · ' + bits.join(' ') : '');
    if (hasPrompt(s) && !onThisFrame(s)) b.classList.add('away');
    b.append(sw, nm);
    // selecting a subject that lives on another frame goes there, otherwise
    // its marks would be invisible and the next click would land on the wrong
    // frame and be silently ignored
    b.addEventListener('click', () => {
      S.active = i;
      if (s.promptFrame !== null && s.promptFrame !== S.promptFrame && !S.tracked) {
        $('#sPF').value = s.promptFrame; $('#vPF').textContent = String(s.promptFrame);
        showPromptFrame(s.promptFrame);
      }
      renderSubjects(); drawOverlay();
    });
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
  paintOffFrame();
}

const pimg = $('#pimg'), pov = $('#pov'), pctx = pov.getContext('2d');

let promptSeq = 0;
async function showPromptFrame(n) {
  S.promptFrame = n;
  S.previewMasks = null; $('#pvinfo').hidden = true;
  pov.width = S.W; pov.height = S.H;
  drawOverlay();
  renderSubjects();
  const seq = ++promptSeq;
  try {
    const got = await E().frameURL(n);
    if (seq !== promptSeq) { if (got.revoke) URL.revokeObjectURL(got.url); return; }
    if (S.frameURL) URL.revokeObjectURL(S.frameURL);
    S.frameURL = got.revoke ? got.url : null;
    pimg.src = got.url;
  } catch (err) {
    toast('could not read frame ' + n + ': ' + why(err), true);
  }
}
$('#sPF').addEventListener('input', (e) => {
  $('#vPF').textContent = e.target.value; showPromptFrame(+e.target.value);
});

function drawOverlay() {
  if (!S.W || S.kind !== 'video') return;
  pov.width = S.W; pov.height = S.H;
  pctx.clearRect(0, 0, S.W, S.H);
  S.subjects.forEach((s, i) => {
    if (!onThisFrame(s)) return;              // its marks belong to another frame
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
  drawPaths();
  drawPreviewMasks();
}

/* ---- lasso / polygon shapes ------------------------------------------------
 * A shape is {op:'add'|'sub', pts:[[x,y],…]} in clip-native pixels. They are
 * rasterised to a binary PNG at the clip's own size and sent as a mask prompt;
 * EdgeTAM takes a mask OR points+box for a given frame and object, never both,
 * so a subject with shapes ignores its clicks (the UI says so). */
function strokeShape(p, col, open, hover) {
  if (!p.pts.length) return;
  const path = () => {
    pctx.beginPath();
    pctx.moveTo(p.pts[0][0], p.pts[0][1]);
    for (let i = 1; i < p.pts.length; i++) pctx.lineTo(p.pts[i][0], p.pts[i][1]);
    if (hover) pctx.lineTo(hover[0], hover[1]);
    if (!open) pctx.closePath();
  };
  pctx.save();
  pctx.lineJoin = 'round'; pctx.lineCap = 'round';
  pctx.setLineDash([]); pctx.lineWidth = 5; pctx.strokeStyle = 'rgba(0,0,0,.55)';
  path(); pctx.stroke();
  pctx.setLineDash(open ? [9, 7] : []);
  pctx.lineWidth = 2.5; pctx.strokeStyle = col; path(); pctx.stroke();
  if (open) {
    pctx.setLineDash([]); pctx.fillStyle = col;
    for (const q of p.pts) pctx.fillRect(q[0] - 3, q[1] - 3, 6, 6);
  }
  pctx.restore();
}

function drawPaths() {
  S.subjects.forEach((s, i) => {
    if (!onThisFrame(s)) return;
    const col = s.palette[s.palette.length - 1];
    pctx.globalAlpha = i === S.active ? 1 : 0.4;
    (s.paths || []).forEach((p) => strokeShape(p, p.op === 'add' ? col : '#ff9d7c', false));
    pctx.globalAlpha = 1;
  });
  if (S.curPath && S.subjects[S.active]) {
    const col = S.subjects[S.active].palette.slice(-1)[0];
    strokeShape(S.curPath, S.curPath.op === 'add' ? col : '#ff9d7c', true,
                S.tool === 'poly' ? S.hoverXY : null);
  }
}

function drawPreviewMasks() {
  if (!S.previewMasks) return;
  S.subjects.forEach((s) => {
    const im = S.previewMasks[String(s.id)];
    if (!im || !im.complete) return;
    const t = document.createElement('canvas');
    t.width = S.W; t.height = S.H;
    const g = t.getContext('2d');
    g.drawImage(im, 0, 0, S.W, S.H);          // white-ish where the mask is
    g.globalCompositeOperation = 'source-in';
    g.fillStyle = s.palette[s.palette.length - 1];
    g.fillRect(0, 0, S.W, S.H);
    pctx.save(); pctx.globalAlpha = 0.45; pctx.drawImage(t, 0, 0); pctx.restore();
  });
}

/* one binary mask per subject, at the clip's own resolution */
function subjectMaskDataURL(s) {
  const shapes = (s.paths || []).filter((p) => p.pts.length > 2);
  if (!shapes.length) return null;
  const c = document.createElement('canvas');
  c.width = S.W; c.height = S.H;
  const g = c.getContext('2d');
  g.fillStyle = '#000'; g.fillRect(0, 0, S.W, S.H);
  if (!shapes.some((p) => p.op === 'add')) { g.fillStyle = '#fff'; g.fillRect(0, 0, S.W, S.H); }
  for (const p of shapes) {
    g.fillStyle = p.op === 'add' ? '#fff' : '#000';
    g.beginPath();
    g.moveTo(p.pts[0][0], p.pts[0][1]);
    for (let i = 1; i < p.pts.length; i++) g.lineTo(p.pts[i][0], p.pts[i][1]);
    g.closePath(); g.fill();
  }
  return c.toDataURL('image/png');
}

function povXY(e) {
  const r = pov.getBoundingClientRect();
  return [clamp((e.clientX - r.left) / r.width * S.W, 0, S.W - 1),
          clamp((e.clientY - r.top) / r.height * S.H, 0, S.H - 1)];
}
let down = null;
const MINSTEP = 4;                      // lasso: drop points closer than this
function commitPath() {
  const c = S.curPath;
  S.curPath = null; S.hoverXY = null;
  if (c && c.pts.length >= 3) {
    const s = S.subjects[S.active];
    claimFrame(s);
    s.paths.push(c); S.previewMasks = null;
  }
  renderSubjects(); drawOverlay();
}
pov.addEventListener('pointerdown', (e) => {
  if (!S.subjects.length) return;
  const act = S.subjects[S.active];
  if (!onThisFrame(act)) {
    toast(`subject #${act.id} was prompted on frame ${act.promptFrame} — `
      + 'jump back to it, or add a new subject for this frame', true);
    return;
  }
  const p = povXY(e), neg = e.shiftKey || e.altKey;
  if (S.tool === 'point') {
    pov.setPointerCapture(e.pointerId);
    down = { xy: p, moved: false, neg };
    return;
  }
  if (S.tool === 'lasso') {
    pov.setPointerCapture(e.pointerId);
    S.curPath = { op: neg ? 'sub' : 'add', pts: [p] };
    down = { lasso: true };
  } else {                                                   // polygon
    if (!S.curPath) S.curPath = { op: neg ? 'sub' : 'add', pts: [p] };
    else S.curPath.pts.push(p);
    S.hoverXY = p;
  }
  drawOverlay();
});
pov.addEventListener('pointermove', (e) => {
  const p = povXY(e);
  if (S.tool === 'poly') { if (S.curPath) { S.hoverXY = p; drawOverlay(); } return; }
  if (!down) return;
  if (down.lasso) {
    const last = S.curPath.pts[S.curPath.pts.length - 1];
    if (Math.hypot(p[0] - last[0], p[1] - last[1]) < MINSTEP) return;
    S.curPath.pts.push(p); drawOverlay(); return;
  }
  if (Math.abs(p[0] - down.xy[0]) > 5 || Math.abs(p[1] - down.xy[1]) > 5) down.moved = true;
  if (down.moved) {
    S.dragBox = [Math.min(down.xy[0], p[0]), Math.min(down.xy[1], p[1]),
                 Math.max(down.xy[0], p[0]), Math.max(down.xy[1], p[1])];
    drawOverlay();
  }
});
pov.addEventListener('pointerup', (e) => {
  if (!down) return;
  if (down.lasso) { down = null; commitPath(); return; }
  const p = povXY(e), s = S.subjects[S.active];
  claimFrame(s);
  if (down.moved && S.dragBox) s.box = S.dragBox.map(Math.round);
  else s.points.push([Math.round(p[0]), Math.round(p[1]), down.neg ? 0 : 1]);
  down = null; S.dragBox = null; S.previewMasks = null;
  renderSubjects(); drawOverlay();
});
pov.addEventListener('dblclick', (e) => {
  if (S.tool === 'poly' && S.curPath) { e.preventDefault(); commitPath(); }
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && S.tool === 'poly' && S.curPath) { e.preventDefault(); commitPath(); }
  if (e.key === 'Escape' && S.curPath) { S.curPath = null; S.hoverXY = null; drawOverlay(); }
});

/* ---- prompt tool ---- */
const TOOLHINT = {
  point: "click what you want · shift-click what you don't · drag a box",
  lasso: 'drag around the subject · shift-drag to subtract · esc cancels',
  poly: 'click each corner · double-click or enter to close · esc cancels',
};
const TOOLNOTE = {
  point: 'Click = keep this · shift-click = not this · drag = a box. '
       + 'The tracker re-derives the outline itself on every frame.',
  lasso: 'Drag to draw around the subject · shift-drag subtracts · esc cancels. '
       + 'A drawn shape replaces this subject\'s clicks and box — EdgeTAM takes '
       + 'a mask prompt or points, never both on one frame.',
  poly: 'Click each corner, double-click or enter to close · shift for a subtracting '
      + 'shape · esc cancels. A drawn shape replaces this subject\'s clicks and box.',
};
function paintTool() {
  $$('#ptool .chip').forEach((c) => c.setAttribute('aria-pressed',
    String(c.dataset.tool === S.tool)));
  $('#vTool').textContent = S.tool === 'point' ? 'point / box'
    : S.tool === 'lasso' ? 'lasso' : 'polygon';
  $('#toolnote').textContent = TOOLNOTE[S.tool];
  $('#phint').textContent = TOOLHINT[S.tool];
  $('#bUndo').hidden = S.tool === 'point';
  pov.style.cursor = S.tool === 'poly' ? 'copy' : 'crosshair';
}
$$('#ptool .chip').forEach((c) => c.addEventListener('click', () => {
  if (S.curPath) commitPath();
  S.tool = c.dataset.tool; paintTool(); backToPrompt(); drawOverlay();
}));
$('#bUndo').addEventListener('click', () => {
  const s = S.subjects[S.active];
  if (!s) return;
  if (S.curPath) { S.curPath = null; S.hoverXY = null; }
  else if (s.paths.length) s.paths.pop();
  renderSubjects(); drawOverlay();
});
paintTool();

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
  $('#vTQ').textContent = t ? t.label : `${S.trackSize} px`;
}

$('#bTrack').addEventListener('click', track);

/* The engine-neutral prompt: clip-pixel coordinates, one prompt frame each. */
function promptPayload(only) {
  return S.subjects
    .filter((s) => !only || only.includes(s))
    .map((s) => {
      const mask = subjectMaskDataURL(s);
      const frameIdx = frameOf(s);
      return mask ? { id: s.id, mask, frameIdx }
                  : { id: s.id, points: s.points, box: s.box, frameIdx };
    });
}

/* ---- preview: the first-frame prediction only, no propagation ---- */
$('#bPrev').addEventListener('click', previewFrame);
async function previewFrame() {
  if (S.curPath) commitPath();
  backToPrompt();
  // one frame, so only the subjects that were actually prompted on it
  const here = S.subjects.filter((s) => hasPrompt(s) && onThisFrame(s));
  if (!here.length) {
    toast('nothing is prompted on frame ' + S.promptFrame + ' yet', true);
    return;
  }
  const btn = $('#bPrev'); btn.disabled = true;
  const info = $('#pvinfo'); info.hidden = false; info.classList.remove('err');
  info.textContent = 'predicting this frame…';
  try {
    const r = await E().previewFrame({
      frameIdx: S.promptFrame, imageSize: S.trackSize,
      objects: promptPayload(here),
    }, (m) => { info.textContent = m; });
    const imgs = {};
    for (const o of r.objects) imgs[String(o.id)] = o.image;
    S.previewMasks = imgs;
    drawOverlay();
    info.textContent = `frame ${r.frameIdx} · ${r.objects.length} subject`
      + `${r.objects.length > 1 ? 's' : ''} in ${r.elapsedS.toFixed(2)} s `
      + `(${r.imageSize} px) · `
      + r.objects.map((o) => '#' + o.id + ' ' + o.area + ' px').join(' · ')
      + (r.note ? ' · ' + r.note : '');
  } catch (err) {
    info.classList.add('err');
    info.textContent = 'preview failed: ' + why(err);
  }
  btn.disabled = false;
}

async function track() {
  if (S.curPath) commitPath();
  const bad = S.subjects.filter((s) => !hasPrompt(s));
  if (bad.length) { toast('subject #' + bad[0].id + ' has no prompt yet', true); return; }
  const btn = $('#bTrack'); btn.disabled = true;
  $('#tinfo').hidden = true; $('#tinfo').textContent = '';
  const prog = $('#prog'); prog.hidden = false;
  const bar = $('.bar i', prog), lab = $('span', prog);
  bar.style.width = '0%';
  lab.textContent = 'loading model…';
  try {
    const st = await E().track(
      { objects: promptPayload(), imageSize: S.trackSize },
      (p) => {
        bar.style.width = (p.total ? (p.done / p.total) * 100 : 0).toFixed(1) + '%';
        lab.textContent = p.text;
      });
    prog.hidden = true; S.tracked = true;
    const spread = new Set(S.subjects.map(frameOf));
    const box = $('#tinfo'); box.hidden = false; box.classList.remove('err');
    box.textContent = `tracked ${st.frames} frames in ${st.elapsedS.toFixed(1)} s `
      + `(${st.fps.toFixed(1)} fps) on ${st.device} ${st.backend || ''} · `
      + `${S.subjects.length} subject${S.subjects.length > 1 ? 's' : ''}`
      + (spread.size > 1 ? ` prompted on ${spread.size} different frames` : '')
      + (st.note ? ' · ' + st.note : '');
    $('#s2sum').textContent = `${st.frames}f · ${st.fps.toFixed(1)} fps`;
    $('#pwrap').hidden = true; $('#vwrap').hidden = false;
    $('#offframe').hidden = true;
    $('#composeui').hidden = false;
    $('#bgui').hidden = S.P.compose !== 'cutout';
    dropCache(); buildTargets(); renderModes(); openStep(3);
    await draw();
  } catch (err) {
    prog.hidden = true;
    const box = $('#tinfo'); box.hidden = false; box.classList.add('err');
    box.textContent = 'track failed: ' + why(err);
  }
  btn.disabled = false;
}

/* ===================================================== frames + masks cache */
const CACHE = new Map();

async function frameAt(i) {
  const hit = CACHE.get(i);
  if (hit) { CACHE.delete(i); CACHE.set(i, hit); return hit; }
  const ids = usingSubjects() ? S.subjects.map((s) => s.id) : [];
  const [frame, ...masks] = await Promise.all([
    E().frame(i),
    ...ids.map((id) => E().mask(id, i)),
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
  const btn = $('#bSeed'); btn.disabled = true;
  S.P.seed = 1 + Math.floor(Math.random() * 100000);
  try {
    BLUE = await E().blueNoise(64, S.P.seed);
  } catch (e) {
    BLUE = new Float32Array(4096).map((_, i) => Dither.hash01(i >> 6, i & 63, 5, S.P.seed));
  }
  Dither.setBlueNoise(BLUE);
  DOTS.key = null;
  btn.disabled = false;
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
$('#bExport').addEventListener('click', () => (S.kind === 'image' ? exportPNG() : exportClip()));

/* One frame of the finished picture at the clip's own resolution — what the
 * browser engine feeds its recorder, frame by frame. */
async function composeAt(i) {
  const rec = await frameAt(i);
  const c = ctx2d(S.W, S.H, 'exp');
  c.clearRect(0, 0, S.W, S.H);
  c.drawImage(rec.frame, 0, 0);
  const src = c.getImageData(0, 0, S.W, S.H).data;
  const masks = usingSubjects()
    ? rec.masks.map((m, k) => bitmapAlpha(m, S.W, S.H, 'x' + k)) : [];
  const pal = palettesForRender();
  const out = (S.P.mode === 'dots' && masks.length)
    ? renderDots(src, S.W, S.H, masks, S.P, pal, S.bg, BLUE).out
    : Dither.composeFrame(src, S.W, S.H, masks, S.P, pal, S.bg);
  return new ImageData(out, S.W, S.H);
}

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

async function exportClip() {
  const btn = $('#bExport'); btn.disabled = true;
  $('#dl').hidden = true; $('#outvid').hidden = true;
  const info = $('#rinfo'); info.hidden = true; info.textContent = '';
  const prog = $('#rprog'); prog.hidden = false;
  const bar = $('.bar i', prog), lab = $('span', prog);
  bar.style.width = '0%'; lab.textContent = 'starting…';
  stop();
  try {
    const params = Object.assign({}, S.P, {
      bg: S.bg, palette: S.palette, fps: S.fps,
      subjects: usingSubjects()
        ? S.subjects.map((s) => ({ id: s.id, palette: s.palette })) : [],
    });
    const r = await E().exportClip(params, (p) => {
      bar.style.width = (p.total ? (p.done / p.total) * 100 : 0).toFixed(1) + '%';
      lab.textContent = p.text;
    }, composeAt);
    prog.hidden = true;
    if (S.exportURL) { URL.revokeObjectURL(S.exportURL); S.exportURL = null; }
    if (r.url.startsWith('blob:')) S.exportURL = r.url;
    const dl = $('#dl');
    dl.href = r.url;
    dl.download = `${S.fileName || 'dither'}-${S.P.mode}.${r.ext}`;
    dl.hidden = false;
    if (r.playable) { const v = $('#outvid'); v.src = r.url; v.hidden = false; }
    const box = $('#rinfo'); box.hidden = false; box.classList.remove('err');
    box.textContent = `rendered ${r.frames} frames in ${r.elapsedS.toFixed(1)} s `
      + `(${r.fps.toFixed(1)} fps)`
      + (r.bytes ? ` · ${(r.bytes / 1e6).toFixed(1)} MB` : '')
      + (r.note ? ` · ${r.note}` : '');
    $('#s5sum').textContent = 'ready';
  } catch (err) {
    prog.hidden = true;
    const box = $('#rinfo'); box.hidden = false; box.classList.add('err');
    box.textContent = 'render failed: ' + why(err);
  }
  btn.disabled = false;
}

/* ======================================================== the engine chip */
function paintEngine() {
  const e = E();
  const chip = $('#engine'), name = $('#engName');
  name.textContent = e.label + (e.sublabel ? ' · ' + e.sublabel : '');
  chip.dataset.engine = e.id;
  chip.title = e.id === 'browser'
    ? 'Tracking, dithering and encoding all run in this tab. Nothing is uploaded.'
    : `Tracking and encoding run on ${e.baseUrl || 'this origin'}.`;
  $('#dev').hidden = true;
  const mode = S.enginePref.mode;
  $$('#engpop .opt').forEach((b) => b.setAttribute('aria-pressed',
    String(b.dataset.eng === mode
      || (mode === 'auto' && b.dataset.eng === (e.id === 'browser' ? 'browser' : 'local')))));
  const ext = e.supports.exportExt.toUpperCase();
  $('#bExport').textContent = S.kind === 'image' ? 'Download PNG' : 'Render ' + ext;
}

function openEnginePop(on) {
  const pop = $('#engpop');
  pop.hidden = !on;
  $('#engine').setAttribute('aria-expanded', String(!!on));
  if (on) {
    const p = S.enginePref;
    $('#engUrl').value = p.url || '';
    $('#engKey').value = p.key || '';
    $('#engcustom').hidden = p.mode !== 'custom';
  }
}
$('#engine').addEventListener('click', (e) => {
  e.stopPropagation();
  openEnginePop($('#engpop').hidden);
});
document.addEventListener('click', (e) => {
  if (!$('#engpop').hidden && !$('#engpop').contains(e.target)) openEnginePop(false);
});
$$('#engpop .opt').forEach((b) => b.addEventListener('click', async () => {
  const eng = b.dataset.eng;
  if (eng === 'custom') { $('#engcustom').hidden = false; $('#engUrl').focus(); return; }
  await switchEngine({ mode: eng, url: '', key: '' });
}));
$('#engGo').addEventListener('click', async () => {
  await switchEngine({ mode: 'custom', url: $('#engUrl').value.trim(),
                       key: $('#engKey').value.trim() });
});

/* Changing engine throws the clip away: the frames live inside whichever engine
 * decoded them, and re-uploading silently would be a surprise on a metered
 * connection. Say so rather than pretending the switch is free. */
async function switchEngine(pref) {
  const stat = $('#engstat');
  stat.textContent = 'checking…'; stat.classList.remove('err');
  try {
    const r = await chooseEngine(pref);
    if (r.warn) { stat.classList.add('err'); stat.textContent = r.warn; }
    else stat.textContent = '';
    savePref(pref);
    S.enginePref = pref;
    const had = S.kind !== 'none';
    S.engine = r.engine;
    S.meta = await r.engine.meta();
    await afterEngine();
    if (had) {
      resetClip();
      toast('switched to ' + r.engine.label + ' — drop the file again');
    }
    openEnginePop(false);
  } catch (err) {
    stat.classList.add('err'); stat.textContent = why(err);
  }
}

function resetClip() {
  S.kind = 'none'; S.job = null; S.bitmap = null; S.tracked = false;
  S.subjects = []; S.nextId = 1; S.scope = 'whole'; S.promptFrame = 0;
  S.previewMasks = null; S.curPath = null;
  dropCache();
  $('#upstat').hidden = true; $('#tinfo').hidden = true; $('#rinfo').hidden = true;
  $('#dl').hidden = true; $('#outvid').hidden = true; $('#pvinfo').hidden = true;
  $('#pwrap').hidden = true; $('#vwrap').hidden = true;
  showSteps('none');
}

/* Metadata that depends on which engine is live: palettes, kernels, the noise
 * tile, the tracker resolutions, and whether the models are actually there. */
async function afterEngine() {
  paintEngine();
  buildTrackSizes();
  const sel = $('#sAlgo');
  sel.textContent = '';
  (S.meta.kernels.length ? S.meta.kernels
    : Object.entries(Dither.KERNELS).map(([id, v]) => ({ id, name: v.name })))
    .forEach((k) => {
      const o = document.createElement('option');
      o.value = k.id; o.textContent = k.name; sel.append(o);
    });
  sel.value = S.P.algo;
  renderPalettes();
  try {
    BLUE = await E().blueNoise(64, S.P.seed);
  } catch (e) {
    BLUE = new Float32Array(4096).map((_, i) => Dither.hash01(i >> 6, i & 63, 5, S.P.seed));
  }
  Dither.setBlueNoise(BLUE);
  DOTS.key = null;
  setMode(S.P.mode);
  buildTargets();
  await checkModels();
}

/* The ONNX weights are ~130 MB and deliberately not in the repo. Say that
 * plainly where it matters — the Subjects step — rather than failing on Track. */
async function checkModels() {
  const box = $('#nomodels');
  S.modelsMissing = null;
  if (E().id !== 'browser') { box.hidden = true; $('#bTrack').disabled = false; return; }
  try {
    await E().init();
    box.hidden = true; $('#bTrack').disabled = false;
  } catch (err) {
    S.modelsMissing = why(err);
    box.hidden = false;
    box.textContent = 'Subject tracking needs the EdgeTAM model files, and they '
      + 'are not here. ' + S.modelsMissing + '  Stills and whole-frame clips work '
      + 'without them.';
    $('#bTrack').disabled = true;
    $('#bPrev').disabled = true;
  }
}

/* ------------------------------------------------------------------ boot */
(async function boot() {
  S.enginePref = loadPref();
  let picked;
  try {
    picked = await chooseEngine(S.enginePref);
  } catch (e) {
    picked = { engine: new BrowserEngine(), pref: S.enginePref, tried: [] };
  }
  S.engine = picked.engine;
  S.engineTried = picked.tried;
  if (picked.warn) toast(picked.warn, true);
  try {
    S.meta = await S.engine.meta();
  } catch (e) {
    // last resort: the dither engine ships its own palette/mode tables, so the
    // still and whole-frame flows survive an engine that cannot introspect
    S.meta = { palettes: Dither.PALETTES, modes: Dither.MODES, stable: Dither.STABLE,
               kernels: [], subject_colors: ['#b0413e'], device: '?',
               track_sizes: [{ size: 768, id: 'balanced', label: 'balanced', fps: 0 }],
               default_track_size: 768 };
  }
  await afterEngine();
  showSteps('none');
  window.DV = S;
  window.DV_maskURL = subjectMaskDataURL;
  window.DV_draw = draw;
  window.DV_engine = () => ({ id: S.engine.id, label: S.engine.label,
                              baseUrl: S.engine.baseUrl || '',
                              supports: S.engine.supports,
                              modelsMissing: S.modelsMissing,
                              tried: S.engineTried });
  window.DV_switchEngine = switchEngine;
  window.DV_composeAt = composeAt;
  window.DV_ready = true;
})();
