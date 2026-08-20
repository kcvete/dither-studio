/* ---------------------------------------------------------------------------
   DITHER STUDIO — one flow for three jobs.

     still            drop an image, dither it, download a PNG (never leaves the tab)
     still + subject  point at something in the photograph and EdgeTAM cuts it
                      out on the spot — one frame, no propagation, ~0.15 s, so
                      the mask follows every click live
     clip             drop a video, every frame gets dithered, export an MP4
     clip + subject   point at something, EdgeTAM tracks it through the clip, and
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
  // still. `stillMasks` is id -> a soft mask image at S.W x S.H; it is to a
  // photograph what the per-frame mask files are to a clip.
  bitmap: null, natW: 0, natH: 0, fileName: '',
  stillMasks: new Map(), stillURL: null, stillFile: null, pngAlpha: false,
  // source file + trim (video only) — kept so a trim can re-open the same file
  srcFile: null, srcDuration: 0, trim: null, recordedS: 0, photo: null,
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
  paletteTouched: false,             // has anyone chosen one yet?
  dotsTuned: false,                  // has the dot count been set for this still?
  bg: '#c9d4c5',
  target: 'bg',                      // which palette the editor is editing: 'bg' | subject id
  meta: null,
  compare: false, split: 0.5,
  // engine
  engine: null, enginePref: null, modelsMissing: null,
  exportURL: null, frameURL: null,
  // export
  format: '', gifFps: 15,
  // sequence: captured segments and static shapes, in order. Survives loading
  // another clip on purpose — a morph from one clip to another needs both.
  library: [], seqDoc: null,
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

/* Which of the four things that can occupy the stage is showing. Every place
 * that used to poke #pwrap/#vwrap by hand goes through this, because a still
 * with subjects has the same two views a clip does. */
function showStage(which) {
  $('#pwrap').hidden = which !== 'prompt';
  $('#vwrap').hidden = which !== 'result';
  $('#camwrap').hidden = which !== 'camera';
  $('#seqwrap').hidden = which !== 'sequence';
  $('#empty').hidden = which !== 'empty';
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
$('#sSec').addEventListener('input', (e) => {
  $('#vSec').textContent = e.target.value + ' s';
  paintTrim();
});

function take(f) {
  const isVid = /^video\//.test(f.type) || /\.(mp4|mov|m4v|webm)$/i.test(f.name);
  $('#vidopts').hidden = !isVid;
  $('#trimui').hidden = true;
  if (!isVid) { S.srcFile = null; return loadStill(f); }
  // The clip loads whole, immediately, and the trim bar appears next to it:
  // making every drop wait for a second click would be worse than the one
  // extra decode a trim costs.
  S.srcFile = f;
  S.trim = null;
  const done = uploadClip(f);
  buildStrip(f);
  return done;
}

/* ======================================================= trim: filmstrip
 * Twelve thumbnails, two handles, and a "use this range" that re-opens the
 * same file over the chosen seconds. The browser engine decodes only that
 * range; the server engine hands ffmpeg -ss/-t. Nothing else in the app
 * knows a trim happened — the clip that comes back is just shorter.
 */
const STRIP_N = 12;
let stripSeq = 0;

async function buildStrip(file) {
  const seq = ++stripSeq;
  const cv = $('#stripcv'), g = cv.getContext('2d');
  g.fillStyle = '#0a1310'; g.fillRect(0, 0, cv.width, cv.height);
  $('#trimui').hidden = false;
  $('#trimnote').textContent = 'reading the clip…';
  const url = URL.createObjectURL(file);
  const v = document.createElement('video');
  v.preload = 'auto'; v.muted = true; v.playsInline = true; v.src = url;
  try {
    await new Promise((ok, no) => {
      v.onloadedmetadata = ok;
      v.onerror = () => no(new Error('cannot read that clip'));
      setTimeout(() => no(new Error('timed out reading the header')), 20000);
    });
    let dur = v.duration;
    if (!isFinite(dur) || dur <= 0) {
      // MediaRecorder WebM has no duration in its header until it has been
      // played through; seeking past the end is the usual way to find it out
      dur = await probeDuration(v);
    }
    S.srcDuration = dur;
    S.trim = { start: 0, end: dur };
    paintTrim();
    const tw = Math.floor(cv.width / STRIP_N), th = cv.height;
    for (let i = 0; i < STRIP_N; i++) {
      if (seq !== stripSeq) return;
      const t = (i + 0.5) / STRIP_N * dur;
      try { await seekTo(v, Math.min(t, Math.max(0, dur - 0.05))); } catch (e) { break; }
      const vw = v.videoWidth || 16, vh = v.videoHeight || 9;
      const k = Math.max(tw / vw, th / vh);
      g.drawImage(v, (tw - vw * k) / 2 + i * tw, (th - vh * k) / 2, vw * k, vh * k);
    }
    $('#trimnote').textContent = '';
  } catch (err) {
    $('#trimnote').textContent = 'no filmstrip for this file (' + err.message
      + ') — the trim handles still work';
  } finally {
    v.src = ''; v.load?.();
    URL.revokeObjectURL(url);
  }
}

function seekTo(v, t) {
  return new Promise((ok, no) => {
    const done = () => { v.removeEventListener('seeked', done); clearTimeout(bail); ok(); };
    const bail = setTimeout(() => { v.removeEventListener('seeked', done);
      no(new Error('seek stalled')); }, 8000);
    v.addEventListener('seeked', done);
    v.currentTime = Math.max(0, t);
  });
}

/** Duration of a stream whose header does not carry one (MediaRecorder WebM). */
function probeDuration(v) {
  return new Promise((ok) => {
    const done = () => {
      v.removeEventListener('durationchange', done);
      ok(isFinite(v.duration) && v.duration > 0 ? v.duration : 0);
    };
    v.addEventListener('durationchange', done);
    setTimeout(done, 3000);
    try { v.currentTime = 1e6; } catch (e) { done(); }
  });
}

function paintTrim() {
  if (!S.trim) return;
  const { start, end } = S.trim, dur = S.srcDuration || 1;
  const a = clamp(start / dur, 0, 1), b = clamp(end / dur, 0, 1);
  $('#trimdim').style.left = '0'; $('#trimdim').style.width = (a * 100) + '%';
  $('#trimdim2').style.right = '0'; $('#trimdim2').style.width = ((1 - b) * 100) + '%';
  $('#hIn').style.left = (a * 100) + '%';
  $('#hOut').style.left = (b * 100) + '%';
  const cap = Math.min(+$('#sSec').value, end - start);
  $('#vTrim').textContent = `${start.toFixed(1)} – ${end.toFixed(1)} s · `
    + `${(end - start).toFixed(1)} s`
    + (cap < end - start ? ` (first ${cap.toFixed(1)} s used)` : '');
}

function dragHandle(el, which) {
  el.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    el.setPointerCapture(e.pointerId);
    const strip = $('#strip');
    const move = (ev) => {
      const r = strip.getBoundingClientRect();
      const t = clamp((ev.clientX - r.left) / r.width, 0, 1) * (S.srcDuration || 1);
      if (which === 'in') S.trim.start = Math.min(t, S.trim.end - 0.1);
      else S.trim.end = Math.max(t, S.trim.start + 0.1);
      paintTrim();
    };
    const up = () => {
      el.removeEventListener('pointermove', move);
      el.removeEventListener('pointerup', up);
    };
    el.addEventListener('pointermove', move);
    el.addEventListener('pointerup', up);
    move(e);
  });
}
dragHandle($('#hIn'), 'in');
dragHandle($('#hOut'), 'out');

$('#bTrim').addEventListener('click', () => {
  if (!S.srcFile || !S.trim) return;
  uploadClip(S.srcFile, { start: S.trim.start, end: S.trim.end });
});
$('#bTrimAll').addEventListener('click', () => {
  if (!S.srcFile) return;
  S.trim = { start: 0, end: S.srcDuration || 0 };
  paintTrim();
  uploadClip(S.srcFile);
});

/* ============================================================== camera ===
 * getUserMedia -> live preview -> MediaRecorder -> a WebM blob that goes
 * through exactly the same path a dropped file does. Which means it works on
 * both engines: the browser one decodes the blob, the server one uploads it
 * (server.py already accepts .webm, and ffmpeg reads what Chrome writes).
 */
const CAM = { stream: null, rec: null, chunks: [], t0: 0, timer: 0 };
const CAM_MAX_S = 30;

async function camOpen() {
  if (CAM.stream) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    toast('this browser has no camera API', true); return;
  }
  try {
    CAM.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
    });
  } catch (err) {
    toast('no camera: ' + (err.message || err.name), true);
    return;
  }
  const v = $('#camvid');
  v.srcObject = CAM.stream;
  await v.play().catch(() => {});
  const t = CAM.stream.getVideoTracks()[0].getSettings();
  $('#camnote').textContent = `${t.width || '?'}×${t.height || '?'}`
    + (t.frameRate ? ` · ${Math.round(t.frameRate)} fps` : '')
    + ` · up to ${CAM_MAX_S} s · nothing leaves the tab until you export`;
  $('#camui').hidden = false;
  $('#camwrap').hidden = false;
  $('#pwrap').hidden = true; $('#vwrap').hidden = true; $('#empty').hidden = true;
  $('#bCam').textContent = 'camera on';
}

function camClose() {
  camStop(true);
  if (CAM.stream) CAM.stream.getTracks().forEach((t) => t.stop());
  CAM.stream = null;
  $('#camvid').srcObject = null;
  $('#camui').hidden = true;
  $('#bCam').textContent = 'record from camera';
  restoreStage();
}

/** Back to whatever the current source was showing before the camera opened. */
function restoreStage() {
  if (S.kind === 'none') return showStage('empty');
  const prompting = S.scope === 'track'
    && (S.kind === 'image' ? !S.stillMasks.size : !S.tracked);
  showStage(prompting ? 'prompt' : 'result');
}

/** A photo: the frame the preview is showing, at the camera's own resolution,
 *  handed to the still flow as a PNG. Same path an uploaded image takes, so it
 *  dithers, re-palettes and exports client-side like any other still. */
async function camSnap() {
  if (!CAM.stream) return;
  const v = $('#camvid');
  const t = CAM.stream.getVideoTracks()[0].getSettings();
  const w = v.videoWidth || t.width || 1280, h = v.videoHeight || t.height || 720;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.getContext('2d').drawImage(v, 0, 0, w, h);
  const blob = await new Promise((ok) => c.toBlob(ok, 'image/png'));
  if (!blob) { toast('the camera frame could not be read', true); return; }
  const name = `photo-${new Date().toISOString().slice(11, 19).replace(/:/g, '')}.png`;
  S.photo = { w, h, bytes: blob.size };
  camClose();
  take(new File([blob], name, { type: 'image/png' }));
  toast(`photo ${w}\u00d7${h} — the camera is off`);
}

function camStart() {
  if (!CAM.stream || CAM.rec) return;
  const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mime = types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
  if (!mime) { toast('this browser cannot record WebM', true); return; }
  CAM.chunks = [];
  CAM.rec = new MediaRecorder(CAM.stream, { mimeType: mime, videoBitsPerSecond: 8e6 });
  CAM.rec.ondataavailable = (e) => { if (e.data && e.data.size) CAM.chunks.push(e.data); };
  CAM.rec.onstop = () => {
    const blob = new Blob(CAM.chunks, { type: 'video/webm' });
    CAM.rec = null;
    const secs = (performance.now() - CAM.t0) / 1000;
    const f = new File([blob], `camera-${new Date().toISOString().slice(11, 19)
      .replace(/:/g, '')}.webm`, { type: 'video/webm' });
    camClose();
    $('#upstat').hidden = false;
    $('#upstat').textContent = `recorded ${secs.toFixed(1)} s · `
      + `${(blob.size / 1e6).toFixed(1)} MB · trim it below`;
    S.recordedS = +secs.toFixed(2);
    take(f);
  };
  CAM.rec.start(200);
  CAM.t0 = performance.now();
  $('#camdot').hidden = false;
  $('#bRec').textContent = 'stop';
  CAM.timer = setInterval(() => {
    const s = (performance.now() - CAM.t0) / 1000;
    $('#camtime').textContent = s.toFixed(1) + ' s';
    if (s >= CAM_MAX_S) camStop();
  }, 100);
}

function camStop(silent) {
  clearInterval(CAM.timer); CAM.timer = 0;
  $('#camdot').hidden = true;
  $('#bRec').textContent = 'record';
  if (!CAM.rec) return;
  if (silent) { CAM.rec.onstop = null; CAM.rec.stop(); CAM.rec = null; return; }
  CAM.rec.stop();
}

$('#bSnap').addEventListener('click', camSnap);
$('#bCam').addEventListener('click', () => (CAM.stream ? camClose() : camOpen()));
$('#bRec').addEventListener('click', () => (CAM.rec ? camStop() : camStart()));
$('#bCamOff').addEventListener('click', camClose);

function showSteps(kind) {
  $('#st2').hidden = kind === 'none';
  paintStep2(kind);
  $('#st3').hidden = $('#st4').hidden = $('#st5').hidden = kind === 'none';
  // the sequence step outlives the clip: a morph from one clip into another
  // needs the segment captured from the first one to still be there
  $('#st6').hidden = kind !== 'video' && !S.library.length;
  buildSubjectPicker();
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
    S.tracked = false; S.subjects = []; S.nextId = 1; S.scope = 'whole';
    S.promptFrame = 0; S.previewMasks = null; S.curPath = null;
    S.paletteTouched = false; S.dotsTuned = false;
    dropStill();
    S.stillFile = f;
    S.stillURL = URL.createObjectURL(f);
    // The still prompts, segments and previews at S.W x S.H — the preview
    // budget — and only the PNG goes back to the file's own resolution. One
    // coordinate space for clicks, masks and the overlay, on both engines.
    const [pw, ph] = previewSize();
    S.W = pw; S.H = ph;
    box.textContent = `${bmp.width} × ${bmp.height} · ${(f.size / 1024).toFixed(0)} KB · stays in this tab`;
    $('#s1sum').textContent = `${bmp.width}×${bmp.height}`;
    showSteps('image');
    showStage('result');
    $('#bPlay').hidden = $('#sFrame').hidden = $('#fcount').hidden = true;
    $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
    $('#outimg').hidden = true; $('#pvinfo').hidden = true; $('#tinfo').hidden = true;
    $('#s5sum').textContent = ''; $('#bExport').textContent = 'Download PNG';
    $('#fmtui').hidden = true; $('#trimui').hidden = true;
    $('#offframe').hidden = true;
    setScope('whole');
    buildTargets(); renderModes(); paintCompose(); paintAlphaUI(); openStep(2);
    await draw();
  } catch (err) {
    box.classList.add('err'); box.textContent = 'could not read that image: ' + err.message;
  }
}

/** Throw away everything that belonged to the previous still. */
function dropStill() {
  S.stillMasks.forEach((m) => m && m.close && m.close());
  S.stillMasks.clear();
  if (S.stillURL) { URL.revokeObjectURL(S.stillURL); S.stillURL = null; }
  S.stillFile = null;
  stillJobKey = null;
  DOTS_CACHE = null;
}

async function uploadClip(f, trim) {
  const box = $('#upstat'); box.hidden = false; box.classList.remove('err');
  box.textContent = (E().id === 'browser' ? 'decoding ' : 'uploading ') + f.name + '…';
  busy(true);
  const t = trim || S.trim || null;
  const cut = t && (t.start > 0.05 || (S.srcDuration && t.end < S.srcDuration - 0.05));
  try {
    const j = await E().open(f, {
      maxSeconds: +$('#sSec').value,
      trimStart: cut ? t.start : 0,
      trimEnd: cut ? t.end : null,
      onProgress: (p) => { if (p.text) box.textContent = p.text; },
    });
    S.trim = t;
    dropStill();
    S.kind = 'video'; S.job = j.job; S.nFrames = j.nFrames; S.W = j.w; S.H = j.h; S.fps = j.fps;
    S.fileName = f.name.replace(/\.[^.]+$/, '');
    S.tracked = false; S.subjects = []; S.nextId = 1; S.cur = 0; S.promptFrame = 0;
    S.scope = 'whole';
    dropCache();
    box.textContent = `${j.nFrames} frames · ${j.w}×${j.h} · ${j.fps} fps`
      + (cut ? ` · trimmed ${t.start.toFixed(1)}–${t.end.toFixed(1)} s` : '')
      + (E().id === 'browser' ? ' · stays in this tab' : '');
    $('#s1sum').textContent = `${j.nFrames}f`;
    $('#sPF').max = j.nFrames - 1; $('#sPF').value = 0; $('#vPF').textContent = '0';
    $('#sFrame').max = j.nFrames - 1;
    $('#bPlay').hidden = $('#sFrame').hidden = $('#fcount').hidden = false;
    $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
    $('#tinfo').hidden = true; $('#s5sum').textContent = '';
    $('#outimg').hidden = true;
    buildFormats();
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
  const still = S.kind === 'image';
  $$('[data-scope]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.scope === v)));
  $('#trackui').hidden = v !== 'track';
  $('#wholenote').hidden = v === 'track';
  paintCompose();
  if (v === 'track') {
    if (!S.subjects.length) addSubject();
    if (still) {
      showStage(S.stillMasks.size ? 'result' : 'prompt');
      showStillPrompt();
    } else {
      showStage(S.tracked ? 'result' : 'prompt');
      if (!S.tracked) showPromptFrame(S.promptFrame);
    }
  } else {
    showStage('result');
    draw();
  }
  buildTargets(); renderModes(); paintAlphaUI();
  paintScopeSummary();
}

function paintScopeSummary() {
  const still = S.kind === 'image';
  if (S.scope !== 'track') {
    $('#s2sum').textContent = still ? 'whole image' : 'whole clip';
    return;
  }
  const n = S.subjects.length;
  $('#s2sum').textContent = still
    ? (S.stillMasks.size ? `${S.stillMasks.size} selected` : `${n} subj`)
    : (S.tracked ? `${n} tracked` : `${n} subj${n > 1 ? 's' : ''}`);
}

/* Compose (flat background vs keep the scene) only means something when part
 * of the picture is left alone: a subject cut out of it, or the dots look,
 * which paints on a background of its own whatever else is going on. */
const composeMatters = () => usingSubjects()
  || (S.kind === 'image' && S.P.mode === 'dots');
const flatBg = () => composeMatters() && S.P.compose === 'cutout';
function paintCompose() {
  $('#composeui').hidden = !composeMatters();
  $('#bgui').hidden = !flatBg();
}

/* Step 2 is "track subjects through a clip" or "select subjects in a photo".
 * Same tools, same prompts, same six-subject ceiling; what changes is that a
 * photograph has no frames to propagate through, so there is no prompt-frame
 * slider, no Track run and no progress bar — the mask is simply there. */
function paintStep2(kind) {
  const still = kind === 'image';
  const whole = $('#scope [data-scope="whole"]');
  const track = $('#scope [data-scope="track"]');
  if (whole) whole.textContent = still ? 'whole image' : 'whole clip';
  if (track) track.textContent = still ? 'select subjects' : 'track subjects';
  $('#st2 .sh span').textContent = still ? 'Subject' : 'Subjects';
  $('#pfui').hidden = still;
  $('#bPrev').hidden = still;
  $('#stillnote').hidden = !still;
  $('#tqlbl').textContent = still ? 'Selection quality' : 'Tracking quality';
  $('#tqnote').textContent = still
    ? 'Your picture keeps its own resolution — this only changes the square the '
      + 'model looks at, and so how fine an outline it can cut.'
    : 'Your clip keeps its own resolution — this only changes the square the '
      + 'tracker looks at, and so how fine an outline it can draw.';
  $('#bTrack').textContent = still ? 'Use this selection' : 'Track';
  $('#wholenote').textContent = still
    ? 'Every pixel of the image gets dithered. Switch to select subjects to '
      + 'dither only what you point at — or to cut it out of its background.'
    : 'Every pixel of every frame gets dithered. Switch to track subjects to '
      + 'dither only what you point at.';
  if (still) { $('#prog').hidden = true; $('#tinfo').hidden = true; }
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
  if (S.kind === 'none' || S.scope !== 'track') return;
  S.playing = false;
  showStage('prompt');
  if (S.kind === 'image') showStillPrompt();
  else showPromptFrame(S.promptFrame);
}
$('#bAdd').addEventListener('click', () => { addSubject(); backToPrompt(); });
$('#bClr').addEventListener('click', () => {
  S.subjects.forEach((s) => { s.points = []; s.box = null; s.paths = []; s.promptFrame = null; });
  S.curPath = null; S.previewMasks = null; $('#pvinfo').hidden = true;
  if (S.kind === 'image') {
    S.stillMasks.forEach((m) => m && m.close && m.close());
    S.stillMasks.clear(); S.tracked = false;
    buildTargets(); renderModes(); paintCompose(); paintAlphaUI();
  }
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
    if (s.promptFrame !== null && S.kind !== 'image') bits.push('@ ' + s.promptFrame);
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
        if (S.kind === 'image') {
          const m = S.stillMasks.get(s.id);
          if (m && m.close) m.close();
          S.stillMasks.delete(s.id);
          S.tracked = S.stillMasks.size > 0;
          paintCompose(); paintAlphaUI(); draw();
        }
        renderSubjects(); drawOverlay(); buildTargets();
      });
      b.append(x);
    }
    wrap.append(b);
  });
  $('#vSubs').textContent = `${S.subjects.length} / ${MAX_SUBJECTS}`;
  paintScopeSummary();
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

/** The still, on the prompt stage. One picture, no seeking, no frame index. */
function showStillPrompt() {
  if (S.kind !== 'image') return;
  S.promptFrame = 0;
  pov.width = S.W; pov.height = S.H;
  if (S.stillURL && pimg.src !== S.stillURL) pimg.src = S.stillURL;
  drawOverlay();
  renderSubjects();
}

function drawOverlay() {
  if (!S.W || S.kind === 'none') return;
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

/* A mask image carries its coverage in the RED channel and is fully opaque —
 * that is what makes it readable by `bitmapAlpha` and by the exporters. Tinting
 * it with `source-in` therefore keeps every pixel, which washed the whole frame
 * instead of showing the subject. Move the coverage into alpha first.
 *
 * Cached against the image object: a lasso drag repaints the overlay on every
 * pointermove, and a megapixel loop per move is not a thing to do twice. */
const TINT = new WeakMap();
function tintedMask(im, col) {
  const rec = TINT.get(im);
  if (rec && rec.col === col && rec.w === S.W && rec.h === S.H) return rec.canvas;
  const t = document.createElement('canvas');
  t.width = S.W; t.height = S.H;
  const g = t.getContext('2d', { willReadFrequently: true });
  g.drawImage(im, 0, 0, S.W, S.H);
  const px = g.getImageData(0, 0, S.W, S.H), d = px.data;
  const [r, gr, b] = Dither.hexRGB(col);
  for (let p = 0, n = S.W * S.H * 4; p < n; p += 4) {
    d[p + 3] = d[p]; d[p] = r; d[p + 1] = gr; d[p + 2] = b;
  }
  g.putImageData(px, 0, 0);
  TINT.set(im, { col, canvas: t, w: S.W, h: S.H });
  return t;
}

function drawPreviewMasks() {
  const still = S.kind === 'image';
  if (!still && !S.previewMasks) return;
  S.subjects.forEach((s) => {
    const im = still ? S.stillMasks.get(s.id) : S.previewMasks[String(s.id)];
    // an <img> that has not loaded yet is `complete === false`; an ImageBitmap
    // has no such property at all, and must not be skipped for lacking one
    if (!im || im.complete === false) return;
    pctx.save();
    pctx.globalAlpha = 0.55;
    pctx.drawImage(tintedMask(im, s.palette[s.palette.length - 1]), 0, 0, S.W, S.H);
    pctx.restore();
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
    afterPromptEdit();
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
  afterPromptEdit();
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
  else if (s.paths.length) { s.paths.pop(); afterPromptEdit(); }
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

$('#bTrack').addEventListener('click', () => {
  if (S.kind === 'image') return useStillSelection();
  return track();
});

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

/* ===================================== a still: segment it, live =========
 * A photograph has one frame, so there is nothing to propagate: the whole of
 * "select a subject" is the conditioning step the clip flow runs on frame 0.
 * It costs an image encode and the SAM heads — about 0.15 s — which is fast
 * enough that there is no button. Every click, box or drawn shape re-runs it
 * and the mask overlay follows.
 *
 * The picture is handed to the engine once (the browser keeps it as a clip of
 * one frame; the server takes it as a one-frame job), and every prompt after
 * that reuses it. */
let stillJobKey = null;
let segBusy = false, segAgain = false;

async function ensureStillJob() {
  const key = [E().id, S.fileName, S.natW, S.natH, S.W, S.H].join('|');
  if (stillJobKey === key) return;
  const c = ctx2d(S.W, S.H, 'seg');
  c.clearRect(0, 0, S.W, S.H);
  c.drawImage(S.bitmap, 0, 0, S.W, S.H);
  const blob = await new Promise((ok) => c.canvas.toBlob(ok, 'image/png'));
  if (!blob) throw new Error('could not read the image back out of the canvas');
  await E().openStill(blob, { w: S.W, h: S.H, maxSide: PREVIEW_MAX,
                              name: (S.fileName || 'still') + '.png' });
  stillJobKey = key;
}

/** Called after anything that changes a prompt. Coalesces: a click that lands
 *  while an inference is in flight schedules exactly one more run. */
function afterPromptEdit() {
  if (S.kind === 'image' && S.scope === 'track') segmentStill();
}

async function segmentStill() {
  if (S.kind !== 'image' || S.scope !== 'track') return;
  if (!(E().segmentImage && E().openStill && E().supports.stillSubjects)) {
    const info = $('#pvinfo'); info.hidden = false; info.classList.add('err');
    info.textContent = 'this engine cannot segment a still — switch to the '
      + 'browser engine, or update the server';
    return;
  }
  const here = S.subjects.filter(hasPrompt);
  if (!here.length) {
    S.stillMasks.forEach((m) => m && m.close && m.close());
    S.stillMasks.clear(); S.tracked = false;
    $('#pvinfo').hidden = true;
    buildTargets(); renderModes(); paintCompose(); paintAlphaUI();
    drawOverlay(); paintScopeSummary();
    return;
  }
  if (segBusy) { segAgain = true; return; }
  segBusy = true;
  const info = $('#pvinfo'); info.hidden = false; info.classList.remove('err');
  if (!info.textContent) info.textContent = 'reading the selection…';
  try {
    await ensureStillJob();
    const r = await E().segmentImage(
      { objects: promptPayload(here), imageSize: S.trackSize },
      (m) => { info.textContent = m; });
    const live = new Set(here.map((x) => x.id));
    S.stillMasks.forEach((m, id) => {
      if (!live.has(id)) { if (m && m.close) m.close(); S.stillMasks.delete(id); }
    });
    for (const o of r.objects) {
      const id = +o.id, old = S.stillMasks.get(id);
      if (old && old.close && old !== o.image) old.close();
      S.stillMasks.set(id, o.image);
    }
    S.tracked = S.stillMasks.size > 0;
    info.classList.remove('err');
    info.textContent = `${r.objects.length} subject${r.objects.length > 1 ? 's' : ''} `
      + `in ${r.elapsedS.toFixed(2)} s (${r.imageSize} px) · `
      + r.objects.map((o) => '#' + o.id + ' ' + o.area + ' px').join(' · ')
      + (r.note ? ' · ' + r.note : '');
    buildTargets(); renderModes(); paintCompose(); paintAlphaUI();
    drawOverlay(); paintScopeSummary();
    DOTS_CACHE = null;
    if (!$('#vwrap').hidden) await draw();
  } catch (err) {
    info.classList.add('err');
    info.textContent = 'selection failed: ' + why(err);
  }
  segBusy = false;
  if (segAgain) { segAgain = false; segmentStill(); }
}

/** "Use this selection": the masks are already there, this is only the move
 *  from the prompt stage to the picture. `backToPrompt` is the way back. */
function useStillSelection() {
  if (!S.stillMasks.size) {
    toast('click the subject first — the outline appears as you do', true);
    return;
  }
  showStage('result');
  buildTargets(); renderModes(); paintCompose(); paintAlphaUI();
  openStep(3);
  draw();
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
    DOTS_CACHE = null;
    buildSubjectPicker();
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
  const ids = (S.kind === 'video' && usingSubjects()) ? S.subjects.map((s) => s.id) : [];
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
/* "there are masks, and they belong to subjects" — true for a tracked clip and
 * for a still whose subjects have been segmented. Everything downstream (the
 * compose split, the per-subject palettes, the dots renderer, the exports)
 * asks this and not the source kind. */
const usingSubjects = () => S.scope === 'track' && S.subjects.length > 0
  && (S.kind === 'video' ? S.tracked
    : S.kind === 'image' ? S.stillMasks.size > 0 : false);

/* A whole-image dither has no mask; the dots renderer still wants one, because
 * density is measured inside a mask. This is the "all of it" mask. */
let FULLMASK = { w: 0, h: 0, m: null };
function fullMask(w, h) {
  if (FULLMASK.w !== w || FULLMASK.h !== h) {
    FULLMASK = { w, h, m: new Float32Array(w * h).fill(1) };
  }
  return FULLMASK.m;
}

/** The still's per-subject coverage, resampled to whatever size is being
 *  rendered. Subjects with no mask yet contribute an empty one so the array
 *  stays index-aligned with the palette list. */
function stillMasksAt(w, h, slot) {
  const pre = slot || 'sm';
  return S.subjects.map((s, k) => {
    const im = S.stillMasks.get(s.id);
    return im ? bitmapAlpha(im, w, h, pre + k) : new Float32Array(w * h);
  });
}

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

/* Which cells are lit, per subject. Split out of renderDots so the .dots
 * export and the picture cannot disagree: both go through this. */
function dotsOn(srcData, W, H, masks, P, tile) {
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

  const ons = [];
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
    ons.push(on);
  }
  return { F, on: ons };
}

/** Lit cells -> the integer dot centres, in cell-scan order. This is what a
 *  .dots.gz stores, and what the player draws. */
function dotXY(F, on) {
  const { N, cx, cy } = F;
  let n = 0;
  for (let q = 0; q < N; q++) if (on[q]) n++;
  const xy = new Uint16Array(n * 2);
  let i = 0;
  for (let q = 0; q < N; q++) {
    if (!on[q]) continue;
    xy[i++] = Math.round(cx[q]); xy[i++] = Math.round(cy[q]);
  }
  return xy;
}

function renderDots(srcData, W, H, masks, P, palettes, bg, tile) {
  const { F, on: ons } = dotsOn(srcData, W, H, masks, P, tile);
  const { N, thr, cx, cy } = F;
  const out = new Uint8ClampedArray(W * H * 4), bgc = Dither.hexRGB(bg);
  // P.alpha: leave the flat background transparent and let only the dots be
  // opaque. `overlay` keeps the scene, so there is nothing to key out.
  const bga = (P.alpha && P.compose !== 'overlay') ? 0 : 255;
  if (P.compose === 'overlay') {
    for (let p = 0, n = W * H * 4; p < n; p += 4) {
      const lum = (0.2126 * srcData[p] + 0.7152 * srcData[p + 1] + 0.0722 * srcData[p + 2]) / 255;
      const g = (lum * 0.55 + 0.22) * 1.15;
      out[p] = g * bgc[0]; out[p + 1] = g * bgc[1]; out[p + 2] = g * bgc[2]; out[p + 3] = 255;
    }
  } else {
    for (let p = 0, n = W * H * 4; p < n; p += 4) {
      out[p] = bgc[0]; out[p + 1] = bgc[1]; out[p + 2] = bgc[2]; out[p + 3] = bga;
    }
  }

  const dp = P.dotpx | 0, half = dp >> 1;
  let lit = 0;
  for (let k = 0; k < ons.length; k++) {
    const on = ons[k];
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
  // a still with no subject selected is dotted whole-image, the way the
  // original demo did it: one mask covering everything, density from luminance
  const dotMasks = masks.length ? masks
    : (S.kind === 'image' ? [fullMask(W, H)] : null);
  if (S.P.mode === 'dots' && dotMasks) {
    const r = renderDots(srcData, W, H, dotMasks, S.P, pal, S.bg, BLUE);
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
    c.clearRect(0, 0, w, h);
    c.drawImage(S.bitmap, 0, 0, w, h);
    const src = c.getImageData(0, 0, w, h).data;
    if (seq !== drawSeq) return;
    const masks = usingSubjects() ? stillMasksAt(w, h) : [];
    const r = paint($('#vcv'), src, w, h, masks, { original: S.bitmap });
    $('#fps').textContent = `${w}×${h} · ${r.ms.toFixed(0)} ms`
      + (S.P.mode === 'dots' ? ` · ${r.lit} dots` : '');
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
/* Dots need a mask to measure density inside. A tracked clip has one per
 * subject; a still is its own mask when nothing is selected — which is where
 * this look came from in the first place. A whole-frame CLIP still needs a
 * tracked subject: there is nothing to hold the dots still against otherwise. */
const dotsAvailable = () => S.kind === 'image' || usingSubjects();

function setMode(id) {
  S.P.mode = id;
  if (id === 'dots' && S.kind === 'image' && !usingSubjects()) tuneWholeImageDots();
  const dotsUI = $('#dotsexp');
  if (dotsUI) dotsUI.hidden = !(id === 'dots' && dotsAvailable());
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
  paintCompose(); paintAlphaUI();
  renderModes();
}

/* Two defaults that are right for a subject and wrong for a whole picture, and
 * are therefore adjusted once, the first time dots is chosen on a still that
 * has nothing selected. Both sliders stay where they land and anything the user
 * has already touched is left alone.
 *
 *   palette   black-and-white is right for a dither, which covers every pixel,
 *             and wrong for dots, which paint ON a background: white dots on
 *             the default sage are invisible. A whole picture takes the same
 *             pairing a subject gets — the one the look was designed around.
 *   count     8,000 is a subject-sized number. A whole 720p frame at cell 4 is
 *             57,600 cells, and 8,000 of them lit is a scatter with no picture
 *             in it — measured: the tree and the wall only come out of the
 *             noise somewhere north of half the cells. `fill` never bites at
 *             this scale (its 0.7 x 57,600 is far above n), so `n` is the knob
 *             that matters, and it is aimed at 55 % coverage.
 */
function tuneWholeImageDots() {
  if (S.meta && !S.paletteTouched) {
    S.palette = [S.bg, subjectColor(0)];
    renderSwatches();
  }
  if (S.dotsTuned) return;
  S.dotsTuned = true;
  const [w, h] = previewSize();
  const cells = Math.max(1, ((w / S.P.cell) | 0) * ((h / S.P.cell) | 0));
  const el = $('#sN');
  const n = Math.min(+el.max, Math.max(+el.min,
    Math.round(cells * 0.55 / 500) * 500));
  S.P.n = n; el.value = String(n); $('#vN').textContent = String(n);
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
    const ok = m.id !== 'dots' || dotsAvailable();
    if (!ok) { b.classList.add('off'); b.title = 'track a subject first'; }
    if (S.kind === 'video' && S.meta.stable[m.id] === false) {
      const w = document.createElement('i'); w.className = 'fl'; w.textContent = '≈';
      w.title = 'flickers frame to frame'; b.append(w);
    }
    b.addEventListener('click', () => {
      if (!ok) { toast('on a clip the dots look needs a tracked subject', true); return; }
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
$('#sN').addEventListener('input', () => { S.dotsTuned = true; });
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
  paintCompose(); paintAlphaUI(); buildTargets();
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
  S.paletteTouched = true;
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
async function composeAt(i, opts) {
  const rec = await frameAt(i);
  const c = ctx2d(S.W, S.H, 'exp');
  c.clearRect(0, 0, S.W, S.H);
  c.drawImage(rec.frame, 0, 0);
  const src = c.getImageData(0, 0, S.W, S.H).data;
  const masks = usingSubjects()
    ? rec.masks.map((m, k) => bitmapAlpha(m, S.W, S.H, 'x' + k)) : [];
  const pal = palettesForRender();
  // `alpha` is the transparent exports: same pixels, background keyed out
  const P = (opts && opts.alpha) ? Object.assign({}, S.P, { alpha: true }) : S.P;
  const out = (S.P.mode === 'dots' && masks.length)
    ? renderDots(src, S.W, S.H, masks, P, pal, S.bg, BLUE).out
    : Dither.composeFrame(src, S.W, S.H, masks, P, pal, S.bg);
  return new ImageData(out, S.W, S.H);
}

/* Whether "transparent background" is a question worth asking: only where the
 * picture has a flat background to remove. A whole-image dither covers every
 * pixel, so there is nothing to key out and the checkbox stays away. */
const alphaMatters = () => S.kind === 'image' && S.P.compose !== 'overlay'
  && (S.P.mode === 'dots' || usingSubjects());
function paintAlphaUI() {
  const box = $('#pngalpha');
  if (!box) return;
  box.hidden = !alphaMatters();
  if (box.hidden) return;
  $('#alphanote').textContent = usingSubjects()
    ? 'The subject stays; everything that would have been flat background '
      + 'becomes transparent.'
    : 'Only the dots stay opaque; the flat background becomes transparent.';
}
$('#cAlpha').addEventListener('change', (e) => { S.pngAlpha = e.target.checked; });

/** The finished still, at the file's own resolution: the same renderer the
 *  preview runs, with the masks resampled up and — for a cutout — the flat
 *  background optionally left transparent. */
function composeStill(w, h, opts) {
  const c = ctx2d(w, h, 'exp');
  c.clearRect(0, 0, w, h);
  c.drawImage(S.bitmap, 0, 0, w, h);
  const src = c.getImageData(0, 0, w, h).data;
  const masks = usingSubjects() ? stillMasksAt(w, h, 'xm') : [];
  const P = (opts && opts.alpha) ? Object.assign({}, S.P, { alpha: true }) : S.P;
  const pal = palettesForRender();
  if (S.P.mode === 'dots') {
    const r = renderDots(src, w, h, masks.length ? masks : [fullMask(w, h)],
                         P, pal, S.bg, BLUE);
    return { out: r.out, lit: r.lit, masks };
  }
  return { out: Dither.composeFrame(src, w, h, masks, P, pal, S.bg), lit: 0, masks };
}

async function exportPNG() {
  const info = $('#rinfo'); info.hidden = true; info.textContent = '';
  busy(true);
  await sleep(16);
  try {
    // re-render at the source's native resolution, not the preview's
    const alpha = alphaMatters() && S.pngAlpha;
    const { out, lit } = composeStill(S.natW, S.natH, { alpha });
    const cv = document.createElement('canvas');
    cv.width = S.natW; cv.height = S.natH;
    cv.getContext('2d').putImageData(new ImageData(out, S.natW, S.natH), 0, 0);
    const blob = await new Promise((r) => cv.toBlob(r, 'image/png'));
    const dl = $('#dl');
    if (dl.dataset.url) URL.revokeObjectURL(dl.dataset.url);
    const url = URL.createObjectURL(blob);
    dl.dataset.url = url; dl.href = url;
    dl.download = `${S.fileName || 'dither'}-${S.P.mode}`
      + (alpha ? '-alpha' : '') + '.png';
    dl.hidden = false;
    const box = $('#rinfo'); box.hidden = false; box.classList.remove('err');
    box.textContent = `${S.natW}×${S.natH} PNG · ${(blob.size / 1024).toFixed(0)} KB`
      + (S.P.mode === 'dots' ? ` · ${lit} dots` : '')
      + (usingSubjects() ? ` · ${S.stillMasks.size} subject`
        + `${S.stillMasks.size > 1 ? 's' : ''}, ${S.P.compose}` : '')
      + (alpha ? ' · transparent background' : '');
    $('#s5sum').textContent = 'ready';
  } catch (err) {
    const box = $('#rinfo'); box.hidden = false; box.classList.add('err');
    box.textContent = 'export failed: ' + err.message;
  }
  busy(false);
}

/* ---- the format select ------------------------------------------------
 * The list comes from the live engine, unavailable entries included: a
 * greyed-out "MP4 · needs the local server" says more than a menu that
 * quietly has three items in the tab and five on the server. */
function engineFormats() {
  return (E() && E().supports && E().supports.formats) || [];
}
function currentFormat() {
  const list = engineFormats();
  return list.find((f) => f.id === S.format)
    || list.find((f) => f.available) || { id: 'webm', ext: 'webm', alpha: false };
}
function buildFormats() {
  const sel = $('#sFmt'), list = engineFormats();
  $('#fmtui').hidden = S.kind !== 'video' || !list.length;
  sel.textContent = '';
  list.forEach((f) => {
    const o = document.createElement('option');
    o.value = f.id;
    o.textContent = f.label + (f.available ? '' : ' — unavailable here');
    o.disabled = !f.available;
    sel.append(o);
  });
  if (!list.some((f) => f.id === S.format && f.available)) {
    const first = list.find((f) => f.available);
    S.format = first ? first.id : '';
  }
  sel.value = S.format;
  paintFormat();
}
function paintFormat() {
  const f = currentFormat();
  $('#vFmt').textContent = f.ext ? f.ext.toUpperCase() : '';
  $('#fmtnote').textContent = f.note || '';
  $('#giffps').hidden = f.id !== 'gif';
  $('#bExport').textContent = S.kind === 'image' ? 'Download PNG'
    : 'Render ' + (f.ext || '').toUpperCase();
}
$('#sFmt').addEventListener('change', (e) => { S.format = e.target.value; paintFormat(); });
$$('[data-gfps]').forEach((b) => b.addEventListener('click', () => {
  S.gifFps = +b.dataset.gfps;
  $$('[data-gfps]').forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
}));

async function exportClip() {
  const btn = $('#bExport'); btn.disabled = true;
  $('#dl').hidden = true; $('#outvid').hidden = true; $('#outimg').hidden = true;
  const info = $('#rinfo'); info.hidden = true; info.textContent = '';
  const prog = $('#rprog'); prog.hidden = false;
  const bar = $('.bar i', prog), lab = $('span', prog);
  bar.style.width = '0%'; lab.textContent = 'starting…';
  stop();
  try {
    const fmt = currentFormat();
    const params = Object.assign({}, S.P, {
      bg: S.bg, palette: S.palette, fps: S.fps,
      format: fmt.id, gif_fps: S.gifFps,
      subjects: usingSubjects()
        ? S.subjects.map((s) => ({ id: s.id, palette: s.palette })) : [],
    });
    const r = await E().exportClip(params, (p) => {
      bar.style.width = (p.total ? (p.done / p.total) * 100 : 0).toFixed(1) + '%';
      lab.textContent = p.text;
    }, (i) => composeAt(i, { alpha: fmt.alpha }));
    prog.hidden = true;
    if (S.exportURL) { URL.revokeObjectURL(S.exportURL); S.exportURL = null; }
    if (r.url.startsWith('blob:')) S.exportURL = r.url;
    const dl = $('#dl');
    dl.href = r.url;
    dl.download = `${S.fileName || 'dither'}-${S.P.mode}.${r.ext}`;
    dl.hidden = false;
    if (r.image) { const im = $('#outimg'); im.src = r.url; im.hidden = false; }
    else if (r.playable) { const v = $('#outvid'); v.src = r.url; v.hidden = false; }
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

/* ================================================ dot data + sequences ===
 * Three things that are really one thing:
 *
 *   .dots.gz    the dots as positions, not pixels (web/player/dither-player.js
 *               carries the format spec). Both engines emit it: the browser
 *               computes it here from the same `dotsOn` the preview paints
 *               with, the server renders it in render.render_dots and we decode
 *               what comes back — same bytes either way.
 *   player      the same file, played back on a canvas. It is a replay, not a
 *               re-dither: identical integer positions, identical squares.
 *   sequence    segments of clips and static shapes, morphed into each other.
 *               The tween lives in the player module and runs HERE, in JS; a
 *               video of it is made by shipping the finished dot positions to
 *               the server and letting it feed ffmpeg. That way there is one
 *               implementation of the morph rather than two that drift.
 */
let DP = null;                      // the player module, imported on first use
let PLAYER = null;                  // its Player over #seqcv
let DOTS_CACHE = null;              // { key, doc, bytes } for the current clip

async function playerLib() {
  if (DP) return DP;
  await import('./player/dither-player.js');
  DP = globalThis.DitherPlayer;
  return DP;
}

const dotsReady = () => S.kind === 'video' && usingSubjects();

function dotsParams() {
  return {
    cell: S.P.cell, dotpx: S.P.dotpx, n: S.P.n, fill: S.P.fill,
    stray: S.P.stray, band: S.P.band, gamma: S.P.gamma, invert: S.P.invert,
    seed: S.P.seed, bg: S.bg, fps: S.fps,
    subjects: S.subjects.map((x) => ({ id: x.id, palette: x.palette })),
  };
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

/** A still as a one-frame dots document. The player shows it as a static
 *  frame; it is the same file a clip writes, with n_frames = 1. Computed at
 *  the picture's own resolution so the positions match the exported PNG. */
async function dotsDocStill() {
  const P = await playerLib();
  const w = S.natW, h = S.natH;
  const c = ctx2d(w, h, 'exp');
  c.clearRect(0, 0, w, h);
  c.drawImage(S.bitmap, 0, 0, w, h);
  const src = c.getImageData(0, 0, w, h).data;
  const use = usingSubjects();
  const masks = use ? stillMasksAt(w, h, 'xm') : [fullMask(w, h)];
  const r = dotsOn(src, w, h, masks, S.P, BLUE);
  const cols = use ? S.subjects.map((x) => x.palette[x.palette.length - 1])
    : [S.palette[S.palette.length - 1]];
  const doc = { w, h, fps: 1, dotpx: S.P.dotpx,
                palette: [S.bg].concat(cols), bgIndex: 0, bg: S.bg,
                subjects: cols.map((col) => ({ color: col })),
                frames: [r.on.map((o) => dotXY(r.F, o))] };
  return { key: 'still', doc, bytes: await P.pack(doc) };
}

/** The current clip as a dots document, cached against the look it was made
 *  with — the sequence step asks for this repeatedly. A still short-circuits:
 *  one frame, nothing to cache against. */
async function dotsDoc(onProgress) {
  if (S.kind === 'image') return dotsDocStill();
  const P = await playerLib();
  const params = dotsParams();
  const key = JSON.stringify([E().id, S.job, S.nFrames, params]);
  if (DOTS_CACHE && DOTS_CACHE.key === key) return DOTS_CACHE;
  let bytes, doc;
  if (E().exportDots) {
    const r = await E().exportDots(params, onProgress);
    bytes = r.bytes;
    doc = await P.unpack(bytes);
  } else {
    const frames = [];
    for (let i = 0; i < S.nFrames; i++) {
      const rec = await frameAt(i);
      const c = ctx2d(S.W, S.H, 'exp');
      c.clearRect(0, 0, S.W, S.H);
      c.drawImage(rec.frame, 0, 0);
      const src = c.getImageData(0, 0, S.W, S.H).data;
      const masks = rec.masks.map((m, k) => bitmapAlpha(m, S.W, S.H, 'x' + k));
      const r = dotsOn(src, S.W, S.H, masks, S.P, BLUE);
      frames.push(r.on.map((o) => dotXY(r.F, o)));
      if (onProgress) onProgress({ done: i + 1, total: S.nFrames,
                                   text: `${i + 1}/${S.nFrames}` });
      if (i % 8 === 0) await sleep(0);
    }
    const cols = S.subjects.map((x) => x.palette[x.palette.length - 1]);
    doc = { w: S.W, h: S.H, fps: S.fps, dotpx: S.P.dotpx,
            palette: [S.bg].concat(cols), bgIndex: 0, bg: S.bg,
            subjects: cols.map((c) => ({ color: c })), frames };
    bytes = await P.pack(doc);
  }
  DOTS_CACHE = { key, doc, bytes };
  return DOTS_CACHE;
}

async function exportDots(asJSON) {
  const btn = asJSON ? $('#bDotsJson') : $('#bDots');
  btn.disabled = true;
  const info = $('#rinfo'); info.hidden = false; info.classList.remove('err');
  info.textContent = 'reading dot positions…';
  const t0 = performance.now();
  try {
    const P = await playerLib();
    const { doc, bytes } = await dotsDoc((pr) => {
      info.textContent = `dot positions ${pr.text}`;
    });
    const counts = doc.frames.map((f) => f.reduce((a, x) => a + (x.length >> 1), 0));
    const mean = counts.reduce((a, b) => a + b, 0) / Math.max(1, counts.length);
    const one = doc.frames.length === 1;
    const name = `${S.fileName || 'dither'}.dots`;
    let size;
    if (asJSON) {
      const j = new Blob([JSON.stringify(P.toJSON(doc))], { type: 'application/json' });
      size = j.size;
      download(j, name + '.json');
    } else {
      size = bytes.length;
      download(new Blob([bytes], { type: 'application/octet-stream' }), name + '.gz');
    }
    info.textContent = (one ? `1 frame · ${mean.toFixed(0)} dots `
      : `${doc.frames.length} frames · ${mean.toFixed(0)} dots/frame `)
      + `· ${(size / 1024).toFixed(0)} KB ${asJSON ? 'JSON' : '.dots.gz'} `
      + `· ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  } catch (err) {
    info.classList.add('err');
    info.textContent = 'dot export failed: ' + why(err);
  }
  btn.disabled = false;
}
$('#bDots').addEventListener('click', () => exportDots(false));
$('#bDotsJson').addEventListener('click', () => exportDots(true));

/* ---------------------------------------------------------- static shapes
 * A shape becomes dots through the same pipeline a clip does: draw it dark on
 * light, hand it to `dotsOn` with a full-frame mask, keep the positions. So a
 * ring is dithered, not plotted. */
function drawRing(g, W, H) {
  g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);
  const S0 = Math.min(W, H);
  g.save(); g.translate(W / 2, H / 2);
  const r = S0 * 0.32;
  const grad = g.createRadialGradient(0, 0, r * 0.62, 0, 0, r * 1.06);
  grad.addColorStop(0, '#fff'); grad.addColorStop(0.35, '#111');
  grad.addColorStop(0.7, '#111'); grad.addColorStop(1, '#fff');
  g.fillStyle = grad;
  g.beginPath(); g.arc(0, 0, r * 1.06, 0, Math.PI * 2); g.fill();
  g.restore();
}

function drawCoral(g, W, H) {
  g.fillStyle = '#fff'; g.fillRect(0, 0, W, H);
  const S0 = Math.min(W, H);
  g.save(); g.translate(W / 2, H * 0.9); g.scale(S0, S0);
  const rnd = DP.mulberry32(1337);
  g.lineCap = 'round';
  (function branch(x, y, ang, len, wid, depth) {
    if (depth > 8 || len < 0.006) return;
    const ex = x + Math.cos(ang) * len, ey = y + Math.sin(ang) * len;
    const cx = x + Math.cos(ang - 0.25) * len * 0.55;
    const cy = y + Math.sin(ang - 0.25) * len * 0.55;
    g.lineWidth = wid;
    g.strokeStyle = 'rgba(0,0,0,' + (0.30 + 0.055 * depth).toFixed(3) + ')';
    g.beginPath(); g.moveTo(x, y); g.quadraticCurveTo(cx, cy, ex, ey); g.stroke();
    const n = rnd() < 0.22 ? 3 : 2;
    for (let i = 0; i < n; i++) {
      const spread = (i - (n - 1) / 2) * (0.42 + rnd() * 0.24);
      branch(ex, ey, ang + spread + (rnd() - 0.5) * 0.16,
             len * (0.70 + rnd() * 0.14), wid * 0.70, depth + 1);
    }
  })(0, 0, -Math.PI / 2, 0.20, 0.030, 0);
  g.restore();
}

async function addShape(kind, bitmap) {
  await playerLib();
  const W = seqW(), H = seqH();
  const c = ctx2d(W, H, 'shape');
  if (bitmap) {
    c.fillStyle = '#fff'; c.fillRect(0, 0, W, H);
    const k = Math.min(W / bitmap.width, H / bitmap.height) * 0.86;
    const w = bitmap.width * k, h = bitmap.height * k;
    c.drawImage(bitmap, (W - w) / 2, (H - h) / 2, w, h);
  } else if (kind === 'coral') drawCoral(c, W, H);
  else drawRing(c, W, H);
  const src = c.getImageData(0, 0, W, H).data;
  const mask = new Float32Array(W * H).fill(1);
  const r = dotsOn(src, W, H, [mask], S.P, BLUE);
  const xy = dotXY(r.F, r.on[0]);
  if (!xy.length) { toast('that shape came out empty', true); return; }
  const hold = Math.max(1, +$('#seqHold').value || 30);
  S.library.push({ name: kind, kind: 'shape', color: seqColor(),
                   frames: new Array(hold).fill(xy), w: W, h: H,
                   dots: xy.length >> 1 });
  renderLibrary();
  toast(`${kind}: ${xy.length >> 1} dots`);
}

const seqW = () => (S.library[0] ? S.library[0].w : (S.W || 1280));
const seqH = () => (S.library[0] ? S.library[0].h : (S.H || 720));
const seqColor = () => (S.library[0] ? S.library[0].color
  : (S.subjects[0] ? S.subjects[0].palette.slice(-1)[0] : '#b0413e'));

/* ------------------------------------------------------------- the library */
async function captureSegment() {
  const btn = $('#bCap'); btn.disabled = true;
  const info = $('#seqinfo'); info.hidden = false; info.classList.remove('err');
  info.textContent = 'reading dot positions…';
  try {
    const { doc } = await dotsDoc((pr) => { info.textContent = 'dots ' + pr.text; });
    const k = Math.max(0, +$('#seqSubj').value || 0);
    const len = Math.max(2, +$('#seqLen').value);
    const start = Math.min(S.cur, Math.max(0, doc.frames.length - 2));
    const frames = [];
    for (let i = start; i < Math.min(doc.frames.length, start + len); i++) {
      frames.push(doc.frames[i][k] || new Uint16Array(0));
    }
    const dots = frames.reduce((a, f) => a + (f.length >> 1), 0) / frames.length;
    S.library.push({
      name: `${S.fileName || 'clip'} #${(S.subjects[k] || {}).id || k + 1} `
        + `${start}–${start + frames.length - 1}`,
      kind: 'segment', color: doc.subjects[k].color, frames,
      w: doc.w, h: doc.h, dots: Math.round(dots),
    });
    renderLibrary();
    info.textContent = `captured ${frames.length} frames from ${start}, `
      + `${Math.round(dots)} dots a frame`;
  } catch (err) {
    info.classList.add('err');
    info.textContent = 'capture failed: ' + why(err);
  }
  btn.disabled = false;
}

function renderLibrary() {
  const wrap = $('#seqlist'); wrap.textContent = '';
  S.library.forEach((it, i) => {
    const b = document.createElement('span');
    b.className = 'chip seqit';
    const sw = document.createElement('span');
    sw.className = 'sw'; sw.style.background = it.color;
    const nm = document.createElement('span');
    nm.textContent = `${i + 1}. ${it.name} · ${it.frames.length}f · ${it.dots} dots`;
    b.append(sw, nm);
    const mk = (label, fn, title) => {
      const x = document.createElement('button');
      x.className = 'lnk'; x.textContent = label; x.title = title;
      x.addEventListener('click', (e) => { e.stopPropagation(); fn(); });
      b.append(x);
    };
    if (i > 0) mk('↑', () => { const t = S.library[i - 1]; S.library[i - 1] = it;
                               S.library[i] = t; renderLibrary(); }, 'move earlier');
    mk('✕', () => { S.library.splice(i, 1); renderLibrary(); }, 'remove');
    wrap.append(b);
  });
  const n = S.library.length;
  $('#vSeq').textContent = n ? `${n} item${n > 1 ? 's' : ''}` : 'nothing captured yet';
  $('#s6sum').textContent = n ? `${n} · ${(n - 1)} morph${n === 2 ? '' : 's'}` : '';
  const mixed = S.library.some((x) => x.w !== seqW() || x.h !== seqH());
  $('#seqwarn').hidden = !mixed;
  ['#bSeqPrev', '#bSeqDots', '#bSeqVideo'].forEach((id) => { $(id).disabled = n < 1; });
}

function buildSubjectPicker() {
  const sel = $('#seqSubj'); sel.textContent = '';
  S.subjects.forEach((x, i) => {
    const o = document.createElement('option');
    o.value = String(i); o.textContent = '#' + x.id;
    sel.append(o);
  });
  $('#bCap').disabled = !dotsReady();
  $('#capnote').textContent = dotsReady()
    ? 'Captures from the frame the transport is on, in the current look.'
    : 'Track a subject and pick the dots look to capture from this clip. '
      + 'Static shapes work without one.';
}

/* ------------------------------------------------------------ the sequence */
async function buildSeq() {
  const P = await playerLib();
  if (!S.library.length) throw new Error('nothing in the sequence yet');
  const doc = P.buildSequence(S.library.map((x) => ({
    name: x.name, frames: x.frames, color: x.color,
  })), {
    w: seqW(), h: seqH(), fps: S.fps || 30, dotpx: S.P.dotpx, bg: S.bg,
    color: seqColor(), durationMs: +$('#seqDur').value, seed: S.P.seed,
  });
  S.seqDoc = doc;
  return doc;
}

async function seqPreview() {
  const info = $('#seqinfo'); info.hidden = false; info.classList.remove('err');
  try {
    const P = await playerLib();
    const doc = await buildSeq();
    $('#seqwrap').hidden = false;
    $('#pwrap').hidden = true; $('#vwrap').hidden = true; $('#camwrap').hidden = true;
    $('#empty').hidden = true;
    if (!PLAYER) {
      PLAYER = new P.Player($('#seqcv'), { loop: true, onFrame: (f) => {
        $('#seqframe').textContent = `${f} / ${doc.frames.length - 1}`;
      } });
    }
    PLAYER.setDoc(doc);
    PLAYER.play();
    $('#bSeqPlay').textContent = 'pause';
    const morphs = doc.marks.filter((m) => m.kind === 'morph');
    info.textContent = `${doc.frames.length} frames · ${doc.fps} fps · `
      + `${(doc.frames.length / doc.fps).toFixed(1)} s · ${morphs.length} morph`
      + `${morphs.length === 1 ? '' : 's'} of `
      + `${morphs.map((m) => m.frames).join('/')} frames`;
  } catch (err) {
    info.classList.add('err');
    info.textContent = 'preview failed: ' + why(err);
  }
}

async function seqExportDots() {
  const info = $('#seqinfo'); info.hidden = false; info.classList.remove('err');
  try {
    const P = await playerLib();
    const doc = await buildSeq();
    const bytes = await P.pack(doc);
    download(new Blob([bytes], { type: 'application/octet-stream' }),
             'sequence.dots.gz');
    info.textContent = `${doc.frames.length} frames · `
      + `${(bytes.length / 1024).toFixed(0)} KB .dots.gz`;
  } catch (err) {
    info.classList.add('err');
    info.textContent = 'export failed: ' + why(err);
  }
}

/** A sequence as a video. The server rasterises the dot positions it is given
 *  (one implementation of the morph, in JS); without one, MediaRecorder does
 *  the same job in the tab and gives WebM. */
async function seqExportVideo() {
  const info = $('#seqinfo'); info.hidden = false; info.classList.remove('err');
  const btn = $('#bSeqVideo'); btn.disabled = true;
  try {
    const P = await playerLib();
    const doc = await buildSeq();
    const bytes = await P.pack(doc);
    let r;
    if (E().renderSequence) {
      info.textContent = 'rendering on the server…';
      r = await E().renderSequence(bytes, currentFormat().id);
    } else {
      info.textContent = 'recording in the tab…';
      r = await recordSequence(doc, (pr) => {
        info.textContent = `recording ${pr.done}/${pr.total}`;
      });
    }
    const a = $('#seqdl');
    a.href = r.url; a.download = 'sequence.' + r.ext; a.hidden = false;
    const v = $('#seqvid');
    if (r.ext !== 'gif' && r.ext !== 'mov') { v.src = r.url; v.hidden = false; }
    info.textContent = `${r.frames} frames · ${r.ext.toUpperCase()} · `
      + `${(r.bytes / 1e6).toFixed(2)} MB`
      + (r.elapsedS ? ` · ${r.elapsedS.toFixed(1)} s` : '');
  } catch (err) {
    info.classList.add('err');
    info.textContent = 'render failed: ' + why(err);
  }
  btn.disabled = false;
}

/** MediaRecorder over the player's own rasteriser — the browser engine's
 *  answer to "render this sequence". */
async function recordSequence(doc, onProgress) {
  const P = await playerLib();
  const cv = document.createElement('canvas');
  cv.width = doc.w; cv.height = doc.h;
  const g = cv.getContext('2d');
  const types = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mime = types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
  if (!mime) throw new Error('this browser has no MediaRecorder WebM encoder');
  const img = g.createImageData(doc.w, doc.h);
  const stream = cv.captureStream(0);
  const vtrack = stream.getVideoTracks()[0];
  const chunks = [];
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12e6 });
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  const stopped = new Promise((ok) => { rec.onstop = ok; });
  const t0 = performance.now();
  rec.start();
  const dt = 1000 / doc.fps;
  for (let i = 0; i < doc.frames.length; i++) {
    const fs = performance.now();
    P.paintFrame(img.data, doc.w, doc.h, doc, doc.frames[i], { bg: doc.bg });
    g.putImageData(img, 0, 0);
    vtrack.requestFrame();
    if (onProgress) onProgress({ done: i + 1, total: doc.frames.length });
    await sleep(Math.max(0, dt - (performance.now() - fs)));
  }
  await sleep(dt * 2);
  rec.stop(); await stopped; vtrack.stop();
  const blob = new Blob(chunks, { type: mime });
  return { url: URL.createObjectURL(blob), ext: 'webm', bytes: blob.size,
           frames: doc.frames.length,
           elapsedS: (performance.now() - t0) / 1000 };
}

$('#bCap').addEventListener('click', captureSegment);
$('#seqLen').addEventListener('input', (e) => { $('#vSeqLen').textContent = e.target.value + ' frames'; });
$('#seqHold').addEventListener('input', (e) => { $('#vSeqHold').textContent = e.target.value + ' frames'; });
$('#seqDur').addEventListener('input', (e) => { $('#vSeqDur').textContent = e.target.value + ' ms'; });
$$('#shapes [data-shape]').forEach((b) => b.addEventListener('click', () => addShape(b.dataset.shape)));
$('#bShapeImg').addEventListener('click', () => $('#shapeFile').click());
$('#shapeFile').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try { await addShape(f.name.replace(/\.[^.]+$/, ''), await createImageBitmap(f)); }
  catch (err) { toast('could not read that image: ' + err.message, true); }
});
$('#bSeqClear').addEventListener('click', () => { S.library = []; renderLibrary(); });
$('#bSeqPrev').addEventListener('click', seqPreview);
$('#bSeqDots').addEventListener('click', seqExportDots);
$('#bSeqVideo').addEventListener('click', seqExportVideo);
$('#bSeqPlay').addEventListener('click', () => {
  if (!PLAYER) return;
  PLAYER.toggle();
  $('#bSeqPlay').textContent = PLAYER.playing ? 'pause' : 'play';
});
$('#bSeqBack').addEventListener('click', () => {
  $('#seqwrap').hidden = true;
  if (PLAYER) PLAYER.pause();
  $('#empty').hidden = S.kind !== 'none';
  $('#vwrap').hidden = S.kind === 'none';
});

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
  buildFormats();
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
    // meta() first, then publish: window.DV_engine() and paintEngine() must
    // never see an engine whose capabilities have not been read yet
    const meta = await r.engine.meta();
    savePref(pref);
    S.enginePref = pref;
    const had = S.kind !== 'none';
    S.engine = r.engine;
    S.meta = meta;
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
  dropCache(); dropStill();
  $('#upstat').hidden = true; $('#tinfo').hidden = true; $('#rinfo').hidden = true;
  $('#dl').hidden = true; $('#outvid').hidden = true; $('#pvinfo').hidden = true;
  $('#outimg').hidden = true; $('#fmtui').hidden = true; $('#trimui').hidden = true;
  S.srcFile = null; S.trim = null; S.srcDuration = 0;
  showStage('empty');
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
    box.textContent = 'Selecting a subject — in a photograph or a clip — needs the '
      + 'EdgeTAM model files, and they are not here. ' + S.modelsMissing
      + '  Whole-image stills and whole-frame clips, dots included, work without '
      + 'them.';
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
  window.DV_still = {
    segment: segmentStill, use: useStillSelection, doc: dotsDocStill,
    masks: () => Array.from(S.stillMasks.keys()),
    alpha: (on) => { S.pngAlpha = !!on; $('#cAlpha').checked = !!on; },
    /* mask coverage in the picture's own pixels — what the verifiers assert on */
    areas: () => {
      const out = {};
      const ms = stillMasksAt(S.W, S.H, 'vm');
      S.subjects.forEach((sub, k) => {
        let n = 0;
        for (let q = 0; q < ms[k].length; q++) if (ms[k][q] >= 0.5) n++;
        out[sub.id] = n;
      });
      return out;
    },
  };
  window.DV_formats = engineFormats;
  window.DV_camera = { open: camOpen, start: camStart, stop: camStop,
                       close: camClose, snap: camSnap,
                       state: () => ({
                         live: !!CAM.stream, recording: !!CAM.rec,
                         recordedS: S.recordedS, photo: S.photo || null,
                         track: CAM.stream
                           ? CAM.stream.getVideoTracks()[0].getSettings() : null,
                       }) };
  window.DV_trim = (start, end) => {
    S.trim = { start, end }; paintTrim();
    return { start, end, duration: S.srcDuration };
  };
  window.DV_dots = { doc: dotsDoc, params: dotsParams, lib: playerLib,
                     library: () => S.library, build: buildSeq,
                     capture: captureSegment, shape: addShape,
                     preview: seqPreview, player: () => PLAYER };
  window.DV_setFormat = (id) => {
    S.format = id; $('#sFmt').value = id; paintFormat();
    return currentFormat();
  };
  window.DV_ready = true;
})();
