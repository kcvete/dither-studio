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
import * as CV from './canvas.js';

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
  // the source clip's own pixels (the estimate needs them before any decode)
  srcW: 0, srcH: 0,
  // set by DV_limit() before a drop: take only the first N seconds. Nothing in
  // the UI sets it — it is how the verifiers ask for a shorter clip.
  pendingLimit: 0,
  // true while a long clip is sitting in the trim bar waiting for a click —
  // the consent gate. Never a refusal: "whole clip" clears it.
  awaitingChoice: false,
  // the job the current frames were extracted from, so a second trim can be
  // re-cut from the source the server already holds
  srcJob: null,
  /* THE ACTIVE RANGE. `range` is an inclusive [in, out] pair of frame indices
   * into the clip that is open -- null means the whole of it. A trim dragged
   * AFTER the tracking sets this instead of re-extracting: the frames and the
   * per-frame masks on disk do not move, so a narrower range is a window on
   * them and costs nothing. `jobStart` is where those frames begin in the
   * SOURCE file's own seconds, which is what turns the trim bar's seconds into
   * frame indices. `extend` holds a range the user asked for that runs past
   * what was extracted -- the offer, not a refusal. */
  range: null, jobStart: 0, extend: null,
  // clip
  job: null, nFrames: 0, W: 0, H: 0, fps: 30,
  // `promptFrame` is where the SCRUBBER is. Each subject remembers the frame it
  // was actually prompted on -- a ball that flies in at frame 80 does not exist
  // on frame 0, so one prompt frame per clip was never enough.
  scope: 'whole', subjects: [], active: 0, nextId: 1, promptFrame: 0,
  tool: 'point', curPath: null, hoverXY: null, previewMasks: null,
  promptMode: 'add',                 // ⊕ keep / ⊖ remove — the visible polarity
  coachSeen: null,                   // which coach moments have fired (per source)
  trackSize: 1024, tracked: false, playing: false, cur: 0,
  // look
  P: {
    mode: 'bluenoise', algo: 'floyd-steinberg', matrix: 4, serpentine: false,
    strength: 1, brightness: 0, contrast: 1, gamma: 1, invert: false, pixel: 1,
    compose: 'cutout', seed: 7,
    n: 8000, cell: 4, dotpx: 3, fill: 0.7, stray: 0.02, band: 9,
  },
  palette: ['#000000', '#ffffff'],   // background / whole-frame palette
  lookPreset: 'custom',              // which look-preset tile is lit
  paletteTouched: false,             // has anyone chosen one yet?
  dotsTuned: false,                  // has the dot count been set for this still?
  bg: '#c9d4c5',
  target: 'bg',                      // which palette the editor is editing: 'bg' | subject id
  meta: null,
  compare: false, split: 0.5,
  // engine
  engine: null, enginePref: null, modelsMissing: null,
  exportURL: null, frameURL: null, exportOrigURL: null,
  // export
  format: '', gifFps: 15,
  // "also save the original (matched cut)": the render's own frames, undithered.
  // Remembered for the session, off in a fresh tab.
  saveOriginal: false,
  // sequence. `library` is the session's pool of captured dot clouds and
  // survives loading another clip on purpose — a morph from one clip into
  // another needs both, and only one can be open at a time. `strip` is the
  // sequence itself: instances of pool items, each carrying its own trim,
  // colour and the transition that leads into it.
  view: 'studio', library: [], strip: [], sel: null, seqDoc: null,
  returnToSeq: false, pendingImage: null,
  seq: { dotpx: 3, bg: '#c9d4c5', fps: 30, format: '', preset: 'source', w: 0, h: 0 },
  /* THE CANVAS. `preset` is one of web/canvas.js's, and 'source' -- the
   * default -- means everything below is inert and the tool behaves exactly as
   * it did before there was a canvas at all. `follow` is the framing: 'auto'
   * decides between holding still and tracking the subject by asking whether
   * the subject ever leaves a fixed frame. `dx`/`dy` are the manual bias, as a
   * FRACTION of the source frame, that dragging the preview adds to the
   * smoothed path — a fraction and not pixels, because the preview, a still's
   * native-resolution export and the server are three different grids. */
  canvas: { preset: 'source', w: 1080, h: 1920, follow: 'auto', zoom: 1,
            dx: 0, dy: 0 },
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
  // the hero demo runs only while the landing is what is on stage
  if (typeof HERO !== 'undefined' && HERO) {
    if (which === 'empty'
        && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      HERO.play();
    } else HERO.pause();
  }
}

/* Only the studio rail accordions: the sequence panel's sections are a
 * different set of steps and must not close because the studio moved on. */
function openStep(n) {
  $$('#studiopanel .step').forEach((el) => el.setAttribute(
    'data-open', el.id === 'st' + n ? '1' : '0'));
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
/* ------------------------------------------------------- clip estimates ===
 * There is no length cap. What replaces it is arithmetic done out loud before
 * anything is decoded: how many frames that range is, what they weigh, and how
 * long tracking them will take at the quality that is selected. Over a minute
 * the note appears and the clip waits for a click -- it is still never
 * refused, and the trim bar works before AND after.
 */
const LONG_S = 60;              // over this: say the number, wait for a click
const TAB_MEM_WARN = 2e9;       // over this in-tab: suggest the local server
const DECODE_FPS = 30;
const DECODE_H = 720;

const fmtBytes = (b) => (b >= 1e9 ? (b / 1e9).toFixed(1) + ' GB'
  : b >= 1e6 ? Math.round(b / 1e6) + ' MB' : Math.round(b / 1e3) + ' KB');
const fmtDur = (s) => (s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
  : `${s < 10 ? s.toFixed(1) : Math.round(s)}s`);

/* The tracker resolutions, in words a first-timer can pick between. The
 * engine's own ids/labels survive as the secondary text on each chip. */
const TQ_HUMAN = { fast: 'Draft', balanced: 'Standard', best: 'Fine' };
const tqName = (t) => (t && (TQ_HUMAN[t.id] || t.label.split(' ')[0])) || '';

/** Seconds the current trim covers (the whole clip when nothing is trimmed). */
function trimSeconds() {
  if (S.trim) return Math.max(0, S.trim.end - S.trim.start);
  return S.srcDuration || 0;
}

/** Everything the estimate line says, as numbers. */
function clipEstimate() {
  const secs = trimSeconds();
  const n = Math.max(1, Math.round(secs * DECODE_FPS));
  const vw = S.srcW || S.W || 1280, vh = S.srcH || S.H || 720;
  const h = DECODE_H, w = Math.max(2, Math.round(vw * (h / vh) / 2) * 2);
  const jpeg = n * 90e3 * (w * h) / (1280 * 720);
  const lru = 40 * w * h * 4;
  const masks = n * 192 * 192 * 4 * Math.max(1, S.subjects.length || 1);
  const t = (S.meta.track_sizes || []).find((x) => x.size === S.trackSize)
    || (S.meta.track_sizes || [])[0];
  const fps = (t && t.fps) || 0;
  return { secs, n, w, h, jpeg, tabBytes: jpeg + lru + masks,
           trackFps: fps, trackS: fps ? n / fps : 0,
           quality: tqName(t), size: t ? t.size : 0 };
}

/** The Track button carries the honest estimate: what it costs, before it is
 *  pressed. Stills keep their own label ("Use this selection"). */
function paintTrackCTA() {
  const b = $('#bTrack');
  if (!b || S.kind !== 'video') return;
  if (b.dataset.running === '1') return;      // progress copy owns it mid-run
  const nSub = Math.max(1, S.subjects.length || 1);
  let est = 0;
  try { est = clipEstimate().trackS * nSub; } catch (e) { est = 0; }
  b.textContent = (nSub > 1 ? `Track ${nSub} subjects` : 'Track subject')
    + (est ? ` — ≈ ${fmtDur(est)}` : '');
}

function paintEstimate() {
  paintTrackCTA();
  if (S.kind === 'image' || !S.srcFile) { $('#vidopts').hidden = true; return; }
  const e = clipEstimate();
  const browser = E().id === 'browser';
  $('#vEst').textContent = `${e.secs.toFixed(1)} s · ${e.n} frames`;
  $('#estline').innerHTML = `${e.w}×${e.h} @ ${DECODE_FPS} fps · frames `
    + `<b>≈ ${fmtBytes(browser ? e.tabBytes : e.jpeg)}</b> `
    + (browser ? 'in this tab' : 'on the server')
    + (e.trackS ? ` · tracking one subject <b>≈ ${fmtDur(e.trackS)}</b>`
        + (e.quality ? ` at ${e.quality} · ${e.size} px` : '') : '');
  const warn = browser && e.tabBytes > TAB_MEM_WARN;
  $('#estwarn').hidden = !warn;
  if (warn) {
    $('#estwarn').textContent = `≈ ${fmtBytes(e.tabBytes)} of frames and masks `
      + 'in this tab — it will work, but the local server engine keeps them on '
      + 'disk instead. Switch engines, or trim.';
  }
  const long = e.secs > LONG_S;
  $('#estlong').hidden = !long;
  if (long) {
    $('#estlong').textContent = `long clip: tracking ≈ ${fmtDur(e.trackS)}`
      + ' — consider trimming. You can also trim afterwards, and re-cut '
      + 'without uploading again.';
  }
}

function take(f) {
  const isVid = /^video\//.test(f.type) || /\.(mp4|mov|m4v|webm)$/i.test(f.name);
  $('#vidopts').hidden = !isVid;
  $('#trimui').hidden = true;
  if (!isVid) { S.srcFile = null; return loadStill(f); }
  // A short clip loads whole, immediately, and the trim bar appears next to
  // it. A long one shows its arithmetic first and waits for a click -- not a
  // cap, a sentence: nothing is refused and "whole clip" is right there.
  S.srcFile = f;
  S.trim = null;
  return takeClip(f);
}

async function takeClip(f) {
  const box = $('#upstat'); box.hidden = false; box.classList.remove('err');
  box.textContent = 'reading ' + f.name + '…';
  const strip = buildStrip(f);            // thumbnails, in the background
  let dur = 0;
  try { dur = await probeFile(f); } catch (err) { dur = 0; }
  S.srcDuration = dur;
  S.trim = { start: 0, end: dur };
  if (S.pendingLimit > 0) {
    S.trim.end = Math.min(dur || S.pendingLimit, S.pendingLimit);
    S.pendingLimit = 0;
  }
  paintTrim(); paintEstimate();
  S.awaitingChoice = false;
  if (!dur || trimSeconds() <= LONG_S) return uploadClip(f, S.trim);
  const e = clipEstimate();
  S.awaitingChoice = true;
  box.textContent = `${f.name} · ${dur.toFixed(1)} s · ${e.n} frames · tracking `
    + `≈ ${fmtDur(e.trackS)} — press “use this range” for the trim below, `
    + 'or “whole clip”.';
  await strip;
}

/** Duration + natural size of a file, from its header alone. No frames. */
function probeFile(f) {
  const url = URL.createObjectURL(f);
  const v = document.createElement('video');
  v.preload = 'metadata'; v.muted = true; v.playsInline = true; v.src = url;
  return new Promise((ok, no) => {
    v.onloadedmetadata = () => ok();
    v.onerror = () => no(new Error('cannot read that clip'));
    setTimeout(() => no(new Error('timed out reading the header')), 20000);
  }).then(async () => {
    let d = v.duration;
    if (!isFinite(d) || d <= 0) d = await probeDuration(v);
    S.srcW = v.videoWidth || 0; S.srcH = v.videoHeight || 0;
    return d;
  }).finally(() => { v.src = ''; v.load?.(); URL.revokeObjectURL(url); });
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
    // takeClip() owns S.trim -- it read the same header first. The strip only
    // needs the duration to space its twelve thumbnails.
    if (!S.srcDuration) { S.srcDuration = dur; paintTrim(); paintEstimate(); }
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
  $('#vTrim').textContent = `${start.toFixed(1)} – ${end.toFixed(1)} s · `
    + `${(end - start).toFixed(1)} s`;
  paintEstimate();
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

/* ============================================ the range, after the track ===
 * The frames of a clip are extracted once, and the tracker writes one mask
 * file per frame beside them. Both are numbered from 0 and neither moves. So
 * once a clip is tracked, "trim it" does not mean "extract a different clip"
 * -- it means "use frames a..b of the ones that are already here", which needs
 * no ffmpeg, no upload, and above all no second track.
 *
 * That is the whole feature: below, the trim bar decides which of the two
 * things it is being asked for.
 *
 *   nothing tracked yet   re-extract, exactly as before -- there is nothing to
 *                         lose, and tracking fewer frames is cheaper than
 *                         tracking frames you are going to throw away
 *   tracked, narrower     set S.range. No network call at all.
 *   tracked, wider        say which frames are not tracked, and offer to do it
 *
 * Everything downstream -- the preview, the render, every export format, the
 * matched original cut, the .dots.gz, and what "add to the sequence" seeds an
 * item's in/out with -- reads activeRange().
 */

/** The active window, resolved and clamped: {in, out, n, whole}. */
function activeRange() {
  const n = S.kind === 'video' ? (S.nFrames | 0) : 0;
  if (n <= 0) return { in: 0, out: 0, n: 0, whole: true };
  const r = S.range || { in: 0, out: n - 1 };
  const a = clamp(r.in | 0, 0, n - 1);
  const b = clamp(r.out | 0, a, n - 1);
  return { in: a, out: b, n: b - a + 1, whole: a === 0 && b === n - 1 };
}

/** Seconds of the SOURCE file that the extracted frames cover. */
function jobWindow() {
  const span = (S.nFrames | 0) / Math.max(1, S.fps);
  return { start: S.jobStart || 0, end: (S.jobStart || 0) + span };
}

/** Set the window and repaint everything that depends on it. */
function setRange(a, b) {
  const n = S.nFrames | 0;
  if (n <= 0) return activeRange();
  const lo = clamp(a | 0, 0, n - 1);
  const hi = clamp(b | 0, lo, n - 1);
  S.range = (lo === 0 && hi === n - 1) ? null : { in: lo, out: hi };
  S.extend = null;
  DOTS_CACHE = null;                 // the dots doc is per range
  const r = activeRange();
  paintRange(); paintTrimOffer();
  if (S.cur < r.in || S.cur > r.out) { stop(); draw(r.in); }
  return r;
}

/** Whether this engine can render a window out of frames it already has.
 *  The browser one always can; a server older than the frame_range field
 *  cannot, and then a trim has to go back to being a re-extract. */
const canWindow = () => !!(E() && (E().id === 'browser'
  || (E().supports && E().supports.frameRange)));

/** What the trim bar is actually being asked for, given what is on disk. */
function trimPlan(t) {
  if (S.kind !== 'video' || !S.job || !S.nFrames) return { kind: 'open' };
  // nothing tracked: re-extracting is both cheaper downstream and lossless
  if (!S.tracked || !S.subjects.length || !canWindow()) return { kind: 'open' };
  const j = jobWindow();
  const eps = 0.5 / Math.max(1, S.fps);         // half a frame of slack
  const a = clamp(Math.round((t.start - j.start) * S.fps), 0, S.nFrames - 1);
  const b = clamp(Math.round((t.end - j.start) * S.fps) - 1, a, S.nFrames - 1);
  if (t.start < j.start - eps || t.end > j.end + eps) {
    /* The frames of the range being ASKED for, numbered as the re-extraction
     * would number them, so the message can name the ones that do not exist
     * yet rather than gesturing at seconds. */
    const total = Math.max(1, Math.round((t.end - t.start) * S.fps));
    const off = Math.round((j.start - t.start) * S.fps);   // where the tracked
    const head = Math.max(0, off);                         // part lands
    const tail = Math.min(total - 1, off + S.nFrames - 1);
    return { kind: 'extend', job: j, total,
             missing: [].concat(head > 0 ? [[0, head - 1]] : [],
                                tail < total - 1 ? [[tail + 1, total - 1]] : []) };
  }
  return { kind: 'window', in: a, out: b };
}

/** "use this range" / "whole clip", once a clip is open. */
function applyTrim(t0) {
  if (!S.srcFile || !t0) return;
  // the source is as long as it is: a range that runs off the end asks for
  // the end, exactly as ffmpeg's -t already clamps it
  const dur = S.srcDuration || 0;
  const t = dur > 0
    ? { start: clamp(t0.start, 0, dur), end: clamp(t0.end, 0, dur) }
    : { start: t0.start, end: t0.end };
  const plan = trimPlan(t);
  if (plan.kind === 'open') {
    return uploadClip(S.srcFile, { start: t.start, end: t.end },
                      { recut: S.kind === 'video' && !!S.job });
  }
  if (plan.kind === 'window') {
    const r = setRange(plan.in, plan.out);
    const box = $('#upstat'); box.hidden = false; box.classList.remove('err');
    box.textContent = r.whole
      ? `the whole clip · ${S.nFrames} frames · nothing re-tracked`
      : `frames ${r.in}–${r.out} of ${S.nFrames} · ${(r.n / Math.max(1, S.fps)).toFixed(1)} s`
        + ' · same frames, same masks, nothing re-tracked';
    toast(r.whole ? 'back to the whole tracked clip'
      : `${r.n} frames — the tracking is untouched`);
    return Promise.resolve(r);
  }
  S.extend = Object.assign({ trim: { start: t.start, end: t.end } }, plan);
  paintTrimOffer();
  // the offer lives next to the trim bar, so make sure that step is open --
  // a panel nobody can see is the same as no panel
  openStep(1);
  $('#trimui').hidden = false;
  return Promise.resolve(null);
}

$('#bTrim').addEventListener('click', () => {
  if (!S.srcFile || !S.trim) return;
  applyTrim({ start: S.trim.start, end: S.trim.end });
});
$('#bTrimAll').addEventListener('click', () => {
  if (!S.srcFile) return;
  S.trim = { start: 0, end: S.srcDuration || 0 };
  paintTrim();
  applyTrim(S.trim);
});

/* ---- the offer -------------------------------------------------------
 * A range that runs past what was extracted cannot be served out of frames
 * that do not exist. Rather than silently re-cutting (which is what threw the
 * tracking away in the first place), the page says which frames are missing
 * and what getting them costs, and waits.
 *
 * WHAT IS SHIPPED HERE IS A FULL RE-TRACK of the wider range, not a tail
 * propagated out of the existing memory. EdgeTAM's inference state is built by
 * predictor.init_state() over the whole frames directory and is torn down at
 * the end of every /track call (server/server.py, _track_worker) -- there is no
 * memory bank left to walk forward from, on either engine. Keeping one alive
 * across requests is a different piece of work; pretending the cheap thing
 * happened would be worse than saying the number.
 */
function paintTrimOffer() {
  const box = $('#trimoffer');
  if (!box) return;
  const x = S.extend;
  box.hidden = !x;
  if (!x) return;
  const e = clipEstimate();          // S.trim is already the asked-for range
  const one = x.missing.length === 1 && x.missing[0][0] === x.missing[0][1];
  const list = x.missing.map(([a, b]) => (a === b ? `frame ${a}`
    : `frames ${a}–${b}`)).join(' and ');
  $('#trimoffernote').textContent =
    `${list} ${one ? "isn't" : "aren't"} tracked yet. The clip on disk covers `
    + `${x.job.start.toFixed(1)}–${x.job.end.toFixed(1)} s (${S.nFrames} frames); `
    + `this range is ${x.total}. The tracker keeps no memory between runs, so `
    + `the wider range is re-extracted and tracked in full — about `
    + `${fmtDur(e.trackS)} for ${S.subjects.length} subject`
    + `${S.subjects.length > 1 ? 's' : ''}, with the prompts you already placed.`;
  $('#bExtend').textContent = `re-cut and track ${x.total} frames`;
}

$('#bExtendNo').addEventListener('click', () => {
  S.extend = null;
  paintTrimOffer();
  const r = activeRange();
  S.trim = { start: jobWindow().start + r.in / Math.max(1, S.fps),
             end: jobWindow().start + (r.out + 1) / Math.max(1, S.fps) };
  paintTrim();
});
$('#bExtend').addEventListener('click', () => extendAndTrack());

/** Take the wider range and track it, carrying the prompts across.
 *
 *  The prompts are clip-pixel coordinates on a numbered frame. The pixels do
 *  not move (the decode height is fixed), but the numbering does: a range that
 *  starts earlier pushes every old index forward by the difference. */
async function extendAndTrack() {
  const x = S.extend;
  if (!x || !S.srcFile) return;
  S.extend = null; paintTrimOffer();
  const shift = Math.round((jobWindow().start - x.trim.start) * S.fps);
  const keep = S.subjects.map((s) => Object.assign({}, s, {
    promptFrame: s.promptFrame === null ? null : s.promptFrame + shift,
  }));
  await uploadClip(S.srcFile, x.trim, { recut: !!S.job, keep });
  if (!S.subjects.length || S.kind !== 'video') return;
  setScope('track');
  await track();
}

/* ============================================================== camera ===
 * getUserMedia -> live preview -> MediaRecorder -> a WebM blob that goes
 * through exactly the same path a dropped file does. Which means it works on
 * both engines: the browser one decodes the blob, the server one uploads it
 * (server.py already accepts .webm, and ffmpeg reads what Chrome writes).
 */
const CAM = { stream: null, rec: null, chunks: [], t0: 0, timer: 0 };
// A sanity stop, not a length limit: a forgotten recording should not fill the
// disk with an 8 Mbit/s WebM. Five minutes is ~300 MB and 9,000 frames.
const CAM_MAX_S = 300;

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
    + ` · stops itself at ${Math.round(CAM_MAX_S / 60)} min`
    + ' · nothing leaves the tab until you export';
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
  if (S.view === 'sequence') return showStage('sequence');
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
  $('#empty').hidden = kind !== 'none' || S.view === 'sequence';
  // renumber visible steps so the studio rail always reads 1,2,3…
  let n = 0;
  $$('#studiopanel .step').forEach((st) => {
    if (!st.hidden) $('.sh i', st).textContent = ++n;
  });
  paintToSeq();
  paintCanvasUI();
  if (S.view === 'sequence') renderAdd();
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
    S.paletteTouched = false; S.dotsTuned = false; S.coachSeen = null;
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
    S.range = null; S.extend = null; paintRange(); paintTrimOffer();
    $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
    $('#sharerow') && ($('#sharerow').hidden = true);
    $('#outimg').hidden = true; $('#pvinfo').hidden = true; $('#tinfo').hidden = true;
    $('#s5sum').textContent = ''; $('#bExport').textContent = 'Download PNG';
    $('#fmtui').hidden = true; $('#trimui').hidden = true;
    $('#origui').hidden = true; $('#dlorig').hidden = true;
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

async function uploadClip(f, trim, opts = {}) {
  const box = $('#upstat'); box.hidden = false; box.classList.remove('err');
  box.textContent = (E().id === 'browser' ? 'decoding ' : 'uploading ') + f.name + '…';
  busy(true);
  S.awaitingChoice = false;
  const t = trim || S.trim || null;
  const cut = t && (t.start > 0.05 || (S.srcDuration && t.end < S.srcDuration - 0.05));
  // A second trim of a clip that is already here does not go up again: the
  // server kept source.mp4 and the tab kept the File handle, so both engines
  // can re-cut from what they have.
  const recut = !!opts.recut && E().supports && E().supports.reextract
    && (E().id === 'browser' || S.srcJob === S.job);
  try {
    const args = {
      trimStart: cut ? t.start : 0,
      trimEnd: cut ? t.end : null,
      onProgress: (p) => { if (p.text) box.textContent = p.text; },
    };
    const j = recut ? await E().reopen(args) : await E().open(f, args);
    S.srcJob = j.job;
    S.trim = t;
    dropStill();
    S.kind = 'video'; S.job = j.job; S.nFrames = j.nFrames; S.W = j.w; S.H = j.h; S.fps = j.fps;
    S.fileName = f.name.replace(/\.[^.]+$/, '');
    S.tracked = false; S.subjects = []; S.nextId = 1; S.cur = 0; S.promptFrame = 0;
    S.scope = 'whole'; S.coachSeen = null;
    /* Where these frames start in the source file's own seconds. It is what
     * lets the trim bar (which speaks seconds) address frames on disk, so a
     * later trim can be a window instead of a second extraction. */
    S.jobStart = +j.trimStart || 0;
    S.range = null; S.extend = null;
    /* A re-cut that is carrying prompts forward: same subjects, same marks,
     * frame numbers moved by however much the range's start moved. */
    if (opts.keep && opts.keep.length) {
      S.subjects = opts.keep.map((s) => Object.assign({}, s, {
        promptFrame: s.promptFrame === null ? null
          : clamp(s.promptFrame, 0, j.nFrames - 1),
      }));
      S.nextId = Math.max(...S.subjects.map((s) => s.id | 0)) + 1;
      S.scope = 'track';
    }
    dropCache();
    box.textContent = `${j.nFrames} frames · ${j.w}×${j.h} · ${j.fps} fps`
      + ` · ${(j.nFrames / Math.max(1, j.fps)).toFixed(1)} s`
      + (cut ? ` · trimmed ${t.start.toFixed(1)}–${t.end.toFixed(1)} s` : '')
      + (recut ? ' · re-cut, nothing re-uploaded' : '')
      + (E().id === 'browser' ? ' · stays in this tab' : '');
    paintEstimate();
    $('#s1sum').textContent = `${j.nFrames}f`;
    $('#sPF').max = j.nFrames - 1; $('#sPF').value = 0; $('#vPF').textContent = '0';
    $('#sFrame').max = j.nFrames - 1;
    $('#bPlay').hidden = $('#sFrame').hidden = $('#fcount').hidden = false;
    $('#dl').hidden = true; $('#outvid').hidden = true; $('#rinfo').hidden = true;
    $('#sharerow') && ($('#sharerow').hidden = true);
    $('#tinfo').hidden = true; $('#s5sum').textContent = '';
    $('#outimg').hidden = true; $('#dlorig').hidden = true;
    buildFormats();
    if (S.P.mode === 'dots') setMode('bluenoise');
    showSteps('video'); setScope(S.scope);
    buildTargets(); renderModes(); renderSubjects(); paintRange(); paintTrimOffer();
    openStep(2);
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
    coach('start', (COARSE ? 'tap' : 'click')
      + ' the thing you want — the outline appears as you do');
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
  // a cutout and an overlay want different crops (one clamps, one does not),
  // so the canvas note and the follow-or-hold answer move with this
  paintCanvasUI();
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
  $('#tqlbl').textContent = still ? 'Detail' : 'Detail';
  $('#tqnote').textContent = still
    ? 'Your picture keeps its own resolution — this only changes the square the '
      + 'model looks at, and so how fine an outline it can cut.'
    : 'Your clip keeps its own resolution — this only changes the square the '
      + 'tracker looks at, and so how fine an outline it can draw.';
  if (still) $('#bTrack').textContent = 'Use this selection';
  else paintTrackCTA();
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
                    points: [], box: null, paths: [], promptFrame: null,
                    // mask polish, 0-100; 0 is off and is the default. See
                    // web/polish.js — the same algorithm the server runs.
                    polish: 0 });
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
  renderPolish();
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
      // SAM-style marker glyphs: shape carries the meaning, colour reinforces.
      // keep = subject colour, white ring, white +; remove = red, white ring, −
      const keep = !!p[2], R = 8;
      pctx.beginPath(); pctx.arc(p[0], p[1], R, 0, Math.PI * 2);
      pctx.fillStyle = keep ? col : '#E6193B'; pctx.fill();
      pctx.lineWidth = 2; pctx.strokeStyle = '#ffffffdd'; pctx.stroke();
      pctx.strokeStyle = '#fff'; pctx.lineWidth = 2.4; pctx.lineCap = 'round';
      pctx.beginPath();
      pctx.moveTo(p[0] - 3.6, p[1]); pctx.lineTo(p[0] + 3.6, p[1]);
      if (keep) { pctx.moveTo(p[0], p[1] - 3.6); pctx.lineTo(p[0], p[1] + 3.6); }
      pctx.stroke();
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

function paintPathButtons() {
  const box = $('#pathbtns');
  if (!box) return;
  box.hidden = !(S.curPath && S.tool === 'poly');
}
$('#bPathOk') && $('#bPathOk').addEventListener('click', () => commitPath());
$('#bPathNo') && $('#bPathNo').addEventListener('click', () => {
  S.curPath = null; S.hoverXY = null; drawOverlay();
});
function drawPaths() {
  paintPathButtons();
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

/* ---- ⊕ / ⊖: the visible polarity ---------------------------------------
 * The taught path on every pointer. Shift/alt still mean "the opposite of ⊕"
 * for pointer-fine muscle memory; on touch, a tap that lands INSIDE the tinted
 * mask subtracts implicitly (Roboflow's smart default) so one hand is enough. */
function paintPolarity() {
  $$('#polarity .chip').forEach((c) => c.setAttribute(
    'aria-pressed', String(c.dataset.polarity === S.promptMode)));
  $('#vPolarity').textContent = S.promptMode === 'sub' ? 'remove' : 'keep';
}
$$('#polarity .chip').forEach((c) => c.addEventListener('click', () => {
  S.promptMode = c.dataset.polarity;
  paintPolarity();
}));
paintPolarity();

/** Is (x, y) — prompt-canvas pixels — inside the active subject's live mask? */
function maskHit(x, y) {
  const s = S.subjects[S.active];
  if (!s) return false;
  const im = S.kind === 'image' ? S.stillMasks.get(s.id)
    : (S.previewMasks || {})[String(s.id)];
  if (!im || im.complete === false) return false;
  try {
    const c = ctx2d(S.W, S.H, 'hit');
    c.clearRect(0, 0, S.W, S.H);
    c.drawImage(im, 0, 0, S.W, S.H);
    return c.getImageData(x | 0, y | 0, 1, 1).data[0] > 127;
  } catch (e) { return false; }
}

/* ---- the coach: one floating pill over the canvas ------------------------
 * Fires once per moment per source, never a modal, dies on tap or on its own.
 * The SAM 2 pattern, with remove.bg's failure copy for the empty-mask case. */
function coach(id, msg, again) {
  if (!S.coachSeen) S.coachSeen = new Set();
  if (S.coachSeen.has(id) && !again) return;
  S.coachSeen.add(id);
  const el = $('#coach');
  if (!el) return;
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(coach._t);
  coach._t = setTimeout(() => { el.hidden = true; }, 7000);
}
document.addEventListener('DOMContentLoaded', () => {});
$('#coach') && $('#coach').addEventListener('click', () => { $('#coach').hidden = true; });

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
  hideDemoHint();
  if (!S.subjects.length) return;
  const act = S.subjects[S.active];
  if (!onThisFrame(act)) {
    toast(`subject #${act.id} was prompted on frame ${act.promptFrame} — `
      + 'jump back to it, or add a new subject for this frame', true);
    return;
  }
  const p = povXY(e);
  const neg = e.shiftKey || e.altKey || S.promptMode === 'sub';
  if (S.tool === 'point') {
    pov.setPointerCapture(e.pointerId);
    down = { xy: p, moved: false, neg, touch: e.pointerType === 'touch' };
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
  if (down.moved && S.dragBox) {
    claimFrame(s);
    s.box = S.dragBox.map(Math.round);
  } else {
    // a tap on an existing marker deletes it (SAM 2) — the outline re-cuts
    const near = s.points.findIndex((q) =>
      Math.hypot(q[0] - p[0], q[1] - p[1]) < 10);
    if (near >= 0) {
      s.points.splice(near, 1);
    } else {
      claimFrame(s);
      // touch default: a tap inside the tinted mask means "not this bit" —
      // no toggle to remember, one-handed refinement (⊕/⊖ stays the override)
      const neg = down.neg
        || (down.touch && S.promptMode === 'add' && maskHit(p[0], p[1]));
      s.points.push([Math.round(p[0]), Math.round(p[1]), neg ? 0 : 1]);
    }
  }
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

/* ---- prompt tool ----
 * Two sets of words for the same tools: a phone has no shift key and no esc,
 * so on a coarse pointer every hint speaks in taps and the ⊕/⊖ pair. */
const COARSE = window.matchMedia
  && window.matchMedia('(pointer: coarse)').matches;
const TOOLHINT = COARSE ? {
  point: 'tap = keep · ⊖ mode = remove · drag a box',
  lasso: 'draw around it with one finger · ⊖ mode subtracts',
  poly: 'tap each corner · ✓ closes · ✕ cancels',
} : {
  point: "click what you want · shift-click what you don't · drag a box",
  lasso: 'drag around the subject · shift-drag to subtract · esc cancels',
  poly: 'click each corner · double-click or enter to close · esc cancels',
};
const TOOLNOTE = COARSE ? {
  point: 'Tap = keep this · tap inside the tint (or ⊖ mode) = not this · '
       + 'drag = a box. The tracker re-derives the outline itself on every frame.',
  lasso: 'Draw around the subject with one finger · switch to ⊖ to subtract. '
       + 'A drawn shape replaces this subject\'s taps and box.',
  poly: 'Tap each corner, then ✓ to close · ⊖ makes a subtracting shape. '
      + 'A drawn shape replaces this subject\'s taps and box.',
} : {
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
  $('#vTool').textContent = S.tool === 'point' ? 'tap / box'
    : S.tool === 'lasso' ? 'draw around' : 'trace corners';
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
    b.title = t.label;
    b.innerHTML = `${tqName(t)} · ${t.size} px`
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
  $('#vTQ').textContent = t ? tqName(t) : `${S.trackSize} px`;
  // the estimate quotes THIS quality's fps, so it moves when the chip does
  paintEstimate();
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
  else if (S.kind === 'video' && S.scope === 'track') autoPreview();
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
    coachOnMasks(r.objects);
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

/* ---- the live mask on a clip -------------------------------------------
 * The magic moment, with no extra button: every prompt edit auto-runs the
 * same single-frame prediction "preview this frame" always ran, and the tint
 * paints the instant it lands. #bPrev survives as the manual re-check.
 * Coalesced exactly like the still path: a click that lands while a
 * prediction is in flight schedules exactly one more run. */
let pvBusy = false, pvAgain = false;

/** The empty-selection failure branch is a routine outcome, not an exception:
 *  say so, in remove.bg's words, and keep the number honest in #pvinfo. */
function coachOnMasks(objects) {
  const frame = Math.max(1, S.W * S.H);
  const biggest = objects.reduce((a, o) => Math.max(a, o.area || 0), 0);
  if (biggest < frame * 0.002) {
    coach('empty', 'that selection came back almost empty — try tapping the '
      + 'middle of the thing you want', true);
  } else {
    coach('refine', 'not what you expected? add a few more taps until the '
      + 'whole thing is selected');
  }
}

async function runPreview(here) {
  const info = $('#pvinfo'); info.hidden = false; info.classList.remove('err');
  if (!info.textContent) info.textContent = 'predicting this frame…';
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
    coachOnMasks(r.objects);
  } catch (err) {
    info.classList.add('err');
    info.textContent = 'preview failed: ' + why(err);
    throw err;
  }
}

/** Auto-run after a prompt edit on a clip. Never toasts, never blocks. */
async function autoPreview() {
  if (S.kind !== 'video' || S.scope !== 'track') return;
  if (S.modelsMissing) return;
  const here = S.subjects.filter((s) => hasPrompt(s) && onThisFrame(s));
  if (!here.length) { S.previewMasks = null; drawOverlay(); return; }
  if (pvBusy) { pvAgain = true; return; }
  pvBusy = true;
  try { await runPreview(here); } catch (e) { /* #pvinfo already says */ }
  pvBusy = false;
  if (pvAgain) { pvAgain = false; autoPreview(); }
}

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
  if (pvBusy) { pvAgain = true; return; }
  const btn = $('#bPrev'); btn.disabled = true;
  pvBusy = true;
  try { await runPreview(here); } catch (e) { /* said in #pvinfo */ }
  pvBusy = false;
  btn.disabled = false;
  if (pvAgain) { pvAgain = false; autoPreview(); }
}

/* ---- the wait is the show -----------------------------------------------
 * During Track the stage stops being a frozen prompt frame: on the server
 * engines the masks stream in as frames complete, and the preview plays them
 * — the advancing picture IS the progress bar (SAM 2's pattern). The browser
 * engine keeps its masks private until the run ends, so there the numbers
 * carry the wait. A wake lock keeps a phone from sleeping through it. */
let WLOCK = null;
async function holdWake() {
  try { WLOCK = navigator.wakeLock ? await navigator.wakeLock.request('screen') : null; }
  catch (e) { WLOCK = null; }
}
function releaseWake() {
  try { if (WLOCK) WLOCK.release(); } catch (e) { /* released with the tab */ }
  WLOCK = null;
}

const TSTREAM = { busy: false, at: 0 };
async function paintTrackedFrame(i) {
  if (TSTREAM.busy) return;
  TSTREAM.busy = true;
  try {
    // the frames always exist (extracted up front); a mask may not be on disk
    // yet for this exact index, and then the raw frame still advances
    const frame = await E().frame(i);
    const masks = await Promise.all(S.subjects.map(
      (x) => E().mask(x.id, i).catch(() => null)));
    const cv = $('#vcv');
    if (cv.width !== S.W || cv.height !== S.H) { cv.width = S.W; cv.height = S.H; }
    const g = cv.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.drawImage(frame, 0, 0);
    S.subjects.forEach((x, k) => {
      if (!masks[k]) return;
      g.save(); g.globalAlpha = 0.5;
      g.drawImage(tintedMask(masks[k], x.palette[x.palette.length - 1]),
                  0, 0, S.W, S.H);
      g.restore();
    });
    $('#fcount').textContent = `${i} / ${S.nFrames - 1}`;
    $('#sFrame').value = i;
    frame.close && frame.close();
    masks.forEach((m) => m && m.close && m.close());
  } catch (e) { /* a frame mid-extraction — routine */ }
  TSTREAM.busy = false;
}

async function track() {
  if (S.curPath) commitPath();
  const bad = S.subjects.filter((s) => !hasPrompt(s));
  if (bad.length) { toast('subject #' + bad[0].id + ' has no prompt yet', true); return; }
  while (pvBusy || segBusy) await sleep(60);   // one tracker session, one user
  const btn = $('#bTrack'); btn.disabled = true;
  btn.dataset.running = '1';
  btn.textContent = 'tracking…';
  $('#tinfo').hidden = true; $('#tinfo').textContent = '';
  const prog = $('#prog'); prog.hidden = false;
  const bar = $('.bar i', prog), lab = $('span', prog);
  bar.style.width = '0%';
  lab.textContent = 'loading model…';
  holdWake();
  if (window.DV_sheet && window.matchMedia('(max-width: 767px)').matches) {
    window.DV_sheet('collapsed');
  }
  // masks stream in per frame on the server engines: play them as they land
  const streamed = E().id !== 'browser';
  if (streamed) { stop(); showStage('result'); }
  const t0 = performance.now();
  try {
    const st = await E().track(
      { objects: promptPayload(), imageSize: S.trackSize },
      (p) => {
        bar.style.width = (p.total ? (p.done / p.total) * 100 : 0).toFixed(1) + '%';
        const el = (performance.now() - t0) / 1000;
        const left = (p.done && p.total && el > 2)
          ? (p.total - p.done) * (el / p.done) : 0;
        lab.textContent = (p.text || '')
          + (left > 1 ? ` · ≈ ${fmtDur(left)} left` : '');
        btn.textContent = p.total
          ? `tracking · ${p.done}/${p.total}` : 'tracking…';
        if (streamed && p.done > 1) {
          const now = performance.now();
          if (now - TSTREAM.at > 350) {
            TSTREAM.at = now;
            paintTrackedFrame(Math.min(p.done - 1, S.nFrames - 1));
          }
        }
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
    dropCache(); buildTargets(); renderModes(); renderPolish(); openStep(3);
    DOTS_CACHE = null;
    coach('tracked', 'if the outline slips somewhere, scrub to that frame, '
      + 'add or remove a tap, and Track again');
    paintToSeq();
    if (S.returnToSeq) toast('tracked — "add to the sequence" is in the header');
    if (S.demoRun) { S.demoRun = false; applyDemoLook(); }
    // the finished result plays itself from the top — the reveal, unasked
    await draw(activeRange().in);
    play();
  } catch (err) {
    prog.hidden = true;
    const box = $('#tinfo'); box.hidden = false; box.classList.add('err');
    box.textContent = 'track failed: ' + why(err);
    backToPrompt();
  }
  releaseWake();
  btn.disabled = false;
  delete btn.dataset.running;
  paintTrackCTA();
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
  dropPolish();
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

/* ============================================================ mask polish ===
 * The tracker's masks are per-frame and slightly restless: the outline wobbles
 * a pixel or two, pinholes open and close, and a subject that is barely moving
 * still shimmers. web/polish.js smooths that — temporally, motion-aware, so a
 * ball crossing the frame keeps its own mask while a body gets the whole
 * window — and this is where the browser runs it.
 *
 * Two things matter here beyond calling it:
 *
 *   PARITY   the result is quantised to 8 bits before use, because the server
 *            writes its polished masks as PNGs and the two engines have to
 *            hand the renderer the same numbers. server/polish.py is the same
 *            arithmetic; server/parity.py gates it.
 *   COST     polishing a whole 720p frame per subject per draw would make
 *            scrubbing crawl, so it runs on the union bounding box of the
 *            window's non-zero pixels, padded by more than the filters reach.
 *            Outside that box every mask is exactly zero (they arrive as 8-bit
 *            images), so a padded crop and the whole frame give the same
 *            answer — which is a claim the parity gate also covers.
 */
/* A mask WELL: where raw masks come from, how big they are, and the two LRU
 * caches that make polishing them affordable. The studio has one — the clip on
 * the stage, through the engine — and every sequence item that came from a
 * clip has its own, because an item outlives the clip it was captured from and
 * still has to be able to re-polish at a new strength. */
const PM_RAW_MAX = 64, PM_OUT_MAX = 48;
function newWell(id, cfg) {
  return Object.assign({ id, key: null, raw: new Map(), out: new Map(), pool: [] },
                       cfg);
}
function wellReset(W) { W.raw.clear(); W.out.clear(); }

/* The studio's well: the clip that is open, fetched through the engine. */
const PM = newWell('pm', {
  get: (objId, j) => E().mask(objId, j),
  size: () => ({ w: S.W, h: S.H, n: S.nFrames }),
});

const polishOn = () => S.kind === 'video' && usingSubjects()
  && S.subjects.some((s) => (s.polish | 0) > 0);

function polishKey() {
  return JSON.stringify([E().id, S.job, S.W, S.H,
                         S.subjects.map((s) => [s.id, s.polish | 0])]);
}
function dropPolish() { PM.key = null; wellReset(PM); }
function checkPolishKey() {
  const k = polishKey();
  if (PM.key !== k) { PM.key = k; wellReset(PM); }
}
function lru(map, max) {
  while (map.size > max) { const k = map.keys().next().value; map.delete(k); }
}

/** One subject's raw mask on one frame: 8-bit coverage, its stats and the box
 *  outside which it is exactly zero. Fetched through the well, not through
 *  frameAt — polish wants six neighbouring MASKS, not six neighbouring frames. */
async function rawMask(W, objId, j) {
  const key = objId + ':' + j;
  const hit = W.raw.get(key);
  if (hit) { W.raw.delete(key); W.raw.set(key, hit); return hit; }
  const bmp = await W.get(objId, j);
  const { w, h } = W.size();
  const c = ctx2d(w, h, W.id + 'raw');
  c.clearRect(0, 0, w, h);
  c.drawImage(bmp, 0, 0, w, h);
  if (bmp.close) bmp.close();
  const d = c.getImageData(0, 0, w, h).data;
  const u8 = new Uint8Array(w * h);
  let n = 0, sx = 0, sy = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0, q = 0; y < h; y++) {
    for (let x = 0; x < w; x++, q++) {
      const v = d[q * 4];
      u8[q] = v;
      if (!v) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
      if (v >= 128) { n++; sx += x; sy += y; }   // >= 0.5 in 8-bit terms
    }
  }
  const rec = { u8, box: x1 < 0 ? null : { x0, y0, x1, y1 },
                st: n ? { area: n, cx: sx / n, cy: sy / n }
                      : { area: 0, cx: 0, cy: 0 } };
  W.raw.set(key, rec); lru(W.raw, PM_RAW_MAX);
  return rec;
}

/** A cropped float view of a raw mask, from the well's shared pool. */
function crop(W, rec, box, w, slot) {
  const cw = box.x1 - box.x0 + 1, ch = box.y1 - box.y0 + 1;
  let a = W.pool[slot];
  if (!a || a.length < cw * ch) a = W.pool[slot] = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) {
    const src = (box.y0 + y) * w + box.x0, dst = y * cw;
    for (let x = 0; x < cw; x++) a[dst + x] = rec.u8[src + x] / 255;
  }
  return a.length === cw * ch ? a : a.subarray(0, cw * ch);
}

/** Subject `objId`'s polished mask on frame `i`, as the renderer wants it.
 *  `slot` names the scratch Float32Array it is expanded into — callers that
 *  hold several masks at once must pass different ones. */
async function polishedMask(W, objId, i, strength, slot) {
  const { w, h, n: nF } = W.size();
  const key = objId + ':' + strength + ':' + i;
  const hit = W.out.get(key);
  if (hit) {
    W.out.delete(key); W.out.set(key, hit);
    return expand(hit, w * h, slot);
  }
  const P = MaskPolish.params(strength);
  const lo = Math.max(0, i - P.radius), hi = Math.min(nF - 1, i + P.radius);
  const recs = [];
  for (let j = lo; j <= hi; j++) recs.push(await rawMask(W, objId, j));
  const pad = 2 * P.morph + P.blur + 2;
  let box = null;
  for (const r of recs) {
    if (!r.box) continue;
    box = box ? { x0: Math.min(box.x0, r.box.x0), y0: Math.min(box.y0, r.box.y0),
                  x1: Math.max(box.x1, r.box.x1), y1: Math.max(box.y1, r.box.y1) }
              : Object.assign({}, r.box);
  }
  const out = new Uint8Array(w * h);
  if (box) {
    box = { x0: clamp(box.x0 - pad, 0, w - 1), y0: clamp(box.y0 - pad, 0, h - 1),
            x1: clamp(box.x1 + pad, 0, w - 1), y1: clamp(box.y1 + pad, 0, h - 1) };
    const cw = box.x1 - box.x0 + 1, ch = box.y1 - box.y0 + 1;
    const win = recs.map((r, n) => crop(W, r, box, w, n));
    const st = recs.map((r) => r.st);
    const got = MaskPolish.polishFrame(win, i - lo, cw, ch, strength, st);
    const q = MaskPolish.quantise(got);
    for (let y = 0; y < ch; y++) out.set(q.subarray(y * cw, y * cw + cw),
                                        (box.y0 + y) * w + box.x0);
  }
  W.out.set(key, out); lru(W.out, PM_OUT_MAX);
  return expand(out, w * h, slot);
}

const EXP = {};
function expand(u8, n, slot) {
  let a = EXP[slot];
  if (!a || a.length !== n) a = EXP[slot] = new Float32Array(n);
  for (let q = 0; q < n; q++) a[q] = u8[q] / 255;
  return a;
}

/** The masks one frame is composed with — polished where a subject asks for
 *  it. Preview and export both come through here, which is what makes the two
 *  the same picture. */
async function masksFor(i, rec, slot) {
  if (!usingSubjects()) return [];
  if (S.kind !== 'video') return stillMasksAt(S.W, S.H, slot);
  if (!polishOn()) return rec.masks.map((m, k) => bitmapAlpha(m, S.W, S.H, slot + k));
  checkPolishKey();
  const out = [];
  for (let k = 0; k < S.subjects.length; k++) {
    out.push((S.subjects[k].polish | 0) > 0
      ? await polishedMask(PM, S.subjects[k].id, i, S.subjects[k].polish | 0,
                           'pf' + k)
      : bitmapAlpha(rec.masks[k], S.W, S.H, slot + k));
  }
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
  // the jitter that keeps a dot cloud from looking like graph paper. At cell 1
  // there is no room for it and no need: one cell is one pixel, and jittering
  // it by a pixel would turn a Bayer screen into noise. server/render.py has
  // the same line.
  const jit = cell > 1 ? 0.8 : 0;
  for (let i = 0; i < gh; i++) for (let j = 0; j < gw; j++) {
    const q = i * gw + j;
    thr[q] = tile[(i % 64) * 64 + (j % 64)];
    cx[q] = j * cell + cell / 2 + (Dither.hash01(i, j, 1, seed) - 0.5) * cell * jit;
    cy[q] = i * cell + cell / 2 + (Dither.hash01(i, j, 2, seed) - 0.5) * cell * jit;
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

/* ------------------------------------------------- the modes, on the dot grid
 * A sequence is dot positions and nothing else, so "which cells are lit" is
 * the only question a look has to answer, and every dither mode can answer it.
 *
 *   dots       answers it the way the studio's dots look does: blue noise, a
 *              target count, a fill ratio, stray dots in a halo around the
 *              subject. That is `dotsOn` above, unchanged.
 *   everything the per-cell tone field becomes a gw x gh greyscale image — one
 *   else       pixel per cell — and goes through web/dither.js exactly as a
 *              picture would, black on white, the subject's coverage as the
 *              gate. A cell is lit where that comes back black.
 *
 * So Bayer, halftone, blue noise, white noise, error diffusion and Riemersma
 * all produce clouds that morph like any other: they are dots on the same
 * grid, they just disagree about which ones survive. Error diffusion and
 * Riemersma flicker frame to frame on a clip exactly as they do in the studio
 * — the mode chips say so — but they morph, so nothing is greyed out.
 *
 * `n`, `fill`, `stray` and `halo` belong to the dots renderer alone; the
 * inspector hides them for the other modes rather than pretending.
 */
const MODE_PAL = ['#ffffff', '#000000'];
function dotsOnMode(srcData, W, H, masks, P, tile) {
  if (!P.mode || P.mode === 'dots') return dotsOn(srcData, W, H, masks, P, tile);
  const cell = P.cell | 0;
  const F = dotFields(W, H, cell, P.seed, tile);
  const { gw, gh, N } = F;
  const K = masks.length;
  const lum = new Float32Array(N), cov = [];
  for (let k = 0; k < K; k++) cov.push(new Float32Array(N));
  const useH = gh * cell, useW = gw * cell;
  for (let y = 0; y < useH; y++) {
    const row = ((y / cell) | 0) * gw, base = y * W * 4, mrow = y * W;
    for (let x = 0; x < useW; x++) {
      const p = base + x * 4, q = row + ((x / cell) | 0);
      lum[q] += (0.2126 * srcData[p] + 0.7152 * srcData[p + 1]
                 + 0.0722 * srcData[p + 2]) / 255;
      for (let k = 0; k < K; k++) cov[k][q] += masks[k][mrow + x];
    }
  }
  const cc = cell * cell;
  const img = new Uint8ClampedArray(N * 4);
  for (let q = 0; q < N; q++) {
    const v = Math.round(clamp(lum[q] / cc, 0, 1) * 255);
    img[q * 4] = v; img[q * 4 + 1] = v; img[q * 4 + 2] = v; img[q * 4 + 3] = 255;
  }
  // one owner per cell, as in dotsOn: two subjects never light the same dot
  const owner = new Int8Array(N).fill(-1);
  for (let q = 0; q < N; q++) {
    let bv = 0, best = -1;
    for (let k = 0; k < K; k++) if (cov[k][q] > bv) { bv = cov[k][q]; best = k; }
    owner[q] = best;
  }
  const p = { mode: P.mode, algo: P.algo || 'floyd-steinberg',
              matrix: P.matrix || 4, serpentine: !!P.serpentine,
              strength: P.strength === undefined ? 1 : P.strength,
              seed: P.seed, brightness: 0, contrast: 1,
              gamma: P.gamma, invert: P.invert, palette: MODE_PAL };
  const out = new Uint8ClampedArray(N * 4);
  const gate = new Float32Array(N);
  const ons = [];
  for (let k = 0; k < K; k++) {
    for (let q = 0; q < N; q++) gate[q] = owner[q] === k ? 1 : 0;
    out.fill(255);
    Dither.ditherRGBA(img, out, gw, gh, p, gate);
    const on = new Uint8Array(N);
    for (let q = 0; q < N; q++) if (gate[q] > 0 && out[q * 4] < 128) on[q] = 1;
    ons.push(on);
  }
  return { F, on: ons };
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

/* ================================================== the canvas (aspect) ===
 * Everything above renders at the size of what came in. This is the part that
 * says "no, 1080x1920" — and it is deliberately one affine map (web/canvas.js)
 * applied in one place, because the same three numbers have to reach the
 * preview, the export, the matched original cut and the .dots.gz or the four
 * of them stop being the same picture.
 *
 * Two behaviours, chosen by what is on screen rather than by a mode switch:
 *
 *   cutout      the background is flat, so nothing can fall off the edge. The
 *               crop is NOT clamped to the source and the subject sits where
 *               the path puts it; the dots are re-measured on the canvas, so a
 *               9:16 export is 1080x1920 of real dots and not an upscale.
 *   overlay /   the footage is visible, so the crop window is clamped inside
 *   whole-frame the source and there is no zoom by default: the crop is the
 *               largest rectangle of the target aspect that fits, scaled up to
 *               the target's pixels. That upscale is real and the UI says so.
 */
const CPATH = { key: null, centers: null, union: null, n: 0 };

/** The source's own pixel size for the flow that is open. */
function srcSize() {
  if (S.kind === 'image') return [S.natW || 1, S.natH || 1];
  return [S.W || 1, S.H || 1];
}

/** The output size, or null when the canvas is the source's own.
 *
 *  `CANVAS_OFF` is how the SEQUENCE captures: an item is the material, at the
 *  frame it was tracked in, and the strip has a frame size of its own — so a
 *  studio set to 9:16 must not bake 9:16 into the dot cloud it hands over. */
let CANVAS_OFF = 0;
async function withoutCanvas(fn) {
  CANVAS_OFF++;
  try { return await fn(); } finally { CANVAS_OFF--; }
}
function canvasTarget() {
  if (CANVAS_OFF) return null;
  const [sw, sh] = srcSize();
  return CV.targetSize(S.canvas, sw, sh);
}
const canvasOn = () => !!canvasTarget();

/** Whether the crop has to stay inside the source: true wherever real footage
 *  is visible, false for a cutout, which has nothing behind it to run out of. */
const canvasClamps = () => !(usingSubjects() && S.P.compose !== 'overlay');

/** The framing actually in force. 'auto' answers itself, and answers it
 *  against the CROP THAT IS SET rather than once and for all: a subject that
 *  never leaves a 16:9 frame may well leave a 1:1 one. Cheap enough to ask on
 *  every paint — it is one rectangle against another. */
function framing() {
  const f = S.canvas.follow;
  if (f === 'follow' || f === 'static') return f;
  if (!CPATH.union || !CPATH.centers || CPATH.centers.length < 2) return 'static';
  const [sw, sh] = srcSize();
  const t = canvasTarget() || { w: sw, h: sh };
  const crop = CV.cropRect(sw, sh, t.w, t.h, S.canvas.zoom);
  const u = CPATH.union;
  return CV.fitsStatic({ x0: u.x0 * sw, y0: u.y0 * sh, x1: u.x1 * sw, y1: u.y1 * sh },
                       crop) ? 'static' : 'follow';
}

/* Which path we are holding. The centres are NORMALISED (0..1 of the source),
 * so one path serves the preview, a still's native-resolution export and the
 * server, none of which are looking at the same pixel grid. */
function pathKey() {
  return JSON.stringify([S.kind, E() && E().id, S.job, S.nFrames,
                         // a still has no job id to change, so it is identified
                         // by the file itself — two photographs dropped one
                         // after the other must not share a crop centre
                         S.kind === 'image' ? [S.fileName, S.natW, S.natH] : 0,
                         usingSubjects() ? S.subjects.map((s) => s.id) : [],
                         S.kind === 'image' ? S.stillMasks.size : 0]);
}

/** One frame's subject centroid, as a fraction of the frame, from the masks
 *  themselves. Small on purpose: a centroid does not need 720p. */
const PATH_W = 160;
function centroidFromBitmaps(bmps, sw, sh) {
  const w = PATH_W, h = Math.max(1, Math.round(w * sh / sw));
  let n = 0, sx = 0, sy = 0, x0 = w, y0 = h, x1 = -1, y1 = -1;
  bmps.forEach((bmp, k) => {
    const c = ctx2d(w, h, 'cp' + (k & 3));
    c.clearRect(0, 0, w, h);
    c.drawImage(bmp, 0, 0, w, h);
    const d = c.getImageData(0, 0, w, h).data;
    for (let y = 0, q = 0; y < h; y++) {
      for (let x = 0; x < w; x++, q++) {
        if (d[q * 4] < 128) continue;
        n++; sx += x; sy += y;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  });
  if (!n) return { ok: false, x: 0.5, y: 0.5, box: null };
  return { ok: true, x: (sx / n + 0.5) / w, y: (sy / n + 0.5) / h,
           box: { x0: x0 / w, y0: y0 / h, x1: (x1 + 1) / w, y1: (y1 + 1) / h } };
}

/** Build (or reuse) the per-frame crop centres for whatever is open.
 *
 *  A tracked clip walks its masks once — through the server's own arithmetic
 *  where there is a server, because 900 mask PNGs over HTTP to work out 900
 *  centres is silly — smooths the result over +/-15 frames, and remembers the
 *  union box so "auto" can answer the hold-still question. Everything else
 *  (a whole-frame clip, a still with no subject) is one fixed centre. */
async function ensureCanvasPath(onProgress) {
  const key = pathKey();
  if (CPATH.key === key) return CPATH;
  const centre = { ok: true, x: 0.5, y: 0.5, box: null };
  if (S.kind === 'image') {
    const ms = usingSubjects()
      ? S.subjects.map((s) => S.stillMasks.get(s.id)).filter(Boolean) : [];
    const c = ms.length ? centroidFromBitmaps(ms, S.W || 1, S.H || 1) : centre;
    Object.assign(CPATH, { key, centers: [[c.x, c.y]], union: c.box, n: 1 });
    return CPATH;
  }
  if (S.kind !== 'video' || !usingSubjects() || !S.tracked) {
    Object.assign(CPATH, { key, centers: [[0.5, 0.5]], union: null,
                           n: S.nFrames || 1 });
    return CPATH;
  }
  const n = S.nFrames;
  const ids = S.subjects.map((s) => s.id);
  let raw = null;
  if (E().centroids) {
    try { raw = await E().centroids(ids); } catch (err) { raw = null; }
  }
  if (!raw) {
    raw = [];
    for (let i = 0; i < n; i++) {
      const bmps = await Promise.all(ids.map((id) => E().mask(id, i)));
      raw.push(centroidFromBitmaps(bmps, S.W, S.H));
      bmps.forEach((b) => b.close && b.close());
      if (onProgress && (i & 7) === 0) {
        onProgress({ done: i + 1, total: n, text: `framing ${i + 1}/${n}` });
      }
      if ((i & 7) === 7) await sleep(0);
    }
  }
  const sm = CV.smoothPath(raw, 15);
  let union = null;
  raw.forEach((p) => {
    if (!p.ok || !p.box) return;
    union = union ? { x0: Math.min(union.x0, p.box.x0), y0: Math.min(union.y0, p.box.y0),
                      x1: Math.max(union.x1, p.box.x1), y1: Math.max(union.y1, p.box.y1) }
                  : Object.assign({}, p.box);
  });
  Object.assign(CPATH, { key, n, centers: sm.map((p) => [p.x, p.y]), union });
  return CPATH;
}

/** The centre the crop sits on for frame `i`, in source pixels. Falls back to
 *  the frame's own centre until the path has been built. */
function centreAt(i, sw, sh) {
  const p = CPATH.centers;
  if (!p || !p.length) return { x: sw / 2, y: sh / 2 };
  if (framing() === 'static' || p.length === 1) {
    if (CPATH.union && framing() === 'static' && p.length > 1) {
      return { x: (CPATH.union.x0 + CPATH.union.x1) / 2 * sw,
               y: (CPATH.union.y0 + CPATH.union.y1) / 2 * sh };
    }
    const c = p[0];
    return { x: c[0] * sw, y: c[1] * sh };
  }
  const c = p[clamp(i | 0, 0, p.length - 1)];
  return { x: c[0] * sw, y: c[1] * sh };
}

/** The affine map for one frame, or null when there is no canvas. */
function canvasPlanAt(i, sw, sh) {
  const t = canvasTarget();
  if (!t) return null;
  const c = centreAt(i, sw, sh);
  const plan = CV.place({ sw, sh, tw: t.w, th: t.h,
                          cx: c.x + S.canvas.dx * sw,
                          cy: c.y + S.canvas.dy * sh,
                          zoom: S.canvas.zoom, clamp: canvasClamps() });
  return Object.assign(plan, { tw: t.w, th: t.h, overlay: canvasClamps() });
}

/* Scratch coverage buffers for the warped masks — one per subject slot. */
const WBUF = {};
function wbuf(n, slot) {
  let a = WBUF[slot];
  if (!a || a.length !== n) a = WBUF[slot] = new Float32Array(n);
  return a;
}

/** Source pixels and masks, both mapped onto the canvas. `drawSrc` paints the
 *  source at its own coordinates; the transform does the rest. */
function onCanvas(drawSrc, sw, sh, masks, plan, slot) {
  const { tw, th } = plan;
  const c = ctx2d(tw, th, slot);
  c.setTransform(1, 0, 0, 1, 0, 0);
  c.clearRect(0, 0, tw, th);
  if (plan.overlay) { c.fillStyle = '#000'; c.fillRect(0, 0, tw, th); }
  c.imageSmoothingQuality = 'high';
  c.setTransform(plan.k, 0, 0, plan.k, plan.x0, plan.y0);
  drawSrc(c);
  c.setTransform(1, 0, 0, 1, 0, 0);
  const src = c.getImageData(0, 0, tw, th).data;
  const out = masks.map((m, k) => CV.warpMask(m, sw, sh, wbuf(tw * th, slot + k),
                                              tw, th, plan, CV.maskBox(m, sw, sh)));
  return { src, masks: out, W: tw, H: th };
}

/** What the export ships to the server: the map itself, already worked out,
 *  one entry per frame of the window. Nothing about HOW the crop was chosen
 *  crosses the wire — only where it ended up — so server/render.py has no
 *  second opinion about following, clamping or smoothing to drift from. */
function canvasPayload(rng) {
  const t = canvasTarget();
  if (!t) return null;
  const [sw, sh] = srcSize();
  const r = rng || activeRange();
  const place = [];
  if (S.kind === 'image' || framing() === 'static') {
    const p = canvasPlanAt(0, sw, sh);
    place.push([+p.x0.toFixed(3), +p.y0.toFixed(3)]);
    return { w: t.w, h: t.h, k: +p.k.toFixed(6), place };
  }
  let k = 1;
  for (let i = r.in; i <= r.out; i++) {
    const p = canvasPlanAt(i, sw, sh);
    k = p.k;
    place.push([+p.x0.toFixed(3), +p.y0.toFixed(3)]);
  }
  return { w: t.w, h: t.h, k: +k.toFixed(6), place };
}

/* ------------------------------------------------------------ the controls */
/** '1080×1920 · 9:16' — what the export is about to be. */
function canvasLabel() {
  const t = canvasTarget();
  if (!t) return 'source';
  const p = CV.presetOf(S.canvas.preset);
  return `${t.w}×${t.h}` + (p.id === 'custom' ? '' : ` · ${p.label}`);
}
/** The bit that goes in a filename: 9:16 is not a legal one. */
function canvasSlug() {
  const t = canvasTarget();
  if (!t) return '';
  return S.canvas.preset === 'custom' ? `${t.w}x${t.h}`
    : S.canvas.preset.replace(':', 'x');
}

const canvasOffered = () => S.kind === 'image' || S.kind === 'video';

function buildCanvasPresets() {
  const wrap = $('#canvaspresets');
  if (!wrap || wrap.childElementCount) return;
  CV.PRESETS.forEach((p) => {
    const b = document.createElement('button');
    b.className = 'chip'; b.dataset.preset = p.id; b.textContent = p.label;
    b.title = p.note;
    b.addEventListener('click', () => setCanvasPreset(p.id));
    wrap.append(b);
  });
}

function paintCanvasUI() {
  const box = $('#canvasui');
  if (!box) return;
  box.hidden = !canvasOffered();
  if (box.hidden) return;
  buildCanvasPresets();
  $$('#canvaspresets .chip').forEach((b) => b.setAttribute(
    'aria-pressed', String(b.dataset.preset === S.canvas.preset)));
  $('#canvascustom').hidden = S.canvas.preset !== 'custom';
  const t = canvasTarget();
  $('#vCanvas').textContent = canvasLabel();
  $('#canvasopts').hidden = !t;
  $('#cvhint').hidden = !t || S.kind === 'none';
  $('#vstage').dataset.canvas = t ? '1' : '0';
  $$('#framing .chip').forEach((b) => b.setAttribute(
    'aria-pressed', String(b.dataset.framing === S.canvas.follow)));
  $('#vFraming').textContent = S.canvas.follow === 'auto'
    ? 'auto · ' + framing() : framing();
  $('#sZoom').value = String(S.canvas.zoom);
  $('#vZoom').textContent = S.canvas.zoom.toFixed(2) + '×';
  const note = $('#canvasnote');
  if (!t) {
    note.textContent = 'The export is whatever came in. Pick a shape and both '
      + 'the render and the matched original cut come out at exactly that size.';
    note.classList.remove('warn');
    return;
  }
  const [sw, sh] = srcSize();
  const plan = canvasPlanAt(S.cur | 0, sw, sh);
  const up = plan ? plan.k : 1;
  const cropping = canvasClamps();       // real footage: a crop, and an upscale
  const lines = [`${t.w}×${t.h}.`];
  if (cropping) {
    lines.push(`A ${(t.w / up).toFixed(0)}×${(t.h / up).toFixed(0)} crop of the `
      + `${sw}×${sh} source, scaled ${up.toFixed(2)}× to fill it`
      + (up > 1.05 ? ' — real footage, really upscaled.' : '.'));
    if (S.kind === 'video' && usingSubjects()) {
      lines.push(framing() === 'follow'
        ? 'The crop follows the tracked subject, smoothed over ±15 frames.'
        : 'The crop holds still on the subject’s whole-clip box; the subject '
          + 'moves inside it.');
    } else {
      lines.push('Nothing is tracked here, so the crop is centred. Drag the '
        + 'picture to move it.');
    }
  } else {
    lines.push('A cutout has no background to run out of, so the dots are '
      + 'measured on the canvas itself: they come out at '
      + `${t.w}×${t.h}, crisp, not scaled up. Dot size stays ${S.P.dotpx} px `
      + 'of the OUTPUT.');
    if (S.kind === 'video' && usingSubjects()) {
      lines.push(framing() === 'follow'
        ? 'The frame follows the subject, smoothed over ±15 frames.'
        : 'The frame holds still; the subject moves inside it.');
    }
  }
  if (S.canvas.dx || S.canvas.dy) lines.push('Nudged by hand — “recentre” undoes it.');
  note.textContent = lines.join(' ');
  note.classList.toggle('warn', cropping && up > 1.6);
}

/** Change the shape. The path is rebuilt (it decides follow-or-hold against
 *  the new crop) and the picture redrawn — asynchronously, because a tracked
 *  clip's path is a walk over its masks. */
async function applyCanvas() {
  paintCanvasUI();
  if (canvasOn()) {
    try { await ensureCanvasPath(); } catch (err) { /* the fallback centre */ }
  }
  paintCanvasUI();                        // the framing answer may have moved
  await draw();
  return canvasTarget();
}

function setCanvasPreset(id) {
  S.canvas.preset = id;
  S.canvas.dx = 0; S.canvas.dy = 0;
  return applyCanvas();
}

$('#sZoom') && $('#sZoom').addEventListener('input', (e) => {
  S.canvas.zoom = +e.target.value;
  applyCanvas();
});
$$('#framing .chip').forEach((b) => b.addEventListener('click', () => {
  S.canvas.follow = b.dataset.framing;
  applyCanvas();
}));
$('#bCanvasCentre') && $('#bCanvasCentre').addEventListener('click', () => {
  S.canvas.dx = 0; S.canvas.dy = 0;
  applyCanvas();
});
['#cvW', '#cvH'].forEach((sel) => {
  const el = $(sel);
  if (!el) return;
  el.addEventListener('change', () => {
    S.canvas.w = Math.max(16, +$('#cvW').value || 1080);
    S.canvas.h = Math.max(16, +$('#cvH').value || 1920);
    applyCanvas();
  });
});

/* Drag the picture to move the frame. The bias is stored as a fraction of the
 * source, so it survives a switch between the preview's pixels and the
 * export's, and it is ADDED to the smoothed path rather than replacing it —
 * a followed subject stays followed, just off to one side. */
let CVDRAG = null;
$('#vcv').addEventListener('pointerdown', (e) => {
  if (!canvasOn() || S.compare) return;
  const [sw, sh] = srcSize();
  const plan = canvasPlanAt(S.cur | 0, sw, sh);
  if (!plan) return;
  const r = $('#vcv').getBoundingClientRect();
  CVDRAG = { x: e.clientX, y: e.clientY, dx: S.canvas.dx, dy: S.canvas.dy,
             // client px -> canvas px -> source px -> fraction of the source
             sx: (plan.tw / r.width) / plan.k / sw,
             sy: (plan.th / r.height) / plan.k / sh };
  $('#vcv').setPointerCapture(e.pointerId);
  e.preventDefault();
});
$('#vcv').addEventListener('pointermove', (e) => {
  if (!CVDRAG) return;
  S.canvas.dx = CVDRAG.dx - (e.clientX - CVDRAG.x) * CVDRAG.sx;
  S.canvas.dy = CVDRAG.dy - (e.clientY - CVDRAG.y) * CVDRAG.sy;
  paintCanvasUI();
  if (!S.playing) draw();
});
const cvDragEnd = () => { CVDRAG = null; };
$('#vcv').addEventListener('pointerup', cvDragEnd);
$('#vcv').addEventListener('pointercancel', cvDragEnd);

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
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.putImageData(new ImageData(out, W, H), 0, 0);
  if (S.compare && opts && opts.original) {
    const x = Math.round(clamp(S.split, 0, 1) * W);
    if (x > 0) {
      g.save(); g.beginPath(); g.rect(0, 0, x, H); g.clip();
      // the "before" half is the same frame on the same canvas: a wipe that
      // compared a cropped render against an uncropped source would be
      // comparing two different pictures
      const pl = opts.plan;
      if (pl) {
        g.setTransform(pl.k, 0, 0, pl.k, pl.x0, pl.y0);
        g.drawImage(opts.original, 0, 0, opts.srcW, opts.srcH);
      } else {
        g.drawImage(opts.original, 0, 0, W, H);
      }
      g.restore();
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
    let src = c.getImageData(0, 0, w, h).data;
    if (seq !== drawSeq) return;
    let masks = usingSubjects() ? stillMasksAt(w, h) : [];
    let W = w, H = h;
    const plan = canvasPlanAt(0, w, h);
    if (plan) {
      const on = onCanvas((g) => g.drawImage(S.bitmap, 0, 0, w, h), w, h,
                          masks, plan, 'cvs');
      src = on.src; masks = on.masks; W = on.W; H = on.H;
    }
    const r = paint($('#vcv'), src, W, H, masks,
                    { original: S.bitmap, plan, srcW: w, srcH: h });
    $('#fps').textContent = `${W}×${H} · ${r.ms.toFixed(0)} ms`
      + (S.P.mode === 'dots' ? ` · ${r.lit} dots` : '');
    paintCanvasUI();
    scheduleLookThumbs();
    return;
  }
  if (S.kind !== 'video') return;
  const idx = i === undefined ? S.cur : i;
  const bmp = await frameAt(idx);
  if (seq !== drawSeq) return;
  const c = ctx2d(S.W, S.H, 'src');
  c.drawImage(bmp.frame, 0, 0);
  let src = c.getImageData(0, 0, S.W, S.H).data;
  let masks = await masksFor(idx, bmp, 'm');
  if (seq !== drawSeq) return;
  let W = S.W, H = S.H;
  const plan = canvasPlanAt(idx, S.W, S.H);
  if (plan) {
    const on = onCanvas((g) => g.drawImage(bmp.frame, 0, 0), S.W, S.H,
                        masks, plan, 'cvs');
    src = on.src; masks = on.masks; W = on.W; H = on.H;
  }
  const r = paint($('#vcv'), src, W, H, masks,
                  { original: bmp.frame, plan, srcW: S.W, srcH: S.H });
  S.cur = idx;
  $('#fcount').textContent = `${idx} / ${S.nFrames - 1}`;
  $('#sFrame').value = idx;
  paintRange();
  $('#fps').textContent = `${(1000 / Math.max(r.ms, 0.01)).toFixed(1)} fps`
    + (S.P.mode === 'dots' ? ` · ${r.lit} dots` : '');
  scheduleLookThumbs();
  return r.lit;
}

/* ------------------------------------------------------------- transport */
/** The active range, stated beside the frame counter. `full clip ↺` only
 *  appears when there is something to go back to. */
function paintRange() {
  const lbl = $('#rangelbl'), btn = $('#bRangeAll');
  if (!lbl || !btn) return;
  if (S.kind !== 'video' || !S.nFrames) {
    lbl.hidden = true; btn.hidden = true; return;
  }
  const r = activeRange();
  lbl.hidden = false;
  lbl.textContent = `frames ${r.in}–${r.out} of ${S.nFrames}`;
  lbl.classList.toggle('on', !r.whole);
  btn.hidden = r.whole;
}
$('#bRangeAll').addEventListener('click', () => {
  if (S.kind !== 'video' || !S.nFrames) return;
  setRange(0, S.nFrames - 1);
  S.trim = { start: jobWindow().start, end: jobWindow().end };
  paintTrim();
  toast('the whole tracked clip again');
});

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
    // the preview plays the range, not the clip -- that is what makes a trim
    // after the track something you can watch before you export it
    const r = activeRange();
    const next = (S.cur + 1 > r.out || S.cur + 1 < r.in) ? r.in : S.cur + 1;
    for (let k = 1; k <= 3; k++) {
      frameAt(next + k > r.out ? r.in + ((next + k - r.in) % r.n) : next + k);
    }
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
  const dev = $('#devexp');
  if (dev) dev.hidden = !(id === 'dots' && dotsAvailable());
  const ed = id === 'errordiff';
  $('#edui').hidden = !ed;
  $('#mxui').hidden = !(id === 'ordered' || id === 'halftone');
  $('#dotsui').hidden = id !== 'dots';
  const adv = $('#advui');
  if (adv) adv.hidden = id !== 'dots';
  $('#pxui').hidden = id === 'dots';
  renderPolish();
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

/* ---------------------------------------------------------- mask polish UI
 * One row per subject: a toggle and a strength. Video only — a still has one
 * frame, and everything temporal about this needs neighbours. Default off,
 * because polish is a decision about a subject ("this is a body, steady it")
 * and not a global improvement. */
function renderPolish() {
  const box = $('#polui');
  if (!box) return;
  const show = S.kind === 'video' && usingSubjects();
  box.hidden = !show;
  if (!show) return;
  const wrap = $('#pollist');
  wrap.textContent = '';
  S.subjects.forEach((s, k) => {
    const on = (s.polish | 0) > 0;
    const row = document.createElement('div');
    row.className = 'mini';
    const t = document.createElement('button');
    t.className = 'chip pol';
    t.setAttribute('aria-pressed', String(on));
    t.title = on ? 'turn polish off for this subject' : 'polish this subject';
    const sw = document.createElement('span');
    sw.className = 'sw'; sw.style.background = s.palette[s.palette.length - 1];
    const nm = document.createElement('span');
    nm.textContent = '#' + s.id;
    t.append(sw, nm);
    const sl = document.createElement('input');
    sl.type = 'range'; sl.min = '10'; sl.max = '100'; sl.step = '5';
    sl.value = String(on ? (s.polish | 0) : 70);
    sl.disabled = !on;
    const v = document.createElement('b');
    v.textContent = on ? String(s.polish | 0) : 'off';
    t.addEventListener('click', () => setPolish(k, on ? 0 : (+sl.value || 70)));
    sl.addEventListener('input', () => setPolish(k, +sl.value));
    row.append(t, sl, v);
    wrap.append(row);
  });
  const lit = S.subjects.filter((s) => (s.polish | 0) > 0);
  $('#vPol').textContent = lit.length
    ? `${lit.length}/${S.subjects.length} on` : 'off';
}

function setPolish(k, v) {
  const s = S.subjects[k];
  if (!s) return;
  s.polish = clamp(v | 0, 0, 100);
  dropPolish();
  DOTS_CACHE = null;
  renderPolish();
  draw();
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
      setMode(m.id); markCustom(); draw();
    });
    wrap.append(b);
  });
  paintLookRow();
}

/* ================================================= look presets ==========
 * A look is a whole answer — style + colours + the dot dials — applied in one
 * tap. Presets are pure parameter dictionaries over the existing engine: no
 * new render path, and every slider underneath stays live. Touching one flips
 * the row to "Custom"; the tiles are small live renders of the actual frame.
 */
const LOOK_BASE = {
  algo: 'floyd-steinberg', matrix: 4, serpentine: false, strength: 1,
  brightness: 0, contrast: 1, gamma: 1, invert: false, pixel: 1,
  n: 8000, cell: 4, dotpx: 3, fill: 0.7, stray: 0.02, band: 9,
};
const LOOKS = [
  { id: 'solvd', name: 'Solvd',
    P: { mode: 'dots', n: 9000, cell: 4, dotpx: 3, fill: 0.7, stray: 0.02, band: 9 },
    bg: '#c9d4c5', ink: ['#0f1f18', '#b0413e'], palette: ['#c9d4c5', '#0f1f18'] },
  { id: 'newsprint', name: 'Newsprint',
    P: { mode: 'halftone', matrix: 8, pixel: 2 },
    bg: '#f6ece2', ink: ['#1c1b18'], palette: ['#f6ece2', '#1c1b18'] },
  { id: 'gameboy', name: 'Game Boy',
    P: { mode: 'ordered', matrix: 4, pixel: 3 },
    bg: '#9bbc0f', ink: ['#0f380f'],
    palette: ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'] },
  { id: 'blueprint', name: 'Blueprint',
    P: { mode: 'ordered', matrix: 8, pixel: 2 },
    bg: '#10214b', ink: ['#dce8ff'], palette: ['#10214b', '#3b5bbf', '#dce8ff'] },
  { id: 'ember', name: 'Ember',
    P: { mode: 'bluenoise', pixel: 2 },
    bg: '#e8804a', ink: ['#f6ece2'], palette: ['#e8804a', '#f6ece2'] },
  { id: 'ghost', name: 'Ghost',
    P: { mode: 'dots', n: 5000, cell: 5, dotpx: 2, fill: 0.6, stray: 0.05, band: 14 },
    bg: '#0f1f18', ink: ['#e8efe6'], palette: ['#0f1f18', '#e8efe6'] },
  { id: 'comic', name: 'Comic',
    P: { mode: 'errordiff', algo: 'atkinson', pixel: 2 },
    bg: '#ffffff', ink: ['#d02f26'], palette: ['#000000', '#ffffff', '#d02f26'] },
  { id: 'terminal', name: 'Terminal',
    P: { mode: 'ordered', matrix: 4, pixel: 2 },
    bg: '#001a05', ink: ['#2dff6a'], palette: ['#001a05', '#2dff6a'] },
  { id: 'film', name: 'Film grain',
    P: { mode: 'whitenoise' },
    bg: '#000000', ink: ['#ffffff'], palette: ['#000000', '#555555', '#aaaaaa', '#ffffff'] },
];
let APPLYING_LOOK = false;

/** The 30-second demo script's payoff: the Solvd look, applied for you. */
function applyDemoLook() {
  const solvd = LOOKS[0];
  try { applyLook(solvd); } catch (e) { /* the preset row is still there */ }
  toast('the Solvd look — everything in Look updates live');
}

/** Any manual change to a look control makes the look "Custom". */
function markCustom() {
  if (APPLYING_LOOK || S.lookPreset === 'custom') return;
  S.lookPreset = 'custom';
  paintLookRow();
}

/** Write every look control's DOM state from S.P — one place, so a preset and
 *  a verifier-driven change repaint the same way. */
function syncLookUI() {
  const set = (id, v, out, txt) => {
    const el = $(id); if (el) el.value = String(v);
    if (out) $(out).textContent = txt;
  };
  set('#sN', S.P.n, '#vN', String(S.P.n));
  set('#sCell', S.P.cell, '#vCell', S.P.cell + ' px');
  set('#sDot', S.P.dotpx, '#vDot', S.P.dotpx + ' px');
  set('#sFill', S.P.fill, '#vFill', S.P.fill.toFixed(2));
  set('#sStray', S.P.stray, '#vStray', S.P.stray.toFixed(3));
  set('#sBand', S.P.band, '#vBand', String(S.P.band));
  set('#sStr', S.P.strength, '#vStr', S.P.strength.toFixed(2));
  set('#sPx', S.P.pixel, '#vPx', S.P.pixel + '×');
  set('#sBri', S.P.brightness, '#vBri', S.P.brightness.toFixed(2));
  set('#sCon', S.P.contrast, '#vCon', S.P.contrast.toFixed(2));
  set('#sGam', S.P.gamma, '#vGam', S.P.gamma.toFixed(2));
  $('#tInv').setAttribute('aria-pressed', String(!!S.P.invert));
  $('#tSerp').setAttribute('aria-pressed', String(!!S.P.serpentine));
  const alg = $('#sAlgo'); if (alg) alg.value = S.P.algo;
  $$('[data-mx]').forEach((o) => o.setAttribute('aria-pressed',
    String(+o.dataset.mx === S.P.matrix)));
}

function applyLook(l) {
  if (l.P.mode === 'dots' && !dotsAvailable()) {
    toast('on a clip the dots look needs a tracked subject', true);
    return;
  }
  APPLYING_LOOK = true;
  try {
    Object.assign(S.P, LOOK_BASE, l.P);
    S.bg = l.bg; const cbg = $('#cBg'); if (cbg) cbg.value = l.bg;
    S.paletteTouched = true; S.dotsTuned = true;
    S.palette = l.palette.slice();
    S.subjects.forEach((s, i) => { s.palette = [l.bg, l.ink[i % l.ink.length]]; });
    S.lookPreset = l.id;
    syncLookUI();
    setMode(S.P.mode);
    renderSwatches(); renderSubjects(); buildTargets(); renderPolish();
    DOTS_CACHE = null;
    draw();
  } finally { APPLYING_LOOK = false; }
  paintLookRow();
}

/** A tiny render of the current frame under one preset — a live thumbnail. */
const LT = { w: 72, h: 40, src: null, at: 0 };
async function lookThumbSource() {
  const { w, h } = LT;
  const c = ctx2d(w, h, 'lt');
  c.clearRect(0, 0, w, h);
  if (S.kind === 'image' && S.bitmap) c.drawImage(S.bitmap, 0, 0, w, h);
  else if (S.kind === 'video' && S.nFrames) {
    try {
      const rec = await frameAt(S.cur | 0);
      c.drawImage(rec.frame, 0, 0, w, h);
    } catch (e) { return null; }
  } else {
    // no source yet: a soft diagonal ramp, so the tiles still show their looks
    const g = c.createLinearGradient(0, 0, w, h);
    g.addColorStop(0, '#e8e8e8'); g.addColorStop(0.5, '#777');
    g.addColorStop(1, '#111');
    c.fillStyle = g; c.fillRect(0, 0, w, h);
    c.fillStyle = '#ddd'; c.beginPath();
    c.arc(w * 0.62, h * 0.45, h * 0.3, 0, Math.PI * 2); c.fill();
  }
  return c.getImageData(0, 0, w, h);
}

function renderLookThumb(cv, l, srcData) {
  const { w, h } = LT;
  if (cv.width !== w) { cv.width = w; cv.height = h; }
  const g = cv.getContext('2d');
  if (!srcData || !BLUE) { g.fillStyle = l.bg; g.fillRect(0, 0, w, h); return; }
  const P = Object.assign({}, LOOK_BASE, l.P,
                          { compose: 'cutout', seed: S.P.seed, pixel: 1 });
  let out;
  if (P.mode === 'dots') {
    P.n = 700; P.cell = 2; P.dotpx = 1;
    out = renderDots(srcData.data, w, h, [fullMask(w, h)], P,
                     [l.palette, [l.bg, l.ink[0]]], l.bg, BLUE).out;
  } else {
    out = Dither.composeFrame(srcData.data, w, h, [], P, [l.palette], l.bg);
  }
  g.putImageData(new ImageData(out, w, h), 0, 0);
}

function paintLookRow() {
  const wrap = $('#looks');
  if (!wrap) return;
  $$('#looks .chip').forEach((b) => b.setAttribute(
    'aria-pressed', String(b.dataset.look === (S.lookPreset || 'custom'))));
  const cur = LOOKS.find((x) => x.id === S.lookPreset);
  $('#vLook').textContent = cur ? cur.name : 'custom';
  // dots-based tiles are gated exactly like the dots mode chip
  $$('#looks .chip[data-dots="1"]').forEach((b) => {
    const ok = dotsAvailable();
    b.classList.toggle('off', !ok);
    b.title = ok ? '' : 'track a subject first';
  });
}

let LOOKS_BUILT = false;
function buildLookRow() {
  const wrap = $('#looks');
  if (!wrap || LOOKS_BUILT) return;
  LOOKS_BUILT = true;
  LOOKS.forEach((l) => {
    const b = document.createElement('button');
    b.className = 'chip look';
    b.dataset.look = l.id;
    if (l.P.mode === 'dots') b.dataset.dots = '1';
    const cv = document.createElement('canvas');
    cv.width = LT.w; cv.height = LT.h;
    const nm = document.createElement('span');
    nm.textContent = l.name;
    b.append(cv, nm);
    b.addEventListener('click', () => applyLook(l));
    wrap.append(b);
  });
  const cust = document.createElement('button');
  cust.className = 'chip look custom';
  cust.dataset.look = 'custom';
  const nm = document.createElement('span');
  nm.textContent = 'Custom';
  cust.title = 'your own mix — touch any dial below and the look is yours';
  cust.append(nm);
  wrap.append(cust);
  paintLookRow();
}

let LT_TIMER = 0;
/** Repaint the tiles from the frame that is on screen, at most ~every 1.5 s. */
function scheduleLookThumbs(force) {
  if (!LOOKS_BUILT) buildLookRow();
  // tiles that nobody can see are not repainted mid-playback
  const st3 = $('#st3');
  if (!force && st3 && (st3.hidden || st3.getAttribute('data-open') !== '1')) return;
  const now = performance.now();
  if (!force && now - LT.at < 1500) {
    if (!LT_TIMER) LT_TIMER = setTimeout(() => { LT_TIMER = 0; scheduleLookThumbs(); },
                                         1600 - (now - LT.at));
    return;
  }
  LT.at = now;
  lookThumbSource().then((src) => {
    if (!src) return;
    $$('#looks .chip.look').forEach((b) => {
      const l = LOOKS.find((x) => x.id === b.dataset.look);
      const cv = b.querySelector('canvas');
      if (l && cv) renderLookThumb(cv, l, src);
    });
  }).catch(() => {});
}

function bindSlider(id, out, key, fmt, int) {
  const el = $(id);
  el.addEventListener('input', () => {
    S.P[key] = int ? parseInt(el.value, 10) : parseFloat(el.value);
    $(out).textContent = fmt(S.P[key]);
    markCustom();
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
  markCustom();
  draw();
});
$('#tSerp').addEventListener('click', () => {
  S.P.serpentine = !S.P.serpentine;
  $('#tSerp').setAttribute('aria-pressed', String(S.P.serpentine));
  draw();
});
$('#sAlgo').addEventListener('change', (e) => { S.P.algo = e.target.value; markCustom(); draw(); });
$$('[data-mx]').forEach((b) => b.addEventListener('click', () => {
  S.P.matrix = +b.dataset.mx;
  $$('[data-mx]').forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
  markCustom();
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
  markCustom();
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
  let src = c.getImageData(0, 0, S.W, S.H).data;
  let masks = await masksFor(i, rec, 'x');
  let W = S.W, H = S.H;
  const plan = canvasPlanAt(i, S.W, S.H);
  if (plan) {
    const on = onCanvas((g) => g.drawImage(rec.frame, 0, 0), S.W, S.H,
                        masks, plan, 'cvx');
    src = on.src; masks = on.masks; W = on.W; H = on.H;
  }
  const pal = palettesForRender();
  // `alpha` is the transparent exports: same pixels, background keyed out
  const P = (opts && opts.alpha) ? Object.assign({}, S.P, { alpha: true }) : S.P;
  const out = (S.P.mode === 'dots' && masks.length)
    ? renderDots(src, W, H, masks, P, pal, S.bg, BLUE).out
    : Dither.composeFrame(src, W, H, masks, P, pal, S.bg);
  return new ImageData(out, W, H);
}

/** The clip's own frame `i`, at the render's size and nothing else done to it.
 *  Deliberately composeAt() with the dither taken out: the same frameAt cache,
 *  the same canvas, the same S.W x S.H — so frame i of the original cut is the
 *  picture frame i of the dithered render was made from. */
async function originalAt(i) {
  const rec = await frameAt(i);
  const plan = canvasPlanAt(i, S.W, S.H);
  if (plan) {
    // the matched cut follows the identical crop path — that is the whole of
    // "the two lay on top of each other in an edit" once there is a crop
    const c = ctx2d(plan.tw, plan.th, 'exo');
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, plan.tw, plan.th);
    c.fillStyle = '#000'; c.fillRect(0, 0, plan.tw, plan.th);
    c.imageSmoothingQuality = 'high';
    c.setTransform(plan.k, 0, 0, plan.k, plan.x0, plan.y0);
    c.drawImage(rec.frame, 0, 0);
    c.setTransform(1, 0, 0, 1, 0, 0);
    return c.getImageData(0, 0, plan.tw, plan.th);
  }
  const c = ctx2d(S.W, S.H, 'exp');
  c.clearRect(0, 0, S.W, S.H);
  c.drawImage(rec.frame, 0, 0);
  return c.getImageData(0, 0, S.W, S.H);
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
  let src = c.getImageData(0, 0, w, h).data;
  let masks = usingSubjects() ? stillMasksAt(w, h, 'xm') : [];
  let W = w, H = h;
  const plan = canvasPlanAt(0, w, h);
  if (plan) {
    const on = onCanvas((g) => g.drawImage(S.bitmap, 0, 0, w, h), w, h,
                        masks, plan, 'cvi');
    src = on.src; masks = on.masks; W = on.W; H = on.H;
  }
  const P = (opts && opts.alpha) ? Object.assign({}, S.P, { alpha: true }) : S.P;
  const pal = palettesForRender();
  if (S.P.mode === 'dots') {
    const r = renderDots(src, W, H, masks.length ? masks : [fullMask(W, H)],
                         P, pal, S.bg, BLUE);
    return { out: r.out, lit: r.lit, masks, w: W, h: H };
  }
  return { out: Dither.composeFrame(src, W, H, masks, P, pal, S.bg),
           lit: 0, masks, w: W, h: H };
}

async function exportPNG() {
  const info = $('#rinfo'); info.hidden = true; info.textContent = '';
  $('#dlorig').hidden = true;               // a still has no matched cut
  busy(true);
  await sleep(16);
  try {
    // re-render at the source's native resolution, not the preview's
    const alpha = alphaMatters() && S.pngAlpha;
    const { out, lit, w: ow, h: oh } = composeStill(S.natW, S.natH, { alpha });
    const cv = document.createElement('canvas');
    cv.width = ow; cv.height = oh;
    cv.getContext('2d').putImageData(new ImageData(out, ow, oh), 0, 0);
    const blob = await new Promise((r) => cv.toBlob(r, 'image/png'));
    const dl = $('#dl');
    if (dl.dataset.url) URL.revokeObjectURL(dl.dataset.url);
    const url = URL.createObjectURL(blob);
    dl.dataset.url = url; dl.href = url;
    dl.download = `${S.fileName || 'dither'}-${S.P.mode}`
      + (canvasOn() ? '-' + canvasSlug() : '') + (alpha ? '-alpha' : '') + '.png';
    dl.hidden = false;
    offerShare(url, dl.download, 'image/png');
    const box = $('#rinfo'); box.hidden = false; box.classList.remove('err');
    box.textContent = `${ow}×${oh} PNG · ${(blob.size / 1024).toFixed(0)} KB`
      + (canvasOn() ? ` · ${canvasLabel()}` : '')
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
const FMT_SHORT = { mp4: 'MP4', webm: 'WebM', gif: 'GIF',
                    'webm-alpha': 'Alpha', prores: 'ProRes' };
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
  // the select stays in the DOM (and settable); these chips are its face
  const chips = $('#fmtchips');
  if (chips) {
    chips.textContent = '';
    list.forEach((f) => {
      const b = document.createElement('button');
      b.className = 'chip';
      b.dataset.fmt = f.id;
      b.textContent = FMT_SHORT[f.id] || (f.ext || f.id).toUpperCase();
      b.title = f.label + (f.available ? (f.note ? ' — ' + f.note : '')
                                       : ' — ' + (f.note || 'unavailable here'));
      if (!f.available) b.classList.add('off');
      b.addEventListener('click', () => {
        if (!f.available) { toast(f.note || 'unavailable on this engine', true); return; }
        S.format = f.id; sel.value = f.id; paintFormat();
      });
      chips.append(b);
    });
  }
  paintFormat();
  buildSeqFormats();
}
function paintFormat() {
  const f = currentFormat();
  $$('#fmtchips .chip').forEach((b) => b.setAttribute(
    'aria-pressed', String(b.dataset.fmt === f.id)));
  $('#vFmt').textContent = f.ext ? f.ext.toUpperCase() : '';
  $('#fmtnote').textContent = f.note || '';
  $('#giffps').hidden = f.id !== 'gif';
  $('#bExport').textContent = S.kind === 'image' ? 'Download PNG'
    : 'Render ' + (f.ext || '').toUpperCase();
  paintOriginalUI();
}

/* ---- the original cut ------------------------------------------------
 * An export that is meant for an edit is a PAIR: the dithered render and the
 * clip it was made from, cut to exactly the same frames. Video only — a still
 * already has its source on disk — and only where the engine can write it (an
 * older server has no /original route).
 *
 * Containers: the tab has no H.264 encoder, so the browser engine always pairs
 * with WebM. The server follows the render's format where that means anything
 * and falls back to MP4 where it does not: a GIF of the original would be
 * decimated to gif_fps, and an alpha format has nothing to key out of footage
 * nobody dithered.
 */
const ORIG_KEY = 'dither-studio.saveOriginal';
const ORIGINAL_EXT = { mp4: 'mp4', webm: 'webm', gif: 'mp4',
                       'webm-alpha': 'mp4', prores: 'mp4' };
const originalOffered = () => S.kind === 'video'
  && !!(E() && E().supports && E().supports.original);

function originalExt() {
  return E() && E().id === 'browser'
    ? 'webm' : (ORIGINAL_EXT[currentFormat().id] || 'mp4');
}

function paintOriginalUI() {
  const box = $('#origui');
  if (!box) return;
  box.hidden = !originalOffered();
  $('#cOrig').checked = S.saveOriginal;
  if (box.hidden) return;
  const ext = originalExt(), f = currentFormat();
  const reason = E().id === 'browser'
    ? 'the tab has no other encoder.'
    : (f.id === 'gif'
      ? "a GIF of the original would be decimated to the GIF's frame rate, and "
        + 'pairing a GIF with a GIF is pointless.'
      : 'an alpha container has nothing to key out of footage nobody dithered.');
  const swapped = ext !== f.ext
    ? ` The ${(f.ext || '').toUpperCase()} pairs with an ${ext.toUpperCase()}: ${reason}`
    : '';
  $('#orignote').textContent =
    `A second file beside the render — same trim, same frames, same size, same `
    + `rate, no dither — so the two lay on top of each other in an edit.${swapped}`;
}

$('#cOrig').addEventListener('change', (e) => {
  S.saveOriginal = e.target.checked;
  try { sessionStorage.setItem(ORIG_KEY, S.saveOriginal ? '1' : '0'); } catch (err) { /* blocked */ }
  paintOriginalUI();
});
try { S.saveOriginal = sessionStorage.getItem(ORIG_KEY) === '1'; } catch (e) { /* blocked */ }

$('#sFmt').addEventListener('change', (e) => { S.format = e.target.value; paintFormat(); });
$$('[data-gfps]').forEach((b) => b.addEventListener('click', () => {
  S.gifFps = +b.dataset.gfps;
  $$('[data-gfps]').forEach((o) => o.setAttribute('aria-pressed', String(o === b)));
}));

/* ---- the share sheet (phones mostly). The File is prepared as soon as the
 * export lands, because navigator.share must be called synchronously in the
 * tap handler — no await between the tap and the call. Falls back silently:
 * the #dl anchor is always there (Android in-app WebViews have no Web Share). */
async function offerShare(url, name, mime) {
  const row = $('#sharerow');
  if (!row) return;
  row.hidden = true; S.shareFile = null;
  if (!navigator.canShare || !navigator.share) return;
  try {
    const blob = await (await fetch(url)).blob();
    const f = new File([blob], name, { type: mime || blob.type });
    if (!navigator.canShare({ files: [f] })) return;
    S.shareFile = f;
    $('#bShare').textContent = `share… · ${(blob.size / 1e6).toFixed(1)} MB`;
    row.hidden = false;
  } catch (e) { /* the download button is the fallback */ }
}
$('#bShare') && $('#bShare').addEventListener('click', () => {
  const f = S.shareFile;
  if (!f) return;
  navigator.share({ files: [f] }).catch((e) => {
    if (e && e.name !== 'AbortError') {
      toast('sharing failed — use download instead', true);
    }
  });
});

async function exportClip() {
  const btn = $('#bExport'); btn.disabled = true;
  $('#sharerow') && ($('#sharerow').hidden = true);
  $('#dl').hidden = true; $('#dlorig').hidden = true;
  $('#dl').textContent = 'download';
  $('#outvid').hidden = true; $('#outimg').hidden = true;
  const info = $('#rinfo'); info.hidden = true; info.textContent = '';
  const prog = $('#rprog'); prog.hidden = false;
  const bar = $('.bar i', prog), lab = $('span', prog);
  bar.style.width = '0%'; lab.textContent = 'starting…';
  stop();
  holdWake();
  try {
    const fmt = currentFormat();
    const rng = activeRange();
    if (canvasOn()) {
      lab.textContent = 'framing…';
      await ensureCanvasPath((pr) => { lab.textContent = pr.text; });
    }
    const canvas = canvasPayload(rng);
    const params = Object.assign({}, S.P, {
      bg: S.bg, palette: S.palette, fps: S.fps,
      format: fmt.id, gif_fps: S.gifFps,
      /* THE CANVAS, already worked out. `place` is one [x0, y0] per frame of
       * the window (or one for the whole of it, when the frame holds still),
       * and `k` the scale: server/render.py applies exactly these numbers, and
       * so does the browser engine below, so the render, the matched cut and
       * the dot data cannot disagree about where the crop was. */
      canvas,
      // the browser engine records onto a canvas of `source`'s size
      ...(canvas ? { source: { w: canvas.w, h: canvas.h,
                               nFrames: S.nFrames, fps: S.fps } } : {}),
      // the window, on both engines: server/render.frame_range slices the
      // frames directory, the browser engine walks the same indices
      frame_in: rng.in, frame_out: rng.out,
      subjects: usingSubjects()
        ? S.subjects.map((s) => ({ id: s.id, palette: s.palette,
                                   polish: s.polish | 0 })) : [],
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
    dl.download = `${S.fileName || 'dither'}-${S.P.mode}`
      + (canvas ? '-' + canvasSlug() : '') + `.${r.ext}`;
    dl.hidden = false;
    if (r.image) { const im = $('#outimg'); im.src = r.url; im.hidden = false; }
    else if (r.playable) { const v = $('#outvid'); v.src = r.url; v.hidden = false; }
    offerShare(r.url, 'clip.' + r.ext, fmt.mime);
    const box = $('#rinfo'); box.hidden = false; box.classList.remove('err');
    box.textContent = `rendered ${r.frames} frames in ${r.elapsedS.toFixed(1)} s `
      + `(${r.fps.toFixed(1)} fps)`
      + (rng.whole ? '' : ` · frames ${rng.in}–${rng.out} of ${S.nFrames}`)
      + (canvas ? ` · ${canvas.w}×${canvas.h}` : '')
      + (r.bytes ? ` · ${(r.bytes / 1e6).toFixed(1)} MB` : '')
      + (r.note ? ` · ${r.note}` : '');
    $('#s5sum').textContent = 'ready';
    if (S.saveOriginal && originalOffered()) {
      box.textContent += ' · ' + await exportOriginalCut(params, r, prog, bar, lab);
    }
  } catch (err) {
    prog.hidden = true;
    const box = $('#rinfo'); box.hidden = false; box.classList.add('err');
    box.textContent = 'render failed: ' + why(err);
  }
  releaseWake();
  btn.disabled = false;
}

/** The second file: the render's own frames, undithered.
 *
 *  Runs after the render, never instead of it — if this fails the dithered
 *  export is still on the page and still downloadable, and the stat line says
 *  what went wrong. The count is the assertion: the pair is only useful if it
 *  is frame for frame, so a mismatch is an error rather than a note. The
 *  server is told the same number and refuses on its side too.
 */
async function exportOriginalCut(params, r, prog, bar, lab) {
  // the frames the render consumed: the active range, not the whole clip
  const want = activeRange().n;
  /* If the dither ran slower than real time the tab's recorder wrote a file at
   * that slower rate (exportWebM says so in its note). The pair is only usable
   * if both files carry the same rate, so the original is handed over at the
   * pace the render actually achieved rather than at the clip's own — but only
   * where the two files are the same kind of thing, since a GIF render is
   * decimated to gif_fps and pairs with a full-rate video. The server engine
   * encodes at a fixed -r and never needs any of this. */
  const paced = r.frames === want && r.elapsedS ? (r.elapsedS * 1000) / r.frames : 0;
  const realMs = 1000 / Math.max(1, S.fps);
  try {
    prog.hidden = false;
    bar.style.width = '0%'; lab.textContent = 'the original cut…';
    const o = await E().exportOriginal(
      Object.assign({}, params, { expect_frames: want,
                                  pace_ms: paced > realMs * 1.05 ? paced : 0 }),
      (p) => {
        bar.style.width = (p.total ? (p.done / p.total) * 100 : 0).toFixed(1) + '%';
        lab.textContent = 'original · ' + p.text;
      },
      originalAt);
    prog.hidden = true;
    if (o.frames !== want) {
      throw new Error(`${o.frames} frames against the clip's ${want} — `
        + 'the cut would not line up');
    }
    if (S.exportOrigURL) URL.revokeObjectURL(S.exportOrigURL);
    S.exportOrigURL = o.url.startsWith('blob:') ? o.url : null;
    const a = $('#dlorig');
    a.href = o.url;
    a.download = `${S.fileName || 'dither'}-${S.P.mode}`
      + (canvasOn() ? '-' + canvasSlug() : '') + `.original.${o.ext}`;
    a.hidden = false;
    $('#dl').textContent = 'download the dithered';
    return `original cut: ${o.frames} frames, ${o.ext.toUpperCase()}`
      + (o.bytes ? `, ${(o.bytes / 1e6).toFixed(1)} MB` : '');
  } catch (err) {
    prog.hidden = true;
    return 'the original cut failed: ' + why(err);
  }
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

function dotsParams(rng) {
  const r = rng || activeRange();
  return {
    cell: S.P.cell, dotpx: S.P.dotpx, n: S.P.n, fill: S.P.fill,
    stray: S.P.stray, band: S.P.band, gamma: S.P.gamma, invert: S.P.invert,
    seed: S.P.seed, bg: S.bg, fps: S.fps,
    frame_in: r.in, frame_out: r.out,
    // the dot data carries the canvas too: a .dots.gz written for a 9:16
    // export has 9:16 in its header and positions inside it
    canvas: canvasPayload(r),
    subjects: S.subjects.map((x) => ({ id: x.id, palette: x.palette,
                                      polish: x.polish | 0 })),
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
async function dotsDocStill(whole) {
  const P = await playerLib();
  const w = S.natW, h = S.natH;
  const c = ctx2d(w, h, 'exp');
  c.clearRect(0, 0, w, h);
  c.drawImage(S.bitmap, 0, 0, w, h);
  const src = c.getImageData(0, 0, w, h).data;
  const use = !whole && usingSubjects();
  let masks = use ? stillMasksAt(w, h, 'xm') : [fullMask(w, h)];
  let sd = src, W = w, H = h;
  const plan = canvasPlanAt(0, w, h);
  if (plan) {
    const on = onCanvas((g) => g.drawImage(S.bitmap, 0, 0, w, h), w, h,
                        masks, plan, 'cvd');
    sd = on.src; W = on.W; H = on.H;
    masks = use ? on.masks : [fullMask(W, H)];
  }
  const r = dotsOn(sd, W, H, masks, S.P, BLUE);
  const cols = use ? S.subjects.map((x) => x.palette[x.palette.length - 1])
    : [S.palette[S.palette.length - 1]];
  const doc = { w: W, h: H, fps: 1, dotpx: S.P.dotpx,
                palette: [S.bg].concat(cols), bgIndex: 0, bg: S.bg,
                subjects: cols.map((col) => ({ color: col })),
                frames: [r.on.map((o) => dotXY(r.F, o))] };
  return { key: 'still', doc, bytes: await P.pack(doc) };
}

/** The current clip as a dots document, cached against the look it was made
 *  with — the sequence step asks for this repeatedly. A still short-circuits:
 *  one frame, nothing to cache against. */
async function dotsDoc(onProgress, rng) {
  if (S.kind === 'image') return dotsDocStill();
  const P = await playerLib();
  // the active range by default; the sequence asks for the whole clip, because
  // a strip entry carries its own in/out and can be widened later
  const R0 = rng || activeRange();
  if (canvasOn()) await ensureCanvasPath(onProgress);
  const params = dotsParams(R0);
  const key = JSON.stringify([E().id, S.job, S.nFrames, params]);
  if (DOTS_CACHE && DOTS_CACHE.key === key) return DOTS_CACHE;
  let bytes, doc;
  if (E().exportDots) {
    const r = await E().exportDots(params, onProgress);
    bytes = r.bytes;
    doc = await P.unpack(bytes);
  } else {
    const frames = [];
    let DW = S.W, DH = S.H;
    for (let i = R0.in; i <= R0.out; i++) {
      const rec = await frameAt(i);
      const c = ctx2d(S.W, S.H, 'exp');
      c.clearRect(0, 0, S.W, S.H);
      c.drawImage(rec.frame, 0, 0);
      let src = c.getImageData(0, 0, S.W, S.H).data;
      let masks = await masksFor(i, rec, 'x');
      let W = S.W, H = S.H;
      const plan = canvasPlanAt(i, S.W, S.H);
      if (plan) {
        const on = onCanvas((g) => g.drawImage(rec.frame, 0, 0), S.W, S.H,
                            masks, plan, 'cvd');
        src = on.src; masks = on.masks; W = on.W; H = on.H;
      }
      DW = W; DH = H;
      const r = dotsOn(src, W, H, masks, S.P, BLUE);
      frames.push(r.on.map((o) => dotXY(r.F, o)));
      if (onProgress) onProgress({ done: frames.length, total: R0.n,
                                   text: `${frames.length}/${R0.n}` });
      if (i % 8 === 0) await sleep(0);
    }
    const cols = S.subjects.map((x) => x.palette[x.palette.length - 1]);
    doc = { w: DW, h: DH, fps: S.fps, dotpx: S.P.dotpx,
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

/* ==================================================== the sequence flow ===
 * A sequence is its own view, not a step: the header switches between STUDIO
 * (drop something, cut a subject out of it, pick a look, export it) and
 * SEQUENCE (a strip of dot clouds with a transition between each pair).
 *
 * Three lists hold it:
 *
 *   S.library   the pool. Everything captured this session — a tracked clip's
 *               subjects, a dithered still, a rasterised shape — kept as dot
 *               positions, per subject, at full length. It outlives the clip it
 *               came from on purpose: a morph from one clip into another needs
 *               both, and only one of them can be loaded at a time.
 *   S.strip     what is actually in the sequence: instances that point at a
 *               pool item and carry their own options — which subject, in/out,
 *               how long a still holds, an optional colour override, and the
 *               transition that leads INTO them.
 *   S.trans     nothing. The join lives on the item that follows it, so
 *               dragging an item to a new position takes its transition with
 *               it instead of leaving it behind.
 *
 * The document is built by web/player/dither-player.js, played by the same
 * player the .dots.gz files use, and rendered to video by the server route
 * that has always rasterised dot positions (/api/sequence) — one implementation
 * of every transition, in JS, and the server only ever feeds an encoder.
 */
let SEQID = 1;
const libOf = (id) => S.library.find((x) => x.id === id);
const KINDS = ['morph', 'scatter', 'cut', 'density'];
const EMPTY_XY = new Uint16Array(0);

/* The look of the SEQUENCE, as opposed to the look of an item in it: the
 * canvas. One background, one dot size, one frame rate for the whole strip.
 * Dot size is here and not on the item because a .dots.gz stores exactly one
 * of it — the whole point of the format is that size and colour are not baked
 * into the positions — so a per-item dot size could not survive an export. */
function seqLook() {
  return { dotpx: S.seq.dotpx, bg: S.seq.bg, fps: S.seq.fps,
           cell: S.P.cell, seed: S.P.seed };
}

/* ===================================================== an item's own look ===
 * Everything that decides where an item's dots land. A strip entry carries its
 * own copy, so two entries made from the same capture diverge freely: change
 * the mode on item 2 and items 1 and 3 do not move a dot.
 *
 * `colors` and `polish` are keyed by SUBJECT id, not by index, so they survive
 * a subject picker change. `colors` is the only thing in here that does not
 * change the positions — which is why `lookKey` leaves it out, and why
 * re-colouring an item costs nothing.
 */
const LOOK_GEOM = ['mode', 'algo', 'matrix', 'serpentine', 'strength', 'cell',
                   'n', 'fill', 'stray', 'band', 'gamma', 'invert', 'seed'];
/* The bounds the inspector's sliders use. Wider than the studio's on purpose:
 * an item in a sequence is often much smaller in frame than the thing that was
 * dithered to make it — and `cell` reaches 1, where a pixel dither mode is a
 * true pixel dither rather than a screen at dot size. */
const LOOK_RANGE = { n: [200, 60000], cell: [1, 16], band: [0, 30] };

/* Choosing a pixel dither mode drops the cell to 1 — Bayer at cell 4 is a
 * chunky screen, Bayer at cell 1 is Bayer — and choosing Dots again puts the
 * cell the item was captured at back. `dotCell` remembers it; it is not part of
 * LOOK_GEOM, so it never enters the derivation key. */
function modeSwitch(look, mode) {
  if (mode === look.mode) return { mode };
  if (mode === 'dots') return { mode, cell: look.dotCell || look.cell };
  if (look.mode === 'dots') return { mode, cell: 1, dotCell: look.cell };
  return { mode };
}

/** The look an item is captured at: the studio's dot settings, its subject
 *  colours and its mask polish, copied. The MODE is always `dots` — the studio
 *  hands the sequence a dot cloud whatever it is showing itself, and that is
 *  what has always gone in, so adding an item still changes nothing on screen.
 *  Every other mode is one click away on the item afterwards. */
function lookFromStudio(subjects) {
  const P = S.P;
  const look = {
    mode: 'dots', algo: P.algo, matrix: P.matrix, serpentine: P.serpentine,
    strength: P.strength, cell: P.cell, n: P.n, fill: P.fill, stray: P.stray,
    band: P.band, gamma: P.gamma, invert: P.invert, seed: P.seed,
    colors: {}, polish: {},
  };
  (subjects || []).forEach((s) => {
    if (s.color) look.colors[s.id] = s.color;
    if ((s.polish | 0) > 0) look.polish[s.id] = s.polish | 0;
  });
  return look;
}

const cloneLook = (l) => JSON.parse(JSON.stringify(l));

/** The identity of a derivation: the geometry knobs plus the polish, with the
 *  colours left out because they do not move a dot. */
function lookKey(look) {
  const pol = Object.keys(look.polish || {}).sort()
    .map((k) => k + '=' + look.polish[k]);
  return JSON.stringify(LOOK_GEOM.map((k) => look[k])) + '|' + pol.join(',');
}

/* ==================================================== the derivation cache ===
 * An item is a LIVE reference: it keeps its source, not a picture, and its
 * dots are worked out on demand at whatever look it is wearing. That is only
 * affordable with a cache, so this is it — one slot per (library item, look),
 * each holding the frames that have actually been asked for.
 *
 * Capture seeds the slot for the look the item was captured at with the exact
 * positions the studio produced, so an item nobody has touched is byte for
 * byte what it always was, and costs nothing to draw.
 */
const DERIVED = new Map();
const DERIVED_MAX = 32;

function derivedSlot(item, look) {
  const key = item.id + '|' + lookKey(look);
  let d = DERIVED.get(key);
  if (d) { DERIVED.delete(key); } else { d = { key, frames: new Map() }; }
  DERIVED.set(key, d);
  // oldest first, but never the capture slots: those hold the exact positions
  // the studio produced, which is the whole of "an untouched item is what it
  // always was" and the only copy of them there is
  for (const [k, v] of DERIVED) {
    if (DERIVED.size <= DERIVED_MAX) break;
    if (!v.pinned && k !== key) DERIVED.delete(k);
  }
  return d;
}
const derivedPeek = (item, look) => DERIVED.get(item.id + '|' + lookKey(look));

/** The capture: the positions the studio made, at the look it made them at. */
function seedDerived(item, look, frames) {
  const d = derivedSlot(item, look);
  frames.forEach((f, i) => d.frames.set(i, f));
  d.pinned = true;
  return d;
}

/** Redraw one frame of one item at one look. This is the whole of it: the
 *  source frame, the masks (polished if the look asks), the dot grid. Both
 *  engines come through here, so there is one answer and not two. */
async function deriveFrame(item, look, i) {
  const src = item.src, W = item.w, H = item.h;
  const c = ctx2d(W, H, 'dv');
  let masks;
  if (src.kind === 'shape') {
    paintShape(c, src, W, H);
    masks = [fullMask(W, H)];
  } else if (src.kind === 'still') {
    c.clearRect(0, 0, W, H);
    c.drawImage(src.bitmap, 0, 0, W, H);
    masks = src.masks
      ? src.masks.map((m, k) => (m.image
          ? bitmapAlpha(m.image, W, H, 'dm' + k) : new Float32Array(W * H)))
      : [fullMask(W, H)];
  } else {
    const bmp = await src.source.frame(i);
    c.clearRect(0, 0, W, H);
    c.drawImage(bmp, 0, 0, W, H);
    if (bmp.close) bmp.close();
    masks = [];
    for (let k = 0; k < item.tracks.length; k++) {
      const id = item.tracks[k].id;
      const strength = (look.polish || {})[id] | 0;
      if (strength > 0) {
        masks.push(await polishedMask(src.well, id, i, strength,
                                      'iw' + item.id + '.' + k));
      } else {
        const mb = await src.source.mask(id, i);
        masks.push(bitmapAlpha(mb, W, H, 'dm' + k));
        if (mb.close) mb.close();
      }
    }
  }
  const data = c.getImageData(0, 0, W, H).data;
  const r = dotsOnMode(data, W, H, masks, look, BLUE);
  return r.on.map((o) => dotXY(r.F, o));
}

/* Derivations run one at a time. Two overlapping builds — a debounced preview
 * and an explicit one, say — would otherwise decode the same frames twice and
 * fight over the cache's eviction order. */
let DERIVE_Q = Promise.resolve();
const serialise = (fn) => {
  const next = DERIVE_Q.then(fn, fn);
  DERIVE_Q = next.catch(() => {});
  return next;
};

/** Make sure frames `from`..`to` of `item` exist at `look`. Only the frames an
 *  item actually uses are ever derived, which is why dragging a trim is cheap
 *  and changing the mode on a 45-frame item is a second, not a minute. */
function deriveRange(item, look, from, to, onProgress) {
  return serialise(() => deriveRangeNow(item, look, from, to, onProgress));
}

async function deriveRangeNow(item, look, from, to, onProgress) {
  const d = derivedSlot(item, look);
  const need = [];
  for (let i = from; i <= to; i++) if (!d.frames.has(i)) need.push(i);
  if (!need.length) return d;
  await playerLib();
  for (let k = 0; k < need.length; k++) {
    d.frames.set(need[k], await deriveFrame(item, look, need[k]));
    if (onProgress) onProgress({ done: k + 1, total: need.length });
    if ((k & 3) === 3) await sleep(0);
  }
  return d;
}

/* ------------------------------------------------------------- the pool */
/** Everything the current source could contribute, as pool items. */
async function captureClip() {
  // the whole tracked clip, whatever the studio's range is: the item is the
  // material, and stripAdd() seeds the ENTRY's in/out from the active range
  const whole = { in: 0, out: Math.max(0, S.nFrames - 1),
                  n: Math.max(1, S.nFrames), whole: true };
  const { doc } = await withoutCanvas(() => dotsDoc(
    (pr) => seqInfo('reading dot positions ' + pr.text), whole));
  const source = E().snapshot ? E().snapshot() : null;
  if (!source) throw new Error('this engine has no handle on that clip');
  const subs = doc.subjects.map((sub, k) => ({
    id: (S.subjects[k] || {}).id || (k + 1),
    color: sub.color,
    polish: (S.subjects[k] || {}).polish | 0,
  }));
  const item = {
    id: SEQID++, kind: 'clip', name: S.fileName || 'clip',
    w: doc.w, h: doc.h, fps: doc.fps, nFrames: doc.frames.length,
    tracks: subs.map((s) => ({ id: s.id, color: s.color })),
    look: lookFromStudio(subs),
    src: { kind: 'clip', source,
           // its own polish well: this item can be re-polished long after the
           // clip it came from has left the studio
           well: newWell('iw' + SEQID, {
             get: (objId, j) => source.mask(objId, j),
             size: () => ({ w: doc.w, h: doc.h, n: doc.frames.length }),
           }) },
  };
  seedDerived(item, item.look, doc.frames);
  S.library.push(item);
  return item;
}

/** An independent copy of a mask image, so closing the original cannot take it
 *  away. Returns null for a subject that has none. */
async function cloneMask(img) {
  if (!img) return null;
  try { return await createImageBitmap(img); } catch (e) { return img; }
}

/** Everything about the studio that decides what a still capture contains, so
 *  adding subject #1 and then subject #2 of the same photograph reuses one
 *  library entry instead of dithering it twice. */
function stillCapKey(whole) {
  return JSON.stringify([!!whole, S.fileName, S.natW, S.natH, S.bg,
    S.P.cell, S.P.n, S.P.fill, S.P.stray, S.P.band, S.P.gamma, S.P.invert,
    S.P.seed, S.palette,
    whole ? [] : S.subjects.map((x) => [x.id, x.palette])]);
}

/** The picture in the studio, as an item. `whole` ignores whatever is selected
 *  and takes the entire frame; otherwise the subjects come across as one track
 *  each and the strip entry picks which of them to show. */
async function captureStill(opts) {
  const whole = !!(opts && opts.whole);
  const key = stillCapKey(whole);
  const had = S.library.find((x) => x.kind === 'still' && x.capKey === key
                                 && x.src.bitmap === S.bitmap);
  if (had) return had;
  const { doc } = await withoutCanvas(() => dotsDocStill(whole));
  const use = !whole && usingSubjects();
  const subs = doc.subjects.map((sub, k) => ({
    id: use ? ((S.subjects[k] || {}).id || (k + 1)) : 1,
    color: sub.color, polish: 0,
  }));
  const item = {
    id: SEQID++, kind: 'still',
    name: (S.fileName || 'still') + (whole && usingSubjects() ? ' (whole)' : ''),
    w: doc.w, h: doc.h, fps: 30, nFrames: 1, capKey: key,
    tracks: subs.map((s) => ({ id: s.id, color: s.color })),
    look: lookFromStudio(subs),
    src: { kind: 'still', bitmap: S.bitmap, whole,
           // a COPY of each mask: dropStill() closes the studio's when another
           // picture is loaded, and this item has to outlive that
           masks: use ? await Promise.all(S.subjects.map(async (s) => ({
             id: s.id, image: await cloneMask(S.stillMasks.get(s.id)),
           }))) : null },
  };
  seedDerived(item, item.look, [doc.frames[0]]);
  S.library.push(item);
  return item;
}

/* ---------------------------------------------------------- static shapes
 * A shape becomes dots through the same pipeline a clip does: draw it dark on
 * light, hand it to the dot grid with a full-frame mask, keep the positions.
 * So a ring is dithered, not plotted — and because the item keeps the recipe
 * rather than the result, changing its look redraws it. */
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

/** A shape source, painted. `src.shape` is 'ring' | 'coral' | a bitmap's name. */
function paintShape(c, src, W, H) {
  if (src.bitmap) {
    c.fillStyle = '#fff'; c.fillRect(0, 0, W, H);
    const k = Math.min(W / src.bitmap.width, H / src.bitmap.height) * 0.86;
    const w = src.bitmap.width * k, h = src.bitmap.height * k;
    c.drawImage(src.bitmap, (W - w) / 2, (H - h) / 2, w, h);
  } else if (src.shape === 'coral') drawCoral(c, W, H);
  else drawRing(c, W, H);
}

/* ------------------------------------------------------- the seq's frame ===
 * A .dots.gz has exactly one frame size, so the strip has one too. The default
 * is the first item's — which is what it always was — and the presets are the
 * studio's, out of the same table. Items that are not that shape are FITTED
 * into it (contain, centred) rather than left sitting off centre: a dot cloud
 * is positions, so re-placing one is arithmetic and loses nothing.
 */
const seqSource = () => ({ w: S.library[0] ? S.library[0].w : (S.W || 1280),
                           h: S.library[0] ? S.library[0].h : (S.H || 720) });
function seqTarget() {
  const src = seqSource();
  return CV.targetSize({ preset: S.seq.preset, w: S.seq.w, h: S.seq.h },
                       src.w, src.h) || src;
}
const seqW = () => seqTarget().w;
const seqH = () => seqTarget().h;

/** One item's tracks, placed on the sequence's frame. A no-op when the item
 *  already is that shape, which is the common case and the old behaviour.
 *
 *  Two rules, both of them about what dot positions can and cannot survive:
 *
 *    NEVER MAGNIFY.  Scaling positions up spreads the cloud without growing
 *      the dots, so a 2.6x "fit" is the same subject with gaps between its
 *      dots. The scale is therefore min(1, contain) — shrink to fit if the
 *      item is too big for the frame, otherwise leave the spacing exactly as
 *      it was captured. (The studio's canvas has no such limit because it
 *      re-measures the dots on the new frame; a strip item is positions, and
 *      re-deriving every item on every preset click is not what this control
 *      is for.)
 *    CENTRE THE CLOUD, not the frame the cloud came out of. A subject that sat
 *      in the left third of a 16:9 clip belongs in the middle of a 9:16
 *      sequence, and the empty pixels around it were never in the file.
 */
function fitTracks(tracks, iw, ih, W, H) {
  if (!iw || !ih || (iw === W && ih === H)) return tracks;
  const k = Math.min(1, W / iw, H / ih);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  tracks.forEach((t) => t.frames.forEach((f) => {
    for (let i = 0; i < f.length; i += 2) {
      if (f[i] < x0) x0 = f[i];
      if (f[i] > x1) x1 = f[i];
      if (f[i + 1] < y0) y0 = f[i + 1];
      if (f[i + 1] > y1) y1 = f[i + 1];
    }
  }));
  const cx = x1 < x0 ? iw / 2 : (x0 + x1) / 2;
  const cy = y1 < y0 ? ih / 2 : (y0 + y1) / 2;
  const ox = W / 2 - cx * k, oy = H / 2 - cy * k;
  return tracks.map((t) => ({
    color: t.color,
    frames: t.frames.map((f) => {
      const out = new Uint16Array(f.length);
      for (let i = 0; i < f.length; i += 2) {
        out[i] = clamp(Math.round(f[i] * k + ox), 0, W - 1);
        out[i + 1] = clamp(Math.round(f[i + 1] * k + oy), 0, H - 1);
      }
      return out;
    }),
  }));
}
const seqColor = () => (S.library[0] ? S.library[0].tracks[0].color
  : (S.subjects[0] ? S.subjects[0].palette.slice(-1)[0] : '#b0413e'));

async function addShape(kind, bitmap) {
  await playerLib();
  const W = seqW(), H = seqH();
  const src = { kind: 'shape', shape: kind, bitmap: bitmap || null };
  const color = seqColor();
  const item = { id: SEQID++, kind: 'shape', name: kind, w: W, h: H, fps: 30,
                 nFrames: 1, tracks: [{ id: 1, color }],
                 look: lookFromStudio([{ id: 1, color }]), src };
  const xy = (await deriveFrame(item, item.look, 0))[0];
  if (!xy || !xy.length) throw new Error('that shape came out empty');
  seedDerived(item, item.look, [[xy]]);
  S.library.push(item);
  return item;
}

/* --------------------------------------------------------------- the strip */
function stripAdd(item, opts) {
  const n = item.nFrames;
  const clip = n > 1;
  const inst = Object.assign({
    uid: SEQID++, lib: item.id, subject: 'all',
    // default to the WHOLE clip; the item panel's in/out trims it down
    in: 0,
    out: clip ? n - 1 : 0,
    hold: 30, color: null,
    look: cloneLook(item.look),
    trans: { kind: 'morph', ms: 900 },
  }, opts || {});
  if (!inst.look) inst.look = cloneLook(item.look);
  S.strip.push(inst);
  S.sel = { type: 'item', i: S.strip.length - 1 };
  renderSeq();
  return inst;
}

/** The frames an entry uses out of its item: [first, last]. */
function stripRange(inst) {
  const it = libOf(inst.lib);
  const n = it ? it.nFrames : 1;
  if (n <= 1) return [0, 0];
  const a = clamp(inst.in | 0, 0, n - 1);
  return [a, clamp(inst.out | 0, a, n - 1)];
}

/** How long an entry runs — arithmetic, not a count of what has been derived,
 *  so a trim reads back immediately and a still still holds. */
const stripLen = (inst) => {
  const it = libOf(inst.lib);
  if (!it) return 0;
  if (it.nFrames > 1) { const [a, b] = stripRange(inst); return b - a + 1; }
  return Math.max(1, inst.hold | 0);
};

/** Which subject indices this entry draws. */
function stripPick(inst, it) {
  if (inst.subject === 'all') return it.tracks.map((t, k) => k);
  const k = +inst.subject;
  return [Number.isFinite(k) && it.tracks[k] ? k : 0];
}

/** The colour one of an entry's tracks is drawn in: the whole-item override
 *  first (that is what a palette preset sets), then the item's own per-subject
 *  colour, then the colour the subject had when it was captured. */
function trackColor(inst, it, k) {
  if (inst.color) return inst.color;
  const id = it.tracks[k].id;
  return (inst.look.colors || {})[id] || it.tracks[k].color;
}

/** One strip entry as the player wants it: tracks of dot frames, trimmed.
 *  Reads the derivation cache — `ensureStrip` fills it — and falls back to an
 *  empty frame for anything not derived yet, so nothing here can block. */
function stripTracks(inst) {
  const it = libOf(inst.lib);
  if (!it) return [];
  const d = derivedPeek(it, inst.look);
  const [a, b] = stripRange(inst);
  return stripPick(inst, it).map((k) => {
    let frames;
    if (it.nFrames > 1) {
      frames = [];
      for (let i = a; i <= b; i++) {
        const f = d && d.frames.get(i);
        frames.push((f && f[k]) || EMPTY_XY);
      }
    } else {
      const f = d && d.frames.get(0);
      frames = new Array(Math.max(1, inst.hold | 0))
        .fill((f && f[k]) || EMPTY_XY);
    }
    return { frames, color: trackColor(inst, it, k) };
  });
}

/** Derive whatever the strip is currently asking for. Every build goes through
 *  this, which is what keeps the preview, the .dots.gz and the MP4 the same
 *  picture: they all read the cache this fills. */
async function ensureStrip(onProgress) {
  let derived = 0;
  for (let i = 0; i < S.strip.length; i++) {
    const inst = S.strip[i];
    const it = libOf(inst.lib);
    if (!it) continue;
    const [a, b] = stripRange(inst);
    await deriveRange(it, inst.look, a, b, (pr) => {
      derived = pr.done;
      if (onProgress) onProgress({ i, name: it.name, done: pr.done, total: pr.total });
    });
  }
  return derived;
}

function seqItems() {
  const W = seqW(), H = seqH();
  return S.strip.map((inst, i) => {
    const it = libOf(inst.lib) || { name: '?' };
    return { name: `${i + 1}. ${it.name}`,
             tracks: fitTracks(stripTracks(inst), it.w, it.h, W, H),
             cell: inst.look ? inst.look.cell : 0,
             transition: i > 0 ? inst.trans : null };
  });
}

async function buildSeq() {
  const P = await playerLib();
  if (!S.strip.length) throw new Error('nothing in the sequence yet');
  await ensureStrip((pr) => seqInfo(
    `redrawing ${pr.name} · ${pr.done}/${pr.total} frames`));
  const doc = P.buildSequence(seqItems(), Object.assign({
    w: seqW(), h: seqH(), color: seqColor(), durationMs: 900,
  }, seqLook()));
  S.seqDoc = doc;
  return doc;
}

/* An item's look changed: re-derive just that item and play the strip again.
 * Debounced, because these come off sliders. */
let SEQ_PENDING = null;
function seqTouch(quick) {
  renderStrip();
  clearTimeout(SEQ_PENDING);
  SEQ_PENDING = setTimeout(() => { seqPreviewSafe(); },
                           quick ? 0 : 220);
}

/* ------------------------------------------------------------ the view */
function setView(v, opts) {
  S.view = v;
  $('#studiopanel').hidden = v !== 'studio';
  $('#seqpanel').hidden = v !== 'sequence';
  $$('#viewbar .chip[data-view]').forEach((b) => b.setAttribute(
    'aria-pressed', String(b.dataset.view === v)));
  if (v === 'sequence') {
    showStage('sequence');
    renderSeq();
    // opening the view rebuilds and plays: the strip may have been edited
    // since the last preview, and a stage showing a stale document (or a black
    // rectangle) is worse than the tenth of a second the rebuild costs.
    // `skipPreview` is for seqAdd, which awaits its own.
    if (opts && opts.skipPreview) return paintToSeq();
    if (S.strip.length) seqPreviewSafe();
    else if (PLAYER) PLAYER.pause();
  } else {
    if (PLAYER) PLAYER.pause();
    restoreStage();
  }
  paintToSeq();
}

/** The studio's one link back: everything the current source could add. */
function paintToSeq() {
  const b = $('#bToSeq');
  const can = seqCandidates();
  b.hidden = S.view !== 'studio' || !can.length;
  b.textContent = S.returnToSeq ? '→ add to the sequence' : '+ to the sequence';
  $('#viewbar').dataset.pending = S.returnToSeq ? '1' : '0';
}

/** What the current source could contribute right now, as add buttons.
 *
 *  A still that has been segmented offers its subjects ONE AT A TIME as well as
 *  together and as the whole frame — a photograph with two people in it is
 *  usually two items, not one — and the cut-out entries come first because that
 *  is what someone who bothered to click on something wants. */
function seqCandidates() {
  const out = [];
  const name = S.fileName || 'this picture';
  if (S.kind === 'video' && dotsReady()) {
    const r = activeRange();
    out.push({ id: 'clip', label: 'this clip · '
      + S.subjects.length + ' subject' + (S.subjects.length > 1 ? 's' : '')
      + (r.whole ? '' : ` · frames ${r.in}–${r.out}`),
      note: 'the tracked subjects of ' + (S.fileName || 'this clip')
        + ', at the current look'
        + (r.whole ? '' : `. The entry starts trimmed to frames ${r.in}–${r.out}`
          + ' — the item keeps all ' + S.nFrames + '.') });
  }
  if (S.kind === 'image') {
    const cut = usingSubjects();
    if (cut) {
      S.subjects.forEach((sub, k) => out.push({
        id: 'still', arg: { subject: k },
        label: `this still · #${sub.id}`,
        note: `subject #${sub.id}, cut out of ${name}, as dots`,
      }));
      if (S.subjects.length > 1) {
        out.push({ id: 'still', arg: { subject: 'all' },
          label: `this still · all ${S.subjects.length}`,
          note: `every subject cut out of ${name}, as one item` });
      }
    }
    out.push({ id: 'still', arg: { whole: true },
      label: cut ? 'this still · whole picture' : 'this still',
      note: 'the whole frame of ' + name + ', as dots' });
  }
  return out;
}

async function seqAdd(what, arg) {
  const t0 = performance.now();
  busy(true);
  try {
    let item, opts = null;
    if (what === 'clip') {
      item = await captureClip();
      // the studio's active range becomes the entry's in/out. The item itself
      // holds every tracked frame, so the inspector can widen it again.
      const r = activeRange();
      opts = { in: r.in, out: r.out };
    } else if (what === 'still') {
      item = await captureStill(arg);
      if (arg && arg.subject !== undefined) opts = { subject: arg.subject };
    } else if (what === 'shape') item = await addShape(arg);
    else if (what === 'image') item = await addShape(arg.name, arg.bitmap);
    else if (what === 'lib') item = libOf(arg);
    if (!item) throw new Error('nothing to add');
    stripAdd(item, opts);
    setView('sequence', { skipPreview: true });
    await seqPreviewSafe();
    seqInfo(`added ${item.name} · ${item.tracks.length} track`
      + `${item.tracks.length > 1 ? 's' : ''} · `
      + `${item.nFrames} frame`
      + `${item.nFrames > 1 ? 's' : ''} · `
      + `${((performance.now() - t0) / 1000).toFixed(1)} s`);
    S.returnToSeq = false;
  } catch (err) {
    seqInfo('could not add that: ' + why(err), true);
  }
  busy(false);
}

function seqInfo(msg, bad) {
  const el = $('#seqinfo');
  el.hidden = false; el.classList.toggle('err', !!bad);
  el.textContent = msg;
}

/* ----------------------------------------------------------- the panel */
function renderAdd() {
  const wrap = $('#seqadd');
  wrap.textContent = '';
  const mk = (label, title, fn) => {
    const b = document.createElement('button');
    b.className = 'chip'; b.textContent = label; b.title = title || '';
    b.addEventListener('click', fn);
    wrap.append(b);
    return b;
  };
  seqCandidates().forEach((c) => mk('+ ' + c.label, c.note,
                                    () => seqAdd(c.id, c.arg)));
  mk('+ ring', 'a dithered ring', () => seqAdd('shape', 'ring'));
  mk('+ coral', 'a dithered branching form', () => seqAdd('shape', 'coral'));
  mk('+ image…', 'a picture — whole, or with a subject clicked out of it',
     () => $('#shapeFile').click());
  /* A picture waiting to be told what it is: the whole frame, or one thing
   * clicked out of it. The second answer is the studio's still-subject step,
   * because that is where the live segmentation lives — the sequence sends the
   * picture there and the header carries it back. */
  if (S.pendingImage) {
    const lbl = document.createElement('span');
    lbl.className = 'seqsep';
    lbl.textContent = `${S.pendingImage.name} — add it as`;
    wrap.append(lbl);
    mk('whole image', 'rasterised through the same dots pipeline', () => {
      const p = S.pendingImage; S.pendingImage = null;
      seqAdd('image', { name: p.name, bitmap: p.bitmap });
    }).classList.add('go');
    mk('select a subject…', 'open it in the studio and click the thing you '
       + 'want cut out', imageToStudio);
    mk('cancel', '', () => {
      S.pendingImage = null; renderAdd(); $('#seqinfo').hidden = true;
    });
  }
  if (S.library.length) {
    const lbl = document.createElement('span');
    lbl.className = 'seqsep';
    lbl.textContent = 'captured this session — click to add (again)';
    wrap.append(lbl);
    S.library.forEach((it, i) => mk(`${i + 1}. ${it.name}`,
      `${it.kind} · ${it.tracks.length} track(s) · `
        + `${it.nFrames} frame(s)`,
      () => seqAdd('lib', it.id)));
  }
  $('#seqsrc').textContent = S.library.length
    ? `${S.library.length} in the library` : '';
  $('#addnote').textContent = seqCandidates().length ? ''
    : 'Nothing loaded to capture from — add a shape, or bring in a clip or a '
      + 'picture and come back.';
}

/* =========================================================== the inspector ===
 * Clicking a card in the strip opens THAT ITEM's look — the whole of it, the
 * same controls the studio has, scoped to one item. Mode, per-subject colour,
 * the dot sliders, the mask polish, the trim. Changing any of them re-derives
 * that item's dots and nothing else's.
 *
 * What is deliberately NOT here: the background, the canvas size and the dot
 * size. Those are the sequence's, not the item's — one .dots.gz has one
 * background and one dot square, and the rail says so in as many words.
 */
function renderInspector() {
  const box = $('#seqinspect');
  box.textContent = '';
  const sel = S.sel;
  const note = (t, cls) => {
    const d = document.createElement('div');
    d.className = 'note' + (cls ? ' ' + cls : ''); d.textContent = t;
    box.append(d); return d;
  };
  if (!sel || !S.strip.length) {
    note('Click an item or a join in the strip below the stage.');
    $('#sq2sum').textContent = '';
    return;
  }
  const lbl = (t, v) => {
    const d = document.createElement('div');
    d.className = 'lbl';
    const a = document.createElement('span'); a.textContent = t;
    const b = document.createElement('span'); b.textContent = v || '';
    d.append(a, b); box.append(d);
    return b;
  };
  const slider = (min, max, step, val, onIn) => {
    const i = document.createElement('input');
    i.type = 'range'; i.min = String(min); i.max = String(max);
    i.step = String(step); i.value = String(val);
    i.addEventListener('input', () => onIn(+i.value));
    box.append(i);
    return i;
  };
  const chipRow = (seg) => {
    const d = document.createElement('div');
    d.className = 'chips' + (seg ? ' seg' : '');
    box.append(d); return d;
  };
  const chip = (row, label, on, title, fn) => {
    const b = document.createElement('button');
    b.className = 'chip'; b.textContent = label; b.title = title || '';
    b.setAttribute('aria-pressed', String(!!on));
    b.addEventListener('click', fn);
    row.append(b);
    return b;
  };

  if (sel.type === 'join') {
    const inst = S.strip[sel.i];
    $('#sq2sum').textContent = 'join ' + sel.i;
    lbl('Transition', inst.trans.kind);
    const chips = chipRow(true);
    (DP ? DP.TRANSITIONS : KINDS.map((id) => ({ id, name: id })))
      .forEach((t) => {
        const b = chip(chips, t.name, inst.trans.kind === t.id, t.note || '',
                       () => { inst.trans.kind = t.id; renderSeq(); seqPreviewSafe(); });
        b.dataset.kind = t.id;
      });
    const tn = document.createElement('div');
    tn.className = 'note';
    tn.textContent = ((DP && DP.TRANSITIONS.find((x) => x.id === inst.trans.kind))
      || {}).note || '';
    box.append(tn);
    if (inst.trans.kind !== 'cut') {
      const out = lbl('Length', inst.trans.ms + ' ms');
      slider(100, 2500, 1, inst.trans.ms, (v) => {
        inst.trans.ms = Math.round(v / 50) * 50;
        out.textContent = inst.trans.ms + ' ms';
        renderStrip();
      });
    }
    return;
  }

  const inst = S.strip[sel.i];
  const it = libOf(inst.lib);
  if (!it) { note('that item is gone'); return; }
  const look = inst.look;
  $('#sq2sum').textContent = `${sel.i + 1}. ${it.name}`;

  /* Any change to the look: keep it, re-derive this item, play again. `redraw`
   * says whether the panel itself has to be rebuilt (a chip changed what the
   * other controls are) or whether a slider is mid-drag and must not lose the
   * pointer. */
  const setLook = (patch, redraw) => {
    Object.assign(look, patch);
    if (redraw) renderSeq();
    seqTouch();
  };

  note('The look is per item. Background, dot size and canvas belong to the '
       + 'whole sequence — they are in Look, below.', 'small');
  lbl(it.kind === 'clip' ? 'Clip' : it.kind === 'still' ? 'Still' : 'Shape',
      `${it.w}×${it.h} · ${it.nFrames}f`);

  /* ---------------------------------------------------------- subject */
  if (it.tracks.length > 1) {
    lbl('Subject', inst.subject === 'all' ? 'all'
      : '#' + (it.tracks[inst.subject] || it.tracks[0]).id);
    const chips = chipRow(true);
    const mk = (v, label, col) => {
      const b = document.createElement('button');
      b.className = 'chip';
      b.setAttribute('aria-pressed', String(String(inst.subject) === String(v)));
      if (col) {
        const sw = document.createElement('span');
        sw.className = 'sw'; sw.style.background = col;
        b.classList.add('sub'); b.append(sw);
      }
      const t = document.createElement('span'); t.textContent = label;
      b.append(t);
      b.addEventListener('click', () => { inst.subject = v; renderSeq(); seqTouch(); });
      chips.append(b);
    };
    mk('all', 'all');
    it.tracks.forEach((t, k) => mk(k, '#' + t.id, trackColor(inst, it, k)));
  }

  /* ------------------------------------------------------------- mode */
  const modes = (S.meta && S.meta.modes) || [{ id: 'dots', name: 'Dots' }];
  const cur = modes.find((m) => m.id === look.mode) || modes[0];
  lbl('Mode', cur.name);
  const mrow = chipRow(true);
  modes.forEach((m) => {
    const flick = it.nFrames > 1 && S.meta && S.meta.stable
      && S.meta.stable[m.id] === false;
    const b = chip(mrow, m.name, look.mode === m.id,
                   (m.note || '') + (m.id === 'dots' ? ''
                     : ' — run on the dot grid, one pixel per cell'),
                   () => setLook(modeSwitch(look, m.id), true));
    b.dataset.mode = m.id;
    if (flick) b.classList.add('warn');
  });
  note(cur.id === 'dots'
    ? (cur.note || '') + ' — the density knobs below are its own'
    : (cur.note || '') + ': the cell tones go through the same dither the '
      + 'studio uses, so the lit cells become the dots. At cell 1 that is a '
      + 'true pixel dither — tens of thousands of dots a frame — and a '
      + 'transition thins it to ' + (DP ? DP.PARTICLE_CAP : 8000).toLocaleString()
      + ' particles for the flight, handing the rest back on arrival.');

  if (look.mode === 'errordiff') {
    lbl('Kernel', '');
    const sel2 = document.createElement('select');
    ((S.meta && S.meta.kernels) || [{ id: 'floyd-steinberg', name: 'Floyd–Steinberg' }])
      .forEach((a) => {
        const o = document.createElement('option');
        o.value = a.id; o.textContent = a.name;
        sel2.append(o);
      });
    sel2.value = look.algo;
    sel2.addEventListener('change', () => setLook({ algo: sel2.value }));
    box.append(sel2);
    const r = chipRow();
    chip(r, 'serpentine', look.serpentine,
         'alternate the scan direction each row',
         () => setLook({ serpentine: !look.serpentine }, true));
  }
  if (look.mode === 'ordered' || look.mode === 'halftone') {
    const mo = lbl('Matrix', look.matrix + '×' + look.matrix);
    const r = chipRow(true);
    [2, 4, 8, 16].forEach((n) => chip(r, n + '×' + n, look.matrix === n, '',
                                      () => setLook({ matrix: n }, true)));
    mo.textContent = look.matrix + '×' + look.matrix;
  }

  /* ------------------------------------------------------- dot sliders */
  const cellOut = lbl('Cell', look.cell + ' px');
  slider(LOOK_RANGE.cell[0], LOOK_RANGE.cell[1], 1, look.cell, (v) => {
    look.cell = v; cellOut.textContent = v + ' px'; seqTouch();
  });
  if (look.mode === 'dots') {
    const nOut = lbl('Count', look.n.toLocaleString());
    slider(LOOK_RANGE.n[0], LOOK_RANGE.n[1], 100, look.n, (v) => {
      look.n = v; nOut.textContent = v.toLocaleString(); seqTouch();
    });
    const fOut = lbl('Fill', look.fill.toFixed(2));
    slider(5, 100, 1, Math.round(look.fill * 100), (v) => {
      look.fill = v / 100; fOut.textContent = look.fill.toFixed(2); seqTouch();
    });
    const sOut = lbl('Stray', look.stray.toFixed(3));
    slider(0, 300, 1, Math.round(look.stray * 1000), (v) => {
      look.stray = v / 1000; sOut.textContent = look.stray.toFixed(3); seqTouch();
    });
    const bOut = lbl('Halo', look.band + ' cells');
    slider(LOOK_RANGE.band[0], LOOK_RANGE.band[1], 1, look.band, (v) => {
      look.band = v; bOut.textContent = v + ' cells'; seqTouch();
    });
  }
  const gOut = lbl('Gamma', look.gamma.toFixed(2));
  slider(20, 300, 1, Math.round(look.gamma * 100), (v) => {
    look.gamma = v / 100; gOut.textContent = look.gamma.toFixed(2); seqTouch();
  });
  const irow = chipRow();
  chip(irow, 'invert', look.invert, 'dots on the light half instead of the dark',
       () => setLook({ invert: !look.invert }, true));
  chip(irow, 'reseed', false, 'a different blue-noise offset for this item',
       () => setLook({ seed: 1 + Math.floor(Math.random() * 100000) }, true));

  /* ---------------------------------------------------------- colours */
  lbl('Colour', inst.color ? 'one for the item' : 'per subject');
  const cw = document.createElement('div');
  cw.className = 'chips';
  it.tracks.forEach((t, k) => {
    const lab = document.createElement('label');
    lab.className = 'chip sw1';
    const col = document.createElement('input');
    col.type = 'color';
    col.value = trackColor(inst, it, k);
    col.dataset.sub = String(t.id);
    col.addEventListener('input', () => {
      inst.color = null;
      look.colors = Object.assign({}, look.colors, { [t.id]: col.value });
      renderSeq();
    });
    lab.append(col);
    if (it.tracks.length > 1) {
      const n = document.createElement('span');
      n.textContent = '#' + t.id;
      lab.append(n);
    }
    cw.append(lab);
  });
  const reset = document.createElement('button');
  reset.className = 'chip'; reset.textContent = 'as captured';
  reset.setAttribute('aria-pressed', String(!inst.color && !Object.keys(look.colors || {}).length));
  reset.addEventListener('click', () => {
    inst.color = null; look.colors = {}; renderSeq();
  });
  cw.append(reset);
  box.append(cw);
  if (S.meta && S.meta.palettes) {
    const pw = document.createElement('div');
    pw.className = 'chips';
    S.meta.palettes.forEach((pal) => {
      const b = document.createElement('button');
      b.className = 'chip pal';
      const pv = document.createElement('span'); pv.className = 'pv';
      pal.colors.slice(0, 5).forEach((c) => {
        const sp = document.createElement('b'); sp.style.background = c; pv.append(sp);
      });
      const nm = document.createElement('span'); nm.textContent = pal.name;
      b.append(pv, nm);
      b.title = 'this item’s subjects take this palette’s colours in order';
      b.addEventListener('click', () => {
        const cols = pal.colors.length > 1 ? pal.colors.slice(1) : pal.colors;
        inst.color = null;
        look.colors = {};
        it.tracks.forEach((t, k) => { look.colors[t.id] = cols[k % cols.length]; });
        renderSeq();
      });
      pw.append(b);
    });
    box.append(pw);
  }

  /* ----------------------------------------------------------- polish */
  if (it.kind === 'clip') {
    const lit = it.tracks.filter((t) => ((look.polish || {})[t.id] | 0) > 0);
    lbl('Mask polish', lit.length ? `${lit.length}/${it.tracks.length} on` : 'off');
    it.tracks.forEach((t) => {
      const v = (look.polish || {})[t.id] | 0;
      const on = v > 0;
      const row = document.createElement('div');
      row.className = 'mini';
      const b = document.createElement('button');
      b.className = 'chip pol';
      b.setAttribute('aria-pressed', String(on));
      b.dataset.sub = String(t.id);
      const sw = document.createElement('span');
      sw.className = 'sw'; sw.style.background = t.color;
      const nm = document.createElement('span'); nm.textContent = '#' + t.id;
      b.append(sw, nm);
      const sl = document.createElement('input');
      sl.type = 'range'; sl.min = '10'; sl.max = '100'; sl.step = '5';
      sl.value = String(on ? v : 70);
      sl.disabled = !on;
      const out = document.createElement('b');
      out.textContent = on ? String(v) : 'off';
      const put = (x, redraw) => {
        const pol = Object.assign({}, look.polish);
        if (x > 0) pol[t.id] = x; else delete pol[t.id];
        setLook({ polish: pol }, redraw);
      };
      b.addEventListener('click', () => put(on ? 0 : (+sl.value || 70), true));
      sl.addEventListener('input', () => { out.textContent = sl.value; put(+sl.value); });
      row.append(b, sl, out);
      box.append(row);
    });
    note('Polish steadies this item’s masks before the dots are measured — '
         + 'the first pass costs a moment a frame.', 'small');
  }

  /* ------------------------------------------------------ trim / hold */
  if (it.nFrames > 1) {
    const io = lbl('In / out', `${inst.in} – ${inst.out} · ${stripLen(inst)}f`);
    const say = () => { io.textContent = `${inst.in} – ${inst.out} · ${stripLen(inst)}f`; };
    slider(0, it.nFrames - 1, 1, inst.in, (v) => {
      inst.in = Math.min(v, inst.out); say(); seqTouch();
    });
    slider(0, it.nFrames - 1, 1, inst.out, (v) => {
      inst.out = Math.max(v, inst.in); say(); seqTouch();
    });
  } else {
    const ho = lbl('Hold', inst.hold + ' frames');
    slider(1, 150, 1, inst.hold, (v) => {
      inst.hold = v; ho.textContent = v + ' frames'; seqTouch();
    });
  }

  const row = document.createElement('div');
  row.className = 'row';
  const rm = document.createElement('button');
  rm.className = 'btn'; rm.textContent = 'remove from the strip';
  rm.addEventListener('click', () => {
    S.strip.splice(sel.i, 1); S.sel = null; renderSeq();
    if (S.strip.length) seqPreviewSafe(); else if (PLAYER) PLAYER.pause();
  });
  row.append(rm);
  box.append(row);
}

/* ------------------------------------------------------------- the strip */
function thumb(cv, inst) {
  const it = libOf(inst.lib);
  const g = cv.getContext('2d');
  const w = cv.width, h = cv.height;
  g.fillStyle = S.seq.bg; g.fillRect(0, 0, w, h);
  if (!it) return;
  const k = Math.min(w / it.w, h / it.h);
  const ox = (w - it.w * k) / 2, oy = (h - it.h * k) / 2;
  stripTracks(inst).forEach((t) => {
    const xy = t.frames[0] || new Uint16Array(0);
    g.fillStyle = t.color;
    for (let i = 0; i < xy.length; i += 2) {
      g.fillRect(Math.round(ox + xy[i] * k), Math.round(oy + xy[i + 1] * k), 1, 1);
    }
  });
}

function renderStrip() {
  const wrap = $('#strip2');
  wrap.textContent = '';
  S.strip.forEach((inst, i) => {
    if (i > 0) {
      const j = document.createElement('button');
      j.className = 'join';
      j.dataset.i = String(i);
      j.setAttribute('aria-pressed',
                     String(S.sel && S.sel.type === 'join' && S.sel.i === i));
      const k = document.createElement('b');
      k.textContent = inst.trans.kind === 'density' ? 'density' : inst.trans.kind;
      const ms = document.createElement('em');
      ms.textContent = inst.trans.kind === 'cut' ? '—' : inst.trans.ms + ' ms';
      j.append(k, ms);
      j.title = 'click to choose the transition · shift-click cycles it';
      j.addEventListener('click', (e) => {
        if (e.shiftKey) {
          inst.trans.kind = KINDS[(KINDS.indexOf(inst.trans.kind) + 1) % KINDS.length];
        }
        S.sel = { type: 'join', i };
        renderSeq();
      });
      wrap.append(j);
    }
    const card = document.createElement('div');
    card.className = 'card';
    card.draggable = true;
    card.dataset.i = String(i);
    card.setAttribute('aria-pressed',
                      String(S.sel && S.sel.type === 'item' && S.sel.i === i));
    const cv = document.createElement('canvas');
    cv.width = 112; cv.height = 63;
    card.append(cv);
    const it = libOf(inst.lib) || { name: '?', kind: '?' };
    const nm = document.createElement('b');
    nm.textContent = `${i + 1}. ${it.name}`;
    const sub = document.createElement('em');
    const tracks = stripTracks(inst);
    sub.textContent = `${stripLen(inst)}f · `
      + (tracks.length > 1 ? `${tracks.length} subjects` : it.kind);
    card.append(nm, sub);
    card.addEventListener('click', () => { S.sel = { type: 'item', i }; renderSeq(); });
    card.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/plain', String(i));
      e.dataTransfer.effectAllowed = 'move';
      card.classList.add('drag');
    });
    card.addEventListener('dragend', () => card.classList.remove('drag'));
    card.addEventListener('dragover', (e) => { e.preventDefault(); card.classList.add('over'); });
    card.addEventListener('dragleave', () => card.classList.remove('over'));
    card.addEventListener('drop', (e) => {
      e.preventDefault();
      card.classList.remove('over');
      const from = +e.dataTransfer.getData('text/plain');
      if (Number.isNaN(from) || from === i) return;
      const [moved] = S.strip.splice(from, 1);
      S.strip.splice(i, 0, moved);
      S.sel = { type: 'item', i };
      renderSeq();
    });
    wrap.append(card);
    thumb(cv, inst);
  });
  if (!S.strip.length) {
    const empty = document.createElement('div');
    empty.className = 'seqempty';
    empty.textContent = 'nothing in the strip yet — add something on the left';
    wrap.append(empty);
  }
  const frames = S.strip.reduce((a, x) => a + stripLen(x), 0);
  const joins = Math.max(0, S.strip.length - 1);
  $('#sq1sum').textContent = S.strip.length
    ? `${S.strip.length} item${S.strip.length > 1 ? 's' : ''}` : '';
  $('#sq4sum').textContent = S.strip.length
    ? `${frames}f + ${joins} join${joins === 1 ? '' : 's'}` : '';
  /* The old warning was "these items are different sizes, so the rest will sit
   * off centre" — which they no longer do: fitTracks centres every one of them
   * on the strip's frame. What is left worth saying is the ACCIDENTAL case:
   * the frame is whatever the first item happened to be and the others are
   * being placed against it without anyone having chosen that. Choose a preset
   * and the Canvas step's own note explains what it does instead. */
  const mixed = (S.seq.preset || 'source') === 'source' && S.strip.some((x) => {
    const it = libOf(x.lib);
    return it && (it.w !== seqW() || it.h !== seqH());
  });
  $('#seqwarn').hidden = !mixed;
  if (mixed) {
    $('#seqwarn').textContent = `Some items are not ${seqW()}×${seqH()}, which `
      + "is the first item's frame — their dots are centred on it, and shrunk "
      + 'to fit only if they would not otherwise. Pick a frame below to say '
      + 'which shape you meant.';
  }
  paintSeqCanvas();
  ['#bSeqPrev', '#bSeqDots', '#bSeqVideo'].forEach((id) => {
    $(id).disabled = !S.strip.length;
  });
}

function renderSeq() {
  renderAdd();
  renderStrip();
  renderInspector();
}

/* ------------------------------------------------------ preview + export */
/* Every UI path into the preview goes through this: seqPreview() rejects so
 * window.DV_seq.preview() can be awaited in a test, and an unhandled rejection
 * from a click handler would show up as a page error. */
const seqPreviewSafe = () => seqPreview().catch(() => {});

async function seqPreview() {
  try {
    const P = await playerLib();
    const doc = await buildSeq();
    showStage('sequence');
    if (!PLAYER) {
      PLAYER = new P.Player($('#seqcv'), { loop: true, onFrame: (f) => {
        const d = S.seqDoc;
        if (!d) return;
        $('#seqframe').textContent = `${f} / ${d.frames.length - 1}`;
        const m = d.marks.filter((x) => x.start <= f).pop();
        $('#seqmark').textContent = m ? (m.kind === 'item' ? m.name : m.kind) : '';
      } });
    }
    PLAYER.setDoc(doc);
    PLAYER.play();
    $('#bSeqPlay').textContent = 'pause';
    renderStrip();          // the cards read the same cache the build just filled
    const joins = doc.marks.filter((m) => m.kind !== 'item');
    seqInfo(`${doc.frames.length} frames · ${doc.fps} fps · `
      + `${(doc.frames.length / doc.fps).toFixed(1)} s · `
      + `${doc.subjects.length} colour${doc.subjects.length > 1 ? 's' : ''}`
      + (joins.length ? ' · ' + joins.map((m) => `${m.kind} ${m.frames}f`).join(' · ')
        : ''));
    return doc;
  } catch (err) {
    seqInfo('preview failed: ' + why(err), true);
    throw err;
  }
}

async function seqExportDots() {
  try {
    const P = await playerLib();
    const doc = await buildSeq();
    const bytes = await P.pack(doc);
    download(new Blob([bytes], { type: 'application/octet-stream' }),
             'sequence.dots.gz');
    seqInfo(`${doc.frames.length} frames · `
      + `${(bytes.length / 1024).toFixed(0)} KB .dots.gz`);
  } catch (err) {
    seqInfo('export failed: ' + why(err), true);
  }
}

/** A sequence as a video.
 *
 *  With a server: the finished dot positions go up to /api/sequence and it
 *  rasterises them into any of the five containers. Without one: the SAME
 *  encoder path a clip export uses in the tab — `engine.exportClip` with a
 *  `source` override and a renderFrame that paints sequence frames — so a
 *  sequence gets the browser's GIF and alpha WebM for free rather than a
 *  second, weaker recorder of its own. */
async function seqExportVideo() {
  const btn = $('#bSeqVideo'); btn.disabled = true;
  try {
    const P = await playerLib();
    const doc = await buildSeq();
    const fmt = seqFormat();
    let r;
    if (E().renderSequence) {
      seqInfo('rendering on the server…');
      const bytes = await P.pack(doc);
      r = await E().renderSequence(bytes, fmt.id);
    } else {
      seqInfo('encoding in the tab…');
      const img = new ImageData(doc.w, doc.h);
      const opts = fmt.alpha ? { bg: null, transparent: true } : { bg: doc.bg };
      r = await E().exportClip({
        format: fmt.id, fps: doc.fps, gif_fps: S.gifFps, bg: doc.bg,
        palette: doc.palette, subjects: [],
        source: { w: doc.w, h: doc.h, nFrames: doc.frames.length, fps: doc.fps },
      }, (pr) => seqInfo(`encoding ${pr.text || pr.done + '/' + pr.total}`),
         (i) => {
           P.paintFrame(img.data, doc.w, doc.h, doc, doc.frames[i], opts);
           return img;
         });
    }
    const a = $('#seqdl');
    a.href = r.url; a.download = 'sequence.' + r.ext; a.hidden = false;
    const v = $('#seqvid');
    if (r.playable !== false && r.ext !== 'gif' && r.ext !== 'mov') {
      v.src = r.url; v.hidden = false;
    } else { v.hidden = true; }
    seqInfo(`${r.frames} frames · ${r.ext.toUpperCase()} · `
      + `${(r.bytes / 1e6).toFixed(2)} MB`
      + (r.elapsedS ? ` · ${r.elapsedS.toFixed(1)} s` : '')
      + (r.note ? ` · ${r.note}` : ''));
  } catch (err) {
    seqInfo('render failed: ' + why(err), true);
  }
  btn.disabled = false;
}

/* --------------------------------------------------- the sequence's look */
function seqFormat() {
  const list = engineFormats();
  return list.find((f) => f.id === S.seq.format && f.available)
    || list.find((f) => f.available) || { id: 'webm', ext: 'webm' };
}

function buildSeqFormats() {
  const sel = $('#sSeqFmt'), list = engineFormats();
  $('#seqfmt').hidden = !list.length;
  sel.textContent = '';
  list.forEach((f) => {
    const o = document.createElement('option');
    o.value = f.id;
    o.textContent = f.label + (f.available ? '' : ' — unavailable here');
    o.disabled = !f.available;
    sel.append(o);
  });
  sel.value = seqFormat().id;
  S.seq.format = sel.value;
  $('#vSeqFmt').textContent = (seqFormat().ext || '').toUpperCase();
}

/* The strip's frame, as chips. No custom entry here: a sequence's size has to
 * be one number for the whole document and the presets are what anyone
 * actually wants; a bespoke one is a studio export's job. */
function buildSeqPresets() {
  const wrap = $('#seqpresets');
  if (!wrap || wrap.childElementCount) return;
  CV.PRESETS.filter((p) => p.id !== 'custom').forEach((p) => {
    const b = document.createElement('button');
    b.className = 'chip'; b.dataset.preset = p.id; b.textContent = p.label;
    b.title = p.note;
    b.addEventListener('click', () => {
      S.seq.preset = p.id;
      paintSeqCanvas();
      seqTouch(true);
    });
    wrap.append(b);
  });
}

function paintSeqCanvas() {
  const wrap = $('#seqpresets');
  if (!wrap) return;
  buildSeqPresets();
  $$('#seqpresets .chip').forEach((b) => b.setAttribute(
    'aria-pressed', String(b.dataset.preset === (S.seq.preset || 'source'))));
  const src = seqSource();
  $('#vSeqSize').textContent = `${seqW()}×${seqH()}`;
  $('#sq3sum').textContent = `${seqW()}×${seqH()}`;
  $('#seqcanvasnote').textContent = S.seq.preset && S.seq.preset !== 'source'
    ? `Every frame of the sequence — the preview, the .dots.gz and the video — `
      + `is ${seqW()}×${seqH()}. Each item's cloud is centred on it, and shrunk `
      + 'only if it would not fit; nothing is magnified, because spreading dots '
      + 'apart does not make them bigger.'
    : `The first item's own frame: ${src.w}×${src.h}.`;
}

function renderSeqPalettes() {
  const wrap = $('#seqpals');
  if (!wrap || !S.meta) return;
  wrap.textContent = '';
  S.meta.palettes.forEach((p) => {
    const b = document.createElement('button');
    b.className = 'chip pal';
    const pv = document.createElement('span'); pv.className = 'pv';
    p.colors.slice(0, 5).forEach((c) => {
      const s = document.createElement('b'); s.style.background = c; pv.append(s);
    });
    const nm = document.createElement('span'); nm.textContent = p.name;
    b.append(pv, nm);
    b.title = 'background from the first colour, item colours from the rest';
    b.addEventListener('click', () => {
      // the background is the palette's darkest end, the items take the rest in
      // order — so one click re-colours the whole strip and keeps items apart
      const cols = p.colors.slice();
      S.seq.bg = cols[0];
      $('#cSeqBg').value = cols[0];
      const rest = cols.length > 1 ? cols.slice(1) : cols;
      S.strip.forEach((inst, i) => { inst.color = rest[i % rest.length]; });
      renderSeq();
    });
    wrap.append(b);
  });
}

/* ------------------------------------------------------------- handlers */
$$('#viewbar .chip[data-view]').forEach((b) => b.addEventListener('click', () => {
  setView(b.dataset.view);
}));
$('#bToSeq').addEventListener('click', () => {
  const can = seqCandidates();
  if (can.length) seqAdd(can[0].id, can[0].arg);
});
$('#bSeqNew').addEventListener('click', () => {
  S.returnToSeq = true;
  setView('studio');
  openStep(1);
  toast('drop or record something — then "add to the sequence"');
});
$('#shapeFile').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  try {
    S.pendingImage = { file: f, name: f.name.replace(/\.[^.]+$/, ''),
                       bitmap: await createImageBitmap(f) };
    renderAdd();
    seqInfo('whole picture, or click a subject out of it?');
  } catch (err) { toast('could not read that image: ' + err.message, true); }
  e.target.value = '';
});

/** The other answer: send the picture to the studio's still-subject step and
 *  let the header bring the cutout back. One segmentation flow, not two. */
async function imageToStudio() {
  const p = S.pendingImage;
  if (!p) return;
  S.pendingImage = null;
  S.returnToSeq = true;
  setView('studio');
  busy(true);
  try {
    await take(p.file);
    openStep(2);
    if (S.scope !== 'track') setScope('track');
    toast('click what you want cut out — then "→ add to the sequence"');
  } catch (err) {
    toast('could not open that image: ' + why(err), true);
  }
  busy(false);
}
$('#bSeqClear').addEventListener('click', () => {
  S.strip = []; S.sel = null; S.seqDoc = null;
  if (PLAYER) PLAYER.pause();
  renderSeq();
  seqInfo('the strip is empty — the library kept everything');
});
$('#bSeqPrev').addEventListener('click', seqPreviewSafe);
$('#bSeqDots').addEventListener('click', seqExportDots);
$('#bSeqVideo').addEventListener('click', seqExportVideo);
$('#bSeqPlay').addEventListener('click', () => {
  if (!PLAYER) return;
  PLAYER.toggle();
  $('#bSeqPlay').textContent = PLAYER.playing ? 'pause' : 'play';
});
$('#sSeqFmt').addEventListener('change', (e) => {
  S.seq.format = e.target.value;
  $('#vSeqFmt').textContent = (seqFormat().ext || '').toUpperCase();
});
$('#sSeqDot').addEventListener('input', (e) => {
  S.seq.dotpx = +e.target.value;
  $('#vSeqDot').textContent = S.seq.dotpx + ' px';
  if (PLAYER && S.seqDoc) PLAYER.set({ dotpx: S.seq.dotpx });
  renderStrip();
});
$('#cSeqBg').addEventListener('input', (e) => {
  S.seq.bg = e.target.value;
  if (PLAYER && S.seqDoc) PLAYER.set({ bg: S.seq.bg });
  renderStrip();
});

/* ===================================================== the storage line
 * jobs/ is the server's scratch directory: frames, masks and renders for every
 * clip anyone ever dropped. It swept itself into 5.4 GB in two days before the
 * janitor existed, so the number is worth showing next to the button that
 * hands it back. Server engine only -- the browser engine keeps nothing on
 * disk, and a server too old to advertise `gc` is never asked.
 */
const gcSize = (mb) => (mb >= 1024 ? (mb / 1024).toFixed(1) + ' GB'
                                   : Math.round(mb) + ' MB');

async function paintStorage() {
  const bar = $('#gcbar');
  if (E().id === 'browser' || !(S.meta && S.meta.gc)) { bar.hidden = true; return; }
  try {
    const g = await E().api('/api/gc/status');
    $('#gcuse').textContent = `storage: ${gcSize(g.usage_mb)} · ${g.jobs} job`
      + (g.jobs === 1 ? '' : 's') + (g.over_budget ? ' · over budget' : '');
    $('#gcbar').title = `jobs/ on ${E().baseUrl || 'this machine'}. `
      + `Swept every ${g.every_h} h: anything untouched for ${g.max_age_days} days `
      + `goes, then the oldest until it fits ${gcSize(g.budget_mb)}. `
      + `Nothing used in the last ${g.keep_hours} h is ever touched, and a `
      + 'camera recording keeps its original.';
    bar.hidden = false;
  } catch (e) {
    bar.hidden = true;
  }
}

$('#bGC').addEventListener('click', async () => {
  const b = $('#bGC');
  b.disabled = true; b.textContent = 'sweeping…';
  try {
    const g = await E().api('/api/gc/run', { method: 'POST' });
    const r = g.ran || {};
    const n = (r.deleted || []).length, t = (r.trimmed || []).length;
    toast(n || t
      ? `freed ${gcSize((r.freed_bytes || 0) / 1048576)} · ${n} job`
        + `${n === 1 ? '' : 's'} deleted${t ? `, ${t} trimmed to the original` : ''}`
      : 'nothing to clean up yet — everything here is recent');
  } catch (err) {
    toast(why(err), true);
  }
  await paintStorage();
  b.disabled = false; b.textContent = 'clean up';
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
  S.srcW = 0; S.srcH = 0; S.srcJob = null;
  S.range = null; S.extend = null; S.jobStart = 0;
  // the SHAPE is kept — someone making a set of 9:16 clips should not have to
  // say so again for each one — but the hand-nudged bias and the crop path
  // belong to the source that has just gone
  S.canvas.dx = 0; S.canvas.dy = 0;
  CPATH.key = null; CPATH.centers = null; CPATH.union = null;
  paintRange(); paintTrimOffer();
  $('#vidopts').hidden = true;
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
  buildLookRow();
  scheduleLookThumbs(true);
  await checkModels();
  paintStorage();
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

/* ============================================== first-run: the demo =======
 * The landing is already running: a prebaked demo.dots.gz plays in the hero
 * (the shipped player — no model, no GPU, works in every webview), and the
 * remove.bg-style sample row skips a first-timer straight into the flow.
 * The 30-second script: tap the sample clip → one pulsing in-canvas hint →
 * tap the athlete → instant tint → Track → Solvd applies itself → playing.
 * Any real action dismisses the hint; nothing here is a modal. (spec §5) */
let HERO = null;
const DEMO = { hint: false, at: [545, 205] };   // the athlete, in clip pixels

function showDemoHint() {
  const el = $('#demohint');
  if (!el || !DEMO.hint) return;
  el.style.left = (DEMO.at[0] / (S.W || 1280) * 100) + '%';
  el.style.top = (DEMO.at[1] / (S.H || 720) * 100) + '%';
  el.hidden = false;
}
function hideDemoHint() {
  DEMO.hint = false;
  const el = $('#demohint');
  if (el) el.hidden = true;
}

async function tryDemo(kind) {
  busy(true);
  try {
    const name = kind === 'still' ? 'sample.jpg' : 'sample.mp4';
    const res = await fetch('./demo/' + name);
    if (!res.ok) throw new Error('the sample is not on this host');
    const blob = await res.blob();
    const f = new File([blob], name,
                       { type: kind === 'still' ? 'image/jpeg' : 'video/mp4' });
    S.demoRun = kind === 'clip';
    await take(f);
    // straight to the subject step, with the one hint — unless the models are
    // missing, in which case the step's own plain-language note explains
    if (!S.modelsMissing) {
      setScope('track');
      openStep(2);
      DEMO.hint = true;
      showDemoHint();
    }
  } catch (err) {
    toast(why(err), true);
    S.demoRun = false;
  }
  busy(false);
}

function paintHeroLooks() {
  const wrap = $('#herolooks');
  if (!wrap || wrap.childElementCount) return;
  const picks = ['solvd', 'gameboy', 'newsprint', 'blueprint', 'ember', 'terminal'];
  const tiles = [];
  picks.forEach((id) => {
    const l = LOOKS.find((x) => x.id === id);
    if (!l) return;
    const fig = document.createElement('figure');
    const cv = document.createElement('canvas');
    cv.width = LT.w; cv.height = LT.h;
    const cap = document.createElement('figcaption');
    cap.textContent = l.name;
    fig.append(cv, cap);
    wrap.append(fig);
    tiles.push([cv, l]);
  });
  lookThumbSource().then((src) => {
    tiles.forEach(([cv, l]) => renderLookThumb(cv, l, src));
  }).catch(() => {});
}

async function initHero() {
  paintHeroLooks();
  const ring = $('#dShapeCv');
  if (ring) drawRing(ring.getContext('2d'), ring.width, ring.height);
  const hc = $('#herocv');
  if (!hc) return;
  try {
    const res = await fetch('./demo/demo.dots.gz');
    if (!res.ok) throw new Error('no demo asset');
    const bytes = new Uint8Array(await res.arrayBuffer());
    const P = await playerLib();
    const doc = await P.unpack(bytes);
    HERO = new P.Player(hc, { loop: true });
    HERO.setDoc(doc);
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) HERO.draw(0);
    else if (!$('#empty').hidden) HERO.play();
    else HERO.draw(0);
  } catch (e) {
    const w = $('#herowrap');
    if (w) w.hidden = true;      // a static host without the asset: plain hero
  }
}
$('#bPickCam') && $('#bPickCam').addEventListener('click', () => camOpen());
$('#dTryClip') && $('#dTryClip').addEventListener('click', () => tryDemo('clip'));
$('#dTryStill') && $('#dTryStill').addEventListener('click', () => tryDemo('still'));
$('#dTryShape') && $('#dTryShape').addEventListener('click', () => seqAdd('shape', 'ring'));

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
  S.seq.dotpx = S.P.dotpx;
  S.seq.bg = S.bg;
  $('#cSeqBg').value = S.bg;
  renderSeqPalettes();
  setView('studio');
  showSteps('none');
  renderSeq();
  window.DV = S;
  window.DV_maskURL = subjectMaskDataURL;
  // an explicit frame request pauses playback first — scrubbing means pause,
  // and the verifiers depend on the frame they asked for staying put
  window.DV_draw = (i) => { if (i !== undefined) stop(); return draw(i); };
  window.DV_engine = () => ({ id: S.engine.id, label: S.engine.label,
                              baseUrl: S.engine.baseUrl || '',
                              supports: S.engine.supports,
                              modelsMissing: S.modelsMissing,
                              tried: S.engineTried });
  window.DV_switchEngine = switchEngine;
  window.DV_composeAt = composeAt;
  /* the undithered frame the matched cut is made of — the verifiers compare
     the exported original against this, which is the only ground truth the
     browser engine has (the server has jobs/<id>/frames/ on disk) */
  window.DV_originalAt = originalAt;
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
  /* mask polish, for the verifiers and for anyone driving the page from a
   * console: strengths in, the algorithm's own numbers out. */
  window.DV_polish = {
    set: (id, strength) => {
      const k = S.subjects.findIndex((x) => String(x.id) === String(id));
      if (k < 0) throw new Error('no subject ' + id);
      setPolish(k, strength);
      return S.subjects[k].polish;
    },
    all: (strength) => { S.subjects.forEach((x, k) => setPolish(k, strength));
                         return S.subjects.map((x) => x.polish); },
    get: () => S.subjects.map((x) => ({ id: x.id, polish: x.polish | 0 })),
    params: (strength) => MaskPolish.params(strength),
    /* the polished mask for one subject on one frame, 8-bit, as the renderer
     * sees it — the browser half of the preview/export parity check */
    mask: async (id, frame) => {
      const k = S.subjects.findIndex((x) => String(x.id) === String(id));
      checkPolishKey();
      const m = await polishedMask(PM, S.subjects[k].id, frame,
                                   S.subjects[k].polish | 0, 'pf' + k);
      const u8 = new Uint8Array(m.length);
      for (let q = 0; q < m.length; q++) u8[q] = Math.round(m[q] * 255);
      return Array.from(u8);
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
  /* The clip-length story, callable. There is no cap to set, so what a test
   * needs instead is (a) a way to ask for a shorter range before the drop,
   * (b) the numbers the estimate line is showing, and (c) the re-cut that
   * "use this range" does after a clip is already open. */
  window.DV_limit = (seconds) => { S.pendingLimit = +seconds || 0; return S.pendingLimit; };
  window.DV_estimate = () => Object.assign(clipEstimate(), {
    longNote: $('#estlong').hidden ? '' : $('#estlong').textContent,
    memNote: $('#estwarn').hidden ? '' : $('#estwarn').textContent,
    line: $('#estline').textContent,
    head: $('#vEst').textContent,
    longThresholdS: LONG_S,
  });
  /* The trim bar's two buttons, so a test can drive either without a mouse.
   * `recut` is what makes a second trim free: same file, no upload. */
  window.DV_useRange = (start, end) => {
    if (start !== undefined) S.trim = { start, end };
    paintTrim();
    return $('#bTrim').click();
  };
  window.DV_wholeClip = () => $('#bTrimAll').click();
  /* The range, callable. `set` takes FRAME indices (what the exports use);
   * `seconds` takes the trim bar's own seconds and goes through exactly the
   * button's decision, so a test can prove that narrowing after a track fires
   * no /track and no /reextract. `offer` is what the extend panel is saying. */
  window.DV_range = {
    get: () => Object.assign(activeRange(), { jobStart: S.jobStart,
                                              nFrames: S.nFrames,
                                              fps: S.fps }),
    set: (a, b) => setRange(a, b),
    full: () => { $('#bRangeAll').click(); return activeRange(); },
    seconds: (start, end) => { S.trim = { start, end }; paintTrim();
                               return applyTrim(S.trim); },
    plan: (start, end) => trimPlan({ start, end }),
    offer: () => ($('#trimoffer').hidden ? null : {
      note: $('#trimoffernote').textContent,
      button: $('#bExtend').textContent,
      missing: S.extend ? S.extend.missing : null,
      total: S.extend ? S.extend.total : 0,
    }),
    extend: () => extendAndTrack(),
    label: () => ($('#rangelbl').hidden ? '' : $('#rangelbl').textContent),
    resetShown: () => !$('#bRangeAll').hidden,
  };
  window.DV_dots = { doc: dotsDoc, params: dotsParams, lib: playerLib,
                     library: () => S.library, build: buildSeq,
                     preview: seqPreview, player: () => PLAYER };
  /* The sequence view, for the verifiers: everything the UI does, callable. */
  window.DV_seq = {
    view: (v) => { setView(v); return S.view; },
    add: (what, arg) => seqAdd(what, arg),
    library: () => S.library.map((x) => ({ id: x.id, name: x.name, kind: x.kind,
                                           w: x.w, h: x.h, nFrames: x.nFrames,
                                           look: x.look,
                                           tracks: x.tracks.map((t) => ({
                                             id: t.id, color: t.color,
                                             frames: x.nFrames })) })),
    strip: () => S.strip.map((x, i) => {
      const it = libOf(x.lib) || {};
      return { i, lib: x.lib, name: it.name, kind: it.kind, subject: x.subject,
               in: x.in, out: x.out, hold: x.hold, color: x.color,
               look: cloneLook(x.look),
               colors: it.tracks ? it.tracks.map((t, k) => trackColor(x, it, k)) : [],
               frames: stripLen(x), trans: i > 0 ? x.trans : null };
    }),
    set: (i, opts) => { Object.assign(S.strip[i], opts); renderSeq();
                        return S.strip[i]; },
    /* one item's own look — the inspector's controls, callable */
    itemLook: (i) => cloneLook(S.strip[i].look),
    setLook: async (i, patch) => {
      const look = S.strip[i].look, p = patch || {};
      // same coupling the mode chips have: picking a mode picks the cell that
      // suits it, unless the caller says otherwise in the same breath
      Object.assign(look,
        (p.mode && p.cell === undefined) ? modeSwitch(look, p.mode) : {}, p);
      renderSeq();
      const doc = await buildSeq();
      renderStrip();
      return { look: cloneLook(S.strip[i].look), frames: doc.frames.length };
    },
    /* the dots one item is actually made of right now, frame by frame: the
     * cache the preview and both exports all read */
    itemDots: (i) => {
      const inst = S.strip[i];
      return stripTracks(inst).map((t) => t.frames.map((f) => Array.from(f)));
    },
    modes: () => (S.meta && S.meta.modes) || [],
    /* what "+ image…" is waiting to be told, and the two answers */
    pending: () => (S.pendingImage ? S.pendingImage.name : null),
    candidates: () => seqCandidates().map((c) => ({ id: c.id, arg: c.arg || null,
                                                    label: c.label })),
    cap: () => (DP ? DP.PARTICLE_CAP : null),
    trans: (i, kind, ms) => {
      const inst = S.strip[i];
      if (!inst || !i) throw new Error('no join before item ' + i);
      inst.trans = { kind, ms: ms === undefined ? inst.trans.ms : ms };
      renderSeq();
      return inst.trans;
    },
    move: (from, to) => { const [m] = S.strip.splice(from, 1);
                          S.strip.splice(to, 0, m); renderSeq();
                          return S.strip.map((x) => x.lib); },
    select: (type, i) => { S.sel = { type, i }; renderSeq(); return S.sel; },
    /* the strip's own frame size — the same preset table as the studio's */
    canvas: (preset) => {
      if (preset) S.seq.preset = preset;
      paintSeqCanvas();
      return { preset: S.seq.preset || 'source', w: seqW(), h: seqH(),
               source: seqSource() };
    },
    look: (o) => {
      Object.assign(S.seq, o || {});
      $('#sSeqDot').value = String(S.seq.dotpx);
      $('#vSeqDot').textContent = S.seq.dotpx + ' px';
      $('#cSeqBg').value = S.seq.bg;
      if (PLAYER && S.seqDoc) PLAYER.set({ dotpx: S.seq.dotpx, bg: S.seq.bg });
      renderStrip();
      return S.seq;
    },
    build: buildSeq,
    preview: seqPreview,
    dots: seqExportDots,
    video: seqExportVideo,
    format: (id) => { S.seq.format = id; $('#sSeqFmt').value = id;
                      $('#vSeqFmt').textContent = (seqFormat().ext || '').toUpperCase();
                      return seqFormat(); },
    doc: () => S.seqDoc,
    player: () => PLAYER,
    transitions: () => (DP ? DP.TRANSITIONS : null),
  };
  /* The canvas, callable: the aspect-ratio control, its framing decision and
   * the map it produces, so a verifier can assert on the geometry rather than
   * on a screenshot. */
  window.DV_canvas = {
    presets: () => CV.PRESETS.map((p) => ({ id: p.id, label: p.label,
                                            w: p.w || 0, h: p.h || 0 })),
    get: () => Object.assign({}, S.canvas, {
      target: canvasTarget(), framing: framing(), clamps: canvasClamps(),
      label: canvasLabel(), slug: canvasSlug() }),
    set: async (preset, opts) => {
      if (opts) Object.assign(S.canvas, opts);
      if (preset) { S.canvas.preset = preset; S.canvas.dx = 0; S.canvas.dy = 0; }
      await applyCanvas();
      return window.DV_canvas.get();
    },
    framing: async (mode) => { S.canvas.follow = mode; await applyCanvas();
                               return framing(); },
    nudge: async (dx, dy) => { S.canvas.dx = dx; S.canvas.dy = dy;
                               await applyCanvas();
                               return { dx: S.canvas.dx, dy: S.canvas.dy }; },
    /* the crop centre, in source pixels, for one frame — what a test compares
     * against the subject's own mask centroid */
    at: (i) => {
      const [sw, sh] = srcSize();
      const p = canvasPlanAt(i | 0, sw, sh);
      return p ? { k: p.k, x0: p.x0, y0: p.y0, cx: p.cx, cy: p.cy,
                   tw: p.tw, th: p.th, sw, sh } : null;
    },
    path: async () => {
      await ensureCanvasPath();
      return { n: CPATH.n, mode: framing(), union: CPATH.union,
               centers: CPATH.centers };
    },
    payload: (rng) => canvasPayload(rng),
    note: () => $('#canvasnote').textContent,
  };
  window.DV_setFormat = (id) => {
    S.format = id; $('#sFmt').value = id; paintFormat();
    return currentFormat();
  };
  initHero();
  window.DV_ready = true;
})();

/* ===================================================== the mobile shell ===
 * Below 768 px: #panel is a bottom sheet with three detents, the five steps
 * get a tab bar, the canvas keeps the rest. Pure additive controller — the
 * tabs call the same openStep()/setView() the desktop headers call, the sheet
 * repositions #panel with a transform, and at desktop widths every listener
 * here is inert. No DOM id moves, no state renames. (docs/ux-spec.md §2.2) */
(function mobileShell() {
  const tabs = $('#mtabs'), grab = $('#grab'), panel = $('#panel');
  if (!tabs || !grab || !panel) return;
  const mq = window.matchMedia('(max-width: 767px)');

  function setSheet(d) {
    document.body.dataset.sheet = d;
    panel.style.transform = '';
    document.body.classList.remove('sheetdrag');
  }
  window.DV_sheet = setSheet;            // additive hook; track() peeks it

  function paintTabs() {
    const open = $$('#studiopanel .step').find(
      (el) => el.getAttribute('data-open') === '1');
    const cur = S.view === 'sequence' ? 'seq' : (open ? open.id.slice(2) : '1');
    $$('#mtabs .tab').forEach((b) => {
      b.setAttribute('aria-pressed', String(b.dataset.mtab === cur));
      if (b.dataset.mtab !== 'seq') {
        const st = $('#st' + b.dataset.mtab);
        b.classList.toggle('off', !!(st && st.hidden));
      }
    });
  }
  $$('#mtabs .tab').forEach((b) => b.addEventListener('click', () => {
    const t = b.dataset.mtab;
    if (t === 'seq') setView('sequence');
    else {
      if (S.view === 'sequence') setView('studio');
      openStep(+t);
    }
    setSheet('half');
    paintTabs();
  }));
  // whatever opens a step — header tap, code, a verifier — the tabs follow
  new MutationObserver(paintTabs).observe(panel, {
    subtree: true, attributes: true, attributeFilter: ['data-open', 'hidden'] });

  /* drag the grabber between detents; a plain tap toggles collapsed ⁄ half */
  let drag = null;
  const tabH = () => (tabs.getBoundingClientRect().height || 56);
  grab.addEventListener('pointerdown', (e) => {
    if (!mq.matches) return;
    grab.setPointerCapture(e.pointerId);
    const natTop = window.innerHeight - tabH() - panel.offsetHeight;
    drag = { y0: e.clientY, ty0: panel.getBoundingClientRect().top - natTop,
             moved: false };
    document.body.classList.add('sheetdrag');
    e.preventDefault();
  });
  grab.addEventListener('pointermove', (e) => {
    if (!drag) return;
    const dy = e.clientY - drag.y0;
    if (Math.abs(dy) > 5) drag.moved = true;
    const ty = clamp(drag.ty0 + dy, 0, panel.offsetHeight - 46);
    panel.style.transform = `translateY(${ty}px)`;
  });
  const dragEnd = (e) => {
    if (!drag) return;
    const d = drag; drag = null;
    if (!d.moved) {
      setSheet(document.body.dataset.sheet === 'collapsed' ? 'half' : 'collapsed');
      return;
    }
    const ty = clamp(d.ty0 + (e.clientY - d.y0), 0, panel.offsetHeight - 46);
    const visible = panel.offsetHeight - ty;
    setSheet(visible < panel.offsetHeight * 0.3 ? 'collapsed'
      : visible < panel.offsetHeight * 0.78 ? 'half' : 'full');
  };
  grab.addEventListener('pointerup', dragEnd);
  grab.addEventListener('pointercancel', dragEnd);

  /* touching the picture asks for the picture: the sheet gets out of the way */
  $('#stage').addEventListener('pointerdown', (e) => {
    if (!mq.matches) return;
    if (e.target.closest('#transport') || e.target.closest('#strip2')
        || e.target.closest('#toast') || e.target.closest('#pathbtns')) return;
    if (document.body.dataset.sheet !== 'collapsed') setSheet('collapsed');
  }, { capture: true, passive: true });

  const boot = () => { if (mq.matches) { setSheet('half'); paintTabs(); } };
  mq.addEventListener ? mq.addEventListener('change', boot) : 0;
  boot();
})();
