#!/usr/bin/env python3
"""Dither Video -- local FastAPI backend.

    upload video -> frames -> EdgeTAM point/box tracking -> blue-noise dither -> mp4

Everything lives under jobs/<job-id>/. Nothing leaves the machine.
"""
import base64
import contextlib
import io
import json
import os
import shutil
import sys
import subprocess
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

import dither as DI
import dots as DT
import edgetam_util as EU
import jobsgc
import polish as PL
import render as R

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repo root: web/, env/, jobs/ live there
# coreml/ is model tooling and lives at the repo root, next to onnxexport/,
# not inside server/ -- so the root has to be importable or the CoreML
# accelerator silently falls through to torch.
sys.path.insert(0, ROOT)
JOBS = os.path.join(ROOT, "jobs")
WEB = os.path.join(ROOT, "web")
CKPT = os.path.join(ROOT, "env", "EdgeTAM", "checkpoints", "edgetam.pt")
CFG = "configs/edgetam.yaml"
DEVICE = os.environ.get("DV_DEVICE", "mps")
COREML_DIR = os.path.join(ROOT, "env", "coreml")
MAX_OBJECTS = 6
API_VERSION = 2
# Frames on disk, measured: a 1280x720 `-q:v 3` JPEG out of ffmpeg is ~90 KB.
# The disk check below scales it by the real pixel count.
JPEG_BYTES_PER_FRAME = 90_000
# .dots.gz carries n_frames as a uint16 (see dots.py). 65,535 frames is 36
# minutes at 30 fps; the encoders raise rather than silently wrap.
DOTS_MAX_FRAMES = 65535
# Frames are decoded to this height (width follows the source aspect). 720 is
# the historical value and stays the default; a 9:16 clip wants 1280 so its
# masks come back at the resolution the phone will see.
DECODE_HEIGHT = max(64, int(os.environ.get("DV_DECODE_HEIGHT", "720")))

# Optional shared secret. Unset (the normal local case) = wide open on
# 127.0.0.1. Set it and every /api/* call must carry `Authorization: Bearer
# <key>`; that is the whole of what a rented-GPU deployment needs from this
# file. There is no billing, no accounts and no rate limiting here on purpose
# -- put those in front of it if you sell access.
API_KEY = os.environ.get("DV_API_KEY", "").strip()
# A page served from somewhere else (GitHub Pages, a file:// build) has to be
# able to reach this API, so CORS is open by default. Narrow it with
# DV_CORS_ORIGINS="https://example.com,https://other.example".
CORS_ORIGINS = [o.strip() for o in
                os.environ.get("DV_CORS_ORIGINS", "*").split(",") if o.strip()]

# Tracking backends, fastest first. `DV_BACKEND` picks one; the default walks
# down the list until one builds, so a machine without the CoreML export (or
# without coremltools) still runs. Measured on the reference clip, 1 subject,
# 150 frames at 1280x720, M4 Pro -- see bench/results.md and the README.
#
#   coreml          15.0 fps   image encoder + memory attention + memory encoder
#                              as CoreML graphs, the rest on MPS. IoU 0.997 vs
#                              fp32, worst frame 0.958.
#   torch-compiled  11.6 fps   torch.compile on the image encoder; the first
#                              track in a process pays ~20 s of inductor time.
#   torch-half       9.3 fps   model.half() under autocast.
#   torch            8.3 fps   fp16 autocast on fp32 weights (the old default).
#   torch-fp32       8.0 fps   no autocast; this is the mask reference.
BACKENDS = ("coreml", "torch-compiled", "torch-half", "torch", "torch-fp32")

# The tracker resizes every frame to a square before it sees it; that square is
# the single biggest speed/quality knob in the model. The clip itself is never
# touched -- masks come back at the source resolution either way. fps is the
# median of the CoreML backend on the reference clip, 1 subject.
TRACK_SIZES = [
    {"size": 512,  "id": "fast",     "label": "fast · prototyping", "fps": 27.0},
    {"size": 768,  "id": "balanced", "label": "balanced · default", "fps": 20.9},
    {"size": 1024, "id": "best",     "label": "best · production", "fps": 13.9},
]
# 768 by default (user decision 2026-08-19): visually indistinguishable from
# 1024 in the dithered result at 1.5x the speed -- see bench/res_compare_sheet.png.
# 512 = quick prototyping, 1024 = production renders.
DEFAULT_TRACK_SIZE = 768
BACKEND = os.environ.get("DV_BACKEND", "").strip().lower()
if not BACKEND:
    BACKEND = "torch-fp32" if os.environ.get("DV_FP32", "0") == "1" else "auto"
if DEVICE != "mps" and BACKEND in ("auto", "coreml"):
    BACKEND = "torch-fp32" if DEVICE == "cpu" else BACKEND

os.makedirs(JOBS, exist_ok=True)

# jobs/ is a scratch directory and nothing here ever removed anything from it,
# so two days of use grew it to 5.4 GB. The janitor sweeps on startup and every
# DV_JOBS_GC_EVERY_H hours; see server/jobsgc.py for the whole policy.
GC = jobsgc.GC(JOBS)


@contextlib.asynccontextmanager
async def _lifespan(_app):
    GC.start()
    yield


app = FastAPI(title="Dither Studio", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_credentials=False, allow_methods=["*"],
                   allow_headers=["*"], expose_headers=["Content-Length"])


@app.middleware("http")
async def _require_key(request: Request, call_next):
    """Gate /api/* behind DV_API_KEY when one is set. The page itself, the
    static assets and the CORS preflight stay open -- a browser cannot put a
    header on the request that loads the HTML."""
    path = request.url.path
    if API_KEY and path.startswith("/api/") and request.method != "OPTIONS":
        got = request.headers.get("authorization", "")
        if got[:7].lower() != "bearer " or got[7:].strip() != API_KEY:
            return JSONResponse({"detail": "bad or missing bearer token"},
                                status_code=401)
    # Any touch of a job -- a frame, a mask, a status poll -- says "this one is
    # in use", and that is the only thing standing between an open clip and the
    # garbage collector. Reading a frame changes nothing on disk otherwise.
    bits = path.split("/", 4)
    if len(bits) > 3 and bits[1] == "api" and bits[2] in ("jobs", "sequence"):
        GC.touch(bits[3])
    return await call_next(request)

_state_lock = threading.Lock()      # guards the in-memory job table
_gpu_lock = threading.Lock()        # only one EdgeTAM run at a time
_model_lock = threading.Lock()
_predictors = {}                    # image_size -> predictor
_backend = None                     # the backend that actually built
_jobs = {}                          # job_id -> status dict


# --------------------------------------------------------------- job helpers
def job_dir(jid):
    d = os.path.join(JOBS, jid)
    if not os.path.isdir(d) or os.path.sep in jid or jid.startswith("."):
        raise HTTPException(404, "no such job")
    return d


def meta_path(jid):
    return os.path.join(JOBS, jid, "meta.json")


def read_meta(jid):
    with open(meta_path(jid)) as f:
        return json.load(f)


def write_meta(jid, meta):
    with open(meta_path(jid), "w") as f:
        json.dump(meta, f, indent=2)


def new_status(n_frames):
    return {
        "state": "idle", "done_frames": 0, "n_frames": n_frames,
        "elapsed_s": 0.0, "fps": 0.0, "error": None, "device": DEVICE,
        "backend": _resolved_backend(),
        "precision": _precision(_backend or BACKEND),
        "image_size": DEFAULT_TRACK_SIZE,
        "objects": [],
        "prompt_frames": {},
        "render": {"state": "idle", "done_frames": 0, "n_frames": n_frames,
                   "elapsed_s": 0.0, "fps": 0.0, "error": None},
    }


def status_of(jid):
    with _state_lock:
        st = _jobs.get(jid)
        if st is None:
            meta = read_meta(jid)
            st = new_status(meta["n_frames"])
            # recover state from disk for jobs from a previous server run
            mroot = os.path.join(JOBS, jid, "masks")
            if os.path.isdir(mroot):
                objs = sorted(os.listdir(mroot))
                if objs:
                    st["state"] = "done"
                    st["objects"] = objs
                    st["done_frames"] = len(os.listdir(os.path.join(mroot, objs[0])))
            if os.path.exists(os.path.join(JOBS, jid, "out.mp4")):
                st["render"]["state"] = "done"
            _jobs[jid] = st
        return json.loads(json.dumps(st))


# ------------------------------------------------------------------ ffmpeg
def ffprobe_json(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,duration",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise HTTPException(400, "ffprobe failed: " + out.stderr[-400:])
    return json.loads(out.stdout)


# Extraction progress, keyed by a client-invented ticket. The upload POST is
# one long request, so the only way to show a 2-minute clip going by is a
# second, cheap request the page can poll while the first one runs. Bounded:
# tickets fall out after an hour.
_extract = {}
_extract_lock = threading.Lock()


def _extract_set(ticket, **kw):
    if not ticket:
        return
    with _extract_lock:
        st = _extract.setdefault(ticket, {"phase": "queued", "done": 0,
                                          "total": 0, "t": time.time()})
        st.update(kw)
        st["t"] = time.time()
        if len(_extract) > 64:
            cut = time.time() - 3600
            for k in [k for k, v in _extract.items() if v["t"] < cut]:
                _extract.pop(k, None)


def plan_extraction(probe, fps, trim_start=0.0, trim_end=None):
    """(t0, duration or None, expected frames, out w, out h) for a trim window.

    There is no cap. `duration=None` means "to the end of the clip" -- ffmpeg
    gets no `-t` at all, which is the honest way to say it when the container's
    own duration is a lie (MediaRecorder WebM) or missing.
    """
    st = (probe.get("streams") or [{}])[0]
    src_dur = float(probe.get("format", {}).get("duration")
                    or st.get("duration") or 0.0)
    t0 = max(0.0, float(trim_start or 0.0))
    dur = None
    if trim_end:
        dur = max(0.05, float(trim_end) - t0)
    if src_dur > 0:
        avail = max(0.0, src_dur - t0)
        dur = avail if dur is None else min(dur, avail)
    span = dur if dur is not None else max(0.0, src_dur - t0)
    n = int(round(span * fps)) if span > 0 else 0
    w0, h0 = int(st.get("width") or 0), int(st.get("height") or 0)
    if w0 and h0:
        h = DECODE_HEIGHT
        w = max(2, int(round(w0 * (h / h0) / 2)) * 2)
    else:
        w, h = 1280, DECODE_HEIGHT
    return t0, dur, n, w, h


def check_disk(target_dir, n_frames, w, h):
    """Refuse before ffmpeg fills the volume, and say what it would have cost."""
    per = JPEG_BYTES_PER_FRAME * max(1, w * h) / (1280 * 720)
    need = int(max(1, n_frames) * per * 1.25) + (64 << 20)
    try:
        free = shutil.disk_usage(target_dir).free
    except OSError:
        return                                  # can't tell: don't invent a wall
    if free < need:
        raise HTTPException(
            507, "not enough disk for %d frames: %.1f GB needed, %.1f GB free. "
                 "Trim the clip, or free some space."
                 % (n_frames, need / 1e9, free / 1e9))


def pad_for(n_frames):
    """Digits in a frame's filename. 4 keeps every existing job on disk
    readable; past ~9,000 frames it widens so `sorted()` stays in order."""
    return 4 if int(n_frames or 0) < 9000 else 6


def pad_of(meta):
    return int(meta.get("pad") or 4)


def fname(n, pad=4, ext=".jpg"):
    return "%0*d%s" % (pad, int(n), ext)


def extract_frames(src, frames_dir, fps=30, trim_start=0.0, trim_end=None,
                   probe=None, ticket=None, pad=None):
    """Decode the clip to frames -- all of it, or exactly the trim range.

    `-ss` goes before `-i` so ffmpeg seeks instead of decoding-and-throwing-away;
    modern ffmpeg is frame accurate there. Nothing is capped: a 4 s window out
    of a 30 s clip is 4 s of frames, and no window at all is the whole clip.
    ffmpeg's own `-progress` is piped back so a long extraction is visible.
    """
    os.makedirs(frames_dir, exist_ok=True)
    probe = probe if probe is not None else ffprobe_json(src)
    t0, dur, n_expect, w, h = plan_extraction(probe, fps, trim_start, trim_end)
    check_disk(os.path.dirname(frames_dir) or ".", n_expect, w, h)
    pad = pad_for(n_expect) if pad is None else int(pad)
    _extract_set(ticket, phase="extract", done=0, total=n_expect)

    cmd = ["ffmpeg", "-v", "error", "-y", "-progress", "pipe:1", "-nostats"]
    if t0 > 0.01:
        cmd += ["-ss", "%.3f" % t0]
    cmd += ["-i", src]
    if dur is not None:
        cmd += ["-t", "%.3f" % dur]
    cmd += ["-vf", "scale=-2:%d,fps=%d" % (DECODE_HEIGHT, fps),
            "-q:v", "3", "-start_number", "0",
            os.path.join(frames_dir, "%%0%dd.jpg" % pad)]

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    err = []
    # stderr has to be drained on its own thread or a chatty failure deadlocks
    # against the -progress pipe we are reading here.
    t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    t.start()
    for line in p.stdout:
        if line.startswith("frame="):
            try:
                _extract_set(ticket, phase="extract",
                             done=int(line.split("=", 1)[1].strip() or 0),
                             total=n_expect)
            except ValueError:
                pass
    p.wait()
    t.join(timeout=5)
    if p.returncode != 0:
        _extract_set(ticket, phase="error")
        raise HTTPException(400, "ffmpeg failed: " + ("".join(err))[-600:])
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    _extract_set(ticket, phase="done", done=len(files), total=len(files))
    return files


# ------------------------------------------------------------------- model
def _resolved_backend():
    """The backend in use, or None while nothing has been built yet and the
    choice is still 'whichever of BACKENDS comes up first'."""
    return _backend or (None if BACKEND == "auto" else BACKEND)


def _precision(name):
    """What the UI shows next to the device. Everything but the reference
    backend runs the model in fp16."""
    return "fp32" if name == "torch-fp32" else "fp16"


def _build(name, image_size):
    """Build the predictor for one backend. Raises if that backend can't run."""
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    if not os.path.exists(CKPT):
        raise RuntimeError("checkpoint missing: run ./setup.sh")
    p = build_sam2_video_predictor(
        CFG, CKPT, device=torch.device(DEVICE),
        hydra_overrides_extra=EU.hydra_overrides(image_size))
    EU.set_image_size(p, image_size)
    if name in ("torch-half", "torch-compiled", "coreml"):
        # half weights under autocast: autocast still fixes up the fp32 tensors
        # the prompt encoder builds for itself, and the weights stop being cast
        # on every call. Measured 9.3 vs 8.3 fps against plain autocast.
        p = p.half()
    if name == "torch-compiled":
        p.forward_image = torch.compile(p.forward_image, dynamic=False)
    if name == "coreml":
        from coreml import accel
        d = os.path.join(COREML_DIR, str(image_size))
        if accel.install(p, directory=d) is None:
            raise RuntimeError("no CoreML export in %s: run ./setup.sh" % d)
        p._coreml_accel.warm(range(1, MAX_OBJECTS + 1))
    return p


def get_predictor(image_size=DEFAULT_TRACK_SIZE):
    """Returns (predictor, backend-name) for one tracking resolution.

    One predictor is cached per resolution -- the model is small and the
    CoreML graphs are per-size, so switching quality in the UI should not
    re-pay the build."""
    global _backend
    image_size = int(image_size)
    with _model_lock:
        if image_size not in _predictors:
            order = list(BACKENDS) if BACKEND == "auto" else \
                [BACKEND] + [b for b in BACKENDS if b != BACKEND]
            if _backend is not None:            # stay on one backend per process
                order = [_backend] + [b for b in order if b != _backend]
            errors = []
            for name in order:
                t = time.perf_counter()
                try:
                    _predictors[image_size] = _build(name, image_size)
                except Exception as e:                      # noqa: BLE001
                    errors.append("%s: %s: %s" % (name, type(e).__name__, e))
                    print("[model] backend %r unavailable at %d px (%s)"
                          % (name, image_size, e), flush=True)
                    continue
                _backend = name
                print("[model] EdgeTAM loaded on %s at %d px, backend %s, in %.2fs"
                      % (DEVICE, image_size, name, time.perf_counter() - t), flush=True)
                break
            else:
                raise HTTPException(500, "no tracking backend would build: "
                                    + " | ".join(errors))
        return _predictors[image_size], _backend


def _sync():
    if DEVICE == "mps":
        import torch
        torch.mps.synchronize()


def _decode_mask(data_url, w, h):
    """data: URL (or bare base64) of a PNG -> HxW bool numpy array."""
    b64 = data_url.split(",", 1)[-1]
    try:
        im = Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, "bad mask image: %s" % e)
    if im.size != (w, h):
        im = im.resize((w, h), Image.NEAREST)
    a = np.array(im.convert("L"))
    if not a.any():
        raise HTTPException(400, "mask selection is empty")
    return a > 127


def _apply_prompts(predictor, state, pairs, w, h):
    """Feed prompts to the predictor. `pairs` is [(object, frame_idx)].

    SAM2/EdgeTAM keeps prompts per *object*, not per state, so subjects
    prompted on different frames all go into a single inference state --
    `_track_worker` then decides which propagation loop walks it. A subject
    that has not entered the shot yet comes back empty from either loop,
    because its object score stays negative until it is actually there.
    """
    out = None
    for o, frame_idx in pairs:
        if o.mask:
            out = predictor.add_new_mask(
                state, frame_idx=frame_idx, obj_id=int(o.id),
                mask=_decode_mask(o.mask, w, h))
            continue
        pts = np.array([[p[0], p[1]] for p in o.points], np.float32) \
            if o.points else None
        lbl = np.array([int(p[2]) for p in o.points], np.int32) if o.points else None
        box = np.array(o.box, np.float32) if o.box else None
        out = predictor.add_new_points_or_box(
            state, frame_idx=frame_idx, obj_id=int(o.id),
            points=pts, labels=lbl, box=box)
    return out


def _soft_png(logits):
    """model logits -> the same soft 0-255 L PNG the tracker writes to disk"""
    soft = 1.0 / (1.0 + np.exp(-logits))
    return Image.fromarray((soft * 255.0).round().clip(0, 255).astype(np.uint8), "L")


# -------------------------------------------------------------------- API
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(WEB, "index.html")) as f:
        return HTMLResponse(f.read())


@app.get("/api/meta")
def api_meta():
    """The cheapest possible "is there a Dither Studio server here?" answer.

    The page GETs this with a short timeout on load. Present and `ok` means the
    remote engine is available (and, on this Mac, faster than the browser one);
    a timeout, a 404 or a CORS failure means fall back to the browser engine.
    Deliberately tiny -- it must not touch torch, the checkpoint or the disk.
    """
    return {"ok": True, "name": "dither-studio", "api": API_VERSION,
            "device": DEVICE, "backend": _resolved_backend(),
            "auth": "bearer" if API_KEY else "none",
            "max_objects": MAX_OBJECTS,
            "track_sizes": [t["size"] for t in TRACK_SIZES],
            "default_track_size": DEFAULT_TRACK_SIZE,
            "per_object_prompt_frames": True,
            "segment_image": True,
            "uncapped": True,          # no 10 s / 300 frame ceiling on upload
            "reextract": True,         # POST /api/jobs/<id>/reextract
            "extract_progress": True,  # GET /api/extract/<ticket>
            "gc": True,                # GET /api/gc/status, POST /api/gc/run
            "original": True,          # POST /api/jobs/<id>/original
            # render / original / dots all take an inclusive frame_in..frame_out
            # window over the frames already on disk -- a trim after the track
            # costs nothing and re-tracks nothing.
            "frame_range": True,
            "dots_max_frames": DOTS_MAX_FRAMES,
            "formats": list(R.FORMATS)}


@app.get("/api/bluenoise")
def bluenoise(n: int = 64, seed: int = 7):
    tile = R.blue_noise(n, seed)
    return {"n": n, "seed": seed, "tile": [round(float(v), 6) for v in tile.ravel()]}


@app.get("/api/palettes")
def palettes():
    """Everything the client needs to build its controls."""
    return {
        "palettes": DI.PALETTES,
        "modes": DI.MODES,
        "kernels": [{"id": k, "name": v["name"]} for k, v in DI.KERNELS.items()],
        "stable": DI.STABLE,
        "defaults": R.DEFAULTS,
        "subject_colors": R.SUBJECT_COLORS,
        "device": DEVICE,
        "backend": _resolved_backend(),
        "backends": list(BACKENDS),
        "track_sizes": TRACK_SIZES,
        "default_track_size": DEFAULT_TRACK_SIZE,
        "precision": _precision(_backend or BACKEND),
        "max_objects": MAX_OBJECTS,
        "segment_image": True,
        "uncapped": True,
        "reextract": True,
        "extract_progress": True,
        "gc": True,
        "original": True,
        "frame_range": True,
        "dots_max_frames": DOTS_MAX_FRAMES,
        "jpeg_bytes_per_frame": JPEG_BYTES_PER_FRAME,
        "decode_height": DECODE_HEIGHT,
        "formats": [dict(id=k, **{x: v[x] for x in ("ext", "mime", "alpha", "label")})
                    for k, v in R.FORMATS.items()],
    }


@app.get("/api/gc/status")
def gc_status():
    """What jobs/ is holding, and what the janitor last did about it.

    The page shows the one number a person cares about -- how much disk this
    tool is sitting on -- next to the button that hands it back.
    """
    return GC.status()


@app.post("/api/gc/run")
def gc_run():
    """Sweep now. Same policy as the timed run, so the button cannot delete
    anything the background thread would not have."""
    rep = GC.run("manual")
    out = GC.status()
    out["ran"] = rep
    return out


IMAGE_MAX_SIDE = 4096


@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...), max_side: int = Form(1600)):
    """A still, as a one-frame job.

    Everything downstream of a job -- /preview, /frame, /mask -- already works
    on a single frame, so a still needs no second code path on this side: it
    becomes a job whose n_frames is 1 and whose fps means nothing. Selecting a
    subject in a photograph is then literally the same call the clip flow's
    "preview this frame" button makes, which is why there is no propagation and
    no second set of numbers to keep honest.

    The page sends the picture already scaled to the size it prompts at, so the
    click coordinates and the mask that comes back are in the same pixels.
    `max_side` is the backstop for anything that arrives bigger.
    """
    raw = await file.read()
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        im = im.convert("RGB")
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, "could not read that image: %s" % e)
    w0, h0 = im.size
    side = max(64, min(IMAGE_MAX_SIDE, int(max_side or 1600)))
    if max(w0, h0) > side:
        k = side / float(max(w0, h0))
        im = im.resize((max(1, round(w0 * k)), max(1, round(h0 * k))),
                       Image.LANCZOS)
    w, h = im.size
    jid = uuid.uuid4().hex[:12]
    d = os.path.join(JOBS, jid)
    os.makedirs(os.path.join(d, "frames"), exist_ok=True)
    im.save(os.path.join(d, "frames", "0000.jpg"), quality=95)
    meta = {
        "job": jid, "kind": "image", "filename": file.filename,
        "n_frames": 1, "w": w, "h": h, "fps": 1,
        "natural_w": w0, "natural_h": h0,
        "created": time.time(),
    }
    write_meta(jid, meta)
    with _state_lock:
        _jobs[jid] = new_status(1)
    print("[image] %s: %dx%d (from %dx%d)" % (jid, w, h, w0, h0), flush=True)
    return {"job": jid, "kind": "image", "n_frames": 1, "w": w, "h": h,
            "natural_w": w0, "natural_h": h0}


def _open_clip(src, jid, filename, fps, trim_start, trim_end, ticket):
    """Probe + extract + meta for one job directory. Blocking; callers hand it
    to the threadpool so the event loop stays free to answer /api/extract."""
    d = os.path.join(JOBS, jid)
    probe = ffprobe_json(src)
    files = extract_frames(src, os.path.join(d, "frames"), fps,
                           trim_start, trim_end, probe=probe, ticket=ticket)
    if not files:
        raise HTTPException(400, "no frames extracted")
    w, h = Image.open(os.path.join(d, "frames", files[0])).size
    meta = {
        "job": jid, "source": os.path.basename(src), "filename": filename,
        "n_frames": len(files), "w": w, "h": h, "fps": fps,
        "pad": len(os.path.splitext(files[0])[0]),
        "source_duration_s": float(probe.get("format", {}).get("duration") or 0),
        "trim_start": float(trim_start or 0.0),
        "trim_end": float(trim_end) if trim_end else None,
        "created": time.time(),
    }
    write_meta(jid, meta)
    with _state_lock:
        _jobs[jid] = new_status(len(files))
    return {"job": jid, "n_frames": len(files), "w": w, "h": h, "fps": fps,
            "source_duration_s": meta["source_duration_s"],
            "trim_start": meta["trim_start"], "trim_end": meta["trim_end"],
            "seconds": round(len(files) / max(1, fps), 3),
            "dots_max_frames": DOTS_MAX_FRAMES}


@app.get("/api/extract/{ticket}")
def extract_progress(ticket: str):
    """Where an in-flight /api/upload has got to. The upload is one long POST;
    this is the only thing the page can ask while it is still running."""
    with _extract_lock:
        st = _extract.get(ticket)
    return st or {"phase": "queued", "done": 0, "total": 0}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...),
                 fps: int = Form(30),
                 trim_start: float = Form(0.0),
                 trim_end: float | None = Form(None),
                 ticket: str | None = Form(None),
                 # accepted and ignored: older pages still send them, and the
                 # answer to both is now "no cap"
                 max_seconds: float | None = Form(None),
                 max_frames: int | None = Form(None)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".mp4", ".mov", ".m4v", ".webm"):
        raise HTTPException(400, "expected .mp4 / .mov / .webm")
    jid = uuid.uuid4().hex[:12]
    d = os.path.join(JOBS, jid)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "source" + ext)
    _extract_set(ticket, phase="upload", done=0, total=0)
    await run_in_threadpool(_save_upload, file, src)
    return await run_in_threadpool(_open_clip, src, jid, file.filename, fps,
                                   trim_start, trim_end, ticket)


def _save_upload(file, dst):
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f, 1 << 20)


class ReextractReq(BaseModel):
    trim_start: float = 0.0
    trim_end: float | None = None
    fps: int | None = None
    ticket: str | None = None


@app.post("/api/jobs/{jid}/reextract")
async def reextract(jid: str, req: ReextractReq):
    """A different range out of a clip that is already here.

    The source file stays in the job directory, so changing your mind about the
    trim costs one ffmpeg run and no upload. It lands in a NEW job -- the old
    one's masks and renders belong to the old range and must not be confused
    with the new one's.
    """
    d = job_dir(jid)
    meta = read_meta(jid)
    name = meta.get("source") or ""
    src = os.path.join(d, name)
    if not name or not os.path.exists(src):
        raise HTTPException(404, "this job has no source clip to re-extract "
                                 "from -- upload the file again")
    njid = uuid.uuid4().hex[:12]
    nd = os.path.join(JOBS, njid)
    os.makedirs(nd, exist_ok=True)
    nsrc = os.path.join(nd, name)
    try:
        os.link(src, nsrc)          # same volume: no second copy of the bytes
    except OSError:
        shutil.copyfile(src, nsrc)
    return await run_in_threadpool(
        _open_clip, nsrc, njid, meta.get("filename") or name,
        int(req.fps or meta.get("fps") or 30), req.trim_start, req.trim_end,
        req.ticket)


@app.get("/api/jobs/{jid}/meta")
def get_meta(jid: str):
    job_dir(jid)
    return read_meta(jid)


@app.get("/api/jobs/{jid}/frame/{n}")
def get_frame(jid: str, n: int):
    p = os.path.join(job_dir(jid), "frames",
                     fname(n, pad_of(read_meta(jid)), ".jpg"))
    if not os.path.exists(p):
        raise HTTPException(404, "no such frame")
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{jid}/mask/{obj}/{n}")
def get_mask(jid: str, obj: str, n: int, polish: int = 0):
    """One soft mask. `?polish=0-100` hands back the polished one instead --
    the same file the render would use, built and cached on first ask. The page
    polishes in the tab rather than fetching this, but it is the only way to
    see the server's own answer, which is what the parity check compares."""
    if not obj.isalnum():
        raise HTTPException(400, "bad object id")
    d = job_dir(jid)
    if polish > 0:
        if not os.path.isdir(os.path.join(d, "masks", obj)):
            raise HTTPException(404, "no such mask")
        md, _ = PL.polished_dir(d, obj, polish)
        p = os.path.join(md, fname(n, pad_of(read_meta(jid)), ".png"))
    else:
        p = os.path.join(d, "masks", obj,
                         fname(n, pad_of(read_meta(jid)), ".png"))
    if not os.path.exists(p):
        raise HTTPException(404, "no such mask")
    return FileResponse(p, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{jid}/status")
def get_status(jid: str):
    job_dir(jid)
    return status_of(jid)


class TrackObject(BaseModel):
    id: int
    points: list[list[float]] = []      # [[x, y, label], ...]  label 1=fg 0=bg
    box: list[float] | None = None      # [x0, y0, x1, y1]
    # A lasso/polygon selection, rasterised client-side to a binary PNG at the
    # clip's own resolution and sent as a data URL. EdgeTAM takes a mask prompt
    # OR points+box for a given (frame, object) -- `add_new_mask` drops the
    # frame's point inputs and `add_new_points_or_box` drops its mask input --
    # so when this is present it is what the tracker sees.
    mask: str | None = None
    # The frame this subject was prompted on. A tennis ball that flies in at
    # frame 80 does not exist on frame 0, so each subject carries its own.
    # None = fall back to the request-level frame_idx (the old single-frame
    # behaviour, kept so an older client still works).
    frame_idx: int | None = None


class TrackReq(BaseModel):
    frame_idx: int = 0                      # default for objects without one
    objects: list[TrackObject]
    image_size: int = DEFAULT_TRACK_SIZE    # tracker input square, not the clip

    def frames(self):
        """[(object, its prompt frame)] in request order."""
        return [(o, self.frame_idx if o.frame_idx is None else o.frame_idx)
                for o in self.objects]


@app.post("/api/jobs/{jid}/track")
def track(jid: str, req: TrackReq):
    d = job_dir(jid)
    meta = read_meta(jid)
    if not req.objects:
        raise HTTPException(400, "no objects")
    if len(req.objects) > MAX_OBJECTS:
        raise HTTPException(400, "at most %d objects" % MAX_OBJECTS)
    for o in req.objects:
        if not o.points and not o.box and not o.mask:
            raise HTTPException(400, "object %s has no prompt" % o.id)
    for o, fi in req.frames():
        if not (0 <= fi < meta["n_frames"]):
            raise HTTPException(400, "object %s: frame_idx %d out of range"
                                % (o.id, fi))
    if req.image_size not in [t["size"] for t in TRACK_SIZES]:
        raise HTTPException(400, "image_size must be one of %s"
                            % [t["size"] for t in TRACK_SIZES])

    with _state_lock:
        st = _jobs.get(jid) or new_status(meta["n_frames"])
        if st["state"] in ("tracking", "loading"):
            raise HTTPException(409, "already tracking")
        st.update(state="loading", done_frames=0, elapsed_s=0.0, fps=0.0, error=None,
                  n_frames=meta["n_frames"],
                  objects=[str(o.id) for o in req.objects],
                  prompt_frames={str(o.id): fi for o, fi in req.frames()})
        _jobs[jid] = st

    # the rasterised mask is big and reproducible from the paths the client
    # keeps, so record that it was used rather than storing it in meta.json
    prompts = [{**o.model_dump(exclude={"mask"}), "mask": bool(o.mask),
                "frame_idx": fi} for o, fi in req.frames()]
    write_meta(jid, {**meta, "prompts": {"frame_idx": req.frame_idx,
                                        "image_size": req.image_size,
                                        "objects": prompts}})
    threading.Thread(target=_track_worker, args=(jid, d, req), daemon=True).start()
    return {"job": jid, "state": "loading", "image_size": req.image_size,
            "objects": [str(o.id) for o in req.objects],
            "prompt_frames": {str(o.id): fi for o, fi in req.frames()}}


def _set(jid, **kw):
    with _state_lock:
        _jobs[jid].update(kw)


def _track_worker(jid, d, req):
    import torch
    t0 = time.perf_counter()
    try:
        with _gpu_lock:
            predictor, backend = get_predictor(req.image_size)
            _set(jid, backend=backend, precision=_precision(backend),
                 image_size=req.image_size)
            frames_dir = os.path.join(d, "frames")
            mroot = os.path.join(d, "masks")
            shutil.rmtree(mroot, ignore_errors=True)
            for o in req.objects:
                os.makedirs(os.path.join(mroot, str(o.id)), exist_ok=True)

            cast = (contextlib.nullcontext() if backend == "torch-fp32"
                    or DEVICE != "mps"
                    else torch.autocast(DEVICE, dtype=torch.float16))
            with torch.inference_mode(), cast:
                state = predictor.init_state(frames_dir, offload_video_to_cpu=True)
                _sync()
                n_frames = state["num_frames"]
                _set(jid, state="tracking", n_frames=n_frames)

                meta = read_meta(jid)
                mpad = pad_of(meta)
                pairs = req.frames()
                _apply_prompts(predictor, state, pairs, meta["w"], meta["h"])
                _sync()
                start = min(fi for _, fi in pairs)

                seen = set()

                def drain(gen):
                    for fidx, obj_ids, masks in gen:
                        arr = masks.float().cpu().numpy()
                        for k, oid in enumerate(obj_ids):
                            soft = 1.0 / (1.0 + np.exp(-arr[k, 0]))
                            Image.fromarray(
                                (soft * 255.0).round().clip(0, 255).astype(np.uint8),
                                mode="L"
                            ).save(os.path.join(mroot, str(oid),
                                                fname(fidx, mpad, ".png")))
                        seen.add(int(fidx))
                        el = time.perf_counter() - t0
                        _set(jid, done_frames=len(seen), elapsed_s=round(el, 2),
                             fps=round(len(seen) / el, 2) if el > 0 else 0.0)

                # A click on a middle frame must fill the whole clip, so every
                # subject gets walked out of its prompt frame both ways.
                #
                # Which loop does that depends on how many prompt frames there
                # are. Upstream's batched one shares a single memory bank across
                # all objects, and its pre-pass consolidates every prompt frame
                # across every object before tracking starts -- so a frame
                # prompted for one subject enters the OTHER subjects' memory as
                # a conditioning frame holding a NO_OBJ placeholder, and kills
                # their tracks from there on. That can only happen when the
                # subjects were prompted on more than one frame, so that is
                # exactly when we swap in the per-object loop (SAM 2.1's fix,
                # see edgetam_util.propagate_per_object), which does both
                # directions itself. One prompt frame keeps the batched path
                # unchanged: reverse from it, then forward from it.
                if len({fi for _, fi in pairs}) > 1:
                    drain(EU.propagate_per_object(predictor, state))
                else:
                    if start > 0:
                        drain(predictor.propagate_in_video(
                            state, start_frame_idx=start, reverse=True))
                    drain(predictor.propagate_in_video(
                        state, start_frame_idx=start, reverse=False))

                del state
            if DEVICE == "mps":
                torch.mps.empty_cache()

        el = time.perf_counter() - t0
        _set(jid, state="done", elapsed_s=round(el, 2),
             fps=round(len(seen) / el, 2) if el > 0 else 0.0)
        print("[track] %s: %d frames, %d obj, %.1fs (%.2f fps, %s @ %d px)"
              % (jid, len(seen), len(req.objects), el, len(seen) / max(el, 1e-6),
                 backend, req.image_size), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _set(jid, state="error", error="%s: %s" % (type(e).__name__, e))


@app.post("/api/jobs/{jid}/preview")
def preview(jid: str, req: TrackReq):
    """Run the first-frame prediction only -- no propagation.

    This is the "is my prompt good enough?" button. It builds a one-frame
    inference state over a symlink to the prompt frame, so it costs an image
    encode and the SAM heads rather than decoding the whole clip, and returns
    the same soft masks the tracker would write for that frame.
    """
    import torch
    d = job_dir(jid)
    meta = read_meta(jid)
    if not req.objects:
        raise HTTPException(400, "no objects")
    if len(req.objects) > MAX_OBJECTS:
        raise HTTPException(400, "at most %d objects" % MAX_OBJECTS)
    for o in req.objects:
        if not o.points and not o.box and not o.mask:
            raise HTTPException(400, "object %s has no prompt" % o.id)
    if not (0 <= req.frame_idx < meta["n_frames"]):
        raise HTTPException(400, "frame_idx out of range")
    if req.image_size not in [t["size"] for t in TRACK_SIZES]:
        raise HTTPException(400, "bad image_size")
    # One frame, so only the subjects whose own prompt frame is this one.
    pairs = [(o, fi) for o, fi in req.frames() if fi == req.frame_idx]
    if not pairs:
        raise HTTPException(400, "no subject is prompted on frame %d"
                            % req.frame_idx)

    mpad = pad_of(meta)
    src = os.path.join(d, "frames", fname(req.frame_idx, mpad, ".jpg"))
    one = os.path.join(d, "preview", fname(req.frame_idx, mpad, ""))
    os.makedirs(one, exist_ok=True)
    link = os.path.join(one, "0000.jpg")
    if not os.path.exists(link):
        os.symlink(os.path.relpath(src, one), link)

    t0 = time.perf_counter()
    if not _gpu_lock.acquire(timeout=0.05 if _predictors else 120):
        raise HTTPException(409, "busy tracking")
    try:
        predictor, backend = get_predictor(req.image_size)
        cast = (contextlib.nullcontext() if backend == "torch-fp32" or DEVICE != "mps"
                else torch.autocast(DEVICE, dtype=torch.float16))
        with torch.inference_mode(), cast:
            state = predictor.init_state(one, offload_video_to_cpu=True)
            out = _apply_prompts(predictor, state,
                                 [(o, 0) for o, _ in pairs],
                                 meta["w"], meta["h"])
            _, obj_ids, masks = out
            arr = masks.float().cpu().numpy()
            del state
    finally:
        _gpu_lock.release()
    if DEVICE == "mps":
        torch.mps.empty_cache()

    objs = []
    for k, oid in enumerate(obj_ids):
        buf = io.BytesIO()
        _soft_png(arr[k, 0]).save(buf, format="PNG")
        objs.append({"id": str(oid),
                     "area": int((arr[k, 0] > 0).sum()),
                     "mask": "data:image/png;base64,"
                             + base64.b64encode(buf.getvalue()).decode()})
    el = time.perf_counter() - t0
    print("[preview] %s: frame %d, %d obj, %.2fs (%s @ %d px)"
          % (jid, req.frame_idx, len(objs), el, backend, req.image_size), flush=True)
    return {"job": jid, "frame_idx": req.frame_idx, "elapsed_s": round(el, 3),
            "backend": backend, "image_size": req.image_size, "objects": objs}


def _resolve_subjects(jid, d, subjects, do_polish=True, progress=None):
    """[{id, palette, dot, polish}] -> what the renderer wants: a mask directory
    per subject, plus the palettes.

    `polish` (0-100) is the only thing here that does work. It swaps the
    tracker's own masks for a polished copy -- server/polish.py, the same
    algorithm web/polish.js runs in the tab -- built once per (subject,
    strength) and cached under jobs/<jid>/polish/, because two renders of the
    same clip at the same strength want the same masks and polishing 189 frames
    costs a few seconds.
    """
    mroot = os.path.join(d, "masks")
    out = []
    for i, s in enumerate(subjects):
        oid = str(s.get("id"))
        md = os.path.join(mroot, oid)
        if not os.path.isdir(md):
            raise HTTPException(400, "no masks for subject %s - track first" % oid)
        strength = int(round(float(s.get("polish") or 0)))
        if strength > 0 and do_polish:
            if progress:
                progress("polishing subject #%s (%d)" % (oid, strength))
            t0 = time.perf_counter()
            md, info = PL.polished_dir(d, oid, strength)
            if info:
                print("[polish] %s: subject %s at %d -> %d frames in %.1fs "
                      "(radius %d, morph %d, blur %d)"
                      % (jid, oid, strength, info["frames"],
                         time.perf_counter() - t0, info["radius"], info["morph"],
                         info["blur"]), flush=True)
        out.append({"masks": md, "palette": s.get("palette"),
                    "dot": s.get("dot") or R.SUBJECT_COLORS[i % 6],
                    "polish": strength})
    return out


class RenderReq(BaseModel):
    # [{id, palette:[hex,...], polish:0-100}] — [] = whole frame. `polish` is
    # the mask smoother; 0 (the default) leaves the tracker's masks alone.
    subjects: list[dict] = []
    # look
    mode: str = "dots"                 # dots|bluenoise|ordered|halftone|whitenoise|errordiff|riemersma
    algo: str = "floyd-steinberg"      # errordiff kernel
    matrix: int = 4                    # ordered / halftone matrix size
    serpentine: bool = False
    strength: float = 1.0
    palette: list[str] | None = None   # background / whole-frame palette
    # tone
    brightness: float = 0.0
    contrast: float = 1.0
    gamma: float = 1.0
    invert: bool = False
    pixel: int = 1
    # composition
    compose: str = "cutout"            # cutout|overlay
    bg: str = "#c9d4c5"
    # dots-only
    n: int = 8000
    cell: int = 4
    dotpx: int = 3
    fill: float = 0.7
    stray: float = 0.02
    band: int = 9
    seed: int = 7
    fps: int | None = None
    # container. mp4|webm|gif|webm-alpha|prores — see render.FORMATS. The two
    # alpha formats key the flat background out and leave only the dots.
    format: str = "mp4"
    gif_fps: int | None = None
    # The frame window, inclusive, as indices into jobs/<id>/frames. Omitted =
    # the whole clip, which is what every client did before this existed.
    frame_in: int = 0
    frame_out: int | None = None


def _window(meta, frame_in, frame_out, what="render"):
    """(in, out, count) for a request's frame window, or a 400 saying why not.

    The window is a view on frames that already exist. It never re-extracts and
    never re-tracks: masks are numbered by the same index the frames are, so a
    narrower range is a slice and nothing more.
    """
    n = int(meta.get("n_frames") or 0)
    a = int(frame_in or 0)
    b = n - 1 if frame_out is None else int(frame_out)
    if n <= 0:
        raise HTTPException(400, "this job has no frames")
    if a < 0 or b >= n or a > b:
        raise HTTPException(
            400, "frames %d–%d are outside this clip, which has %d "
                 "(0–%d). Re-extract to cover a wider range."
                 % (a, b, n, n - 1))
    return a, b, b - a + 1


@app.post("/api/jobs/{jid}/render")
def start_render(jid: str, req: RenderReq):
    d = job_dir(jid)
    meta = read_meta(jid)
    # validated here, polished in the worker: building the polished masks for
    # four subjects takes seconds and this call has to answer straight away
    resolved = _resolve_subjects(jid, d, req.subjects, do_polish=False)
    if req.mode == "dots" and not resolved:
        raise HTTPException(400, "the dots look needs at least one tracked subject")
    if req.format not in R.FORMATS:
        raise HTTPException(400, "format must be one of %s" % list(R.FORMATS))
    f_in, f_out, n_win = _window(meta, req.frame_in, req.frame_out)

    with _state_lock:
        st = _jobs.get(jid) or new_status(meta["n_frames"])
        if st["render"]["state"] == "rendering":
            raise HTTPException(409, "already rendering")
        st["render"] = {"state": "rendering", "done_frames": 0,
                        "n_frames": n_win, "elapsed_s": 0.0,
                        "fps": 0.0, "error": None, "format": req.format,
                        "bytes": 0, "stage": "rendering"}
        _jobs[jid] = st

    params = req.model_dump()
    params.pop("subjects", None)
    if params.get("fps") is None:
        params["fps"] = meta.get("fps", 30)
    threading.Thread(target=_render_worker, args=(jid, d, resolved, params),
                     daemon=True).start()
    return {"job": jid, "state": "rendering", "subjects": len(resolved),
            "format": req.format, "frame_in": f_in, "frame_out": f_out,
            "frames": n_win,
            "url": "/api/jobs/%s/output/%s" % (jid, req.format)}


def _render_worker(jid, d, subs, params):
    t0 = time.perf_counter()
    fmt = params.get("format") or "mp4"
    out = os.path.join(d, R.FORMATS[fmt]["file"])
    try:
        for k, s in enumerate(subs):
            if not s.get("polish"):
                continue
            with _state_lock:
                _jobs[jid]["render"]["stage"] = "polishing subject %d/%d" % (
                    k + 1, len(subs))
            tp = time.perf_counter()
            s["masks"], info = PL.polished_dir(d, os.path.basename(s["masks"]),
                                               s["polish"])
            if info:
                print("[polish] %s: subject %s at %d -> %d frames in %.1fs "
                      "(radius %d, morph %d, blur %d)"
                      % (jid, os.path.basename(os.path.dirname(info["dir"])),
                         s["polish"], info["frames"], time.perf_counter() - tp,
                         info["radius"], info["morph"], info["blur"]), flush=True)
        with _state_lock:
            _jobs[jid]["render"]["stage"] = "rendering"

        def prog(done, total):
            el = time.perf_counter() - t0
            with _state_lock:
                _jobs[jid]["render"].update(
                    done_frames=done, n_frames=total, elapsed_s=round(el, 2),
                    fps=round(done / el, 2) if el > 0 else 0.0)

        info = R.render(os.path.join(d, "frames"), subs, out, params, prog)
        el = time.perf_counter() - t0
        with _state_lock:
            _jobs[jid]["render"].update(state="done", elapsed_s=round(el, 2),
                                        fps=round(info["frames"] / max(el, 1e-6), 2),
                                        format=fmt, bytes=info.get("bytes", 0))
        print("[render] %s: %d frames (%d–%d) -> %s in %.1fs (%.1f MB)"
              % (jid, info["frames"], info.get("first_frame", 0),
                 info.get("last_frame", info["frames"] - 1), fmt, el,
                 info.get("bytes", 0) / 1e6), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        with _state_lock:
            _jobs[jid]["render"].update(state="error",
                                        error="%s: %s" % (type(e).__name__, e))


class OriginalReq(BaseModel):
    """The matched cut: the render's own frames, re-encoded without the dither.

    `format` is the format the RENDER asked for -- the original follows it
    where that makes sense and falls back to MP4 where it does not (see
    render.ORIGINAL_FORMAT). `expect_frames` is the render's frame count; when
    it is sent and does not match what is on disk, this refuses rather than
    writing a file that would not line up.
    """
    format: str = "mp4"
    fps: int | None = None
    expect_frames: int | None = None
    # the same window the render used; the pair is only a pair if both files
    # were cut from the same frames
    frame_in: int = 0
    frame_out: int | None = None


@app.post("/api/jobs/{jid}/original")
def export_original(jid: str, req: OriginalReq):
    """Write out.original.<ext> beside the render: the same frames, undithered.

    Synchronous, like /dots and for the same reason -- there is nothing to
    compute, only JPEGs to decode and hand to ffmpeg, which runs at hundreds of
    frames a second.
    """
    d = job_dir(jid)
    meta = read_meta(jid)
    if req.format not in R.FORMATS:
        raise HTTPException(400, "format must be one of %s" % list(R.FORMATS))
    frames_dir = os.path.join(d, "frames")
    if not os.path.isdir(frames_dir) or not R.count_frames(frames_dir):
        raise HTTPException(404, "this job has no frames to cut from")
    f_in, f_out, have = _window(meta, req.frame_in, req.frame_out, "original cut")
    if req.expect_frames is not None and int(req.expect_frames) != have:
        raise HTTPException(409, "the render used %d frames and frames %d–%d are "
                                 "%d — the original cut would not line up"
                                 % (int(req.expect_frames), f_in, f_out, have))
    fps = int(req.fps or meta.get("fps", 30))
    fmt = R.original_format(req.format)
    out = os.path.join(d, R.original_file(req.format))
    t0 = time.perf_counter()
    info = R.render_original(frames_dir, out, {"format": req.format, "fps": fps,
                                               "frame_in": f_in,
                                               "frame_out": f_out})
    el = time.perf_counter() - t0
    print("[original] %s: %d frames (%d–%d) -> %s in %.1fs (%.1f MB)"
          % (jid, info["frames"], f_in, f_out, fmt, el,
             info.get("bytes", 0) / 1e6), flush=True)
    return {"job": jid, "frames": info["frames"], "w": info["w"], "h": info["h"],
            "fps": fps, "format": fmt, "ext": R.FORMATS[fmt]["ext"],
            "frame_in": f_in, "frame_out": f_out,
            "bytes": info["bytes"], "elapsed_s": round(el, 2),
            "matched": fmt == req.format,
            "url": "/api/jobs/%s/original/%s" % (jid, fmt)}


@app.get("/api/jobs/{jid}/original/{fmt}")
def get_original(jid: str, fmt: str):
    """The matched cut for one format. `fmt` is the ORIGINAL's format, which
    is what POST /original hands back in its `url`."""
    if fmt not in R.FORMATS:
        raise HTTPException(400, "unknown format %r" % fmt)
    f = R.FORMATS[fmt]
    p = os.path.join(job_dir(jid), R.original_file(fmt))
    if not os.path.exists(p):
        raise HTTPException(404, "no original cut yet")
    return FileResponse(p, media_type=f["mime"], headers={
        # the pair, named as a pair: dither-<id>.mp4 / dither-<id>.original.mp4
        "Content-Disposition": 'inline; filename="dither-%s.original.%s"'
                               % (jid, f["ext"]),
        "Accept-Ranges": "bytes"})


class DotsReq(BaseModel):
    """The dots look, exported as data instead of pixels. Same knobs as a
    render; everything that is not a dots knob is ignored."""
    subjects: list[dict] = []
    cell: int = 4
    dotpx: int = 3
    n: int = 8000
    fill: float = 0.7
    stray: float = 0.02
    band: int = 9
    gamma: float = 1.0
    invert: bool = False
    seed: int = 7
    bg: str = "#c9d4c5"
    fps: int | None = None
    frame_in: int = 0
    frame_out: int | None = None
    json: bool = False          # also write the readable .dots.json variant


@app.post("/api/jobs/{jid}/dots")
def export_dots(jid: str, req: DotsReq):
    """Render the dot positions for a tracked job and write out.dots.gz.

    Synchronous on purpose: this is ~30 ms a frame, i.e. a few seconds for the
    longest clip the tool accepts, and it has no encoder to feed.
    """
    d = job_dir(jid)
    meta = read_meta(jid)
    subs = _resolve_subjects(jid, d, req.subjects)
    if not subs:
        raise HTTPException(400, "dot data needs at least one tracked subject")
    f_in, f_out, _ = _window(meta, req.frame_in, req.frame_out, "dot export")
    params = req.model_dump()
    params.pop("subjects", None)
    params.pop("json", None)
    if params.get("fps") is None:
        params["fps"] = meta.get("fps", 30)
    t0 = time.perf_counter()
    doc = R.render_dots(os.path.join(d, "frames"), subs, params)
    gz = DT.pack(doc)
    out = os.path.join(d, "out.dots.gz")
    with open(out, "wb") as f:
        f.write(gz)
    counts = [int(sum(len(x) for x in fr)) for fr in doc["frames"]]
    res = {"job": jid, "frames": len(doc["frames"]), "bytes": len(gz),
           "frame_in": f_in, "frame_out": f_out,
           "raw_bytes": len(DT.encode(doc)), "w": doc["w"], "h": doc["h"],
           "fps": doc["fps"], "dotpx": doc["dotpx"], "subjects": len(doc["subjects"]),
           "palette": doc["palette"],
           "dots_mean": round(sum(counts) / max(1, len(counts)), 1),
           "dots_max": max(counts), "elapsed_s": round(time.perf_counter() - t0, 2),
           "url": "/api/jobs/%s/out.dots.gz" % jid}
    if req.json:
        jp = os.path.join(d, "out.dots.json")
        with open(jp, "w") as f:
            json.dump(DT.to_json(doc), f)
        res["json_bytes"] = os.path.getsize(jp)
        res["json_url"] = "/api/jobs/%s/out.dots.json" % jid
    print("[dots] %s: %d frames (%d–%d), %.1f dots/frame, %d B gz in %.1fs"
          % (jid, res["frames"], f_in, f_out, res["dots_mean"], res["bytes"],
             res["elapsed_s"]),
          flush=True)
    return res


@app.get("/api/jobs/{jid}/out.dots.gz")
def get_dots(jid: str):
    p = os.path.join(job_dir(jid), "out.dots.gz")
    if not os.path.exists(p):
        raise HTTPException(404, "no dot data yet")
    # not Content-Encoding: gzip -- the gzip IS the file, and a browser that
    # transparently decoded it would hand the player the wrong bytes
    return FileResponse(p, media_type="application/octet-stream", headers={
        "Content-Disposition": 'inline; filename="dither-%s.dots.gz"' % jid})


@app.get("/api/jobs/{jid}/out.dots.json")
def get_dots_json(jid: str):
    p = os.path.join(job_dir(jid), "out.dots.json")
    if not os.path.exists(p):
        raise HTTPException(404, "no dot data yet")
    return FileResponse(p, media_type="application/json")


@app.post("/api/sequence")
async def sequence(file: UploadFile = File(...), format: str = Form("mp4"),
                   gif_fps: int = Form(15)):
    """Rasterise a .dots.gz (a sequence, a morph, anything) into a video.

    This is how a morph becomes an MP4. The tween itself is built in JS --
    web/player/dither-player.js -- and shipped here as dot positions, so there
    is exactly one implementation of the transition and the server does the one
    thing it is better at: feeding an encoder.
    """
    if format not in R.FORMATS:
        raise HTTPException(400, "format must be one of %s" % list(R.FORMATS))
    raw = await file.read()
    try:
        doc = DT.unpack(raw) if raw[:2] == b"\x1f\x8b" else DT.decode(raw)
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, "not a .dots file: %s" % e)
    sid = "seq-" + uuid.uuid4().hex[:10]
    d = os.path.join(JOBS, sid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "in.dots.gz"), "wb") as f:
        f.write(raw)
    out = os.path.join(d, R.FORMATS[format]["file"])
    t0 = time.perf_counter()
    info = DT.rasterise(doc, out, format, gif_fps)
    el = time.perf_counter() - t0
    print("[sequence] %s: %d frames -> %s in %.1fs"
          % (sid, info["frames"], format, el), flush=True)
    return {"sequence": sid, "frames": info["frames"], "w": info["w"],
            "h": info["h"], "format": format, "bytes": os.path.getsize(out),
            "elapsed_s": round(el, 2),
            "url": "/api/sequence/%s/output/%s" % (sid, format)}


@app.get("/api/sequence/{sid}/dots.gz")
def get_sequence_dots(sid: str):
    """The dot data a sequence was rendered from — what the player plays."""
    p = os.path.join(job_dir(sid), "in.dots.gz")
    if not os.path.exists(p):
        raise HTTPException(404, "no such sequence")
    return FileResponse(p, media_type="application/octet-stream", headers={
        "Content-Disposition": 'inline; filename="%s.dots.gz"' % sid})


@app.get("/api/sequence/{sid}/output/{fmt}")
def get_sequence(sid: str, fmt: str):
    if fmt not in R.FORMATS:
        raise HTTPException(400, "unknown format %r" % fmt)
    f = R.FORMATS[fmt]
    p = os.path.join(job_dir(sid), f["file"])
    if not os.path.exists(p):
        raise HTTPException(404, "not rendered yet")
    return FileResponse(p, media_type=f["mime"], headers={
        "Content-Disposition": 'inline; filename="dither-%s.%s"' % (sid, f["ext"]),
        "Accept-Ranges": "bytes"})


@app.get("/api/jobs/{jid}/output/{fmt}")
def get_output(jid: str, fmt: str):
    """The rendered file for one format. One file per format per job, so
    exporting an MP4 does not throw away the GIF you rendered a minute ago."""
    if fmt not in R.FORMATS:
        raise HTTPException(400, "unknown format %r" % fmt)
    f = R.FORMATS[fmt]
    p = os.path.join(job_dir(jid), f["file"])
    if not os.path.exists(p):
        raise HTTPException(404, "not rendered yet")
    # inline (not attachment) so the in-page <video>/<img> can play it; the UI's
    # download link carries a `download` attribute of its own
    return FileResponse(p, media_type=f["mime"], headers={
        "Content-Disposition": 'inline; filename="dither-%s.%s"' % (jid, f["ext"]),
        "Accept-Ranges": "bytes"})


@app.get("/api/jobs/{jid}/out.mp4")
def get_out(jid: str):
    """The pre-formats route, kept so an older client still works."""
    return get_output(jid, "mp4")


# The page is served from web/ as-is -- the same bytes GitHub Pages would
# serve. /static stays as an alias so old bookmarks keep working.
app.mount("/static", StaticFiles(directory=WEB), name="static")
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DV_PORT", 8765)),
                log_level="info")
