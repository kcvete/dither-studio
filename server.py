#!/usr/bin/env python3
"""Dither Video -- local FastAPI backend.

    upload video -> frames -> EdgeTAM point/box tracking -> blue-noise dither -> mp4

Everything lives under jobs/<job-id>/. Nothing leaves the machine.
"""
import contextlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import dither as DI
import render as R

HERE = os.path.dirname(os.path.abspath(__file__))
JOBS = os.path.join(HERE, "jobs")
STATIC = os.path.join(HERE, "static")
CKPT = os.path.join(HERE, "env", "EdgeTAM", "checkpoints", "edgetam.pt")
CFG = "configs/edgetam.yaml"
DEVICE = os.environ.get("DV_DEVICE", "mps")
# fp16 autocast on MPS: measured 1.19x faster tracking (7.42 -> 8.80 fps, A/B in one
# process, order alternated) with global mask IoU 0.9985 vs fp32 over 150 frames,
# worst frame 0.9609. Set DV_FP32=1 to turn it off.
FP16 = DEVICE == "mps" and os.environ.get("DV_FP32", "0") != "1"
MAX_OBJECTS = 6

os.makedirs(JOBS, exist_ok=True)

app = FastAPI(title="Dither Video")

_state_lock = threading.Lock()      # guards the in-memory job table
_gpu_lock = threading.Lock()        # only one EdgeTAM run at a time
_model_lock = threading.Lock()
_predictor = None
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
        "precision": "fp16" if FP16 else "fp32",
        "objects": [],
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


def extract_frames(src, frames_dir, max_seconds, fps, max_frames):
    os.makedirs(frames_dir, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", src,
           "-t", str(max_seconds),
           "-vf", "scale=-2:720,fps=%d" % fps,
           "-frames:v", str(max_frames),
           "-q:v", "3", "-start_number", "0",
           os.path.join(frames_dir, "%04d.jpg")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise HTTPException(400, "ffmpeg failed: " + p.stderr[-600:])
    return sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))


# ------------------------------------------------------------------- model
def get_predictor():
    global _predictor
    with _model_lock:
        if _predictor is None:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
            if not os.path.exists(CKPT):
                raise HTTPException(500, "checkpoint missing: run ./setup.sh")
            t = time.perf_counter()
            _predictor = build_sam2_video_predictor(
                CFG, CKPT, device=torch.device(DEVICE))
            print("[model] EdgeTAM loaded on %s in %.2fs" % (DEVICE, time.perf_counter() - t),
                  flush=True)
        return _predictor


def _sync():
    if DEVICE == "mps":
        import torch
        torch.mps.synchronize()


# -------------------------------------------------------------------- API
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(STATIC, "index.html")) as f:
        return HTMLResponse(f.read())


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
        "precision": "fp16" if FP16 else "fp32",
        "max_objects": MAX_OBJECTS,
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), max_seconds: float = Form(10.0),
                 fps: int = Form(30), max_frames: int = Form(300)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".mp4", ".mov", ".m4v", ".webm"):
        raise HTTPException(400, "expected .mp4 / .mov")
    jid = uuid.uuid4().hex[:12]
    d = os.path.join(JOBS, jid)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(d, "source" + ext)
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)

    probe = ffprobe_json(src)
    files = extract_frames(src, os.path.join(d, "frames"), max_seconds, fps, max_frames)
    if not files:
        raise HTTPException(400, "no frames extracted")
    w, h = Image.open(os.path.join(d, "frames", files[0])).size
    meta = {
        "job": jid, "source": os.path.basename(src), "filename": file.filename,
        "n_frames": len(files), "w": w, "h": h, "fps": fps,
        "source_duration_s": float(probe.get("format", {}).get("duration") or 0),
        "created": time.time(),
    }
    write_meta(jid, meta)
    with _state_lock:
        _jobs[jid] = new_status(len(files))
    return {"job": jid, "n_frames": len(files), "w": w, "h": h, "fps": fps,
            "source_duration_s": meta["source_duration_s"]}


@app.get("/api/jobs/{jid}/meta")
def get_meta(jid: str):
    job_dir(jid)
    return read_meta(jid)


@app.get("/api/jobs/{jid}/frame/{n}")
def get_frame(jid: str, n: int):
    p = os.path.join(job_dir(jid), "frames", "%04d.jpg" % n)
    if not os.path.exists(p):
        raise HTTPException(404, "no such frame")
    return FileResponse(p, media_type="image/jpeg",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/jobs/{jid}/mask/{obj}/{n}")
def get_mask(jid: str, obj: str, n: int):
    if not obj.isalnum():
        raise HTTPException(400, "bad object id")
    p = os.path.join(job_dir(jid), "masks", obj, "%04d.png" % n)
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


class TrackReq(BaseModel):
    frame_idx: int = 0
    objects: list[TrackObject]


@app.post("/api/jobs/{jid}/track")
def track(jid: str, req: TrackReq):
    d = job_dir(jid)
    meta = read_meta(jid)
    if not req.objects:
        raise HTTPException(400, "no objects")
    if len(req.objects) > MAX_OBJECTS:
        raise HTTPException(400, "at most %d objects" % MAX_OBJECTS)
    for o in req.objects:
        if not o.points and not o.box:
            raise HTTPException(400, "object %s has no prompt" % o.id)
    if not (0 <= req.frame_idx < meta["n_frames"]):
        raise HTTPException(400, "frame_idx out of range")

    with _state_lock:
        st = _jobs.get(jid) or new_status(meta["n_frames"])
        if st["state"] in ("tracking", "loading"):
            raise HTTPException(409, "already tracking")
        st.update(state="loading", done_frames=0, elapsed_s=0.0, fps=0.0, error=None,
                  n_frames=meta["n_frames"],
                  objects=[str(o.id) for o in req.objects])
        _jobs[jid] = st

    prompts = [o.model_dump() for o in req.objects]
    write_meta(jid, {**meta, "prompts": {"frame_idx": req.frame_idx, "objects": prompts}})
    threading.Thread(target=_track_worker, args=(jid, d, req), daemon=True).start()
    return {"job": jid, "state": "loading", "objects": [str(o.id) for o in req.objects]}


def _set(jid, **kw):
    with _state_lock:
        _jobs[jid].update(kw)


def _track_worker(jid, d, req):
    import torch
    t0 = time.perf_counter()
    try:
        with _gpu_lock:
            predictor = get_predictor()
            frames_dir = os.path.join(d, "frames")
            mroot = os.path.join(d, "masks")
            shutil.rmtree(mroot, ignore_errors=True)
            for o in req.objects:
                os.makedirs(os.path.join(mroot, str(o.id)), exist_ok=True)

            cast = (torch.autocast("mps", dtype=torch.float16) if FP16
                    else contextlib.nullcontext())
            with torch.inference_mode(), cast:
                state = predictor.init_state(frames_dir, offload_video_to_cpu=True)
                _sync()
                n_frames = state["num_frames"]
                _set(jid, state="tracking", n_frames=n_frames)

                for o in req.objects:
                    pts = np.array([[p[0], p[1]] for p in o.points], np.float32) \
                        if o.points else None
                    lbl = np.array([int(p[2]) for p in o.points], np.int32) \
                        if o.points else None
                    box = np.array(o.box, np.float32) if o.box else None
                    predictor.add_new_points_or_box(
                        state, frame_idx=req.frame_idx, obj_id=int(o.id),
                        points=pts, labels=lbl, box=box)
                _sync()

                seen = set()

                def drain(gen):
                    for fidx, obj_ids, masks in gen:
                        arr = masks.float().cpu().numpy()
                        for k, oid in enumerate(obj_ids):
                            soft = 1.0 / (1.0 + np.exp(-arr[k, 0]))
                            Image.fromarray(
                                (soft * 255.0).round().clip(0, 255).astype(np.uint8),
                                mode="L"
                            ).save(os.path.join(mroot, str(oid), "%04d.png" % int(fidx)))
                        seen.add(int(fidx))
                        el = time.perf_counter() - t0
                        _set(jid, done_frames=len(seen), elapsed_s=round(el, 2),
                             fps=round(len(seen) / el, 2) if el > 0 else 0.0)

                # a click on a middle frame must fill the whole clip, so run both ways
                if req.frame_idx > 0:
                    drain(predictor.propagate_in_video(
                        state, start_frame_idx=req.frame_idx, reverse=True))
                drain(predictor.propagate_in_video(
                    state, start_frame_idx=req.frame_idx, reverse=False))

                del state
            if DEVICE == "mps":
                torch.mps.empty_cache()

        el = time.perf_counter() - t0
        _set(jid, state="done", elapsed_s=round(el, 2),
             fps=round(len(seen) / el, 2) if el > 0 else 0.0)
        print("[track] %s: %d frames, %d obj, %.1fs (%.2f fps, %s)"
              % (jid, len(seen), len(req.objects), el, len(seen) / max(el, 1e-6),
                 "fp16" if FP16 else "fp32"), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _set(jid, state="error", error="%s: %s" % (type(e).__name__, e))


class RenderReq(BaseModel):
    subjects: list[dict] = []          # [{id, palette:[hex,...]}] — [] = whole frame
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


@app.post("/api/jobs/{jid}/render")
def start_render(jid: str, req: RenderReq):
    d = job_dir(jid)
    meta = read_meta(jid)
    mroot = os.path.join(d, "masks")
    resolved = []
    for i, s in enumerate(req.subjects):
        oid = str(s.get("id"))
        md = os.path.join(mroot, oid)
        if not os.path.isdir(md):
            raise HTTPException(400, "no masks for subject %s - track first" % oid)
        resolved.append({"masks": md,
                         "palette": s.get("palette"),
                         "dot": s.get("dot") or R.SUBJECT_COLORS[i % 6]})
    if req.mode == "dots" and not resolved:
        raise HTTPException(400, "the dots look needs at least one tracked subject")

    with _state_lock:
        st = _jobs.get(jid) or new_status(meta["n_frames"])
        if st["render"]["state"] == "rendering":
            raise HTTPException(409, "already rendering")
        st["render"] = {"state": "rendering", "done_frames": 0,
                        "n_frames": meta["n_frames"], "elapsed_s": 0.0,
                        "fps": 0.0, "error": None}
        _jobs[jid] = st

    params = req.model_dump()
    params.pop("subjects", None)
    if params.get("fps") is None:
        params["fps"] = meta.get("fps", 30)
    threading.Thread(target=_render_worker, args=(jid, d, resolved, params),
                     daemon=True).start()
    return {"job": jid, "state": "rendering", "subjects": len(resolved)}


def _render_worker(jid, d, subs, params):
    t0 = time.perf_counter()
    out = os.path.join(d, "out.mp4")
    try:
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
                                        fps=round(info["frames"] / max(el, 1e-6), 2))
        print("[render] %s: %d frames in %.1fs" % (jid, info["frames"], el), flush=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        with _state_lock:
            _jobs[jid]["render"].update(state="error",
                                        error="%s: %s" % (type(e).__name__, e))


@app.get("/api/jobs/{jid}/out.mp4")
def get_out(jid: str):
    p = os.path.join(job_dir(jid), "out.mp4")
    if not os.path.exists(p):
        raise HTTPException(404, "not rendered yet")
    # inline (not attachment) so the in-page <video> can play it; the UI's
    # download link carries a `download` attribute of its own
    return FileResponse(p, media_type="video/mp4", headers={
        "Content-Disposition": 'inline; filename="dither-%s.mp4"' % jid,
        "Accept-Ranges": "bytes"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DV_PORT", 8765)),
                log_level="info")
