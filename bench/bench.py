#!/usr/bin/env python3
"""Benchmark harness for EdgeTAM subject tracking backends.

    env/venv/bin/python bench/bench.py --backend torch-fp16 --runs 2

Runs a named backend over the reference clip's frames with the reference
prompt, times the model, scores every mask against the reference masks
(binary IoU at 0.5) and appends a row to bench/results.md.

A "backend" here is just a bag of options; `BACKENDS` names the useful
combinations. `--opt k=v` overrides any of them ad hoc.
"""
import argparse
import contextlib
import json
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'server'))
sys.path.insert(0, HERE)

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from PIL import Image

REF = os.environ.get(
    "DV_BENCH_CLIP",
    "/private/tmp/claude-501/-Users-kevincvetezar/012258e2-fe5c-46ee-8648-eeafdcc38f82"
    "/scratchpad/parkour")
CKPT = os.path.join(ROOT, "env", "EdgeTAM", "checkpoints", "edgetam.pt")
CFG = "configs/edgetam.yaml"

# ---------------------------------------------------------------- backends
BASE = dict(
    device="mps",
    precision="fp16",        # fp32 | fp16 (autocast) | half (model.half())
    image_size=1024,
    channels_last=False,
    compile="none",          # none | default | reduce-overhead | max-autotune
    compile_what="track",    # track | encoder | memattn
    num_maskmem=7,
    max_obj_ptrs=16,
    coreml=False,
    coreml_prefetch=True,    # run the CoreML image encoder a frame ahead
    coreml_stages='encoder,memattn,memenc',
    coreml_dir="",           # defaults to env/coreml/<image_size>
    postprocess=True,        # fill_hole_area / non-overlap constraints
    offload_video=True,
    fastpath=False,          # real-arithmetic RoPE (fastpath.py)
    prefetch=0,              # image-encoder batch size (0 = upstream, one frame)
    posenc_expand=False,     # stop re-copying the cached FPN position encodings
)

BACKENDS = {
    # --- what the app can be set to (DV_BACKEND) -------------------------
    "torch":             dict(precision="fp16"),          # the old default
    "torch-half":        dict(precision="half"),
    "torch-compiled":    dict(precision="half", compile="default",
                              compile_what="encoder"),
    "coreml":            dict(precision="half", coreml=True),
    "coreml-768":        dict(precision="half", coreml=True, image_size=768),
    "coreml-512":        dict(precision="half", coreml=True, image_size=512),
    # --- everything else that was measured -------------------------------
    "torch-fp32":        dict(precision="fp32"),
    "torch-fast":        dict(precision="fp16", fastpath=True),
    "torch-fast-half":   dict(precision="half", fastpath=True),
    "torch-cl":          dict(precision="fp16", channels_last=True),
    "torch-lean":        dict(precision="fp16", num_maskmem=3, max_obj_ptrs=4),
    "torch-768":         dict(precision="fp16", image_size=768),
    "torch-512":         dict(precision="fp16", image_size=512),
    "torch-pf4":         dict(precision="half", fastpath=True, prefetch=4,
                              posenc_expand=True),
    "torch-pe":          dict(precision="half", fastpath=True, posenc_expand=True),
    "torch-compiled-track": dict(precision="fp16", compile="default",
                                 compile_what="track"),
    "torch-compiled-ro": dict(precision="fp16", compile="reduce-overhead",
                              compile_what="encoder"),
    "coreml-fp32":       dict(precision="fp32", coreml=True),
    "coreml-nopf":       dict(precision="half", coreml=True, coreml_prefetch=False),
    "coreml-encoder-only":  dict(precision="half", coreml=True,
                                 coreml_stages="encoder"),
    "coreml-memory-only":   dict(precision="half", coreml=True,
                                 coreml_stages="memattn,memenc"),
    "coreml-sam-compiled": dict(precision="half", coreml=True, compile="default",
                                compile_what="samheads"),
}


def opts_for(name, extra):
    o = dict(BASE)
    if name in BACKENDS:
        o.update(BACKENDS[name])
    elif name != "custom":
        raise SystemExit("unknown backend %r; known: %s" % (name, ", ".join(BACKENDS)))
    for kv in extra or []:
        k, _, v = kv.partition("=")
        if k not in o:
            raise SystemExit("unknown option %r" % k)
        cur = o[k]
        if isinstance(cur, bool):
            o[k] = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            o[k] = int(v)
        else:
            o[k] = v
    return o


# ------------------------------------------------------------------- IoU
def load_ref_masks(d, n):
    out = []
    for i in range(n):
        p = os.path.join(d, "%04d.png" % i)
        out.append(np.array(Image.open(p)) > 127)
    return out


def iou(a, b):
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    return 1.0 if union == 0 else float(inter) / float(union)


# ----------------------------------------------------------------- runner
def build_predictor(o):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    import edgetam_util
    ov = edgetam_util.hydra_overrides(o["image_size"])
    p = build_sam2_video_predictor(CFG, CKPT, device=torch.device(o["device"]),
                                   hydra_overrides_extra=ov,
                                   apply_postprocessing=o["postprocess"])
    # these two are plain runtime attributes; setting them through hydra would
    # resize `maskmem_tpos_enc` and break the checkpoint load, so set them after.
    p.num_maskmem = o["num_maskmem"]
    p.max_obj_ptrs_in_encoder = o["max_obj_ptrs"]
    import fastpath
    if o["fastpath"]:
        fastpath.install(p)
    if o["posenc_expand"]:
        fastpath.install_posenc_expand()
    if o["prefetch"] > 1:
        fastpath.install_prefetch(p, o["prefetch"])
    edgetam_util.set_image_size(p, o["image_size"])
    if o["precision"] == "half":
        p = p.half()
        # init_state keeps the decoded frames in fp32 and `_get_image_feature`
        # forces `.float()`, so the encoder would get an fp32 tensor against
        # fp16 weights; cast at the door instead.
        _fwd = p.forward_image
        p.forward_image = lambda img, _f=_fwd: _f(img.half())
    if o["channels_last"]:
        p.image_encoder = p.image_encoder.to(memory_format=torch.channels_last)
        p.memory_encoder = p.memory_encoder.to(memory_format=torch.channels_last)
    return p


def apply_compile(p, o):
    import torch
    if o["compile"] == "none":
        return p, None
    mode = None if o["compile"] == "default" else o["compile"]
    kw = dict(dynamic=False, fullgraph=False)
    if mode:
        kw["mode"] = mode
    what = o["compile_what"]
    if what == "track":
        p.track_step = torch.compile(p.track_step, **kw)
    elif what == "encoder":
        p.forward_image = torch.compile(p.forward_image, **kw)
    elif what == "memattn":
        p.memory_attention = torch.compile(p.memory_attention, **kw)
    elif what == "samheads":
        p._forward_sam_heads = torch.compile(p._forward_sam_heads, **kw)
    elif what == "both":
        p.forward_image = torch.compile(p.forward_image, **kw)
        p.track_step = torch.compile(p.track_step, **kw)
    else:
        raise SystemExit("bad compile_what")
    return p, what


def run_once(o, n_expect, ref_masks, warmup_frames=0, quiet=False):
    import torch
    dev = o["device"]

    def sync():
        if dev == "mps":
            torch.mps.synchronize()
        elif dev == "cuda":
            torch.cuda.synchronize()

    prompt = json.load(open(os.path.join(REF, "prompt.json")))
    frames_dir = os.path.join(REF, "frames")

    wall0 = time.perf_counter()
    t = time.perf_counter()
    p = build_predictor(o)
    load_s = time.perf_counter() - t

    accel = None
    if o["coreml"]:
        from coreml import accel as A
        cdir = o["coreml_dir"] or A.dir_for(o["image_size"])
        accel = A.install(p, directory=cdir, verbose=not quiet,
                          prefetch=o["coreml_prefetch"],
                          stages=tuple(o["coreml_stages"].split(",")))
        if accel is None:
            raise SystemExit("coreml backend requested but %s has no manifest.json"
                             % cdir)
        t = time.perf_counter()
        accel.warm((1,))
        if not quiet:
            print("[bench] coreml warm %.1fs" % (time.perf_counter() - t), flush=True)

    p, _ = apply_compile(p, o)

    # `half` still runs under autocast: the prompt encoder builds fp32 coordinate
    # tensors of its own and would hit fp16 weights without it.
    if o["precision"] in ("fp16", "half"):
        cast = torch.autocast(dev, dtype=torch.float16)
    else:
        cast = contextlib.nullcontext()

    ious = []
    with torch.inference_mode(), cast:
        t = time.perf_counter()
        state = p.init_state(frames_dir, offload_video_to_cpu=o["offload_video"])
        sync()
        init_s = time.perf_counter() - t
        n = state["num_frames"]

        t = time.perf_counter()
        p.add_new_points_or_box(
            state, frame_idx=int(prompt.get("frame", 0)), obj_id=1,
            points=np.array([prompt["point"]], np.float32),
            labels=np.array([1], np.int32),
            box=np.array(prompt["box"], np.float32))
        sync()
        prompt_s = time.perf_counter() - t

        prop_s = 0.0
        score_s = 0.0
        count = 0
        gen = p.propagate_in_video(state)
        while True:
            t = time.perf_counter()
            try:
                fidx, obj_ids, masks = next(gen)
            except StopIteration:
                prop_s += time.perf_counter() - t
                break
            arr = masks[0, 0].float().cpu().numpy()
            sync()
            prop_s += time.perf_counter() - t
            t = time.perf_counter()
            if ref_masks is not None:
                ious.append(iou(arr > 0, ref_masks[int(fidx)]))
            score_s += time.perf_counter() - t
            count += 1
        del state
    if dev == "mps":
        torch.mps.empty_cache()
    wall_s = time.perf_counter() - wall0

    model_s = init_s + prompt_s + prop_s
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    gpu_mb = 0.0
    if dev == "mps":
        try:
            gpu_mb = torch.mps.driver_allocated_memory() / 1e6
        except Exception:
            pass
    r = {
        "frames": count,
        "wall_s": round(wall_s, 2),
        "model_s": round(model_s, 3),
        "prop_s": round(prop_s, 3),
        "init_s": round(init_s, 3),
        "prompt_s": round(prompt_s, 3),
        "load_s": round(load_s, 2),
        "fps_model": round(count / max(prop_s, 1e-9), 2),
        "fps_track": round(count / max(model_s, 1e-9), 2),
        "iou_mean": round(float(np.mean(ious)), 4) if ious else None,
        "iou_min": round(float(np.min(ious)), 4) if ious else None,
        "iou_p1": round(float(np.percentile(ious, 1)), 4) if ious else None,
        "peak_mem_mb": round(peak_mb, 1),
        "gpu_mem_mb": round(gpu_mb, 1),
    }
    if accel is not None:
        r["coreml"] = accel.summary()
    del p
    return r


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch-fp16")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--note", default="")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--skip-first", action="store_true",
                    help="drop run 1 from the summary (torch.compile warm-up)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    o = opts_for(args.backend, args.opt)
    ref_dir = os.path.join(REF, "masks_edgetam")
    n_frames = len([f for f in os.listdir(os.path.join(REF, "frames"))
                    if f.endswith(".jpg")])
    ref = load_ref_masks(ref_dir, n_frames)

    print("[bench] backend=%s opts=%s" % (args.backend, {
        k: v for k, v in o.items() if BASE.get(k) != v}), flush=True)

    runs = []
    for i in range(args.runs):
        r = run_once(o, n_frames, ref, quiet=i > 0)
        runs.append(r)
        print("[bench] run %d/%d %s" % (i + 1, args.runs, json.dumps(
            {k: r[k] for k in ("fps_model", "fps_track", "wall_s", "iou_mean",
                               "iou_min", "peak_mem_mb")})), flush=True)

    if args.skip_first and len(runs) > 1:
        runs = runs[1:]
    best = max(runs, key=lambda r: r["fps_model"])
    med = median([r["fps_model"] for r in runs])
    med_wall = median([r["wall_s"] for r in runs])
    summary = {
        "backend": args.backend,
        "fps_model_best": best["fps_model"],
        "fps_model_median": round(med, 2),
        "fps_track_best": max(r["fps_track"] for r in runs),
        "wall_s_best": min(r["wall_s"] for r in runs),
        "wall_s_median": round(med_wall, 2),
        "iou_mean": best["iou_mean"],
        "iou_min": min(r["iou_min"] for r in runs if r["iou_min"] is not None),
        "peak_mem_mb": max(r["peak_mem_mb"] for r in runs),
        "runs": len(runs),
        "opts": {k: v for k, v in o.items() if BASE.get(k) != v},
        "note": args.note,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": summary, "runs": runs}, f, indent=2)

    if not args.no_append:
        md = os.path.join(HERE, "results.md")
        new = not os.path.exists(md)
        with open(md, "a") as f:
            if new:
                f.write("# EdgeTAM tracking benchmark\n\n"
                        "1 subject, 150 frames, 1280x720 source, M4 Pro / 24 GB.\n"
                        "`fps_model` = frames / propagate seconds (includes the GPU->CPU\n"
                        "mask copy, excludes PNG writing). `wall_s` = whole run including\n"
                        "model load and frame decode. IoU is binary-at-0.5 against\n"
                        "`masks_edgetam/` (torch fp32 MPS).\n\n"
                        "| backend | fps best | fps med | wall_s | IoU mean | IoU min "
                        "| peak MB | runs | note |\n"
                        "|---|---|---|---|---|---|---|---|---|\n")
            f.write("| %s | %.2f | %.2f | %.1f | %s | %s | %.0f | %d | %s |\n" % (
                args.backend, summary["fps_model_best"], summary["fps_model_median"],
                summary["wall_s_median"], summary["iou_mean"], summary["iou_min"],
                summary["peak_mem_mb"], summary["runs"],
                (args.note or "") + (" " + json.dumps(summary["opts"])
                                     if summary["opts"] else "")))
        print("[bench] appended to " + md, flush=True)


if __name__ == "__main__":
    main()
