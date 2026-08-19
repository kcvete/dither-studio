#!/usr/bin/env python3
"""Per-stage wall-clock profile of one EdgeTAM propagate pass.

    env/venv/bin/python bench/profile_stages.py [--precision fp16] [--fastpath]

Wraps the modules that make up a track step, synchronising MPS around each so
the numbers are real GPU time and not queue-submission time.
"""
import argparse, contextlib, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.environ.setdefault("TQDM_DISABLE", "1")
import numpy as np, torch
from bench import REF, CKPT, CFG, build_predictor, BASE, opts_for

T = {}


def timed(name, fn, sync):
    def wrap(*a, **k):
        sync()
        t = time.perf_counter()
        r = fn(*a, **k)
        sync()
        T[name] = T.get(name, 0.0) + time.perf_counter() - t
        return r
    return wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch-fp16")
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--frames", type=int, default=60)
    a = ap.parse_args()
    o = opts_for(a.backend, a.opt)
    dev = o["device"]

    def sync():
        if dev == "mps":
            torch.mps.synchronize()

    p = build_predictor(o)
    if o["coreml"]:
        from coreml import accel as A
        acc = A.install(p, directory=o["coreml_dir"] or A.dir_for(o["image_size"]),
                        prefetch=o["coreml_prefetch"],
                        stages=tuple(o["coreml_stages"].split(",")))
        acc.warm((1,))
    if o["precision"] in ("fp16", "half"):
        cast = torch.autocast(dev, dtype=torch.float16)
    else:
        cast = contextlib.nullcontext()

    p.forward_image = timed("image_encoder", p.forward_image, sync)
    p.memory_attention.forward = timed("memory_attention", p.memory_attention.forward, sync)
    p._encode_new_memory = timed("memory_encoder+perceiver", p._encode_new_memory, sync)
    p._forward_sam_heads = timed("sam_heads", p._forward_sam_heads, sync)
    p._prepare_memory_conditioned_features = timed(
        "prep_memcond(total)", p._prepare_memory_conditioned_features, sync)
    p.track_step = timed("track_step(total)", p.track_step, sync)
    p.image_encoder.trunk.forward = timed("  trunk(repvit)", p.image_encoder.trunk.forward, sync)
    p.image_encoder.neck.forward = timed("  neck(fpn)", p.image_encoder.neck.forward, sync)
    p.memory_encoder.forward = timed("  mem_enc.conv", p.memory_encoder.forward, sync)
    p.spatial_perceiver.forward = timed("  spatial_perceiver", p.spatial_perceiver.forward, sync)
    for i, l in enumerate(p.memory_attention.layers):
        l.self_attn.forward = timed("  ma%d.self_attn" % i, l.self_attn.forward, sync)
        l.cross_attn_image.forward = timed("  ma%d.cross_attn" % i, l.cross_attn_image.forward, sync)

    prompt = json.load(open(os.path.join(REF, "prompt.json")))
    with torch.inference_mode(), cast:
        state = p.init_state(os.path.join(REF, "frames"), offload_video_to_cpu=True)
        p.add_new_points_or_box(state, frame_idx=0, obj_id=1,
                                points=np.array([prompt["point"]], np.float32),
                                labels=np.array([1], np.int32),
                                box=np.array(prompt["box"], np.float32))
        sync()
        T.clear()
        n = 0
        t0 = time.perf_counter()
        for fidx, oids, masks in p.propagate_in_video(state):
            _ = masks[0, 0].float().cpu().numpy()
            n += 1
            if n >= a.frames:
                break
        sync()
        wall = time.perf_counter() - t0

    print("\nbackend=%s frames=%d wall=%.2fs (%.2f fps)" % (a.backend, n, wall, n / wall))
    tot = T.get("track_step(total)", wall)
    for k in ("image_encoder", "  trunk(repvit)", "  neck(fpn)",
              "prep_memcond(total)", "memory_attention",
              "  ma0.self_attn", "  ma0.cross_attn", "  ma1.self_attn", "  ma1.cross_attn",
              "sam_heads", "memory_encoder+perceiver", "  mem_enc.conv",
              "  spatial_perceiver", "track_step(total)"):
        if k in T:
            print("%-28s %7.2f s  %5.1f%%  %6.1f ms/frame"
                  % (k, T[k], 100 * T[k] / tot, 1000 * T[k] / n))
    print("%-28s %7.2f s  %5.1f%%  %6.1f ms/frame"
          % ("[outside track_step]", wall - tot, 100 * (wall - tot) / tot,
             1000 * (wall - tot) / n))


if __name__ == "__main__":
    main()
