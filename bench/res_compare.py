#!/usr/bin/env python3
"""Does the tracker's input resolution show up in the dithered output?

    env/venv/bin/python bench/res_compare.py

Tracks the reference clip at 1024 / 768 / 512 with the CoreML backend, renders
each mask set through the real renderer (Dots, defaults, Sage), and writes

    bench/res_compare.mp4         three-up, labelled with size + fps
    bench/res_compare_sheet.png   4 frames x 3 sizes, 1:1 crops on the subject

The crops are the point: a 3-up at 640 px wide hides exactly the edge detail
the question is about.
"""
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import render as R
from bench import REF, build_predictor, opts_for, load_ref_masks, iou

SIZES = (1024, 768, 512)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
SHEET_FRAMES = (12, 55, 98, 140)
CROP = 420


def track_timed(size, mask_dir):
    """Track the clip at one input size. Returns (fps, per-frame IoU vs the
    1024 fp32 reference). Mask writing is outside the timed region."""
    import torch
    o = opts_for("coreml", ["image_size=%d" % size])
    p = build_predictor(o)
    from coreml import accel as A
    acc = A.install(p, directory=A.dir_for(size), verbose=False)
    if acc is None:
        raise SystemExit("no CoreML export for %d" % size)
    acc.warm((1,))
    prompt = json.load(open(os.path.join(REF, "prompt.json")))
    ref = load_ref_masks(os.path.join(REF, "masks_edgetam"), 150)
    shutil.rmtree(mask_dir, ignore_errors=True)
    os.makedirs(mask_dir, exist_ok=True)
    ious, model_s = [], 0.0
    with torch.inference_mode(), torch.autocast("mps", dtype=torch.float16):
        st = p.init_state(os.path.join(REF, "frames"), offload_video_to_cpu=True)
        p.add_new_points_or_box(
            st, frame_idx=0, obj_id=1,
            points=np.array([prompt["point"]], np.float32),
            labels=np.array([1], np.int32),
            box=np.array(prompt["box"], np.float32))
        torch.mps.synchronize()
        gen = p.propagate_in_video(st)
        while True:
            t = time.perf_counter()
            try:
                fidx, oids, masks = next(gen)
            except StopIteration:
                model_s += time.perf_counter() - t
                break
            arr = masks[0, 0].float().cpu().numpy()
            model_s += time.perf_counter() - t
            soft = 1.0 / (1.0 + np.exp(-arr))
            ious.append(iou(arr > 0, ref[int(fidx)]))
            Image.fromarray((soft * 255).round().clip(0, 255).astype(np.uint8), "L") \
                .save(os.path.join(mask_dir, "%04d.png" % int(fidx)))
    del p
    torch.mps.empty_cache()
    return len(ious) / model_s, ious


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.path.join(HERE, "_res"))
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    frames = os.path.join(REF, "frames")
    os.makedirs(a.work, exist_ok=True)
    info = {}
    for s in SIZES:
        md = os.path.join(a.work, str(s), "1")
        fps, ious = track_timed(s, md)
        info[s] = {"fps": round(fps, 2), "iou_mean": round(float(np.mean(ious)), 4),
                   "iou_min": round(float(np.min(ious)), 4)}
        print("[res] %d: %.2f fps, IoU %.4f / %.4f"
              % (s, fps, info[s]["iou_mean"], info[s]["iou_min"]), flush=True)
        mp4 = os.path.join(a.work, "%d.mp4" % s)
        R.render(frames, [{"masks": md, "dot": "#b0413e", "palette": None}], mp4,
                 dict(mode="dots", compose="cutout", bg="#c9d4c5", fps=30))
        print("[res] %d: rendered %s" % (s, mp4), flush=True)

    # ------------------------------------------------------------ three-up
    ins, filt = [], []
    for i, s in enumerate(SIZES):
        ins += ["-i", os.path.join(a.work, "%d.mp4" % s)]
        filt.append("[%d:v]scale=640:360,drawtext=fontfile=%s:text='%d px  %.1f fps':"
                    "x=10:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=0x000000AA:"
                    "boxborderw=6[v%d]" % (i, FONT, s, info[s]["fps"], i))
    fc = ";".join(filt) + ";" + "".join("[v%d]" % i for i in range(3)) + "hstack=inputs=3[o]"
    out_mp4 = os.path.join(a.out, "res_compare.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y"] + ins +
                   ["-filter_complex", fc, "-map", "[o]", "-c:v", "libx264",
                    "-crf", "18", "-pix_fmt", "yuv420p", out_mp4], check=True)
    print("[res] wrote " + out_mp4, flush=True)

    # ---------------------------------------------------------- crop sheet
    font = ImageFont.truetype(FONT, 22)
    small = ImageFont.truetype(FONT, 18)
    pad, head = 8, 34
    W = 3 * (CROP + pad) + pad
    H = head + 4 * (CROP + pad + 20) + pad
    sheet = Image.new("RGB", (W, H), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    for ci, s in enumerate(SIZES):
        d.text((pad + ci * (CROP + pad) + 4, 6),
               "%d px  ·  %.1f fps  ·  IoU %.3f" % (s, info[s]["fps"], info[s]["iou_mean"]),
               font=font, fill=(235, 235, 235))
    for ri, fr in enumerate(SHEET_FRAMES):
        # centre the crop on the 1024-run's mask so all three show the same pixels
        m = np.array(Image.open(os.path.join(a.work, "1024", "1", "%04d.png" % fr)))
        ys, xs = np.nonzero(m > 127)
        cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (360, 640)
        for ci, s in enumerate(SIZES):
            png = os.path.join(a.work, "f%d_%d.png" % (fr, s))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i",
                            os.path.join(a.work, "%d.mp4" % s), "-vf",
                            "select=eq(n\\,%d)" % fr, "-vframes", "1", png], check=True)
            im = Image.open(png).convert("RGB")
            x0 = max(0, min(im.width - CROP, cx - CROP // 2))
            y0 = max(0, min(im.height - CROP, cy - CROP // 2))
            x = pad + ci * (CROP + pad)
            y = head + ri * (CROP + pad + 20)
            sheet.paste(im.crop((x0, y0, x0 + CROP, y0 + CROP)), (x, y))
            d.text((x + 4, y + CROP + 2), "frame %d · %d px" % (fr, s),
                   font=small, fill=(190, 190, 190))
    out_png = os.path.join(a.out, "res_compare_sheet.png")
    sheet.save(out_png)
    print("[res] wrote " + out_png, flush=True)
    with open(os.path.join(a.out, "res_compare.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
