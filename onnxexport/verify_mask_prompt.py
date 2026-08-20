#!/usr/bin/env python3
"""Check `heads_mask.onnx` against torch's `add_new_mask` on a real frame.

    env/venv/bin/python onnxexport/verify_mask_prompt.py [--job <dir>]

A lasso/polygon prompt goes to EdgeTAM as a *mask* prompt, and EdgeTAM's
`use_mask_input_as_output_without_sam: true` sends that down
`_use_mask_as_output`: the drawn mask becomes the output logits directly and
the SAM decoder is run only to produce the object pointer. This rasterises a
polygon with PIL, feeds it to the real video predictor, and scores the exported
graph against what the predictor stored.

Two ONNX runs are reported:

* **graph-only** — the graph is fed torch's own backbone features, so the
  number is the heads-graph rewrite alone;
* **end-to-end** — the graph is fed `encoder.onnx`'s features from the same
  JPEG, so the number also carries the encoder's own fp32/fp16 drift.
"""
import argparse
import os
import shutil
import sys
import tempfile
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'server'))
sys.path.insert(0, os.path.join(ROOT, 'env', 'EdgeTAM'))

import numpy as np                                          # noqa: E402
import onnxruntime as ort                                   # noqa: E402
import torch                                                # noqa: E402
from PIL import Image, ImageDraw                            # noqa: E402

from onnxexport.verify_loop import MEAN, STD, sess           # noqa: E402


def load_frame(path, size):
    """`sam2.utils.misc._load_img_as_tensor`'s preprocessing.

    Note the BICUBIC: EdgeTAM's loader calls `Image.resize` with no filter
    argument, so it gets PIL's default. `verify_loop.load_frame` uses BILINEAR
    instead, because that is what a browser canvas does — and the two disagree
    by 0.203 max-abs on `f2`, which would swamp the number this script is
    trying to measure.
    """
    im = Image.open(path).convert('RGB').resize((size, size), Image.BICUBIC)
    a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def find_job(root):
    for d in sorted(os.listdir(os.path.join(root, 'jobs'))):
        f = os.path.join(root, 'jobs', d, 'frames', '0000.jpg')
        if os.path.exists(f):
            return os.path.join(root, 'jobs', d)
    raise SystemExit('no jobs/*/frames/0000.jpg found')


def polygon(w, h):
    """A rough closed outline over the middle of the frame, in clip pixels."""
    cx, cy, rx, ry = w * 0.5, h * 0.52, w * 0.20, h * 0.34
    pts = [(0.00, -1.00), (0.45, -0.86), (0.78, -0.40), (1.00, 0.10),
           (0.72, 0.62), (0.30, 0.95), (-0.22, 1.00), (-0.70, 0.66),
           (-1.00, 0.05), (-0.80, -0.50), (-0.40, -0.90)]
    return [(cx + x * rx, cy + y * ry) for x, y in pts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', default=None)
    ap.add_argument('--models', default=os.path.join(ROOT, 'web', 'models'))
    ap.add_argument('--image-size', type=int, default=768)
    a = ap.parse_args()
    job = a.job or find_job(ROOT)
    frame = os.path.join(job, 'frames', '0000.jpg')
    print('[verify] job %s' % job, flush=True)

    W, H = Image.open(frame).size
    poly = polygon(W, H)
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon(poly, fill=255)
    mask_clip = np.asarray(im) > 127
    print('[verify] frame %dx%d, polygon covers %.2f%% of it'
          % (W, H, 100.0 * mask_clip.mean()), flush=True)

    # ------------------------------------------------------------- torch
    from sam2.build_sam import build_sam2_video_predictor
    import edgetam_util
    ckpt = os.path.join(ROOT, 'env', 'EdgeTAM', 'checkpoints', 'edgetam.pt')
    pred = build_sam2_video_predictor(
        'configs/edgetam.yaml', ckpt, device=torch.device('cpu'),
        hydra_overrides_extra=edgetam_util.hydra_overrides(a.image_size))
    edgetam_util.set_image_size(pred, a.image_size)
    pred.eval()

    tmp = tempfile.mkdtemp()
    try:
        shutil.copy(frame, os.path.join(tmp, '0000.jpg'))
        with torch.inference_mode():
            st = pred.init_state(tmp, offload_video_to_cpu=True)
            pred.add_new_mask(st, frame_idx=0, obj_id=1, mask=mask_clip)
            # what add_new_mask actually handed the model: the polygon resized
            # to image_size with an antialiased bilinear and re-thresholded
            mask_full = st['mask_inputs_per_obj'][0][0].float()   # [1,1,S,S]
            out = st['temp_output_dict_per_obj'][0]['cond_frame_outputs'][0]
            t_low = out['pred_masks'].float().cpu().numpy()       # [1,1,192,192]
            t_ptr = out['obj_ptr'].float().cpu().numpy()          # [1,256]
            t_osl = out['object_score_logits'].float().cpu().numpy()
            _, _, vfeats, _, fsz = pred._get_image_feature(st, 0, 1)
            hi_res = [x.permute(1, 2, 0).reshape(1, -1, *s)
                      for x, s in zip(vfeats[:-1], fsz[:-1])]
            t_pix = vfeats[-1].permute(1, 2, 0).reshape(
                1, -1, *fsz[-1]).contiguous()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    S = a.image_size
    mk = mask_full.cpu().numpy().astype(np.float32)
    print('[verify] torch  osl=%.4f  |ptr|=%.4f  mask logits in [%.2f, %.2f]'
          % (t_osl[0, 0], np.abs(t_ptr).max(), t_low.min(), t_low.max()),
          flush=True)

    # -------------------------------------------------------------- onnx
    hm = sess(os.path.join(a.models, 'heads_mask.onnx'))
    names = [i.name for i in hm.get_inputs()]
    print('[verify] heads_mask.onnx inputs: %s' % names, flush=True)

    def run(pix, f0, f1, s, dt=np.float32):
        feed = {'pix_feat': pix.astype(dt), 'mask_full': mk.astype(dt)}
        for n, v in (('f0', f0), ('f1', f1)):
            if n in [i.name for i in s.get_inputs()]:
                feed[n] = v.astype(dt)
        m, i, p, o = s.run(None, feed)
        return (m.astype(np.float32), i.astype(np.float32),
                p.astype(np.float32), o.astype(np.float32))

    f0t = hi_res[0].numpy().astype(np.float32)
    f1t = hi_res[1].numpy().astype(np.float32)
    m, iou, ptrs, osl = run(t_pix.numpy().astype(np.float32), f0t, f1t, hm)
    score(a, 'graph-only  fp32', m, iou, ptrs, osl, t_low, t_ptr, t_osl)

    enc = sess(os.path.join(a.models, 'encoder.onnx'))
    img = load_frame(frame, S)
    f0, f1, f2 = enc.run(None, {'image': img.astype(np.float32)})
    print('[verify] encoder f2 vs torch max_abs=%.3e'
          % np.abs(f2 - t_pix.numpy()).max(), flush=True)
    m, iou, ptrs, osl = run(f2, f0, f1, hm)
    score(a, 'end-to-end  fp32', m, iou, ptrs, osl, t_low, t_ptr, t_osl)

    p16 = os.path.join(a.models, 'heads_mask.fp16.onnx')
    if os.path.exists(p16):
        h16 = sess(p16)
        e16 = sess(os.path.join(a.models, 'encoder.fp16.onnx'))
        g0, g1, g2 = e16.run(None, {'image': img.astype(np.float16)})
        m, iou, ptrs, osl = run(g2, g0, g1, h16, np.float16)
        score(a, 'end-to-end  fp16', m, iou, ptrs, osl, t_low, t_ptr, t_osl)


def score(a, tag, m, iou, ptrs, osl, t_low, t_ptr, t_osl):
    k = 0                       # every slot holds the same thing; see the docstring
    low = m[:, k:k + 1]
    ptr = ptrs[0, k]
    d_low = float(np.abs(low - t_low).max())
    d_ptr = float(np.abs(ptr - t_ptr[0]).max())
    g, r = low[0, 0] > 0, t_low[0, 0] > 0
    u = np.logical_or(g, r).sum()
    iou_bin = 1.0 if u == 0 else float(np.logical_and(g, r).sum() / u)
    spread_m = float(np.abs(m - m[:, :1]).max())
    spread_p = float(np.abs(ptrs - ptrs[:, :1]).max())
    print('[%s] low_res_masks max_abs=%.3e   obj_ptr max_abs=%.3e'
          % (tag, d_low, d_ptr))
    print('%s  binarised IoU=%.6f   osl onnx=%.4f torch=%.4f   '
          'ious=%s' % (' ' * (len(tag) + 2), iou_bin, osl[0, 0], t_osl[0, 0],
                       np.array2string(iou[0], precision=3)))
    print('%s  slot spread: masks=%.3e ptrs=%.3e (both must be 0)'
          % (' ' * (len(tag) + 2), spread_m, spread_p), flush=True)
    _ = a


if __name__ == '__main__':
    main()
