#!/usr/bin/env python3
"""Run the exported graphs as the browser will, and score them against the
server's masks.

This is the reference implementation of the tracking loop that
`static/track-web/track.js` mirrors: same memory-bank bookkeeping, same
padding, same token selection. Getting it right here first means a JS bug
later is a JS bug, not a design bug.

    PYTHONPATH=<pylibs> env/venv/bin/python onnxexport/verify_loop.py \
        --clip <dir> --models static/track-web/models [--fp16] [--limit N]
"""
import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
NEG = -1e4          # additive mask for padded memory slots (fp16-safe)


def sess(path, providers=('CPUExecutionProvider',)):
    o = ort.SessionOptions()
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, o, providers=list(providers))


def load_frame(path, size):
    im = Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR)
    a = np.asarray(im, np.float32) / 255.0
    a = (a - MEAN) / STD
    return np.ascontiguousarray(a.transpose(2, 0, 1)[None])


def bilinear_to(logits, w, h):
    """192x192 logits -> (h,w), matching F.interpolate(align_corners=False)."""
    im = Image.fromarray(logits.astype(np.float32), mode='F')
    return np.asarray(im.resize((w, h), Image.BILINEAR), np.float32)


class Bank:
    """`_prepare_memory_conditioned_features`, minus PyTorch.

    Slot layout is the one `coreml/wrappers.py` assumes: `nspat` blocks of 512
    latents then `nptr` pointer tokens. Unused slots stay zero and are masked
    out of the cross-attention, which is what lets a single fixed-shape graph
    serve the cold-start frames too.
    """

    def __init__(self, man, tpos):
        self.NS, self.NP = man['nspat'], man['nptr']
        self.D = man['mem_dim']
        self.CH = man['ptr_tokens']          # 256 / 64 = 4 tokens per pointer
        self.MAXPTR = man['max_obj_ptrs']
        self.MEMLEN = man['memlen']
        self.tpos = tpos                     # [7,1,1,64]
        self.spatial = {}                    # frame -> (lat[512,64], lpos[512,64])
        self.ptr = {}                        # frame -> [256]
        self.cond = set()

    def add(self, idx, lat, lpos, ptr, is_cond):
        self.spatial[idx] = (lat, lpos)
        self.ptr[idx] = ptr
        if is_cond:
            self.cond.add(idx)

    def build(self, t):
        mem = np.zeros((self.MEMLEN, 1, self.D), np.float32)
        pos = np.zeros((self.MEMLEN, 1, self.D), np.float32)
        msk = np.full((1, 1, 1, self.MEMLEN), NEG, np.float32)

        picks = [(0, c) for c in sorted(self.cond)]
        for t_pos in range(1, self.NS):
            prev = t - (self.NS - t_pos)
            if prev in self.spatial and prev not in self.cond:
                picks.append((t_pos, prev))

        for slot, (t_pos, f) in enumerate(picks):
            lat, lpos = self.spatial[f]
            a, b = slot * 512, slot * 512 + 512
            mem[a:b, 0] = lat
            pos[a:b, 0] = lpos + self.tpos[self.NS - t_pos - 1].reshape(1, self.D)
            msk[0, 0, 0, a:b] = 0.0

        ptrs = [self.ptr[c] for c in sorted(self.cond) if c <= t]
        for d in range(1, self.MAXPTR):
            f = t - d
            if f < 0:
                break
            if f in self.ptr and f not in self.cond:
                ptrs.append(self.ptr[f])
        base = self.NS * 512
        for i, p in enumerate(ptrs[:self.MAXPTR]):
            a = base + i * self.CH
            mem[a:a + self.CH, 0] = p.reshape(self.CH, self.D)
            msk[0, 0, 0, a:a + self.CH] = 0.0
        return mem, pos, msk


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--clip', default='/private/tmp/claude-501/'
                    '-Users-kevincvetezar/012258e2-fe5c-46ee-8648-eeafdcc38f82/'
                    'scratchpad/parkour')
    ap.add_argument('--models', default=os.path.join(here, 'static', 'track-web', 'models'))
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--ref', default='masks_edgetam')
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.models, 'manifest.json')))
    S, G = man['image_size'], man['grid']
    cb = np.fromfile(os.path.join(a.models, 'consts.bin'), np.float32)
    C = {k: cb[v['offset'] // 4: v['offset'] // 4 + v['count']].reshape(v['shape'])
         for k, v in man['consts'].items()}
    no_mem = C['no_mem_embed'].astype(np.float32).reshape(1, -1, 1, 1)
    tpos = C['maskmem_tpos_enc'].astype(np.float32)          # [7,1,1,64]

    sfx = '.fp16' if a.fp16 else ''
    dt = np.float16 if a.fp16 else np.float32
    enc = sess(os.path.join(a.models, f'encoder{sfx}.onnx'))
    mat = sess(os.path.join(a.models, f'memattn{sfx}.onnx'))
    hds = sess(os.path.join(a.models, f'heads{sfx}.onnx'))
    hpr = sess(os.path.join(a.models, f'heads_prompt{sfx}.onnx'))
    mec = sess(os.path.join(a.models, 'memenc.onnx'))        # always fp32

    frames = sorted(os.listdir(os.path.join(a.clip, 'frames')))
    if a.limit:
        frames = frames[:a.limit]
    pr = json.load(open(os.path.join(a.clip, 'prompt.json')))
    W, H = Image.open(os.path.join(a.clip, 'frames', frames[0])).size
    sx, sy = S / W, S / H
    bx = pr['box']
    pc = np.array([[[bx[0] * sx, bx[1] * sy], [bx[2] * sx, bx[3] * sy],
                    [pr['point'][0] * sx, pr['point'][1] * sy]]], np.float32)
    pl = np.array([[2., 3., 1.]], np.float32)

    bank = Bank(man, tpos)
    ious, times = [], {k: [] for k in ('enc', 'mat', 'hds', 'mec', 'all')}
    refdir = os.path.join(a.clip, a.ref)

    for t, fn in enumerate(frames):
        t_all = time.perf_counter()
        img = load_frame(os.path.join(a.clip, 'frames', fn), S)
        t0 = time.perf_counter()
        f0, f1, f2 = enc.run(None, {'image': img.astype(dt)})
        times['enc'].append(time.perf_counter() - t0)
        f2f = f2.astype(np.float32)

        if t == 0:
            pix = f2f + no_mem      # heads_prompt can also do this in-graph
        else:
            mem, mpos, msk = bank.build(t)
            t0 = time.perf_counter()
            out = mat.run(None, {'feat': f2.astype(dt), 'memory': mem.astype(dt),
                                 'memory_pos': mpos.astype(dt),
                                 'mem_mask': msk.astype(dt)})[0]
            times['mat'].append(time.perf_counter() - t0)
            pix = out.astype(np.float32)

        t0 = time.perf_counter()
        if t == 0:
            masks, iou, ptrs, osl = hpr.run(None, {
                'pix_feat': pix.astype(dt), 'f0': f0, 'f1': f1,
                'point_coords': pc.astype(dt), 'point_labels': pl.astype(dt),
                'add_no_mem': np.zeros(1, dt)})
            k = 0                                    # >1 point -> single-mask token
        else:
            masks, iou, ptrs, osl = hds.run(None, {
                'pix_feat': pix.astype(dt), 'f0': f0, 'f1': f1,
                'point_coords': np.zeros((1, 1, 2), dt),
                'point_labels': -np.ones((1, 1), dt)})
            k = 1 + int(np.argmax(iou.astype(np.float32)[0, 1:]))
        times['hds'].append(time.perf_counter() - t0)
        low = masks.astype(np.float32)[:, k:k + 1]
        ptr = ptrs.astype(np.float32)[0, k]
        is_obj = (osl.astype(np.float32) > 0).astype(np.float32)

        t0 = time.perf_counter()
        feed = {'pix_feat': f2f, 'low_res_mask': low}
        if 'is_obj' in [i.name for i in mec.get_inputs()]:
            feed['is_obj'] = is_obj      # EdgeTAM has no no_obj_embed_spatial
        lat, lpos = mec.run(None, feed)
        times['mec'].append(time.perf_counter() - t0)
        bank.add(t, lat[0], lpos[0], ptr, is_cond=(t == 0))
        times['all'].append(time.perf_counter() - t_all)

        ref = np.asarray(Image.open(os.path.join(refdir, '%04d.png' % t)), np.uint8)
        got = bilinear_to(low[0, 0], W, H) > 0
        r = ref > 127
        inter = np.logical_and(got, r).sum()
        union = np.logical_or(got, r).sum()
        ious.append(1.0 if union == 0 else inter / union)
        if t % 25 == 0:
            print('  frame %3d iou=%.4f' % (t, ious[-1]), flush=True)

    ious = np.array(ious)
    print('\nIoU vs %s: mean=%.4f min=%.4f  (>=0.9: %d/%d)'
          % (a.ref, ious.mean(), ious.min(), int((ious >= 0.9).sum()), len(ious)))
    for k in ('enc', 'mat', 'hds', 'mec', 'all'):
        if times[k]:
            print('  %-4s %7.1f ms' % (k, 1000 * np.mean(times[k])))
    return ious


if __name__ == '__main__':
    main()
