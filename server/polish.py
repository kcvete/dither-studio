#!/usr/bin/env python3
"""Mask polish, server side — a numpy transcription of web/polish.js.

The algorithm, the reasoning behind the motion gate and the meaning of every
number live in that file's header; this one exists so the render path can do
the same thing to a directory of mask PNGs, and so server/parity.py can prove
the two agree.

Parity is not a coincidence, it is a constraint:

  * every stage rounds to float32 exactly where the JS writes into a
    Float32Array, and computes in float64 exactly where JS arithmetic does
    (which is everywhere: JS has no float32 arithmetic, only float32 storage);
  * the term order inside the blur is the same as the JS term order;
  * round-half-up (floor(v + 0.5)) instead of numpy's round-half-to-even.

    polish_sequence(masks, strength)      list of HxW float32 -> same
    polish_dir(src, dst, strength)        mask PNGs -> polished mask PNGs
    polished_dir(job_dir, oid, strength)  the cached directory for one subject
"""
import math
import os
import shutil

import numpy as np
from PIL import Image

ALPHA = 0.35


def _round0(v):
    """floor(v + 0.5) — JavaScript's Math.round, not numpy's round-half-even."""
    return int(math.floor(v + 0.5))


def params(strength):
    u = min(1.0, max(0.0, float(strength or 0) / 100.0))
    return dict(u=u, radius=_round0(u * 3), morph=_round0(u * 2),
                blur=_round0(u * 2), alpha=ALPHA)


def load_mask(path, w=None, h=None):
    """A soft mask PNG as float32 0..1, quantised the way the browser's
    bitmapAlpha quantises an ImageBitmap: uint8 / 255 in double, then float32."""
    im = Image.open(path).convert('L')
    if w and h and im.size != (w, h):
        im = im.resize((w, h))
    return (np.asarray(im, np.float64) / 255.0).astype(np.float32)


def save_mask(m, path):
    """float32 0..1 -> the 0..255 L PNG the tracker itself writes."""
    a = np.clip(np.floor(m.astype(np.float64) * 255.0 + 0.5), 0, 255).astype(np.uint8)
    Image.fromarray(a, mode='L').save(path)


def stats(m):
    """area (soft >= 0.5) and its centroid."""
    b = m >= 0.5
    n = int(b.sum())
    if not n:
        return (0, 0.0, 0.0)
    ys, xs = np.nonzero(b)
    return (n, float(xs.sum()) / n, float(ys.sum()) / n)


def weights(win, c, radius):
    """Per-frame temporal weights — the motion gate, in the open.

    `win` is [(area, cx, cy), ...] for a contiguous run of frames, `c` the index
    of the one being polished.
    """
    w = np.zeros(len(win), np.float64)
    w[c] = 1.0
    size = math.sqrt(win[c][0])
    if radius <= 0 or size < 1:
        return w
    tol = ALPHA * size
    for direction in (-1, 1):
        disp = 0.0
        for d in range(1, radius + 1):
            j = c + direction * d
            prev = j - direction
            if j < 0 or j >= len(win):
                break
            if win[j][0] <= 0 or win[prev][0] <= 0:
                break
            dx = win[j][1] - win[prev][1]
            dy = win[j][2] - win[prev][2]
            disp += math.sqrt(dx * dx + dy * dy)
            tri = (radius + 1 - d) / (radius + 1)
            gate = min(1.0, max(0.0, 1.0 - disp / tol))
            wj = tri * gate
            if wj <= 0:
                break
            w[j] = wj
    return w


# ------------------------------------------------------------ spatial bits
def _extreme(src, r, take_max):
    """Separable min/max over a (2r+1) square, edges clamped."""
    h, wd = src.shape
    xi = np.clip(np.arange(wd)[None, :] + np.arange(-r, r + 1)[:, None], 0, wd - 1)
    cols = src[:, xi]                                    # (h, 2r+1, w)
    tmp = (cols.max(1) if take_max else cols.min(1)).astype(np.float32)
    yi = np.clip(np.arange(h)[None, :] + np.arange(-r, r + 1)[:, None], 0, h - 1)
    rows = tmp[yi, :]                                    # (2r+1, h, w)
    return (rows.max(0) if take_max else rows.min(0)).astype(np.float32)


def dilate(a, r):
    return _extreme(a, r, True) if r > 0 else a


def erode(a, r):
    return _extreme(a, r, False) if r > 0 else a


def close(a, r):
    return erode(dilate(a, r), r)


def open_(a, r):
    return dilate(erode(a, r), r)


def blur121(src):
    """One separable [1 2 1]/4 pass, edges clamped, terms in the JS order."""
    h, wd = src.shape
    s = src.astype(np.float64)
    left = s[:, np.clip(np.arange(wd) - 1, 0, wd - 1)]
    right = s[:, np.clip(np.arange(wd) + 1, 0, wd - 1)]
    tmp = ((left + 2.0 * s + right) / 4.0).astype(np.float32).astype(np.float64)
    up = tmp[np.clip(np.arange(h) - 1, 0, h - 1), :]
    dn = tmp[np.clip(np.arange(h) + 1, 0, h - 1), :]
    return ((up + 2.0 * tmp + dn) / 4.0).astype(np.float32)


# ----------------------------------------------------------------- polish
def polish_frame(win, c, strength, st=None):
    """One frame. `win` is a contiguous list of float32 masks, `c` the centre."""
    p = params(strength)
    mid = win[c]
    if p['u'] <= 0:
        return mid.astype(np.float32, copy=True)
    cur = None
    if p['radius'] > 0 and len(win) > 1:
        S = st if st is not None else [stats(m) for m in win]
        wt = weights(S, c, p['radius'])
        total = float(wt.sum())
        if total > 0 and any(wt[j] > 0 for j in range(len(wt)) if j != c):
            acc = np.zeros(mid.shape, np.float32)
            for j, wj in enumerate(wt):
                if wj <= 0:
                    continue
                acc = (acc.astype(np.float64) + wj * win[j].astype(np.float64)) \
                    .astype(np.float32)
            cur = (acc.astype(np.float64) / total).astype(np.float32)
    if cur is None:
        cur = mid.astype(np.float32, copy=True)
    if p['morph'] > 0:
        cur = open_(close(cur, p['morph']), p['morph'])
    for _ in range(p['blur']):
        cur = blur121(cur)
    return cur


def polish_sequence(masks, strength):
    p = params(strength)
    if p['u'] <= 0:
        return [m.astype(np.float32, copy=True) for m in masks]
    st = [stats(m) for m in masks]
    out = []
    r = p['radius']
    for i in range(len(masks)):
        lo, hi = max(0, i - r), min(len(masks) - 1, i + r)
        out.append(polish_frame(masks[lo:hi + 1], i - lo, strength, st[lo:hi + 1]))
    return out


# ------------------------------------------------------------- directories
def _frames(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith('.png'))


def polish_dir(src, dst, strength, progress=None):
    """A directory of soft mask PNGs -> a directory of polished ones.

    Two passes and a ring buffer rather than the whole sequence in memory: a
    189-frame 720p subject is 700 MB as float32 and 4 MB as a window of seven.
    """
    files = _frames(src)
    if not files:
        raise RuntimeError('no mask PNGs in ' + src)
    p = params(strength)
    r = p['radius']
    os.makedirs(dst, exist_ok=True)

    st = []
    for f in files:
        st.append(stats(load_mask(os.path.join(src, f))))

    cache = {}

    def get(i):
        m = cache.get(i)
        if m is None:
            m = load_mask(os.path.join(src, files[i]))
            cache[i] = m
        return m

    for i, f in enumerate(files):
        lo, hi = max(0, i - r), min(len(files) - 1, i + r)
        win = [get(k) for k in range(lo, hi + 1)]
        save_mask(polish_frame(win, i - lo, strength, st[lo:hi + 1]),
                  os.path.join(dst, f))
        for k in list(cache):
            if k < lo:
                del cache[k]
        if progress:
            progress(i + 1, len(files))
    return {'frames': len(files), 'strength': int(strength), 'radius': r,
            'morph': p['morph'], 'blur': p['blur'], 'dir': dst}


def polished_dir(job_dir, oid, strength, rebuild=False):
    """The cached polished masks for one subject at one strength.

    jobs/<jid>/masks/<oid>       the tracker's own output, never touched
    jobs/<jid>/polish/<oid>/<s>  this, built once and reused by every render
    """
    src = os.path.join(job_dir, 'masks', str(oid))
    s = int(round(float(strength)))
    if s <= 0:
        return src, None
    dst = os.path.join(job_dir, 'polish', str(oid), str(s))
    done = os.path.join(dst, '.done')
    if os.path.exists(done) and not rebuild:
        return dst, None
    shutil.rmtree(dst, ignore_errors=True)
    info = polish_dir(src, dst, s)
    with open(done, 'w') as f:
        f.write(str(info['frames']))
    return dst, info


if __name__ == '__main__':
    import argparse, json, time
    ap = argparse.ArgumentParser(description='polish a directory of mask PNGs')
    ap.add_argument('--masks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strength', type=float, default=70)
    a = ap.parse_args()
    t0 = time.perf_counter()
    info = polish_dir(a.masks, a.out, a.strength)
    info['elapsed_s'] = round(time.perf_counter() - t0, 2)
    print(json.dumps(info, indent=2))
