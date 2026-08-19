#!/usr/bin/env python3
"""Flicker-free blue-noise threshold dither renderer, multi-subject edition.

Turns per-frame soft masks + frames into a temporally stable particle animation:
flat background, small solid square dots, density follows tone. Each tracked
subject gets its own dot colour.

Temporal stability: NO error diffusion. One fixed blue-noise threshold field is
generated once over the cell grid and reused for every frame, plus per-cell fixed
jitter, so dots stay put and only switch on/off as the underlying tone changes.

The jitter / stray fields use a portable integer hash (not numpy's RNG) so the
browser preview in static/app.js can reproduce the exact same layout.
"""
import os
import subprocess
import numpy as np
from PIL import Image

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)

DEFAULTS = dict(
    mode="cutout", bg="#c9d4c5", n=8000, cell=4, dotpx=3, invert=False,
    gamma=1.0, fill=0.7, stray=0.02, band=9, fps=30, seed=7,
)

PALETTES = [
    {"name": "sage",   "bg": "#c9d4c5", "dots": ["#b0413e", "#2f4f4a", "#7a6a4f", "#3c5a7a", "#8a5a8a", "#4a7a4a"]},
    {"name": "forest", "bg": "#0f1f18", "dots": ["#d7e3d5", "#e8a04a", "#8fc7ff", "#f28b82", "#c5b0ff", "#9fe0a8"]},
    {"name": "ember",  "bg": "#e8804a", "dots": ["#f6ece2", "#2b1a12", "#ffd9a0", "#6b2d1a", "#fff1d6", "#9c3b1e"]},
    {"name": "mist",   "bg": "#aebfab", "dots": ["#ffffff", "#28352c", "#e2b8a0", "#5c7a6a", "#f0e6d2", "#3d4f5c"]},
]


# ---------------------------------------------------------------- blue noise
def _lowpass(v, sigma):
    n = v.shape[0]
    f = np.fft.fftfreq(n)
    k = f[:, None] ** 2 + f[None, :] ** 2
    g = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * k)
    return np.real(np.fft.ifft2(np.fft.fft2(v) * g))


def _rank_uniform(a):
    flat = a.ravel()
    order = np.argsort(flat, kind='stable')
    r = np.empty(flat.size, np.int64)
    r[order] = np.arange(flat.size)
    return ((r + 0.5) / flat.size).reshape(a.shape)


def blue_noise(n=64, seed=7, iters=40, sigma=1.6):
    """Void-and-cluster-ish: iterated high-pass + histogram remap -> blue noise tile."""
    v = np.random.default_rng(seed).random((n, n))
    for _ in range(iters):
        v = _rank_uniform(v - _lowpass(v, sigma))
    return v


# ------------------------------------------------- portable per-cell hash rng
_M = 0xFFFFFFFF


def _u32(x):
    return np.asarray(x, dtype=np.int64) & _M


def hash01(i, j, salt, seed):
    """Deterministic uniform [0,1) per (i, j, salt). Mirrored in static/app.js."""
    x = _u32(_u32(i * 73856093) ^ _u32(j * 19349663) ^
             _u32(salt * 83492791) ^ _u32(seed * 2654435761))
    x = _u32(x ^ (x >> 16))
    x = _u32(x * 0x7feb352d)
    x = _u32(x ^ (x >> 15))
    x = _u32(x * 0x846ca68b)
    x = _u32(x ^ (x >> 16))
    return x.astype(np.float64) / 4294967296.0


def cell_fields(gh, gw, seed):
    """jx, jy, stray_r for a gh x gw cell grid."""
    ii = np.arange(gh, dtype=np.int64)[:, None]
    jj = np.arange(gw, dtype=np.int64)[None, :]
    ii = np.broadcast_to(ii, (gh, gw))
    jj = np.broadcast_to(jj, (gh, gw))
    return (hash01(ii, jj, 1, seed), hash01(ii, jj, 2, seed), hash01(ii, jj, 3, seed))


# ------------------------------------------------------------------- helpers
def hexcol(s):
    s = s.lstrip('#')
    if len(s) == 3:
        s = ''.join(c * 2 for c in s)
    return np.array([int(s[i:i + 2], 16) for i in (0, 2, 4)], np.uint8)


def block_mean(a, cell):
    h, w = a.shape
    gh, gw = h // cell, w // cell
    return a[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean((1, 3))


def _shift(a, d, axis):
    """Shift by d along axis, filling with zeros (no wrap-around)."""
    out = np.zeros_like(a)
    if axis == 0:
        if d > 0:
            out[d:, :] = a[:-d, :]
        else:
            out[:d, :] = a[-d:, :]
    else:
        if d > 0:
            out[:, d:] = a[:, :-d]
        else:
            out[:, :d] = a[:, -d:]
    return out


def dilate(a, r):
    """Cross-shaped max dilation of radius r. Edges clamp instead of wrapping --
    the original np.roll version leaked a subject's halo to the opposite edge."""
    out = a
    for d in range(1, r + 1):
        out = np.maximum.reduce([out,
                                 _shift(a, d, 0), _shift(a, -d, 0),
                                 _shift(a, d, 1), _shift(a, -d, 1)])
    return out


def gain_for_count(w, thr, target):
    lo, hi = 1e-3, 1e3
    for _ in range(24):
        mid = (lo * hi) ** 0.5
        if np.count_nonzero(np.minimum(w * mid, 1.0) > thr) < target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def _list_frames(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.png')))


def _load_mask(mdir, i, h, w, last):
    p = os.path.join(mdir, '%04d.png' % i)
    if os.path.exists(p):
        im = Image.open(p).convert('L')
        if im.size != (w, h):
            im = im.resize((w, h))
        return np.asarray(im, np.float32) / 255.0
    return last if last is not None else np.zeros((h, w), np.float32)


# -------------------------------------------------------------------- render
def render(frames_dir, subjects, out_path, params=None, progress=None):
    """frames_dir  -- directory of %04d.jpg
       subjects    -- [{'masks': <dir of %04d.png>, 'dot': '#rrggbb'}, ...]
       out_path    -- .mp4
       params      -- overrides for DEFAULTS
       progress    -- callable(done, total)
    """
    a = dict(DEFAULTS)
    a.update({k: v for k, v in (params or {}).items() if v is not None})
    files = _list_frames(frames_dir)
    if not files:
        raise RuntimeError('no frames in ' + frames_dir)
    if not subjects:
        raise RuntimeError('no subjects to render')

    H, W = np.asarray(Image.open(os.path.join(frames_dir, files[0])).convert('RGB')).shape[:2]
    cell, dp = int(a['cell']), int(a['dotpx'])
    gh, gw = H // cell, W // cell
    seed = int(a['seed'])

    tile = blue_noise(64, seed)
    thr = np.tile(tile, (gh // 64 + 1, gw // 64 + 1))[:gh, :gw]
    jx, jy, stray_r = cell_fields(gh, gw, seed)
    jx = (jx - .5) * cell * .8
    jy = (jy - .5) * cell * .8
    cy = np.broadcast_to(np.arange(gh)[:, None] * cell + cell / 2.0 + jy, (gh, gw))
    cx = np.broadcast_to(np.arange(gw)[None, :] * cell + cell / 2.0 + jx, (gh, gw))
    off = np.arange(dp) - dp // 2
    oy, ox = np.meshgrid(off, off, indexing='ij')
    oy, ox = oy.ravel()[None, :], ox.ravel()[None, :]

    bg = hexcol(a['bg'])
    dots = [hexcol(s.get('dot', '#b0413e')) for s in subjects]
    nsub = len(subjects)

    p = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', '%dx%d' % (W, H), '-r', str(int(a['fps'])), '-i', '-',
         '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
         '-movflags', '+faststart', out_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    last = [None] * nsub
    total = len(files)
    try:
        for i, fn in enumerate(files):
            rgb = np.asarray(Image.open(os.path.join(frames_dir, fn)).convert('RGB'), np.float32) / 255.0
            lum = rgb @ LUM
            tone = lum if a['invert'] else 1.0 - lum
            if float(a['gamma']) != 1.0:
                tone = np.clip(tone, 0, 1) ** float(a['gamma'])

            wgts, mgs = [], []
            for k in range(nsub):
                m = last[k] = _load_mask(subjects[k]['masks'], i, H, W, last[k])
                wgts.append(block_mean(m * tone, cell))
                mgs.append(block_mean(m, cell))

            # one cell belongs to exactly one subject (the one covering it most),
            # so overlapping masks never double-paint or fight over colour
            stack = np.stack(mgs)
            owner = stack.argmax(0)
            covered = stack.max(0) > 0
            any_mg = stack.max(0)

            if a['mode'] == 'overlay':
                g = (lum * 0.55 + 0.22)[:, :, None]
                canvas = np.clip(g * bg[None, None, :].astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
            else:
                canvas = np.broadcast_to(bg, (H, W, 3)).copy()

            for k in range(nsub):
                mine = covered & (owner == k)
                wgt = np.where(mine, wgts[k], 0.0)
                if a['n']:
                    # never light more than `fill` of the cells the subject covers,
                    # else the dithered texture collapses into a solid silhouette
                    tgt = min(int(a['n']), max(1, int(float(a['fill']) * np.count_nonzero(wgt > 0))))
                    wgt = np.minimum(wgt * gain_for_count(wgt, thr, tgt), 1.0)
                on = wgt > thr
                if float(a['stray']) > 0:
                    band = (dilate(mgs[k], int(a['band'])) > .15) & (any_mg <= .15)
                    nb = int(band.sum())
                    if nb:
                        on |= band & (stray_r < min(1.0, float(a['stray']) * max(on.sum(), 1) / nb))
                if not on.any():
                    continue
                yy = np.clip(np.rint(cy[on])[:, None] + oy, 0, H - 1).astype(np.int32)
                xx = np.clip(np.rint(cx[on])[:, None] + ox, 0, W - 1).astype(np.int32)
                canvas[yy.ravel(), xx.ravel()] = dots[k]

            p.stdin.write(canvas.tobytes())
            if progress:
                progress(i + 1, total)
    finally:
        try:
            p.stdin.close()
        except Exception:
            pass
    err = p.stderr.read().decode('utf-8', 'replace')
    if p.wait() != 0:
        raise RuntimeError('ffmpeg failed: ' + err[-800:])
    return {'out': out_path, 'frames': total, 'w': W, 'h': H}


if __name__ == '__main__':
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--masks', action='append', required=True,
                    help='dir[:#rrggbb], repeatable')
    ap.add_argument('--out', required=True)
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument('--' + k, action='store_true')
        else:
            ap.add_argument('--' + k, type=type(v), default=v)
    args = ap.parse_args()
    subs = []
    pal = PALETTES[0]['dots']
    for idx, spec in enumerate(args.masks):
        d, _, c = spec.partition(':')
        subs.append({'masks': d, 'dot': c or pal[idx % len(pal)]})
    prm = {k: getattr(args, k) for k in DEFAULTS}
    print(json.dumps(render(args.frames, subs, args.out, prm), indent=2))
