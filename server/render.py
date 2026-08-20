#!/usr/bin/env python3
"""Video renderer.

Two families of look, both driven by the shared engine in dither.py:

  dots        the original flicker-free particle renderer — a fixed blue-noise
              threshold field over a cell grid plus fixed per-cell jitter, so
              dots stay put between frames and only switch on and off as tone
              changes. Density follows the subject's tone.
  everything  per-pixel dithering (Bayer / blue noise / halftone / white noise /
  else        error diffusion / Riemersma) through dither.dither_rgb.

Subjects are optional. With no subjects the whole frame is dithered; with
subjects each one is dithered with its own palette and, in `cutout` compose
mode, everything else is flat background.

web/dither.js + web/app.js reproduce all of this in the browser; parity.py
is the gate that keeps the two honest.
"""
import os
import subprocess

import numpy as np
from PIL import Image

import dither as DI

LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)

DEFAULTS = dict(
    # what the pixels become
    mode="dots", algo="floyd-steinberg", matrix=4, serpentine=False, strength=1.0,
    # tone shaping (shared by every mode)
    brightness=0.0, contrast=1.0, gamma=1.0, invert=False,
    # chunky-pixel scale
    pixel=1,
    # composition
    compose="cutout", bg="#c9d4c5",
    # dots-mode knobs
    n=8000, cell=4, dotpx=3, fill=0.7, stray=0.02, band=9,
    # output
    fps=30, seed=7,
)

PALETTES = DI.PALETTES
MODES = DI.MODES
KERNELS = DI.KERNELS

# per-subject default dot colours, used when the client does not send a palette
SUBJECT_COLORS = ["#b0413e", "#2f4f4a", "#7a6a4f", "#3c5a7a", "#8a5a8a", "#4a7a4a"]


# ------------------------------------------------------------- blue noise
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
    """Void-and-cluster-ish: iterated high-pass + histogram remap -> blue noise."""
    v = np.random.default_rng(seed).random((n, n))
    for _ in range(iters):
        v = _rank_uniform(v - _lowpass(v, sigma))
    return v


# ------------------------------------------------------------ dots helpers
hash01 = DI.hash01
hexcol = DI.hexcol


def cell_fields(gh, gw, seed):
    ii = np.broadcast_to(np.arange(gh, dtype=np.int64)[:, None], (gh, gw))
    jj = np.broadcast_to(np.arange(gw, dtype=np.int64)[None, :], (gh, gw))
    return (hash01(ii, jj, 1, seed).astype(np.float64),
            hash01(ii, jj, 2, seed).astype(np.float64),
            hash01(ii, jj, 3, seed).astype(np.float64))


def block_mean(a, cell):
    h, w = a.shape
    gh, gw = h // cell, w // cell
    return a[:gh * cell, :gw * cell].reshape(gh, cell, gw, cell).mean((1, 3))


def _shift(a, d, axis):
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
    """Cross-shaped max dilation, radius r. Edges clamp instead of wrapping."""
    out = a
    for d in range(1, r + 1):
        out = np.maximum.reduce([out, _shift(a, d, 0), _shift(a, -d, 0),
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


# --------------------------------------------------------- pixel scaling
def downscale(a, p):
    """Box-mean downscale by integer factor p, rounded half-up (== JS Math.round)."""
    if p <= 1:
        return a
    h, w = a.shape[:2]
    h2, w2 = h // p, w // p
    if h2 < 1 or w2 < 1:
        return a
    trimmed = a[:h2 * p, :w2 * p].astype(np.float64)
    if a.ndim == 3:
        m = trimmed.reshape(h2, p, w2, p, a.shape[2]).mean((1, 3))
    else:
        m = trimmed.reshape(h2, p, w2, p).mean((1, 3))
    if np.issubdtype(a.dtype, np.integer):
        return np.floor(m + 0.5).clip(0, 255).astype(a.dtype)
    return m.astype(a.dtype)


def upscale(a, p, h, w):
    """Nearest-neighbour upscale back to h x w (clamping the trailing edge)."""
    if p <= 1:
        return a
    yi = np.clip(np.arange(h) // p, 0, a.shape[0] - 1)
    xi = np.clip(np.arange(w) // p, 0, a.shape[1] - 1)
    return a[yi][:, xi]


# --------------------------------------------------------------- I/O bits
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


def _params(overrides):
    a = dict(DEFAULTS)
    a.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return a


def _subject_palette(s, i, bg):
    pal = s.get('palette')
    if pal:
        return list(pal)
    return [bg, s.get('dot') or SUBJECT_COLORS[i % len(SUBJECT_COLORS)]]


# ------------------------------------------------------------ dots renderer
def _frame_dots(rgb, masks, a, F, dots_cols, bg):
    H, W = rgb.shape[:2]
    cell = int(a['cell'])
    thr, cy, cx, stray_r, gh, gw = F
    lum = rgb.astype(np.float32) / 255.0 @ LUM
    tone = lum if a['invert'] else 1.0 - lum
    if float(a['gamma']) != 1.0:
        tone = np.clip(tone, 0, 1) ** float(a['gamma'])

    K = len(masks)
    wgts = [block_mean(m * tone, cell) for m in masks]
    mgs = [block_mean(m, cell) for m in masks]
    stack = np.stack(mgs)
    owner = stack.argmax(0)
    covered = stack.max(0) > 0
    any_mg = stack.max(0)

    if a['compose'] == 'overlay':
        g = (lum * 0.55 + 0.22)[:, :, None]
        canvas = np.clip(g * np.array(hexcol(bg), np.float32)[None, None, :] * 1.15,
                         0, 255).astype(np.uint8)
    else:
        canvas = np.broadcast_to(np.array(hexcol(bg), np.uint8), (H, W, 3)).copy()

    dp = int(a['dotpx'])
    off = np.arange(dp) - dp // 2
    oy, ox = np.meshgrid(off, off, indexing='ij')
    oy, ox = oy.ravel()[None, :], ox.ravel()[None, :]

    for k in range(K):
        mine = covered & (owner == k)
        wgt = np.where(mine, wgts[k], 0.0)
        if a['n']:
            tgt = min(int(a['n']), max(1, int(float(a['fill']) * np.count_nonzero(wgt > 0))))
            wgt = np.minimum(wgt * gain_for_count(wgt, thr, tgt), 1.0)
        on = wgt > thr
        if float(a['stray']) > 0 and int(a['band']) > 0:
            band = (dilate(mgs[k], int(a['band'])) > .15) & (any_mg <= .15)
            nb = int(band.sum())
            if nb:
                on |= band & (stray_r < min(1.0, float(a['stray']) * max(on.sum(), 1) / nb))
        if not on.any():
            continue
        yy = np.clip(np.rint(cy[on])[:, None] + oy, 0, H - 1).astype(np.int32)
        xx = np.clip(np.rint(cx[on])[:, None] + ox, 0, W - 1).astype(np.int32)
        canvas[yy.ravel(), xx.ravel()] = np.array(hexcol(dots_cols[k]), np.uint8)
    return canvas


# ------------------------------------------------------- per-pixel renderer
def _frame_pixels(rgb, masks, a, blue, palettes, bg):
    """masks may be empty -> the whole frame is dithered with palettes[0]."""
    H, W = rgb.shape[:2]
    P = max(1, int(a['pixel']))
    base = dict(a)
    base['_blue'] = blue

    small = downscale(rgb, P)
    h2, w2 = small.shape[:2]

    if not masks:
        p = dict(base); p['palette'] = palettes[0]
        out = DI.dither_rgb(small, p)
        return upscale(out, P, H, W)

    ms = [downscale(m, P) for m in masks]
    stack = np.stack(ms)
    owner = stack.argmax(0)
    inside = stack.max(0) >= 0.5

    if a['compose'] == 'overlay':
        p = dict(base); p['palette'] = palettes[0]
        out = DI.dither_rgb(small, p)
    else:
        out = np.broadcast_to(np.array(hexcol(bg), np.uint8), (h2, w2, 3)).copy()

    for k in range(len(masks)):
        gate = (inside & (owner == k)).astype(np.float32)
        if not gate.any():
            continue
        p = dict(base)
        p['palette'] = palettes[k + 1] if len(palettes) > k + 1 else palettes[0]
        d = DI.dither_rgb(small, p, gate)
        g3 = gate.astype(bool)[:, :, None]
        out = np.where(g3, d, out)
    return upscale(out, P, H, W)


# --------------------------------------------------------------- render
def render(frames_dir, subjects, out_path, params=None, progress=None):
    """frames_dir -- directory of %04d.jpg
       subjects   -- [] for whole-frame, else [{'masks': dir, 'dot'|'palette': ...}]
       params     -- overrides for DEFAULTS
       progress   -- callable(done, total)
    """
    a = _params(params)
    files = _list_frames(frames_dir)
    if not files:
        raise RuntimeError('no frames in ' + frames_dir)
    if a['mode'] == 'dots' and not subjects:
        raise RuntimeError('the dots look needs at least one tracked subject')

    H, W = np.asarray(Image.open(os.path.join(frames_dir, files[0])).convert('RGB')).shape[:2]
    seed = int(a['seed'])
    bg = a['bg']
    blue = blue_noise(64, seed).astype(np.float32)

    palettes = [list(a.get('palette') or PALETTES[0]['colors'])]
    for i, s in enumerate(subjects):
        palettes.append(_subject_palette(s, i, bg))
    dots_cols = [palettes[i + 1][-1] for i in range(len(subjects))]

    F = None
    if a['mode'] == 'dots':
        cell = int(a['cell'])
        gh, gw = H // cell, W // cell
        thr = np.tile(blue, (gh // 64 + 1, gw // 64 + 1))[:gh, :gw]
        jx, jy, stray_r = cell_fields(gh, gw, seed)
        cy = np.broadcast_to(np.arange(gh)[:, None] * cell + cell / 2.0
                             + (jy - .5) * cell * .8, (gh, gw))
        cx = np.broadcast_to(np.arange(gw)[None, :] * cell + cell / 2.0
                             + (jx - .5) * cell * .8, (gh, gw))
        F = (thr, cy, cx, stray_r, gh, gw)

    p = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', '%dx%d' % (W, H), '-r', str(int(a['fps'])), '-i', '-',
         '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
         '-movflags', '+faststart', out_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    last = [None] * len(subjects)
    total = len(files)
    try:
        for i, fn in enumerate(files):
            rgb = np.asarray(Image.open(os.path.join(frames_dir, fn)).convert('RGB'))
            masks = []
            for k, s in enumerate(subjects):
                last[k] = _load_mask(s['masks'], i, H, W, last[k])
                masks.append(last[k])
            if a['mode'] == 'dots':
                canvas = _frame_dots(rgb, masks, a, F, dots_cols, bg)
            else:
                canvas = _frame_pixels(rgb, masks, a, blue, palettes, bg)
            p.stdin.write(np.ascontiguousarray(canvas, np.uint8).tobytes())
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
    ap = argparse.ArgumentParser(description='render a dithered mp4 from frames (+ optional masks)')
    ap.add_argument('--frames', required=True)
    ap.add_argument('--masks', action='append', default=[],
                    help='maskdir[:#rrggbb], repeatable; omit for whole-frame')
    ap.add_argument('--out', required=True)
    ap.add_argument('--palette', default=None, help='comma-separated hex colours')
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ap.add_argument('--' + k, action='store_true')
        else:
            ap.add_argument('--' + k, type=type(v), default=v)
    args = ap.parse_args()
    subs = []
    for idx, spec in enumerate(args.masks):
        d, _, c = spec.partition(':')
        subs.append({'masks': d, 'dot': c or SUBJECT_COLORS[idx % len(SUBJECT_COLORS)]})
    prm = {k: getattr(args, k) for k in DEFAULTS}
    if args.palette:
        prm['palette'] = args.palette.split(',')
    print(json.dumps(render(args.frames, subs, args.out, prm), indent=2))
