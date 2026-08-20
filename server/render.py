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
import math
import os
import subprocess
import threading

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
    # the frame window this render covers, as INDICES into the job's frames
    # directory (which is also how the mask files are numbered). Both are
    # inclusive; frame_out=None means "to the last frame there is". The
    # defaults are the whole clip, so a caller that never heard of a range
    # gets exactly what it always got.
    frame_in=0, frame_out=None,
    # container: see FORMATS. 'format' picks the encoder, gif_fps only bites for GIF.
    format="mp4", gif_fps=None,
    # THE CANVAS. None -- the default -- means the output is the frames' own
    # size, which is what every caller before this one got. Otherwise a dict:
    #
    #   {"w": 1080, "h": 1920, "k": 2.667, "place": [[x0, y0], ...]}
    #
    # which is one affine map per frame of the window (or a single entry, used
    # for all of them, when the frame holds still):
    #
    #   X = x * k + x0        x, y  in the source frame's pixels
    #   Y = y * k + y0        X, Y  in the canvas's
    #
    # The CLIENT works those numbers out -- web/canvas.js -- because it is the
    # one that can see the tracked subject and the user's dragging, and it
    # hands over only where the crop ENDED UP. Nothing about following,
    # smoothing or clamping is implemented here, so there is no second opinion
    # to drift from the preview's. The matched original cut and the .dots
    # export take the same block, which is what keeps the three files framed
    # identically.
    canvas=None,
)


# --------------------------------------------------------------- formats
# One rendered frame sequence, five containers. `alpha` decides whether the
# frames are handed to ffmpeg as rgb24 or rgba -- and, upstream of that,
# whether the flat background is painted or left transparent.
#
# GIF is here because the looks this tool produces are 2-4 flat colours, which
# is exactly what a 256-entry global palette is good at; a 150-frame 720p dots
# render is smaller as a GIF than as an MP4 (see the README's formats table).
FORMATS = {
    "mp4":        {"ext": "mp4",  "file": "out.mp4",         "mime": "video/mp4",
                   "alpha": False, "label": "MP4 · H.264"},
    "webm":       {"ext": "webm", "file": "out.webm",        "mime": "video/webm",
                   "alpha": False, "label": "WebM · VP9"},
    "gif":        {"ext": "gif",  "file": "out.gif",         "mime": "image/gif",
                   "alpha": False, "label": "GIF · looping"},
    "webm-alpha": {"ext": "webm", "file": "out.alpha.webm",  "mime": "video/webm",
                   "alpha": True,  "label": "WebM · VP9 + alpha"},
    "prores":     {"ext": "mov",  "file": "out.mov",         "mime": "video/quicktime",
                   "alpha": True,  "label": "ProRes 4444 · alpha"},
}


def ffmpeg_cmd(fmt, W, H, fps, out_path, gif_fps=None, input_args=None):
    """The encoder invocation for one format, reading raw frames on stdin.

    `input_args` replaces the stdin input for a caller that has a better one --
    the original cut hands ffmpeg the JPEGs themselves rather than decoding
    them in Python only to hand them straight back. Everything downstream of
    the input, which is the part that decides what the file IS, is the same.
    """
    f = FORMATS[fmt]
    pix = "rgba" if f["alpha"] else "rgb24"
    cmd = ["ffmpeg", "-v", "error", "-y"] + (list(input_args) if input_args else
          ["-f", "rawvideo", "-pix_fmt", pix,
           "-s", "%dx%d" % (W, H), "-r", str(int(fps)), "-i", "-"])
    if fmt == "mp4":
        cmd += ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart"]
    elif fmt == "webm":
        cmd += ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0",
                "-pix_fmt", "yuv420p", "-row-mt", "1"]
    elif fmt == "webm-alpha":
        # yuva420p is what makes the alpha survive; -auto-alt-ref 0 is required
        # by libvpx-vp9 when the alpha plane is present.
        cmd += ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0",
                "-pix_fmt", "yuva420p", "-row-mt", "1", "-auto-alt-ref", "0"]
    elif fmt == "prores":
        cmd += ["-c:v", "prores_ks", "-profile:v", "4444",
                "-pix_fmt", "yuva444p10le", "-alpha_bits", "16", "-vendor", "apl0"]
    elif fmt == "gif":
        # One pass: split the piped stream, build a palette from the whole clip
        # (stats_mode=full), then map through it. dither=none because the frames
        # are already flat colours -- an extra dither would fuzz the dots.
        g = int(gif_fps or min(int(fps), 30))
        cmd += ["-filter_complex",
                "fps=%d,split[a][b];[a]palettegen=stats_mode=full:max_colors=256[p];"
                "[b][p]paletteuse=dither=none" % g,
                "-loop", "0"]
    else:
        raise RuntimeError("unknown format %r" % fmt)
    return cmd + [out_path]

# ---------------------------------------------------------------- canvas
def canvas_plan(a, n_frames=1):
    """Validate a request's canvas block, or None.

    Everything that can be wrong with it is wrong here rather than three
    hundred frames into an encode: a size that is not a positive even pair (no
    H.264 encoder takes an odd dimension), a scale that is not finite, an empty
    or wrongly-sized place list.
    """
    c = a.get("canvas")
    if not c:
        return None
    try:
        tw, th = int(c["w"]), int(c["h"])
        k = float(c["k"])
        place = [(float(p[0]), float(p[1])) for p in (c.get("place") or [])]
    except (KeyError, TypeError, ValueError, IndexError) as e:
        raise RuntimeError("bad canvas block: %s" % e)
    if tw < 2 or th < 2 or tw % 2 or th % 2:
        raise RuntimeError("canvas must be a positive even size, got %dx%d" % (tw, th))
    if not (k > 0) or not math.isfinite(k):
        raise RuntimeError("canvas scale must be positive and finite, got %r" % k)
    if not place:
        raise RuntimeError("canvas has no placement")
    if len(place) != 1 and len(place) != n_frames:
        raise RuntimeError("canvas has %d placements for %d frames"
                           % (len(place), n_frames))
    return {"tw": tw, "th": th, "k": k, "place": place}


def _at(plan, i):
    p = plan["place"]
    return p[0] if len(p) == 1 else p[min(i, len(p) - 1)]


def _affine(plan, i):
    """PIL's AFFINE coefficients for the INVERSE map (canvas -> source), which
    is the direction a resampler reads in."""
    k = plan["k"]
    x0, y0 = _at(plan, i)
    return (1.0 / k, 0.0, -x0 / k, 0.0, 1.0 / k, -y0 / k)


def warp_rgb(rgb, plan, i):
    """One frame, mapped onto the canvas. Bilinear, black outside the source --
    which only shows where a crop was deliberately taken wider than the frame;
    a `cutout` paints its flat background over it anyway."""
    im = Image.fromarray(np.ascontiguousarray(rgb, np.uint8), "RGB")
    out = im.transform((plan["tw"], plan["th"]), Image.AFFINE, _affine(plan, i),
                       resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return np.asarray(out, np.uint8)


def warp_mask(m, plan, i):
    """The same map over a float coverage mask. 'F' mode, so the soft edge
    survives the resample instead of being rounded to 8 bits twice."""
    im = Image.fromarray(np.ascontiguousarray(m, np.float32), "F")
    out = im.transform((plan["tw"], plan["th"]), Image.AFFINE, _affine(plan, i),
                       resample=Image.BILINEAR, fillcolor=0.0)
    return np.asarray(out, np.float32)


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


def frame_range(files, a):
    """(the files this render covers, the index of the first one).

    A trim after the tracking is a WINDOW on frames that already exist, not a
    new extraction: jobs/<id>/frames/ and jobs/<id>/masks/<obj>/ are both
    numbered from 0 and stay exactly as they were. So everything a narrowed
    range needs is a slice plus the offset the masks have to be read at --
    frame k of the output is frames[base + k] and masks/<obj>/(base + k).png.
    """
    n = len(files)
    i0 = max(0, int(a.get("frame_in") or 0))
    i1 = a.get("frame_out")
    i1 = n - 1 if i1 is None else min(n - 1, int(i1))
    if i0 > i1:
        raise RuntimeError("empty frame range: %d..%s of %d frames"
                           % (i0, i1, n))
    return files[i0:i1 + 1], i0


def _load_mask(mdir, i, h, w, last):
    # 4 digits for every clip that fits in 9,999 frames, 6 past that -- see
    # server.pad_for. Reads accept either, so an old job still renders.
    p = os.path.join(mdir, '%04d.png' % i)
    if not os.path.exists(p):
        p = os.path.join(mdir, '%06d.png' % i)
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
def dots_fields(H, W, a, blue):
    """The fixed per-cell fields the dots look is built on: the blue-noise
    threshold, the jittered cell centres and the stray-dot lottery. They depend
    only on the frame size, the cell size and the seed -- which is exactly why
    dots hold still between frames."""
    cell = int(a['cell'])
    seed = int(a['seed'])
    gh, gw = H // cell, W // cell
    thr = np.tile(blue, (gh // 64 + 1, gw // 64 + 1))[:gh, :gw]
    jx, jy, stray_r = cell_fields(gh, gw, seed)
    # the jitter that keeps a dot cloud from looking like graph paper. At cell 1
    # there is no room for it and no need: one cell is one pixel, and jittering
    # it by a pixel would turn a Bayer screen into noise. web/app.js dotFields
    # has the same line.
    jit = .8 if cell > 1 else 0.0
    cy = np.broadcast_to(np.arange(gh)[:, None] * cell + cell / 2.0
                         + (jy - .5) * cell * jit, (gh, gw))
    cx = np.broadcast_to(np.arange(gw)[None, :] * cell + cell / 2.0
                         + (jx - .5) * cell * jit, (gh, gw))
    return (thr, cy, cx, stray_r, gh, gw)


def dots_on(rgb, masks, a, F):
    """Which cells are lit, per subject, on one frame.

    The single source of truth for both the painted frame and the .dots export
    -- the dot data has to be the dots you can see, not a second opinion."""
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

    out = []
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
        out.append(on)
    return out


def dot_positions(on, F):
    """Lit cells -> the integer (x, y) the renderer actually draws at, in
    cell-scan order (row-major), which is what makes the delta coding in
    dots.py small.

    floor(v + 0.5), not np.rint: numpy rounds halves to even and JavaScript's
    Math.round rounds them up, and a cell centre lands exactly on .5 often
    enough to matter -- about one to three dots a frame moved by a pixel
    between the two engines, and a .dots.gz written here then failed to replay
    byte-identically against the browser's own render. This is the JS rule.
    """
    thr, cy, cx, stray_r, gh, gw = F
    ys = np.floor(cy[on] + 0.5).astype(np.int32)
    xs = np.floor(cx[on] + 0.5).astype(np.int32)
    return np.stack([xs, ys], 1)


def _frame_dots(rgb, masks, a, F, dots_cols, bg, alpha=False):
    """One dots frame. With `alpha`, the flat background is left transparent
    and only the dots themselves are opaque -- that is the whole of what the
    ProRes 4444 / WebM-alpha exports need. `overlay` compose has no background
    to key out, so it stays fully opaque."""
    H, W = rgb.shape[:2]
    lum = rgb.astype(np.float32) / 255.0 @ LUM
    ons = dots_on(rgb, masks, a, F)

    if a['compose'] == 'overlay':
        g = (lum * 0.55 + 0.22)[:, :, None]
        canvas = np.clip(g * np.array(hexcol(bg), np.float32)[None, None, :] * 1.15,
                         0, 255).astype(np.uint8)
    else:
        canvas = np.broadcast_to(np.array(hexcol(bg), np.uint8), (H, W, 3)).copy()
    if alpha:
        a0 = 255 if a['compose'] == 'overlay' else 0
        canvas = np.concatenate(
            [canvas, np.full((H, W, 1), a0, np.uint8)], axis=2)

    dp = int(a['dotpx'])
    off = np.arange(dp) - dp // 2
    oy, ox = np.meshgrid(off, off, indexing='ij')
    oy, ox = oy.ravel()[None, :], ox.ravel()[None, :]

    for k, on in enumerate(ons):
        if not on.any():
            continue
        pos = dot_positions(on, F)
        yy = np.clip(pos[:, 1][:, None] + oy, 0, H - 1).astype(np.int32)
        xx = np.clip(pos[:, 0][:, None] + ox, 0, W - 1).astype(np.int32)
        canvas[yy.ravel(), xx.ravel()] = np.array(
            hexcol(dots_cols[k]) + ([255] if alpha else []), np.uint8)
    return canvas


# ------------------------------------------------------- per-pixel renderer
def _frame_pixels(rgb, masks, a, blue, palettes, bg, alpha=False):
    """masks may be empty -> the whole frame is dithered with palettes[0].

    With `alpha`, everything that is flat background becomes transparent. That
    only means something in `cutout` compose with subjects; a whole-frame dither
    has no background to remove, so it comes back fully opaque."""
    H, W = rgb.shape[:2]
    P = max(1, int(a['pixel']))
    base = dict(a)
    base['_blue'] = blue

    small = downscale(rgb, P)
    h2, w2 = small.shape[:2]

    if not masks:
        p = dict(base); p['palette'] = palettes[0]
        out = DI.dither_rgb(small, p)
        if alpha:
            out = np.concatenate(
                [out, np.full(out.shape[:2] + (1,), 255, np.uint8)], axis=2)
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
    if alpha:
        if a['compose'] == 'overlay':
            av = np.full((h2, w2, 1), 255, np.uint8)
        else:
            av = (inside[:, :, None] * 255).astype(np.uint8)
        out = np.concatenate([out, av], axis=2)
    return upscale(out, P, H, W)


# --------------------------------------------------------------- render
def render(frames_dir, subjects, out_path, params=None, progress=None):
    """frames_dir -- directory of numbered .jpg frames
       subjects   -- [] for whole-frame, else [{'masks': dir, 'dot'|'palette': ...}]
       params     -- overrides for DEFAULTS
       progress   -- callable(done, total)
    """
    a = _params(params)
    files = _list_frames(frames_dir)
    if not files:
        raise RuntimeError('no frames in ' + frames_dir)
    files, base = frame_range(files, a)
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

    # the canvas: the OUTPUT size, which is the frames' own unless one is asked
    # for. The dot grid is built on the output, not the source, which is what
    # makes a 1080x1920 cutout real dots at 1080x1920 rather than an upscale.
    plan = canvas_plan(a, len(files))
    OW, OH = (plan['tw'], plan['th']) if plan else (W, H)

    F = dots_fields(OH, OW, a, blue) if a['mode'] == 'dots' else None

    fmt = str(a.get('format') or 'mp4')
    if fmt not in FORMATS:
        raise RuntimeError('unknown format %r (have %s)'
                           % (fmt, ', '.join(FORMATS)))
    alpha = FORMATS[fmt]['alpha']
    p = subprocess.Popen(
        ffmpeg_cmd(fmt, OW, OH, a['fps'], out_path, a.get('gif_fps')),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    last = [None] * len(subjects)
    total = len(files)
    try:
        for i, fn in enumerate(files):
            rgb = np.asarray(Image.open(os.path.join(frames_dir, fn)).convert('RGB'))
            masks = []
            for k, s in enumerate(subjects):
                last[k] = _load_mask(s['masks'], base + i, H, W, last[k])
                masks.append(last[k])
            if plan:
                rgb = warp_rgb(rgb, plan, i)
                masks = [warp_mask(m, plan, i) for m in masks]
            if a['mode'] == 'dots':
                canvas = _frame_dots(rgb, masks, a, F, dots_cols, bg, alpha)
            else:
                canvas = _frame_pixels(rgb, masks, a, blue, palettes, bg, alpha)
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
    return {'out': out_path, 'frames': total, 'w': OW, 'h': OH, 'format': fmt,
            'src_w': W, 'src_h': H, 'canvas': bool(plan),
            'first_frame': base, 'last_frame': base + total - 1,
            'bytes': os.path.getsize(out_path)}


# ------------------------------------------------------- the original cut
# The dithered render and the untouched clip, frame for frame.
#
# Both read the SAME jobs/<id>/frames/*.jpg -- the trimmed, fps-normalised,
# 720p-or-native ground truth that the tracker and the dither already agreed
# on -- through the same _list_frames() ordering and the same encoder call. So
# the pair lands in an edit with identical frame counts, rate and size, and
# frame N of one IS frame N of the other. Nothing here re-reads the source
# file, because the source file is not what was rendered.
#
# Not every container makes sense for a copy of the clip: pairing a GIF with a
# GIF is pointless (and the GIF is decimated to gif_fps anyway), and an alpha
# format has nothing to key out of footage that was never dithered. Those fall
# back to H.264.
ORIGINAL_FORMAT = {"mp4": "mp4", "webm": "webm", "gif": "mp4",
                   "webm-alpha": "mp4", "prores": "mp4"}


def original_format(fmt):
    """The container the matched cut is written in for a given render format."""
    return ORIGINAL_FORMAT.get(str(fmt or "mp4"), "mp4")


def original_file(fmt):
    """out.mp4 -> out.original.mp4. Beside the render, never over it."""
    return FORMATS[original_format(fmt)]["file"].replace("out.", "out.original.", 1)


def count_frames(frames_dir, params=None):
    """How many frames render() would consume from this directory.

    With `params` carrying a frame_in/frame_out window, this is the length of
    that window -- which is what the matched cut has to agree with.
    """
    files = _list_frames(frames_dir)
    if not files or params is None:
        return len(files)
    return len(frame_range(files, _params(params))[0])


def _numbered_run(frames_dir, files):
    """ffmpeg -i arguments for a contiguous run of numbered frames, or None.

    The frames are written `%04d.jpg` from 0 (server.extract_frames), so the
    whole directory is normally one image2 pattern -- which means ffmpeg reads
    the JPEGs with its own decoder and nothing is re-encoded through a second
    one on the way. A directory with a hole in the numbering is not that, and
    gets the frame-by-frame path instead, because a pattern would silently stop
    at the hole.
    """
    stems = [os.path.splitext(f)[0] for f in files]
    pad = len(stems[0])
    if not all(x.isdigit() and len(x) == pad for x in stems):
        return None
    start = int(stems[0])
    if [int(x) for x in stems] != list(range(start, start + len(stems))):
        return None
    ext = os.path.splitext(files[0])[1]
    return ["-f", "image2", "-start_number", str(start),
            "-i", os.path.join(frames_dir, "%%0%dd%s" % (pad, ext))]


def _run_counting(cmd, total, progress=None):
    """Run ffmpeg with -progress on stdout; return the frames it actually wrote.

    The count is the point. A file that is one frame short of the render is
    worse than no file at all, so this reads ffmpeg's own counter rather than
    trusting that the input it was pointed at was the input we meant.
    """
    cmd = list(cmd)
    cmd[4:4] = ["-progress", "pipe:1", "-nostats"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    err = []
    # stderr on its own thread: a chatty failure would otherwise deadlock
    # against the -progress pipe being read here (same shape as extract_frames)
    t = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
    t.start()
    done = 0
    for line in p.stdout:
        if line.startswith("frame="):
            try:
                done = int(line.split("=", 1)[1].strip() or 0)
            except ValueError:
                pass
            if progress:
                progress(min(done, total), total)
    p.wait()
    t.join(timeout=5)
    if p.returncode != 0:
        raise RuntimeError('ffmpeg failed: ' + ("".join(err))[-800:])
    return done


def render_original(frames_dir, out_path, params=None, progress=None):
    """The same frames, re-encoded and otherwise untouched.

    The frame list is render()'s own -- _list_frames(), same directory, same
    order -- so there is no second notion of "the range" that could drift from
    the one the dither used. What differs is only that these frames go to the
    encoder as they are.
    """
    a = _params(params)
    files = _list_frames(frames_dir)
    if not files:
        raise RuntimeError('no frames in ' + frames_dir)
    files, base = frame_range(files, a)
    fmt = original_format(a.get('format'))
    fps = int(a['fps'])
    H, W = np.asarray(Image.open(os.path.join(frames_dir, files[0]))
                      .convert('RGB')).shape[:2]
    total = len(files)

    # A canvas is a MOVING crop, so the image2 fast path -- which hands ffmpeg
    # the JPEGs untouched -- cannot express it. The pair has to be framed
    # identically or it is not a pair, so a canvas takes the frame-by-frame
    # path and the same warp the render used.
    plan = canvas_plan(a, total)
    if plan:
        OW, OH = plan['tw'], plan['th']
        p = subprocess.Popen(ffmpeg_cmd(fmt, OW, OH, fps, out_path),
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i, fn in enumerate(files):
                rgb = np.asarray(Image.open(os.path.join(frames_dir, fn))
                                 .convert('RGB'))
                p.stdin.write(np.ascontiguousarray(warp_rgb(rgb, plan, i),
                                                   np.uint8).tobytes())
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
        return {'out': out_path, 'frames': total, 'w': OW, 'h': OH,
                'format': fmt, 'src_w': W, 'src_h': H, 'canvas': True,
                'first_frame': base, 'last_frame': base + total - 1,
                'bytes': os.path.getsize(out_path)}

    run = _numbered_run(frames_dir, files)
    if run:
        cmd = ffmpeg_cmd(fmt, W, H, fps, out_path,
                         input_args=["-framerate", str(fps)] + run)
        # -start_number only says where to BEGIN; the image2 demuxer would
        # happily read past the window's last frame to the end of the
        # directory, so the count is pinned on the output side too.
        cmd = cmd[:-1] + ["-frames:v", str(total)] + cmd[-1:]
        wrote = _run_counting(cmd, total, progress)
        if wrote != total:
            raise RuntimeError('the original cut got %d frames, the render had %d'
                               % (wrote, total))
    else:
        # holes in the numbering: hand the frames over one at a time, in the
        # order render() would have read them
        p = subprocess.Popen(ffmpeg_cmd(fmt, W, H, fps, out_path),
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i, fn in enumerate(files):
                rgb = np.asarray(Image.open(os.path.join(frames_dir, fn))
                                 .convert('RGB'))
                p.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())
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
    return {'out': out_path, 'frames': total, 'w': W, 'h': H, 'format': fmt,
            'first_frame': base, 'last_frame': base + total - 1,
            'bytes': os.path.getsize(out_path)}

# ---------------------------------------------------------- dots as data
def render_dots(frames_dir, subjects, params=None, progress=None):
    """The same dots, as positions instead of pixels.

    Returns a dots.py document: one (N, 2) int array of dot centres per subject
    per frame. It is the export path for .dots.gz, and it runs the same
    `dots_on` the renderer paints with, so the data cannot drift from the video.
    """
    a = _params(params)
    a['mode'] = 'dots'
    files = _list_frames(frames_dir)
    if not files:
        raise RuntimeError('no frames in ' + frames_dir)
    files, base = frame_range(files, a)
    if not subjects:
        raise RuntimeError('the dots look needs at least one tracked subject')
    H, W = np.asarray(Image.open(os.path.join(frames_dir, files[0])).convert('RGB')).shape[:2]
    bg = a['bg']
    blue = blue_noise(64, int(a['seed'])).astype(np.float32)
    plan = canvas_plan(a, len(files))
    OW, OH = (plan['tw'], plan['th']) if plan else (W, H)
    F = dots_fields(OH, OW, a, blue)
    cols = [_subject_palette(s, i, bg)[-1] for i, s in enumerate(subjects)]

    last = [None] * len(subjects)
    frames = []
    total = len(files)
    for i, fn in enumerate(files):
        rgb = np.asarray(Image.open(os.path.join(frames_dir, fn)).convert('RGB'))
        masks = []
        for k, s in enumerate(subjects):
            last[k] = _load_mask(s['masks'], base + i, H, W, last[k])
            masks.append(last[k])
        if plan:
            rgb = warp_rgb(rgb, plan, i)
            masks = [warp_mask(m, plan, i) for m in masks]
        frames.append([dot_positions(on, F) for on in dots_on(rgb, masks, a, F)])
        if progress:
            progress(i + 1, total)
    return dict(w=OW, h=OH, fps=int(a['fps']), dotpx=int(a['dotpx']),
                palette=[bg] + cols, bg=bg, bg_index=0, first_frame=base,
                subjects=[{'color': c} for c in cols], frames=frames)


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
            ap.add_argument('--' + k, type=(int if v is None else type(v)), default=v)
    args = ap.parse_args()
    subs = []
    for idx, spec in enumerate(args.masks):
        d, _, c = spec.partition(':')
        subs.append({'masks': d, 'dot': c or SUBJECT_COLORS[idx % len(SUBJECT_COLORS)]})
    prm = {k: getattr(args, k) for k in DEFAULTS}
    if args.palette:
        prm['palette'] = args.palette.split(',')
    print(json.dumps(render(args.frames, subs, args.out, prm), indent=2))
