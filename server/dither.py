#!/usr/bin/env python3
"""Dither engine — the Python mirror of static/dither.js.

Threshold modes are vectorised numpy. The two serial modes (error diffusion and
Riemersma) call into env/libcdither.dylib through ctypes; a pure-Python fallback
exists but is ~500x slower and only meant to keep the tool working if the C
build is unavailable.

Any rule changed here must be changed in static/dither.js too — the browser
preview and this exporter are supposed to produce the same pixels.
"""
import ctypes
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        p = os.path.join(os.path.dirname(HERE), 'env', 'libcdither.dylib')
        if not os.path.exists(p):
            _LIB = False
            return False
        l = ctypes.CDLL(p)
        F = ctypes.POINTER(ctypes.c_float)
        U = ctypes.POINTER(ctypes.c_ubyte)
        l.error_diffuse.argtypes = [F, ctypes.c_int, ctypes.c_int, U, ctypes.c_int,
                                    F, ctypes.c_int, ctypes.c_float, ctypes.c_int,
                                    ctypes.c_float, F]
        l.riemersma.argtypes = [F, ctypes.c_int, ctypes.c_int, U, ctypes.c_int,
                                ctypes.c_int, ctypes.c_float, ctypes.c_float, F]
        _LIB = l
    return _LIB


# ----------------------------------------------------------------- palettes
PALETTES = [
    {"id": "bw",         "name": "Black & White",   "colors": ["#000000", "#ffffff"]},
    {"id": "sage",       "name": "Sage",            "colors": ["#c9d4c5", "#b0413e"]},
    {"id": "forest",     "name": "Forest",          "colors": ["#0f1f18", "#d7e3d5"]},
    {"id": "ember",      "name": "Ember",           "colors": ["#e8804a", "#f6ece2"]},
    {"id": "mist",       "name": "Mist",            "colors": ["#7d8f80", "#c2cfbd", "#ffffff"]},
    {"id": "gameboy",    "name": "Game Boy DMG",    "colors": ["#0f380f", "#306230", "#8bac0f", "#9bbc0f"]},
    {"id": "red",        "name": "Red Mono",        "colors": ["#1a0000", "#ff2d2d"]},
    {"id": "green",      "name": "Green Mono",      "colors": ["#001a05", "#2dff6a"]},
    {"id": "blue",       "name": "Blue Mono",       "colors": ["#00081a", "#3d8bff"]},
    {"id": "amber",      "name": "Amber Mono",      "colors": ["#1a1000", "#ffb000"]},
    {"id": "cmyk",       "name": "CMYK",            "colors": ["#000000", "#00ffff", "#ff00ff", "#ffff00", "#ffffff"]},
    {"id": "rgby",       "name": "RGBY",            "colors": ["#000000", "#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ffffff"]},
    {"id": "bwr",        "name": "Black White Red", "colors": ["#000000", "#ffffff", "#d02f26"]},
    {"id": "purpgreen",  "name": "Purple & Green",  "colors": ["#2b1b46", "#7b5ea7", "#8fd694", "#f2f0e6"]},
    {"id": "blueyellow", "name": "Blue & Yellow",   "colors": ["#10214b", "#3b5bbf", "#f4c542", "#fdf6e3"]},
    {"id": "c64",        "name": "Commodore 64",    "colors": ["#000000", "#626262", "#898989", "#adadad", "#ffffff",
                                                               "#9f4e44", "#cb7e75", "#6d5412", "#a1683c", "#c9d487",
                                                               "#9ae29b", "#5cab5e", "#6abfc6", "#887ecb", "#50459b", "#a057a3"]},
    {"id": "grey4",      "name": "4 Greys",         "colors": ["#000000", "#555555", "#aaaaaa", "#ffffff"]},
    {"id": "grey8",      "name": "8 Greys",         "colors": ["#000000", "#242424", "#484848", "#6d6d6d",
                                                               "#919191", "#b6b6b6", "#dadada", "#ffffff"]},
]

# --------------------------------------------------- error-diffusion kernels
KERNELS = {
    "floyd-steinberg": {"name": "Floyd–Steinberg", "div": 16,
                        "k": [(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)]},
    "false-fs":        {"name": "False Floyd–Steinberg", "div": 8,
                        "k": [(1, 0, 3), (0, 1, 3), (1, 1, 2)]},
    "jarvis":          {"name": "Jarvis–Judice–Ninke", "div": 48,
                        "k": [(1, 0, 7), (2, 0, 5),
                              (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
                              (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)]},
    "stucki":          {"name": "Stucki", "div": 42,
                        "k": [(1, 0, 8), (2, 0, 4),
                              (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
                              (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)]},
    "atkinson":        {"name": "Atkinson", "div": 8,
                        "k": [(1, 0, 1), (2, 0, 1), (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)]},
    "burkes":          {"name": "Burkes", "div": 32,
                        "k": [(1, 0, 8), (2, 0, 4),
                              (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)]},
    "sierra3":         {"name": "Sierra 3", "div": 32,
                        "k": [(1, 0, 5), (2, 0, 3),
                              (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
                              (-1, 2, 2), (0, 2, 3), (1, 2, 2)]},
    "sierra2":         {"name": "Sierra 2", "div": 16,
                        "k": [(1, 0, 4), (2, 0, 3),
                              (-2, 1, 1), (-1, 1, 2), (0, 1, 3), (1, 1, 2), (2, 1, 1)]},
    "sierra-lite":     {"name": "Sierra 2-4A (lite)", "div": 4,
                        "k": [(1, 0, 2), (-1, 1, 1), (0, 1, 1)]},
    "fan93":           {"name": "Fan 93", "div": 16,
                        "k": [(1, 0, 7), (-2, 1, 1), (-1, 1, 3), (0, 1, 5)]},
    "shiau-fan":       {"name": "Shiau–Fan", "div": 8,
                        "k": [(1, 0, 4), (-2, 1, 1), (-1, 1, 1), (0, 1, 2)]},
    "shiau-fan2":      {"name": "Shiau–Fan 2", "div": 16,
                        "k": [(1, 0, 8), (-3, 1, 1), (-2, 1, 1), (-1, 1, 2), (0, 1, 4)]},
    "stevenson-arce":  {"name": "Stevenson–Arce", "div": 200,
                        "k": [(2, 0, 32),
                              (-3, 1, 12), (-1, 1, 26), (1, 1, 30), (3, 1, 16),
                              (-2, 2, 12), (0, 2, 26), (2, 2, 12),
                              (-3, 3, 5), (-1, 3, 12), (1, 3, 12), (3, 3, 5)]},
    "simple-2d":       {"name": "Simple 2D", "div": 2, "k": [(1, 0, 1), (0, 1, 1)]},
}

MODES = [
    {"id": "dots",       "name": "Dots",            "note": "particle swarm, flicker-free"},
    {"id": "bluenoise",  "name": "Blue noise",      "note": "organic grain, flicker-free"},
    {"id": "ordered",    "name": "Bayer",           "note": "crosshatch, flicker-free"},
    {"id": "halftone",   "name": "Halftone",        "note": "newsprint screen, flicker-free"},
    {"id": "whitenoise", "name": "White noise",     "note": "raw grain, flicker-free"},
    {"id": "errordiff",  "name": "Error diffusion", "note": "sharpest detail — flickers on video"},
    {"id": "riemersma",  "name": "Riemersma",       "note": "Hilbert curve — flickers on video"},
]

STABLE = {"dots": True, "ordered": True, "halftone": True, "bluenoise": True,
          "whitenoise": True, "errordiff": False, "riemersma": False}


# ------------------------------------------------------- threshold matrices
def bayer(n):
    m = np.array([[0, 2], [3, 1]], np.int64)
    while m.shape[0] < n:
        s = m.shape[0]
        o = np.empty((s * 2, s * 2), np.int64)
        v = m * 4
        o[:s, :s] = v
        o[:s, s:] = v + 2
        o[s:, :s] = v + 3
        o[s:, s:] = v + 1
        m = o
    return ((m + 0.5) / (n * n)).astype(np.float32)


def clustered(n):
    """45-degree rotated clustered-dot screen (classic newsprint halftone)."""
    y, x = np.mgrid[0:n, 0:n]
    cx = (x + 0.5) / n - 0.5
    cy = (y + 0.5) / n - 0.5
    u, v = cx + cy, cx - cy
    d = np.sqrt(u * u + v * v).ravel()
    order = np.argsort(d, kind='stable')
    r = np.empty(n * n, np.int64)
    r[order] = np.arange(n * n)
    return ((r + 0.5) / (n * n)).reshape(n, n).astype(np.float32)


_MATRIX_CACHE = {}


def matrix(kind, n):
    key = (kind, n)
    if key not in _MATRIX_CACHE:
        _MATRIX_CACHE[key] = clustered(n) if kind == 'halftone' else bayer(n)
    return _MATRIX_CACHE[key]


# ------------------------------------------------------------ portable hash
_M = 0xFFFFFFFF


def _u32(x):
    return np.asarray(x, dtype=np.int64) & _M


def hash01(i, j, salt, seed):
    """Deterministic uniform [0,1). Mirrored in static/dither.js."""
    x = _u32(_u32(i * 73856093) ^ _u32(j * 19349663) ^
             _u32(salt * 83492791) ^ _u32(seed * 2654435761))
    x = _u32(x ^ (x >> 16))
    x = _u32(x * 0x7feb352d)
    x = _u32(x ^ (x >> 15))
    x = _u32(x * 0x846ca68b)
    x = _u32(x ^ (x >> 16))
    return (x.astype(np.float64) / 4294967296.0).astype(np.float32)


# ------------------------------------------------------------------ helpers
def hexcol(s):
    s = s.lstrip('#')
    if len(s) == 3:
        s = ''.join(c * 2 for c in s)
    return [int(s[i:i + 2], 16) for i in (0, 2, 4)]


def palette_bytes(colors):
    return np.array([hexcol(c) for c in colors], np.uint8).reshape(-1)


def tone_lut(p):
    """256-entry LUT: brightness / contrast / gamma / invert. Mirrors toneLUT()."""
    br = float(p.get('brightness', 0.0))
    co = float(p.get('contrast', 1.0))
    ga = float(p.get('gamma', 1.0))
    # float64 throughout, cast once at the end: JS Numbers are f64 and only the
    # Float32Array store rounds, so computing this in f32 would drift by an ULP
    # and error diffusion amplifies that into visible pixel flips
    v = np.arange(256, dtype=np.float64) / 255.0
    v = (v - 0.5) * co + 0.5 + br
    v = np.clip(v, 0, 1)
    if ga != 1.0:
        v = v ** ga
    if p.get('invert'):
        v = 1.0 - v
    return (v * 255.0).astype(np.float32)


def _nearest_index(flat, pal):
    """flat: (N,3) float32 0..255 ; pal: (P,3) uint8 -> (N,) index of nearest."""
    flat = flat.astype(np.float64, copy=False)
    p = pal.astype(np.float64)
    best = None
    bestd = None
    # chunk so a 16-colour palette on a 1MP frame stays inside a sane buffer
    for i in range(p.shape[0]):
        d = ((flat - p[i]) ** 2).sum(1)
        if best is None:
            best = np.zeros(flat.shape[0], np.int32)
            bestd = d
        else:
            m = d < bestd
            best[m] = i
            bestd = np.where(m, d, bestd)
    return best


def extract_palette(rgb, k):
    """Median-cut. rgb: (H,W,3) uint8 -> list of #rrggbb."""
    h, w = rgb.shape[:2]
    step = max(1, int(math.sqrt((h * w) / 20000)))
    px = rgb[::step, ::step].reshape(-1, 3).astype(np.int32)
    boxes = [px]
    while len(boxes) < k:
        bi, br = -1, -1
        for i, b in enumerate(boxes):
            if len(b) < 2:
                continue
            r = (b.max(0) - b.min(0)).max()
            if r > br:
                br, bi = r, i
        if bi < 0:
            break
        b = boxes[bi]
        ch = int(np.argmax(b.max(0) - b.min(0)))
        b = b[np.argsort(b[:, ch], kind='stable')]
        mid = len(b) // 2
        boxes[bi:bi + 1] = [b[:mid], b[mid:]]
    out = []
    for b in boxes:
        if not len(b):
            continue
        m = b.mean(0).round().astype(int).clip(0, 255)
        out.append('#%02x%02x%02x' % tuple(m))
    return out


# --------------------------------------------------------------------- core
def dither_rgb(rgb, p, gate=None):
    """rgb: (H,W,3) uint8. gate: (H,W) float32 0..1 or None.
    Returns (H,W,3) uint8. Where gate<=0 the original pixels are preserved."""
    h, w = rgb.shape[:2]
    pal = palette_bytes(p['palette']).reshape(-1, 3)
    np_ = pal.shape[0]
    lut = tone_lut(p)
    seed = int(p.get('seed', 7))
    strength = float(p.get('strength', 1.0))
    spread = 255.0 * strength / max(1, np_ - 1)
    mode = p.get('mode', 'bluenoise')

    src = lut[rgb.reshape(-1, 3).astype(np.int32)].astype(np.float32)  # (N,3) 0..255

    if mode in ('errordiff', 'riemersma'):
        buf = np.ascontiguousarray(src.reshape(-1), np.float32)
        g = None if gate is None else np.ascontiguousarray(gate.reshape(-1), np.float32)
        lib = _lib()
        if lib:
            F = ctypes.POINTER(ctypes.c_float)
            palc = np.ascontiguousarray(pal.reshape(-1), np.uint8)
            gp = g.ctypes.data_as(F) if g is not None else None
            if mode == 'errordiff':
                kd = KERNELS.get(p.get('algo'), KERNELS['floyd-steinberg'])
                kern = np.ascontiguousarray(
                    np.array(kd['k'], np.float32).reshape(-1), np.float32)
                lib.error_diffuse(buf.ctypes.data_as(F), w, h,
                                  palc.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), np_,
                                  kern.ctypes.data_as(F), len(kd['k']), float(kd['div']),
                                  1 if p.get('serpentine') else 0, strength, gp)
            else:
                lib.riemersma(buf.ctypes.data_as(F), w, h,
                              palc.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), np_,
                              int(p.get('queue', 16)), float(p.get('ratio', 16)),
                              strength, gp)
        else:
            _serial_fallback(buf, w, h, pal, np_, p, strength, g, mode)
        out = buf.reshape(h, w, 3).clip(0, 255).astype(np.uint8)
    else:
        if mode in ('ordered', 'halftone'):
            mn = int(p.get('matrix', 4))
            m = matrix('halftone' if mode == 'halftone' else 'bayer', mn)
            thr = np.tile(m, (h // mn + 1, w // mn + 1))[:h, :w]
        elif mode == 'bluenoise':
            m = p['_blue']           # 64x64 float32, injected by the caller
            thr = np.tile(m, (h // 64 + 1, w // 64 + 1))[:h, :w]
        else:                        # whitenoise
            yy = np.broadcast_to(np.arange(h, dtype=np.int64)[:, None], (h, w))
            xx = np.broadcast_to(np.arange(w, dtype=np.int64)[None, :], (h, w))
            thr = hash01(yy, xx, 11, seed)
        off = ((thr - 0.5) * spread).reshape(-1, 1)
        idx = _nearest_index(np.clip(src + off, 0, 255), pal)
        out = pal[idx].reshape(h, w, 3)

    if gate is not None:
        keep = (gate <= 0)[:, :, None]
        out = np.where(keep, rgb, out)
    return np.ascontiguousarray(out, np.uint8)


def _serial_fallback(buf, w, h, pal, np_, p, strength, gate, mode):
    """Pure-Python error diffusion. Correct but ~500x slower than the C path."""
    palf = pal.astype(np.float32)
    if mode == 'riemersma':
        raise RuntimeError('riemersma needs env/libcdither.dylib (run ./setup.sh)')
    kd = KERNELS.get(p.get('algo'), KERNELS['floyd-steinberg'])
    kern, div = kd['k'], float(kd['div'])
    serp = bool(p.get('serpentine'))
    for y in range(h):
        rev = serp and (y & 1)
        rng = range(w - 1, -1, -1) if rev else range(w)
        for x in rng:
            i = y * w + x
            q = i * 3
            c = buf[q:q + 3].copy()
            k = int(((palf - c) ** 2).sum(1).argmin())
            buf[q:q + 3] = palf[k]
            if gate is not None and gate[i] <= 0:
                continue
            e = (c - palf[k]) * strength / div
            for dx, dy, wt in kern:
                nx, ny = x + (-dx if rev else dx), y + dy
                if nx < 0 or nx >= w or ny >= h:
                    continue
                nq = (ny * w + nx) * 3
                buf[nq:nq + 3] += e * wt
