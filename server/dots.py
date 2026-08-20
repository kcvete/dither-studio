#!/usr/bin/env python3
"""The .dots.gz format, server side.

A .dots.gz is the dot positions themselves -- for every frame, for every
subject, the integer centre of every dot the renderer lit -- rather than a
picture of them. web/player/dither-player.js is the reference implementation
and carries the format's full spec in its header; this file mirrors it byte for
byte so a file written here plays there and vice versa.

    encode(doc) -> bytes          the raw body
    pack(doc)   -> bytes          gzipped, i.e. the .dots.gz file
    decode/unpack                 the way back
    rasterise(doc, out, fmt)      dots -> mp4/webm/gif/... via ffmpeg

`doc` is a dict:
    {w, h, fps, dotpx, palette:[ '#rrggbb', ... ], bg_index, subjects:[{color}],
     frames: [ [ (N,2) int array of (x, y) per subject ] per frame ]}
"""
import gzip
import io
import json
import subprocess

import numpy as np

MAGIC = b"DOTS"
VERSION = 1


# ------------------------------------------------------------------ colours
def hexcol(s):
    s = str(s).lstrip('#')
    if len(s) == 3:
        s = ''.join(c * 2 for c in s)
    return [int(s[i:i + 2], 16) for i in (0, 2, 4)]


def colhex(r, g, b):
    return '#%02x%02x%02x' % (int(r), int(g), int(b))


# ------------------------------------------------------------- varint bits
def _varint(out, v):
    v &= 0xffffffff
    while v >= 0x80:
        out.append((v & 0x7f) | 0x80)
        v >>= 7
    out.append(v)


def _svarint(out, v):
    """zigzag: a small negative delta costs one byte, not five"""
    _varint(out, ((v << 1) ^ (v >> 31)) & 0xffffffff)


class _Reader:
    def __init__(self, b):
        self.b = b
        self.n = 0

    def u8(self):
        v = self.b[self.n]
        self.n += 1
        return v

    def u16(self):
        v = self.b[self.n] | (self.b[self.n + 1] << 8)
        self.n += 2
        return v

    def varint(self):
        v = s = 0
        while True:
            c = self.b[self.n]
            self.n += 1
            v |= (c & 0x7f) << s
            s += 7
            if not c & 0x80:
                return v

    def svarint(self):
        v = self.varint()
        return (v >> 1) ^ -(v & 1)


# ------------------------------------------------------------------ encode
def encode(doc):
    pal = list(doc['palette'])
    subs = doc.get('subjects') or [{'color': pal[-1]}]

    def idx_of(c):
        if c in pal:
            return pal.index(c)
        pal.append(c)
        return len(pal) - 1

    sub_idx = [idx_of(s['color']) for s in subs]
    bg_index = doc.get('bg_index')
    if bg_index is None:
        bg_index = idx_of(doc.get('bg') or pal[0])

    out = bytearray()
    out += MAGIC
    out.append(VERSION)
    out.append(0)                                   # flags
    for v in (int(doc['w']), int(doc['h']), len(doc['frames'])):
        out.append(v & 0xff)
        out.append((v >> 8) & 0xff)
    out.append(int(doc.get('fps', 30)))
    out.append(int(doc.get('dotpx', 3)))
    out.append(len(pal))
    out.append(len(subs))
    out.append(bg_index)
    out.append(0)                                   # reserved
    for c in pal:
        out += bytes(hexcol(c))
    for i in sub_idx:
        out.append(i)

    for frame in doc['frames']:
        for k in range(len(subs)):
            xy = frame[k] if k < len(frame) else np.zeros((0, 2), np.int32)
            xy = np.asarray(xy, np.int32).reshape(-1, 2)
            _varint(out, len(xy))
            px = py = 0
            for x, y in xy:
                _svarint(out, int(x) - px)
                _svarint(out, int(y) - py)
                px, py = int(x), int(y)
    return bytes(out)


def decode(b):
    b = bytes(b)
    if b[:4] != MAGIC:
        raise ValueError('not a .dots file')
    r = _Reader(b)
    r.n = 4
    version = r.u8()
    r.u8()
    if version != VERSION:
        raise ValueError('unsupported .dots version %d' % version)
    w, h, n_frames = r.u16(), r.u16(), r.u16()
    fps, dotpx, n_pal, n_sub = r.u8(), r.u8(), r.u8(), r.u8()
    bg_index = r.u8()
    r.u8()
    palette = [colhex(r.u8(), r.u8(), r.u8()) for _ in range(n_pal)]
    subjects = [{'color': palette[r.u8()]} for _ in range(n_sub)]
    frames = []
    for _ in range(n_frames):
        per = []
        for _ in range(n_sub):
            n = r.varint()
            xy = np.empty((n, 2), np.int32)
            px = py = 0
            for i in range(n):
                px += r.svarint()
                py += r.svarint()
                xy[i, 0] = px
                xy[i, 1] = py
            per.append(xy)
        frames.append(per)
    return dict(w=w, h=h, fps=fps, dotpx=dotpx, palette=palette,
                bg_index=bg_index, bg=palette[bg_index], subjects=subjects,
                frames=frames)


def pack(doc):
    buf = io.BytesIO()
    # mtime=0 so the same dots always give the same bytes
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9, mtime=0) as f:
        f.write(encode(doc))
    return buf.getvalue()


def unpack(b):
    return decode(gzip.decompress(bytes(b)))


def to_json(doc):
    return {
        'format': 'dither-studio/dots', 'version': VERSION,
        'w': doc['w'], 'h': doc['h'], 'fps': doc['fps'], 'dotpx': doc['dotpx'],
        'palette': doc['palette'], 'bgIndex': doc.get('bg_index', 0),
        'subjects': [{'color': s['color']} for s in doc['subjects']],
        'frames': [[np.asarray(xy, np.int32).reshape(-1).tolist() for xy in f]
                   for f in doc['frames']],
    }


def from_json(j):
    pal = j['palette']
    return dict(w=j['w'], h=j['h'], fps=j['fps'], dotpx=j['dotpx'], palette=pal,
                bg_index=j.get('bgIndex', 0), bg=pal[j.get('bgIndex', 0)],
                subjects=j['subjects'],
                frames=[[np.asarray(a, np.int32).reshape(-1, 2) for a in f]
                        for f in j['frames']])


# --------------------------------------------------------------- rasterise
def paint(doc, i, alpha=False):
    """One frame as an (H, W, 3|4) uint8 array.

    The rule this shares with app.js::renderDots, render.py::_frame_dots and
    the player: a dot is a dotpx square centred on its integer position with
    half = dotpx >> 1, and every pixel of it is CLAMPED into the frame rather
    than clipped away. Change it in one place and the replay stops matching.
    """
    W, H = int(doc['w']), int(doc['h'])
    dp = max(1, int(doc.get('dotpx', 3)))
    half = dp >> 1
    ch = 4 if alpha else 3
    bg = hexcol(doc.get('bg') or doc['palette'][doc.get('bg_index', 0)])
    canvas = np.empty((H, W, ch), np.uint8)
    canvas[:, :, :3] = np.array(bg, np.uint8)
    if alpha:
        canvas[:, :, 3] = 0
    off = np.arange(dp) - half
    oy, ox = np.meshgrid(off, off, indexing='ij')
    oy, ox = oy.ravel()[None, :], ox.ravel()[None, :]
    for k, xy in enumerate(doc['frames'][i]):
        xy = np.asarray(xy, np.int32).reshape(-1, 2)
        if not len(xy):
            continue
        col = hexcol(doc['subjects'][k]['color'])
        yy = np.clip(xy[:, 1][:, None] + oy, 0, H - 1)
        xx = np.clip(xy[:, 0][:, None] + ox, 0, W - 1)
        canvas[yy.ravel(), xx.ravel()] = np.array(col + ([255] if alpha else []),
                                                  np.uint8)
    return canvas


def rasterise(doc, out_path, fmt='mp4', gif_fps=None, progress=None):
    """Replay a dots document into a video file, no source clip involved."""
    import render as R
    W, H = int(doc['w']), int(doc['h'])
    fps = int(doc.get('fps', 30))
    alpha = R.FORMATS[fmt]['alpha']
    p = subprocess.Popen(R.ffmpeg_cmd(fmt, W, H, fps, out_path, gif_fps),
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    total = len(doc['frames'])
    try:
        for i in range(total):
            p.stdin.write(np.ascontiguousarray(paint(doc, i, alpha), np.uint8).tobytes())
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
    return {'out': out_path, 'frames': total, 'w': W, 'h': H, 'format': fmt}


if __name__ == '__main__':
    import argparse
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description='inspect or rasterise a .dots.gz')
    ap.add_argument('file')
    ap.add_argument('--out', default=None, help='render it to this video file')
    ap.add_argument('--format', default='mp4')
    ap.add_argument('--json', default=None, help='write the JSON variant here')
    args = ap.parse_args()
    raw = open(args.file, 'rb').read()
    doc = unpack(raw) if raw[:2] == b'\x1f\x8b' else decode(raw)
    counts = [sum(len(np.asarray(x).reshape(-1, 2)) for x in f) for f in doc['frames']]
    print(json.dumps({'w': doc['w'], 'h': doc['h'], 'fps': doc['fps'],
                      'dotpx': doc['dotpx'], 'frames': len(doc['frames']),
                      'subjects': len(doc['subjects']), 'palette': doc['palette'],
                      'bytes': len(raw), 'dots_min': min(counts), 'dots_max': max(counts),
                      'dots_mean': round(sum(counts) / max(1, len(counts)), 1)}, indent=1))
    if args.json:
        json.dump(to_json(doc), open(args.json, 'w'))
    if args.out:
        print(json.dumps(rasterise(doc, args.out, args.format), indent=1))
