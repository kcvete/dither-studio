#!/usr/bin/env python3
"""Engine parity gate: web/dither.js vs server/dither.py, pixel-exact.

Runs every mode, all 14 error-diffusion kernels (serpentine on and off), three
palettes and the tone controls through both implementations and requires byte
identical output. Run with GATE=1 to repeat it through a subject mask.

    env/venv/bin/python server/parity.py        # whole frame
    GATE=1 env/venv/bin/python server/parity.py # through a mask
"""
import base64, json, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dither as D, render as R
from PIL import Image

W, H = 160, 96
rng = np.random.default_rng(3)
yy, xx = np.mgrid[0:H, 0:W]
rgb = np.stack([
    (xx / (W - 1) * 255), (yy / (H - 1) * 255),
    (((xx // 8 + yy // 8) % 2) * 200 + 30)], -1)
rgb = np.clip(rgb + rng.normal(0, 12, rgb.shape), 0, 255).astype(np.uint8)
rgba = np.dstack([rgb, np.full((H, W), 255, np.uint8)])
gate = (((xx - W / 2) ** 2 / (W * 0.32) ** 2 + (yy - H / 2) ** 2 / (H * 0.36) ** 2) < 1).astype(np.float32)
blue = R.blue_noise(64, 7).astype(np.float32)

PAL = {
    'bw': ['#000000', '#ffffff'],
    'gb': ['#0f380f', '#306230', '#8bac0f', '#9bbc0f'],
    'c64': D.PALETTES[15]['colors'],
}
cases = []
def add(name, **kw):
    p = dict(palette=PAL['bw'], strength=1.0, seed=7, gamma=1.0, contrast=1.0,
             brightness=0.0, invert=False)
    p.update(kw); p['_name'] = name; cases.append(p)

for pn, pal in PAL.items():
    add(f'bluenoise/{pn}', mode='bluenoise', palette=pal)
    add(f'whitenoise/{pn}', mode='whitenoise', palette=pal)
    for m in (2, 4, 8, 16):
        add(f'ordered{m}/{pn}', mode='ordered', matrix=m, palette=pal)
    add(f'halftone8/{pn}', mode='halftone', matrix=8, palette=pal)
    add(f'riemersma/{pn}', mode='riemersma', palette=pal)
    for algo in D.KERNELS:
        add(f'ed-{algo}/{pn}', mode='errordiff', algo=algo, palette=pal)
        add(f'edS-{algo}/{pn}', mode='errordiff', algo=algo, serpentine=True, palette=pal)
# tone controls + strength
add('tone', mode='bluenoise', gamma=1.8, contrast=1.4, brightness=-0.1, invert=True, palette=PAL['gb'])
add('strength', mode='errordiff', algo='atkinson', strength=0.55, palette=PAL['c64'])

USE_GATE = os.environ.get('GATE') == '1'
payload = dict(w=W, h=H, blue=[float(v) for v in blue.ravel()],
               src=base64.b64encode(rgba.tobytes()).decode(),
               gate=base64.b64encode(gate.tobytes()).decode() if USE_GATE else None,
               cases=cases)
S = os.environ.get('TMPDIR', '/tmp').rstrip('/')
json.dump(payload, open(f'{S}/pin.json', 'w'))
subprocess.run(['node', 'parity.mjs', f'{S}/pin.json', f'{S}/pout.json'],
               cwd=os.path.dirname(os.path.abspath(__file__)), check=True)
js = json.load(open(f'{S}/pout.json'))

bad, ok = [], 0
for c in cases:
    p = dict(c); p['_blue'] = blue
    g = gate if USE_GATE else None
    py = D.dither_rgb(rgb, p, g)
    jsarr = np.frombuffer(base64.b64decode(js[c['_name']]), np.uint8).reshape(H, W, 4)[:, :, :3]
    diff = int((py != jsarr).any(-1).sum())
    if diff: bad.append((c['_name'], diff, round(100 * diff / (W * H), 3)))
    else: ok += 1
print(json.dumps({'gate': USE_GATE, 'cases': len(cases), 'pixel_identical': ok,
                  'mismatched': bad[:14], 'n_mismatched': len(bad)}, indent=1))
sys.exit(1 if bad else 0)
