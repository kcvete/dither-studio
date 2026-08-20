#!/usr/bin/env python3
"""Engine parity gate: web/dither.js vs server/dither.py, pixel-exact.

Two gates, both byte-for-byte:

  the kernel gate   every mode, all 14 error-diffusion kernels (serpentine on
                    and off), three palettes and the tone controls, through
                    dither_rgb / ditherRGBA. 110 cases. GATE=1 repeats the whole
                    set through a subject mask.
  the compose gate  the thing a picture is actually made of: whole frames and
                    MASKED ones -- two subjects with a palette each, cutout and
                    overlay, chunky pixels, and the transparent variant the
                    alpha exports use -- through render._frame_pixels /
                    composeFrame. This is the still cutout PNG's own code path.

    env/venv/bin/python server/parity.py        # whole frame
    GATE=1 env/venv/bin/python server/parity.py # kernels through a mask
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

# ------------------------------------------------------- the compose gate
# Two overlapping subjects, so `owner` (which subject a contested pixel belongs
# to) is exercised, plus a soft edge so the >= 0.5 `inside` rule is too.
m0 = np.clip(1.6 - 3.2 * np.sqrt(((xx - W * .38) / (W * .30)) ** 2
                                 + ((yy - H * .50) / (H * .42)) ** 2), 0, 1).astype(np.float32)
m1 = np.clip(1.6 - 3.2 * np.sqrt(((xx - W * .62) / (W * .26)) ** 2
                                 + ((yy - H * .46) / (H * .38)) ** 2), 0, 1).astype(np.float32)
MASKS = [m0, m1]
BG = '#c9d4c5'
PALS = [PAL['bw'], ['#c9d4c5', '#b0413e'], ['#c9d4c5', '#2f4f4a']]

compose = []
def addc(name, masks, **kw):
    p = dict(mode='bluenoise', algo='floyd-steinberg', matrix=4, serpentine=False,
             strength=1.0, brightness=0.0, contrast=1.0, gamma=1.0, invert=False,
             pixel=1, compose='cutout', seed=7, alpha=False)
    p.update(kw)
    p['_name'] = name
    p['_masks'] = masks
    p['_palettes'] = PALS
    p['_bg'] = BG
    compose.append(p)

addc('whole', [])
addc('whole-alpha', [], alpha=True)
addc('whole-px3', [], pixel=3)
addc('cutout-1', [0])
addc('cutout-2', [0, 1])
addc('cutout-2-alpha', [0, 1], alpha=True)
addc('cutout-2-px3', [0, 1], pixel=3)
addc('cutout-2-px3-alpha', [0, 1], pixel=3, alpha=True)
addc('overlay-2', [0, 1], compose='overlay')
addc('overlay-2-alpha', [0, 1], compose='overlay', alpha=True)
addc('cutout-ed', [0, 1], mode='errordiff', algo='atkinson')
addc('cutout-ed-serp', [0, 1], mode='errordiff', algo='floyd-steinberg', serpentine=True)
addc('cutout-halftone', [0, 1], mode='halftone', matrix=8)
addc('cutout-riemersma', [0, 1], mode='riemersma')
addc('cutout-tone', [0, 1], gamma=1.7, contrast=1.3, brightness=-0.08, invert=True)

USE_GATE = os.environ.get('GATE') == '1'
payload = dict(w=W, h=H, blue=[float(v) for v in blue.ravel()],
               src=base64.b64encode(rgba.tobytes()).decode(),
               gate=base64.b64encode(gate.tobytes()).decode() if USE_GATE else None,
               masks=[base64.b64encode(m.tobytes()).decode() for m in MASKS],
               cases=cases, compose=compose)
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

cbad, cok = [], 0
for c in compose:
    a = dict(R.DEFAULTS)
    a.update({k: v for k, v in c.items() if not k.startswith('_')})
    masks = [MASKS[i] for i in c['_masks']]
    py = R._frame_pixels(rgb, masks, a, blue, c['_palettes'], c['_bg'],
                         bool(c['alpha']))
    jsarr = np.frombuffer(base64.b64decode(js['C:' + c['_name']]), np.uint8) \
        .reshape(H, W, 4)
    chans = 4 if c['alpha'] else 3
    if not c['alpha']:
        jsarr = jsarr[:, :, :3]
    diff = int((py != jsarr).any(-1).sum())
    if diff:
        cbad.append((c['_name'], chans, diff, round(100 * diff / (W * H), 3)))
    else:
        cok += 1

print(json.dumps({'gate': USE_GATE, 'cases': len(cases), 'pixel_identical': ok,
                  'mismatched': bad[:14], 'n_mismatched': len(bad),
                  'compose_cases': len(compose), 'compose_identical': cok,
                  'compose_mismatched': cbad[:14]}, indent=1))
sys.exit(1 if bad or cbad else 0)
