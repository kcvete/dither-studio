# Dither Studio

A local, offline dithering tool for **stills, clips, and clips where you only want
one thing dithered**. Point at a person in a video, the tracker follows them for the
whole clip, and only they turn into dots — or turn the whole frame into a Game Boy
screen. Nothing is uploaded anywhere; it all runs on this Mac.

![tracked subjects, per-subject palettes](docs/c-mixed.png)

---

## Run it

```bash
cd ~/dither-video
./run.sh
```

`run.sh` is idempotent. It calls `setup.sh` (creates `env/venv`, clones
`env/EdgeTAM`, fetches the checkpoint, compiles `env/libcdither.dylib` — all no-ops
after the first time), starts `server.py` on `http://127.0.0.1:8765`, and opens the
browser. If 8765 is taken it walks forward to the next free port and says which one.
`DV_PORT=` overrides, `DV_NO_OPEN=1` skips the browser.

## What you can do

Drop **an image** or **a clip** (or paste one). The steps adapt to what you gave it.

### 1 · Source
Images stay in the tab — they are never uploaded. Clips are decoded server-side to
720p / 30 fps JPEG frames, capped by the *max length* slider (10 s default, 300 frames).

### 2 · Subjects — clips only
Two choices:

* **whole clip** — every pixel of every frame gets dithered.
* **track subjects** — scrub to any frame and prompt what you care about:
  click = keep this, shift-click = not this, drag = a box. `+ add subject` for
  another object (up to 6), each with its own palette. Press **Track** and EdgeTAM
  follows them forward *and* backward through the clip, so a click on a middle
  frame still fills the whole thing.

  There is deliberately no lasso here: the video tracker consumes points and boxes
  and re-derives the outline itself on every frame.

### 3 · Look
Seven algorithms, each labelled with how it behaves on video:

| | | stability |
|---|---|---|
| **Dots** | the particle swarm — density follows tone, dots hold their position | flicker-free |
| **Blue noise** | organic grain from a void-and-cluster tile | flicker-free |
| **Bayer** | ordered crosshatch, 2×2 / 4×4 / 8×8 / 16×16 | flicker-free |
| **Halftone** | 45° rotated clustered-dot newsprint screen, same matrix sizes | flicker-free |
| **White noise** | raw hashed grain | flicker-free |
| **Error diffusion** | 14 kernels, optional serpentine — the sharpest detail | **flickers on video** |
| **Riemersma** | Hilbert-curve error diffusion, no directional bias | **flickers on video** |

The error-diffusion kernels are Floyd–Steinberg, False Floyd–Steinberg,
Jarvis–Judice–Ninke, Stucki, Atkinson, Burkes, Sierra 3, Sierra 2, Sierra 2-4A,
Fan 93, Shiau–Fan, Shiau–Fan 2, Stevenson–Arce and Simple 2D.

The flicker labels are the honest part: threshold modes reuse one fixed field every
frame, so dots stay put and only switch on and off as tone changes. Error diffusion
recomputes a chaotic error field per frame, so it boils. Both are offered; the chip
carries a `≈` marker on video so you know which you picked.

Also here: dither strength, pixel size (chunky-pixel scale, box-downsample then
nearest-upscale), brightness / contrast / gamma / invert, and **reseed** for a new
noise field. For tracked clips, *background* chooses between a flat colour and the
dithered scene.

### 4 · Palette
18 presets — Black & White, Sage, Forest, Ember, Mist, Game Boy DMG, four monochromes,
CMYK, RGBY, Black White Red, Purple & Green, Blue & Yellow, Commodore 64, 4 Greys,
8 Greys — plus **from image** (median-cut extraction from the current frame) and a
free colour editor. With tracked subjects you get one palette per subject *plus* one
for the background, switchable at the top of the step.

### 5 · Export
Images render at full source resolution in the browser and download as PNG. Clips go
to the server, which renders the identical parameters with numpy + the C dither loop
and encodes with ffmpeg (libx264, crf 18), then offers a download and an inline player.

**Compare** (in the transport bar) drags a before/after divider across the frame, and
it keeps working while the clip plays.

## Architecture

```
dither.py            the engine: palettes, threshold matrices, 14 kernels,
                     Riemersma, median-cut, tone LUT, pixel scale
static/dither.js     the same engine in JS — used for the live preview AND for
                     the client-side PNG export
cdither.c            the serial inner loops (error diffusion, Riemersma) in C;
                     built to env/libcdither.dylib, loaded with ctypes
parity.py + .mjs     the gate that keeps the two engines byte-identical
render.py            video renderer: frames + optional masks -> mp4 (also a CLI)
server.py            FastAPI on 127.0.0.1:8765
static/              index.html + app.js + style.css — vanilla, no build, no CDN
env/                 venv + EdgeTAM checkout + checkpoint + libcdither  (gitignored)
jobs/<id>/           source, frames/, masks/<obj>/, out.mp4              (gitignored)
verify.mjs           headless end-to-end check of all three flows
docs/                verification screenshots + verify-report.json
```

API:

```
POST /api/upload                    mp4/mov -> jobs/<id>/frames/%04d.jpg (720p, 30fps)
GET  /api/jobs/<id>/meta
GET  /api/jobs/<id>/frame/<n>       jpeg
POST /api/jobs/<id>/track           {frame_idx, objects:[{id, points, box}]}
GET  /api/jobs/<id>/status          {state, done_frames, fps, precision, render:{…}}
GET  /api/jobs/<id>/mask/<obj>/<n>  png (soft mask)
POST /api/jobs/<id>/render          {mode, algo, matrix, palette, subjects:[…], …}
GET  /api/jobs/<id>/out.mp4
GET  /api/bluenoise                 the 64x64 threshold tile as JSON
GET  /api/palettes                  palettes, modes, kernels, defaults, device
GET  /                              static/index.html
```

### Two engines, one output

The browser preview is not an approximation of the export — it is the same algorithm.
`parity.py` runs 110 cases (every mode, all 14 kernels with serpentine on and off,
three palettes, the tone controls) through both implementations and requires
**byte-identical** output, then repeats the whole set through a subject mask.

Getting there needed three fixes worth remembering, because each produced *visible*
pixel differences that error diffusion then amplified:

* **Float width.** JavaScript Numbers are f64 and only the `Float32Array` *store*
  rounds. The C loop now uses `double` locals with float32 storage to match, and
  `tone_lut()` computes in float64 and casts once.
* **FMA contraction.** `cc -O3` fuses `a + b*c` into one instruction that rounds
  once where JS rounds twice, so the build uses `-ffp-contract=off`.
* **RNG.** Both sides use a portable integer hash (`hash01`) instead of
  `numpy.random` / `Math.random` for jitter and stray fields.

## Verification

`verify.mjs` drives a real headless browser against a real server, a real EdgeTAM
run and real ffmpeg. No mocks. It uploads files, drags boxes and clicks points on
the canvas, switches every algorithm, samples the preview canvas' pixels, exports,
and `ffprobe`s the result.

```bash
./run.sh &
node verify.mjs http://127.0.0.1:8765 /path/to/clip.mp4 /path/to/still.jpg
env/venv/bin/python parity.py && GATE=1 env/venv/bin/python parity.py
```

Latest run (M4 Pro, 24 GB, macOS 26.1, torch 2.13 / MPS), 150-frame 1280×720 clip:

| | result |
|---|---|
| engine parity | **110/110 byte-identical**, and 110/110 again through a mask |
| still: 6 algorithms | 2-colour output each, 1280×720 |
| still: 14 kernels | **14 distinct** images, no two kernels alike |
| still: palettes | Game Boy → 4 colours, from-image → 4, pixel-scale 4× → 2 |
| still: PNG export | 1280×720, 494 KB, downloaded and probed |
| clip whole-frame | Bayer 8×8 + Game Boy, preview 58.5 fps, render 150 frames in 10.5 s |
| clip tracked | 2 subjects, 150/150 frames in 40–48 s (3.1–3.7 fps) on MPS fp16 |
| tracked → dots | preview 37.9 fps · 3989 dots, render 3.9 s, ffprobe 150 frames |
| tracked → Atkinson, 3 palettes | 12 distinct colours, render 12.2 s, ffprobe 150 frames |
| preview vs exported MP4 | **97.7 %** of pixels within 30 RGB units |
| console / page errors | **0 / 0** |

The remaining 2.3 % of the preview-vs-export comparison is not an engine difference —
it is the browser's JPEG decoder disagreeing with Pillow's by a level or two on the
source frames, plus h264 quantisation on the way out. Feed both the same decoded
pixels and they agree exactly, which is what `parity.py` measures.

## Tracking performance

EdgeTAM's README quotes **15.7 FPS on iPhone 15 Pro Max** and **150.9 FPS on A100**,
footnoting that the A100 number is *"obtained with torch compile"*; the iPhone number
is the **CoreML export on the Neural Engine**. This tool runs the PyTorch graph on
**MPS**, a third and slower path.

Measured breakdown, 1 subject, 150 frames, 23.0 s of propagate:

| stage | s | share | per frame |
|---|---|---|---|
| memory attention | 12.18 | 53 % | 81 ms |
| image encoder | 4.81 | 21 % | 32 ms |
| memory encoder | 3.28 | 14 % | 22 ms |
| SAM heads | 2.60 | 11 % | 17 ms |
| GPU→CPU + PNG write | ~1.0 | ~4 % | 6 ms |

So it is the model, not this tool's I/O. Tried and rejected: `bfloat16` (2× worse —
the RepViT encoder falls back), `torch.compile` on memory attention (2× worse —
inductor's MPS path loses on dynamic shapes), shortening the memory bank from 7 to 3
frames (inside run-to-run noise).

**fp16 autocast is on by default** — A/B in one process with the order alternated:
7.42 → 8.80 fps (**1.19×**), global mask IoU **0.9985** against fp32 over 150 frames,
worst frame 0.9609. `DV_FP32=1` turns it off.

A CoreML port was investigated in depth and *is* viable — the hard part (memory
attention, which needs a real-arithmetic rewrite of its complex-number RoPE) converts
with 1.3e-6 parity and benchmarks at 19.1 ms on the GPU against 70.9 ms on MPS,
giving a measured 2.5× on the three stages that are 88 % of runtime, i.e. ~14-17 fps
end to end. It is 3-5 days of work plus hardening, needs fp32-on-GPU for the memory
path (fp16-on-ANE produces latent errors 4× the signal), and is not done here.

## Limits

* **macOS + Apple Silicon.** Tracking is MPS (`DV_DEVICE=cpu` works, far slower).
  No CUDA extension is built.
* **The C library is required for error diffusion and Riemersma.** `setup.sh` builds
  it. The pure-Python fallback is ~500× slower and Riemersma has no fallback at all.
* **Error diffusion and Riemersma flicker on video.** That is inherent, not a bug —
  the UI marks them. Use dots / blue noise / Bayer / halftone for stable motion.
* **Short clips.** 720p / 30 fps, 300 frames / 10 s by default.
* **One track at a time.** A process-wide lock serialises EdgeTAM; a second request
  gets a 409.
* **Objects cost time.** ~6 fps for one subject, ~3-4 for two, ~1.4 for four. Track
  times swing up to 1.7× run to run under sustained load (thermals).
* **Tracking quality is EdgeTAM's.** Fast motion, blur and occlusion make masks drift;
  the fix is a better prompt, not a renderer setting.
* **Preview cost.** The preview dithers at full resolution on the main thread and
  caches 40 frames of decoded bitmaps. Heavy settings (small cell, many subjects,
  Riemersma) drop below the clip's own frame rate; the fps counter tells you.
* **Stills preview at 1600 px** on the long edge and re-render at native resolution
  only for the download, so a large photo stays responsive while you drag sliders.
* **Compare is preview-only** — not baked into the export.
* **No audio.** Video export is picture only.
* **Jobs are never garbage collected.** ~13 MB per 150-frame clip; delete by hand.
* **EdgeTAM patch.** `setup.sh` rewrites two `.view(...)` calls to `.reshape(...)` in
  `sam2/modeling/perceiver.py`; upstream throws *"view size is not compatible with
  input tensor's size and stride"* as soon as more than one object is tracked.
