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
`env/EdgeTAM`, fetches the checkpoint, compiles `env/libcdither.dylib`, exports the
CoreML tracking graphs to `env/coreml/` — all no-ops after the first time), starts
`server.py` on `http://127.0.0.1:8765`, and opens the browser. If 8765 is taken it walks forward to the next free port and says which one.
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

  **Tracking quality** — *fast (512) / balanced (768) / best (1024)*, with the
  measured fps on each chip — is the square EdgeTAM resizes every frame to before
  it looks at it. Your clip keeps its own resolution either way; only the
  tracker's internal view changes, and with it how fine an outline it can draw.
  See *Tracking performance* below for what that actually costs.

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
coreml/              EdgeTAM -> CoreML: traceable wrappers, the exporter, and the
                     accelerator that swaps three modules on a live predictor
bench/               tracking benchmark harness, stage profiler, CoreML
                     microbenchmark, the resolution A/B, results.md
server.py            FastAPI on 127.0.0.1:8765
static/              index.html + app.js + style.css — vanilla, no build, no CDN
env/                 venv + EdgeTAM checkout + checkpoint + libcdither +
                     env/coreml/*.mlpackage                        (gitignored)
jobs/<id>/           source, frames/, masks/<obj>/, out.mp4              (gitignored)
verify.mjs           headless end-to-end check of all three flows
docs/                verification screenshots + verify-report.json
```

API:

```
POST /api/upload                    mp4/mov -> jobs/<id>/frames/%04d.jpg (720p, 30fps)
GET  /api/jobs/<id>/meta
GET  /api/jobs/<id>/frame/<n>       jpeg
POST /api/jobs/<id>/track           {frame_idx, image_size, objects:[{id, points, box}]}
GET  /api/jobs/<id>/status          {state, done_frames, fps, backend, image_size, render:{…}}
GET  /api/jobs/<id>/mask/<obj>/<n>  png (soft mask)
POST /api/jobs/<id>/render          {mode, algo, matrix, palette, subjects:[…], …}
GET  /api/jobs/<id>/out.mp4
GET  /api/bluenoise                 the 64x64 threshold tile as JSON
GET  /api/palettes                  palettes, modes, kernels, defaults, device,
                                    backend, track_sizes
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
the canvas, switches every algorithm and every tracking quality, samples the preview
canvas' pixels, exports, and `ffprobe`s the result.

```bash
./run.sh &
node verify.mjs http://127.0.0.1:8765 /path/to/clip.mp4 /path/to/still.jpg
env/venv/bin/python parity.py && GATE=1 env/venv/bin/python parity.py
```

Latest run (M4 Pro, 24 GB, macOS 26.1, torch 2.13 / MPS + CoreML), 150-frame
1280×720 clip:

| | result |
|---|---|
| engine parity | **110/110 byte-identical**, and 110/110 again through a mask |
| still: 6 algorithms | 2-colour output each, 1280×720 |
| still: 14 kernels | **14 distinct** images, no two kernels alike |
| still: palettes | Game Boy → 4 colours, from-image → 4, pixel-scale 4× → 2 |
| still: PNG export | 1280×720, 494 KB, downloaded and probed |
| clip whole-frame | Bayer 8×8 + Game Boy, preview 57.8 fps, render 150 frames in 10.4 s |
| clip tracked, 2 subjects, best quality | 150/150 frames in 26.9 s (**5.6 fps**) on the CoreML backend |
| clip tracked, 1 subject, fast quality | 150/150 frames in 11.1 s (**13.6 fps**), tracker at 512 px, masks still 1280×720 |
| tracked → dots | preview 42.4 fps · 3991 dots, render 3.8 s, ffprobe 150 frames |
| tracked → Atkinson, 3 palettes | 12 distinct colours, render 12.2 s, ffprobe 150 frames |
| preview vs exported MP4 | **97.8 %** of pixels within 30 RGB units |
| console / page errors | **0 / 0** |

Those two tracked rows are end to end — model build, frame decode, propagate and
PNG writing — which is why they sit below the propagate-only fps in
`bench/results.md`.

The remaining 2.2 % of the preview-vs-export comparison is not an engine difference —
it is the browser's JPEG decoder disagreeing with Pillow's by a level or two on the
source frames, plus h264 quantisation on the way out. Feed both the same decoded
pixels and they agree exactly, which is what `parity.py` measures.

## Tracking performance

EdgeTAM's README quotes **15.7 FPS on iPhone 15 Pro Max** and **150.9 FPS on A100**,
footnoting that the A100 number is *"obtained with torch compile"*; the iPhone number
is the **CoreML export on the Neural Engine**. This tool used to run the PyTorch graph
on **MPS**, a third and slower path. It no longer does — the three modules that are
~90 % of a track step now run as CoreML graphs, and MPS keeps the rest.

Where the time went on MPS (1 subject, 1024 px input, per frame, wall clock with
`torch.mps.synchronize()` around each stage — `bench/profile_stages.py`):

| stage | ms | share |
|---|---|---|
| memory attention | 42.9 | 38 % |
| image encoder (RepViT + FPN neck) | 39.8 | 35 % |
| memory encoder + spatial perceiver | 18.3 | 16 % |
| SAM heads | 10.0 | 9 % |
| everything else | ~2 | 2 % |

So four modules, three of which convert cleanly. `coreml/export.py` writes one
static-shape `.mlpackage` per (graph, object count); `coreml/accel.py` swaps the
three module calls on a live predictor and leaves the memory-bank bookkeeping, the
SAM heads and the propagate loop exactly as upstream wrote them. Every shim checks
its tensors against the exported shape and falls back to PyTorch when they don't
match, which is what makes the cold-start frames (the memory bank is still filling
for the first 16) and un-exported object counts simply work.

Per-call cost of the exported graphs on this machine (`bench/micro_coreml.py`):

| graph | precision | unit | ms | on MPS |
|---|---|---|---|---|
| image encoder | fp16 | GPU | 15.2 | ~40 |
| memory attention | fp16 | GPU | 18.2 | ~43 |
| memory encoder + perceiver | fp32 compute, fp16 in | GPU | 4.8 | ~18 |

Three findings behind those choices:

* **Memory attention needed a real-arithmetic RoPE.** It rotates queries and keys
  with complex tensors (`torch.polar`, `view_as_complex`), and MIL has no complex
  type, so conversion dies with `KeyError: np.int32(9)`. `coreml/wrappers.py`
  recomputes the identical rotation as `a·cos − b·sin, a·sin + b·cos`; parity
  against the stock module is **1.3e-6**.
* **The Neural Engine is fast and wrong here.** The encoder runs 13.2 ms on the ANE
  against 15.2 on the GPU, but its fp16 accumulation moved frame 147 of the reference
  clip to **IoU 0.893** where the same graph on the GPU holds 0.958 (clip mean
  0.993 → 0.998). The 2 ms is not worth a visibly wrong frame, so every graph is
  pinned to `CPU_AND_GPU`.
* **The memory encoder stays fp32.** Its latents *are* the memory bank, so error
  there compounds down the clip; fp16 compute puts an error 4× the signal into them.
  Its *inputs* are fp16 — `pix_feat` is the encoder's fp16 output anyway.

The image encoder also runs one frame ahead on a worker thread: it is a pure
function of the frame, so it does not have to wait for the memory bank. That is
worth 14.6 → 15.4 fps on its own. The numpy → MPS copy stays on the calling
thread; two threads driving the Metal queue trips an `IOGPUMetalCommandBuffer`
assertion.

### Measured

`bench/bench.py` runs any backend over a fixed 150-frame 1280×720 clip with a fixed
prompt, scores every mask against the fp32 MPS output and appends to
`bench/results.md`:

```bash
env/venv/bin/python bench/bench.py --backend coreml --runs 3
```

`fps` is frames ÷ propagate seconds (the GPU→CPU mask copy is in, PNG writing is
out). IoU is binary-at-0.5 against `masks_edgetam/`, per frame. Run-to-run spread
from thermals is up to 1.7×, so these are **one run of each backend per round, three
rounds, interleaved** — a straight sequence of backends measures the machine's
temperature, not the code.

| backend | `DV_BACKEND` | fps best | fps med | IoU mean | worst frame |
|---|---|---|---|---|---|
| CoreML for the three heavy stages | `coreml` **(default)** | **15.57** | **15.40** | 0.9968 | 0.9580 |
| …without the encoder prefetch thread | — | 14.73 | 14.60 | 0.9968 | 0.9580 |
| `torch.compile` on the image encoder | `torch-compiled` | 11.85 | 11.83 | 0.9961 | 0.9589 |
| `model.half()` under autocast | `torch-half` | 9.48 | 9.38 | 0.9963 | 0.9589 |
| fp16 autocast — *the previous default* | `torch` | 9.52 | 9.42 | 0.9984 | 0.9592 |
| no autocast — *the mask reference* | `torch-fp32` | 7.99 | 7.93 | 1.0000 | 1.0000 |

**1.6× over the previous default, 1.9× over fp32**, with the whole-clip mean IoU
at 0.997 and no frame below 0.958. `bench/results.md` has the full set, including
the eight things that did not move the needle.

`DV_BACKEND=` picks one; anything that will not build falls through to the next
row down, so a machine without coremltools still tracks:

```bash
DV_BACKEND=torch ./run.sh      # force the old path
DV_BACKEND=coreml ./run.sh     # the default anyway
```

### Tracking quality — the other 2×

EdgeTAM resizes every frame to a square before it looks at it, and that square is
a bigger lever than any backend. **Step 2 exposes it** as *fast / balanced / best*;
the clip itself is never touched, so masks and exports stay at the source
resolution whatever you pick.

| tracking quality | square | fps (CoreML) | fps (torch) | IoU vs 1024 fp32 | worst frame |
|---|---|---|---|---|---|
| best **(default)** | 1024 | 13.9 | 9.4 | 0.9968 | 0.958 |
| balanced | 768 | 20.9 | 15.4 | 0.9668 | 0.936 |
| fast | 512 | 27.0 | 27.1 | 0.9395 | 0.894 |

(One interleaved session, three rounds, so the 1024 row reads a little slower
than the backend table above — same code, hotter machine.)

Those IoU numbers look alarming and mostly are not, because a dot render throws
away most of what they measure. `bench/res_compare.py` tracks the reference clip
at all three sizes, renders each through the real Dots renderer with the Sage
palette, and writes `bench/res_compare.mp4` (three-up) and
`bench/res_compare_sheet.png` (4 frames × 3 sizes, 1:1 crops). Looking at them:

* **On a typical frame, 768 is indistinguishable from 1024.** Silhouette, limb
  separation, stray dots — the same picture.
* **On the worst frames it is not.** On the three frames where the two masks
  disagree most (148, 85, 149 — the landing, where the subject folds and limbs
  touch), 1024 keeps a clean sage channel between two body parts that 768
  narrows and 512 nearly closes. At 3× zoom it is obvious; at 1:1, side by side,
  it is visible if you know where to look.
* **512 is distinguishable at 1:1 on ordinary frames**: the silhouette is
  consistently fatter, thin limbs go blobby, and the edge bleeds a few dots.
* **No size flickers more than any other.** Mean frame-to-frame change in mask
  area is 1.45 % / 1.41 % / 1.48 % at 1024 / 768 / 512. Lower resolution costs
  outline accuracy, not temporal stability.

So the default stays **best (1024)**. 768 is the right pick for a long clip or a
subject that never gets tangled, and it is nearly 21 fps.

### What did not work

| tried | result |
|---|---|
| `torch.compile(track_step)` | 8.7 fps median, and ~30 s of inductor on the first track |
| `torch.compile(forward_image)` | the only compile placement that helps at all; still below CoreML, and the first track in a process stalls ~20 s |
| `channels_last` on the encoder + memory encoder | inside the noise |
| real-arithmetic RoPE in PyTorch (`bench/fastpath.py`) | inside the noise — the complex path is not the MPS bottleneck it looked like |
| batching the image encoder 4 frames at a time (`--backend torch-pf4`) | inside the noise |
| `expand` instead of `repeat` for the FPN position encodings | inside the noise |
| `num_maskmem` 7→3 and `max_obj_ptrs_in_encoder` 16→4 | ~1.2× at best, IoU mean 0.9855 / worst frame 0.9495 — a real quality cost for a small win |
| `image_size` 768 | 1.5×, but IoU mean **0.9676** — fails the gate |
| `image_size` 512 | 3×, but IoU mean **0.9393**, worst frame 0.894 — fails the gate |
| `bfloat16` | 2× worse; the RepViT encoder falls back |
| CoreML `CompiledMLModel` instead of `MLModel` | indistinguishable end to end |
| compiling the SAM heads on top of CoreML | 13.8 vs 15.0 fps — worse |
| sam3.cpp (ggml/Metal EdgeTAM) | its own README benchmarks EdgeTAM at **0.4 s/frame on an M4 Pro / 24 GB** — the same machine — i.e. 2.5 fps, a third of the plain torch path. Not tried locally. |
| MLX | no EdgeTAM port exists in Python. The SAM 2.1 hiera-small MLX video predictor was measured on this clip at **2.0 fps**. Adapting `mlx-sam` to EdgeTAM (RepViT trunk, 2-D spatial perceiver, split cross-attention keys) is a week, not a day. |
| ONNX Runtime + CoreML EP | no published EdgeTAM ONNX export includes memory attention — `onnx-community/EdgeTAM-ONNX`, the AXERA and Qualcomm AI Hub exports are all image-path only, and Qualcomm's own driver runs the memory path in eager PyTorch. The one complete export (`Rushour0/websam`) publishes no weights. More than a day of work. |
| HF `transformers.EdgeTamVideoModel` | exists since transformers 4.57, but it is the same PyTorch graph on the same MPS device — the ceiling is the `torch` row above, not the CoreML one. Not tried. |

## Limits

* **macOS + Apple Silicon.** Tracking is CoreML + MPS. `DV_BACKEND=` picks
  `coreml` (default) / `torch-compiled` / `torch-half` / `torch` / `torch-fp32`;
  an unavailable backend falls through to the next one, so a machine without
  coremltools still tracks. `DV_DEVICE=cpu` works, far slower. No CUDA extension.
* **The C library is required for error diffusion and Riemersma.** `setup.sh` builds
  it. The pure-Python fallback is ~500× slower and Riemersma has no fallback at all.
* **Error diffusion and Riemersma flicker on video.** That is inherent, not a bug —
  the UI marks them. Use dots / blue noise / Bayer / halftone for stable motion.
* **Short clips.** 720p / 30 fps, 300 frames / 10 s by default.
* **One track at a time.** A process-wide lock serialises EdgeTAM; a second request
  gets a 409.
* **Objects cost time.** The fps in the quality chips is for one subject; every
  extra object is another batch row through the memory attention and the SAM
  heads. Track times swing up to 1.7× run to run under sustained load (thermals),
  which is why `bench/bench.py` interleaves.
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
