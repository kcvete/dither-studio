# Dither Studio

Turn a photograph into computational structure. One unit, one palette, one logic,
repeated until the picture is made of it.

That idea is having a moment. When Solvd rebranded around it, the reason the
identity held together was not the look — it was that
[Afternow built them a dither tool](https://www.linkedin.com/posts/filip-justic_illustrations-no-one-drew-solvd-inc-builds-activity-7495381583588339712-qOF2),
so every new asset came out of the same machine instead of somebody's hand. A
brand system needs that. So does anyone who wants more than one image.

Dither Studio is that tool, in the open, and it goes one step further: it does
**video**, and it can dither **one thing in the video**. Point at a person and
EdgeTAM follows them for the whole clip — they turn into dots and the background
stays where it is, or the other way round. Drop a still and it is a still
ditherer with fourteen error-diffusion kernels and eighteen palettes.

![tracked subjects, per-subject palettes](docs/c-mixed.png)

**It is free and it runs in your browser.** No account, no upload, no server —
the tracker, the dither engine and the video encoder are all in the tab. There
is also an optional local server that does the same work faster on an Apple
Silicon Mac, and the seam for a hosted one; the page picks whichever is there.

---

## Quickstart

### The page, on its own

```sh
python3 -m http.server -d web 8080     # or any static host
open http://127.0.0.1:8080/
```

That is the whole product. `web/` is self-contained and deploys to GitHub Pages
or Cloudflare Pages with no build step — see [`web/README.md`](web/README.md).
Subject tracking needs ~83 MB of model weights that are **not** in git; the page
says so plainly and still does stills and whole-frame clips without them.

### With the local accelerator

On an Apple Silicon Mac, `./run.sh` builds everything and opens the page against
a local server that tracks about **1.7x faster**:

```sh
./run.sh
```

`run.sh` is idempotent. It calls `setup.sh` — creates `env/venv`, clones
`env/EdgeTAM`, fetches the checkpoint, compiles `env/libcdither.dylib`, exports
the CoreML graphs *and* the ONNX graphs, vendors onnxruntime-web — then starts
`server/server.py` on `http://127.0.0.1:8765` and opens the browser. First run is
a few minutes and about 330 MB on disk; every run after is a no-op. If 8765 is
taken it walks forward to the next free port and says which one. `DV_PORT=`
overrides, `DV_NO_OPEN=1` skips the browser,
`DV_SKIP_WEB_MODELS=1 ./setup.sh` skips the browser-engine models.

The page served by that server is the same directory. There is no fork.

## Three tiers, one codebase

| | **Browser** | **Local server** | **Hosted** |
|---|---|---|---|
| what it is | `web/`, on any static host | `server/`, on your machine | `server/`, on a rented GPU |
| price | free | free | yours to set |
| tracking | 12.4 fps | 20.9 fps | ~150 fps on an A100 |
| your frames | never leave the tab | never leave the machine | uploaded |
| video out | WebM (VP9) | H.264 MP4 | H.264 MP4 |
| status | shipped | shipped | the seam is here, the billing is not |

The third column is a deployment, not a feature branch: the same `server.py`, on
a CUDA box, with `DV_API_KEY` set. See *Hosting a paid backend* below.

## Architecture

```
                    ┌──────────────────────── web/ ───────────────────────────┐
                    │  index.html · app.js · style.css                        │
   a file  ────────▶│                                                         │
                    │  engines/index.js ── GET /api/meta, 1.5 s timeout       │
                    │        │                                                │
                    │        ├── server answered? ──▶ engines/remote.js ──────┼──▶ ┌─── server/ ────┐
                    │        │                        {baseUrl, apiKey?}      │    │ server.py      │
                    │        │                                                │    │  ├ ffmpeg      │
                    │        └── nothing there?   ──▶ engines/browser.js      │    │  ├ EdgeTAM     │
                    │                                  │                      │    │  │  └ coreml/  │
                    │                                  ├ <video> ▶ canvas     │    │  ├ dither.py   │
                    │                                  ├ track.js ▶ models/   │    │  │  └ cdither.c│
                    │                                  │   (onnxruntime-web,  │    │  └ render.py   │
                    │                                  │    WebGPU)           │    │     └ ffmpeg   │
                    │                                  └ MediaRecorder ▶ WebM │    └────────────────┘
                    │                                                         │
                    │  dither.js ◀── the preview, and the browser export ──────┤
                    └─────────────────────────────────────────────────────────┘
                                          ▲                                          ▲
                                          └───────── same algorithm, byte for byte ──┘
                                                     (server/parity.py is the gate)
```

```
web/            the deployable page. engines/, dither.js, track.js, models/
server/         the optional accelerator. FastAPI + numpy + the C dither loop
coreml/         EdgeTAM -> CoreML: traceable wrappers, exporter, live-swap accel
onnxexport/     EdgeTAM -> ONNX: the five graphs the browser engine runs
bench/          tracking benchmarks, stage profiler, the resolution A/B
verify.mjs      headless end-to-end check, server engine
verify-web.mjs  headless end-to-end check, browser engine + the engine seam
env/            venv, EdgeTAM checkout, checkpoint, libcdither, CoreML  (ignored)
jobs/<id>/      source, frames/, masks/<obj>/, out.mp4                  (ignored)
docs/           verification screenshots, reports, the tracking test clip
```

## Which engine am I on?

A chip in the header always says, and always switches:

![the engine chip and its switcher](docs/w-engine-popover.png)

On load the page does one thing: `GET /api/meta` against its own origin with a
1.5 second timeout. An answer means a Dither Studio server is there, and it wins,
because on the machine that has one it is faster. Anything else — 404, timeout,
CORS, a page sitting on GitHub Pages — and everything runs in the tab. A choice
you make by hand is remembered and beats the probe.

**Browser · free** · **Local server** · **Custom URL** (with an optional API key,
for a backend you are paying for). Switching engines drops the loaded clip,
because the frames live inside whichever engine decoded them; the page says so
rather than silently re-uploading.

## What you can do

Drop **an image** or **a clip** (or paste one). The steps adapt to what you gave
it, and to which engine is live.

### 1 · Source
Images stay in the tab on either engine — they are never uploaded. Clips are
decoded to 720p / 30 fps, capped by the *max length* slider (10 s, 300 frames):
in the browser engine by a `<video>` seek loop into JPEG blobs, on the server by
ffmpeg into `jobs/<id>/frames/`. Both produce the same frame grid, so frame 42
is the same picture either way.

### 2 · Subjects — clips only
Two choices:

* **whole clip** — every pixel of every frame gets dithered.
* **track subjects** — scrub to any frame and prompt what you care about:
  click = keep this, shift-click = not this, drag = a box. `+ add subject` for
  another object (up to 6), each with its own palette. Press **Track** and
  EdgeTAM follows them forward *and* backward through the clip.

  **Each subject remembers its own frame.** A ball that flies into shot at frame
  80 does not exist on frame 0, so it cannot be prompted there. Scrub to where it
  appears, add a subject, and click it: the chip reads `#2 · 1pt+box @ 80`, and
  that is the frame that subject is conditioned on. Marks are drawn only on their
  own frame; a subject that lives elsewhere is dimmed, with a
  *"#1 prompted @ 0 — jump"* line that takes you back to it. Before its subject
  arrives, a mask is legitimately **empty** — no dots, nothing to erase, no
  lingering ghost. That is the right answer about a ball that is not in the shot.

  ![two subjects, two prompt frames](docs/w-entry-prompts-remote.png)

  **Prompt tool** — *point / box*, *lasso* or *polygon*. Clicks and a box are the
  fast path and the one the tracker likes best: it re-derives the outline itself
  on every frame. When that is not enough, draw the subject instead — freehand
  with the lasso, corner by corner with the polygon, shift to subtract a shape.
  The drawing is rasterised to a binary mask and sent as a **mask prompt**, on
  both engines: the browser one has a fifth ONNX graph (`heads_mask`) for exactly
  this, because EdgeTAM's mask path skips the memory attention entirely rather
  than sharing the click path.

  A subject uses one or the other, never both: EdgeTAM's `add_new_mask` drops the
  frame's point inputs and `add_new_points_or_box` drops its mask, so a subject
  with a drawn shape ignores its clicks. The chip says which it has.

  **Preview this frame** runs only the first-frame prediction — no propagation —
  and paints the mask the tracker would produce over your prompt. **0.13 s** on
  the server, **0.14 s** in the browser once the graphs are warm. It answers "is
  this click enough?" before you spend the whole clip on it. It covers only the
  subjects prompted on the frame you are looking at.

  **Tracking quality** — *fast (512) / balanced (768) / best (1024)*, with the
  measured fps on each chip — is the square EdgeTAM resizes every frame to. Your
  clip keeps its own resolution either way. The browser engine offers **768
  only**, because that is the one resolution it ships models for; exporting all
  three would triple an 83 MB download for a knob that mostly matters when you
  are waiting on a server.

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

The flicker labels are the honest part: threshold modes reuse one fixed field
every frame, so dots stay put and only switch on and off as tone changes. Error
diffusion recomputes a chaotic error field per frame, so it boils. Both are
offered; the chip carries a `≈` marker on video so you know which you picked.

Also here: dither strength, pixel size (chunky-pixel scale, box-downsample then
nearest-upscale), brightness / contrast / gamma / invert, and **reseed** for a
new noise field. For tracked clips, *background* chooses between a flat colour
and the dithered scene.

### 4 · Palette
18 presets — Black & White, Sage, Forest, Ember, Mist, Game Boy DMG, four
monochromes, CMYK, RGBY, Black White Red, Purple & Green, Blue & Yellow,
Commodore 64, 4 Greys, 8 Greys — plus **from image** (median-cut extraction from
the current frame) and a free colour editor. With tracked subjects you get one
palette per subject *plus* one for the background.

### 5 · Export
Stills render at full source resolution in the tab and download as PNG, on both
engines. Clips depend on the engine:

* **browser** — every frame is dithered into a canvas and fed to `MediaRecorder`
  through a capture stream, giving **WebM (VP9)**. Writing H.264 in the tab would
  mean vendoring ~32 MB of ffmpeg.wasm to keep the no-CDN rule — a bigger
  download than the tracker itself. The recorder timestamps each frame when the
  page hands it over, so the loop is paced to the clip's own frame interval; if a
  frame takes longer to dither than that, the file plays slow and the export line
  tells you so.
* **server** — numpy plus the C dither loop, encoded by ffmpeg to **H.264 MP4**
  at crf 18.

**Compare** (in the transport bar) drags a before/after divider across the frame,
and it keeps working while the clip plays.

## The two engines

### One dither engine, three implementations

The browser preview is not an approximation of the export — it is the same
algorithm. `server/parity.py` runs 110 cases (every mode, all 14 kernels with
serpentine on and off, three palettes, the tone controls) through the Python and
the JavaScript implementations and requires **byte-identical** output, then
repeats the whole set through a subject mask. The browser engine's export uses
the same `dither.js` the preview does, so it is identical by construction.

Getting there needed three fixes worth remembering, because each produced
*visible* pixel differences that error diffusion then amplified:

* **Float width.** JavaScript Numbers are f64 and only the `Float32Array` *store*
  rounds. The C loop uses `double` locals with float32 storage to match, and
  `tone_lut()` computes in float64 and casts once.
* **FMA contraction.** `cc -O3` fuses `a + b*c` into one instruction that rounds
  once where JS rounds twice, so the build uses `-ffp-contract=off`.
* **RNG.** Both sides use a portable integer hash (`hash01`) instead of
  `numpy.random` / `Math.random` for jitter and stray fields.

The one thing that is *not* shared is the blue-noise tile: the server generates
it with an FFT high-pass and numpy's RNG, which is not portable. So the server's
default seed-7 tile ships as `web/bluenoise.json` and both engines start from the
same field; **reseed** in the browser generates a fresh tile with the same
construction and a portable hash, which is a different realisation of the same
spectrum.

### One tracker, two ports

`coreml/` runs EdgeTAM's three heaviest stages as CoreML graphs with PyTorch
holding the rest. `onnxexport/` goes further and exports *five* graphs, because a
browser has no PyTorch to fall back to when a shape is unusual: the memory
attention is exported once at full memory length with an additive key mask, the
SAM heads become a graph with `NonZero` re-derived as arithmetic, and the mask
prompt gets its own graph. `web/track.js` reimplements sam2's memory-bank
bookkeeping in JS around them. `docs/track-web.md` is the full account.

They agree. Against the same 1024 px fp32 torch reference on the same clip, the
server's 768 px torch path scores IoU **0.9668** and the ONNX export scores
**0.9681** (fp32) / **0.9666** (fp16). In the actual browser it is 0.9535, and
the gap is the canvas resampler, not the port — feeding the Python loop a box
filter instead of bilinear moves it to 0.9295 with the model untouched.

### Performance

150-frame 1280×720 clip, one subject, 768 px tracker input, M4 Pro / 24 GB.

| | tracking | first-frame preview | video export |
|---|---|---|---|
| **browser** (WebGPU fp16) | **12.4 fps** (80.7 ms/frame) | 0.14 s | 29 fps → WebM |
| **browser** (WASM, 8 threads) | 2.05 fps | — | — |
| **local server** (CoreML) | **20.9 fps** | 0.13 s | 14.5 fps → MP4 |
| local server (torch MPS) | 15.4 fps | — | — |
| **A100** (EdgeTAM's own figure) | **~150 fps** | — | — |

The A100 row is EdgeTAM's published number, *"obtained with torch compile"* — not
measured here. It is in the table because it is the reason the third tier is
worth building: the same `server.py` on rented silicon is an order of magnitude
faster than the Mac under it.

Per stage in the browser, each graph run 20× in isolation with a CPU readback
(the only way to get an honest number out of an async backend):

| | WebGPU fp16 | WebGPU fp32 | WASM fp16 |
|---|---|---|---|
| image encoder | 24.5 ms | 28.6 ms | 175 ms |
| memory attention | 42.2 ms | 49.5 ms | 252 ms |
| SAM heads | 7.4 ms | 12.3 ms | 23 ms |
| memory encoder | 14.6 ms | 13.7 ms | 58 ms |

They sum to more than the 80.7 ms end-to-end figure, which is the point of
chaining: `encoder → memattn → heads` is wired with
`preferredOutputLocation: 'gpu-buffer'`, so 9.4 MB of feature maps per frame
never come back to JS.

**Download.** The page itself is ~200 KB. The first time you track something, the
browser engine pulls **83 MB** — 55.3 MB of fp16 ONNX graphs and 27.7 MB of
onnxruntime-web — and caches it. Nothing comes from a CDN.

**Multiple subjects.** The server batches every subject through one propagate
pass. The browser tracks them one at a time, because a `WebTracker` is a
single-object memory bank, so N subjects cost N × the time. Two subjects over
149 frames: 16.1 s on the server, 25.7 s in the browser.

## Hosting a paid backend

The seam is built; the billing is not, and this repository is not going to grow
it.

`server.py` reads one optional environment variable:

```sh
DV_API_KEY=$(openssl rand -hex 24) DV_PORT=8765 env/venv/bin/python server/server.py
```

With it set, every `/api/*` request must carry `Authorization: Bearer <key>` or
gets a 401. The page and the static assets stay open, because a browser cannot
put a header on the request that loads the HTML. `GET /api/meta` reports
`"auth": "bearer"` so a client knows before it tries. `DV_CORS_ORIGINS` narrows
CORS from the default `*`.

On the page: the engine chip → **Custom URL** → the base URL and the key. The
same `engines/remote.js` drives it; there is no separate client and no premium
code path.

A deployment sketch, unbuilt but not hand-wavy:

```
  Cloudflare / nginx            a CUDA box (A10G, L4, A100…)
  ┌──────────────────┐          ┌───────────────────────────────────┐
  │ TLS              │          │ DV_API_KEY=<per-customer>         │
  │ rate limit       │─────────▶│ DV_DEVICE=cuda DV_BACKEND=torch-  │
  │ key -> customer  │          │            compiled               │
  │ meter /track     │          │ server/server.py                  │
  └──────────────────┘          │ jobs/ on a scratch disk, GC'd     │
                                └───────────────────────────────────┘
```

What you would actually have to do:

* **CUDA.** `DV_DEVICE=cuda`. The CoreML backend is Apple-only and falls through
  to `torch-compiled` on its own; EdgeTAM's 150 fps figure *is* the compiled
  torch path, so that is the right backend there. The `_gpu_lock` in `server.py`
  serialises one track at a time per process — run one process per GPU behind
  the proxy rather than trying to share.
* **Jobs are never garbage collected.** ~13 MB per 150-frame clip, and they hold
  the customer's frames. A hosted deployment needs a reaper and a retention
  policy before it needs a payment form.
* **Metering.** Count tracked frames, not requests: `POST /track` returns
  immediately and the work is in the worker. `GET /status` already reports
  `done_frames`, `elapsed_s` and `image_size`.
* **The free tier does not get worse.** The browser engine is the product for
  most people, and it costs the operator nothing. The paid tier buys speed,
  larger clips and 1024 px tracking — not the feature list.

## Verification

Two headless suites, both against a real server, a real EdgeTAM run and real
ffmpeg. No mocks.

```sh
./run.sh &
node verify.mjs     http://127.0.0.1:8765 clip.mp4 still.jpg
node verify-web.mjs http://127.0.0.1:8765 clip.mp4 docs/entry-clip.mp4 still.jpg
env/venv/bin/python server/parity.py && GATE=1 env/venv/bin/python server/parity.py
```

`verify.mjs` drives the **server engine** through five flows: a still through
every algorithm, a whole-frame clip, two tracked subjects, one subject at a
non-default tracking quality, and a polygon mask prompt with a frame preview.

`verify-web.mjs` drives the **browser engine** and the seam between the two: the
auto probe and the manual switch, a still, a whole-frame clip, a tracked subject,
a polygon through the `heads_mask` graph, two subjects prompted on two different
frames — and the same two-frame test on the server engine, so the feature is
checked on both. It also starts a second server with `DV_API_KEY` set and checks
401 / 401 / 200. It picks a browser rather than assuming one: headless Chromium
with the WebGPU flags, then `channel:'chrome'`, then the WASM backend over a
shorter clip with the report saying so.

Latest run (M4 Pro, 24 GB, macOS 26.1, torch 2.13 / MPS + CoreML; headless
Chromium with a real WebGPU adapter), 150-frame 1280×720 clip:

| | result |
|---|---|
| engine parity | **110/110 byte-identical**, and 110/110 again through a mask |
| `verify.mjs` | 5 flows, **0 console errors** |
| `verify-web.mjs` | 8 flows, **56/56 assertions**, **0 console errors** |
| still: 14 kernels | **14 distinct** images, no two kernels alike |
| browser: clip decode | 150 frames in **4.9 s**, in the tab |
| browser: frame-0 preview | **0.14 s** once the graphs are warm (1.6 s including the load) |
| browser: track | 150/150 frames in 13.0 s (**11.5 fps**), WebGPU fp16 |
| browser: mask prompt | tracked from a polygon alone, non-empty on 150/150 frames |
| browser: dots preview | 49.5 fps · 774 dots |
| browser: export | 150 frames of VP9 WebM in 5.3 s, 1280×720, ffprobed |
| server: track, 2 subjects | 150/150 in 14.0 s (**10.7 fps**) end to end, CoreML |
| server: track, 1 subject @ 512 px | 150/150 in 10.4 s (**14.4 fps**), masks still 1280×720 |
| server: track from a polygon | mask-prompt vs box-prompt IoU **0.978 mean / 0.928 worst** |
| server: export | 150 frames of H.264 in 10.3 s, ffprobed |
| preview vs exported MP4 | **97.8 %** of pixels within 30 RGB units |
| `DV_API_KEY` | bare 401 · wrong key 401 · right key 200 · the page still 200 |

The remaining 2.2 % of the preview-vs-export comparison is not an engine
difference — it is the browser's JPEG decoder disagreeing with Pillow's by a
level or two, plus h264 quantisation on the way out. Feed both the same decoded
pixels and they agree exactly, which is what `parity.py` measures.

### Subjects that arrive mid-clip

`docs/entry-clip.mp4` is five seconds of a locked-off park shot
([Mixkit](https://mixkit.co/free-stock-video/view-of-a-park-while-a-girl-runs-across-4831/),
free licence) that a jogger runs into. She is **not in frame until frame 38**.
Both engines are given two subjects: a tree prompted on frame 0, and the jogger
prompted on frame 48, ten frames after she appears.

| | tree, prompted @ 0 | jogger, prompted @ 48 |
|---|---|---|
| server engine | non-empty on **149/149** frames | empty 0–37, **first mask on frame 38** |
| browser engine | non-empty on **149/149** frames | empty 0–38, **first mask on frame 39** |

Two subjects over 149 frames took **16.1 s** on the server and **29.5 s** in the
browser, which is the one-pass-per-subject cost showing up.

Neither was told when she arrives. The server finds frame 38 because SAM2's
`max_cond_frames_in_attn` is -1, so a conditioning frame in the *future*
participates in the memory attention from frame 0 onwards and the object score
simply stays negative until she is there. The browser gets to the same place from
the other direction, tracking her backwards out of frame 48 until she leaves.
One frame apart, from entirely separate code.

The renderer follows: at frame 10 the dot count is the tree's alone — **1346**,
with not one stray dot where she will be — and at frame 100 it is **1787**, the
tree plus her. Dots pop rather than fade, which is what a threshold field does
and what the aesthetic wants.

![frame 10, before she arrives](docs/w-entry-f10-remote.png)
![frame 100, both subjects](docs/w-entry-f100-remote.png)

One real bug fell out of writing that test. SAM2 consolidates *every* prompt
frame across *every* object before propagation begins, and on frame 48 the tree
had not been tracked yet — so it got the `NO_OBJ_SCORE` placeholder and vanished
for exactly one frame at 30 fps. `_fill_foreign_cond_holes` in `server.py` copies
the neighbouring frame, which is what the browser engine produces anyway.

## Licence

**Apache-2.0**, matching EdgeTAM, whose weights this cannot work without. MIT
would have been fine for the JavaScript on its own, but a two-licence repository
where the model half is Apache and the code half is MIT is a paperwork tax on
everyone downstream for no benefit. Apache-2.0 also carries an explicit patent
grant, which matters more than usual for something that ships model weights.

What that means in practice:

| | licence | committed here? |
|---|---|---|
| this code (`web/`, `server/`, `coreml/`, `onnxexport/`, `bench/`) | Apache-2.0 | yes |
| [EdgeTAM](https://github.com/facebookresearch/EdgeTAM) + its checkpoint | Apache-2.0 | no — `setup.sh` clones and downloads |
| the derived ONNX / CoreML graphs | Apache-2.0 (derived from the checkpoint) | no — regenerated or released separately |
| [onnxruntime-web](https://github.com/microsoft/onnxruntime) in `web/ort/` | MIT | no — `setup.sh` fetches it from npm |
| `docs/entry-clip.mp4` | Mixkit Free License | yes, **as a test fixture only** |

The Mixkit clip is in the repository because the tracking tests need a real
video where something enters the shot; it is not redistributable as a stock
asset and it is not part of the software. Everything else used while building
this was a test input and is not here. `NOTICE` has the full attributions, and
anyone bundling `web/ort/` must carry the MIT notice with it.

## Contributing

The bar is the same one the code holds itself to: **no mocks in the tests, and a
number in the commit message**. `verify.mjs`, `verify-web.mjs` and `parity.py`
run against a real server, a real EdgeTAM and real ffmpeg, and they should stay
that way — a suite that can pass while the tool is broken is worse than none.

Practically:

* Run both verifiers before opening a PR, and paste what they printed. Zero
  console errors is a hard gate, not an aspiration.
* Touching `dither.js` or `dither.py` means running
  `server/parity.py` **and** `GATE=1 server/parity.py`. They are byte-for-byte
  equal today and that is worth defending.
* Anything that changes tracking speed should come with a `bench/bench.py` run —
  interleaved, three rounds. A straight sequence of backends measures the
  machine's temperature, not your patch.
* Keep the browser engine honest. If a feature only works with a server, say so
  in the UI rather than hiding the button.
* Do not commit weights. `setup.sh` regenerates every binary in this repo.

Open questions worth an issue before code: a WebCodecs decode path (faster than
the seek loop, but frame-accuracy needs proving), 512 and 1024 ONNX exports
behind an opt-in download, and batching multiple subjects into one browser pass.

## How the server tracker got fast

The browser port is documented in `docs/track-web.md`. This is the
CoreML side, and the measurements behind the 20.9 fps above.


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
| **best · production** | 1024 | 13.9 | 9.4 | 0.9968 | 0.958 |
| **balanced · default** | 768 | 20.9 | 15.4 | 0.9668 | 0.936 |
| **fast · prototyping** | 512 | 27.0 | 27.1 | 0.9395 | 0.894 |

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

**The default is balanced (768)** — 20.9 fps, and on everything except a frame
where limbs touch it is the same picture. Drop to **fast (512)** while you are
still choosing an algorithm and a palette; go to **best (1024)** for the render
you keep, where the extra 7 seconds on a 5-second clip buys back the notches
between limbs. That is a judgement about a dot render, not about masks: by IoU
alone 768 fails the ≥ 0.98 bar the backends are held to (0.967), which is exactly
why the setting exists rather than a silently lowered default.

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

### The browser engine
* **WebGPU or nothing much.** Chrome, Edge and current Safari have it. Without it
  the page falls back to multi-threaded WASM at **2.05 fps** — usable for a short
  clip, not for a 300-frame one — and says so on the chip.
* **83 MB before the first track.** Cached afterwards, but it is a real cost on a
  first visit, and the weights are not in git (see `web/README.md`).
* **One tracker resolution.** 768 px only. 512 and 1024 would triple the
  download.
* **Subjects cost linearly.** One full pass each: N subjects, N × the time. The
  server batches them into one.
* **WebM, not MP4.** `MediaRecorder` gives VP9. Older Safari will not play it.
* **The export is paced in real time.** A frame that takes longer to dither than
  the clip's frame interval makes the file play slow; the export line says when
  that happened.
* **Memory.** A 150-frame 720p clip is ~15 MB of JPEG blobs plus ~22 MB of mask
  logits per subject. A 300-frame clip with six subjects is not a good idea in a
  tab.

### The server engine
* **macOS + Apple Silicon, for the fast path.** Tracking is CoreML + MPS.
  `DV_BACKEND=` picks `coreml` (default) / `torch-compiled` / `torch-half` /
  `torch` / `torch-fp32`; an unavailable backend falls through to the next one,
  so a machine without coremltools still tracks. `DV_DEVICE=cpu` works, far
  slower. `DV_DEVICE=cuda` is untested here but is the documented path for a
  hosted deployment.
* **The C library is required for error diffusion and Riemersma.** `setup.sh`
  builds it. The pure-Python fallback is ~500× slower and Riemersma has no
  fallback at all.
* **One track at a time.** A process-wide lock serialises EdgeTAM; a second
  request gets a 409.
* **Jobs are never garbage collected.** ~13 MB per 150-frame clip; delete by
  hand. A hosted deployment needs a reaper before it needs a payment form.
* **`DV_API_KEY` is authentication, not authorisation.** One key, all or
  nothing, no accounts, no rate limiting, no metering. Put those in front of it.

### Both
* **A drawn shape and clicks are exclusive.** Per frame and object EdgeTAM takes
  a mask prompt or points+box, never both — that is upstream's design, not a
  shortcut here. Draw the shape *or* click, and use **preview this frame** to
  find out which you need.
* **Error diffusion and Riemersma flicker on video.** That is inherent, not a bug
  — the UI marks them. Use dots / blue noise / Bayer / halftone for stable
  motion.
* **Short clips.** 720p / 30 fps, 300 frames / 10 s by default.
* **Objects cost time**, and track times swing up to 1.7× run to run under
  sustained load (thermals), which is why `bench/bench.py` interleaves.
* **Tracking quality is EdgeTAM's.** Fast motion, blur and occlusion make masks
  drift; the fix is a better prompt, not a renderer setting. A subject that
  leaves and re-enters is re-identified when EdgeTAM manages it and not when it
  does not — the jogger in `docs/entry-clip.mp4` survives a tree occlusion around
  frames 59–69, which is the model's doing, not this tool's.
* **Switching engines drops the clip.** The frames live inside whichever engine
  decoded them. The page says so rather than silently re-uploading.
* **Preview cost.** The preview dithers at full resolution on the main thread and
  caches 40 frames of decoded bitmaps. Heavy settings (small cell, many subjects,
  Riemersma) drop below the clip's own frame rate; the fps counter tells you.
* **Stills preview at 1600 px** on the long edge and re-render at native
  resolution only for the download, so a large photo stays responsive while you
  drag sliders.
* **Compare is preview-only** — not baked into the export.
* **No audio.** Video export is picture only.
* **EdgeTAM patch.** `setup.sh` rewrites two `.view(...)` calls to `.reshape(...)`
  in `sam2/modeling/perceiver.py`; upstream throws *"view size is not compatible
  with input tensor's size and stride"* as soon as more than one object is
  tracked.
