# `web/` — the page

This directory is the whole product. Copy it to any static host and it works:
drop a still and dither it, drop a clip and dither every frame, or point at
something and have only that dithered — cut out of a photograph on the spot,
followed by EdgeTAM through a clip.
There is no build step, no bundler, no framework and nothing is fetched from a
CDN at run time.

```
index.html        the page: two views, Studio and Sequence
app.js            Studio: source -> subjects -> look -> palette -> export
                  Sequence: a strip of dot clouds and the joins between them.
                  A strip item is a live reference — its source plus its own
                  copy of the look — and its dots are re-derived on demand and
                  cached per (item, look).
dither.js         the dithering engine (server/dither.py mirrors it exactly)
canvas.js         the CANVAS: aspect-ratio presets, the one affine map from
                  source pixels to output pixels, the gaussian that turns a
                  per-frame mask centroid into a camera move, and the mask
                  warp. No DOM, no engine — the export ships the map it
                  produces to server/render.py, which only applies it
polish.js         mask polish — motion-aware temporal smoothing of the tracker's
                  masks (server/polish.py mirrors it exactly)
style.css
engines/
  index.js        which engine to run, and the /api/meta probe that decides
  browser.js      everything in the tab: decode, track, dither, encode.
                  `snapshot()` hands out a detached handle on the open clip,
                  which is what lets a sequence item outlive it
  encode.js       how frames become a file: WebCodecs VideoEncoder with the
                  timestamps written down (i * 1e6 / fps), muxed to WebM or
                  MP4. This is why an export is the clip's own length however
                  long the render took
  decode.js       how a clip becomes frames: WebCodecs in a Worker, WebCodecs
                  on the main thread, or the original <video> seek loop, picked
                  by feature detection and verified by trying. All three
                  produce the same grid — frame i is the picture at
                  t0 + (i + 0.5)/fps — and the same JPEG bytes
  remote.js       the REST API in server/ — local or rented, same client.
                  `snapshot()` is the same idea: the two job routes with the
                  id already bound
track.js          EdgeTAM's tracking loop in JS over the ONNX graphs
workers/
  decode-worker.js  a module Worker: demux + VideoDecoder + JPEG, so a decode
                    never touches the main thread
  decode-core.js    the decode itself, importable from the worker OR the page
  demux-mp4.js      a minimal MP4/MOV sample table reader — ordinary moov+stbl
                    and fragmented moof+trun, edit lists included. Written
                    here; nothing vendored
  demux-webm.js     the same for WebM/Matroska, including the unknown-size
                    Segments and Clusters MediaRecorder writes
vendor/
  gifenc.js       a GIF89a/LZW encoder, ~220 lines, no dependencies
  webm-muxer.js   Vanilagy's WebM/Matroska muxer, MIT, vendored verbatim
  mp4-muxer.js    the same for MP4. The pair is what turns the browser's own
                  VideoEncoder output into a file; see NOTICE. Nothing is
                  fetched at run time and they are imported lazily, so a page
                  that only dithers stills never loads either
player/
  dither-player.js  plays a .dots.gz on a canvas: the codec, the four
                    transitions, the sequence builder and the player, one file,
                    no dependencies. Carries the format spec. A join thins a
                    cloud over 8,000 dots down to particles for the flight and
                    hands the rest back in place at the two ends, so a pixel
                    dither mode morphs like a swarm.
  dither-player.mjs the same thing as ES module exports
  demo.html         a page around it — drop a file, scrub, recolour, resize
bluenoise.json    the server's default 64x64 threshold tile, so both engines
                  start from the same noise
models/           the ONNX graphs                        (weights NOT committed)
ort/              the onnxruntime-web build              (NOT committed)
track-probe.html  a bench page: per-stage timings and IoU against the server
```

## Deploying it

Nothing to build.

```sh
# GitHub Pages
git subtree push --prefix web origin gh-pages

# Cloudflare Pages / Netlify / anything
#   build command:      (none)
#   publish directory:  web
```

Every asset path in `index.html` is relative, so a subdirectory deployment
(`https://you.github.io/dither-studio/`) works without configuration.

`player/` is independent of the rest: `dither-player.js` is a single file with no
imports, so a `.dots.gz` can be played on any page anywhere by dropping it in.

**Except for the models.** `models/*.onnx` (~130 MB) and `ort/` (~39 MB) are
deliberately not in git — see below. Without them the page still does stills and
whole-frame clips; subject tracking says so plainly in step 2 instead of failing
on the Track button.

### One header worth setting

The page runs fine as plain static files. If your host lets you set headers,
these two turn on the multi-threaded WASM fallback for visitors whose browser
has no WebGPU:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

They are not required. WebGPU — the fast path, and what a current Chrome, Edge
or Safari 26 uses — needs neither. Without them the page detects that it is not
cross-origin isolated and asks onnxruntime for **one** WASM thread rather than
eight: eight would not be a slower fast path, it would be a failed init, because
threads need `SharedArrayBuffer` and that needs these headers. One thread tracks
at ~0.5 fps, which is a still and a handful of frames, not a clip.

**Browsers.** Fastest in Chrome and Safari; Firefox works but is slower. All
three write WebM and MP4 through `VideoEncoder` and all three produce a file the
clip's own length. Firefox's *alpha* export can come up a frame or two short —
its `requestFrame()` queues the grab for the next paint, so two grabs inside one
paint collapse — and it says so when it does. See *Browsers* in the root README
for the measured table.

## The models

`models/manifest.json` and `models/consts.bin` **are** committed: they are small,
and they are what tells the page what the model set can do (`image_size`, the
memory-bank shapes, `has_mask_prompt`, and `tiers` — which tracker squares this
deployment carries). The weights are not.

Three squares, three directories:

```
models/          768 px, the default and the only one a first visit downloads
models/512/      512 px, "fast"
models/1024/    1024 px, "best"
```

Each has its own `manifest.json` and `consts.bin`, both committed, and its own
graphs, which are not. The page reads `tiers` out of the default manifest and
offers exactly those chips — it does not probe, because probing would mean a
404 per absent square in the console of every page that only has the default
set. If a square is listed and its files are missing, the track happens at
768 px and the line says so.

Two ways to get them.

**Export them yourself.** `./setup.sh` at the repo root does it, along with
everything else:

```sh
./setup.sh                      # clones EdgeTAM, fetches the checkpoint,
                                # exports ONNX -> web/models/, vendors web/ort/
DV_SKIP_WEB_MODELS=1 ./setup.sh # skip both, server-only install
```

Or just the ONNX half, if the rest is already built:

```sh
env/venv/bin/python onnxexport/export_onnx.py --image-size 768 \
  --tiers 512,768,1024                       # -> web/models/
env/venv/bin/python onnxexport/export_onnx.py --image-size 512 \
  --out web/models/512
env/venv/bin/python onnxexport/export_onnx.py --image-size 1024 \
  --out web/models/1024
```

`--tiers` only records a list in the default manifest; it exports nothing extra.
`DV_MODELS_TIERS=0 ./setup.sh` keeps the install to the default square.

**Download them.** For anyone who is not going to run PyTorch, the graphs are
attached to the [`models-v1.1`](https://github.com/kcvete/dither-studio/releases/tag/models-v1.1)
release. One command from the repo root, and no python at all:

```sh
./setup.sh --page-only          # models + onnxruntime-web, nothing else
DV_MODELS_TIERS=0 ./setup.sh --page-only   # just the default 768 px square
```

Or by hand:

```sh
B=https://github.com/kcvete/dither-studio/releases/download/models-v1.1
curl -fL $B/dither-studio-models-v1.1.tar.gz      | tar xz -C web/  # 768 px
curl -fL $B/dither-studio-models-v1.1-512.tar.gz  | tar xz -C web/  # 512 px
curl -fL $B/dither-studio-models-v1.1-1024.tar.gz | tar xz -C web/  # 1024 px
npm pack onnxruntime-web@1.27   # -> web/ort/, see setup.sh for which files
```

| tarball | | |
|---|---|---|
| `dither-studio-models-v1.1.tar.gz` | 53 MB | 768 px fp16 + both memory encoders — the default, and the only one a first visit needs |
| `dither-studio-models-v1.1-512.tar.gz` | 51 MB | 512 px fp16, into `models/512/` |
| `dither-studio-models-v1.1-1024.tar.gz` | 55 MB | 1024 px fp16, into `models/1024/` |
| `dither-studio-models-v1.1-fp32.tar.gz` | 89 MB | 768 px fp32, for a GPU with no `shader-f16` |

`SHA256SUMS` is on the release beside them. The fp32 set is the one the page
asks for when the adapter reports no `shader-f16`; if it is not deployed, that
machine runs fp16 on WASM instead and says so. `tools/build-models-release.sh`
is what builds all four out of a checkout that has exported them.

A full `./setup.sh` can pull the same tarball instead of exporting:

```sh
DV_MODELS=download ./setup.sh
```

The weights are derived from the EdgeTAM checkpoint and carry its Apache-2.0
licence; `ort/` is onnxruntime-web, MIT. See `NOTICE` at the repo root.

### What is in there

Five graphs plus a small constants blob. fp16 and fp32 builds of each; the page
loads fp16 by default.

| file | fp16 | what it is |
|---|---|---|
| `encoder.fp16.onnx` | 10.0 MB | RepViT trunk + FPN neck |
| `memattn.fp16.onnx` | 9.3 MB | memory attention, fixed length + additive key mask |
| `heads.fp16.onnx` | 10.4 MB | SAM prompt encoder + mask decoder, tracking frames |
| `heads_prompt.fp16.onnx` | 10.3 MB | the same with a variable prompt-point count |
| `heads_mask.fp16.onnx` | 8.6 MB | a drawn shape as a mask prompt |
| `memenc.f16in.onnx` | 6.7 MB | memory encoder + spatial perceiver (computes in fp32) |
| `consts.bin` + `manifest.json` | 4 KB | `no_mem_embed`, the temporal encodings, shapes |

**55.3 MB** for the fp16 set, cached by the browser after the first load.
`docs/track-web.md` explains why each one is shaped the way it is.

### One 404 is expected

On a host with no Dither Studio server, the page's engine probe requests
`/api/meta`, gets a 404 and falls back to the browser engine — which is the
whole design. Browsers log every failed request, so devtools shows one 404 line
on load. It is not a bug and there is no way to suppress it: `fetch` cannot ask
the browser not to report a failure.

(Serving the page from `localhost` adds a second line — the probe also tries
`127.0.0.1:8765`, the default server port, so `./run.sh` on another port is still
found. A real deployment never sees that one.)

## The two engines

The page picks one on load: `GET /api/meta` against its own origin, with a short
timeout. An answer means a Dither Studio server is there and the remote engine
is used; anything else — 404, timeout, CORS, a page on GitHub Pages — and it runs
in the tab. The chip in the header names the winner and switches it by hand:
**Browser**, **Local server**, or **Custom URL** with an optional API key.

|  | browser | server |
|---|---|---|
| tracking | 12.4 fps (WebGPU fp16, M4 Pro) | 20.9 fps (CoreML, same Mac, same 768 px) |
| tracker resolutions | 768 only — one exported set | 512 / 768 / 1024 |
| multiple subjects | one pass each, so N subjects cost N x | one batched propagate pass |
| video export | WebM (VP9) **or H.264 MP4**, via WebCodecs `VideoEncoder` + a vendored muxer | H.264 MP4 via ffmpeg |
| the matched cut | the decoded frames back through the same encoder, same container as the render | `jobs/<id>/frames/*.jpg` re-encoded, MP4 (WebM beside a WebM render) |
| a trim after the track | a window on the frames and mask logits already in memory — nothing re-decoded | `frame_in`/`frame_out` on `/render`, `/original` and `/dots`, a slice of the frames and masks already on disk |
| frames | never leave the tab | uploaded, decoded to JPEG under `jobs/` |
| still export | PNG (RGBA when you ask for a transparent background), in the tab | the same, in the tab |
| subject in a still | one frame through `encoder` + `heads_prompt`, nothing uploaded | `POST /api/upload_image`, then one `/preview` per click |

### The export, and the length of what comes out

**The file is `nFrames / fps` seconds long, always.** It did not use to be.
The export was `MediaRecorder` over a canvas capture stream, and a recorder
stamps a frame with the moment it *arrives* — so the file's time base was the
wall clock of the render, and a 3.0 s clip whose dots took 84 s to draw came out
as an 84 s file. The progress line even said so, which made it a documented
defect rather than a hidden one.

`VideoEncoder` takes the timestamp as an argument (`engines/encode.js`): frame
*i* is stamped `i × 1e6 / fps` µs and carries `1e6 / fps` of duration, so how
long the render took cannot reach the output. `VideoEncoder.isConfigSupported`
is probed at the size the export will use — VP9 → VP8 → AV1 for WebM, H.264 →
HEVC → AV1 for MP4 — and the first yes wins; a container this browser cannot
write is greyed in the format chips with that reason. The muxers are Vanilagy's
`webm-muxer` and `mp4-muxer`, MIT, vendored verbatim into `vendor/` (see
NOTICE). About 130 KB for the pair, and the reason "writing H.264 in the tab
needs ~32 MB of ffmpeg.wasm" was never true: the *encoder* was in the platform,
and what was missing was a box writer.

Two things still go through `MediaRecorder`: the **alpha** WebM, because a WebM
alpha plane is a second bitstream in a Matroska `BlockAdditional` and
`VideoEncoder` will not produce one, and any browser with no `VideoEncoder`.
Those render every frame first, into a store of lossless PNGs, and then
**replay** the finished frames to the recorder on the clip's own clock — a
decoded PNG is microseconds against a dither's tens of milliseconds, so the two
clocks are the same clock. Dithered output is two to four flat colours, so the
store is tens of kilobytes a frame; past 256 MB the replay is chunked and the
recorder paused between runs.

`?slowrender=100` (or `window.DV_slowRender(100)`) sleeps 100 ms after every
frame the export dithers and changes nothing else. It exists so the property
above can be asserted rather than believed: `verify-web.mjs`'s `exportTiming`
flow throttles the render to ~4× slower than real time and demuxes what comes
out. One cosmetic caveat — `webm-muxer` writes the Matroska `Duration` as the
last timestamp, one frame interval short of the playable length, so ffprobe
reports 1.966 for a 60-frame 30 fps WebM that plays every one of them. MP4 is
exact.

## Hacking on it

```sh
# any static server will do; python's is enough
python3 -m http.server -d web 8080
```

Two things to know:

* `dither.js` is a classic script, not a module. Its top-level `const Dither`
  lands in the global lexical scope, which module scripts can see — that is why
  `index.html` loads it with a plain `<script>` before `app.js`. It is also
  `require()`-able from Node, which is how `server/parity.mjs` checks it against
  the Python engine byte for byte.
* `import()` resolves against the importing module's URL, `fetch()` against the
  document's. `engines/browser.js` lives one directory deeper than the page, so
  it resolves the onnxruntime path against `document.baseURI` explicitly. Get
  this wrong and the runtime 404s at `engines/ort/`.
