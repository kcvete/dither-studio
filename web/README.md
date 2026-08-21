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

**Browsers.** Fastest in Chrome and Safari; Firefox works but is slower, and its
WebM export can come up a frame short on a slow render (it says so when it
does). See *Browsers* in the root README for the measured table.

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
| video export | WebM (VP9) via MediaRecorder | H.264 MP4 via ffmpeg |
| the matched cut | the decoded frames back through the same recorder, WebM | `jobs/<id>/frames/*.jpg` re-encoded, MP4 (WebM beside a WebM render) |
| a trim after the track | a window on the frames and mask logits already in memory — nothing re-decoded | `frame_in`/`frame_out` on `/render`, `/original` and `/dots`, a slice of the frames and masks already on disk |
| frames | never leave the tab | uploaded, decoded to JPEG under `jobs/` |
| still export | PNG (RGBA when you ask for a transparent background), in the tab | the same, in the tab |
| subject in a still | one frame through `encoder` + `heads_prompt`, nothing uploaded | `POST /api/upload_image`, then one `/preview` per click |

The WebM trade-off is the one worth knowing about. Writing H.264 in the tab
would mean shipping an encoder, and ffmpeg.wasm is ~32 MB that would have to be
vendored to keep the no-CDN rule — a bigger download than the tracker. WebM
plays everywhere except older Safari; if you need MP4, the local server is one
`./run.sh` away.

The export is also paced in real time, because MediaRecorder timestamps each
frame when the page hands it over. If a frame takes longer to dither than the
clip's own frame interval, the result plays slow, and the export line says so
rather than letting you find out in QuickTime. *Also save the original* is
handed over at whatever pace the dithered pass actually managed, so the pair
keeps one duration between them even when that happens.

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
