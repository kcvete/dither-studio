# Changelog

All notable changes to Dither Studio are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format, aiming at
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). **No versioned
release has been cut yet** — there are no tags, so everything built so far sits
under Unreleased, grouped by what it touches rather than by date. The first
tagged release will close this section.

## [Unreleased]

### Engines and tracking

- Two engines behind one page. On load, `web/engines/index.js` probes
  `GET /api/meta` with a 1.5 s timeout: an answer means a server is there and it
  wins; anything else — 404, timeout, CORS, GitHub Pages — and everything runs in
  the tab. A header chip says which engine is live and switches between
  **Browser**, **Local server** and a **Custom URL** with an optional API key.
- EdgeTAM subject tracking fully in the browser over ONNX graphs on WebGPU:
  12.4 fps at fp16, IoU 0.95 against the server, ~47 MB downloaded. The
  serverless path is the product, not a degraded mode.
- The server tracker got ~1.6x faster by moving EdgeTAM's three heaviest stages
  (memory attention, image encoder, memory encoder + spatial perceiver) onto
  CoreML, with PyTorch kept as the fallback for any tensor that does not match an
  exported shape. A tracking-quality setting exposes the trade.
- Prompting: click points, boxes, lasso and polygon mask prompts, with a
  first-frame preview before committing to a track; 768 px is the default.
- Per-subject prompt frames — prompt a thing on the frame it walks into, and the
  tracker walks it backwards out of that frame until the object score goes
  negative. Subjects that arrive mid-clip work on both engines; each gets its own
  memory bank on the server.
- One frame is enough: single-image segmentation on both engines, ~0.09 s at
  768 px, no propagation and no button — the outline is re-cut after every click.
- Mask polish: motion-aware temporal smoothing that steadies a mask without
  smearing a moving subject, mirrored byte-for-byte in JS and Python.

### Dither and look

- The dithering engine: 14 error-diffusion kernels (serpentine on or off),
  ordered and threshold modes, 18 palettes, tone controls and a dot mode — for
  stills, whole clips and tracked subjects alike. `web/dither.js` and
  `server/dither.py` are pixel-identical by construction, and `server/parity.py`
  is the gate that keeps them that way.
- A before/after wipe slider over the live preview; dots on a still, and a
  subject cut out of one as a transparent PNG.
- The canvas: aspect-ratio presets, so 16:9 goes in and 9:16 comes out — one
  affine map from source to output pixels, an auto-reframe that turns per-frame
  mask centroids into a camera move, and a hand-nudge over it.

### Sequences and the dot-data player

- The dots as data: a `.dots.gz` document format, a dependency-free player that
  carries the format spec (`web/player/dither-player.js`), and morphs between
  subjects.
- The sequence view got a room of its own — a strip of dot clouds, drag to
  reorder, with morph, scatter and density-fade joins between items. Every item
  wears its own look (mode, palette, cell size, gamma, polish), and changing one
  leaves the others hashing identically. An item defaults to the whole clip
  rather than an arbitrary 45 frames.
- Particles for the flight: a join thins a cloud over 8,000 dots down to
  particles mid-morph and hands the rest back in place at both ends, so a pixel
  dither mode morphs like a swarm.

### Source, trim and capture

- Record from the camera — a photo or a clip — then cut the bit you meant. No cap
  on clip length on either engine: a long clip is quoted honestly, in time and
  disk, and then taken.
- Trim after the track without losing the track: narrowing the range fires no
  re-upload, no re-extract and no re-track; widening it offers the extra frames
  and only does the work if you take the offer.

### Export

- Five containers, not one: H.264 MP4, VP9 WebM, GIF (a ~220-line home-grown
  encoder), an alpha variant and ProRes 4444.
- Export the clip beside its dither — a matched cut of the original, the same
  frames, rate and shape, named alongside the dithered file. `.dots.gz` export
  from both the studio and the sequence view.

### UI and UX

- A desktop and mobile redesign, specified first in `docs/` and then built:
  humanized vocabulary, look presets, save-as chips and a share sheet.
- Mobile shell: canvas-first layout, a bottom sheet that is sized rather than
  translated so sticky CTAs pin, and a tab bar.
- First-run demo — the landing page is already running something. Instant mask on
  clips, touch-friendly polarity, and tracking presented as playback rather than
  as a progress bar.
- Smaller passes: the frame block promoted under Style, the compare control
  labelled, sequence empty-state copy, a tidier mobile transport, an engine chip
  that stops truncating.

### Server and storage

- Restructured for open source: `web/` is the deployable page, `server/` the
  optional accelerator, with `coreml/`, `onnxexport/` and `bench/` beside them.
  `setup.sh` became the single bootstrap — venv, pinned EdgeTAM checkout,
  checkpoint, the C dither library, the CoreML and ONNX graphs, and a vendored
  onnxruntime-web.
- The scratch directory stops growing forever: `jobs/` is swept on an age limit
  and a size budget (14 days / 2 GB by default) with a 48 h grace window for work
  in flight, plus a `POST /api/gc/run` hook and hard-link-aware accounting.
- `DV_API_KEY` gates every `/api/*` route behind a bearer token when set, leaving
  the page itself open; `DV_CORS_ORIGINS` narrows CORS.

### Verification and documentation

- `verify.mjs` and `verify-web.mjs`: two headless end-to-end suites over the
  server engine and the browser engine, against a real server, a real EdgeTAM run
  and real ffmpeg. No mocks.
- `server/parity.py`: the kernel gate (110 cases, `GATE=1` repeats them through a
  subject mask), the compose gate and the polish gate, all byte-for-byte.
  `server/jobsgc_check.py` checks the janitor's policy against fabricated job
  trees in a second, with no server at all.
- Open-source documentation — README, `web/README.md`, LICENSE (Apache-2.0,
  matching EdgeTAM) and NOTICE — with every performance number measured rather
  than quoted.

### Fixed

- The Track CTA now follows the subject count instead of going stale, and stale
  single-frame predictions are dropped. Never snap a camera frame that has not
  been delivered yet; camera checks no longer assume an environment.
- The playback loop dies with its source rather than outliving it, mid-track
  streaming no longer 404-probes for frames that do not exist yet, and the paths
  and framing left behind by the open-source restructure were fixed.
