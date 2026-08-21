# Contributing to Dither Studio

No framework, no bundler, no build step — getting to a running copy is fast. The
tests are real, which is the part worth reading before you send a patch.

## Getting a copy running

### 1. The page, on its own (zero dependencies)

`web/` is the whole product. It needs nothing installed:

```sh
python3 -m http.server -d web 8080
open http://127.0.0.1:8080/
```

Everything in the tab: decode, dither, track, encode. Subject **tracking** also
needs the ONNX graphs in `web/models/` and the onnxruntime-web build in
`web/ort/` — neither is in git (~170 MB); `./setup.sh` below produces both.
Without them the page still does stills and whole-frame clips, and says so
plainly in step 2 rather than failing on a button. Editing `dither.js`,
`canvas.js`, `polish.js`, `app.js`, `style.css`, the sequence view or the
`.dots.gz` player? This is the only path you need.

### 2. The full environment (`./setup.sh`)

Apple Silicon Mac. Builds the optional local accelerator and the model files:

```sh
./setup.sh        # ~330 MB, a few minutes the first time, a no-op after
./run.sh          # setup, then server on 127.0.0.1:8765, then opens the page
```

`setup.sh` creates `env/venv` with python3.13, clones EdgeTAM into `env/EdgeTAM`
at a pinned commit, downloads the `edgetam.pt` checkpoint, exports the CoreML
**and** ONNX graphs, vendors onnxruntime-web 1.27 into `web/ort/`, and compiles
`env/libcdither.dylib` from `server/cdither.c`. It is idempotent.
`DV_SKIP_WEB_MODELS=1` skips the browser-engine export, `DV_PORT=` moves the
server, `DV_NO_OPEN=1` skips launching the browser. Nothing under `env/` or
`jobs/` is committed, and **no weights ever get committed**.

## Running the suites

| suite | command | where it runs |
|---|---|---|
| engine parity | `env/venv/bin/python server/parity.py` (and `GATE=1 …`) | CI **and** locally |
| jobs/ janitor | `env/venv/bin/python server/jobsgc_check.py` | CI **and** locally |
| syntax + codec round-trip | `npm test` (`tests/syntax-check.mjs`, `tests/player-roundtrip.mjs`) | CI **and** locally |
| end to end, server engine | `node verify.mjs` | **local only** |
| end to end, browser engine | `node verify-web.mjs` | **local only** |

**`parity.py`** is the pixel-exact gate between `web/dither.js` and
`server/dither.py`: 110 kernel cases (every mode, all 14 error-diffusion kernels
with serpentine on and off, three palettes, the tone controls), 15 compose cases
and 27 polish cases; `GATE=1` repeats the kernel set through a subject mask. It
exits nonzero on any mismatch and prints a JSON report on stdout. Needs numpy,
Pillow and `node` on `PATH` — it shells out to `server/parity.mjs` for the
JavaScript side.

**`jobsgc_check.py`** is the `jobs/` garbage-collector suite — age limit, budget
eviction oldest-first, the 48 h grace window, hard-link accounting — against
fabricated temp directories. Stdlib only, about a second, prints `all green`.

**`verify.mjs` / `verify-web.mjs`** are the end-to-end suites: Playwright driving
the real page against a real server, a real EdgeTAM run and real ffmpeg. No mocks.
Deliberately **not** in CI — they need the weights, a GPU-ish Mac and minutes:

```sh
./run.sh &
node verify.mjs        # writes docs/verify-report.json
node verify-web.mjs    # writes docs/verify-web-report.json
```

`verify-web.mjs` needs a Chromium with WebGPU; it tries headless Chromium with
the WebGPU flags, then the `chrome` channel, then falls back to the WASM backend
over a shorter clip and says so in its report.

## What CI checks

CI runs on Ubuntu, and only what can honestly run there: `npm run check` (every
hand-written `.js`/`.mjs` parsed), the `.dots.gz` codec round-trip, both
`server/parity.py` gates and `server/jobsgc_check.py` — with
`env/libcdither.dylib` compiled from source in the job. Anything model-, GPU- or
ffmpeg-shaped stays local.

`package.json` names the checks (`check`, `test`, `parity`, `parity:gate`,
`jobsgc`, `verify`, `verify:web`) and has no dependencies. Its scripts call
`python3`; use the `env/venv/bin/python` forms above if your system Python has no
numpy and Pillow.

## Before you open a PR

1. `env/venv/bin/python server/parity.py && GATE=1 env/venv/bin/python server/parity.py`
2. `env/venv/bin/python server/jobsgc_check.py`
3. `npm test` — syntax over every `.js`/`.mjs`, plus the `.dots.gz` round-trip
4. Touched anything the browser runs? `node verify-web.mjs`, and paste what it
   printed. **Zero console errors is a hard gate, not an aspiration.**
5. Touched the server, the renderer or the job lifecycle? `node verify.mjs`.
6. UI changes: attach a screenshot.
7. Tracking-speed changes: attach a `bench/bench.py` run — interleaved, three
   rounds. A straight sequence of backends measures the machine's temperature,
   not your patch.

## House rules

**No new runtime dependencies.** The page has no framework, no bundler and
fetches **nothing from a CDN at run time**. The only third-party code the browser
build ships is onnxruntime-web (vendored by `setup.sh`, not committed) and a
~220-line home-grown GIF encoder in `web/vendor/gifenc.js`. A new dependency
needs an argument, not a preference.

**Keep the browser engine honest.** The free, serverless path is the product for
most people. If a feature only works with a server, say so in the UI rather than
hiding the button. And no mocks in the tests: a suite that can pass while the
tool is broken is worse than none.

**Parity is byte-for-byte.** `web/dither.js` and `server/dither.py` produce
identical pixels today, and so do `web/polish.js` and `server/polish.py`. If you
change one, change the other in the same commit and run both gates.

## Code style

Every file opens with a prose block comment explaining **why** it exists and what
it is responsible for — not what the next line does. Read a few
(`web/canvas.js`, `server/server.py`, `web/player/dither-player.js`) before
writing one; matching that register matters more here than any lint rule.
JavaScript is two-space indent, plain ES modules, no transpilation. Python is
stdlib-plus-numpy in style: readable over clever.

## Commit messages

The history uses **short, human sentences that say what changed and, where there
is one, the number that proves it** — sentence case, no trailing full stop:

```
Trim after the track without losing the track
Every item in the strip wears its own look
Track 1.6x faster: EdgeTAM's three heavy stages on CoreML, plus a tracking-quality setting
```

More recent commits put a short prefix in front of that same sentence when it
helps scanning. The three in use are `fix:`, `UX:` and `docs:`:

```
fix: playback loop dies with its source; mid-track streaming never 404-probes
UX: sheet sized not translated (sticky CTAs pin), mobile transport tidy
docs: UX specification for desktop + mobile redesign
```

This is **not** Conventional Commits — no `feat:`, no scopes, no
`BREAKING CHANGE:` footer, and the prefix is optional. Write the sentence first;
add a prefix only if it is a fix, a UX pass or docs.

## Worth an issue before code

A WebCodecs decode path (faster than the seek loop, but frame-accuracy needs
proving), 512 and 1024 px ONNX exports behind an opt-in download, and batching
multiple subjects into one browser tracking pass.
