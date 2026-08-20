# The browser tracker — EdgeTAM with no server

How `web/track.js` and `web/models/` work: the image encoder, memory attention,
SAM heads, mask prompt and memory encoder as five ONNX graphs on
onnxruntime-web's WebGPU backend, with sam2's memory-bank bookkeeping
reimplemented in JavaScript. Nothing is uploaded and nothing is fetched from a
CDN.

This started as a measuring instrument — *can Dither Studio ship as a static
site?* — and the answer turned out to be yes, so it is now the free tier and the
default engine anywhere there is no server. `web/track-probe.html` is what is
left of the instrument: a bench page that runs the graphs in isolation, prints
per-stage timings and scores an IoU against the server's masks.

## Building the graphs

The weights are large and are **not** committed. `./setup.sh` does all of this;
these are the individual steps.

```sh
# the five graphs (needs the EdgeTAM checkpoint under env/)
env/venv/bin/pip install onnx onnxruntime
env/venv/bin/python onnxexport/export_onnx.py --image-size 768   # -> web/models/

# the runtime
npm install onnxruntime-web@1.27
cp node_modules/onnxruntime-web/dist/ort.all.bundle.min.mjs web/ort/
cp node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded{,.jsep}.{mjs,wasm} web/ort/
```

`onnxexport/verify_loop.py` is the same loop in Python against the CPU execution
provider — the reference this port is checked against, and the place to debug a
numeric difference before blaming JS. `onnxexport/verify_mask_prompt.py` does
the same for the mask-prompt graph against torch's `add_new_mask`.

## Running the bench page

`web/track-probe.html` hard-codes a clip directory (`../parkour/`) holding
`frames/%04d.jpg`, `prompt.json` and `masks_edgetam/%04d.png` — the server's own
output for a clip, which `bench/bench.py` produces. It is a development harness,
not something a visitor is meant to find; serve `web/` with cross-origin
isolation (multi-threaded WASM needs `SharedArrayBuffer`; the WebGPU path does
not, but the comparison should be fair) and open `/track-probe.html`.

## What the five graphs are

| graph | in | out | precision |
|---|---|---|---|
| `encoder` | 1×3×768×768 image | `f0` 1×32×192², `f1` 1×64×96², `f2` 1×256×48² | fp16 |
| `memattn` | `f2`, memory 3648×1×64, its pos enc, additive key mask | conditioned 1×256×48² | fp16 |
| `heads` | conditioned features, `f0`, `f1`, prompt points | 4 masks 192², 4 IoUs, 4 pointers, object score | fp16 |
| `heads_mask` | `f2`, 1×1×768×768 binary mask | the same four, all four slots identical | fp16 |
| `memenc` | `f2`, chosen 192² mask | 512 memory latents + pos | fp32 |

`heads_prompt` is `heads` with a dynamic prompt-point count and the
`no_mem_embed` add folded in — the conditioning frame only, once per clip.

`heads_mask` is the conditioning frame for a **lasso or polygon** prompt, which
EdgeTAM takes as a *mask* prompt. `use_mask_input_as_output_without_sam: true`
in `edgetam.yaml` sends that down `_use_mask_as_output`, so the drawn mask *is*
the answer — ±10 logits downsampled 4×, IoU a dummy 1.0, object score
`20·any(mask>0) − 10` — and the SAM decoder runs only to produce the object
pointer from a dense mask embedding. All four token slots therefore hold the
same mask and the same pointer, so the caller's existing "pick token k" code
works with any k. Two consequences worth knowing:

* it takes **no `add_no_mem`**. `_track_step`'s mask branch never calls
  `_prepare_memory_conditioned_features`, so `directly_add_no_mem_embed` does
  not apply to a mask prompt at all — adding it moves the pointer by ~8e-2.
* it takes **no `f0`/`f1`**. They are arguments to the wrapper but are pruned
  out of the graph: only `output_upscaling` uses the high-res features, and
  this graph discards the decoder's masks.

`_use_mask_as_output` downsamples with `F.interpolate(antialias=True)`, which
does not export. At an exact 1/4 scale that op is a separable 8-tap triangle
filter renormalised at the edges, i.e. a fixed 8×8 conv with stride 4 and
pad 2 times a constant 192² normaliser — exact, not an approximation
(`manifest.checks.mask_down4_max_abs`, 7.2e-07).
`onnxexport/verify_mask_prompt.py` scores the whole graph against a real
`add_new_mask` call on a real frame.

Three choices differ from the CoreML split in `coreml/`, all because a browser
has no PyTorch to fall back to when a shape is unusual:

* **the memory attention is exported once, at the full memory length, with an
  additive key mask.** The bank holds 7 spatial blocks and 16 object pointers
  once it is warm, but frames 1..6 have fewer. CoreML handles those by falling
  back to torch; here the empty slots are zero-filled and masked to -1e4, which
  makes the softmax ignore them and reproduces the shorter attention exactly.
* **the SAM heads are a graph**, not a torch module. The stock prompt encoder
  writes `point_embedding[labels == -1] = 0.0`, which traces to
  `NonZero` + `ScatterND`; `NonZero` is not in the WebGPU operator set and the
  shape it produces is dynamic, so the whole decoder would fall to the CPU
  backend. `wrappers_onnx.HeadsGraph` re-derives it as arithmetic.
* **the memory encoder swallows the 192² → 768² mask upsample**, so the only
  tensor that comes back to JS per frame is four 192×192 mask candidates.

`encoder → memattn → heads` is wired with `preferredOutputLocation:
'gpu-buffer'`, so the 9.4 MB of feature maps per frame never leave the GPU.
That is also why `memenc.f16in.onnx` exists: the memory encoder computes in
fp32 (its latents *are* the memory bank), but in the fp16 build its `pix_feat`
arrives as the encoder's fp16 buffer, so it widens on the first op instead.

## Measured on an M4 Pro / 24 GB, Chrome, 150 frames of 1280×720

| | WebGPU fp16 | WebGPU fp32 | WASM fp16 (8 threads) |
|---|---|---|---|
| end-to-end | **12.4 fps** (80.7 ms/frame) | 9.9 fps (101.5 ms) | 2.05 fps (489 ms) |
| encoder | 24.5 ms | 28.6 ms | 175 ms |
| memory attention | 42.2 ms | 49.5 ms | 252 ms |
| heads | 7.4 ms | 12.3 ms | 23 ms |
| memory encoder | 14.6 ms | 13.7 ms | 58 ms |
| download | 46.7 MB | 85.7 MB | 46.6 MB |
| load + compile | 0.4–1.0 s | 0.4 s | 0.6 s |
| IoU vs the server's 1024px masks | 0.9535 | 0.9568 | — |

Per-stage rows are each graph run 20× in isolation with a CPU readback, which
is the only way to get an honest number out of an async backend; they sum to
more than the chained end-to-end figure, which is the point of chaining.

For scale, the server-side path on the same machine and the same clip is
20.9 fps (CoreML) / 15.4 fps (MPS) at the same 768 px.

The IoU reference is the 1024 px torch run, so some of the gap is resolution,
not the port: the server's *own* 768 px torch path scores 0.9668 against it, and
this export scores 0.9681 (fp32) / 0.9666 (fp16) when it is fed frames resized
the way torch resizes them. The browser's 0.9535 is the canvas resampler —
re-running the Python loop with a box filter instead of bilinear moves it to
0.9295 with the model untouched.
