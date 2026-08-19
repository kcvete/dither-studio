# EdgeTAM tracking benchmark

`bench/bench.py`, 1 subject, 150 frames, 1280×720 source, M4 Pro / 24 GB,
macOS 26.1, torch 2.13, coremltools 9.0.

* **fps** = frames ÷ propagate seconds. The GPU→CPU mask copy is inside;
  PNG writing is outside.
* **IoU** = binary-at-0.5 against `masks_edgetam/`, the torch fp32 MPS output,
  per frame. `IoU min` is the worst single frame of the clip.
* Run-to-run spread from thermals reaches 1.7×, so a straight sequence of
  backends measures the machine's temperature, not the code. Everything below
  is **one run of each backend per round, three rounds, interleaved**; `fps med`
  is the median of the three rounds.

## Backends at 1024 px

| backend | fps best | fps med | wall_s med | IoU mean | IoU min | peak MB |
|---|---|---|---|---|---|---|
| torch-fp32 | 7.99 | 7.93 | 23.2 | 1.0000 | 1.0000 | 2744 |
| torch *(the old default)* | 9.52 | 9.42 | 20.2 | 0.9984 | 0.9592 | 2749 |
| torch-half | 9.48 | 9.38 | 20.2 | 0.9963 | 0.9589 | 2749 |
| torch-cl *(channels_last)* | 9.41 | 9.27 | 20.4 | 0.9982 | 0.9589 | 2749 |
| torch-fast *(real-arithmetic RoPE)* | 9.56 | 9.35 | 20.4 | 0.9984 | 0.9592 | 2759 |
| torch-pf4 *(encoder batched 4 frames)* | 9.00 | 8.85 | 21.5 | 0.9961 | 0.9589 | 2742 |
| torch-lean *(num_maskmem 3, obj ptrs 4)* | 10.40 | 10.25 | 18.8 | 0.9855 | 0.9495 | 2750 |
| torch-compiled *(compile the image encoder, half)* | 11.85 | 11.83 | 15.4 | 0.9961 | 0.9589 | 2957 |
| torch-compiled-fp16 *(same, autocast weights)* | 11.71 | 11.63 | 15.6 | 0.9970 | 0.9586 | 2905 |
| coreml-nopf *(no encoder prefetch thread)* | 14.73 | 14.60 | 15.9 | 0.9968 | 0.9580 | 2816 |
| **coreml** *(shipped default)* | **15.57** | **15.40** | 14.6 | **0.9968** | **0.9580** | 2822 |

`torch.compile` needs a warm-up run in the same process; the compiled rows are
measured after it (`--skip-first`), and the first track in a fresh process pays
~20 s (image encoder) or ~30 s (whole track step).

## Tracker input resolution

Second interleaved session, three rounds, all five measured together. The clip
is always 1280×720 — this is only the square EdgeTAM resizes each frame to.

| backend | fps best | fps med | IoU mean | IoU min | peak MB |
|---|---|---|---|---|---|
| coreml @ 1024 | 14.73 | 13.90 | 0.9968 | 0.9580 | 2761 |
| coreml @ 768 | 22.51 | 20.86 | 0.9668 | 0.9361 | 1948 |
| coreml @ 512 | 27.55 | 27.03 | 0.9395 | 0.8938 | 1315 |
| torch @ 768 | 15.88 | 15.37 | 0.9676 | 0.9369 | 1902 |
| torch @ 512 | 27.69 | 27.05 | 0.9393 | 0.8941 | 1292 |

CoreML is worth 1.6× at 1024 and 1.36× at 768; at 512 the model is small enough
that it makes no difference at all. Silhouette flicker, measured as the mean
frame-to-frame change in mask area over the clip, is **1.45 % / 1.41 % / 1.48 %**
at 1024 / 768 / 512 — resolution does not make the dither boil.

## Per-stage, per-call

`bench/profile_stages.py` (wall clock with `torch.mps.synchronize()` around each
stage, so the numbers are GPU time, not queue time), 1024 px, 1 subject:

| stage | MPS ms/frame | share |
|---|---|---|
| memory attention | 42.9 | 38 % |
| image encoder | 39.8 | 35 % |
| memory encoder + spatial perceiver | 18.3 | 16 % |
| SAM heads | 10.0 | 9 % |

`bench/micro_coreml.py`, one exported graph, pre-made numpy inputs:

| graph | fp32 ALL | fp32 GPU | fp16 GPU | fp16 ANE |
|---|---|---|---|---|
| image encoder | 16.9 | 15.2 | — | 13.2 |
| memory attention | 32.4 | 26.0 | 18.2 | 33.1 |
| memory encoder | 4.8 | 4.8 | — | 35.7 |

The ANE is the fastest place for the image encoder and the wrong one: its fp16
accumulation drops frame 147 of the reference clip to IoU 0.893 where the same
graph on the GPU holds 0.958 (clip mean 0.993 → 0.998). Everything is pinned to
`CPU_AND_GPU`.
