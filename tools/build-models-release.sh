#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the model tarballs for a GitHub release.
#
# The browser engine's graphs are ~450 MB across three tracker resolutions and
# two precisions. None of it is in git; it comes from a release, and this is
# what makes the files that go in one. It does not publish anything -- it
# writes dist/ and prints the `gh release create` line to run by hand.
#
#   ./tools/build-models-release.sh [version]      default: v1.1
#
# Six tarballs, all extracting into web/ so `tar xz -C web` is the install:
#
#   dither-studio-models-<v>.tar.gz            models/        768 px, fp16
#   dither-studio-models-<v>-512.tar.gz        models/512/    512 px, fp16
#   dither-studio-models-<v>-1024.tar.gz       models/1024/  1024 px, fp16
#   dither-studio-models-<v>-fp32.tar.gz       models/        768 px, fp32
#   dither-studio-models-<v>-512-fp32.tar.gz   models/512/    512 px, fp32
#   dither-studio-models-<v>-1024-fp32.tar.gz  models/1024/  1024 px, fp32
#
# The first is the only one a first visit downloads. The other squares load
# when their chip is picked.
#
# EVERY fp16 tarball is self-sufficient: it carries `memenc.onnx` as well as
# `memenc.f16in.onnx`, because the memory encoder computes in fp32 whatever the
# rest does and the WASM path wants the fp32-input build of it. That is what
# lets a deployment ship fp16 only — a GPU with no `shader-f16` falls to fp16
# on WASM and says so, and never asks for a graph that is not there.
#
# The -fp32 tarballs are the faster answer for that machine (WebGPU fp32 rather
# than WASM fp16, ~20% slower than fp16 instead of ~6x). They are separate
# downloads because they are 83 MB per square, and there is now one per square
# rather than one for 768 only: a deployment that installs the fp32 set has to
# be able to install it for every chip it offers, or picking a chip lands on
# exactly the half-installed tier this script exists to make impossible.
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V="${1:-v1.1}"
OUT="$HERE/dist"
M="$HERE/web/models"
rm -rf "$OUT"; mkdir -p "$OUT"

need() { [ -f "$1" ] || { echo "[release] missing $1 - run setup.sh / export_onnx.py first"; exit 1; }; }

# --- the default set: 768 px, fp16, plus the two memory encoders (the fp32 one
#     is what the WASM path uses, and it is small)
need "$M/manifest.json"; need "$M/encoder.fp16.onnx"; need "$M/memenc.f16in.onnx"
tar czf "$OUT/dither-studio-models-$V.tar.gz" -C "$HERE/web" \
  models/manifest.json models/consts.bin \
  models/encoder.fp16.onnx models/memattn.fp16.onnx models/heads.fp16.onnx \
  models/heads_prompt.fp16.onnx models/heads_mask.fp16.onnx \
  models/memenc.onnx models/memenc.f16in.onnx

# --- the other two squares
for S in 512 1024; do
  need "$M/$S/manifest.json"; need "$M/$S/encoder.fp16.onnx"
  tar czf "$OUT/dither-studio-models-$V-$S.tar.gz" -C "$HERE/web" \
    "models/$S/manifest.json" "models/$S/consts.bin" \
    "models/$S/encoder.fp16.onnx" "models/$S/memattn.fp16.onnx" \
    "models/$S/heads.fp16.onnx" "models/$S/heads_prompt.fp16.onnx" \
    "models/$S/heads_mask.fp16.onnx" \
    "models/$S/memenc.onnx" "models/$S/memenc.f16in.onnx"
done

# --- fp32: the faster fallback for a GPU without shader-f16, one per square.
#     memenc has no fp16 build, so the same models/memenc.onnx rides in both.
need "$M/encoder.onnx"
tar czf "$OUT/dither-studio-models-$V-fp32.tar.gz" -C "$HERE/web" \
  models/manifest.json models/consts.bin \
  models/encoder.onnx models/memattn.onnx models/heads.onnx \
  models/heads_prompt.onnx models/heads_mask.onnx models/memenc.onnx

for S in 512 1024; do
  need "$M/$S/encoder.onnx"
  tar czf "$OUT/dither-studio-models-$V-$S-fp32.tar.gz" -C "$HERE/web" \
    "models/$S/manifest.json" "models/$S/consts.bin" \
    "models/$S/encoder.onnx" "models/$S/memattn.onnx" "models/$S/heads.onnx" \
    "models/$S/heads_prompt.onnx" "models/$S/heads_mask.onnx" \
    "models/$S/memenc.onnx"
done

# --- and the check that each tarball is COMPLETE. A tier missing one graph is
#     the failure this whole layout has to make impossible: the page offers the
#     chip (the tier list is in the committed manifest), the visitor presses it,
#     and one 404 lands in the middle of the load.
for T in "$OUT"/*.tar.gz; do
  case "$T" in
    *-fp32.tar.gz) WANT="encoder memattn heads heads_prompt heads_mask" ;;
    *)             WANT="encoder.fp16 memattn.fp16 heads.fp16 heads_prompt.fp16 heads_mask.fp16" ;;
  esac
  LIST="$(tar tzf "$T")"
  for F in $WANT manifest consts memenc; do
    case "$F" in
      manifest) N="manifest.json" ;; consts) N="consts.bin" ;;
      *)        N="$F.onnx" ;;
    esac
    echo "$LIST" | grep -q "/$N\$" \
      || { echo "[release] $(basename "$T") is missing $N"; exit 1; }
  done
  case "$T" in
    *-fp32.tar.gz) ;;
    *) echo "$LIST" | grep -q '/memenc.f16in.onnx$' \
         || { echo "[release] $(basename "$T") is missing memenc.f16in.onnx"; exit 1; } ;;
  esac
  echo "[release] $(basename "$T") complete"
done

( cd "$OUT" && shasum -a 256 ./*.tar.gz | sed 's| \./| |' > SHA256SUMS )
ls -la "$OUT"
cat "$OUT/SHA256SUMS"
cat <<TXT

[release] nothing has been published. To publish, from the repo root:

  gh release create models-$V \\
    dist/dither-studio-models-$V.tar.gz \\
    dist/dither-studio-models-$V-512.tar.gz \\
    dist/dither-studio-models-$V-1024.tar.gz \\
    dist/dither-studio-models-$V-fp32.tar.gz \\
    dist/dither-studio-models-$V-512-fp32.tar.gz \\
    dist/dither-studio-models-$V-1024-fp32.tar.gz \\
    dist/SHA256SUMS \\
    --repo kcvete/dither-studio \\
    --title "EdgeTAM ONNX graphs $V" \\
    --notes "Browser-engine graphs at three tracker resolutions. \\
models-$V.tar.gz is the default 768 px fp16 set; -512 and -1024 add the other \\
two squares under models/512 and models/1024; the -fp32 tarballs are the same \\
three squares in fp32, for a GPU with no shader-f16. Each extracts with: \\
tar xz -C web"

If models-$V already exists, add ONLY the two new squares -- gzip stamps the
time, so the fp16 tarballs above are not byte-identical to the ones already
published and re-uploading them would change checksums nobody asked to change:

  gh release upload models-$V \\
    dist/dither-studio-models-$V-512-fp32.tar.gz \\
    dist/dither-studio-models-$V-1024-fp32.tar.gz \\
    --repo kcvete/dither-studio

SHA256SUMS in dist/ describes THIS build of all six. Upload it only together
with all six (--clobber), or the sums in the release will not match its assets.
TXT
