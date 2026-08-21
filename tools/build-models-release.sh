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
# Four tarballs, all extracting into web/ so `tar xz -C web` is the install:
#
#   dither-studio-models-<v>.tar.gz         models/          768 px, fp16
#   dither-studio-models-<v>-512.tar.gz     models/512/      512 px, fp16
#   dither-studio-models-<v>-1024.tar.gz    models/1024/    1024 px, fp16
#   dither-studio-models-<v>-fp32.tar.gz    models/          768 px, fp32
#
# The first is the only one a first visit downloads. The other squares load
# when their chip is picked. The fp32 set is for a GPU with no `shader-f16`;
# without it that machine falls to fp16 on WASM and says so.
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

# --- fp32, 768 px only: the fallback for a GPU without shader-f16
need "$M/encoder.onnx"
tar czf "$OUT/dither-studio-models-$V-fp32.tar.gz" -C "$HERE/web" \
  models/manifest.json models/consts.bin \
  models/encoder.onnx models/memattn.onnx models/heads.onnx \
  models/heads_prompt.onnx models/heads_mask.onnx models/memenc.onnx

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
    dist/SHA256SUMS \\
    --repo kcvete/dither-studio \\
    --title "EdgeTAM ONNX graphs $V" \\
    --notes "Browser-engine graphs at three tracker resolutions. \\
models-$V.tar.gz is the default 768 px fp16 set; -512 and -1024 add the other \\
two squares under models/512 and models/1024; -fp32 is the 768 px set for a \\
GPU with no shader-f16. Each extracts with: tar xz -C web"
TXT
