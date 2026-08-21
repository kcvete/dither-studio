#!/usr/bin/env bash
# Idempotent environment bootstrap for Dither Studio.
# Creates env/venv (python 3.13) + env/EdgeTAM (editable install) + checkpoint,
# then the CoreML graphs, the ONNX graphs and the vendored onnxruntime-web.
#
#   ./setup.sh --page-only    just the static page: pre-exported ONNX graphs
#                             from the models-v1 release + onnxruntime-web.
#                             No venv, no PyTorch, no checkpoint.
#   DV_MODELS=download        full install, but download the ONNX graphs
#                             instead of exporting them
#   DV_SKIP_WEB_MODELS=1      server only: no ONNX graphs, no web/ort
#   DV_EDGETAM_CKPT=<path>    reuse an edgetam.pt you already have
#   DV_PYTHON=<path>          which python builds the venv (default 3.13)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVD="$HERE/env"
VENV="$ENVD/venv"
REPO="$ENVD/EdgeTAM"
CKPT="$REPO/checkpoints/edgetam.pt"
EDGETAM_COMMIT="7711e012a30a2402c4eaab637bdb00a521302c91"
ORT_VERSION="1.27"
# where the pre-exported browser-engine graphs live, for DV_MODELS=download.
# Three tarballs now, one per tracker square: the default 768 px set is what a
# first visit downloads, and 512/1024 land under web/models/512 and
# web/models/1024 so their chips have something to load. All extract into web/.
MODELS_RELEASE="${DV_MODELS_RELEASE:-models-v1.1}"
MODELS_VER="${MODELS_RELEASE#models-}"
MODELS_BASE="https://github.com/kcvete/dither-studio/releases/download/$MODELS_RELEASE"
MODELS_URL="$MODELS_BASE/dither-studio-models-$MODELS_VER.tar.gz"
# the other two squares; DV_MODELS_TIERS=0 keeps the download to the default set
MODELS_TIERS="${DV_MODELS_TIERS:-512 1024}"
PY="${DV_PYTHON:-/opt/homebrew/opt/python@3.13/bin/python3.13}"
[ -x "$PY" ] || PY="$(command -v python3.13)"

mkdir -p "$ENVD"

# onnxruntime-web itself. Vendored, not pulled from a CDN at run time, because
# the whole point of the browser engine is that the page talks to nobody.
vendor_ort() {
  local ORT="$HERE/web/ort" PKG D v
  if [ -f "$ORT/ort.all.bundle.min.mjs" ]; then return 0; fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "[setup] no npm - skipping onnxruntime-web (see web/README.md)"
    return 0
  fi
  echo "[setup] fetching onnxruntime-web (once, ~39 MB)"
  PKG="$ENVD/ortpkg"
  mkdir -p "$PKG" "$ORT"
  ( cd "$PKG" && npm install --silent --no-audit --no-fund --prefix "$PKG" \
      onnxruntime-web@"$ORT_VERSION" >/dev/null 2>&1 ) || true
  D="$PKG/node_modules/onnxruntime-web/dist"
  if [ -f "$D/ort.all.bundle.min.mjs" ]; then
    # only the two builds the page can actually load: .jsep is the WebGPU one,
    # the bare one is the multi-threaded WASM fallback. The asyncify and jspi
    # variants are another 39 MB nothing here asks for.
    cp "$D/ort.all.bundle.min.mjs" "$ORT/"
    for v in "" ".jsep"; do
      cp "$D/ort-wasm-simd-threaded$v.mjs" "$D/ort-wasm-simd-threaded$v.wasm" "$ORT/"
    done
    echo "[setup] onnxruntime-web ok ($(du -sh "$ORT" | cut -f1))"
  else
    echo "[setup] onnxruntime-web download failed - see web/README.md"
  fi
}

# Every square is either COMPLETE or absent -- there is no third state that
# works. A tier missing one graph is a quality chip the page offers and then
# fails to load: web/track.js names the file and web/engines/browser.js drops
# back to 768 px, but that is a rescue, not an install. Say so here instead.
models_complete() {
  local root="$1"; shift
  local ok=1
  for S in "$@"; do
    [ "$S" = "0" ] && continue
    local d="$root"; [ "$S" = "768" ] || d="$root/$S"
    [ -d "$d" ] || continue
    for F in manifest.json consts.bin encoder.fp16.onnx memattn.fp16.onnx \
             heads.fp16.onnx heads_prompt.fp16.onnx heads_mask.fp16.onnx \
             memenc.onnx memenc.f16in.onnx; do
      if [ ! -s "$d/$F" ]; then
        echo "[setup] WARNING: ${S}px is incomplete - $d/$F is missing."
        echo "[setup]          that chip will fall back to 768 px in the page."
        ok=0; break
      fi
    done
    # fp32 is optional, but half of it is not
    if [ -f "$d/encoder.onnx" ]; then
      for F in memattn.onnx heads.onnx heads_prompt.onnx heads_mask.onnx; do
        [ -s "$d/$F" ] || { echo "[setup] WARNING: ${S}px fp32 is incomplete - $d/$F is missing"; ok=0; break; }
      done
    fi
  done
  [ "$ok" = "1" ] && echo "[setup] model squares complete: $*"
  return 0
}

# --page-only: everything the static page needs and nothing else. No venv, no
# PyTorch, no EdgeTAM clone, no checkpoint -- the graphs come pre-exported from
# the release and onnxruntime-web from npm. This is the install for anyone who
# is hosting web/ and never running the accelerator.
if [ "${1:-}" = "--page-only" ] || [ "${DV_PAGE_ONLY:-0}" = "1" ]; then
  echo "[setup] page only: models from $MODELS_RELEASE, no python environment"
  if [ -f "$HERE/web/models/encoder.fp16.onnx" ]; then
    echo "[setup] models already there"
  else
    curl -fL --progress-bar "$MODELS_URL" | tar xz -C "$HERE/web" \
      || { echo "[setup] model download failed: $MODELS_URL"; exit 1; }
    echo "[setup] models ok"
  fi
  for S in $MODELS_TIERS; do
    [ "$S" = "0" ] && continue
    if [ -f "$HERE/web/models/$S/encoder.fp16.onnx" ]; then
      echo "[setup] ${S}px models already there"
    else
      echo "[setup] the ${S}px tracker set"
      curl -fL --progress-bar "$MODELS_BASE/dither-studio-models-$MODELS_VER-$S.tar.gz" \
        | tar xz -C "$HERE/web" \
        || echo "[setup] ${S}px download failed - that chip will track at 768 instead"
    fi
  done
  # DV_MODELS_FP32=1 adds the fp32 graphs for every square. Nothing needs them:
  # a GPU without shader-f16 runs the fp16 graphs on WASM and says so. They are
  # the FASTER answer for that machine (WebGPU fp32, ~20% slower than fp16,
  # against WASM's ~6x), at ~85 MB per square. All or nothing, per square --
  # half a square installed is a chip that 404s in the middle of a load.
  if [ "${DV_MODELS_FP32:-0}" = "1" ]; then
    for S in "" $MODELS_TIERS; do
      [ "$S" = "0" ] && continue
      D="$HERE/web/models${S:+/$S}"; SFX="${S:+-$S}-fp32"
      if [ -f "$D/encoder.onnx" ]; then
        echo "[setup] ${S:-768}px fp32 already there"
      else
        echo "[setup] the ${S:-768}px fp32 set"
        curl -fL --progress-bar "$MODELS_BASE/dither-studio-models-$MODELS_VER$SFX.tar.gz" \
          | tar xz -C "$HERE/web" \
          || echo "[setup] ${S:-768}px fp32 download failed - that square stays fp16-only"
      fi
    done
  fi
  models_complete "$HERE/web/models" 768 $MODELS_TIERS
  vendor_ort
  echo "[setup] ok - now serve web/ with any static server"
  exit 0
fi

if [ ! -d "$REPO/.git" ]; then
  echo "[setup] cloning EdgeTAM -> $REPO"
  git clone --quiet https://github.com/facebookresearch/EdgeTAM.git "$REPO"
  git -C "$REPO" checkout --quiet "$EDGETAM_COMMIT"
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] creating venv ($PY)"
  "$PY" -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip wheel setuptools
fi

if ! "$VENV/bin/python" -c "import sam2, fastapi, timm, cv2" >/dev/null 2>&1; then
  echo "[setup] installing python deps (this can take a few minutes)"
  "$VENV/bin/pip" install --quiet torch torchvision
  ( cd "$REPO" && SAM2_BUILD_CUDA=0 SAM2_BUILD_ALLOW_ERRORS=1 "$VENV/bin/pip" install --quiet -e . )
  "$VENV/bin/pip" install --quiet timm fastapi "uvicorn[standard]" python-multipart pillow numpy opencv-python
fi

# coremltools drives the fastest tracking backend. It is optional: if the wheel
# or the export below fails, server.py falls back to the torch backends.
if ! "$VENV/bin/python" -c "import coremltools" >/dev/null 2>&1; then
  echo "[setup] installing coremltools (optional, powers DV_BACKEND=coreml)"
  "$VENV/bin/pip" install --quiet coremltools || echo "[setup] coremltools install failed - torch backends still work"
fi

# EdgeTAM's spatial perceiver does `.expand(B,-1,-1).view(...)`, which throws
# "view size is not compatible with input tensor's size and stride" as soon as
# more than one object is tracked (B > 1). reshape() is the documented fix.
PERC="$REPO/sam2/modeling/perceiver.py"
if ! grep -q "DITHER_VIDEO_MULTIOBJ_PATCH" "$PERC"; then
  echo "[setup] patching perceiver.py for multi-object batches"
  python3 - "$PERC" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace("self.latents_2d.unsqueeze(0).expand(B, -1, -1).view(-1, 1, C)",
              "self.latents_2d.unsqueeze(0).expand(B, -1, -1).reshape(-1, 1, C)")
s = s.replace("latents_2d = latents_2d.view(B, num_window, num_window, C)",
              "latents_2d = latents_2d.reshape(B, num_window, num_window, C)")
s = "# DITHER_VIDEO_MULTIOBJ_PATCH: .view -> .reshape so B>1 works\n" + s
open(p, "w").write(s)
PY
fi

if [ ! -f "$CKPT" ]; then
  mkdir -p "$(dirname "$CKPT")"
  # DV_EDGETAM_CKPT lets a second checkout reuse a copy you already have
  # instead of pulling 100 MB again.
  if [ -n "${DV_EDGETAM_CKPT:-}" ] && [ -f "$DV_EDGETAM_CKPT" ]; then
    echo "[setup] copying checkpoint from \$DV_EDGETAM_CKPT"
    cp "$DV_EDGETAM_CKPT" "$CKPT"
  else
    echo "[setup] downloading checkpoint"
    curl -fsSL -o "$CKPT" https://huggingface.co/spaces/facebook/EdgeTAM/resolve/main/checkpoints/edgetam.pt
  fi
fi

# The three EdgeTAM stages that dominate tracking time, converted to CoreML:
# image encoder + memory attention + memory encoder, one static-shape graph per
# object count -- and one set per tracking resolution the UI offers, because the
# shapes differ. ~20 s and 55-68 MB each (183 MB for all three), and worth
# 9.4 -> 15.4 fps at 1024 px.
# Cached: a size is rebuilt only if the checkpoint or the exporter is newer.
if "$VENV/bin/python" -c "import coremltools" >/dev/null 2>&1; then
  for SZ in 1024 768 512; do
    CML="$ENVD/coreml/$SZ/manifest.json"
    if [ -f "$CML" ] && [ ! "$CKPT" -nt "$CML" ] && [ ! "$HERE/coreml/export.py" -nt "$CML" ] \
       && [ ! "$HERE/coreml/wrappers.py" -nt "$CML" ]; then
      continue
    fi
    echo "[setup] exporting CoreML graphs at $SZ px (once, ~15 s)"
    "$VENV/bin/python" "$HERE/coreml/export.py" --batch 1,2,3 --image-size "$SZ" \
      >/dev/null 2>&1 \
      && echo "[setup] CoreML export $SZ ok" \
      || echo "[setup] CoreML export $SZ failed - that quality falls back to torch"
  done
fi

# ---------------------------------------------------------------- browser engine
# The same three stages plus the SAM heads, exported to ONNX so the page can
# track without a server at all. ~130 MB of weights + a 41 MB onnxruntime build,
# neither committed. Skip with DV_SKIP_WEB_MODELS=1 if you only ever want the
# server path.
if [ "${DV_SKIP_WEB_MODELS:-0}" != "1" ]; then
  WEBM="$HERE/web/models"
  MAN="$WEBM/manifest.json"
  NEED=0
  for g in encoder memattn heads heads_prompt heads_mask; do
    [ -f "$WEBM/$g.fp16.onnx" ] || NEED=1
  done
  [ -f "$WEBM/memenc.onnx" ] || NEED=1
  [ "$HERE/onnxexport/export_onnx.py" -nt "$MAN" ] && NEED=1
  [ "$HERE/onnxexport/wrappers_onnx.py" -nt "$MAN" ] && NEED=1
  [ "$CKPT" -nt "$MAN" ] && NEED=1
  # DV_MODELS=download pulls the same graphs from the release instead of
  # spending 90 s and a PyTorch install re-deriving them from the checkpoint.
  if [ "$NEED" = "1" ] && [ "${DV_MODELS:-export}" = "download" ]; then
    echo "[setup] downloading the ONNX graphs from $MODELS_RELEASE (~53 MB)"
    if curl -fL --progress-bar "$MODELS_URL" | tar xz -C "$HERE/web"; then
      echo "[setup] models ok"; NEED=0
      for S in $MODELS_TIERS; do
        [ "$S" = "0" ] && continue
        [ -f "$HERE/web/models/$S/encoder.fp16.onnx" ] && continue
        curl -fL --progress-bar "$MODELS_BASE/dither-studio-models-$MODELS_VER-$S.tar.gz" \
          | tar xz -C "$HERE/web" \
          || echo "[setup] ${S}px download failed - that chip will track at 768 instead"
      done
    else
      echo "[setup] model download failed - falling back to the export"
    fi
  fi
  if [ "$NEED" = "1" ]; then
    if ! "$VENV/bin/python" -c "import onnx, onnxruntime" >/dev/null 2>&1; then
      echo "[setup] installing onnx + onnxruntime (for the ONNX export)"
      "$VENV/bin/pip" install --quiet onnx onnxruntime || true
    fi
    if "$VENV/bin/python" -c "import onnx, onnxruntime" >/dev/null 2>&1; then
      echo "[setup] exporting the ONNX graphs for the browser engine (once, ~90 s)"
      # --tiers records in the default manifest which squares this checkout
      # carries, so the page offers exactly those chips and probes for nothing
      "$VENV/bin/python" "$HERE/onnxexport/export_onnx.py" --image-size 768 \
        --tiers "$(echo 768 $MODELS_TIERS | tr ' ' ,)" >/dev/null 2>&1 \
        && echo "[setup] ONNX export ok" \
        || echo "[setup] ONNX export failed - the browser engine will say so in the UI"
      # the other squares, exported the same way into their own directories.
      # DV_MODELS_TIERS=0 skips them; they are ~140 MB each.
      for S in $MODELS_TIERS; do
        [ "$S" = "0" ] && continue
        [ -f "$HERE/web/models/$S/encoder.fp16.onnx" ] && continue
        echo "[setup] exporting the ${S}px graphs (once, ~90 s)"
        "$VENV/bin/python" "$HERE/onnxexport/export_onnx.py" --image-size "$S" \
          --out "$HERE/web/models/$S" >/dev/null 2>&1 \
          && echo "[setup] ${S}px export ok" \
          || echo "[setup] ${S}px export failed - that chip will track at 768 instead"
      done
    else
      echo "[setup] no onnx/onnxruntime - skipping the browser engine models"
    fi
  fi

  models_complete "$WEBM" 768 $MODELS_TIERS
  vendor_ort
fi

# the serial dither modes (error diffusion, Riemersma) run a per-pixel loop that
# would take minutes per frame in Python; this is that loop in C.
LIB="$ENVD/libcdither.dylib"
if [ ! -f "$LIB" ] || [ "$HERE/server/cdither.c" -nt "$LIB" ]; then
  echo "[setup] building libcdither"
  # -ffp-contract=off keeps clang from fusing multiply-add, which would round once
  # where JavaScript rounds twice and break pixel parity with web/dither.js
  cc -O3 -ffp-contract=off -shared -fPIC -o "$LIB" "$HERE/server/cdither.c"
fi

echo "[setup] ok"
