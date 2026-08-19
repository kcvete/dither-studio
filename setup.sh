#!/usr/bin/env bash
# Idempotent environment bootstrap for Dither Video.
# Creates env/venv (python 3.13) + env/EdgeTAM (editable install) + checkpoint.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVD="$HERE/env"
VENV="$ENVD/venv"
REPO="$ENVD/EdgeTAM"
CKPT="$REPO/checkpoints/edgetam.pt"
EDGETAM_COMMIT="7711e012a30a2402c4eaab637bdb00a521302c91"
PY="${DV_PYTHON:-/opt/homebrew/opt/python@3.13/bin/python3.13}"
[ -x "$PY" ] || PY="$(command -v python3.13)"

mkdir -p "$ENVD"

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
  SRC="/private/tmp/claude-501/-Users-kevincvetezar/012258e2-fe5c-46ee-8648-eeafdcc38f82/scratchpad/parkour/env/EdgeTAM/checkpoints/edgetam.pt"
  if [ -f "$SRC" ]; then
    echo "[setup] copying checkpoint from local cache"
    cp "$SRC" "$CKPT"
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

# the serial dither modes (error diffusion, Riemersma) run a per-pixel loop that
# would take minutes per frame in Python; this is that loop in C.
LIB="$ENVD/libcdither.dylib"
if [ ! -f "$LIB" ] || [ "$HERE/cdither.c" -nt "$LIB" ]; then
  echo "[setup] building libcdither"
  # -ffp-contract=off keeps clang from fusing multiply-add, which would round once
  # where JavaScript rounds twice and break pixel parity with static/dither.js
  cc -O3 -ffp-contract=off -shared -fPIC -o "$LIB" "$HERE/cdither.c"
fi

echo "[setup] ok"
