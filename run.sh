#!/usr/bin/env bash
# Idempotent launcher: build the env if missing, start the server, open the UI.
# Defaults to 127.0.0.1:8765; falls forward to the next free port if that one
# is taken by something else.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="${DV_PORT:-8765}"

"$HERE/setup.sh"

is_ours() { curl -sf "http://127.0.0.1:$1/api/palettes" >/dev/null 2>&1; }
in_use()  { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

PORT=""
for p in $(seq "$BASE" $((BASE + 10))); do
  if is_ours "$p"; then echo "[run] server already up on :$p"; PORT="$p"; break; fi
  if ! in_use "$p"; then PORT="$p"; break; fi
done
[ -n "$PORT" ] || { echo "[run] no free port in $BASE..$((BASE+10))"; exit 1; }

if ! is_ours "$PORT"; then
  [ "$PORT" = "$BASE" ] || echo "[run] :$BASE is taken, using :$PORT"
  echo "[run] starting server on http://127.0.0.1:$PORT"
  DV_PORT="$PORT" "$HERE/env/venv/bin/python" "$HERE/server.py" &
  SRV=$!
  for _ in $(seq 1 60); do is_ours "$PORT" && break; sleep 0.5; done
fi

[ "${DV_NO_OPEN:-0}" = "1" ] || open "http://127.0.0.1:$PORT/"
wait ${SRV:-} 2>/dev/null || true
