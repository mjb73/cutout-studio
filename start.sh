#!/usr/bin/env bash
# PhotoCraft AI — the panel.  UI on :7887, MCP on :7887/mcp (streamable HTTP).
#
#   ./start.sh                    # http://127.0.0.1:7887
#   CUTOUT_PORT=9000 ./start.sh
#
# LD_LIBRARY_PATH is cleared because a linuxbrew shell injects libraries that break the venv's
# compiled modules.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
PORT="${CUTOUT_PORT:-7887}"
LOG="${CUTOUT_LOG:-/tmp/cutout-studio.log}"

if [ ! -d ".venv" ]; then
  echo "[start] creating .venv"
  "${PYTHON:-python3}" -m venv .venv
fi
if [ ! -f ".venv/.requirements-installed" ]; then
  echo "[start] installing requirements"
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
  touch .venv/.requirements-installed
fi

exec env -u LD_LIBRARY_PATH "$DIR/.venv/bin/python" -m uvicorn server:app \
  --host 127.0.0.1 --port "$PORT" >> "$LOG" 2>&1
