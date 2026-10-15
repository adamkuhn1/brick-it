#!/usr/bin/env bash
# Starts the API (port 8000) and the web app (port 5173). Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."
trap 'kill 0' EXIT
uv run uvicorn server.app:app --host 127.0.0.1 --port 8000 &
(cd web && npm run dev -- --host 127.0.0.1) &
sleep 2
echo
echo "  Open http://127.0.0.1:5173"
echo
wait
