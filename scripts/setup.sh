#!/usr/bin/env bash
# One-time setup: Python deps, TripoSR code + weights, web deps.
# Pass --no-model to skip the ~3 GB model install (you can still upload .glb/.obj files).
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "Install uv first: https://docs.astral.sh/uv/"; exit 1; }
command -v npm >/dev/null || { echo "Install Node.js 20+ first."; exit 1; }

if [[ "${1:-}" == "--no-model" ]]; then
  uv sync
else
  if [[ ! -d vendor/TripoSR ]]; then
    git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR
  fi
  uv sync --extra model
  echo "Downloading model weights (first time only)…"
  uv run python - <<'PY'
from huggingface_hub import hf_hub_download
import rembg
hf_hub_download("stabilityai/TripoSR", "config.yaml")
hf_hub_download("stabilityai/TripoSR", "model.ckpt")
rembg.new_session("u2net")
print("Model ready.")
PY
fi

(cd web && npm install)
echo "Setup done. Run ./scripts/dev.sh"
