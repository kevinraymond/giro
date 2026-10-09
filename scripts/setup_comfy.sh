#!/usr/bin/env bash
# Recreate giro's pinned, headless ComfyUI in vendor/comfyui.
# No custom nodes: MiniMax H3, SAM3, Qwen-Image-Edit and Pixal3D (the proxy orbit's default proxy, which
# needs v0.34 or later) are all core nodes at this commit.
set -euo pipefail

COMFY_COMMIT=daeb5e53681e2b10a3f0727d9ec5bc90784bee10  # ComfyUI v0.38.2 (Oct 9, 2026; was v0.31.1)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/vendor/comfyui"

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/Comfy-Org/ComfyUI.git "$DEST"
fi
git -C "$DEST" fetch --quiet https://github.com/Comfy-Org/ComfyUI.git "$COMFY_COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$COMFY_COMMIT"

uv venv --allow-existing --python 3.13.9 "$DEST/.venv"
# comfy-lock.txt is the exact freeze of a working install (torch 2.12.1+cu130).
uv pip install --python "$DEST/.venv/bin/python" -r "$ROOT/scripts/comfy-lock.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

echo "ComfyUI $(git -C "$DEST" describe --tags --always) ready in $DEST"
