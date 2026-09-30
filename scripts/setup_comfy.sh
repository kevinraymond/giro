#!/usr/bin/env bash
# Recreate giro's pinned, headless ComfyUI in vendor/comfyui.
# No custom nodes: MiniMax H3, SAM3 and Qwen-Image-Edit are all core nodes at this commit.
set -euo pipefail

COMFY_COMMIT=fe4195f7f4275f2626cbafc703acc3ddde1e5490  # ComfyUI v0.31.1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/vendor/comfyui"

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/comfyanonymous/ComfyUI.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$COMFY_COMMIT" 2>/dev/null || true
git -C "$DEST" checkout --quiet "$COMFY_COMMIT"

uv venv --allow-existing --python 3.13.9 "$DEST/.venv"
# comfy-lock.txt is the exact freeze of a working install (torch 2.12.1+cu130).
uv pip install --python "$DEST/.venv/bin/python" -r "$ROOT/scripts/comfy-lock.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

echo "ComfyUI $(git -C "$DEST" describe --tags --always) ready in $DEST"
