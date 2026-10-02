#!/usr/bin/env bash
# Install giro's pinned Depth Anything 3 (feedforward camera poses for the pose fallback, M5) in
# vendor/depth-anything-3, in its own venv (torch), with the Apache-licensed DA3-BASE weights
# (docs/FINDINGS.md, "Pose fallback"; the Giant and Large checkpoints are CC BY-NC).
set -euo pipefail

DA3_COMMIT=3d835ec1a5802d64a8b8b15f817a1ab54809bfe4  # main, 2026-07-27
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/vendor/depth-anything-3"

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin "$DA3_COMMIT" || true
git -C "$DEST" checkout --quiet "$DA3_COMMIT"
[ -d "$DEST/.venv" ] || uv venv --python 3.12 "$DEST/.venv"
VIRTUAL_ENV="$DEST/.venv" uv pip install -e "$DEST" addict  # addict: imported by the model, not declared
"$DEST/.venv/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('depth-anything/DA3-BASE'))"
echo "Depth Anything 3 ready: $DEST"
