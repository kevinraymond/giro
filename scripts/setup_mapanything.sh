#!/usr/bin/env bash
# Install giro's pinned MapAnything (feedforward camera poses, M5) in vendor/map-anything,
# in its own venv (torch), with the Apache-licensed weights (docs/FINDINGS.md, "Camera poses").
set -euo pipefail

MAPANYTHING_TAG=v1.1.3  # 9d1db2dd
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/vendor/map-anything"

if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 --branch "$MAPANYTHING_TAG" https://github.com/facebookresearch/map-anything.git "$DEST"
fi
git -C "$DEST" fetch --quiet --depth 1 origin tag "$MAPANYTHING_TAG" || true
git -C "$DEST" checkout --quiet "$MAPANYTHING_TAG"
uv venv --python 3.12 "$DEST/.venv"
VIRTUAL_ENV="$DEST/.venv" uv pip install -e "$DEST"
# Apache 2.0 weights (the default facebook/map-anything checkpoint is CC-BY-NC).
"$DEST/.venv/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('facebook/map-anything-apache'))"
echo "MapAnything ready: $DEST"
