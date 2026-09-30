#!/usr/bin/env bash
# Build giro's pinned Brush (headless brush-cli) in vendor/brush.
set -euo pipefail

BRUSH_COMMIT=6378a76a  # main, 2026-09-26 "Improve training efficiency (#554)"
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST="$ROOT/vendor/brush"

if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/ArthurBrussee/brush.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$BRUSH_COMMIT"
# --locked: Cargo.lock pins the git revisions of Burn and the wgpu fork.
cargo build --manifest-path "$DEST/Cargo.toml" --release --locked -p brush-cli
echo "brush-cli ready: $DEST/target/release/brush-cli"
