#!/usr/bin/env bash
# Install giro's pinned splat-transform (npm) in vendor/splat-transform.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SRC="$ROOT/scripts/splat-transform"
DEST="$ROOT/vendor/splat-transform"

mkdir -p "$DEST"
cp "$SRC/package.json" "$DEST/"
if [ -f "$SRC/package-lock.json" ]; then
  cp "$SRC/package-lock.json" "$DEST/"
  npm ci --prefix "$DEST" --no-audit --no-fund
else
  npm install --prefix "$DEST" --no-audit --no-fund
  cp "$DEST/package-lock.json" "$SRC/"  # first install: record the lockfile
fi
echo "splat-transform ready: $DEST/node_modules/.bin/splat-transform"
