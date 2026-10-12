#!/usr/bin/env bash
# giro's container entrypoint. `serve` (the default) starts the web app on every GPU the container
# sees; anything else runs as a giro subcommand (`giro models fetch`, `giro run ...`).
set -euo pipefail
cd /app

if [ "${1:-serve}" != "serve" ]; then
  exec giro "$@"
fi
shift || true

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "giro: no NVIDIA GPU in the container. Run with --gpus all (Docker) or --device nvidia.com/gpu=all (Podman)." >&2
  exit 1
fi
n=$(nvidia-smi -L | wc -l)
case "$n" in
  1) gpus=0 ;;
  *) gpus=0,1 ;;  # giro drives at most two GPUs (giro.comfy.server.PORTS)
esac

if ! giro models list | grep -q "GB  diffusion/wan22/wan2.2_fun_control_high_noise"; then
  echo "giro: the proxy orbit's weights are not in /models yet. Fetch them (about 60 GB for the default groups):" >&2
  echo "      docker run --rm -v <models>:/models <image> models fetch" >&2
fi

exec giro serve --host 0.0.0.0 --port 8470 --https-port 8443 --gpus "$gpus" "$@"
