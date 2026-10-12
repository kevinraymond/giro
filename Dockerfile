# syntax=docker/dockerfile:1
# giro in a container: code and runtime only. Model weights live on a volume at /models
# (`giro models fetch` fills it; see README, "Running it with Docker"), jobs at /app/data.
#
#   docker build -t giro .
#   docker run --rm --gpus all -p 8470:8470 -p 8443:8443 -v ~/models:/models -v ./data:/app/data giro
#
# Four stages: COLMAP (CUDA) and brush-cli are compiled once, the UI is built with Node, and the
# runtime image gets the binaries plus giro's and ComfyUI's pinned Python environments.
# CUDA 13 (torch 2.12.1+cu130 in scripts/comfy-lock.txt) needs an NVIDIA driver of 580 or newer.

ARG CUDA_VERSION=13.0.2
ARG UBUNTU_VERSION=24.04
# The GPUs the COLMAP binary runs on: Ampere, Ada, Hopper, Blackwell (sm_80 ... sm_120).
ARG CUDA_ARCHITECTURES="80;86;89;90;100;120"
ARG COLMAP_VERSION=4.2.1
ARG BRUSH_COMMIT=6378a76a
ARG UV_VERSION=0.11.6

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM node:22-bookworm-slim AS node

# ---- COLMAP with CUDA, no GUI ---------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS colmap
ARG CUDA_ARCHITECTURES
ARG COLMAP_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential ca-certificates \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev \
        libopenimageio-dev openimageio-tools libmetis-dev libgoogle-glog-dev libgtest-dev libgmock-dev \
        libsqlite3-dev libcgal-dev libceres-dev libsuitesparse-dev libcurl4-openssl-dev libssl-dev \
        libopenblas-dev liblapack-dev libgl1-mesa-dev libglew-dev \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/include/opencv4  # Ubuntu's openimageio config wants the directory
RUN git clone --depth 1 --branch ${COLMAP_VERSION} https://github.com/colmap/colmap.git /colmap \
    && cmake -S /colmap -B /colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DGUI_ENABLED=OFF -DCUDA_ENABLED=ON -DTESTS_ENABLED=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" -DBLA_VENDOR=OpenBLAS \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
    && ninja -C /colmap/build install

# ---- brush-cli (Gaussian splat training; wgpu over Vulkan at run time) ----------------------
FROM rust:1.97-bookworm AS brush
ARG BRUSH_COMMIT
RUN apt-get update && apt-get install -y --no-install-recommends cmake pkg-config libssl-dev clang \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/ArthurBrussee/brush.git /brush && git -C /brush checkout --quiet ${BRUSH_COMMIT}
RUN --mount=type=cache,target=/usr/local/cargo/registry --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/brush/target \
    cargo build --manifest-path /brush/Cargo.toml --release --locked -p brush-cli \
    && cp /brush/target/release/brush-cli /brush-cli

# ---- the web UI ------------------------------------------------------------------------------
FROM node AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

# ---- runtime ---------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ENV DEBIAN_FRONTEND=noninteractive
# ffmpeg for frames; libvulkan1 for brush-cli (wgpu); the rest are COLMAP's shared libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git ffmpeg libvulkan1 libegl1 libx11-6 libxext6 libxcb1 libgomp1 \
        libboost-program-options1.83.0 libomp5 libmetis5 libceres4t64 libopenimageio2.4t64 \
        libgoogle-glog0v6t64 libcurl4 libssl3t64 libopenblas0 libsqlite3-0 libgl1 libopengl0 libglew2.2 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=colmap /colmap-install/ /usr/local/
# Node 22 for splat-transform (SPZ/SOG export).
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules/npm /usr/local/lib/node_modules/npm
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
COPY --from=uv /uv /usr/local/bin/uv

RUN (userdel -r ubuntu 2>/dev/null || true) && useradd --create-home --uid 1000 --shell /bin/bash giro && mkdir -p /app/data /models && chown -R giro:giro /app /models
WORKDIR /app
USER giro
ENV UV_LINK_MODE=copy UV_PYTHON=3.12 UV_PYTHON_INSTALL_DIR=/home/giro/.local/share/uv/python

# The pinned headless ComfyUI (torch cu130) and splat-transform first: they do not change with
# giro's source, so a source change rebuilds only giro's own environment.
COPY --chown=giro:giro scripts/ scripts/
RUN --mount=type=cache,target=/home/giro/.cache/uv,uid=1000 scripts/setup_comfy.sh
RUN --mount=type=cache,target=/home/giro/.npm,uid=1000 scripts/setup_splat_transform.sh
COPY --chown=giro:giro pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=giro:giro src/ src/
RUN --mount=type=cache,target=/home/giro/.cache/uv,uid=1000 uv sync --frozen --no-dev
COPY --from=brush --chown=giro:giro /brush-cli vendor/brush/target/release/brush-cli
COPY --chown=giro:giro docs/ docs/
COPY --chown=giro:giro docker/entrypoint.sh docker/entrypoint.sh
COPY --from=ui --chown=giro:giro /ui/dist ui/dist
# The page of the exported all-in-one splat.html (its bundle is in ui/dist).
COPY --chown=giro:giro ui/standalone/template.html ui/standalone/template.html

# /models: the weights (giro models fetch); /app/data: jobs, logs, the self-signed certificate.
ENV GIRO_MODELS=/models HF_HOME=/models/hf PATH=/app/.venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
VOLUME ["/models", "/app/data"]
EXPOSE 8470 8443
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["serve"]
