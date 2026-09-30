"""Splat tools: splat-transform for conversion, transforms and GPU renders.

splat-transform (like PlayCanvas and SuperSplat) reads a PLY in the usual
3DGS file convention, y down, and works in its own y-up frame, turned 180
degrees about Z from the file: internal = FLIP @ file. The helpers here take
file coordinates (the COLMAP world Brush trains in) unless they say viewer.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SPLAT_TRANSFORM = ROOT / "vendor" / "splat-transform" / "node_modules" / ".bin" / "splat-transform"
FLIP = np.diag([-1.0, -1.0, 1.0])  # file <-> splat-transform's frame (its own inverse)


class ToolFailed(RuntimeError):
    pass


def splat_transform(*args: str, gpu: int | None = None) -> str:
    if not SPLAT_TRANSFORM.exists():
        raise ToolFailed(f"splat-transform not installed; run scripts/setup_splat_transform.sh ({SPLAT_TRANSFORM})")
    cmd = [str(SPLAT_TRANSFORM), "-q", "-w", *(["-g", str(gpu)] if gpu is not None else []), *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
        raise ToolFailed(f"splat-transform failed: {' | '.join(tail)}")
    return r.stdout


@dataclass
class Camera:
    """A look-at camera. Coordinates are in the PLY file's frame unless viewer=True."""

    position: Sequence[float]
    target: Sequence[float]
    up: Sequence[float]
    fov_deg: float = 40.0  # vertical
    viewer: bool = False

    def internal(self) -> dict[str, list[float]]:
        f = np.eye(3) if self.viewer else FLIP
        return {k: [round(float(x), 6) for x in f @ np.asarray(v, dtype=float)]
                for k, v in (("position", self.position), ("target", self.target), ("up", self.up))}


def render(ply: Path, cameras: Sequence[Camera], size: tuple[int, int],
           background: Sequence[float] = (0.0, 0.0, 0.0), gpu: int | None = None) -> list[Image.Image]:
    """Render `ply` from each camera; returns RGB images.

    One splat-transform run per camera (~1 s each): with --camera-track, frames after
    the first show blocky artifacts that single renders of the same view do not.
    """
    images = []
    with tempfile.TemporaryDirectory(prefix="giro-render-") as tmp:
        out = Path(tmp) / "view.webp"
        for camera in cameras:
            cam = camera.internal()
            splat_transform(str(ply), str(out),
                            "--resolution", f"{size[0]}x{size[1]}",
                            "--background", ",".join(str(c) for c in background),
                            "--camera-pos", ",".join(map(str, cam["position"])),
                            "--camera-target", ",".join(map(str, cam["target"])),
                            "--camera-up", ",".join(map(str, cam["up"])),
                            "--camera-fov", str(camera.fov_deg), gpu=gpu)
            images.append(Image.open(out).convert("RGB"))
    return images


def framing_distance(size: Sequence[float], fov_deg: float = 40.0, margin: float = 1.1) -> float:
    """Camera distance at which a box of `size` (width, height, depth) fits the vertical FOV
    from any direction (its bounding sphere does)."""
    radius = 0.5 * float(np.linalg.norm(size))
    return margin * radius / np.sin(np.radians(fov_deg) / 2)


def orbit_cameras(center: Sequence[float], radius: float, n: int,
                  fov_deg: float = 40.0, elevation_deg: float = 10.0, viewer: bool = True) -> list[Camera]:
    """n cameras on a ring around the +Y axis through `center` (viewer frame), looking at it,
    starting in front (+Z) and going counterclockwise seen from above."""
    c = np.asarray(center, dtype=float)
    el = np.radians(elevation_deg)
    cams = []
    for i in range(n):
        az = 2 * np.pi * i / n
        offset = radius * np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
        cams.append(Camera(c + offset, c, (0.0, 1.0, 0.0), fov_deg, viewer=viewer))
    return cams


def sheet(images: Sequence[Image.Image], cols: int, width: int | None = None) -> Image.Image:
    """Tile images into a grid, each scaled to `width` (default: its own)."""
    if width:
        images = [im.resize((width, round(im.height * width / im.width))) for im in images]
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (w * cols, h * rows))
    for i, im in enumerate(images):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out
