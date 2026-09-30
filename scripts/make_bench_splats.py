"""Test splats for measuring the Quest's WebXR budget (docs/FINDINGS.md, "VR on Quest 3"): copies of one real
exported splat side by side, rows of 4, all in front of a viewer 1.5 m away (VrView).

    uv run python scripts/make_bench_splats.py data/jobs/<id>/best/export/splat.ply 2 4 6 8

writes data/bench/crowd_<n>k.spz for each copy count (SPZ v3, as giro exports).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

from giro import render, splat

src = Path(sys.argv[1])
out = Path(__file__).resolve().parents[1] / "data" / "bench"
out.mkdir(parents=True, exist_ok=True)
one = splat.read_ply(src)
for copies in map(int, sys.argv[2:]):
    parts = []
    for i in range(copies):
        row, col = divmod(i, 4)
        n_in_row = min(4, copies - row * 4)
        dx = (col - (n_in_row - 1) / 2) * 0.75  # meters, canonical frame (y up)
        dz = -row * 1.2
        part = one.copy()
        # PLY file frame = diag(-1, -1, 1) @ canonical
        part["x"] -= dx
        part["z"] += dz
        parts.append(part)
    tiled = np.concatenate(parts)
    name = f"crowd_{round(len(tiled) / 1000)}k"
    with tempfile.TemporaryDirectory() as tmp:
        ply = Path(tmp) / f"{name}.ply"
        splat.write_ply(ply, tiled)
        render.splat_transform(str(ply), str(out / f"{name}.spz"), "--spz-version", "3")
    print(f"{name}: {len(tiled):,} splats, {(out / f'{name}.spz').stat().st_size / 2**20:.1f} MB")
