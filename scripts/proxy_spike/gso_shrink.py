"""Decimate the view-LoRA data's Pixal3D meshes (board #3709): a Pixal3D mesh is ~21M faces and
200 MB, and v2 keeps one per object for ~1,500 objects; the controls sample 6M points from it, so
1M faces lose nothing visible.

    uv run --with fast-simplification --with scipy python gso_shrink.py SRC_DIR DST_DIR [--faces 1000000]

For every SRC_DIR/<name>/work/mesh.npz: quadric decimation (fast-simplification, MIT) to --faces,
colors carried over from the nearest original vertex, the original's box saved as `bounds` (so
texture_common.samples() centers and scales it exactly as before) -> DST_DIR/<name>/work/mesh.npz
(float32 vertices, int32 faces, float16 colors). SRC_DIR may equal DST_DIR (in place). Skips meshes
already decimated.
"""
import argparse
import time
from pathlib import Path

import fast_simplification
import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("dst", type=Path)
ap.add_argument("--faces", type=int, default=1_000_000)
ap.add_argument("--names", default="")
ap.add_argument("--workers", type=int, default=4, help="nearest-vertex lookup threads (all 32 cores overheat the CPU)")
args = ap.parse_args()

names = args.names.split(",") if args.names else sorted(p.parent.parent.name for p in args.src.glob("*/work/mesh.npz"))
for i, name in enumerate(names):
    src, dst = args.src / name / "work" / "mesh.npz", args.dst / name / "work" / "mesh.npz"
    if not src.exists():
        continue
    if dst.exists():
        with np.load(dst) as d:
            if "bounds" in d.files:
                continue
    if time.time() - src.stat().st_mtime < 120:  # maybe still being written (GiroSaveMesh)
        continue
    t0 = time.monotonic()
    try:
        m = np.load(src)
        v, f, c = m["vertices"].astype(np.float32), m["faces"], m["colors"]
    except Exception as e:  # noqa: BLE001
        print(f"[{i + 1}/{len(names)}] {name}: unreadable ({e}), skipped", flush=True)
        continue
    bounds = np.stack([v.min(0), v.max(0)])
    if len(f) > args.faces:
        v2, f2 = fast_simplification.simplify(v, f.astype(np.int32), target_reduction=1 - args.faces / len(f))
        _, idx = cKDTree(v).query(v2, workers=args.workers)
        c2 = c[idx]
    else:
        v2, f2, c2 = v, f, c
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez(tmp, vertices=v2.astype(np.float32), faces=f2.astype(np.int32), colors=c2.astype(np.float16), bounds=bounds)
    tmp.replace(dst)
    print(f"[{i + 1}/{len(names)}] {name}: {len(f):,} -> {len(f2):,} faces, {dst.stat().st_size / 1e6:.0f} MB, "
          f"{time.monotonic() - t0:.0f} s", flush=True)
