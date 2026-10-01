"""Camera poses from Depth Anything 3 (feedforward), written as a COLMAP text model.

Runs inside vendor/depth-anything-3/.venv (torch, depth_anything_3), not giro's own environment:

    python da3_poses.py OUT_DIR IMAGE... --root ATTEMPT [--size W H] [--points N] [--masks MASK...]

Images are resized (never cropped) to --size, whose sides must be multiples of 14 and whose
aspect ratio must be within 2% of theirs; the predicted intrinsics scale back to full resolution
per axis. Frames (every
image but the ones under hero/) share one SIMPLE_PINHOLE camera with the median focal
length, like COLMAP with single_camera_per_folder; the hero gets its own. With --masks,
the 3D points (Brush's initialization) are kept only inside the subject masks, and only
the more confident half of them.

OUT_DIR gets cameras.txt, images.txt (names relative to the attempt, e.g. frames/00012.png),
points3D.txt and stats.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image

from depth_anything_3.api import DepthAnything3

MODEL = "depth-anything/DA3-BASE"  # Apache 2.0 weights (docs/FINDINGS.md, "Pose fallback")


def rotmat_to_qvec(r: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) of a rotation matrix."""
    k = np.array([
        [r[0, 0] - r[1, 1] - r[2, 2], 0, 0, 0],
        [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], 0, 0],
        [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], 0],
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r[0, 0] + r[1, 1] + r[2, 2]],
    ]) / 3.0
    w, v = np.linalg.eigh(k)
    q = v[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--root", type=Path, required=True, help="attempt dir: image names are relative to it")
    ap.add_argument("--size", type=int, nargs=2, default=[378, 504], metavar=("W", "H"))
    ap.add_argument("--points", type=int, default=100_000, help="3D points to keep for Brush")
    ap.add_argument("--masks", type=Path, nargs="*", default=[])
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    t0 = time.monotonic()

    # Not resolved: frames may be symlinks into other dirs, and names must stay relative to the attempt.
    root = args.root.absolute()
    names = [str(p.absolute().relative_to(root)) for p in args.images]
    sizes = [Image.open(p).size for p in args.images]
    w_in, h_in = args.size
    for (w, h), name in zip(sizes, names):
        if abs((w / h) / (w_in / h_in) - 1) > 0.02:
            raise SystemExit(f"{name} is {w}x{h}, not the aspect ratio of --size {w_in}x{h_in}")
    rgb = [np.asarray(Image.open(p).convert("RGB").resize((w_in, h_in), Image.Resampling.BICUBIC)) for p in args.images]

    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    t_load = time.monotonic()
    # Already at --size, so DA3's own resize (longest side to process_res) is a no-op.
    pred = model.inference(rgb, process_res=max(w_in, h_in), process_res_method="upper_bound_resize")
    t_infer = time.monotonic()
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    if pred.depth.shape[1:] != (h_in, w_in):
        raise SystemExit(f"DA3 resized the images to {pred.depth.shape[2]}x{pred.depth.shape[1]}; use a --size "
                         "whose sides are multiples of 14")

    hero = [n.startswith("hero/") for n in names]
    w2c = pred.extrinsics[:, :3, :4]
    focals = [float((k[0, 0] * w / w_in + k[1, 1] * h / h_in) / 2) for k, (w, h) in zip(pred.intrinsics, sizes)]
    frame_f = float(np.median([f for f, h_ in zip(focals, hero) if not h_]))

    args.out.mkdir(parents=True, exist_ok=True)
    frame_size = next(s for s, h_ in zip(sizes, hero) if not h_)
    cams = [f"1 SIMPLE_PINHOLE {frame_size[0]} {frame_size[1]} {frame_f:.6f} {frame_size[0] / 2} {frame_size[1] / 2}"]
    hero_idx = [i for i, h_ in enumerate(hero) if h_]
    if hero_idx:
        (w, h), f = sizes[hero_idx[0]], focals[hero_idx[0]]
        cams.append(f"2 SIMPLE_PINHOLE {w} {h} {f:.6f} {w / 2} {h / 2}")
    (args.out / "cameras.txt").write_text("# giro: Depth Anything 3\n" + "\n".join(cams) + "\n")
    with open(args.out / "images.txt", "w") as f:
        for i, (name, e) in enumerate(zip(names, w2c)):
            q = rotmat_to_qvec(e[:, :3])
            f.write(f"{i + 1} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in e[:, 3])} "
                    f"{2 if hero[i] else 1} {name}\n\n")  # no 2D observations

    # Brush initializes from the sparse points: unproject the depth maps inside the masks.
    masks = {m.absolute().relative_to(root / "masks").as_posix(): m for m in args.masks}
    ys, xs = np.mgrid[0:h_in, 0:w_in] + 0.5
    threshold = float(np.median(pred.conf))
    pts, cols = [], []
    for name, e, k, d, c, img in zip(names, w2c, pred.intrinsics, pred.depth, pred.conf, rgb):
        valid = (c > threshold) & (d > 0)
        if name in masks:
            m = Image.open(masks[name]).convert("L").resize((w_in, h_in), Image.Resampling.NEAREST)
            valid &= np.asarray(m) > 127
        cam = np.stack([(xs[valid] - k[0, 2]) / k[0, 0] * d[valid], (ys[valid] - k[1, 2]) / k[1, 1] * d[valid], d[valid]], 1)
        pts.append((cam - e[:, 3]) @ e[:, :3])  # camera -> world: R^T (x - t)
        cols.append(img[valid])
    pts_all, cols_all = np.concatenate(pts), np.concatenate(cols)
    if len(pts_all) > args.points:
        keep = np.random.default_rng(0).choice(len(pts_all), args.points, replace=False)
        pts_all, cols_all = pts_all[keep], cols_all[keep]
    with open(args.out / "points3D.txt", "w") as f:
        for i, (p, c) in enumerate(zip(pts_all, cols_all)):
            f.write(f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0\n")

    (args.out / "stats.json").write_text(json.dumps({
        "images": len(names), "model": args.model, "model_size": [w_in, h_in], "frame_focal_px": round(frame_f, 2),
        "focal_spread": round(float(np.std([f for f, h_ in zip(focals, hero) if not h_]) / frame_f), 4),
        "hero_focal_px": round(focals[hero_idx[0]], 2) if hero_idx else None,
        "points": len(pts_all), "load_s": round(t_load - t0, 1), "infer_s": round(t_infer - t_load, 1),
        "peak_vram_gb": round(peak_gb, 2),
    }, indent=2) + "\n")
    print(json.dumps({"ok": True, "infer_s": round(t_infer - t_load, 1), "peak_vram_gb": round(peak_gb, 2)}))


if __name__ == "__main__":
    main()
