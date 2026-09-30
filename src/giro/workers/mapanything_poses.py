"""Camera poses from MapAnything (feedforward), written as a COLMAP text model.

Runs inside vendor/map-anything/.venv (torch, mapanything), not giro's own environment:

    python mapanything_poses.py OUT_DIR IMAGE... [--size W H] [--points N] [--masks MASK...]

Images are resized (never cropped) to --size, which must keep their aspect ratio, so the
predicted intrinsics scale back to full resolution by one factor per image. Frames (every
image but the ones under hero/) share one SIMPLE_PINHOLE camera with the median focal
length, like COLMAP with single_camera_per_folder; the hero gets its own. With --masks,
the 3D points (Brush's initialization) are kept only inside the subject masks.

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

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

MODEL = "facebook/map-anything-apache"  # Apache 2.0 weights (docs/FINDINGS.md, "Camera poses")


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
    ap.add_argument("--focal", type=float, help="known focal length of the frames, in their pixels")
    ap.add_argument("--hero-focal", type=float, help="known focal length of the hero image, in its pixels")
    args = ap.parse_args()
    t0 = time.monotonic()

    names = [str(p.resolve().relative_to(args.root.resolve())) for p in args.images]
    sizes = [Image.open(p).size for p in args.images]
    w_in, h_in = args.size
    for (w, h), name in zip(sizes, names):
        if abs(w / h - w_in / h_in) > 1e-3:
            raise SystemExit(f"{name} is {w}x{h}, not the aspect ratio of --size {w_in}x{h_in}")

    views = load_images([str(p) for p in args.images], resize_mode="fixed_size", size=(w_in, h_in))
    for view, (w, h), name in zip(views, sizes, names):
        f = args.hero_focal if name.startswith("hero/") else args.focal
        if f:  # known intrinsics, at model resolution (a pure resize: one scale factor)
            fm = f * w_in / w
            view["intrinsics"] = torch.tensor([[[fm, 0, w_in / 2], [0, fm, h_in / 2], [0, 0, 1]]], dtype=torch.float32)
    model = MapAnything.from_pretrained(MODEL).to("cuda").eval()
    t_load = time.monotonic()
    with torch.no_grad():
        outputs = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                              apply_mask=True, mask_edges=True)
    t_infer = time.monotonic()
    peak_gb = torch.cuda.max_memory_allocated() / 2**30

    hero = [n.startswith("hero/") for n in names]
    focals, poses = [], []
    for out, (w, h) in zip(outputs, sizes):
        k = out["intrinsics"][0].float().cpu().numpy()
        s = w / w_in  # model pixels -> image pixels (same in x and y)
        focals.append(float((k[0, 0] + k[1, 1]) / 2 * s))
        poses.append(out["camera_poses"][0].float().cpu().numpy())  # cam2world
    predicted_f = float(np.median([f for f, h_ in zip(focals, hero) if not h_]))
    # A known focal length was an input: the camera files carry it, not the model's estimate.
    frame_f = args.focal or predicted_f
    if args.hero_focal:
        focals = [args.hero_focal if h_ else f for f, h_ in zip(focals, hero)]

    args.out.mkdir(parents=True, exist_ok=True)
    frame_size = next(s for s, h_ in zip(sizes, hero) if not h_)
    cams = [f"1 SIMPLE_PINHOLE {frame_size[0]} {frame_size[1]} {frame_f:.6f} {frame_size[0] / 2} {frame_size[1] / 2}"]
    hero_idx = [i for i, h_ in enumerate(hero) if h_]
    if hero_idx:
        (w, h), f = sizes[hero_idx[0]], focals[hero_idx[0]]
        cams.append(f"2 SIMPLE_PINHOLE {w} {h} {f:.6f} {w / 2} {h / 2}")
    (args.out / "cameras.txt").write_text("# giro: MapAnything\n" + "\n".join(cams) + "\n")

    lines = []
    for i, (name, c2w) in enumerate(zip(names, poses)):
        r = c2w[:3, :3].T
        t = -r @ c2w[:3, 3]
        q = rotmat_to_qvec(r)
        lines.append(f"{i + 1} {' '.join(f'{x:.9f}' for x in q)} {' '.join(f'{x:.9f}' for x in t)} "
                     f"{2 if hero[i] else 1} {name}\n")  # no 2D observations
    (args.out / "images.txt").write_text("".join(line + "\n" for line in lines))

    # Brush initializes from the sparse points: sample the predicted point maps.
    masks = {m.resolve().relative_to(args.root.resolve() / "masks").as_posix(): m for m in args.masks}
    pts, cols = [], []
    for out, name in zip(outputs, names):
        p = out["pts3d"][0].float().cpu().numpy()
        valid = out["mask"][0].squeeze(-1).cpu().numpy().astype(bool) & (out["depth_z"][0].squeeze(-1).cpu().numpy() > 0)
        if name in masks:
            m = Image.open(masks[name]).convert("L").resize((w_in, h_in), Image.Resampling.NEAREST)
            valid &= np.asarray(m) > 127
        pts.append(p[valid])
        cols.append((out["img_no_norm"][0].float().cpu().numpy()[valid] * 255).astype(np.uint8))
    pts_all, cols_all = np.concatenate(pts), np.concatenate(cols)
    if len(pts_all) > args.points:
        keep = np.random.default_rng(0).choice(len(pts_all), args.points, replace=False)
        pts_all, cols_all = pts_all[keep], cols_all[keep]
    with open(args.out / "points3D.txt", "w") as f:
        for i, (p, c) in enumerate(zip(pts_all, cols_all)):
            f.write(f"{i + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0\n")

    (args.out / "stats.json").write_text(json.dumps({
        "images": len(names), "model_size": [w_in, h_in], "frame_focal_px": round(frame_f, 2),
        "predicted_focal_px": round(predicted_f, 2), "focal_given": bool(args.focal),
        "focal_spread": round(float(np.std([f for f, h_ in zip(focals, hero) if not h_]) / predicted_f), 4),
        "hero_focal_px": round(focals[hero_idx[0]], 2) if hero_idx else None,
        "points": len(pts_all), "load_s": round(t_load - t0, 1), "infer_s": round(t_infer - t_load, 1),
        "peak_vram_gb": round(peak_gb, 2),
    }, indent=2) + "\n")
    print(json.dumps({"ok": True, "infer_s": round(t_infer - t_load, 1), "peak_vram_gb": round(peak_gb, 2)}))


if __name__ == "__main__":
    main()
