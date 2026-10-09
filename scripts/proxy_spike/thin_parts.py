"""Parts the proxy mesh lacks (a scooter's front fork, mirror stalks, kickstand), from per-view
depth: MoGe-2 (MIT) estimates each source view's depth, aligned to the mesh's own depth from that
camera (scale and shift, robust fit inside the subject's mask). Where a view's surface is clearly
nearer than the mesh, or the mesh has nothing at all, the view sees something the mesh does not
have. Those pixels

  - are kept off the mesh when painting (WORK/thin/exclude/<name>.png, project_texture.py
    --exclude): the fork in front of the wheel no longer lands on the wheel disc, and
  - are lifted to 3D at their aligned depth and kept where other views agree (depth-map
    fusion): WORK/thin/extra.npz (xyz, rgb), which project_texture.py --extra renders with the
    mesh, so the splat gets the fork.

    thin_parts.py ATTEMPT ANCHOR_DIR WORK GPU --skip 08,10,...

Reads WORK/texture.json (hero fit), WORK/anchor_cameras.json; writes WORK/thin/ (depth/*.npz,
exclude/*.png, extra.npz, flagged.jpg, thin.json).
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient
from texture_common import PoseCamera, anchor_mask, load_cameras, project, samples, zbuffer

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("anchor_dir", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--skip", default="02,05")
ap.add_argument("--near", type=float, default=0.03, help="nearer than the mesh by this much (proxy units, ~1 tall)")
ap.add_argument("--agree", type=float, default=0.015, help="another view's depth within this much confirms a point")
ap.add_argument("--votes", type=int, default=2, help="other views that must confirm a point")
ap.add_argument("--max-off", type=float, default=0.12, help="points farther than this from the mesh are dropped")
ap.add_argument("--min-px", type=int, default=40, help="flagged pieces smaller than this are noise")
ap.add_argument("--smooth-px", type=float, default=25.0, help="the depth difference's smooth part, removed first")
ap.add_argument("--edge-px", type=float, default=5.0, help="no-mesh pixels count only this far outside the mesh")
ap.add_argument("--edge-band", type=int, default=4, help="pixels around the mesh's depth edges not flagged as nearer")
ap.add_argument("--max-half-width", type=float, default=15.0, help="flagged pieces wider than twice this are shape mismatch")
args = ap.parse_args()
attempt, adir, work = args.attempt.resolve(), args.anchor_dir.resolve(), args.work.resolve()
dev = torch.device(f"cuda:{args.gpu}")
out = work / "thin"
for d in ("depth", "exclude"):
    (out / d).mkdir(parents=True, exist_ok=True)

fits = json.loads((work / "texture.json").read_text())["fits"]
cams = load_cameras(work / "anchor_cameras.json")
skip = set(args.skip.split(","))
views = [("hero", attempt / "hero" / "hero.png", np.asarray(Image.open(attempt / "proxy" / "hero_mask.png")) > 127,
          PoseCamera.of(campath.PathCamera.from_json(fits["hero"]["camera"])))]
views += [(n, adir / f"{n}.png", anchor_mask(adir, n), cams[n]) for n in sorted(cams) if n not in skip]


# 1. MoGe depth per view, at the view's own field of view (the fov is over the smaller side, the width).
async def depth() -> None:
    todo = [v for v in views if not (out / "depth" / f"{v[0]}.npz").exists()]
    if not todo:
        return
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                wf = {"moge": {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge_2_vitl_normal_fp16.safetensors"}}}
                for name, path, _, cam in todo:
                    image = await comfy.upload_image(path)
                    wf |= {
                        f"img_{name}": {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}},
                        f"geo_{name}": {"class_type": "MoGeInference", "inputs": {
                            "moge_model": ["moge", 0], "image": [f"img_{name}", 0], "resolution_level": 9,
                            "fov_x_degrees": cam.fov, "batch_size": 1, "force_projection": True, "apply_mask": True,
                            "refine_steps": 0}},
                        f"save_{name}": {"class_type": "GiroSaveMoGe", "inputs": {
                            "moge_geometry": [f"geo_{name}", 0], "path": str(out / "depth" / f"{name}.npz")}},
                    }
                async for _ in comfy.run(wf):
                    pass
            finally:
                await comfy.free()

asyncio.run(depth())
xyz, _, _, _ = samples(work, 3_000_000, dev)


def robust_fit(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """s, t minimizing |s a + t - b| over the inliers (three trimmed least-squares passes)."""
    s, t = float(np.median(b) / np.median(a)), 0.0
    keep = np.ones(len(a), bool)
    for _ in range(3):
        r = s * a + t - b
        mad = np.median(np.abs(r - np.median(r))) + 1e-6
        keep = np.abs(r - np.median(r)) < 3 * mad
        A = np.stack([a[keep], np.ones(keep.sum())], 1)
        s, t = np.linalg.lstsq(A, b[keep], rcond=None)[0]
    return float(s), float(t)


# 2. Align each view's depth to the mesh, flag what the mesh lacks, lift it to 3D.
per_view, report, tiles = [], {}, []
for name, path, mask, cam in views:
    img = np.asarray(Image.open(path).convert("RGB"), np.float32) / 255
    h, w = mask.shape
    g = np.load(out / "depth" / f"{name}.npz")
    d = g["depth"]
    if d.shape != (h, w):
        d = np.asarray(Image.fromarray(d.astype(np.float32)).resize((w, h), Image.NEAREST))
    u, v, z = project(cam, w, h, xyz)
    mesh = zbuffer(u, v, z, w, h)[3].cpu().numpy()
    inner = ndimage.binary_erosion(mask, iterations=4)
    ok = inner & np.isfinite(d) & (d > 0)
    both = ok & np.isfinite(mesh)
    s, t = robust_fit(d[both], mesh[both])
    aligned = s * d + t
    res = np.median(np.abs(aligned[both] - mesh[both]))
    # Only local differences: MoGe's depth bends away from the mesh's smoothly (near parts come out
    # nearer in every view), so the difference's own smooth part is removed first (normalized
    # Gaussian over the pixels both have), and what is left must be narrow.
    diff = np.where(both, aligned - mesh, 0.0)
    low = ndimage.gaussian_filter(diff, args.smooth_px) / np.maximum(ndimage.gaussian_filter(both.astype(float), args.smooth_px), 1e-6)
    nearer = both & (diff - low < -args.near)
    # Not at the mesh's own depth edges (panel outlines, tire rims): both depths blur across them.
    m_f = np.where(np.isfinite(mesh), mesh, np.nanmax(np.where(np.isfinite(mesh), mesh, np.nan)) + 1.0)
    edge = np.hypot(*np.gradient(m_f)) > args.near / 2
    nearer &= ~ndimage.binary_dilation(edge, iterations=args.edge_band)
    # No mesh at all: only clear of the mesh's outline (a band there is just silhouettes disagreeing).
    off_mesh = ndimage.distance_transform_edt(~np.isfinite(mesh)) > args.edge_px
    missing = ok & ~np.isfinite(mesh) & off_mesh
    flag = ndimage.binary_opening(nearer | missing, iterations=1)
    lab, n = ndimage.label(flag)
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum(flag, lab, idx)
    thick = ndimage.maximum(ndimage.distance_transform_edt(flag), lab, idx)  # half the widest width
    flag = np.isin(lab, idx[(sizes >= args.min_px) & (thick <= args.max_half_width)])
    Image.fromarray(ndimage.binary_dilation(flag, iterations=3).astype(np.uint8) * 255).save(out / "exclude" / f"{name}.png")
    ys, xs = np.nonzero(flag)
    f = cam.focal(w, h)
    zz = aligned[ys, xs]
    pc = np.stack([(xs + 0.5 - w / 2) / f * zz, (ys + 0.5 - h / 2) / f * zz, zz], 1)
    r, tv = cam.world_to_camera()
    world = (pc - tv) @ r
    per_view.append((name, cam, w, h, aligned, mask, world, img[ys, xs]))
    report[name] = {"scale": round(s, 4), "shift": round(t, 4), "median_residual": round(float(res), 4),
                    "nearer_px": int(nearer.sum()), "missing_px": int(missing.sum()), "flagged_px": int(flag.sum())}
    th = 320
    tw = round(th * w / h)
    over = img.copy()
    over[flag] = over[flag] * 0.3 + np.array([1.0, 0.0, 0.8]) * 0.7
    tiles.append(Image.fromarray((over * 255).astype(np.uint8)).resize((tw, th)))
    print(f"{name}: depth fit residual {res:.4f}, flagged {flag.sum():,} px (nearer {nearer.sum():,}, no mesh {missing.sum():,})", flush=True)

# 3. Keep the lifted points other views confirm (their aligned depth agrees where they see them).
mesh_tree = cKDTree(xyz[::10].cpu().numpy())
keep_xyz, keep_rgb, support = [], [], []
for i, (name, cam, w, h, aligned, mask, world, rgb) in enumerate(per_view):
    if not len(world):
        continue
    votes = np.zeros(len(world), int)
    pts = torch.from_numpy(world.astype(np.float32)).to(dev)
    for j, (_, cam_j, wj, hj, aligned_j, mask_j, _, _) in enumerate(per_view):
        if j == i:
            continue
        uj, vj, zj = (x.cpu().numpy() for x in project(cam_j, wj, hj, pts))
        inside = (zj > 1e-3) & (uj >= 0) & (uj < wj) & (vj >= 0) & (vj < hj)
        ui, vi = uj[inside].astype(int), vj[inside].astype(int)
        dj = aligned_j[vi, ui]
        agree = mask_j[vi, ui] & np.isfinite(dj) & (np.abs(dj - zj[inside]) < args.agree)
        votes[np.nonzero(inside)[0][agree]] += 1
    good = votes >= args.votes
    if good.any():  # thin parts hang on the body (a fork by its wheel); far ones are depth noise
        good[good] &= mesh_tree.query(world[good])[0] < args.max_off
    keep_xyz.append(world[good])
    keep_rgb.append(rgb[good])
    support.append(votes[good])
    report[name]["lifted"] = int(len(world))
    report[name]["confirmed"] = int(good.sum())
xyz_extra = np.concatenate(keep_xyz) if keep_xyz else np.zeros((0, 3))
rgb_extra = np.concatenate(keep_rgb) if keep_rgb else np.zeros((0, 3))
np.savez_compressed(out / "extra.npz", xyz=xyz_extra.astype(np.float32), rgb=rgb_extra.astype(np.float32),
                    support=np.concatenate(support) if support else np.zeros(0, int))
(out / "thin.json").write_text(json.dumps(report, indent=1))
cols = 7
W = max(t.width for t in tiles)
sheet = Image.new("RGB", (W * cols, 320 * ((len(tiles) + cols - 1) // cols)), (40, 40, 40))
for k, tile in enumerate(tiles):
    sheet.paste(tile, ((k % cols) * W, (k // cols) * 320))
sheet.save(out / "flagged.jpg", quality=88)
print(f"{len(xyz_extra):,} confirmed points -> {out / 'extra.npz'}; {out / 'flagged.jpg'}", flush=True)
