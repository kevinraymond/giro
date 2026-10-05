"""Dense multi-view stereo on the registered source views, the photogrammetry way: COLMAP's
PatchMatch stereo (CUDA) and fusion over the hero and the anchors with the cameras we solved
(register_photometric.py) and each anchor's warp baked in (warp_views.py), then a comparison with
the proxy mesh: fused points the mesh lacks (a fork in front of a wheel, mirror stalks) become
WORK/dense/extra.npz for project_texture.py --extra.

Unlike one depth guess per image (thin_parts.py, MoGe), PatchMatch keeps a depth only where
several views agree on the colors there, and fusion only where the depths agree in 3D.

    dense_views.py ATTEMPT ANCHOR_DIR WORK GPU --skip 08,10,...

Writes WORK/dense/ (the COLMAP workspace, fused.ply, extra.npz, dense.json, new.jpg).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from giro import path as campath
from giro import splat
from giro.stages.fallback import _qvec
from texture_common import PoseCamera, Source, anchor_mask, bake, load_cameras, render_points, samples

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("anchor_dir", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--skip", default="02,05")
ap.add_argument("--neighbors", type=int, default=6)
ap.add_argument("--max-angle", type=float, default=60.0, help="neighbors look within this many degrees")
ap.add_argument("--band", type=float, default=0.25, help="depth search this far around the mesh (proxy units)")
ap.add_argument("--min-views", type=int, default=3, help="fusion: a point needs this many views")
ap.add_argument("--off", type=float, default=0.015, help="fused points this far from the mesh are new geometry")
ap.add_argument("--max-off", type=float, default=0.12)
ap.add_argument("--size", type=int, default=1200, help="PatchMatch's max image size")
ap.add_argument("--reuse", action="store_true", help="keep an earlier run's PatchMatch depth maps; redo the model and fusion")
ap.add_argument("--fusion", default="", help="extra stereo_fusion options, e.g. '--StereoFusion.max_reproj_error 4'")
args = ap.parse_args()
attempt, adir, work = args.attempt.resolve(), args.anchor_dir.resolve(), args.work.resolve()
dev = torch.device(f"cuda:{args.gpu}")
out = work / "dense"
if out.exists() and not args.reuse:
    shutil.rmtree(out)
for d in ("sparse", "images", "masks"):
    (out / d).mkdir(parents=True, exist_ok=True)

fits = json.loads((work / "texture.json").read_text())["fits"]
cams = load_cameras(work / "anchor_cameras.json")
warps = torch.load(work / "warps.pt") if (work / "warps.pt").exists() else {}
skip = set(args.skip.split(","))
views = [("hero", Image.open(attempt / "hero" / "hero.png").convert("RGB"),
          np.asarray(Image.open(attempt / "proxy" / "hero_mask.png")) > 127,
          PoseCamera.of(campath.PathCamera.from_json(fits["hero"]["camera"])))]
views += [(n, Image.open(adir / f"{n}.png").convert("RGB"), anchor_mask(adir, n), cams[n]) for n in sorted(cams) if n not in skip]
xyz, _, _, _ = samples(work, 3_000_000, dev)

# 1. Images with the warps baked in (each pixel resampled from where the warp says it lies), the
#    background blacked out, and the masks fusion uses.
lines_c, lines_i, depth = [], [], {}
for k, (name, img, mask, cam) in enumerate(views):
    w, h = img.size
    pix, m = bake(Source(name, img, mask, cam, 1.0, dev, warp=warps.get(name)))
    Image.fromarray(pix).save(out / "images" / f"{name}.png")
    Image.fromarray(m.astype(np.uint8) * 255).save(out / "masks" / f"{name}.png.png")  # fusion: <image name>.png
    f = cam.focal(w, h)
    lines_c.append(f"{k + 1} PINHOLE {w} {h} {f} {f} {w / 2} {h / 2}\n")
    rot, t = cam.world_to_camera()
    lines_i.append(f"{k + 1} {' '.join(map(str, _qvec(np.asarray(rot))))} {' '.join(map(str, t))} {k + 1} {name}.png\n\n")
    pc = xyz @ torch.tensor(np.asarray(rot), device=dev, dtype=torch.float32).T + torch.tensor(np.asarray(t), device=dev, dtype=torch.float32)
    depth[name] = (float(pc[:, 2].min()), float(pc[:, 2].max()))
# Sparse points with tracks: COLMAP's fusion picks the images to fuse together by the points they
# share (with none it fuses nothing). Mesh samples, each tracked in the views that see it.
from texture_common import project, zbuffer  # noqa: E402

track_pts = xyz[torch.randperm(len(xyz), generator=torch.Generator().manual_seed(1))[:20000].to(dev)]
tracks: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(track_pts))}
seen_by = []
for k, (name, img, _, cam) in enumerate(views):
    w, h = img.size
    u, v, z = project(cam, w, h, xyz)
    zb = zbuffer(u, v, z, w, h)[3]
    tu, tv, tz = project(cam, w, h, track_pts)
    ok = (tz > 1e-3) & (tu >= 0) & (tu < w) & (tv >= 0) & (tv < h)
    seen = ok.clone()
    seen[ok] = tz[ok] <= zb[tv[ok].long(), tu[ok].long()] + 0.005
    ids = torch.nonzero(seen)[:, 0].cpu().numpy()
    for j, pid in enumerate(ids):
        tracks[int(pid)].append((k + 1, j))
    seen_by.append((ids, tu[seen].cpu().numpy(), tv[seen].cpu().numpy()))
obs_lines = [" ".join(f"{x:.2f} {y:.2f} {int(pid) + 1 if len(tracks[int(pid)]) >= 2 else -1}" for pid, x, y in zip(ids, us, vs))
             for ids, us, vs in seen_by]
lines_i = [ln.replace("\n\n", "\n" + o + "\n") for ln, o in zip(lines_i, obs_lines)]
pcols = (np.full(3, 128)).tolist()
(out / "sparse" / "cameras.txt").write_text("".join(lines_c))
(out / "sparse" / "images.txt").write_text("".join(lines_i))
(out / "sparse" / "points3D.txt").write_text("".join(
    f"{i + 1} {' '.join(f'{x:.6f}' for x in track_pts[i].tolist())} {' '.join(map(str, pcols))} 0 "
    f"{' '.join(f'{a} {b}' for a, b in tr)}\n" for i, tr in tracks.items() if len(tr) >= 2))

# 2. COLMAP: undistort (identity for pinhole; it lays out the workspace), PatchMatch, fusion.
env = os.environ | {"CUDA_VISIBLE_DEVICES": str(args.gpu)}


def colmap(*a: str) -> None:
    r = subprocess.run(["colmap", *a], env=env, capture_output=True, text=True)
    (out / "colmap.log").open("a").write(r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"colmap {a[0]} failed; see {out / 'colmap.log'}")


if args.reuse and (out / "ws" / "stereo" / "depth_maps").exists():
    shutil.rmtree(out / "ws" / "sparse")
    (out / "ws" / "sparse").mkdir()
    colmap("model_converter", "--input_path", str(out / "sparse"), "--output_path", str(out / "ws" / "sparse"), "--output_type", "BIN")
else:
    colmap("image_undistorter", "--image_path", str(out / "images"), "--input_path", str(out / "sparse"),
           "--output_path", str(out / "ws"), "--output_type", "COLMAP")
# Neighbors: the views looking most alike in direction (no sparse points to choose them from).
fwd = {n: np.asarray(c.world_to_camera()[0])[2] for n, _, _, c in views}
cfg = []
for n, *_ in views:
    ang = {o: math.degrees(math.acos(np.clip(fwd[n] @ fwd[o], -1, 1))) for o in fwd if o != n}
    near = [o for o in sorted(ang, key=ang.get) if ang[o] <= args.max_angle][: args.neighbors]
    cfg.append(f"{n}.png\n{', '.join(f'{o}.png' for o in near) if near else '__all__'}\n")
(out / "ws" / "stereo" / "patch-match.cfg").write_text("".join(cfg))
lo = min(d[0] for d in depth.values()) - args.band
hi = max(d[1] for d in depth.values()) + args.band
if not (args.reuse and (out / "ws" / "stereo" / "depth_maps").exists()):
    colmap("patch_match_stereo", "--workspace_path", str(out / "ws"), "--PatchMatchStereo.geom_consistency", "1",
           "--PatchMatchStereo.depth_min", str(max(lo, 0.05)), "--PatchMatchStereo.depth_max", str(hi),
           "--PatchMatchStereo.max_image_size", str(args.size), "--PatchMatchStereo.gpu_index", "0")
colmap("stereo_fusion", "--workspace_path", str(out / "ws"), "--input_type", "geometric",
       "--output_path", str(out / "fused.ply"), "--StereoFusion.mask_path", str(out / "masks"),
       "--StereoFusion.min_num_pixels", str(args.min_views), *args.fusion.split())

# 3. Against the mesh: fused points off its surface (but near it) are what it lacks.
ply = splat.read_ply(out / "fused.ply")
pts = np.stack([ply["x"], ply["y"], ply["z"]], 1).astype(np.float32)
cols = np.stack([ply["red"], ply["green"], ply["blue"]], 1).astype(np.float32) / 255
dist, _ = cKDTree(xyz.cpu().numpy()).query(pts)
new = (dist > args.off) & (dist < args.max_off)
np.savez_compressed(out / "extra.npz", xyz=pts[new], rgb=cols[new])
report = {"fused": int(len(pts)), "on_mesh": int((dist <= args.off).sum()), "new": int(new.sum()),
          "far": int((dist >= args.max_off).sum()), "median_dist": round(float(np.median(dist)), 4),
          "depth_range": [round(lo, 3), round(hi, 3)]}
(out / "dense.json").write_text(json.dumps(report, indent=1))
print(report, flush=True)

# A look: the mesh in gray, fused points on it in green, new ones in red, from four sides.
hero = campath.PathCamera.from_json(fits["hero"]["camera"])
all_xyz = torch.cat([xyz, torch.from_numpy(pts).to(dev)])
vals = torch.cat([torch.full((len(xyz), 3), 0.35, device=dev),
                  torch.from_numpy(np.where(new[:, None], [1.0, 0.1, 0.1], [0.2, 0.9, 0.3]).astype(np.float32)).to(dev)])
tiles = []
for yaw in (0, 90, 180, 270):
    img, _ = render_points(all_xyz, vals, campath.PathCamera(hero.yaw + yaw, 10.0, hero.distance, hero.target, hero.fov), 600, 800)
    tiles.append(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)))
sheet = Image.new("RGB", (2400, 800))
for i, t in enumerate(tiles):
    sheet.paste(t, (i * 600, 0))
sheet.save(out / "new.jpg", quality=88)
