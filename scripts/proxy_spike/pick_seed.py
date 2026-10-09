"""Pick the proxy seed that agrees best with the registered anchor views: each candidate mesh
(proxy_mesh.py's mesh_<model>_<seed>.npz) gets every anchor's camera refined against its
silhouette (from register_anchors.py's fit to the attempt's proxy), then is scored by edge
agreement: how many of the anchors' strong image edges lie on the mesh's depth edges, and the
reverse (F1, 3 px). The image-to-3D models give a different shape per seed; on the scooter some
model the front fork and shock and some do not. Silhouette IoU (within 0.004 over six seeds) and
color disagreement after painting do not see a 1-2 cm fork; edges do, weakly (Oct 5: F1 0.393-
0.408, the fork seeds 48 and 50 on top, 46 without one below). So the pick is a ranking, and
SEEDS_DIR/seeds.jpg (shaded meshes, front-wheel close-ups) is there to overrule it by eye
(--seed).

    pick_seed.py ATTEMPT ANCHOR_DIR SEEDS_DIR WORK GPU [--min-iou 0.75] [--seed mesh_pixal3d_50.npz]

Writes SEEDS_DIR/seeds.json (scores) and sets up WORK for project_texture.py: mesh.npz (the
winner) and candidates/proxy_<attempt seed>.ply (its points, which the camera fits read).
"""
import argparse
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from scipy import ndimage, optimize

from giro import path as campath
from giro.stages.proxy import Points, fit_hero_camera, iou
from texture_common import anchor_mask, project, render_points, samples, write_points_ply, zbuffer

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("anchor_dir", type=Path)
ap.add_argument("seeds_dir", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--min-iou", type=float, default=0.75, help="anchors registered worse than this to the proxy are left out")
ap.add_argument("--height", type=int, default=256, help="silhouettes are compared at this height")
ap.add_argument("--seed", help="take this mesh file instead of the best score (still scores all)")
args = ap.parse_args()
attempt, adir, sdir, work = (p.resolve() for p in (args.attempt, args.anchor_dir, args.seeds_dir, args.work))
dev = torch.device(f"cuda:{args.gpu}")
reg = json.loads((adir / "registration.json").read_text())
anchors = [n for n in sorted(reg) if reg[n]["iou"] >= args.min_iou]
proxy = json.loads((attempt / "proxy" / "proxy.json").read_text())
hero_img = Image.open(attempt / "hero" / "hero.png").convert("RGB")
hero_mask = np.asarray(Image.open(attempt / "proxy" / "hero_mask.png")) > 127
masks = {}
for n in anchors:
    m = anchor_mask(adir, n)
    size = (round(args.height * m.shape[1] / m.shape[0]), args.height)
    masks[n] = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size)) > 127


def refine(points: Points, cam: campath.PathCamera, mask: np.ndarray) -> tuple[float, campath.PathCamera]:
    h, w = mask.shape
    rot, _ = cam.world_to_camera()

    def moved(x: np.ndarray) -> campath.PathCamera:
        target = np.asarray(cam.target) + x[3] * cam.distance * rot[0] + x[4] * cam.distance * rot[1]
        return replace(cam, yaw=cam.yaw + x[0], pitch=cam.pitch + x[1], distance=cam.distance * math.exp(x[2]),
                       target=tuple(target))

    res = optimize.minimize(lambda x: -iou(points.silhouette(moved(x), w, h), mask), np.zeros(5), method="Nelder-Mead",
                            options={"initial_simplex": np.vstack([np.zeros(5), np.diag([3.0, 3.0, 0.05, 0.02, 0.02])]),
                                     "maxiter": 150, "xatol": 0.05, "fatol": 1e-4})
    return -float(res.fun), moved(res.x)


def edge_f1(xyz: torch.Tensor, cam: campath.PathCamera, name: str) -> float:
    """Anchor image edges (top 10% gradient in the mask) against the mesh's depth edges, 3 px."""
    img = np.asarray(Image.open(adir / f"{name}.png").convert("L"), np.float32) / 255
    h, w = img.shape
    mask = ndimage.binary_erosion(anchor_mask(adir, name), iterations=6)
    u, v, z = project(cam, w, h, xyz)
    d = zbuffer(u, v, z, w, h)[3].cpu().numpy()
    fin = np.isfinite(d)
    d = np.where(fin, d, d[fin].max() + 0.5)
    medge = (np.hypot(ndimage.sobel(d, 0), ndimage.sobel(d, 1)) > 0.04) & mask
    g = ndimage.gaussian_filter(img, 1.2)
    gi = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    iedge = (gi > np.percentile(gi[mask], 90)) & mask
    if not medge.any() or not iedge.any():
        return 0.0
    rec = (ndimage.distance_transform_edt(~medge)[iedge] <= 3).mean()
    prec = (ndimage.distance_transform_edt(~iedge)[medge] <= 3).mean()
    return float(2 * rec * prec / max(rec + prec, 1e-9))


scores = {}
for mesh in sorted(sdir.glob("mesh_*.npz")):
    ply = sdir / f"points_{mesh.stem}.ply"
    if not ply.exists():
        write_points_ply(sdir, ply, mesh=mesh.name)
    points = Points(ply)
    keep = slice(None, None, 3)  # a third of the points: the silhouettes stay closed at this size
    points.xyz, points.rgb = points.xyz[keep], points.rgb[keep]
    hero_cam, fit = fit_hero_camera(points, hero_img, hero_mask, proxy["hero_camera"]["fov"])
    fitted = {n: refine(points, campath.PathCamera.from_json(reg[n]["camera"]), masks[n]) for n in anchors}
    per = {n: round(f[0], 4) for n, f in fitted.items()}
    xyz, _, _, _ = samples(sdir, 3_000_000, dev, mesh.name)
    edges = {n: round(edge_f1(xyz, f[1], n), 4) for n, f in fitted.items()}
    score = float(np.mean(list(edges.values())))
    scores[mesh.name] = {"score": round(score, 4), "hero_iou": fit["iou"], "anchor_iou_mean": round(float(np.mean(list(per.values()))), 4),
                         "edge_f1": edges, "anchor_iou": per}
    print(f"{mesh.name}: edge F1 {score:.4f} (silhouette IoU: hero {fit['iou']:.3f}, anchors {np.mean(list(per.values())):.3f})", flush=True)

# A contact sheet to overrule the pick by eye: each seed shaded, whole and at the front wheel.
hero = campath.PathCamera.from_json(proxy["hero_camera"])
light = torch.tensor([0.4, -0.7, -0.6], device=dev)
light /= light.norm()
rows = []
for name in sorted(scores, key=lambda k: -scores[k]["score"]):
    xyz, nrm, _, _ = samples(sdir, 3_000_000, dev, name)
    shade = (0.25 + 0.75 * (nrm @ light).abs())[:, None].repeat(1, 3)
    row = Image.new("RGB", (4 * 300, 400))
    for k, (yaw, zoom) in enumerate(((0, 1.0), (300, 0.55), (60, 0.55), (180, 1.0))):
        tgt = np.asarray(hero.target) + (np.array([0, 0.25, -0.25]) if zoom < 1 else 0)
        cam = campath.PathCamera(hero.yaw + yaw, 5.0, hero.distance * zoom, tuple(tgt.tolist()), hero.fov)
        img, _ = render_points(xyz, shade, cam, 300, 400, nrm=nrm)
        row.paste(Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)), (k * 300, 0))
    ImageDraw.Draw(row).text((6, 6), f"{name}  edge F1 {scores[name]['score']:.4f}", fill=(255, 255, 0))
    rows.append(row)
sheet = Image.new("RGB", (1200 * 2, 400 * ((len(rows) + 1) // 2)))
for i, r in enumerate(rows):
    sheet.paste(r, ((i % 2) * 1200, (i // 2) * 400))
sheet.save(sdir / "seeds.jpg", quality=88)

best = args.seed or max(scores, key=lambda k: scores[k]["score"])
(sdir / "seeds.json").write_text(json.dumps({"best": best, "anchors": anchors, "scores": scores}, indent=1))
work.mkdir(parents=True, exist_ok=True)
shutil.copy(sdir / best, work / "mesh.npz")
(work / "candidates").mkdir(exist_ok=True)
shutil.copy(sdir / f"points_{Path(best).stem}.ply", work / "candidates" / f"proxy_{proxy['seed']}.ply")
print(f"best: {best} -> {work}", flush=True)
