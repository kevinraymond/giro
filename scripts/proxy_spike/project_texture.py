"""Projection texturing: paint a proxy mesh from the hero and registered anchor views, the way
photogrammetry textures its mesh from the photos, then render a dense multi-ring orbit of it and
lay that out as an attempt the stages can train (dataset -> export). No video model.

    project_texture.py ATTEMPT ANCHOR_DIR WORK GPU [--cameras WORK/anchor_cameras.json]
        [--select-power 6] [--level-voxel 0.1,0.03] [--out attempt]   (--help for all)

WORK holds proxy_mesh.py's output (mesh.npz, candidates/proxy_<seed>.ply). Steps:

1. The mesh in the splat frame (as GiroMeshToSplat moves it), sampled densely by area: each
   sample keeps its face normal and the mesh's own color (the fallback where no view sees it).
2. The hero's camera is fitted to this mesh (the proxy stage's fit); each anchor's camera is
   refined from its registration (registration.json) against this mesh's silhouette.
3. Every source view sees a sample where it lands inside the view's mask and no nearer sample
   covers that pixel (a z-buffer). Its weight there is |cos| of the viewing angle to the 4th
   power times a feather from the mask's edge (the hero counts double, it is the real photo).
   Each anchor's colors are matched to the hero (per channel gain and offset) on the samples
   both see well. Colors blend with the weights squared, so the best-facing view dominates and
   seams are feathered.
4. RINGS of cameras around the hero's target render the painted samples (z-buffer, the nearest
   layer averaged) as frames/, masks/frames/, with exact COLMAP poses in poses/colmap; the hero
   goes in as the hero. WORK/attempt is then trained by `giro stages --from dataset`.

Writes WORK/sources.jpg (each source with its refined fit outlined), WORK/coverage.jpg (renders
colored by which source painted them), WORK/texture.json (fits, gains, coverage).
"""
import argparse
import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

from giro import path as campath
from giro.stages.fallback import _qvec
from giro.stages.proxy import Points, fit_hero_camera, iou
from texture_common import Source, anchor_mask, fingerprint, load_cameras, render_points, samples, smooth_field

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("anchor_dir", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--skip", default="02,05", help="anchors left out")
ap.add_argument("--points", type=int, default=6_000_000, help="surface samples")
ap.add_argument("--paint-tol", type=float, default=0.012, help="depth tolerance for painting, proxy units")
ap.add_argument("--cameras", type=Path, help="register_photometric.py's anchor cameras (else silhouette fits)")
ap.add_argument("--warps", type=Path, help="warp_views.py's per-view warp grids")
ap.add_argument("--exclude", type=Path, help="thin_parts.py's exclude/ masks: those pixels paint nothing")
ap.add_argument("--extra", type=Path, help="thin_parts.py's extra.npz: points the mesh lacks, rendered with it")
ap.add_argument("--select-power", type=float, default=0.0,
                help="view selection: weights times their local average (1 cm voxels, sigma 2) to this power, "
                     "so each region takes its regionally best view (0: off)")
ap.add_argument("--level-voxel", default="",
                help="seam leveling: each anchor's low frequencies (voxels this size, sigma 1.5) pulled to the "
                     "other views' blend, coarse to fine for a list (0.1,0.03); the hero is kept (empty: off)")
ap.add_argument("--views", type=Path, help="paint from this attempt's frames (frames/, masks/frames/, "
                "cameras.json; e.g. enhanced renders) instead of the anchors; the hero stays")
ap.add_argument("--views-every", type=int, default=2, help="with --views, every Nth frame")
ap.add_argument("--save-texture", type=Path, help="write the painted samples' colors here (patch_region.py reads them)")
ap.add_argument("--texture", type=Path, help="render this saved texture (patch_region.py's output) instead of painting")
ap.add_argument("--rings", help="cameras to render, PITCH:VIEWS,... (default -20:24,0:48,20:48,40:36,60:24,80:8)")
ap.add_argument("--out", default="attempt", help="attempt directory name under WORK")
ap.add_argument("--exposure", choices=["match", "off"], default="match",
                help="match each anchor's color spread to the hero's (or the views matched before); off: as generated. "
                     "Oct 5, knight: on polished armor nearly every gain hit the 0.7 clamp, compounding down the "
                     "chain to the rear anchors (dark, flat)")
ap.add_argument("--mesh-fallback", default="", metavar="LO,HI",
                help="where the views disagree (the w-weighted spread of their colors, before view selection) more than "
                     "LO, blend toward the mesh's own colors, fully at HI (e.g. 0.08,0.16; empty: off)")
ap.add_argument("--supersample", type=int, default=2,
                help="render the frames this many times larger and shrink them: antialiased edges, and soft masks "
                     "(coverage) to train with dataset.mask_erode_px=0 (Oct 5, tank: cleaner silhouettes; 1: hard edges)")
args = ap.parse_args()
attempt, adir, work = args.attempt.resolve(), args.anchor_dir.resolve(), args.work.resolve()
skip = set(args.skip.split(","))
n_points = args.points
cameras_file = args.cameras.resolve() if args.cameras else None
dev = torch.device(f"cuda:{args.gpu}")
FRAME_SIZE = (768, 1024)
RINGS = [(-20, 24), (0, 48), (20, 48), (40, 36), (60, 24), (80, 8)]  # (pitch, views); + looks down
if args.rings:
    RINGS = [(float(p), int(n)) for p, n in (r.split(":") for r in args.rings.split(","))]
FIT_REFINE_HEIGHT = 512
HERO_WEIGHT = 2.0
FEATHER_PX = 8
seed = json.loads((attempt / "proxy" / "proxy.json").read_text())["seed"]
report: dict = {}

# 1. The mesh, in the splat frame, sampled by area.
xyz, nrm, base_rgb, spacing = samples(work, n_points, dev)
print(f"{n_points:,} samples, spacing {spacing:.5f}", flush=True)

# 2. Cameras fitted to this mesh.
points = Points(work / "candidates" / f"proxy_{seed}.ply")
hero_img = Image.open(attempt / "hero" / "hero.png").convert("RGB")
hero_mask = np.asarray(Image.open(attempt / "proxy" / "hero_mask.png")) > 127
fov = json.loads((attempt / "proxy" / "proxy.json").read_text())["hero_camera"]["fov"]


def refine(cam: campath.PathCamera, mask: np.ndarray) -> tuple[campath.PathCamera, float]:
    h0, w0 = mask.shape
    size = (round(FIT_REFINE_HEIGHT * w0 / h0), FIT_REFINE_HEIGHT)
    mk = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size)) > 127
    rot, _ = cam.world_to_camera()

    def moved(x: np.ndarray) -> campath.PathCamera:
        target = np.asarray(cam.target) + x[3] * cam.distance * rot[0] + x[4] * cam.distance * rot[1]
        return replace(cam, yaw=cam.yaw + x[0], pitch=cam.pitch + x[1], distance=cam.distance * math.exp(x[2]),
                       target=tuple(target))

    res = optimize.minimize(lambda x: -iou(points.silhouette(moved(x), *size), mk), np.zeros(5), method="Nelder-Mead",
                            options={"initial_simplex": np.vstack([np.zeros(5), np.diag([2.0, 2.0, 0.03, 0.01, 0.01])]),
                                     "maxiter": 300, "xatol": 0.02, "fatol": 1e-5})
    return moved(res.x), -res.fun


hero_cam, fit = fit_hero_camera(points, hero_img, hero_mask, fov)
hero_cam, hero_iou = refine(hero_cam, hero_mask)
print(f"hero: yaw {hero_cam.yaw:.1f} pitch {hero_cam.pitch:.1f} IoU {hero_iou:.3f}", flush=True)
sources = [("hero", hero_img, hero_mask, hero_cam, HERO_WEIGHT)]
reg = {} if args.views or args.texture else json.loads((adir / "registration.json").read_text())
solved = load_cameras(cameras_file) if cameras_file else {}
if args.views:
    vdir = args.views.resolve()
    for k, c in enumerate(json.loads((vdir / "cameras.json").read_text())["frames"]):
        if k % args.views_every == 0:
            name = f"frames/{k:05d}.png"
            sources.append((name, Image.open(vdir / name).convert("RGB"), np.asarray(Image.open(vdir / "masks" / name)) > 127,
                            campath.PathCamera.from_json(c), 1.0))
    print(f"{len(sources) - 1} views from {vdir.name}", flush=True)
for name in sorted(reg):
    if name in skip:
        continue
    mask = anchor_mask(adir, name)
    if name in solved:
        cam = solved[name]
        print(f"{name}: solved camera ({cameras_file.name})", flush=True)
    else:
        cam, fit_iou = refine(campath.PathCamera.from_json(reg[name]["camera"]), mask)
        print(f"{name}: yaw {cam.yaw:.1f} pitch {cam.pitch:.1f} IoU {reg[name]['iou']:.3f} (proxy) -> {fit_iou:.3f}", flush=True)
    sources.append((name, Image.open(adir / f"{name}.png").convert("RGB"), mask, cam, 1.0))
report["fits"] = {n: {"camera": c.to_json(), "iou": round(float(iou(points.silhouette(c, mk.shape[1], mk.shape[0]), mk)), 4)}
                  for n, _, mk, c, _ in sources}


# 3. Paint: per source, colors and weights for every sample.
# Depth tolerances, proxy units (~1 tall): samples this close behind the nearest depth around
# their pixel count as the same surface. Painting is looser (a source pixel spans more depth on a
# slanted surface); rendering tighter (keeps the far side out of the frames).
PAINT_TOL = args.paint_tol
RENDER_TOL = 0.005
warps = torch.load(args.warps) if args.warps else {}
cols, weights = [], []
for name, img, mask, cam, boost in sources:
    ex = args.exclude / f"{name}.png" if args.exclude else None
    if ex is not None and ex.exists():
        mask = mask & ~(np.asarray(Image.open(ex)) > 127)
    rgb, wt = Source(name, img, mask, cam, boost, dev, FEATHER_PX, warps.get(name)).paint(xyz, nrm, PAINT_TOL)
    cols.append(rgb.half() if args.views else rgb)
    weights.append(wt)
    if not args.views:
        print(f"{name}: paints {(wt > 0).float().mean():.1%} of the surface", flush=True)

if not (args.views or args.texture):  # anchors only: rendered views already agree in exposure, and n^2 is slow for ~100
    # Exposure: each anchor matched to the hero (or to the blend of the views already matched).
    gains = {}
    order = sorted(range(1, len(sources)), key=lambda i: -float(((weights[i] > 0.1) & (weights[0] > 0.1)).sum()))
    for i in order if args.exposure == "match" else []:
        ref_w = torch.stack([weights[j] for j in [0] + [k for k in order if k in gains]]).sum(0) if gains else weights[0]
        ref_c = (sum(weights[j][:, None] * cols[j] for j in [0] + [k for k in order if k in gains])
                 / ref_w.clamp_min(1e-6)[:, None]) if gains else cols[0]
        both = (weights[i] > 0.1) & (ref_w > 0.1)
        if both.sum() < 2000:
            gains[i] = None
            continue
        a, b = cols[i][both], ref_c[both]
        # Spreads matched, not a least-squares fit: views a degree or two apart barely correlate per
        # sample, and a regression slope then shrinks toward 0 (washed out, shadows lifted).
        gain = b.std(0) / a.std(0).clamp_min(1e-6)
        gain = gain.clamp(0.7, 1.4)
        offset = b.mean(0) - gain * a.mean(0)
        cols[i] = (cols[i] * gain + offset).clamp(0, 1)
        gains[i] = (gain.tolist(), offset.tolist(), int(both.sum()))
    report["exposure"] = {sources[i][0]: gv for i, gv in gains.items()}

    # Agreement: how far each source's colors are from the other sources' blend where both see the
    # surface well (mean absolute difference, 0-1). Misregistered views disagree more.
    report["disagreement"] = {}
    for i, (name, *_rest) in enumerate(sources):
        others = sum(weights[j] for j in range(len(sources)) if j != i)
        blend = sum(weights[j][:, None] * cols[j] for j in range(len(sources)) if j != i) / others.clamp_min(1e-6)[:, None]
        both = (weights[i] > 0.1) & (others > 0.1)
        report["disagreement"][name] = round(float((cols[i][both] - blend[both]).abs().mean()), 4) if both.sum() > 2000 else None
    print("disagreement:", report["disagreement"], flush=True)

    # Seam leveling (cf. Waechter et al., ECCV 2014): each anchor's colors corrected by the smooth
    # difference to the other views' blend where both see the surface; detail stays, lighting evens out.
for level_voxel in [float(x) for x in args.level_voxel.split(",") if x]:
    for i in range(1, len(sources)):
        others = [j for j in range(len(sources)) if j != i]
        w2 = torch.stack([weights[j] ** 2 for j in others])
        blend = (w2[:, :, None] * torch.stack([cols[j] for j in others])).sum(0) / w2.sum(0).clamp_min(1e-12)[:, None]
        ow = torch.stack([weights[j] for j in others]).sum(0)
        both = torch.minimum(weights[i], ow) * ((weights[i] > 0.05) & (ow > 0.05))
        offset, conf = smooth_field(xyz, blend - cols[i], both, level_voxel, 1.5)
        seen = weights[i] > 0
        cols[i] = torch.where((seen & (conf > 1e-6))[:, None], (cols[i] + offset).clamp(0, 1), cols[i])
        report.setdefault("leveling", {}).setdefault(str(level_voxel), {})[sources[i][0]] = round(float(offset[seen].abs().mean()), 4)
    print(f"leveling at {level_voxel}, mean offset:", report["leveling"][str(level_voxel)], flush=True)

raw_weights = weights  # before view selection: how much each view saw, for the fallback's spread
# View selection: each region takes its regionally best view (soft, so seams stay feathered).
if args.select_power:
    local = torch.stack([smooth_field(xyz, wt[:, None], torch.ones_like(wt), 0.01, 2.0)[0][:, 0].clamp_min(0)
                         for wt in weights])
    rel = local / local.max(0).values.clamp_min(1e-9)  # 1 for the regionally best view
    weights = [wt * rel[i] ** args.select_power for i, wt in enumerate(weights)]

total = torch.zeros_like(weights[0])
acc = torch.zeros_like(base_rgb)
best = torch.zeros_like(weights[0])
winner = torch.full_like(weights[0], -1, dtype=torch.long)
for i, (wt, c) in enumerate(zip(weights, cols)):  # one view at a time: ~100 views do not fit stacked
    w2 = wt**2
    total += w2
    acc += w2[:, None] * c.float()
    winner = torch.where(w2 > best, i, winner)
    best = torch.maximum(best, w2)
painted = total > 1e-8
rgb = torch.where(painted[:, None], acc / total.clamp_min(1e-12)[:, None], base_rgb)
winner = torch.where(painted, winner, -1)
if args.mesh_fallback and not args.texture:
    # Where the views disagree, none of them is to be trusted: the mesh's own colors (the 3D model's,
    # from the hero) are coherent if plain. Spread: the weighted standard deviation of the views' colors.
    lo, hi = (float(x) for x in args.mesh_fallback.split(","))
    t2 = torch.zeros_like(total)
    m1 = torch.zeros_like(base_rgb)
    m2 = torch.zeros_like(base_rgb)
    for wt, c in zip(raw_weights, cols):
        w2 = wt**2
        t2 += w2
        m1 += w2[:, None] * c.float()
        m2 += w2[:, None] * c.float() ** 2
    mean = m1 / t2.clamp_min(1e-12)[:, None]
    spread = (m2 / t2.clamp_min(1e-12)[:, None] - mean**2).clamp_min(0).mean(1).sqrt()
    a = ((spread - lo) / (hi - lo)).clamp(0, 1) * painted
    rgb = rgb * (1 - a[:, None]) + base_rgb * a[:, None]
    report["mesh_fallback"] = {"mean": round(float(a[painted].mean()), 4), "full": round(float((a[painted] > 0.99).float().mean()), 4)}
    print(f"mesh fallback: mean blend {report['mesh_fallback']['mean']:.1%}, fully mesh on "
          f"{report['mesh_fallback']['full']:.1%} of the painted surface", flush=True)
    del m1, m2
if args.texture:
    saved = torch.load(args.texture)
    assert saved["points"] == n_points, f"{args.texture} was painted on {saved['points']} samples, not {n_points}"
    assert abs(saved["fingerprint"] - fingerprint(xyz)) < 1e-3, f"{args.texture} was painted on other samples"
    rgb, winner = saved["rgb"].to(dev).float(), saved["winner"].to(dev)
    painted = winner >= 0
    winner = torch.where(painted, 0, -1)  # the coverage sheet shows painted vs not (the sources are gone)
if args.save_texture:
    torch.save({"rgb": rgb.half().cpu(), "winner": winner.cpu(), "points": n_points, "fingerprint": fingerprint(xyz)},
               args.save_texture)
report["painted"] = round(float(painted.float().mean()), 4)
print(f"painted {painted.float().mean():.1%} of the surface; the rest keeps the mesh's colors", flush=True)
del acc, best

# Check sheet: each source with its fit outlined.
tiles = []
for name, img, mask, cam, _ in sources[:12]:
    th = 320
    tw = round(th * img.width / img.height)
    sil = points.silhouette(cam, tw, th)
    edge = sil & ~ndimage.binary_erosion(sil, iterations=2)
    base = np.asarray(img.resize((tw, th)), dtype=np.float64) / 255
    base[edge] = [1.0, 0.2, 0.2]
    tile = Image.fromarray((base * 255).astype(np.uint8))
    ImageDraw.Draw(tile).text((4, 4), f"{name} {report['fits'][name]['iou']:.2f}", fill=(255, 255, 0))
    tiles.append(tile)
sheet = Image.new("RGB", (sum(t.width for t in tiles[:6]), 640), (40, 40, 40))
x = [0, 0]
for i, t in enumerate(tiles):
    row = i // 6
    sheet.paste(t, (x[row], row * 320))
    x[row] += t.width
sheet.save(work / "sources.jpg", quality=88)


# 4. Render the orbit and lay out the attempt.
if args.extra:  # points the mesh lacks (thin_parts.py), rendered with the painted samples
    extra = np.load(args.extra)
    xyz_r = torch.cat([xyz, torch.from_numpy(extra["xyz"]).to(dev)])
    nrm_r = torch.cat([nrm, torch.zeros(len(extra["xyz"]), 3, device=dev)])  # no normals: no slope slack
    rgb = torch.cat([rgb, torch.from_numpy(extra["rgb"]).to(dev)])
    winner = torch.cat([winner, torch.full((len(extra["xyz"]),), -1, device=dev, dtype=winner.dtype)])
    report["extra_points"] = len(extra["xyz"])
else:
    xyz_r, nrm_r = xyz, nrm


def render_view(cam: campath.PathCamera, w: int, h: int, values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return render_points(xyz_r, values, cam, w, h, RENDER_TOL, nrm=nrm_r)


out = work / args.out
if out.exists():
    shutil.rmtree(out)
for d in ("frames", "masks/frames", "masks/hero", "hero", "poses/colmap/model_txt", "poses/colmap/model", ".stages"):
    (out / d).mkdir(parents=True)
shutil.copy(attempt / "hero" / "hero.png", out / "hero" / "hero.png")
Image.fromarray(hero_mask.astype(np.uint8) * 255, "L").save(out / "masks" / "hero" / "hero.png")
w, h = FRAME_SIZE
cams = [replace(hero_cam, yaw=hero_cam.yaw + 360 * k / n, pitch=float(pitch)) for pitch, n in RINGS for k in range(n)]
palette = torch.tensor(np.array([[0.6, 0.6, 0.6]] + [list(np.random.default_rng(i).uniform(0.2, 1, 3)) for i in range(len(sources))]),
                       device=dev, dtype=torch.float32)
palette[1] = torch.tensor([1.0, 1.0, 1.0], device=dev)  # the hero paints white
cov_tiles = []
lines = []


def shrink(img: np.ndarray, mask: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k x k blocks to one pixel: the color averaged over the block's subject pixels only (so an
    edge pixel keeps the subject's color, not a blend with the background) and the subject's
    coverage of the block as the mask."""
    hh, ww = mask.shape[0] // k, mask.shape[1] // k
    m = mask.astype(np.float32).reshape(hh, k, ww, k)
    num = (img.reshape(hh, k, ww, k, -1) * m[..., None]).sum((1, 3))
    cov = m.sum((1, 3))
    return num / np.maximum(cov, 1e-6)[..., None], cov / (k * k)


ss = args.supersample
for i, cam in enumerate(cams):
    img, mask = render_view(cam, w * ss, h * ss, rgb)
    alpha = mask.astype(np.float32)
    if ss > 1:
        img, alpha = shrink(img, mask, ss)
    name = f"frames/{i:05d}.png"
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out / name)
    Image.fromarray((alpha * 255).round().astype(np.uint8), "L").save(out / "masks" / name)
    rot, t = cam.world_to_camera()
    lines.append(f"{i + 1} {' '.join(f'{x:.12g}' for x in _qvec(rot))} {' '.join(f'{x:.12g}' for x in t)} 1 {name}\n\n")
    if i % 8 == 0:
        cov, _ = render_view(cam, w // 4, h // 4, palette[winner + 1])
        cov_tiles.append(Image.fromarray((np.clip(cov, 0, 1) * 255).astype(np.uint8)))
rot, t = hero_cam.world_to_camera()
lines.append(f"{len(cams) + 1} {' '.join(f'{x:.12g}' for x in _qvec(rot))} {' '.join(f'{x:.12g}' for x in t)} 2 hero/hero.png\n\n")
mt = out / "poses" / "colmap" / "model_txt"
(mt / "images.txt").write_text("".join(lines))
hw, hh = hero_img.size
(mt / "cameras.txt").write_text(f"1 SIMPLE_PINHOLE {w} {h} {hero_cam.focal(w, h)} {w / 2} {h / 2}\n"
                                f"2 SIMPLE_PINHOLE {hw} {hh} {hero_cam.focal(hw, hh)} {hw / 2} {hh / 2}\n")
init = torch.randperm(n_points, device=dev)[:100_000]
pts, pc = xyz[init].cpu().numpy(), (rgb[init].clamp(0, 1) * 255).round().int().cpu().numpy()
(mt / "points3D.txt").write_text("".join(f"{k + 1} {x:.6f} {y:.6f} {z:.6f} {r} {gg} {b} 0\n" for k, ((x, y, z), (r, gg, b)) in enumerate(zip(pts, pc))))
subprocess.run(["colmap", "model_converter", "--input_path", str(mt), "--output_path", str(out / "poses" / "colmap" / "model"),
                "--output_type", "BIN"], check=True, capture_output=True)
front = -hero_cam.forward()
front[1] = 0.0
(out / "poses" / "frame.json").write_text(json.dumps({
    "center": list(hero_cam.target), "up": [0.0, -1.0, 0.0], "front": (front / np.linalg.norm(front)).tolist(),
    "source": "projection texture"}, indent=2) + "\n")
(out / "cameras.json").write_text(json.dumps({"preset": "rings", "width": w, "height": h, "hero": hero_cam.to_json(),
                                              "frames": [c.to_json() for c in cams]}, indent=1))
shutil.copy(attempt / ".stages" / "orbit_video.json", out / ".stages" / "orbit_video.json")
(out / "metrics.json").write_text(json.dumps({"texture": {"sources": [s[0] for s in sources], "n_frames": len(cams),
                                                          "painted": report["painted"]}}, indent=1))
cs = Image.new("RGB", (cov_tiles[0].width * 6, cov_tiles[0].height * math.ceil(len(cov_tiles) / 6)))
for i, t in enumerate(cov_tiles):
    cs.paste(t, ((i % 6) * t.width, (i // 6) * t.height))
cs.save(work / "coverage.jpg", quality=88)
(work / "texture.json").write_text(json.dumps(report, indent=1))
print(f"{len(cams)} frames -> {out}", flush=True)
