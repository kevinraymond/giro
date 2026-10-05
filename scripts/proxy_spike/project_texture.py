"""Projection texturing: paint a proxy mesh from the hero and registered anchor views, the way
photogrammetry textures its mesh from the photos, then render a dense multi-ring orbit of it and
lay that out as an attempt the stages can train (dataset -> export). No video model.

    project_texture.py ATTEMPT ANCHOR_DIR WORK GPU [SKIP=02,05] [N_POINTS=6000000] [PAINT_TOL=0.012]

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
import json
import math
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

from giro import path as campath
from giro.stages.fallback import _qvec
from giro.stages.masks import Masks, combine
from giro.stages.proxy import Points, fit_hero_camera, iou

attempt, adir, work = (Path(a).resolve() for a in sys.argv[1:4])
gpu = int(sys.argv[4])
skip = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else {"02", "05"}
n_points = int(sys.argv[6]) if len(sys.argv) > 6 else 6_000_000
dev = torch.device(f"cuda:{gpu}")
FRAME_SIZE = (768, 1024)
RINGS = [(-20, 24), (0, 48), (20, 48), (40, 36), (60, 24), (80, 8)]  # (pitch, views); + looks down
FIT_REFINE_HEIGHT = 512
HERO_WEIGHT = 2.0
FEATHER_PX = 8
seed = json.loads((attempt / "proxy" / "proxy.json").read_text())["seed"]
report: dict = {}

# 1. The mesh, in the splat frame, sampled by area.
m = np.load(work / "mesh.npz")
v = torch.from_numpy(m["vertices"]).to(dev) * torch.tensor([1.0, -1.0, -1.0], device=dev)
f = torch.from_numpy(m["faces"]).to(dev)
col = torch.from_numpy(m["colors"]).to(dev).clamp(0, 1)
lo, hi = v.min(0).values, v.max(0).values
v = (v - (lo + hi) / 2) / float(hi[1] - lo[1])
tri = v[f]
cross = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
area = cross.norm(dim=1) / 2
normals = cross / (2 * area[:, None] + 1e-12)
g = torch.Generator(device=dev).manual_seed(0)
pick = torch.multinomial(area / area.sum(), n_points, replacement=True, generator=g)
r1, r2 = torch.rand(n_points, generator=g, device=dev), torch.rand(n_points, generator=g, device=dev)
s1 = r1.sqrt()
bary = torch.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
xyz = (tri[pick] * bary[:, :, None]).sum(1)
nrm = normals[pick]
base_rgb = (col[f[pick]] * bary[:, :, None]).sum(1)
del tri, cross, area, normals, col
spacing = math.sqrt(float((torch.linalg.cross(v[f][:, 1] - v[f][:, 0], v[f][:, 2] - v[f][:, 0]).norm(dim=1) / 2).sum()) / n_points)
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
reg = json.loads((adir / "registration.json").read_text())
p = Masks.defaults
for name in sorted(reg):
    if name in skip:
        continue
    raw = adir / "raw"
    subject = np.asarray(Image.open(raw / "subject" / "anchors" / f"{name}.png")) > 127
    bg = raw / "background" / "anchors" / f"{name}.png"
    mask = combine(subject, np.asarray(Image.open(bg)) > 127 if bg.exists() else None, p["touch_px"], p["gap_px"], p["max_add"])
    cam, fit_iou = refine(campath.PathCamera.from_json(reg[name]["camera"]), mask)
    print(f"{name}: yaw {cam.yaw:.1f} pitch {cam.pitch:.1f} IoU {reg[name]['iou']:.3f} (proxy) -> {fit_iou:.3f}", flush=True)
    sources.append((name, Image.open(adir / f"{name}.png").convert("RGB"), mask, cam, 1.0))
report["fits"] = {n: {"camera": c.to_json(), "iou": round(float(iou(points.silhouette(c, mk.shape[1], mk.shape[0]), mk)), 4)}
                  for n, _, mk, c, _ in sources}


def project(cam: campath.PathCamera, w: int, h: int, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rot, t = cam.world_to_camera()
    rot_t, t_t = torch.tensor(rot, device=dev, dtype=torch.float32), torch.tensor(t, device=dev, dtype=torch.float32)
    pc = pts @ rot_t.T + t_t
    z = pc[:, 2]
    fl = cam.focal(w, h)
    zs = torch.where(z > 1e-3, z, torch.ones_like(z))
    return fl * pc[:, 0] / zs + w / 2, fl * pc[:, 1] / zs + h / 2, z


def zbuffer(u: torch.Tensor, vv: torch.Tensor, z: torch.Tensor, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per sample: in view, its pixel index, and the nearest depth around that pixel. The nearest
    depth is taken over 3x3 pixels: a pixel the front surface's samples happen to miss would
    otherwise let the surface behind it through (speckles of the wrong part's color)."""
    ok = (z > 1e-3) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
    pix = torch.where(ok, vv.long().clamp(0, h - 1) * w + u.long().clamp(0, w - 1), torch.zeros_like(z, dtype=torch.long))
    zmin = torch.full((w * h,), float("inf"), device=dev)
    zmin.scatter_reduce_(0, pix[ok], z[ok], reduce="amin")
    zmin = -F.max_pool2d(-zmin.reshape(1, 1, h, w), 3, stride=1, padding=1).reshape(-1)
    return ok, pix, zmin[pix]


def sample(img: torch.Tensor, u: torch.Tensor, vv: torch.Tensor, w: int, h: int) -> torch.Tensor:
    grid = torch.stack([2 * u / w - 1, 2 * vv / h - 1], 1)[None, None]
    return F.grid_sample(img[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T


# 3. Paint: per source, colors and weights for every sample.
# Depth tolerances, proxy units (~1 tall): samples this close behind the nearest depth around
# their pixel count as the same surface. Painting is looser (a source pixel spans more depth on a
# slanted surface); rendering tighter (keeps the far side out of the frames).
PAINT_TOL = float(sys.argv[7]) if len(sys.argv) > 7 else 0.012
RENDER_TOL = 0.005
cols, weights = [], []
for name, img, mask, cam, boost in sources:
    w, h = img.size
    u, vv, z = project(cam, w, h, xyz)
    ok, pix, zmin = zbuffer(u, vv, z, w, h)
    visible = ok & (z <= zmin + PAINT_TOL)
    inner = ndimage.binary_erosion(mask, iterations=2)
    feather = np.clip(ndimage.distance_transform_edt(inner) / FEATHER_PX, 0, 1).astype(np.float32)
    fe = sample(torch.from_numpy(feather).to(dev)[None], u, vv, w, h)[:, 0]
    rot, _ = cam.world_to_camera()
    to_cam = torch.tensor(cam.position(), device=dev, dtype=torch.float32) - xyz
    cos = (nrm * to_cam).sum(1).abs() / to_cam.norm(dim=1)
    wt = torch.where(visible, boost * cos**4 * fe, torch.zeros_like(cos))
    rgb = sample(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1).to(dev), u, vv, w, h)
    cols.append(rgb)
    weights.append(wt)
    print(f"{name}: paints {(wt > 0).float().mean():.1%} of the surface", flush=True)

# Exposure: each anchor matched to the hero (or to the blend of the views already matched).
gains = {}
order = sorted(range(1, len(sources)), key=lambda i: -float(((weights[i] > 0.1) & (weights[0] > 0.1)).sum()))
for i in order:
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

W = torch.stack(weights) ** 2
total = W.sum(0)
painted = total > 1e-8
rgb = torch.where(painted[:, None], (W[:, :, None] * torch.stack(cols)).sum(0) / total.clamp_min(1e-12)[:, None], base_rgb)
winner = torch.where(painted, W.argmax(0), torch.full_like(total, -1, dtype=torch.long))
report["painted"] = round(float(painted.float().mean()), 4)
print(f"painted {painted.float().mean():.1%} of the surface; the rest keeps the mesh's colors", flush=True)
del W

# Check sheet: each source with its fit outlined.
tiles = []
for name, img, mask, cam, _ in sources:
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
def render_view(cam: campath.PathCamera, w: int, h: int, values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    u, vv, z = project(cam, w, h, xyz)
    ok, pix, zmin = zbuffer(u, vv, z, w, h)
    front = ok & (z <= zmin + RENDER_TOL)
    acc = torch.zeros((w * h, values.shape[1]), device=dev).index_add_(0, pix[front], values[front])
    n = torch.zeros(w * h, device=dev).index_add_(0, pix[front], torch.ones_like(z[front]))
    img = (acc / n.clamp_min(1)[:, None]).reshape(h, w, -1)
    filled = (n > 0).reshape(h, w)
    # close pinholes between samples from the neighbors
    for _ in range(4):  # the 3x3 depth test also drops a pixel or two beside nearer edges
        k = torch.ones((1, 1, 3, 3), device=dev)
        num = F.conv2d((img * filled[..., None]).permute(2, 0, 1)[:, None], k, padding=1)[:, 0].permute(1, 2, 0)
        den = F.conv2d(filled[None, None].float(), k, padding=1)[0, 0]
        img = torch.where(filled[..., None], img, num / den.clamp_min(1)[..., None])
        filled = filled | (den > 0)
    mask = ndimage.binary_opening(ndimage.binary_closing(filled.cpu().numpy(), iterations=2), iterations=1)
    return img.cpu().numpy(), mask


out = work / "attempt"
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
for i, cam in enumerate(cams):
    img, mask = render_view(cam, w, h, rgb)
    name = f"frames/{i:05d}.png"
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out / name)
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(out / "masks" / name)
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
