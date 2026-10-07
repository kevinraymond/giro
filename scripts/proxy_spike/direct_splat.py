"""Route B (board #3726): optimize the splat directly against the generated views; the textured mesh is
only the initialization and a weak prior. The texture route bakes the views into one color per surface
sample and trains on renders of that, and the four-stage chain test showed most sharpness is lost in that
bake (#3720). Here each view supervises the splat itself:

- the hero and the route's anchors (warps baked in, each its own pinhole camera, as finetune_anchors.py)
  are trained at full weight, per pixel times a confidence: each view's share of the surface it sees,
  |cos|^4 x visibility (project_texture's weights) raised to --power and normalized over the views, so a
  surface two views see well is split between them rather than averaged at double weight; pixels inside
  the mask that the mesh misses (thin parts it lacks) get --off-mesh;
- the route's renders of the textured mesh (WORK/attempt/dataset) are the prior: weight --prior where the
  views see the surface well, rising to 1 where none does (top, underside: the bake is all there is);
- --bilagrid: a bilateral grid per anchor (an affine color transform varying over the image and with
  brightness) absorbs exposure and white balance that view disagrees on; the hero has none, so the
  splat's colors follow the hero. The grids are dropped at export.

Silhouettes: the renders' masks are exact; an anchor's mask (SAM) counts --sil outside, minus a band of
4 px at its edge. gsplat (Apache-2.0) with its MCMC densification capped at --max-splats, SH degree 3.

    uv run --group direct python direct_splat.py WORK ANGLES OUT GPU [--views both|renders|anchors]
        [--prior 0.1] [--power 2] [--bilagrid] [--iters 30000] [--max-splats 300000]
        [--library NAME --image SOURCE]

WORK is a texture_route.py work dir (mesh.npz, anchor_cameras.json, warps.pt, attempt/ trained; its
route.json one level up says which anchors were used); ANGLES its anchor views (NN.png, raw/ masks).
OUT/attempt becomes an attempt (train/final.ply, then crop -> export by `giro stages --from crop`);
OUT/diag holds each anchor next to the splat's render from its camera.
"""
import argparse
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy
from PIL import Image
from scipy import ndimage

from giro import splat
from texture_common import PoseCamera, Source, anchor_mask, bake, load_cameras, project, render_points, samples

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
C0 = 0.28209479177387814
ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path)
ap.add_argument("angles", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--views", choices=["both", "renders", "anchors"], default="both",
                help="renders: the route's renders only (B0, gsplat vs Brush); anchors: hero + anchors, no prior")
ap.add_argument("--prior", type=float, default=0.1, help="the renders' weight where the views see the surface well")
ap.add_argument("--power", type=float, default=2.0, help="ownership sharpness: weights^power normalized over views")
ap.add_argument("--off-mesh", type=float, default=0.25, help="confidence of view pixels inside the mask the mesh misses")
ap.add_argument("--sil", type=float, default=0.5, help="weight of an anchor's background (its silhouette)")
ap.add_argument("--hero-weight", type=float, default=2.0, help="the hero's boost in the ownership (project_texture's)")
ap.add_argument("--gen-frac", type=float, default=0.5, help="share of steps on the hero + anchors (rest: renders)")
ap.add_argument("--bilagrid", action="store_true")
ap.add_argument("--iters", type=int, default=30_000)
ap.add_argument("--max-splats", type=int, default=300_000)
ap.add_argument("--init", type=int, default=150_000, help="Gaussians sampled on the mesh to start")
ap.add_argument("--ssim", type=float, default=0.2)
ap.add_argument("--depth", type=float, default=0.0,
                help="the mesh as geometry prior: weight of |rendered depth - mesh depth| / mesh depth wherever the mesh "
                     "covers a pixel, every view (B1 without it fit disagreeing views with semi-transparent layers)")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--library", help="also make a library job with this name")
ap.add_argument("--image", type=Path, help="the job's source image, for the library")
args = ap.parse_args()
work, adir, out = args.work.resolve(), args.angles.resolve(), args.out.resolve()
dev = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(dev)
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
base = work / "attempt"
(out / "diag").mkdir(parents=True, exist_ok=True)
t_start = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t_start:6.0f} s] {msg}", flush=True)


def qvec_to_rot(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class View:
    """One training image on the GPU: colors, mask (soft), per-pixel weight inside and outside the mask,
    the camera as gsplat wants it, and an optional bilateral grid index."""

    def __init__(self, name: str, kind: str, rgb: np.ndarray, mask: np.ndarray, rot: np.ndarray, t: np.ndarray,
                 fx: float, fy: float, cx: float, cy: float):
        self.name, self.kind = name, kind
        self.h, self.w = mask.shape
        self.rgb = torch.from_numpy(rgb.astype(np.float32) / 255).to(dev)
        self.mask = torch.from_numpy(mask.astype(np.float32)).to(dev)[..., None]
        self.cpu: tuple[torch.Tensor, ...] | None = None
        vm = np.eye(4, dtype=np.float32)
        vm[:3, :3], vm[:3, 3] = rot, t
        self.viewmat = torch.from_numpy(vm).to(dev)
        self.K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=dev)
        self.cam = PoseCamera(tuple(tuple(map(float, r)) for r in rot), tuple(map(float, t)),
                              math.degrees(2 * math.atan((min(self.w, self.h) / 2) / fx)))
        self.weight = torch.ones_like(self.mask)  # per pixel; set below
        self.grid: int | None = None


# The route's dataset: renders (frames/) and the hero (hero/hero.png; its copies are Brush's weighting).
ds = base / "dataset"
intr = {}
for ln in (ds / "sparse" / "0" / "cameras.txt").read_text().splitlines():
    if ln and not ln.startswith("#"):
        p = ln.split()
        vals = list(map(float, p[4:]))
        fx, fy, cx, cy = (vals[0], vals[0], vals[1], vals[2]) if p[1] == "SIMPLE_PINHOLE" else vals[:4]
        intr[int(p[0])] = (fx, fy, cx, cy)
renders, hero = [], None
img_lines = [ln for ln in (ds / "sparse" / "0" / "images.txt").read_text().splitlines() if ln and not ln.startswith("#")]
for ln in img_lines:
    p = ln.split()
    if len(p) < 10 or not p[9].endswith(".png"):
        continue
    name = p[9]
    if (name.startswith("hero/") and name != "hero/hero.png") or not (ds / "images" / name).exists():
        continue  # the dataset stage may leave a listed frame out
    rgb = np.asarray(Image.open(ds / "images" / name).convert("RGB"))
    mask = np.asarray(Image.open(ds / "masks" / name).convert("L")).astype(np.float32) / 255
    v = View(name, "hero" if name.startswith("hero/") else "render", rgb, mask, qvec_to_rot(list(map(float, p[1:5]))),
             np.array(list(map(float, p[5:8]))), *intr[int(p[8])])
    if v.kind == "hero":
        hero = v
    else:
        renders.append(v)
assert hero is not None
log(f"{len(renders)} renders, hero {hero.w}x{hero.h}")

# The anchors the route used, warps baked in (as finetune_anchors.py).
route = json.loads((work.parent / "route.json").read_text())
dropped = set(route["anchors"]["silhouette_dropped"]) | set(route["anchors"].get("color_dropped", []))
cams = load_cameras(work / "anchor_cameras.json")
warps = torch.load(work / "warps.pt") if (work / "warps.pt").exists() else {}
anchors = []
for n in sorted(cams):
    if n in dropped:
        continue
    img = Image.open(adir / f"{n}.png").convert("RGB")
    pix, m = bake(Source(n, img, anchor_mask(adir, n), cams[n], 1.0, dev, warp=warps.get(n)))
    w, h = img.size
    f = cams[n].focal(w, h)
    rot, t = cams[n].world_to_camera()
    anchors.append(View(f"anchor/{n}", "anchor", pix, m.astype(np.float32), np.asarray(rot), np.asarray(t), f, f, w / 2, h / 2))
log(f"{len(anchors)} anchors: {' '.join(a.name[7:] for a in anchors)}")

# Confidence from the mesh: each generated view's weight per surface sample (|cos|^4, z-buffer, mask feather),
# its share over the views, and the views' total coverage (for the prior).
xyz, nrm, mesh_rgb, spacing = samples(work, 2_000_000, dev)
gen = [hero] + anchors
W = []
for v in gen:
    img = Image.fromarray((v.rgb.cpu().numpy() * 255).astype(np.uint8))
    src = Source(v.name, img, v.mask[..., 0].cpu().numpy() > 0.5, v.cam, args.hero_weight if v.kind == "hero" else 1.0, dev)
    W.append(src.paint(xyz, nrm, 0.012)[1])
W = torch.stack(W)
own = W.clamp_min(0) ** args.power
own = own / own.sum(0, keepdim=True).clamp_min(1e-8)
coverage = (W.sum(0) / 0.5).clamp(0, 1)
for i, v in enumerate(gen):
    conf, on_mesh = render_points(xyz, own[i][:, None], v.cam, v.w, v.h, 0.012, nrm=nrm)
    conf = torch.from_numpy(np.where(on_mesh[..., None], conf, args.off_mesh)).float().to(dev)
    inside = v.mask > 0.5
    m = v.mask[..., 0].cpu().numpy() > 0.5
    band = torch.from_numpy(ndimage.binary_dilation(m, iterations=4) & ~ndimage.binary_erosion(m, iterations=4)).to(dev)[..., None]
    bg = 1.0 if v.kind == "hero" else args.sil
    v.weight = torch.where(inside, conf, torch.full_like(conf, bg))
    v.weight = torch.where(band & ~inside, torch.zeros_like(conf), v.weight)
    if v.kind == "hero":
        v.weight = torch.where(inside, v.weight.clamp_min(0.5), v.weight)  # the real photo always counts
    Image.fromarray((v.weight[..., 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).save(
        out / "diag" / f"conf-{v.name.removesuffix('.png').replace('/', '-')}.png")
prior_w = (args.prior + (1 - args.prior) * (1 - coverage))[:, None]
for v in renders:
    pw, on_mesh = render_points(xyz, prior_w, v.cam, v.w, v.h, 0.012, nrm=nrm)
    pw = torch.from_numpy(np.where(on_mesh[..., None], pw, 1.0)).float().to(dev)
    v.weight = pw if args.views == "both" else torch.ones_like(pw)
for v in renders + gen:  # on the CPU from here on (188 renders x 3 maps would fill the GPU)
    v.cpu = tuple(x.to(torch.float16).cpu().pin_memory() for x in (v.rgb, v.mask, v.weight))
    del v.rgb, v.mask, v.weight
    v.depth = None
    if args.depth > 0:  # the mesh's depth at every pixel it covers (0 elsewhere), float32: depths ~2 need the precision
        zc = project(v.cam, v.w, v.h, xyz)[2]
        d, on_mesh = render_points(xyz, zc[:, None], v.cam, v.w, v.h, 0.012, nrm=nrm)
        v.depth = torch.from_numpy(np.where(on_mesh[..., None], d, 0.0)).float().pin_memory()
torch.cuda.empty_cache()
log(f"confidence: coverage>0.5 on {float((coverage > 0.5).float().mean()):.2f} of the surface, "
    f"mean prior weight {float(prior_w.mean()):.2f}")
del W, own

if args.views == "renders":
    gen_views, prior_views = [], renders
elif args.views == "anchors":
    gen_views, prior_views = gen, []
else:
    gen_views, prior_views = gen, renders

# Bilateral grids (one per anchor; the hero has none): 12 affine coefficients over (brightness 8, rows 16, cols 16).
grids = None
if args.bilagrid:
    for k, v in enumerate(anchors):
        v.grid = k
    eye = torch.eye(3, 4, device=dev).reshape(12, 1, 1, 1)
    grids = torch.nn.Parameter(eye.expand(12, 8, 16, 16).repeat(len(anchors), 1, 1, 1, 1).contiguous())


def apply_grid(rgb: torch.Tensor, alpha: torch.Tensor, k: int) -> torch.Tensor:
    """rgb (h, w, 3) premultiplied by alpha (h, w, 1): colors through grid k's local affine transform."""
    h, w = rgb.shape[:2]
    col = rgb / alpha.clamp_min(1e-3)
    gray = (col * torch.tensor([0.299, 0.587, 0.114], device=dev)).sum(-1).clamp(0, 1)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=dev), torch.linspace(-1, 1, w, device=dev), indexing="ij")
    coords = torch.stack([xx, yy, gray * 2 - 1], -1)[None, None]  # (1, 1, h, w, 3): x, y, brightness
    a = F.grid_sample(grids[k][None], coords, align_corners=True)[0, :, 0].permute(1, 2, 0).reshape(h, w, 3, 4)
    return (a[..., :3] @ rgb[..., None])[..., 0] + a[..., 3] * alpha


def tv(g: torch.Tensor) -> torch.Tensor:
    return sum(((g.diff(dim=d)) ** 2).mean() for d in (1, 2, 3))


def ssim_map(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(h, w, 3) -> (h, w, 1), an 11x11 Gaussian window."""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    g = torch.exp(-0.5 * ((torch.arange(11, device=dev) - 5) / 1.5) ** 2)
    k = (g[:, None] * g[None]) / g.sum() ** 2
    k = k.expand(3, 1, 11, 11)

    def blur(z: torch.Tensor) -> torch.Tensor:
        return F.conv2d(z, k, padding=5, groups=3)

    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx**2, blur(y * y) - my**2, blur(x * y) - mx * my
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx**2 + my**2 + c1) * (sxx + syy + c2))
    return s.mean(1)[0][..., None]


# Gaussians from the mesh: positions and colors of surface samples, flat discs along the surface.
pick = torch.randperm(len(xyz), device=dev)[: args.init]
n0 = len(pick)
s0 = math.log(spacing * math.sqrt(len(xyz) / n0) * 0.7)
z = F.normalize(nrm[pick] + 1e-6, dim=1)
quat = torch.cat([1 + z[:, 2:3], -z[:, 1:2], z[:, 0:1], torch.zeros_like(z[:, :1])], 1)  # rotates +z onto the normal
quat = torch.where(z[:, 2:3] < -0.999, torch.tensor([0.0, 1.0, 0.0, 0.0], device=dev), quat)
params = torch.nn.ParameterDict({
    "means": torch.nn.Parameter(xyz[pick].clone()),
    "scales": torch.nn.Parameter(torch.tensor([s0, s0, s0 - 2.0], device=dev).expand(n0, 3).clone()),
    "quats": torch.nn.Parameter(F.normalize(quat, dim=1)),
    "opacities": torch.nn.Parameter(torch.full((n0,), math.log(0.5 / 0.5), device=dev)),
    "sh0": torch.nn.Parameter(((mesh_rgb[pick] - 0.5) / C0)[:, None, :].clone()),
    "shN": torch.nn.Parameter(torch.zeros(n0, 15, 3, device=dev)),
})
del xyz, nrm, mesh_rgb
scene_scale = 1.0  # samples() normalizes the subject to height 1
lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
optimizers = {k: torch.optim.Adam([params[k]], lr=lrs[k], eps=1e-15) for k in params}
means_decay = 0.01 ** (1 / args.iters)
strategy = MCMCStrategy(cap_max=args.max_splats, refine_start_iter=500, refine_stop_iter=int(args.iters * 0.8),
                        refine_every=100, min_opacity=0.005)
strategy.check_sanity(params, optimizers)
state = strategy.initialize_state()
grid_opt = torch.optim.Adam([grids], lr=2e-3, eps=1e-15) if grids is not None else None


def render(v: View, sh_degree: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    img, alpha, info = rasterization(params["means"], params["quats"], torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
                                     colors, v.viewmat[None], v.K[None], v.w, v.h, sh_degree=sh_degree,
                                     rasterize_mode="antialiased", render_mode="RGB+ED")
    return img[0], alpha[0], info


log(f"training {args.iters} steps: {len(gen_views)} generated views (share {args.gen_frac if gen_views and prior_views else 1}), "
    f"{len(prior_views)} renders, init {n0} Gaussians, cap {args.max_splats}, grids {'on' if grids is not None else 'off'}")
for step in range(args.iters):
    if gen_views and (not prior_views or random.random() < args.gen_frac):
        v = random.choice(gen_views)
    else:
        v = random.choice(prior_views)
    sh_degree = min(3, step // 1000)
    rgbd, alpha, info = render(v, sh_degree)
    rgb, depth = rgbd[..., :3], rgbd[..., 3:]
    strategy.step_pre_backward(params, optimizers, state, step, info)
    if v.grid is not None:
        rgb = apply_grid(rgb, alpha, v.grid)
    v_rgb, v_mask, v_weight = (x.to(dev, non_blocking=True).float() for x in v.cpu)
    bgc = torch.rand(3, device=dev)
    pred = rgb + (1 - alpha) * bgc
    gt = v_rgb * v_mask + (1 - v_mask) * bgc
    err = (1 - args.ssim) * (pred - gt).abs().mean(-1, keepdim=True) + args.ssim * (1 - ssim_map(pred, gt))
    loss = (v_weight * err).mean()
    if v.depth is not None:
        d_mesh = v.depth.to(dev, non_blocking=True)
        on = (d_mesh > 0).float()
        loss = loss + args.depth * (on * (depth - d_mesh).abs() / d_mesh.clamp_min(1e-3)).mean()
    loss = loss + 0.01 * torch.sigmoid(params["opacities"]).mean() + 0.01 * torch.exp(params["scales"]).mean()
    if v.grid is not None:
        loss = loss + 10 * tv(grids[v.grid])
    loss.backward()
    for opt in optimizers.values():
        opt.step()
        opt.zero_grad(set_to_none=True)
    if grid_opt is not None:
        grid_opt.step()
        grid_opt.zero_grad(set_to_none=True)
    for g in optimizers["means"].param_groups:
        g["lr"] *= means_decay
    strategy.step_post_backward(params, optimizers, state, step, info, lr=optimizers["means"].param_groups[0]["lr"])
    if step % 1000 == 0 or step == args.iters - 1 or (step < 1000 and step % 50 == 0):
        log(f"step {step}: loss {float(loss):.4f}, {len(params['means'])} Gaussians, max scale "
            f"{float(torch.exp(params['scales']).max()):.3f}, max radius {int(info['radii'].max())} px, {v.name}, "
            f"{torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB")

# The splat as Brush writes it (INRIA layout: f_rest channel-major), into an attempt for crop -> export.
att = out / "attempt"
if att.exists():
    shutil.rmtree(att)
att.mkdir()
for name in ("frames", "masks", "hero", "poses"):
    (att / name).symlink_to(base / name)
shutil.copy(base / "cameras.json", att / "cameras.json")
shutil.copytree(base / ".stages", att / ".stages")
for st in ("train", "crop", "canonicalize", "export"):
    (att / ".stages" / f"{st}.json").unlink(missing_ok=True)
metrics = json.loads((base / "metrics.json").read_text())
(att / "metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k in ("texture", "dataset")}, indent=1))
(att / "train").mkdir()
with torch.no_grad():
    n = len(params["means"])
    fields = (["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
               "f_dc_0", "f_dc_1", "f_dc_2"] + [f"f_rest_{i}" for i in range(45)])
    rec = np.zeros(n, dtype=[(k, "<f4") for k in fields])
    cols = [params["means"], params["scales"], params["opacities"][:, None], F.normalize(params["quats"], dim=1),
            params["sh0"][:, 0], params["shN"].permute(0, 2, 1).reshape(n, 45)]
    flat = torch.cat(cols, 1).cpu().numpy()
    for i, k in enumerate(fields):
        rec[k] = flat[:, i]
splat.write_ply(att / "train" / "final.ply", rec)
log(f"wrote {n} Gaussians")

# Each anchor (and the hero) next to the splat from its camera, without its grid: what the views became.
with torch.no_grad():
    for v in gen:
        rgb, alpha, _ = render(v, 3)
        rgb = rgb[..., :3]
        pred = (rgb + (1 - alpha) * 0.5).clamp(0, 1)
        v_rgb, v_mask, _ = (x.to(dev).float() for x in v.cpu)
        gt = v_rgb * v_mask + (1 - v_mask) * 0.5
        pair = torch.cat([gt, pred], 1).cpu().numpy()
        Image.fromarray((pair * 255).astype(np.uint8)).save(out / "diag" / f"pair-{v.name.removesuffix('.png').replace('/', '-')}.jpg", quality=92)

params_cli = []
for st in ("crop", "canonicalize", "export"):
    for k, v in json.loads((base / ".stages" / f"{st}.json").read_text())["params"].items():
        params_cli += ["-p", f"{st}.{k}={json.dumps(v)}"]
subprocess.run(["uv", "run", "giro", "stages", str(att), "--from", "crop", "--gpu", str(args.gpu), *params_cli], cwd=ROOT, check=True)
(out / "direct.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "anchors": [a.name for a in anchors],
                                             "n_gaussians": n, "seconds": round(time.monotonic() - t_start)}, indent=1))
if args.library:
    image = args.image.resolve() if args.image else base / "hero" / "hero.png"
    height = json.loads((base / ".stages" / "canonicalize.json").read_text())["params"]["height_m"]
    subprocess.run(["uv", "run", "python", "make_review_jobs.py", args.library, str(image), str(height), f"1={att}"], cwd=HERE, check=True)
log(f"done: {att}")
