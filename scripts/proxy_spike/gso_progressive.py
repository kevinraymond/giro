"""Progressive control images for the view-LoRA v2 (board #3712): what the texture route shows the
LoRA at inference once earlier views are painted. Run after gso_controls.py (cameras) and
gso_render.py --aligned (ground truth at those cameras).

    gso_progressive.py RENDERS_DIR CONTROLS_DIR GPU [--names a,b | --manifest CSV --split train,heldout]
        [--shard J/K] [--points 6000000] [--better 1.5] [--supersample 2]

Per object (CONTROLS_DIR/<name>: work/mesh.npz, gt_cameras.json with hero_camera, target/<view>.png):
1. The mesh sampled; painted from the hero as gso_controls.py does (Pixal3D's colors elsewhere); the
   hero's surface locked with its quality (|cos|^4), as progressive_paint.py starts.
2. Every target rendered from that state: control/<view>.png (+ _mask: coverage), the v1 control
   (k = 0, the ablation's input).
3. The targets ordered as the route paints, by angle from the hero; each target t gets a k drawn
   uniformly from 0..(its place in the order): it is rendered after the first k views in the order
   have been painted, each from its ground truth (teacher forcing: the GT stands in for the LoRA's
   accepted view), overwriting where that view sees the surface `--better` times better than what
   painted it, then locking it (progressive_paint.py's masked rule): prog/<view>.png (+ _mask).
4. prog/prog.json: per target its k, the views painted before it and the nearest of them (the
   camera closest in direction), the image 3 the dataset gives the LoRA (none when k = 0).
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from giro import path as campath
from texture_common import Source, project, render_points, sample, samples, slope_slack, zbuffer

ap = argparse.ArgumentParser()
ap.add_argument("renders", type=Path)
ap.add_argument("controls", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--names", default="")
ap.add_argument("--manifest", type=Path)
ap.add_argument("--split", default="train,heldout")
ap.add_argument("--shard", default="", metavar="J/K")
ap.add_argument("--points", type=int, default=6_000_000, help="surface samples (v2 renders are 1080x1440, 2x supersampled)")
ap.add_argument("--paint-tol", type=float, default=0.012)
ap.add_argument("--better", type=float, default=1.5, help="paint where a view's quality is this many times the locked one")
ap.add_argument("--supersample", type=int, default=2)
args = ap.parse_args()
dev = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")  # -1: the CPU, for a smoke test
BASE_WEIGHT = 0.02  # gso_controls.py's

if args.names:
    names = args.names.split(",")
elif args.manifest:
    with open(args.manifest) as f:
        names = [r["name"] for r in csv.DictReader(f) if r["split"] in args.split.split(",")]
else:
    names = sorted(p.name for p in args.controls.iterdir() if (p / "target" / "done").exists())
if args.shard:
    j, k = map(int, args.shard.split("/"))
    names = names[j::k]


def view(xyz: torch.Tensor, nrm: torch.Tensor, cam: campath.PathCamera, w: int, h: int
         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per sample: in this view's front layer, where it lands, and its quality (progressive_paint.py's)."""
    u, v, z = project(cam, w, h, xyz)
    ok, _, zmin, _ = zbuffer(u, v, z, w, h, slope_slack(cam, w, h, xyz, nrm, z))
    front = ok & (z <= zmin + 0.005)
    to_cam = torch.tensor(np.asarray(cam.position()), device=dev, dtype=torch.float32) - xyz
    cos = (nrm * to_cam).sum(1).abs() / to_cam.norm(dim=1)
    return front, u, v, cos**4


def save_render(xyz: torch.Tensor, rgb: torch.Tensor, nrm: torch.Tensor, cam: campath.PathCamera, w: int, h: int,
                stem: Path) -> None:
    """2x supersampled, color averaged over subject subpixels, coverage as the mask (gso_controls.py's)."""
    ss = args.supersample
    img, m = render_points(xyz, rgb, cam, w * ss, h * ss, nrm=nrm)
    m = m.astype(np.float32)
    blocks = lambda a: a.reshape(h, ss, w, ss, *a.shape[2:]).sum((1, 3))  # noqa: E731
    cov = blocks(m) / ss**2
    color = blocks(img * m[..., None]) / np.maximum(blocks(m), 1e-6)[..., None]
    Image.fromarray((color * cov[..., None] * 255).clip(0, 255).astype(np.uint8)).save(stem.with_suffix(".png"))
    Image.fromarray((cov * 255).astype(np.uint8), "L").save(stem.parent / f"{stem.name}_mask.png")


def one(name: str) -> str | None:
    od = args.controls / name
    if (od / "prog" / "prog.json").exists() or not (od / "target" / "done").exists():
        return None
    g = json.loads((od / "gt_cameras.json").read_text())
    views = g["views"]
    if "hero_camera" in g:
        hero_cam = campath.PathCamera.from_json(g["hero_camera"])
    else:  # v1 files: a view's proxy camera is the hero's with yaw and pitch replaced
        c = campath.PathCamera.from_json(views[0]["proxy_camera"])
        hero_cam = campath.PathCamera(g["hero_fit"]["yaw"], g["hero_fit"]["pitch"], c.distance, c.target, c.fov)
    cams = {v["name"]: campath.PathCamera.from_json(v["proxy_camera"]) for v in views}
    rd = args.renders / name
    hero = Image.open(rd / "hero.png").convert("RGB")
    mask = np.asarray(Image.open(rd / "hero_rgba.png").split()[3]) > 127
    w, h = hero.size
    xyz, nrm, base, _ = samples(od / "work", args.points, dev)
    col, wt = Source("hero", hero, mask, hero_cam, 1.0, dev).paint(xyz, nrm, args.paint_tol)
    rgb = (col * wt[:, None] + base * BASE_WEIGHT) / (wt[:, None] + BASE_WEIGHT)
    front, u, v, q = view(xyz, nrm, hero_cam, w, h)
    inside = sample(torch.from_numpy(mask.astype(np.float32)).to(dev)[None], u, v, w, h)[:, 0] > 0.5
    locked = torch.where(front & inside, q, torch.zeros_like(q))
    (od / "control").mkdir(exist_ok=True)
    (od / "prog").mkdir(exist_ok=True)
    for vv in views:  # k = 0 for every target: the v1-style control
        save_render(xyz, rgb, nrm, cams[vv["name"]], w, h, od / "control" / vv["name"])
    hf = hero_cam.forward()
    order = sorted(cams, key=lambda n: -float(np.dot(cams[n].forward(), hf)))
    rng = random.Random(f"{name}/prog")
    k_of = {n: rng.randint(0, i) for i, n in enumerate(order)}
    meta = {}
    for k in range(len(order)):
        for t in [n for n in order if k_of[n] == k]:
            done = order[:k]
            near = max(done, key=lambda n: float(np.dot(cams[n].forward(), cams[t].forward()))) if done else None
            save_render(xyz, rgb, nrm, cams[t], w, h, od / "prog" / t)
            meta[t] = {"k": k, "painted": done, "neighbor": near}
        # paint view order[k] from its ground truth, where it sees the surface clearly better than the lock
        s = order[k]
        gt = Image.open(od / "target" / f"{s}.png")
        front, u, v, q = view(xyz, nrm, cams[s], w, h)
        need = front & (q > args.better * locked)
        # progressive_paint.py's region: where most of what the view sees needs it, opened, inside the GT's
        # (eroded) alpha, feathered at its edge
        frac, _ = render_points(xyz, need.float()[:, None], cams[s], w, h, nrm=nrm)
        inside_gt = ndimage.binary_erosion(np.asarray(gt.split()[3]) > 127, iterations=2)
        regen = ndimage.binary_opening((frac[..., 0] > 0.5) & inside_gt, iterations=2)
        alpha = np.clip(ndimage.gaussian_filter(regen.astype(np.float32), 1.5), 0, 1) * regen
        a = sample(torch.from_numpy(alpha.astype(np.float32)).to(dev)[None], u, v, w, h)[:, 0] * need
        new = sample(torch.from_numpy(np.asarray(gt.convert("RGB"), np.float32) / 255).permute(2, 0, 1).to(dev), u, v, w, h)
        rgb = rgb * (1 - a[:, None]) + new * a[:, None]
        locked = torch.where(a > 0.5, q, locked)
    (od / "prog" / "prog.json").write_text(json.dumps({"order": order, "better": args.better, "targets": meta}, indent=1) + "\n")
    return f"{len(order)} targets, k {np.mean([m['k'] for m in meta.values()]):.1f} on average"


t_all = time.monotonic()
for i, name in enumerate(names):
    t0 = time.monotonic()
    msg = one(name)
    if msg:
        print(f"[{i + 1}/{len(names)}] {name}: {msg}, {time.monotonic() - t0:.0f} s", flush=True)
print(f"done in {time.monotonic() - t_all:.0f} s", flush=True)
