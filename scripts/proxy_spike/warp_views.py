"""Per-view warp grids for the anchor views (the non-rigid half of Zhou & Koltun's "Color Map
Optimization", SIGGRAPH 2014): each anchor gets a coarse grid of 2D offsets, in pixels, that
bends where its pixels land on the mesh, so it agrees with the other views locally where a rigid
camera cannot (a rim off-center on its tire after a 1 degree pose error; the anchors' own small
shape drift). The hero is the reference and is not warped.

For OUTER rounds, coarse to fine (images blurred by each of SIGMAS px), every anchor's grid is
fitted with Adam to the blend of all the other views at the samples it sees: squared color error
(after a per-channel gain and offset, which absorb exposure), weighted by its own and the others'
weights, plus a smoothness penalty on neighboring grid nodes and a small one on the offsets.

    warp_views.py ATTEMPT ANCHOR_DIR WORK GPU --skip 08,10,... [--cell 56]

Reads WORK/texture.json (the hero's fit) and WORK/anchor_cameras.json (register_photometric.py);
writes WORK/warps.pt ({name: (2, rows, cols) offsets}) for project_texture.py --warps, and
WORK/warps.json (per view: offsets, error before and after).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from giro import path as campath
from texture_common import PoseCamera, Source, anchor_mask, load_cameras, sample, samples

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("anchor_dir", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--skip", default="02,05")
ap.add_argument("--cell", type=int, default=56, help="grid spacing in pixels")
ap.add_argument("--smooth", type=float, default=2e-4, help="penalty per px^2 between neighboring nodes")
ap.add_argument("--max-px", type=float, default=40.0)
args = ap.parse_args()
attempt, adir, work = args.attempt.resolve(), args.anchor_dir.resolve(), args.work.resolve()
dev = torch.device(f"cuda:{args.gpu}")
skip = set(args.skip.split(","))
TOL, OUTER, STEPS, SIGMAS = 0.012, 3, 120, (4.0, 1.5)

fits = json.loads((work / "texture.json").read_text())["fits"]
cams = load_cameras(work / "anchor_cameras.json")
xyz, nrm, _, _ = samples(work, 3_000_000, dev)
hero = Source("hero", Image.open(attempt / "hero" / "hero.png").convert("RGB"),
              np.asarray(Image.open(attempt / "proxy" / "hero_mask.png")) > 127,
              PoseCamera.of(campath.PathCamera.from_json(fits["hero"]["camera"])), 2.0, dev)
sources = [hero] + [Source(n, Image.open(adir / f"{n}.png").convert("RGB"), anchor_mask(adir, n), cams[n], 1.0, dev)
                    for n in sorted(cams) if n not in skip]


def blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, math.ceil(2.5 * sigma))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=img.device, dtype=torch.float32) / sigma) ** 2)
    k = (k / k.sum()).reshape(1, 1, 1, -1)
    c = img.shape[0]
    x = F.conv2d(img[:, None], k, padding=(0, r))
    return F.conv2d(x, k.transpose(2, 3), padding=(r, 0))[:, 0].reshape(c, *img.shape[1:])


# What each view sees, once: sample indices, where they land, weights before the feather.
seen = []
for s in sources:
    u, v, wt = s.visible(xyz, nrm, TOL)
    idx = torch.nonzero(wt > 0)[:, 0]
    seen.append((idx, u[idx], v[idx], wt[idx]))
grids = [torch.zeros(2, math.ceil(s.h / args.cell) + 1, math.ceil(s.w / args.cell) + 1, device=dev) for s in sources]
gains = [torch.ones(3, device=dev) for _ in sources]
offsets = [torch.zeros(3, device=dev) for _ in sources]


def colors(i: int, img: torch.Tensor, grid: torch.Tensor, gain: torch.Tensor, offset: torch.Tensor
           ) -> tuple[torch.Tensor, torch.Tensor]:
    s, (_, u, v, wt) = sources[i], seen[i]
    uw, vw = s.warped(u, v, grid if i else None)
    fe = sample(s.feather, uw, vw, s.w, s.h)[:, 0]
    return sample(img, uw, vw, s.w, s.h) * gain + offset, wt * fe


report: dict = {s.name: {} for s in sources[1:]}
for sigma in SIGMAS:
    imgs = [blur(s.rgb, sigma) for s in sources]
    for rnd in range(OUTER):
        # Every view's current colors, summed over the samples (weights squared, as the blend does).
        num = torch.zeros(len(xyz), 3, device=dev)
        den = torch.zeros(len(xyz), device=dev)
        current = []
        with torch.no_grad():
            for i in range(len(sources)):
                c, w = colors(i, imgs[i], grids[i], gains[i], offsets[i])
                current.append((c, w))
                num.index_add_(0, seen[i][0], (w**2)[:, None] * c)
                den.index_add_(0, seen[i][0], w**2)
        for i in range(1, len(sources)):
            idx = seen[i][0]
            c0, w0 = current[i]
            loo_den = den[idx] - w0**2
            target = (num[idx] - (w0**2)[:, None] * c0) / loo_den.clamp_min(1e-9)[:, None]
            conf = loo_den.clamp_min(0).sqrt()
            grid = grids[i].clone().requires_grad_(True)
            gain = gains[i].clone().requires_grad_(True)
            offset = offsets[i].clone().requires_grad_(True)
            opt = torch.optim.Adam([{"params": [grid], "lr": 0.4 * sigma}, {"params": [gain, offset], "lr": 2e-3}])
            first = last = None
            for step in range(STEPS):
                c, w = colors(i, imgs[i], grid, gain, offset)
                a = torch.minimum(w, conf)
                err = (a[:, None] * (c - target) ** 2).sum() / a.sum().clamp_min(1e-9)
                smooth = ((grid[:, 1:] - grid[:, :-1]) ** 2).mean() + ((grid[:, :, 1:] - grid[:, :, :-1]) ** 2).mean()
                loss = err + args.smooth * smooth + 1e-6 * (grid**2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    grid.clamp_(-args.max_px, args.max_px)
                first = float(err) if first is None else first
                last = float(err)
            grids[i], gains[i], offsets[i] = grid.detach(), gain.detach(), offset.detach()
            mag = grids[i].norm(dim=0)
            report[sources[i].name][f"sigma{sigma}_round{rnd + 1}"] = {
                "err": [round(first, 6), round(last, 6)], "mean_px": round(float(mag.mean()), 2), "max_px": round(float(mag.max()), 2)}
        print(f"blur {sigma} round {rnd + 1}: " + ", ".join(
            f"{s.name} {report[s.name][f'sigma{sigma}_round{rnd + 1}']['mean_px']:.1f}px" for s in sources[1:]), flush=True)

torch.save({s.name: g.cpu() for s, g in zip(sources[1:], grids[1:])}, work / "warps.pt")
(work / "warps.json").write_text(json.dumps(report, indent=1))
print(work / "warps.pt")
