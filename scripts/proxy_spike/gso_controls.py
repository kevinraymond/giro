"""Control images for the view-LoRA pilot (board #3699): for each Google Scanned Object rendered by
gso_render.py, what the texture route would show at each target camera, and that camera mapped into
the object's own frame so gso_render.py --aligned can render the matching ground truth.

    GIRO_COMFY_DIR=vendor/comfyui-next gso_controls.py RENDERS_DIR OUT_DIR GPU [--names a,b | --only N]
        [--split train,heldout --manifest CSV] [--shard J/K] [--seed 46] [--points 3000000]

Per object, as the route does it (proxy stage, proxy_mesh.py, project_texture.py):
1. Pixal3D on the hero (its alpha as the mask, where the route has SAM's), the mesh kept (GiroSaveMesh)
   and its surface as a splat for the hero camera fit (the proxy stage's fit_hero_camera, fov 35).
2. The mesh sampled densely; each sample painted from the hero where the hero sees it (|cos|^4 times
   the mask-edge feather, the route's weight), elsewhere Pixal3D's own colors.
3. The targets as the route makes its rings: the hero camera with yaw = hero yaw + the plan's
   relative yaw and the plan's pitch, in the proxy's frame; rendered 2x supersampled on black
   (control/<view>.png, control/<view>_mask.png: coverage).
4. Each target camera mapped into the GT frame through the hero: the proxy's hero camera and the
   GT hero camera see the same image, so camera coordinates agree up to scale s = GT distance /
   proxy distance (gt_cameras.json, COLMAP world-to-camera). The caption's camera is the route's:
   yaw relative to the hero, the camera's own pitch in the proxy frame.

Writes OUT_DIR/<name>/{work/mesh.npz, control/, gt_cameras.json, hero_fit.jpg} and OUT_DIR/controls.log.

v2 (board #3709): --mesh-from V1_DIR reuses v1's Pixal3D mesh and hero fit (gso_shrink.py first puts
the decimated mesh in OUT_DIR/<name>/work/), so only the new plan's cameras are mapped, no Pixal3D;
--no-render skips the control renders (gso_progressive.py renders v2's controls from the decimated
mesh). gt_cameras.json also keeps the proxy's hero camera ("hero_camera"). ComfyUI starts only when
an object needs Pixal3D.
"""
import argparse
import asyncio
import csv
import json
import shutil
import time
from contextlib import AsyncExitStack
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient
from giro.stages.proxy import Points, fit_hero_camera, fit_sheet, trellis_workflow
from texture_common import Source, render_points, samples

ap = argparse.ArgumentParser()
ap.add_argument("renders", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--names", default="")
ap.add_argument("--manifest", type=Path, help="gso_manifest.py's manifest.csv, with --split")
ap.add_argument("--split", default="train,heldout")
ap.add_argument("--only", type=int, default=0)
ap.add_argument("--shard", default="", metavar="J/K", help="only every K-th object, from the J-th (0-based)")
ap.add_argument("--seed", type=int, default=46)
ap.add_argument("--points", type=int, default=3_000_000, help="surface samples")
ap.add_argument("--paint-tol", type=float, default=0.012, help="depth tolerance for painting (project_texture.py's)")
ap.add_argument("--supersample", type=int, default=2)
ap.add_argument("--mesh-from", type=Path, help="v1 controls dir: reuse its hero fit (mesh: OUT_DIR/<name>/work/mesh.npz)")
ap.add_argument("--no-render", action="store_true", help="map the cameras only, no control renders")
args = ap.parse_args()
dev = torch.device(f"cuda:{args.gpu}")
PARAMS = {"n_gaussians": 262_144}
FOV = 35.0  # the proxy stage's hero fov
BASE_WEIGHT = 0.02  # Pixal3D's own color, where the hero's weight fades out at grazing angles

if args.names:
    names = args.names.split(",")
elif args.manifest:
    with open(args.manifest) as f:
        names = [r["name"] for r in csv.DictReader(f) if r["split"] in args.split.split(",")]
else:
    names = sorted(p.name for p in args.renders.iterdir() if (p / "cameras.json").exists())
if args.only:
    names = names[:args.only]
if args.shard:  # J/K: every K-th object from the J-th, over the fixed list, so runs on two GPUs never overlap
    j, k = map(int, args.shard.split("/"))
    names = names[j::k]
args.out.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)
    with open(args.out / "controls.log", "a") as f:
        f.write(msg + "\n")


def gt_camera(c_proxy: campath.PathCamera, hero_p: campath.PathCamera, hero_g: campath.PathCamera) -> dict:
    """c_proxy in the GT frame: x_g = s M x_p + b with M = R_g^T R_p, b = R_g^T (s t_p - t_g), from
    identifying the two hero cameras' coordinates (scaled by s); returns COLMAP's R, t there."""
    rp, tp = hero_p.world_to_camera()
    rg, tg = hero_g.world_to_camera()
    s = hero_g.distance / hero_p.distance
    m = rg.T @ rp
    b = rg.T @ (s * tp - tg)
    rc, tc = c_proxy.world_to_camera()
    r = rc @ m.T
    return {"rot": r.tolist(), "t": (s * tc - r @ b).tolist(), "fov": c_proxy.fov, "scale": s}


def reused_fit(name: str) -> tuple[campath.PathCamera, dict] | None:
    """v1's proxy hero camera for `name`: saved as hero_camera, or (v1 files) any view's proxy
    camera turned back to the hero's yaw and pitch (a view's camera is the hero's with those replaced)."""
    g = args.mesh_from / name / "gt_cameras.json"
    if not g.exists():
        return None
    d = json.loads(g.read_text())
    if "hero_camera" in d:
        return campath.PathCamera.from_json(d["hero_camera"]), {"iou": d["hero_fit"]["iou"]}
    c = campath.PathCamera.from_json(d["views"][0]["proxy_camera"])
    return replace(c, yaw=d["hero_fit"]["yaw"], pitch=d["hero_fit"]["pitch"]), {"iou": d["hero_fit"]["iou"]}


def paint_and_render(work: Path, hero: Image.Image, mask: np.ndarray, hero_cam: campath.PathCamera, plan: list[dict],
                     od: Path) -> list[dict]:
    if args.no_render:
        return [{"name": v["name"], "rel_yaw": round(v["rel_yaw"] % 360, 2), "pitch": round(v["pitch"], 2),
                 "proxy_camera": replace(hero_cam, yaw=hero_cam.yaw + v["rel_yaw"], pitch=v["pitch"]).to_json()} for v in plan]
    xyz, nrm, base, _ = samples(work, args.points, dev)
    src = Source("hero", hero, mask, hero_cam, 1.0, dev)
    col, wt = src.paint(xyz, nrm, args.paint_tol)
    rgb = (col * wt[:, None] + base * BASE_WEIGHT) / (wt[:, None] + BASE_WEIGHT)
    (od / "control").mkdir(parents=True, exist_ok=True)
    w, h = hero.size
    ss = args.supersample
    views = []
    for v in plan:
        cam = replace(hero_cam, yaw=hero_cam.yaw + v["rel_yaw"], pitch=v["pitch"])
        img, m = render_points(xyz, rgb, cam, w * ss, h * ss, nrm=nrm)
        m = m.astype(np.float32)
        blocks = lambda a: a.reshape(h, ss, w, ss, *a.shape[2:]).sum((1, 3))  # noqa: E731
        cov = blocks(m) / ss**2
        color = blocks(img * m[..., None]) / np.maximum(blocks(m), 1e-6)[..., None]
        Image.fromarray((color * cov[..., None] * 255).clip(0, 255).astype(np.uint8)).save(od / "control" / f"{v['name']}.png")
        Image.fromarray((cov * 255).astype(np.uint8), "L").save(od / "control" / f"{v['name']}_mask.png")
        views.append({"name": v["name"], "rel_yaw": round(v["rel_yaw"] % 360, 2), "pitch": round(v["pitch"], 2),
                      "proxy_camera": cam.to_json(), "coverage": round(float(cov.mean()), 4)})
    return views


async def main() -> None:
    async with AsyncExitStack() as stack:
        comfy = None

        async def need_comfy() -> ComfyClient:
            nonlocal comfy
            if comfy is None:
                lease = stack.enter_context(await asyncio.to_thread(server.Lease, args.gpu))
                comfy = await stack.enter_async_context(ComfyClient(lease.url))
                stack.push_async_callback(comfy.free)
            return comfy

        for i, name in enumerate(names):
            od = args.out / name
            if (od / "gt_cameras.json").exists():
                continue
            work = od / "work"
            work.mkdir(parents=True, exist_ok=True)
            rd = args.renders / name
            cams = json.loads((rd / "cameras.json").read_text())
            hero_g = campath.PathCamera.from_json(cams["views"][0])
            plan = cams["views"][1:]
            rgba = Image.open(rd / "hero_rgba.png")
            hero = Image.open(rd / "hero.png").convert("RGB")
            mask = np.asarray(rgba.split()[3]) > 127
            hero_rgba = hero.copy()
            hero_rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255, "L"))
            hero_rgba.save(work / "hero_rgba.png")
            t0 = time.monotonic()
            reuse = reused_fit(name) if args.mesh_from else None
            if reuse is not None and (work / "mesh.npz").exists():
                hero_cam, fit = reuse
                t_gen = t_fit = 0.0
            else:
                cf = await need_comfy()
                image = await cf.upload_image(work / "hero_rgba.png")
                wf = trellis_workflow(image, [args.seed], work / "candidates", PARAMS, "pixal3d")
                wf["save_mesh"] = {"class_type": "GiroSaveMesh", "inputs": {"mesh": [f"paint_{args.seed}", 0], "path": str(work / "mesh.npz")}}
                async for _ in cf.run(wf):
                    pass
                t_gen = time.monotonic() - t0
                points = Points(work / "candidates" / f"proxy_{args.seed}.ply")
                hero_cam, fit = fit_hero_camera(points, hero, mask, FOV)
                fit_sheet(points, hero_cam, hero, mask).save(od / "hero_fit.jpg", quality=85)
                t_fit = time.monotonic() - t0 - t_gen
            views = paint_and_render(work, hero, mask, hero_cam, plan, od)
            for v in views:
                c = campath.PathCamera.from_json(v["proxy_camera"])
                v |= gt_camera(c, hero_cam, hero_g)
            t_all = time.monotonic() - t0
            (od / "gt_cameras.json").write_text(json.dumps({
                "object": name, "hero_fit": {"iou": fit["iou"], "yaw": hero_cam.yaw, "pitch": hero_cam.pitch,
                                             "distance": hero_cam.distance, "gt_pitch": hero_g.pitch,
                                             "gt_distance": hero_g.distance},
                "hero_camera": hero_cam.to_json(), "reused": reuse is not None,
                "caption": "yaw = rel_yaw (camera yaw - hero yaw, giro.path sign: + moves the camera to the "
                           "hero camera's right), pitch = the camera's pitch in the proxy frame (+ looks down)",
                "seconds": {"pixal3d": round(t_gen, 1), "fit": round(t_fit, 1), "total": round(t_all, 1)},
                "views": views}, indent=1) + "\n")
            shutil.rmtree(work / "candidates", ignore_errors=True)  # the splat was only for the fit
            log(f"[{i + 1}/{len(names)}] {name}: hero IoU {fit['iou']:.3f}, pitch {hero_cam.pitch:.1f} "
                f"(GT {hero_g.pitch:.1f}), Pixal3D {t_gen:.0f} s, fit {t_fit:.0f} s, total {t_all:.0f} s"
                + (" (v1 mesh and fit)" if reuse is not None else ""))


asyncio.run(main())
