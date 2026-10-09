"""Pixal3D multi-view proxies: the hero plus registered anchor views condition one mesh, instead
of the hero alone (TencentARC's multi-view weights, Aug 31 2026; Comfy-Org/Pixal3D
pixal3d_multiview_bf16). Writes OUT/mesh_pixal3dmv-<mode><k>_<seed>.npz, which pick_seed.py
scores next to the single-view seeds.

    pixal3d_mv.py ATTEMPT ROUTE_OUT OUT GPU --mode rig|posed
        [--views 4] [--seeds 201,202,203]

ROUTE_OUT is a texture_route.py output (angles/ with registration.json, work/ with mesh.npz,
texture.json's hero fit and anchor_cameras.json, all in work/mesh.npz's splat frame).

- rig: ComfyUI's Pixal3DMultiViewConditioning, its fixed rig: front = the hero, left, back,
  right = the accepted anchors nearest those directions at low pitch, each cropped to its mask
  (pad 1.1) like the template; one fov (--rig-fov).
- posed: giro's GiroPixal3DPosedConditioning with each view's registered camera, turned into
  Pixal3D's projection world (z up, front camera at -y; mesh vertex v sits at (vx, -vz, vy)).
  Each view is cropped square about its principal point, so the pinhole stays centered, and
  the crop's fov is exact. --views K takes the hero and the K-1 anchors that spread best
  (farthest-point over view directions). Before running, the route's mesh is projected through
  every view's camera and its silhouette IoU with the view's mask is printed (a frame check).
"""
import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient
from giro.stages.proxy import trellis_workflow
from texture_common import PoseCamera, anchor_mask

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("route", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--mode", choices=["rig", "posed"], default="posed")
ap.add_argument("--views", type=int, default=4, help="posed: the hero and this many minus one anchors")
ap.add_argument("--max-pitch", type=float, default=50.0, help="posed: anchors looking down/up more than this are left out")
ap.add_argument("--rig-fov", type=float, default=20.0)
ap.add_argument("--seeds", default="201,202,203")
ap.add_argument("--unet", default="pixal3d/pixal3d_multiview_bf16.safetensors")
ap.add_argument("--dry", action="store_true", help="write the views and the frame check, run nothing")
args = ap.parse_args()
attempt, route, out = args.attempt.resolve(), args.route.resolve(), args.out.resolve()
seeds = [int(s) for s in args.seeds.split(",")]
work, adir = route / "work", route / "angles"
tag = f"{args.mode}{args.views if args.mode == 'posed' else 4}"
vdir = out / f"views-{tag}"
vdir.mkdir(parents=True, exist_ok=True)

# The views: the hero (its fit to the route's mesh) and the photometrically registered anchors.
hero_img = Image.open(attempt / "hero" / "hero.png").convert("RGB")
hero_mask = np.asarray(Image.open(attempt / "proxy" / "hero_mask.png")) > 127
fits = json.loads((work / "texture.json").read_text())["fits"]
cams = json.loads((work / "anchor_cameras.json").read_text())["cameras"]
views = [("hero", hero_img, hero_mask, PoseCamera.of(campath.PathCamera.from_json(fits["hero"]["camera"])))]
for name in sorted(cams):
    if cams[name].get("accepted"):
        views.append((name, Image.open(adir / f"{name}.png").convert("RGB"), anchor_mask(adir, name),
                      PoseCamera.from_json(cams[name]["camera"])))

# Splat frame (texture_common.samples: y-up vertices flipped to y down, centered, 1 tall) and back.
v = np.load(work / "mesh.npz")["vertices"].astype(np.float64)
flip = np.array([1.0, -1.0, -1.0])
lo, hi = (v * flip).min(0), (v * flip).max(0)
center, height = (lo + hi) / 2, float(hi[1] - lo[1])
M = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # splat-frame direction -> projection world (A @ diag(flip))
GL = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> OpenGL


def to_world(cam: PoseCamera) -> np.ndarray:
    """c2w (OpenGL axes) in Pixal3D's projection world of a camera posed in the splat frame."""
    r, t = cam.world_to_camera()
    rw = GL @ r @ M.T
    tw = GL @ (height * t - r @ center)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = rw.T, -rw.T @ tw
    return c2w


def direction(cam: PoseCamera) -> np.ndarray:
    p = cam.position()  # the splat frame is centered on the subject
    return p / np.linalg.norm(p)


def pitch_of(cam: PoseCamera) -> float:
    return math.degrees(math.asin(np.clip(-direction(cam)[1], -1, 1)))  # y down: above is -y


def centered_crop(img: Image.Image, mask: np.ndarray, fov_small: float) -> tuple[Image.Image, float]:
    """Square crop about the image center that holds the mask (5% margin), black outside the mask
    and the image; its horizontal fov."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    half = 1.05 * max(np.abs(xs - w / 2).max(), np.abs(ys - h / 2).max(), 8)
    side = int(math.ceil(2 * half))
    rgb = np.asarray(img).astype(np.uint8) * mask[..., None]
    canvas = np.zeros((side, side, 3), np.uint8)
    x0, y0 = int(round(w / 2 - side / 2)), int(round(h / 2 - side / 2))
    sx, sy = max(0, -x0), max(0, -y0)
    ix, iy = max(0, x0), max(0, y0)
    cw, ch = min(w - ix, side - sx), min(h - iy, side - sy)
    canvas[sy:sy + ch, sx:sx + cw] = rgb[iy:iy + ch, ix:ix + cw]
    fov = 2 * math.degrees(math.atan(math.tan(math.radians(fov_small) / 2) * side / min(w, h)))
    return Image.fromarray(canvas).resize((1024, 1024), Image.LANCZOS), fov


def project_iou(c2w: np.ndarray, fov_x: float, crop: Image.Image) -> float:
    """The route's mesh through Pixal3D's projection (_project_points_to_image) vs the crop's mask."""
    w = v @ np.array([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]]).T  # vertex -> projection world
    w2c = np.linalg.inv(c2w)
    p = w @ w2c[:3, :3].T + w2c[:3, 3]
    f = 512 / math.tan(math.radians(fov_x) / 2)
    z = -p[:, 2]
    ok = z > 1e-6
    u = (f * p[ok, 0] / z[ok] + 512).astype(int)
    vv = (-f * p[ok, 1] / z[ok] + 512).astype(int)
    keep = (u >= 0) & (u < 1024) & (vv >= 0) & (vv < 1024)
    sil = np.zeros((1024, 1024), bool)
    sil[vv[keep], u[keep]] = True
    from scipy import ndimage
    sil = ndimage.binary_closing(ndimage.binary_dilation(sil, iterations=2), iterations=3)
    m = np.asarray(crop).max(-1) > 8
    return float((sil & m).sum() / max((sil | m).sum(), 1))


rig_dirs = {"left": np.array([1.0, 0, 0]), "back": np.array([0, 0, 1.0]), "right": np.array([-1.0, 0, 0])}
if args.mode == "rig":
    # rig 'left' (azimuth 90) is the camera at world +x = splat +x: the side on the front view's right
    picked = {"front": views[0]}
    low = [x for x in views[1:] if abs(pitch_of(x[3])) < 15]
    for side, d in rig_dirs.items():
        picked[side] = max(low, key=lambda x: float(direction(x[3]) @ d))
    meta = {}
    for side, (name, img, mask, cam) in picked.items():
        rgba = np.dstack([np.asarray(img), mask.astype(np.uint8) * 255])
        Image.fromarray(rgba, "RGBA").save(vdir / f"{side}.png")
        meta[side] = {"source": name, "pitch": round(pitch_of(cam), 1),
                      "azimuth": round(math.degrees(math.atan2(direction(cam)[0], -direction(cam)[2])), 1)}
    (vdir / "views.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
else:
    pool = [x for x in views[1:] if abs(pitch_of(x[3])) <= args.max_pitch]
    chosen = [views[0]]
    while len(chosen) < args.views and pool:
        far = max(pool, key=lambda x: min(1 - float(direction(x[3]) @ direction(c[3])) for c in chosen))
        chosen.append(far)
        pool.remove(far)
    entries = []
    for k, (name, img, mask, cam) in enumerate(chosen):
        crop, fov_x = centered_crop(img, mask, cam.fov)
        crop.save(vdir / f"view_{k:02d}.png")
        c2w = to_world(cam)
        check = project_iou(c2w, fov_x, crop)
        entries.append({"image": f"view_{k:02d}.png", "source": name, "c2w": c2w.tolist(), "fov_x": fov_x,
                        "pitch": round(pitch_of(cam), 1), "iou_route_mesh": round(check, 3)})
        print(f"{name}: pitch {pitch_of(cam):.1f} fov_x {fov_x:.1f} route-mesh IoU {check:.3f}", flush=True)
    (vdir / "cameras.json").write_text(json.dumps({"views": entries}, indent=1))
if args.dry:
    sys.exit(0)


def workflow(uploads: dict[str, str]) -> dict:
    params = json.loads((attempt / ".stages" / "proxy.json").read_text())["params"]
    wf = trellis_workflow(uploads.get("front", "unused.png"), seeds, out / f"candidates-{tag}", params, "pixal3d")
    for k in ("moge", "geometry", "fov"):
        wf.pop(k, None)
    wf["unet"]["inputs"]["unet_name"] = args.unet
    if args.mode == "posed":
        for k in ("rgba", "mask", "crop"):
            wf.pop(k)
        wf["cond"] = {"class_type": "GiroPixal3DPosedConditioning", "inputs": {"clip_vision_model": ["dino", 0], "views_dir": str(vdir)}}
    else:
        cond = {"clip_vision_model": ["dino", 0], "fov": args.rig_fov}
        for side, name in uploads.items():
            wf[f"{side}_rgba"] = {"class_type": "LoadImage", "inputs": {"image": name, "upload": "image"}}
            wf[f"{side}_mask"] = {"class_type": "InvertMask", "inputs": {"mask": [f"{side}_rgba", 1]}}
            wf[f"{side}_crop"] = {"class_type": "ImageCropToMask", "inputs": {
                "images": [f"{side}_rgba", 0], "masks": [f"{side}_mask", 0], "width": 1024, "height": 1024,
                "pad_factor": 1.1, "grow_mask": 0, "background": "#000000"}}
            cond[side] = [f"{side}_crop", 0]
        for k in ("rgba", "mask", "crop"):
            wf.pop(k)
        wf["cond"] = {"class_type": "Pixal3DMultiViewConditioning", "inputs": cond}
    for s in seeds:
        wf[f"save_mesh_{s}"] = {"class_type": "GiroSaveMesh", "inputs": {"mesh": [f"paint_{s}", 0],
                                                                       "path": str(out / f"mesh_pixal3dmv-{tag}_{s}.npz")}}
    return wf


async def main() -> None:
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                uploads = {}
                if args.mode == "rig":
                    for side in ("front", "left", "back", "right"):
                        uploads[side] = await comfy.upload_image(vdir / f"{side}.png")
                async for _ in comfy.run(workflow(uploads)):
                    pass
            finally:
                await comfy.free()

asyncio.run(main())
print(out)
