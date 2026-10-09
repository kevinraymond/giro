"""Place registered anchor views in an attempt's refined world, so they can be trained on.

register_anchors.py fits each anchor's camera to the proxy, in the proxy's frame. The attempt's
poses live in the refined world bundle adjustment left; the similarity between the path's nominal
cameras and the refined ones (as path_poses.reconstruct fits it) carries the anchors across. The
fit is then refined against the attempt's trained splat (its centers, brought back into the proxy's
frame), whose shape is closer to the anchors' than the proxy's.

    anchor_views.py ATTEMPT ANCHOR_DIR GPU

Writes ANCHOR_DIR/mapped.json (per anchor: COLMAP pose and focal in the refined world, the proxy
and splat fit IoUs) and ANCHOR_DIR/mapped.jpg (anchor, the trained splat rendered from its camera,
and the two blended) to check the placement by eye.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import optimize

from giro import path as campath
from giro import render
from giro.stages.fallback import _centers, _pose, similarity
from giro.stages.masks import Masks, combine
from giro.stages.path_poses import path_cameras
from giro.stages.poses import read_images_txt
from giro.stages.proxy import REFINE_HEIGHT, Points, iou

attempt, adir, gpu = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
reg = json.loads((adir / "registration.json").read_text())


def anchor_mask(name: str) -> np.ndarray:
    raw, p = adir / "raw", Masks.defaults
    subject = np.asarray(Image.open(raw / "subject" / "anchors" / f"{name}.png")) > 127
    bg = raw / "background" / "anchors" / f"{name}.png"
    return combine(subject, np.asarray(Image.open(bg)) > 127 if bg.exists() else None, p["touch_px"], p["gap_px"], p["max_add"])


# Proxy frame -> refined world, from the frames (the hero shares frame 0's pose).
cams = path_cameras(attempt)
refined = {n: _pose(im) for n, im in read_images_txt(attempt / "poses" / "colmap" / "model_txt" / "images.txt").items()
           if n.startswith("frames/")}
names = sorted(refined)
nominal = {n: cams[n][0].world_to_camera() for n in names}
s, r, t = similarity(_centers(nominal, names), _centers(refined, names))
resid = np.linalg.norm(s * (r @ _centers(nominal, names).T).T + t - _centers(refined, names), axis=1)
radius = float(np.mean([cams[n][0].distance for n in names]))
print(f"similarity: scale {s:.4f}, rotation {np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))):.2f} deg, "
      f"residual median {np.median(resid) / (s * radius):.4f} radii", flush=True)

# The trained splat's centers in the proxy's frame.
ply = attempt / "train" / "final.ply"
splat_pts = Points(ply, min_opacity=0.3)
splat_pts.xyz = ((splat_pts.xyz - t) @ r) / s


def refine(cam: campath.PathCamera, mask: np.ndarray) -> tuple[campath.PathCamera, float, float]:
    h0, w0 = mask.shape
    big = (round(REFINE_HEIGHT * w0 / h0), REFINE_HEIGHT)
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(big)) > 127
    rot, _ = cam.world_to_camera()

    def moved(x: np.ndarray) -> campath.PathCamera:
        target = np.asarray(cam.target) + x[3] * cam.distance * rot[0] + x[4] * cam.distance * rot[1]
        return replace(cam, yaw=cam.yaw + x[0], pitch=cam.pitch + x[1], distance=cam.distance * np.exp(x[2]),
                       target=tuple(target))

    before = iou(splat_pts.silhouette(cam, *big), m)
    res = optimize.minimize(lambda x: -iou(splat_pts.silhouette(moved(x), *big), m), np.zeros(5), method="Nelder-Mead",
                            options={"initial_simplex": np.vstack([np.zeros(5), np.diag([4.0, 3.0, 0.05, 0.02, 0.02])]),
                                     "maxiter": 200, "xatol": 0.05, "fatol": 1e-4})
    return moved(res.x), before, -res.fun


out, tiles = {}, []
for name, rec in sorted(reg.items()):
    img = Image.open(adir / f"{name}.png").convert("RGB")
    w, h = img.size
    mask = anchor_mask(name)
    cam0 = campath.PathCamera.from_json(rec["camera"])
    cam, before, after = refine(cam0, mask)
    # Into the refined world: center s R c + t, rotation R_wc R^T.
    rot_p, _ = cam.world_to_camera()
    rot = rot_p @ r.T
    center = s * r @ cam.position() + t
    tvec = -rot @ center
    focal = cam.focal(w, h)
    out[name] = {"rot": rot.tolist(), "tvec": tvec.tolist(), "focal": focal, "width": w, "height": h,
                 "proxy_iou": rec["iou"], "splat_iou_before": round(before, 4), "splat_iou": round(after, 4),
                 "moved_deg": [round(cam.yaw - cam0.yaw, 2), round(cam.pitch - cam0.pitch, 2)],
                 "camera_proxy_frame": cam.to_json()}
    print(name, f"proxy IoU {rec['iou']:.3f}, splat IoU {before:.3f} -> {after:.3f}, "
          f"moved yaw {cam.yaw - cam0.yaw:+.1f} pitch {cam.pitch - cam0.pitch:+.1f}", flush=True)

    # Render the trained splat from the anchor's camera (PLY frame = refined world).
    th = 384
    tw = round(th * w / h)
    vfov = float(np.degrees(2 * np.arctan((h / 2) / focal)))
    view = render.render(ply, [render.Camera(center, center + rot[2], -rot[1], vfov)], (tw, th), gpu=gpu)[0]
    a = img.resize((tw, th))
    blend = Image.blend(a, view, 0.5)
    tile = Image.new("RGB", (3 * tw, th))
    for i, im in enumerate((a, view, blend)):
        tile.paste(im, (i * tw, 0))
    ImageDraw.Draw(tile).text((4, 4), f"{name}: splat IoU {before:.2f} -> {after:.2f}", fill=(255, 255, 0))
    tiles.append(tile)

(adir / "mapped.json").write_text(json.dumps({"similarity": {"s": s, "r": r.tolist(), "t": t.tolist()}, "anchors": out}, indent=1))
cols = 2
W, H = tiles[0].size
sheet = Image.new("RGB", (W * cols, H * ((len(tiles) + cols - 1) // cols)), (40, 40, 40))
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % cols) * W, (i // cols) * H))
sheet.save(adir / "mapped.jpg", quality=88)
print(adir / "mapped.jpg")
