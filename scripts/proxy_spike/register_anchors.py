"""Register anchor views to a proxy: SAM masks each anchor (the masks stage's prompts), then the proxy
stage's hero-camera fit finds the camera from which the proxy's silhouette matches it best. Writes
OUT/registration.json (per anchor: yaw/pitch relative to the hero camera, IoU) and OUT/registered.jpg
(each anchor with the fitted proxy silhouette outlined).

    register_anchors.py ATTEMPT ANCHOR_DIR GPU
"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient
from giro.stages.masks import Masks, combine, sam_workflow
from giro.stages.proxy import Points, fit_hero_camera

attempt, adir, gpu = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
anchors = sorted(p for p in adir.glob("[0-9][0-9].png"))
raw = adir / "raw"


async def masks() -> None:
    with await asyncio.to_thread(server.Lease, gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                async for _ in comfy.run(sam_workflow({"anchors": anchors}, raw, Masks.defaults)):
                    pass
            finally:
                await comfy.free()

asyncio.run(masks())
hero_cam = campath.PathCamera.from_json(json.loads((attempt / "proxy" / "proxy.json").read_text())["hero_camera"])
points = Points(attempt / "proxy" / "proxy.ply")
p = Masks.defaults
result, tiles = {}, []
for a in anchors:
    subject = np.asarray(Image.open(raw / "subject" / "anchors" / a.name)) > 127
    bg_path = raw / "background" / "anchors" / a.name
    mask = combine(subject, np.asarray(Image.open(bg_path)) > 127 if bg_path.exists() else None, p["touch_px"], p["gap_px"], p["max_add"])
    img = Image.open(a).convert("RGB")
    cam, fit = fit_hero_camera(points, img, mask, hero_cam.fov)
    rel_yaw = ((cam.yaw - hero_cam.yaw + 180) % 360) - 180
    result[a.stem] = {"yaw_from_hero": round(rel_yaw, 1), "pitch": round(cam.pitch, 1), "iou": fit["iou"], "camera": cam.to_json()}
    h = 320
    w = round(h * img.width / img.height)
    sil = points.silhouette(cam, w, h)
    edge = sil & ~ndimage.binary_erosion(sil, iterations=2)
    base = np.asarray(img.resize((w, h)), dtype=np.float64) / 255
    base[edge] = [1.0, 0.2, 0.2]
    tile = Image.fromarray((base * 255).astype(np.uint8))
    ImageDraw.Draw(tile).text((4, 4), f"{a.stem}: yaw {rel_yaw:+.0f} pitch {cam.pitch:.0f} IoU {fit['iou']:.2f}", fill=(255, 255, 0))
    tiles.append(tile)
    print(a.stem, result[a.stem]["yaw_from_hero"], result[a.stem]["pitch"], fit["iou"], flush=True)
(adir / "registration.json").write_text(json.dumps(result, indent=1))
cols = 6
W = max(t.width for t in tiles)
sheet = Image.new("RGB", (W * cols, 320 * ((len(tiles) + cols - 1) // cols)), (40, 40, 40))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * W, (i // cols) * 320))
sheet.save(adir / "registered.jpg", quality=90)
