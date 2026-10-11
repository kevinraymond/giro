"""The symmetry-prior runs' splats (board #3822) at the cameras that matter: train/final.ply (the path
cameras' frame, poses mapper=path) rendered at the hero camera turned by the given yaw offsets at the
hero's pitch, `zoom` times nearer; the first column is the mirrored hero, the column at the mirrored
camera's offset is where the pin should show. Every attempt must share the proxy (the same hero camera).

    mirror_views.py OUT.jpg GPU OFFSETS ZOOM LABEL ATTEMPT [LABEL ATTEMPT ...]   (OFFSETS: 0,41,82,120,160)
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from giro import path as campath
from giro import splat

out, gpu = Path(sys.argv[1]), int(sys.argv[2])
offsets, zoom = [float(v) for v in sys.argv[3].split(",")], float(sys.argv[4])
pairs = list(zip(sys.argv[5::2], map(Path, sys.argv[6::2])))
dev = torch.device(f"cuda:{gpu}")
W, H = 576, 768
hero = campath.PathCamera.from_json(json.loads((pairs[0][1] / "proxy" / "proxy.json").read_text())["hero_camera"])
sign = 1.0
for _, a in pairs:  # the mirror path turns the shorter way to the mirrored camera
    cj = json.loads((a / "cameras.json").read_text())
    if "mirror" in cj:
        sign = 1.0 if cj["frames"][1]["yaw"] >= cj["frames"][0]["yaw"] else -1.0
cams = [replace(hero, yaw=hero.yaw + sign * o, distance=hero.distance / zoom) for o in offsets]


def load(path: Path) -> dict:
    s = splat.read_ply(path)
    g = lambda *k: torch.from_numpy(np.stack([s[x] for x in k], 1)).float().to(dev)  # noqa: E731
    n = len(s)
    return {"means": g("x", "y", "z"), "scales": torch.exp(g("scale_0", "scale_1", "scale_2")),
            "quats": g("rot_0", "rot_1", "rot_2", "rot_3"), "opac": torch.sigmoid(g("opacity")[:, 0]),
            "sh": torch.cat([g("f_dc_0", "f_dc_1", "f_dc_2")[:, None],
                             g(*[f"f_rest_{i}" for i in range(45)]).reshape(n, 3, 15).permute(0, 2, 1)], 1)}


def draw(sp: dict, cam: campath.PathCamera) -> Image.Image:
    from gsplat import rasterization

    r, t = cam.world_to_camera()
    vm = torch.eye(4, device=dev)
    vm[:3, :3], vm[:3, 3] = torch.tensor(r, device=dev), torch.tensor(t, device=dev)
    f = cam.focal(W, H)
    K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev, dtype=torch.float32)
    img, a, _ = rasterization(sp["means"], sp["quats"], sp["scales"], sp["opac"], sp["sh"], vm[None], K[None], W, H,
                              sh_degree=3, rasterize_mode="antialiased")
    im = (img[0].clamp(0, 1) + (1 - a[0]) * 0.5).cpu().numpy()
    return Image.fromarray((im * 255).astype(np.uint8))


ref = next((a / "proxy_render" / "mirror_hero.png" for _, a in pairs if (a / "proxy_render" / "mirror_hero.png").exists()), None)
th = 384
tw = round(th * W / H)
rows = []
with torch.no_grad():
    for label, a in pairs:
        sp = load(a / "train" / "final.ply")
        tiles = [Image.open(ref).convert("RGB") if ref else Image.new("RGB", (W, H))] + [draw(sp, c) for c in cams]
        row = Image.new("RGB", (tw * len(tiles), th + 16), (40, 40, 40))
        d = ImageDraw.Draw(row)
        for k, t in enumerate(tiles):
            row.paste(t.resize((tw, th), Image.LANCZOS), (k * tw, 16))
            if k:
                d.text((k * tw + 4, 2), f"{sign * offsets[k - 1]:+.0f} deg", fill=(200, 200, 200))
        d.text((4, 2), label + ("  (mirrored hero)" if ref else ""), fill=(255, 255, 0))
        rows.append(row)
sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), (40, 40, 40))
for i, r in enumerate(rows):
    sheet.paste(r, (0, i * r.height))
sheet.save(out, quality=90)
