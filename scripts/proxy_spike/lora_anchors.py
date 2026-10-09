"""Anchor views from the GSO view LoRA (board #3699, #3704) instead of fal's Multiple-Angles LoRA: each
anchor is generated at a known camera from the route's own render, so its pose is exact by construction.

    lora_anchors.py ATTEMPT WORK OUT GPU [--lora qwen/gso-view-v1-750.safetensors] [--grid -15,5,30,55]
        [--azimuths 8 | --views YAW:PITCH,...] [--size 768x1024] [--fill 0.65] [--seed 11]
    lora_anchors.py ATTEMPT WORK OUT GPU --finalize      (after register_anchors.py OUT: exact cameras in)

WORK holds the route's mesh.npz and texture.json (the hero's camera fitted to that mesh). As the LoRA was
trained (gso_controls.py, gso_dataset.py): the mesh is painted from the hero where the hero sees it, Pixal3D's
own colors elsewhere, and rendered on black at the target camera (image 1); the hero is the subject on a flat
gray backdrop (image 2, like the GSO heroes: gray 127, the subject ~2/3 of the frame); the prompt is
gso_dataset.caption(yaw relative to the hero camera, giro.path's sign; the camera's pitch in this frame).
Cameras are the hero's with yaw and pitch replaced and the distance set so the subject fills --fill of the
frame (the hero's own framing is landscape for some subjects; the LoRA saw portrait 768x1024 only).
Graph: texture_common.view_lora_workflow with Lightning (the route's setting; scored best for v1 at 750).

Writes OUT/NN.png (the anchors, the route's ANCHOR_DIR layout), OUT/input/NN.png (image 1), OUT/hero_ref.png
(image 2), OUT/views.json (cameras, captions) and OUT/sheet.jpg. --finalize replaces register_anchors.py's
silhouette-fitted cameras in OUT/registration.json with the exact ones (IoU: the mesh's silhouette at the exact
camera against the anchor's SAM mask), so the route refines from the true pose.
"""
import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from giro import path as campath
from gso_dataset import caption
from texture_common import Source, anchor_mask, render_points, samples, view_lora_workflow

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--lora", default="qwen/gso-view-v1-750.safetensors")
ap.add_argument("--grid", default="-15,5,30,55", help="pitches (the training rings span -15..65)")
ap.add_argument("--azimuths", type=int, default=8, help="per pitch, from the hero's yaw")
ap.add_argument("--views", default="", metavar="YAW:PITCH,...",
                help="these cameras (yaw relative to the hero, giro.path's sign) instead of --grid x --azimuths")
ap.add_argument("--size", default="768x1024", help="WxH of image 1 and the anchors (training size)")
ap.add_argument("--fill", type=float, default=0.65, help="the subject's larger extent, as a share of the frame")
ap.add_argument("--points", type=int, default=3_000_000)
ap.add_argument("--paint-tol", type=float, default=0.012)
ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--tries", type=int, default=4, help="new seeds while the output's silhouette disagrees with image 1's")
ap.add_argument("--min-iou", type=float, default=0.75, help="accept an output whose silhouette IoU with image 1's reaches this")
ap.add_argument("--finalize", action="store_true")
args = ap.parse_args()
attempt, work, out = args.attempt.resolve(), args.work.resolve(), args.out.resolve()
W, H = map(int, args.size.split("x"))
dev = torch.device(f"cuda:{args.gpu}")
BASE_WEIGHT = 0.02  # gso_controls.py's: Pixal3D's own color where the hero's weight fades out
SS = 2


def bbox_frac(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return 0.0
    return max((ys.max() - ys.min() + 1) / mask.shape[0], (xs.max() - xs.min() + 1) / mask.shape[1])


def framed(cam: campath.PathCamera, xyz: torch.Tensor, nrm: torch.Tensor) -> campath.PathCamera:
    """The camera moved along its axis so the subject spans --fill of the frame (two perspective steps)."""
    for _ in range(3):
        _, m = render_points(xyz, torch.ones_like(xyz), cam, W // 4, H // 4, nrm=nrm)
        f = bbox_frac(m)
        if f <= 0:
            break
        cam = replace(cam, distance=cam.distance * f / args.fill)
    return cam


def silhouette(img: np.ndarray) -> np.ndarray:
    """The subject on black: pixels off black, holes filled (dark parts inside the outline)."""
    return ndimage.binary_fill_holes(ndimage.binary_opening(img.max(-1) > 0.01, iterations=2))  # the LoRA paints pure black


def sil_iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def hero_ref(hero: Image.Image, mask: np.ndarray) -> Image.Image:
    """The hero's subject on gray 127 in a WxH frame, its larger extent --fill of the frame."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb = np.asarray(hero.convert("RGB"), np.float32)[y0:y1, x0:x1]
    m = mask[y0:y1, x0:x1, None].astype(np.float32)
    sub = Image.fromarray((rgb * m + 127 * (1 - m)).astype(np.uint8))
    s = args.fill * min(W / sub.width, H / sub.height)
    sub = sub.resize((max(1, round(sub.width * s)), max(1, round(sub.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (127, 127, 127))
    canvas.paste(sub, ((W - sub.width) // 2, (H - sub.height) // 2))
    return canvas


hero_cam = campath.PathCamera.from_json(json.loads((work / "texture.json").read_text())["fits"]["hero"]["camera"])

if args.finalize:
    xyz, nrm, _, _ = samples(work, args.points, dev)
    reg = json.loads((out / "registration.json").read_text())
    views = json.loads((out / "views.json").read_text())["views"]
    for v in views:
        cam = campath.PathCamera.from_json(v["camera"])
        mask = anchor_mask(out, v["name"])
        _, sil = render_points(xyz, torch.ones_like(xyz), cam, mask.shape[1], mask.shape[0], nrm=nrm)
        i = float((sil & mask).sum() / max((sil | mask).sum(), 1))
        old = reg[v["name"]]
        reg[v["name"]] = {"yaw_from_hero": round(((cam.yaw - hero_cam.yaw + 180) % 360) - 180, 1), "pitch": round(cam.pitch, 1),
                          "iou": round(i, 4), "camera": cam.to_json(), "silhouette_fit": {k: old[k] for k in ("yaw_from_hero", "pitch", "iou")}}
        print(f"{v['name']}: exact yaw {reg[v['name']]['yaw_from_hero']:+.0f} pitch {cam.pitch:.0f} IoU {i:.3f} "
              f"(silhouette fit: yaw {old['yaw_from_hero']:+.0f} pitch {old['pitch']:.0f} IoU {old['iou']:.3f})", flush=True)
    (out / "registration.json").write_text(json.dumps(reg, indent=1))
    raise SystemExit

out.mkdir(parents=True, exist_ok=True)
(out / "input").mkdir(exist_ok=True)
hero = Image.open(attempt / "hero" / "hero.png").convert("RGB")
hmask = np.asarray(Image.open(attempt / "proxy" / "hero_mask.png").convert("L")) > 127
hero_ref(hero, hmask).save(out / "hero_ref.png")

xyz, nrm, base, _ = samples(work, args.points, dev)
col, wt = Source("hero", hero, hmask, hero_cam, 1.0, dev).paint(xyz, nrm, args.paint_tol)
rgb = (col * wt[:, None] + base * BASE_WEIGHT) / (wt[:, None] + BASE_WEIGHT)

views = []
if args.views:
    plan = [(float(y), float(p)) for y, p in (v.split(":") for v in args.views.split(","))]
else:
    plan = [(360.0 * k / args.azimuths, float(p)) for p in args.grid.split(",") for k in range(args.azimuths)]
for rel, p in plan:
    if True:
        name = f"{len(views):02d}"
        cam = framed(replace(hero_cam, yaw=hero_cam.yaw + rel, pitch=p), xyz, nrm)
        img, m = render_points(xyz, rgb, cam, W * SS, H * SS, nrm=nrm)
        m = m.astype(np.float32)
        blocks = lambda a: a.reshape(H, SS, W, SS, *a.shape[2:]).sum((1, 3))  # noqa: E731
        cov = blocks(m) / SS**2
        color = blocks(img * m[..., None]) / np.maximum(blocks(m), 1e-6)[..., None]
        Image.fromarray((color * cov[..., None] * 255).clip(0, 255).astype(np.uint8)).save(out / "input" / f"{name}.png")
        Image.fromarray((cov * 255).astype(np.uint8), "L").save(out / "input" / f"{name}_mask.png")
        views.append({"name": name, "rel_yaw": round(rel % 360, 2), "pitch": p, "caption": caption(rel, p), "camera": cam.to_json()})
(out / "views.json").write_text(json.dumps({"lora": args.lora, "hero_camera": hero_cam.to_json(), "views": views}, indent=1))


async def generate() -> None:
    from giro.comfy import server
    from giro.comfy.client import ComfyClient, Done

    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                ref = await comfy.upload_image(out / "hero_ref.png", subfolder="giro/lora_anchors")
                # The LoRA sometimes returns the hero's own view instead of image 1's (Oct 6, scooter: about half
                # the first tries): the pose is known, so a try whose silhouette disagrees with image 1's is redone.
                tries_dir = out / "tries"
                tries_dir.mkdir(exist_ok=True)
                for v in views:
                    name = v["name"]
                    target = np.asarray(Image.open(out / "input" / f"{name}_mask.png")) > 127  # image 1's coverage
                    c1 = None
                    scores = []
                    for t in range(args.tries):
                        f = tries_dir / f"{name}-{t}.png"
                        if t == 0 and (out / f"{name}.png").exists() and not f.exists():
                            f.write_bytes((out / f"{name}.png").read_bytes())  # a first try from before retries existed
                        if not f.exists():
                            c1 = c1 or await comfy.upload_image(out / "input" / f"{name}.png", subfolder="giro/lora_anchors_in")
                            done = None
                            wf = view_lora_workflow(c1, ref, v["caption"], args.lora, args.seed + int(name) + 1000 * t, lightning=True,
                                                    prefix="giro/lora_anchors")
                            async for ev in comfy.run(wf):
                                if isinstance(ev, Done):
                                    done = ev
                            await comfy.download(done.outputs["save"]["images"][0], f)
                        img = np.asarray(Image.open(f).convert("RGB").resize(target.shape[::-1]), np.float32) / 255
                        scores.append(sil_iou(silhouette(img), target))
                        if scores[-1] >= args.min_iou:
                            break
                    best = int(np.argmax(scores))
                    (out / f"{name}.png").write_bytes((tries_dir / f"{name}-{best}.png").read_bytes())
                    v["tries"], v["pick"] = [round(x, 3) for x in scores], best
                    print(f"{name}: yaw {v['rel_yaw']:.0f} pitch {v['pitch']:.0f} silhouette IoU {[round(x, 2) for x in scores]} -> try {best}", flush=True)
            finally:
                await comfy.free()

asyncio.run(generate())
(out / "views.json").write_text(json.dumps({"lora": args.lora, "hero_camera": hero_cam.to_json(), "views": views}, indent=1))
th = 256
tiles = []
for v in views:
    a = Image.open(out / "input" / f"{v['name']}.png").convert("RGB").resize((th * W // H, th))
    b = Image.open(out / f"{v['name']}.png").convert("RGB").resize((th * W // H, th))
    t = Image.new("RGB", (2 * a.width, th))
    t.paste(a, (0, 0))
    t.paste(b, (a.width, 0))
    ImageDraw.Draw(t).text((4, 4), f"{v['name']} yaw {v['rel_yaw']:.0f} pitch {v['pitch']:.0f}", fill=(255, 255, 0))
    tiles.append(t)
cols = 4
sheet = Image.new("RGB", (tiles[0].width * cols, th * ((len(tiles) + cols - 1) // cols)))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * t.width, (i // cols) * th))
sheet.save(out / "sheet.jpg", quality=88)
