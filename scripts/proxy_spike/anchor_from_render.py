"""Anchors made from the textured mesh's own renders instead of from the hero alone (cf. TEXTure,
Text2Tex): a render at an exact pose, made a clean photo of the same view by Qwen-Image-Edit 2511
with the hero as picture 2, lines up with the mesh by construction (no registration, no warps) and
starts from what the neighboring views painted, so its detail continues theirs. A test makes the
edits of a few frames at a few denoise strengths and checks they keep the mesh's edges; with
--attempt-out, every Nth frame is edited at the first strength and laid out as an attempt that
project_texture.py --views paints back (and trains, with the base's poses).

    anchor_from_render.py WORK GPU --hero HERO.png --frames 42,96,138 [--denoise 0.4,0.6,0.8]
        [--subject "desert tan M1 Abrams tank"] [--out WORK/from_render]
    anchor_from_render.py WORK GPU --hero HERO.png --frames every:2 --denoise 0.6 --attempt-out WORK/attempt-qwen

WORK holds a textured attempt (WORK/attempt: frames/, masks/frames/, cameras.json) and mesh.npz.
Writes OUT/<frame>-<denoise>.png and OUT/sheet.jpg (per frame: render, then each edit with the
mesh's depth edges in red), and OUT/scores.json: edge F1 of each image against the mesh's depth
edges (the render's own F1 is the baseline: an edit that keeps it did not move the geometry).
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient, Done
from texture_common import project, qwen_edit_workflow, samples, zbuffer

ap = argparse.ArgumentParser()
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--hero", type=Path, required=True)
ap.add_argument("--frames", required=True, help="frame indices of WORK/attempt, comma separated, or every:N")
ap.add_argument("--denoise", default="0.4,0.6,0.8")
ap.add_argument("--subject", default="object")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", type=Path)
ap.add_argument("--attempt-out", type=Path, help="lay the edits out as an attempt here (frames/, masks/frames/, cameras.json)")
args = ap.parse_args()
work = args.work.resolve()
out = (args.out or work / "from_render").resolve()
out.mkdir(parents=True, exist_ok=True)
dev = torch.device(f"cuda:{args.gpu}")
strengths = [float(d) for d in args.denoise.split(",")]
cams_json = json.loads((work / "attempt" / "cameras.json").read_text())
if args.frames.startswith("every:"):
    frames = list(range(0, len(cams_json["frames"]), int(args.frames.split(":")[1])))
else:
    frames = [int(f) for f in args.frames.split(",")]
if args.attempt_out:
    strengths = strengths[:1]
    dest = args.attempt_out.resolve()
    for d in ("frames", "masks/frames"):
        (dest / d).mkdir(parents=True, exist_ok=True)
    (dest / "cameras.json").write_text(json.dumps(cams_json, indent=1))
W, H = cams_json["width"], cams_json["height"]
xyz, _, _, _ = samples(work, 3_000_000, dev)
PROMPT = (f"Picture 1 is a rough render of a {args.subject}; picture 2 is a photo of the same {args.subject}. "
          f"Make picture 1 a clean, sharp, realistic photo of the {args.subject}, exactly this view and framing: "
          f"keep every shape, edge and part where it is, and turn the smeared, patchy, noisy surfaces into the clean, "
          f"coherent detail and paint that belong there, lit evenly like picture 2. Black background.")


def depth_edges(cam: campath.PathCamera) -> tuple[np.ndarray, np.ndarray]:
    u, v, z = project(cam, W, H, xyz)
    d = zbuffer(u, v, z, W, H)[3].cpu().numpy()
    mask = np.isfinite(d)
    d = np.where(mask, d, np.nanmax(np.where(mask, d, np.nan)) + 0.5)
    return (np.hypot(ndimage.sobel(d, 0), ndimage.sobel(d, 1)) > 0.04) & ndimage.binary_dilation(mask, iterations=2), mask


def edge_f1(img: np.ndarray, medge: np.ndarray, mask: np.ndarray) -> float:
    """The image's edges (top 10% gradient in the mask) against the mesh's depth edges, 3 px (pick_seed.py's)."""
    g = img.mean(-1)
    gi = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    iedge = (gi > np.percentile(gi[mask], 90)) & mask
    rec = (ndimage.distance_transform_edt(~medge)[iedge] <= 3).mean()
    prec = (ndimage.distance_transform_edt(~iedge)[medge] <= 3).mean()
    return float(2 * rec * prec / max(rec + prec, 1e-9))


async def main() -> None:
    scores: dict = {}
    rows = []
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                hero = await comfy.upload_image(args.hero.resolve())
                for f in frames:
                    cam = campath.PathCamera.from_json(cams_json["frames"][f])
                    medge, mask = depth_edges(cam)
                    src = work / "attempt" / "frames" / f"{f:05d}.png"
                    render = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255
                    scores[f] = {"yaw": round(cam.yaw, 1), "pitch": cam.pitch, "render": round(edge_f1(render, medge, mask), 3)}
                    image = await comfy.upload_image(src)
                    tiles = [render]
                    for d in strengths:
                        done = None
                        async for ev in comfy.run(qwen_edit_workflow(image, PROMPT, args.seed, "giro/from_render", ref=hero, denoise=d)):
                            if isinstance(ev, Done):
                                done = ev
                        assert done is not None
                        dst = out / f"{f:05d}-{d}.png"
                        await comfy.download(done.outputs["save"]["images"][0], dst)
                        edit = np.asarray(Image.open(dst).convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255
                        scores[f][str(d)] = round(edge_f1(edit, medge, mask), 3)
                        tiles.append(edit)
                        if args.attempt_out:  # the edit inside the render's mask (Qwen redraws the background too)
                            sil = np.asarray(Image.open(work / "attempt" / "masks" / "frames" / f"{f:05d}.png")) > 127
                            Image.fromarray((edit * sil[..., None] * 255).round().astype(np.uint8)).save(dest / "frames" / f"{f:05d}.png")
                            Image.fromarray(sil.astype(np.uint8) * 255, "L").save(dest / "masks" / "frames" / f"{f:05d}.png")
                    print(f"frame {f} (yaw {cam.yaw:.0f}, pitch {cam.pitch:.0f}): edge F1 {scores[f]}", flush=True)
                    if not args.attempt_out:
                        over = [np.where(medge[..., None], [1.0, 0.15, 0.15], t) for t in tiles]
                        rows.append((f, np.concatenate(tiles, 1), np.concatenate(over, 1)))
            finally:
                await comfy.free()
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    if args.attempt_out:
        return
    labels = ["render"] + [f"denoise {d}" for d in strengths]
    tw = W // 2
    sheet = Image.new("RGB", (tw * len(labels), H // 2 * 2 * len(rows)))
    for r, (f, clean, over) in enumerate(rows):
        for k, img in enumerate((clean, over)):
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((tw * len(labels), H // 2))
            draw = ImageDraw.Draw(im)
            for j, lab in enumerate(labels):
                key = "render" if j == 0 else str(strengths[j - 1])
                draw.text((j * tw + 6, 6), f"frame {f} {lab}  F1 {scores[f][key]}" if k == 0 else "mesh depth edges in red",
                          fill=(255, 255, 0))
            sheet.paste(im, (0, (2 * r + k) * (H // 2)))
    sheet.save(out / "sheet.jpg", quality=88)

asyncio.run(main())
