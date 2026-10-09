"""A/B of dx8152's Qwen-Image-Edit-2511-Gaussian-Splash LoRA (Apache-2.0 through its base; trained on
pairs made with Apple's research-only Sharp) against the route's own render cleanup (anchor_from_render.py,
#3685). The LoRA was trained for exactly this input: picture 1 a view rendered from a 3D model of the
photo, with holes, picture 2 the photo; it fixes picture 1's perspective and fills the blanks.

    splash_test.py ATTEMPT WORK GPU --frames 42,96,138 [--texture WORK/texture-0.pt] [--out DIR]

For each frame of WORK/attempt (its renders, exact poses), makes:
  A    the route's recipe: 2511 + Lightning 4 steps, hero as picture 2, our prompt, denoise 0.6
  B1   + the Splash LoRA, its own prompt and settings (10 steps, full denoise)
  B06  + the Splash LoRA, its prompt, denoise 0.6
  C    the LoRA's training input: the render with the surface no view painted left as black holes
       (from --texture's winner < 0), Splash LoRA, full denoise (only with --texture)
Writes OUT/<frame>-<variant>.png, OUT/sheet.jpg (rows per frame: inputs and edits, then the same with
the mesh's depth edges in red), OUT/anaglyph-<frame>.jpg (edit red, render cyan) and OUT/scores.json
(edge F1 against the mesh's depth edges; the render's own is the baseline).
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
from texture_common import fingerprint, project, qwen_edit_workflow, render_points, samples, zbuffer

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path, help="the source attempt (hero/hero.png)")
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--frames", default="42,96,138")
ap.add_argument("--texture", type=Path, help="a painted texture with winner (project_texture.py --save-texture), for C")
ap.add_argument("--subject", default="object")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
work, out = args.work.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
dev = torch.device(f"cuda:{args.gpu}")
cams_json = json.loads((work / "attempt" / "cameras.json").read_text())
W, H = cams_json["width"], cams_json["height"]
frames = [int(f) for f in args.frames.split(",")]
LORA = "qwen/qwen-image-edit-2511-gaussian-splash.safetensors"
SPLASH_PROMPT = "高斯泼溅,参考图2的场景图，修复图1的场景图透视并修复空白区域"  # the LoRA's default, as trained
OUR_PROMPT = (f"Picture 1 is a rough render of a {args.subject}; picture 2 is a photo of the same {args.subject}. "
              f"Make picture 1 a clean, sharp, realistic photo of the {args.subject}, exactly this view and framing: "
              f"keep every shape, edge and part where it is, and turn the smeared, patchy, noisy surfaces into the clean, "
              f"coherent detail and paint that belong there, lit evenly like picture 2. Black background.")
VARIANTS = {"A": dict(prompt=OUR_PROMPT, denoise=0.6),
            "B1": dict(prompt=SPLASH_PROMPT, denoise=1.0, lora=LORA, steps=10),
            "B06": dict(prompt=SPLASH_PROMPT, denoise=0.6, lora=LORA, steps=10)}
if args.texture:
    VARIANTS["C"] = dict(prompt=SPLASH_PROMPT, denoise=1.0, lora=LORA, steps=10, holes=True)

if args.texture:
    saved = torch.load(args.texture)
    xyz, nrm, _, _ = samples(work, saved["points"], dev)
    assert abs(saved["fingerprint"] - fingerprint(xyz)) < 1e-3, f"{args.texture} was painted on other samples"
    rgb, painted = saved["rgb"].to(dev).float(), (saved["winner"].to(dev) >= 0).float()
    print(f"{args.texture.name}: {painted.mean():.1%} of the surface painted", flush=True)
else:
    xyz, nrm, _, _ = samples(work, 3_000_000, dev)


def depth_edges(cam: campath.PathCamera) -> tuple[np.ndarray, np.ndarray]:
    u, v, z = project(cam, W, H, xyz)
    d = zbuffer(u, v, z, W, H)[3].cpu().numpy()
    mask = np.isfinite(d)
    d = np.where(mask, d, np.nanmax(np.where(mask, d, np.nan)) + 0.5)
    return (np.hypot(ndimage.sobel(d, 0), ndimage.sobel(d, 1)) > 0.04) & ndimage.binary_dilation(mask, iterations=2), mask


def edge_f1(img: np.ndarray, medge: np.ndarray, mask: np.ndarray) -> float:
    g = img.mean(-1)
    gi = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    iedge = (gi > np.percentile(gi[mask], 90)) & mask
    rec = (ndimage.distance_transform_edt(~medge)[iedge] <= 3).mean()
    prec = (ndimage.distance_transform_edt(~iedge)[medge] <= 3).mean()
    return float(2 * rec * prec / max(rec + prec, 1e-9))


def holes_render(cam: campath.PathCamera) -> np.ndarray:
    """The texture's render with what no view painted left black."""
    vals, mask = render_points(xyz, torch.cat([rgb * painted[:, None], painted[:, None]], 1), cam, W, H, nrm=nrm)
    p = vals[..., 3]
    img = np.where((p > 0.5)[..., None], vals[..., :3] / np.clip(p, 1e-6, None)[..., None], 0.0)
    return np.clip(img, 0, 1) * mask[..., None]


async def main() -> None:
    scores: dict = {}
    rows = []
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                hero = await comfy.upload_image(args.attempt.resolve() / "hero" / "hero.png")
                for f in frames:
                    cam = campath.PathCamera.from_json(cams_json["frames"][f])
                    medge, mask = depth_edges(cam)
                    src = work / "attempt" / "frames" / f"{f:05d}.png"
                    render = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255
                    inputs = {"render": (src, render)}
                    if args.texture:
                        holes = holes_render(cam)
                        hp = out / f"{f:05d}-holes.png"
                        Image.fromarray((holes * 255).round().astype(np.uint8)).save(hp)
                        inputs["holes"] = (hp, holes)
                    uploaded = {k: await comfy.upload_image(p) for k, (p, _) in inputs.items()}
                    scores[f] = {"yaw": round(cam.yaw, 1), "pitch": cam.pitch, "render": round(edge_f1(render, medge, mask), 3)}
                    tiles, labels = [im for _, im in inputs.values()], list(inputs)
                    for name, v in VARIANTS.items():
                        wf = qwen_edit_workflow(uploaded["holes" if v.get("holes") else "render"], v["prompt"], args.seed,
                                                "giro/splash", ref=hero, denoise=v["denoise"], lora=v.get("lora"),
                                                steps=v.get("steps", 4))
                        done = None
                        async for ev in comfy.run(wf):
                            if isinstance(ev, Done):
                                done = ev
                        assert done is not None
                        dst = out / f"{f:05d}-{name}.png"
                        await comfy.download(done.outputs["save"]["images"][0], dst)
                        edit = np.asarray(Image.open(dst).convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255
                        scores[f][name] = round(edge_f1(edit, medge, mask), 3)
                        tiles.append(edit)
                        labels.append(name)
                        ana = np.stack([edit.mean(-1), render.mean(-1), render.mean(-1)], -1)
                        Image.fromarray((ana * 255).astype(np.uint8)).save(out / f"anaglyph-{f:05d}-{name}.jpg", quality=85)
                    print(f"frame {f} (yaw {cam.yaw:.0f}, pitch {cam.pitch:.0f}): edge F1 {scores[f]}", flush=True)
                    over = [np.where(medge[..., None], [1.0, 0.15, 0.15], t) for t in tiles]
                    rows.append((f, labels, np.concatenate(tiles, 1), np.concatenate(over, 1)))
            finally:
                await comfy.free()
    (out / "scores.json").write_text(json.dumps(scores, indent=1))
    n = len(rows[0][1])
    tw, th = W // 2, H // 2
    sheet = Image.new("RGB", (tw * n, th * 2 * len(rows)))
    for r, (f, labels, clean, over) in enumerate(rows):
        for k, img in enumerate((clean, over)):
            im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((tw * n, th))
            draw = ImageDraw.Draw(im)
            for j, lab in enumerate(labels):
                f1 = scores[f].get(lab, scores[f]["render"])
                draw.text((j * tw + 6, 6), f"{f} {lab} F1 {f1}" if k == 0 else "mesh depth edges", fill=(255, 255, 0))
            sheet.paste(im, (0, (2 * r + k) * th))
    sheet.save(out / "sheet.jpg", quality=88)

asyncio.run(main())
