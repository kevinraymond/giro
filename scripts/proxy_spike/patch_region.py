"""Repaint one region of a projection texture with an image edit, after all the anchors are in:
render the textured mesh from a view, find the region with SAM (a text prompt, e.g. "license
plate"), move in close on it, edit that close render with Qwen-Image-Edit 2511 (Apache-2.0,
Lightning 4 steps), and paint the edited pixels back onto the mesh from that one camera, inside
the region only. One edited view defines the region, so there are no drifting versions of it
for the blend to average (the anchors each invent their own plate text).

    patch_region.py ATTEMPT WORK GPU --texture IN.pt --save OUT.pt --find "license plate" \\
        --prompt "..." [--yaw 180 --pitch 10]

ATTEMPT gives the hero (for the camera frame); WORK/texture.json the hero's fit; IN.pt is
project_texture.py --save-texture's output. Writes OUT.pt for project_texture.py --texture, and
WORK/patch-<slug>/ (renders, masks, the edit, before/after).
"""
import argparse
import asyncio
import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient, Done
from giro.stages.masks import Masks, sam_workflow
from texture_common import project, render_points, sample, samples, zbuffer

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--texture", type=Path, required=True)
ap.add_argument("--save", type=Path, required=True)
ap.add_argument("--find", required=True, help="SAM text prompt for the region")
ap.add_argument("--prompt", required=True, help="the edit")
ap.add_argument("--yaw", type=float, default=180.0, help="look-from direction, degrees from the hero's")
ap.add_argument("--pitch", type=float, default=10.0)
ap.add_argument("--fill", type=float, default=0.35, help="the region's share of the close view's height")
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
work = args.work.resolve()
dev = torch.device(f"cuda:{args.gpu}")
out = work / f"patch-{re.sub(r'[^a-z0-9]+', '-', args.find.lower()).strip('-')}"
out.mkdir(exist_ok=True)

saved = torch.load(args.texture)
xyz, _, _, _ = samples(work, saved["points"], dev)
rgb = saved["rgb"].to(dev).float()
hero = campath.PathCamera.from_json(json.loads((work / "texture.json").read_text())["fits"]["hero"]["camera"])
S = args.size


def save_render(cam: campath.PathCamera, name: str) -> tuple[Path, np.ndarray]:
    img, mask = render_points(xyz, rgb, cam, S, S)
    path = out / f"{name}.png"
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)
    return path, mask


async def find(comfy: ComfyClient, path: Path) -> np.ndarray:
    params = Masks.defaults | {"subject_prompt": args.find, "background_prompt": ""}
    async for _ in comfy.run(sam_workflow({"region": [path]}, out / "raw", params)):
        pass
    m = out / "raw" / "subject" / "region" / path.name
    return np.asarray(Image.open(m)) > 127 if m.exists() else np.zeros((S, S), bool)


def edit_workflow(image: str) -> dict:
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "cfgnorm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "lightning": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["cfgnorm", 0], "strength_model": 1.0,
                      "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}},
        "scaled": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["image", 0]}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scaled", 0], "vae": ["vae", 0]}},
        "pos": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scaled", 0], "prompt": args.prompt}},
        "posr": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["pos", 0], "reference_latents_method": "index_timestep_zero"}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scaled", 0], "prompt": ""}},
        "negr": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg", 0], "reference_latents_method": "index_timestep_zero"}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["lightning", 0], "positive": ["posr", 0], "negative": ["negr", 0],
                   "latent_image": ["latent", 0], "seed": args.seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
                   "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "giro/patch"}},
    }


def surface_point(cam: campath.PathCamera, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """The region's 3D center (the samples landing in it, nearest layer) and its height in the world."""
    u, v, z = project(cam, S, S, xyz)
    ok, pix, zmin, _ = zbuffer(u, v, z, S, S)
    m = torch.from_numpy(mask.reshape(-1)).to(dev)
    hit = ok & (z <= zmin + 0.005) & m[pix]
    pts = xyz[hit].cpu().numpy()
    if len(pts) < 50:
        raise SystemExit(f"SAM found no '{args.find}' on the surface from this view")
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    return pts.mean(0), float(np.linalg.norm(hi - lo))


async def main() -> None:
    global rgb
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                # 1. Find the region from a full view, then move in close on it.
                wide = replace(hero, yaw=hero.yaw + args.yaw, pitch=args.pitch)
                wide_path, _ = save_render(wide, "wide")
                m_wide = await find(comfy, wide_path)
                center, extent = surface_point(wide, m_wide)
                fov = math.radians(wide.fov)
                dist = extent / args.fill / (2 * math.tan(fov / 2))
                close = replace(wide, target=tuple(center.tolist()), distance=max(dist, 0.15))
                close_path, _ = save_render(close, "close")
                mask = await find(comfy, close_path)
                if mask.sum() < 100:
                    raise SystemExit(f"SAM found no '{args.find}' in the close view")
                # 2. Edit the close render.
                image = await comfy.upload_image(close_path)
                done = None
                async for ev in comfy.run(edit_workflow(image)):
                    if isinstance(ev, Done):
                        done = ev
                assert done is not None
                await comfy.download(done.outputs["save"]["images"][0], out / "edited_raw.png")
            finally:
                await comfy.free()

    before = Image.open(close_path).convert("RGB")
    edited = Image.open(out / "edited_raw.png").convert("RGB").resize((S, S), Image.LANCZOS)
    # 3. Composite inside the region (grown a little, feathered), paint it back from the close camera.
    grown = ndimage.binary_dilation(mask, iterations=6)
    alpha = np.asarray(Image.fromarray(grown.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(3)), dtype=np.float32) / 255
    comp = np.asarray(before, np.float32) * (1 - alpha[..., None]) + np.asarray(edited, np.float32) * alpha[..., None]
    Image.fromarray(comp.astype(np.uint8)).save(out / "edited.png")
    Image.fromarray((grown * 255).astype(np.uint8)).save(out / "mask.png")
    u, v, z = project(close, S, S, xyz)
    ok, _, zmin, _ = zbuffer(u, v, z, S, S)
    vis = ok & (z <= zmin + 0.005)
    a = sample(torch.from_numpy(alpha).to(dev)[None], u, v, S, S)[:, 0] * vis
    new = sample(torch.from_numpy(comp / 255).permute(2, 0, 1).to(dev), u, v, S, S)
    rgb = rgb * (1 - a[:, None]) + new * a[:, None]
    winner = saved["winner"].to(dev)
    winner = torch.where(a > 0.5, torch.full_like(winner, len(saved.get("patches", [])) + 1000), winner)
    patches = saved.get("patches", []) + [{"find": args.find, "prompt": args.prompt, "camera": close.to_json(),
                                          "samples": int((a > 0.5).sum())}]
    torch.save({"rgb": rgb.half().cpu(), "winner": winner.cpu(), "points": saved["points"], "patches": patches}, args.save)
    after, _ = render_points(xyz, rgb, close, S, S)
    sheet = Image.new("RGB", (3 * S, S))
    for k, im in enumerate([before, edited, Image.fromarray((np.clip(after, 0, 1) * 255).astype(np.uint8))]):
        sheet.paste(im, (k * S, 0))
    sheet.resize((3 * S // 2, S // 2)).save(out / "before-edit-after.jpg", quality=90)
    print(f"patched {int((a > 0.5).sum()):,} samples -> {args.save}; {out / 'before-edit-after.jpg'}", flush=True)

asyncio.run(main())
