"""Anchor views straight from the hero with fal's Multiple-Angles LoRA for Qwen-Image-Edit 2511
(Apache-2.0; trained on Gaussian-splat renders to move the camera around the subject on command).
Default: 8 azimuths at eye level and 4 elevated; "all": the LoRA's 8 azimuths x 4 elevations
(low-angle, eye-level, elevated, high-angle), 32 views. One review sheet. No proxy, no video.

    anchors_angles.py HERO_PNG OUT GPU [all]
"""
import asyncio
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw

from giro.comfy import server
from giro.comfy.client import ComfyClient, Done

hero_png, out, gpu = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
AZIMUTHS = ["front view", "front-right quarter view", "right side view", "back-right quarter view", "back view",
            "back-left quarter view", "left side view", "front-left quarter view"]
VIEWS = [(a, "eye-level shot") for a in AZIMUTHS] + [(a, "elevated shot") for a in AZIMUTHS[::2]]
if len(sys.argv) > 4 and sys.argv[4] == "all":
    VIEWS = [(a, e) for e in ("low-angle shot", "eye-level shot", "elevated shot", "high-angle shot") for a in AZIMUTHS]


def workflow(hero: str, stamp: str) -> dict:
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "cfgnorm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "lightning": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["cfgnorm", 0], "strength_model": 1.0,
                      "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"}},
        "angles": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["lightning", 0], "strength_model": 1.0,
                   "lora_name": "qwen/qwen-image-edit-2511-multiple-angles-lora.safetensors"}},
        "hero": {"class_type": "LoadImage", "inputs": {"image": hero, "upload": "image"}},
        "scaled": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["hero", 0]}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scaled", 0], "vae": ["vae", 0]}},
        "neg": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scaled", 0], "prompt": ""}},
        "negr": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["neg", 0], "reference_latents_method": "index_timestep_zero"}},
    }
    for i, (az, el) in enumerate(VIEWS):
        wf |= {
            f"pos{i}": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "image1": ["scaled", 0],
                        "prompt": f"<sks> {az} {el} medium shot"}},
            f"posr{i}": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": [f"pos{i}", 0],
                         "reference_latents_method": "index_timestep_zero"}},
            f"s{i}": {"class_type": "KSampler", "inputs": {"model": ["angles", 0], "positive": [f"posr{i}", 0], "negative": ["negr", 0],
                      "latent_image": ["latent", 0], "seed": 1, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            f"d{i}": {"class_type": "VAEDecode", "inputs": {"samples": [f"s{i}", 0], "vae": ["vae", 0]}},
            f"save{i}": {"class_type": "SaveImage", "inputs": {"images": [f"d{i}", 0], "filename_prefix": f"giro/angles-{stamp}-{i:02d}"}},
        }
    return wf


async def main() -> None:
    stamp = time.strftime("%H%M%S")
    with await asyncio.to_thread(server.Lease, gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                hero = await comfy.upload_image(hero_png)
                t0 = time.monotonic()
                done = None
                async for ev in comfy.run(workflow(hero, stamp)):
                    if isinstance(ev, Done):
                        done = ev
                print(f"{len(VIEWS)} views in {time.monotonic() - t0:.0f}s", flush=True)
                for i in range(len(VIEWS)):
                    await comfy.download(done.outputs[f"save{i}"]["images"][0], out / f"{i:02d}.png")
            finally:
                await comfy.free()
    h = 320
    tiles = [Image.open(hero_png).convert("RGB")] + [Image.open(out / f"{i:02d}.png").convert("RGB") for i in range(len(VIEWS))]
    labels = ["hero"] + [f"{az.replace(' view', '')}, {el.replace(' shot', '')}" for az, el in VIEWS]
    tiles = [t.resize((round(t.width * h / t.height), h)) for t in tiles]
    cols = 7
    w = max(t.width for t in tiles)
    sheet = Image.new("RGB", (w * cols, (h + 18) * ((len(tiles) + cols - 1) // cols)), (40, 40, 40))
    for i, (t, label) in enumerate(zip(tiles, labels)):
        x, y = (i % cols) * w, (i // cols) * (h + 18)
        sheet.paste(t, (x, y + 18))
        ImageDraw.Draw(sheet).text((x + 4, y + 3), label, fill=(255, 255, 0))
    sheet.save(out / "angles.jpg", quality=90)


asyncio.run(main())
