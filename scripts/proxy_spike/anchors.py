"""Anchor views for a proxy orbit: the proxy rendered at new angles, redrawn by Qwen-Image-Edit 2511
as photos of the hero's subject (image 1 = the proxy's render at that angle, the image edited;
image 2 = the hero, the reference; with the hero as image 1 the model mostly redrew the hero's view).
The shape comes from the proxy, the look from the hero, the camera is known. A first look, no video.

    anchors.py ATTEMPT OUT GPU [STEPS]

ATTEMPT has proxy/proxy.ply, proxy/proxy.json and hero/hero.png (a proxy orbit attempt). Writes
OUT/<view>-<variant>.png, the proxy renders, and OUT/anchors.jpg (hero | render | anchor per row).
STEPS 40 (default) runs at cfg 4; 4 uses the Lightning LoRA at cfg 1.
"""
import asyncio
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient, Done

attempt, out, gpu = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
steps = int(sys.argv[4]) if len(sys.argv) > 4 else 40
out.mkdir(parents=True, exist_ok=True)
hero_cam = campath.PathCamera.from_json(json.loads((attempt / "proxy" / "proxy.json").read_text())["hero_camera"])
hero_img = Image.open(attempt / "hero" / "hero.png")
W, H = (768, 1024) if hero_img.height >= hero_img.width else (1024, 768)
VIEWS = {  # offsets from the hero's camera
    "side-right": replace(hero_cam, yaw=hero_cam.yaw + 90),
    "back": replace(hero_cam, yaw=hero_cam.yaw + 180),
    "side-left": replace(hero_cam, yaw=hero_cam.yaw + 270),
    "above": replace(hero_cam, yaw=hero_cam.yaw + 45, pitch=50.0),
}
PROMPT = {
    "color": ("Image 1 is a rough 3D model of the object in image 2. Turn image 1 into a photograph of the exact object "
              "in image 2: keep the camera angle, the outline and the proportions of image 1 exactly, and give it the "
              "colors, materials, markings and fine details of the object in image 2, as they would look from this "
              "angle. Plain light gray studio background, soft even light, sharp focus."),
    "depth": ("Image 1 is a depth map of the object in image 2 (near is bright). Turn image 1 into a photograph of the "
              "exact object in image 2 seen from this angle: its outline and shape must match image 1 exactly; its "
              "colors, materials, markings and fine details must match image 2. Plain light gray studio background, "
              "soft even light, sharp focus."),
}
NEG = "blurry, deformed, extra parts, cartoon, 3d render, text, watermark"


def workflow(hero: str, renders: dict[str, str], stamp: str) -> dict:
    wf = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_int8_convrot.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "cfgnorm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "hero": {"class_type": "LoadImage", "inputs": {"image": hero, "upload": "image"}},
        "hero_scaled": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["hero", 0]}},
    }
    model = ["cfgnorm", 0]
    if steps <= 8:
        wf["lightning"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["cfgnorm", 0], "strength_model": 1.0,
                           "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"}}
        model = ["lightning", 0]
    for key, name in renders.items():
        variant = key.split("-")[-1]
        wf |= {
            f"r_{key}": {"class_type": "LoadImage", "inputs": {"image": name, "upload": "image"}},
            f"rs_{key}": {"class_type": "FluxKontextImageScale", "inputs": {"image": [f"r_{key}", 0]}},
            f"pos_{key}": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0],
                           "image1": [f"rs_{key}", 0], "image2": ["hero_scaled", 0], "prompt": PROMPT[variant]}},
            f"neg_{key}": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "vae": ["vae", 0],
                           "image1": [f"rs_{key}", 0], "image2": ["hero_scaled", 0], "prompt": NEG}},
            f"posr_{key}": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": [f"pos_{key}", 0],
                            "reference_latents_method": "index_timestep_zero"}},
            f"negr_{key}": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": [f"neg_{key}", 0],
                            "reference_latents_method": "index_timestep_zero"}},
            f"lat_{key}": {"class_type": "VAEEncode", "inputs": {"pixels": [f"rs_{key}", 0], "vae": ["vae", 0]}},
            f"s_{key}": {"class_type": "KSampler", "inputs": {"model": model, "positive": [f"posr_{key}", 0], "negative": [f"negr_{key}", 0],
                         "latent_image": [f"lat_{key}", 0], "seed": 1, "steps": steps, "cfg": 1.0 if steps <= 8 else 4.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            f"d_{key}": {"class_type": "VAEDecode", "inputs": {"samples": [f"s_{key}", 0], "vae": ["vae", 0]}},
            f"save_{key}": {"class_type": "SaveImage", "inputs": {"images": [f"d_{key}", 0], "filename_prefix": f"giro/anchor-{stamp}-{key}"}},
        }
    return wf


async def main() -> None:
    stamp = time.strftime("%H%M%S")
    render_wf = {"proxy": {"class_type": "GiroLoadSplat", "inputs": {"path": str(attempt / "proxy" / "proxy.ply")}}}
    for style in ("color", "depth"):
        render_wf[f"r_{style}"] = {"class_type": "GiroRenderSplatCameras", "inputs": {
            "splat": ["proxy", 0], "width": W, "height": H, "render_style": style, "background": "#b0b0b0" if style == "color" else "#000000",
            "cameras": json.dumps([c.render_json() for c in VIEWS.values()])}}
        render_wf[f"save_{style}"] = {"class_type": "SaveImage", "inputs": {"images": [f"r_{style}", 0], "filename_prefix": f"giro/anchor-{stamp}-proxy-{style}"}}
    with await asyncio.to_thread(server.Lease, gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                done = None
                async for ev in comfy.run(render_wf):
                    if isinstance(ev, Done):
                        done = ev
                renders = {}
                for style in ("color", "depth"):
                    imgs = sorted(done.outputs[f"save_{style}"]["images"], key=lambda x: x["filename"])
                    for view, img in zip(VIEWS, imgs):
                        dest = out / f"proxy-{view}-{style}.png"
                        await comfy.download(img, dest)
                        renders[f"{view}-{style}"] = await comfy.upload_image(dest)
                hero = await comfy.upload_image(attempt / "hero" / "hero.png")
                t0 = time.monotonic()
                done = None
                async for ev in comfy.run(workflow(hero, renders, stamp)):
                    if isinstance(ev, Done):
                        done = ev
                print(f"{len(renders)} anchors in {time.monotonic() - t0:.0f}s", flush=True)
                for key in renders:
                    img = done.outputs[f"save_{key}"]["images"][0]
                    await comfy.download(img, out / f"{key}.png")
            finally:
                await comfy.free()
    # review sheet: one row per view: hero | proxy color | anchor (color) | proxy depth | anchor (depth)
    h = 384
    rows = []
    for view in VIEWS:
        tiles = [hero_img] + [Image.open(out / n) for n in (f"proxy-{view}-color.png", f"{view}-color.png",
                                                           f"proxy-{view}-depth.png", f"{view}-depth.png")]
        tiles = [t.convert("RGB").resize((round(t.width * h / t.height), h)) for t in tiles]
        row = Image.new("RGB", (sum(t.width for t in tiles), h + 22), (40, 40, 40))
        x = 0
        for t in tiles:
            row.paste(t, (x, 22))
            x += t.width
        ImageDraw.Draw(row).text((6, 4), f"{view}: hero | proxy color | anchor from color | proxy depth | anchor from depth",
                                 fill=(255, 255, 0))
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out / "anchors.jpg", quality=90)


asyncio.run(main())
