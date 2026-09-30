"""Generate test subjects for the pipeline with Qwen-Image 2512 in giro's ComfyUI.

    uv run scripts/make_subjects.py [--gpu 1] [--seed 1] [NAME ...]

Writes data/samples/<name>.png at 3:4 (the orbit video's aspect ratio). The
prompts ask for what orbits well: one subject, whole and centered, standing
free in an empty room with space on every side.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from giro.comfy import ComfyClient, Done, server

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"

LIGHTNING = "Qwen-Image-2512-Lightning-8steps-V1.0-fp32.safetensors"

SETTING = (
    "Full-body photograph, the whole subject in frame with space above and below, centered, "
    "standing free in the middle of an empty room with a plain light wall and a wooden floor, "
    "soft even studio lighting, eye-level camera, sharp focus, realistic."
)
SUBJECTS = {
    "raincoat": "A young woman in a bright yellow hooded raincoat, dark blue jeans and red rubber boots, "
                "standing still with her arms at her sides, looking at the camera.",
    "scooter": "A vintage mint-green motor scooter with chrome details and a tan leather seat, "
               "parked on its center stand, seen from a three-quarter front angle.",
    "knight": "A knight in full polished steel plate armor with a red surcoat and a closed helmet, "
              "standing still, holding a sword pointed down in front of him with both hands.",
}


def workflow(prompt: str, seed: int, width: int, height: int, prefix: str) -> dict:
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name": "qwen_image_2512_fp8_e4m3fn.safetensors", "weight_dtype": "default"}},
        # The pre-merged *_scaled_8steps checkpoint loads as FLUX in this Comfy and gives noise.
        "lightning": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["unet", 0], "lora_name": LIGHTNING, "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["lightning", 0], "shift": 3.0}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["shift", 0], "positive": ["pos", 0], "negative": ["neg", 0], "latent_image": ["latent", 0],
            "seed": seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": prefix}},
    }


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", default=list(SUBJECTS))
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--width", type=int, default=1104)   # Qwen-Image's native 3:4 size
    p.add_argument("--height", type=int, default=1472)
    args = p.parse_args()
    was_up = server.is_up(args.gpu)
    url = await asyncio.to_thread(server.start, args.gpu)
    try:
        async with ComfyClient(url) as comfy:
            for name in args.names:
                prompt = f"{SUBJECTS[name]} {SETTING}"
                async for event in comfy.run(workflow(prompt, args.seed, args.width, args.height, f"giro/subjects/{name}")):
                    if isinstance(event, Done):
                        image = next(img for out in event.outputs.values() for img in out.get("images", []))
                        dest = SAMPLES / f"{name}.png"
                        await comfy.download(image, dest)
                        print(f"{name}: {dest}")
            await comfy.free()
    finally:
        if not was_up:
            server.stop(args.gpu)


if __name__ == "__main__":
    asyncio.run(main())
