"""TripoSplat on giro heroes: hero.png + SAM hero mask -> RGBA -> ComfyUI TripoSplat -> PLY."""
import asyncio, sys, time
from pathlib import Path
from PIL import Image
from giro.comfy import server
from giro.comfy.client import ComfyClient, Done

OUT = Path(sys.argv[1]); GPU = int(sys.argv[2]); N = int(sys.argv[3])
SUBJECTS = dict(a.split("=") for a in sys.argv[4:])

def workflow(image, n, seed=46):
    return {
        "hero": {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}},
        "mask": {"class_type": "InvertMask", "inputs": {"mask": ["hero", 1]}},
        "prep": {"class_type": "TripoSplatPreprocessImage", "inputs": {"image": ["hero", 0], "mask": ["mask", 0], "erode_radius": 1, "size": 1024}},
        "dino": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "dino_v3_vit_h.safetensors"}},
        "flux_vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "decoder": {"class_type": "VAELoader", "inputs": {"vae_name": "triposplat_vae_decoder_fp16.safetensors"}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "triposplat_fp16.safetensors", "weight_dtype": "default"}},
        "cond": {"class_type": "TripoSplatConditioning", "inputs": {"clip_vision": ["dino", 0], "vae": ["flux_vae", 0], "image": ["prep", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1], "latent_image": ["cond", 2],
                   "seed": seed, "steps": 20, "cfg": 3.0, "sampler_name": "dpmpp_2m", "scheduler": "simple", "denoise": 1.0}},
        "decode": {"class_type": "VAEDecodeTripoSplat", "inputs": {"samples": ["sample", 0], "vae": ["decoder", 0], "num_gaussians": n, "seed": seed}},
        "file": {"class_type": "SplatToFile3D", "inputs": {"splat": ["decode", 0], "format": "ply"}},
        "save": {"class_type": "SaveGLB", "inputs": {"mesh": ["file", 0], "filename_prefix": f"giro/tripo-{time.strftime('%H%M%S')}"}},
        "prep_out": {"class_type": "SaveImage", "inputs": {"images": ["prep", 0], "filename_prefix": "giro/tripo-prep"}},
    }

async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with await asyncio.to_thread(server.Lease, GPU) as lease:
        async with ComfyClient(lease.url) as comfy:
            for name, attempt in SUBJECTS.items():
                a = Path(attempt)
                rgb = Image.open(a / "hero" / "hero.png").convert("RGB")
                m = Image.open(a / "masks" / "hero" / "hero.png").convert("L").resize(rgb.size)
                rgba = rgb.copy(); rgba.putalpha(m)
                src = OUT / f"{name}_rgba.png"; rgba.save(src)
                t0 = time.monotonic(); done = None
                async for ev in comfy.run(workflow(await comfy.upload_image(src), N)):
                    if isinstance(ev, Done): done = ev
                dt = time.monotonic() - t0
                stats = await comfy.system_stats()
                for node, out in done.outputs.items():
                    for kind, items in out.items():
                        for it in items if isinstance(items, list) else []:
                            if isinstance(it, dict) and "filename" in it:
                                ext = Path(it["filename"]).suffix
                                dest = OUT / (f"{name}{ext}" if node == "save" else f"{name}_prep{ext}")
                                await comfy.download(it, dest)
                                print(name, node, kind, it["filename"], "->", dest.name, flush=True)
                print(f"{name}: {dt:.1f}s", flush=True)
            await comfy.free()
asyncio.run(main())
