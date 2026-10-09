"""Proxy orbit experiment: super-resolve an attempt's frames with SeedVR2 (Apache-2.0, native in
giro's ComfyUI) into a copy of the attempt, to train on them and compare with the Wan refine pass.

    seedvr2_frames.py SRC_ATTEMPT DST_ATTEMPT GPU SCALE [UNET] [COLOR]

UNET defaults to seedvr2/seedvr2_7b_fp8_e4m3fn.safetensors (Comfy-Org/SeedVR2), COLOR to lab.
Then: uv run giro stages DST_ATTEMPT --gpu GPU   (masks, poses, training at the new size)
"""
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

from giro.comfy import server
from giro.comfy.client import ComfyClient, Done, Progress

src, dst = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
GPU, SCALE = int(sys.argv[3]), float(sys.argv[4])
UNET = sys.argv[5] if len(sys.argv) > 5 else "seedvr2/seedvr2_7b_fp8_e4m3fn.safetensors"
COLOR = sys.argv[6] if len(sys.argv) > 6 else "lab"

frames = sorted((src / "frames_raw").glob("*.png"))
w0, h0 = Image.open(frames[0]).size
W, H = round(w0 * SCALE / 16) * 16, round(h0 * SCALE / 16) * 16

# The copy: everything the stages after the video read, without what they would redo.
dst.mkdir(parents=True, exist_ok=True)
for d in ("hero", "proxy", "proxy_render"):
    if (src / d).exists() and not (dst / d).exists():
        shutil.copytree(src / d, dst / d)
(dst / ".stages").mkdir(exist_ok=True)
for f in ("orbit_video.json", "proxy.json"):
    shutil.copy(src / ".stages" / f, dst / ".stages" / f)
for f in ("orbit.json", "video.mp4"):
    shutil.copy(src / f, dst / f)
cams = json.loads((src / "cameras.json").read_text())
(dst / "cameras.json").write_text(json.dumps(cams | {"width": W, "height": H, "seedvr2": {"unet": UNET, "scale": SCALE}}, indent=1) + "\n")

wf = {
    "frames": {"class_type": "GiroLoadImages", "inputs": {"paths": json.dumps([str(f) for f in frames]), "width": W, "height": H}},
    "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "seedvr2_ema_vae_fp16.safetensors"}},
    "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
    "pre": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["frames", 0]}},
    "encode": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["pre", 0], "vae": ["vae", 0], "tile_size": 512, "overlap": 128,
                                                          "temporal_size": 64, "temporal_overlap": 8}},
    "chunk": {"class_type": "SeedVR2TemporalChunk", "inputs": {"latent": ["encode", 0], "temporal_overlap": 0, "chunking_mode": "auto"}},
    "cond": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["unet", 0], "vae_conditioning": ["chunk", 0]}},
    "sample": {"class_type": "KSampler", "inputs": {"model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1], "latent_image": ["chunk", 0],
                                                   "seed": 1, "steps": 1, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
    "merge": {"class_type": "SeedVR2TemporalMerge", "inputs": {"latents": ["sample", 0], "temporal_overlap": ["chunk", 1]}},
    "decode": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["merge", 0], "vae": ["vae", 0], "tile_size": 512, "overlap": 128,
                                                          "temporal_size": 64, "temporal_overlap": 8}},
    "post": {"class_type": "SeedVR2PostProcessing", "inputs": {"images": ["decode", 0], "original_resized_images": ["frames", 0],
                                                              "color_correction_method": COLOR}},
    "save": {"class_type": "SaveImage", "inputs": {"images": ["post", 0], "filename_prefix": f"giro/seedvr2-{time.strftime('%H%M%S')}"}},
}


async def main() -> None:
    out = dst / "frames_raw"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    t0 = time.monotonic()
    with await asyncio.to_thread(server.Lease, GPU) as lease:
        async with ComfyClient(lease.url) as comfy:
            done = None
            try:
                async for ev in comfy.run(wf):
                    if isinstance(ev, Done):
                        done = ev
                    elif isinstance(ev, Progress):
                        print(f"  {ev.value}/{ev.max} at {time.monotonic() - t0:.0f}s", flush=True)
                images = sorted(done.outputs["save"]["images"], key=lambda x: x["filename"])
                for i, img in enumerate(images):
                    await comfy.download(img, out / f"{i:05d}.png")
            finally:
                await comfy.free()
    print(f"{len(images)} frames at {W}x{H} in {time.monotonic() - t0:.0f}s", flush=True)
    if len(images) != len(frames):
        raise SystemExit(f"SeedVR2 returned {len(images)} frames for {len(frames)}")


asyncio.run(main())
