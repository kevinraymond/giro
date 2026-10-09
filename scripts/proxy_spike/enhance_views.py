"""Enhance a projection-textured attempt's renders with SeedVR2 (the pipeline's upscale workflow),
ring by ring, so its temporal window sees each ring as the short orbit video it is.

    enhance_views.py BASE_ATTEMPT OUT_ATTEMPT GPU [SCALE=2]

OUT_ATTEMPT gets frames2x/ (SeedVR2's output, SCALE times the size), frames/ (those brought back
to the base size, which training and project_texture.py --views read) and the base's masks,
hero, poses and cameras (copied or linked), so it trains as it is (`giro stages --from dataset`)
and can be painted back onto the mesh (project_texture.py --views OUT_ATTEMPT).
"""
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

from giro import workflows
from giro.comfy import ComfyClient, Done, server
from giro.stages.orbit import output_images

base, out, gpu = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
scale = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
cams = json.loads((base / "cameras.json").read_text())
w, h = cams["width"], cams["height"]
size = (round(w * scale / 16) * 16, round(h * scale / 16) * 16)
rings: list[list[int]] = []
for i, c in enumerate(cams["frames"]):
    if rings and abs(cams["frames"][rings[-1][-1]]["pitch"] - c["pitch"]) < 1e-6:
        rings[-1].append(i)
    else:
        rings.append([i])

out.mkdir(parents=True)
for name in ("masks", "hero", "poses"):
    (out / name).symlink_to(base / name)
for name in ("cameras.json", "metrics.json"):
    shutil.copy(base / name, out / name)
(out / ".stages").mkdir()
shutil.copy(base / ".stages" / "orbit_video.json", out / ".stages" / "orbit_video.json")
(out / "frames2x").mkdir()
(out / "frames").mkdir()


async def main() -> None:
    stamp = f"giro/{time.strftime('%Y%m%d-%H%M%S')}-enhance"
    with await asyncio.to_thread(server.Lease, gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                for k, ring in enumerate(rings):
                    names = [f"{i:05d}.png" for i in ring]
                    prompt = workflows.build("upscale_seedvr2", frames=json.dumps([str(base / "frames" / n) for n in names]),
                                             width=size[0], height=size[1], unet="seedvr2/seedvr2_7b_fp8_e4m3fn.safetensors",
                                             color="lab", seed=1, output_prefix=f"{stamp}/ring{k}")
                    t0 = time.monotonic()
                    done = None
                    async for event in comfy.run(prompt):
                        if isinstance(event, Done):
                            done = event
                    assert done is not None
                    images = output_images(done, "save")
                    assert len(images) == len(names), f"SeedVR2 returned {len(images)} frames for {len(names)}"
                    for n, img in zip(names, images):
                        await comfy.download(img, out / "frames2x" / n)
                        Image.open(out / "frames2x" / n).convert("RGB").resize((w, h), Image.LANCZOS).save(out / "frames" / n)
                    print(f"ring {k + 1}/{len(rings)}: {len(names)} frames in {time.monotonic() - t0:.0f} s", flush=True)
            finally:
                await comfy.free()

asyncio.run(main())
print(out)
