"""A video model's pass over a projection-texture attempt's renders: each ring of cameras is one
closed-loop clip (its first frame again at the end: 25, 49, 37 or 9 frames, all 4k+1 as Wan
wants), re-denoised by Wan 2.2 Fun Control's low-noise expert from SIGMA under the mesh's depth
(the pipeline's refine pass, workflows.build_refine), with the hero as the appearance reference.
The renders are one consistent object from every side; the video model's temporal attention is
meant to make their detail coherent from view to view, as a video orbit's is, while the depth
control and the low starting noise keep the geometry and the layout.

    video_pass.py SOURCE_ATTEMPT WORK BASE_NAME OUT_NAME GPU [--sigma 0.35] [--steps 8] [--rings 0,1]

BASE_NAME is a rendered attempt in WORK (frames/, cameras.json); WORK/candidates/proxy_<seed>.ply
is the mesh's points (the depth control). OUT_NAME is a trainable attempt with the new frames and
the base's masks, hero and poses.
"""
import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path

from PIL import Image

from giro import path as campath
from giro import workflows
from giro.comfy import ComfyClient, Done, server
from giro.stages.orbit import output_images

ap = argparse.ArgumentParser()
ap.add_argument("source", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("base")
ap.add_argument("out_name")
ap.add_argument("gpu", type=int)
ap.add_argument("--sigma", type=float, default=0.35)
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--rings", help="only these rings (indices), e.g. for a quick look; the rest keep the base frames")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
source, work = args.source.resolve(), args.work.resolve()
base, out = work / args.base, work / args.out_name
cams_json = json.loads((base / "cameras.json").read_text())
W, H = cams_json["width"], cams_json["height"]
cams = [campath.PathCamera.from_json(c) for c in cams_json["frames"]]
rings: list[list[int]] = []
for i, c in enumerate(cams):
    if rings and abs(cams[rings[-1][-1]].pitch - c.pitch) < 1e-6:
        rings[-1].append(i)
    else:
        rings.append([i])
todo = [int(x) for x in args.rings.split(",")] if args.rings else list(range(len(rings)))
seed = json.loads((source / "proxy" / "proxy.json").read_text())["seed"]
points = work / "candidates" / f"proxy_{seed}.ply"

if out.exists():
    shutil.rmtree(out)
out.mkdir()
for name in ("masks", "hero", "poses"):
    (out / name).symlink_to((base / name).resolve())
for name in ("cameras.json", "metrics.json"):
    shutil.copy(base / name, out / name)
(out / ".stages").mkdir()
shutil.copy(base / ".stages" / "orbit_video.json", out / ".stages" / "orbit_video.json")
(out / "frames").mkdir()
for f in (base / "frames").glob("*.png"):
    shutil.copy(f, out / "frames" / f.name)


async def main() -> None:
    stamp = f"giro/{time.strftime('%H%M%S')}-videopass"
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                hero = await comfy.upload_image(source / "hero" / "hero.png")
                for k in todo:
                    ring = rings[k] + rings[k][:1]  # closed loop: the first frame again, 4k+1 frames
                    names = [f"{i:05d}.png" for i in ring]
                    start = await comfy.upload_image(base / "frames" / names[0])
                    prompt = workflows.build_refine(
                        W, H, args.sigma, args.steps,
                        frames=json.dumps([str(base / "frames" / n) for n in names]),
                        image=hero, start=start, prompt=workflows.PROXY_ORBIT_PROMPT, seed=args.seed + k,
                        proxy=str(points), cameras=json.dumps([cams[i].render_json() for i in ring]),
                        length=len(ring), output_prefix=f"{stamp}/ring{k}")
                    t0 = time.monotonic()
                    done = None
                    async for ev in comfy.run(prompt):
                        if isinstance(ev, Done):
                            done = ev
                    assert done is not None
                    images = output_images(done, "frames")
                    assert len(images) == len(ring), f"Wan returned {len(images)} frames for {len(ring)}"
                    for n, img in zip(names[:-1], images[:-1]):  # the repeated first frame is dropped
                        await comfy.download(img, out / "frames" / n)
                        im = Image.open(out / "frames" / n).convert("RGB")
                        if im.size != (W, H):
                            im.resize((W, H), Image.LANCZOS).save(out / "frames" / n)
                    print(f"ring {k} (pitch {cams[ring[0]].pitch:.0f}, {len(ring)} frames): {time.monotonic() - t0:.0f} s", flush=True)
            finally:
                await comfy.free()

asyncio.run(main())
print(out)
