"""A video model's pass over a projection-texture attempt's renders: each ring of cameras is one
closed loop (its first frame again at the end), in chained clips of at most --max-frames (4k+1
frames, as Wan wants; each clip starts from the frame the one before refined last), re-denoised by Wan 2.2 Fun Control's low-noise expert from SIGMA under the mesh's depth
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
ap.add_argument("--max-frames", type=int, default=49, help="longest clip; longer rings are chained clips")
args = ap.parse_args()
MAX_STEPS = args.max_frames - 1
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
    done_names: set[str] = set()
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                hero = await comfy.upload_image(source / "hero" / "hero.png")
                for k in todo:
                    loop = rings[k] + rings[k][:1]  # closed loop: the first frame again
                    # Clips of at most MAX_STEPS steps, a multiple of 4 (4k+1 frames, as Wan wants),
                    # chained: each starts from the frame the one before refined last.
                    steps = len(loop) - 1
                    n_clips = next(c for c in range(1, steps + 1) if steps % c == 0 and steps // c <= MAX_STEPS and (steps // c) % 4 == 0)
                    per = steps // n_clips
                    t0 = time.monotonic()
                    for c in range(n_clips):
                        clip = loop[c * per:(c + 1) * per + 1]
                        names = [f"{i:05d}.png" for i in clip]
                        first = out / "frames" / names[0] if c else base / "frames" / names[0]
                        start = await comfy.upload_image(first)
                        prompt = workflows.build_refine(
                            W, H, args.sigma, args.steps,
                            frames=json.dumps([str(base / "frames" / n) for n in names]),
                            image=hero, start=start, prompt=workflows.PROXY_ORBIT_PROMPT, seed=args.seed + k,
                            proxy=str(points), cameras=json.dumps([cams[i].render_json() for i in clip]),
                            length=len(clip), output_prefix=f"{stamp}/ring{k}-{c}")
                        done = None
                        async for ev in comfy.run(prompt):
                            if isinstance(ev, Done):
                                done = ev
                        assert done is not None
                        images = output_images(done, "frames")
                        assert len(images) == len(clip), f"Wan returned {len(images)} frames for {len(clip)}"
                        # A clip's last frame is the next clip's first (or, last of all, the ring's first):
                        # kept only where nothing refined it yet.
                        for i, (n, img) in enumerate(zip(names, images)):
                            if i == len(names) - 1 and (c == n_clips - 1 or n in done_names):
                                continue
                            if i == 0 and c > 0:
                                continue
                            await comfy.download(img, out / "frames" / n)
                            done_names.add(n)
                            im = Image.open(out / "frames" / n).convert("RGB")
                            if im.size != (W, H):
                                im.resize((W, H), Image.LANCZOS).save(out / "frames" / n)
                    print(f"ring {k} (pitch {cams[loop[0]].pitch:.0f}, {steps} views, {n_clips} clip(s) of {per + 1} frames): "
                          f"{time.monotonic() - t0:.0f} s", flush=True)
            finally:
                await comfy.free()

asyncio.run(main())
print(out)
