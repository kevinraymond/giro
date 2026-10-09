"""Route A' pilot (board #3727): do views generated together agree better than views generated one by one?
Qwen-Image-Edit 2511 is an MMDiT, so the panels of one image attend to each other; a 2x2 grid of four target
views in one pass is the zero-training version of joint multi-view generation.

    grid_pilot.py ANGLES WORK OUT GPU [--groups 10,11,12,13/18,19,20,21/11,12,19,20] [--panel 768x1024,384x512]

ANGLES is a lora_anchors.py output (input/NN.png = image 1 per view on black, hero_ref.png, views.json with exact
cameras, NN.png = the route's one-by-one anchors); WORK holds the mesh those cameras belong to. Per group of four
views and per panel size, image 1 is the four proxy renders as a 2x2 grid, image 2 the hero, with (a) the view
LoRA v1 + Lightning (as the route) and (b) plain 2511 + Lightning; the panels are cut back out.

Agreement: every output (one-by-one anchors, each grid's panels) is projected onto the mesh's samples at its exact
camera (texture_common.Source.paint), and for each pair of views in a group the mean absolute color difference
over the samples both see well (weight > 0.1, 0-1 scale); lower is better. Silhouette IoU per panel against image
1's coverage says whether the pose held. Also peak VRAM and seconds per call. Writes OUT/report.json and
OUT/sheet-<group>.jpg (rows: proxy renders, one-by-one, grid variants).
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from giro import path as campath
from texture_common import Source, samples, view_lora_workflow

ap = argparse.ArgumentParser()
ap.add_argument("angles", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--groups", default="10,11,12,13/18,19,20,21/11,12,19,20")
ap.add_argument("--panel", default="768x1024,384x512", help="panel sizes WxH (the grid is 2x2 of these)")
ap.add_argument("--lora", default="qwen/gso-view-v1-750.safetensors")
ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()
adir, work, out = args.angles.resolve(), args.work.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
dev = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(dev)
views = {v["name"]: v for v in json.loads((adir / "views.json").read_text())["views"]}
groups = [g.split(",") for g in args.groups.split("/")]
panels = [tuple(map(int, p.split("x"))) for p in args.panel.split(",")]
PROMPT = ("Image 1 is a 2x2 grid of four views of the same object from different viewpoints. Render every panel of "
          "image 1 as a real photo of the object in image 2, each panel from its own viewpoint, all four consistent "
          "with each other.")
VARIANTS = {"lora": args.lora, "base": None}


def grid_image(names: list[str], pw: int, ph: int) -> Image.Image:
    g = Image.new("RGB", (2 * pw, 2 * ph))
    for i, n in enumerate(names):
        g.paste(Image.open(adir / "input" / f"{n}.png").convert("RGB").resize((pw, ph), Image.LANCZOS), ((i % 2) * pw, (i // 2) * ph))
    return g


def cut(img: Image.Image, i: int) -> Image.Image:
    w, h = img.width // 2, img.height // 2
    return img.crop(((i % 2) * w, (i // 2) * h, (i % 2 + 1) * w, (i // 2 + 1) * h))


async def generate() -> dict:
    from giro.comfy import server
    from giro.comfy.client import ComfyClient, Done

    timings = {}
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                ref = await comfy.upload_image(adir / "hero_ref.png", subfolder="giro/grid_pilot")
                for gi, names in enumerate(groups):
                    for pw, ph in panels:
                        src = out / f"g{gi}-{pw}x{ph}-input.png"
                        grid_image(names, pw, ph).save(src)
                        c1 = await comfy.upload_image(src, subfolder="giro/grid_pilot_in")
                        for var, lora in VARIANTS.items():
                            f = out / f"g{gi}-{pw}x{ph}-{var}.png"
                            if f.exists():
                                continue
                            wf = view_lora_workflow(c1, ref, PROMPT, lora, args.seed + gi, lightning=True, prefix="giro/grid_pilot")
                            peak[0] = 0
                            t0 = time.monotonic()
                            done = None
                            async for ev in comfy.run(wf):
                                if isinstance(ev, Done):
                                    done = ev
                            await comfy.download(done.outputs["save"]["images"][0], f)
                            timings[f.stem] = {"s": round(time.monotonic() - t0, 1), "peak_vram_mb": peak[0]}
                            print(f"{f.name}: {timings[f.stem]}", flush=True)
            finally:
                await comfy.free()
    return timings


def vram_watch(stop: list) -> list:
    """Peak memory used on the GPU (ComfyUI is another process, so nvidia-smi's view; the first call of a
    variant includes loading it)."""
    import subprocess
    import threading

    peak = [0]

    def loop() -> None:
        while not stop:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(args.gpu)],
                               capture_output=True, text=True)
            try:
                peak[0] = max(peak[0], int(r.stdout.strip()))
            except ValueError:
                pass
            time.sleep(0.5)

    threading.Thread(target=loop, daemon=True).start()
    return peak


stop: list = []
peak = vram_watch(stop)
timings = asyncio.run(generate())
stop.append(1)

# Agreement on the mesh.
xyz, nrm, _, _ = samples(work, 2_000_000, dev)


def project(img: Image.Image, name: str) -> tuple[torch.Tensor, torch.Tensor, float]:
    cov = np.asarray(Image.open(adir / "input" / f"{name}_mask.png").convert("L")) > 127
    img = img.convert("RGB").resize(cov.shape[::-1], Image.LANCZOS)
    sil = ndimage.binary_fill_holes(ndimage.binary_opening(np.asarray(img).max(-1) > 3, iterations=2))
    iou = float((sil & cov).sum() / max((sil | cov).sum(), 1))
    cam = campath.PathCamera.from_json(views[name]["camera"])
    col, wt = Source(name, img, cov, cam, 1.0, dev).paint(xyz, nrm, 0.012)
    return col, wt, iou


def agreement(outs: dict[str, Image.Image]) -> dict:
    proj = {n: project(im, n) for n, im in outs.items()}
    diffs = []
    names = list(outs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (ca, wa, _), (cb, wb, _) = proj[names[i]], proj[names[j]]
            both = (wa > 0.1) & (wb > 0.1)
            if both.sum() > 2000:
                diffs.append(float((ca[both] - cb[both]).abs().mean()))
    return {"pair_diff": round(float(np.mean(diffs)), 4) if diffs else None, "pairs": len(diffs),
            "iou": [round(p[2], 3) for p in proj.values()]}


report = {"calls": timings, "groups": {}}
th = 256
for gi, names in enumerate(groups):
    rows = [("proxy", {n: Image.open(adir / "input" / f"{n}.png") for n in names}),
            ("one-by-one (route)", {n: Image.open(adir / f"{n}.png") for n in names})]
    for pw, ph in panels:
        for var in VARIANTS:
            f = out / f"g{gi}-{pw}x{ph}-{var}.png"
            if f.exists():
                g = Image.open(f)
                rows.append((f"grid {pw}x{ph} {var}", {n: cut(g, i) for i, n in enumerate(names)}))
    rep = {}
    for label, outs in rows[1:]:
        rep[label] = agreement(outs)
        print(f"group {gi} {names} {label}: {rep[label]}", flush=True)
    report["groups"][",".join(names)] = rep
    tw = th * 3 // 4
    sheet = Image.new("RGB", (tw * 4 + 180, th * len(rows)), (40, 40, 40))
    for r, (label, outs) in enumerate(rows):
        d = ImageDraw.Draw(sheet)
        d.text((4, r * th + 4), label, fill=(255, 255, 0))
        if label in rep:
            d.text((4, r * th + 20), f"diff {rep[label]['pair_diff']}", fill=(255, 255, 255))
            d.text((4, r * th + 36), f"IoU {rep[label]['iou']}", fill=(255, 255, 255))
        for i, n in enumerate(names):
            sheet.paste(outs[n].convert("RGB").resize((tw, th)), (180 + i * tw, r * th))
    sheet.save(out / f"sheet-g{gi}.jpg", quality=90)
(out / "report.json").write_text(json.dumps(report, indent=1))
print(json.dumps({k: v for k, v in report.items() if k != "groups"}), flush=True)
