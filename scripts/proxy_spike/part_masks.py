"""Part labels for the texture route's source views (parts.py): SAM 3.1 per part prompt on the hero and
every anchor, saved as label images WORK/parts/<name>.png (0 = no part, k = the k-th part) for
project_texture.py --parts, plus WORK/parts/check.jpg (each view with its parts tinted).

    part_masks.py ATTEMPT ANCHOR_DIR WORK GPU --parts "tire,wheel rim"
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from PIL import Image

from giro.comfy import server
from giro.comfy.client import ComfyClient
from parts import sam_labels, save_labels

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("anchor_dir", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--parts", required=True, help="SAM prompts, comma separated; a later part overrides an earlier one")
ap.add_argument("--threshold", type=float, help="SAM detection score threshold (default: the masks stage's 0.5); lower finds "
                "half-hidden instances (a scooter's rear wheel)")
args = ap.parse_args()
parts = [p.strip() for p in args.parts.split(",") if p.strip()]
adir, out = args.anchor_dir.resolve(), args.work.resolve() / "parts"
paths = {"hero": args.attempt.resolve() / "hero" / "hero.png"}
paths |= {n: adir / f"{n}.png" for n in sorted(json.loads((adir / "registration.json").read_text()))}
TINT = np.array([[0, 0, 0], [255, 60, 60], [60, 160, 255], [80, 220, 80], [240, 200, 40]], np.float32)


async def main() -> None:
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                labels = await sam_labels(comfy, paths, parts, out / "raw", args.threshold)
            finally:
                await comfy.free()
    save_labels(out, parts, labels)
    tiles = []
    for name, lab in list(labels.items())[:16]:
        img = np.asarray(Image.open(paths[name]).convert("RGB"), np.float32)
        tint = TINT[np.minimum(lab, len(TINT) - 1)]
        img = np.where(lab[..., None] > 0, 0.45 * img + 0.55 * tint, img)
        tiles.append(Image.fromarray(img.astype(np.uint8)).resize((240, round(240 * img.shape[0] / img.shape[1]))))
    h = max(t.height for t in tiles)
    sheet = Image.new("RGB", (240 * 8, h * ((len(tiles) + 7) // 8)), (30, 30, 30))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 8) * 240, (i // 8) * h))
    sheet.save(out / "check.jpg", quality=88)
    share = {n: [round(float((lab == k).mean()), 4) for k in range(1, len(parts) + 1)] for n, lab in labels.items()}
    print(f"{len(labels)} views labeled ({', '.join(parts)}); share of each image per part: {share}", flush=True)

asyncio.run(main())
