"""Generative fill for the parts of a projection texture no source view painted (a tank's tracks
and road wheels, under the hull; they keep the proxy model's own darker colors): over the
attempt's training cameras, greedily the one that sees the most unpainted surface; render it, fill
the gaps from their painted neighbors (so the edit is not steered by the fallback colors), let
Qwen-Image-Edit 2511 make it a clean photo of the same subject, match its colors to the painted
part, and paint it back onto the unpainted samples that view sees only, which then count as
painted. Until VIEWS views or nothing much is left.

    fill_unseen.py ATTEMPT WORK GPU --texture IN.pt --save OUT.pt --subject "tank" [--views 8]

ATTEMPT is the source attempt (for nothing but the hero's frame in WORK/texture.json); WORK holds
the textured attempt (WORK/attempt/cameras.json). Writes OUT.pt for project_texture.py --texture
and WORK/fill/ (per view: render, pre-filled, edit, after).
"""
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient, Done
from texture_common import fingerprint, project, qwen_edit_workflow, render_points, sample, samples, slope_slack, zbuffer

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--texture", type=Path, required=True)
ap.add_argument("--save", type=Path, required=True)
ap.add_argument("--subject", default="object", help="what the photo shows, for the edit prompt")
ap.add_argument("--views", type=int, default=8)
ap.add_argument("--min-px", type=int, default=3000, help="stop when the best view sees fewer unpainted pixels")
ap.add_argument("--cameras", type=Path, help="default: WORK/attempt/cameras.json")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()
work = args.work.resolve()
dev = torch.device(f"cuda:{args.gpu}")
out = work / "fill"
out.mkdir(exist_ok=True)
saved = torch.load(args.texture)
xyz, nrm, _, _ = samples(work, saved["points"], dev)
assert abs(saved["fingerprint"] - fingerprint(xyz)) < 1e-3, f"{args.texture} was painted on other samples"
rgb = saved["rgb"].to(dev).float()
winner = saved["winner"].to(dev)
cams_json = json.loads((args.cameras or work / "attempt" / "cameras.json").read_text())
W, H = cams_json["width"], cams_json["height"]
cams = [campath.PathCamera.from_json(c) for c in cams_json["frames"]]
PROMPT = (f"A clean, realistic photo of this {args.subject}, exactly this view: fill in the smudged, blurry "
          f"areas with the detail that belongs there, matching the rest. Black background.")


def unpainted_px(cam: campath.PathCamera, size: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray]:
    ind, mask = render_points(xyz, (winner < 0).float()[:, None], cam, *size, nrm=nrm)
    gap = (ind[..., 0] > 0.5) & mask
    return int(gap.sum()), gap, mask


def prefill(img: np.ndarray, gap: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Gaps filled from their painted neighbors, coarse to fine (normalized Gaussian blur)."""
    valid = (mask & ~gap).astype(np.float32)
    out_img = img.copy()
    todo = gap.copy()
    for sigma in (3, 8, 24, 64):
        num = np.stack([ndimage.gaussian_filter(img[..., c] * valid, sigma) for c in range(3)], -1)
        den = ndimage.gaussian_filter(valid, sigma)[..., None]
        ok = todo & (den[..., 0] > 0.02)
        out_img[ok] = (num / np.maximum(den, 1e-6))[ok]
        todo &= ~ok
    return out_img


def match_colors(edit: np.ndarray, ref: np.ndarray, where: np.ndarray) -> np.ndarray:
    a, b = edit[where], ref[where]
    return np.clip((edit - a.mean(0)) / (a.std(0) + 1e-6) * b.std(0) + b.mean(0), 0, 1)


async def main() -> None:
    global rgb, winner
    report = []
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                for k in range(args.views):
                    counts = [unpainted_px(c, (W // 4, H // 4))[0] for c in cams]
                    best = int(np.argmax(counts))
                    n, gap, mask = unpainted_px(cams[best], (W, H))
                    if n < args.min_px:
                        print(f"view {k + 1}: best camera sees {n} unpainted px, stopping", flush=True)
                        break
                    cam = cams[best]
                    img, _ = render_points(xyz, rgb, cam, W, H, nrm=nrm)
                    img = np.clip(img, 0, 1) * mask[..., None]
                    filled = prefill(img, gap, mask)
                    Image.fromarray((img * 255).astype(np.uint8)).save(out / f"{k:02d}-render.png")
                    Image.fromarray((filled * 255).astype(np.uint8)).save(out / f"{k:02d}-prefilled.png")
                    # Masked sampling: only the gaps (grown, so the edit blends in) may change.
                    paint = ndimage.binary_dilation(gap, iterations=8) & ndimage.binary_dilation(mask, iterations=4)
                    Image.fromarray(paint.astype(np.uint8) * 255).convert("RGB").save(out / f"{k:02d}-mask.png")
                    image = await comfy.upload_image(out / f"{k:02d}-prefilled.png")
                    mask_up = await comfy.upload_image(out / f"{k:02d}-mask.png")
                    done = None
                    async for ev in comfy.run(qwen_edit_workflow(image, PROMPT, args.seed + k, "giro/fill", mask=mask_up)):
                        if isinstance(ev, Done):
                            done = ev
                    assert done is not None
                    await comfy.download(done.outputs["save"]["images"][0], out / f"{k:02d}-edit.png")
                    edit = np.asarray(Image.open(out / f"{k:02d}-edit.png").convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255
                    edit = match_colors(edit, img, mask & ~gap)
                    grown = ndimage.binary_dilation(gap, iterations=2) & mask
                    alpha = np.clip(ndimage.gaussian_filter(grown.astype(np.float32), 1.5), 0, 1) * grown
                    # Paint back: unpainted samples in this view's nearest layer, where the edit is used.
                    u, v, z = project(cam, W, H, xyz)
                    ok, _, zmin, _ = zbuffer(u, v, z, W, H, slope_slack(cam, W, H, xyz, nrm, z))
                    a = sample(torch.from_numpy(alpha).to(dev)[None], u, v, W, H)[:, 0]
                    take = ok & (z <= zmin + 0.005) & (winner < 0) & (a > 0.5)
                    new = sample(torch.from_numpy(edit).permute(2, 0, 1).to(dev), u, v, W, H)
                    rgb = torch.where(take[:, None], new, rgb)
                    winner = torch.where(take, torch.full_like(winner, 3000 + k), winner)
                    after, _ = render_points(xyz, rgb, cam, W, H, nrm=nrm)
                    tile = np.concatenate([img, filled, edit * mask[..., None], np.clip(after, 0, 1)], 1)
                    Image.fromarray((tile * 255).astype(np.uint8)).resize((W, H // 4)).save(out / f"{k:02d}-sheet.jpg", quality=88)
                    report.append({"camera": best, "unpainted_px": n, "painted_samples": int(take.sum())})
                    print(f"view {k + 1}: camera {best} (yaw {cam.yaw:.0f}, pitch {cam.pitch:.0f}) saw {n:,} unpainted px; "
                          f"painted {int(take.sum()):,} samples; still unpainted {float((winner < 0).float().mean()):.1%}", flush=True)
            finally:
                await comfy.free()
    torch.save({"rgb": rgb.half().cpu(), "winner": winner.cpu(), "points": saved["points"], "fingerprint": saved["fingerprint"],
                "patches": saved.get("patches", []) + [{"fill": report, "subject": args.subject}]}, args.save)
    (out / "fill.json").write_text(json.dumps(report, indent=1))

asyncio.run(main())
