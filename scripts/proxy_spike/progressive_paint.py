"""Progressive painting (cf. TEXTure, Text2Tex): one view at a time, from the hero outward, the textured
mesh's render is made a clean photo by Qwen-Image-Edit 2511 (hero as picture 2) only where this view
sees the surface clearly better than whatever painted it so far; everything else stays as it is and
steers the edit, so each view continues the detail its neighbors painted instead of redrawing it.
The edit is painted back by overwriting, not blending (94 independent edits averaged by view
selection lost their crispness, Oct 5), and those samples are locked with this view's quality.

    progressive_paint.py ATTEMPT WORK GPU --texture IN.pt --save OUT.pt --subject "tank"
        [--keys 20:8] [--hero-lock 1] [--every 2] [--denoise 0.6] [--better 1.5] [--out-dir progressive]

First the key views (--keys PITCH:N,...: N cameras around at that pitch, from the hero's yaw) are edited
whole, unmasked (a masked edit only redraws its mask; the whole-image edits are the crisp ones), each
painted where it sees the surface better than what painted it; then every Nth camera, masked.

ATTEMPT is the source attempt (hero, hero mask); WORK holds mesh.npz, texture.json (the hero's fitted
camera) and a textured attempt (WORK/attempt/cameras.json: the cameras to paint from). A sample's
quality from a view is |cos| of its viewing angle to the 4th power (project_texture.py's weight);
the hero's surface starts locked with --hero-lock times that. Writes OUT.pt for project_texture.py --texture
and WORK/progressive/ (per view: render, regenerate mask, edit, after).
"""
import argparse
import asyncio
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from giro import path as campath
from giro.comfy import server
from giro.comfy.client import ComfyClient, Done
from gso_dataset import caption
from parts import PartView, gate, sam_labels
from texture_common import (fingerprint, project, qwen_edit_workflow, render_points, sample, samples, slope_slack,
                            view_lora_workflow, zbuffer)

ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--texture", type=Path, required=True)
ap.add_argument("--save", type=Path, required=True)
ap.add_argument("--subject", default="object")
ap.add_argument("--every", type=int, default=2, help="every Nth camera of WORK/attempt (0: the key views only, no masked pass)")
ap.add_argument("--denoise", type=float, default=0.6)
ap.add_argument("--better", type=float, default=1.5, help="regenerate where this view's quality is this many times the locked one")
ap.add_argument("--min-px", type=int, default=1500, help="skip views with fewer pixels to regenerate")
ap.add_argument("--keys", default="20:8", help="key views edited whole first, PITCH:N,... ('' for none)")
ap.add_argument("--best-of", type=int, default=3, help="key views: this many edits, the one that keeps its render best")
ap.add_argument("--hero-lock", type=float, default=1.0, help="the hero's surface starts locked with this times its quality")
ap.add_argument("--out-dir", default="progressive")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--lora", default="", help="regenerate with the GSO view LoRA (a loras/ path, e.g. qwen/gso-view-v1-750.safetensors) "
                "instead of plain 2511 edits: the render as image 1, --lora-ref as image 2, the camera as the caption; "
                "full denoise with Lightning, as scored (board #3719); the edit is still painted only where this view is better")
ap.add_argument("--lora-min-iou", type=float, default=0.8,
                help="with --lora: paint an edit only if its silhouette (off black) matches the render's this well; the LoRA "
                     "sometimes turns the view (Oct 6, scooter), and a turned edit painted back doubles the subject")
ap.add_argument("--lora-tries", type=int, default=3, help="with --lora: seeds per view until one passes --lora-min-iou")
ap.add_argument("--lora-ref", type=Path, help="with --lora: the hero as the LoRA saw it (lora_anchors.py's hero_ref.png)")
ap.add_argument("--parts", default="", help="part-aware paint-back (parts.py): SAM labels each edit with these prompts (the "
                "same list as project_texture.py --parts, whose sample labels the texture carries) and an edit pixel "
                "paints only samples of its own part")
ap.add_argument("--part-edge-px", type=float, default=3.0, help="with --parts: edit pixels this close to a part boundary fade out")
args = ap.parse_args()
attempt, work = args.attempt.resolve(), args.work.resolve()
dev = torch.device(f"cuda:{args.gpu}")
out = work / args.out_dir
out.mkdir(exist_ok=True)
saved = torch.load(args.texture)
xyz, nrm, _, _ = samples(work, saved["points"], dev)
assert abs(saved["fingerprint"] - fingerprint(xyz)) < 1e-3, f"{args.texture} was painted on other samples"
rgb = saved["rgb"].to(dev).float()
# Which edit painted each sample: 2000 + its number here (patch_region.py uses 1000 + n), so the
# source-view map covers this layer too.
winner = saved["winner"].to(dev)
parts = [p.strip() for p in args.parts.split(",") if p.strip()]
sample_parts = None
if parts:
    assert saved.get("parts") == parts, f"{args.texture} carries part labels for {saved.get('parts')}, not {parts}"
    sample_parts = saved["labels"].to(dev).long()
cams_json = json.loads((work / "attempt" / "cameras.json").read_text())
W, H = cams_json["width"], cams_json["height"]
hero = campath.PathCamera.from_json(json.loads((work / "texture.json").read_text())["fits"]["hero"]["camera"])


def prompt(cam: campath.PathCamera) -> str:
    """The edit prompt, saying where this view is relative to the hero's: unmasked, a view from behind
    was once turned into a front (a gun muzzle and a number painted on a tank's turret rear)."""
    d = (cam.yaw - hero.yaw) % 360
    side = ("from about the same side as picture 2" if d < 30 or d > 330 else
            "from the side opposite to picture 2, so what faces the camera in picture 2 faces away here" if 150 < d < 210 else
            "from a side a quarter turn around from picture 2" if 60 < d < 120 or 240 < d < 300 else
            "from a three-quarter view between the side of picture 2 and the opposite side")
    above = ", looking down from above" if cam.pitch >= 45 else ""
    return (f"Picture 1 is a render of a {args.subject} seen {side}{above}; picture 2 is a photo of the same {args.subject}. "
            f"Make picture 1 a clean, sharp, realistic photo of the {args.subject}, exactly this view and framing: "
            f"keep every shape, edge and part where it is, continue the detail and paint of the clean parts, and turn "
            f"the smeared, patchy, noisy surfaces into the clean, coherent detail that belongs there, lit evenly like "
            f"picture 2. Do not add any part, marking or text that is not already in picture 1. Black background.")


def edges(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    g = img.mean(-1)
    gi = np.hypot(ndimage.sobel(g, 0), ndimage.sobel(g, 1))
    return (gi > np.percentile(gi[mask], 90)) & mask


def agreement(edit: np.ndarray, render: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """How well an edit keeps its render: edge F1 between the two (3 px), minus twice the 98th percentile
    of their blurred color difference (a part or marking the render lacks is a large local change)."""
    e1, e2 = edges(edit, mask), edges(render, mask)
    rec = (ndimage.distance_transform_edt(~e2)[e1] <= 3).mean()
    prec = (ndimage.distance_transform_edt(~e1)[e2] <= 3).mean()
    f1 = float(2 * rec * prec / max(rec + prec, 1e-9))
    diff = np.abs(np.stack([ndimage.gaussian_filter(edit[..., c] - render[..., c], 4) for c in range(3)], -1)).mean(-1)
    p98 = float(np.percentile(diff[mask], 98))
    return f1 - 2 * p98, f1, p98


def view(cam: campath.PathCamera, w: int, h: int, mask: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per sample: in this view's front layer (and inside `mask`), where it lands, and its quality."""
    u, v, z = project(cam, w, h, xyz)
    ok, pix, zmin, _ = zbuffer(u, v, z, w, h, slope_slack(cam, w, h, xyz, nrm, z))
    front = ok & (z <= zmin + 0.005)
    if mask is not None:
        front &= torch.from_numpy(mask.reshape(-1)).to(dev)[pix]
    to_cam = torch.tensor(np.asarray(cam.position()), device=dev, dtype=torch.float32) - xyz
    cos = (nrm * to_cam).sum(1).abs() / to_cam.norm(dim=1)
    return front, u, v, cos**4


# The hero's surface starts locked: it is the real photo (project_texture.py counts it double).
hero_mask = np.asarray(Image.open(attempt / "proxy" / "hero_mask.png")) > 127
front, _, _, q = view(hero, hero_mask.shape[1], hero_mask.shape[0], hero_mask)
locked = torch.where(front, args.hero_lock * q, torch.zeros_like(q))
print(f"hero locks {front.float().mean():.1%} of the samples", flush=True)


def angle_from_hero(c: campath.PathCamera) -> float:
    return math.degrees(math.acos(float(np.clip(np.dot(c.forward(), hero.forward()), -1, 1))))


all_cams = [campath.PathCamera.from_json(c) for c in cams_json["frames"]]


def nearest(yaw: float, pitch: float) -> int:
    return min(range(len(all_cams)), key=lambda i: abs(all_cams[i].pitch - pitch) * 10
               + abs((all_cams[i].yaw - yaw + 180) % 360 - 180))


keys = [nearest(hero.yaw + 360 * k / int(n), float(p)) for p, n in (spec.split(":") for spec in args.keys.split(",") if spec)
        for k in range(int(n))]
keys = sorted(dict.fromkeys(keys), key=lambda i: angle_from_hero(all_cams[i]))
order = [] if args.every <= 0 else sorted(range(0, len(all_cams), args.every), key=lambda i: angle_from_hero(all_cams[i]))


async def paint_view(comfy: ComfyClient, ref: str, i: int, whole: bool, better: float, report: list) -> None:
    """Edit camera i's render (whole, or masked to what it sees better than `better` times the locked
    quality) and paint the edit back there by overwriting; lock what it painted."""
    global rgb, locked, winner
    cam = all_cams[i]
    front, u, v, q = view(cam, W, H)
    need = front & (q > better * locked)
    img, mask = render_points(xyz, rgb, cam, W, H, nrm=nrm)
    frac, _ = render_points(xyz, need.float()[:, None], cam, W, H, nrm=nrm)
    regen = ndimage.binary_opening((frac[..., 0] > 0.5) & mask, iterations=2)
    if regen.sum() < args.min_px:
        return
    k = len(report)
    img = np.clip(img, 0, 1) * mask[..., None]
    Image.fromarray((img * 255).astype(np.uint8)).save(out / f"{k:02d}-render.png")
    image = await comfy.upload_image(out / f"{k:02d}-render.png")
    mask_up = None
    if not whole:  # the edit may change the regenerated pixels (grown, so it blends in), nothing else
        edit_mask = ndimage.binary_dilation(regen, iterations=8) & ndimage.binary_dilation(mask, iterations=4)
        Image.fromarray(edit_mask.astype(np.uint8) * 255).convert("RGB").save(out / f"{k:02d}-mask.png")
        mask_up = await comfy.upload_image(out / f"{k:02d}-mask.png")
    tries = []
    n_tries = max(args.best_of if whole else 1, args.lora_tries if args.lora else 1)
    for t in range(n_tries):
        done = None
        if args.lora:  # whole-image generation; only the regenerated region is painted back below
            wf = view_lora_workflow(image, ref, caption(cam.yaw - hero.yaw, cam.pitch), args.lora, args.seed + 100 * t + k,
                                    lightning=True, prefix="giro/progressive")
        else:
            wf = qwen_edit_workflow(image, prompt(cam), args.seed + 100 * t + k, "giro/progressive", mask=mask_up, ref=ref, denoise=args.denoise)
        async for ev in comfy.run(wf):
            if isinstance(ev, Done):
                done = ev
        assert done is not None
        dst = out / (f"{k:02d}-edit.png" if not whole else f"{k:02d}-edit-{t}.png")
        await comfy.download(done.outputs["save"]["images"][0], dst)
        e = np.asarray(Image.open(dst).convert("RGB").resize((W, H), Image.LANCZOS), np.float32) / 255
        if args.lora:  # (score, silhouette IoU, edge F1): only edits that keep the view count
            sil = ndimage.binary_fill_holes(ndimage.binary_opening(e.max(-1) > 0.01, iterations=2))
            iou = float((sil & mask).sum() / max((sil | mask).sum(), 1))
            ok = iou >= args.lora_min_iou
            sc = agreement(e, img, mask)
            tries.append(((sc[0] if ok else -9.0, iou, sc[1]), e))
            if ok and not whole:
                break
            continue
        tries.append((agreement(e, img, mask) if whole else (0.0, 0.0, 0.0), e))
    best = max(range(len(tries)), key=lambda t: tries[t][0][0])
    if args.lora and tries[best][0][0] <= -9.0:
        print(f"{'key' if whole else 'view'}: camera {i} (yaw {cam.yaw - hero.yaw:+.0f}, pitch {cam.pitch:.0f}) skipped: "
              f"no edit kept the view (silhouette IoU {[round(sc[1], 2) for sc, _ in tries]})", flush=True)
        report.append({"camera": i, "whole": whole, "skipped": True, "tries": [[round(x, 3) for x in sc] for sc, _ in tries]})
        return
    edit = tries[best][1]
    # Overwrite, feathered only at the regenerated region's edge; lock what the edit painted.
    alpha = np.clip(ndimage.gaussian_filter(regen.astype(np.float32), 1.5), 0, 1) * regen
    a = sample(torch.from_numpy(alpha).to(dev)[None], u, v, W, H)[:, 0] * need
    if sample_parts is not None:  # the edit's own parts (SAM on it): its rim pixels never land on tire samples
        path = out / (f"{k:02d}-edit-{best}.png" if whole else f"{k:02d}-edit.png")
        lab = (await sam_labels(comfy, {"edit": path}, parts, out / "parts-raw" / f"{k:02d}"))["edit"]
        a = a * gate(sample_parts, *PartView(lab, args.part_edge_px, dev, (W, H)).at(u, v))
    new = sample(torch.from_numpy(edit).permute(2, 0, 1).to(dev), u, v, W, H)
    rgb = rgb * (1 - a[:, None]) + new * a[:, None]
    locked = torch.where(a > 0.5, q, locked)
    winner = torch.where(a > 0.5, torch.full_like(winner, 2000 + k), winner)
    after, _ = render_points(xyz, rgb, cam, W, H, nrm=nrm)
    tile = np.concatenate([img, np.repeat(regen[..., None], 3, -1).astype(np.float32), edit * mask[..., None], np.clip(after, 0, 1)], 1)
    Image.fromarray((tile * 255).astype(np.uint8)).resize((W, H // 4)).save(out / f"{k:02d}-sheet.jpg", quality=88)
    report.append({"camera": i, "whole": whole, "yaw": round(cam.yaw - hero.yaw, 1), "pitch": cam.pitch,
                   "regen_px": int(regen.sum()), "painted": int((a > 0.5).sum()), "pick": best,
                   "tries": [[round(x, 3) for x in sc] for sc, _ in tries] if whole or args.lora else None})
    print(f"{'key' if whole else 'view'} {k + 1}: camera {i} (yaw {cam.yaw - hero.yaw:+.0f}, pitch {cam.pitch:.0f}) "
          f"regenerates {int(regen.sum()):,} px, paints {int((a > 0.5).sum()):,} samples"
          + (f"; tries (score, edge F1, p98 color) {[[round(x, 3) for x in sc] for sc, _ in tries]}, picked {best}" if whole else ""), flush=True)


async def main() -> None:
    report: list = []
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                ref = await comfy.upload_image(args.lora_ref if args.lora else attempt / "hero" / "hero.png")
                for i in keys:
                    await paint_view(comfy, ref, i, True, 1.0, report)
                for i in order:
                    await paint_view(comfy, ref, i, False, args.better, report)
            finally:
                await comfy.free()
    torch.save({"rgb": rgb.half().cpu(), "winner": winner.cpu(), "points": saved["points"], "fingerprint": saved["fingerprint"],
                "patches": saved.get("patches", []) + [{"progressive": report, "subject": args.subject}]}
               | {k: saved[k] for k in ("labels", "parts") if k in saved}, args.save)
    (out / "progressive.json").write_text(json.dumps(report, indent=1))

asyncio.run(main())
