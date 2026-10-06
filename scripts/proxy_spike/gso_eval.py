"""Scores the GSO view-LoRA pilot (board #3704) on the held-out objects: each checkpoint against plain
Qwen-Image-Edit 2511 with the same inputs, prompt and seed.

    gso_eval.py gen OUT GPU [--shard J/K] [--variants base,500,750,1000,250] [--steps 25] [--cfg 3]
        [--lightning]
    gso_eval.py masks OUT GPU [--shard J/K]   (BiRefNet foreground masks of the outputs, for alignment)
    gso_eval.py score OUT
    gso_eval.py sheets OUT
    v2 (board #3713): [--dataset DIR --controls DIR --renders DIR --size 1080x1440 --gen-size WxH
        --lora-template 'qwen/gso-view-{}.safetensors' --variants base,v2-250,... --image3 --sheet-variants ...]

Pairs: per held-out object two kept views (dataset/heldout/pairs.csv), picked deterministically: an
eye-level one (|pitch| < 15) and a high one (pitch > 30), each the one whose yaw offset from the hero is
closest to 90 deg (both seen and unseen surface in view). Inputs as trained: image 1 = control1 (the
proxy render), image 2 = control2 (the hero), the caption from target/<id>.txt; 768x1024, no Lightning
(--lightning: 4 steps, cfg 1, for the route's setting), euler/simple, ModelSamplingAuraFlow 3.1.

Score, against the ground truth at the target camera (controls/<obj>/target/<view>.png, RGBA):
- seen / unseen: a target pixel is seen when its surface point (GT depth, back-projected) projects into
  the hero inside the frame with the hero's GT depth within tolerance, i.e. the hero had that pixel;
  PSNR and spatial LPIPS (AlexNet) over GT-alpha & seen, GT-alpha & unseen, and the whole GT alpha;
- alignment: silhouette IoU of the output (BiRefNet's foreground from `masks`, so invented backgrounds
  don't count; else pixels off black) vs the control's coverage mask (what the route paints onto) and
  vs the GT alpha.
Writes OUT/gen/<variant>/<id>.png, OUT/gen/<variant>/times.json, OUT/scores.csv, OUT/summary.txt and
OUT/sheets/*.jpg (rows: proxy | hero | base | 500 | 750 | 1000 | GT).
"""
import argparse
import asyncio
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

P = Path("/home/kevin/ai/giro/data/gso-pilot")

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["gen", "masks", "score", "sheets"])
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int, nargs="?", default=0)
ap.add_argument("--shard", default="0/1", metavar="J/K")
ap.add_argument("--variants", default="base,500,750,1000,250")
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--cfg", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--lightning", action="store_true", help="the route's Lightning 4-step setting; variants get an -l suffix")
ap.add_argument("--only", type=int, default=0, help="first N pairs (loader check)")
ap.add_argument("--ids", default="", help="these pair ids instead of the picked pairs (comma-separated)")
ap.add_argument("--unet", default="qwen_image_edit_2511_int8_convrot.safetensors",
                help="base model file; qwen_image_edit_2511_fp8_e4m3fn.safetensors is a plain fp8 cast of the diffusers weights (variants get -f)")
ap.add_argument("--ref-method", default="index_timestep_zero", help="FluxKontextMultiReferenceLatentMethod: 2511's native "
                "'index_timestep_zero' matches the trainer's samples (black background, pose held); 'index' (2509) is weaker (variants get -i)")
ap.add_argument("--swap", action="store_true", help="hero as image 1, proxy as image 2 (test; variants get -s)")
ap.add_argument("--no-cfgnorm", action="store_true", help="drop CFGNorm (the trainer's sampler has none); variants get -n")
# v2 (board #3713); the defaults are v1's
ap.add_argument("--dataset", type=Path, default=P / "dataset" / "heldout", help="held-out split (v2: ~/ai/datasets/v2/dataset/heldout)")
ap.add_argument("--controls", type=Path, default=P / "controls", help="gso_controls.py's output (v2: ~/ai/datasets/v2/controls)")
ap.add_argument("--renders", type=Path, default=Path("/home/kevin/ai/datasets/gso/renders"), help="gso_render.py's heroes (v2: ~/ai/datasets/v2/renders)")
ap.add_argument("--size", default="768x1024", help="ground-truth size WxH (v2: 1080x1440)")
ap.add_argument("--gen-size", default="", help="WxH to generate at (image 1 and 3 resized first; default: image 1's own size)")
ap.add_argument("--lora-template", default="qwen/gso-view-v1-{}.safetensors",
                help="variant -> LoRA file (v2: 'qwen/gso-view-{}.safetensors' with variants v2-<step>, gso_v2_watch.py's links)")
ap.add_argument("--image3", action="store_true", help="also pass control3/<id>.png as image 3 (v2's nearest accepted view)")
ap.add_argument("--sheet-variants", default="base,500,750,1000", help="the generated columns of `sheets`")
args = ap.parse_args()
out = args.out.resolve()
DS, CONTROLS, RENDERS = args.dataset, args.controls, args.renders
W, H = map(int, args.size.split("x"))


def pairs() -> list[dict]:
    rows = [r for r in csv.DictReader(open(DS / "pairs.csv")) if r["kept"] == "1"]
    by_obj: dict[str, list[dict]] = {}
    for r in rows:
        by_obj.setdefault(r["object"], []).append(r)
    picked = []
    for obj in sorted(by_obj):
        vs = by_obj[obj]
        off = lambda r: abs(min(float(r["rel_yaw"]), 360 - float(r["rel_yaw"])) - 90)  # noqa: E731
        for bucket, test in (("eye", lambda p: abs(p) < 15), ("high", lambda p: p > 30)):
            cand = [r for r in vs if test(float(r["pitch"]))]
            if cand:
                r = min(cand, key=lambda r: (off(r), r["view"]))
                picked.append(r | {"bucket": bucket})
    return picked


def workflow(c1: str, c2: str, prompt: str, lora: str | None, seed: int, c3: str | None = None) -> dict:
    from texture_common import view_lora_workflow

    return view_lora_workflow(c1, c2, prompt, lora, seed, c3, lightning=args.lightning, steps=args.steps, cfg=args.cfg,
                              unet=args.unet, ref_method=args.ref_method, cfgnorm=not args.no_cfgnorm, gen_size=args.gen_size)


async def gen() -> None:
    from giro.comfy import server
    from giro.comfy.client import ComfyClient, Done

    j, k = map(int, args.shard.split("/"))
    todo = ([r for r in csv.DictReader(open(DS / "pairs.csv")) if r["id"] in args.ids.split(",")] if args.ids else pairs())[j::k]
    if args.only:
        todo = todo[:args.only]
    suffix = ("-l" if args.lightning else "") + ("-n" if args.no_cfgnorm else "") + ("-f" if "fp8" in args.unet else "") + ("-i" if args.ref_method != "index_timestep_zero" else "") + ("-s" if args.swap else "")
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                for v in args.variants.split(","):
                    vd = out / "gen" / (v + suffix)
                    vd.mkdir(parents=True, exist_ok=True)
                    tf = vd / f"times-{j}.json"
                    times = json.loads(tf.read_text()) if tf.exists() else {}
                    for n, r in enumerate(todo):
                        dst = vd / f"{r['id']}.png"
                        if dst.exists():
                            continue
                        # separate subfolders: both files are named <id>.png, and one upload would replace the other
                        c1 = await comfy.upload_image(DS / "control1" / f"{r['id']}.png", subfolder="giro/gso_c1")
                        c2 = await comfy.upload_image(DS / "control2" / f"{r['id']}.png", subfolder="giro/gso_c2")
                        if args.swap:
                            c1, c2 = c2, c1
                        c3 = await comfy.upload_image(DS / "control3" / f"{r['id']}.png", subfolder="giro/gso_c3") if args.image3 else None
                        prompt = (DS / "target" / f"{r['id']}.txt").read_text().strip()
                        seed = args.seed + sum(map(ord, r["id"])) % 10_000
                        t0 = time.monotonic()
                        done = None
                        lora = None if v == "base" else args.lora_template.format(v)
                        async for ev in comfy.run(workflow(c1, c2, prompt, lora, seed, c3)):
                            if isinstance(ev, Done):
                                done = ev
                        await comfy.download(done.outputs["save"]["images"][0], dst)
                        times[r["id"]] = round(time.monotonic() - t0, 1)
                        tf.write_text(json.dumps(times, indent=1))
                        print(f"[{v}{suffix} {n + 1}/{len(todo)}] {r['id']} {times[r['id']]} s", flush=True)
            finally:
                await comfy.free()


async def masks() -> None:
    from giro.comfy import server
    from giro.comfy.client import ComfyClient, Done

    j, k = map(int, args.shard.split("/"))
    files = sorted(f for f in (out / "gen").glob("*/*.png") if not f.name.endswith("_mask.png"))[j::k]
    with await asyncio.to_thread(server.Lease, args.gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                for n, f in enumerate(files):
                    dst = f.with_name(f.stem + "_mask.png")
                    if dst.exists():
                        continue
                    name = await comfy.upload_image(f, subfolder=f"giro/gso_m/{f.parent.name}")
                    wf = {"img": {"class_type": "LoadImage", "inputs": {"image": name, "upload": "image"}},
                          "bgm": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": "birefnet.safetensors"}},
                          "rm": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["bgm", 0], "image": ["img", 0]}},
                          "mi": {"class_type": "MaskToImage", "inputs": {"mask": ["rm", 0]}},
                          "save": {"class_type": "SaveImage", "inputs": {"images": ["mi", 0], "filename_prefix": "giro/gso_mask"}}}
                    done = None
                    async for ev in comfy.run(wf):
                        if isinstance(ev, Done):
                            done = ev
                    await comfy.download(done.outputs["save"]["images"][0], dst)
                    if n % 25 == 0:
                        print(f"[masks {n + 1}/{len(files)}]", flush=True)
            finally:
                await comfy.free(unload_models=False)  # BiRefNet is small; don't evict a generation run's models


# --- scoring ---------------------------------------------------------------------------------------

S_FORWARD = lambda yaw, pitch: np.array([-math.cos(math.radians(pitch)) * math.sin(math.radians(yaw)),  # noqa: E731
                                          math.sin(math.radians(pitch)),
                                          math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))])


def axes(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """As gso_render.py: camera x (right), y (down), forward, position in the splat frame."""
    if "rot" in cam:
        r, t = np.asarray(cam["rot"]), np.asarray(cam["t"])
        return r[0], r[1], r[2], -r.T @ t
    f = S_FORWARD(cam["yaw"], cam["pitch"])
    pos = np.asarray(cam["target"]) - cam["distance"] * f
    x = np.cross([0.0, 1.0, 0.0], f)
    x /= np.linalg.norm(x)
    return x, np.cross(f, x), f, pos


def seen_mask(obj: str, view: str) -> np.ndarray:
    """Target pixels whose surface point the hero sees (GT depth both ways)."""
    gt = json.loads((CONTROLS / obj / "gt_cameras.json").read_text())
    tcam = next(v for v in gt["views"] if v["name"] == view)
    hcam = json.loads((RENDERS / obj / "cameras.json").read_text())["views"][0]
    zt = np.load(CONTROLS / obj / "target" / "depth" / f"{view}.npy").astype(np.float32)
    zh = np.load(RENDERS / obj / "depth" / "hero.npy").astype(np.float32)
    x, y, f, pos = axes(tcam)
    fl = (min(W, H) / 2) / math.tan(math.radians(tcam["fov"]) / 2)
    vv, uu = np.mgrid[0:H, 0:W] + 0.5
    d = ((uu - W / 2) / fl)[..., None] * x + ((vv - H / 2) / fl)[..., None] * y + f
    ok = np.isfinite(zt)
    pts = pos + d * np.where(ok, zt, 0)[..., None]
    hx, hy, hf, hpos = axes(hcam)
    hfl = (min(W, H) / 2) / math.tan(math.radians(hcam["fov"]) / 2)
    q = pts - hpos
    z = q @ hf
    u = W / 2 + hfl * (q @ hx) / np.maximum(z, 1e-6)
    v = H / 2 + hfl * (q @ hy) / np.maximum(z, 1e-6)
    ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
    inside = ok & (z > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    seen = np.zeros((H, W), bool)
    # nearest hero depth in a 3x3 window: robust at depth edges
    zh_min = ndimage.minimum_filter(np.where(np.isfinite(zh), zh, np.inf), size=3)
    zz = zh_min[vi.clip(0, H - 1), ui.clip(0, W - 1)]
    tol = 0.01 + 0.004 * hcam["distance"]
    seen[inside] = (np.abs(z[inside] - zz[inside]) < tol)
    return seen


def silhouette(rgb: np.ndarray) -> np.ndarray:
    m = rgb.max(axis=2) > 18
    m = ndimage.binary_opening(m, iterations=1)
    lab, n = ndimage.label(m)
    if n > 1:  # keep components holding most of the area (the subject), drop specks
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        keep = np.flatnonzero(sizes >= 0.02 * sizes.max()) + 1
        m = np.isin(lab, keep)
    return ndimage.binary_fill_holes(m)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def score() -> None:
    import lpips
    import torch

    net = lpips.LPIPS(net="alex", spatial=True, verbose=False).eval()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(dev)
    variants = [d.name for d in sorted((out / "gen").iterdir()) if d.is_dir()]
    rows = []
    for r in pairs():
        obj, view, rid = r["object"], r["view"], r["id"]
        gt_rgba = np.asarray(Image.open(CONTROLS / obj / "target" / f"{view}.png").convert("RGBA")).astype(np.float32) / 255
        alpha = gt_rgba[..., 3] > 0.5
        gt = gt_rgba[..., :3] * gt_rgba[..., 3:]
        seen = seen_mask(obj, view) & alpha
        unseen = alpha & ~seen
        cover = np.asarray(Image.open(CONTROLS / obj / "control" / f"{view}_mask.png").resize((W, H), Image.NEAREST)) > 127
        gt_t = torch.tensor(gt).permute(2, 0, 1)[None].to(dev) * 2 - 1
        for v in variants:
            f = out / "gen" / v / f"{rid}.png"
            if not f.exists():
                continue
            o = np.asarray(Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS)).astype(np.float32) / 255
            with torch.no_grad():
                lp = net(torch.tensor(o).permute(2, 0, 1)[None].to(dev) * 2 - 1, gt_t)[0, 0].cpu().numpy()
            se = ((o - gt) ** 2).mean(axis=2)
            mf = f.with_name(f.stem + "_mask.png")
            if mf.exists():  # BiRefNet's foreground: an invented background is not the subject
                sil = ndimage.binary_fill_holes(np.asarray(Image.open(mf).convert("L").resize((W, H))) > 127)
            else:
                sil = silhouette((o * 255).astype(np.uint8))
            row = {"id": rid, "object": obj, "view": view, "bucket": r["bucket"], "pitch": r["pitch"], "rel_yaw": r["rel_yaw"],
                   "variant": v, "seen_frac": round(float(seen.sum() / max(alpha.sum(), 1)), 3),
                   "iou_control": round(iou(sil, cover), 4), "iou_gt": round(iou(sil, alpha), 4),
                   "control_iou_gt": round(iou(cover, alpha), 4)}
            for name, m in (("all", alpha), ("seen", seen), ("unseen", unseen)):
                ok = m.sum() > 500
                row[f"psnr_{name}"] = round(float(-10 * np.log10(se[m].mean() + 1e-10)), 3) if ok else ""
                row[f"lpips_{name}"] = round(float(lp[m].mean()), 4) if ok else ""
            rows.append(row)
        print(f"scored {rid}", flush=True)
    with open(out / "scores.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # summary: means per variant, over the pairs every variant has
    ids = set.intersection(*[{x["id"] for x in rows if x["variant"] == v} for v in variants])
    keys = ["psnr_all", "lpips_all", "psnr_seen", "lpips_seen", "psnr_unseen", "lpips_unseen", "iou_control", "iou_gt"]
    lines = [f"{len(ids)} pairs common to all variants ({', '.join(variants)})", ""]
    for bucket in ("all", "eye", "high"):
        lines.append(f"bucket {bucket}:")
        lines.append("variant  " + "  ".join(f"{k:>12}" for k in keys))
        for v in variants:
            sel = [x for x in rows if x["variant"] == v and x["id"] in ids and (bucket == "all" or x["bucket"] == bucket)]
            vals = []
            for k in keys:
                xs = [x[k] for x in sel if x[k] != ""]
                vals.append(f"{np.mean(xs):12.4f}" if xs else f"{'':>12}")
            lines.append(f"{v:8} " + "  ".join(vals) + f"   (n={len(sel)})")
        lines.append("")
    # paired: how often each LoRA variant beats base per pair
    for v in variants:
        if v.startswith("base"):
            continue
        b = {x["id"]: x for x in rows if x["variant"] == ("base-l" if v.endswith("-l") else "base")}
        s = {x["id"]: x for x in rows if x["variant"] == v}
        common = [i for i in s if i in b]
        if not common:
            continue
        win = lambda k, lower: sum((s[i][k] < b[i][k]) if lower else (s[i][k] > b[i][k])  # noqa: E731
                                   for i in common if s[i][k] != "" and b[i][k] != "")
        lines.append(f"{v} vs base, pairs better of {len(common)}: lpips_seen {win('lpips_seen', True)}, "
                     f"psnr_seen {win('psnr_seen', False)}, lpips_unseen {win('lpips_unseen', True)}, "
                     f"iou_control {win('iou_control', False)}, iou_gt {win('iou_gt', False)}")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def sheets() -> None:
    cols = [("proxy", None), ("hero", None), *[(v, v) for v in args.sheet_variants.split(",")], ("GT", None)]
    th, tw = 320, 240
    sd = out / "sheets"
    sd.mkdir(parents=True, exist_ok=True)
    ps = pairs()
    per = 6
    for s in range(0, len(ps), per):
        chunk = ps[s:s + per]
        sheet = Image.new("RGB", (tw * len(cols), th * len(chunk)), (40, 40, 40))
        d = ImageDraw.Draw(sheet)
        for i, r in enumerate(chunk):
            for c, (lab, v) in enumerate(cols):
                if lab == "proxy":
                    f = DS / "control1" / f"{r['id']}.png"
                elif lab == "hero":
                    f = DS / "control2" / f"{r['id']}.png"
                elif lab == "GT":
                    f = DS / "target" / f"{r['id']}.png"
                else:
                    f = out / "gen" / v / f"{r['id']}.png"
                if f.exists():
                    sheet.paste(Image.open(f).convert("RGB").resize((tw, th), Image.LANCZOS), (c * tw, i * th))
                txt = f"{lab}" if c else f"{r['object'][:26]} {r['view']} y{float(r['rel_yaw']):.0f} p{float(r['pitch']):.0f}"
                d.rectangle([c * tw, i * th, c * tw + (len(txt) * 6 + 6), i * th + 14], fill=(0, 0, 0))
                d.text((c * tw + 3, i * th + 2), txt, fill=(255, 255, 255))
        sheet.save(sd / f"sheet-{s // per:02d}.jpg", quality=88)
    print(f"{(len(ps) + per - 1) // per} sheets in {sd}")


if args.cmd == "gen":
    asyncio.run(gen())
elif args.cmd == "masks":
    asyncio.run(masks())
elif args.cmd == "score":
    score()
else:
    sheets()
