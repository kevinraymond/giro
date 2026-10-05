"""Fine-tune a trained projection-texture splat on the registered anchor views: the Gaussians are
the geometry the views refine, as a mesh is in photogrammetry. The anchors (warps baked in, each
its own pinhole camera) carry what the mesh lacks, a fork in front of a wheel; the attempt's
renders stay in, so the overall shape holds; the splat starts from the trained one (Brush's
init.ply) at a lower position learning rate, with a little room to grow.

    finetune_anchors.py ATTEMPT_DIR ANCHOR_DIR WORK OUT_NAME GPU --skip 08,10,... [--copies 2]
        [--iters 12000] [--lr-mean 5e-6] [--max-splats 330000]

ATTEMPT_DIR is a trained attempt made by project_texture.py in WORK (its train/final.ply, frames,
dataset/); WORK holds texture.json, anchor_cameras.json, warps.pt. Writes WORK/OUT_NAME, an
attempt trained to the end (crop, canonicalize, export by `giro stages --from crop`).
"""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from giro.stages.fallback import _qvec
from texture_common import Source, anchor_mask, bake, load_cameras

ROOT = Path(__file__).resolve().parents[2]
BRUSH = ROOT / "vendor" / "brush" / "target" / "release" / "brush-cli"
ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("anchor_dir", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("out_name")
ap.add_argument("gpu", type=int)
ap.add_argument("--skip", default="02,05")
ap.add_argument("--copies", type=int, default=2, help="how many times each anchor is trained on")
ap.add_argument("--iters", type=int, default=12000)
ap.add_argument("--lr-mean", type=float, default=5e-6, help="Brush's default is 2e-5")
ap.add_argument("--max-splats", type=int, default=330_000)
args = ap.parse_args()
base, adir, work = args.attempt.resolve(), args.anchor_dir.resolve(), args.work.resolve()
dev = torch.device(f"cuda:{args.gpu}")
out = work / args.out_name
if out.exists():
    shutil.rmtree(out)
out.mkdir()
for name in ("frames", "masks", "hero", "poses"):
    (out / name).symlink_to(base / name)
for name in ("cameras.json",):
    shutil.copy(base / name, out / name)
shutil.copytree(base / ".stages", out / ".stages")
for st in ("train", "crop", "canonicalize", "export"):
    (out / ".stages" / f"{st}.json").unlink(missing_ok=True)
metrics = json.loads((base / "metrics.json").read_text())
(out / "metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k in ("texture", "dataset")}, indent=1))

# The base's dataset (renders, hero copies, eroded masks), plus the anchors, plus the trained splat as init.ply.
ds, bds = out / "dataset", base / "dataset"
(ds / "sparse" / "0").mkdir(parents=True)
for kind in ("images", "masks"):
    for sub in (bds / kind).iterdir():
        (ds / kind).mkdir(parents=True, exist_ok=True)
        (ds / kind / sub.name).symlink_to(sub.resolve())
lines_c = [ln for ln in (bds / "sparse" / "0" / "cameras.txt").read_text().splitlines() if ln and not ln.startswith("#")]
lines_i = [ln for ln in (bds / "sparse" / "0" / "images.txt").read_text().splitlines() if not ln.startswith("#")]
shutil.copy(bds / "sparse" / "0" / "points3D.txt", ds / "sparse" / "0" / "points3D.txt")
next_cam = max(int(ln.split()[0]) for ln in lines_c) + 1
next_img = max(int(h.split()[0]) for h in lines_i[0::2]) + 1
cams = load_cameras(work / "anchor_cameras.json")
warps = torch.load(work / "warps.pt") if (work / "warps.pt").exists() else {}
skip = set(args.skip.split(","))
(ds / "images" / "zanc").mkdir()
(ds / "masks" / "zanc").mkdir()
added = 0
for n in sorted(cams):
    if n in skip:
        continue
    img = Image.open(adir / f"{n}.png").convert("RGB")
    pix, m = bake(Source(n, img, anchor_mask(adir, n), cams[n], 1.0, dev, warp=warps.get(n)))
    w, h = img.size
    f = cams[n].focal(w, h)
    lines_c.append(f"{next_cam} PINHOLE {w} {h} {f} {f} {w / 2} {h / 2}")
    rot, t = cams[n].world_to_camera()
    for k in range(args.copies):
        name = f"zanc/{n}_{k}.png"
        Image.fromarray(pix).save(ds / "images" / name)
        Image.fromarray(m.astype(np.uint8) * 255).save(ds / "masks" / name)
        lines_i += [f"{next_img} {' '.join(map(str, _qvec(np.asarray(rot))))} {' '.join(map(str, t))} {next_cam} {name}", ""]
        next_img += 1
        added += 1
    next_cam += 1
(ds / "sparse" / "0" / "cameras.txt").write_text("\n".join(lines_c) + "\n")
(ds / "sparse" / "0" / "images.txt").write_text("\n".join(lines_i) + "\n")
shutil.copy(base / "train" / "final.ply", ds / "init.ply")
print(f"{added} anchor views added; starting from {base.name}'s splat", flush=True)

# Brush from the trained splat: no eval split (every view trains), a gentler position learning rate.
(out / "train").mkdir()
env = os.environ | {"CUBECL_WGPU_DEFAULT_DEVICE": f"DiscreteGpu({args.gpu})", "RUST_LOG": "brush_cli=info", "NO_COLOR": "1"}
cmd = [str(BRUSH), str(ds), "--total-train-iters", str(args.iters), "--export-every", str(args.iters),
       "--export-path", str(out / "train"), "--export-name", "export_{iter}.ply", "--sh-degree", "3",
       "--max-resolution", "1920", "--max-splats", str(args.max_splats), "--render-mode", "mip",
       "--alpha-mode", "transparent", "--lr-mean", str(args.lr_mean), "--growth-stop-iter", str(int(args.iters * 0.6))]
with open(out / "train" / "brush.log", "w") as log:
    subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
shutil.copy(sorted((out / "train").glob("export_*.ply"))[-1], out / "train" / "final.ply")
params = []
for st in ("crop", "canonicalize", "export"):
    for k, v in json.loads((base / ".stages" / f"{st}.json").read_text())["params"].items():
        params += ["-p", f"{st}.{k}={json.dumps(v)}"]
subprocess.run(["uv", "run", "giro", "stages", str(out), "--from", "crop", "--gpu", str(args.gpu), *params], cwd=ROOT, check=True)
print(out, flush=True)
