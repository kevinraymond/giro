"""Per sweep cell: Gaussians kept, splat-transform fill ratio (~overdraw layers), and PSNR of the
CROPPED splat on the held-out views against the masked frames (one evaluator for every variant).
Also decimates each subject's base crop to the capped cells' counts, to compare 'train to the
budget' against 'train big, decimate'. Caches per-variant results in analysis.json."""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from giro import render, splat
from giro.stages.crop import load_views

ROOT = Path(sys.argv[1])
ST = Path("vendor/splat-transform/node_modules/.bin/splat-transform")
cache_path = ROOT / "analysis.json"
cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

def fill(ply):
    t = subprocess.run([str(ST), "-q", str(ply), "--stats", "json", "null"], capture_output=True, text=True).stdout
    return json.loads(t[t.index("{"):])["stats"][0]["fillRatio"]

def psnr(attempt, ply):
    names = sorted(str(p.relative_to(attempt / "dataset" / "images")) for p in (attempt / "dataset" / "images").rglob("*.png"))
    views = {v.name: v for v in load_views(attempt / "poses" / "colmap" / "model_txt")}
    vals = []
    for name in names[::8]:
        v = views[name]
        gt = np.asarray(Image.open(attempt / "dataset" / "images" / name).convert("RGB"), dtype=np.float32) / 255
        m = (np.asarray(Image.open(attempt / "dataset" / "masks" / name).convert("L")) > 127)[..., None]
        h, w = gt.shape[:2]
        out = np.asarray(render.render(ply, [v.camera()], (w, h))[0], dtype=np.float32) / 255
        mse = float(np.mean((out - gt * m) ** 2))
        vals.append(10 * np.log10(1 / max(mse, 1e-10)))
    return round(float(np.mean(vals)), 2)

def measure(key, attempt, ply):
    if key not in cache:
        cache[key] = {"n": len(splat.read_ply(ply)), "fill": round(fill(ply), 1), "psnr": psnr(attempt, ply)}
        cache_path.write_text(json.dumps(cache, indent=1))
        print(key, cache[key], flush=True)
    return cache[key]

for subject in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name != "sheets"):
    cells = sorted(c for c in subject.iterdir() if (c / "export" / "splat.ply").exists())
    for c in cells:
        measure(f"{subject.name}/{c.name}", c, c / "crop" / "cropped.ply")
    base = subject / "base"
    if not (base / "crop" / "cropped.ply").exists():
        continue
    for c in cells:
        if not c.name.startswith("cap"):
            continue
        n = cache[f"{subject.name}/{c.name}"]["n"]
        for flag, tag in (("-d", "dec"), ("--decimate-adaptive", "adec")):
            key = f"{subject.name}/base-{tag}-to-{c.name}"
            if key in cache:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                small = Path(tmp) / "small.ply"
                subprocess.run([str(ST), "-q", "-w", str(base / "crop" / "cropped.ply"), str(small), flag, str(n)], check=True)
                measure(key, base, small)
print(f"{'variant':<34} {'kept':>8} {'fill':>6} {'crop PSNR':>9}")
for k, v in sorted(cache.items()):
    print(f"{k:<34} {v['n']:>8,} {v['fill']:>6} {v['psnr']:>9}")
