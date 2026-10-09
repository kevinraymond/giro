"""Per subject: giro's canonical splat (top row) vs TripoSplat (bottom row), same six views."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import ImageDraw
from giro import render, splat

OUT = Path("data/sweeps/20261003-triposplat"); ST = "vendor/splat-transform/node_modules/.bin/splat-transform"
SIZE = (384, 512); BG = (0.5, 0.5, 0.5)

def fill(ply):
    t = subprocess.run([ST, "-q", str(ply), "--stats", "json", "null"], capture_output=True, text=True).stdout
    return json.loads(t[t.index("{"):])["stats"][0]["fillRatio"]

def views(center, dist, front_deg=0.0):
    c = np.asarray(center, float); cams = []
    for az, el in ((0, 5), (90, 5), (180, 5), (270, 5), (30, 65), (30, -35)):
        a, e = np.radians(az + front_deg), np.radians(el)
        cams.append(render.Camera(c + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)]), c, (0, 1, 0), 40.0, viewer=True))
    return cams

for pair in sys.argv[1:]:
    name, attempt = pair.split("="); a = Path(attempt)
    m = json.loads((a / "metrics.json").read_text()); h = m["canonicalize"]["height_m"]
    w, d = json.loads((a / "canonical" / "transform.json").read_text())["footprint_m"]
    gply = a / "canonical" / "splat.ply"
    g = render.render(gply, views((0, h / 2, 0), render.framing_distance((w, h, d))), SIZE, background=BG)
    tply = OUT / f"{name}.ply"
    p = splat.positions(splat.read_ply(tply)); lo, hi = p.min(0), p.max(0)
    # file frame -> viewer frame is a 180 degree turn about Z (x, y negated)
    center = (-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2); size = hi - lo
    dist = render.framing_distance((size[0], size[1], size[2]))
    ref = np.asarray(g[0].resize((96, 128)), dtype=np.float32)
    best = min(range(0, 360, 15), key=lambda az: float(np.mean((np.asarray(render.render(tply, views(center, dist, az)[:1], (96, 128), background=BG)[0], dtype=np.float32) - ref) ** 2)))
    t = render.render(tply, views(center, dist, best), SIZE, background=BG)
    sheet = render.sheet(g + t, cols=6, width=320)
    dr = ImageDraw.Draw(sheet)
    ng, nt = len(splat.read_ply(gply)), len(p)
    line_g = f"giro: {ng:,} Gaussians, fill {fill(gply):.0f}"; line_t = f"TripoSplat: {nt:,} Gaussians, fill {fill(tply):.0f}, front at {best} deg"
    dr.text((8, 8), line_g, fill=(255, 255, 0)); dr.text((8, sheet.height // 2 + 8), line_t, fill=(255, 255, 0))
    sheet.save(OUT / f"compare-{name}.jpg", quality=90)
    n80 = OUT / "n80k" / f"{name}.ply"
    print(name, "|", line_g, "|", line_t, "| tripo 80K fill", round(fill(n80), 1) if n80.exists() else "-", flush=True)
