"""Side-by-side review sheets of several attempts' canonical splats, one row per attempt, every row from the
same cameras (framed on the first attempt's canonical box): an orbit sheet (8 around at 10 deg, 2 from 45 deg
above) and a close-up sheet (4 eye-level views from half the distance).

    compare_sheet.py OUT_PREFIX GPU LABEL ATTEMPT [LABEL ATTEMPT ...]

Writes OUT_PREFIX-compare.jpg and OUT_PREFIX-closeup.jpg.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from giro import render

prefix, gpu = Path(sys.argv[1]), int(sys.argv[2])
pairs = list(zip(sys.argv[3::2], map(Path, sys.argv[4::2])))
first = pairs[0][1]
h = json.loads((first / "metrics.json").read_text())["canonicalize"]["height_m"]
w, d = json.loads((first / "canonical" / "transform.json").read_text())["footprint_m"]
center, dist = (0, h / 2, 0), render.framing_distance((w, h, d))
size = (512, 512) if w > h else (384, 512)
orbit = render.orbit_cameras(center, dist, 8, elevation_deg=10) + render.orbit_cameras(center, dist, 2, elevation_deg=45)
close = render.orbit_cameras(center, dist * 0.55, 8, elevation_deg=8)[::2]


def sheet(cams: list, name: str, width: int) -> None:
    rows = []
    for label, a in pairs:
        imgs = render.render(a / "canonical" / "splat.ply", cams, size, background=(0.5, 0.5, 0.5), gpu=gpu)
        row = render.sheet(imgs, cols=len(cams), width=width)
        ImageDraw.Draw(row).text((6, 6), label, fill=(255, 255, 0))
        rows.append(row)
    out = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    out.save(prefix.parent / f"{prefix.name}-{name}.jpg", quality=88)
    print(prefix.parent / f"{prefix.name}-{name}.jpg", flush=True)


sheet(orbit, "compare", 220)
sheet(close, "closeup", 380)
