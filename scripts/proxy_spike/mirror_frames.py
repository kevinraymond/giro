"""Frame sheet for the symmetry-prior runs (board #3822): per attempt, the orbit's frames at the given
indices (frames/, after SeedVR2) and, first, the mirrored hero (proxy_render/mirror_hero.png, from the
pinned run) with the frame at the mirrored camera beside it, so the pin's hold can be judged by eye.

    mirror_frames.py OUT.jpg INDICES LABEL ATTEMPT [LABEL ATTEMPT ...]   (INDICES: 0,8,16,24,40,60,80)
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

out, idx = Path(sys.argv[1]), [int(i) for i in sys.argv[2].split(",")]
pairs = list(zip(sys.argv[3::2], map(Path, sys.argv[4::2])))
th = 256
rows = []
ref = next((a / "proxy_render" / "mirror_hero.png" for _, a in pairs if (a / "proxy_render" / "mirror_hero.png").exists()), None)
for label, a in pairs:
    frames = sorted((a / "frames").glob("*.png")) or sorted((a / "frames_raw").glob("*.png"))
    cams = json.loads((a / "cameras.json").read_text())
    tiles = [Image.open(ref).convert("RGB") if ref else Image.new("RGB", (3, 4))] + [Image.open(frames[i]).convert("RGB") for i in idx if i < len(frames)]
    tw = round(th * tiles[1].width / tiles[1].height)
    row = Image.new("RGB", (tw * len(tiles), th + 16), (40, 40, 40))
    d = ImageDraw.Draw(row)
    for k, t in enumerate(tiles):
        row.paste(t.resize((tw, th), Image.LANCZOS), (k * tw, 16))
        if k:
            c = cams["frames"][idx[k - 1]]
            d.text((k * tw + 4, 2), f"#{idx[k - 1]} yaw {c['yaw'] - cams['hero']['yaw']:.0f} pitch {c['pitch']:.0f}", fill=(200, 200, 200))
    d.text((4, 2), f"{label}" + ("  (mirrored hero)" if ref else ""), fill=(255, 255, 0))
    rows.append(row)
sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), (40, 40, 40))
y = 0
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height
sheet.save(out, quality=88)
