"""Compare proxy models on giro's heroes without making videos: run only the proxy stage per
(subject, model), then score each proxy. Signals (no ground truth beyond the hero):

- iou: silhouette IoU at the fitted hero camera (what the proxy stage keeps the best seed by)
- ring_likeness: DINOv2 likeness to the hero of the proxy's color renders at 8 views around it
  (hero yaw + 45 k, at the hero's pitch), the same views the evaluator scores the final splat at

    uv run --group eval scripts/proxy_spike/proxy_bench.py OUT GPU MODEL[,MODEL] SRC_ATTEMPT...

Each SRC_ATTEMPT's hero/hero.png is copied to OUT/<model>/<subject>/; set GIRO_COMFY_DIR for a
ComfyUI that has the model's nodes (pixal3d, trellis2 need v0.34+). Writes OUT/summary.tsv and a
sheet per proxy.
"""
import json
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import Models  # noqa: E402

from giro import path as campath  # noqa: E402
from giro.stages import Ctx  # noqa: E402
from giro.stages.proxy import Points, Proxy, fit_sheet  # noqa: E402

out, gpu, models = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3].split(",")
sources = [Path(a) for a in sys.argv[4:]]
scorer = Models()
rows = []
for model in models:
    for src in sources:
        subject = src.name.replace("proxy-", "").split("-s")[0]
        a = (out / model / subject).resolve()
        (a / "hero").mkdir(parents=True, exist_ok=True)
        shutil.copy(src / "hero" / "hero.png", a / "hero" / "hero.png")
        ctx = Ctx(gpu=gpu, on_log=lambda st, m: print(f"  {m}", flush=True))
        t0 = time.monotonic()
        try:
            Proxy().execute(a, {"model": model, "n_seeds": 2}, ctx)
        except Exception as e:  # noqa: BLE001 - record and go on
            print(f"{model} {subject}: FAILED {e}", flush=True)
            rows.append([model, subject, "", "", "", f"failed: {e}"])
            continue
        seconds = time.monotonic() - t0
        proxy = json.loads((a / "proxy" / "proxy.json").read_text())
        hero = Image.open(a / "hero" / "hero.png").convert("RGB")
        mask = np.asarray(Image.open(a / "proxy" / "hero_mask.png")) > 127
        cam = campath.PathCamera.from_json(proxy["hero_camera"])
        points = Points(a / "proxy" / "proxy.ply")
        h = 384
        w = round(h * hero.width / hero.height)
        views = []
        for k in range(8):
            c = replace(cam, yaw=cam.yaw + 45 * k)
            sil = points.silhouette(c, w, h)
            rgb = np.where(sil[..., None], points.color(c, w, h), 0.5)
            views.append(Image.fromarray((rgb * 255).astype(np.uint8)))
        hm = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h))) > 127
        hero_gray = Image.fromarray((np.where(hm[..., None], np.asarray(hero.resize((w, h)), dtype=np.float32) / 255, 0.5) * 255).astype(np.uint8))
        e = scorer.embed([hero_gray] + views)
        likeness = [round(float(x), 4) for x in e[1:] @ e[0]]
        fit_sheet(points, cam, hero, mask).save(out / f"{model}-{subject}-fit.jpg", quality=88)
        (a / "bench.json").write_text(json.dumps({"iou": proxy["iou"], "likeness": likeness, "seconds": round(seconds, 1)}, indent=1))
        rows.append([model, subject, proxy["iou"], round(float(np.mean(likeness[1:])), 4), round(seconds, 1), ""])
        print(model, subject, rows[-1][2:], flush=True)
with open(out / "summary.tsv", "a") as f:
    for r in rows:
        f.write("\t".join(map(str, r)) + "\n")
