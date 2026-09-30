"""Review sheets for finished jobs: is the exported splat clean enough to use as is?

    uv run scripts/m3_report.py JOB_DIR [JOB_DIR ...] [-o data/m3/report]

For every passing attempt: the hero image next to the cropped splat rendered
from the hero camera, then the canonical splat from 8 directions around it,
4 from above and 4 from below, on a mid-gray background so both dark
floaters and pale halos show. Writes <out>/<job>-s<seed>.jpg and prints a
summary line per attempt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from giro import render
from giro.job import Job
from giro.stages.crop import load_views


def review(attempt: Path, out: Path) -> str:
    metrics = json.loads((attempt / "metrics.json").read_text())
    height = metrics["canonicalize"]["height_m"]
    views = load_views(attempt / "poses" / "colmap" / "model_txt")
    hero = next(v for v in views if v.name.startswith("hero/"))
    size = (384, 512)
    top = [
        Image.open(attempt / hero.name).convert("RGB").resize(size),
        render.render(attempt / "crop" / "cropped.ply", [hero.camera()], size, background=(0.5, 0.5, 0.5))[0],
    ]
    width, depth = json.loads((attempt / "canonical" / "transform.json").read_text())["footprint_m"]
    center = (0.0, height / 2, 0.0)
    dist = render.framing_distance((width, height, depth))
    cams = render.orbit_cameras(center, dist, 8, elevation_deg=10)
    cams += render.orbit_cameras(center, dist, 4, elevation_deg=55)
    cams += render.orbit_cameras(center, dist, 4, elevation_deg=-30)
    ring = render.render(attempt / "canonical" / "splat.ply", cams, size, background=(0.5, 0.5, 0.5))
    sheet = render.sheet(top + ring, cols=6, width=256)
    exp, crop, train = metrics["export"], metrics["crop"], metrics["train"]
    line = (f"{exp['n_gaussians']:,} Gaussians (kept {crop['kept_frac']:.0%} of {crop['n_input']:,}), "
            f"eval PSNR {train['eval_psnr']:.1f}, {height} m, spz {exp['splat_spz_mb']} MB")
    ImageDraw.Draw(sheet).text((8, 8), line, fill=(255, 255, 0))
    sheet.save(out, quality=90)
    return line


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jobs", nargs="+", type=Path)
    p.add_argument("-o", "--out", type=Path, default=Path("data/m3/report"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.jobs:
        job = Job.load(path)
        for a in job.attempts:
            if a.status != "passed":
                print(f"{path.name} s{a.seed}: {a.status} {a.reason}")
                continue
            dest = args.out / f"{path.name}-s{a.seed}.jpg"
            best = " (best)" if job.ranking and job.ranking[0] == a.seed else ""
            print(f"{path.name} s{a.seed}{best}: {review(job.attempt_dir(a.seed), dest)} -> {dest}")


if __name__ == "__main__":
    main()
