"""Orbit-video parameter sweep: gate pass rate and cost per (length, steps).

Every cell runs the same seeds with no rerolls, so cells differ only in the
parameters. All cells share one GPU pool: videos run one at a time on the video
GPU while passing attempts train on the other.

    uv run scripts/sweep_orbit.py data/samples/adventurer_7.png --seeds 1,2,3,4 \\
        --cells 124x20,192x20,158x20,124x8,192x8

A cell is LENGTHxSTEPS, optionally followed by orbit params: 73x28:width=768:height=768:lora=h3/x.safetensors.
prompt=frozen stands for the H3 360 orbit LoRA's prompt, prompt=wan_orbit for the Wan orbit LoRA's
(workflows.FROZEN_ORBIT_PROMPT, WAN_ORBIT_PROMPT); model=wan22 runs Wan 2.2 I2V instead of MiniMax H3.

Writes data/sweeps/<stamp>/: one job per cell, summary.txt, results.json and
sheets/ (a 16-frame contact sheet per attempt, to check the gate by eye).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

from giro import workflows
from giro.job import Job, JobSpec, run_jobs

ROOT = Path(__file__).resolve().parents[1]


def contact_sheet(frames_raw: Path, dest: Path) -> None:
    n = len(list(frames_raw.glob("*.png")))
    every = max(1, math.ceil(n / 16))
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-pattern_type", "glob", "-i", str(frames_raw / "*.png"),
        "-vf", f"select='not(mod(n\\,{every}))',scale=192:-1,tile=8x2", "-frames:v", "1", str(dest),
    ], check=True)


def summarize(jobs: list[Job]) -> tuple[str, list[dict]]:
    rows, lines = [], []
    lines.append(f"{'cell':<24} {'pass':>5} {'video s':>8} {'views':>6} {'sweep':>6} {'PSNR':>6}  rejections")
    for job in jobs:
        o = job.spec.orbit
        cell = job.path.name.split("-", 2)[-1]  # the cell's tag, without the job's timestamp
        video_s, views, sweeps, psnrs, reasons = [], [], [], [], []
        for a in job.attempts:
            metrics_path = job.attempt_dir(a.seed) / "metrics.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            if "seconds" in metrics.get("orbit_video", {}):
                video_s.append(metrics["orbit_video"]["seconds"])
            if a.status == "passed":
                views.append(a.gate.get("n_views", 0))
                sweeps.append(a.gate.get("azimuth_coverage", 0))
                psnrs.append(a.eval_psnr or 0)
            elif a.status in ("rejected", "error"):
                reasons.append(f"s{a.seed} {a.status}: {a.reason}")
            rows.append({"cell": cell, "length": o["length"], "steps": o["steps"], "seed": a.seed,
                         "status": a.status, "reason": a.reason, "eval_psnr": a.eval_psnr, "gate": a.gate,
                         "video_seconds": metrics.get("orbit_video", {}).get("seconds"),
                         "dedup": metrics.get("dedup", {}), "gpus": a.gpus})

        def mean(xs: list[float]) -> str:
            return f"{statistics.mean(xs):.0f}" if xs else "-"

        passed = sum(a.status == "passed" for a in job.attempts)
        lines.append(f"{cell:<24} {passed:>2}/{len(job.attempts):<2} {mean(video_s):>8} {mean(views):>6} "
                     f"{mean(sweeps):>6} {(f'{statistics.mean(psnrs):.1f}' if psnrs else '-'):>6}")
        lines += [f"{'':<26}{r}" for r in reasons]
    return "\n".join(lines), rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path)
    p.add_argument("--seeds", default="1,2,3,4")
    p.add_argument("--cells", default="124x20,192x20,158x20,124x8,192x8", help="LENGTHxSTEPS,...")
    p.add_argument("--video-gpus", default="1")
    p.add_argument("--post-gpus", default="0,1")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "sweeps" / time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--resume", type=Path, metavar="SWEEP_DIR", help="continue an interrupted sweep")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    video_gpus = [int(g) for g in args.video_gpus.split(",")]
    post_gpus = [int(g) for g in args.post_gpus.split(",")]
    if args.resume:
        args.out = args.resume
    args.out.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out / "sweep.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    jobs = [Job.load(d) for d in sorted(args.out.iterdir()) if (d / "job.json").exists()] if args.resume else []
    for cell in [] if args.resume else args.cells.split(","):
        size, *extra = cell.split(":")
        length, steps = (int(x) for x in size.split("x"))
        orbit: dict[str, object] = {"steps": steps}
        for pair in extra:
            k, _, raw = pair.partition("=")
            try:
                orbit[k] = json.loads(raw)
            except json.JSONDecodeError:
                orbit[k] = raw
        orbit["length"] = workflows.snap_length(length, str(orbit.get("model") or "h3"))
        prompts = {"frozen": workflows.FROZEN_ORBIT_PROMPT, "wan_orbit": workflows.WAN_ORBIT_PROMPT}
        if orbit.get("prompt") in prompts:
            orbit["prompt"] = prompts[str(orbit["prompt"])]
        spec = JobSpec(image=str(args.image.resolve()), want=len(seeds), max_attempts=len(seeds), seeds=seeds,
                       orbit=orbit, video_gpus=video_gpus, post_gpus=post_gpus)
        tag = (f"-{orbit['model']}" if orbit.get("model") else "") + ("-lora" if orbit.get("lora") else "")
        size_tag = f"-{orbit['width']}x{orbit['height']}" if "width" in orbit else ""
        jobs.append(Job.create(args.out, spec, f"L{length}-S{steps}{size_tag}{tag}"))
    log(f"sweep {args.out}: {len(jobs)} cells x {len(seeds)} seeds")
    t0 = time.monotonic()
    jobs = asyncio.run(run_jobs(jobs, log, sorted(set(video_gpus) | set(post_gpus))))
    log(f"sweep finished in {(time.monotonic() - t0) / 60:.1f} min")

    sheets = args.out / "sheets"
    sheets.mkdir(exist_ok=True)
    for job in jobs:
        for a in job.attempts:
            frames_raw = job.attempt_dir(a.seed) / "frames_raw"
            if any(frames_raw.glob("*.png")):
                contact_sheet(frames_raw, sheets / f"{job.path.name.split('-', 2)[-1]}_s{a.seed}_{a.status}.png")
    table, rows = summarize(jobs)
    (args.out / "summary.txt").write_text(table + "\n")
    (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
