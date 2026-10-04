"""Wrap finished sweep attempts into library jobs, so they can be looked at in giro's UI and in
the headset. The attempts are copied (the UI may write to them), never moved.

    make_review_jobs.py NAME SOURCE_IMAGE HEIGHT_M SEED=ATTEMPT_DIR [SEED=ATTEMPT_DIR ...]

The job lands in data/jobs/<stamp>-NAME with each attempt as attempts/<SEED>, ranked like a real
job. Restart `giro serve` afterwards: it reads the jobs at startup.
"""
import json
import shutil
import sys
from pathlib import Path

from giro.job import Attempt, Job, JobSpec
from giro.stages import poses

ROOT = Path(__file__).resolve().parents[2] / "data" / "jobs"


def attempt_record(seed: int, path: Path) -> Attempt:
    metrics = json.loads((path / "metrics.json").read_text())
    stage_params = {}
    for f in (path / ".stages").glob("*.json"):
        if f.stem != "orbit_video":
            params = json.loads(f.read_text()).get("params", {})
            if f.stem == "proxy" and params.get("model"):
                stage_params["proxy"] = {"model": params["model"]}
    gpus = {s: v["gpu"] for s, v in metrics.items() if isinstance(v, dict) and "gpu" in v}
    return Attempt(
        seed=seed, status="passed",
        eval_psnr=metrics.get("train", {}).get("eval_psnr"),
        gate={k: v for k, v in metrics.get("gate", {}).items() if k != "reasons"},
        gaussians=metrics.get("export", {}).get("n_gaussians"),
        poses=poses.active_source(path), params=stage_params, gpus=gpus,
    )


def main() -> None:
    name, image, height = sys.argv[1], Path(sys.argv[2]), float(sys.argv[3])
    pairs = [(int(a.split("=", 1)[0]), Path(a.split("=", 1)[1])) for a in sys.argv[4:]]
    orbit = json.loads((pairs[0][1] / ".stages" / "orbit_video.json").read_text())["params"]
    orbit = {k: v for k, v in orbit.items() if k != "seed"}
    spec = JobSpec(image=str(image.resolve()), want=len(pairs), max_attempts=len(pairs), seeds=[s for s, _ in pairs],
                   orbit=orbit, params={"canonicalize": {"height_m": height}}, video_gpus=[1], post_gpus=[1])
    job = Job.create(ROOT, spec, name)
    for seed, src in pairs:
        shutil.copytree(src, job.attempt_dir(seed), symlinks=True)
        job.attempts.append(attempt_record(seed, job.attempt_dir(seed)))
    job.rank()
    job.status = "done"
    job.save()
    print(job.path, "ranked", job.ranking)


if __name__ == "__main__":
    main()
