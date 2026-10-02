"""poses/colmap: SIFT features, exhaustive matching, incremental or global mapping.

The frames share one SIMPLE_PINHOLE camera and the hero gets its own, via
single_camera_per_folder over images/frames and images/hero.

poses/colmap/model (and model_txt) is the attempt's active camera model, the one the
gate, training, crop and canonicalize read. It links to COLMAP's own reconstruction
(also linked as model_colmap), or, after the job runner switched to the pose fallback,
to poses/fallback/model: that one is a COLMAP model too, bundle-adjusted from
feedforward poses (stages/fallback.py).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from giro import gpu
from giro.stages.base import Ctx, Rejected, Stage, StageFailed

COLMAP = "colmap"
SIFT_VRAM_MB = 4_000
CRASH_RETRIES = 2


def _link_dir(src: Path, dest: Path) -> int:
    """Real directory of per-file symlinks (COLMAP may not follow a symlinked dir)."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*.png")):
        (dest / f.name).symlink_to(f.resolve())
        n += 1
    return n


def read_images_txt(path: Path) -> dict[str, dict[str, Any]]:
    """COLMAP images.txt -> {name: {id, qvec, tvec, camera_id, n_points}}."""
    images: dict[str, dict[str, Any]] = {}
    # Two lines per image; the second (its 2D points) may be empty, so only comments are dropped.
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    for header, points in zip(lines[0::2], lines[1::2]):
        f = header.split()
        n_points = sum(1 for pid in points.split()[2::3] if pid != "-1")
        images[f[9]] = {
            "id": int(f[0]), "qvec": [float(x) for x in f[1:5]], "tvec": [float(x) for x in f[5:8]],
            "camera_id": int(f[8]), "n_points": n_points,
        }
    return images


SOURCES = {"colmap": "poses_colmap", "fallback": "poses_fallback"}  # pose source -> the stage that made it


def active_source(attempt: Path) -> str:
    """Which reconstruction poses/colmap/model links to: "colmap" or "fallback"."""
    model = attempt / "poses" / "colmap" / "model"
    fallback = attempt / "poses" / "fallback"
    return "fallback" if model.exists() and model.resolve().is_relative_to(fallback.resolve()) else "colmap"


def activate(attempt: Path, source: str) -> None:
    """Point poses/colmap/model(_txt) at COLMAP's own reconstruction or at the fallback's."""
    work = attempt / "poses" / "colmap"
    for link in ("model", "model_txt"):  # attempts from before the fallback link COLMAP's model only here
        own = work / link.replace("model", "model_colmap")
        if not own.is_symlink() and (work / link).is_symlink() and active_source(attempt) == "colmap":
            own.symlink_to((work / link).readlink())
    targets = {"colmap": ("model_colmap", "model_colmap_txt"),
               "fallback": ("../fallback/model", "../fallback/model_txt")}[source]
    for link, target in zip(("model", "model_txt"), targets):
        if not (work / target).exists():
            raise FileNotFoundError(f"poses/colmap/{target} does not exist")
        (work / link).unlink(missing_ok=True)
        (work / link).symlink_to(target)


def read_cameras_txt(path: Path) -> dict[int, dict[str, Any]]:
    """COLMAP cameras.txt -> {camera_id: {model, width, height, fx, fy, cx, cy}} (pinhole models only)."""
    cameras: dict[int, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split()
        model, w, h, p = f[1], int(f[2]), int(f[3]), [float(x) for x in f[4:]]
        if model == "SIMPLE_PINHOLE":
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        elif model == "PINHOLE":
            fx, fy, cx, cy = p[:4]
        else:
            raise ValueError(f"camera model {model} is not supported")
        cameras[int(f[0])] = {"model": model, "width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return cameras


class ColmapPoses(Stage):
    name = "poses_colmap"
    # min_reg_rate stops the pipeline before training on a failed reconstruction;
    # the M2 gate applies the real thresholds.
    # use_masks: features only on the subject (masks/ from the masks stage). A video that
    # turns the subject in front of a still background then reads as an orbit, and repeated
    # background patterns (a row of windows) can no longer pull cameras out of place.
    defaults = {"mapper": "incremental", "max_features": 8192, "min_reg_rate": 0.5, "use_masks": True}
    inputs = ("frames", "hero")
    outputs = ("poses/colmap/model",)
    gpu_mb = SIFT_VRAM_MB

    def inputs_for(self, params: dict[str, Any]) -> tuple[str, ...]:
        return self.inputs + (("masks/frames", "masks/hero") if params["use_masks"] else ())

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        work = attempt / "poses" / "colmap"
        if work.exists():
            shutil.rmtree(work)
        images = work / "images"
        n_frames = _link_dir(attempt / "frames", images / "frames")
        n_hero = _link_dir(attempt / "hero", images / "hero")
        n_inputs = n_frames + n_hero
        db = work / "database.db"
        sparse = work / "sparse"
        sparse.mkdir(parents=True)
        log_path = work / "colmap.log"

        device = ctx.gpu if ctx.gpu is not None else gpu.pick(SIFT_VRAM_MB)
        env = gpu.cuda_env(device)
        ctx.log(f"GPU {device}: {n_frames} frames + {n_hero} hero")

        def colmap(step: str, frac: float, *args: str) -> str:
            ctx.check_cancelled()
            ctx.progress(frac, step)
            for tries in range(CRASH_RETRIES + 1):
                if tries and "--output_path" in args:  # start the retry from an empty output
                    out = Path(args[args.index("--output_path") + 1])
                    shutil.rmtree(out)
                    out.mkdir()
                with open(log_path, "a") as log:
                    log.write(f"\n==== colmap {' '.join(args)}\n")
                r = subprocess.run([COLMAP, *args, "--log_color", "0"], env=env, capture_output=True, text=True)
                with open(log_path, "a") as log:
                    log.write(r.stdout + r.stderr)
                # The mapper has segfaulted at random mid-reconstruction and then succeeded
                # on the same input, so a crash (not an ordinary failure) is retried.
                crashed = r.returncode < 0 or "*** Aborted at" in r.stderr + r.stdout
                if not crashed or tries == CRASH_RETRIES:
                    break
                ctx.log(f"colmap {args[0]} crashed (exit {r.returncode}); retrying")
            if r.returncode != 0:
                tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                raise StageFailed(f"colmap {args[0]} failed: {' | '.join(tail)}")
            return r.stdout + r.stderr

        mask_args: list[str] = []
        if params["use_masks"]:
            # COLMAP looks up <image_name>.png under mask_path; black pixels get no features.
            mask_dir = work / "masks"
            for group in ("frames", "hero"):
                (mask_dir / group).mkdir(parents=True)
                for img in sorted((images / group).glob("*.png")):
                    src = attempt / "masks" / group / img.name
                    if not src.exists():
                        raise StageFailed(f"use_masks is on but masks/{group}/{img.name} is missing; run the masks stage")
                    (mask_dir / group / f"{img.name}.png").symlink_to(src.resolve())
            mask_args = ["--ImageReader.mask_path", str(mask_dir)]
        colmap("extracting features", 0.0, "feature_extractor", *mask_args,
               "--database_path", str(db), "--image_path", str(images),
               "--ImageReader.camera_model", "SIMPLE_PINHOLE",
               "--ImageReader.single_camera_per_folder", "1",
               "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0",
               "--SiftExtraction.max_num_features", str(params["max_features"]))
        colmap("matching all pairs", 0.25, "exhaustive_matcher",
               "--database_path", str(db),
               "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0")
        not_orbiting = (
            "the video likely does not orbit the subject (or the subject changes shape between frames)"
            if params["use_masks"] else
            "the video likely does not orbit (for example the subject turns while the background stays still)"
        )
        mapper = {"incremental": "mapper", "global": "global_mapper"}[params["mapper"]]
        try:
            colmap(f"reconstructing ({params['mapper']})", 0.55, mapper,
                   "--database_path", str(db), "--image_path", str(images), "--output_path", str(sparse))
        except StageFailed:
            if "Failed to create any sparse model" not in log_path.read_text():
                raise
            ctx.metric("n_registered", 0)
            ctx.metric("reg_rate", 0.0)
            ctx.metric("hero_registered", False)
            raise Rejected(f"no camera poses could be recovered; {not_orbiting}") from None

        # The mapper may split the scene into several models; keep the largest.
        best, best_n = None, -1
        for model in sorted(p for p in sparse.iterdir() if p.is_dir()):
            txt = model.parent / f"{model.name}_txt"
            txt.mkdir()
            colmap("reading model", 0.9, "model_converter", "--input_path", str(model),
                   "--output_path", str(txt), "--output_type", "TXT")
            n = len(read_images_txt(txt / "images.txt"))
            if n > best_n:
                best, best_n = model, n
        if best is None:
            raise Rejected("COLMAP produced no model: the frames could not be registered")
        (work / "model_colmap").symlink_to(best.relative_to(work))
        (work / "model_colmap_txt").symlink_to((best.parent / f"{best.name}_txt").relative_to(work))
        activate(attempt, "colmap")

        stats = colmap("analyzing model", 0.95, "model_analyzer", "--path", str(best))
        registered = read_images_txt(work / "model_txt" / "images.txt")
        reproj = re.search(r"Mean reprojection error:\s*([\d.]+)", stats)
        ctx.metric("n_models", len(list(sparse.glob("*_txt"))))
        ctx.metric("n_registered", len(registered))
        ctx.metric("reg_rate", round(len(registered) / n_inputs, 4))
        ctx.metric("reproj_err", float(reproj.group(1)) if reproj else None)
        ctx.metric("hero_registered", any(name.startswith("hero/") for name in registered))
        ctx.progress(1.0, f"{len(registered)}/{n_inputs} images registered")
        if len(registered) / n_inputs < params["min_reg_rate"]:
            raise Rejected(f"only {len(registered)} of {n_inputs} images got a camera pose; {not_orbiting}")
