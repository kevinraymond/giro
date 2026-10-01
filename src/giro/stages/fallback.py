"""poses_fallback: camera poses from Depth Anything 3 when COLMAP's fail the gate.

A feedforward model poses every frame, including fast, smeared ones COLMAP cannot
register, but its poses are too loose to train on (a few dB lower). So they are only
the start: COLMAP triangulates the COLMAP stage's own matches from them and bundle-adjusts
twice, which brings the cameras to COLMAP's precision. Frames that too few triangulated
points see are left where bundle adjustment cannot place them; those keep their DA3 pose,
moved into the refined model's frame by a similarity fitted on the other cameras (the
hybrid). On the gated attempts this is as good as COLMAP where COLMAP works, and on
L124-S8/1, where COLMAP drops three smeared frames and leaves a 57-degree jump, it places
all of them (docs/FINDINGS.md, "Pose fallback").

DA3-BASE because it is Apache-licensed and refines as well as the larger (non-commercial)
checkpoints. It does not help turntables (a subject spinning in a still room): every
feedforward model reads those as a camera standing still, and masked COLMAP handles them.

Runs only when the job runner asks (after a gate rejection), so it is not in PIPELINE.
Its model lands in poses/fallback/; the runner makes it the active one (poses.activate).
"""

from __future__ import annotations

import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from giro import gpu
from giro.stages.base import Ctx, Rejected, Stage, StageFailed
from giro.stages.poses import read_images_txt

ROOT = Path(__file__).resolve().parents[3]
DA3_PYTHON = ROOT / "vendor" / "depth-anything-3" / ".venv" / "bin" / "python"
WORKER = ROOT / "src" / "giro" / "workers" / "da3_poses.py"
DA3_VRAM_MB = 9_000  # DA3-BASE on ~120 images at 378x504: 8 GB peak (any shape: long side 504)


def model_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """DA3 input size for frames of this shape: both sides multiples of 14 (DA3's patch, so the
    model resizes nothing further), the long side within 56 px of the one given, and the aspect
    ratio as close as that allows (768x1024 -> 378x504, 1344x768 -> 448x252)."""
    best = None
    for long_ in range(long_side - 56, long_side + 57, 14):
        short = max(14, round(long_ * min(width, height) / max(width, height) / 14) * 14)
        w, h = (long_, short) if width >= height else (short, long_)
        err = abs((w / h) / (width / height) - 1)
        if best is None or err < best[0] - 1e-9 or (abs(err - best[0]) < 1e-9 and abs(long_ - long_side) < best[2]):
            best = (err, (w, h), abs(long_ - long_side))
    return best[1]


def _centers(poses: dict[str, tuple[np.ndarray, np.ndarray]], names: list[str]) -> np.ndarray:
    return np.array([-poses[n][0].T @ poses[n][1] for n in names])


def similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t minimizing |s R src + t - dst| (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, d, vt = np.linalg.svd(xd.T @ xs / len(src))
    flip = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        flip[2, 2] = -1
    r = u @ flip @ vt
    s = float(np.trace(np.diag(d) @ flip) / xs.var(0).sum())
    return s, r, mu_d - s * r @ mu_s


def hybrid(ff: dict[str, tuple[np.ndarray, np.ndarray]], refined: dict[str, tuple[np.ndarray, np.ndarray]],
           constrained: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World-to-camera (R, t) for every refined image: the refined pose where bundle adjustment
    constrained it, else the feedforward pose moved into the refined frame."""
    s, r, t = similarity(_centers(ff, constrained), _centers(refined, constrained))
    out = {}
    for name in refined:
        if name in constrained:
            out[name] = refined[name]
            continue
        r_ff, t_ff = ff[name]
        center = s * r @ (-r_ff.T @ t_ff) + t
        rot = r_ff @ r.T
        out[name] = (rot, -rot @ center)
    return out


def _pose(im: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = im["qvec"]
    return Rotation.from_quat([x, y, z, w]).as_matrix(), np.asarray(im["tvec"], dtype=float)


def _qvec(r: np.ndarray) -> list[float]:
    x, y, z, w = Rotation.from_matrix(r).as_quat()
    return [w, x, y, z] if w >= 0 else [-w, -x, -y, -z]


class PoseFallback(Stage):
    name = "poses_fallback"
    defaults = {
        "enabled": True,
        "model": "depth-anything/DA3-BASE",
        "long_side": 504,       # model input: the frames resized to this long side (sides multiples of 14)
        "min_points": 15,       # an image fewer triangulated points see keeps its feedforward pose
        "max_unrefined": 0.1,   # ... but at most this share of the images
    }
    inputs = ("frames", "hero", "masks/frames", "masks/hero", "poses/colmap/database.db")
    outputs = ("poses/fallback/model", "poses/fallback/model_txt")
    gpu_mb = DA3_VRAM_MB

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        if not DA3_PYTHON.exists():
            raise StageFailed("Depth Anything 3 is not installed; run scripts/setup_da3.sh")
        colmap_dir = attempt / "poses" / "colmap"
        if not (colmap_dir / "database.db").exists():
            raise StageFailed("the COLMAP stage left no feature database to refine against")
        work = attempt / "poses" / "fallback"
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        log_path = work / "fallback.log"
        images = sorted((attempt / "frames").glob("*.png")) + sorted((attempt / "hero").glob("*.png"))
        with Image.open(images[0]) as im:
            size = model_size(*im.size, params["long_side"])
        masks = [m for p in images if (m := attempt / "masks" / p.parent.name / p.name).exists()]

        def run_logged(step: str, frac: float, cmd: list[str], env: dict[str, str] | None = None) -> str:
            ctx.check_cancelled()
            ctx.progress(frac, step)
            r = subprocess.run(cmd, env=env, capture_output=True, text=True)
            with open(log_path, "a") as log:
                log.write(f"\n==== {' '.join(cmd[:3])} ...\n{r.stdout}{r.stderr}")
            if r.returncode != 0:
                tail = (r.stderr or r.stdout).strip().splitlines()[-3:]
                raise StageFailed(f"{step} failed: {' | '.join(tail)}")
            return r.stdout + r.stderr

        def colmap(step: str, frac: float, *args: str) -> str:
            return run_logged(step, frac, ["colmap", *args, "--log_color", "0"])

        device = ctx.gpu if ctx.gpu is not None else gpu.pick(DA3_VRAM_MB)
        ctx.log(f"GPU {device}: Depth Anything 3 on {len(images)} images")
        ff_txt = work / "feedforward"
        run_logged("posing the frames (Depth Anything 3)", 0.0,
                   [str(DA3_PYTHON), str(WORKER), str(ff_txt), *map(str, images), "--root", str(attempt),
                    "--size", *map(str, size), "--model", params["model"],
                    *(["--masks", *map(str, masks)] if masks else [])], env=gpu.cuda_env(device))

        # Start model: the feedforward poses and focal lengths, under the COLMAP database's ids.
        db = work / "database.db"
        shutil.copyfile(colmap_dir / "database.db", db)
        with sqlite3.connect(db) as con:
            db_images = {n: (i, c) for i, n, c in con.execute("SELECT image_id, name, camera_id FROM images")}
            db_cams = {c: (w, h) for c, w, h in con.execute("SELECT camera_id, width, height FROM cameras")}
        ff = read_images_txt(ff_txt / "images.txt")
        ff_focal = {int(f[0]): float(f[4]) for f in (line.split() for line in (ff_txt / "cameras.txt").read_text().splitlines())
                    if f and not f[0].startswith("#")}
        hero_cams = {c for n, (i, c) in db_images.items() if n.startswith("hero/")}
        start = work / "start"
        start.mkdir()
        (start / "cameras.txt").write_text("".join(
            f"{c} SIMPLE_PINHOLE {w} {h} {ff_focal[2 if c in hero_cams else 1]} {w / 2} {h / 2}\n"
            for c, (w, h) in db_cams.items()))
        (start / "images.txt").write_text("".join(
            f"{i} {' '.join(map(str, ff[n]['qvec']))} {' '.join(map(str, ff[n]['tvec']))} {c} {n}\n\n"
            for n, (i, c) in sorted(db_images.items(), key=lambda x: x[1][0]) if n in ff))
        (start / "points3D.txt").write_text("")

        model = start
        for rnd in (1, 2):
            tri, ba = work / f"triangulated{rnd}", work / f"adjusted{rnd}"
            tri.mkdir()
            ba.mkdir()
            colmap(f"triangulating ({rnd}/2)", 0.4 + 0.25 * (rnd - 1), "point_triangulator",
                   "--database_path", str(db), "--image_path", str(colmap_dir / "images"),
                   "--input_path", str(model), "--output_path", str(tri), "--clear_points", "1")
            colmap(f"bundle adjustment ({rnd}/2)", 0.5 + 0.25 * (rnd - 1), "bundle_adjuster",
                   "--input_path", str(tri), "--output_path", str(ba),
                   "--BundleAdjustment.refine_focal_length", "1", "--BundleAdjustment.refine_principal_point", "0",
                   "--BundleAdjustment.refine_extra_params", "0")
            model = ba
        stats = colmap("analyzing the model", 0.9, "model_analyzer", "--path", str(model))
        adjusted_txt = work / "adjusted_txt"
        adjusted_txt.mkdir()
        colmap("reading the model", 0.92, "model_converter", "--input_path", str(model),
               "--output_path", str(adjusted_txt), "--output_type", "TXT")

        # The hybrid: images bundle adjustment could not constrain keep their feedforward pose.
        adjusted = read_images_txt(adjusted_txt / "images.txt")
        constrained = [n for n, im in adjusted.items() if im["n_points"] >= params["min_points"]]
        unrefined = sorted(set(adjusted) - set(constrained))
        n_inputs = len(images)
        ctx.metric("n_refined", len(constrained))
        ctx.metric("unrefined", unrefined)
        if len(constrained) < 3 or len(unrefined) > params["max_unrefined"] * n_inputs:
            raise Rejected(f"bundle adjustment could constrain only {len(constrained)} of {n_inputs} images "
                           f"from the Depth Anything 3 poses (at most {params['max_unrefined']:.0%} may stay unrefined)")
        poses = hybrid({n: _pose(ff[n]) for n in adjusted}, {n: _pose(im) for n, im in adjusted.items()}, constrained)

        out_txt = work / "model_txt"
        out_txt.mkdir()
        shutil.copy(adjusted_txt / "cameras.txt", out_txt)
        shutil.copy(adjusted_txt / "points3D.txt", out_txt)
        lines = [line for line in (adjusted_txt / "images.txt").read_text().splitlines() if not line.startswith("#")]
        with open(out_txt / "images.txt", "w") as f:
            for header, points in zip(lines[0::2], lines[1::2]):
                h = header.split()
                rot, t = poses[h[9]]
                f.write(f"{h[0]} {' '.join(f'{x:.12g}' for x in _qvec(rot))} {' '.join(f'{x:.12g}' for x in t)} "
                        f"{h[8]} {h[9]}\n{points}\n")
        (work / "model").mkdir()
        colmap("writing the model", 0.96, "model_converter", "--input_path", str(out_txt),
               "--output_path", str(work / "model"), "--output_type", "BIN")

        reproj = re.search(r"Mean reprojection error:\s*([\d.]+)", stats)
        focal = [float(line.split()[4]) for line in (out_txt / "cameras.txt").read_text().splitlines()
                 if line and not line.startswith("#")]
        ctx.metric("n_registered", len(poses))
        ctx.metric("reg_rate", round(len(poses) / n_inputs, 4))
        ctx.metric("reproj_err", float(reproj.group(1)) if reproj else None)
        ctx.metric("hero_registered", any(n.startswith("hero/") for n in poses))
        ctx.metric("focal_px", [round(f, 1) for f in focal])
        ctx.progress(1.0, f"{len(constrained)}/{n_inputs} images refined"
                     + (f", {len(unrefined)} kept from Depth Anything 3" if unrefined else ""))
