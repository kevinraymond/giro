"""dataset and train: assemble Brush's input with symlinks, then train with brush-cli."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from giro import gpu
from giro.stages.base import Ctx, Stage, StageFailed
from giro.stages.poses import read_images_txt

ROOT = Path(__file__).resolve().parents[3]
BRUSH = ROOT / "vendor" / "brush" / "target" / "release" / "brush-cli"
BRUSH_VRAM_MB = 8_000

EVAL_LINE = re.compile(r"Eval iter (\d+): PSNR ([\d.]+), ssim ([\d.]+)")
REFINE_LINE = re.compile(r"Refine iter (\d+), (\d+) splats")
LOADED_LINE = re.compile(r"Loaded dataset with (\d+) training, (\d+) eval views")


class Dataset(Stage):
    """dataset/ in the COLMAP layout Brush reads: images/{frames,hero} and
    sparse/0, all symlinks. Never put a .ply here: Brush would start from it."""

    name = "dataset"
    # masks: also link masks/ (mirroring images/), which Brush turns into alpha; see Train.alpha_mode
    # mask_erode_px: SAM masks run a pixel or two into the background; eroded, that sliver
    # trains as background instead of as a pale rim around the subject.
    defaults = {"eval_split_every": 8, "masks": True, "mask_erode_px": 2}
    inputs = ("poses/colmap/model",)
    outputs = ("dataset",)

    def inputs_for(self, params: dict[str, Any]) -> tuple[str, ...]:
        return self.inputs + (("masks/frames", "masks/hero") if params["masks"] else ())

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        out = attempt / "dataset"
        if out.exists():
            shutil.rmtree(out)
        model = (attempt / "poses" / "colmap" / "model").resolve()
        sparse = out / "sparse" / "0"
        sparse.mkdir(parents=True)
        for f in model.iterdir():
            (sparse / f.name).symlink_to(f)

        registered = sorted(read_images_txt(attempt / "poses" / "colmap" / "model_txt" / "images.txt"))
        # Brush sorts views by name and holds out every Nth index starting at 0.
        # The hero sorts last; if it would land on an eval index, leave out one
        # frame so the hero is always trained on.
        split = params["eval_split_every"]
        skipped = []
        if split and "hero/hero.png" in registered and registered.index("hero/hero.png") % split == 0:
            skipped.append(registered[registered.index("hero/hero.png") - 1])
        for name in registered:
            if name in skipped:
                continue
            dest = out / "images" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to((attempt / name).resolve())
            if params["masks"]:
                mask = attempt / "masks" / name
                if not mask.exists():
                    raise StageFailed(f"masks is on but masks/{name} is missing; run the masks stage")
                (out / "masks" / name).parent.mkdir(parents=True, exist_ok=True)
                if params["mask_erode_px"]:
                    m = ndimage.binary_erosion(np.asarray(Image.open(mask)) > 127, iterations=params["mask_erode_px"])
                    Image.fromarray(m.astype(np.uint8) * 255, "L").save(out / "masks" / name)
                else:
                    (out / "masks" / name).symlink_to(mask.resolve())
        n = len(registered) - len(skipped)
        ctx.metric("n_views", n)
        ctx.metric("n_eval", len(range(0, n, split)) if split else 0)
        if skipped:
            ctx.log(f"left out {skipped[0]} so the hero is not held out for eval")


class Train(Stage):
    name = "train"
    defaults = {
        "total_train_iters": 30000,
        "eval_split_every": 8,
        "eval_every": 1000,
        "export_every": 5000,
        "sh_degree": 3,
        "max_resolution": 1920,
        "seed": 42,
        "eval_save_to_disk": False,  # writes train/eval_<iter>/ renders of the held-out views
        # With dataset/masks: "masked" ignores the background pixels, "transparent" trains
        # them to alpha 0, so the background is emptied rather than left untrained: no glow
        # or wisps around the subject (docs/FINDINGS.md, "Masks"). Eval PSNR then scores the subject only.
        "alpha_mode": "transparent",
    }
    inputs = ("dataset",)
    outputs = ("train/final.ply",)
    gpu_mb = BRUSH_VRAM_MB

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        if not BRUSH.exists():
            raise StageFailed(f"brush-cli not built; run scripts/setup_brush.sh ({BRUSH})")
        out = attempt / "train"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        device = ctx.gpu if ctx.gpu is not None else gpu.pick(BRUSH_VRAM_MB)
        env = gpu.cuda_env(device) | {
            # Brush runs on wgpu/Vulkan and ignores CUDA_VISIBLE_DEVICES.
            "CUBECL_WGPU_DEFAULT_DEVICE": f"DiscreteGpu({device})",
            "RUST_LOG": "brush_cli=info",
            "NO_COLOR": "1",
        }
        total = params["total_train_iters"]
        cmd = [
            str(BRUSH), str(attempt / "dataset"),
            "--total-train-iters", str(total),
            "--eval-split-every", str(params["eval_split_every"]),
            "--eval-every", str(params["eval_every"]),
            "--export-every", str(params["export_every"]),
            "--export-path", str(out),
            "--export-name", "export_{iter}.ply",
            "--sh-degree", str(params["sh_degree"]),
            "--max-resolution", str(params["max_resolution"]),
            "--seed", str(params["seed"]),
        ]
        if params["eval_save_to_disk"]:
            cmd.append("--eval-save-to-disk")
        if (attempt / "dataset" / "masks").exists():
            cmd += ["--alpha-mode", params["alpha_mode"]]
        ctx.log(f"GPU {device}: {total} iterations")
        t0 = time.monotonic()
        seen_exports: set[Path] = set()
        with open(out / "brush.log", "w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert proc.stdout
            for line in proc.stdout:
                log.write(line)
                if ctx.is_cancelled():
                    proc.terminate()
                    proc.wait()
                    ctx.check_cancelled()
                if m := LOADED_LINE.search(line):
                    ctx.metric("train_views", int(m.group(1)))
                    ctx.metric("eval_views", int(m.group(2)))
                elif m := REFINE_LINE.search(line):
                    it = int(m.group(1))
                    ctx.metric("n_splats", int(m.group(2)), step=it)
                    ctx.progress(it / total, f"iter {it}, {int(m.group(2)):,} splats")
                elif m := EVAL_LINE.search(line):
                    it, psnr, ssim = int(m.group(1)), float(m.group(2)), float(m.group(3))
                    ctx.metric("eval_psnr", psnr, step=it)
                    ctx.metric("eval_ssim", ssim, step=it)
                    ctx.progress(it / total, f"iter {it}, eval PSNR {psnr:.2f}")
                # Exports are written in place; announce the previous one once a newer exists.
                exports = sorted(out.glob("export_*.ply"))
                for ready in exports[:-1]:
                    if ready not in seen_exports:
                        seen_exports.add(ready)
                        ctx.preview(ready)
            code = proc.wait()
        if code != 0:
            tail = (out / "brush.log").read_text().strip().splitlines()[-3:]
            raise StageFailed(f"brush-cli exited with {code}: {' | '.join(tail)}")
        exports = sorted(out.glob("export_*.ply"))
        if not exports:
            raise StageFailed("brush-cli finished without exporting a .ply")
        shutil.copyfile(exports[-1], out / "final.ply")
        ctx.preview(out / "final.ply")
        ctx.metric("train_seconds", round(time.monotonic() - t0, 1))
        ctx.metric("ply_mb", round((out / "final.ply").stat().st_size / 2**20, 1))
