"""extract and dedup: from the orbit video to the frames used for poses."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from giro.stages.base import Ctx, Rejected, Stage, StageFailed


class Extract(Stage):
    """frames_raw/ as lossless PNG. The orbit stage already writes PNG frames,
    so this only decodes video.mp4 when frames_raw is missing."""

    name = "extract"
    inputs = ("video.mp4",)
    outputs = ("frames_raw",)

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        out = attempt / "frames_raw"
        if out.is_dir() and any(out.glob("*.png")):
            ctx.log("frames_raw already holds the orbit's PNG frames")
        else:
            out.mkdir(exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-y", "-i", str(attempt / "video.mp4"), str(out / "%05d.png")],
                check=True,
            )
        ctx.metric("n_raw", len(list(out.glob("*.png"))))


def _gray(path: Path, width: int) -> np.ndarray:
    im = Image.open(path).convert("L")
    height = round(im.height * width / im.width)
    return np.asarray(im.resize((width, height), Image.BILINEAR), dtype=np.float64) / 255.0


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    c1, c2 = 0.01**2, 0.03**2
    ma, mb = gaussian_filter(a, sigma), gaussian_filter(b, sigma)
    va = gaussian_filter(a * a, sigma) - ma * ma
    vb = gaussian_filter(b * b, sigma) - mb * mb
    cov = gaussian_filter(a * b, sigma) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))).mean())


class Dedup(Stage):
    """Trim the static runs at both ends, drop interior near-duplicates, then
    subsample evenly in cumulative visual change (a stand-in for angle, since
    the orbit's angular speed is not constant)."""

    name = "dedup"
    defaults = {
        "static_ssim": 0.98,     # a frame this similar to the first/last frame is part of a static run
        "duplicate_ssim": 0.98,  # an interior frame this similar to the last kept one is dropped
        "target": 120,
        "analysis_width": 160,
    }
    inputs = ("frames_raw",)
    outputs = ("frames", "dedup.json")

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        raw = sorted((attempt / "frames_raw").glob("*.png"))
        if len(raw) < 3:
            raise StageFailed(f"only {len(raw)} frames in frames_raw")
        gray = []
        for i, p in enumerate(raw):
            gray.append(_gray(p, params["analysis_width"]))
            if i % 16 == 0:
                ctx.progress(0.5 * i / len(raw), "measuring frame similarity")

        first, last = gray[0], gray[-1]
        start = 0
        while start < len(raw) - 1 and ssim(gray[start], first) > params["static_ssim"]:
            start += 1
        end = len(raw) - 1
        while end > start and ssim(gray[end], last) > params["static_ssim"]:
            end -= 1
        if end - start < 2:
            raise Rejected("the video is static: no camera motion was found")

        kept = [start]
        change = [0.0]  # cumulative dissimilarity along the kept frames
        for i in range(start + 1, end + 1):
            s = ssim(gray[i], gray[kept[-1]])
            if s < params["duplicate_ssim"]:
                kept.append(i)
                change.append(change[-1] + (1.0 - s))
            if i % 16 == 0:
                ctx.progress(0.5 + 0.4 * (i - start) / (end - start + 1), "dropping near-duplicates")

        selected = kept
        if len(kept) > params["target"]:
            marks = np.linspace(0.0, change[-1], params["target"])
            picks = np.searchsorted(change, marks).clip(0, len(kept) - 1)
            selected = sorted({kept[j] for j in picks})

        out = attempt / "frames"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        for i in selected:
            # Hard links: frames/ costs no space and keeps the raw names.
            (out / raw[i].name).hardlink_to(raw[i])

        (attempt / "dedup.json").write_text(json.dumps({
            "static_start": start, "static_end": len(raw) - 1 - end,
            "kept": [raw[i].name for i in selected],
        }, indent=2) + "\n")
        ctx.metric("n_raw", len(raw))
        ctx.metric("trimmed_start", start)
        ctx.metric("trimmed_end", len(raw) - 1 - end)
        ctx.metric("near_duplicates", (end - start + 1) - len(kept))
        ctx.metric("n_frames", len(selected))
        if len(selected) < params["target"]:
            ctx.log(f"{len(selected)} distinct frames, fewer than the {params['target']} target; a longer video gives more views")
        ctx.progress(1.0, f"{len(selected)} frames kept")
