"""masks: a subject mask for every frame and the hero, from SAM3.1 in giro's ComfyUI.

Two text-prompted detections run on each image independently (no tracking,
so nothing drifts over the orbit):

- subject:    the subject prompt ("main subject") finds the body, but can miss
              thin held or attached parts such as a sword or a mirror
- background: the room prompt ("wall, floor, ...") finds what is not the subject

The mask is the subject plus every non-background region that touches it. So
a sword held by the subject is kept, while a window the room prompt missed,
standing apart from the subject, is not. Both raw masks are kept in
masks/raw/ so the combination can be retuned without running SAM again.

Layout mirrors dataset/images/ for Brush: masks/frames/<frame>.png and
masks/hero/hero.png, 8-bit, 255 = subject.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from PIL import Image
from scipy import ndimage

from giro.comfy import ComfyClient, ComfyError, Progress, server
from giro.stages.base import Ctx, Stage, StageFailed

SAM_VRAM_MB = 6_000
SAM_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"


def sam_workflow(groups: dict[str, list[Path]], raw: Path, params: dict[str, Any]) -> dict[str, Any]:
    """Detect subject and background on each group of same-sized images; the
    raw masks go to raw/<prompt>/<group>/<name>."""
    wf: dict[str, Any] = {
        "sam": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CHECKPOINT}},
    }
    prompts = {"subject": params["subject_prompt"], "background": params["background_prompt"]}
    for kind, text in prompts.items():
        if text:
            wf[f"{kind}_text"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": text}}
    for group, paths in groups.items():
        wf[f"{group}_images"] = {"class_type": "GiroLoadImages", "inputs": {
            "paths": json.dumps([str(p) for p in paths]), "width": 0, "height": 0}}
        for kind, text in prompts.items():
            if not text:
                continue
            wf[f"{group}_{kind}"] = {"class_type": "SAM3_Detect", "inputs": {
                "model": ["sam", 0], "image": [f"{group}_images", 0], "conditioning": [f"{kind}_text", 0],
                "threshold": params["threshold"], "refine_iterations": 2, "individual_masks": False}}
            wf[f"{group}_{kind}_save"] = {"class_type": "GiroSaveMasks", "inputs": {
                "masks": [f"{group}_{kind}", 0],
                "paths": json.dumps([str(raw / kind / group / p.name) for p in paths])}}
    return wf


def combine(subject: np.ndarray, background: np.ndarray | None, touch_px: int = 3,
            gap_px: int = 4, max_add: float = 0.25) -> np.ndarray:
    """Subject plus the non-background pieces that touch it (boolean masks).

    The background is first grown by gap_px, which closes the thin gaps between
    neighboring background segments (a wall and a window) that would otherwise
    reach the subject as strips. A piece larger than max_add times the subject is
    a background region the room prompt missed (a doorway), not a held object.
    """
    if background is None:
        return subject
    closed = ndimage.binary_dilation(background, iterations=gap_px) if gap_px else background
    added = ~closed & ~subject
    labels, n = ndimage.label(added)
    if n == 0:
        return subject
    near = ndimage.binary_dilation(subject, iterations=touch_px) if touch_px else subject
    touching = np.unique(labels[near & added])
    touching = touching[touching > 0]
    sizes = ndimage.sum_labels(added, labels, touching)
    keep = touching[sizes <= max_add * subject.sum()]
    return subject | np.isin(labels, keep)


class Masks(Stage):
    name = "masks"
    defaults = {
        # "held object" catches what "main subject" alone misses: a sword, a bag.
        "subject_prompt": "main subject, held object:2",
        # :N lets several walls (a corner) or floor patches count; "" turns the background off.
        "background_prompt": "wall:4, floor:2, window:4, door:2, baseboard:2, ceiling:2",
        "threshold": 0.5,
        "touch_px": 3,
        "gap_px": 4,
        "max_add": 0.25,
        "min_area": 0.005,  # a mask smaller than this fraction of the image counts as missing
    }
    inputs = ("frames", "hero")
    outputs = ("masks/frames", "masks/hero/hero.png")
    gpu_mb = SAM_VRAM_MB

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        out = attempt / "masks"
        raw = out / "raw"
        groups = {
            "frames": sorted((attempt / "frames").glob("*.png")),
            "hero": [attempt / "hero" / "hero.png"],
        }
        # SAM's raw masks are kept while the params that shape them are unchanged; then only
        # new or changed images (a gap fill's frames) are detected again.
        record = attempt / ".stages" / f"{self.name}.json"
        previous = json.loads(record.read_text()).get("params") if record.exists() else None
        if previous != params and out.exists():
            shutil.rmtree(out)
        for group in groups:
            shutil.rmtree(out / group, ignore_errors=True)  # combined masks: rebuilt below from raw/

        def detected(group: str, p: Path) -> bool:
            m = raw / "subject" / group / p.name
            return m.exists() and m.stat().st_mtime_ns >= p.stat().st_mtime_ns

        todo = {g: [p for p in paths if not detected(g, p)] for g, paths in groups.items()}
        todo = {g: paths for g, paths in todo.items() if paths}
        n_images = sum(len(v) for v in todo.values())
        ctx.metric("n_detected", n_images)
        if todo:
            try:
                asyncio.run(self._detect(todo, raw, params, n_images, ctx))
            except ComfyError as e:
                ctx.check_cancelled()
                raise StageFailed(f"ComfyUI: {e}") from None
            except aiohttp.ClientConnectionError:
                ctx.check_cancelled()
                raise StageFailed("lost the connection to ComfyUI (it stopped or crashed); Retry runs this step again") from None

        ctx.progress(0.9, "combining subject and background masks")
        areas: dict[str, float] = {}
        for group, paths in groups.items():
            for p in paths:
                subject = np.asarray(Image.open(raw / "subject" / group / p.name)) > 127
                bg_path = raw / "background" / group / p.name
                background = np.asarray(Image.open(bg_path)) > 127 if bg_path.exists() else None
                mask = combine(subject, background, params["touch_px"], params["gap_px"], params["max_add"])
                dest = out / group / p.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask.astype(np.uint8) * 255, "L").save(dest)
                areas[f"{group}/{p.name}"] = round(float(mask.mean()), 5)

        frame_areas = np.array([a for k, a in areas.items() if k.startswith("frames/")])
        missing = [k for k, a in areas.items() if a < params["min_area"]]
        (out / "masks.json").write_text(json.dumps({"areas": areas, "missing": missing}, indent=2) + "\n")
        ctx.metric("n_masks", len(areas))
        ctx.metric("hero_area", areas["hero/hero.png"])
        ctx.metric("mean_area", round(float(frame_areas.mean()), 4))
        ctx.metric("area_cv", round(float(frame_areas.std() / max(frame_areas.mean(), 1e-9)), 3))
        ctx.metric("n_missing", len(missing))
        if areas["hero/hero.png"] < params["min_area"]:
            raise StageFailed(f"SAM found no subject in the hero image for the prompt {params['subject_prompt']!r}")
        ctx.progress(1.0, f"{len(areas)} masks, subject covers {frame_areas.mean():.0%} of a frame on average")

    async def _detect(self, groups: dict[str, list[Path]], raw: Path, params: dict[str, Any],
                      n_images: int, ctx: Ctx) -> None:
        device = ctx.gpu if ctx.gpu is not None else server.pick_gpu(SAM_VRAM_MB)
        ctx.log(f"GPU {device}: starting ComfyUI if needed")
        with await asyncio.to_thread(server.Lease, device) as lease:
            await self._run_sam(lease.url, groups, raw, params, n_images, ctx)

    async def _run_sam(self, url: str, groups: dict[str, list[Path]], raw: Path, params: dict[str, Any],
                       n_images: int, ctx: Ctx) -> None:
        n_passes = n_images * sum(bool(params[k]) for k in ("subject_prompt", "background_prompt"))
        async with ComfyClient(url) as comfy:
            async def interrupt_on_cancel() -> None:
                while not ctx.is_cancelled():
                    await asyncio.sleep(0.5)
                await comfy.interrupt()

            watcher = asyncio.create_task(interrupt_on_cancel())
            try:
                per_node: dict[str | None, int] = {}  # each SAM3_Detect reports images done
                async for event in comfy.run(sam_workflow(groups, raw, params)):
                    if ctx.is_cancelled():
                        await comfy.interrupt()
                        ctx.check_cancelled()
                    if isinstance(event, Progress):
                        per_node[event.node] = event.value
                        ctx.progress(0.9 * min(1.0, sum(per_node.values()) / n_passes), "detecting subject and background")
            finally:
                watcher.cancel()
                await comfy.free()
