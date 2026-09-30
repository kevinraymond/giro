"""edit: change the hero image before any orbit is generated, with
Qwen-Image-Edit-2511 (int8) in giro's ComfyUI. The usual use is swapping the background
when the subject stands near a wall that a full orbit would run into.

It runs in the job directory, not an attempt: input/source.png -> input/edited.png.
`fast` uses the 4-step Lightning LoRA (cfg 1); otherwise 40 steps at cfg 4, as the
ComfyUI template does. Output resolution is FluxKontextImageScale's nearest preferred
size (about 1 MP; 880x1184 for a 3:4 image).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import aiohttp

from giro import workflows
from giro.comfy import ComfyClient, ComfyError, Done, Preview, Progress, server
from giro.stages.base import Ctx, Stage, StageFailed

# int8 diffusion model (20.5 GB) plus the 7B text encoder; weights stream from RAM like the video's.
EDIT_VRAM_MB = 20_000


class EditImage(Stage):
    name = "edit"
    defaults = {"prompt": "", "negative": "", "seed": None, "fast": True}
    inputs = ("input/source.png",)
    outputs = ("input/edited.png",)
    gpu_mb = EDIT_VRAM_MB

    def run(self, job_dir: Path, params: dict[str, Any], ctx: Ctx) -> None:
        if not params["prompt"].strip():
            raise StageFailed("describe the edit, e.g. 'replace the background with a plain studio backdrop'")
        if params["seed"] is None:
            raise StageFailed("edit needs a seed")
        try:
            asyncio.run(self._edit(job_dir, params, ctx))
        except ComfyError as e:
            ctx.check_cancelled()
            raise StageFailed(f"ComfyUI: {e}") from None
        except aiohttp.ClientConnectionError:
            ctx.check_cancelled()
            raise StageFailed("lost the connection to ComfyUI (it stopped or crashed); try again") from None

    async def _edit(self, job_dir: Path, params: dict[str, Any], ctx: Ctx) -> None:
        device = ctx.gpu if ctx.gpu is not None else server.pick_gpu(EDIT_VRAM_MB)
        ctx.log(f"GPU {device}: starting ComfyUI if needed")
        with await asyncio.to_thread(server.Lease, device) as lease:
            async with ComfyClient(lease.url) as comfy:
                try:
                    await self._sample(comfy, job_dir, params, ctx)
                finally:
                    await comfy.free()

    async def _sample(self, comfy: ComfyClient, job_dir: Path, params: dict[str, Any], ctx: Ctx) -> None:
        fast = bool(params["fast"])
        prompt = workflows.build(
            "edit_image",
            image=await comfy.upload_image(job_dir / "input" / "source.png"),
            prompt=params["prompt"], negative=params["negative"], seed=int(params["seed"]),
            steps=4 if fast else 40, cfg=1.0 if fast else 4.0,
            output_prefix=f"giro/edit-{time.strftime('%Y%m%d-%H%M%S')}",
        )
        if not fast:  # sample the base model: drop the Lightning LoRA
            prompt["sampler"]["inputs"]["model"] = ["cfgnorm", 0]
            del prompt["lightning"]
        preview = job_dir / "input" / "edit_preview.jpg"
        done: Done | None = None
        ctx.progress(0.02, "loading the edit model")
        async for event in comfy.run(prompt):
            if ctx.is_cancelled():
                await comfy.interrupt()
                ctx.check_cancelled()
            if isinstance(event, Progress):
                ctx.progress(0.1 + 0.8 * event.value / event.max, f"step {event.value}/{event.max}")
            elif isinstance(event, Preview):
                preview.write_bytes(event.image)
                ctx.preview(preview)
            elif isinstance(event, Done):
                done = event
        assert done is not None
        images = [img for out in done.outputs.values() for img in out.get("images", [])]
        if not images:
            raise StageFailed("ComfyUI finished the edit without an image")
        tmp = job_dir / "input" / "edited.tmp.png"
        await comfy.download(images[0], tmp)
        tmp.replace(job_dir / "input" / "edited.png")
        ctx.progress(1.0, "done")
