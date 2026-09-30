"""gapfill: regenerate only the arc of an orbit where the camera path jumps.

When the gate's only complaint is a jump between two consecutive posed frames
(fast, smeared frames COLMAP could not place, or a stretch it misplaced), the
video model is asked again for just that arc: first-last-frame video from the
posed frame before the jump to the one after it. The in-between frames replace
the unposed ones, and masks, poses and gate run again. A fill costs ~40 s of
video instead of a 7-minute reroll, and keeps the seed (docs/FINDINGS.md, "Gap fill";
the idea is OrbitForge's coverage-aware completion, arXiv:2606.24799).

Generated frames are named after the frame before the jump, 00074_01.png ...
00074_20.png, which sorts between 00074.png and 00075.png in every bytewise
sort (Python, COLMAP, Brush, JS). The originals of the replaced frames stay in
frames_raw/; everything a fill did is listed in gapfill/gapfill.json so it can
be shown and undone.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from PIL import Image

from giro import workflows
from giro.comfy import ComfyClient, ComfyError, Done, Progress, server
from giro.stages.base import Ctx, Stage, StageFailed
from giro.stages.orbit import ORBIT_VRAM_MB, events, output_images

MANIFEST = Path("gapfill") / "gapfill.json"


@dataclass
class Gap:
    first: str  # the posed frame before the jump, "frames/00074.png"
    last: str   # the posed frame after it
    deg: float  # the jump, along the orbit's direction
    missing: list[str]  # frames between the two (none were posed); the fill replaces them
    length: int  # frames to generate, both ends included (on the video model's 17k+5 grid)


def plan(attempt: Path, params: dict[str, Any]) -> tuple[list[Gap], str]:
    """The jumps a fill can repair, or ([], why not) from the attempt's gate.json."""
    path = attempt / "gate.json"
    if not path.exists():
        return [], "the gate has not run"
    gate = json.loads(path.read_text())
    if gate.get("passed"):
        return [], "the gate passed"
    failing = {c["metric"] for c in gate.get("checks", []) if not c["pass"]}
    if failing != {"max_step_deg"}:
        return [], f"the gate also failed on {', '.join(sorted(failing - {'max_step_deg'}))}"
    threshold = next(c["threshold"] for c in gate["checks"] if c["metric"] == "max_step_deg")
    ring = gate.get("ring", {})
    names, azimuths = ring.get("frames", []), np.asarray(ring.get("azimuths", []), dtype=float)
    if len(names) < 3 or len(names) != len(azimuths):
        return [], "the gate recorded no camera ring"
    steps = (np.diff(azimuths) + 180.0) % 360.0 - 180.0
    forward = steps * (1.0 if steps.sum() >= 0 else -1.0)
    small = np.abs(forward[np.abs(steps) <= threshold])
    typical = float(np.median(small)) if len(small) else threshold / 2
    frames = sorted(f"frames/{p.name}" for p in (attempt / "frames").glob("*.png"))
    gaps = []
    for i in np.flatnonzero(np.abs(steps) > threshold):
        deg = float(forward[i])
        if deg <= 0:
            return [], f"the camera snaps back {-deg:.0f} degrees at {names[i + 1]}, which a fill cannot repair"
        if deg > params["max_gap_deg"]:
            return [], f"a {deg:.0f}-degree jump is more than a fill covers (max_gap_deg {params['max_gap_deg']:.0f})"
        n_new = max(1, round(deg / typical) - 1)
        gaps.append(Gap(
            first=names[i], last=names[i + 1], deg=round(deg, 1),
            missing=[f for f in frames if names[i] < f < names[i + 1]],
            length=min(workflows.snap_length(n_new + 2), workflows.snap_length(params["max_length"])),
        ))
    if len(gaps) > params["max_gaps"]:
        return [], f"{len(gaps)} jumps; a fill repairs at most {params['max_gaps']}"
    return gaps, ""


def applied(attempt: Path) -> dict[str, Any] | None:
    """The fill in effect (its manifest), or None; a dedup rerun rebuilds frames/ and undoes it."""
    path = attempt / MANIFEST
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    inserted = [n for gap in manifest["gaps"] for n in gap["inserted"]]
    return manifest if all((attempt / n).exists() for n in inserted) else None


def summary(manifest: dict[str, Any]) -> list[str]:
    """One line per filled gap, for the attempt's record and the UI."""
    return [f"{Path(g['first']).stem}→{Path(g['last']).stem}: {g['deg']:.0f}° jump, "
            f"{len(g['inserted'])} generated frames" + (f" replace {len(g['dropped'])}" if g["dropped"] else "")
            for g in manifest["gaps"]]


def revert(attempt: Path) -> None:
    """Undo an earlier fill: remove the generated frames, restore the replaced ones."""
    path = attempt / MANIFEST
    if path.exists():
        for gap in json.loads(path.read_text())["gaps"]:
            for name in gap["inserted"]:
                (attempt / name).unlink(missing_ok=True)
            for name in gap["dropped"]:
                raw = attempt / "frames_raw" / Path(name).name
                if raw.exists() and not (attempt / name).exists():
                    (attempt / name).hardlink_to(raw)
    shutil.rmtree(attempt / "gapfill", ignore_errors=True)


class GapFill(Stage):
    """Runs only when the job runner asks (after a gate rejection), so it is not in PIPELINE."""

    name = "gapfill"
    defaults = {
        "enabled": True,
        "max_gap_deg": 90.0,  # beyond this the model may take the other way round
        "max_gaps": 2,
        "max_length": 39,     # frames per arc, both ends included
        "steps": 20,
        "seed": None,         # the attempt's seed; gap k uses seed + k
    }
    inputs = ("gate.json",)
    outputs = (str(MANIFEST),)
    gpu_mb = ORBIT_VRAM_MB

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        revert(attempt)
        gaps, why = plan(attempt, params)
        if not gaps:
            raise StageFailed(f"nothing to fill: {why}")
        t0 = time.monotonic()
        try:
            asyncio.run(self._generate(attempt, gaps, params, ctx))
        except ComfyError as e:
            ctx.check_cancelled()
            raise StageFailed(f"ComfyUI: {e}") from None
        except aiohttp.ClientConnectionError:
            ctx.check_cancelled()
            raise StageFailed("lost the connection to ComfyUI (it stopped or crashed); Retry runs this step again") from None

        # Only now touch frames/: a failed or cancelled fill leaves the attempt as it was.
        records = []
        for gap in gaps:
            arc = sorted((attempt / "gapfill" / Path(gap.first).stem).glob("*.png"))
            if len(arc) < 3:
                raise StageFailed(f"the video model returned {len(arc)} frames for the arc after {gap.first}")
            for name in gap.missing:
                (attempt / name).unlink()  # a hard link; the original stays in frames_raw/
            inserted = []
            for k, src in enumerate(arc[1:-1], start=1):  # both ends are the posed frames themselves
                name = f"frames/{Path(gap.first).stem}_{k:02d}.png"
                (attempt / name).hardlink_to(src)
                inserted.append(name)
            records.append(asdict(gap) | {"inserted": inserted, "dropped": gap.missing})
        manifest = {"gaps": records, "seconds": round(time.monotonic() - t0, 1)}
        (attempt / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
        ctx.metric("n_gaps", len(records))
        ctx.metric("n_inserted", sum(len(r["inserted"]) for r in records))
        ctx.metric("n_dropped", sum(len(r["dropped"]) for r in records))
        ctx.metric("filled", summary(manifest))
        ctx.progress(1.0, "; ".join(summary(manifest)))

    async def _generate(self, attempt: Path, gaps: list[Gap], params: dict[str, Any], ctx: Ctx) -> None:
        device = ctx.gpu if ctx.gpu is not None else server.pick_gpu(ORBIT_VRAM_MB)
        ctx.log(f"GPU {device}: starting ComfyUI if needed")
        with await asyncio.to_thread(server.Lease, device) as lease:
            async with ComfyClient(lease.url) as comfy:
                try:
                    for k, gap in enumerate(gaps):
                        await self._arc(comfy, attempt, gap, k, len(gaps), params, ctx)
                finally:
                    await comfy.free()

    async def _arc(self, comfy: ComfyClient, attempt: Path, gap: Gap, k: int, n: int,
                   params: dict[str, Any], ctx: Ctx) -> None:
        with Image.open(attempt / gap.first) as im:
            width, height = im.size
        seed = (params["seed"] or 0) + k
        ctx.log(f"gap {k + 1}/{n}: {gap.first} -> {gap.last}, {gap.deg:.0f} degrees, {gap.length} frames, seed {seed}")
        prompt = workflows.build_arc(
            await comfy.upload_image(attempt / gap.first), await comfy.upload_image(attempt / gap.last),
            prompt=workflows.GAP_PROMPT, width=width, height=height, length=gap.length, seed=seed,
            steps=params["steps"], output_prefix=f"giro/gapfill-{time.strftime('%Y%m%d-%H%M%S')}-{seed}/frame")
        done: Done | None = None
        async for event in events(comfy, prompt, ctx):
            if isinstance(event, Progress):
                ctx.progress((k + 0.9 * event.value / event.max) / n, f"gap {k + 1}/{n}: step {event.value}/{event.max}")
            elif isinstance(event, Done):
                done = event
        assert done is not None
        out = attempt / "gapfill" / Path(gap.first).stem
        out.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(output_images(done)):
            await comfy.download(img, out / f"{i:05d}.png")
