"""export: delivery files for the canonical splat, and its size against the VR budget.

- splat.ply   master copy, full SH (INRIA PLY, the canonical frame)
- splat.sog   compressed, for the web viewer
- splat.spz   SPZ v3: Spark (2.3.1 too) rejects the v4 that splat-transform writes by default

`box` (min and max corners in meters, in the canonical frame: y up, feet at 0) trims
what the auto-crop left, e.g. a stand or a stray patch of floor; it never rescales.

Spark's guidance for Quest 3 standalone is at most ~500K splats (docs/FINDINGS.md, "VR on Quest 3").
Above `vr_budget`, a decimated splat.vr.sog is written as well.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

from giro import render, splat
from giro.stages.base import Ctx, Stage, StageFailed


class Export(Stage):
    name = "export"
    defaults = {"formats": ["sog", "spz"], "vr_budget": 500_000, "box": None}
    inputs = ("canonical/splat.ply",)
    outputs = ("export/splat.ply",)

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        out = attempt / "export"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        master = out / "splat.ply"
        if params["box"]:
            lo_hi = [float(x) for x in params["box"]]
            if len(lo_hi) != 6 or any(lo_hi[i] >= lo_hi[i + 3] for i in range(3)):
                raise StageFailed(f"box must be min x,y,z then max x,y,z, got {params['box']}")
            # splat-transform's -B works in the y-up frame, the canonical frame (docs/FINDINGS.md, "Export and viewing").
            render.splat_transform(str(attempt / "canonical" / "splat.ply"), "-B", ",".join(f"{x:.6g}" for x in lo_hi),
                                   str(master), gpu=ctx.gpu)
        else:
            shutil.copyfile(attempt / "canonical" / "splat.ply", master)
        n = len(splat.read_ply(master))
        ctx.metric("n_gaussians", n)
        ctx.metric("vr_budget", params["vr_budget"])
        ctx.metric("within_vr_budget", n <= params["vr_budget"])

        for i, fmt in enumerate(params["formats"]):
            ctx.progress(i / (len(params["formats"]) + 1), f"writing {fmt}")
            extra = ["--spz-version", "3"] if fmt == "spz" else []
            render.splat_transform(str(master), str(out / f"splat.{fmt}"), *extra, gpu=ctx.gpu)
        if n > params["vr_budget"]:
            ctx.progress(0.9, f"decimating {n:,} to {params['vr_budget']:,} for VR")
            with tempfile.TemporaryDirectory(prefix="giro-export-") as tmp:
                small = Path(tmp) / "vr.ply"
                render.splat_transform(str(master), str(small), "-d", str(params["vr_budget"]), gpu=ctx.gpu)
                render.splat_transform(str(small), str(out / "splat.vr.sog"), gpu=ctx.gpu)
        for f in sorted(out.iterdir()):
            ctx.metric(f"{f.name.replace('.', '_')}_mb", round(f.stat().st_size / 2**20, 2))
        ctx.progress(1.0, f"{n:,} Gaussians ({'within' if n <= params['vr_budget'] else 'over'} the VR budget)")
