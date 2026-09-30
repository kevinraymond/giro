"""The worker contract every pipeline stage implements.

A stage runs inside one attempt directory: `run(attempt, params, ctx)`.
Stages are idempotent. A stage records a key built from its params and the
fingerprint of its inputs, and it is skipped when that key is unchanged and
its outputs still exist.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class Cancelled(Exception):
    pass


class StageFailed(Exception):
    """A stage could not produce its outputs; the message is shown to the user."""


class Rejected(StageFailed):
    """The stage judged the video unusable (static, not an orbit, ...): another seed may
    do better. Any other StageFailed is a fault in giro or the machine, not in the video."""


@dataclass
class Ctx:
    """What a stage may report while it runs. The CLI prints; the orchestrator
    will publish the same calls as websocket events."""

    stage: str = ""
    on_progress: Callable[[str, float, str], None] | None = None
    on_metric: Callable[[str, str, Any, int | None], None] | None = None  # stage, name, value, step
    on_log: Callable[[str, str], None] | None = None
    on_preview: Callable[[str, Path], None] | None = None
    on_start: Callable[[str], None] | None = None  # the stage runs (it was not skipped)
    is_cancelled: Callable[[], bool] = lambda: False
    gpu: int | None = None  # set by the scheduler; None lets the stage pick one
    metrics: dict[str, Any] = field(default_factory=dict)

    def progress(self, frac: float, msg: str = "") -> None:
        if self.on_progress:
            self.on_progress(self.stage, frac, msg)

    def metric(self, name: str, value: Any, step: int | None = None) -> None:
        """Record a measurement; `step` (e.g. a training iteration) makes a series of it."""
        self.metrics[name] = value
        if self.on_metric:
            self.on_metric(self.stage, name, value, step)

    def log(self, msg: str) -> None:
        if self.on_log:
            self.on_log(self.stage, msg)

    def preview(self, path: Path) -> None:
        if self.on_preview:
            self.on_preview(self.stage, path)

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise Cancelled(self.stage)


class Stage:
    name: str = ""
    defaults: dict[str, Any] = {}
    inputs: tuple[str, ...] = ()   # paths relative to the attempt dir
    outputs: tuple[str, ...] = ()
    gpu_mb: int = 0  # VRAM it needs; 0 means it runs on the CPU and needs no GPU lease

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        raise NotImplementedError

    def execute(self, attempt: Path, params: dict[str, Any] | None, ctx: Ctx, force: bool = False) -> bool:
        """Run unless an identical earlier run is recorded. Returns True if it ran."""
        # Absolute: external tools (Brush) resolve relative paths against their own bases.
        attempt = attempt.resolve()
        merged = self.defaults | (params or {})
        ctx.stage = self.name
        ctx.metrics = {}
        ctx.check_cancelled()  # a stage queued behind a cancel must not start
        record = attempt / ".stages" / f"{self.name}.json"
        key = self._key(attempt, merged)
        if not force and record.exists() and all((attempt / o).exists() for o in self.outputs):
            done = json.loads(record.read_text())
            if done.get("key") == key:
                ctx.metrics.update(done.get("metrics", {}))
                ctx.log("unchanged since last run, skipped")
                return False
        t0 = time.monotonic()
        if ctx.on_start:
            ctx.on_start(self.name)
        try:
            self.run(attempt, merged, ctx)
        except StageFailed as e:
            # Keep what was measured: the gate and UI explain failures from it.
            _merge_metrics(attempt, self.name, ctx.metrics | {"failed": str(e)})
            raise
        record.parent.mkdir(exist_ok=True)
        record.write_text(json.dumps({
            "key": key, "params": merged, "seconds": round(time.monotonic() - t0, 1),
            "metrics": {k: v for k, v in ctx.metrics.items()},
        }, indent=2) + "\n")
        _merge_metrics(attempt, self.name, ctx.metrics)
        return True

    def inputs_for(self, params: dict[str, Any]) -> tuple[str, ...]:
        """The inputs a run with these params reads (some params add inputs)."""
        return self.inputs

    def _key(self, attempt: Path, params: dict[str, Any]) -> str:
        h = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode())
        for rel in self.inputs_for(params):
            path = attempt / rel
            files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
            for f in files:
                if f.exists():
                    st = f.stat()
                    h.update(f"{f.relative_to(attempt)}:{st.st_size}:{st.st_mtime_ns}".encode())
        return h.hexdigest()[:16]


def _merge_metrics(attempt: Path, stage: str, metrics: dict[str, Any]) -> None:
    path = attempt / "metrics.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[stage] = metrics
    path.write_text(json.dumps(data, indent=2) + "\n")
