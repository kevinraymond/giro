"""GPU inventory and placement (developed on a 2x RTX 4090 box).

GPUs are named by PCI order (nvidia-smi index). GPU 1 is headless, so it is
tried first; GPU 0 also drives the desktop.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

PREFERENCE = (1, 0)


@dataclass
class GpuState:
    index: int
    free_mb: int
    total_mb: int


def states() -> list[GpuState]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return [GpuState(*(int(x) for x in line.split(","))) for line in out.strip().splitlines()]


def pick(need_mb: int, reclaimable: set[int] = frozenset()) -> int:
    """First GPU in preference order with `need_mb` free (or whose memory giro
    itself can reclaim); otherwise the one with the most free memory."""
    by_index = {s.index: s for s in states()}
    for gpu in PREFERENCE:
        if gpu in by_index and (by_index[gpu].free_mb >= need_mb or gpu in reclaimable):
            return gpu
    return max(by_index.values(), key=lambda s: s.free_mb).index


def cuda_env(gpu: int) -> dict[str, str]:
    """Environment that makes CUDA device 0 the given PCI-ordered GPU."""
    return os.environ | {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu)}
