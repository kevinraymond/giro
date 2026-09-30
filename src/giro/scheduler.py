"""GPU leases for the job runner.

Each GPU runs one heavy job at a time. A request names the GPUs it accepts in
order of preference and a priority; when a GPU frees up it goes to the
highest-priority waiter that accepts it, first come first served within a
priority. The runner gives later pipeline stages higher priority, so attempts
already in flight finish before new videos start.
"""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass(order=True)
class _Waiter:
    priority: int  # lower is served first
    seq: int
    prefs: Sequence[int] = field(compare=False)
    granted: asyncio.Future[int] = field(compare=False)


class GpuPool:
    def __init__(self, gpus: Sequence[int]):
        self.gpus = tuple(gpus)
        self._free = set(gpus)
        self._waiters: list[_Waiter] = []
        self._seq = itertools.count()

    def busy(self) -> set[int]:
        return set(self.gpus) - self._free

    @asynccontextmanager
    async def lease(self, prefs: Sequence[int], priority: int = 0) -> AsyncIterator[int]:
        usable = [g for g in prefs if g in self.gpus]
        if not usable:
            raise ValueError(f"none of GPUs {list(prefs)} are in the pool {list(self.gpus)}")
        waiter = _Waiter(priority, next(self._seq), usable, asyncio.get_running_loop().create_future())
        self._waiters.append(waiter)
        self._grant()
        try:
            gpu = await waiter.granted
        except asyncio.CancelledError:
            if waiter in self._waiters:
                self._waiters.remove(waiter)
            elif waiter.granted.done() and not waiter.granted.cancelled():
                self._release(waiter.granted.result())
            raise
        try:
            yield gpu
        finally:
            self._release(gpu)

    def _release(self, gpu: int) -> None:
        self._free.add(gpu)
        self._grant()

    def _grant(self) -> None:
        for waiter in sorted(self._waiters):
            gpu = next((g for g in waiter.prefs if g in self._free), None)
            if gpu is not None:
                self._free.remove(gpu)
                self._waiters.remove(waiter)
                waiter.granted.set_result(gpu)
