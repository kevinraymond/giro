"""The orchestrator's core: every job the server knows, the runners working on them,
and the stream of events the UI listens to.

All state changes happen on the server's event loop. Stages run in worker
threads and report through `publish`, which hops back onto the loop. Every
user action on an attempt is one of: change its params or verdict, then queue
it again (stages whose inputs and params are unchanged are skipped), or stop it.
"""

from __future__ import annotations

import asyncio
import random
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from giro import stages
from giro.comfy import server
from giro.index import Index
from giro.job import TERMINAL, Job, JobLocked, JobSpec, Runner, locked_elsewhere
from giro.scheduler import GpuPool

# Events kept in the index for replay; progress is only held as the latest value.
LOGGED = {"metric", "preview", "log"}


class ActionError(Exception):
    """The action does not apply to the job or attempt in its current state (HTTP 409)."""


class Manager:
    def __init__(self, root: Path, index: Index, gpus: list[int]):
        self.root = root
        self.index = index
        self.pool = GpuPool(gpus)
        self.jobs: dict[str, Job] = {}
        self.runners: dict[str, Runner] = {}
        self.cancels: dict[str, threading.Event] = {}
        self.run_tasks: dict[str, asyncio.Task[None]] = {}
        self.edit_tasks: dict[str, asyncio.Task[None]] = {}
        # (job, seed) -> stage -> latest progress event
        self.progress: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
        self.previews: dict[tuple[str, int], dict[str, Any]] = {}  # latest preview event per attempt
        self.subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None

    # ---- lifecycle -------------------------------------------------------------------------

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self.root.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.root.iterdir()):
            if not (path / "job.json").exists():
                continue
            try:
                job = Job.load(path)
            except (ValueError, TypeError, KeyError) as e:
                print(f"giro serve: skipping {path.name}: unreadable job.json ({e})", flush=True)
                continue
            self._attach(job)
            if job.edit.get("status") == "running":  # the server died mid-edit
                job.edit |= {"status": "error", "error": "interrupted; try again"}
                job.save()
            if job.status == "running" and not locked_elsewhere(path):  # the process running it died
                job.status = "interrupted"
                for a in job.attempts:
                    if a.status in ("queued", "running"):
                        a.status = "cancelled"
                job.save()
            else:
                self.index.put_job(job.data())
        self.index.drop_missing(set(self.jobs))

    async def stop(self) -> None:
        for cancel in self.cancels.values():
            cancel.set()
        for task in self.edit_tasks.values():
            task.cancel()
        tasks = [*self.run_tasks.values(), *self.edit_tasks.values()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._stop_idle_comfy()

    def _attach(self, job: Job) -> None:
        job.on_save = lambda data: self.publish({"type": "job", "job": data["id"], "data": data})
        self.jobs[job.id] = job

    # ---- events ----------------------------------------------------------------------------

    def publish(self, event: dict[str, Any]) -> None:
        """Deliver an event to the index and every subscriber. Safe from any thread."""
        event = event | {"ts": time.time()}
        if threading.get_ident() == self._loop_thread or self.loop is None:
            self._deliver(event)
        else:
            self.loop.call_soon_threadsafe(self._deliver, event)

    def _deliver(self, event: dict[str, Any]) -> None:
        kind = event["type"]
        if kind == "job":
            self.index.put_job(event["data"])
        elif kind == "start":
            # The stage runs again: what it reported last time is stale.
            self.index.clear_events(event["job"], event["seed"], [event["stage"]])
            self.progress.get((event["job"], event["seed"]), {}).pop(event["stage"], None)
        elif kind == "progress":
            self.progress.setdefault((event["job"], event["seed"]), {})[event["stage"]] = event
        elif kind in LOGGED:
            self.index.add_event(event)
            if kind == "preview":
                self.previews[(event["job"], event["seed"])] = event
        for queue in list(self.subscribers):
            if queue.qsize() < 2000:  # a stalled client loses events rather than memory
                queue.put_nowait(event)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.subscribers.discard(queue)

    def snapshot(self) -> dict[str, Any]:
        """What a newly connected client needs before the event stream makes sense."""
        return {
            "type": "hello",
            "jobs": [self.jobs[j].data() for j in sorted(self.jobs, reverse=True)],
            "progress": [ev for stages_ in self.progress.values() for ev in stages_.values()],
            "previews": list(self.previews.values()),
            "gpus": list(self.pool.gpus),
        }

    # ---- jobs ------------------------------------------------------------------------------

    def job(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError:
            raise KeyError(f"no job {job_id}") from None

    def is_running(self, job_id: str) -> bool:
        runner = self.runners.get(job_id)
        return runner is not None and runner.running

    def refresh(self, job_id: str) -> Job:
        """Reload a job that another process is running, so its state here is current."""
        job = self.job(job_id)
        if not self.is_running(job_id) and locked_elsewhere(job.path):
            job = Job.load(job.path)
            self._attach(job)
            self.index.put_job(job.data())
        return job

    def create_job(self, image: bytes, filename: str, spec: JobSpec, name: str | None = None,
                   start: bool = True) -> Job:
        with tempfile.TemporaryDirectory(prefix="giro-upload-") as tmp:
            path = Path(tmp) / (Path(filename).name or "image.png")
            path.write_bytes(image)
            spec.image = str(path)
            job = Job.create(self.root, spec, name)
        # The upload is gone; the job's own copy is the image from now on.
        job.spec.image = str((job.path / "input" / "source.png").resolve())
        self._attach(job)
        if not start:
            job.status = "draft"
        job.save()
        if start:
            self._run(job)
        return job

    def start_job(self, job_id: str, use_edit: bool) -> None:
        """Start a draft job's orbits, from the edited image or the original."""
        job = self.job(job_id)
        if job.status != "draft":
            raise ActionError("the job has already been started")
        if job.edit.get("status") == "running":
            raise ActionError("wait for the edit to finish")
        if use_edit and not (job.path / "input" / "edited.png").exists():
            raise ActionError("there is no edited image to use")
        job.spec.use_edit = use_edit
        self._run(job)

    def edit(self, job_id: str, prompt: str, seed: int | None, fast: bool) -> None:
        """Edit a draft job's image (background swap); the result is input/edited.png."""
        job = self.job(job_id)
        if job.status != "draft":
            raise ActionError("edit the image before starting the orbits")
        if job.edit.get("status") == "running":
            raise ActionError("an edit is already running")
        if not prompt.strip():
            raise ActionError("describe the edit")
        seed = seed if seed is not None else random.randrange(2**32)
        job.edit = {"status": "running", "prompt": prompt, "seed": seed, "fast": fast}
        job.save()
        self.edit_tasks[job_id] = asyncio.create_task(self._drive_edit(job, {"prompt": prompt, "seed": seed, "fast": fast}))

    async def _drive_edit(self, job: Job, params: dict[str, Any]) -> None:
        t0 = time.monotonic()
        stage = stages.EDIT
        root = job.path.resolve()
        ctx = stages.Ctx(
            on_progress=lambda st, frac, msg: self.publish(
                {"type": "progress", "job": job.id, "seed": None, "stage": st, "frac": round(frac, 4), "msg": msg}),
            on_preview=lambda st, p: self.publish(
                {"type": "preview", "job": job.id, "seed": None, "stage": st, "path": str(Path(p).resolve().relative_to(root))}),
            on_start=lambda st: self.publish({"type": "start", "job": job.id, "seed": None, "stage": st}),
            is_cancelled=lambda: job.id not in self.jobs,  # deleted meanwhile
        )
        try:
            gpus = list(dict.fromkeys(job.spec.video_gpus + job.spec.post_gpus))
            # Someone is waiting on this: it goes ahead of queued pipeline stages.
            async with self.pool.lease(gpus, priority=-100) as gpu:
                ctx.gpu = gpu
                await asyncio.to_thread(stage.execute, job.path, params, ctx)
            job.edit |= {"status": "done", "seconds": round(time.monotonic() - t0, 1)}
        except stages.StageFailed as e:
            job.edit |= {"status": "error", "error": str(e)}
        except stages.Cancelled:
            return
        except Exception as e:  # noqa: BLE001 - reported on the job
            job.edit |= {"status": "error", "error": f"{type(e).__name__}: {e}"}
        finally:
            self.edit_tasks.pop(job.id, None)
        if job.id in self.jobs:
            job.save()

    def _run(self, job: Job) -> None:
        if self.is_running(job.id):
            return
        if locked_elsewhere(job.path):
            raise ActionError("another giro process (a CLI run?) is running this job")
        cancel = threading.Event()
        log_file = job.path / "job.log"

        def log(msg: str) -> None:
            with open(log_file, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

        runner = Runner(job, log, self.pool, cancel, on_event=self.publish)
        self.runners[job.id], self.cancels[job.id] = runner, cancel
        runner.running = True  # from now, not only once run() starts: actions must join this run
        self.run_tasks[job.id] = asyncio.create_task(self._drive(job, runner))

    async def _drive(self, job: Job, runner: Runner) -> None:
        try:
            await runner.run()
        except JobLocked as e:  # lost a race with another process
            runner.log(str(e))
        except Exception as e:  # noqa: BLE001 - a bug in the runner must not take the server down
            runner.running = False
            job.status = "error"
            runner.log(f"runner crashed: {type(e).__name__}: {e}")
            job.save()
        finally:
            self.run_tasks.pop(job.id, None)
            if not self.run_tasks:
                await self._stop_idle_comfy()

    async def _stop_idle_comfy(self) -> None:
        """Stop the ComfyUI instances this server started, unless something is using them."""
        def stop() -> None:
            for g in self.pool.gpus:
                if g in server.PORTS:
                    server.stop_if_idle(g)
        await asyncio.to_thread(stop)

    def resume(self, job_id: str) -> None:
        job = self.job(job_id)
        if self.is_running(job_id):
            raise ActionError("the job is already running")
        self._run(job)

    def cancel(self, job_id: str) -> None:
        if not self.is_running(job_id):
            raise ActionError("the job is not running")
        self.cancels[job_id].set()

    def add_seeds(self, job_id: str, n: int) -> None:
        """Aim for n more passing attempts, with the budget raised to match."""
        job = self.job(job_id)
        job.spec.want += n
        job.spec.max_attempts += n
        job.save()
        self._run(job)

    def delete(self, job_id: str) -> None:
        job = self.job(job_id)
        if self.is_running(job_id):
            raise ActionError("cancel the job first")
        if job_id in self.edit_tasks:
            raise ActionError("wait for the edit to finish")
        shutil.rmtree(job.path)
        del self.jobs[job_id]
        self.runners.pop(job_id, None)
        self.cancels.pop(job_id, None)
        self.index.drop_missing(set(self.jobs))
        self.publish({"type": "deleted", "job": job_id})

    # ---- attempts --------------------------------------------------------------------------

    def _requeue(self, job: Job, seed: int, change: Callable[[], None] | None = None) -> None:
        attempt = job.attempt(seed)
        runner = self.runners.get(job.id)
        if runner and runner.running and runner.is_active(attempt):
            raise ActionError(f"seed {seed} is still running")
        if change:
            change()
        if runner and runner.running:
            runner.launch(attempt)
            job.save()
        else:
            attempt.status, attempt.reason = "queued", ""
            job.save()
            self._run(job)

    def retry(self, job_id: str, seed: int) -> None:
        """Run an attempt again; finished stages are skipped, so this redoes what failed."""
        job = self.job(job_id)
        if job.attempt(seed).status not in TERMINAL | {"cancelled"}:
            raise ActionError(f"seed {seed} has not finished")
        self._requeue(job, seed)

    def set_params(self, job_id: str, seed: int, params: dict[str, dict[str, Any]]) -> None:
        """Change stage params for one attempt and rerun what they affect, e.g. the crop's tau."""
        job = self.job(job_id)
        for stage, values in params.items():
            if stage not in stages.BY_NAME or stage == stages.ORBIT.name:
                raise ActionError(f"no per-attempt params for stage {stage!r}")
            unknown = set(values) - set(stages.BY_NAME[stage].defaults)
            if unknown:
                raise ActionError(f"{stage} has no params {sorted(unknown)}")
        attempt = job.attempt(seed)

        def change() -> None:
            for stage, values in params.items():
                attempt.params[stage] = attempt.params.get(stage, {}) | values
        self._requeue(job, seed, change)

    def override(self, job_id: str, seed: int, verdict: str) -> None:
        """The user's verdict over the gate's: "pass" trains a rejected orbit anyway,
        "fail" drops a passing one from the ranking (a new seed replaces it)."""
        job = self.job(job_id)
        attempt = job.attempt(seed)
        if verdict == "pass":
            if attempt.status != "rejected" or attempt.stage != "gate":
                raise ActionError("only an attempt the gate rejected can be passed by hand")

            def change() -> None:
                attempt.override = "pass"
            self._requeue(job, seed, change)
        elif verdict == "fail":
            if attempt.status != "passed":
                raise ActionError("only a passing attempt can be failed by hand")
            attempt.override, attempt.status, attempt.reason = "fail", "rejected", "rejected by you"
            job.rank()
            job.save()
            self._run(job)  # replaces it with a new seed if the budget allows
        else:
            raise ActionError(f"verdict must be 'pass' or 'fail', not {verdict!r}")

    def discard(self, job_id: str, seed: int) -> None:
        """Drop an attempt (stopping it if it runs); a new seed replaces it while the budget lasts."""
        job = self.job(job_id)
        attempt = job.attempt(seed)
        runner = self.runners.get(job_id)
        if runner and runner.running and runner.is_active(attempt):
            runner.stop(attempt)
            return
        attempt.status, attempt.reason = "discarded", "discarded by you"
        job.rank()
        job.save()
        self._run(job)
