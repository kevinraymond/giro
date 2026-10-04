"""giro's HTTP and WebSocket API, served by `giro serve`.

    GET  /api/jobs                                  every job, newest first
    POST /api/jobs                                  multipart: image + spec (JSON) -> the new job, started
                                                    (or a draft with spec.start = false)
    GET  /api/jobs/{id}                             job.json
    POST /api/jobs/{id}/edit           {prompt, seed?, fast}   draft jobs: background edit -> input/edited.png
    POST /api/jobs/{id}/start          {use_edit}   start a draft job's orbits
    POST /api/jobs/{id}/resume | cancel             run again (finished stages are skipped) / stop
    POST /api/jobs/{id}/seeds          {n}          aim for n more passing attempts
    DELETE /api/jobs/{id}
    GET  /api/jobs/{id}/attempts/{seed}             attempt, metrics, gate verdict, artifacts
    GET  /api/jobs/{id}/attempts/{seed}/events      metric / preview / log history (?types=metric)
    POST /api/jobs/{id}/attempts/{seed}/override    {verdict: pass|fail}
    POST /api/jobs/{id}/attempts/{seed}/discard | retry
    POST /api/jobs/{id}/attempts/{seed}/params      {stage: {key: value}}: e.g. crop tau, height_m
    GET  /api/jobs/{id}/attempts/{seed}/frames      every extracted frame: kept, posed, azimuth, error
    GET  /api/jobs/{id}/attempts/{seed}/frames.jpg  the frames as one thumbnail strip
    GET  /api/jobs/{id}/attempts/{seed}/cameras     cameras and sparse points in the viewer frame
    GET  /api/jobs/{id}/attempts/{seed}/training    PSNR and splat count by iteration, checkpoints
    GET  /api/jobs/{id}/thumb                       small JPEG of the job's image
    GET  /api/jobs/{id}/export.zip                  offline HTML report + splats (PLY, SOG, SPZ) of the passing attempts
    GET  /api/stages                                stage names and default params
    GET  /files/{id}/{path}                         anything in a job directory
    POST /api/xr/stats | GET /api/xr/stats          frame-time reports from the VR page (data/xr/stats.jsonl)
    GET  /bench/{file}                              test splats for VR measurements (data/bench)
    WS   /api/events                                hello snapshot, then job/progress/metric/preview/log/start

The UI (ui/dist) is served at / when it has been built.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError

from giro import inspection, report, stages
from giro.index import Index
from giro.job import JobSpec, locked_elsewhere
from giro.manager import ActionError, Manager

UI_DIST = Path(__file__).resolve().parents[2] / "ui" / "dist"
MAX_UPLOAD = 64 * 2**20

# Artifacts an attempt may have, relative to its directory, in pipeline order.
ARTIFACTS = [
    "hero/hero.png", "video.mp4", "gate.json", "crop/preview_hero.jpg", "canonical/turnaround.jpg",
    "canonical/transform.json", "export/splat.ply", "export/splat.sog", "export/splat.spz", "export/splat.vr.sog",
]


class NewJob(BaseModel):
    """What the new-job form sends besides the image."""
    name: str | None = Field(None, pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    want: int = Field(3, ge=1, le=12)
    max_attempts: int = Field(6, ge=1, le=32)
    seeds: list[int] = []
    height_m: float | None = Field(None, gt=0.01, le=100)
    subject: str | None = None
    orbit: dict[str, Any] = {}                 # orbit_video params (length, steps, width, height, prompt)
    crop: list[float] | None = Field(None, min_length=4, max_length=4)  # (left, top, right, bottom) of the image, 0-1
    params: dict[str, dict[str, Any]] = {}     # per-stage overrides
    video_gpus: list[int] = [1]
    post_gpus: list[int] = [0, 1]
    start: bool = True  # False: a draft, to try background edits before starting (POST .../start)

    def spec(self) -> JobSpec:
        params = {k: dict(v) for k, v in self.params.items()}
        if self.height_m is not None:
            params.setdefault("canonicalize", {})["height_m"] = self.height_m
        if self.subject:
            params.setdefault("masks", {})["subject_prompt"] = self.subject
        for stage, values in params.items():
            if stage not in stages.BY_NAME:
                raise ValueError(f"unknown stage {stage!r}")
            known = stages.BY_NAME[stage]
            unknown = set(values) - set(known.defaults) - set(known.extra_params)
            if unknown:
                raise ValueError(f"{stage} has no params {sorted(unknown)}")
        orbit = {k: v for k, v in self.orbit.items() if k != "seed"}
        unknown = set(orbit) - set(stages.ORBIT.defaults) - set(stages.ORBIT.extra_params)
        if unknown:
            raise ValueError(f"orbit_video has no params {sorted(unknown)}")
        for side in ("width", "height"):  # the video model's grid (MiniMaxH3ImageToVideo: step 32)
            v = orbit.get(side)
            if v is not None and (not isinstance(v, int) or v % 32 or not 256 <= v <= 2048):
                raise ValueError(f"orbit_video {side} must be a multiple of 32 from 256 to 2048, not {v!r}")
        if self.crop is not None:
            left, top, right, bottom = self.crop
            if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError(f"crop must be (left, top, right, bottom) within 0-1, not {self.crop}")
        return JobSpec(image="", want=self.want, max_attempts=max(self.max_attempts, self.want), seeds=self.seeds,
                       orbit=orbit, params=params, crop=self.crop,
                       video_gpus=self.video_gpus, post_gpus=self.post_gpus)


class Seeds(BaseModel):
    n: int = Field(1, ge=1, le=12)


class Verdict(BaseModel):
    verdict: str


class EditRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    seed: int | None = Field(None, ge=0, lt=2**63)
    fast: bool = True


class StartRequest(BaseModel):
    use_edit: bool = False


def create_app(root: Path, db: Path, gpus: list[int]) -> FastAPI:
    manager = Manager(root, Index(db), gpus)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await manager.start()
        yield
        await manager.stop()

    app = FastAPI(title="giro", lifespan=lifespan)
    app.state.manager = manager

    @app.exception_handler(KeyError)
    async def not_found(request: Request, e: KeyError) -> JSONResponse:
        return JSONResponse({"detail": str(e.args[0]) if e.args else "not found"}, status_code=404)

    @app.exception_handler(ActionError)
    async def conflict(request: Request, e: ActionError) -> JSONResponse:
        return JSONResponse({"detail": str(e)}, status_code=409)

    def attempt_path(job_id: str, seed: int) -> Path:
        job = manager.job(job_id)
        job.attempt(seed)
        return job.attempt_dir(seed)

    # ---- jobs ------------------------------------------------------------------------------

    @app.get("/api/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        return [manager.jobs[j].data() for j in sorted(manager.jobs, reverse=True)]

    @app.post("/api/jobs", status_code=201)
    async def create_job(image: UploadFile = File(...), spec: str = Form("{}")) -> dict[str, Any]:
        data = await image.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"image larger than {MAX_UPLOAD // 2**20} MB")
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
        except (UnidentifiedImageError, OSError):
            raise HTTPException(422, "not an image giro can read") from None
        try:
            request = NewJob.model_validate(json.loads(spec))
            job_spec = request.spec()
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            raise HTTPException(422, str(e)) from None
        bad = set(job_spec.video_gpus + job_spec.post_gpus) - set(manager.pool.gpus)
        if bad:
            raise HTTPException(422, f"GPUs {sorted(bad)} are not in the server's pool {list(manager.pool.gpus)}")
        name = request.name or Path(image.filename or "image").stem[:64] or "image"
        job = manager.create_job(data, image.filename or "image.png", job_spec, name, start=request.start)
        return job.data()

    @app.post("/api/jobs/{job_id}/edit")
    async def edit_image(job_id: str, body: EditRequest) -> dict[str, Any]:
        manager.edit(job_id, body.prompt, body.seed, body.fast)
        return manager.job(job_id).data()

    @app.post("/api/jobs/{job_id}/start")
    async def start_job(job_id: str, body: StartRequest) -> dict[str, Any]:
        manager.start_job(job_id, body.use_edit)
        return manager.job(job_id).data()

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = manager.refresh(job_id)
        return job.data() | {"running": manager.is_running(job_id), "elsewhere": locked_elsewhere(job.path)}

    @app.post("/api/jobs/{job_id}/resume")
    async def resume(job_id: str) -> dict[str, Any]:
        manager.resume(job_id)
        return manager.job(job_id).data()

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel(job_id: str) -> dict[str, Any]:
        manager.cancel(job_id)
        return manager.job(job_id).data()

    @app.post("/api/jobs/{job_id}/seeds")
    async def add_seeds(job_id: str, body: Seeds) -> dict[str, Any]:
        manager.add_seeds(job_id, body.n)
        return manager.job(job_id).data()

    @app.delete("/api/jobs/{job_id}", status_code=204)
    async def delete_job(job_id: str) -> None:
        manager.delete(job_id)

    # ---- attempts --------------------------------------------------------------------------

    @app.get("/api/jobs/{job_id}/attempts/{seed}")
    async def get_attempt(job_id: str, seed: int) -> dict[str, Any]:
        job = manager.job(job_id)
        attempt = job.attempt(seed)
        path = job.attempt_dir(seed)

        def read(name: str) -> Any:
            f = path / name
            return json.loads(f.read_text()) if f.exists() else None

        files = {name: (path / name).stat().st_size for name in ARTIFACTS if (path / name).exists()}
        previews = sorted(p.name for p in (path / "previews").glob("*")) if (path / "previews").is_dir() else []
        checkpoints = sorted(p.name for p in (path / "train").glob("export_*.ply")) if (path / "train").is_dir() else []
        return {
            "job": job_id, "attempt": job.data()["attempts"][job.attempts.index(attempt)],
            "rank": job.ranking.index(seed) + 1 if seed in job.ranking else None,
            "running": (runner := manager.runners.get(job_id)) is not None and runner.is_active(attempt),
            "metrics": read("metrics.json") or {}, "gate": read("gate.json"), "dedup": read("dedup.json"),
            "files": files, "previews": previews, "checkpoints": checkpoints,
        }

    @app.get("/api/jobs/{job_id}/attempts/{seed}/events")
    async def attempt_events(job_id: str, seed: int, types: str = "", since: float = 0.0) -> list[dict[str, Any]]:
        attempt_path(job_id, seed)
        return manager.index.events(job_id, seed, [t for t in types.split(",") if t] or None, since)

    @app.post("/api/jobs/{job_id}/attempts/{seed}/override")
    async def override(job_id: str, seed: int, body: Verdict) -> dict[str, Any]:
        manager.override(job_id, seed, body.verdict)
        return manager.job(job_id).data()

    @app.post("/api/jobs/{job_id}/attempts/{seed}/discard")
    async def discard(job_id: str, seed: int) -> dict[str, Any]:
        manager.discard(job_id, seed)
        return manager.job(job_id).data()

    @app.post("/api/jobs/{job_id}/attempts/{seed}/retry")
    async def retry(job_id: str, seed: int) -> dict[str, Any]:
        manager.retry(job_id, seed)
        return manager.job(job_id).data()

    @app.post("/api/jobs/{job_id}/attempts/{seed}/params")
    async def set_params(job_id: str, seed: int, body: dict[str, dict[str, Any]]) -> dict[str, Any]:
        manager.set_params(job_id, seed, body)
        return manager.job(job_id).data()

    @app.get("/api/jobs/{job_id}/attempts/{seed}/frames")
    async def attempt_frames(job_id: str, seed: int) -> dict[str, Any]:
        return await asyncio.to_thread(inspection.frames, attempt_path(job_id, seed))

    @app.get("/api/jobs/{job_id}/attempts/{seed}/frames.jpg")
    async def attempt_frame_sheet(job_id: str, seed: int) -> FileResponse:
        sheet = await asyncio.to_thread(inspection.frame_sheet, attempt_path(job_id, seed))
        if sheet is None:
            raise HTTPException(404, "no frames yet")
        return FileResponse(sheet, headers={"Cache-Control": "no-cache"})

    @app.get("/api/jobs/{job_id}/attempts/{seed}/training")
    async def attempt_training(job_id: str, seed: int) -> dict[str, Any]:
        return await asyncio.to_thread(inspection.training, attempt_path(job_id, seed))

    @app.get("/api/jobs/{job_id}/attempts/{seed}/cameras")
    async def attempt_cameras(job_id: str, seed: int) -> dict[str, Any]:
        data = await asyncio.to_thread(inspection.cameras, attempt_path(job_id, seed))
        if data is None:
            raise HTTPException(404, "no camera poses yet")
        return data

    @app.get("/api/jobs/{job_id}/export.zip")
    async def export_job(job_id: str) -> FileResponse:
        """The job as one zip (giro/report.py), built when asked: attempts may have changed since."""
        job = manager.job(job_id)
        out = job.path / "archive" / f"{job.path.name}.zip"
        await asyncio.to_thread(report.build, job, out)
        return FileResponse(out, filename=out.name, media_type="application/zip")

    @app.get("/api/jobs/{job_id}/thumb")
    async def thumb(job_id: str) -> FileResponse:
        """A small JPEG of the image the job orbits (the edited one if it uses it), made once."""
        job = manager.job(job_id)
        source = job.hero_source()
        out = job.path / "input" / f"thumb-{source.stem}.jpg"
        if not out.exists():
            def make() -> None:
                with Image.open(source) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.thumbnail((320, 320))
                    tmp = out.with_suffix(".tmp.jpg")
                    im.save(tmp, quality=85)
                    tmp.replace(out)
            await asyncio.to_thread(make)
        return FileResponse(out, headers={"Cache-Control": "no-cache"})  # changes when a draft starts from its edit

    xr_log = root.parent / "xr" / "stats.jsonl"

    @app.post("/api/xr/stats", status_code=204)
    async def xr_stats(body: dict[str, Any]) -> None:
        """A VR page's frame times (VrView.tsx), kept for measuring the Quest's budget."""
        xr_log.parent.mkdir(parents=True, exist_ok=True)
        with open(xr_log, "a") as f:
            f.write(json.dumps({"ts": time.time()} | body) + "\n")
        manager.publish({"type": "xr_stats", "job": "", **body})

    @app.get("/api/xr/stats")
    async def xr_stats_list(since: float = 0.0) -> list[dict[str, Any]]:
        if not xr_log.exists():
            return []
        rows = [json.loads(line) for line in xr_log.read_text().splitlines() if line.strip()]
        return [r for r in rows if r["ts"] > since]

    @app.get("/api/stages")
    async def list_stages() -> list[dict[str, Any]]:
        return [{"name": s.name, "defaults": s.defaults, "gpu": bool(s.gpu_mb)} for s in stages.PIPELINE]

    # ---- files and events ------------------------------------------------------------------

    @app.get("/files/{job_id}/{rel:path}")
    async def files(job_id: str, rel: str) -> FileResponse:
        base = manager.job(job_id).path.resolve()
        path = (base / rel).resolve()
        if not path.is_relative_to(base) or not path.is_file():
            raise HTTPException(404, "no such file")
        # Checkpoints and previews are rewritten in place: never let the browser keep a stale one.
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.websocket("/api/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        queue = manager.subscribe()

        async def send() -> None:
            await ws.send_json(manager.snapshot())
            while True:
                await ws.send_json(await queue.get())

        async def until_closed() -> None:
            # Without this, a quiet stream never notices the client left, and the handler
            # (waiting on the queue) keeps the server from shutting down.
            while (await ws.receive())["type"] != "websocket.disconnect":
                pass

        tasks = [asyncio.create_task(send()), asyncio.create_task(until_closed())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            manager.unsubscribe(queue)

    bench = root.parent / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    app.mount("/bench", StaticFiles(directory=bench), name="bench")
    if UI_DIST.is_dir():
        app.mount("/", StaticFiles(directory=UI_DIST, html=True), name="ui")
    return app
