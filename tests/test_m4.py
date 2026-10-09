import io
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from giro import stages
from giro.api import create_app
from giro.comfy import server
from giro.job import Attempt, Job, JobSpec
from giro.stages import Rejected, Stage


@pytest.fixture(autouse=True)
def _h3_default(monkeypatch):
    """These tests drive the job runner with fake H3-shaped stages; new jobs would otherwise get the
    proxy orbit, whose real proxy stage needs ComfyUI on a GPU."""
    monkeypatch.setattr(stages, "DEFAULT_MODEL", "h3")

RUNS: list[tuple[int, str]] = []  # (seed, stage) of every stage that actually ran


class _Orbit(Stage):
    name = "orbit_video"
    defaults = {"seed": None, "width": 8, "height": 8}
    gpu_mb = 1
    outputs = ("video.mp4",)

    def run(self, attempt, params, ctx):
        RUNS.append((params["seed"], self.name))
        for i in range(4):
            ctx.progress(i / 4, f"step {i}")
        ctx.preview(attempt / "video.mp4")
        (attempt / "video.mp4").write_text(str(params["seed"]))


class _Gate(Stage):
    """Rejects odd seeds."""
    name = "gate"
    defaults = {"enforce": True}
    inputs = ("video.mp4",)
    outputs = ("gate.json",)

    def run(self, attempt, params, ctx):
        seed = int((attempt / "video.mp4").read_text())
        RUNS.append((seed, self.name))
        ctx.metric("azimuth_coverage", 356.0 if seed % 2 == 0 else 180.0)
        (attempt / "gate.json").write_text(json.dumps({"passed": seed % 2 == 0}))
        if seed % 2 and params["enforce"]:
            raise Rejected("the camera covers only 180 degrees around the subject (need 330)")


class _Train(Stage):
    name = "train"
    gpu_mb = 1
    inputs = ("gate.json",)
    outputs = ("final.ply",)
    hold = threading.Event()  # set by default; a test clears it to keep a seed in training

    def run(self, attempt, params, ctx):
        seed = int((attempt / "video.mp4").read_text())
        RUNS.append((seed, self.name))
        while not self.hold.wait(0.02):
            ctx.check_cancelled()
        ctx.metric("eval_psnr", 20.0 + seed)
        (attempt / "final.ply").write_text("ply")


class _Crop(Stage):
    name = "crop"
    defaults = {"tau": 0.8}
    inputs = ("final.ply",)
    outputs = ("crop.json",)

    def run(self, attempt, params, ctx):
        RUNS.append((int((attempt / "video.mp4").read_text()), self.name))
        ctx.metric("tau", params["tau"])
        (attempt / "crop.json").write_text(json.dumps(params))


class _Canonicalize(Stage):
    name = "canonicalize"
    defaults = {"height_m": 1.7}
    inputs = ("crop.json",)
    outputs = ("canonical.json",)

    def run(self, attempt, params, ctx):
        (attempt / "canonical.json").write_text(json.dumps(params))


def _pipeline():
    orbit = _Orbit()
    return orbit, [orbit, _Gate(), _Train(), _Crop(), _Canonicalize()]


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    RUNS.clear()
    _Train.hold.set()
    orbit, pipeline = _pipeline()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", pipeline)
    monkeypatch.setattr(stages, "BY_NAME", {s.name: s for s in pipeline})
    monkeypatch.setattr(server, "is_up", lambda g: False)
    app = create_app(tmp_path / "jobs", tmp_path / "giro.sqlite", [0, 1])
    with TestClient(app) as c:
        c.root = tmp_path
        yield c


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (24, 32), "green").save(buf, format="PNG")
    return buf.getvalue()


def _start(client, **spec) -> dict:
    r = client.post("/api/jobs", files={"image": ("hero.png", _png(), "image/png")}, data={"spec": json.dumps(spec)})
    assert r.status_code == 201, r.text
    return r.json()


def _wait(client, job_id: str, until=lambda job: not job["running"], timeout=10.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if until(job):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job did not settle: {job}")


def _by_seed(job: dict) -> dict[int, dict]:
    return {a["seed"]: a for a in job["attempts"]}


def test_job_runs_through_the_api_and_streams_events(client):
    with client.websocket_connect("/api/events") as ws:
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["jobs"] == []
        job = _start(client, want=2, max_attempts=4, seeds=[1, 2, 3, 4], height_m=1.2)
        seen = []
        while not any(e["type"] == "job" and e["data"]["status"] == "done" for e in seen):
            seen.append(ws.receive_json())
    job = _wait(client, job["id"])
    assert job["status"] == "done" and job["ranking"] == [4, 2]
    assert job["spec"]["params"]["canonicalize"] == {"height_m": 1.2}
    kinds = {e["type"] for e in seen}
    assert {"job", "progress", "metric", "preview", "start"} <= kinds
    preview = next(e for e in seen if e["type"] == "preview" and e["seed"] == 2)
    assert preview["path"] == "attempts/2/video.mp4"
    assert client.get(f"/files/{job['id']}/{preview['path']}").text == "2"

    history = client.get(f"/api/jobs/{job['id']}/attempts/4/events", params={"types": "metric"}).json()
    assert {(e["stage"], e["name"]) for e in history} == {("gate", "azimuth_coverage"), ("train", "eval_psnr"),
                                                          ("crop", "tau")}
    detail = client.get(f"/api/jobs/{job['id']}/attempts/4").json()
    assert detail["rank"] == 1 and detail["metrics"]["train"]["eval_psnr"] == 24.0
    assert client.get("/api/jobs").json()[0]["id"] == job["id"]


def test_new_params_rerun_only_the_stages_they_affect(client):
    job = _wait(client, _start(client, want=1, max_attempts=1, seeds=[2])["id"])
    RUNS.clear()
    r = client.post(f"/api/jobs/{job['id']}/attempts/2/params", json={"crop": {"tau": 0.5}})
    assert r.status_code == 200, r.text
    _wait(client, job["id"])
    assert RUNS == [(2, "crop")]
    assert client.get(f"/api/jobs/{job['id']}/attempts/2").json()["metrics"]["crop"]["tau"] == 0.5
    # Only the new run's metric is kept for the stage.
    history = client.get(f"/api/jobs/{job['id']}/attempts/2/events", params={"types": "metric"}).json()
    assert [e["value"] for e in history if e["stage"] == "crop"] == [0.5]
    bad = client.post(f"/api/jobs/{job['id']}/attempts/2/params", json={"crop": {"nope": 1}})
    assert bad.status_code == 409 and "nope" in bad.json()["detail"]


def test_override_trains_a_rejected_orbit_and_fail_rerolls(client):
    job = _wait(client, _start(client, want=1, max_attempts=2, seeds=[1, 2])["id"])
    assert _by_seed(job)[1]["status"] == "rejected" and job["ranking"] == [2]

    assert client.post(f"/api/jobs/{job['id']}/attempts/1/override", json={"verdict": "pass"}).status_code == 200
    job = _wait(client, job["id"])
    assert _by_seed(job)[1]["status"] == "passed" and job["ranking"] == [2, 1]

    assert client.post(f"/api/jobs/{job['id']}/attempts/2/override", json={"verdict": "fail"}).status_code == 200
    job = _wait(client, job["id"])
    assert _by_seed(job)[2]["status"] == "rejected" and _by_seed(job)[2]["reason"] == "rejected by you"
    assert job["ranking"] == [1]
    assert len(job["attempts"]) == 2  # budget spent: no reroll
    r = client.post(f"/api/jobs/{job['id']}/attempts/2/override", json={"verdict": "pass"})
    assert r.status_code == 409  # rejected by the user, not by the gate


def test_discard_stops_a_running_attempt_and_rerolls(client):
    _Train.hold.clear()
    job = _start(client, want=1, max_attempts=2, seeds=[2, 4])
    _wait(client, job["id"], until=lambda j: _by_seed(j)[2]["stage"] == "train")
    assert client.post(f"/api/jobs/{job['id']}/attempts/2/discard").status_code == 200
    _wait(client, job["id"], until=lambda j: 4 in _by_seed(j))
    _Train.hold.set()
    job = _wait(client, job["id"])
    assert _by_seed(job)[2]["status"] == "discarded"
    assert job["ranking"] == [4] and job["status"] == "done"


def test_add_seeds_and_errors(client):
    job = _wait(client, _start(client, want=1, max_attempts=1, seeds=[2, 4])["id"])
    assert client.post(f"/api/jobs/{job['id']}/seeds", json={"n": 1}).status_code == 200
    job = _wait(client, job["id"])
    assert job["ranking"] == [4, 2] and job["spec"]["want"] == 2

    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get(f"/api/jobs/{job['id']}/attempts/99").status_code == 404
    assert client.get(f"/files/{job['id']}/../../giro.sqlite").status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/cancel").status_code == 409
    r = client.post("/api/jobs", files={"image": ("x.png", b"not an image", "image/png")}, data={"spec": "{}"})
    assert r.status_code == 422
    r = client.post("/api/jobs", files={"image": ("x.png", _png(), "image/png")},
                    data={"spec": json.dumps({"params": {"crop": {"nope": 1}}})})
    assert r.status_code == 422
    assert client.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert client.get("/api/jobs").json() == []


def test_a_job_left_running_by_a_dead_server_is_interrupted_and_resumes(tmp_path, monkeypatch):
    RUNS.clear()
    _Train.hold.set()
    orbit, pipeline = _pipeline()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", pipeline)
    monkeypatch.setattr(stages, "BY_NAME", {s.name: s for s in pipeline})
    monkeypatch.setattr(server, "is_up", lambda g: False)
    image = tmp_path / "hero.png"
    image.write_bytes(_png())
    job = Job.create(tmp_path / "jobs", JobSpec(image=str(image), want=1, max_attempts=1, seeds=[2]), "t")
    job.attempts = [Attempt(2, status="running", stage="train")]
    job.save()

    with TestClient(create_app(tmp_path / "jobs", tmp_path / "giro.sqlite", [0, 1])) as client:
        data = client.get(f"/api/jobs/{job.id}").json()
        assert data["status"] == "interrupted" and data["attempts"][0]["status"] == "cancelled"
        assert client.post(f"/api/jobs/{job.id}/resume").status_code == 200
        data = _wait(client, job.id)
        assert data["status"] == "done" and data["ranking"] == [2]


def test_a_job_another_process_runs_is_left_alone(tmp_path, monkeypatch):
    import fcntl

    orbit, pipeline = _pipeline()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", pipeline)
    monkeypatch.setattr(stages, "BY_NAME", {s.name: s for s in pipeline})
    image = tmp_path / "hero.png"
    image.write_bytes(_png())
    job = Job.create(tmp_path / "jobs", JobSpec(image=str(image), want=1, max_attempts=1, seeds=[2]), "t")
    job.attempts = [Attempt(2, status="running", stage="train")]
    job.save()
    with open(job.path / "job.lock", "w") as held:  # as a CLI 'giro job' would
        fcntl.flock(held, fcntl.LOCK_EX)
        with TestClient(create_app(tmp_path / "jobs", tmp_path / "giro.sqlite", [0, 1])) as client:
            data = client.get(f"/api/jobs/{job.id}").json()
            assert data["status"] == "running" and data["elsewhere"] and not data["running"]
            r = client.post(f"/api/jobs/{job.id}/resume")
            assert r.status_code == 409 and "another giro process" in r.json()["detail"]


def test_comfy_is_stopped_only_by_its_starter_and_only_when_unused(tmp_path, monkeypatch):
    import os

    from giro.comfy import server as srv

    stopped = []
    monkeypatch.setattr(srv, "RUN_DIR", tmp_path)
    monkeypatch.setattr(srv, "start", lambda gpu, timeout=120.0: "http://127.0.0.1:1")
    monkeypatch.setattr(srv, "stop", lambda gpu: stopped.append(gpu) or True)

    (tmp_path / "comfy-gpu0.owner").write_text("1")  # another process started it
    assert not srv.stop_if_idle(0)
    (tmp_path / "comfy-gpu0.owner").write_text(str(os.getpid()))
    with srv.Lease(0) as lease:  # in use (by any process)
        assert lease.url == "http://127.0.0.1:1"
        assert not srv.stop_if_idle(0)
    assert stopped == []
    assert srv.stop_if_idle(0) and stopped == [0]
    assert not (tmp_path / "comfy-gpu0.owner").exists()


class _Edit(Stage):
    """Paints the image red, like a background swap would change it."""
    name = "edit"
    defaults = {"prompt": "", "negative": "", "seed": None, "fast": True}
    inputs = ("input/source.png",)
    outputs = ("input/edited.png",)
    gpu_mb = 1

    def run(self, job_dir, params, ctx):
        ctx.progress(0.5, "step 2/4")
        Image.new("RGB", (24, 32), "red").save(job_dir / "input" / "edited.png")


def test_draft_edit_then_start_from_the_edited_image(client, monkeypatch):
    monkeypatch.setattr(stages, "EDIT", _Edit())
    job = _start(client, want=1, max_attempts=1, seeds=[2], start=False)
    assert job["status"] == "draft" and job["attempts"] == []
    assert client.post(f"/api/jobs/{job['id']}/start", json={"use_edit": True}).status_code == 409  # nothing edited yet

    r = client.post(f"/api/jobs/{job['id']}/edit", json={"prompt": "plain studio backdrop", "seed": 7})
    assert r.status_code == 200, r.text
    data = _wait(client, job["id"], until=lambda j: j["edit"].get("status") != "running")
    assert data["edit"] == {"status": "done", "prompt": "plain studio backdrop", "seed": 7, "fast": True,
                            "seconds": data["edit"]["seconds"]}
    assert client.get(f"/files/{job['id']}/input/edited.png").status_code == 200

    assert client.post(f"/api/jobs/{job['id']}/start", json={"use_edit": True}).status_code == 200
    data = _wait(client, job["id"])
    assert data["status"] == "done" and data["spec"]["use_edit"]
    hero = Image.open(client.root / "jobs" / job["id"] / "attempts" / "2" / "hero" / "hero.png")
    assert hero.getpixel((0, 0)) == (255, 0, 0)  # the orbit started from the edited image
    r = client.post(f"/api/jobs/{job['id']}/edit", json={"prompt": "again"})
    assert r.status_code == 409  # started jobs keep their image


class _FakeComfy:
    """Stands in for ComfyClient: every prompt yields progress, a preview and 3 frames (or one edit)."""

    prompts: list[dict] = []

    def __init__(self, url):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def upload_image(self, path, subfolder="giro"):
        return f"giro/{path.name}"

    async def run(self, prompt):
        from giro.comfy import Done, Preview, Progress

        _FakeComfy.prompts.append(prompt)
        n = 3 if "frames" in prompt else 1
        yield Progress("sampler", 1, 2)
        yield Preview(_png(), "image/png")
        yield Done("p1", {"save": {"images": [{"filename": f"f{i:05d}.png"} for i in range(n)]}})

    async def download(self, image, dest):
        Image.new("RGB", (24, 32), "red").save(dest)
        return dest

    async def interrupt(self):
        pass

    async def free(self, unload_models=True):
        pass


class _FakeLease:
    url = "http://127.0.0.1:1"

    def __init__(self, gpu, timeout=120.0):
        self.gpu = gpu

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def test_orbit_and_edit_stages_talk_to_comfy_end_to_end(tmp_path, monkeypatch):
    """The Comfy-backed stages from upload to their files, with ComfyUI faked (not the GPU)."""
    from giro.stages import edit as edit_mod
    from giro.stages import orbit as orbit_mod

    for mod in (orbit_mod, edit_mod):
        monkeypatch.setattr(mod, "ComfyClient", _FakeComfy)
        monkeypatch.setattr(mod.server, "Lease", _FakeLease)
    _FakeComfy.prompts = []
    attempt = tmp_path / "a"
    (attempt / "hero").mkdir(parents=True)
    Image.new("RGB", (24, 32), "green").save(attempt / "hero" / "hero.png")
    ctx = stages.Ctx(gpu=1)
    orbit_mod.OrbitVideo().execute(attempt, {"seed": 5, "width": 24, "height": 32, "length": 22}, ctx)
    assert sorted(p.name for p in (attempt / "frames_raw").iterdir()) == ["00000.png", "00001.png", "00002.png"]
    assert (attempt / "video.mp4").stat().st_size > 0
    assert json.loads((attempt / "orbit.json").read_text())["gpu"] == 1
    assert ctx.metrics["gpu"] == 1 and ctx.metrics["n_frames"] == 3

    job = tmp_path / "job"
    (job / "input").mkdir(parents=True)
    Image.new("RGB", (24, 32), "green").save(job / "input" / "source.png")
    edit_mod.EditImage().execute(job, {"prompt": "studio backdrop", "seed": 3, "fast": False}, stages.Ctx(gpu=1))
    assert Image.open(job / "input" / "edited.png").getpixel((0, 0)) == (255, 0, 0)
    quality = _FakeComfy.prompts[-1]
    assert "lightning" not in quality and quality["sampler"]["inputs"]["model"] == ["cfgnorm", 0]
    assert quality["sampler"]["inputs"]["steps"] == 40 and quality["positive"]["inputs"]["prompt"] == "studio backdrop"
