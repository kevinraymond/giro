import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from giro import job as job_mod
from giro import stages
from giro.scheduler import GpuPool
from giro.stages import Ctx, Rejected, Stage, StageFailed
from giro.stages.gate import Gate, analyze_ring


@pytest.fixture(autouse=True)
def _h3_default(monkeypatch):
    """These tests drive the job runner with fake H3-shaped stages; new jobs would otherwise get the
    proxy orbit, whose real proxy stage needs ComfyUI on a GPU."""
    monkeypatch.setattr(stages, "DEFAULT_MODEL", "h3")


def _look_at(center: np.ndarray, target=np.zeros(3), up=np.array([0.0, 1.0, 0.0])):
    """World-to-camera (qvec, tvec) for a camera at `center` looking at `target` (COLMAP axes)."""
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    r = np.stack([x, y, z])  # rows: camera axes in world coordinates
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    q = [w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)]
    return q, list(-r @ center)


def _ring(degrees, radius=4.0, offset=np.array([1.0, 0.5, -2.0])):
    poses = [_look_at(offset + radius * np.array([math.cos(math.radians(a)), 0.0, math.sin(math.radians(a))]), offset)
             for a in degrees]
    return [q for q, _ in poses], [t for _, t in poses]


def test_full_orbit_measures_as_full():
    ring = analyze_ring(*_ring(np.linspace(0, 356, 90)))
    assert ring["azimuth_coverage"] == pytest.approx(356, abs=0.5)
    assert ring["azimuth_monotonic"] == 1.0
    assert ring["radius"] == pytest.approx(4.0, rel=1e-3)
    assert ring["loop_closure"] == pytest.approx(2 * math.sin(math.radians(2)), abs=1e-3)
    assert ring["radius_cv"] < 1e-6


def test_partial_orbit_and_back_and_forth():
    partial = analyze_ring(*_ring(np.linspace(0, 220, 80)))
    assert partial["azimuth_coverage"] == pytest.approx(220, abs=0.5)
    assert partial["loop_closure"] > 1.8
    wobble = analyze_ring(*_ring([0, 10, 20, 15, 25, 35, 30, 40, 50, 60]))
    assert wobble["azimuth_monotonic"] == pytest.approx(7 / 9, abs=1e-3)
    # Out 25 degrees and back (seen in the length-158 sweep): nets to 0 but covers 25.
    swing = analyze_ring(*_ring(list(np.linspace(0, 25, 26)) + list(np.linspace(24, 0, 25))))
    assert swing["azimuth_coverage"] == pytest.approx(0, abs=0.5)
    assert swing["azimuth_span"] == pytest.approx(25, abs=0.5)


def _attempt_with_model(tmp_path: Path, degrees) -> Path:
    txt = tmp_path / "poses" / "colmap" / "model_txt"
    txt.mkdir(parents=True)
    names = [f"frames/{i:05d}.png" for i in range(len(degrees))] + ["hero/hero.png"]
    qs, ts = _ring(list(degrees) + [degrees[0]])
    lines = [f"{i + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {n}\n\n" for i, (n, q, t) in enumerate(zip(names, qs, ts))]
    (txt / "images.txt").write_text("".join(lines))
    for n in names:
        (tmp_path / n).parent.mkdir(exist_ok=True)
        Image.new("RGB", (4, 4)).save(tmp_path / n)
    (tmp_path / "metrics.json").write_text(json.dumps({"poses_colmap": {"reproj_err": 0.9}}))
    return tmp_path


def test_gate_passes_orbit_and_rejects_half_orbit_with_reason(tmp_path: Path):
    good = _attempt_with_model(tmp_path / "good", np.linspace(0, 356, 90))
    ctx = Ctx()
    Gate().execute(good, None, ctx)
    assert ctx.metrics["passed"] is True

    bad = _attempt_with_model(tmp_path / "bad", np.linspace(0, 180, 90))
    with pytest.raises(StageFailed, match="covers only 180 degrees"):
        Gate().execute(bad, None, Ctx())
    verdict = json.loads((bad / "gate.json").read_text())
    assert not verdict["passed"]
    failed = {c["metric"] for c in verdict["checks"] if not c["pass"]}
    assert failed == {"azimuth_coverage", "loop_closure"}


def test_pool_serves_priority_then_arrival_and_respects_preferences():
    async def scenario():
        pool = GpuPool([0, 1])
        order = []
        release = asyncio.Event()

        async def job(name, prefs, priority, hold=False):
            async with pool.lease(prefs, priority) as gpu:
                order.append((name, gpu))
                if hold:
                    await release.wait()

        holders = [asyncio.create_task(job("video-a", [1], 0, hold=True)),
                   asyncio.create_task(job("post-a", [0, 1], -3, hold=True))]
        await asyncio.sleep(0)
        waiting = [asyncio.create_task(job("video-b", [1], 0)),
                   asyncio.create_task(job("post-b", [0, 1], -3))]
        await asyncio.sleep(0)
        assert order == [("video-a", 1), ("post-a", 0)]
        release.set()
        await asyncio.gather(*holders, *waiting)
        return order

    order = asyncio.run(scenario())
    # post-b outranks video-b; video-b only accepts GPU 1.
    assert order[2][0] == "post-b"
    assert ("video-b", 1) in order


class _FakeOrbit(Stage):
    name = "orbit_video"
    defaults = {"seed": None, "width": 8, "height": 8}
    gpu_mb = 1
    outputs = ("video.mp4",)

    def run(self, attempt, params, ctx):
        (attempt / "video.mp4").write_text(str(params["seed"]))


class _FakeGate(Stage):
    """Rejects odd seeds, like a gate would reject a turntable video."""
    name = "gate"
    inputs = ("video.mp4",)
    outputs = ("gate.json",)

    def run(self, attempt, params, ctx):
        seed = int((attempt / "video.mp4").read_text())
        ctx.metric("azimuth_coverage", 356.0 if seed % 2 == 0 else 180.0)
        ctx.metric("n_views", 85)
        if seed % 2:
            raise Rejected("the camera covers only 180 degrees around the subject (need 330)")
        (attempt / "gate.json").write_text("{}")


class _FakeTrain(Stage):
    name = "train"
    gpu_mb = 1
    inputs = ("gate.json",)
    outputs = ("final.ply",)

    def run(self, attempt, params, ctx):
        ctx.metric("eval_psnr", 20.0 + int((attempt / "video.mp4").read_text()))
        (attempt / "final.ply").write_text("ply")


def test_job_rerolls_rejected_seeds_and_ranks_by_psnr(tmp_path: Path, monkeypatch):
    orbit = _FakeOrbit()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, _FakeGate(), _FakeTrain()])
    monkeypatch.setattr(job_mod.server, "is_up", lambda g: False)
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)

    spec = job_mod.JobSpec(image=str(image), want=2, max_attempts=4, seeds=[1, 2, 3, 4, 5, 6])
    job = job_mod.Job.create(tmp_path / "jobs", spec, "t")
    job = asyncio.run(job_mod.Runner(job, lambda msg: None).run())

    by_seed = {a.seed: a for a in job.attempts}
    assert sorted(by_seed) == [1, 2, 3, 4]  # two rejected seeds were rerolled, then the budget ran out
    assert by_seed[1].status == by_seed[3].status == "rejected"
    assert "covers only 180 degrees" in by_seed[1].reason
    assert by_seed[1].gate["azimuth_coverage"] == 180.0
    assert job.ranking == [4, 2] and job.status == "done"
    assert (job.path / "best").resolve() == job.attempt_dir(4).resolve()
    saved = json.loads((job.path / "job.json").read_text())
    assert saved["best"] == 4


def test_run_jobs_shares_one_pool(tmp_path: Path, monkeypatch):
    orbit = _FakeOrbit()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, _FakeGate(), _FakeTrain()])
    monkeypatch.setattr(job_mod.server, "is_up", lambda g: False)
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)

    jobs = [job_mod.Job.create(tmp_path / "jobs", job_mod.JobSpec(image=str(image), want=2, max_attempts=2, seeds=seeds), name)
            for name, seeds in (("a", [1, 2]), ("b", [3, 4]))]
    done = asyncio.run(job_mod.run_jobs(jobs, lambda msg: None, [0, 1]))
    assert [j.ranking for j in done] == [[2], [4]]  # odd seeds rejected, no budget left to reroll
    assert all(j.status == "done" for j in done)


def test_cancel_stops_queued_attempts_from_starting(tmp_path: Path, monkeypatch):
    import threading

    cancel = threading.Event()
    started = []

    class CancellingOrbit(_FakeOrbit):
        def run(self, attempt, params, ctx):
            started.append(params["seed"])
            cancel.set()  # Ctrl-C arrives during the first video
            ctx.check_cancelled()

    orbit = CancellingOrbit()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, _FakeGate(), _FakeTrain()])
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)
    job = job_mod.Job.create(tmp_path / "jobs", job_mod.JobSpec(image=str(image), want=3, seeds=[2, 4, 6]), "c")
    job = asyncio.run(job_mod.Runner(job, lambda msg: None, GpuPool([1]), cancel).run())

    assert started == [2]
    assert job.status == "cancelled"
    assert [a.status for a in job.attempts] == ["cancelled"] * 3


def test_resume_finishes_a_cancelled_job_without_redoing_work(tmp_path: Path, monkeypatch):
    import threading

    cancel = threading.Event()
    runs = []

    class CountingOrbit(_FakeOrbit):
        def run(self, attempt, params, ctx):
            runs.append(params["seed"])
            super().run(attempt, params, ctx)
            if len(runs) == 2:
                cancel.set()

    orbit = CountingOrbit()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, _FakeGate(), _FakeTrain()])
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)
    job = job_mod.Job.create(tmp_path / "jobs", job_mod.JobSpec(image=str(image), want=3, max_attempts=3, seeds=[2, 4, 6]), "r")
    job = asyncio.run(job_mod.Runner(job, lambda msg: None, GpuPool([1]), cancel).run())
    assert job.status == "cancelled" and runs == [2, 4]

    resumed = asyncio.run(job_mod.Runner(job_mod.Job.load(job.path), lambda msg: None, GpuPool([1])).run())
    assert runs == [2, 4, 6]  # seeds 2 and 4 kept their videos
    assert resumed.status == "done" and resumed.ranking == [6, 4, 2]


def test_a_fault_is_an_error_not_a_rejection(tmp_path: Path, monkeypatch):
    class CrashingGate(_FakeGate):
        def run(self, attempt, params, ctx):
            raise StageFailed("colmap mapper failed: SIGSEGV")  # not a verdict on the video

    orbit = _FakeOrbit()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, CrashingGate(), _FakeTrain()])
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)
    job = job_mod.Job.create(tmp_path / "jobs", job_mod.JobSpec(image=str(image), want=1, max_attempts=6, seeds=[2]), "e")
    job = asyncio.run(job_mod.Runner(job, lambda msg: None, GpuPool([1]), __import__("threading").Event()).run())
    assert [(a.seed, a.status) for a in job.attempts] == [(2, "error")]  # no seeds burned on rerolls
    assert job.status == "error"


FAKE_COLMAP = """#!/bin/bash
# Stands in for colmap: the mapper segfaults on its first run, then works.
cmd=$1; shift
arg() { while [ $# -gt 0 ]; do [ "$1" = "$want" ] && { echo "$2"; return; }; shift; done; }
case $cmd in
  mapper)
    if [ ! -e "$STATE/crashed" ]; then touch "$STATE/crashed"; echo "*** Aborted at 1 (unix time)" >&2; kill -SEGV $$; fi
    want=--output_path; mkdir -p "$(arg "$@")/0" ;;
  model_converter)
    want=--output_path; out=$(arg "$@")
    printf '1 1 0 0 0 0 0 4 1 frames/00000.png\\n\\n2 1 0 0 0 0 0 4 2 hero/hero.png\\n\\n' > "$out/images.txt" ;;
  model_analyzer) echo "Mean reprojection error: 0.5px" ;;
esac
exit 0
"""


def test_colmap_crash_is_retried(tmp_path: Path, monkeypatch):
    from giro.stages import poses

    fake = tmp_path / "colmap"
    fake.write_text(FAKE_COLMAP)
    fake.chmod(0o755)
    monkeypatch.setattr(poses, "COLMAP", str(fake))
    monkeypatch.setenv("STATE", str(tmp_path))
    attempt = tmp_path / "attempt"
    for name in ("frames/00000.png", "hero/hero.png"):
        (attempt / name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4)).save(attempt / name)

    ctx, logs = Ctx(gpu=0), []
    ctx.on_log = lambda stage, msg: logs.append(msg)
    poses.ColmapPoses().execute(attempt, {"use_masks": False}, ctx)
    assert (tmp_path / "crashed").exists()
    assert any("crashed" in m and "retrying" in m for m in logs)
    assert ctx.metrics["n_registered"] == 2 and ctx.metrics["reg_rate"] == 1.0
