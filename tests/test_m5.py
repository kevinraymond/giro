import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from giro import job as job_mod
from giro import stages
from giro.stages import Ctx, Rejected, Stage, gapfill
from giro.stages.masks import Masks

MAX_STEP = 30.0


def _gate_json(attempt: Path, names: list[str], azimuths: list[float], failing: set[str]) -> None:
    metrics = ["reg_rate", "n_views", "azimuth_coverage", "max_step_deg", "loop_closure"]
    checks = [{"metric": m, "value": 0, "op": "<=", "threshold": MAX_STEP if m == "max_step_deg" else 0,
               "pass": m not in failing} for m in metrics]
    (attempt / "gate.json").write_text(json.dumps({
        "passed": not failing, "reasons": sorted(failing), "checks": checks,
        "ring": {"frames": names, "azimuths": azimuths},
    }))


def _frames(attempt: Path, n: int, size=(24, 32)) -> list[str]:
    (attempt / "frames_raw").mkdir(parents=True, exist_ok=True)
    (attempt / "frames").mkdir(exist_ok=True)
    for i in range(n):
        raw = attempt / "frames_raw" / f"{i:05d}.png"
        if not raw.exists():
            Image.new("RGB", size, (i, 0, 0)).save(raw)
            (attempt / "frames" / raw.name).hardlink_to(raw)  # like dedup
    return [f"frames/{i:05d}.png" for i in range(n)]


def _jumpy_orbit(attempt: Path, gap=(40, 44), step=3.0, jump=None, failing=frozenset({"max_step_deg"})):
    """An orbit of 3-degree steps whose frames gap[0]+1 .. gap[1]-1 got no camera; across them the
    camera moves `jump` degrees (default: the fast, smeared stretch, at 5x the usual speed)."""
    names = _frames(attempt, 100)
    posed, az, a = [], [], 0.0
    for i, n in enumerate(names):
        if gap[0] < i < gap[1]:
            continue
        if posed and i == gap[1]:
            a += jump if jump is not None else step * (gap[1] - gap[0]) * 5
        elif posed:
            a += step
        posed.append(n)
        az.append((a + 180) % 360 - 180)
    _gate_json(attempt, posed, az, set(failing))


def test_plan_fills_a_forward_jump_and_names_the_frames_it_replaces(tmp_path):
    _jumpy_orbit(tmp_path)
    gaps, why = gapfill.plan(tmp_path, gapfill.GapFill.defaults)
    assert why == "" and len(gaps) == 1
    g = gaps[0]
    assert (g.first, g.last) == ("frames/00040.png", "frames/00044.png")
    assert g.deg == pytest.approx(60.0)
    assert g.missing == ["frames/00041.png", "frames/00042.png", "frames/00043.png"]
    assert g.length == 22  # 60 / 3 -> 19 new frames + both ends = 21, on the 17k+5 grid


def test_plan_leaves_what_a_fill_cannot_repair(tmp_path):
    cases = {
        "coverage": dict(failing={"max_step_deg", "azimuth_coverage"}),
        "backwards": dict(jump=-60.0),  # the camera snaps back instead of rushing ahead
        "huge": dict(gap=(40, 48)),  # 8 frames at 5x speed: 120 degrees
    }
    for name, kw in cases.items():
        attempt = tmp_path / name
        attempt.mkdir()
        _jumpy_orbit(attempt, **kw)
        gaps, why = gapfill.plan(attempt, gapfill.GapFill.defaults)
        assert gaps == [], name
        assert {"coverage": "also failed on azimuth_coverage", "backwards": "snaps back 60 degrees", "huge": "more than a fill covers"}[name] in why


class _FakeComfy:
    """Stands in for ComfyClient: a prompt of length N yields N frames."""

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
        from giro.comfy import Done, Progress

        _FakeComfy.prompts.append(prompt)
        n = prompt["condition"]["inputs"]["length"]
        yield Progress("sampler", 1, 1)
        yield Done("p1", {"frames": {"images": [{"filename": f"f{i:05d}.png"} for i in range(n)]}})

    async def download(self, image, dest):
        Image.new("RGB", (24, 32), "blue").save(dest)
        return dest

    async def interrupt(self):
        pass

    async def free(self, unload_models=True):
        pass


class _FakeLease:
    url = "http://127.0.0.1:1"

    def __init__(self, gpu, timeout=120.0):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


@pytest.fixture
def fake_comfy(monkeypatch):
    monkeypatch.setattr(gapfill, "ComfyClient", _FakeComfy)
    monkeypatch.setattr(gapfill.server, "Lease", _FakeLease)
    _FakeComfy.prompts = []


def test_gapfill_splices_the_arc_and_reverts(tmp_path, fake_comfy):
    _jumpy_orbit(tmp_path)
    before = sorted(p.name for p in (tmp_path / "frames").iterdir())
    ctx = Ctx(gpu=1)
    gapfill.GapFill().execute(tmp_path, {"seed": 7}, ctx)

    prompt = _FakeComfy.prompts[0]
    assert prompt["hero"]["inputs"]["image"] == "giro/00040.png"
    assert prompt["last"]["inputs"]["image"] == "giro/00044.png"
    assert prompt["condition"]["inputs"]["first_frame"] == ["hero", 0]
    assert prompt["condition"]["inputs"]["last_frame"] == ["last", 0]
    assert prompt["noise"]["inputs"]["noise_seed"] == 7

    names = sorted(p.name for p in (tmp_path / "frames").iterdir())
    at = names.index("00040.png")
    assert names[at + 1:at + 21] == [f"00040_{k:02d}.png" for k in range(1, 21)]  # both ends are not repeated
    assert names[at + 21] == "00044.png"  # 41-43 are gone
    assert not (tmp_path / "frames" / "00041.png").exists() and (tmp_path / "frames_raw" / "00041.png").exists()
    manifest = gapfill.applied(tmp_path)
    assert manifest and manifest["gaps"][0]["dropped"] == ["frames/00041.png", "frames/00042.png", "frames/00043.png"]
    assert gapfill.summary(manifest) == ["00040→00044: 60° jump, 20 generated frames replace 3"]
    assert ctx.metrics["n_inserted"] == 20 and ctx.metrics["n_dropped"] == 3

    gapfill.revert(tmp_path)
    assert sorted(p.name for p in (tmp_path / "frames").iterdir()) == before
    assert gapfill.applied(tmp_path) is None


def test_masks_detect_only_new_frames_while_params_hold(tmp_path, monkeypatch):
    asked: list[list[str]] = []

    async def fake_detect(self, groups, raw, params, n_images, ctx):
        asked.append(sorted(p.name for paths in groups.values() for p in paths))
        for group, paths in groups.items():
            for p in paths:
                for kind in ("subject", "background"):
                    dest = raw / kind / group / p.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("L", Image.open(p).size, 255 if kind == "subject" else 0).save(dest)

    monkeypatch.setattr(Masks, "_detect", fake_detect)
    _frames(tmp_path, 4)
    (tmp_path / "hero").mkdir()
    Image.new("RGB", (24, 32)).save(tmp_path / "hero" / "hero.png")

    Masks().execute(tmp_path, None, Ctx())
    assert asked[-1] == ["00000.png", "00001.png", "00002.png", "00003.png", "hero.png"]
    Image.new("RGB", (24, 32), "blue").save(tmp_path / "frames" / "00001_01.png")  # a gap fill's frame
    (tmp_path / "frames" / "00002.png").unlink()  # one it replaced
    ctx = Ctx()
    Masks().execute(tmp_path, None, ctx)
    assert asked[-1] == ["00001_01.png"] and ctx.metrics["n_detected"] == 1
    assert sorted(p.name for p in (tmp_path / "masks" / "frames").iterdir()) == ["00000.png", "00001.png", "00001_01.png", "00003.png"]
    Masks().execute(tmp_path, {"threshold": 0.6}, Ctx())  # new params: everything again
    assert len(asked[-1]) == 5


class _Orbit(Stage):
    name = "orbit_video"
    defaults = {"seed": None, "width": 24, "height": 32}
    gpu_mb = 1
    outputs = ("frames",)

    def run(self, attempt, params, ctx):
        _frames(attempt, 100)
        (attempt / "seed").write_text(str(params["seed"]))


class _Gate(Stage):
    """Seed 1: a jump a fill repairs. Seed 2: also a half orbit, so it is rerolled."""

    name = "gate"
    inputs = ("frames",)
    outputs = ("gate.json",)

    def run(self, attempt, params, ctx):
        seed = int((attempt / "seed").read_text())
        if any("_" in p.name for p in (attempt / "frames").iterdir()):
            names = sorted(f"frames/{p.name}" for p in (attempt / "frames").iterdir())
            _gate_json(attempt, names, [(3.0 * i + 180) % 360 - 180 for i in range(len(names))], set())
            ctx.metric("passed", True)
            return
        _jumpy_orbit(attempt, failing={"max_step_deg"} | ({"azimuth_coverage"} if seed == 2 else set()))
        raise Rejected("the camera path jumps 60 degrees between two consecutive frames")


class _Train(Stage):
    name = "train"
    inputs = ("gate.json",)
    outputs = ("final.ply",)

    def run(self, attempt, params, ctx):
        ctx.metric("eval_psnr", 30.0)
        (attempt / "final.ply").write_text("ply")


def test_job_fills_a_jumpy_orbit_instead_of_rerolling(tmp_path, monkeypatch, fake_comfy):
    orbit = _Orbit()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, _Gate(), _Train()])
    monkeypatch.setattr(job_mod.server, "is_up", lambda g: False)
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)
    log: list[str] = []

    spec = job_mod.JobSpec(image=str(image), want=1, max_attempts=3, seeds=[2, 1])
    job = job_mod.Job.create(tmp_path / "jobs", spec, "t")
    job = asyncio.run(job_mod.Runner(job, log.append).run())

    by_seed = {a.seed: a for a in job.attempts}
    assert by_seed[2].status == "rejected" and by_seed[2].filled == []  # not a fill's case: rerolled
    assert any("no gap fill (the gate also failed on azimuth_coverage)" in m for m in log)
    assert by_seed[1].status == "passed" and by_seed[1].eval_psnr == 30.0
    assert by_seed[1].filled == ["00040→00044: 60° jump, 20 generated frames replace 3"]
    assert by_seed[1].gpus["gapfill"] == 1  # a video GPU
    assert job.ranking == [1] and len(job.attempts) == 2
    saved = json.loads((job.path / "job.json").read_text())
    assert saved["attempts"][1]["filled"] == by_seed[1].filled
    metrics = json.loads((job.attempt_dir(1) / "metrics.json").read_text())
    assert metrics["gapfill"]["n_inserted"] == 20 and metrics["gate"]["passed"] is True


def test_a_fill_that_still_fails_the_gate_is_rejected_once(tmp_path, monkeypatch, fake_comfy):
    class _StubbornGate(_Gate):
        def run(self, attempt, params, ctx):
            _jumpy_orbit(attempt) if not gapfill.applied(attempt) else None
            raise Rejected("the camera path jumps 60 degrees between two consecutive frames")

    orbit = _Orbit()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, _StubbornGate(), _Train()])
    monkeypatch.setattr(job_mod.server, "is_up", lambda g: False)
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)
    spec = job_mod.JobSpec(image=str(image), want=1, max_attempts=1, seeds=[1])
    job = asyncio.run(job_mod.Runner(job_mod.Job.create(tmp_path / "jobs", spec, "t"), lambda m: None).run())
    (attempt,) = job.attempts
    assert attempt.status == "rejected" and len(attempt.filled) == 1
    assert len(_FakeComfy.prompts) == 1  # filled once, not in a loop


# --- pose fallback (Depth Anything 3 + COLMAP refinement) ---

from scipy.spatial.transform import Rotation  # noqa: E402

from giro.stages import fallback, poses  # noqa: E402
from giro.stages.gate import Gate  # noqa: E402


def test_hybrid_keeps_refined_poses_and_moves_the_rest_into_their_frame():
    rng = np.random.default_rng(0)
    ff = {f"frames/{i:05d}.png": (Rotation.random(random_state=i).as_matrix(), rng.normal(size=3)) for i in range(12)}
    # The refined model is the feedforward one under a similarity (bundle adjustment's own gauge).
    s, r, t = 2.5, Rotation.from_euler("xyz", [20, -35, 60], degrees=True).as_matrix(), np.array([1.0, -2.0, 0.5])

    def moved(rot, tv):
        center = s * r @ (-rot.T @ tv) + t
        new = rot @ r.T
        return new, -new @ center

    refined = {n: moved(*p) for n, p in ff.items()}
    unconstrained = ["frames/00003.png", "frames/00007.png"]
    for n in unconstrained:  # what bundle adjustment leaves where it had nothing to go on
        refined[n] = (np.eye(3), np.zeros(3))
    out = fallback.hybrid(ff, refined, [n for n in ff if n not in unconstrained])
    for n in ff:
        want = moved(*ff[n])
        assert np.allclose(out[n][0], want[0], atol=1e-9) and np.allclose(out[n][1], want[1], atol=1e-9)


def _models(attempt: Path, old_layout=False) -> None:
    """COLMAP's sparse/0 (+_txt) and a fallback model, each with a marker file."""
    work = attempt / "poses" / "colmap"
    for d in ("sparse/0", "sparse/0_txt", "../fallback/model", "../fallback/model_txt"):
        (work / d).mkdir(parents=True)
        (work / d / "which").write_text(d)
    if old_layout:  # before the fallback: model links straight to sparse/0
        (work / "model").symlink_to("sparse/0")
        (work / "model_txt").symlink_to("sparse/0_txt")
    else:
        (work / "model_colmap").symlink_to("sparse/0")
        (work / "model_colmap_txt").symlink_to("sparse/0_txt")
        poses.activate(attempt, "colmap")


@pytest.mark.parametrize("old_layout", [False, True])
def test_activate_switches_the_model_every_stage_reads_and_back(tmp_path, old_layout):
    _models(tmp_path, old_layout)
    model_txt = tmp_path / "poses" / "colmap" / "model_txt"
    assert poses.active_source(tmp_path) == "colmap" and (model_txt / "which").read_text() == "sparse/0_txt"
    poses.activate(tmp_path, "fallback")
    assert poses.active_source(tmp_path) == "fallback"
    assert (model_txt / "which").read_text() == "../fallback/model_txt"
    poses.activate(tmp_path, "colmap")
    assert poses.active_source(tmp_path) == "colmap" and (model_txt / "which").read_text() == "sparse/0_txt"


def test_gate_takes_the_reprojection_error_of_the_active_cameras(tmp_path):
    from test_m2 import _ring  # the synthetic orbit

    _models(tmp_path)
    names = [f"frames/{i:05d}.png" for i in range(90)] + ["hero/hero.png"]
    qs, ts = _ring(list(np.linspace(0, 356, 90)) + [0.0])
    for d in ("sparse/0_txt", "../fallback/model_txt"):
        (tmp_path / "poses" / "colmap" / d / "images.txt").write_text(
            "".join(f"{i + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {n}\n\n"
                    for i, (n, q, t) in enumerate(zip(names, qs, ts))))
    for n in names:
        (tmp_path / n).parent.mkdir(exist_ok=True)
        Image.new("RGB", (4, 4)).save(tmp_path / n)
    (tmp_path / "metrics.json").write_text(json.dumps({"poses_colmap": {"reproj_err": 2.0},
                                                       "poses_fallback": {"reproj_err": 0.9}}))
    with pytest.raises(Rejected, match="reprojection error 2.00"):
        Gate().execute(tmp_path, None, Ctx())
    poses.activate(tmp_path, "fallback")
    ctx = Ctx()
    Gate().execute(tmp_path, None, ctx)
    assert ctx.metrics["passed"] is True and ctx.metrics["reproj_err"] == 0.9 and ctx.metrics["poses"] == "fallback"


class _Poses(Stage):
    """COLMAP's stage, reduced to the files the runner and the fallback look for."""

    name = "poses_colmap"
    inputs = ("frames",)
    outputs = ("poses/colmap/model",)

    def run(self, attempt, params, ctx):
        work = attempt / "poses" / "colmap"
        shutil.rmtree(work, ignore_errors=True)
        (work / "sparse" / "0").mkdir(parents=True)
        (work / "sparse" / "0_txt").mkdir()
        (work / "database.db").write_text("")
        (work / "model_colmap").symlink_to("sparse/0")
        (work / "model_colmap_txt").symlink_to("sparse/0_txt")
        poses.activate(attempt, "colmap")


class _Fallback(Stage):
    name = "poses_fallback"
    defaults = {"enabled": True}
    inputs = ("poses/colmap/database.db",)
    outputs = ("poses/fallback/model",)
    gpu_mb = 1
    runs = 0

    def run(self, attempt, params, ctx):
        _Fallback.runs += 1
        for d in ("model", "model_txt"):
            (attempt / "poses" / "fallback" / d).mkdir(parents=True, exist_ok=True)


class _PoseGate(Stage):
    """COLMAP's cameras jump 60 degrees; the fallback's pass unless `fallback_passes` is off."""

    name = "gate"
    inputs = ("poses/colmap/model_txt", "frames")
    outputs = ("gate.json",)
    fallback_passes = True

    def run(self, attempt, params, ctx):
        if poses.active_source(attempt) == "fallback" and self.fallback_passes:
            ctx.metric("passed", True)
            return
        if gapfill.applied(attempt):
            raise Rejected("still jumps after the fill")
        _jumpy_orbit(attempt)
        raise Rejected("the camera path jumps 60 degrees between two consecutive frames")


def _pose_job(tmp_path, monkeypatch, gate):
    orbit = _Orbit()
    _Fallback.runs = 0
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, _Poses(), gate, _Train()])
    monkeypatch.setattr(stages, "FALLBACK", _Fallback())
    monkeypatch.setattr(job_mod.server, "is_up", lambda g: False)
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)
    log: list[str] = []
    spec = job_mod.JobSpec(image=str(image), want=1, max_attempts=1, seeds=[1])
    job = asyncio.run(job_mod.Runner(job_mod.Job.create(tmp_path / "jobs", spec, "t"), log.append).run())
    return job.attempts[0], log


def test_job_trains_on_the_fallback_cameras_when_colmaps_fail_the_gate(tmp_path, monkeypatch, fake_comfy):
    attempt, log = _pose_job(tmp_path, monkeypatch, _PoseGate())
    assert attempt.status == "passed" and attempt.poses == "fallback" and attempt.filled == []
    assert _Fallback.runs == 1 and _FakeComfy.prompts == []  # no gap fill needed
    assert any("trying the pose fallback" in m for m in log)


def test_rejected_fallback_goes_back_to_colmap_and_gap_fill(tmp_path, monkeypatch, fake_comfy):
    gate = _PoseGate()
    gate.fallback_passes = False
    attempt, log = _pose_job(tmp_path, monkeypatch, gate)
    assert any("rejected the pose fallback too; back to COLMAP's cameras" in m for m in log)
    assert len(_FakeComfy.prompts) == 1  # the fill was planned from COLMAP's jump
    # After the fill COLMAP ran again, so the fallback got one more try; then the attempt is rejected.
    assert _Fallback.runs == 2 and attempt.status == "rejected" and attempt.poses == "colmap"


# --- video shape, size and crop ---

from giro.api import NewJob  # noqa: E402
from giro.hero import fit_to_aspect  # noqa: E402


@pytest.fixture(autouse=True)
def _h3_default(monkeypatch):
    """These tests drive the job runner with fake H3-shaped stages; new jobs would otherwise get the
    proxy orbit, whose real proxy stage needs ComfyUI on a GPU."""
    monkeypatch.setattr(stages, "DEFAULT_MODEL", "h3")


def test_hero_keeps_the_chosen_region_at_the_video_aspect(tmp_path):
    src = tmp_path / "src.png"
    im = Image.new("RGB", (1000, 800), "black")
    im.paste((255, 0, 0), (600, 100, 900, 500))  # the subject, right of center
    im.save(src)
    centered, removed = fit_to_aspect(src, 768, 1024)  # 600x800 from the middle
    assert centered.size == (600, 800) and removed == pytest.approx(0.4)
    region, removed = fit_to_aspect(src, 768, 1024, crop=(0.55, 0.05, 0.95, 0.65))  # zoomed onto the subject
    assert region.size == (360, 480) and removed == pytest.approx(1 - 360 * 480 / 800_000)
    assert region.getpixel((180, 240)) == (255, 0, 0)


def test_new_job_checks_the_video_size_and_crop():
    spec = NewJob(orbit={"width": 1344, "height": 768}, crop=[0.1, 0.0, 0.6, 0.9]).spec()
    assert spec.orbit == {"width": 1344, "height": 768, "model": "h3"} and spec.crop == [0.1, 0.0, 0.6, 0.9]
    with pytest.raises(ValueError, match="multiple of 32"):
        NewJob(orbit={"width": 1000}).spec()
    with pytest.raises(ValueError, match="crop must be"):
        NewJob(crop=[0.5, 0.0, 0.4, 1.0]).spec()


@pytest.mark.parametrize("frame", [(768, 1024), (1344, 768), (1184, 672), (672, 1184), (896, 896), (736, 1088)])
def test_fallback_feeds_da3_the_frames_shape_on_its_patch_grid(frame):
    w, h = fallback.model_size(*frame, 504)
    assert w % 14 == 0 and h % 14 == 0 and abs(max(w, h) - 504) <= 56
    assert abs((w / h) / (frame[0] / frame[1]) - 1) < 0.005  # within the worker's 2%, with room to spare


def test_a_rerun_leases_no_gpu_for_stages_it_reuses(tmp_path, monkeypatch):
    orbit = _Orbit()
    monkeypatch.setattr(stages, "ORBIT", orbit)
    monkeypatch.setattr(stages, "PIPELINE", [orbit, _Train()])
    monkeypatch.setattr(job_mod.server, "is_up", lambda g: False)
    image = tmp_path / "subject.png"
    Image.new("RGB", (16, 16), "green").save(image)
    spec = job_mod.JobSpec(image=str(image), want=1, max_attempts=1, seeds=[1])
    job = asyncio.run(job_mod.Runner(job_mod.Job.create(tmp_path / "jobs", spec, "t"), lambda m: None).run())
    (attempt,) = job.attempts
    assert attempt.gpus == {"orbit_video": 1} and orbit.is_current(job.attempt_dir(1), {"seed": 1} | spec.orbit)
    attempt.status, attempt.gpus = "queued", {}  # rerun, e.g. after new params for a later stage
    job = asyncio.run(job_mod.Runner(job, lambda m: None).run())
    assert job.attempts[0].status == "passed" and job.attempts[0].gpus == {}  # the video was reused without a lease


def test_a_rejection_after_a_failed_fallback_is_the_gates(tmp_path, monkeypatch, fake_comfy):
    gate = _PoseGate()
    gate.fallback_passes = False
    attempt, _ = _pose_job(tmp_path, monkeypatch, gate)
    assert attempt.status == "rejected" and attempt.stage == "gate"


# --- job export ---

def test_export_packs_a_report_whose_links_all_resolve(tmp_path):
    import re
    import zipfile

    from giro import report

    image = tmp_path / "subject.png"
    Image.new("RGB", (32, 32), "green").save(image)
    spec = job_mod.JobSpec(image=str(image), want=1, max_attempts=2, seeds=[1, 2])
    job = job_mod.Job.create(tmp_path / "jobs", spec, "t")
    (job.path / "input").mkdir(exist_ok=True)
    Image.new("RGB", (32, 32), "green").save(job.path / "input" / "source.png")
    good = job.attempt_dir(1)
    _frames(good, 6)
    for rel in ("hero/hero.png", "crop/preview_hero.jpg", "canonical/turnaround.jpg"):
        (good / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 32), "red").save(good / rel)
    (good / "export").mkdir()
    for ext in ("ply", "sog", "spz"):
        (good / "export" / f"splat.{ext}").write_bytes(b"x" * 100)
    (good / "video.mp4").write_bytes(b"mp4")
    (good / "gate.json").write_text(json.dumps({"passed": True, "checks": [
        {"metric": "reg_rate", "value": 1.0, "op": ">=", "threshold": 0.9, "pass": True}], "ring": {"azimuth_span": 358.0}}))
    (good / "metrics.json").write_text(json.dumps({"train": {"eval_psnr": 30.0, "eval_ssim": 0.95}}))
    job.attempts = [job_mod.Attempt(1, status="passed", eval_psnr=30.0, gaussians=1000,
                                    params={"masks": {"subject_prompt": "woman:2"}}),
                    job_mod.Attempt(2, status="rejected", stage="gate", reason="the camera covers only 62 degrees")]
    job.ranking = [1]

    out = report.build(job, tmp_path / "out.zip")
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        page = zf.read(f"{job.path.name}/index.html").decode()
    links = set(re.findall(r'(?:src|href)="([^"#:]+)"', page))
    assert links and all(f"{job.path.name}/{link}" in names for link in links), links - {n.split("/", 1)[1] for n in names}
    for ext in ("ply", "sog", "spz"):
        assert f"{job.path.name}/seed-1/splat.{ext}" in names
    assert "woman:2" in page and "100.0%" in page  # the seed's own prompt; fractions as percentages
    assert "the camera covers only 62 degrees" in page  # the rejected seed gets its line
