import asyncio
import json
from pathlib import Path

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
