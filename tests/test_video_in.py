"""Video in: the user's own orbit video or frames in place of a generated orbit."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from giro import stages
from giro.job import Job, JobSpec
from giro.stages import video_in


def _frames(directory: Path, n: int = 12, size: tuple[int, int] = (64, 48)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        im = Image.new("RGB", size, (20 * i % 256, 90, 160))
        im.paste((255, 255, 255), (4 * i, 8, 4 * i + 8, 32))  # something that moves frame to frame
        im.save(directory / f"frame_{i:03d}.png")
        out.append(directory / f"frame_{i:03d}.png")
    return out


def _video(path: Path, n: int = 12, size: tuple[int, int] = (64, 48), fps: int = 24) -> Path:
    frames = path.parent / f"{path.stem}_src"
    _frames(frames, n, size)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(frames / "frame_%03d.png"),
                    "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture
def ctx() -> stages.Ctx:
    return stages.Ctx(on_progress=lambda *a: None, on_log=lambda *a: None)


def test_probe_reads_a_video_and_a_frames_directory(tmp_path):
    video = _video(tmp_path / "orbit.mp4", n=12, size=(64, 48))
    p = video_in.probe(video)
    assert (p.width, p.height, p.frames) == (64, 48, 12) and p.fps == pytest.approx(24)
    frames = _frames(tmp_path / "frames", n=5, size=(32, 32))
    assert video_in.probe(frames[0].parent) == video_in.Probe(32, 32, 5, 16.0)
    with pytest.raises(ValueError):
        video_in.probe(tmp_path / "missing.mp4")
    (tmp_path / "text.mp4").write_text("not a video")
    with pytest.raises(ValueError):
        video_in.probe(tmp_path / "text.mp4")


def test_video_in_writes_frames_and_a_preview_and_skips_when_unchanged(tmp_path, ctx):
    video = _video(tmp_path / "orbit.mp4", n=12)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    stage = stages.VIDEO_IN
    assert stage.execute(attempt, {"source": str(video)}, ctx)
    assert len(list((attempt / "frames_raw").glob("*.png"))) == 12
    assert (attempt / "video.mp4").exists()
    record = json.loads((attempt / "orbit.json").read_text())
    assert (record["width"], record["height"], record["frames"]) == (64, 48, 12)
    assert ctx.metrics["n_raw"] == 12
    assert not stage.execute(attempt, {"source": str(video)}, ctx)  # unchanged: skipped
    # A rewritten video at the same path is a new input.
    _video(tmp_path / "orbit.mp4", n=16)
    assert stage.execute(attempt, {"source": str(video)}, ctx)
    assert len(list((attempt / "frames_raw").glob("*.png"))) == 16
    # The extract stage finds the frames ready and leaves them alone.
    assert stages.BY_NAME["extract"].execute(attempt, {}, ctx)
    assert ctx.metrics["n_raw"] == 16


def test_video_in_samples_long_videos_and_scales_big_frames(tmp_path, ctx):
    video = _video(tmp_path / "long.mp4", n=40, size=(256, 128))
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    stages.VIDEO_IN.execute(attempt, {"source": str(video), "max_frames": 10, "max_side": 128}, ctx)
    frames = sorted((attempt / "frames_raw").glob("*.png"))
    assert len(frames) == 10
    with Image.open(frames[0]) as im:
        assert (im.width, im.height) == (128, 64)


def test_video_in_takes_a_frames_directory_relative_to_the_attempt(tmp_path, ctx):
    _frames(tmp_path / "job" / "input" / "frames", n=30, size=(50, 70))
    attempt = tmp_path / "job" / "attempts" / "1"
    attempt.mkdir(parents=True)
    stages.VIDEO_IN.execute(attempt, {"source": "../../input/frames", "max_frames": 15}, ctx)
    frames = sorted((attempt / "frames_raw").glob("*.png"))
    assert len(frames) == 15
    with Image.open(frames[0]) as im:
        assert (im.width, im.height) == (50, 70)
    assert (attempt / "video.mp4").exists()


def test_video_in_refuses_too_few_frames(tmp_path, ctx):
    _frames(tmp_path / "few", n=3)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    with pytest.raises(stages.StageFailed, match="only 3 frames"):
        stages.VIDEO_IN.execute(attempt, {"source": str(tmp_path / "few")}, ctx)
    with pytest.raises(stages.StageFailed, match="no source"):
        stages.VIDEO_IN.execute(attempt, {}, ctx)


def test_the_video_pipeline_is_h3s_with_the_users_video_first():
    names = [s.name for s in stages.pipeline(stages.VIDEO_MODEL)]
    assert names == [s.name for s in stages.pipeline(None)]
    assert stages.pipeline(stages.VIDEO_MODEL)[0] is stages.VIDEO_IN
    assert stages.pipeline(None)[0] is stages.ORBIT
    assert stages.mode_params(stages.VIDEO_MODEL, "poses_colmap") == {}  # COLMAP maps, as for H3


def test_job_create_copies_the_video_and_takes_the_first_frame_as_hero(tmp_path):
    video = _video(tmp_path / "scooter orbit.mp4", n=12, size=(64, 48))
    job = Job.create(tmp_path / "jobs", JobSpec(image="", video=str(video), want=1, max_attempts=1))
    assert job.path.name.endswith("scooter orbit")
    assert Path(job.spec.video) == job.path / "input" / "video.mp4"
    assert Path(job.spec.image) == job.path / "input" / "source.png"
    with Image.open(job.spec.image) as im:
        assert (im.width, im.height) == (64, 48)
    assert job.spec.orbit["model"] == stages.VIDEO_MODEL
    assert job.spec.orbit["source"] == job.spec.video
    assert (job.spec.orbit["width"], job.spec.orbit["height"]) == (64, 48)
    reloaded = Job.load(job.path)
    assert reloaded.spec.video == job.spec.video


def test_job_create_keeps_a_given_hero_beside_a_frames_directory(tmp_path):
    frames = _frames(tmp_path / "frames", n=10, size=(40, 60))
    hero = tmp_path / "hero.png"
    Image.new("RGB", (80, 120), "red").save(hero)
    job = Job.create(tmp_path / "jobs", JobSpec(image=str(hero), video=str(frames[0].parent)))
    assert Path(job.spec.video) == job.path / "input" / "frames"
    assert len(list((job.path / "input" / "frames").glob("*.png"))) == 10
    with Image.open(job.spec.image) as im:
        assert im.getpixel((0, 0)) == (255, 0, 0)


def test_probe_applies_a_phones_rotation_flag_and_reads_mov(tmp_path, ctx):
    plain = _video(tmp_path / "plain.mov", n=12, size=(128, 64))
    rotated = tmp_path / "upright.mov"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-display_rotation", "90", "-i", str(plain), "-c", "copy",
                    str(rotated)], check=True)
    assert (video_in.probe(plain).width, video_in.probe(plain).height) == (128, 64)
    p = video_in.probe(rotated)
    assert (p.width, p.height) == (64, 128)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    stages.VIDEO_IN.execute(attempt, {"source": str(rotated)}, ctx)
    with Image.open(sorted((attempt / "frames_raw").glob("*.png"))[0]) as im:
        assert (im.width, im.height) == (64, 128)  # decoded upright, not squashed into the stored size
    with Image.open(video_in.first_frame(rotated, tmp_path / "hero.png")) as im:
        assert (im.width, im.height) == (64, 128)
