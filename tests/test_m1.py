from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from giro import workflows
from giro.stages import Ctx
from giro.stages.frames import Dedup, ssim
from giro.stages.train import Dataset


def test_snap_length_matches_minimax_grid():
    assert [workflows.snap_length(n) for n in (1, 5, 124, 125, 192)] == [5, 5, 124, 141, 192]


def test_build_sets_named_inputs_and_rejects_unknown():
    prompt = workflows.build("orbit_video", image="giro/hero.png", seed=7, length=141)
    assert prompt["hero"]["inputs"]["image"] == "giro/hero.png"
    assert prompt["noise"]["inputs"]["noise_seed"] == 7
    assert prompt["condition"]["inputs"]["first_frame"] == prompt["condition"]["inputs"]["last_frame"] == ["hero", 0]
    with pytest.raises(KeyError):
        workflows.build("orbit_video", nope=1)


def test_ssim_identity_and_difference():
    rng = np.random.default_rng(0)
    a = rng.random((64, 48))
    assert ssim(a, a) == pytest.approx(1.0)
    assert ssim(a, rng.random((64, 48))) < 0.2


def _frame(path: Path, shift: int) -> None:
    # A textured image translated by `shift` px stands in for camera motion.
    rng = np.random.default_rng(1)
    base = (rng.random((96, 72 + 200)) * 255).astype(np.uint8)
    Image.fromarray(base[:, shift : shift + 72]).save(path)


def test_dedup_trims_static_runs_and_keeps_motion(tmp_path: Path):
    raw = tmp_path / "frames_raw"
    raw.mkdir()
    shifts = [0] * 5 + list(range(0, 120, 6)) + [114] * 5
    for i, s in enumerate(shifts):
        _frame(raw / f"{i:05d}.png", s)
    Dedup().execute(tmp_path, {"target": 120}, Ctx())
    kept = sorted(p.name for p in (tmp_path / "frames").iterdir())
    # Shifts 0 and 114 belong to the static runs; shifts 6..108 remain (indices 6..23).
    assert kept == [f"{i:05d}.png" for i in range(6, 24)]


def test_dataset_keeps_hero_out_of_eval(tmp_path: Path):
    # 8 frames + hero: the hero sorts to index 8, an eval index for split 8.
    colmap = tmp_path / "poses" / "colmap"
    model = colmap / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(b"")
    (colmap / "model").symlink_to("sparse/0")
    txt = colmap / "model_txt"
    txt.mkdir()
    names = [f"frames/{i:05d}.png" for i in range(8)] + ["hero/hero.png"]
    lines = []
    for i, name in enumerate(names):
        lines.append(f"{i + 1} 1 0 0 0 0 0 0 1 {name}")
        lines.append("")
    (txt / "images.txt").write_text("\n".join(lines) + "\n")
    for name in names:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"png")

    ctx = Ctx()
    Dataset().execute(tmp_path, {"eval_split_every": 8, "masks": False}, ctx)
    views = sorted(str(p.relative_to(tmp_path / "dataset" / "images")) for p in (tmp_path / "dataset" / "images").rglob("*.png"))
    assert "hero/hero.png" in views and views.index("hero/hero.png") % 8 != 0
    assert ctx.metrics["n_views"] == 8
