import json
import math

import numpy as np

from giro import path as campath
from giro import symmetry
from giro.stages.orbit import OrbitVideo


def _box_car(yaw_deg: float, offset: float) -> np.ndarray:
    """A toy car: a long box with a cabin toward one end, mirror-symmetric left to right, turned
    `yaw_deg` about the vertical and moved `offset` along x."""
    rng = np.random.default_rng(0)
    body = rng.uniform([-0.2, -0.1, -0.5], [0.2, 0.1, 0.5], (20000, 3))
    cabin = rng.uniform([-0.15, -0.25, 0.1], [0.15, -0.1, 0.4], (6000, 3))
    x = np.concatenate([body, cabin])
    a = math.radians(yaw_deg)
    rot = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    return x @ rot.T + [offset, 0, 0]


def test_the_plane_is_found_and_its_mirror_camera_sees_the_flip():
    x = _box_car(30.0, 0.05)
    pl = symmetry.plane(x)
    a = math.radians(30.0)
    true_n = np.array([math.cos(a), 0, -math.sin(a)])  # the box's x axis, turned
    assert abs(abs(pl.normal @ true_n) - 1) < 1e-3
    assert abs(abs(pl.d) - abs(0.05 * true_n[0])) < 0.01
    assert pl.score < 2 * pl.floor + 0.01
    hero = campath.PathCamera(10.0, 5.0, 2.0, (0.05, 0.0, 0.0))
    m = symmetry.mirror(hero, pl)
    # its pose is F R H with the reflection's translation
    r, t = hero.world_to_camera()
    h = np.eye(3) - 2 * np.outer(pl.normal, pl.normal)
    r2 = np.diag([-1.0, 1, 1]) @ r @ h
    rm, tm = m.world_to_camera()
    assert np.allclose(rm, r2, atol=1e-9) and np.allclose(tm, np.diag([-1.0, 1, 1]) @ t - 2 * pl.d * (r2 @ pl.normal), atol=1e-9)
    # mirroring twice is the hero again; the camera moved by twice its angle to the plane
    back = symmetry.mirror(m, pl)
    assert np.allclose(back.position(), hero.position()) and abs(m.pitch - hero.pitch) < 1e-9
    assert abs(symmetry.separation(hero, m) - symmetry.separation(m, back)) < 1e-6


def test_the_mirror_path_reaches_the_mirrored_camera_then_finishes_the_turns(tmp_path):
    hero = campath.PathCamera(350.0, -3.0, 3.0, (0.1, -0.1, 0.0))
    (tmp_path / "proxy").mkdir()
    (tmp_path / "proxy" / "proxy.json").write_text(json.dumps({"hero_camera": hero.to_json(), "bounds": [[-1, -1, -1], [1, 1, 1]]}))
    m = campath.PathCamera(70.0, -3.0, 3.0, (-0.05, -0.1, 0.1))
    cams, clips = OrbitVideo()._cameras(tmp_path, {"width": 576, "height": 768}, 81, {"camera": m.to_json(), "frames": 17})
    assert [len(c) for c in clips] == [17, 65] and len(cams["frames"]) == 81
    end = clips[0][-1]
    assert np.allclose(end.position(), m.position()) and np.allclose(end.target, m.target)
    assert clips[1][0] == end and clips[1][-1].pitch == 45.0
    assert abs(clips[1][-1].yaw - hero.yaw - 720) < 1e-9  # two turns in all, the same way round
    assert cams["mirror"]["frames"] == 17


def test_dataset_trains_on_the_mirrored_hero_at_the_pinned_frame(tmp_path):
    from PIL import Image

    from giro.stages import Ctx
    from giro.stages.train import Dataset

    colmap = tmp_path / "poses" / "colmap"
    (colmap / "sparse" / "0").mkdir(parents=True)
    (colmap / "model").symlink_to("sparse/0")
    txt = colmap / "model_txt"
    txt.mkdir()
    names = [f"frames/{i:05d}.png" for i in range(30)] + ["hero/hero.png"]
    lines = []
    for i, name in enumerate(names):
        lines += [f"{i + 1} 1 0 0 0 0.5 0 {i} {2 if name.startswith('hero') else 1} {name}", ""]
    (txt / "images.txt").write_text("\n".join(lines) + "\n")
    (txt / "cameras.txt").write_text("1 SIMPLE_PINHOLE 8 8 10 4 4\n2 SIMPLE_PINHOLE 16 16 20 8 8\n")
    (txt / "points3D.txt").write_text("")
    hero = np.zeros((16, 16, 3), np.uint8)
    hero[:, :4] = 255  # bright on the left
    for name in names:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        Image.fromarray(hero).save(tmp_path / name)
        (tmp_path / "masks" / name).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(hero[..., 0]).save(tmp_path / "masks" / name)
    (tmp_path / "cameras.json").write_text(json.dumps({"mirror": {"frames": 17, "pin": True}}))

    ctx = Ctx()
    Dataset().execute(tmp_path, {"eval_split_every": 8, "mask_erode_px": 0, "hero_copies": 5, "mirror_copies": 5}, ctx)
    images = tmp_path / "dataset" / "images"
    views = sorted(str(p.relative_to(images)) for p in images.rglob("*.png"))
    mirrors = [v for v in views if "_mirror" in v]
    assert mirrors == [f"frames/00016_mirror{k}.png" for k in range(1, 6)] and ctx.metrics["mirror_copies"] == 5
    assert all(views.index(v) % 8 != 0 for v in mirrors + [v for v in views if v.startswith("hero/")])
    assert len(views) >= 31 + 4 + 5 - 3  # at most a frame or so left out per run of copies
    flipped = np.asarray(Image.open(images / mirrors[0]))
    assert flipped[:, -4:].min() == 255 and flipped[:, :4].max() == 0
    assert np.asarray(Image.open(tmp_path / "dataset" / "masks" / mirrors[0]))[:, -1].min() == 255
    written = (tmp_path / "dataset" / "sparse" / "0" / "images.txt").read_text().split("\n")
    line = next(ln for ln in written if ln.endswith("00016_mirror2.png")).split()
    assert line[1:9] == ["1", "0", "0", "0", "0.5", "0", "16", "2"]  # frame 16's pose, the hero's camera
