import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from giro import render, splat
from giro.stages import Ctx
from giro.stages.canonicalize import Canonicalize, orbit_frame
from giro.stages.crop import View, hull_votes
from giro.stages.masks import combine

F = 500.0
W, H = 400, 400


def _look_at(center, target=np.zeros(3), up=np.array([0.0, 1.0, 0.0])):
    """World-to-camera rotation and translation, COLMAP axes (image y down = world +up)."""
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    rot = np.stack([x, np.cross(z, x), z])
    return rot, -rot @ center


def _views(n=40, radius=4.0, hero_deg=90.0):
    """Frames on a ring around the origin in the y = 0 plane, plus a hero camera. As in
    COLMAP, world +Y is down the image (so the subject's up is -Y)."""
    views = []
    for i, a in enumerate(list(np.linspace(0, 360, n, endpoint=False)) + [hero_deg]):
        center = radius * np.array([math.cos(math.radians(a)), 0.0, math.sin(math.radians(a))])
        rot, t = _look_at(center)
        name = "hero/hero.png" if i == n else f"frames/{i:05d}.png"
        views.append(View(name, rot, t, F, F, W / 2, H / 2, W, H))
    return views


def _disc_masks(views, radius=0.5):
    """Silhouettes of a sphere of `radius` at the origin."""
    masks = []
    yy, xx = np.mgrid[0:H, 0:W]
    for v in views:
        dist = np.linalg.norm(v.center)
        r_px = F * radius / math.sqrt(dist**2 - radius**2)
        masks.append((xx - W / 2) ** 2 + (yy - H / 2) ** 2 <= r_px**2)
    return masks


def test_combine_keeps_held_parts_and_drops_strips_and_rooms():
    subject = np.zeros((100, 100), bool)
    subject[30:70, 40:60] = True
    background = ~subject
    background[45:48, 60:90] = False    # a sword, 3 px wide, sticking out of the subject
    background[20:22, 0:40] = False     # a 2 px seam between wall segments, touching nothing
    background[50:51, 0:40] = False     # a 1 px seam that reaches the subject
    background[0:30, 60:100] = False    # a doorway the room prompt missed, next to the subject
    background[25:30, 60:100] = False
    mask = combine(subject, background, touch_px=3, gap_px=1, max_add=0.25)
    assert mask[46, 60:89].all()        # sword kept
    assert not mask[20:22, 0:30].any()  # far seam dropped
    assert not mask[50, 0:35].any()     # 1 px seam closed by the gap
    assert not mask[0:25, 70:100].any()  # doorway too large to be held
    assert mask[subject].all()
    assert (combine(subject, None) == subject).all()


def test_hull_keeps_the_subject_and_drops_the_room():
    views = _views()
    masks = _disc_masks(views)
    rng = np.random.default_rng(0)
    inside = rng.normal(size=(500, 3))
    inside = 0.4 * inside / np.linalg.norm(inside, axis=1, keepdims=True) * rng.uniform(0, 1, (500, 1))
    near = np.array([[0.0, 0.0, 1.0], [0.9, 0.3, 0.0]])  # beside the subject, inside the ring
    wall = np.array([[0.0, 0.0, -9.0]])                   # the room, behind the cameras on the far side
    n_vis, n_in = hull_votes(np.vstack([inside, near, wall]), views, masks)
    frac_in = n_in / np.maximum(n_vis, 1)
    assert (frac_in[:500] == 1.0).all() and (n_vis[:500] == len(views)).all()
    assert (frac_in[500:502] < 0.8).all()
    # Seen only by the few views facing it: dropped by the visibility rule even if every
    # one of those views wrongly had it inside the mask.
    assert n_vis[502] < 0.5 * len(views)


def test_orbit_frame_up_and_front():
    views = _views(hero_deg=90.0)
    center, m = orbit_frame(views)
    assert np.allclose(center, 0, atol=1e-6)
    assert np.allclose(m[1], [0, -1, 0], atol=1e-6)  # image up is world -Y
    assert np.allclose(m[2], [0, 0, 1], atol=1e-6)   # toward the hero at 90 degrees (+Z)
    assert np.isclose(np.linalg.det(m), 1.0)


def _write_model(attempt: Path, views) -> None:
    txt = attempt / "poses" / "colmap" / "model_txt"
    txt.mkdir(parents=True)
    (txt / "cameras.txt").write_text(f"1 SIMPLE_PINHOLE {W} {H} {F} {W / 2} {H / 2}\n")
    lines = []
    for i, v in enumerate(views):
        x, y, z, w = Rotation.from_matrix(v.rot).as_quat()
        lines += [f"{i + 1} {w} {x} {y} {z} {' '.join(map(str, v.trans))} 1 {v.name}", ""]
    (txt / "images.txt").write_text("\n".join(lines) + "\n")


@pytest.mark.skipif(not render.SPLAT_TRANSFORM.exists(), reason="splat-transform not installed")
def test_canonicalize_stands_the_subject_up(tmp_path):
    views = _views(hero_deg=90.0)
    _write_model(tmp_path, views)
    # A column 2 units tall (world y from +1 at the feet to -1 at the head), 0.5 wide.
    rng = np.random.default_rng(1)
    pts = np.stack([rng.uniform(-0.25, 0.25, 4000), rng.uniform(-1, 1, 4000), rng.uniform(-0.1, 0.1, 4000)], 1)
    pts[0] = [0, -1.0, 0.1]  # the top of the head, toward the hero
    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2", "opacity", "rot_0", "rot_1", "rot_2", "rot_3",
             "f_dc_0", "f_dc_1", "f_dc_2"]
    g = np.zeros(len(pts), dtype=[(n, "<f4") for n in names])
    g["x"], g["y"], g["z"] = pts.T
    g["rot_0"], g["scale_0"], g["scale_1"], g["scale_2"] = 1.0, -5, -5, -5
    (tmp_path / "crop").mkdir()
    splat.write_ply(tmp_path / "crop" / "cropped.ply", g)

    stage = Canonicalize()
    stage.execute(tmp_path, {"height_m": 1.8}, Ctx())
    viewer = splat.positions(splat.read_ply(tmp_path / "canonical" / "splat.ply")) @ render.FLIP
    lo, hi = np.percentile(viewer, 0.5, 0), np.percentile(viewer, 99.5, 0)
    assert lo[1] == pytest.approx(0, abs=0.01) and hi[1] == pytest.approx(1.8, abs=0.01)
    assert abs(lo[0] + hi[0]) < 0.01 and abs(lo[2] + hi[2]) < 0.01  # footprint centered
    assert viewer[0][1] > 1.75 and viewer[0][2] > 0  # head on top, facing +Z
    transform = json.loads((tmp_path / "canonical" / "transform.json").read_text())
    assert transform["scale"] == pytest.approx(0.9, rel=0.01)


def test_dataset_links_eroded_masks_that_mirror_images(tmp_path):
    from PIL import Image

    from giro.stages.train import Dataset
    views = _views(n=4)
    _write_model(tmp_path, views)
    model = tmp_path / "poses" / "colmap" / "model"
    model.mkdir()
    for v in views:
        (tmp_path / v.name).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (W, H)).save(tmp_path / v.name)
        m = np.zeros((H, W), np.uint8)
        m[100:300, 150:250] = 255
        (tmp_path / "masks" / v.name).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(m).save(tmp_path / "masks" / v.name)
    Dataset().execute(tmp_path, {"eval_split_every": 0, "masks": True, "mask_erode_px": 2}, Ctx())
    for v in views:
        assert (tmp_path / "dataset" / "images" / v.name).exists()
        mask = np.asarray(Image.open(tmp_path / "dataset" / "masks" / v.name)) > 127
        assert mask[102:298, 152:248].all() and not mask[101, 200] and not mask[200, 151]
