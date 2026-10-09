import json
from dataclasses import replace

import numpy as np
from PIL import Image

from giro import path as campath
from giro import splat, stages, workflows
from giro.stages import Ctx
from giro.stages.proxy import Points, fit_hero_camera
from giro.stages.train import Dataset


def test_paths_start_at_the_hero_camera_and_climb():
    hero = campath.PathCamera(30.0, 5.0, 1.5, (0.0, 0.1, 0.0))
    ring = campath.plan(hero, "ring", 81, pitch_end=45.0)
    assert ring[0] == hero and {c.pitch for c in ring} == {5.0}
    assert abs(ring[1].yaw - ring[0].yaw - 360 / 81) < 1e-9
    spiral = campath.plan(hero, "spiral", 81, turns=2.0, pitch_end=45.0)
    assert spiral[0] == hero and spiral[-1].pitch == 45.0 and abs(spiral[-1].yaw - 30.0 - 720 * 80 / 81) < 1e-9
    wave = campath.plan(hero, "wave", 80, pitch_end=45.0)
    assert wave[0].pitch == 5.0 and abs(wave[40].pitch - 45.0) < 1e-9


def test_path_camera_looks_at_its_target_from_its_distance():
    cam = campath.PathCamera(70.0, 30.0, 2.0, (0.1, -0.2, 0.05))
    rot, t = cam.world_to_camera()
    assert np.allclose(rot @ rot.T, np.eye(3))
    p = rot @ np.asarray(cam.target) + t
    assert np.allclose(p[:2], 0) and abs(p[2] - 2.0) < 1e-9
    # positive pitch: the camera is above the target (the splat frame's y points down)
    assert cam.position()[1] < cam.target[1]


def test_proxy_orbit_workflow_renders_depth_at_the_video_size_with_the_hero_first():
    prompt = workflows.build_orbit("wan22-control", image="giro/hero.png", seed=3, width=768, height=1024,
                                   length=workflows.snap_length(80, "wan22-control"), proxy="/x/proxy.ply", cameras="[]")
    assert prompt["depth"]["inputs"]["width"] == 768 and prompt["depth"]["inputs"]["height"] == 1024
    assert prompt["condition"]["inputs"]["start_image"] == prompt["condition"]["inputs"]["ref_image"] == ["hero", 0]
    assert prompt["condition"]["inputs"]["control_video"] == ["depth", 0]
    assert prompt["condition"]["inputs"]["length"] == 81


def test_proxy_mode_adds_the_proxy_stage_and_its_params():
    assert [s.name for s in stages.pipeline(stages.PROXY_MODEL)][:2] == ["proxy", "orbit_video"]
    assert stages.pipeline(None) is stages.PIPELINE
    assert stages.mode_params(stages.PROXY_MODEL, "poses_colmap") == {"mapper": "path"}
    assert stages.mode_params("h3", "poses_colmap") == {}
    assert stages.mode_params(stages.PROXY_MODEL, "proxy") == {"model": "pixal3d"}


def test_proxy_orbit_starts_from_the_hero_cut_out_on_black(tmp_path):
    (tmp_path / "hero").mkdir()
    (tmp_path / "proxy").mkdir()
    Image.new("RGB", (8, 8), (200, 100, 50)).save(tmp_path / "hero" / "hero.png")
    mask = np.zeros((8, 8), np.uint8)
    mask[2:6, 2:6] = 255
    Image.fromarray(mask, "L").save(tmp_path / "proxy" / "hero_mask.png")
    out = stages.ORBIT._video_hero(tmp_path, {"model": stages.PROXY_MODEL})
    px = np.asarray(Image.open(out))
    assert out == tmp_path / "proxy_render" / "hero_black.png" and not list((tmp_path / "hero").glob("*black*"))
    assert (px[3, 3] == (200, 100, 50)).all() and (px[0, 0] == 0).all()
    assert stages.ORBIT._video_hero(tmp_path, {"model": stages.PROXY_MODEL, "hero_bg": "keep"}).name == "hero.png"
    assert stages.ORBIT._video_hero(tmp_path, {"model": "h3"}).name == "hero.png"


def _figure(path, n=6000, seed=0):
    """A figure ~1 tall in the splat frame (y down): a body, an arm sticking out to +x, a red
    front (+z side) and a blue back."""
    rng = np.random.default_rng(seed)
    body = rng.uniform([-0.15, -0.5, -0.1], [0.15, 0.5, 0.1], (n, 3))
    arm = rng.uniform([0.15, -0.3, -0.05], [0.45, -0.2, 0.05], (n // 4, 3))
    xyz = np.concatenate([body, arm])
    color = np.where(xyz[:, 2:3] > 0, [0.9, 0.1, 0.1], [0.1, 0.1, 0.9])
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity"]
    s = np.zeros(len(xyz), dtype=[(k, "<f4") for k in names])
    s["x"], s["y"], s["z"] = xyz.T
    for i in range(3):
        s[f"f_dc_{i}"] = (color[:, i] - 0.5) / 0.28209479177387814
    s["opacity"] = 5.0
    splat.write_ply(path, s)


def test_hero_camera_fit_recovers_the_view_front_and_back_apart(tmp_path):
    _figure(tmp_path / "proxy.ply")
    points = Points(tmp_path / "proxy.ply")
    truth = campath.PathCamera(20.0, 8.0, 2.2, tuple(points.center), 35.0)
    w, h = 192, 256
    mask = points.silhouette(truth, w, h)
    hero = Image.fromarray((points.color(truth, w, h) * 255).astype(np.uint8))
    cam, fit = fit_hero_camera(points, hero, mask, 35.0)
    assert fit["iou"] > 0.9
    assert abs((cam.yaw - truth.yaw + 180) % 360 - 180) < 8 and abs(cam.pitch - truth.pitch) < 6
    # the figure seen from the back has a mirrored silhouette: the fit must not pick it
    back = replace(truth, yaw=truth.yaw + 180)
    assert abs((cam.yaw - back.yaw + 180) % 360 - 180) > 90


def test_dataset_trains_on_hero_copies_away_from_eval_indices(tmp_path):
    colmap = tmp_path / "poses" / "colmap"
    model = colmap / "sparse" / "0"
    model.mkdir(parents=True)
    (model / "images.bin").write_bytes(b"")
    (colmap / "model").symlink_to("sparse/0")
    txt = colmap / "model_txt"
    txt.mkdir()
    names = [f"frames/{i:05d}.png" for i in range(14)] + ["hero/hero.png"]
    lines = []
    for i, name in enumerate(names):
        cam = 2 if name.startswith("hero") else 1
        lines += [f"{i + 1} 1 0 0 0 0.5 0 {i} {cam} {name}", ""]
    (txt / "images.txt").write_text("\n".join(lines) + "\n")
    (txt / "cameras.txt").write_text("1 SIMPLE_PINHOLE 8 8 10 4 4\n2 SIMPLE_PINHOLE 16 16 20 8 8\n")
    (txt / "points3D.txt").write_text("")
    for name in names:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"png")

    ctx = Ctx()
    Dataset().execute(tmp_path, {"eval_split_every": 8, "masks": False, "hero_copies": 5}, ctx)
    views = sorted(str(p.relative_to(tmp_path / "dataset" / "images")) for p in (tmp_path / "dataset" / "images").rglob("*.png"))
    heroes = [v for v in views if v.startswith("hero/")]
    assert heroes == ["hero/hero.png"] + [f"hero/hero_{k}.png" for k in range(2, 6)]
    assert all(views.index(v) % 8 != 0 for v in heroes)
    written = (tmp_path / "dataset" / "sparse" / "0" / "images.txt").read_text().split("\n")
    copies = [ln for ln in written if ln.endswith("hero_3.png")]
    assert copies and copies[0].split()[1:9] == ["1", "0", "0", "0", "0.5", "0", "14", "2"]
    assert ctx.metrics["hero_copies"] == 5
    assert json.loads((tmp_path / ".stages" / "dataset.json").read_text())["params"]["hero_copies"] == 5


def test_a_chained_clip_starts_where_the_last_one_ended():
    first = campath.plan(campath.PathCamera(10.0, 5.0, 1.5), "spiral", 81, turns=2.0, pitch_end=45.0)
    second = campath.segment(first[-1], 81, turns=1.0, pitch_end=70.0, distance_end=1.0, target_end=(0.0, -0.3, 0.0))
    assert second[0] == first[-1]
    assert second[-1].pitch == 70.0 and second[-1].distance == 1.0 and second[-1].target == (0.0, -0.3, 0.0)
    # eased: the first steps barely change elevation
    assert second[1].pitch - second[0].pitch < (70.0 - 45.0) / 80


def test_refine_sigmas_start_at_the_asked_noise_level_on_the_shifted_schedule():
    sigmas = workflows.refine_sigmas(0.35, 8, shift=8.0)
    assert sigmas[0] == 0.35 and sigmas[-1] == 0.0 and len(sigmas) == 9
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))
    prompt = workflows.build_refine(768, 1024, 0.35, 8, frames="[]", image="h", start="s", prompt="p", seed=1,
                                    proxy="/x.ply", cameras="[]", length=81, output_prefix="o")
    assert prompt["condition"]["inputs"]["width"] == prompt["depth"]["inputs"]["width"] == prompt["frames_in"]["inputs"]["width"] == 768
    assert prompt["sample"]["inputs"]["latent_image"] == ["encode", 0]


def test_new_jobs_get_the_proxy_orbit_and_old_attempts_stay_h3():
    orbit = stages.new_orbit({"steps": 20, "width": None})
    assert orbit["model"] == stages.DEFAULT_MODEL == stages.PROXY_MODEL
    assert (orbit["width"], orbit["height"], orbit["length"]) == (576, 768, 81)
    assert stages.new_orbit({"model": "wan22-control", "width": 768, "height": 576})["width"] == 768
    h3 = stages.new_orbit({"model": "h3"})
    assert h3 == {"model": "h3"}  # H3's size and length come from orbit_video's defaults
    # an attempt that recorded no model (all of them before the switch) is H3
    assert stages.pipeline(None) is stages.PIPELINE and stages.mode_params(None, "dataset") == {}
