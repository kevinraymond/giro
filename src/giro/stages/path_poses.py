"""Poses of a proxy orbit from its known camera path (the poses stage's mapper "path").

The frames were rendered from cameras giro chose (cameras.json), so those cameras are the
start: COLMAP triangulates the stage's own masked matches from them and bundle-adjusts twice
(the pose fallback's recipe), which absorbs where Wan's video strays a little from the depth it
was given. Cameras that too few points constrain keep their path pose, moved into the refined
frame (fallback.hybrid). On the Oct 3 spike this posed 81 of 81 frames of a two-turn spiral to
45 degrees, where COLMAP's own mapper posed 27 (docs/FINDINGS.md, "Proxy orbit").

Where bundle adjustment moves the cameras far from the path (MAX_MOVED), its matches are too few
or poor to trust, and the path's cameras are kept as they are: the video follows them closely
(the gate checks how closely, from the subject's silhouettes).

The hero is the video's first frame (start_image), so it is not adjusted on its own: bundle
adjustment trades its focal length for distance (on the first run it came out 1.49x longer and
half an orbit radius further back). It takes frame 0's refined pose, with the frames' focal
length scaled to its size.

The proxy's own frame (center, up, front) is carried into the refined world through the
similarity between path and refined cameras, and written to poses/frame.json for canonicalize.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

import numpy as np

from giro import path as campath
from giro.stages.base import Ctx, Rejected
from giro.stages.fallback import _centers, _pose, _qvec, hybrid, similarity
from giro.stages.poses import read_images_txt

MIN_POINTS = 15  # an image bundle adjustment constrains sees at least this many points
# Signs that bundle adjustment had too little to go on, so the path's cameras are kept unadjusted.
# Good attempts (Oct 4): median movement 0.010-0.035 orbit radii, focal 1-5% off, 0-8% of frames
# unconstrained; the glossy scooter: 0.046-0.065, 11.5% and 20%.
MAX_MOVED = 0.04
MAX_FOCAL_CHANGE = 0.08
MAX_UNCONSTRAINED = 0.15
HERO, FIRST = "hero/hero.png", "frames/00000.png"


def path_cameras(attempt: Path) -> dict[str, tuple[campath.PathCamera, int, int]]:
    """Image name -> (camera, width, height) for every frame and the hero."""
    data = json.loads((attempt / "cameras.json").read_text())
    proxy = json.loads((attempt / "proxy" / "proxy.json").read_text())
    out = {f"frames/{i:05d}.png": (campath.PathCamera.from_json(c), data["width"], data["height"])
           for i, c in enumerate(data["frames"])}
    hw, hh = proxy["hero_size"]
    out["hero/hero.png"] = (campath.PathCamera.from_json(data["hero"]), hw, hh)
    return out


def reconstruct(attempt: Path, work: Path, colmap: Callable[..., str], ctx: Ctx, adjust: bool = True) -> Path:
    """Triangulate and bundle-adjust from the path cameras; returns the model's directory
    (work/sparse/0, with a TXT copy in work/sparse/0_txt) and records the metrics. adjust=False
    keeps the path's cameras as they are and only triangulates points from them."""
    db, images, sparse = work / "database.db", work / "images", work / "sparse"
    cams = path_cameras(attempt)
    with sqlite3.connect(db) as con:
        db_images = {n: (i, c) for i, n, c in con.execute("SELECT image_id, name, camera_id FROM images")}
        db_cams = {c: (w, h) for c, w, h in con.execute("SELECT camera_id, width, height FROM cameras")}
    missing = [n for n in db_images if n not in cams]
    if missing:
        raise Rejected(f"{len(missing)} images have no path camera (first: {missing[0]}); was cameras.json rewritten?")

    # One intrinsic per COLMAP camera (frames share one, the hero has its own), from the path's fov.
    focal: dict[int, float] = {}
    for name, (image_id, cam_id) in db_images.items():
        cam, w, h = cams[name]
        focal.setdefault(cam_id, cam.focal(w, h))
    start = work / "path_start"
    start.mkdir()
    (start / "cameras.txt").write_text("".join(
        f"{c} SIMPLE_PINHOLE {w} {h} {focal[c]} {w / 2} {h / 2}\n" for c, (w, h) in db_cams.items()))
    nominal = {n: cams[n][0].world_to_camera() for n in db_images}
    (start / "images.txt").write_text("".join(
        f"{i} {' '.join(map(str, _qvec(nominal[n][0])))} {' '.join(map(str, nominal[n][1]))} {c} {n}\n\n"
        for n, (i, c) in sorted(db_images.items(), key=lambda x: x[1][0]) if n != HERO))
    (start / "points3D.txt").write_text("")

    model = start
    for rnd in (1, 2) if adjust else (1,):
        tri, ba = work / f"path_triangulated{rnd}", work / f"path_adjusted{rnd}"
        tri.mkdir()
        colmap(f"triangulating from the path ({rnd}/2)", 0.55 + 0.15 * (rnd - 1), "point_triangulator",
               "--database_path", str(db), "--image_path", str(images),
               "--input_path", str(model), "--output_path", str(tri), "--clear_points", "1")
        model = tri
        if adjust:
            ba.mkdir()
            colmap(f"bundle adjustment ({rnd}/2)", 0.62 + 0.15 * (rnd - 1), "bundle_adjuster",
                   "--input_path", str(tri), "--output_path", str(ba),
                   "--BundleAdjustment.refine_focal_length", "1", "--BundleAdjustment.refine_principal_point", "0",
                   "--BundleAdjustment.refine_extra_params", "0")
            model = ba
    adjusted_txt = work / "path_adjusted_txt"
    adjusted_txt.mkdir()
    colmap("reading model", 0.85, "model_converter", "--input_path", str(model),
           "--output_path", str(adjusted_txt), "--output_type", "TXT")
    adjusted = read_images_txt(adjusted_txt / "images.txt")
    constrained = [n for n, im in adjusted.items() if im["n_points"] >= MIN_POINTS]
    if len(constrained) < 3:
        raise Rejected(f"bundle adjustment constrained only {len(constrained)} of {len(adjusted)} images: "
                       "the frames do not match each other (the video likely ignored the control)")
    refined = {n: _pose(im) for n, im in adjusted.items()}
    poses = hybrid({n: nominal[n] for n in adjusted}, refined, constrained)

    # How far bundle adjustment moved the cameras from the path, in orbit radii, after a similarity fit.
    frames_c = [n for n in constrained if n.startswith("frames/")]
    s, r, t = similarity(_centers(nominal, constrained), _centers(refined, constrained))
    radius = float(np.mean([cams[n][0].distance for n in frames_c])) if frames_c else 1.0
    moved = np.linalg.norm((s * (r @ _centers(nominal, frames_c).T).T + t) - _centers(refined, frames_c), axis=1) / (s * radius)
    if adjust:
        # Few or poor matches (a glossy scooter seen from above) let bundle adjustment pull the
        # cameras off a path the video follows; the path's own cameras train a clean splat there.
        frame_cam = db_images[FIRST][1]
        adj_focal = next(float(ln.split()[4]) for ln in (adjusted_txt / "cameras.txt").read_text().splitlines()
                         if ln and not ln.startswith("#") and int(ln.split()[0]) == frame_cam)
        first_cam, fw, fh = cams[FIRST]
        n_frames = sum(n.startswith("frames/") for n in adjusted)
        doubts = {
            "moved": float(np.median(moved)) if len(moved) else 0.0,
            "focal": abs(adj_focal / first_cam.focal(fw, fh) - 1),
            "unconstrained": 1 - len(frames_c) / max(1, n_frames),
        }
        if doubts["moved"] > MAX_MOVED or doubts["focal"] > MAX_FOCAL_CHANGE or doubts["unconstrained"] > MAX_UNCONSTRAINED:
            ctx.metric("adjustment_doubts", {k: round(v, 4) for k, v in doubts.items()})
            ctx.log(f"bundle adjustment looks unreliable (median move {doubts['moved']:.1%} of the radius, focal "
                    f"{doubts['focal']:.1%} off, {doubts['unconstrained']:.0%} of frames unconstrained); keeping the path's cameras")
            for d in [work / "path_start", work / "path_adjusted_txt", *work.glob("path_triangulated*"), *work.glob("path_adjusted?")]:
                shutil.rmtree(d)
            return reconstruct(attempt, work, colmap, ctx, adjust=False)

    # The proxy's frame (y down) in the refined world; front points toward the hero camera.
    hero_cam = cams["hero/hero.png"][0]
    front = -hero_cam.forward()
    front[1] = 0.0
    center = np.asarray(hero_cam.target, dtype=float)
    (attempt / "poses" / "frame.json").write_text(json.dumps({
        "center": (s * r @ center + t).tolist(), "up": (r @ np.array([0.0, -1.0, 0.0])).tolist(),
        "front": (r @ (front / np.linalg.norm(front))).tolist(), "source": "proxy path",
    }, indent=2) + "\n")

    # The final model: refined cameras, hybrid poses for the unconstrained ones.
    out_txt = sparse / "0_txt"
    out_txt.mkdir(parents=True)
    (out_txt / "points3D.txt").write_text((adjusted_txt / "points3D.txt").read_text())
    cam_lines = [ln for ln in (adjusted_txt / "cameras.txt").read_text().splitlines() if ln and not ln.startswith("#")]
    frame_cam = db_images[FIRST][1]
    frame_focal = next(float(ln.split()[4]) for ln in cam_lines if int(ln.split()[0]) == frame_cam)
    if HERO in db_images:  # the hero: frame 0's pose, the frames' focal length at its size
        hero_id, hero_cam = db_images[HERO]
        hw, hh = db_cams[hero_cam]
        cam_lines = [ln for ln in cam_lines if int(ln.split()[0]) != hero_cam]
        cam_lines.append(f"{hero_cam} SIMPLE_PINHOLE {hw} {hh} {frame_focal * hw / db_cams[frame_cam][0]} {hw / 2} {hh / 2}")
        poses[HERO] = poses[FIRST]
    (out_txt / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
    lines = [ln for ln in (adjusted_txt / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    with open(out_txt / "images.txt", "w") as f:
        for header, points in zip(lines[0::2], lines[1::2]):
            h = header.split()
            rot, tv = poses[h[9]]
            f.write(f"{h[0]} {' '.join(f'{x:.12g}' for x in _qvec(rot))} {' '.join(f'{x:.12g}' for x in tv)} {h[8]} {h[9]}\n{points}\n")
        if HERO in db_images:
            rot, tv = poses[HERO]
            f.write(f"{hero_id} {' '.join(f'{x:.12g}' for x in _qvec(rot))} {' '.join(f'{x:.12g}' for x in tv)} {hero_cam} {HERO}\n\n")
    (sparse / "0").mkdir()
    colmap("writing model", 0.9, "model_converter", "--input_path", str(out_txt),
           "--output_path", str(sparse / "0"), "--output_type", "BIN")

    stats = colmap("analyzing model", 0.95, "model_analyzer", "--path", str(sparse / "0"))
    ctx.metric("n_constrained", len(constrained))
    ctx.metric("path_adjusted", adjust)
    ctx.metric("path_moved_median", round(float(np.median(moved)), 4) if len(moved) else None)
    ctx.metric("path_moved_max", round(float(moved.max()), 4) if len(moved) else None)
    # The hero stands in for frame 0, so what the gate needs is that frame 0 is well placed.
    ctx.metric("hero_constrained", FIRST in constrained)
    first_cam, fw, fh = cams[FIRST]
    ctx.metric("focal_ratio", round(frame_focal / first_cam.focal(fw, fh), 4))
    pts = re.search(r"Points:\s*(\d+)", stats)
    ctx.metric("n_points", int(pts.group(1)) if pts else None)
    return sparse / "0"
