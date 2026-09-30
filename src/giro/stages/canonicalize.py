"""canonicalize: stand the cropped subject upright at the origin, at real-world size.

In the viewer's frame (y up, as PlayCanvas, SuperSplat and splat-transform
show a 3DGS PLY): the orbit axis becomes +Y, the hero view faces +Z (the
side a default camera looks at), the subject's lowest point is at y = 0 with
its footprint centered on the origin, and its height is `height_m` meters.

- up:    normal of the camera ring (PCA of the frame cameras), signed to agree
         with the cameras' image-up direction
- front: from the orbit center toward the hero camera, in the ring plane

The PLY on disk keeps the usual 3DGS file convention (y down), so viewers
that read PLY show it upright; transform.json records the mapping.
splat-transform applies it, which also rotates the spherical harmonics.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from giro import render, splat
from giro.stages.base import Ctx, Stage, StageFailed
from giro.stages.crop import View, load_views, robust_extent


def orbit_frame(views: list[View]) -> tuple[np.ndarray, np.ndarray]:
    """(orbit center, rotation M whose rows are the canonical X, Y (up), Z (front) axes in world)."""
    frames = [v for v in views if v.name.startswith("frames/")]
    if len(frames) < 3:
        raise StageFailed("need at least 3 posed frames to find the orbit")
    centers = np.array([v.center for v in frames])
    axes = np.array([v.rot[2] for v in frames])
    proj = np.eye(3)[None] - axes[:, :, None] * axes[:, None, :]
    center = np.linalg.lstsq(proj.sum(0), np.einsum("nij,nj->i", proj, centers), rcond=None)[0]

    up = np.linalg.svd(centers - centers.mean(0))[2][2]
    image_up = -np.mean([v.rot[1] for v in frames], axis=0)  # camera +y points down the image
    if up @ image_up < 0:
        up = -up

    hero = next((v for v in views if v.name.startswith("hero/")), frames[0])
    front = hero.center - center
    front -= (front @ up) * up
    front /= np.linalg.norm(front)
    right = np.cross(up, front)
    return center, np.stack([right, up, front])


class Canonicalize(Stage):
    name = "canonicalize"
    defaults = {"height_m": 1.7}
    inputs = ("crop/cropped.ply", "poses/colmap/model_txt")
    outputs = ("canonical/splat.ply", "canonical/transform.json")

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        out = attempt / "canonical"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        views = load_views(attempt / "poses" / "colmap" / "model_txt")
        center, m = orbit_frame(views)
        points = splat.positions(splat.read_ply(attempt / "crop" / "cropped.ply"))

        local = (points - center) @ m.T  # viewer-frame axes, orbit center at the origin, world units
        lo, hi = robust_extent(local)
        height = float(hi[1] - lo[1])
        scale = params["height_m"] / height
        offset = -scale * np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])

        # splat-transform: internal = FLIP @ file; then translate, rotate, scale, translate.
        # viewer = scale * m @ (world - center) + offset, so rotate by m @ FLIP after moving
        # the center to the origin.
        euler = Rotation.from_matrix(m @ render.FLIP).as_euler("xyz", degrees=True)
        t_center = -(render.FLIP @ center)
        render.splat_transform(
            str(attempt / "crop" / "cropped.ply"),
            "-t", ",".join(f"{x:.9g}" for x in t_center),
            "-r", ",".join(f"{x:.9g}" for x in euler),
            "-s", f"{scale:.9g}",
            "-t", ",".join(f"{x:.9g}" for x in offset),
            str(out / "splat.ply"),
        )
        check = splat.positions(splat.read_ply(out / "splat.ply")) @ render.FLIP  # file -> viewer
        expected = scale * local + offset
        err = float(np.abs(check - expected).max())
        if err > 1e-3 * params["height_m"]:
            raise StageFailed(f"splat-transform moved the Gaussians {err:.4f} m away from where giro expected")

        (out / "transform.json").write_text(json.dumps({
            "convention": "viewer = scale * R @ (world - center) + offset; PLY file = diag(-1,-1,1) @ viewer",
            "center": center.tolist(), "R": m.tolist(), "scale": scale, "offset": offset.tolist(),
            "height_m": params["height_m"],
            "footprint_m": [round(float(scale * (hi[0] - lo[0])), 3), round(float(scale * (hi[2] - lo[2])), 3)],
        }, indent=2) + "\n")
        ctx.metric("scale", round(scale, 5))
        ctx.metric("height_m", params["height_m"])
        ctx.metric("width_m", round(float(scale * (hi[0] - lo[0])), 3))
        ctx.metric("depth_m", round(float(scale * (hi[2] - lo[2])), 3))

        ctx.progress(0.7, "rendering turnaround")
        h = params["height_m"]
        size = (scale * (hi[0] - lo[0]), h, scale * (hi[2] - lo[2]))
        dist = render.framing_distance(size)
        cams = render.orbit_cameras((0.0, h / 2, 0.0), dist, 8, elevation_deg=10)
        cams += render.orbit_cameras((0.0, h / 2, 0.0), dist, 4, elevation_deg=55)
        cams += render.orbit_cameras((0.0, h / 2, 0.0), dist, 4, elevation_deg=-30)
        images = render.render(out / "splat.ply", cams, (384, 512), background=(0.5, 0.5, 0.5))
        render.sheet(images, cols=8).save(out / "turnaround.jpg", quality=90)
        ctx.preview(out / "turnaround.jpg")
        ctx.progress(1.0, f"{h} m tall, scale {scale:.3f}")
