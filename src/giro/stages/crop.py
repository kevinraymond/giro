"""crop: keep the Gaussians of the subject, by visual hull over the masks.

Each Gaussian center is projected into every posed view (frames and hero).
It is kept when it is in frame in at least half of the views and lands inside
the (slightly dilated) subject mask in at least `tau` of those. Then near-invisible
Gaussians (low opacity) and oversized ones (floaters, blobs smeared over the
background) are dropped too.

No occlusion reasoning is needed: a silhouette contains everything of the
subject along each ray, so the subject's Gaussians are inside every mask,
while background Gaussians fall outside the silhouette from most directions.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from giro import render, splat
from giro.stages.base import Ctx, Stage, StageFailed
from giro.stages.gate import qvec_to_rotmat
from giro.stages.poses import read_cameras_txt, read_images_txt


@dataclass
class View:
    name: str
    rot: np.ndarray    # world -> camera
    trans: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def center(self) -> np.ndarray:
        return -self.rot.T @ self.trans

    def project(self, points: np.ndarray, near: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel coordinates (u, v) and whether each point is in front of the camera."""
        cam = points @ self.rot.T + self.trans
        z = cam[:, 2]
        front = z > near
        zs = np.where(front, z, 1.0)
        return self.fx * cam[:, 0] / zs + self.cx, self.fy * cam[:, 1] / zs + self.cy, front

    def camera(self) -> render.Camera:
        fov = float(np.degrees(2 * np.arctan(self.height / (2 * self.fy))))
        return render.Camera(self.center, self.center + self.rot[2], -self.rot[1], fov)


def load_views(model_txt: Path) -> list[View]:
    cameras = read_cameras_txt(model_txt / "cameras.txt")
    views = []
    for name, im in sorted(read_images_txt(model_txt / "images.txt").items()):
        c = cameras[im["camera_id"]]
        views.append(View(name, qvec_to_rotmat(im["qvec"]), np.asarray(im["tvec"], dtype=float),
                          c["fx"], c["fy"], c["cx"], c["cy"], c["width"], c["height"]))
    return views


def hull_votes(points: np.ndarray, views: list[View], masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per point: in how many views it is in the frustum, and in how many of those inside the mask."""
    n_vis = np.zeros(len(points), dtype=np.int32)
    n_in = np.zeros(len(points), dtype=np.int32)
    for view, mask in zip(views, masks):
        u, v, front = view.project(points)
        h, w = mask.shape
        vis = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui, vi = u[vis].astype(np.int32), v[vis].astype(np.int32)
        n_vis += vis
        n_in[vis] += mask[vi, ui]
    return n_vis, n_in


def robust_extent(points: np.ndarray, q: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis low and high percentiles (q and 100-q) of the points."""
    return np.percentile(points, q, axis=0), np.percentile(points, 100 - q, axis=0)


class Crop(Stage):
    name = "crop"
    defaults = {
        "tau": 0.8,             # fraction of the views that must see a Gaussian inside the mask
        "dilate_px": 4,         # grow each mask first: SAM edges are soft and poses not exact
        # The subject is in frame in (nearly) every view of an orbit; the room behind the cameras
        # only in the few that face it, where a stray mask edge could otherwise win the vote.
        "min_visible_frac": 0.5,
        "min_opacity": 0.05,
        "max_scale": 0.1,       # largest axis (std dev) as a fraction of the subject's size
    }
    inputs = ("train/final.ply", "masks/frames", "masks/hero/hero.png", "poses/colmap/model_txt")
    # votes.f32: per Gaussian of train/final.ply, the fraction of views that see it inside the
    # mask (-1: seen by too few views, -2: pruned for opacity or size), so the UI can preview tau.
    outputs = ("crop/cropped.ply", "crop/crop.json", "crop/votes.f32")

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        out = attempt / "crop"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        ctx.progress(0.0, "loading splat and masks")
        gaussians = splat.read_ply(attempt / "train" / "final.ply")
        points = splat.positions(gaussians)
        views = load_views(attempt / "poses" / "colmap" / "model_txt")
        masks = []
        for view in views:
            path = attempt / "masks" / view.name
            if not path.exists():
                raise StageFailed(f"no mask for {view.name}; run the masks stage")
            mask = np.asarray(Image.open(path)) > 127
            if params["dilate_px"]:
                mask = ndimage.binary_dilation(mask, iterations=params["dilate_px"])
            masks.append(mask)

        ctx.progress(0.3, f"voting over {len(views)} views")
        n_vis, n_in = hull_votes(points, views, masks)
        in_hull = (n_vis >= params["min_visible_frac"] * len(views)) & (n_in >= params["tau"] * np.maximum(n_vis, 1))
        if in_hull.sum() < 100:
            raise StageFailed(f"only {in_hull.sum()} Gaussians fall inside the subject masks; the masks or poses are wrong")

        lo, hi = robust_extent(points[in_hull])
        size = float(np.linalg.norm(hi - lo))
        opaque = splat.opacities(gaussians) >= params["min_opacity"]
        small = splat.scales(gaussians).max(axis=1) <= params["max_scale"] * size
        keep = in_hull & opaque & small
        cropped = gaussians[keep]
        splat.write_ply(out / "cropped.ply", cropped)
        visible = n_vis >= params["min_visible_frac"] * len(views)
        votes = np.where(visible, n_in / np.maximum(n_vis, 1), -1.0).astype("<f4")
        votes[~(opaque & small)] = -2.0
        votes.tofile(out / "votes.f32")

        counts = {
            "n_input": len(gaussians),
            "n_hull": int(in_hull.sum()),
            "n_low_opacity": int((in_hull & ~opaque).sum()),
            "n_oversized": int((in_hull & opaque & ~small).sum()),
            "n_kept": int(keep.sum()),
        }
        (out / "crop.json").write_text(json.dumps(counts | {"subject_size": round(size, 4)}, indent=2) + "\n")
        for k, v in counts.items():
            ctx.metric(k, v)
        ctx.metric("kept_frac", round(counts["n_kept"] / counts["n_input"], 4))

        ctx.progress(0.8, "rendering the hero view")
        hero = next((v for v in views if v.name.startswith("hero/")), views[0])
        size_px = (hero.width * 512 // hero.height, 512)
        before, after = render.render(attempt / "train" / "final.ply", [hero.camera()], size_px)[0], \
            render.render(out / "cropped.ply", [hero.camera()], size_px)[0]
        reference = Image.open(attempt / hero.name).convert("RGB").resize(size_px)
        render.sheet([reference, before, after], cols=3).save(out / "preview_hero.jpg", quality=90)
        ctx.preview(out / "preview_hero.jpg")
        ctx.progress(1.0, f"kept {counts['n_kept']:,} of {counts['n_input']:,} Gaussians")
