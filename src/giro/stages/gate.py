"""gate: decide from the recovered cameras whether an attempt is worth training.

The frames of a good orbit video give cameras on one ring around the subject,
visited in order, sweeping (nearly) a full turn and returning to the start.
Everything is measured from the COLMAP model, in time order of the frames:

- orbit center: least-squares closest point to all optical axes
- orbit plane: PCA of the camera centers
- azimuth: angle of each camera center around the center, in that plane
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from giro.stages.base import Ctx, Rejected, Stage
from giro.stages.poses import SOURCES, active_source, read_images_txt


def qvec_to_rotmat(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def analyze_ring(qvecs: list[list[float]], tvecs: list[list[float]]) -> dict[str, Any]:
    """Orbit geometry of world-to-camera poses given in time order."""
    rots = [qvec_to_rotmat(q) for q in qvecs]
    centers = np.array([-r.T @ np.asarray(t) for r, t in zip(rots, tvecs)])
    axes = np.array([r[2] for r in rots])  # camera +z (viewing direction) in world coordinates

    # Closest point to all optical axes: sum_i (I - d d^T)(p - c_i) = 0.
    proj = np.eye(3)[None] - axes[:, :, None] * axes[:, None, :]
    center = np.linalg.lstsq(proj.sum(0), np.einsum("nij,nj->i", proj, centers), rcond=None)[0]

    _, sing, vt = np.linalg.svd(centers - centers.mean(0))
    e1, e2 = vt[0], vt[1]
    rel = centers - center
    in_plane = np.stack([rel @ e1, rel @ e2], axis=1)
    radius = np.linalg.norm(in_plane, axis=1)
    azimuth = np.degrees(np.arctan2(in_plane[:, 1], in_plane[:, 0]))

    steps = (np.diff(azimuth) + 180.0) % 360.0 - 180.0  # wrapped into [-180, 180)
    direction = 1.0 if steps.sum() >= 0 else -1.0
    forward = steps * direction
    turned = np.concatenate([[0.0], np.cumsum(steps)])  # unwrapped azimuth relative to the first frame
    mean_radius = float(radius.mean())
    return {
        "azimuth_coverage": round(float(abs(steps.sum())), 1),  # net: where the orbit ends up
        "azimuth_span": round(float(turned.max() - turned.min()), 1),  # range: what the camera saw
        "azimuth_monotonic": round(float((forward > 0).mean()), 3) if len(steps) else 0.0,
        "max_step_deg": round(float(np.abs(steps).max()), 1) if len(steps) else 0.0,
        "loop_closure": round(float(np.linalg.norm(centers[-1] - centers[0]) / mean_radius), 3),
        "radius_cv": round(float(radius.std() / mean_radius), 3),
        "planarity": round(float(sing[2] / sing[0]), 4),
        "radius": round(mean_radius, 3),
        "azimuths": [round(float(a), 2) for a in azimuth],
    }


# (metric, comparison, threshold param, plain-language reason when it fails)
CHECKS: list[tuple[str, str, str, str]] = [
    ("reg_rate", ">=", "min_reg_rate",
     "only {value:.0%} of the images got a camera pose (need {threshold:.0%})"),
    ("hero_registered", "==", "require_hero",
     "the hero image got no camera pose, so its detail cannot be used"),
    ("n_views", ">=", "min_views",
     "only {value} frames got a camera pose (need {threshold})"),
    ("reproj_err", "<=", "max_reproj_err",
     "the frames are geometrically inconsistent: reprojection error {value:.2f} px (need at most {threshold})"),
    ("azimuth_coverage", ">=", "min_azimuth_coverage",
     "the camera covers only {azimuth_span:.0f} degrees around the subject and ends {value:.0f} degrees "
     "from where it started (need {threshold:.0f} net)"),
    ("azimuth_monotonic", ">=", "min_azimuth_monotonic",
     "the camera does not move steadily one way: only {value:.0%} of steps go forward (need {threshold:.0%})"),
    ("max_step_deg", "<=", "max_step_deg",
     "the camera path jumps {value:.0f} degrees between two consecutive frames (allowed {threshold:.0f}): "
     "views are missing, or COLMAP misplaced some cameras (repeated patterns such as rows of windows cause this)"),
    ("loop_closure", "<=", "max_loop_closure",
     "the orbit does not return to where it started: the last frame is {value:.2f} radii from the first (allowed {threshold})"),
    ("radius_cv", "<=", "max_radius_cv",
     "the camera distance to the subject varies by {value:.0%} (allowed {threshold:.0%})"),
]


# A proxy orbit's cameras come from a known path (poses mapper "path"), which need not be a ring
# (a spiral climbs): its own checks replace the ring's.
PATH_CHECKS: list[tuple[str, str, str, str]] = [
    ("reg_rate", ">=", "min_reg_rate",
     "only {value:.0%} of the images got a camera pose (need {threshold:.0%})"),
    ("hero_constrained", "==", "require_hero",
     "bundle adjustment could not place the hero (too few matches with the frames), so its detail cannot be used"),
    ("n_views", ">=", "min_views",
     "only {value} frames got a camera pose (need {threshold})"),
    ("reproj_err", "<=", "max_reproj_err",
     "the frames are geometrically inconsistent: reprojection error {value:.2f} px (need at most {threshold})"),
    ("constrained_rate", ">=", "min_constrained_rate",
     "bundle adjustment placed only {value:.0%} of the frames from their matches (need {threshold:.0%}); the others keep "
     "their path pose unchecked"),
    ("band_control_iou_min", ">=", "min_band_control_iou",
     "at one elevation of the path the subject's silhouette matches the proxy's only {value:.0%} on average "
     "(need {threshold:.0%}): the video drifted from the control there"),
    ("path_moved_median", "<=", "max_path_moved_median",
     "bundle adjustment moved the cameras a median {value:.0%} of the orbit radius from the path (allowed {threshold:.0%}): "
     "the video does not follow the proxy's depth"),
    ("control_iou_mean", ">=", "min_control_iou",
     "the subject's silhouette matches the proxy's only {value:.0%} on average (need {threshold:.0%}): "
     "the video does not follow the proxy's depth"),
]


def path_values(attempt: Path, registered: dict[str, dict[str, Any]], poses_metrics: dict[str, Any]) -> dict[str, Any]:
    """Measurements of a path orbit: constrained frames overall and per elevation band, and how
    well each frame's subject mask matches the proxy silhouette it was rendered from."""
    from giro import path as campath
    from giro.stages.path_poses import MIN_POINTS

    cams = [campath.PathCamera.from_json(c) for c in json.loads((attempt / "cameras.json").read_text())["frames"]]
    frames = sorted((attempt / "frames").glob("*.png"))
    constrained = {n for n, im in registered.items() if im["n_points"] >= MIN_POINTS}
    iou_of: dict[str, float] = {}
    for f in frames:
        sil, mask = attempt / "proxy_render" / "silhouettes" / f.name, attempt / "masks" / "frames" / f.name
        if sil.exists() and mask.exists():
            a = np.asarray(Image.open(sil).convert("L")) > 127
            b = np.asarray(Image.open(mask).convert("L").resize(a.shape[::-1])) > 127
            iou_of[f"frames/{f.name}"] = float((a & b).sum() / max(1, (a | b).sum()))
    ious = list(iou_of.values())
    # Per elevation band: the share of frames bundle adjustment placed (few on smooth, glossy
    # surfaces seen from above: a scooter's band above 40 degrees had 17% while its video followed
    # the depth), and how well the video followed the proxy there, which is what catches drift.
    by_band, iou_band = {}, {}
    for band, idx in campath.elevation_bands(cams).items():
        names = [f"frames/{i:05d}.png" for i in idx if (attempt / "frames" / f"{i:05d}.png").exists()]
        if len(names) >= 5:  # a band the path only grazes says little
            by_band[band] = round(sum(n in constrained for n in names) / len(names), 3)
            if any(n in iou_of for n in names):
                iou_band[band] = round(float(np.mean([iou_of[n] for n in names if n in iou_of])), 3)
    return {
        "hero_constrained": bool(poses_metrics.get("hero_constrained")),
        "constrained_rate": round(sum(f"frames/{f.name}" in constrained for f in frames) / max(1, len(frames)), 3),
        "band_constrained": {str(k): v for k, v in sorted(by_band.items())},
        "band_constrained_min": min(by_band.values()) if by_band else 0.0,
        "band_control_iou": {str(k): v for k, v in sorted(iou_band.items())},
        "band_control_iou_min": min(iou_band.values()) if iou_band else None,
        "path_moved_median": poses_metrics.get("path_moved_median"),
        "path_moved_max": poses_metrics.get("path_moved_max"),
        "control_iou_mean": round(float(np.mean(ious)), 3) if ious else None,
        "control_iou_min": round(float(np.min(ious)), 3) if ious else None,
    }


class Gate(Stage):
    name = "gate"
    defaults = {
        "min_reg_rate": 0.90,
        "require_hero": True,
        "min_views": 60,
        "max_reproj_err": 1.5,
        "min_azimuth_coverage": 330.0,
        "min_azimuth_monotonic": 0.95,
        "max_step_deg": 30.0,
        "max_loop_closure": 0.3,
        "max_radius_cv": 0.1,
        "enforce": True,  # False records the verdict without stopping the pipeline
    }
    # Also read, for path orbits only (PATH_CHECKS), so ring attempts keep their stage key.
    path_defaults = {
        "min_constrained_rate": 0.6,
        "min_band_control_iou": 0.85,
        # Oct 4: 0.010-0.035 on good attempts; 0.065 on a glossy scooter whose few matches pulled
        # the cameras off the path and broke the splat
        "max_path_moved_median": 0.05,
        "min_control_iou": 0.7,
    }
    extra_params = tuple(path_defaults)
    inputs = ("poses/colmap/model_txt", "frames", "hero")
    outputs = ("gate.json",)

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        registered = read_images_txt(attempt / "poses" / "colmap" / "model_txt" / "images.txt")
        n_inputs = len(list((attempt / "frames").glob("*.png"))) + len(list((attempt / "hero").glob("*.png")))
        frames = sorted(name for name in registered if name.startswith("frames/"))
        source = active_source(attempt)  # COLMAP's own cameras, or the pose fallback's
        poses_metrics = json.loads((attempt / "metrics.json").read_text()).get(SOURCES[source], {})

        values: dict[str, Any] = {
            "reg_rate": round(len(registered) / n_inputs, 4) if n_inputs else 0.0,
            "hero_registered": any(name.startswith("hero/") for name in registered),
            "n_views": len(frames),
            "reproj_err": poses_metrics.get("reproj_err"),
        }
        ring: dict[str, Any] = {}
        is_path = poses_metrics.get("mapper") == "path"
        if is_path:
            values |= path_values(attempt, registered, poses_metrics)
            params = self.path_defaults | params
        elif len(frames) >= 3:
            ring = analyze_ring([registered[n]["qvec"] for n in frames], [registered[n]["tvec"] for n in frames])
            values |= {k: v for k, v in ring.items() if k != "azimuths"}

        checks, reasons = [], []
        for metric, op, param, reason in PATH_CHECKS if is_path else CHECKS:
            value, threshold = values.get(metric), params[param]
            if op == "==" and threshold is False:
                continue  # requirement switched off
            ok = value is not None and {
                ">=": lambda: value >= threshold, "<=": lambda: value <= threshold, "==": lambda: value == threshold,
            }[op]()
            checks.append({"metric": metric, "value": value, "op": op, "threshold": threshold, "pass": ok})
            if not ok:
                reasons.append(reason.format(**values | {"value": value if value is not None else math.nan,
                                                         "threshold": threshold}))
        passed = not reasons

        for k, v in values.items():
            ctx.metric(k, v)
        ctx.metric("poses", source)
        ctx.metric("passed", passed)
        if reasons:
            ctx.metric("reasons", reasons)
        (attempt / "gate.json").write_text(json.dumps({
            "passed": passed, "reasons": reasons, "checks": checks, "poses": source,
            "ring": ring | {"frames": frames},
        }, indent=2) + "\n")
        ctx.progress(1.0, "passed" if passed else "failed")
        if not passed and params["enforce"]:
            raise Rejected("; ".join(reasons))
