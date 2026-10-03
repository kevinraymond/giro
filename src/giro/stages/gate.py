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
        if len(frames) >= 3:
            ring = analyze_ring([registered[n]["qvec"] for n in frames], [registered[n]["tvec"] for n in frames])
            values |= {k: v for k, v in ring.items() if k != "azimuths"}

        checks, reasons = [], []
        for metric, op, param, reason in CHECKS:
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
