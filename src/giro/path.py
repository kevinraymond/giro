"""Camera paths for the proxy orbit: where the proxy is rendered from, frame by frame.

Cameras live in the proxy splat's own frame, as ComfyUI's splat renderer has it: y points
down, a camera at yaw 0 and pitch 0 looks along +z, and a positive pitch looks down on the
subject from above. The same cameras are rendered (GiroRenderSplatCameras) and handed to COLMAP
as the frames' starting poses, so the two can never disagree.

The first camera of every path is the hero's (fitted by the proxy stage), so the hero can be
the video's exact first frame.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

PRESETS = ("ring", "spiral", "wave", "loop")


@dataclass(frozen=True)
class PathCamera:
    yaw: float       # degrees about the vertical axis
    pitch: float     # degrees; positive looks down from above
    distance: float  # from the target, in proxy units (the proxy is ~1 tall)
    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fov: float = 35.0  # degrees over the image's smaller side (ComfyUI's renderer)

    def forward(self) -> np.ndarray:
        y, p = math.radians(self.yaw), math.radians(self.pitch)
        return np.array([-math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])

    def position(self) -> np.ndarray:
        return np.asarray(self.target, dtype=float) - self.distance * self.forward()

    def world_to_camera(self) -> tuple[np.ndarray, np.ndarray]:
        """COLMAP's pose (x right, y down, z forward): rotation and translation."""
        f = self.forward()
        x = np.cross([0.0, 1.0, 0.0], f)
        x /= np.linalg.norm(x)
        rot = np.stack([x, np.cross(f, x), f])
        return rot, -rot @ self.position()

    def focal(self, width: int, height: int) -> float:
        return (min(width, height) / 2) / math.tan(math.radians(self.fov) / 2)

    def render_json(self) -> dict[str, Any]:
        """What GiroRenderSplatCameras takes."""
        return {"position": [round(float(v), 6) for v in self.position()],
                "target": [round(float(v), 6) for v in self.target], "fov": self.fov}

    def to_json(self) -> dict[str, Any]:
        return asdict(self) | {"target": list(self.target)}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> PathCamera:
        return cls(data["yaw"], data["pitch"], data["distance"], tuple(data["target"]), data.get("fov", 35.0))


def plan(start: PathCamera, preset: str, frames: int, turns: float = 1.0, pitch_end: float | None = None
         ) -> list[PathCamera]:
    """`frames` cameras from `start`: `turns` circles around the target at start's distance.

    ring: at start's pitch. spiral: the pitch goes from start's to `pitch_end` over the clip.
    wave: up to `pitch_end` and back down once. The last camera stops one step short of where a
    whole number of turns would close the circle (the next frame would repeat the first).
    loop: a wave whose last camera is the first one again (whole turns), so the clip can end on
    the hero as well as start on it.
    """
    if preset not in PRESETS:
        raise ValueError(f"unknown path preset {preset!r}; one of {', '.join(PRESETS)}")
    end = start.pitch if pitch_end is None or preset == "ring" else pitch_end
    cams = []
    for i in range(frames):
        t = i / (frames - 1) if preset == "loop" else i / frames
        if preset in ("wave", "loop"):
            pitch = start.pitch + (end - start.pitch) * 0.5 * (1 - math.cos(2 * math.pi * t))
        else:
            pitch = start.pitch + (end - start.pitch) * (i / max(1, frames - 1))
        cams.append(replace(start, yaw=start.yaw + 360.0 * turns * t, pitch=pitch))
    return cams


def segment(start: PathCamera, frames: int, turns: float = 1.0, pitch_end: float | None = None,
            distance_end: float | None = None, target_end: tuple[float, float, float] | None = None) -> list[PathCamera]:
    """A clip that continues a path from `start` (its first camera is `start` itself): `turns`
    circles while pitch, distance and target ease to their end values (the last camera)."""
    end_pitch = start.pitch if pitch_end is None else pitch_end
    end_dist = start.distance if distance_end is None else distance_end
    t0 = np.asarray(start.target, dtype=float)
    t1 = t0 if target_end is None else np.asarray(target_end, dtype=float)
    cams = []
    for i in range(frames):
        u = i / max(1, frames - 1)
        e = 0.5 * (1 - math.cos(math.pi * u))  # ease in and out: no jolt where clips meet
        cams.append(replace(start, yaw=start.yaw + 360.0 * turns * u, pitch=start.pitch + (end_pitch - start.pitch) * e,
                            distance=start.distance + (end_dist - start.distance) * e,
                            target=tuple(float(v) for v in t0 + (t1 - t0) * e)))
    return cams


def elevation_bands(cams: list[PathCamera], width: float = 20.0) -> dict[int, list[int]]:
    """Frame indices per elevation band (keyed by the band's lower edge in degrees)."""
    bands: dict[int, list[int]] = {}
    for i, c in enumerate(cams):
        bands.setdefault(int(math.floor(c.pitch / width) * width), []).append(i)
    return bands
