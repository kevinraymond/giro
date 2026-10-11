"""The subject's mirror symmetry, from its proxy: most people and man-made objects are close to
mirror-symmetric, so the hero mirrored across the symmetry plane is a real-looking view from the
mirrored camera, made of nothing invented (board #3822).

plane() searches the proxy's points for the plane n.x = d that maps them onto themselves best: a
distance field of the points, scored by how far the mirrored points land from the surface (trimmed
mean over the closest 90%, so one-sided parts do not decide it), the best few candidates refined
by Nelder-Mead. vertical=True keeps the plane upright (n.y = 0), the plane of an upright subject
in the proxy's frame; the mirrored camera of a path camera is then a path camera too (mirror()).

On the v2 held-out objects (scripts/proxy_spike/symmetry_mirror.py, Oct 9) the mirrored hero beat
the hero-painted proxy at the mirrored camera on 26 of 29 objects whose mirrored camera is 60 deg
or more from the hero; heroes that face straight along the plane gain nothing (the mirror shows the
same side).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage, optimize

from giro import path as campath


@dataclass(frozen=True)
class Plane:
    normal: np.ndarray  # unit
    d: float            # n.x = d
    score: float        # trimmed mean distance of the mirrored points from the surface (proxy units)
    floor: float        # the points' own spacing: a perfectly symmetric proxy scores about this

    @property
    def tilt(self) -> float:
        """Degrees the plane leans from upright."""
        return math.degrees(math.asin(min(1.0, abs(float(self.normal[1])))))

    def reflect(self, x: np.ndarray) -> np.ndarray:
        return x - 2 * (x @ self.normal - self.d)[..., None] * self.normal

    def to_json(self) -> dict:
        return {"normal": [round(float(v), 6) for v in self.normal], "d": round(self.d, 6), "score": round(self.score, 6),
                "floor": round(self.floor, 6), "tilt": round(self.tilt, 2)}


def _distance_field(x: torch.Tensor, res: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Distance to the nearest occupied voxel over the points' padded box, as grid_sample's volume."""
    lo, hi = x.min(0).values, x.max(0).values
    pad = 0.15 * float((hi - lo).max())
    lo, size = lo - pad, float((hi - lo).max()) + 2 * pad
    vox = size / res
    idx = ((x - lo) / vox).long().clamp(0, res - 1).cpu().numpy()
    occ = np.zeros((res, res, res), bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    df = torch.from_numpy(ndimage.distance_transform_edt(~occ) * vox).float().to(x.device)
    return df.permute(2, 1, 0)[None, None], lo, size  # (1, 1, z, y, x)


def plane(xyz: np.ndarray, vertical: bool = True, device: str | torch.device = "cpu", res: int = 128) -> Plane:
    x = torch.as_tensor(np.asarray(xyz), dtype=torch.float32, device=device)
    vol, lo, size = _distance_field(x, res)
    g = torch.Generator().manual_seed(0)
    pick = lambda n: x[torch.randperm(len(x), generator=g)[:n].to(x.device)]  # noqa: E731
    probe, more = pick(8000), pick(30000)
    c = x.mean(0)

    def score(normals: torch.Tensor, ds: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        dist = pts[None] @ normals[:, :, None]
        refl = pts[None] - 2 * (dist - ds[:, None, None]) * normals[:, None, :]
        grid = ((refl - lo) / size * 2 - 1)[:, :, None, None, :]
        v = F.grid_sample(vol.expand(len(normals), -1, -1, -1, -1), grid, align_corners=False, padding_mode="border")
        v = v.reshape(len(normals), -1)
        return v.sort(1).values[:, : int(0.9 * v.shape[1])].mean(1)

    def unit(p: np.ndarray) -> np.ndarray:
        if vertical:
            return np.array([math.cos(p[0]), 0.0, math.sin(p[0])])
        th, ph = p[0], p[1]
        return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])

    if vertical:
        a = torch.arange(360, device=x.device).float() * math.pi / 360  # 0.5 deg steps over 180
        normals = torch.stack([torch.cos(a), torch.zeros_like(a), torch.sin(a)], 1)
    else:  # a Fibonacci hemisphere (a plane and its flipped normal are the same plane)
        i = torch.arange(3000, device=x.device).float() + 0.5
        zc = i / 3000
        phi = math.pi * (1 + 5**0.5) * i
        r = (1 - zc**2).sqrt()
        normals = torch.stack([r * torch.cos(phi), r * torch.sin(phi), zc], 1)
    with torch.no_grad():
        s = torch.cat([score(n_, n_ @ c, probe) for n_ in normals.split(250)])
    seeds: list[int] = []
    for j in s.argsort().tolist():  # the best few distinct planes
        if all(abs(float(normals[j] @ normals[k])) < 0.97 for k in seeds):
            seeds.append(j)
        if len(seeds) == 4:
            break
    best: tuple[float, np.ndarray, float] | None = None
    for j in seeds:
        n0 = normals[j].double().cpu().numpy()
        p0 = [math.atan2(n0[2], n0[0])] if vertical else [math.acos(np.clip(n0[2], -1, 1)), math.atan2(n0[1], n0[0])]

        def f(p: np.ndarray) -> float:
            n_ = torch.tensor(unit(p[:-1]), device=x.device).float()[None]
            with torch.no_grad():
                return float(score(n_, torch.tensor([p[-1]], device=x.device).float(), more)[0])
        r = optimize.minimize(f, [*p0, float(normals[j] @ c)], method="Nelder-Mead",
                              options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        if best is None or r.fun < best[0]:
            best = (float(r.fun), unit(r.x[:-1]), float(r.x[-1]))
    assert best is not None
    spacing = float(torch.cdist(probe[:2000], pick(200000)).topk(2, largest=False).values[:, 1].mean())
    return Plane(best[1], best[2], best[0], spacing)


def mirror(cam: campath.PathCamera, pl: Plane) -> campath.PathCamera:
    """The camera that sees the mirrored subject as `cam` sees the subject, flipped left to right
    (its pose is F R H, H the reflection, F = diag(-1, 1, 1)). Exact for an upright plane; a leaning
    plane's mirrored camera would roll, which a path camera cannot."""
    f = pl.reflect(cam.position() + cam.forward()) - pl.reflect(cam.position())
    target = pl.reflect(np.asarray(cam.target, dtype=float))
    return campath.PathCamera(yaw=math.degrees(math.atan2(-f[0], f[2])), pitch=math.degrees(math.asin(np.clip(f[1], -1, 1))),
                              distance=cam.distance, target=tuple(float(v) for v in target), fov=cam.fov)


def separation(a: campath.PathCamera, b: campath.PathCamera) -> float:
    """Degrees between two cameras' directions from the first one's target."""
    u, v = a.position() - np.asarray(a.target), b.position() - np.asarray(a.target)
    return math.degrees(math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))
