"""What project_texture.py and register_features.py share: the proxy mesh as dense surface
samples in the splat frame, cameras with a free pose, projection and the z-buffer test."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from giro import path as campath
from giro.stages.masks import Masks, combine


@dataclass(frozen=True)
class PoseCamera:
    """A camera with any rotation (COLMAP's world-to-camera R, t), and the fov over the image's
    smaller side like PathCamera, so it stands in for one (Points.silhouette, project)."""

    rot: tuple[tuple[float, ...], ...]
    t: tuple[float, ...]
    fov: float

    def world_to_camera(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.rot), np.asarray(self.t)

    def position(self) -> np.ndarray:
        r, t = self.world_to_camera()
        return -r.T @ t

    def focal(self, width: int, height: int) -> float:
        return (min(width, height) / 2) / math.tan(math.radians(self.fov) / 2)

    def to_json(self) -> dict[str, Any]:
        return {"rot": [list(r) for r in self.rot], "t": list(self.t), "fov": self.fov}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> PoseCamera:
        return cls(tuple(tuple(r) for r in d["rot"]), tuple(d["t"]), d["fov"])

    @classmethod
    def of(cls, cam: campath.PathCamera | PoseCamera) -> PoseCamera:
        r, t = cam.world_to_camera()
        return cls(tuple(tuple(map(float, row)) for row in r), tuple(map(float, t)), cam.fov)


def samples(work: Path, n: int, dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """WORK/mesh.npz in the splat frame (as GiroMeshToSplat moves it), sampled by area: positions,
    face normals, the mesh's colors, and the spacing between samples."""
    m = np.load(work / "mesh.npz")
    v = torch.from_numpy(m["vertices"]).to(dev) * torch.tensor([1.0, -1.0, -1.0], device=dev)
    f = torch.from_numpy(m["faces"]).to(dev)
    col = torch.from_numpy(m["colors"]).to(dev).clamp(0, 1)
    lo, hi = v.min(0).values, v.max(0).values
    v = (v - (lo + hi) / 2) / float(hi[1] - lo[1])
    tri = v[f]
    cross = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = cross.norm(dim=1) / 2
    normals = cross / (2 * area[:, None] + 1e-12)
    g = torch.Generator(device=dev).manual_seed(0)
    pick = torch.multinomial(area / area.sum(), n, replacement=True, generator=g)
    r1, r2 = torch.rand(n, generator=g, device=dev), torch.rand(n, generator=g, device=dev)
    s1 = r1.sqrt()
    bary = torch.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)
    xyz = (tri[pick] * bary[:, :, None]).sum(1)
    rgb = (col[f[pick]] * bary[:, :, None]).sum(1)
    return xyz, normals[pick], rgb, math.sqrt(float(area.sum()) / n)


def project(cam: Any, w: int, h: int, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rot, t = cam.world_to_camera()
    rot_t = torch.tensor(np.asarray(rot), device=pts.device, dtype=torch.float32)
    t_t = torch.tensor(np.asarray(t), device=pts.device, dtype=torch.float32)
    pc = pts @ rot_t.T + t_t
    z = pc[:, 2]
    fl = cam.focal(w, h)
    zs = torch.where(z > 1e-3, z, torch.ones_like(z))
    return fl * pc[:, 0] / zs + w / 2, fl * pc[:, 1] / zs + h / 2, z


def zbuffer(u: torch.Tensor, vv: torch.Tensor, z: torch.Tensor, w: int, h: int
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per sample: in view, its pixel index, and the nearest depth around that pixel; and that
    depth per pixel (inf where nothing). The nearest depth is taken over 3x3 pixels: a pixel the
    front surface's samples happen to miss would otherwise let the surface behind it through
    (speckles of the wrong part's color)."""
    ok = (z > 1e-3) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
    pix = torch.where(ok, vv.long().clamp(0, h - 1) * w + u.long().clamp(0, w - 1), torch.zeros_like(z, dtype=torch.long))
    zmin = torch.full((w * h,), float("inf"), device=z.device)
    zmin.scatter_reduce_(0, pix[ok], z[ok], reduce="amin")
    zmin = -F.max_pool2d(-zmin.reshape(1, 1, h, w), 3, stride=1, padding=1).reshape(-1)
    return ok, pix, zmin[pix], zmin.reshape(h, w)


def sample(img: torch.Tensor, u: torch.Tensor, vv: torch.Tensor, w: int, h: int) -> torch.Tensor:
    grid = torch.stack([2 * u / w - 1, 2 * vv / h - 1], 1)[None, None]
    return F.grid_sample(img[None], grid, mode="bilinear", align_corners=False)[0, :, 0].T


def anchor_mask(adir: Path, name: str) -> np.ndarray:
    raw, p = adir / "raw", Masks.defaults
    subject = np.asarray(Image.open(raw / "subject" / "anchors" / f"{name}.png")) > 127
    bg = raw / "background" / "anchors" / f"{name}.png"
    return combine(subject, np.asarray(Image.open(bg)) > 127 if bg.exists() else None, p["touch_px"], p["gap_px"], p["max_add"])


def load_cameras(path: Path) -> dict[str, PoseCamera]:
    return {k: PoseCamera.from_json(v["camera"]) for k, v in json.loads(path.read_text())["cameras"].items()}


class Source:
    """One view that paints the mesh: its image and mask-edge feather on the GPU, and optionally
    a warp: a coarse grid of 2D offsets in pixels, (2, rows, cols) over the image, added to where
    each sample lands before its color (and feather) is read (warp_views.py)."""

    def __init__(self, name: str, img: Image.Image, mask: np.ndarray, cam: Any, boost: float, dev: torch.device,
                 feather_px: float = 8.0, warp: torch.Tensor | None = None):
        self.name, self.img, self.mask, self.cam, self.boost = name, img, mask, cam, boost
        self.w, self.h = img.size
        inner = ndimage.binary_erosion(mask, iterations=2)
        feather = np.clip(ndimage.distance_transform_edt(inner) / feather_px, 0, 1).astype(np.float32)
        self.feather = torch.from_numpy(feather).to(dev)[None]
        self.rgb = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255).permute(2, 0, 1).to(dev)
        self.warp = warp.to(dev) if warp is not None else None

    def warped(self, u: torch.Tensor, vv: torch.Tensor, warp: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        warp = self.warp if warp is None else warp
        if warp is None:
            return u, vv
        d = sample(warp, u, vv, self.w, self.h)  # the grid spans the image, read bilinearly
        return u + d[:, 0], vv + d[:, 1]

    def visible(self, xyz: torch.Tensor, nrm: torch.Tensor, tol: float, cam: Any = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Where every sample lands (u, v) and its weight before the feather: |cos| of the viewing
        angle to the 4th power where it lands in view and no nearer surface covers it, else 0."""
        cam = cam or self.cam
        u, vv, z = project(cam, self.w, self.h, xyz)
        ok, _, zmin, _ = zbuffer(u, vv, z, self.w, self.h)
        to_cam = torch.tensor(np.asarray(cam.position()), device=xyz.device, dtype=torch.float32) - xyz
        cos = (nrm * to_cam).sum(1).abs() / to_cam.norm(dim=1)
        return u, vv, torch.where(ok & (z <= zmin + tol), self.boost * cos**4, torch.zeros_like(cos))

    def paint(self, xyz: torch.Tensor, nrm: torch.Tensor, tol: float, cam: Any = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Colors and weights for every sample from camera `cam` (default: its own). A sample is
        painted where it lands inside the mask and no nearer surface covers it; the weight is
        |cos| of the viewing angle to the 4th power, times the feather from the mask's edge."""
        u, vv, wt = self.visible(xyz, nrm, tol, cam)
        uw, vw = self.warped(u, vv)
        return sample(self.rgb, uw, vw, self.w, self.h), wt * sample(self.feather, uw, vw, self.w, self.h)[:, 0]


def smooth_field(xyz: torch.Tensor, values: torch.Tensor, weights: torch.Tensor, voxel: float, sigma_vox: float
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """The weighted average of `values` (N, C) around each sample: binned into a voxel grid,
    blurred with a Gaussian of sigma_vox voxels (normalized convolution, so empty voxels borrow
    from their neighbors), read back per sample. Also returns the blurred weight (confidence)."""
    lo = xyz.min(0).values - 4 * voxel
    dims = (((xyz.max(0).values + 4 * voxel) - lo) / voxel).ceil().long() + 1
    ijk = ((xyz - lo) / voxel).round().long()
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    n_cells, c = int(dims.prod()), values.shape[1]
    num = torch.zeros(n_cells, c, device=xyz.device).index_add_(0, flat, values * weights[:, None])
    den = torch.zeros(n_cells, device=xyz.device).index_add_(0, flat, weights)
    grid = torch.cat([num.T, den[None]]).reshape(1, c + 1, *dims.tolist())
    r = max(1, math.ceil(2.5 * sigma_vox))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=xyz.device, dtype=torch.float32) / sigma_vox) ** 2)
    k = k / k.sum()
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = 2 * r + 1
        pad = [0, 0, 0]
        pad[axis] = r
        grid = F.conv3d(grid.reshape(c + 1, 1, *grid.shape[2:]), k.reshape(shape), padding=tuple(pad)).reshape(1, c + 1, *grid.shape[2:])
    # Read back trilinearly (nearest-voxel reads show the grid as blocks on the surface).
    pos = (xyz - lo) / voxel / (dims - 1).float() * 2 - 1  # voxel centers at the grid's corners
    grid_xyz = pos[:, [2, 1, 0]].reshape(1, -1, 1, 1, 3)  # grid_sample wants (x, y, z) = (dim 4, 3, 2)
    out = F.grid_sample(grid, grid_xyz, mode="bilinear", align_corners=True).reshape(c + 1, -1)
    den_b = out[c]
    return (out[:c].T / den_b.clamp_min(1e-9)[:, None]), den_b


def render_points(xyz: torch.Tensor, values: torch.Tensor, cam: Any, w: int, h: int, tol: float = 0.005
                  ) -> tuple[np.ndarray, np.ndarray]:
    """The samples' `values` seen from `cam`: the nearest layer per pixel (within `tol`) averaged,
    pinholes closed from the neighbors; returns the image (h, w, C) and the subject's mask."""
    dev = xyz.device
    u, vv, z = project(cam, w, h, xyz)
    ok, pix, zmin, _ = zbuffer(u, vv, z, w, h)
    front = ok & (z <= zmin + tol)
    acc = torch.zeros((w * h, values.shape[1]), device=dev).index_add_(0, pix[front], values[front].float())
    n = torch.zeros(w * h, device=dev).index_add_(0, pix[front], torch.ones_like(z[front]))
    img = (acc / n.clamp_min(1)[:, None]).reshape(h, w, -1)
    filled = (n > 0).reshape(h, w)
    for _ in range(4):  # the 3x3 depth test also drops a pixel or two beside nearer edges
        k = torch.ones((1, 1, 3, 3), device=dev)
        num = F.conv2d((img * filled[..., None]).permute(2, 0, 1)[:, None], k, padding=1)[:, 0].permute(1, 2, 0)
        den = F.conv2d(filled[None, None].float(), k, padding=1)[0, 0]
        img = torch.where(filled[..., None], img, num / den.clamp_min(1)[..., None])
        filled = filled | (den > 0)
    mask = ndimage.binary_opening(ndimage.binary_closing(filled.cpu().numpy(), iterations=2), iterations=1)
    return img.cpu().numpy(), mask
