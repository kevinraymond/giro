"""Part-aware painting (board #3720): a source pixel paints a surface sample only when both are the same
part. The source-view map of the scooter showed each wheel painted from 5-8 views whose rim edges land
a little differently on Pixal3D's wheel, so the rim's silver spilled onto the tire wherever the next
view took over (and the progressive edits did it again).

- Part labels per image: SAM 3.1 with a text prompt per part (giro's sam_workflow), as a label image:
  0 = none of the parts, k = the k-th part; a later part overrides an earlier one where both are found
  ("tire,wheel rim": the rim inside the tire is the rim).
- Part labels per sample: every labeled source votes with its painting weight (|cos|^4 times the mask
  feather) where it sees the sample; votes are smoothed over 1 cm voxels and a sample is labeled only
  where one part has a clear majority (else -1, unlabeled).
- The gate: a labeled sample takes paint only from pixels of its own label (0 included: a body panel
  takes no tire pixels); an unlabeled sample takes any pixel (the views disagree there, so no label is
  trustworthy, and refusing all of them would leave it unpainted). Every pixel's weight also fades to
  near 0 within --part-edge-px of a part boundary in its own image, where rim and tire mix.

part_masks.py makes the label images; project_texture.py --parts and progressive_paint.py --parts use them.
"""
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from giro.stages.masks import Masks, sam_workflow
from texture_common import smooth_field

EDGE_FLOOR = 0.02  # a boundary pixel still paints a sample nothing else sees (else it keeps the mesh's color)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def label_file(parts_dir: Path, name: str) -> Path:
    return parts_dir / f"{name.replace('/', '_')}.png"


def label_image(masks: list[np.ndarray]) -> np.ndarray:
    """Masks in part order -> a uint8 label image (later parts win)."""
    lab = np.zeros(masks[0].shape, np.uint8)
    for k, m in enumerate(masks, 1):
        lab[m] = k
    return lab


def edge_weight(lab: np.ndarray, px: float) -> np.ndarray:
    """1 away from part boundaries, falling to EDGE_FLOOR on them: rim and tire pixels mix at their border."""
    boundary = np.zeros(lab.shape, bool)
    boundary[1:] |= lab[1:] != lab[:-1]
    boundary[:-1] |= lab[1:] != lab[:-1]
    boundary[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    boundary[:, :-1] |= lab[:, 1:] != lab[:, :-1]
    if not boundary.any():
        return np.ones(lab.shape, np.float32)
    d = ndimage.distance_transform_edt(~boundary)
    return np.clip(d / max(px, 1e-6), EDGE_FLOOR, 1).astype(np.float32)


async def sam_labels(comfy, paths: dict[str, Path], parts: list[str], raw: Path) -> dict[str, np.ndarray]:
    """SAM 3.1 per part prompt over the images (grouped by size, as sam_workflow wants); returns the
    label image per name, at each image's own size."""
    groups: dict[tuple[int, int], list[tuple[str, Path]]] = {}
    for name, p in paths.items():
        groups.setdefault(Image.open(p).size, []).append((name, p))
    found: dict[str, list[np.ndarray]] = {n: [] for n in paths}
    for part in parts:
        params = Masks.defaults | {"subject_prompt": part, "background_prompt": ""}
        pdir = raw / slug(part)
        wf_groups = {f"g{i}": [p for _, p in items] for i, items in enumerate(groups.values())}
        async for _ in comfy.run(sam_workflow(wf_groups, pdir, params)):
            pass
        for i, (size, items) in enumerate(groups.items()):
            for name, p in items:
                m = pdir / "subject" / f"g{i}" / p.name
                found[name].append(np.asarray(Image.open(m)) > 127 if m.exists() else np.zeros((size[1], size[0]), bool))
    return {n: label_image(ms) for n, ms in found.items()}


def save_labels(parts_dir: Path, parts: list[str], labels: dict[str, np.ndarray]) -> None:
    parts_dir.mkdir(parents=True, exist_ok=True)
    for name, lab in labels.items():
        Image.fromarray(lab).save(label_file(parts_dir, name))
    (parts_dir / "parts.json").write_text(json.dumps({"parts": parts, "names": sorted(labels)}, indent=1))


class PartView:
    """One image's part labels and boundary fade on the GPU, read where samples land."""

    def __init__(self, lab: np.ndarray, edge_px: float, dev: torch.device, size: tuple[int, int] | None = None):
        if size is not None and (lab.shape[1], lab.shape[0]) != size:  # to the image it labels (nearest: labels)
            lab = np.asarray(Image.fromarray(lab).resize(size, Image.NEAREST))
        self.h, self.w = lab.shape
        self.lab = torch.from_numpy(lab.astype(np.float32)).to(dev)[None]
        self.edge = torch.from_numpy(edge_weight(lab, edge_px)).to(dev)[None]

    @classmethod
    def load(cls, parts_dir: Path, name: str, edge_px: float, dev: torch.device, size: tuple[int, int] | None = None):
        f = label_file(parts_dir, name)
        return cls(np.asarray(Image.open(f)), edge_px, dev, size) if f.exists() else None

    def at(self, u: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The label (nearest pixel) and boundary fade (bilinear) where samples land (pixel coords)."""
        grid = torch.stack([2 * u / self.w - 1, 2 * v / self.h - 1], 1)[None, None]
        lab = F.grid_sample(self.lab[None], grid, mode="nearest", align_corners=False)[0, 0, 0].round().long()
        edge = F.grid_sample(self.edge[None], grid, mode="bilinear", align_corners=False)[0, 0, 0]
        return lab, edge


def vote(xyz: torch.Tensor, entries: list[tuple[torch.Tensor, torch.Tensor]], n_parts: int,
         voxel: float = 0.01, agree: float = 0.6) -> torch.Tensor:
    """Sample labels from (pixel label, weight) per labeled source: weighted votes smoothed over `voxel`
    (proxy units, ~1 tall: ~1 cm), a label where one part has `agree` of the vote, else -1."""
    votes = torch.zeros(len(xyz), n_parts + 1, device=xyz.device)
    for lab, wt in entries:
        votes.index_put_((torch.arange(len(xyz), device=xyz.device), lab.clamp(0, n_parts)), wt, accumulate=True)
    total = votes.sum(1)
    share, conf = smooth_field(xyz, votes / total.clamp_min(1e-9)[:, None], total, voxel, 1.0)
    best = share.max(1)
    return torch.where((best.values >= agree) & (conf > 1e-6), best.indices, torch.full_like(best.indices, -1))


def gate(sample_lab: torch.Tensor, pixel_lab: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
    """The weight factor: 0 where a labeled sample would take another part's pixel, else the pixel's
    boundary fade."""
    return ((sample_lab < 0) | (pixel_lab == sample_lab)).float() * edge
