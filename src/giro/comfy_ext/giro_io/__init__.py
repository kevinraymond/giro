# SPDX-License-Identifier: GPL-3.0-or-later
"""giro's file I/O nodes: load images and save masks by absolute path.

ComfyUI is giro's private, localhost-only backend, so stages hand it paths
inside giro's data directory instead of uploading ~120 frames through
/upload/image and downloading the results. Paths outside the data roots
are refused.

Loaded through the `custom_nodes` entry in scripts/extra_model_paths.yaml.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# src/giro/comfy_ext/giro_io/__init__.py -> the repo's data/
_ROOTS = [Path(__file__).resolve().parents[4] / "data"]
_ROOTS += [Path(p) for p in os.environ.get("GIRO_DATA_ROOTS", "").split(os.pathsep) if p]


def _checked(path: str) -> Path:
    p = Path(path).resolve()
    if not any(p.is_relative_to(root.resolve()) for root in _ROOTS):
        raise ValueError(f"giro_io: {path} is outside giro's data roots")
    return p


def _paths(text: str) -> list[str]:
    paths = json.loads(text)
    if not isinstance(paths, list):
        raise ValueError("giro_io: paths must be a JSON list")
    return paths


class GiroLoadImages:
    """A batch of RGB images, all resized to width x height (0: the first image's size)."""

    CATEGORY = "giro"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "paths": ("STRING", {"default": "[]", "multiline": True}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
        }}

    @classmethod
    def IS_CHANGED(cls, paths, width, height):
        return [os.stat(_checked(p)).st_mtime_ns for p in _paths(paths)]

    def load(self, paths, width, height):
        images = []
        for p in _paths(paths):
            im = Image.open(_checked(p)).convert("RGB")
            if not width or not height:
                width, height = im.size
            if im.size != (width, height):
                im = im.resize((width, height), Image.LANCZOS)
            images.append(torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0))
        return (torch.stack(images),)


class GiroLoadMask:
    """One grayscale PNG as a MASK [1, H, W], resized to width x height (0: native)."""

    CATEGORY = "giro"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "path": ("STRING", {"default": ""}),
            "width": ("INT", {"default": 0, "min": 0, "max": 16384}),
            "height": ("INT", {"default": 0, "min": 0, "max": 16384}),
        }}

    @classmethod
    def IS_CHANGED(cls, path, width, height):
        return os.stat(_checked(path)).st_mtime_ns

    def load(self, path, width, height):
        im = Image.open(_checked(path)).convert("L")
        if width and height and im.size != (width, height):
            im = im.resize((width, height), Image.BILINEAR)
        return (torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)[None],)


class GiroSaveMasks:
    """Write mask i to paths[i] as 8-bit grayscale PNG; an empty path skips that mask.
    Masks are resized to each path's (width, height) when `sizes` gives one. Reports
    each mask's foreground fraction so the caller can check them without reading files."""

    CATEGORY = "giro"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "masks": ("MASK",),
            "paths": ("STRING", {"default": "[]", "multiline": True}),
        }, "optional": {
            "sizes": ("STRING", {"default": "[]", "multiline": True}),
        }}

    def save(self, masks, paths, sizes="[]"):
        paths, sizes = _paths(paths), _paths(sizes)
        if masks.ndim == 2:
            masks = masks[None]
        if len(paths) != masks.shape[0]:
            raise ValueError(f"giro_io: {masks.shape[0]} masks but {len(paths)} paths")
        areas = []
        for i, p in enumerate(paths):
            m = masks[i].float().clamp(0, 1)
            if i < len(sizes) and sizes[i]:
                w, h = sizes[i]
                if tuple(m.shape) != (h, w):
                    m = F.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            areas.append(round(float((m > 0.5).float().mean()), 5))
            if not p:
                continue
            dest = _checked(p)
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((m.cpu().numpy() * 255).round().astype(np.uint8), "L").save(dest)
        return {"ui": {"giro_areas": areas}}


NODE_CLASS_MAPPINGS = {
    "GiroLoadImages": GiroLoadImages,
    "GiroLoadMask": GiroLoadMask,
    "GiroSaveMasks": GiroSaveMasks,
}
