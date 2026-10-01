"""Prepare the hero image so it shares the video frames' aspect ratio."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageOps


def fit_to_aspect(src: Path, width: int, height: int,
                  crop: Sequence[float] | None = None) -> tuple[Image.Image, float]:
    """Crop `src` to width:height at its native resolution.

    `crop` is the region to keep as fractions of the image, (left, top, right, bottom), as the
    new-job form's pan and zoom chose it; the largest width:height box centered in it is kept.
    Without it the crop is centered on the whole image.

    Returns the cropped RGB image and the fraction of pixels removed, so the
    caller can warn when a crop discards much of the subject.
    """
    image = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = image.size
    x0, y0, x1, y1 = (round(crop[0] * w), round(crop[1] * h), round(crop[2] * w), round(crop[3] * h)) if crop else (0, 0, w, h)
    rw, rh = max(1, x1 - x0), max(1, y1 - y0)
    target = width / height
    if rw / rh > target:
        new_w, new_h = round(rh * target), rh
    else:
        new_w, new_h = rw, round(rw / target)
    left, top = x0 + (rw - new_w) // 2, y0 + (rh - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped, 1 - (new_w * new_h) / (w * h)
