"""Prepare the hero image so it shares the video frames' aspect ratio."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


def fit_to_aspect(src: Path, width: int, height: int) -> tuple[Image.Image, float]:
    """Center-crop `src` to width:height at its native resolution.

    Returns the cropped RGB image and the fraction of pixels removed, so the
    caller can warn when a crop discards much of the subject.
    """
    image = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    w, h = image.size
    target = width / height
    if w / h > target:
        new_w, new_h = round(h * target), h
    else:
        new_w, new_h = w, round(w / target)
    left, top = (w - new_w) // 2, (h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped, 1 - (new_w * new_h) / (w * h)
