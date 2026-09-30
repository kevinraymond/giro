"""giro ComfyUI extension: video-aware latent previews.

Core Latent2RGBPreviewer shows frame 0 of a video latent. For first-and-last
frame orbits, frame 0 is pinned to the hero image, so the preview never
changes. This shows a strip of interior time slices instead, which lets the
UI watch the orbit form. It adds no nodes.

Loaded through the `custom_nodes` entry in scripts/extra_model_paths.yaml.
"""

import latent_preview
from PIL import Image

TILES = 4

_decode_frame = latent_preview.Latent2RGBPreviewer.decode_latent_to_preview


def _decode_strip(self, x0):
    if x0.ndim != 5 or x0.shape[2] < TILES + 2:
        return _decode_frame(self, x0)
    t = x0.shape[2]
    # Evenly spaced interior slices; the first and last are the keyframes.
    slices = sorted({round(t * (i + 1) / (TILES + 1)) for i in range(TILES)})
    frames = [_decode_frame(self, x0[:, :, i : i + 1]) for i in slices]
    strip = Image.new("RGB", (sum(f.width for f in frames), max(f.height for f in frames)))
    x = 0
    for frame in frames:
        strip.paste(frame, (x, 0))
        x += frame.width
    return strip


latent_preview.Latent2RGBPreviewer.decode_latent_to_preview = _decode_strip

NODE_CLASS_MAPPINGS = {}
