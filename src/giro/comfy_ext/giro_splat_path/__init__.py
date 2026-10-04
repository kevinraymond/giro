"""giro's splat camera-path renderer: a Gaussian splat rendered along an orbit whose
elevation changes, as color or depth frames.

ComfyUI's own Render Splat turns the camera a full circle at one elevation. A proxy-guided
orbit also wants the views from above (docs/FINDINGS.md, "Orbit videos"), so this node takes
a yaw and a pitch per frame: `turns` circles, the pitch going from `pitch_start` to
`pitch_end` ("ramp") or there and back ("wave"). It reuses ComfyUI's rasterizer, and returns
the cameras as JSON so the poses of the generated video are known.

Loaded through the `custom_nodes` entry in scripts/extra_model_paths.yaml.
"""

import json
import math

import torch

import comfy.model_management
from comfy_extras import nodes_gaussian_splat as gs


def path_angles(frames: int, yaw_start: float, turns: float, pitch_start: float, pitch_end: float,
                pitch_mode: str) -> list[tuple[float, float]]:
    """(yaw, pitch) in degrees per frame. The last frame stops one step short of closing the circle."""
    out = []
    for i in range(frames):
        t = i / frames
        if pitch_mode == "wave":  # up and back down once over the clip
            pitch = pitch_start + (pitch_end - pitch_start) * 0.5 * (1 - math.cos(2 * math.pi * t))
        else:
            pitch = pitch_start + (pitch_end - pitch_start) * (i / max(1, frames - 1))
        out.append((yaw_start + 360.0 * turns * t, pitch))
    return out


class GiroRenderSplatPath:
    CATEGORY = "giro"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "cameras")
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "splat": ("SPLAT",),
            "width": ("INT", {"default": 576, "min": 64, "max": 2048, "step": 8}),
            "height": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 8}),
            "frames": ("INT", {"default": 81, "min": 1, "max": 480}),
            "yaw_start": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0}),
            "turns": ("FLOAT", {"default": 1.0, "min": -8.0, "max": 8.0, "step": 0.25}),
            "pitch_start": ("FLOAT", {"default": 5.0, "min": -89.0, "max": 89.0}),
            "pitch_end": ("FLOAT", {"default": 5.0, "min": -89.0, "max": 89.0}),
            "pitch_mode": (["ramp", "wave"],),
            "distance": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 1000.0, "step": 0.01}),
            "fov": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 120.0}),
            "render_style": (["color", "clay", "depth", "normal"],),
            "background": ("STRING", {"default": "#000000"}),
        }}

    def render(self, splat, width, height, frames, yaw_start, turns, pitch_start, pitch_end, pitch_mode,
               distance, fov, render_style, background):
        device = comfy.model_management.get_torch_device()
        xyz, rgb, opacity, scale, rot = gs._gaussian_item(splat, 0, device)
        pivot = torch.zeros(3, device=device)
        bg = gs._hex_to_rgb(background)
        imgs, masks, cams = [], [], []
        for yaw, pitch in path_angles(frames, yaw_start, turns, pitch_start, pitch_end, pitch_mode):
            cam = gs._orbit_camera_info(yaw, pitch, distance, fov, pivot, device)
            img, mask = gs._render_gaussian(xyz, rgb, opacity, scale, rot, width, height, 1.0, bg, cam,
                                            sharpen=2.0, render_style=render_style)
            imgs.append(img)
            masks.append(mask)
            cams.append({"yaw": round(yaw, 4), "pitch": round(pitch, 4), "distance": distance, "fov": fov})
        return (torch.stack(imgs), torch.stack(masks), json.dumps(cams))


NODE_CLASS_MAPPINGS = {"GiroRenderSplatPath": GiroRenderSplatPath}
NODE_DISPLAY_NAME_MAPPINGS = {"GiroRenderSplatPath": "giro: render splat along a path"}
