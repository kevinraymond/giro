"""API-format ComfyUI workflows, stored as JSON templates with named nodes.

Each template has a parameter map from giro parameter names to the
(node, input) they set, so callers never touch node ids.
"""

from __future__ import annotations

import copy
import json
from importlib import resources
from typing import Any

ORBIT_PROMPT = (
    "The character remains completely frozen in place, perfectly still like a statue. "
    "The camera smoothly orbits 360 degrees around the character in one continuous shot "
    "at a constant speed. No character movement, no pose change, no cuts, no eye movement."
)

# Gap fill: only an arc of the orbit, between two frames of it (docs/FINDINGS.md, "Gap fill").
GAP_PROMPT = (
    "The character remains completely frozen in place, perfectly still like a statue. "
    "The camera smoothly orbits around the character in one continuous shot at a slow, constant speed. "
    "No character movement, no pose change, no cuts, no eye movement."
)

# The prompt the 360 orbit LoRA was trained on; its card says to use it verbatim
# (https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA).
FROZEN_ORBIT_PROMPT = (
    "One frozen instant. Only the camera moves. In a continuous 360 orbit. "
    "Preserve every person and object in exactly the same world position, orientation, shape and pose "
    "throughout the shot. Airborne objects remain suspended at the captured height and angle: "
    "no wobbling, shaking, spinning, drifting, falling or continued action. "
    "Keep faces, hands, clothing, liquids and the background motionless while retaining their natural appearance. "
    "Camera parallax is the only source of apparent movement. No cuts, zoom, morphing or added objects."
)

# Wan 2.2 with the ostris orbit LoRA, whose trigger is "orbit 360"
# (https://huggingface.co/ostris/wan22_i2v_14b_orbit_shot_lora).
WAN_ORBIT_PROMPT = (
    "orbit 360 around the subject. The subject stays completely still, frozen like a statue; "
    "only the camera moves, circling all the way around the subject at a constant speed. No cuts, no zoom."
)

# The proxy orbit (Wan 2.2 Fun Control): the depth video already sets the camera path, so the
# prompt only has to keep the subject still (docs/FINDINGS.md, "Proxy orbit").
PROXY_ORBIT_PROMPT = (
    "The subject stays completely still, frozen like a statue, in an empty room. "
    "Only the camera moves, circling around the subject at a constant speed. No cuts, no zoom."
)

_PARAMS: dict[str, dict[str, tuple[str, str]]] = {
    "orbit_video": {
        "image": ("hero", "image"),
        "prompt": ("condition", "prompt"),
        "width": ("condition", "width"),
        "height": ("condition", "height"),
        "length": ("condition", "length"),
        "seed": ("noise", "noise_seed"),
        "steps": ("sigmas", "steps"),
        "sampler": ("sampler", "sampler_name"),
        "scheduler": ("sigmas", "scheduler"),
        "output_prefix": ("frames", "filename_prefix"),
    },
    # Wan 2.2 I2V 14B: a high-noise and a low-noise model, each sampling half of the steps.
    "orbit_video_wan22": {
        "image": ("hero", "image"),
        "prompt": ("positive", "text"),
        "width": ("condition", "width"),
        "height": ("condition", "height"),
        "length": ("condition", "length"),
        "seed": ("sample", "noise_seed"),
        "output_prefix": ("frames", "filename_prefix"),
    },
    # Wan 2.2 Fun Control 14B: the proxy splat rendered as depth from giro's cameras is the
    # control video, the hero both the first frame and the appearance reference.
    "orbit_video_wan22_control": {
        "image": ("hero", "image"),
        "prompt": ("positive", "text"),
        "width": ("condition", "width"),
        "height": ("condition", "height"),
        "length": ("condition", "length"),
        "seed": ("sample", "noise_seed"),
        "output_prefix": ("frames", "filename_prefix"),
        "proxy": ("proxy", "path"),
        "cameras": ("depth", "cameras"),
        "silhouette_paths": ("silhouettes", "paths"),
        "depth_prefix": ("depth_frames", "filename_prefix"),
    },
    # A second pass over a proxy orbit clip at a higher resolution: the clip, upscaled, is noised
    # to `sigmas`' first value and denoised by the low-noise model under the same control.
    "orbit_refine_wan22_control": {
        "frames": ("frames_in", "paths"),
        "image": ("hero", "image"),
        "start": ("start", "image"),
        "prompt": ("positive", "text"),
        "seed": ("noise", "noise_seed"),
        "sigmas": ("sigmas", "sigmas"),
        "proxy": ("proxy", "path"),
        "cameras": ("depth", "cameras"),
        "length": ("condition", "length"),
        "output_prefix": ("frames", "filename_prefix"),
    },
    # SeedVR2 (Apache-2.0): one-step video super-resolution of frames, resized first to the target size.
    "upscale_seedvr2": {
        "frames": ("frames", "paths"),
        "width": ("frames", "width"),
        "height": ("frames", "height"),
        "unet": ("unet", "unet_name"),
        "color": ("post", "color_correction_method"),
        "seed": ("sample", "seed"),
        "output_prefix": ("save", "filename_prefix"),
    },
    "edit_image": {
        "image": ("image", "image"),
        "prompt": ("positive", "prompt"),
        "negative": ("negative", "prompt"),
        "seed": ("sampler", "seed"),
        "steps": ("sampler", "steps"),
        "cfg": ("sampler", "cfg"),
        "output_prefix": ("save", "filename_prefix"),
    },
}


def build(name: str, **params: Any) -> dict[str, Any]:
    """Return a ready-to-queue prompt for workflow `name` with `params` applied."""
    template = json.loads(resources.files(__package__).joinpath(f"{name}.json").read_text())
    prompt = copy.deepcopy(template)
    bindings = _PARAMS[name]
    for key, value in params.items():
        if value is None:
            continue
        if key not in bindings:
            raise KeyError(f"workflow {name!r} has no parameter {key!r}")
        node, field = bindings[key]
        prompt[node]["inputs"][field] = value
    return prompt


def with_lora(prompt: dict[str, Any], lora: str, strength: float = 1.0, loader: str = "unet") -> dict[str, Any]:
    """Put a LoRA (a file under ComfyUI's loras folder) between a model loader and all that reads it."""
    name = "lora" + loader.removeprefix("unet")  # "lora", and "lora_low" after "unet_low"
    for node in prompt.values():
        for key, value in node["inputs"].items():
            if value == [loader, 0]:
                node["inputs"][key] = [name, 0]
    prompt[name] = {"class_type": "LoraLoaderModelOnly",
                    "inputs": {"model": [loader, 0], "lora_name": lora, "strength_model": strength}}
    return prompt


ORBIT_MODELS = {"h3": "orbit_video", "wan22": "orbit_video_wan22", "wan22-control": "orbit_video_wan22_control"}


def build_orbit(model: str = "h3", steps: int | None = None, **params: Any) -> dict[str, Any]:
    """The orbit workflow of a video model ("h3", "wan22" or "wan22-control") with `params` applied."""
    if model not in ORBIT_MODELS:
        raise KeyError(f"no orbit workflow for model {model!r} (have {', '.join(ORBIT_MODELS)})")
    if model == "h3":
        return build("orbit_video", steps=steps, **params)
    prompt = build(ORBIT_MODELS[model], **params)
    if model == "wan22-control":  # the control video is rendered at the video's size
        for key in ("width", "height"):
            if params.get(key):
                prompt["depth"]["inputs"][key] = params[key]
    if steps is not None:  # the high-noise model takes the first half, the low-noise one the rest
        for node in ("sample", "sample_low"):
            prompt[node]["inputs"]["steps"] = steps
        prompt["sample"]["inputs"]["end_at_step"] = prompt["sample_low"]["inputs"]["start_at_step"] = steps // 2
    return prompt


def refine_sigmas(sigma: float, steps: int, shift: float = 8.0) -> list[float]:
    """`steps` sigmas from `sigma` down to 0 on the shifted flow schedule (ModelSamplingSD3's
    sigma = shift t / (1 + (shift - 1) t)), evenly spaced in t like the "simple" scheduler."""
    t0 = sigma / (shift - (shift - 1) * sigma)
    ts = [t0 * (1 - i / steps) for i in range(steps + 1)]
    return [round(shift * t / (1 + (shift - 1) * t), 6) for t in ts]


def build_refine(width: int, height: int, sigma: float, steps: int, from_proxy: bool = False,
                 silhouette_paths: str | None = None, depth_prefix: str | None = None, **params: Any) -> dict[str, Any]:
    """The refine pass at width x height, from `sigma` in `steps` steps. from_proxy: start from the
    proxy's color render along the cameras instead of frames (a clip whose layout and colors come
    from the proxy); silhouette_paths and depth_prefix save the proxy's silhouettes and depth."""
    prompt = build("orbit_refine_wan22_control", sigmas=", ".join(map(str, refine_sigmas(sigma, steps))), **params)
    if from_proxy:
        prompt["frames_in"] = {"class_type": "GiroRenderSplatCameras", "inputs": {
            "splat": ["proxy", 0], "cameras": prompt["depth"]["inputs"]["cameras"], "render_style": "color",
            "background": "#808080"}}
    if silhouette_paths:
        prompt["silhouettes"] = {"class_type": "GiroSaveMasks", "inputs": {"masks": ["depth", 1], "paths": silhouette_paths}}
    if depth_prefix:
        prompt["depth_frames"] = {"class_type": "SaveImage", "inputs": {"images": ["depth", 0], "filename_prefix": depth_prefix}}
    for node in ("frames_in", "depth", "condition"):
        prompt[node]["inputs"]["width"] = width
        prompt[node]["inputs"]["height"] = height
    return prompt


def build_arc(first: str, last: str, **params: Any) -> dict[str, Any]:
    """orbit_video from `first` to a different `last` frame (uploaded image names), for a gap fill."""
    prompt = build("orbit_video", image=first, **params)
    prompt["last"] = {"class_type": "LoadImage", "inputs": {"image": last, "upload": "image"}}
    prompt["condition"]["inputs"]["last_frame"] = ["last", 0]
    return prompt


def snap_length(frames: int, model: str = "h3") -> int:
    """MiniMax H3 frame counts live on a 17k+5 grid, Wan's on 4k+1; snap up like the nodes do."""
    if model.startswith("wan22"):
        return max(5, frames) + (1 - max(5, frames) % 4) % 4
    frames = max(5, frames)
    return frames + (5 - frames % 17) % 17
