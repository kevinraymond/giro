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


def build_arc(first: str, last: str, **params: Any) -> dict[str, Any]:
    """orbit_video from `first` to a different `last` frame (uploaded image names), for a gap fill."""
    prompt = build("orbit_video", image=first, **params)
    prompt["last"] = {"class_type": "LoadImage", "inputs": {"image": last, "upload": "image"}}
    prompt["condition"]["inputs"]["last_frame"] = ["last", 0]
    return prompt


def snap_length(frames: int) -> int:
    """MiniMax H3 frame counts live on a 17k+5 grid; snap up like the node does."""
    frames = max(5, frames)
    return frames + (5 - frames % 17) % 17
