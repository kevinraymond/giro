# SPDX-License-Identifier: GPL-3.0-or-later
"""giro's nodes for the proxy orbit: render a Gaussian splat along a camera path, save and
load splats by path, and Wan 2.2 Fun Control with a first frame.

ComfyUI's own Render Splat turns the camera a full circle at one elevation. A proxy-guided
orbit also wants the views from above (docs/FINDINGS.md, "Orbit videos"):

- GiroRenderSplatPath takes a yaw and a pitch per frame: `turns` circles, the pitch going from
  `pitch_start` to `pitch_end` ("ramp") or there and back ("wave") (the Oct 3 spike).
- GiroRenderSplatCameras renders an explicit camera list that giro computes (giro/path.py),
  so the cameras giro poses the frames with are exactly the ones rendered.

Both reuse ComfyUI's rasterizer. GiroWanFunControlToVideo is ComfyUI's Wan22FunControlToVideo
with its `start_image` input declared: the node's code pins the first frames to it, but its
schema leaves the input out.

Loaded through the `custom_nodes` entry in scripts/extra_model_paths.yaml.
"""

import json
import math
import os
from pathlib import Path

import torch

import comfy.model_management
import comfy.utils
import node_helpers
from comfy_extras import nodes_gaussian_splat as gs
from comfy_extras import nodes_wan

# src/giro/comfy_ext/giro_splat_path/__init__.py -> the repo's data/ (as giro_io)
_ROOTS = [Path(__file__).resolve().parents[4] / "data"]
_ROOTS += [Path(p) for p in os.environ.get("GIRO_DATA_ROOTS", "").split(os.pathsep) if p]


def _checked(path: str) -> Path:
    p = Path(path).resolve()
    if not any(p.is_relative_to(root.resolve()) for root in _ROOTS):
        raise ValueError(f"giro_splat_path: {path} is outside giro's data roots")
    return p


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


class GiroRenderSplatCameras:
    """Render a splat from each camera of a JSON list of {"position", "target", "fov"}, in the
    splat's own frame (y down; fov over the image's smaller side, as ComfyUI's renderer)."""

    CATEGORY = "giro"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "splat": ("SPLAT",),
            "width": ("INT", {"default": 576, "min": 64, "max": 4096, "step": 8}),
            "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
            "cameras": ("STRING", {"multiline": True, "default": "[]"}),
            "render_style": (["color", "clay", "depth", "normal"],),
            "background": ("STRING", {"default": "#000000"}),
        }}

    def render(self, splat, width, height, cameras, render_style, background):
        device = comfy.model_management.get_torch_device()
        xyz, rgb, opacity, scale, rot = gs._gaussian_item(splat, 0, device)
        bg = gs._hex_to_rgb(background)
        to_world = lambda v: [v[0], -v[1], -v[2]]  # noqa: E731 - splat <-> world, as _orbit_camera_info
        imgs, masks = [], []
        for cam in json.loads(cameras):
            info = gs._lookat_camera_info(to_world(cam["position"]), to_world(cam["target"]), cam["fov"], device)
            img, mask = gs._render_gaussian(xyz, rgb, opacity, scale, rot, width, height, 1.0, bg, info,
                                            sharpen=2.0, render_style=render_style)
            imgs.append(img)
            masks.append(mask)
        if not imgs:
            raise ValueError("GiroRenderSplatCameras: no cameras")
        return (torch.stack(imgs), torch.stack(masks))


class GiroMeshToSplat:
    """A mesh's surface as small Gaussians, so a mesh proxy (TRELLIS.2, Pixal3D) renders like
    TripoSplat's: `count` points sampled by area, colored from the vertex colors (gray without),
    moved into the splat frame (the mesh's y-up world turned to the splat's y-down one), centered
    and scaled to `height` tall."""

    CATEGORY = "giro"
    RETURN_TYPES = ("SPLAT",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mesh": ("MESH",),
            "count": ("INT", {"default": 262144, "min": 1024, "max": 4_000_000}),
            "height": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
        }}

    def convert(self, mesh, count, height, seed):
        v = mesh.vertices[0].float().cpu()
        f = mesh.faces[0].long().cpu()
        if mesh.vertex_counts is not None:
            v, f = v[:int(mesh.vertex_counts[0])], f[:int(mesh.face_counts[0])]
        tri = v[f]  # (F, 3, 3)
        area = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).norm(dim=1) / 2
        g = torch.Generator().manual_seed(seed)
        pick = torch.multinomial(area / area.sum(), count, replacement=True, generator=g)
        r1, r2 = torch.rand(count, generator=g), torch.rand(count, generator=g)
        s1 = r1.sqrt()
        bary = torch.stack([1 - s1, s1 * (1 - r2), s1 * r2], 1)  # uniform on each triangle
        pts = (tri[pick] * bary[:, :, None]).sum(1)
        colors = getattr(mesh, "vertex_colors", None)
        if colors is not None:
            c = colors[0].float().cpu()[:, :3]
            if c.max() > 1.5:
                c = c / 255.0
            rgb = (c[f[pick]] * bary[:, :, None]).sum(1).clamp(0, 1)
        else:
            rgb = torch.full((count, 3), 0.6)
        pts = pts * torch.tensor([1.0, -1.0, -1.0])  # y-up world -> splat frame (y down)
        lo, hi = pts.min(0).values, pts.max(0).values
        scale = height / float(hi[1] - lo[1])
        pts = (pts - (lo + hi) / 2) * scale
        # isotropic, about the spacing between samples, so the surface renders closed
        spacing = float((area.sum() * scale**2 / count).sqrt())
        C0 = 0.28209479177387814
        splat = gs.Types.SPLAT(
            pts[None], torch.full((1, count, 3), 0.7 * spacing),
            torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, count, 4).clone(),
            torch.ones((1, count, 1)), ((rgb - 0.5) / C0)[None, :, None, :])
        return (splat,)


class GiroSaveSplat:
    """Write a splat to a PLY file at an absolute path inside giro's data roots."""

    CATEGORY = "giro"
    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"splat": ("SPLAT",), "path": ("STRING", {"default": ""})}}

    def save(self, splat, path):
        p = _checked(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        end = gs._real_len(splat, 0)
        p.write_bytes(gs._gaussian_ply_bytes(splat.positions[0, :end], splat.scales[0, :end],
                                             splat.rotations[0, :end], splat.opacities[0, :end], splat.sh[0, :end]))
        return {}


class GiroLoadSplat:
    """A splat from a PLY file at an absolute path inside giro's data roots."""

    CATEGORY = "giro"
    RETURN_TYPES = ("SPLAT",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"default": ""})}}

    @classmethod
    def IS_CHANGED(cls, path):
        p = _checked(path)
        return f"{p.stat().st_size}:{p.stat().st_mtime_ns}"

    def load(self, path):
        model = gs.Types.File3D(str(_checked(path)), file_format="ply")
        return gs.File3DToSplat.execute(model).args


class GiroWanFunControlToVideo:
    """Wan22FunControlToVideo with `start_image`: its first frames are kept (inpainted around),
    the reference image only steers appearance. With `end_image` too, the last frames are kept
    as well; the image half of the conditioning is then the whole clip encoded with gray between
    the two (as ComfyUI's first/last frame node does), which Fun Control was not trained on."""

    CATEGORY = "giro"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 576, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 768, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 81, "min": 1, "max": 4096, "step": 4}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
                "start_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "control_video": ("IMAGE",),
            },
        }

    def encode(self, positive, negative, vae, width, height, length, ref_image=None, start_image=None,
               end_image=None, control_video=None):
        if end_image is None:
            return nodes_wan.Wan22FunControlToVideo.execute(
                positive, negative, vae, width, height, length, 1,
                ref_image=ref_image, start_image=start_image, control_video=control_video).args
        positive, negative, latent = nodes_wan.Wan22FunControlToVideo.execute(
            positive, negative, vae, width, height, length, 1, ref_image=ref_image, control_video=control_video).args
        known = _known_frames(length, width, height, start_image, end_image)
        n_latent = latent["samples"].shape[2]
        # The mask has four slots per latent frame (so 4 x n_latent slots, a few more than the
        # clip's frames). Given: the start image's slots plus the three the VAE folds into the
        # first latent frame with it, and the end image's slots at the very end.
        slots = 4 * n_latent
        given = set()
        if start_image is not None:
            given |= set(range(min(slots, start_image.shape[0] + 3)))
        given |= set(range(slots - min(length, end_image.shape[0]), slots))
        mask = torch.ones((1, 4, n_latent, height // 8, width // 8))
        for slot in given:
            mask[:, slot % 4, slot // 4] = 0.0
        encoded = vae.encode(known)
        channels = vae.latent_channels
        conditioned = []
        for cond in (positive, negative):
            concat = cond[0][1]["concat_latent_image"].clone()
            n = min(encoded.shape[2], concat.shape[2])
            concat[:, channels:, :n] = encoded[:, :, :n]
            conditioned.append(node_helpers.conditioning_set_values(cond, {"concat_latent_image": concat, "concat_mask": mask}))
        return (conditioned[0], conditioned[1], latent)


def _known_frames(length, width, height, start_image, end_image):
    """A clip of mid-gray frames with the start image's frames at its beginning and the end
    image's at its end, all at width x height."""
    def fitted(images):
        return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)[..., :3]

    clip = torch.full((length, height, width, 3), 0.5)
    if start_image is not None:
        head = fitted(start_image[:length])
        clip[:head.shape[0]] = head
    tail = fitted(end_image[-length:])
    clip[length - tail.shape[0]:] = tail
    return clip


NODE_CLASS_MAPPINGS = {
    "GiroRenderSplatPath": GiroRenderSplatPath,
    "GiroRenderSplatCameras": GiroRenderSplatCameras,
    "GiroSaveSplat": GiroSaveSplat,
    "GiroLoadSplat": GiroLoadSplat,
    "GiroMeshToSplat": GiroMeshToSplat,
    "GiroWanFunControlToVideo": GiroWanFunControlToVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GiroRenderSplatPath": "giro: render splat along a path",
    "GiroRenderSplatCameras": "giro: render splat from cameras",
    "GiroSaveSplat": "giro: save splat (path)",
    "GiroMeshToSplat": "giro: mesh surface as a splat",
    "GiroLoadSplat": "giro: load splat (path)",
    "GiroWanFunControlToVideo": "giro: Wan 2.2 Fun Control with a first frame",
}
