"""proxy: a rough 3D splat of the hero (TripoSplat) and the hero's camera in its frame.

The proxy orbit renders this splat as a depth video along a known camera path and has Wan 2.2
Fun Control repaint it with the hero's appearance, so the video's cameras are known
(docs/FINDINGS.md, "Proxy orbit"). This stage:

1. masks the hero with SAM 3.1 (the masks stage's prompts and combination);
2. runs TripoSplat on the masked hero with `n_seeds` seeds (~10 s each);
3. fits the hero's camera to each proxy on the CPU: the silhouette of the proxy's Gaussian
   centers against the hero mask, over yaw and pitch, with distance and framing solved from the
   silhouettes' boxes, then refined; front and back share a silhouette, so the color against
   the hero breaks the tie;
4. keeps the proxy whose fitted silhouette matches the hero best.

Outputs: proxy/proxy.ply (the splat's own frame: y down, ~1 tall), proxy/proxy.json (the hero
camera as a giro.path.PathCamera, the fit scores) and proxy/fit.jpg (hero mask against the
fitted silhouette, and a turntable of the proxy).
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

from giro import path as campath
from giro import splat
from giro.comfy import ComfyClient, ComfyError, Done, server
from giro.stages.base import Ctx, Stage, StageFailed
from giro.stages.masks import SAM_VRAM_MB, Masks, combine, sam_workflow
from giro.stages.orbit import events

C0 = 0.28209479177387814  # SH band 0: f_dc -> base color
FIT_HEIGHT = 160  # px: silhouettes are compared at this image height
REFINE_HEIGHT = 256


def tripo_workflow(image: str, seeds: list[int], out_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    """TripoSplat on an uploaded RGBA image, one splat per seed, saved as out_dir/proxy_<seed>.ply."""
    wf: dict[str, Any] = {
        "rgba": {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}},
        # LoadImage's mask is 1 - alpha; TripoSplat wants the subject as 1.
        "mask": {"class_type": "InvertMask", "inputs": {"mask": ["rgba", 1]}},
        "prep": {"class_type": "TripoSplatPreprocessImage", "inputs": {
            "image": ["rgba", 0], "mask": ["mask", 0], "erode_radius": 1, "size": 1024}},
        "dino": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "dino_v3_vit_h.safetensors"}},
        "flux_vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "decoder": {"class_type": "VAELoader", "inputs": {"vae_name": "triposplat_vae_decoder_fp16.safetensors"}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "triposplat_fp16.safetensors", "weight_dtype": "default"}},
        "cond": {"class_type": "TripoSplatConditioning", "inputs": {
            "clip_vision": ["dino", 0], "vae": ["flux_vae", 0], "image": ["prep", 0]}},
    }
    for seed in seeds:
        wf[f"sample_{seed}"] = {"class_type": "KSampler", "inputs": {
            "model": ["unet", 0], "positive": ["cond", 0], "negative": ["cond", 1], "latent_image": ["cond", 2],
            "seed": seed, "steps": params["steps"], "cfg": params["cfg"], "sampler_name": "dpmpp_2m",
            "scheduler": "simple", "denoise": 1.0}}
        wf[f"decode_{seed}"] = {"class_type": "VAEDecodeTripoSplat", "inputs": {
            "samples": [f"sample_{seed}", 0], "vae": ["decoder", 0], "num_gaussians": params["n_gaussians"], "seed": seed}}
        wf[f"save_{seed}"] = {"class_type": "GiroSaveSplat", "inputs": {
            "splat": [f"decode_{seed}", 0], "path": str(out_dir / f"proxy_{seed}.ply")}}
    return wf


# The TRELLIS.2 family (MIT, native in ComfyUI from v0.34): files in giro's model store.
TRELLIS_MODELS = {
    "trellis2": {"unet": "trellis2/trellis_2_bf16.safetensors", "dino": "dino_v3_vit_l.safetensors", "pad": 1.0},
    # TRELLIS.2 conditioned pixel-aligned on the image, from its field of view (MoGe-2 estimates it)
    "pixal3d": {"unet": "pixal3d/pixal3d_bf16.safetensors", "dino": "dino_v3_L_naf_fp32.safetensors", "pad": 1.1},
}


def trellis_workflow(image: str, seeds: list[int], out_dir: Path, params: dict[str, Any], model: str) -> dict[str, Any]:
    """TRELLIS.2 or Pixal3D on an uploaded RGBA image (Comfy's template, shape at 1024^3, colors
    painted from the texture stage), each mesh's surface as a splat at out_dir/proxy_<seed>.ply."""
    m = TRELLIS_MODELS[model]
    k = lambda model_, pos, neg, latent, seed, steps, cfg, scheduler: {"class_type": "KSampler", "inputs": {  # noqa: E731
        "model": model_, "positive": pos, "negative": neg, "latent_image": latent, "seed": seed, "steps": steps,
        "cfg": cfg, "sampler_name": "euler", "scheduler": scheduler, "denoise": 1.0}}
    wf: dict[str, Any] = {
        "rgba": {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}},
        "mask": {"class_type": "InvertMask", "inputs": {"mask": ["rgba", 1]}},
        "crop": {"class_type": "ImageCropToMask", "inputs": {"images": ["rgba", 0], "masks": ["mask", 0], "width": 1024,
                                                             "height": 1024, "pad_factor": m["pad"], "grow_mask": 0,
                                                             "background": "#000000"}},
        "dino": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": m["dino"]}},
        "shape_vae": {"class_type": "VAELoader", "inputs": {"vae_name": "trellis_2_shape_vae_bf16.safetensors"}},
        "tex_vae": {"class_type": "VAELoader", "inputs": {"vae_name": "trellis_2_texture_vae_bf16.safetensors"}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        # the template's guidance: CFG only in the late part of the structure and shape passes
        "m_struct_cfg": {"class_type": "CFGOverride", "inputs": {"model": ["unet", 0], "cfg": 1.0, "start_percent": 0.667, "end_percent": 1.0}},
        "m_struct_rescale": {"class_type": "RescaleCFG", "inputs": {"model": ["m_struct_cfg", 0], "multiplier": 0.7}},
        "m_struct": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["m_struct_rescale", 0], "shift": 5.0}},
        "m_shape_cfg": {"class_type": "CFGOverride", "inputs": {"model": ["unet", 0], "cfg": 1.0, "start_percent": 0.769, "end_percent": 1.0}},
        "m_shape": {"class_type": "RescaleCFG", "inputs": {"model": ["m_shape_cfg", 0], "multiplier": 0.5}},
        "empty": {"class_type": "EmptyTrellis2LatentStructure", "inputs": {"batch_size": 1}},
    }
    if model == "pixal3d":
        wf |= {
            "moge": {"class_type": "LoadMoGeModel", "inputs": {"model_name": "moge_2_vitl_normal_fp16.safetensors"}},
            "geometry": {"class_type": "MoGeInference", "inputs": {"moge_model": ["moge", 0], "image": ["crop", 0], "resolution_level": 9,
                                                                   "fov_x_degrees": 0.0, "batch_size": 4, "force_projection": True,
                                                                   "apply_mask": True, "refine_steps": 3}},
            "fov": {"class_type": "MoGeGeometryToFOV", "inputs": {"moge_geometry": ["geometry", 0], "axis": "horizontal", "unit": "degrees"}},
            "cond": {"class_type": "Pixal3DConditioning", "inputs": {"clip_vision_model": ["dino", 0], "image": ["crop", 0], "camera_angle_x": ["fov", 0]}},
        }
    else:
        wf["cond"] = {"class_type": "Trellis2Conditioning", "inputs": {"clip_vision_model": ["dino", 0], "image": ["crop", 0]}}
    for seed in seeds:
        n = lambda name: f"{name}_{seed}"  # noqa: E731
        wf |= {
            n("structure"): k(["m_struct", 0], ["cond", 0], ["cond", 1], ["empty", 0], seed, 12, 7.5, "normal"),
            n("voxel"): {"class_type": "VaeDecodeStructureTrellis2", "inputs": {"samples": [n("structure"), 0], "vae": ["shape_vae", 0], "resolution": "32"}},
            n("shape_stage"): {"class_type": "Trellis2ShapeStage", "inputs": {"positive": ["cond", 0], "negative": ["cond", 1], "voxel": [n("voxel"), 0]}},
            n("shape"): k(["m_shape", 0], [n("shape_stage"), 0], [n("shape_stage"), 1], [n("shape_stage"), 2], seed, 20, 7.5, "normal"),
            n("up_stage"): {"class_type": "Trellis2UpsampleStage", "inputs": {"positive": [n("shape_stage"), 0], "negative": [n("shape_stage"), 1],
                                                                             "shape_latent": [n("shape"), 0], "vae": ["shape_vae", 0],
                                                                             "target_resolution": 1024}},
            n("up"): k(["m_shape", 0], [n("up_stage"), 0], [n("up_stage"), 1], [n("up_stage"), 2], seed, 12, 7.5, "simple"),
            n("mesh"): {"class_type": "VaeDecodeShapeTrellis", "inputs": {"samples": [n("up"), 0], "vae": ["shape_vae", 0]}},
            n("tex_stage"): {"class_type": "Trellis2TextureStage", "inputs": {"positive": [n("up_stage"), 0], "negative": [n("up_stage"), 1],
                                                                             "shape_latent": [n("up"), 0]}},
            n("tex"): k(["unet", 0], [n("tex_stage"), 0], [n("tex_stage"), 1], [n("tex_stage"), 2], seed + 1, 12, 1.0, "normal"),
            n("colors"): {"class_type": "VaeDecodeTextureTrellis", "inputs": {"samples": [n("tex"), 0], "vae": ["tex_vae", 0],
                                                                             "shape_subdivides": [n("mesh"), 1]}},
            n("paint"): {"class_type": "PaintMesh", "inputs": {"mesh": [n("mesh"), 0], "voxel_colors": [n("colors"), 0]}},
            n("splat"): {"class_type": "GiroMeshToSplat", "inputs": {"mesh": [n("paint"), 0], "count": params["n_gaussians"],
                                                                     "height": 1.0, "seed": seed}},
            n("save"): {"class_type": "GiroSaveSplat", "inputs": {"splat": [n("splat"), 0], "path": str(out_dir / f"proxy_{seed}.ply")}},
        }
    return wf


class Points:
    """A proxy's Gaussian centers and base colors, enough to fit cameras with."""

    def __init__(self, ply: Path, min_opacity: float = 0.2) -> None:
        s = splat.read_ply(ply)
        keep = 1 / (1 + np.exp(-s["opacity"].astype(np.float64))) > min_opacity
        self.xyz = splat.positions(s)[keep]
        self.rgb = np.clip(0.5 + C0 * np.stack([s[f"f_dc_{i}"] for i in range(3)], 1)[keep], 0, 1)
        self.lo, self.hi = np.percentile(self.xyz, 0.5, axis=0), np.percentile(self.xyz, 99.5, axis=0)
        self.center = (self.lo + self.hi) / 2
        self.n = len(s)

    def project(self, cam: campath.PathCamera, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rot, t = cam.world_to_camera()
        p = self.xyz @ rot.T + t
        front = p[:, 2] > 1e-3
        f = cam.focal(width, height)
        z = np.where(front, p[:, 2], 1.0)
        return f * p[:, 0] / z + width / 2, f * p[:, 1] / z + height / 2, np.where(front, p[:, 2], np.inf)

    def silhouette(self, cam: campath.PathCamera, width: int, height: int) -> np.ndarray:
        u, v, z = self.project(cam, width, height)
        ok = np.isfinite(z) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        m = np.zeros((height, width), bool)
        m[v[ok].astype(int), u[ok].astype(int)] = True
        return ndimage.binary_closing(ndimage.binary_dilation(m), iterations=2)

    def color(self, cam: campath.PathCamera, width: int, height: int) -> np.ndarray:
        """Nearest point's color per pixel (black where none), dilated a little."""
        u, v, z = self.project(cam, width, height)
        ok = np.isfinite(z) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        order = np.argsort(-z[ok])  # far first, so nearer points overwrite them
        img = np.zeros((height, width, 3))
        img[v[ok][order].astype(int), u[ok][order].astype(int)] = self.rgb[ok][order]
        filled = img.max(axis=2) > 0
        for c in range(3):  # fill pinholes from the neighbors
            img[..., c] = np.where(filled, img[..., c], ndimage.grey_dilation(img[..., c], size=3))
        return img


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


def box(m: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def aligned(points: Points, cam: campath.PathCamera, mask: np.ndarray, rounds: int = 3) -> campath.PathCamera:
    """`cam` moved along its view axis and across the image so the proxy's silhouette box matches
    the hero mask's (the target moves; yaw and pitch stay)."""
    h, w = mask.shape
    goal = box(mask)
    assert goal is not None
    for _ in range(rounds):
        got = box(points.silhouette(cam, w, h))
        if got is None:
            cam = replace(cam, distance=cam.distance * 1.5)
            continue
        cam = replace(cam, distance=cam.distance * (got[3] - got[1]) / (goal[3] - goal[1]))
        got = box(points.silhouette(cam, w, h))
        if got is None:
            continue
        # pixels -> world at the target's depth
        per_px = cam.distance / cam.focal(w, h)
        rot, _ = cam.world_to_camera()
        dx = ((goal[0] + goal[2]) - (got[0] + got[2])) / 2 * per_px
        dy = ((goal[1] + goal[3]) - (got[1] + got[3])) / 2 * per_px
        # moving the target by +x shifts the subject the other way in the image
        cam = replace(cam, target=tuple(np.asarray(cam.target) - dx * rot[0] - dy * rot[1]))
    return cam


def color_error(points: Points, cam: campath.PathCamera, hero: np.ndarray, mask: np.ndarray) -> float:
    """Mean squared difference to the hero inside both silhouettes, each normalized per channel
    (the proxy's colors are paler than the hero's)."""
    h, w = mask.shape
    sil = points.silhouette(cam, w, h)
    both = sil & mask
    if both.sum() < 50:
        return math.inf
    a, b = points.color(cam, w, h)[both], hero[both]
    norm = lambda x: (x - x.mean(0)) / (x.std(0) + 1e-6)  # noqa: E731
    return float(np.mean((norm(a) - norm(b)) ** 2))


def fit_hero_camera(points: Points, hero: Image.Image, hero_mask: np.ndarray, fov: float
                    ) -> tuple[campath.PathCamera, dict[str, Any]]:
    """The camera from which the proxy looks most like the hero (see the module docstring)."""
    aspect = hero.width / hero.height
    small = (round(FIT_HEIGHT * aspect), FIT_HEIGHT)
    mask = np.asarray(Image.fromarray(hero_mask.astype(np.uint8) * 255).resize(small)) > 127
    hero_rgb = np.asarray(hero.convert("RGB").resize(small), dtype=np.float64) / 255
    height = float(points.hi[1] - points.lo[1])
    start = campath.PathCamera(0.0, 0.0, 2.0 * height, tuple(points.center), fov)
    scored = []
    for yaw in range(0, 360, 10):
        for pitch in (-10.0, 0.0, 10.0, 20.0, 30.0):
            cam = aligned(points, replace(start, yaw=float(yaw), pitch=pitch), mask)
            scored.append((iou(points.silhouette(cam, *small), mask), cam))
    scored.sort(key=lambda s: -s[0])
    # Front and back (and mirrored sides) share a silhouette: the color decides among the best.
    top = [s for s in scored[:8] if s[0] >= scored[0][0] - 0.05]
    errors = [color_error(points, cam, hero_rgb, mask) for _, cam in top]
    best_iou, best = top[int(np.argmin(errors))]

    # Refine yaw, pitch, distance and framing at a higher resolution.
    big = (round(REFINE_HEIGHT * aspect), REFINE_HEIGHT)
    mask_big = np.asarray(Image.fromarray(hero_mask.astype(np.uint8) * 255).resize(big)) > 127
    rot, _ = best.world_to_camera()

    def moved(x: np.ndarray) -> campath.PathCamera:
        target = np.asarray(best.target) + x[3] * best.distance * rot[0] + x[4] * best.distance * rot[1]
        return replace(best, yaw=best.yaw + x[0], pitch=best.pitch + x[1], distance=best.distance * math.exp(x[2]),
                       target=tuple(target))

    res = optimize.minimize(lambda x: -iou(points.silhouette(moved(x), *big), mask_big), np.zeros(5),
                            method="Nelder-Mead", options={
                                # degrees, degrees, log distance, framing in distances
                                "initial_simplex": np.vstack([np.zeros(5), np.diag([4.0, 3.0, 0.05, 0.02, 0.02])]),
                                "maxiter": 150, "xatol": 0.05, "fatol": 1e-4})
    cam = moved(res.x)
    cam = replace(cam, yaw=round(cam.yaw % 360.0, 3), pitch=round(cam.pitch, 3), distance=round(cam.distance, 5),
                  target=tuple(round(float(v), 5) for v in cam.target))
    final_iou = iou(points.silhouette(cam, *big), mask_big)
    return cam, {
        "iou": round(final_iou, 4), "coarse_iou": round(best_iou, 4),
        "color_error": round(float(min(errors)), 4),
        "candidates": [{"yaw": c.yaw, "pitch": c.pitch, "iou": round(s, 4), "color_error": round(e, 4)}
                       for (s, c), e in zip(top, errors)],
    }


def fit_sheet(points: Points, cam: campath.PathCamera, hero: Image.Image, hero_mask: np.ndarray) -> Image.Image:
    """Hero with the fitted silhouette's outline, then the proxy around the ring from the hero camera."""
    h = 320
    w = round(h * hero.width / hero.height)
    sil = points.silhouette(cam, w, h)
    edge = sil & ~ndimage.binary_erosion(sil, iterations=2)
    base = np.asarray(hero.convert("RGB").resize((w, h)), dtype=np.float64) / 255
    base[edge] = [1.0, 0.2, 0.2]
    tiles = [Image.fromarray((base * 255).astype(np.uint8))]
    for k in range(8):
        tiles.append(Image.fromarray((points.color(replace(cam, yaw=cam.yaw + 45 * k), w, h) * 255).astype(np.uint8)))
    out = Image.new("RGB", (w * len(tiles), h))
    for i, t in enumerate(tiles):
        out.paste(t, (i * w, 0))
    ImageDraw.Draw(out).text((4, 4), f"yaw {cam.yaw:.0f} pitch {cam.pitch:.0f} d {cam.distance:.2f}", fill=(255, 255, 0))
    return out


class Proxy(Stage):
    name = "proxy"
    defaults = {
        "n_seeds": 4,       # TripoSplat runs; the best fit to the hero is kept
        "first_seed": 46,
        "steps": 20,
        "cfg": 3.0,
        "n_gaussians": 262_144,
        "fov": 35.0,        # the hero camera's, over the image's smaller side
    }
    # Also read: "model", "triposplat" (default), "pixal3d" or "trellis2" (TRELLIS_MODELS; these
    # need ComfyUI v0.34 or later).
    extra_params = ("model",)
    inputs = ("hero/hero.png",)
    outputs = ("proxy/proxy.ply", "proxy/proxy.json")
    gpu_mb = SAM_VRAM_MB

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        out = attempt / "proxy"
        if out.exists():
            shutil.rmtree(out)
        (out / "candidates").mkdir(parents=True)
        hero = Image.open(attempt / "hero" / "hero.png").convert("RGB")
        seeds = [params["first_seed"] + i for i in range(params["n_seeds"])]
        try:
            asyncio.run(self._generate(attempt, hero, seeds, params, ctx))
        except ComfyError as e:
            ctx.check_cancelled()
            raise StageFailed(f"ComfyUI: {e}") from None
        except aiohttp.ClientConnectionError:
            ctx.check_cancelled()
            raise StageFailed("lost the connection to ComfyUI (it stopped or crashed); Retry runs this step again") from None

        hero_mask = np.asarray(Image.open(out / "hero_mask.png")) > 127
        fits = []
        for k, seed in enumerate(seeds):
            ctx.check_cancelled()
            ctx.progress(0.6 + 0.35 * k / len(seeds), f"fitting the hero camera to proxy {k + 1}/{len(seeds)}")
            points = Points(out / "candidates" / f"proxy_{seed}.ply")
            cam, fit = fit_hero_camera(points, hero, hero_mask, params["fov"])
            fits.append({"seed": seed, "camera": cam.to_json(), **fit})
            ctx.log(f"seed {seed}: IoU {fit['iou']:.3f}, yaw {cam.yaw:.0f}, pitch {cam.pitch:.0f}, distance {cam.distance:.2f}")
        best = max(fits, key=lambda f: f["iou"])
        shutil.copyfile(out / "candidates" / f"proxy_{best['seed']}.ply", out / "proxy.ply")
        points = Points(out / "proxy.ply")
        cam = campath.PathCamera.from_json(best["camera"])
        fit_sheet(points, cam, hero, hero_mask).save(out / "fit.jpg", quality=90)
        ctx.preview(out / "fit.jpg")
        (out / "proxy.json").write_text(json.dumps({
            "model": params.get("model", "triposplat"),
            "hero_camera": best["camera"], "seed": best["seed"], "iou": best["iou"],
            "bounds": [points.lo.tolist(), points.hi.tolist()], "n_gaussians": points.n,
            "hero_size": [hero.width, hero.height], "fits": fits,
        }, indent=2) + "\n")
        ctx.metric("seed", best["seed"])
        ctx.metric("hero_iou", best["iou"])
        ctx.metric("hero_yaw", cam.yaw)
        ctx.metric("hero_pitch", cam.pitch)
        ctx.metric("hero_distance", cam.distance)
        ctx.progress(1.0, f"proxy seed {best['seed']}, hero silhouette IoU {best['iou']:.2f}")
        if best["iou"] < 0.5:
            raise StageFailed(f"no proxy matches the hero's silhouette (best IoU {best['iou']:.2f})")

    async def _generate(self, attempt: Path, hero: Image.Image, seeds: list[int], params: dict[str, Any],
                        ctx: Ctx) -> None:
        out = attempt / "proxy"
        device = ctx.gpu if ctx.gpu is not None else server.pick_gpu(SAM_VRAM_MB)
        ctx.log(f"GPU {device}: starting ComfyUI if needed")
        mask_params = Masks.defaults
        with await asyncio.to_thread(server.Lease, device) as lease:
            async with ComfyClient(lease.url) as comfy:
                try:
                    ctx.progress(0.05, "masking the hero")
                    raw = out / "raw"
                    async for _ in events(comfy, sam_workflow({"hero": [attempt / "hero" / "hero.png"]}, raw, mask_params), ctx):
                        pass
                    subject = np.asarray(Image.open(raw / "subject" / "hero" / "hero.png")) > 127
                    bg = raw / "background" / "hero" / "hero.png"
                    mask = combine(subject, np.asarray(Image.open(bg)) > 127 if bg.exists() else None,
                                   mask_params["touch_px"], mask_params["gap_px"], mask_params["max_add"])
                    if mask.mean() < mask_params["min_area"]:
                        raise StageFailed("SAM found no subject in the hero image")
                    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(out / "hero_mask.png")
                    rgba = hero.copy()
                    rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255, "L"))
                    rgba.save(out / "hero_rgba.png")

                    model = params.get("model", "triposplat")
                    ctx.progress(0.2, f"{model}, {len(seeds)} seeds")
                    image = await comfy.upload_image(out / "hero_rgba.png")
                    done: Done | None = None
                    wf = (tripo_workflow(image, seeds, out / "candidates", params) if model == "triposplat"
                          else trellis_workflow(image, seeds, out / "candidates", params, model))
                    async for event in events(comfy, wf, ctx):
                        if isinstance(event, Done):
                            done = event
                    assert done is not None
                finally:
                    await comfy.free()
        missing = [s for s in seeds if not (out / "candidates" / f"proxy_{s}.ply").exists()]
        if missing:
            raise StageFailed(f"{params.get('model', 'triposplat')} wrote no splat for seeds {missing}")
