"""Proxy-guided orbit test: hero -> TripoSplat -> depth turntable -> Wan 2.2 Fun Control -> frames_raw.

    proxy_orbit.py OUT_ATTEMPT SRC_ATTEMPT GPU WIDTH HEIGHT DISTANCE PITCH SEED "PROMPT"
"""
import asyncio, json, shutil, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
from giro.comfy import server
from giro.comfy.client import ComfyClient, Done, Progress

out, src = Path(sys.argv[1]), Path(sys.argv[2])
GPU, W, H, DIST, PITCH, SEED, TEXT = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), int(sys.argv[8]), sys.argv[9]
# optional: TURNS PITCH_END MODE (ramp|wave) for a path whose elevation changes
TURNS, PITCH_END, MODE = (float(sys.argv[10]), float(sys.argv[11]), sys.argv[12]) if len(sys.argv) > 12 else (1.0, PITCH, "ramp")
N = 81
NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

def tripo(image):
    return {
        "rgba": {"class_type": "LoadImage", "inputs": {"image": image, "upload": "image"}},
        "mask": {"class_type": "InvertMask", "inputs": {"mask": ["rgba", 1]}},
        "prep": {"class_type": "TripoSplatPreprocessImage", "inputs": {"image": ["rgba", 0], "mask": ["mask", 0], "erode_radius": 1, "size": 1024}},
        "dino": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "dino_v3_vit_h.safetensors"}},
        "flux_vae": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "decoder": {"class_type": "VAELoader", "inputs": {"vae_name": "triposplat_vae_decoder_fp16.safetensors"}},
        "tripo_unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "triposplat_fp16.safetensors", "weight_dtype": "default"}},
        "cond": {"class_type": "TripoSplatConditioning", "inputs": {"clip_vision": ["dino", 0], "vae": ["flux_vae", 0], "image": ["prep", 0]}},
        "tripo_sample": {"class_type": "KSampler", "inputs": {"model": ["tripo_unet", 0], "positive": ["cond", 0], "negative": ["cond", 1], "latent_image": ["cond", 2],
                         "seed": 46, "steps": 20, "cfg": 3.0, "sampler_name": "dpmpp_2m", "scheduler": "simple", "denoise": 1.0}},
        "splat": {"class_type": "VAEDecodeTripoSplat", "inputs": {"samples": ["tripo_sample", 0], "vae": ["decoder", 0], "num_gaussians": 262144, "seed": 46}},
    }

def turntable(wf, name, style, frames, yaw, w, h):
    wf[f"cam_{name}"] = {"class_type": "CreateCameraInfo", "inputs": {"mode": "orbit", "mode.yaw": yaw, "mode.pitch": PITCH, "mode.distance": DIST,
                         "target_x": 0.0, "target_y": 0.0, "target_z": 0.0, "roll": 0.0, "fov": 35.0, "zoom": 1.0, "camera_type": "perspective"}}
    wf[name] = {"class_type": "RenderSplat", "inputs": {"splat": ["splat", 0], "width": w, "height": h, "frames": frames, "splat_scale": 1.0, "sharpen": 2.0,
                "headlight_shading": 0.0, "opacity_threshold": 0.0, "render_style": style, "background": "#000000", "camera_info": [f"cam_{name}", 0]}}

async def run(comfy, wf, dest_by_node):
    done = None; t0 = time.monotonic()
    async for ev in comfy.run(wf):
        if isinstance(ev, Done): done = ev
        elif isinstance(ev, Progress) and ev.value % 5 == 0: print(f"  step {ev.value}/{ev.max} at {time.monotonic()-t0:.0f}s", flush=True)
    for node, dest in dest_by_node.items():
        dest.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(done.outputs[node]["images"]):
            await comfy.download(img, dest / f"{i:05d}.png")
    return time.monotonic() - t0

async def main():
    out.mkdir(parents=True, exist_ok=True)
    (out / "hero").mkdir(exist_ok=True); shutil.copyfile(src / "hero" / "hero.png", out / "hero" / "hero.png")
    rgb = Image.open(src / "hero" / "hero.png").convert("RGB")
    mask = Image.open(src / "masks" / "hero" / "hero.png").convert("L").resize(rgb.size)
    rgba = rgb.copy(); rgba.putalpha(mask); rgba.save(out / "hero_rgba.png")
    stamp = time.strftime("%H%M%S")
    with await asyncio.to_thread(server.Lease, GPU) as lease:
        async with ComfyClient(lease.url) as comfy:
            up_rgba = await comfy.upload_image(out / "hero_rgba.png"); up_hero = await comfy.upload_image(out / "hero" / "hero.png")
            # A. find the yaw at which the proxy shows its front: the silhouette closest to the hero's
            wf = tripo(up_rgba); turntable(wf, "probe", "color", 36, 0.0, W, H)
            wf["save_probe"] = {"class_type": "SaveImage", "inputs": {"images": ["probe", 0], "filename_prefix": f"giro/proxy-{stamp}-probe"}}
            await run(comfy, wf, {"save_probe": out / "proxy_probe"})
            def norm_sil(m):  # silhouette cropped to its box, as a 64x64 grid
                ys, xs = np.nonzero(m); crop = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                return np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((64, 64))) > 127, crop.shape[1] / crop.shape[0]
            hero_sil, hero_aspect = norm_sil(np.asarray(mask) > 127)
            hero_rgb = np.asarray(rgb.resize((W, H)), dtype=np.float32)
            scores = []
            for i, f in enumerate(sorted((out / "proxy_probe").glob("*.png"))):
                im = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32)
                sil, aspect = norm_sil(im.max(axis=2) > 12)
                iou = (sil & hero_sil).sum() / max(1, (sil | hero_sil).sum())
                scores.append((iou - abs(np.log(aspect / hero_aspect)), i))
            # front and back share a silhouette: break the tie by color against the hero
            top = sorted(scores, reverse=True)[:6]
            def color_err(i):
                im = Image.open(sorted((out / "proxy_probe").glob("*.png"))[i]).convert("RGB")
                a = np.asarray(im.resize((48, 64)), dtype=np.float32); b = np.asarray(rgba.convert("RGB").resize((48, 64)), dtype=np.float32) * (np.asarray(mask.resize((48, 64)))[..., None] > 127)
                return float(np.mean((a - b) ** 2))
            best = min(top, key=lambda s: color_err(s[1]))[1]
            yaw = 360.0 * best / 36
            print(f"front: probe frame {best} (yaw {yaw:.0f}); top silhouettes {[i for _, i in top]}", flush=True)
            # B. depth turntable from the front, and Wan 2.2 Fun Control with the hero as reference
            wf = tripo(up_rgba)
            for name in ("depth", "color"):
                wf[name] = {"class_type": "GiroRenderSplatPath", "inputs": {"splat": ["splat", 0], "width": W, "height": H, "frames": N, "yaw_start": yaw,
                            "turns": TURNS, "pitch_start": PITCH, "pitch_end": PITCH_END, "pitch_mode": MODE, "distance": DIST, "fov": 35.0,
                            "render_style": name, "background": "#000000"}}
            wf |= {
                "hero": {"class_type": "LoadImage", "inputs": {"image": up_hero, "upload": "image"}},
                "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan22/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
                "unet_low": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan22/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
                "shift": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet", 0], "shift": 8.0}},
                "shift_low": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["unet_low", 0], "shift": 8.0}},
                "te": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
                "wan_vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan21/wan_2.1_vae.safetensors"}},
                "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": TEXT}},
                "neg": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": NEG}},
                "control": {"class_type": "Wan22FunControlToVideo", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["wan_vae", 0],
                            "width": W, "height": H, "length": N, "batch_size": 1, "ref_image": ["hero", 0], "control_video": ["depth", 0]}},
                "s1": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["shift", 0], "add_noise": "enable", "noise_seed": SEED, "steps": 20, "cfg": 3.5,
                       "sampler_name": "euler", "scheduler": "simple", "positive": ["control", 0], "negative": ["control", 1], "latent_image": ["control", 2],
                       "start_at_step": 0, "end_at_step": 10, "return_with_leftover_noise": "enable"}},
                "s2": {"class_type": "KSamplerAdvanced", "inputs": {"model": ["shift_low", 0], "add_noise": "disable", "noise_seed": 0, "steps": 20, "cfg": 3.5,
                       "sampler_name": "euler", "scheduler": "simple", "positive": ["control", 0], "negative": ["control", 1], "latent_image": ["s1", 0],
                       "start_at_step": 10, "end_at_step": 10000, "return_with_leftover_noise": "disable"}},
                "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["s2", 0], "vae": ["wan_vae", 0]}},
                "save_frames": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": f"giro/proxy-{stamp}-frame"}},
                "save_depth": {"class_type": "SaveImage", "inputs": {"images": ["depth", 0], "filename_prefix": f"giro/proxy-{stamp}-depth"}},
                "save_color": {"class_type": "SaveImage", "inputs": {"images": ["color", 0], "filename_prefix": f"giro/proxy-{stamp}-color"}},
            }
            for d in ("frames_raw", "proxy_depth", "proxy_color"):
                if (out / d).exists(): shutil.rmtree(out / d)
            secs = await run(comfy, wf, {"save_frames": out / "frames_raw", "save_depth": out / "proxy_depth", "save_color": out / "proxy_color"})
            await comfy.free()
    (out / "proxy.json").write_text(json.dumps({"yaw": yaw, "pitch": PITCH, "pitch_end": PITCH_END, "turns": TURNS, "pitch_mode": MODE, "distance": DIST, "fov": 35.0, "frames": N, "width": W, "height": H,
                                                "seed": SEED, "prompt": TEXT, "seconds": round(secs, 1)}, indent=2))
    print(f"done in {secs:.0f}s: {len(list((out / 'frames_raw').glob('*.png')))} frames", flush=True)
asyncio.run(main())
