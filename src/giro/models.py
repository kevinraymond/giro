"""The model weights giro runs on: where each one comes from, and `giro models` to fetch them.

giro ships no weights. The store (GIRO_MODELS, else base_path in scripts/extra_model_paths.yaml)
holds them in ComfyUI's folder layout, and this module knows the Hugging Face file behind each
path giro's workflows name. `giro models fetch` downloads what is missing, group by group, and
copies each repo's license text beside the weights; `giro models list` shows what is there.

Groups: "proxy" is the default route (the proxy orbit with both proxies, Wan 2.2 Fun Control,
SeedVR2 and SAM 3.1); "edit" the background edit; "i2v" Wan 2.2 I2V for the handoff option;
"trellis2" the optional third proxy; "da3" the pose fallback's weights; "h3" MiniMax H3 with
the 360 orbit LoRA, never fetched by --all: its license excludes the US, EU, UK and Korea.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATHS = ROOT / "scripts" / "extra_model_paths.yaml"


@dataclass(frozen=True)
class Weight:
    path: str     # in the store, ComfyUI's folder layout as scripts/extra_model_paths.yaml maps it
    repo: str     # Hugging Face repo
    file: str     # path inside the repo
    license: str  # the file's own license (a repo's MIT does not cover a DINOv3 encoder inside it)
    group: str


SAM = "SAM License (Meta): commercial use allowed; no military or weapons use; keep the license text with the weights"
DINO = "DINOv3 License (Meta): commercial use allowed; no military or weapons use; keep the license text with the weights"
H3 = "MiniMax H3 Community License: NOT licensed for use in the US, EU, UK or Korea; 'Powered by MiniMax H3' notice required"
WAN = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"

WEIGHTS: tuple[Weight, ...] = (
    # The proxy orbit (the default): Wan 2.2 Fun Control repaints the proxy's depth along the camera path.
    Weight("diffusion/wan22/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors", WAN,
           "split_files/diffusion_models/wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors", "Apache-2.0", "proxy"),
    Weight("diffusion/wan22/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors", WAN,
           "split_files/diffusion_models/wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors", "Apache-2.0", "proxy"),
    Weight("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", WAN,
           "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "Apache-2.0", "proxy"),
    Weight("vae/wan21/wan_2.1_vae.safetensors", WAN, "split_files/vae/wan_2.1_vae.safetensors", "Apache-2.0", "proxy"),
    # SeedVR2 doubles the frames' resolution.
    Weight("diffusion/seedvr2/seedvr2_7b_fp8_e4m3fn.safetensors", "Comfy-Org/SeedVR2",
           "diffusion_models/seedvr2_7b_fp8_e4m3fn.safetensors", "Apache-2.0", "proxy"),
    Weight("vae/seedvr2_ema_vae_fp16.safetensors", "Comfy-Org/SeedVR2", "vae/seedvr2_ema_vae_fp16.safetensors", "Apache-2.0", "proxy"),
    # SAM 3.1 masks the subject in every frame.
    Weight("checkpoints/sam3.1_multiplex_fp16.safetensors", "Comfy-Org/sam3.1", "checkpoints/sam3.1_multiplex_fp16.safetensors", SAM, "proxy"),
    # TripoSplat, the proxy for people and characters.
    Weight("diffusion/triposplat_fp16.safetensors", "VAST-AI/TripoSplat", "diffusion_models/triposplat_fp16.safetensors", "MIT", "proxy"),
    Weight("vae/triposplat_vae_decoder_fp16.safetensors", "VAST-AI/TripoSplat", "vae/triposplat_vae_decoder_fp16.safetensors", "MIT", "proxy"),
    Weight("vae/flux2-vae.safetensors", "VAST-AI/TripoSplat", "vae/flux2-vae.safetensors", "Apache-2.0 (FLUX.2 autoencoder)", "proxy"),
    Weight("clip_vision/dino_v3_vit_h.safetensors", "VAST-AI/TripoSplat", "clip_vision/dino_v3_vit_h.safetensors", DINO, "proxy"),
    # Pixal3D, the proxy for objects, with the TRELLIS.2 decoders and MoGe-2 for the field of view.
    Weight("diffusion/pixal3d/pixal3d_bf16.safetensors", "Comfy-Org/Pixal3D", "diffusion_models/pixal3d_bf16.safetensors", "MIT", "proxy"),
    Weight("clip_vision/dino_v3_L_naf_fp32.safetensors", "Comfy-Org/Pixal3D", "clip_vision/dino_v3_L_naf_fp32.safetensors", DINO, "proxy"),
    Weight("vae/trellis_2_shape_vae_bf16.safetensors", "Comfy-Org/Pixal3D", "vae/trellis_2_shape_vae_bf16.safetensors", "MIT", "proxy"),
    Weight("vae/trellis_2_texture_vae_bf16.safetensors", "Comfy-Org/Pixal3D", "vae/trellis_2_texture_vae_bf16.safetensors", "MIT", "proxy"),
    Weight("geometry_estimation/moge_2_vitl_normal_fp16.safetensors", "Comfy-Org/MoGe",
           "geometry_estimation/moge_2_vitl_normal_fp16.safetensors", "MIT", "proxy"),
    # The background edit before the orbit.
    Weight("diffusion/qwen_image_edit_2511_int8_convrot.safetensors", "Comfy-Org/Qwen-Image-Edit_ComfyUI",
           "split_files/diffusion_models/qwen_image_edit_2511_int8_convrot.safetensors", "Apache-2.0", "edit"),
    Weight("text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "Comfy-Org/Qwen-Image_ComfyUI",
           "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors", "Apache-2.0", "edit"),
    Weight("vae/qwen_image_vae.safetensors", "Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "Apache-2.0", "edit"),
    Weight("loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "lightx2v/Qwen-Image-Edit-2511-Lightning",
           "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "Apache-2.0", "edit"),
    # Wan 2.2 I2V: the handoff option (orbit_video.handoff) finishes the low-noise steps with it.
    Weight("diffusion/wan22/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", WAN,
           "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "Apache-2.0", "i2v"),
    Weight("diffusion/wan22/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", WAN,
           "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "Apache-2.0", "i2v"),
    # TRELLIS.2, an optional proxy (-p proxy.model=trellis2).
    Weight("diffusion/trellis2/trellis_2_bf16.safetensors", "Comfy-Org/TRELLIS.2", "diffusion_models/trellis_2_bf16.safetensors", "MIT", "trellis2"),
    Weight("clip_vision/dino_v3_vit_l.safetensors", "Comfy-Org/TRELLIS.2", "clip_vision/dino_v3_vit_l.safetensors", DINO, "trellis2"),
    # MiniMax H3 with the 360 orbit LoRA (--model h3).
    Weight("diffusion/minimax_h3_fl2va_pruned_int8_convrot.safetensors", "Comfy-Org/MiniMax-H3",
           "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", H3, "h3"),
    Weight("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "Comfy-Org/MiniMax-H3",
           "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", H3, "h3"),
    Weight("vae/minimax_h3_video_vae_fp16.safetensors", "Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", H3, "h3"),
    Weight("loras/h3/minimax_h3_flf2v_orbit360_pablodawson_v1.safetensors", "pablodawson/MiniMax-H3-360-Orbit-LoRA",
           "minimax_h3_flf2v_lora_v1.safetensors", H3 + " (a LoRA on H3)", "h3"),
)
# Not single files: the pose fallback's weights, a Hugging Face snapshot in the hub cache (HF_HOME)
# that vendor/depth-anything-3's worker loads by repo id.
SNAPSHOTS: dict[str, tuple[str, str]] = {"da3": ("depth-anything/DA3-BASE", "Apache-2.0")}

GROUPS: dict[str, str] = {
    "proxy": "the proxy orbit, the default route: both proxies, Wan 2.2 Fun Control, SeedVR2, SAM 3.1",
    "edit": "the background edit (Qwen-Image-Edit 2511)",
    "i2v": "Wan 2.2 I2V, for the handoff option",
    "trellis2": "TRELLIS.2, an optional third proxy",
    "da3": "Depth Anything 3, the pose fallback (needs scripts/setup_da3.sh)",
    "h3": "MiniMax H3 with the 360 orbit LoRA: license-restricted, never part of --all",
}
DEFAULT_GROUPS = ("proxy", "edit")
ALL_GROUPS = tuple(g for g in GROUPS if g != "h3")


def store() -> Path:
    """Where the weights live: GIRO_MODELS, else scripts/extra_model_paths.yaml's base_path."""
    if os.environ.get("GIRO_MODELS"):
        return Path(os.environ["GIRO_MODELS"]).expanduser()
    m = re.search(r"^\s*base_path:\s*(\S+)", MODEL_PATHS.read_text(), re.MULTILINE)
    if not m:
        raise RuntimeError(f"no base_path in {MODEL_PATHS}")
    return Path(m.group(1)).expanduser()


def weights(groups: Iterable[str]) -> list[Weight]:
    wanted = set(groups)
    unknown = wanted - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown model group(s) {sorted(unknown)}; have {', '.join(GROUPS)}")
    return [w for w in WEIGHTS if w.group in wanted]


def missing(groups: Iterable[str], root: Path | None = None) -> list[Weight]:
    root = root or store()
    return [w for w in weights(groups) if not (root / w.path).is_file()]


def describe(root: Path | None = None) -> str:
    """One line per weight: present or missing, size, license."""
    root = root or store()
    lines = [f"models in {root}"]
    for group, what in GROUPS.items():
        lines.append(f"\n[{group}] {what}")
        for w in weights([group]):
            p = root / w.path
            state = f"{p.stat().st_size / 2**30:6.2f} GB" if p.is_file() else "   missing"
            lines.append(f"  {state}  {w.path}  ({w.license.split(':')[0].split(' (')[0]})")
        if group in SNAPSHOTS:
            repo, lic = SNAPSHOTS[group]
            lines.append(f"  snapshot  {repo} in the Hugging Face cache ({lic})")
    return "\n".join(lines)


Download = Callable[[str, str, Path], Path]  # (repo, file, into) -> the downloaded file


def _hf_download(repo: str, file: str, into: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, file, local_dir=into))


def _hf_license_files(repo: str) -> list[str]:
    from huggingface_hub import list_repo_files

    try:
        return [f for f in list_repo_files(repo) if re.fullmatch(r"LICENSE[^/]*|LICENCE[^/]*", f, re.IGNORECASE)]
    except Exception:  # noqa: BLE001 - a license copy is best effort; the weights are not
        return []


def fetch(groups: Iterable[str], root: Path | None = None, log: Callable[[str], None] = print,
          download: Download = _hf_download, license_files: Callable[[str], list[str]] = _hf_license_files) -> list[Weight]:
    """Download the groups' missing weights into the store (resumable: huggingface_hub keeps
    partial files), each repo's license text into <store>/licenses/, and the snapshots into
    the hub cache. Returns what was fetched."""
    root = root or store()
    groups = list(groups)
    if "h3" in groups:
        log(f"WARNING: {H3}. Fetch it only where its license allows you to use it.")
    todo = missing(groups, root)
    done: list[Weight] = []
    scratch = root / ".downloads"
    for w in todo:
        target = root / w.path
        log(f"fetching {w.repo}/{w.file} -> {w.path} ({w.license.split(':')[0].split(' (')[0]})")
        got = download(w.repo, w.file, scratch / w.repo.replace("/", "--"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(got), target)
        done.append(w)
    for repo in sorted({w.repo for w in weights(groups)}):
        for name in license_files(repo):
            copy = root / "licenses" / f"{repo.replace('/', '--')}--{Path(name).name}"
            if not copy.exists():
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(download(repo, name, scratch / repo.replace("/", "--"))), copy)
                log(f"license text: {copy}")
    for group in groups:
        if group in SNAPSHOTS:
            repo, _ = SNAPSHOTS[group]
            log(f"snapshot {repo} -> the Hugging Face cache ({os.environ.get('HF_HOME') or '~/.cache/huggingface'})")
            from huggingface_hub import snapshot_download

            snapshot_download(repo)
    shutil.rmtree(scratch, ignore_errors=True)
    present = [w for w in weights(groups) if (root / w.path).is_file()]
    log(f"{len(done)} fetched, {len(present)} of {len(weights(groups))} present in {root}")
    return done
