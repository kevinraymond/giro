"""giro's own LoRA trainer for Qwen-Image-Edit 2511 (board #3735): the view LoRAs without ai-toolkit's training
loop, on one GPU or several (torchrun: data parallel, every GPU holds the whole 3-bit model and trains on its own
samples; the LoRA's gradients are averaged after each step), and with several target images per sample (the
joint multi-view LoRA: all targets denoised together as one multi-frame segment).

    # 1. cache text embeddings and latents (the text encoder and VAE are not needed while training)
    torchrun --nproc_per_node 2 qwen_edit_lora.py cache DATA CACHE
    # 2. train
    torchrun --nproc_per_node 2 qwen_edit_lora.py train CACHE OUT --name NAME [--steps 1000] [--lr 1e-4]
        [--rank 32] [--init-lora V1.safetensors] [--save-every 250] [--warmup 0] [--cosine]

Run with ai-toolkit's venv (vendor/ai-toolkit/venv/bin/torchrun): it reuses ONLY ai-toolkit's 3-bit quantizer,
because ostris's accuracy recovery adapter (ARA) was trained against exactly that quantization; everything else
here is diffusers (Apache-2.0), transformers and PyTorch.

DATA is ai-toolkit's folder layout (target/<id>.png + <id>.txt caption, control1/<id>.png, control2/..., matched
by name; gso_dataset.py, joint_dataset.py) or a JSONL file with {"id", "targets": [paths], "controls": [paths],
"prompt"} per line (several targets: the joint LoRA).

What v1 was trained with (ai-toolkit, Oct 5), reproduced:
- the base transformer quantized to uint3 with the ARA (rank 16, frozen, always on) on the linears it covers,
  everything else uint8; our LoRA (rank 32, alpha = rank, so ComfyUI's default scale 1) on the 14 linears of
  each of the 60 blocks, saved with ComfyUI's keys (diffusion_model.<module>.lora_A/B.weight).
- text: Qwen2.5-VL on the prompt and every control image at CONDITION_IMAGE_SIZE (diffusers' pipeline);
  controls into the sequence as VAE latents at VAE_IMAGE_SIZE; target latents at their own size; latents
  normalized with the VAE's latents_mean/std.
- flow matching: sigma = t/1000 for t uniform on linspace(1000, 1, 1000); x_t = (1 - sigma) x0 + sigma noise;
  the model predicts noise - x0 on the target tokens; MSE times ai-toolkit's default timestep weighting
  ("weighted"). Qwen 2511's zero_cond_t: the first image segment is the noisy one (timestep sigma), every later
  one a clean reference (timestep 0); several targets therefore go in as ONE segment of K frames, never as K
  segments (diffusers would treat targets 2..K as references).
- AdamW 8-bit (bitsandbytes), lr 1e-4 constant, gradient clipping 1.0, gradient checkpointing, batch 1 per GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
AITK = ROOT / "vendor" / "ai-toolkit"
REPO = "Qwen/Qwen-Image-Edit-2511"
ARA = ("ostris/accuracy_recovery_adapters", "qwen_image_edit_2511_torchao_uint3.safetensors")
LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj",
                "attn.add_v_proj", "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj",
                "txt_mlp.net.2", "img_mod.1", "txt_mod.1")  # v1's 14 per block


def setup() -> tuple[int, int, torch.device]:
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    if world > 1:
        dist.init_process_group("nccl", device_id=dev)
    return rank, world, dev


def log(rank: int, msg: str) -> None:
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def items(data: Path) -> list[dict]:
    if data.suffix == ".jsonl":
        return [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    controls = sorted(d for d in data.iterdir() if d.is_dir() and d.name.startswith("control"))
    out = []
    for t in sorted((data / "target").glob("*.png")):
        out.append({"id": t.stem, "targets": [str(t)], "controls": [str(c / t.name) for c in controls],
                    "prompt": t.with_suffix(".txt").read_text().strip()})
    return out


# ---- cache -------------------------------------------------------------------------------------------------

@torch.no_grad()
def cache(args: argparse.Namespace) -> None:
    from diffusers import AutoencoderKLQwenImage, QwenImageEditPlusPipeline
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (CONDITION_IMAGE_SIZE, VAE_IMAGE_SIZE,
                                                                            calculate_dimensions)

    rank, world, dev = setup()
    todo = [it for i, it in enumerate(items(args.data)) if i % world == rank]
    args.cache.mkdir(parents=True, exist_ok=True)
    todo = [it for it in todo if not (args.cache / f"{it['id']}.pt").exists()]
    log(rank, f"caching {len(todo)} items per rank into {args.cache}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, transformer=None, vae=None, torch_dtype=torch.bfloat16)
    pipe.text_encoder.to(dev)
    vae = AutoencoderKLQwenImage.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.bfloat16).to(dev)
    mean = torch.tensor(vae.config.latents_mean, device=dev).view(1, -1, 1, 1, 1)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=dev).view(1, -1, 1, 1, 1)
    proc = pipe.image_processor

    def encode(img: Image.Image, w: int, h: int) -> torch.Tensor:
        x = proc.preprocess(img, h, w).to(dev, torch.bfloat16).unsqueeze(2)  # (1, 3, 1, h, w), -1..1
        z = vae.encode(x).latent_dist.mode()
        return ((z - mean) * inv_std)[0, :, 0].to(torch.bfloat16).cpu()  # (16, h/8, w/8)

    t0 = time.monotonic()
    for k, it in enumerate(todo):
        ctrl = [Image.open(p).convert("RGB") for p in it["controls"]]
        cond = []
        ctrl_lat = []
        for img in ctrl:
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, img.width / img.height)
            cond.append(proc.resize(img, ch, cw))
            vw, vh = calculate_dimensions(VAE_IMAGE_SIZE, img.width / img.height)
            ctrl_lat.append(encode(img, vw, vh))
        emb, mask = pipe.encode_prompt(prompt=[it["prompt"]], image=cond, device=dev, num_images_per_prompt=1)
        if mask is None:
            mask = torch.ones(emb.shape[:2], dtype=torch.long, device=dev)
        tgt = []
        for p in it["targets"]:
            img = Image.open(p).convert("RGB")
            tgt.append(encode(img, img.width // 32 * 32, img.height // 32 * 32))
        torch.save({"prompt_embeds": emb[0].to(torch.bfloat16).cpu(), "mask": mask[0].cpu(), "targets": tgt, "controls": ctrl_lat},
                   args.cache / f"{it['id']}.pt")
        if rank == 0 and (k % 50 == 0 or k == len(todo) - 1):
            log(rank, f"{k + 1}/{len(todo)} ({(time.monotonic() - t0) / (k + 1):.2f} s/item)")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


# ---- model -------------------------------------------------------------------------------------------------

class LoRALinear(torch.nn.Module):
    """A (quantized) linear plus the frozen accuracy recovery adapter plus our trainable LoRA."""

    def __init__(self, base: torch.nn.Module, in_f: int, out_f: int):
        super().__init__()
        self.base, self.in_f, self.out_f = base, in_f, out_f
        self.ara_A = self.ara_B = self.A = self.B = None
        self.ara_scale = self.scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.ara_A is not None:
            y = y + F.linear(F.linear(x, self.ara_A), self.ara_B) * self.ara_scale
        if self.A is not None:
            y = y + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale
        return y


def load_transformer(args: argparse.Namespace, rank: int, world: int, dev: torch.device) -> tuple[torch.nn.Module, dict]:
    """The transformer, ARA'd and quantized as ai-toolkit does it, our LoRA attached; returns it and our LoRA
    modules by name. Ranks load one after another (each holds the bf16 model in RAM until it is quantized)."""
    sys.path.insert(0, str(AITK))
    from diffusers import QwenImageTransformer2DModel
    from huggingface_hub import hf_hub_download
    from optimum.quanto import freeze
    from safetensors.torch import load_file
    from toolkit.util.quantize import get_qtype, quantize

    for turn in range(world):
        if turn == rank:
            log(rank, f"rank {rank}: loading the transformer")
            tr = QwenImageTransformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.bfloat16)
            tr.requires_grad_(False)
            ara = load_file(args.ara or hf_hub_download(*ARA))
            ara_rank = next(v.shape[0] for k, v in ara.items() if k.endswith("lora_A.weight"))
            ara_names = {k[len("diffusion_model."):].split(".lora_")[0] for k in ara}
            lora_names = {n for n, _ in tr.named_modules() if n.startswith("transformer_blocks.") and n.count(".") >= 2
                          and n.split(".", 2)[2] in LORA_TARGETS}
            wrapped: dict[str, LoRALinear] = {}
            for name, m in list(tr.named_modules()):
                if isinstance(m, torch.nn.Linear) and (name in ara_names or name in lora_names):
                    w = LoRALinear(m, m.in_features, m.out_features)
                    parent, _, leaf = name.rpartition(".")
                    setattr(tr.get_submodule(parent) if parent else tr, leaf, w)
                    wrapped[name] = w
            q3, q8 = get_qtype(args.qtype), get_qtype("uint8")
            for name, w in wrapped.items():
                w.base.to(dev, torch.bfloat16)
                quantize(w.base, weights=q3 if name in ara_names else q8)
                freeze(w.base)
                if name in ara_names:
                    w.ara_A = ara[f"diffusion_model.{name}.lora_A.weight"].to(dev, torch.bfloat16)
                    w.ara_B = ara[f"diffusion_model.{name}.lora_B.weight"].to(dev, torch.bfloat16)
                    w.ara_scale = 1.0  # alpha = rank (ai-toolkit builds the ARA network that way)
            quantize(tr, weights=q8, exclude=[f"{n}*" for n in wrapped], quantize_device=dev, keep_on_quantize_device=True)
            tr.to(dev)
            del ara
            torch.cuda.empty_cache()
            log(rank, f"rank {rank}: on {dev}, {torch.cuda.memory_allocated(dev) / 2**30:.1f} GB (ARA rank {ara_rank} on {len(ara_names)} linears)")
        if world > 1:
            dist.barrier()

    lora = {}
    init = load_file(args.init_lora) if args.init_lora else {}
    dtype = torch.float32 if args.lora_dtype == "fp32" else torch.bfloat16
    for name in sorted(lora_names):
        w = wrapped[name]
        a, b = f"diffusion_model.{name}.lora_A.weight", f"diffusion_model.{name}.lora_B.weight"
        if a in init:
            A, B = init[a].to(dev, dtype), init[b].to(dev, dtype)
        else:
            A = torch.empty(args.rank, w.in_f, device=dev, dtype=dtype)
            torch.nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            B = torch.zeros(w.out_f, args.rank, device=dev, dtype=dtype)
        w.A, w.B = torch.nn.Parameter(A), torch.nn.Parameter(B)
        w.scale = (args.alpha or A.shape[0]) / A.shape[0]
        lora[name] = w
    if world > 1:  # every rank starts from rank 0's weights
        for w in lora.values():
            dist.broadcast(w.A.data, 0)
            dist.broadcast(w.B.data, 0)
    tr.enable_gradient_checkpointing()
    return tr, lora


def save_lora(lora: dict, path: Path, alpha: float | None) -> None:
    from safetensors.torch import save_file

    sd = {}
    for name, w in lora.items():
        sd[f"diffusion_model.{name}.lora_A.weight"] = w.A.detach().to(torch.float16).cpu().contiguous()
        sd[f"diffusion_model.{name}.lora_B.weight"] = w.B.detach().to(torch.float16).cpu().contiguous()
        if alpha and alpha != w.A.shape[0]:
            sd[f"diffusion_model.{name}.alpha"] = torch.tensor(float(alpha))
    tmp = path.with_suffix(".tmp")
    save_file(sd, str(tmp), metadata={"format": "pt", "trainer": "giro qwen_edit_lora.py"})
    tmp.replace(path)


# ---- train -------------------------------------------------------------------------------------------------

def pack(lat: torch.Tensor) -> torch.Tensor:
    """(C, H, W) latents -> (H/2 * W/2, 4C) tokens, diffusers' 2x2 patch order."""
    c, h, w = lat.shape
    return lat.view(c, h // 2, 2, w // 2, 2).permute(1, 3, 0, 2, 4).reshape((h // 2) * (w // 2), c * 4)


def train(args: argparse.Namespace) -> None:
    import bitsandbytes as bnb

    sys.path.insert(0, str(AITK))
    from toolkit.timestep_weighing.default_weighing_scheme import default_weighing_scheme

    rank, world, dev = setup()
    files = sorted(args.cache.glob("*.pt"))
    assert files, f"nothing cached in {args.cache}"
    args.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.out / f"{args.name}.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()} | {"world": world}, indent=1))
    log(rank, f"{len(files)} cached samples, {world} GPU(s), {args.steps} steps")
    tr, lora = load_transformer(args, rank, world, dev)
    params = [p for w in lora.values() for p in (w.A, w.B)]
    log(rank, f"LoRA: {len(lora)} linears, {sum(p.numel() for p in params) / 1e6:.0f}M parameters")
    opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)

    def lr_at(step: int) -> float:
        f = min(1.0, (step + 1) / args.warmup) if args.warmup else 1.0
        if args.cosine:
            f *= 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        return args.lr * f

    timesteps = torch.linspace(1000, 1, 1000)
    weights = torch.tensor(default_weighing_scheme, dtype=torch.float32)
    rng = random.Random(args.seed)
    order: list[int] = []
    ema, t_last = None, time.monotonic()
    tr.train()
    for step in range(args.steps):
        while len(order) < world:  # a fresh shuffle per epoch, the same on every rank
            perm = list(range(len(files)))
            rng.shuffle(perm)
            order += perm
        idx = order[rank]
        order = order[world:]
        g = torch.Generator().manual_seed(args.seed * 1_000_003 + step * world + rank)
        d = torch.load(files[idx], map_location=dev)
        k = int(torch.randint(0, 1000, (1,), generator=g))
        sigma = float(timesteps[k]) / 1000
        x0 = torch.cat([pack(t.float()) for t in d["targets"]])  # K frames, frame-major
        noise = torch.randn(x0.shape, generator=g).to(dev)
        xt = (1 - sigma) * x0 + sigma * noise
        ctrl = [pack(c.float()) for c in d["controls"]]
        _, h, w = d["targets"][0].shape
        img_shapes = [[(len(d["targets"]), h // 2, w // 2)] + [(1, c.shape[1] // 2, c.shape[2] // 2) for c in d["controls"]]]
        hidden = torch.cat([xt] + ctrl)[None].to(torch.bfloat16)
        pred = tr(hidden_states=hidden, timestep=torch.tensor([sigma], device=dev, dtype=torch.bfloat16),
                  encoder_hidden_states=d["prompt_embeds"][None].to(torch.bfloat16),
                  encoder_hidden_states_mask=d["mask"][None].to(torch.long), img_shapes=img_shapes, return_dict=False)[0]
        pred = pred[0, : x0.shape[0]]
        loss = F.mse_loss(pred.float(), (noise - x0).float()) * float(weights[k])
        loss.backward()
        if world > 1:  # average the gradients in place, per tensor (a flat copy would cost ~1 GB at 24 GB)
            for work in [dist.all_reduce(p.grad, async_op=True) for p in params]:
                work.wait()
            for p in params:
                p.grad /= world
        torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lv = loss.detach()
        if world > 1:
            dist.all_reduce(lv)
            lv /= world
        ema = float(lv) if ema is None else 0.98 * ema + 0.02 * float(lv)
        if step % args.log_every == 0 or step == args.steps - 1:
            now = time.monotonic()
            log(rank, f"step {step + 1}/{args.steps}: loss {float(lv):.4f} (ema {ema:.4f}), lr {lr_at(step):.2e}, "
                      f"{(now - t_last) / (args.log_every if step else 1):.1f} s/step, peak {torch.cuda.max_memory_allocated(dev) / 2**30:.1f} GB")
            t_last = now
        if rank == 0 and ((step + 1) % args.save_every == 0 or step == args.steps - 1):
            path = args.out / f"{args.name}_{step + 1:06d}.safetensors"
            save_lora(lora, path, args.alpha)
            log(rank, f"saved {path}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("data", type=Path)
    c.add_argument("cache", type=Path)
    t = sub.add_parser("train")
    t.add_argument("cache", type=Path)
    t.add_argument("out", type=Path)
    t.add_argument("--name", required=True)
    t.add_argument("--steps", type=int, default=1000)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=0)
    t.add_argument("--cosine", action="store_true")
    t.add_argument("--weight-decay", type=float, default=1e-2)
    t.add_argument("--max-grad-norm", type=float, default=1.0)
    t.add_argument("--rank", type=int, default=32)
    t.add_argument("--alpha", type=float, default=None, help="default: the rank (scale 1, ComfyUI's default)")
    t.add_argument("--lora-dtype", choices=["bf16", "fp32"], default="bf16", help="v1 (ai-toolkit) trained bf16 LoRA weights")
    t.add_argument("--init-lora", type=Path, help="start from this LoRA (ComfyUI keys), e.g. v1 step 750")
    t.add_argument("--ara", type=Path, help="the accuracy recovery adapter (default: from the hub cache)")
    t.add_argument("--qtype", default="uint3")
    t.add_argument("--save-every", type=int, default=250)
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    cache(a) if a.cmd == "cache" else train(a)
