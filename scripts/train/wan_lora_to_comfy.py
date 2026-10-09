"""A DiffSynth-Studio Wan LoRA (PEFT keys: blocks.N.self_attn.q.lora_A.default.weight, alpha = rank) in the
key format ComfyUI's LoraLoaderModelOnly maps onto its Wan model (diffusion_model.blocks.N.self_attn.q.lora_A.weight),
bf16, so giro's orbit stage can take it as "lora_low" (a file under ComfyUI's loras folder).

    wan_lora_to_comfy.py IN.safetensors OUT.safetensors
"""
import sys

import torch
from safetensors.torch import load_file, save_file

src, dst = sys.argv[1:3]
sd = load_file(src)
out = {"diffusion_model." + k.replace(".default.", "."): v.to(torch.bfloat16) for k, v in sd.items()}
save_file(out, dst, metadata={"source": src, "format": "comfyui wan lora (from DiffSynth-Studio)"})
print(f"{len(out)} tensors -> {dst}")
