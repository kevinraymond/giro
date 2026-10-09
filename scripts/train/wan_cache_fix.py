"""Post-process a DiffSynth-Studio split-training cache (wan_orbit_lora.sh cache) of Wan 2.2 Fun Control clips so
sft:train sees what giro's inference gives the model (GiroWanFunControlToVideo = ComfyUI's Wan22FunControlToVideo
with start_image), in place:

- y = [control latents (16), mask (4), start-image latents (16)]: DiffSynth leaves the last 20 channels at zero
  (Fun Control without a start image); giro keeps the hero as frame 0: mask 1 at latent frame 0 (ComfyUI inverts
  its concat_mask to this), and the hero's latent there. Wan's VAE is causal, so latent frame 0 of the clip IS the
  hero encoded alone (input_latents[:, :, 0]) and no VAE is needed.
- use_gradient_checkpointing(_offload) on: sft:train takes them from the cache, not the command line.
- the PIL frames dropped (input_video, control_video: ~215 MB of a ~280 MB sample); nothing after the cache reads them.

- the timestep range of the expert being trained (--expert): sft:train takes it from the cache too. low: t ~0-900
  (min 0.358, max 1 of DiffSynth's index range), high: t ~900-1000 (min 0, max 0.358).

    .venv/bin/python wan_cache_fix.py [--expert low|high] CACHE_DIR [CACHE_DIR ...]
"""
import argparse
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--expert", choices=["low", "high"], default="low")
ap.add_argument("roots", nargs="+")
args = ap.parse_args()
lo, hi = (0.358, 1.0) if args.expert == "low" else (0.0, 0.358)
for root in args.roots:
    n = 0
    for p in sorted(Path(root).rglob("*.pth")):
        shared, posi, nega = torch.load(p, map_location="cpu", weights_only=False)
        y, lat = shared["y"], shared["input_latents"]
        assert y.shape[1] == 36 and lat.shape[1] == 16, (p, y.shape, lat.shape)
        y[:, 16:20] = 0.0
        y[:, 16:20, 0] = 1.0
        y[:, 20:36] = 0.0
        y[:, 20:36, 0] = lat[:, :, 0].to(y.dtype)
        shared["use_gradient_checkpointing"] = True
        shared["use_gradient_checkpointing_offload"] = True
        shared["min_timestep_boundary"], shared["max_timestep_boundary"] = lo, hi
        for k in ("input_video", "control_video"):
            shared.pop(k, None)
        torch.save((shared, posi, nega) if isinstance(posi, dict) else [shared, posi, nega], p)
        n += 1
    print(f"{root}: {n} samples fixed ({args.expert} expert)", flush=True)
