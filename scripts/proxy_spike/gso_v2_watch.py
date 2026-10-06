"""Per-checkpoint evaluation for the view-LoRA v2 (board #3713): waits for ai-toolkit to save each
checkpoint, links it where ComfyUI finds LoRAs, and scores it with gso_eval.py (gen, then score), so
the run is judged at every checkpoint instead of only at the last step (v1's step 1000 looked worse
than 500/750 in the trainer's samples).

    gso_v2_watch.py RUN_DIR EVAL_DIR GPU [--name gso_view_lora_v2] [--loras ~/models/loras/qwen] [--poll 120]
        [--eval-args "..."]

RUN_DIR is the trainer's output folder for the run (RUN_DIR/<name>_000000250.safetensors, ...).
Each checkpoint becomes qwen/gso-view-v2-<step>.safetensors and is passed to gso_eval.py as variant
v2-<step>; --eval-args go to both gso_eval.py calls: v2's dataset/controls/renders, --size, --gen-size,
--lora-template 'qwen/gso-view-{}.safetensors' and --image3 (data/gso-pilot/v2/SETUP.md has the line).
"""
import argparse
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("eval", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--name", default="gso_view_lora_v2")
ap.add_argument("--loras", type=Path, default=Path("~/models/loras/qwen").expanduser())
ap.add_argument("--poll", type=int, default=120)
ap.add_argument("--eval-args", default="")
args = ap.parse_args()
done: set[int] = set()
while True:
    for ck in sorted(args.run.glob(f"{args.name}_*.safetensors")):
        m = re.search(r"_(\d{9})\.safetensors$", ck.name)
        if not m or int(m.group(1)) in done or time.time() - ck.stat().st_mtime < 60:
            continue
        step = int(m.group(1))
        link = args.loras / f"gso-view-v2-{step}.safetensors"
        if not link.exists():
            link.symlink_to(ck.resolve())
        extra = args.eval_args.split()
        for cmd in (["gen", str(args.eval), str(args.gpu)], ["score", str(args.eval)]):
            subprocess.run(["uv", "run", "python", "gso_eval.py", *cmd, "--variants", f"v2-{step}", *extra], cwd=HERE, check=False)
        done.add(step)
        print(f"step {step} scored", flush=True)
    if (args.run / f"{args.name}.safetensors").exists() and not any(
            int(re.search(r"_(\d{9})", c.name).group(1)) not in done for c in args.run.glob(f"{args.name}_*.safetensors")):
        break  # the final save is there and every step checkpoint is scored
    time.sleep(args.poll)
