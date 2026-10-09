"""ai-toolkit configs for the view-LoRA v2 (board #3713), written after gso_dataset.py so the sample
prompts name real held-out pairs.

    python3 gso_v2_config.py DATASET_DIR MANIFEST OUT_DIR [--steps 4000] [--batch 4] [--rank 32] [--res 1024]
        [--name gso_view_lora_v2] [--training-folder DIR]

Writes OUT_DIR/v2.yaml (images 1-3: progressive proxy render, hero, nearest earlier view) and
OUT_DIR/v2-2img.yaml (the ablation without image 3), the same otherwise: bf16, no quantization (the
RTX Pro 6000 has 96 GB), resolution buckets around --res, AdamW8bit 1e-4, 100 warmup steps then cosine
to 1e-5 (needs data/gso-pilot/v2/ai-toolkit-warmup.patch), save and sample every 250 steps on six
held-out pairs: two Objaverse vehicles, one Objaverse figure, three GSO, eye level and high, k > 0
where possible.
"""
import argparse
import csv
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dataset", type=Path)
ap.add_argument("manifest", type=Path, help="gso_manifest_v2.py's (for each object's source and group)")
ap.add_argument("out", type=Path)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--name", default="gso_view_lora_v2")
ap.add_argument("--training-folder", type=Path)
args = ap.parse_args()
ds = args.dataset.resolve()
with open(ds / "heldout" / "pairs.csv") as f:
    pairs = [r for r in csv.DictReader(f) if r["kept"] == "1"]
with open(args.manifest) as f:
    meta = {r["name"]: r for r in csv.DictReader(f)}


def pick(source: str, group: str, n: int, high: bool) -> list[dict]:
    """n pairs from different objects: k > 0 first, then eye level or high as asked."""
    rows = [r for r in pairs if meta[r["object"]]["source"] == source and (not group or meta[r["object"]]["group"] == group)]
    rows.sort(key=lambda r: (int(r["k"] or 0) == 0, (float(r["pitch"]) > 30) != high, r["id"]))
    out, seen = [], set()
    for r in rows:
        if r["object"] not in seen and len(out) < n:
            out.append(r)
            seen.add(r["object"])
    return out


samples = (pick("objaverse", "vehicle", 1, False) + pick("objaverse", "vehicle", 2, True)[1:] + pick("objaverse", "figure", 1, False)
           + pick("gso", "", 2, False) + pick("gso", "", 3, True)[2:])


def config(three: bool) -> dict:
    name = args.name + ("" if three else "_2img")
    ctrl = ["control1", "control2"] + (["control3"] if three else [])
    sample_rows = []
    for r in samples:
        s = {"prompt": (ds / "heldout" / "target" / f"{r['id']}.txt").read_text().strip()}
        for i, c in enumerate(ctrl, 1):
            s[f"ctrl_img_{i}"] = str(ds / "heldout" / c / f"{r['id']}.png")
        sample_rows.append(s)
    return {"job": "extension", "config": {"name": name, "process": [{
        "type": "diffusion_trainer",
        "training_folder": str(args.training_folder or args.out / "output"),
        "device": "cuda:0",
        "network": {"type": "lora", "linear": args.rank, "linear_alpha": args.rank},
        "save": {"dtype": "float16", "save_every": 250, "max_step_saves_to_keep": 20},
        "datasets": [{"folder_path": str(ds / "train" / "target"),
                      "control_path": [str(ds / "train" / c) for c in ctrl],
                      "caption_ext": "txt", "caption_dropout_rate": 0.0, "resolution": [args.res],
                      "cache_latents_to_disk": True}],
        "train": {"batch_size": args.batch, "cache_text_embeddings": True, "steps": args.steps,
                  "gradient_accumulation": 1, "timestep_type": "weighted", "train_unet": True,
                  "train_text_encoder": False, "gradient_checkpointing": True, "noise_scheduler": "flowmatch",
                  "optimizer": "adamw8bit", "lr": 1e-4, "lr_scheduler": "cosine",
                  "lr_scheduler_params": {"num_warmup_steps": 100, "eta_min": 1e-5},
                  "dtype": "bf16", "skip_first_sample": True},
        "model": {"name_or_path": "Qwen/Qwen-Image-Edit-2511", "arch": "qwen_image_edit_plus",
                  "quantize": False, "quantize_te": False, "low_vram": False},
        "sample": {"sampler": "flowmatch", "sample_every": 250, "width": 768, "height": 1024,
                   "samples": sample_rows, "neg": "", "seed": 42, "guidance_scale": 3, "sample_steps": 25},
    }]}, "meta": {"name": "[name]", "version": "2.0"}}


args.out.mkdir(parents=True, exist_ok=True)
for three, fname in ((True, "v2.yaml"), (False, "v2-2img.yaml")):
    # JSON is YAML, except that PyYAML reads 1e-05 as a string (YAML 1.1 wants 1.0e-05)
    text = re.sub(r"(?<![\w.])(\d+)e([-+]\d+)", r"\1.0e\2", json.dumps(config(three), indent=2))
    (args.out / fname).write_text(f"# GSO view-LoRA v2 (board #3709), written by gso_v2_config.py\n{text}\n")
print("samples:", [r["id"] for r in samples])
