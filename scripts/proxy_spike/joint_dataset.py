"""Joint multi-view LoRA pilot data (Route A', board #3735): four neighboring views of an object as one 2x2
grid, so Qwen-Image-Edit 2511's joint attention learns to make them agree. Built from the view-LoRA pilot's
single-view data (gso_dataset.py's layout), one sample per group of four views:

    joint_dataset.py SRC_DIR OUT_DIR [--groupings 2] [--panel 384x512] [--seed 0]

SRC_DIR/<split>/{pairs.csv, target/, control1/, control2/} (gso_dataset.py) -> OUT_DIR/<split>/{target,
control1, control2}/<object>__g<k>.png, target/<id>.txt (the caption) and groups.csv. target = the four true
views as a 2x2 grid of --panel panels (768x1024 for 384x512: v1's training size, so a 4090 trains it at v1's
cost), control1 = the four proxy renders the same way, control2 = the hero (image 2, unchanged). Groups: per
object and grouping, a random kept view plus its three nearest unused kept views by camera direction (views
that overlap, so agreement is learnable); --groupings shuffles that many times. Panels are ordered by yaw.
The caption names each panel's camera as the single-view caption does (gso_dataset.caption's convention).
"""
import argparse
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

PROMPT = ("Render each panel of image 1 as a real photo of the object in image 2, from that panel's viewpoint; "
          "the four panels are views of the same object and agree with each other")
POS = ["top-left", "top-right", "bottom-left", "bottom-right"]


def caption(cams: list[tuple[float, float]]) -> str:
    return PROMPT + ": " + "; ".join(f"{p} yaw {round(y) % 360}°, pitch {round(q)}°" for p, (y, q) in zip(POS, cams))


def direction(yaw: float, pitch: float) -> np.ndarray:
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])


def groups_of_four(views: list[dict], rng: random.Random) -> list[list[dict]]:
    """Greedy neighbor groups: a random unused view and the three unused views nearest in direction; leftovers
    are topped up with the nearest views overall."""
    unused = list(views)
    rng.shuffle(unused)
    out = []
    while unused:
        seed = unused.pop(0)
        d = direction(seed["yaw"], seed["pitch"])
        pool = sorted(unused, key=lambda v: -float(direction(v["yaw"], v["pitch"]) @ d))
        group = [seed] + pool[:3]
        for v in pool[:3]:
            unused.remove(v)
        if len(group) < 4:
            rest = sorted((v for v in views if v not in group), key=lambda v: -float(direction(v["yaw"], v["pitch"]) @ d))
            group += rest[: 4 - len(group)]
        out.append(sorted(group, key=lambda v: v["yaw"] % 360))
    return out


def grid(paths: list[Path], pw: int, ph: int) -> Image.Image:
    g = Image.new("RGB", (2 * pw, 2 * ph))
    for i, p in enumerate(paths):
        g.paste(Image.open(p).convert("RGB").resize((pw, ph), Image.LANCZOS), ((i % 2) * pw, (i // 2) * ph))
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--groupings", type=int, default=2)
    ap.add_argument("--panel", default="384x512")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pw, ph = map(int, args.panel.split("x"))
    for split in ("train", "heldout"):
        sdir, odir = args.src / split, args.out / split
        if not (sdir / "pairs.csv").exists():
            continue
        for d in ("target", "control1", "control2"):
            (odir / d).mkdir(parents=True, exist_ok=True)
        objects: dict[str, list[dict]] = {}
        for r in csv.DictReader(open(sdir / "pairs.csv")):
            if r["kept"] == "1":
                objects.setdefault(r["object"], []).append({"id": r["id"], "yaw": float(r["rel_yaw"]), "pitch": float(r["pitch"])})
        rng = random.Random(args.seed)
        rows = []
        for obj, views in sorted(objects.items()):
            if len(views) < 4:
                continue
            k = 0
            for _ in range(args.groupings):
                for g in groups_of_four(views, rng):
                    gid = f"{obj}__g{k}"
                    k += 1
                    grid([sdir / "target" / f"{v['id']}.png" for v in g], pw, ph).save(odir / "target" / f"{gid}.png")
                    grid([sdir / "control1" / f"{v['id']}.png" for v in g], pw, ph).save(odir / "control1" / f"{gid}.png")
                    hero = odir / "control2" / f"{gid}.png"
                    if not hero.exists():
                        os.symlink(os.path.realpath(sdir / "control2" / f"{g[0]['id']}.png"), hero)
                    (odir / "target" / f"{gid}.txt").write_text(caption([(v["yaw"], v["pitch"]) for v in g]))
                    rows.append({"id": gid, "object": obj, "views": " ".join(v["id"].split("__")[1] for v in g)})
        with open(odir / "groups.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "object", "views"])
            w.writeheader()
            w.writerows(rows)
        print(f"{split}: {len(objects)} objects, {len(rows)} grids", flush=True)
