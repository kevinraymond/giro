"""Manifest and a fixed split of Google Scanned Objects for the view-LoRA pilot (board #3699).

    uv run --project ~/ai/datasets/gso/render python gso_manifest.py GSO_DIR [--train 300] [--heldout 30]

Reads GSO_DIR/fuel_list.json (the Gazebo Fuel listing) and GSO_DIR/models/<name>/, writes
GSO_DIR/manifest.csv: name, category, license, bbox (x, y, z in the OBJ's z-up meters), triangles,
texture size, and split (train, heldout or spare). Held out first: vehicles and figures (the kinds
of subjects giro sees), then a spread over the other categories; train: a spread over categories
with shoes capped (a quarter of GSO is shoes). Deterministic (seeded by name).
"""
import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("gso", type=Path)
ap.add_argument("--train", type=int, default=300)
ap.add_argument("--heldout", type=int, default=30)
ap.add_argument("--max-shoes", type=int, default=30, help="shoes in the train split")
args = ap.parse_args()
FIGURE_LIKE = re.compile(r"car|truck|vehicle|bus|train|plane|tractor|bulldozer|robot|figure|dino|horse|"
                         r"elephant|animal|dog|cat|bird|shark|turtle|knight|soldier|doll|mario|batman|lego", re.I)

listing = {m["name"]: m for m in json.loads((args.gso / "fuel_list.json").read_text())}
rows = []
for name, m in sorted(listing.items()):
    d = args.gso / "models" / name
    obj = d / "meshes" / "model.obj"
    if not obj.exists():
        continue
    v, n_tri = [], 0
    with open(obj) as f:
        for line in f:
            if line.startswith("v "):
                v.append(line.split()[1:4])
            elif line.startswith("f "):
                n_tri += len(line.split()) - 3
    v = np.asarray(v, dtype=np.float32)
    ext = v.max(0) - v.min(0)
    tex = d / "materials" / "textures" / "texture.png"
    tw, th = Image.open(tex).size if tex.exists() else (0, 0)
    cat = (m.get("categories") or ["none"])[0]
    rows.append({"name": name, "category": cat, "license": m["license_name"], "bbox_x": f"{ext[0]:.4f}",
                 "bbox_y": f"{ext[1]:.4f}", "bbox_z": f"{ext[2]:.4f}", "triangles": n_tri, "texture": f"{tw}x{th}",
                 "description": m.get("description", "").split("\n")[0][:80], "split": "spare"})


def key(r: dict) -> str:
    return hashlib.sha1(r["name"].encode()).hexdigest()


def spread(pool: list[dict], n: int, cap: dict[str, int] | None = None) -> list[dict]:
    """n rows round-robin over categories (each category's rows in hash order), honoring caps."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in sorted(pool, key=key):
        by_cat[r["category"]].append(r)
    taken: list[dict] = []
    used: dict[str, int] = defaultdict(int)
    while len(taken) < n and any(by_cat.values()):
        for cat in sorted(by_cat):
            if by_cat[cat] and len(taken) < n and used[cat] < (cap or {}).get(cat, 10**9):
                taken.append(by_cat[cat].pop(0))
                used[cat] += 1
        if all(not rs or used[c] >= (cap or {}).get(c, 10**9) for c, rs in by_cat.items()):
            break
    return taken


usable = [r for r in rows if r["texture"] != "0x0"]
figure = [r for r in usable if FIGURE_LIKE.search(r["name"] + " " + r["description"]) or r["category"] == "Action Figures"]
held = spread(figure, args.heldout // 2)
held += spread([r for r in usable if r not in held], args.heldout - len(held), cap={"Shoe": 3})
for r in held:
    r["split"] = "heldout"
train = spread([r for r in usable if r["split"] == "spare"], args.train, cap={"Shoe": args.max_shoes})
for r in train:
    r["split"] = "train"
with open(args.gso / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
count = defaultdict(lambda: defaultdict(int))
for r in rows:
    count[r["split"]][r["category"]] += 1
print(f"{len(rows)} models, {len(usable)} textured")
for s in ("train", "heldout", "spare"):
    print(s, sum(count[s].values()), dict(sorted(count[s].items(), key=lambda kv: -kv[1])))
print("held out:", ", ".join(r["name"] for r in held))
