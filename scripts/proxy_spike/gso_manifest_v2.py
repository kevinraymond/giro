"""The view-LoRA v2 manifest (board #3710): Google Scanned Objects plus the Objaverse and Poly Haven
picks, one table for gso_render.py / gso_controls.py / gso_progressive.py / gso_dataset.py.

    python3 gso_manifest_v2.py OUT_CSV [--gso ~/ai/datasets/gso] [--objaverse ~/ai/datasets/objaverse-v2]
        [--polyhaven ~/ai/datasets/polyhaven] [--cull CULL_TXT] [--max-shoes 60] [--max-furniture 50]

Columns: name, source, category, group, license, author, url, path (the model: a GSO model directory,
a .glb or a .gltf), split (train, heldout, spare). Names: GSO's, ov_<uid> (Objaverse), ph_<id> (Poly
Haven). Splits: GSO keeps v1's held-out 30 (never trained on) and trains on the rest, shoes capped;
Objaverse holds out 20 vehicles and figures (tanks, scooters, motorcycles, cars, aircraft, trucks,
armor, figurines, a statue, a helmet: giro's kinds of subjects) and trains on the rest; Poly Haven
trains, furniture capped. GSO train is capped at --max-gso-train (v1's train objects first) and Poly
Haven at --max-polyhaven, so the GPU work (one Pixal3D run per new object) stays near 1,000 objects. --cull lists names to drop (scenes, ground planes, broken imports found in
the hero sheets). Also writes attribution.csv beside OUT_CSV (CC-BY needs it: name, author, url, license).
Deterministic (hash order).
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("--gso", type=Path, default=Path("~/ai/datasets/gso").expanduser())
ap.add_argument("--objaverse", type=Path, default=Path("~/ai/datasets/objaverse-v2").expanduser())
ap.add_argument("--polyhaven", type=Path, default=Path("~/ai/datasets/polyhaven").expanduser())
ap.add_argument("--glbs", type=Path, default=Path("~/.objaverse/hf-objaverse-v1/glbs").expanduser())
ap.add_argument("--cull", type=Path)
ap.add_argument("--max-shoes", type=int, default=60)
ap.add_argument("--max-furniture", type=int, default=50)
ap.add_argument("--max-gso-train", type=int, default=500, help="v1's 300 first (their Pixal3D meshes are reused)")
ap.add_argument("--max-polyhaven", type=int, default=200)
args = ap.parse_args()
LICENSES = {"by": "CC-BY-4.0", "cc0": "CC0-1.0"}
HELD_OUT = {"army_tank": 2, "motor_scooter": 2, "motorcycle": 2, "car_(automobile)": 1, "race_car": 1, "airplane": 1,
            "helicopter": 1, "pickup_truck": 1, "truck": 1, "armor": 4, "figurine": 2, "statue_(sculpture)": 1, "helmet": 1}
FURNITURE = {"furniture", "seating", "table", "shelves", "bed"}


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


cull = set(args.cull.read_text().split()) if args.cull and args.cull.exists() else set()
rows = []
shoes = 0
with open(args.gso / "manifest.csv") as f:
    gso = sorted(csv.DictReader(f), key=lambda r: (r["split"] != "train", h(r["name"])))
n_gso = 0
for r in gso:
    split = "heldout" if r["split"] == "heldout" else "train"
    if split == "train" and r["category"] == "Shoe":
        shoes += 1
        split = "train" if shoes <= args.max_shoes else "spare"
    if split == "train":
        n_gso += 1
        split = "train" if n_gso <= args.max_gso_train else "spare"
    rows.append({"name": r["name"], "source": "gso", "category": r["category"], "group": "", "license": "CC-BY-4.0",
                 "author": "Google LLC", "url": f"https://app.gazebosim.org/GoogleResearch/fuel/models/{r['name']}",
                 "path": str(args.gso / "models" / r["name"]), "split": split})

glbs = {p.stem: p for p in args.glbs.glob("*/*.glb")}
picked = sorted(json.loads((args.objaverse / "picked.json").read_text()), key=lambda k: h(k["uid"]))
check = json.loads((args.objaverse / "check.json").read_text()) if (args.objaverse / "check.json").exists() else {}
held = defaultdict(int)
for k in picked:
    if k["uid"] not in glbs or not check.get(k["uid"], {}).get("keep", True):  # check.py: appearance, franchise rips
        continue
    name = f"ov_{k['uid']}"
    split = "train"
    if name not in cull and held[k["category"]] < HELD_OUT.get(k["category"], 0):
        held[k["category"]] += 1
        split = "heldout"
    rows.append({"name": name, "source": "objaverse", "category": k["category"], "group": k["group"],
                 "license": LICENSES[k["license"]], "author": k["author"], "url": k["url"], "path": str(glbs[k["uid"]]),
                 "split": split})

ph = args.polyhaven / "picked.json"
furniture = n_ph = 0
for k in sorted(json.loads(ph.read_text()) if ph.exists() else [], key=lambda k: h(k["id"])):
    split = "train"
    if set(k["categories"]) & FURNITURE:
        furniture += 1
        split = "train" if furniture <= args.max_furniture else "spare"
    if split == "train":
        n_ph += 1
        split = "train" if n_ph <= args.max_polyhaven else "spare"
    rows.append({"name": f"ph_{k['id']}", "source": "polyhaven", "category": next((c for c in k["categories"] if not c.startswith("collection")), ""),
                 "group": "", "license": "CC0-1.0", "author": ", ".join(k["authors"]), "url": k["url"],
                 "path": str(args.polyhaven / k["gltf"]), "split": split})

for r in rows:
    if r["name"] in cull:
        r["split"] = "culled"
args.out.parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
with open(args.out.parent / "attribution.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "source", "author", "url", "license"])
    w.writeheader()
    w.writerows({k: r[k] for k in w.fieldnames} for r in rows if r["split"] in ("train", "heldout"))
count = defaultdict(int)
for r in rows:
    count[(r["source"], r["split"])] += 1
print(dict(sorted(count.items())))
