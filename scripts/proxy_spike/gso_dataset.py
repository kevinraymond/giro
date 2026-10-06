"""Lay out the view-LoRA pilot's training data for ai-toolkit (board #3699) from gso_render.py's
heroes, gso_controls.py's control renders and gso_render.py --aligned's targets.

    gso_dataset.py RENDERS_DIR CONTROLS_DIR MANIFEST OUT_DIR [--min-iou 0.6]

ai-toolkit pairs files by name across folders: OUT_DIR/<split>/target/<id>.png (+ <id>.txt, the
caption), control1/<id>.png (image 1: the proxy render at the target camera, the texture route's
look, on black) and control2/<id>.png (image 2: the hero on its gray backdrop, a symlink). The target
is the ground truth at that camera, on black like the renders. id = <object>__<view>.

Caption: the fixed instruction plus the camera as the route knows it at inference (gso_controls.py):
yaw relative to the hero, in [0, 360), giro.path's sign (+ moves the camera to the hero camera's
right), and the camera's pitch in the proxy frame (+ looks down from above). Views whose control
silhouette and target alpha disagree (IoU below --min-iou: the proxy got the shape wrong there) are
left out. Writes OUT_DIR/<split>/pairs.csv.

v2 (board #3709): --control1 prog takes image 1 from gso_progressive.py (the proxy after k earlier
views were painted, as the route shows it mid-way); --control3 adds image 3, the nearest of those
earlier views' ground truth on black (a black frame when k = 0), as control3/<id>.png.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

INSTRUCTION = "Render image 1 as a real photo of the object in image 2, from image 1's viewpoint"


def caption(rel_yaw: float, pitch: float) -> str:
    """The prompt for a target camera; the same wording at inference."""
    return f"{INSTRUCTION}: yaw {round(rel_yaw) % 360}°, pitch {round(pitch)}°"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("renders", type=Path)
    ap.add_argument("controls", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--min-iou", type=float, default=0.6)
    ap.add_argument("--control1", choices=["control", "prog"], default="control", help="image 1: hero-painted only, or progressive")
    ap.add_argument("--control3", action="store_true", help="image 3: the nearest view painted before (gso_progressive.py)")
    args = ap.parse_args()
    with open(args.manifest) as f:
        split_of = {r["name"]: r["split"] for r in csv.DictReader(f)}
    rows: dict[str, list[dict]] = {"train": [], "heldout": []}
    subs = ("target", "control1", "control2") + (("control3",) if args.control3 else ())
    use_prog = args.control1 == "prog" or args.control3

    def on_black(src: Path, dst: Path) -> None:
        im = Image.open(src)
        bg = Image.new("RGB", im.size, (0, 0, 0))
        bg.paste(im, mask=im.split()[3])
        bg.save(dst)

    for od in sorted(p for p in args.controls.iterdir() if (p / "target" / "done").exists()
                       and (not use_prog or (p / "prog" / "prog.json").exists())):
        split = split_of.get(od.name)
        if split not in rows:
            continue
        dd = args.out / split
        for sub in subs:
            (dd / sub).mkdir(parents=True, exist_ok=True)
        views = json.loads((od / "gt_cameras.json").read_text())["views"]
        prog = json.loads((od / "prog" / "prog.json").read_text())["targets"] if use_prog else {}
        for v in views:
            cm = np.asarray(Image.open(od / "control" / f"{v['name']}_mask.png")) > 127
            t = Image.open(od / "target" / f"{v['name']}.png")
            ta = np.asarray(t)[..., 3] > 127
            fit = float((cm & ta).sum() / max(1, (cm | ta).sum()))
            sid = f"{od.name}__{v['name']}"
            keep = fit >= args.min_iou
            pg = prog.get(v["name"], {})
            rows[split].append({"id": sid, "object": od.name, "view": v["name"], "rel_yaw": v["rel_yaw"],
                                "pitch": v["pitch"], "iou": round(fit, 3), "kept": int(keep), "k": pg.get("k", 0),
                                "neighbor": pg.get("neighbor") or ""})
            if not keep:
                continue
            on_black(od / "target" / f"{v['name']}.png", dd / "target" / f"{sid}.png")
            (dd / "target" / f"{sid}.txt").write_text(caption(v["rel_yaw"], v["pitch"]) + "\n")
            if args.control3:
                c3 = dd / "control3" / f"{sid}.png"
                if pg.get("neighbor"):
                    on_black(od / "target" / f"{pg['neighbor']}.png", c3)
                else:
                    Image.new("RGB", t.size, (0, 0, 0)).save(c3)
            c1 = od / args.control1 / f"{v['name']}.png"
            for sub, src in (("control1", c1), ("control2", args.renders / od.name / "hero.png")):
                link = dd / sub / f"{sid}.png"
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(src.resolve(), link)
    for split, rs in rows.items():
        if not rs:
            continue
        with open(args.out / split / "pairs.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
        kept = sum(r["kept"] for r in rs)
        print(f"{split}: {len({r['object'] for r in rs})} objects, {kept}/{len(rs)} pairs kept "
              f"(IoU >= {args.min_iou}; median IoU {np.median([r['iou'] for r in rs]):.2f})")
