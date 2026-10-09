"""Retrain a finished attempt with anchor views added to its training set.

The new attempt OUT shares the source's frames, masks, poses and proxy (symlinks) and reruns
dataset -> export with the source's params; after the dataset stage, each anchor placed by
anchor_views.py (ANCHOR_DIR/mapped.json) is added COPIES times, with its own camera and its SAM
mask. Brush has no per-image weight, hence the copies (as the dataset stage does for the hero).

Brush sorts views by name and holds every 8th out for eval. The anchors are named kf/kf_NNN.png
so they sort after the frames and the hero (whose split stays as it was), and every slot that
would be held out gets a copy of frame 0 instead, so no anchor is.

    anchor_train.py SOURCE_ATTEMPT ANCHOR_DIR OUT GPU [COPIES=3] [SKIP=02,05]
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from giro.stages.fallback import _qvec
from giro.stages.masks import Masks, combine

ROOT = Path(__file__).resolve().parents[2]
src, adir, out = (Path(a).resolve() for a in sys.argv[1:4])
gpu = int(sys.argv[4])
copies = int(sys.argv[5]) if len(sys.argv) > 5 else 3
skip = set(sys.argv[6].split(",")) if len(sys.argv) > 6 else {"02", "05"}
REDONE = {"dataset", "train", "crop", "canonical", "export", "previews"}


def stage_params(*stages: str) -> list[str]:
    args = []
    for stage in stages:
        for k, v in json.loads((src / ".stages" / f"{stage}.json").read_text())["params"].items():
            args += ["-p", f"{stage}.{k}={json.dumps(v)}"]
    return args


def stages(first: str, last: str | None, *params: str) -> None:
    cmd = ["uv", "run", "giro", "stages", str(out), "--from", first, *(["--to", last] if last else []), "--gpu", str(gpu), *params]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


out.mkdir(parents=True)
for entry in src.iterdir():
    if entry.name in REDONE:
        continue
    if entry.name == ".stages":
        shutil.copytree(entry, out / ".stages")
        for stage in REDONE:
            (out / ".stages" / f"{stage}.json").unlink(missing_ok=True)
    elif entry.is_dir():
        (out / entry.name).symlink_to(entry)
    else:
        shutil.copy2(entry, out / entry.name)
metrics = json.loads((out / "metrics.json").read_text())
(out / "metrics.json").write_text(json.dumps({k: v for k, v in metrics.items() if k not in REDONE | {"canonicalize"}}, indent=1))

stages("dataset", "dataset", *stage_params("dataset"))

# Add the anchors to dataset/.
ds = out / "dataset"
sparse = ds / "sparse" / "0"
for f in ("cameras.txt", "images.txt"):
    text = (sparse / f).read_text()
    (sparse / f).unlink()
    (sparse / f).write_text(text)
mapped = json.loads((adir / "mapped.json").read_text())["anchors"]
anchors = [n for n in sorted(mapped) if n not in skip]
lines = [ln for ln in (sparse / "images.txt").read_text().splitlines() if not ln.startswith("#")]
headers = [h.split() for h in lines[0::2]]
n_views = len(headers)
next_image = max(int(h[0]) for h in headers) + 1
cam_lines = [ln for ln in (sparse / "cameras.txt").read_text().splitlines() if ln and not ln.startswith("#")]
next_cam = max(int(ln.split()[0]) for ln in cam_lines) + 1
first = next(h for h in headers if h[9] == "frames/00000.png")
split = json.loads((src / ".stages" / "dataset.json").read_text())["params"]["eval_split_every"]
erode = json.loads((src / ".stages" / "dataset.json").read_text())["params"]["mask_erode_px"]
(ds / "images" / "kf").mkdir()
(ds / "masks" / "kf").mkdir()

anchor_cam: dict[str, int] = {}
p = Masks.defaults
for name in anchors:
    a = mapped[name]
    anchor_cam[name] = next_cam
    cam_lines.append(f"{next_cam} SIMPLE_PINHOLE {a['width']} {a['height']} {a['focal']} {a['width'] / 2} {a['height'] / 2}")
    next_cam += 1
    raw = adir / "raw"
    subject = np.asarray(Image.open(raw / "subject" / "anchors" / f"{name}.png")) > 127
    bg = raw / "background" / "anchors" / f"{name}.png"
    mask = combine(subject, np.asarray(Image.open(bg)) > 127 if bg.exists() else None, p["touch_px"], p["gap_px"], p["max_add"])
    if erode:
        mask = ndimage.binary_erosion(mask, iterations=erode)
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(adir / f"mask_{name}.png")

queue = [name for name in anchors for _ in range(copies)]
added, k = [], 0
with open(sparse / "images.txt", "a") as f:
    while queue:
        kf = f"kf/kf_{k:03d}.png"
        if split and (n_views + k) % split == 0:  # held out: frame 0 again
            f.write(f"{next_image} {' '.join(first[1:9])} {kf}\n\n")
            (ds / "images" / kf).symlink_to((ds / "images" / "frames" / "00000.png").resolve())
            shutil.copy(ds / "masks" / "frames" / "00000.png", ds / "masks" / kf)
            added.append((kf, "eval: frame 0"))
        else:
            name = queue.pop(0)
            a = mapped[name]
            q = _qvec(np.asarray(a["rot"]))
            f.write(f"{next_image} {' '.join(f'{x:.12g}' for x in q)} {' '.join(f'{x:.12g}' for x in a['tvec'])} {anchor_cam[name]} {kf}\n\n")
            (ds / "images" / kf).symlink_to(adir / f"{name}.png")
            (ds / "masks" / kf).symlink_to(adir / f"mask_{name}.png")
            added.append((kf, f"anchor {name}"))
        next_image += 1
        k += 1
(sparse / "cameras.txt").write_text("\n".join(cam_lines) + "\n")
(out / "anchors.json").write_text(json.dumps({"anchor_dir": str(adir), "anchors": anchors, "copies": copies,
                                              "views": dict(added)}, indent=1))
print(f"added {sum(1 for _, w in added if w.startswith('anchor'))} anchor views ({len(anchors)} anchors x{copies}), "
      f"{sum(1 for _, w in added if w.startswith('eval'))} eval fillers", flush=True)

stages("train", None, *stage_params("train", "crop", "canonicalize", "export"))
