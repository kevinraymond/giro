"""What the UI shows about one attempt's frames and cameras.

- frames(): every extracted frame, which ones dedup kept, which got a camera,
  their azimuth around the orbit and their reprojection error.
- frame_sheet(): all extracted frames as one strip of small thumbnails.
- cameras(): the recovered cameras and sparse points in the viewer's frame. Once
  canonicalize has run, that is the export's frame (so cameras and splat line
  up); before, it is the orbit frame (ring axis up, orbit radius 1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from giro.stages import gapfill
from giro.stages.canonicalize import orbit_frame
from giro.stages.crop import load_views
from giro.stages.poses import read_images_txt
from giro.stages.train import EVAL_LINE, REFINE_LINE

THUMB_H = 96  # frame strip thumbnail height (px)
MAX_POINTS = 20_000


def _model(attempt: Path) -> Path:
    return attempt / "poses" / "colmap" / "model_txt"


def _image_errors(model: Path) -> dict[str, float]:
    """Mean reprojection error of the 3D points each image sees, by image name."""
    ids = {im["id"]: name for name, im in read_images_txt(model / "images.txt").items()}
    total: dict[int, float] = {}
    count: dict[int, int] = {}
    for line in (model / "points3D.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        err = float(f[7])
        for image_id in f[8::2]:
            i = int(image_id)
            total[i] = total.get(i, 0.0) + err
            count[i] = count.get(i, 0) + 1
    return {ids[i]: round(total[i] / count[i], 3) for i in total if i in ids}


def training(attempt: Path) -> dict[str, Any]:
    """Held-out PSNR and splat count by iteration, from Brush's log (works for CLI runs too)."""
    psnr, splats = [], []
    log = attempt / "train" / "brush.log"
    if log.exists():
        for line in log.read_text(errors="replace").splitlines():
            if m := EVAL_LINE.search(line):
                psnr.append({"step": int(m.group(1)), "value": float(m.group(2))})
            elif m := REFINE_LINE.search(line):
                splats.append({"step": int(m.group(1)), "value": int(m.group(2))})
    record = attempt / ".stages" / "train.json"
    params = json.loads(record.read_text())["params"] if record.exists() else {}
    checkpoints = sorted(p.name for p in (attempt / "train").glob("export_*.ply")) if (attempt / "train").is_dir() else []
    return {"psnr": psnr, "splats": splats, "total_iters": params.get("total_train_iters"), "checkpoints": checkpoints}


def _frame_files(attempt: Path) -> list[Path]:
    """Every frame the timeline shows, in order: the extracted ones and those a gap fill generated."""
    raw = list((attempt / "frames_raw").glob("*.png")) if (attempt / "frames_raw").is_dir() else []
    fill = gapfill.applied(attempt)
    generated = [attempt / n for g in fill["gaps"] for n in g["inserted"]] if fill else []
    return sorted(raw + generated, key=lambda p: p.name)


def frames(attempt: Path) -> dict[str, Any]:
    files = _frame_files(attempt)
    dedup = json.loads((attempt / "dedup.json").read_text()) if (attempt / "dedup.json").exists() else {}
    kept = set(dedup.get("kept", []))
    fill = gapfill.applied(attempt)
    generated = {Path(n).name for g in fill["gaps"] for n in g["inserted"]} if fill else set()
    replaced = {Path(n).name for g in fill["gaps"] for n in g["dropped"]} if fill else set()
    model = _model(attempt)
    registered: set[str] = set()
    errors: dict[str, float] = {}
    if (model / "images.txt").exists():
        registered = set(read_images_txt(model / "images.txt"))
        errors = _image_errors(model) if (model / "points3D.txt").exists() else {}
    azimuths: dict[str, float] = {}
    gate = attempt / "gate.json"
    if gate.exists():
        ring = json.loads(gate.read_text()).get("ring", {})
        azimuths = dict(zip(ring.get("frames", []), ring.get("azimuths", [])))
    out = []
    for path in files:
        name = path.name
        key = f"frames/{name}"
        out.append({
            "name": name,
            "kept": (name in kept or name in generated) and name not in replaced,
            "posed": key in registered,
            "azimuth": azimuths.get(key),
            "error": errors.get(key),
            "generated": name in generated,
            "replaced": name in replaced,
        })
    return {
        "frames": out,
        "static_start": dedup.get("static_start"),
        "static_end": dedup.get("static_end"),
        "hero": {"posed": "hero/hero.png" in registered, "error": errors.get("hero/hero.png")},
        "filled": gapfill.summary(fill) if fill else [],
        "thumb_height": THUMB_H,
    }


def frame_sheet(attempt: Path) -> Path | None:
    """One JPEG with every frame of the timeline side by side, made once (and again if the frames change)."""
    raw = _frame_files(attempt)
    if not raw:
        return None
    out = attempt / "ui" / "frames.jpg"
    newest = max(p.stat().st_mtime for p in raw)
    # The count too: a gap fill's frames can be older than the last sheet.
    count = out.with_suffix(".count")
    if out.exists() and out.stat().st_mtime >= newest and count.exists() and count.read_text() == str(len(raw)):
        return out
    with Image.open(raw[0]) as first:
        w = round(first.width * THUMB_H / first.height)
    sheet = Image.new("RGB", (w * len(raw), THUMB_H))
    for i, path in enumerate(raw):
        with Image.open(path) as im:
            factor = max(1, im.height // (THUMB_H * 2))
            small = im.convert("RGB").reduce(factor).resize((w, THUMB_H), Image.Resampling.BILINEAR)
        sheet.paste(small, (i * w, 0))
    out.parent.mkdir(exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    sheet.save(tmp, quality=80)
    tmp.replace(out)
    count.write_text(str(len(raw)))
    return out


def cameras(attempt: Path) -> dict[str, Any] | None:
    model = _model(attempt)
    if not (model / "images.txt").exists():
        return None
    views = load_views(model)
    transform = attempt / "canonical" / "transform.json"
    if transform.exists():
        t = json.loads(transform.read_text())
        center, rot, scale, offset = np.array(t["center"]), np.array(t["R"]), t["scale"], np.array(t["offset"])
        frame = "canonical"
    else:
        center, rot = orbit_frame(views)
        radius = np.mean([np.linalg.norm((v.center - center) - ((v.center - center) @ rot[1]) * rot[1])
                          for v in views if v.name.startswith("frames/")])
        scale, offset = 1.0 / float(radius), np.zeros(3)
        frame = "orbit"

    def to_viewer(p: np.ndarray) -> np.ndarray:
        return scale * (p - center) @ rot.T + offset

    errors = _image_errors(model) if (model / "points3D.txt").exists() else {}
    cams = []
    for v in views:
        axes = rot @ v.rot.T  # columns: the camera's right, down, forward in the viewer frame
        cams.append({
            "name": v.name,
            "hero": v.name.startswith("hero/"),
            "position": [round(float(x), 4) for x in to_viewer(v.center)],
            "right": [round(float(x), 4) for x in axes[:, 0]],
            "down": [round(float(x), 4) for x in axes[:, 1]],
            "forward": [round(float(x), 4) for x in axes[:, 2]],
            "fx": v.fx, "fy": v.fy, "width": v.width, "height": v.height,
            "error": errors.get(v.name),
        })

    points, colors = [], []
    path = model / "points3D.txt"
    if path.exists():
        rows = [l.split()[1:7] for l in path.read_text().splitlines() if l and not l.startswith("#")]
        if rows:
            data = np.array(rows, dtype=float)
            if len(data) > MAX_POINTS:
                data = data[np.random.default_rng(0).choice(len(data), MAX_POINTS, replace=False)]
            points = np.round(to_viewer(data[:, :3]), 4).ravel().tolist()
            colors = data[:, 3:6].astype(int).ravel().tolist()
    # COLMAP world -> viewer, for splats still in the world frame (training checkpoints).
    world_to_viewer = np.eye(4)
    world_to_viewer[:3, :3] = scale * rot
    world_to_viewer[:3, 3] = offset - scale * rot @ center
    return {"frame": frame, "cameras": cams, "points": points, "colors": colors,
            "world_to_viewer": np.round(world_to_viewer, 6).tolist()}
