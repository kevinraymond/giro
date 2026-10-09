"""Score attempts' splats with one evaluator, so orbit modes can be compared (#3600).

Brush's eval PSNR scores held-out frames of the generated video: it says how well the video agrees
with itself, not how close the splat is to the hero or how sharp it is (a blurrier video can score
higher). This adds, per attempt:

- hero view: the cropped splat rendered from the hero's camera against the hero, inside the hero
  mask: PSNR and LPIPS (the only real ground truth; the hero is also a training view, so this
  measures fidelity, not generalization)
- held-out PSNR: the held-out frames (Brush's split), cropped splat, for continuity with Brush's number
- sharpness: mean absolute Laplacian inside the subject, on views of the canonical splat at a fixed
  framing, relative to the hero at the same subject height (1.0 = as sharp as the hero)
- likeness: DINOv2 cosine similarity of each canonical view to the masked hero (views around the
  ring, from above and from below)
- a review sheet of the canonical splat with the hero view in the corner

    uv run --group eval scripts/evaluate.py OUT_DIR ATTEMPT [ATTEMPT ...] [--gpu 1]

Writes OUT_DIR/<label>.json, OUT_DIR/<label>-sheet.jpg and OUT_DIR/summary.tsv; an attempt is
labeled <parent>/<name> (a job's attempts/<seed>: <job>/<seed>).

Caveats: hero PSNR counts subject pixels only, so it reads lower than Brush's whole-frame PSNR;
the Laplacian also rewards noise, so a noisy top view can score "sharp" (read it with the sheet);
likeness from above and below is to a front view by nature, so compare it between modes only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from giro import render
from giro.stages.crop import View, load_views

RING_ELEVATIONS = ((10.0, 8), (55.0, 4), (-30.0, 4))  # (degrees, views): as the canonicalize sheet
VIEW_HEIGHT = 512  # px (--view-height); the subject's height at the canonical framing is about 0.75 of it
HERO_LONG_SIDE = 1024


def masked_psnr(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    mse = float(((a - b) ** 2)[m].mean())
    return float(10 * np.log10(1 / max(mse, 1e-10)))


def laplacian_energy(rgb: np.ndarray, m: np.ndarray) -> float:
    """Mean |Laplacian| of luminance inside the eroded subject (the silhouette edge excluded)."""
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    inner = ndimage.binary_erosion(m, iterations=3)
    return float(np.abs(ndimage.laplace(lum))[inner].mean()) if inner.any() else 0.0


def subject_box(m: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def at_height(img: Image.Image, mask: np.ndarray, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the subject and scale it to `height` px tall (so sharpness compares like with like)."""
    x0, y0, x1, y1 = subject_box(mask)
    w = max(1, round((x1 - x0) * height / (y1 - y0)))
    crop = img.crop((x0, y0, x1, y1)).resize((w, height), Image.Resampling.LANCZOS)
    mcrop = Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255).resize((w, height), Image.Resampling.BILINEAR)
    return np.asarray(crop, dtype=np.float32) / 255, np.asarray(mcrop) > 127


def render_alpha(ply: Path, cams: list[render.Camera], size: tuple[int, int], gpu: int | None
                 ) -> list[tuple[Image.Image, np.ndarray]]:
    """Renders on black, with coverage from a second render on white."""
    black = render.render(ply, cams, size, background=(0.0, 0.0, 0.0), gpu=gpu)
    white = render.render(ply, cams, size, background=(1.0, 1.0, 1.0), gpu=gpu)
    out = []
    for b, w in zip(black, white):
        alpha = 1 - (np.asarray(w, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean(axis=2) / 255
        out.append((b, alpha > 0.5))
    return out


class Models:
    """LPIPS and DINOv2, loaded once."""

    def __init__(self) -> None:
        import lpips
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(self.dev).eval()
        self.dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.dino = AutoModel.from_pretrained("facebook/dinov2-base").to(self.dev).eval()

    def lpips_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        t = lambda x: self.torch.from_numpy(x).permute(2, 0, 1)[None].to(self.dev) * 2 - 1  # noqa: E731
        with self.torch.no_grad():
            return float(self.lpips(t(a), t(b)).item())

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.dino_proc(images=images, return_tensors="pt").to(self.dev)
            cls = self.dino(**inputs).last_hidden_state[:, 0]
            return self.torch.nn.functional.normalize(cls, dim=1).cpu().numpy()


def on_gray(rgb: np.ndarray, m: np.ndarray) -> Image.Image:
    return Image.fromarray((np.where(m[..., None], rgb, 0.5) * 255).astype(np.uint8))


def evaluate(attempt: Path, models: Models, gpu: int | None, view_height: int = VIEW_HEIGHT) -> tuple[dict, Image.Image]:
    model_txt = attempt / "poses" / "colmap" / "model_txt"
    views = {v.name: v for v in load_views(model_txt)}
    cropped = attempt / "crop" / "cropped.ply"
    result: dict = {"attempt": str(attempt)}

    hero_rgb = Image.open(attempt / "hero" / "hero.png").convert("RGB")
    hero_mask = np.asarray(Image.open(attempt / "masks" / "hero" / "hero.png").convert("L").resize(hero_rgb.size)) > 127
    hero_at, hero_m_at = at_height(hero_rgb, hero_mask, round(view_height * 0.75))
    result["hero_sharpness"] = round(laplacian_energy(hero_at, hero_m_at), 5)
    hero_embed = models.embed([on_gray(hero_at, hero_m_at)])[0]

    # Hero view: render at the hero camera, compare at a common size inside the hero mask.
    hero_view: View | None = views.get("hero/hero.png")
    hero_tile = None
    if hero_view is not None:
        s = HERO_LONG_SIDE / max(hero_view.width, hero_view.height)
        size = (round(hero_view.width * s), round(hero_view.height * s))
        (img, _), = render_alpha(cropped, [hero_view.camera()], size, gpu)
        r = np.asarray(img, dtype=np.float32) / 255
        gt = np.asarray(hero_rgb.resize(size, Image.Resampling.LANCZOS), dtype=np.float32) / 255
        m = np.asarray(Image.fromarray(hero_mask.astype(np.uint8) * 255).resize(size)) > 127
        gt_masked = gt * m[..., None]
        result["hero_psnr"] = round(masked_psnr(r, gt_masked, m), 3)
        result["hero_lpips"] = round(models.lpips_dist(r * m[..., None], gt_masked), 4)
        hero_tile = Image.fromarray((np.concatenate([gt_masked, r], axis=1) * 255).astype(np.uint8))
    else:
        result["hero_psnr"] = result["hero_lpips"] = None

    # Held-out frames: Brush's split (every Nth of the sorted dataset images, from index 0).
    split = json.loads((attempt / ".stages" / "dataset.json").read_text())["params"].get("eval_split_every", 8)
    names = sorted(str(p.relative_to(attempt / "dataset" / "images")) for p in (attempt / "dataset" / "images").rglob("*.png"))
    held = [n for n in names[::split] if n in views] if split else []
    vals = []
    for name in held:
        v = views[name]
        (img, _), = render_alpha(cropped, [v.camera()], (v.width, v.height), gpu)
        gt = np.asarray(Image.open(attempt / "dataset" / "images" / name).convert("RGB"), dtype=np.float32) / 255
        m = np.asarray(Image.open(attempt / "dataset" / "masks" / name).convert("L")) > 127
        vals.append(masked_psnr(np.asarray(img, dtype=np.float32) / 255, gt * m[..., None], np.ones_like(m)))
    result["heldout_psnr"] = round(float(np.mean(vals)), 3) if vals else None
    result["n_heldout"] = len(vals)

    # Canonical views at a fixed framing: sharpness and likeness to the hero.
    canonical = attempt / "canonical" / "splat.ply"
    tf = json.loads((attempt / "canonical" / "transform.json").read_text())
    h = tf["height_m"]
    w, d = tf["footprint_m"]
    dist = render.framing_distance((w, h, d))
    cams, labels = [], []
    for elev, n in RING_ELEVATIONS:
        cams += render.orbit_cameras((0.0, h / 2, 0.0), dist, n, elevation_deg=elev)
        labels += [f"{elev:+.0f}/{round(360 * i / n)}" for i in range(n)]
    size = (view_height, view_height) if w > h else (round(view_height * 0.75), view_height)
    rendered = render_alpha(canonical, cams, size, gpu)
    sharp, tiles, crops = {}, [], []
    for label, (img, m) in zip(labels, rendered):
        if m.sum() < 100:
            sharp[label] = None
            crops.append(on_gray(np.asarray(img, dtype=np.float32) / 255, m))
        else:
            rgb, mm = at_height(img, m, round(view_height * 0.75))
            sharp[label] = round(laplacian_energy(rgb, mm) / max(result["hero_sharpness"], 1e-9), 3)
            crops.append(on_gray(rgb, mm))
        tiles.append(Image.fromarray((np.where(m[..., None], np.asarray(img, dtype=np.float32) / 255, 0.5) * 255).astype(np.uint8)))
    sims = models.embed(crops) @ hero_embed
    likeness = {label: round(float(s), 4) for label, s in zip(labels, sims)}
    ring = [lab for lab in labels if lab.startswith("+10/")]
    above = [lab for lab in labels if lab.startswith("+55/")]
    below = [lab for lab in labels if lab.startswith("-30/")]
    mean = lambda d, ks: round(float(np.mean([d[k] for k in ks if d[k] is not None])), 4)  # noqa: E731
    result |= {
        "sharpness_ring": mean(sharp, ring), "sharpness_above": mean(sharp, above), "sharpness_below": mean(sharp, below),
        "likeness_front": likeness["+10/0"], "likeness_ring": mean(likeness, ring),
        "likeness_above": mean(likeness, above), "likeness_below": mean(likeness, below),
        "sharpness": sharp, "likeness": likeness,
    }
    metrics = json.loads((attempt / "metrics.json").read_text())
    result["n_gaussians"] = metrics.get("export", {}).get("n_gaussians")
    result["brush_eval_psnr"] = metrics.get("train", {}).get("eval_psnr")

    sheet = render.sheet(tiles, cols=8, width=240)
    if hero_tile is not None:
        hero_tile = hero_tile.resize((round(hero_tile.width * sheet.height / 2 / hero_tile.height), sheet.height // 2))
        out = Image.new("RGB", (sheet.width + hero_tile.width, sheet.height), (128, 128, 128))
        out.paste(hero_tile, (0, 0))
        out.paste(sheet, (hero_tile.width, 0))
        sheet = out
    return result, sheet


COLUMNS = ("hero_psnr", "hero_lpips", "heldout_psnr", "brush_eval_psnr", "sharpness_ring", "sharpness_above",
           "sharpness_below", "likeness_front", "likeness_ring", "likeness_above", "likeness_below", "n_gaussians")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("attempts", type=Path, nargs="+")
    ap.add_argument("--gpu", type=int, default=1, help="GPU for splat-transform's renders")
    ap.add_argument("--view-height", type=int, default=VIEW_HEIGHT,
                    help="canonical views' height in px; sharpness is measured with the subject at 0.75 of it")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    models = Models()
    rows = []
    for attempt in a.attempts:
        path = attempt.resolve()
        # a job's attempts/<seed>: name it after the job
        parent = path.parent.parent if path.parent.name == "attempts" else path.parent
        label = f"{parent.name}/{path.name}"
        result, sheet = evaluate(attempt.resolve(), models, a.gpu, a.view_height)
        result["view_height"] = a.view_height
        stem = label.replace("/", "__")
        (a.out / f"{stem}.json").write_text(json.dumps(result, indent=1) + "\n")
        sheet.save(a.out / f"{stem}-sheet.jpg", quality=90)
        rows.append([label] + [result.get(c) for c in COLUMNS])
        print(label, {c: result.get(c) for c in COLUMNS}, flush=True)
    with open(a.out / "summary.tsv", "a") as f:
        if f.tell() == 0:
            f.write("\t".join(("attempt",) + COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join("" if v is None else str(v) for v in row) + "\n")


if __name__ == "__main__":
    main()
