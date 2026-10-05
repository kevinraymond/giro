"""Register anchor views to the proxy mesh photometrically: each anchor's camera is moved until
the colors it paints onto the mesh agree best with what the other views paint there (Zhou &
Koltun, "Color Map Optimization", SIGGRAPH 2014; Open3D's color_map_optimization), with its
silhouette kept on the mesh's. Every pixel counts, so it works on smooth glossy panels where SIFT
finds too few matches (a scooter: 5-18 per pair, Oct 5).

The hero's camera stays fixed. Anchors take turns (most overlap with the hero first), each against
the blend of all the others, for ROUNDS rounds. Cost: 1 - weighted NCC of luminance over samples
both see well, + SIL_WEIGHT (1 - silhouette IoU). Pose (rotation, translation) and fov are free; an anchor
keeps its silhouette camera if the cost does not improve, it turns more than MAX_TURN_DEG or
its silhouette IoU drops more than MAX_IOU_DROP.

    register_photometric.py ATTEMPT ANCHOR_DIR WORK GPU [SKIP=02,05]

Reads WORK/texture.json (project_texture.py's silhouette fits); writes WORK/anchor_cameras.json
for project_texture.py's CAMERAS argument.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import optimize
from scipy.spatial.transform import Rotation

from giro import path as campath
from texture_common import PoseCamera, Source, anchor_mask, project, samples

attempt, adir, work = (Path(a).resolve() for a in sys.argv[1:4])
dev = torch.device(f"cuda:{int(sys.argv[4])}")
skip = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else {"02", "05"}
SCALE = 0.5        # images are compared at half size, slightly blurred: a smoother cost
ROUNDS = 2
MAX_TURN_DEG = 6.0
TOL = 0.012
MIN_OVERLAP = 3000
SIL_WEIGHT = 1.0    # at 0.5, three anchors traded 0.06-0.12 of silhouette IoU for color agreement
MAX_IOU_DROP = 0.03

fits = json.loads((work / "texture.json").read_text())["fits"]
xyz, nrm, _, _ = samples(work, 3_000_000, dev)
lum_w = torch.tensor([0.299, 0.587, 0.114], device=dev)


def small(img: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    size = (round(img.width * SCALE), round(img.height * SCALE))
    m = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size)) > 127
    return img.resize(size, Image.LANCZOS).filter(ImageFilter.GaussianBlur(1.0)), m


def camera(fit: dict) -> PoseCamera:
    return PoseCamera.of(campath.PathCamera.from_json(fit) if "yaw" in fit else PoseCamera.from_json(fit))


hero_img, hero_mask = small(Image.open(attempt / "hero" / "hero.png").convert("RGB"),
                            np.asarray(Image.open(attempt / "proxy" / "hero_mask.png")) > 127)
sources = [Source("hero", hero_img, hero_mask, camera(fits["hero"]["camera"]), 2.0, dev, 4.0)]
for n in sorted(fits):
    if n != "hero" and n not in skip:
        img, mask = small(Image.open(adir / f"{n}.png").convert("RGB"), anchor_mask(adir, n))
        sources.append(Source(n, img, mask, camera(fits[n]["camera"]), 1.0, dev, 4.0))
painted = [s.paint(xyz, nrm, TOL) for s in sources]


def silhouette_iou(src: Source, cam: PoseCamera) -> float:
    w, h = src.w // 2, src.h // 2
    u, v, z = project(cam, w, h, xyz)
    ok = (z > 1e-3) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    sil = torch.zeros(h * w, device=dev, dtype=torch.bool)
    sil[v[ok].long() * w + u[ok].long()] = True
    sil = torch.nn.functional.max_pool2d(sil.reshape(1, 1, h, w).float(), 3, 1, 1)[0, 0] > 0
    m = torch.from_numpy(np.asarray(Image.fromarray(src.mask.astype(np.uint8) * 255).resize((w, h))) > 127).to(dev)
    return float((sil & m).sum() / (sil | m).sum().clamp_min(1))


def moved(cam: PoseCamera, x: np.ndarray, dist: float) -> PoseCamera:
    """x: rotation vector (degrees, camera frame), translation (camera frame, in units of the
    camera's distance), fov change (degrees)."""
    r, t = cam.world_to_camera()
    center = -r.T @ t
    rd = Rotation.from_rotvec(np.radians(x[:3])).as_matrix()
    r2 = rd @ r
    c2 = center + r.T @ (np.asarray(x[3:6]) * dist)
    return PoseCamera(tuple(map(tuple, r2.tolist())), tuple((-r2 @ c2).tolist()), cam.fov + float(x[6]))


def turn_deg(a: PoseCamera, b: PoseCamera) -> float:
    r = np.asarray(a.rot) @ np.asarray(b.rot).T
    return math.degrees(math.acos(np.clip((np.trace(r) - 1) / 2, -1, 1)))


def wncc(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor) -> float:
    w = w / w.sum()
    am, bm = (w * a).sum(), (w * b).sum()
    cov = (w * (a - am) * (b - bm)).sum()
    return float(cov / ((w * (a - am) ** 2).sum() * (w * (b - bm) ** 2).sum()).sqrt().clamp_min(1e-9))


def cost_parts(i: int, cam: PoseCamera, ref: torch.Tensor, ref_w: torch.Tensor) -> tuple[float, float, int]:
    col, wt = sources[i].paint(xyz, nrm, TOL, cam)
    both = (wt > 0.05) & (ref_w > 0.05)
    n = int(both.sum())
    if n < MIN_OVERLAP:
        return 0.0, silhouette_iou(sources[i], cam), n
    return wncc(col[both] @ lum_w, ref[both] @ lum_w, torch.minimum(wt[both], ref_w[both])), silhouette_iou(sources[i], cam), n


hero_overlap = {i: float(((painted[i][1] > 0.05) & (painted[0][1] > 0.05)).sum()) for i in range(1, len(sources))}
order = sorted(hero_overlap, key=lambda i: -hero_overlap[i])
start = {i: sources[i].cam for i in order}
report: dict = {}
for rnd in range(ROUNDS):
    for i in order:
        others = [j for j in range(len(sources)) if j != i]
        w2 = torch.stack([painted[j][1] ** 2 for j in others])
        ref = (w2[:, :, None] * torch.stack([painted[j][0] for j in others])).sum(0) / w2.sum(0).clamp_min(1e-12)[:, None]
        ref_w = torch.stack([painted[j][1] for j in others]).sum(0)
        cam0 = sources[i].cam
        dist = float(np.linalg.norm(cam0.position()))

        def cost(x: np.ndarray) -> float:
            ncc, iou, n = cost_parts(i, moved(cam0, x, dist), ref, ref_w)
            return (1 - ncc) + SIL_WEIGHT * (1 - iou) + (1.0 if n < MIN_OVERLAP else 0.0)

        before = cost(np.zeros(7))
        res = optimize.minimize(cost, np.zeros(7), method="Nelder-Mead", options={
            "initial_simplex": np.vstack([np.zeros(7), np.diag([0.8, 0.8, 0.8, 0.01, 0.01, 0.01, 1.0])]),
            "maxiter": 500, "xatol": 0.01, "fatol": 1e-5})
        cam = moved(cam0, res.x, dist)
        turned = turn_deg(cam, start[i])
        iou_start = silhouette_iou(sources[i], start[i])
        accept = (res.fun < before - 1e-4 and turned <= MAX_TURN_DEG
                  and silhouette_iou(sources[i], cam) >= iou_start - MAX_IOU_DROP)
        if accept:
            sources[i].cam = cam
            painted[i] = sources[i].paint(xyz, nrm, TOL)
        ncc0, iou0, n0 = cost_parts(i, cam0, ref, ref_w)
        ncc1, iou1, n1 = cost_parts(i, sources[i].cam, ref, ref_w)
        report[sources[i].name] = {
            "round": rnd + 1, "accepted": accept, "ncc": [round(ncc0, 4), round(ncc1, 4)], "iou": [round(iou0, 4), round(iou1, 4)],
            "overlap": n1, "turn_from_start_deg": round(turn_deg(sources[i].cam, start[i]), 2),
            "moved_from_start": round(float(np.linalg.norm(sources[i].cam.position() - start[i].position())), 4),
            "fov": round(sources[i].cam.fov, 2), "camera": sources[i].cam.to_json()}
        print(f"round {rnd + 1} {sources[i].name}: NCC {ncc0:.3f} -> {ncc1:.3f}, IoU {iou0:.3f} -> {iou1:.3f}, "
              f"overlap {n1}, turned {report[sources[i].name]['turn_from_start_deg']} deg, fov {sources[i].cam.fov:.1f}"
              f"{'' if accept else ' (kept)'}", flush=True)
(work / "anchor_cameras.json").write_text(json.dumps({"cameras": report}, indent=1))
print(work / "anchor_cameras.json")
