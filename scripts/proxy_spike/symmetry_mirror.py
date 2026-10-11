"""Symmetry prior feasibility (board #3822): is the hero, mirrored across the subject's symmetry plane,
a believable view from the other side? Measured on the v2 held-out objects, whose truth can be rendered
at any camera (gso_render.py --aligned).

    symmetry_mirror.py plane OUT GPU [--names a,b]   symmetry plane of each proxy, the mirrored camera
    (render)  uv run --project ~/ai/datasets/gso/render python gso_render.py ~/ai/datasets/gso OUT \
                  ~/ai/datasets/v2/manifest.csv --aligned OUT --names a,b --size 768x1024 --depth none --device gpu
    symmetry_mirror.py eval OUT GPU [--names a,b]    mirrored hero vs the truth at the mirrored camera
    symmetry_mirror.py subjects OUT GPU --dirs tank=RUN/proxy,...   the same plane and camera on giro's own proxies

plane: the proxy (V2/controls/NAME/work/mesh.npz, in the splat frame as texture_common.samples puts it)
sampled; its reflective symmetry plane n.x = d searched over normals on a hemisphere, scored by how far
the mirrored samples land from the surface (a distance field, trimmed mean over the closest 90%), the
best few refined by Nelder-Mead. The mirrored camera: R' = F R H, t' = F t - 2d R' n (H = I - 2nn^T,
F = diag(-1, 1, 1)), so it sees the mirrored subject exactly as the hero camera sees the subject, flipped
left to right; for a symmetric subject that is the subject itself. Mapped into the GT frame through the
two hero cameras (gso_controls.gt_camera); OUT/NAME/gt_cameras.json also holds the hero camera itself
("herocheck", which must reproduce the hero render) for gso_render.py.

eval, at the mirrored camera, against the truth there: M = the hero flipped; P = the hero-painted proxy
(the texture route without a generator, gso_controls.py's paint: Pixal3D's colors where the hero does
not see); C = M where the proxy agrees with its own mirror (depth within --agree), P elsewhere. Scores:
LPIPS (VGG, box around the masks on gray, as gso_truth_test.py), PSNR, silhouette IoU with the truth;
the gate the route could use without truth: IoU of the flipped hero mask with the proxy's silhouette.
Writes OUT/symmetry.json and OUT/sheet-*.jpg (hero | truth | M | P | C).

subjects: no truth; the plane, the mirrored camera, the gate and the agreement share on a proxy stage's
own output (Points: TripoSplat or Pixal3D), OUT/subjects.jpg (hero | proxy at the hero | mirrored hero |
proxy at the mirrored camera), OUT/LABEL-mirror.png and OUT/LABEL-agree.jpg (red: the proxy disagrees
with its own mirror there).
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy import ndimage, optimize

from giro import path as campath
from texture_common import PoseCamera, Source, project, render_points, samples, zbuffer

V2 = Path.home() / "ai" / "datasets" / "v2"
W, H = 768, 1024
BASE_WEIGHT = 0.02  # gso_controls.py's
PAINT_TOL = 0.012
ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["plane", "eval", "subjects"])
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--names", default="", help="comma-separated; default: the manifest's held-out split")
ap.add_argument("--points", type=int, default=3_000_000, help="eval: surface samples for the painted proxy")
ap.add_argument("--agree", type=float, default=0.02, help="eval: depth agreement of the proxy with its mirror (units of height)")
ap.add_argument("--dirs", default="", help="subjects: LABEL=PROXY_DIR,... (a proxy stage's proxy/: proxy.ply, proxy.json, hero_rgba.png)")
args = ap.parse_args()
dev = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(dev)
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
names = [] if args.step == "subjects" else args.names.split(",") if args.names else [
    r["name"] for r in csv.DictReader(open(V2 / "manifest.csv")) if r["split"].strip() == "heldout"]
F_FLIP = np.diag([-1.0, 1.0, 1.0])


def gt_camera(c_proxy, hero_p: campath.PathCamera, hero_g: campath.PathCamera) -> dict:
    """gso_controls.gt_camera (it imports a ComfyUI client, so repeated here)."""
    rp, tp = hero_p.world_to_camera()
    rg, tg = hero_g.world_to_camera()
    s = hero_g.distance / hero_p.distance
    m = rg.T @ rp
    b = rg.T @ (s * tp - tg)
    rc, tc = c_proxy.world_to_camera()
    r = rc @ m.T
    return {"rot": r.tolist(), "t": (s * tc - r @ b).tolist(), "fov": c_proxy.fov, "scale": s}


def mirror_camera(cam, n: np.ndarray, d: float) -> PoseCamera:
    r, t = (np.asarray(a, float) for a in cam.world_to_camera())
    hm = np.eye(3) - 2 * np.outer(n, n)
    r2 = F_FLIP @ r @ hm
    t2 = F_FLIP @ t - 2 * d * (r2 @ n)
    return PoseCamera(tuple(map(tuple, r2.tolist())), tuple(t2.tolist()), cam.fov)


def distance_field(x: torch.Tensor, res: int = 128) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Distance (object units) to the nearest occupied voxel over the samples' padded box."""
    lo, hi = x.min(0).values, x.max(0).values
    pad = 0.15 * (hi - lo).max()
    lo, size = lo - pad, float((hi - lo).max() + 2 * pad)
    vox = size / res
    idx = ((x - lo) / vox).long().clamp(0, res - 1).cpu().numpy()
    occ = np.zeros((res, res, res), bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    df = ndimage.distance_transform_edt(~occ) * vox
    return torch.from_numpy(df).float().to(dev), lo, size


def symmetry_plane(x: torch.Tensor) -> dict:
    df, lo, size = distance_field(x)
    vol = df.permute(2, 1, 0)[None, None]  # grid_sample's (D, H, W) = (z, y, x)
    g = torch.Generator(device="cpu").manual_seed(0)
    probe = x[torch.randperm(len(x), generator=g)[:8000].to(dev)]
    c = x.mean(0)

    def score(normals: torch.Tensor, ds: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """Trimmed mean distance of the mirrored points from the surface, per plane."""
        dist = pts[None] @ normals[:, :, None]  # (k, n, 1)
        refl = pts[None] - 2 * (dist - ds[:, None, None]) * normals[:, None, :]
        grid = ((refl - lo) / size * 2 - 1)[:, :, None, None, :]  # (k, n, 1, 1, 3) as x, y, z
        v = F.grid_sample(vol.expand(len(normals), -1, -1, -1, -1), grid, align_corners=False, padding_mode="border")
        v = v.reshape(len(normals), -1)
        k = int(0.9 * v.shape[1])
        return v.sort(1).values[:, :k].mean(1)

    # Fibonacci hemisphere of normals (a plane and its flipped normal are the same plane)
    m = 3000
    i = torch.arange(m, device=dev).float() + 0.5
    zc = i / m  # z in (0, 1]
    phi = math.pi * (1 + 5**0.5) * i
    rr = (1 - zc**2).sqrt()
    normals = torch.stack([rr * torch.cos(phi), rr * torch.sin(phi), zc], 1)
    normals = torch.cat([normals, torch.tensor([[1.0, 0, 0], [0, 1.0, 0]], device=dev)])
    with torch.no_grad():
        s = torch.cat([score(nn_, nn_ @ c, probe) for nn_ in normals.split(250)])
    order = s.argsort()
    best = None
    seeds = []
    for j in order.tolist():  # the best few distinct planes
        if all(abs(float(normals[j] @ normals[k])) < 0.97 for k in seeds):
            seeds.append(j)
        if len(seeds) == 5:
            break
    more = x[torch.randperm(len(x), generator=g)[:30000].to(dev)]
    cands = []
    for j in seeds:
        n0 = normals[j].double().cpu().numpy()
        th0, ph0 = math.acos(np.clip(n0[2], -1, 1)), math.atan2(n0[1], n0[0])

        def f(p):
            th, ph, d = p
            nn_ = torch.tensor([[math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)]], device=dev).float()
            with torch.no_grad():
                return float(score(nn_, torch.tensor([d], device=dev).float(), more)[0])
        r = optimize.minimize(f, [th0, ph0, float(normals[j] @ c)], method="Nelder-Mead",
                              options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        th, ph, d = r.x
        n = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        cands.append({"normal": n, "d": float(d), "score": float(r.fun)})
    cands.sort(key=lambda q: q["score"])
    best = cands[0]
    # spacing of the samples: the score's floor; a symmetric proxy scores near it
    spacing = float(torch.cdist(probe[:2000], x[torch.randperm(len(x), generator=g)[:200000].to(dev)]).topk(2, largest=False).values[:, 1].mean())
    return best | {"second": cands[1]["score"] if len(cands) > 1 else None, "floor": spacing,
                   "vertical_deg": round(math.degrees(math.asin(min(1.0, abs(best["normal"][1])))), 1)}


def hero_rgba(name: str) -> tuple[np.ndarray, np.ndarray]:
    im = np.asarray(Image.open(V2 / "renders" / name / "hero_rgba.png").convert("RGBA").resize((W, H), Image.LANCZOS), np.float32) / 255
    return im[..., :3] * im[..., 3:], im[..., 3] > 0.5


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max(1, (a | b).sum()))


if args.step == "plane":
    for name in names:
        src = V2 / "controls" / name
        gt = json.loads((src / "gt_cameras.json").read_text())
        hero_p = campath.PathCamera.from_json(gt["hero_camera"])
        hero_g = campath.PathCamera.from_json(json.loads((V2 / "renders" / name / "cameras.json").read_text())["views"][0])
        xyz, nrm, _, _ = samples(src / "work", 400_000, dev)
        pl = symmetry_plane(xyz)
        n, d = pl["normal"], pl["d"]
        mcam = mirror_camera(hero_p, n, d)
        # how far the mirrored camera is from the hero, around the target: 0 = the hero faces along the plane
        pos_h, pos_m = np.asarray(hero_p.position()), np.asarray(mcam.position())
        tgt = np.asarray(hero_p.target)
        a, b = pos_h - tgt, pos_m - tgt
        sep = math.degrees(math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
        _, hmask = hero_rgba(name)
        _, psil = render_points(xyz, torch.ones_like(xyz[:, :1]), mcam, W, H, nrm=nrm)
        _, hsil = render_points(xyz, torch.ones_like(xyz[:, :1]), hero_p, W, H, nrm=nrm)
        rec = {"name": name, "normal": [round(v, 5) for v in n.tolist()], "d": round(d, 5),
               "score": round(pl["score"], 5), "second": round(pl["second"], 5), "floor": round(pl["floor"], 5),
               "vertical_deg": pl["vertical_deg"], "separation_deg": round(sep, 1),
               "gate_iou": round(iou(np.fliplr(hmask), psil), 4), "hero_iou": round(iou(hmask, hsil), 4),
               "mirror_camera": mcam.to_json()}
        od = out / name
        od.mkdir(exist_ok=True)
        (od / "plane.json").write_text(json.dumps(rec, indent=1))
        (od / "gt_cameras.json").write_text(json.dumps({"views": [
            {"name": "mirror"} | gt_camera(mcam, hero_p, hero_g),
            {"name": "herocheck"} | gt_camera(PoseCamera.of(hero_p), hero_p, hero_g)]}, indent=1))
        print(f"{name}: score {rec['score']:.4f} (2nd {rec['second']:.4f}, floor {rec['floor']:.4f}), "
              f"plane tilt {rec["vertical_deg"]:.0f} deg, separation {sep:.0f} deg, "
              f"gate IoU {rec['gate_iou']:.3f}, hero IoU {rec['hero_iou']:.3f}", flush=True)
    sys.exit()

if args.step == "subjects":
    from giro.stages.proxy import Points

    def zmap(pts: Points, cam, w: int, h: int) -> np.ndarray:
        u, v, z = pts.project(cam, w, h)
        ok = np.isfinite(z) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        m = np.full((h, w), np.inf)
        np.minimum.at(m, (v[ok].astype(int), u[ok].astype(int)), z[ok])
        return ndimage.grey_erosion(m, size=3)

    tiles, recs = [], []
    for item in args.dirs.split(","):
        label, d = item.split("=")
        d = Path(d).expanduser()
        pj = json.loads((d / "proxy.json").read_text())
        hero_p = campath.PathCamera.from_json(pj["hero_camera"])
        pts = Points(d / "proxy.ply")
        rgba = np.asarray(Image.open(d / "hero_rgba.png").convert("RGBA"), np.float32) / 255
        h, w = rgba.shape[:2]
        hero, hmask = rgba[..., :3] * rgba[..., 3:], rgba[..., 3] > 0.5
        pl = symmetry_plane(torch.from_numpy(pts.xyz).float().to(dev))
        n, dd = pl["normal"], pl["d"]
        mcam = mirror_camera(hero_p, n, dd)
        a, b = np.asarray(hero_p.position()) - hero_p.target, np.asarray(mcam.position()) - hero_p.target
        sep = math.degrees(math.acos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
        psil = pts.silhouette(mcam, w, h)
        gate = iou(np.fliplr(hmask), psil)
        zm, zh = zmap(pts, mcam, w, h), np.fliplr(zmap(pts, hero_p, w, h))
        agree = np.isfinite(zm) & np.isfinite(zh) & (np.abs(zm - zh) < args.agree * hero_p.distance / 2.5)
        agree = ndimage.binary_opening(agree, iterations=2) & np.fliplr(hmask)
        rec = {"label": label, "dir": str(d), "model": pj.get("model"), "separation_deg": round(sep, 1), "gate_iou": round(gate, 4),
               "plane_tilt_deg": pl["vertical_deg"], "score": round(pl["score"], 5), "floor": round(pl["floor"], 5),
               "agree_share": round(float(agree.sum() / max(1, hmask.sum())), 3), "mirror_camera": mcam.to_json(),
               "normal": [round(v, 5) for v in n.tolist()], "d": round(dd, 5)}
        recs.append(rec)
        print(f"{label}: {rec['model']}, separation {sep:.0f} deg, gate IoU {gate:.3f}, plane tilt {pl['vertical_deg']:.0f} deg, "
              f"agree {rec['agree_share']:.2f}, score {pl['score']:.4f} (floor {pl['floor']:.4f})", flush=True)
        flip = np.fliplr(hero)
        shade = np.where(agree[..., None], flip, flip * 0.35 + np.array([0.6, 0.0, 0.0]) * np.fliplr(hmask)[..., None])
        th = 384
        tw = round(th * w / h)
        row = Image.new("RGB", (tw * 4, th + 18), (40, 40, 40))
        for k, (im, mk) in enumerate([(hero, hmask), (pts.color(hero_p, w, h), pts.silhouette(hero_p, w, h)),
                                      (flip, np.fliplr(hmask)), (pts.color(mcam, w, h), psil)]):
            t = ((im * mk[..., None] + 0.5 * (1 - mk[..., None])) * 255).clip(0, 255).astype(np.uint8)
            row.paste(Image.fromarray(t).resize((tw, th), Image.LANCZOS), (k * tw, 18))
        ImageDraw.Draw(row).text((4, 3), f"{label} ({rec['model']})  sep {sep:.0f} deg  gate {gate:.2f}  "
                                 f"agree {rec['agree_share']:.2f}   [hero | proxy@hero | mirrored hero | proxy@mirror]", fill=(255, 255, 0))
        tiles.append(row)
        Image.fromarray((shade * 255).clip(0, 255).astype(np.uint8)).save(out / f"{label}-agree.jpg", quality=88)
        Image.fromarray((flip * 255).astype(np.uint8)).save(out / f"{label}-mirror.png")
    (out / "subjects.json").write_text(json.dumps(recs, indent=1))
    sheet = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)), (40, 40, 40))
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height
    sheet.save(out / "subjects.jpg", quality=88)
    sys.exit()

# eval
import lpips  # noqa: E402

lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)


def truth(name: str, view: str) -> tuple[np.ndarray, np.ndarray]:
    im = np.asarray(Image.open(out / name / "target" / f"{view}.png").convert("RGBA").resize((W, H), Image.LANCZOS), np.float32) / 255
    return im[..., :3] * im[..., 3:], im[..., 3] > 0.5


def on_gray(rgb: np.ndarray, m: np.ndarray) -> torch.Tensor:
    mm = m[..., None].astype(np.float32)
    return torch.from_numpy(rgb * mm + 0.5 * (1 - mm)).float().to(dev)


def scores(pred: torch.Tensor, pm: np.ndarray, gt_img: torch.Tensor, gm: np.ndarray) -> dict:
    ys, xs = np.nonzero(ndimage.binary_dilation(gm | pm, iterations=16))
    box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    with torch.no_grad():
        l = float(lp(pred[box].permute(2, 0, 1)[None] * 2 - 1, gt_img[box].permute(2, 0, 1)[None] * 2 - 1))
    inside = torch.from_numpy(gm | pm).to(dev)
    mse = float(((pred - gt_img)[inside] ** 2).mean())
    return {"lpips": round(l, 4), "psnr": round(10 * math.log10(1 / max(mse, 1e-10)), 2), "iou": round(iou(pm, gm), 4)}


rows, tiles = [], []
for name in names:
    od = out / name
    if not (od / "target" / "mirror.png").exists():
        print(f"{name}: no truth render, skipped", flush=True)
        continue
    rec = json.loads((od / "plane.json").read_text())
    src = V2 / "controls" / name
    gt = json.loads((src / "gt_cameras.json").read_text())
    hero_p = campath.PathCamera.from_json(gt["hero_camera"])
    mcam = PoseCamera.from_json(rec["mirror_camera"])
    hero, hmask = hero_rgba(name)
    chk, chk_m = truth(name, "herocheck")
    t_rgb, t_m = truth(name, "mirror")
    gt_img = on_gray(t_rgb, t_m)
    # M: the hero flipped
    m_rgb, m_m = np.ascontiguousarray(np.fliplr(hero)), np.ascontiguousarray(np.fliplr(hmask))
    # P: the hero-painted proxy at the mirrored camera
    xyz, nrm, base, _ = samples(src / "work", args.points, dev)
    hsrc = Source("hero", Image.fromarray((hero * 255).astype(np.uint8)), hmask, hero_p, 1.0, dev)
    col, wt = hsrc.paint(xyz, nrm, PAINT_TOL)
    rgb = (col * wt[:, None] + base * BASE_WEIGHT) / (wt[:, None] + BASE_WEIGHT)
    p_rgb, p_m = render_points(xyz, rgb, mcam, W, H, nrm=nrm)
    # agreement: the proxy's depth at the mirrored camera vs its depth at the hero camera flipped
    # (= the mirrored proxy seen from the mirrored camera)
    def depth(cam) -> np.ndarray:
        u, vv, z = project(cam, W, H, xyz)
        return zbuffer(u, vv, z, W, H)[3].cpu().numpy()
    dm, dh = depth(mcam), np.fliplr(depth(hero_p))
    agree = np.isfinite(dm) & np.isfinite(dh) & (np.abs(dm - dh) < args.agree)
    agree = ndimage.binary_opening(agree, iterations=2) & m_m
    c_rgb = np.where(agree[..., None], m_rgb, p_rgb)
    c_m = (agree | p_m)
    r = {"name": name, "separation_deg": rec["separation_deg"], "gate_iou": rec["gate_iou"],
         "herocheck_iou": round(iou(chk_m, hmask), 4), "agree_share": round(float(agree.sum() / max(1, m_m.sum())), 3),
         "M": scores(on_gray(m_rgb, m_m), m_m, gt_img, t_m), "P": scores(on_gray(p_rgb, p_m), p_m, gt_img, t_m),
         "C": scores(on_gray(c_rgb, c_m), c_m, gt_img, t_m)}
    rows.append(r)
    print(f"{name}: sep {r['separation_deg']:.0f}, gate {r['gate_iou']:.3f}, check {r['herocheck_iou']:.3f}, agree {r['agree_share']:.2f} | "
          + " | ".join(f"{k} lpips {r[k]['lpips']:.3f} psnr {r[k]['psnr']:.1f} iou {r[k]['iou']:.3f}" for k in "MPC"), flush=True)
    tw, th = 192, 256
    row = Image.new("RGB", (tw * 5, th + 16), (40, 40, 40))
    for k, (im, mk) in enumerate([(hero, hmask), (t_rgb, t_m), (m_rgb, m_m), (p_rgb, p_m), (c_rgb, c_m)]):
        t = (on_gray(im, mk).cpu().numpy() * 255).astype(np.uint8)
        row.paste(Image.fromarray(t).resize((tw, th), Image.LANCZOS), (k * tw, 16))
    dr = ImageDraw.Draw(row)
    dr.text((4, 2), f"{name[:40]}  sep {r['separation_deg']:.0f}  gate {r['gate_iou']:.2f}  LPIPS M {r['M']['lpips']:.3f} "
            f"P {r['P']['lpips']:.3f} C {r['C']['lpips']:.3f}", fill=(255, 255, 0))
    tiles.append(row)
    torch.cuda.empty_cache()
(out / "symmetry.json").write_text(json.dumps(rows, indent=1))
for i in range(0, len(tiles), 10):
    part = tiles[i:i + 10]
    sheet = Image.new("RGB", (part[0].width, sum(t.height for t in part)), (40, 40, 40))
    for j, t in enumerate(part):
        sheet.paste(t, (0, j * t.height))
    sheet.save(out / f"sheet-{i // 10}.jpg", quality=88)
if rows:
    for k in "MPC":
        print(k, {s: round(float(np.mean([r[k][s] for r in rows])), 4) for s in ("lpips", "psnr", "iou")})
    print("M beats P on LPIPS:", sum(r["M"]["lpips"] < r["P"]["lpips"] for r in rows), "of", len(rows))
