"""Poses from the proxy path: the nominal cameras the depth video was rendered from, refined by
COLMAP's triangulation and bundle adjustment on the COLMAP stage's own matches (as the DA3
fallback does). Writes poses/fallback/ and makes it the active model.

    proxy_poses.py ATTEMPT
"""
import json, math, re, shutil, sqlite3, subprocess, sys
from pathlib import Path
import numpy as np
from giro.stages import poses as poses_stage
from giro.stages.fallback import hybrid, _pose, _qvec
from giro.stages.poses import read_images_txt

a = Path(sys.argv[1]).resolve(); pj = json.loads((a / "proxy.json").read_text())
N, W = pj["frames"], pj["width"]
turns, p0, p1, mode = pj.get("turns", 1.0), pj["pitch"], pj.get("pitch_end", pj["pitch"]), pj.get("pitch_mode", "ramp")

def nominal(i):
    t = i / N
    pitch = p0 + (p1 - p0) * (0.5 * (1 - math.cos(2 * math.pi * t)) if mode == "wave" else i / max(1, N - 1))
    y, p = math.radians(pj["yaw"] + 360.0 * turns * t), math.radians(pitch)
    f = np.array([-math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])  # splat frame, y down
    x = np.cross([0.0, 1.0, 0.0], f); x /= np.linalg.norm(x)
    r = np.stack([x, np.cross(f, x), f])
    return r, -r @ (-pj["distance"] * f)

colmap_dir, work = a / "poses" / "colmap", a / "poses" / "fallback"
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
db = work / "database.db"; shutil.copyfile(colmap_dir / "database.db", db)
with sqlite3.connect(db) as con:
    db_images = {n: (i, c) for i, n, c in con.execute("SELECT image_id, name, camera_id FROM images")}
    db_cams = {c: (w, h) for c, w, h in con.execute("SELECT camera_id, width, height FROM cameras")}
ff = {n: nominal(int(Path(n).stem)) for n in db_images if n.startswith("frames/")}
focal = 967.0 * W / 576  # what COLMAP found on a flat proxy ring; bundle adjustment refines it
start = work / "start"; start.mkdir()
(start / "cameras.txt").write_text("".join(f"{c} SIMPLE_PINHOLE {w} {h} {focal} {w / 2} {h / 2}\n" for c, (w, h) in db_cams.items()))
(start / "images.txt").write_text("".join(
    f"{i} {' '.join(map(str, _qvec(ff[n][0])))} {' '.join(map(str, ff[n][1]))} {c} {n}\n\n"
    for n, (i, c) in sorted(db_images.items(), key=lambda x: x[1][0]) if n in ff))
(start / "points3D.txt").write_text("")

def colmap(*args):
    r = subprocess.run(["colmap", *args, "--log_color", "0"], capture_output=True, text=True)
    (work / "fallback.log").open("a").write(r.stdout + r.stderr)
    if r.returncode: raise SystemExit(f"colmap {args[0]} failed: {(r.stderr or r.stdout).strip().splitlines()[-2:]}")
    return r.stdout + r.stderr

model = start
for rnd in (1, 2):
    tri, ba = work / f"triangulated{rnd}", work / f"adjusted{rnd}"; tri.mkdir(); ba.mkdir()
    colmap("point_triangulator", "--database_path", str(db), "--image_path", str(colmap_dir / "images"),
           "--input_path", str(model), "--output_path", str(tri), "--clear_points", "1")
    colmap("bundle_adjuster", "--input_path", str(tri), "--output_path", str(ba), "--BundleAdjustment.refine_focal_length", "1",
           "--BundleAdjustment.refine_principal_point", "0", "--BundleAdjustment.refine_extra_params", "0")
    model = ba
stats = colmap("model_analyzer", "--path", str(model))
adj_txt = work / "adjusted_txt"; adj_txt.mkdir()
colmap("model_converter", "--input_path", str(model), "--output_path", str(adj_txt), "--output_type", "TXT")
adjusted = read_images_txt(adj_txt / "images.txt")
constrained = [n for n, im in adjusted.items() if im["n_points"] >= 15]
refined = {n: _pose(im) for n, im in adjusted.items()}
out = hybrid({n: ff[n] for n in adjusted}, refined, constrained)
# how far bundle adjustment moved the cameras from the nominal path (after a similarity fit)
from giro.stages.fallback import similarity, _centers
s, r, t = similarity(_centers(ff, constrained), _centers(refined, constrained))
moved = np.linalg.norm((s * (r @ _centers(ff, constrained).T).T + t) - _centers(refined, constrained), axis=1) / (s * pj["distance"])
# the proxy's own frame in the refined world: center, up (the splat frame is y down) and front (toward the first camera)
f0 = -nominal(0)[0][2]; f0[1] = 0.0
(a / "poses" / "frame.json").write_text(json.dumps({"center": t.tolist(), "up": (r @ np.array([0.0, -1.0, 0.0])).tolist(),
                                                     "front": (r @ (f0 / np.linalg.norm(f0))).tolist(), "source": "proxy path"}, indent=2))
out_txt = work / "model_txt"; out_txt.mkdir()
shutil.copy(adj_txt / "cameras.txt", out_txt); shutil.copy(adj_txt / "points3D.txt", out_txt)
lines = [l for l in (adj_txt / "images.txt").read_text().splitlines() if not l.startswith("#")]
with open(out_txt / "images.txt", "w") as f:
    for header, points in zip(lines[0::2], lines[1::2]):
        h = header.split(); rot, tv = out[h[9]]
        f.write(f"{h[0]} {' '.join(f'{x:.12g}' for x in _qvec(rot))} {' '.join(f'{x:.12g}' for x in tv)} {h[8]} {h[9]}\n{points}\n")
(work / "model").mkdir()
colmap("model_converter", "--input_path", str(out_txt), "--output_path", str(work / "model"), "--output_type", "BIN")
poses_stage.activate(a, "fallback")
reproj = re.search(r"Mean reprojection error:\s*([\d.]+)", stats); pts = re.search(r"Points:\s*(\d+)", stats)
print(f"{len(adjusted)} images posed, {len(constrained)} refined by bundle adjustment, {pts.group(1) if pts else '?'} points, "
      f"reprojection {reproj.group(1) if reproj else '?'} px, focal {(out_txt / 'cameras.txt').read_text().split()[-3]}, "
      f"moved from the nominal path: median {np.median(moved):.3f}, max {moved.max():.3f} of the orbit radius")
