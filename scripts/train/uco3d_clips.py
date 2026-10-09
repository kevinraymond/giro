"""Real-footage clips for the Route C orbit LoRA (board #3738) from uCO3D (Meta, CC BY 4.0; HF facebook/uco3d):
handheld phone videos circling one object, with cameras, object masks and per-sequence Gaussian splats. Per
sequence, an 81-frame clip laid out like wan_orbit_data.py's synthetic ones:

    uv run --group direct --with h5py --with plyfile --with omegaconf --with opencv-python-headless --with imageio \\
        python scripts/train/uco3d_clips.py UCO3D_DATA OUT GPU [--names a,b] [--step 4.5] [--repo /mnt/t9/uco3d/repo]

- cameras: uCO3D annotates ~200 frames of each video (30, 60, 120 fps or odd rates), unevenly spaced in time (its depth maps exist only for
  those, and 81 even azimuth steps over them repeat frames); every video frame in between gets its camera
  interpolated by timestamp (center linear, rotation slerp; one intrinsic per sequence);
- frames: the camera centers' circle (least-squares look-at point, up = the centers' thinnest spread) gives each
  frame an azimuth; 81 video frames at even azimuth steps of --step degrees (less if the video covers less), so the
  camera moves at a steady speed like giro's path (4.4-8.9 deg/frame) and no frame repeats;
- framing: a 3:4 window of constant size (the object's largest extent / --fill) following the mask's center
  (lightly smoothed: 3 picked frames), resized to 576x768: the object stays centered like a look-at camera;
- target: the frame times uCO3D's mask, on black; control: the depth of the sequence's splat with the background
  Gaussians removed (uCO3D's truncate_bg_gaussians: the object alone, as a proxy), rendered by gsplat at the frame's
  camera, in giro's encoding (GiroRenderSplatCameras' depth style: expected depth where coverage > 0.5, near = 1,
  far = 0, 0 off the object);
- hero / reference: frame 0.

Writes OUT/clips/uco3d_<seq>{.mp4,_control.mp4,_ref.png}, OUT/metadata.csv (DiffSynth-Studio, as wan_orbit_data.py
pack) and OUT/clips.json (per clip: category, azimuth step, coverage, video frames, mask/depth silhouette IoU).
"""
import argparse
import csv
import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation, Slerp

from giro import workflows

W, H, FRAMES = 576, 768, 81
ap = argparse.ArgumentParser()
ap.add_argument("data", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--names", default="", help="comma-separated sequence names (default: all in metadata.sqlite)")
ap.add_argument("--step", type=float, default=4.5, help="azimuth step per frame, degrees")
ap.add_argument("--min-cover", type=float, default=180.0, help="skip clips covering fewer degrees")
ap.add_argument("--min-iou", type=float, default=0.75, help="skip clips whose mask and splat depth agree less (mean IoU)")
ap.add_argument("--fill", type=float, default=0.6, help="the object's largest extent over the window's height")
ap.add_argument("--repo", type=Path, default=Path("/mnt/t9/uco3d/repo"), help="uCO3D's code (its splat loader)")
args = ap.parse_args()
dev = torch.device(f"cuda:{args.gpu}")
pkg = types.ModuleType("uco3d")  # the splat helpers only: uco3d/__init__ pulls in the whole dataset stack
pkg.__path__ = [str(args.repo / "uco3d")]
sys.modules["uco3d"] = pkg
from uco3d.dataset_utils.gauss3d_utils import load_compressed_gaussians, truncate_bg_gaussians  # noqa: E402


def f32(blob: bytes, n: int) -> np.ndarray:
    return np.frombuffer(blob, np.float32, n).astype(np.float64)


def decode(path: Path, frames: list[int], w: int, h: int, gray: bool = False) -> np.ndarray:
    """The given frames (indices in the video, increasing, distinct) as uint8 (N, h, w[, 3])."""
    sel = "+".join(f"eq(n\\,{i})" for i in frames)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"select='{sel}'", "-fps_mode", "passthrough",
                          "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.uint8)
    a = a.reshape(-1, h, w) if gray else a.reshape(-1, h, w, 3)
    if len(a) != len(frames):
        raise RuntimeError(f"{path}: decoded {len(a)} of {len(frames)} frames")
    return a


def probe(path: Path) -> tuple[float, int]:
    """A video's frame rate (uCO3D has 30, 60, 120 and odd rates like 29.58) and frame count."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate,nb_frames",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.split(",")
    num, den = out[0].split("/")
    return float(num) / float(den), int(out[1])


def video_cameras(ts: np.ndarray, R: np.ndarray, T: np.ndarray, fps: float, count: int
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cameras (pytorch3d R, T: X_cam = X R + T) at every video frame between the first and last annotated one."""
    n = np.arange(int(np.ceil(ts[0] * fps)), min(int(np.floor(ts[-1] * fps)) + 1, count))
    t = n / fps
    centers = -np.einsum("ni,nji->nj", T, R)  # C = -T R^T
    c = np.stack([np.interp(t, ts, centers[:, j]) for j in range(3)], 1)
    r = Slerp(ts, Rotation.from_matrix(np.transpose(R, (0, 2, 1))))(t).as_matrix().transpose(0, 2, 1)  # slerp R^T
    return n, r, -np.einsum("ni,nij->nj", c, r)


def azimuths(R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Unwrapped azimuth (degrees, increasing overall) of each camera around the point the cameras look at."""
    centers = -np.einsum("ni,nji->nj", T, R)
    fwd = R[:, :, 2]  # the camera's +z axis in world coordinates (X_cam = X R)
    a, b = np.zeros((3, 3)), np.zeros(3)
    for c, d in zip(centers, fwd / np.linalg.norm(fwd, axis=1, keepdims=True)):
        p = np.eye(3) - np.outer(d, d)
        a += p
        b += p @ c
    rel = centers - np.linalg.solve(a, b)
    _, _, vt = np.linalg.svd(rel - rel.mean(0))
    e1, up = vt[0], vt[2]
    az = np.degrees(np.unwrap(np.arctan2(rel @ np.cross(up, e1), rel @ e1)))
    return az if az[-1] >= az[0] else -az


def pick(az: np.ndarray, step: float) -> tuple[list[int], float]:
    """81 indices at even azimuth steps (step, or the covered span / 80 if shorter), from the start that fits best."""
    order = np.maximum.accumulate(az)  # handheld jitter: never step backwards
    step = min(step, (order[-1] - order[0]) / (FRAMES - 1))
    best = None
    for s in range(0, len(az), 3):
        goals = order[s] + step * np.arange(FRAMES)
        if goals[-1] > order[-1] + 1e-6:
            break
        idx = np.searchsorted(order, goals).clip(0, len(az) - 1)
        err = np.abs(order[idx] - goals).max()
        if best is None or err < best[1]:
            best = (idx, err)
    idx = best[0]
    for i in range(1, len(idx)):  # dense 30 fps frames: a repeat only where the camera paused; take the next frame
        idx[i] = max(idx[i], idx[i - 1] + 1)
    return idx.tolist(), step


def window(cx: float, cy: float, ch: float) -> tuple[float, float, float, float]:
    cw = ch * W / H
    return cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2


def crop(img: np.ndarray, box: tuple, resample) -> np.ndarray:
    return np.asarray(Image.fromarray(img).transform((W, H), Image.EXTENT, box, resample=resample, fillcolor=0))


def render_depth(sp: dict, r: np.ndarray, t: np.ndarray, focal: np.ndarray, pp: np.ndarray, box: tuple, iw: int, ih: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Expected depth and coverage of the splat at a pytorch3d camera (ndc_isotropic intrinsics of the iw x ih frame),
    rendered straight into the crop window `box` at WxH."""
    from gsplat import rasterization

    rcv = r.copy()
    rcv[:, :2] *= -1  # pytorch3d (+x left, +y up) -> OpenCV, as uco3d's opencv_cameras_projection_from_uco3d
    tcv = t.copy()
    tcv[:2] *= -1
    vm = torch.eye(4, device=dev)
    vm[:3, :3] = torch.tensor(rcv.T, device=dev, dtype=torch.float32)
    vm[:3, 3] = torch.tensor(tcv, device=dev, dtype=torch.float32)
    s = min(iw, ih) / 2
    fx, fy = focal * s
    cx, cy = -pp[0] * s + iw / 2, -pp[1] * s + ih / 2
    k = W / (box[2] - box[0])  # crop + resize as an intrinsic change
    K = torch.tensor([[fx * k, 0, (cx - box[0]) * k], [0, fy * k, (cy - box[1]) * k], [0, 0, 1]], device=dev, dtype=torch.float32)
    out, alpha, _ = rasterization(sp["means"], sp["quats"], sp["scales"], sp["opac"], sp["colors"], vm[None], K[None], W, H,
                                  sh_degree=0, render_mode="ED", near_plane=0.01)
    return out[0, ..., 0].cpu().numpy(), alpha[0, ..., 0].cpu().numpy()


db = sqlite3.connect(args.data / "metadata.sqlite")
seqs = db.execute("select sequence_name, category, _video_path, _mask_video_path, _gaussian_splats_dir from sequence_annots").fetchall()
if args.names:
    keep = set(args.names.split(","))
    seqs = [s for s in seqs if s[0] in keep]
cdir = args.out / "clips"
cdir.mkdir(parents=True, exist_ok=True)
rows, info = [], {}
for k, (seq, cat, vpath, mpath, gdir) in enumerate(seqs):
    fr = db.execute("select frame_timestamp, _image_size, _viewpoint_R, _viewpoint_T, _viewpoint_focal_length, "
                    "_viewpoint_principal_point, _viewpoint_intrinsics_format from frame_annots where sequence_name=? "
                    "order by frame_timestamp", (seq,)).fetchall()
    assert all(f[6] == "ndc_isotropic" for f in fr), seq
    _, keep = np.unique([f[0] for f in fr], return_index=True)  # a few sequences repeat a timestamp
    fr = [fr[i] for i in sorted(keep)]
    ts = np.array([f[0] for f in fr])
    fps, count = probe(args.data / vpath)
    ih, iw = (int(v) for v in np.frombuffer(fr[0][1], np.int32, 2))  # stored as (H, W)
    vn, vr, vt = video_cameras(ts, np.stack([f32(f[2], 9).reshape(3, 3) for f in fr]), np.stack([f32(f[3], 3) for f in fr]), fps, count)
    focal, pp = f32(fr[0][4], 2), f32(fr[0][5], 2)
    az = azimuths(vr, vt)
    idx, step = pick(az, args.step)
    cover = float(az[idx[-1]] - az[idx[0]])
    name = f"uco3d_{seq}"
    if cover < args.min_cover:
        print(f"[{k + 1}/{len(seqs)}] {seq} ({cat}): covers {cover:.0f} deg, skipped", flush=True)
        continue
    vid = [int(vn[i]) for i in idx]
    try:
        rgb = decode(args.data / vpath, vid, iw, ih)
        m = [mi.astype(np.float32) / 255 for mi in decode(args.data / mpath, vid, iw, ih, gray=True)]
    except RuntimeError as e:
        print(f"[{k + 1}/{len(seqs)}] {seq} ({cat}): FAILED {e}", flush=True)
        continue
    boxes = []
    for mi in m:
        ys, xs = np.nonzero(mi > 0.5)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None)
    good = [b for b in boxes if b is not None]
    ch = max(max(b[3] - b[1], (b[2] - b[0]) * H / W) for b in good) / args.fill
    cxy = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) if b else (np.nan, np.nan) for b in boxes])
    for j in range(2):  # fill missing boxes, then smooth the window's path
        v = cxy[:, j]
        ok = ~np.isnan(v)
        cxy[:, j] = ndimage.uniform_filter1d(np.interp(np.arange(len(v)), np.flatnonzero(ok), v[ok]), 3, mode="nearest")
    g = truncate_bg_gaussians(load_compressed_gaussians(str(args.data / gdir), load_higher_order_harms=False))
    sp = {"means": g.means, "quats": g.quats, "scales": torch.exp(g.scales), "opac": torch.sigmoid(g.opacities.flatten()),
          "colors": g.sh0[:, None]}
    sp = {kk: v.to(dev, torch.float32) for kk, v in sp.items()}
    tgt, ctl, ious = [], [], []
    for j, i in enumerate(idx):
        box = window(*cxy[j], ch)
        mm = crop((m[j] * 255).astype(np.uint8), box, Image.BILINEAR).astype(np.float32) / 255
        im = crop(rgb[j], box, Image.BICUBIC).astype(np.float32)
        tgt.append((im * mm[..., None] + 0.5).clip(0, 255).astype(np.uint8))
        z, cov = render_depth(sp, vr[i], vt[i], focal, pp, box, iw, ih)
        covm = cov > 0.5
        d = np.zeros((H, W), np.float32)
        if covm.any():
            lo, hi = z[covm].min(), z[covm].max()
            d[covm] = ((hi - z[covm]) / max(hi - lo, 1e-6)).clip(0, 1)
        ctl.append(np.repeat((d * 255 + 0.5).astype(np.uint8)[..., None], 3, -1))
        mb = mm > 0.5
        ious.append(float((mb & covm).sum() / max((mb | covm).sum(), 1)))
    if np.mean(ious) < args.min_iou:
        print(f"[{k + 1}/{len(seqs)}] {seq} ({cat}): mask/depth IoU {np.mean(ious):.2f} (min {np.min(ious):.2f}), skipped", flush=True)
        continue
    Image.fromarray(tgt[0]).save(cdir / f"{name}_ref.png")
    for frames, path in ((tgt, cdir / f"{name}.mp4"), (ctl, cdir / f"{name}_control.mp4")):
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                              "-r", "16", "-i", "-", "-c:v", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p",
                              str(path)], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
        if p.wait():
            raise RuntimeError(f"ffmpeg failed on {path}")
    rows.append({"video": f"clips/{name}.mp4", "control_video": f"clips/{name}_control.mp4", "reference_image": f"clips/{name}_ref.png",
                 "prompt": workflows.PROXY_ORBIT_PROMPT, "control": "real", "preset": "uco3d"})
    info[name] = {"category": cat, "step": round(step, 2), "cover": round(cover, 1), "iou_mean": round(float(np.mean(ious)), 3),
                  "iou_min": round(float(np.min(ious)), 3), "video_frames": vid}
    print(f"[{k + 1}/{len(seqs)}] {seq} ({cat}): {step:.2f} deg/frame, {cover:.0f} deg, crop {ch:.0f} px, "
          f"mask/depth IoU {np.mean(ious):.2f} (min {np.min(ious):.2f})", flush=True)
with open(args.out / "metadata.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
(args.out / "clips.json").write_text(json.dumps(info, indent=1))
print(f"{len(rows)} clips -> {args.out}/metadata.csv", flush=True)
