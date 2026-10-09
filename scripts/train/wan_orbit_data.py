"""Training clips for the Route C orbit LoRA (board #3738): Wan 2.2 Fun Control taught to turn giro's proxy
depth into the TRUE object's look. Per v2 object (~/ai/datasets/v2: GSO / Poly Haven / Objaverse CC-BY, each
with its hero render and Pixal3D proxy), an 81-frame orbit with exact cameras, laid out as giro's proxy orbit
lays its path (giro.path.plan from the proxy's hero camera, frame 0 = the hero):

    wan_orbit_data.py plan OUT [--names a,b | --only N] [--true-share 0.25] [--seed 0]
    uv run --project ~/ai/datasets/gso/render python scripts/proxy_spike/gso_render.py ~/ai/datasets/gso/models \\
        OUT ~/ai/datasets/v2/manifest.csv --aligned OUT --names <OUT/true.txt> --size 576x768 --device gpu --depth all
        (and again with --names <OUT/proxy.txt> --depth none)
    wan_orbit_data.py controls OUT GPU     depth videos through giro's own renderer (GiroRenderSplatCameras)
    wan_orbit_data.py pack OUT             clips/<name>.mp4, <name>_control.mp4, metadata.csv (DiffSynth-Studio)

plan writes OUT/<name>/clip.json (the path and its cameras in the proxy's frame) and gt_cameras.json (the same
cameras mapped into the object's own frame, gso_controls.py's gt_camera), so gso_render.py --aligned renders
the true frames OUT/<name>/target/fNNN.png. The control is the depth of a splat along the same cameras: the
Pixal3D proxy's surface (what inference has, shape errors included) or, for --true-share of the objects, the
true surface (back-projected from the target depth maps into the proxy's frame), so the LoRA sees both.
Frames go on black, as giro's depth control and the truth test's Wan baseline (#3737); the caption is giro's
PROXY_ORBIT_PROMPT. Held out: v2's heldout split and the truth-test objects (bus, shoe, Lego).
"""
import argparse
import asyncio
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "proxy_spike"))
from giro import path as campath  # noqa: E402
from giro import splat, workflows  # noqa: E402
from texture_common import write_points_ply  # noqa: E402

V2 = Path.home() / "ai" / "datasets" / "v2"
W, H, FRAMES = 576, 768, 81
TRUTH_TEST = {"Sonny_School_Bus", "Reebok_REESCULPT_TRAINER_II", "Android_Lego"}
# giro's default path (orbit.py path_defaults) half the time; other presets and pitches for the rest
PRESETS = [("spiral", 0.5), ("ringrise", 0.2), ("wave", 0.3)]

ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["plan", "controls", "pack"])
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int, nargs="?", default=1)
ap.add_argument("--names", default="", help="comma-separated objects instead of v2's train split")
ap.add_argument("--only", type=int, default=0, help="the first N objects of the shuffled train split (0: all)")
ap.add_argument("--true-share", type=float, default=0.25, help="share of objects whose control is the true surface")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
out = args.out.resolve()


def gt_camera(c_proxy: campath.PathCamera, hero_p: campath.PathCamera, hero_g: campath.PathCamera) -> dict:
    """gso_controls.py's (which parses its arguments on import): c_proxy in the GT frame, x_g = s M x_p + b,
    from identifying the two hero cameras' coordinates (scaled by s); COLMAP's R, t there."""
    rp, tp = hero_p.world_to_camera()
    rg, tg = hero_g.world_to_camera()
    s = hero_g.distance / hero_p.distance
    m = rg.T @ rp
    b = rg.T @ (s * tp - tg)
    rc, tc = c_proxy.world_to_camera()
    r = rc @ m.T
    return {"rot": r.tolist(), "t": (s * tc - r @ b).tolist(), "fov": c_proxy.fov, "scale": s}


def clips() -> list[Path]:
    return sorted(p.parent for p in out.glob("*/clip.json"))


def sample_path(rng: random.Random, hero: campath.PathCamera) -> dict:
    preset = rng.choices([p for p, _ in PRESETS], [w for _, w in PRESETS])[0]
    if preset == "spiral" and rng.random() < 0.5:
        turns, pitch_end = 2.0, 45.0  # exactly giro's default
    else:
        turns, pitch_end = rng.choice([1.0, 1.5, 2.0]), round(rng.uniform(-15.0, 60.0), 1)
    cams = campath.plan(hero, preset, FRAMES, turns=turns, pitch_end=pitch_end)
    return {"preset": preset, "turns": turns, "pitch_end": pitch_end, "cameras": [c.to_json() for c in cams]}


if args.step == "plan":
    rows = list(csv.DictReader(open(V2 / "manifest.csv")))
    if args.names:
        names = args.names.split(",")
    else:
        names = [r["name"] for r in rows if r["split"] == "train" and r["name"] not in TRUTH_TEST]
        random.Random(args.seed).shuffle(names)
        names = [n for n in names if all((V2 / "controls" / n / f).exists() for f in ("work/mesh.npz", "gt_cameras.json"))
                 and (V2 / "renders" / n / "cameras.json").exists()]
        if args.only:
            names = names[:args.only]
    lists = {"proxy": [], "true": []}
    for name in names:
        src, od = V2 / "controls" / name, out / name
        gt = json.loads((src / "gt_cameras.json").read_text())
        hero_p = campath.PathCamera.from_json(gt["hero_camera"])
        hero_g = campath.PathCamera.from_json(json.loads((V2 / "renders" / name / "cameras.json").read_text())["views"][0])
        rng = random.Random(f"{args.seed}:{name}")
        clip = sample_path(rng, hero_p) | {"object": name, "control": "true" if rng.random() < args.true_share else "proxy",
                                           "hero_gt": hero_g.to_json()}
        od.mkdir(parents=True, exist_ok=True)
        (od / "clip.json").write_text(json.dumps(clip, indent=1))
        views = [{"name": f"f{i:03d}"} | gt_camera(campath.PathCamera.from_json(c), hero_p, hero_g)
                 for i, c in enumerate(clip["cameras"])]
        (od / "gt_cameras.json").write_text(json.dumps({"views": views}, indent=1))
        lists[clip["control"]].append(name)
    for k, v in lists.items():
        (out / f"{k}.txt").write_text(",".join(v))
    print(f"planned {len(names)} clips: {len(lists['proxy'])} proxy-depth, {len(lists['true'])} true-depth "
          f"(names in {out}/proxy.txt, true.txt)", flush=True)


def true_points(od: Path, clip: dict, n: int = 262_144) -> np.ndarray:
    """The true surface seen along the orbit, in the proxy's frame: every 4th target depth map back-projected
    (gso_render.py's depth_map: camera z, inf off the object) and moved through gt_camera's similarity."""
    name = clip["object"]
    gt = json.loads((V2 / "controls" / name / "gt_cameras.json").read_text())
    hero_p, hero_g = campath.PathCamera.from_json(gt["hero_camera"]), campath.PathCamera.from_json(clip["hero_gt"])
    rp, tp = hero_p.world_to_camera()
    rg, tg = hero_g.world_to_camera()
    s = hero_g.distance / hero_p.distance
    m, b = rg.T @ rp, rg.T @ (s * tp - tg)
    views = json.loads((od / "gt_cameras.json").read_text())["views"]
    pts = []
    for v in views[::4]:
        z = np.load(od / "target" / "depth" / f"{v['name']}.npy").astype(np.float32)
        h, w = z.shape
        fl = (min(w, h) / 2) / np.tan(np.radians(v["fov"]) / 2)
        vv, uu = np.nonzero(np.isfinite(z))
        zz = z[vv, uu]
        xc = np.stack([(uu + 0.5 - w / 2) / fl * zz, (vv + 0.5 - h / 2) / fl * zz, zz], 1)
        r, t = np.asarray(v["rot"]), np.asarray(v["t"])
        pts.append((xc - t) @ r)  # GT frame: R^T (x_cam - t)
    xg = np.concatenate(pts)
    xp = (xg - b) @ m / s  # proxy frame: M^T (x_g - b) / s
    return xp[np.random.default_rng(0).choice(len(xp), min(n, len(xp)), replace=False)].astype(np.float32)


def points_ply(path: Path, xyz: np.ndarray) -> None:
    """A minimal opaque gray splat PLY, as texture_common.write_points_ply writes (depth ignores colors)."""
    rec = np.zeros(len(xyz), dtype=[(k, "<f4") for k in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                                          "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3")])
    for i, k in enumerate("xyz"):
        rec[k] = xyz[:, i]
    for i in range(3):
        rec[f"scale_{i}"] = np.log(0.003)
    rec["opacity"], rec["rot_0"] = 5.0, 1.0
    splat.write_ply(path, rec)


if args.step == "controls":
    async def run() -> None:
        from giro.comfy import server
        from giro.comfy.client import ComfyClient, Done

        todo = [od for od in clips() if not (od / "control" / f"f{FRAMES - 1:03d}.png").exists()]
        with await asyncio.to_thread(server.Lease, args.gpu) as lease:
            async with ComfyClient(lease.url) as comfy:
                try:
                    for k, od in enumerate(todo):
                        clip = json.loads((od / "clip.json").read_text())
                        ply = od / f"{clip['control']}.ply"
                        if clip["control"] == "proxy":
                            write_points_ply(V2 / "controls" / clip["object"] / "work", ply)
                        elif (od / "target" / "done").exists():
                            points_ply(ply, true_points(od, clip))
                        else:
                            print(f"{od.name}: true-depth clip without target depth yet, skipped", flush=True)
                            continue
                        cams = [campath.PathCamera.from_json(c).render_json() for c in clip["cameras"]]
                        wf = {"proxy": {"class_type": "GiroLoadSplat", "inputs": {"path": str(ply)}},
                              "depth": {"class_type": "GiroRenderSplatCameras", "inputs": {
                                  "splat": ["proxy", 0], "width": W, "height": H, "cameras": json.dumps(cams),
                                  "render_style": "depth", "background": "#000000"}},
                              "save": {"class_type": "SaveImage", "inputs": {
                                  "images": ["depth", 0], "filename_prefix": f"giro/routec/{od.name}/depth"}}}
                        done = None
                        async for ev in comfy.run(wf):
                            if isinstance(ev, Done):
                                done = ev
                        (od / "control").mkdir(exist_ok=True)
                        for i, im in enumerate(sorted(done.outputs["save"]["images"], key=lambda d: d["filename"])):
                            await comfy.download(im, od / "control" / f"f{i:03d}.png")
                        print(f"[{k + 1}/{len(todo)}] {od.name}: {clip['control']} depth", flush=True)
                finally:
                    await comfy.free()

    asyncio.run(run())


def on_black(path: Path) -> np.ndarray:
    im = np.asarray(Image.open(path).convert("RGBA"), np.float32) / 255
    return (im[..., :3] * im[..., 3:] * 255 + 0.5).astype(np.uint8)


def encode(frames: list[np.ndarray], path: Path) -> None:
    """Near-lossless H.264 at Wan's 16 fps."""
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", "16", "-i", "-", "-c:v", "libx264", "-crf", "10", "-preset", "slow", "-pix_fmt", "yuv420p",
                          str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed on {path}")


if args.step == "pack":
    cdir = out / "clips"
    cdir.mkdir(exist_ok=True)
    rows = []
    for od in clips():
        names = [f"f{i:03d}.png" for i in range(FRAMES)]
        if not all((od / "target" / n).exists() and (od / "control" / n).exists() for n in names):
            continue
        n = od.name
        if not (cdir / f"{n}_control.mp4").exists():
            tgt = [on_black(od / "target" / f) for f in names]
            Image.fromarray(tgt[0]).save(cdir / f"{n}_ref.png")
            encode(tgt, cdir / f"{n}.mp4")
            encode([np.asarray(Image.open(od / "control" / f).convert("RGB")) for f in names], cdir / f"{n}_control.mp4")
        clip = json.loads((od / "clip.json").read_text())
        rows.append({"video": f"clips/{n}.mp4", "control_video": f"clips/{n}_control.mp4",
                     "reference_image": f"clips/{n}_ref.png", "prompt": workflows.PROXY_ORBIT_PROMPT,
                     "control": clip["control"], "preset": clip["preset"]})
    with open(out / "metadata.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"packed {len(rows)} clips into {out}/metadata.csv", flush=True)
