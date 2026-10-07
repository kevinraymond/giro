"""True-view test for Route B (board #3726): is the softness/ghosting of direct optimization caused by the
generated views disagreeing, or by the trainer and the sparse cameras? On a Google Scanned Object from the v2
view-LoRA data, the same direct_splat.py run is trained on (T) the object's TRUE renders at 16 of its 24
cameras, (G) the GSO view LoRA's generated views at the same 16 cameras, (P) the hero-painted mesh's renders
only, and scored against the true renders at the 8 held-out cameras (by eye: sheet.jpg; LPIPS, PSNR).
If T is crisp and G is not, the inputs are the problem; if T is soft too, 16 views are too sparse.

    gso_truth_test.py prep NAME OUT GPU      route-like layout (direct_splat.py's WORK and ANGLES dirs)
    gso_truth_test.py lora NAME OUT GPU      the view LoRA's views at the training cameras (ComfyUI)
    gso_truth_test.py eval NAME OUT GPU      render OUT/NAME/<run>/final.ply at the held-out cameras, score, sheet

From V2/controls/NAME: work/mesh.npz (Pixal3D on the hero, decimated), target/tNN.png (the true render, RGBA,
1080x1440), control/tNN.png (image 1 as the LoRA was trained: the hero-painted mesh on black), gt_cameras.json
(each view's camera in the mesh's frame, the hero's camera); V2/renders/NAME/hero_rgba.png. Everything is used
at 768x1024 (the LoRA's size; the cameras' fov is over the smaller side, so resizing keeps them exact).

prep writes OUT/NAME/: route.json (no drops), work/{mesh.npz, anchor_cameras.json (training cameras),
attempt/dataset (188 renders of the hero-painted mesh on project_texture.py's rings + the hero, COLMAP text)},
angles-true/{tNN.png, raw/subject/anchors/tNN.png}, split.json. lora writes angles-lora/ the same way (best of
up to 4 seeds by silhouette IoU with image 1, as lora_anchors.py).
"""
import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from giro import path as campath
from giro.stages.fallback import _qvec
from gso_dataset import caption
from texture_common import PoseCamera, Source, render_points, samples, view_lora_workflow

V2 = Path.home() / "ai" / "datasets" / "v2"
W, H = 768, 1024
SS = 2
RINGS = [(-20, 24), (0, 48), (20, 48), (40, 36), (60, 24), (80, 8)]  # project_texture.py's
BASE_WEIGHT = 0.02  # gso_controls.py's: Pixal3D's own color where the hero's weight fades out
ap = argparse.ArgumentParser()
ap.add_argument("step", choices=["prep", "lora", "joint", "lowres", "wan", "selfeval", "eval"])
ap.add_argument("name")
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--lora", default="qwen/gso-view-v1-750.safetensors")
ap.add_argument("--runs", default="T,T0,G,P", help="eval: run dirs under OUT/NAME, in sheet order")
ap.add_argument("--tag", default="joint", help="joint: the views go to OUT/NAME/angles-<tag>")
ap.add_argument("--split", choices=["train", "held"], default="train",
                help="lora/joint: generate at the training cameras or at the held-out ones (angles-<tag>-held: self-consistency)")
ap.add_argument("--pairs", default="Tc:true-held,Gc:lora-held,Jc-500:joint-500-held,Wc:wan-held",
                help="selfeval: RUN:TAG,... = OUT/NAME/RUN/final.ply scored against the views in OUT/NAME/angles-TAG")
ap.add_argument("--hold-every", type=int, default=5, help="wan: every Nth frame (from 2) is held out of training")
ap.add_argument("--no-lightning", action="store_true", help="joint: 25 steps, cfg 3 instead of Lightning's 4")
args = ap.parse_args()
dev = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(dev)
src = V2 / "controls" / args.name
out = args.out.resolve() / args.name
gt = json.loads((src / "gt_cameras.json").read_text())
views = gt["views"]
train = [v for i, v in enumerate(views) if i % 3 != 1]
held = [v for i, v in enumerate(views) if i % 3 == 1]
sel = held if args.split == "held" else train
sfx = "-held" if args.split == "held" else ""


def write_cams(adir: Path, cams: dict) -> None:
    """adir/cameras.json in load_cameras' format (direct_splat.py --cameras, selfeval)."""
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "cameras.json").write_text(json.dumps({"cameras": {n: {"camera": c.to_json()} for n, c in cams.items()}}, indent=1))


def gso_cams(vs: list) -> dict:
    return {v["name"]: PoseCamera.of(campath.PathCamera.from_json(v["proxy_camera"])) for v in vs}


def load_splat(path: Path) -> dict:
    """A trained splat's PLY as gsplat's tensors (on dev)."""
    from giro import splat

    s_ = splat.read_ply(path)
    g = lambda *k: torch.from_numpy(np.stack([s_[x] for x in k], 1)).float().to(dev)  # noqa: E731
    n = len(s_)
    return {"means": g("x", "y", "z"), "scales": torch.exp(g("scale_0", "scale_1", "scale_2")),
            "quats": g("rot_0", "rot_1", "rot_2", "rot_3"), "opac": torch.sigmoid(g("opacity")[:, 0]),
            "sh": torch.cat([g("f_dc_0", "f_dc_1", "f_dc_2")[:, None],
                             g(*[f"f_rest_{i}" for i in range(45)]).reshape(n, 3, 15).permute(0, 2, 1)], 1)}


def draw_splat(sp: dict, cam, w: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    from gsplat import rasterization

    r, t = cam.world_to_camera()
    vm = torch.eye(4, device=dev)
    vm[:3, :3], vm[:3, 3] = torch.tensor(np.asarray(r), device=dev), torch.tensor(np.asarray(t), device=dev)
    f = cam.focal(w, h)
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], device=dev, dtype=torch.float32)
    img, a, _ = rasterization(sp["means"], sp["quats"], sp["scales"], sp["opac"], sp["sh"], vm[None], K[None], w, h,
                              sh_degree=3, rasterize_mode="antialiased")
    return img[0].clamp(0, 1), a[0]
hero_cam = campath.PathCamera.from_json(gt["hero_camera"])


def rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """An RGBA render at WxH: colors on black, alpha > 0.5 as the mask."""
    im = np.asarray(Image.open(path).convert("RGBA").resize((W, H), Image.LANCZOS), np.float32) / 255
    a = im[..., 3:]
    return (im[..., :3] * a * 255).astype(np.uint8), a[..., 0] > 0.5


def save_view(adir: Path, name: str, rgb: np.ndarray, mask: np.ndarray) -> None:
    (adir / "raw" / "subject" / "anchors").mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(adir / f"{name}.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(adir / "raw" / "subject" / "anchors" / f"{name}.png")


def hero_ref(hero: np.ndarray, mask: np.ndarray, fill: float = 0.65) -> Image.Image:
    """lora_anchors.py's image 2: the subject on gray 127, its larger extent `fill` of the frame."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    m = mask[y0:y1, x0:x1, None].astype(np.float32)
    sub = Image.fromarray((hero[y0:y1, x0:x1].astype(np.float32) * m + 127 * (1 - m)).astype(np.uint8))
    s = fill * min(W / sub.width, H / sub.height)
    sub = sub.resize((max(1, round(sub.width * s)), max(1, round(sub.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", (W, H), (127, 127, 127))
    canvas.paste(sub, ((W - sub.width) // 2, (H - sub.height) // 2))
    return canvas


if args.step == "prep":
    work, ds = out / "work", out / "work" / "attempt" / "dataset"
    for d in (ds / "sparse" / "0", ds / "images" / "frames", ds / "masks" / "frames", ds / "images" / "hero", ds / "masks" / "hero"):
        d.mkdir(parents=True, exist_ok=True)
    if not (work / "mesh.npz").exists():
        (work / "mesh.npz").symlink_to(src / "work" / "mesh.npz")
    (out / "route.json").write_text(json.dumps({"anchors": {"silhouette_dropped": [], "color_dropped": []}}))
    (out / "split.json").write_text(json.dumps({"train": [v["name"] for v in train], "held": [v["name"] for v in held]}))
    cams = {v["name"]: {"camera": PoseCamera.of(campath.PathCamera.from_json(v["proxy_camera"])).to_json()} for v in train}
    (work / "anchor_cameras.json").write_text(json.dumps({"cameras": cams}, indent=1))
    for v in train:
        save_view(out / "angles-true", v["name"], *rgba(src / "target" / f"{v['name']}.png"))
    for v in held:
        save_view(out / "angles-true-held", v["name"], *rgba(src / "target" / f"{v['name']}.png"))
    write_cams(out / "angles-true", gso_cams(train))
    write_cams(out / "angles-true-held", gso_cams(held))
    hero, hmask = rgba(V2 / "renders" / args.name / "hero_rgba.png")
    Image.fromarray(hero).save(ds / "images" / "hero" / "hero.png")
    Image.fromarray(hmask.astype(np.uint8) * 255).save(ds / "masks" / "hero" / "hero.png")
    hero_ref(hero, hmask).save(out / "hero_ref.png")
    # The prior: the mesh painted from the hero (Pixal3D's colors elsewhere), as the route's renders.
    xyz, nrm, base, _ = samples(work, 3_000_000, dev)
    col, wt = Source("hero", Image.fromarray(hero), hmask, hero_cam, 1.0, dev).paint(xyz, nrm, 0.012)
    rgb = (col * wt[:, None] + base * BASE_WEIGHT) / (wt[:, None] + BASE_WEIGHT)
    f = hero_cam.focal(W, H)
    lines_c = [f"1 PINHOLE {W} {H} {f} {f} {W / 2} {H / 2}"]
    lines_i = []
    ring = [replace(hero_cam, yaw=hero_cam.yaw + 360 * k / n, pitch=float(p)) for p, n in RINGS for k in range(n)]
    for i, cam in enumerate(ring):
        img, m = render_points(xyz, rgb, cam, W * SS, H * SS, nrm=nrm)
        m = m.astype(np.float32)
        blocks = lambda a: a.reshape(H, SS, W, SS, *a.shape[2:]).sum((1, 3))  # noqa: E731
        cov = blocks(m) / SS**2
        color = blocks(img * m[..., None]) / np.maximum(blocks(m), 1e-6)[..., None]
        Image.fromarray((color * cov[..., None] * 255).clip(0, 255).astype(np.uint8)).save(ds / "images" / "frames" / f"{i:05d}.png")
        Image.fromarray((cov * 255).astype(np.uint8)).save(ds / "masks" / "frames" / f"{i:05d}.png")
        r, t = cam.world_to_camera()
        lines_i += [f"{i + 1} {' '.join(map(str, _qvec(np.asarray(r))))} {' '.join(map(str, t))} 1 frames/{i:05d}.png", ""]
    r, t = hero_cam.world_to_camera()
    lines_i += [f"{len(ring) + 1} {' '.join(map(str, _qvec(np.asarray(r))))} {' '.join(map(str, t))} 1 hero/hero.png", ""]
    (ds / "sparse" / "0" / "cameras.txt").write_text("\n".join(lines_c) + "\n")
    (ds / "sparse" / "0" / "images.txt").write_text("\n".join(lines_i) + "\n")
    print(f"{args.name}: {len(train)} training views, {len(held)} held out, {len(ring)} prior renders", flush=True)

elif args.step == "lora":
    adir = out / f"angles-lora{sfx}"
    (adir / "input").mkdir(parents=True, exist_ok=True)
    write_cams(adir, gso_cams(sel))
    # Image 1 per training camera as gso_controls.py renders it (v2 renders its controls late, so they may not
    # exist yet): the hero-painted mesh on black, 2x supersampled; its coverage is the silhouette to hold.
    hero, hmask = rgba(V2 / "renders" / args.name / "hero_rgba.png")
    xyz, nrm, base, _ = samples(out / "work", 3_000_000, dev)
    col, wt = Source("hero", Image.fromarray(hero), hmask, hero_cam, 1.0, dev).paint(xyz, nrm, 0.012)
    rgb = (col * wt[:, None] + base * BASE_WEIGHT) / (wt[:, None] + BASE_WEIGHT)
    for v in sel:
        img, m = render_points(xyz, rgb, campath.PathCamera.from_json(v["proxy_camera"]), W * SS, H * SS, nrm=nrm)
        m = m.astype(np.float32)
        blocks = lambda a: a.reshape(H, SS, W, SS, *a.shape[2:]).sum((1, 3))  # noqa: E731
        cov = blocks(m) / SS**2
        color = blocks(img * m[..., None]) / np.maximum(blocks(m), 1e-6)[..., None]
        Image.fromarray((color * cov[..., None] * 255).clip(0, 255).astype(np.uint8)).save(adir / "input" / f"{v['name']}.png")
        Image.fromarray((cov * 255).astype(np.uint8)).save(adir / "input" / f"{v['name']}_mask.png")
    del xyz, nrm, base, col, wt, rgb
    torch.cuda.empty_cache()

    async def generate() -> None:
        from giro.comfy import server
        from giro.comfy.client import ComfyClient, Done

        tries_dir = adir / "tries"
        tries_dir.mkdir(parents=True, exist_ok=True)
        with await asyncio.to_thread(server.Lease, args.gpu) as lease:
            async with ComfyClient(lease.url) as comfy:
                try:
                    ref = await comfy.upload_image(out / "hero_ref.png", subfolder="giro/truth_test")
                    report = {}
                    for v in sel:
                        n = v["name"]
                        c1p = adir / "input" / f"{n}.png"
                        target = np.asarray(Image.open(adir / "input" / f"{n}_mask.png")) > 127
                        c1 = await comfy.upload_image(c1p, subfolder="giro/truth_test_in")
                        scores = []
                        for t in range(4):
                            f = tries_dir / f"{n}-{t}.png"
                            if not f.exists():
                                wf = view_lora_workflow(c1, ref, caption(v["rel_yaw"], v["pitch"]), args.lora, 11 + int(n[1:]) + 1000 * t,
                                                        lightning=True, prefix="giro/truth_test")
                                done = None
                                async for ev in comfy.run(wf):
                                    if isinstance(ev, Done):
                                        done = ev
                                await comfy.download(done.outputs["save"]["images"][0], f)
                            img = np.asarray(Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS))
                            sil = ndimage.binary_fill_holes(ndimage.binary_opening(img.max(-1) > 3, iterations=2))
                            scores.append(float((sil & target).sum() / max((sil | target).sum(), 1)))
                            if scores[-1] >= 0.75:
                                break
                        best = int(np.argmax(scores))
                        img = np.asarray(Image.open(tries_dir / f"{n}-{best}.png").convert("RGB").resize((W, H), Image.LANCZOS))
                        sil = ndimage.binary_fill_holes(ndimage.binary_opening(img.max(-1) > 3, iterations=2))
                        save_view(adir, n, (img * sil[..., None]).astype(np.uint8), sil)
                        report[n] = {"ious": [round(s, 3) for s in scores], "pick": best}
                        print(f"{n}: yaw {v['rel_yaw']:.0f} pitch {v['pitch']:.0f} IoU {report[n]['ious']}", flush=True)
                    (adir / "report.json").write_text(json.dumps(report, indent=1))
                finally:
                    await comfy.free()

    asyncio.run(generate())

elif args.step == "joint":
    # The joint LoRA (board #3735): the 16 training cameras as 4 groups of 4 neighbors (joint_dataset.py's
    # grouping and caption), each group one 2x2 grid of image 1s (from the lora step's renders) -> one pass ->
    # the 4 panels cut out at 384x512. Needs `lora` run first (angles-lora/input).
    from joint_dataset import caption as joint_caption
    from joint_dataset import groups_of_four
    import random as _random

    adir = out / f"angles-{args.tag}{sfx}"
    write_cams(adir, gso_cams(sel))
    gdir = adir / "grids"
    gdir.mkdir(parents=True, exist_ok=True)
    pw, ph = W // 2, H // 2
    views_g = [{"id": v["name"], "yaw": v["rel_yaw"], "pitch": v["pitch"]} for v in sel]
    groups = groups_of_four(views_g, _random.Random(0))

    async def generate_joint() -> None:
        from giro.comfy import server
        from giro.comfy.client import ComfyClient, Done

        with await asyncio.to_thread(server.Lease, args.gpu) as lease:
            async with ComfyClient(lease.url) as comfy:
                try:
                    ref = await comfy.upload_image(out / "hero_ref.png", subfolder="giro/truth_test")
                    for gi, g in enumerate(groups):
                        grid_in = Image.new("RGB", (W, H))
                        for i, v in enumerate(g):
                            im = Image.open(out / f"angles-lora{sfx}" / "input" / f"{v['id']}.png").convert("RGB").resize((pw, ph), Image.LANCZOS)
                            grid_in.paste(im, ((i % 2) * pw, (i // 2) * ph))
                        grid_in.save(gdir / f"g{gi}-input.png")
                        c1 = await comfy.upload_image(gdir / f"g{gi}-input.png", subfolder="giro/truth_test_in")
                        wf = view_lora_workflow(c1, ref, joint_caption([(v["yaw"], v["pitch"]) for v in g]), args.lora, 11 + gi,
                                                lightning=not args.no_lightning, prefix="giro/truth_joint")
                        done = None
                        async for ev in comfy.run(wf):
                            if isinstance(ev, Done):
                                done = ev
                        await comfy.download(done.outputs["save"]["images"][0], gdir / f"g{gi}.png")
                        grid_out = Image.open(gdir / f"g{gi}.png").convert("RGB").resize((W, H), Image.LANCZOS)
                        ious = []
                        for i, v in enumerate(g):
                            panel = np.asarray(grid_out.crop(((i % 2) * pw, (i // 2) * ph, (i % 2 + 1) * pw, (i // 2 + 1) * ph)))
                            sil = ndimage.binary_fill_holes(ndimage.binary_opening(panel.max(-1) > 3, iterations=2))
                            target = np.asarray(Image.open(out / f"angles-lora{sfx}" / "input" / f"{v['id']}_mask.png").resize((pw, ph))) > 127
                            ious.append(round(float((sil & target).sum() / max((sil | target).sum(), 1)), 3))
                            save_view(adir, v["id"], (panel * sil[..., None]).astype(np.uint8), sil)
                        print(f"group {gi} {[v['id'] for v in g]}: panel IoU {ious}", flush=True)
                finally:
                    await comfy.free()

    asyncio.run(generate_joint())

elif args.step == "lowres":
    # The true and the one-by-one LoRA views at the joint panels' 384x512, so all three train at one resolution.
    for src_tag in ("true", "lora"):
        sdir, ddir = out / f"angles-{src_tag}", out / f"angles-{src_tag}384"
        for v in train:
            n = v["name"]
            img = Image.open(sdir / f"{n}.png").convert("RGB").resize((W // 2, H // 2), Image.LANCZOS)
            m = Image.open(sdir / "raw" / "subject" / "anchors" / f"{n}.png").convert("L").resize((W // 2, H // 2), Image.BILINEAR)
            save_view(ddir, n, np.asarray(img), np.asarray(m) > 127)


elif args.step == "wan":
    # A Wan 2.2 Fun Control orbit (giro's proxy orbit, wan22-control) driven by this object's mesh in its own frame, so
    # all 81 frames come with exact cameras: hero = frame 0, giro.path's spiral (2 turns, pitch rising to 45), the
    # mesh's surface splat as the depth proxy. Frames held out of training: i % --hold-every == 2 (angles-wan-held).
    from giro import workflows
    from texture_common import write_points_ply

    WW, WH = 576, 768
    wdir = out / "wan"
    (wdir / "silhouettes").mkdir(parents=True, exist_ok=True)
    ply = wdir / "proxy.ply"
    if not ply.exists():
        write_points_ply(out / "work", ply)
    hero, _ = rgba(V2 / "renders" / args.name / "hero_rgba.png")
    Image.fromarray(hero).resize((WW, WH), Image.LANCZOS).save(wdir / "hero.png")
    cams = campath.plan(hero_cam, "spiral", 81, turns=2.0, pitch_end=45.0)

    async def generate_wan() -> None:
        from giro.comfy import server
        from giro.comfy.client import ComfyClient, Done

        with await asyncio.to_thread(server.Lease, args.gpu) as lease:
            async with ComfyClient(lease.url) as comfy:
                try:
                    h = await comfy.upload_image(wdir / "hero.png", subfolder="giro/truth_wan")
                    wf = workflows.build_orbit(
                        "wan22-control", video="fun_control", steps=20, prompt=workflows.PROXY_ORBIT_PROMPT, width=WW, height=WH,
                        length=81, seed=7, image=h, proxy=str(ply.resolve()), cameras=json.dumps([c.render_json() for c in cams]),
                        silhouette_paths=json.dumps([str((wdir / "silhouettes" / f"{i:05d}.png").resolve()) for i in range(81)]),
                        depth_prefix=f"giro/truth_wan/{args.name}/depth", output_prefix=f"giro/truth_wan/{args.name}/frame")
                    done = None
                    async for ev in comfy.run(wf):
                        if isinstance(ev, Done):
                            done = ev
                    for i, im in enumerate(sorted(done.outputs["frames"]["images"], key=lambda d: d["filename"])):
                        await comfy.download(im, wdir / f"f{i:03d}.png")
                finally:
                    await comfy.free()

    if not (wdir / "f080.png").exists():
        asyncio.run(generate_wan())
    tr_c, ho_c = {}, {}
    for i, cam in enumerate(cams):
        n = f"f{i:03d}"
        img = np.asarray(Image.open(wdir / f"{n}.png").convert("RGB"))
        fg = ndimage.binary_fill_holes(ndimage.binary_opening(img.max(-1) > 12, iterations=2))
        sp = wdir / "silhouettes" / f"{i:05d}.png"
        if sp.exists():  # the frame's subject, kept near the proxy's silhouette (Wan may add a floor or a glow)
            sil = np.asarray(Image.open(sp).convert("L").resize((img.shape[1], img.shape[0]))) > 127
            fg &= ndimage.binary_dilation(sil, iterations=12)
        held_frame = i % args.hold_every == 2
        save_view(out / ("angles-wan-held" if held_frame else "angles-wan"), n, (img * fg[..., None]).astype(np.uint8), fg)
        (ho_c if held_frame else tr_c)[n] = PoseCamera.of(cam)
    write_cams(out / "angles-wan", tr_c)
    write_cams(out / "angles-wan-held", ho_c)
    print(f"{args.name}: {len(tr_c)} Wan frames to train on, {len(ho_c)} held out", flush=True)

elif args.step == "selfeval":
    # Self-consistency: each splat against the generator's OWN views at cameras it did not train on (true views: the
    # ceiling). A generator whose views agree in 3D scores like the true views; one that invents each view anew does not.
    import lpips

    from texture_common import load_cameras

    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
    rows, summary = [], {}
    with torch.no_grad():
        for pair in args.pairs.split(","):
            run, tag = pair.split(":")
            adir, ply = out / f"angles-{tag}", out / run / "final.ply"
            if not ply.exists() or not (adir / "cameras.json").exists():
                print(f"{pair}: missing ({ply.exists()=}, cameras {(adir / 'cameras.json').exists()})", flush=True)
                continue
            sp = load_splat(ply)
            sc, tiles = {"lpips": [], "psnr": []}, []
            for n, cam in sorted(load_cameras(adir / "cameras.json").items()):
                g = np.asarray(Image.open(adir / f"{n}.png").convert("RGB"), np.float32) / 255
                gm = np.asarray(Image.open(adir / "raw" / "subject" / "anchors" / f"{n}.png")) > 127
                h_, w_ = gm.shape
                rgb_, a = draw_splat(sp, cam, w_, h_)
                pred = rgb_ + (1 - a) * 0.5
                m = torch.from_numpy(gm).to(dev)[..., None].float()
                tgt = torch.from_numpy(g).to(dev) * m + (1 - m) * 0.5
                ys, xs = np.nonzero(ndimage.binary_dilation(gm, iterations=16))
                box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
                sc["lpips"].append(float(lp(pred[box].permute(2, 0, 1)[None] * 2 - 1, tgt[box].permute(2, 0, 1)[None] * 2 - 1)))
                inside = ((m > 0) | (a > 0.5)).expand_as(pred)
                sc["psnr"].append(10 * np.log10(1 / max(float(((pred - tgt)[inside] ** 2).mean()), 1e-10)))
                if len(tiles) < 6:
                    tiles.append(torch.cat([tgt, pred], 1).cpu().numpy())
            summary[pair] = {k: round(float(np.mean(x)), 4) for k, x in sc.items()} | {"views": len(sc["lpips"])}
            rows.append((pair, tiles))
            print(f"{args.name} {pair}: {summary[pair]}", flush=True)
    (out / "selfeval.json").write_text(json.dumps(summary, indent=1))
    th = 256
    sheet = Image.new("RGB", (180 + 6 * th * 3 // 2, th * len(rows)), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for r, (pair, tiles) in enumerate(rows):
        d.text((4, r * th + 4), pair, fill=(255, 255, 0))
        d.text((4, r * th + 20), f"LPIPS {summary[pair]['lpips']:.3f}", fill=(255, 255, 255))
        d.text((4, r * th + 36), "(view | splat)", fill=(200, 200, 200))
        for c, t in enumerate(tiles):
            sheet.paste(Image.fromarray((t * 255).astype(np.uint8)).resize((th * 3 // 2, th)), (180 + c * th * 3 // 2, r * th))
    sheet.save(out / "selfeval.jpg", quality=90)

else:  # eval
    import lpips
    from gsplat import rasterization

    from giro import splat

    lp = lpips.LPIPS(net="vgg", verbose=False).to(dev)
    runs = [r for r in args.runs.split(",") if (out / r / "final.ply").exists()]

    def load(path: Path) -> dict:
        s = splat.read_ply(path)
        g = lambda *k: torch.from_numpy(np.stack([s[x] for x in k], 1)).float().to(dev)  # noqa: E731
        n = len(s)
        return {"means": g("x", "y", "z"), "scales": torch.exp(g("scale_0", "scale_1", "scale_2")),
                "quats": g("rot_0", "rot_1", "rot_2", "rot_3"), "opac": torch.sigmoid(g("opacity")[:, 0]),
                "sh": torch.cat([g("f_dc_0", "f_dc_1", "f_dc_2")[:, None],
                                 g(*[f"f_rest_{i}" for i in range(45)]).reshape(n, 3, 15).permute(0, 2, 1)], 1)}

    def draw(sp: dict, cam: PoseCamera) -> tuple[torch.Tensor, torch.Tensor]:
        r, t = cam.world_to_camera()
        vm = torch.eye(4, device=dev)
        vm[:3, :3], vm[:3, 3] = torch.tensor(np.asarray(r), device=dev), torch.tensor(np.asarray(t), device=dev)
        f = cam.focal(W, H)
        K = torch.tensor([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], device=dev, dtype=torch.float32)
        img, a, _ = rasterization(sp["means"], sp["quats"], sp["scales"], sp["opac"], sp["sh"], vm[None], K[None], W, H,
                                  sh_degree=3, rasterize_mode="antialiased")
        return img[0].clamp(0, 1), a[0]

    splats = {r: load(out / r / "final.ply") for r in runs}
    metrics = {r: {"lpips": [], "psnr": []} for r in runs}
    tiles = {r: [] for r in ["GT"] + runs}
    with torch.no_grad():
        for v in held:
            cam = PoseCamera.of(campath.PathCamera.from_json(v["proxy_camera"]))
            g_rgb, g_m = rgba(src / "target" / f"{v['name']}.png")
            m = torch.from_numpy(g_m).to(dev)[..., None].float()
            gt_img = torch.from_numpy(g_rgb).to(dev).float() / 255 + (1 - m) * 0.5
            ys, xs = np.nonzero(ndimage.binary_dilation(g_m, iterations=16))
            box = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
            tiles["GT"].append(gt_img)
            for r in runs:
                rgb, a = draw(splats[r], cam)
                pred = rgb + (1 - a) * 0.5
                tiles[r].append(pred)
                pb, gb = pred[box].permute(2, 0, 1)[None] * 2 - 1, gt_img[box].permute(2, 0, 1)[None] * 2 - 1
                metrics[r]["lpips"].append(float(lp(pb, gb)))
                inside = ((m > 0) | (a > 0.5)).expand_as(pred)
                mse = float(((pred - gt_img)[inside] ** 2).mean())
                metrics[r]["psnr"].append(10 * np.log10(1 / max(mse, 1e-10)))
    summary = {r: {k: round(float(np.mean(x)), 4) for k, x in mm.items()} for r, mm in metrics.items()}
    (out / "eval.json").write_text(json.dumps({"held": [v["name"] for v in held], "mean": summary, "per_view": metrics}, indent=1))
    th, tw = 384, 288
    sheet = Image.new("RGB", (100 + tw * len(held), th * len(tiles)), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    for row, (r, ims) in enumerate(tiles.items()):
        d.text((4, row * th + 4), r, fill=(255, 255, 0))
        if r in summary:
            d.text((4, row * th + 20), f"LPIPS {summary[r]['lpips']:.3f}", fill=(255, 255, 255))
            d.text((4, row * th + 36), f"PSNR {summary[r]['psnr']:.1f}", fill=(255, 255, 255))
        for col, im in enumerate(ims):
            sheet.paste(Image.fromarray((im.cpu().numpy() * 255).astype(np.uint8)).resize((tw, th), Image.LANCZOS), (100 + col * tw, row * th))
    sheet.save(out / "sheet.jpg", quality=90)
    print(args.name, json.dumps(summary), flush=True)
