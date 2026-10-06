"""Ground-truth renders of Google Scanned Objects (CC-BY 4.0) for the view-LoRA pilot (board
#3699), in two passes, with cameras in giro's path convention.

    uv run --project ~/ai/datasets/gso/render python gso_render.py GSO_DIR OUT_DIR SPLIT_CSV
        [--names a,b | --only N] [--targets 12] [--aligned CONTROLS_DIR] [--samples 64] [--threads 12]
        [--plan strata|rings] [--size 768x1024] [--depth all|heldout|none] [--device cpu|gpu] [--shard J/K]

v2 (board #3709): a manifest row with a `path` column names its model file (Objaverse .glb, Poly
Haven .gltf, or a GSO model directory); glTF is y up and Blender's importer turns it z up, the GSO
frame. glTF materials keep their own roughness and metal (GSO scans get a fixed matte roughness).
--plan rings lays the targets on 5 rings (pitch -15..65, jittered) instead of stratified pairs.

Runs in its own environment (PyPI bpy, Blender's Cycles on the CPU; trimesh + Embree for depth; no
giro imports), so the camera math below repeats giro.path.PathCamera: cameras live in the splat
frame (y down, a camera at yaw 0 and pitch 0 looks along +z, positive pitch looks down from above;
positive yaw moves the camera to the hero camera's right). The object is moved into that frame (GSO
is z up), centered on its box and scaled to height 1, like the texture route's proxy samples.

Pass 1 (default): OUT_DIR/<name>/hero.png (RGB on a gray backdrop, like a photo), hero_rgba.png,
depth/hero.npy and cameras.json: the hero as a PathCamera and the plan of target views (yaw relative
to the hero, pitch), as the texture route lays out its rings (project_texture.py: yaw = hero yaw +
offset, pitch per ring). With --render-plan the plan is also rendered here (t00.png..., GT frame).

Pass 2 (--aligned CONTROLS_DIR, after gso_controls.py): the targets rendered at the cameras the
control step mapped into this frame (CONTROLS_DIR/<name>/gt_cameras.json: rot, t as COLMAP's
world-to-camera, fov), so a target lines up with its control render up to the proxy's shape error:
CONTROLS_DIR/<name>/target/<view>.png (RGBA) and target/depth/<view>.npy.

Depth: float16 camera z (along the view direction, object units), inf off the object.
"""
import argparse
import os
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import bpy
import numpy as np
import trimesh
from mathutils import Matrix, Vector
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("gso", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("split", type=Path, help="manifest with name and split columns")
ap.add_argument("--only", type=int, default=0, help="the first N objects (0: all)")
ap.add_argument("--names", default="", help="comma-separated object names instead of the split")
ap.add_argument("--targets", type=int, default=12, help="planned target views per object")
ap.add_argument("--render-plan", action="store_true", help="also render the plan in the GT frame (pass 1)")
ap.add_argument("--aligned", type=Path, help="pass 2: render the targets gso_controls.py mapped into this frame")
ap.add_argument("--samples", type=int, default=64)
ap.add_argument("--threads", type=int, default=12, help="Cycles CPU threads (meg: 32 threads ran the package to 100 C, Oct 5-6)")
ap.add_argument("--size", default="768x1024", help="WxH")
ap.add_argument("--fov", type=float, default=35.0, help="degrees over the smaller side (giro's hero camera)")
ap.add_argument("--fill", type=float, default=0.8, help="the object's bounding sphere spans this much of the smaller side")
ap.add_argument("--plan", choices=["strata", "rings"], default="strata", help="target layout (v1: strata)")
ap.add_argument("--depth", choices=["all", "heldout", "none"], default="all",
                help="depth maps for every object, only held-out ones (the evaluation needs them), or none")
ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Cycles on the CPU, or a GPU (OptiX, else CUDA)")
ap.add_argument("--shard", default="", metavar="J/K", help="every K-th object from the J-th (two GPUs)")
args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:])
W, H = (int(v) for v in args.size.split("x"))
BACKDROP = (0.5, 0.5, 0.5)
PITCHES = (-20.0, 60.0)

# splat frame (x, y down, z) <- GSO (x, y, z up): x_s = x, y_s = -z, z_s = y (a rotation, det 1)
S_FROM_G = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])


def forward(yaw: float, pitch: float) -> np.ndarray:
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([-math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])


def axes(cam: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Camera x (right), y (down), forward and position in the splat frame, for a path camera
    (yaw, pitch, distance, target) or a posed one (rot, t: COLMAP's world-to-camera)."""
    if "rot" in cam:
        r, t = np.asarray(cam["rot"]), np.asarray(cam["t"])
        return r[0], r[1], r[2], -r.T @ t
    f = forward(cam["yaw"], cam["pitch"])
    pos = np.asarray(cam["target"]) - cam["distance"] * f
    x = np.cross([0.0, 1.0, 0.0], f)
    x /= np.linalg.norm(x)
    return x, np.cross(f, x), f, pos


def camera_matrix(cam: dict) -> Matrix:
    """The Blender camera's world matrix (Blender world = GSO frame)."""
    x, y, f, pos = axes(cam)
    # Blender cameras look along -Z with +Y up in the image
    rot_s = np.stack([x, -y, -f], 1)  # columns: camera x, y, z axes in the splat frame
    g = S_FROM_G.T
    m = np.eye(4)
    m[:3, :3] = g @ rot_s
    m[:3, 3] = g @ pos
    return Matrix(m.tolist())


def setup_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    if args.device == "gpu":  # CUDA_VISIBLE_DEVICES picks the card
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
                prefs.get_devices()
                devs = [d for d in prefs.devices if d.type == kind]
            except TypeError:
                devs = []
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                sc.cycles.device = "GPU"
                print(f"Cycles on {kind}: {[d.name for d in devs]}", flush=True)
                break
    sc.cycles.samples = args.samples
    sc.cycles.use_denoising = True
    if sc.cycles.device == "GPU" and os.environ.get("GSO_GPU_DENOISE", "1") == "1":
        # OpenImageDenoise on the GPU: on the CPU it took most of a frame's time (Oct 6)
        sc.cycles.denoiser = "OPENIMAGEDENOISE"
        if hasattr(sc.cycles, "denoising_use_gpu"):
            sc.cycles.denoising_use_gpu = True
    sc.render.threads_mode = "FIXED"
    sc.render.threads = args.threads
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = W, H, 100
    sc.render.film_transparent = True
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    world = bpy.data.worlds.new("studio")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
    # soft key, fill and top lights, fixed to the object (an orbit moves the camera, not the lights)
    for name, loc, energy, size in (("key", (2.5, -2.5, 2.5), 450, 3.0), ("fill", (-3.0, -1.0, 1.2), 160, 4.0),
                                    ("top", (0.0, 1.5, 4.0), 220, 4.0)):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size = energy, size
        obj = bpy.data.objects.new(name, light)
        obj.location = loc
        obj.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
        sc.collection.objects.link(obj)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    sc.collection.objects.link(cam)
    sc.camera = cam
    # the fov spans the smaller side, as giro's cameras have it
    if W <= H:
        cam.data.sensor_fit, cam.data.angle_x = "HORIZONTAL", math.radians(args.fov)
    else:
        cam.data.sensor_fit, cam.data.angle_y = "VERTICAL", math.radians(args.fov)
    cam.data.clip_start, cam.data.clip_end = 0.01, 100


KEEP = {"key", "fill", "top", "cam"}


def load_object(model: Path) -> tuple[list, float]:
    """The object, centered on its box and scaled to height 1; returns its parts and bounding radius.
    `model` is a GSO model directory or a .glb/.gltf file."""
    for o in list(bpy.data.objects):
        if o.name not in KEEP:
            bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.images, bpy.data.armatures, bpy.data.actions):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)
    gltf = model.suffix.lower() in (".glb", ".gltf")
    if gltf:
        bpy.ops.import_scene.gltf(filepath=str(model))
    else:
        bpy.ops.wm.obj_import(filepath=str(model / "meshes" / "model.obj"), forward_axis="Y", up_axis="Z")
    bpy.context.view_layer.update()
    parts = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.data.vertices]
    for o in parts:  # skinned glTF meshes: the bind pose, not an armature's deformation
        for m in [m for m in o.modifiers if m.type == "ARMATURE"]:
            o.modifiers.remove(m)
    tex = model / "materials" / "textures" / "texture.png"  # GSO: the .mtl names it beside the .obj
    for img in bpy.data.images:
        if not gltf and img.source == "FILE" and not Path(bpy.path.abspath(img.filepath)).exists() and tex.exists():
            img.filepath = str(tex)
            img.reload()
    pts = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in parts])
    lo, hi = pts.min(0), pts.max(0)
    scale = 1.0 / (hi[2] - lo[2])
    center = (lo + hi) / 2
    for o in parts:
        mw = Matrix.Scale(scale, 4) @ Matrix.Translation(Vector(-center)) @ o.matrix_world
        o.parent = None  # glTF nodes hang under empties; flatten so the world matrix is ours
        o.matrix_world = mw
        for mat in o.data.materials if not gltf else ():  # scans: no shine from the default specular
            bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat and mat.use_nodes else None
            if bsdf is not None:
                bsdf.inputs["Roughness"].default_value = 0.7
    radius = float(np.linalg.norm((pts - center) * scale, axis=1).max())
    return parts, radius


def object_mesh(parts: list) -> trimesh.Trimesh:
    """The scene's triangles in world coordinates, for the depth maps (ray cast with Embree)."""
    vs, fs, n = [], [], 0
    for o in parts:
        me = o.data
        me.calc_loop_triangles()
        mw = np.array(o.matrix_world)
        v = np.array([v.co[:] for v in me.vertices]).reshape(-1, 3)
        f = np.array([t.vertices[:] for t in me.loop_triangles], dtype=np.int64).reshape(-1, 3)
        if not len(f):  # a part with no faces (points or loose edges, seen in Objaverse GLBs)
            continue
        vs.append(v @ mw[:3, :3].T + mw[:3, 3])
        fs.append(f + n)
        n += len(v)
    return trimesh.Trimesh(np.concatenate(vs), np.concatenate(fs), process=False)


def depth_map(mesh: trimesh.Trimesh, cam: dict) -> np.ndarray:
    """Camera z (along the view direction, object units) per pixel, inf off the object: what
    giro's project() calls z."""
    x, y, f, pos = axes(cam)
    fl = (min(W, H) / 2) / math.tan(math.radians(cam["fov"]) / 2)
    vv, uu = np.mgrid[0:H, 0:W] + 0.5
    d_s = ((uu - W / 2) / fl)[..., None] * x + ((vv - H / 2) / fl)[..., None] * y + f
    d_g = d_s.reshape(-1, 3) @ S_FROM_G  # rows: (S_FROM_G.T @ d)
    o_g = np.broadcast_to(S_FROM_G.T @ pos, d_g.shape)
    loc, ray, _ = mesh.ray.intersects_location(o_g, d_g / np.linalg.norm(d_g, axis=1, keepdims=True), multiple_hits=False)
    z = np.full(W * H, np.inf)
    z[ray] = (loc @ S_FROM_G.T - pos) @ f
    return z.reshape(H, W)


def render(cam: dict, path_png: Path, path_depth: Path | None, mesh: trimesh.Trimesh) -> None:
    sc = bpy.context.scene
    sc.camera.matrix_world = camera_matrix(cam)
    sc.render.filepath = str(path_png)
    bpy.ops.render.render(write_still=True)  # RGBA PNG, straight alpha, Standard view transform
    if path_depth is not None:
        path_depth.parent.mkdir(exist_ok=True)
        np.save(path_depth, depth_map(mesh, cam).astype(np.float16))


RINGS = ((-15.0, 0.18), (5.0, 0.26), (25.0, 0.24), (45.0, 0.2), (65.0, 0.12))  # pitch, share of the targets


def plan_cameras(rng: random.Random, radius: float) -> list[dict]:
    distance = radius / math.sin(math.radians(args.fov) / 2) / args.fill
    hero = {"name": "hero", "yaw": rng.uniform(0, 360), "pitch": rng.uniform(5, 20), "distance": distance,
            "target": [0.0, 0.0, 0.0], "fov": args.fov}
    cams = [hero]
    n = args.targets
    if args.plan == "rings":  # yaw stratified per ring from a random offset, pitch jittered +-5
        counts = [max(1, round(n * share)) for _, share in RINGS]
        counts[1] += n - sum(counts)
        k = 0
        for (pitch, _), m in zip(RINGS, counts):
            off = rng.random()
            for j in range(m):
                dyaw = (360 * (j + off) / m) % 360
                cams.append(hero | {"name": f"t{k:02d}", "yaw": hero["yaw"] + dyaw, "pitch": pitch + rng.uniform(-5, 5)})
                k += 1
        for c in cams:
            c["rel_yaw"] = round((c["yaw"] - hero["yaw"]) % 360, 2)
            c["caption_pitch"] = round(c["pitch"], 2)
        return cams
    pitch_bins = np.linspace(*PITCHES, n + 1)
    pitch_order = list(range(n))
    rng.shuffle(pitch_order)
    for k in range(n):  # yaw stratified over the circle, pitch stratified over the range, paired at random
        dyaw = (360 * (k + rng.random()) / n) % 360
        lo, hi = pitch_bins[pitch_order[k]], pitch_bins[pitch_order[k] + 1]
        cams.append(hero | {"name": f"t{k:02d}", "yaw": hero["yaw"] + dyaw, "pitch": rng.uniform(lo, hi)})
    for c in cams:
        c["rel_yaw"] = round((c["yaw"] - hero["yaw"]) % 360, 2)
        c["caption_pitch"] = round(c["pitch"], 2)
    return cams


def main() -> None:
    with open(args.split) as f:
        rows = {r["name"]: r for r in csv.DictReader(f)}
    if args.names:
        names = args.names.split(",")
    else:
        names = [n for n, r in rows.items() if r.get("split") in ("train", "heldout")]

    def model_of(name: str) -> Path:
        p = rows.get(name, {}).get("path")
        return Path(p).expanduser() if p else args.gso / "models" / name

    def depth_of(name: str, path: Path) -> Path | None:
        return path if args.depth == "all" or (args.depth == "heldout" and rows.get(name, {}).get("split") == "heldout") else None
    if args.only:
        names = names[:args.only]
    if args.shard:
        j, k = map(int, args.shard.split("/"))
        names = names[j::k]
    setup_scene()
    t_all = time.monotonic()
    for i, name in enumerate(names):
        t0 = time.monotonic()
        try:
            one(i, name, names, model_of, depth_of)
        except Exception as e:  # one broken model must not stop the pass (Oct 6: an Objaverse GLB did)
            print(f"[{i + 1}/{len(names)}] {name}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        dt = time.monotonic() - t0
        print(f"[{i + 1}/{len(names)}] {name}: {dt:.1f} s", flush=True)
    print(f"done {len(names)} objects in {time.monotonic() - t_all:.0f} s", flush=True)


def one(i: int, name: str, names: list, model_of, depth_of) -> None:
    """Render one object: its hero (and plan), or with --aligned its ground truth."""
    if True:
        if args.aligned:
            cd = args.aligned / name
            if not (cd / "gt_cameras.json").exists() or (cd / "target" / "done").exists():
                return
            parts, _ = load_object(model_of(name))
            mesh = object_mesh(parts)
            views = json.loads((cd / "gt_cameras.json").read_text())["views"]
            for c in views:
                render(c, cd / "target" / f"{c['name']}.png", depth_of(name, cd / "target" / "depth" / f"{c['name']}.npy"), mesh)
            (cd / "target" / "done").write_text("")
        else:
            od = args.out / name
            if (od / "cameras.json").exists():
                return
            od.mkdir(parents=True, exist_ok=True)
            parts, radius = load_object(model_of(name))
            if not parts:
                print(f"[{i + 1}/{len(names)}] {name}: no mesh, skipped", flush=True)
                return
            mesh = object_mesh(parts)
            views = plan_cameras(random.Random(name), radius)
            for c in views if args.render_plan else views[:1]:
                render(c, od / ("hero_rgba.png" if c["name"] == "hero" else f"{c['name']}.png"),
                       depth_of(name, od / "depth" / f"{c['name']}.npy"), mesh)
            rgba = Image.open(od / "hero_rgba.png")
            bg = Image.new("RGB", rgba.size, tuple(int(v * 255) for v in BACKDROP))
            bg.paste(rgba, mask=rgba.split()[3])
            bg.save(od / "hero.png")
            (od / "cameras.json").write_text(json.dumps({
                "frame": "splat (giro.path): y down, yaw 0 / pitch 0 looks along +z, + pitch looks down, "
                         "+ yaw moves the camera to the hero camera's right",
                "caption": "yaw = rel_yaw (target yaw - hero yaw, [0, 360)), pitch = the target camera's own pitch",
                "size": [W, H], "radius": radius, "rendered": "all" if args.render_plan else "hero",
                "views": views}, indent=1) + "\n")


main()
