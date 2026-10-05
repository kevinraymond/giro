"""The projection-texture route end to end, from an attempt whose proxy stage ran (hero, proxy):
no video model. Each step is its own script in this directory; this runs them in order and
skips what OUT already has.

  1. proxy seeds: N Pixal3D meshes (proxy_mesh.py; the side ComfyUI, GIRO_COMFY_DIR)
  2. anchors: the Multiple-Angles LoRA's 32 views of the hero (anchors_angles.py), SAM masks and
     silhouette registration to the attempt's proxy (register_anchors.py)
  3. seed pick by edge agreement with the anchors, contact sheet to overrule it (pick_seed.py)
  4. silhouette fits, photometric registration (anchors that still disagree are dropped), warp
     grids (project_texture.py, register_photometric.py, warp_views.py)
  5. texture with view selection, then region patches (patch_region.py), optionally a generative
     fill of what no view painted (fill_unseen.py), then progressive painting (progressive_paint.py:
     8 guarded whole-image key views, then masked views; --no-progressive skips it, ~25 min), then 188
     renders (2x supersampled, soft masks)
  6. training with the attempt's own settings (giro stages --from dataset), max splats 300K, masks not
     eroded

    texture_route.py ATTEMPT OUT GPU [--seeds 6] [--angles DIR] [--seed mesh_pixal3d_N.npz]
        [--patch "license plate::Make the license plate ...::180::10"] [--fill] [--no-progressive]
        [--subject "desert tan M1 Abrams tank"] [--library NAME --image SOURCE]

A patch is FIND::PROMPT::YAW::PITCH[::PICK] (patch_region.py's arguments). OUT/route.json records
what each step chose; OUT/work/attempt is the trained attempt.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("attempt", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("gpu", type=int)
ap.add_argument("--seeds", type=int, default=6)
ap.add_argument("--first-seed", type=int, default=101)
ap.add_argument("--angles", type=Path, help="anchor views made before (with registration.json)")
ap.add_argument("--seed", help="this proxy mesh instead of the pick")
ap.add_argument("--min-iou", type=float, default=0.75, help="anchors registered worse than this are left out")
ap.add_argument("--min-ncc", type=float, default=0.6, help="anchors whose colors agree less after registration are left out")
ap.add_argument("--patch", action="append", default=[])
ap.add_argument("--subject", default="object", help="what the hero shows, for the edit prompts (fill, progressive), "
                "e.g. 'desert tan M1 Abrams tank'")
ap.add_argument("--fill", nargs="?", const="", metavar="SUBJECT",
                help="generative fill of the surface no view painted (fill_unseen.py); SUBJECT, if given, sets --subject")
ap.add_argument("--exposure", choices=["match", "off"], default="match",
                help="project_texture.py --exposure: off for reflective subjects (Oct 5, knight: the matched gains "
                     "hit their clamp and darkened the rear)")
ap.add_argument("--no-progressive", action="store_true", help="skip progressive painting (progressive_paint.py)")
ap.add_argument("--max-splats", type=int, default=300_000)
ap.add_argument("--library", help="also make a library job with this name")
ap.add_argument("--image", type=Path, help="the job's source image, for the library (default: the hero)")
args = ap.parse_args()
if args.fill:
    args.subject = args.fill
attempt, out = args.attempt.resolve(), args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
work, seeds_dir = out / "work", out / "seeds"
angles = args.angles.resolve() if args.angles else out / "angles"
NEXT = str(ROOT / "vendor" / "comfyui-next")
route: dict = json.loads((out / "route.json").read_text()) if (out / "route.json").exists() else {}
g = str(args.gpu)


def run(*cmd: str, comfy: str | None = None, cwd: Path = HERE) -> str:
    env = os.environ | ({"GIRO_COMFY_DIR": comfy} if comfy else {})
    t0 = time.monotonic()
    r = subprocess.run(["uv", "run", "python", *cmd] if cmd[0].endswith(".py") else ["uv", "run", *cmd],
                       cwd=cwd, env=env, capture_output=True, text=True)
    with open(out / "route.log", "a") as log:
        log.write(f"$ {' '.join(cmd)}\n{r.stdout}{r.stderr}\n")
    if r.returncode:
        sys.exit(f"{cmd[0]} failed ({r.returncode}); see {out / 'route.log'}")
    print(f"  {cmd[0]} {time.monotonic() - t0:.0f} s", flush=True)
    return r.stdout


def comfy_down(comfy: str | None = None) -> None:
    # A ComfyUI left on the GPU would be reused whatever checkout the next step needs.
    run("giro", "comfy", "down", "--gpu", g, comfy=comfy, cwd=ROOT)


def save() -> None:
    (out / "route.json").write_text(json.dumps(route, indent=1))


print("1. proxy seeds", flush=True)
seeds = list(range(args.first_seed, args.first_seed + args.seeds))
if not all((seeds_dir / f"mesh_pixal3d_{s}.npz").exists() for s in seeds):
    comfy_down()
    run("proxy_mesh.py", str(attempt), str(seeds_dir), g, "pixal3d", ",".join(map(str, seeds)), comfy=NEXT)
    comfy_down(NEXT)

print("2. anchor views", flush=True)
if not (angles / "registration.json").exists():
    if not (angles / "31.png").exists():
        run("anchors_angles.py", str(attempt / "hero" / "hero.png"), str(angles), g, "all")
    run("register_anchors.py", str(attempt), str(angles), g)
    comfy_down()
reg = json.loads((angles / "registration.json").read_text())
skip = sorted(n for n, r in reg.items() if r["iou"] < args.min_iou)
route["anchors"] = {"made": len(reg), "silhouette_dropped": skip}

print("3. seed pick", flush=True)
if not (work / "mesh.npz").exists() or args.seed:
    run("pick_seed.py", str(attempt), str(angles), str(seeds_dir), str(work), g, "--min-iou", str(args.min_iou),
        *(["--seed", args.seed] if args.seed else []))
route["seed"] = json.loads((seeds_dir / "seeds.json").read_text())["best"] if not args.seed else args.seed
save()

print("4. cameras", flush=True)
S = ",".join(skip) or "none"
if not (work / "anchor_cameras.json").exists():
    run("project_texture.py", str(attempt), str(angles), str(work), g, "--skip", S, "--out", "scratch")
    run("register_photometric.py", str(attempt), str(angles), str(work), g, S)
cams = json.loads((work / "anchor_cameras.json").read_text())["cameras"]
weak = sorted(n for n, c in cams.items() if c["ncc"][1] < args.min_ncc)
skip2 = sorted(set(skip) | set(weak))
S2 = ",".join(skip2) or "none"
route["anchors"] |= {"color_dropped": weak, "used": len(reg) - len(skip2)}
if not (work / "warps.pt").exists():
    run("warp_views.py", str(attempt), str(angles), str(work), g, "--skip", S2)
save()

print("5. texture", flush=True)
run("project_texture.py", str(attempt), str(angles), str(work), g, "--skip", S2, "--cameras", str(work / "anchor_cameras.json"),
    "--warps", str(work / "warps.pt"), "--select-power", "6", "--exposure", args.exposure, "--save-texture", str(work / "texture-0.pt"),
    "--out", "scratch")
tex = work / "texture-0.pt"
for k, spec in enumerate(args.patch, 1):
    find, prompt, yaw, pitch, *pick = spec.split("::")
    nxt = work / f"texture-{k}.pt"
    run("patch_region.py", str(attempt), str(work), g, "--texture", str(tex), "--save", str(nxt), "--find", find,
        "--prompt", prompt, "--yaw", yaw, "--pitch", pitch, *(["--pick", pick[0]] if pick else []))
    tex = nxt
fill = args.fill is not None
if (fill or not args.no_progressive) and not (work / "attempt" / "cameras.json").exists():
    # the fill and progressive painting take their cameras from the textured mesh's renders
    run("project_texture.py", str(attempt), str(angles), str(work), g, "--texture", str(tex), "--out", "attempt")
if fill:
    nxt = work / "texture-fill.pt"
    run("fill_unseen.py", str(attempt), str(work), g, "--texture", str(tex), "--save", str(nxt), "--subject", args.subject)
    tex = nxt
if not args.no_progressive:
    nxt = work / "texture-prog.pt"
    run("progressive_paint.py", str(attempt), str(work), g, "--texture", str(tex), "--save", str(nxt), "--subject", args.subject)
    tex = nxt
if args.patch or fill or not args.no_progressive:
    comfy_down()
run("project_texture.py", str(attempt), str(angles), str(work), g, "--texture", str(tex), "--out", "attempt")
route["texture"] = json.loads((work / "texture.json").read_text()).get("painted")
route["patches"] = args.patch
route["fill"] = args.subject if fill else None
route["progressive"] = not args.no_progressive
route["exposure"] = args.exposure
save()

print("6. training", flush=True)
params = []
for st in ("dataset", "train", "crop", "canonicalize", "export"):
    for k, v in json.loads((attempt / ".stages" / f"{st}.json").read_text())["params"].items():
        params += ["-p", f"{st}.{k}={json.dumps(v)}"]
# The renders' masks are exact and soft at the edges (project_texture.py --supersample): the erosion meant
# for SAM masks would binarize them and train each part's outer pixels transparent.
run("giro", "stages", str(work / "attempt"), "--from", "dataset", "--gpu", g, *params, "-p", f"train.max_splats={args.max_splats}",
    "-p", "dataset.mask_erode_px=0", cwd=ROOT)
m = json.loads((work / "attempt" / "metrics.json").read_text())
route["result"] = {"n_gaussians": m["export"]["n_gaussians"], "train_seconds": m["train"]["train_seconds"]}
save()
if args.library:
    image = args.image.resolve() if args.image else attempt / "hero" / "hero.png"
    height = json.loads((attempt / ".stages" / "canonicalize.json").read_text())["params"]["height_m"]
    run("make_review_jobs.py", args.library, str(image), str(height), f"1={work / 'attempt'}")
print(json.dumps(route, indent=1))
