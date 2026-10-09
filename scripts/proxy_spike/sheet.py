import json, sys
from pathlib import Path
from giro import render
for a in map(Path, sys.argv[1:]):
    if not (a/"canonical"/"splat.ply").exists(): print("no splat", a); continue
    m=json.loads((a/"metrics.json").read_text()); h=m["canonicalize"]["height_m"]
    w,d=json.loads((a/"canonical"/"transform.json").read_text())["footprint_m"]
    c=(0,h/2,0); dist=render.framing_distance((w,h,d)); size=(512,512) if w>h else (384,512)
    cams=render.orbit_cameras(c,dist,8,elevation_deg=10)+render.orbit_cameras(c,dist,2,elevation_deg=60)+render.orbit_cameras(c,dist,2,elevation_deg=-30)
    render.sheet(render.render(a/"canonical"/"splat.ply",cams,size,background=(0.5,0.5,0.5)),cols=6,width=300).save(a.parent/f"{a.name}-review.jpg",quality=90)
    print(a.name, m["export"]["n_gaussians"], m["train"]["eval_psnr"])
