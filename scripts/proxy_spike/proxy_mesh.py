"""Regenerate an attempt's TRELLIS.2/Pixal3D proxy with its seed and keep the mesh, which the
proxy stage only samples into a splat. Same image (proxy/hero_rgba.png) and params as the stage.

    proxy_mesh.py ATTEMPT OUT_DIR GPU [MODEL SEED,SEED,...]

With MODEL and SEEDS (trellis2 or pixal3d), writes OUT_DIR/mesh_<model>_<seed>.npz for each,
to compare seeds and models.

Writes OUT_DIR/mesh.npz (vertices, faces, colors in the mesh's y-up frame; GiroSaveMesh) and
OUT_DIR/candidates/proxy_<seed>.ply, to compare with the attempt's proxy.ply.
"""
import asyncio
import json
import sys
from pathlib import Path

from giro.comfy import server
from giro.comfy.client import ComfyClient
from giro.stages.proxy import trellis_workflow

attempt, out, gpu = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), int(sys.argv[3])
params = json.loads((attempt / ".stages" / "proxy.json").read_text())["params"]
seed = json.loads((attempt / "proxy" / "proxy.json").read_text())["seed"]
model = sys.argv[4] if len(sys.argv) > 4 else params["model"]
seeds = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [seed]
out.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    with await asyncio.to_thread(server.Lease, gpu) as lease:
        async with ComfyClient(lease.url) as comfy:
            try:
                image = await comfy.upload_image(attempt / "proxy" / "hero_rgba.png")
                wf = trellis_workflow(image, seeds, out / "candidates", params, model)
                for s in seeds:
                    name = "mesh.npz" if len(sys.argv) <= 4 else f"mesh_{model}_{s}.npz"
                    wf[f"save_mesh_{s}"] = {"class_type": "GiroSaveMesh", "inputs": {"mesh": [f"paint_{s}", 0], "path": str(out / name)}}
                async for _ in comfy.run(wf):
                    pass
            finally:
                await comfy.free()

asyncio.run(main())
print(out)
