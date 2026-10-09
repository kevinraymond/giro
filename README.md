# giro

**One image in, a cropped Gaussian splat you can walk around in VR out.**

> **Experimental research prototype.** A personal learning project, not a product. It runs on one
> machine for one user, needs a 24 GB GPU (two for the MiniMax H3 orbit), and changes often. The numbers in
> [docs/FINDINGS.md](docs/FINDINGS.md) come from a handful of subjects and seeds: lab notes, not
> benchmarks.

<p align="center"><img src="docs/images/library.webp" alt="The giro library: six finished splats (two raincoats, two adventurers, a knight, a scooter) with SPZ, SOG and PLY downloads" width="900"></p>

## The question

Video models can "orbit" a subject from one photo: use the photo as both the first and last frame
and ask for a 360° camera move. The result looks convincing but is hallucinated, and some seeds
cheat. The subject spins like a turntable, the camera swings halfway and morphs back, or frames
jump.

giro asks whether **automated checks can turn these unreliable videos into splats clean enough to
skip manual cleanup**. The manual workflow it replaces: reroll seeds by eye, trim frames, run
COLMAP and a trainer, then delete the room by hand.

## How it works

<!-- regenerate with: uv run scripts/pipeline_diagram.py -->
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/pipeline-dark.svg">
  <img src="docs/images/pipeline-light.svg" alt="Pipeline: one image, optionally background-edited, goes into a per-seed loop of orbit video, frames, masks, poses (COLMAP, with a Depth Anything 3 fallback) and a gate. A missing arc goes to gap fill and back to masks; a failure rerolls a new seed; a pass goes on to train, crop, upright and scale, and the exported splat." width="1000">
</picture>

The **gate** is the core idea. It checks the recovered camera ring: at least 330° of sweep,
steady one-way motion, no jump over 30° between frames, a closed loop and a round path. A failing
seed is rerolled. If COLMAP's cameras are what fails, a **pose fallback** poses the frames with
Depth Anything 3 and refines them with COLMAP. If the only problem is one missing arc, **gap fill**
regenerates just that arc.
Passing seeds are ranked by held-out PSNR.

There are two ways to make the orbit:

- **The proxy orbit** (the default since Oct 4, 2026; permissively licensed models): Pixal3D
  turns the hero into a 3D *proxy* mesh (TripoSplat with `-p proxy.model=triposplat`), giro renders
  the proxy's depth along a camera path it chooses (a spiral rising to 45°, starting at the hero's
  own camera), and Wan 2.2 Fun Control repaints that path with the hero, cut out onto black, as the
  first frame. The cameras are known, so COLMAP only
  refines them, and the views from above are real views instead of guesses. SeedVR2 then doubles
  the frames' resolution before training.
- **MiniMax H3 with a 360° orbit LoRA** (`--model h3`): the hero is the first and last frame of a
  generated orbit, and COLMAP recovers the cameras. Its license excludes the US, EU, UK and Korea.

The rest follows from there. Masking the subject before COLMAP rescues "turntable" videos. Brush
trains with a transparent background. The **crop** keeps Gaussians that land inside the subject
mask in most views. The export is upright, scaled to the subject's real height, and compressed for
the web and VR.

## A tour of the UI

`giro serve` runs a local web app. Long steps show live progress, and every decision shows its
reason.

### New job
Drop an image, set the subject's height and the number of good orbits wanted, and optionally
change the background first. Pick the video's shape and size, and drag or zoom the crop frame to
choose which part of the image the orbit starts from.

<img src="docs/images/new-job.webp" alt="New job page: the knight image with its crop frame, crop zoom, video shape and size, name, subject height, orbits wanted, seed limit, segmentation prompt with its count hint" width="900">

### Job
The best result on top and one card per seed below. Here one seed passed. The other was rejected
because it covered only 182° and jumped 169° in one step.

**Export job** saves the whole job as one zip to share or archive: an offline HTML report with
each good seed's result, orbit, cameras, training curve and numbers, plus its splat as PLY, SOG
and SPZ.

<img src="docs/images/job.webp" alt="Job page: One more orbit and Export job in the header, best result viewer on top, two seed cards below, one passed and one rejected with reasons" width="900">

### Result
The cropped splat with its size, its splat count against the Quest budget, downloads, and a
**View in VR** button for the Quest browser.

<img src="docs/images/result.webp" alt="Result tab: the knight splat on a grid, size, splat count bar, held-out PSNR, SPZ/SOG/PLY downloads, View in VR" width="900">

### Orbit
Each gate check against its threshold, and the hero view compared across the input image, the
trained splat and the crop.

<img src="docs/images/orbit.webp" alt="Orbit tab: gate checklist all passing, and a hero / trained / cropped comparison strip" width="900">

### Cameras
The recovered camera ring, colored by reprojection error, with the hero camera outlined. A clean
orbit is one smooth ring.

<img src="docs/images/cameras.webp" alt="Cameras tab: a ring of green camera frustums around the knight, and an azimuth-per-frame plot" width="900">

### Training
A live splat preview while Brush trains, with curves for held-out PSNR and splat count.

<img src="docs/images/training.webp" alt="Training tab: splat preview, held-out PSNR curve rising to 35.8 dB, splat count curve" width="900">

### Crop
Live crop controls: hull strictness, an optional box and subject height, with the kept count
against the VR budget.

<img src="docs/images/crop.webp" alt="Crop tab: splat preview with strictness slider, trim box toggle, subject height, Gaussians kept" width="900">

### Numbers
Every metric from every stage. On a rejected seed, the reasons come first and **Use anyway**
overrides the gate.

<img src="docs/images/rejected.webp" alt="Numbers tab of a rejected seed: three failed-check reasons, then per-stage metric tables" width="900">

## What I learned

The details are in [docs/FINDINGS.md](docs/FINDINGS.md).

- The gate matched a visual check on 20 of 20 videos. Its largest-jump check caught one that swept
  357° and still closed its loop.
- Masking out the room rescues turntable videos. One seed went from 4 to 119 of 119 frames placed.
- Gap fill regenerates a missing arc in about 40 s, against 7 minutes for a reroll. It passed in
  3 of 3 tries.
- Feedforward poses (Depth Anything 3, VGGT, MapAnything) trained splats 2–4 dB worse than
  COLMAP's, but as a starting point for COLMAP's refinement they matched it. The pose fallback
  built on this rescued a seed COLMAP left with a 57° jump, in 12 s.
- Transparent training plus the hull crop needed no manual cleanup on the 4 test subjects,
  including their swords and mirrors.
- The Quest 3 browser stuttered at 245K splats, well below the 400–600K I expected. This needs a
  cleaner measurement.
- The proxy orbit, against H3 with the same training on five subjects, matched or beat it on the
  hero view on three of five (adventurer LPIPS 0.065 against 0.093, knight 0.057 against 0.071) and
  was sharper on all five, with a formed top where H3's is a smear. H3 kept the hero's look around
  the ring better on four of five: the video follows the proxy's shape, and TripoSplat's tank is
  boxier than the real one.
- Training, not the video, was where detail went missing: at an 80K splat cap the splat kept about
  60% of the frames' sharpness. SeedVR2 frames and up to 150K splats for 60K iterations fixed most
  of that; a sharper, slower 768×1024 video alone did not.
- Measure on the hero view. Brush's held-out PSNR scores a video against itself, and a blurrier
  video can score higher. `scripts/evaluate.py` scores the hero view, sharpness and likeness.

**Where it stands (Oct 4, 2026).** The proxy orbit is now the default: anyone can use its models,
it passed the gate on its first try in 14 of 14 runs across five subjects and three seeds, and its
quality is on par with H3's or better except for how faithfully the shape of hard-surface subjects
comes through. Since Oct 9, Pixal3D (MIT) is the proxy, on ComfyUI v0.38.2: TripoSplat gave the
tank a hull about as wide as it is long, Pixal3D a long one with the barrel in line. The video now
starts from the hero cut out onto black, which keeps a floor and wall from tilting around the
subject in the views from above. Its denser splats (up to 150K) still need a check in the headset. Details
and the open questions are in [FINDINGS](docs/FINDINGS.md#proxy-orbit).

## Running it

giro is shared to read and learn from. It is not packaged for easy installation. You will need:

- Linux with a 24 GB NVIDIA GPU: the proxy orbit's steps run one after another on one card, about
  20 minutes per seed. The H3 orbit's video alone peaks at about 22 GB, so it wants a second card.
- Python 3.12 with [uv](https://docs.astral.sh/uv/), Node.js, Rust, FFmpeg, and COLMAP 4.x with CUDA.
- The model weights, which you supply (see [Third-party software and models](#third-party-software-and-models)).

```bash
uv sync
scripts/setup_comfy.sh                  # pinned headless ComfyUI in vendor/
scripts/setup_brush.sh                  # pinned Brush
scripts/setup_splat_transform.sh        # pinned splat-transform
scripts/setup_da3.sh                    # optional pose fallback (Depth Anything 3)
# point scripts/extra_model_paths.yaml at your models; the orbit LoRA goes in
# loras/h3/minimax_h3_flf2v_orbit360_pablodawson_v1.safetensors (or pass -p orbit_video.lora=null)
cd ui && npm ci && npm run build && cd ..

uv run giro serve                       # UI on http://<host>:8470
uv run giro run image.png -o data/try1 --gpu 0  # or one seed from the CLI (the proxy orbit)
uv run giro run image.png -o data/try2 --model h3          # MiniMax H3 instead
uv run giro job export data/jobs/<job>  # a job as one zip: offline HTML report + PLY/SOG/SPZ
```

`just check` lints and runs the tests. Outputs go to `data/`, which git ignores.
`uv run --group eval scripts/evaluate.py OUT ATTEMPT...` scores finished attempts (hero view,
sharpness, likeness) so orbit modes and settings can be compared.

## Repository map

| Path | Contents |
|---|---|
| `src/giro/stages/` | One module per pipeline step (`proxy.py`, `path_poses.py`: the proxy orbit) |
| `src/giro/path.py` | The proxy orbit's camera paths |
| `src/giro/api.py`, `manager.py`, `job.py`, `scheduler.py` | Orchestrator: API, jobs, rerolls, GPU scheduling |
| `src/giro/comfy/`, `src/giro/workflows/` | Headless ComfyUI client and workflows |
| `ui/` | React UI with a [Spark](https://sparkjs.dev) splat viewer and a WebXR page |
| `scripts/` | Pinned setup scripts, sweeps, the evaluator, report and screenshot helpers |
| `docs/FINDINGS.md` | Lab notes |

## Limitations

- The top of the head, the underside and other unseen parts are invented, and they come out soft.
  The proxy orbit sees the top but still invents it, and its shapes are only as good as its proxy.
- The measurements are small: a few subjects and seeds, untuned, on one machine.
- The test images are AI-generated, so held-out PSNR measures consistency with the generated
  video, not with reality.
- There is no authentication. Keep it on your local network.

## Third-party software and models

giro orchestrates the tools below. It ships no weights or third-party binaries. Check each license,
including what it says about outputs, before use.

| Component | Role | License |
|---|---|---|
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Headless inference server (over HTTP) | GPL-3.0 |
| [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) ([ComfyUI files](https://huggingface.co/Comfy-Org/MiniMax-H3)) | Orbit video | [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) |
| [MiniMax-H3 360° Orbit LoRA](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA) | Orbit video (on by default) | MiniMax H3 Community License |
| [Pixal3D](https://huggingface.co/TencentARC/Pixal3D) ([ComfyUI files](https://huggingface.co/Comfy-Org/Pixal3D)) | Proxy mesh (proxy orbit) | MIT |
| [TRELLIS.2](https://huggingface.co/microsoft/TRELLIS.2-4B) shape and texture VAEs, in Pixal3D's ComfyUI files | Pixal3D's decoders | MIT |
| [DINOv3](https://github.com/facebookresearch/dinov3) ViT-L, in Pixal3D's ComfyUI files | Pixal3D's image encoder | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| [MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl-normal) ViT-L | Field of view for Pixal3D | MIT |
| [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) ([weights](https://huggingface.co/VAST-AI/TripoSplat)) | Proxy splat (proxy orbit, `-p proxy.model=triposplat`) | MIT |
| [DINOv3](https://github.com/facebookresearch/dinov3) ViT-H, bundled with TripoSplat | TripoSplat's image encoder | [DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| [FLUX.2 autoencoder](https://github.com/black-forest-labs/flux2#flux2-autoencoder), bundled with TripoSplat | TripoSplat's image conditioning | Apache-2.0 |
| [Wan 2.2 Fun Control A14B](https://huggingface.co/alibaba-pai/Wan2.2-Fun-A14B-Control) ([ComfyUI files](https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged)) | Orbit video (proxy orbit) | Apache-2.0 |
| [SeedVR2](https://github.com/ByteDance-Seed/SeedVR) ([7B](https://huggingface.co/ByteDance-Seed/SeedVR2-7B), [ComfyUI files](https://huggingface.co/Comfy-Org/SeedVR2)) | Frame upscale (proxy orbit) | Apache-2.0 |
| [SageAttention](https://github.com/thu-ml/SageAttention) 1.0.6 | Attention kernels (optional) | BSD-3-Clause |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) ([ComfyUI files](https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI), [Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)) | Background edit | Apache-2.0 |
| [SAM 3.1](https://huggingface.co/facebook/sam3.1) ([ComfyUI files](https://huggingface.co/Comfy-Org/sam3.1)) | Subject masks | [SAM License](https://huggingface.co/Comfy-Org/sam3.1/blob/main/LICENSE) |
| [COLMAP](https://colmap.github.io) | Camera poses | BSD-3-Clause |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) ([DA3-BASE weights](https://huggingface.co/depth-anything/DA3-BASE)) | Pose fallback | Apache-2.0 |
| [MapAnything](https://github.com/facebookresearch/map-anything) ([weights](https://huggingface.co/facebook/map-anything-apache)) | Pose fallback, tried first (worker kept) | Apache-2.0 |
| [Brush](https://github.com/ArthurBrussee/brush) | Splat training | Apache-2.0 |
| [splat-transform](https://github.com/playcanvas/splat-transform) | SPZ/SOG export | MIT |
| [Spark](https://github.com/sparkjsdev/spark), [three.js](https://threejs.org) | Web and WebXR viewer | MIT |
| [FFmpeg](https://ffmpeg.org) | Frame extraction | LGPL/GPL |

**MiniMax H3's license is restrictive; read it before you run giro.** As of October 2026 it
grants use only outside its "Excluded Territories" (the European Union, the United Kingdom, the
Republic of Korea and the United States), and asks people there to contact MiniMax for a license.
It also requires a "Powered by MiniMax H3" notice on products built with it, and it does not allow
using outputs to improve other AI models. This is a summary, not legal advice. The orbit step is
the only part of giro that depends on H3.

**The proxy orbit (`--model wan22-control`, experimental) does not use MiniMax H3.** Its models
allow commercial use in every region. Pixal3D, the TRELLIS.2 VAEs and MoGe-2 are MIT; Tencent's
own Hugging Face repo for Pixal3D refuses downloads from the EU (a download gate, not a term of the
MIT license), and the Comfy-Org repack giro uses is not gated. Pixal3D's ComfyUI files and
TripoSplat's repo both bundle Meta's DINOv3 encoder under the DINOv3 License, the same terms as SAM 3.1's SAM License: royalty-free and
commercial, but no military, warfare, nuclear, espionage or weapons uses, no use by sanctioned
parties, no reverse engineering, the license text must go with the weights, and Meta can change the
terms. The FLUX.2 VAE bundled with TripoSplat is Apache-2.0; the non-commercial license covers
FLUX.2 [dev] itself, not this VAE. Wan 2.2 Fun Control and SeedVR2 are Apache-2.0. This is a
summary, not legal advice.

The sample subjects were generated with [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
and [Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) with its
[Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning), all Apache-2.0.
[Spirula Studio](https://github.com/harry7557558/spirula-studio) was tried along the way (see
[FINDINGS](docs/FINDINGS.md#camera-poses)). Gap fill adapts an idea from OrbitForge
([arXiv:2606.24799](https://arxiv.org/abs/2606.24799)).

## License

giro's own code is [MIT](LICENSE), except its ComfyUI nodes in `src/giro/comfy_ext/`, which are
[GPL-3.0-or-later](src/giro/comfy_ext/LICENSE): they run inside ComfyUI (GPL-3.0) and build on its
internals. The rest of giro talks to ComfyUI over HTTP. Neither license covers the third-party
tools and models above.
