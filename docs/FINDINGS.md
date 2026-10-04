# Findings

Lab notes from building giro, by pipeline step. Nearly everything was measured on four AI-generated
subjects (an adventurer, a woman in a raincoat, a knight and a scooter), with a few seeds each and
default settings, on one machine with two RTX 4090s. Treat the numbers as notes, not benchmarks.

A *seed* is one generated orbit video and everything built from it. The *hero* is the input image,
which also gets a camera. *PSNR* is Brush's held-out PSNR on every 8th frame. It measures
consistency with the generated video, since there is no real ground truth.

## Orbit videos

- The first MiniMax H3 orbit (768×1024, 124 frames, 20 steps) was a real 360° turn with a
  plausible invented room. It took 291 s on a 4090 and peaked at about 22 GB.
- The speed is not constant. The first ~16 and last ~35 frames barely move, so the frames step
  trims both ends and drops near-duplicates.
- Longer clips cost more than linearly: 1.5× the frames took 2.3× the time.
- The same seed gives the same video, so a known-bad seed makes a reliable test fixture.

Clip-length sweep, one subject, 4 seeds per setting:

| Frames × steps | Pass rate | Video time | GPU min per passing seed |
|---|---|---|---|
| 124 × 20 | 2/4 | 293 s | 9.8 |
| 124 × 8 | 0/4 | 138 s | – |
| **158 × 20** (default) | 3/4 | 418 s | 9.3 |
| 192 × 20 | 3/4 | 563 s | 12.5 |
| 192 × 8 | 2/4 | 253 s | 8.4 |

With n = 4, only 124 × 8 is clearly worse. A seed that passes at one setting can fail at another.

An orbit LoRA changed this more than any setting. With the
[360 orbit LoRA](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA) and the prompt it
was trained on, seeds 1–4 with no rerolls:

| Subject, video | Pass rate | Video time | Views | Without the LoRA |
|---|---|---|---|---|
| Adventurer, 768×768, 73 frames, 28 steps | 4/4 | 142 s | 66 | – |
| Adventurer, 768×1024, 158 frames, 20 steps | 4/4 | 437 s | 103 | 3/4 |
| Tank, 1024×768, 158 frames, 20 steps | 4/4 | 424 s | 104 | 1 of 3 |

- The LoRA was trained on orbits of people only, yet the tank, which filled the frame and had
  failed most seeds, passed every time, with a level ring and a rear that reads as a tank.
- With it, the video moves from the first frame: dedup trimmed 1–2 frames at the start instead of
  about 16. giro now uses it by default.
- The top and the underside are still invented: the orbit stays at eye level.
- Wan 2.2 (image-to-video 14B, Apache-2.0) with an
  [orbit LoRA](https://huggingface.co/ostris/wan22_i2v_14b_orbit_shot_lora), first and last frame
  pinned, 576×768 and 81 frames, passed 0 of 4 on the adventurer at 441 s per clip. One clip looked
  like an orbit, but the room changed from wall to wall and the satchel swapped hips. Another
  turned the subject twice in a still room. It stays as `model=wan22` for comparisons.

## Proxy orbit

MiniMax H3's license excludes the US, EU, UK and South Korea, so giro has a second way to make the
orbit from permissively licensed models (`model=wan22-control`):

1. TripoSplat (MIT) turns the masked hero into a rough splat, the *proxy*, in about 10 s per
   seed. giro runs 4 seeds and keeps the one whose silhouette best matches the hero.
2. giro fits the hero's camera to the proxy on the CPU (about 5 s): silhouette IoU over yaw and
   pitch, distance and framing from the silhouettes' boxes, then a Nelder-Mead refinement. Front
   and back share a silhouette, so the color against the hero breaks the tie. On the adventurer
   and the tank the fit reaches IoU 0.94–0.95.
3. The proxy is rendered as depth along a camera path that starts at the hero's camera, and
   Wan 2.2 Fun Control 14B (Apache-2.0) repaints it with the hero as both the first frame and the
   appearance reference. ComfyUI's node implements a first frame but does not declare the input;
   giro's node does.
4. The path's cameras are the poses: COLMAP triangulates its masked matches from them and
   bundle-adjusts twice. The hero is the video's first frame (31 dB against it), so it takes frame
   0's pose; adjusted on its own, bundle adjustment traded its focal length for distance.

The first runs (Oct 3–4, adventurer, seed 1, 576×768, 81 frames, a two-turn spiral to 45°):

- Wan without control, even with an orbit LoRA, passed 0 of 4: the subject was not one consistent
  object. With the proxy's depth it posed every frame, and the camera path is known even where
  COLMAP's own mapper loses it (it posed 27 of 82 frames of the spiral).
- The gate passed outright: 82 of 82 images posed, bundle adjustment moved the cameras a median
  1.8% of the orbit radius from the path, and the subject's mask matched the proxy's silhouette at
  IoU 0.96 on average.
- 13.6 min for the whole attempt on one 4090: proxy 65 s, video 474 s, masks 113 s, poses 5 s,
  training 150 s.

Brush's PSNR does not compare across orbit modes: it scores the video against itself, and a
blurrier video can score higher. giro's evaluator (`scripts/evaluate.py`) scores the cropped and
canonical splats on the hero view (the only real image, subject pixels only, so PSNR reads low),
on sharpness (mean absolute Laplacian inside the subject, as a fraction of the hero's, with the
subject 768 px tall) and on DINOv2 likeness to the hero. The Laplacian also rewards noise, so read
it with close-up renders.

Adventurer, seed 1, every row trained with the hero's camera known:

| Orbit | Hero PSNR / LPIPS | Sharpness ring / above | Likeness front / ring | Time |
|---|---|---|---|---|
| H3 + orbit LoRA, 768×1024, 158 frames, 120K Gaussians | 21.9–22.1 / 0.100 | 0.36–0.38 / 0.17–0.18 | 0.93–0.94 / 0.67–0.70 | – |
| Proxy, 576×768, 80K cap | 22.0 / 0.090 | 0.42 / 0.24 | 0.91 / 0.69 | 13.6 min |
| Proxy, 768×1024 video | 22.1 / 0.086 | 0.38 / 0.20 | 0.93 / 0.71 | 26.5 min |
| Proxy + Wan refine pass to 768×1024 | 22.8 / 0.084 | 0.35 / 0.20 | 0.90 / 0.69 | 22.4 min |
| Proxy + SeedVR2 to 768×1024 | 21.9 / 0.087 | 0.55 / 0.36 | 0.89 / 0.66 | 14.5 min |
| **Proxy + SeedVR2 ×2 + 150K splats, 60K iterations** | **22.7 / 0.065** | **0.62 / 0.40** | 0.92 / 0.69 | 19.9 min |

- Training was the bottleneck, not the video. Rendered at the frames' own cameras, the splat
  keeps about 60% of the frames' sharpness: H3 0.62 → 0.38, proxy 0.51 → 0.31, and a sharper
  768×1024 video (0.60) still trains to 0.33. Up to 150K splats and 60K iterations recover part
  of it (0.39); Brush stops growing at about 157K on its own, mip stays better than without, and
  training on the hero five times is worth 1.5 dB on the hero view.
- SeedVR2 (one-step video super-resolution, Apache-2.0, native in ComfyUI) upscales the 81
  frames in about a minute, to frames sharper than the hero itself, and the splat keeps them:
  the close-ups have wool texture, buckles and a crisp face from every side, and a formed top.
  At ×2 it invents a knit-like texture the hero's cloth does not have; ×1.33 is noisier. H3's
  front face still keeps the hero's identity a little better.
- Wan at 768×1024 makes sharper frames than at 576×768, at 2.6× the time (1.5× with
  SageAttention), but the gain does not survive training at 80K; the refine pass (the clip again at
  768×1024 from σ 0.35) cleans edges and fits the hero best, and softens the cloth.
- More clips along one camera program (a climb to 70°, then close-ups) join without a seam, since
  each starts on the last frame of the one before, but the splat gets blurrier (sharpness 0.28 at
  80K, 0.33 at 150K) and the views from above more like the hero: the clips disagree on fine
  detail, and training averages them.
- Ending the path on the hero (also pinned as the last frame) and starting each clip from the
  proxy's color render instead of noise both traded one score for another; neither is a default.
- The tank shows the limit: Wan follows the proxy's depth faithfully, so TripoSplat's boxier hull
  and flat rear come through. It matches H3 on the hero view (18.5 dB / 0.044 against 17.7 / 0.047)
  and is sharper, but looks less like the hero around the ring (0.58 against 0.88).

The proxy orbit's defaults are the bold row: a two-turn spiral to 45° at 576×768 and 81 frames,
SeedVR2 ×2, and up to 150K splats for 60K iterations (still under the 150K VR budget after the
crop; not yet checked in the headset).

Against H3 with the orbit LoRA, both trained the same way (150K splats, 60K iterations), seed 1:

| Subject | Hero PSNR / LPIPS, H3 → proxy | Sharpness ring, H3 → proxy | Likeness ring, H3 → proxy |
|---|---|---|---|
| Adventurer | 22.6 / 0.093 → **22.7 / 0.065** (seed 2: 22.5 / 0.068) | 0.39 → **0.62** | 0.68 → 0.69 |
| Knight | 21.3 / 0.071 → **22.1 / 0.057** | 0.38 → **0.52** | **0.89** → 0.82 |
| Raincoat | 20.8 / 0.048 → 20.0 / 0.046 | 0.43 → **0.58** | **0.55** → 0.47 |
| Scooter | 19.4 / 0.077 → 18.9 / 0.079 | 0.46 → 0.48 | **0.80** → 0.66 |
| Tank | 18.5 / 0.043 → 18.5 / 0.044 | 0.33 → 0.39 | **0.89** → 0.58 |

- The proxy orbit matches or beats H3 on the hero view and is sharper on every subject, with a top
  that is formed instead of smeared. H3 keeps the hero's look around the ring better on four of
  five: the proxy's shape is TripoSplat's, and the video follows it. A better proxy is the next
  lever for hard-surface subjects.
- Starting the clips from the proxy's color render (`orbit_video.init`) helped the tank (19.6 / 0.040,
  likeness 0.71) and hurt the adventurer (its pale proxy colors came through on the back), so it
  stays an option.
- Two failures, both fixed. Once, Wan returned 81 frames of pure noise; the same run in a fresh
  ComfyUI was fine, and the orbit stage now stops when the first frame (pinned to the hero, normally
  31–32 dB against it) does not match it. And on the glossy scooter, SIFT found too few matches
  from above, bundle adjustment pulled the cameras off the path (median 5–6% of the radius, focal
  13% off) and the splat broke apart. When bundle adjustment looks that unreliable, the poses stage
  now keeps the path's cameras unadjusted: the video follows them closely (silhouette IoU 0.95 per
  elevation band), and the scooter trained clean.

Where it stands, and what is next (Oct 4):

- Worth making the default for a public tool: the license question goes away, the hero view is as
  good or better, the splats are sharper and the top is real coverage. Not done: the shapes of
  hard-surface subjects are the proxy's, measured with one seed per subject (two for the adventurer
  and the tank).
- Next: a better proxy for hard surfaces (TRELLIS, MIT, gives Gaussians directly; Step1X-3D,
  Apache-2.0; TripoSG, MIT, geometry only, is enough for depth), the 135K-splat exports in the
  headset, more seeds, and a way to make chained clips agree (render the first splat into the
  later clips as kept content, as VideoFrom3D does, or per-image appearance in training).

## The quality gate

Failure modes seen, and the check that caught each one:

| What the video did | Caught by |
|---|---|
| Barely moved | Too few distinct frames |
| Orbited ~220°, then morphed back to the front | Coverage, loop closure |
| Swept 357° and closed the loop, with one snap in the middle | Largest jump |
| Swung 25° out and back | Coverage, one-way motion |
| Spun the subject like a turntable in a still room | COLMAP finds no motion (fixed by [masks](#masks)) |
| Smooth, but COLMAP misplaced cameras facing a row of identical windows | Largest jump |

- The gate matched a visual check of contact sheets on 20 of 20 videos.
- Passing seeds sit far inside most thresholds. The tightest margin is the largest jump: passes
  reached 18° and the nearest rejection was 35°, against a limit of 30°.
- Reprojection error never separated good seeds from bad (0.74–0.99 px for both).
- PSNR should only rank seeds that already passed. One rejected seed had the best PSNR in its
  batch, because its held-out views covered only the 220° it actually orbited.

## Camera poses

- COLMAP took 10–50 s per seed, which is small next to a 5–7 minute video. Faster methods only
  help if they rescue seeds that COLMAP cannot.
- MapAnything (feedforward, Apache weights) posed 119 frames in about 16 s. Its ring was loose and
  its focal length about 45% short, and splats trained on its raw poses were blurry (20.7 dB
  against 24.5 dB). It fails on turntable videos.
- Refining MapAnything's poses with COLMAP's triangulation and bundle adjustment matched COLMAP
  (24.45 dB against 24.47 dB). On a seed with three motion-blurred frames that COLMAP could not
  place, it placed all of them. The gate still rejected that seed, correctly, for a 32° jump. The
  pose fallback below grew out of this.
- I also tried the SfM built into
  [Spirula Studio](https://github.com/harry7557558/spirula-studio) (`spirula sfm auto`, with
  giro's masks) on all 28 gated seeds. It gave the same gate verdict as COLMAP on 27 of 28, took
  3–9 s against COLMAP's 8–31 s, and led to Brush PSNR within 0.1–0.5 dB of COLMAP's. The two
  motion-blurred borderline seeds stayed borderline with either tool. giro stays on COLMAP, which
  was already wired in.
- A lower reprojection error did not predict better training results with any pose tool.

## Pose fallback

When the gate rejects COLMAP's cameras, giro poses the frames with Depth Anything 3 (DA3-BASE),
refines them with COLMAP's triangulation and bundle adjustment on COLMAP's own matches, and gates
the result. If that fails too, it goes back to COLMAP's cameras and on to gap fill.

- I compared DA3-Giant, DA3-Base, VGGT and MapAnything on all 28 gated seeds. Raw camera centers
  sat 1.6%, 1.7%, 2.4% and 4.9% of the orbit radius from COLMAP's. Brush on the raw poses lost
  2–4 dB (24.5 dB with COLMAP; 22.4, 20.7, 21.2 and 20.7 dB).
- After refinement all four landed within 0.3 dB of COLMAP (24.2, 23.9, 24.4 and 24.5 dB).
  DA3-Base refines as well as the larger checkpoints, takes about 4 s, and is the only DA3 size
  under Apache-2.0, so giro uses it.
- Bundle adjustment cannot place frames that too few triangulated points see: the same smeared
  frames COLMAP dropped. Plain refinement drops them again and the jump comes back (37–80°).
  giro keeps their DA3 pose instead, moved into the refined frame by a similarity fitted on the
  other cameras.
- On the seed with three smeared frames, this passed the gate with every frame placed: the largest
  jump went from 57° to 20°, in 12 s. It trained to 34.2 dB, the same as refinement without the
  kept frames (34.3 dB) on the same held-out frames. Without masks the jump went from 69° to 20°.
- Every feedforward model failed on turntables: they read the still room as a camera standing
  still. Graying the background with the masks did not fix it. Masked COLMAP handles those.
- SenseNova-Vision-7B-MoT, which writes poses as text tokens, scored 9–11 AUC@30 on 10-frame
  sets spread around the orbit, against 93–96 for DA3 and VGGT, and about 65 on nearby frames.

## Gap fill

A seed failed only because three motion-blurred frames went unplaced, leaving a 58° jump. giro
asked the video model for just that arc (22 frames, from the last good frame before the gap to the
first one after it) and spliced it in. The idea comes from OrbitForge's coverage-aware completion
([arXiv:2606.24799](https://arxiv.org/abs/2606.24799)).

- 3 of 3 fills passed the gate, with the largest jump down from 58° to 12–18°.
- PSNR was 34.4–35.2 dB, against 34.9 dB without the fill. The held-out frames differ, so this
  only shows it is no worse.
- Renders from inside the old gap went from smeared to clean.
- It cost about 40 s of video plus masks and poses for 20 frames. A reroll takes about 7 minutes
  and loses the seed.
- Name spliced frames so every tool sorts them the same way. `ls`, Python and Rust disagreed on
  `00074_01.png` against `00075.png`.

## Masks

- For SAM 3.1, "main subject" missed thin held objects such as swords, and adding "held object"
  caught them. Inverting a "wall, floor" mask pulled in the invented room's doors and seams.
- Masking the room before COLMAP helped on every seed tested. With the room gone, a spinning
  subject looks the same as an orbiting camera:
  - raincoat turntable: 17 → 85 of 85 frames placed, and it passed
  - knight turntable: 4 → 119 of 119, and it passed
  - the row-of-windows case: largest jump from 35° to 13°, and it passed
  - a good orbit: unchanged or slightly better
- Masks now run before poses. They cost 2–3 GPU minutes per seed.
- Unmasked training left halos and floor wisps. Brush's "masked" mode glowed around turntable
  subjects. "Transparent" mode was clean apart from a thin rim, which eroding the masks by 2 px
  removed.

## Training

- Brush ran 30k iterations in about 200 s on a 4090, and PSNR plateaued after about 6k. Side
  views, where the video is least consistent, showed room bleeding into the subject. That is what
  motivated masking and cropping.
- I also trained one scene with
  [Spirula Studio](https://github.com/harry7557558/spirula-studio), with the same frames and
  split, scored by one evaluator:

| Trainer | PSNR (full / subject) | Time |
|---|---|---|
| Brush, 425K splats | 24.7 / 20.2 dB | 267 s |
| Spirula Studio, 200K splats | 22.8 / 18.1 dB | 177 s |
| Spirula Studio, 425K splats | 22.8 / 18.2 dB | 193 s |

Spirula Studio trained 1.4–1.7× faster, and Brush scored higher on this scene. It is one scene and
one seed, and I did not tune Spirula Studio for generated video with a transparent background.
giro stays on Brush, which it was built around.

- A sweep of Brush options on three finished seeds (knight, scooter, tank), each scored after the
  crop: Gaussians kept, splat-transform's fill ratio (about how many layers are blended per pixel,
  which is what the Quest runs out of) and PSNR on the held-out frames.

| Training | Knight: kept / fill / PSNR | Tank | Scooter |
|---|---|---|---|
| Brush defaults | 122K / 62 / 33.5 dB | 156K / 88 / 30.6 dB | 124K / 29 / 23.5 dB |
| `--max-splats 80000` | 68K / 36 / 34.6 dB | 71K / 40 / 31.1 dB | 57K / 18 / 23.6 dB |
| `--max-splats 50000` | 44K / 24 / 34.7 dB | 46K / 25 / 30.8 dB | 39K / 14 / 23.6 dB |
| **80K and `--render-mode mip`** (default) | 70K / 34 / 34.5 dB | 72K / 36 / 31.1 dB | 58K / 18 / 23.6 dB |
| 10k iterations (45–49 s) | 58K / 25 / 33.1 dB | 80K / 32 / 30.4 dB | 75K / 18 / 23.4 dB |
| `--match-alpha-weight 1.0` | 256K / 215 / 33.1 dB | 296K / 151 / 30.3 dB | 290K / 127 / 23.4 dB |

- Capping the count halves the layers and costs nothing after the crop: an uncapped run spends
  a fifth of its Gaussians on near-transparent ones that the crop drops anyway.
- Train to the cap rather than decimating afterwards. The uncapped knight decimated to 44K scored
  32.4 dB, against 34.7 dB trained with a 50K cap.
- A stronger alpha loss doubled the count and the layers for no gain. The LPIPS loss was too slow
  to use: 200 iterations in 10 minutes.
- None of this has been compared in the headset yet.

## Crop

- The visual-hull crop keeps a Gaussian if it lands inside the subject mask in most of the views
  that see it. On its own, that rule kept 21K Gaussians of invented room, seen by only a few
  views.
- Visibility is bimodal: every Gaussian was seen by either under 20% or over 95% of views.
  Requiring at least half removed the room. The crop takes 2 s on the CPU.
- Every passing seed of the 4 subjects was reviewed from 16 directions. All were clean, and thin
  parts such as two swords and a scooter's mirrors survived. Exports were 85–123K Gaussians
  (2.4–3.4 MB as SPZ).

## Export and viewing

- splat-transform and PlayCanvas use the PLY frame rotated 180° about Z. Spark reads the file frame
  as-is, so the splat shows upside down. Rotating 180° about Z fixes it with the hero side facing
  +Z. The 180°-about-X flip in common examples also looks upright, but it turns the subject away.
- Spark 2.2 rejects SPZ v4, which is splat-transform's default, so giro passes `--spz-version 3`.
- Height is measured between the 0.5th and 99.5th percentiles of the Gaussian centers, so thin
  tops such as mirrors can stick out a few percent past the set height.

## VR on Quest 3

- I expected 400–600K splats at 72 Hz, based on Spark's guidance and native apps.
- In a first pass in the Quest browser (Spark 2.2, 90 Hz, SH degree 3), head motion already
  stuttered at 245K. The comfortable budget may be nearer 150K.
- That measurement was noisy: my head was moving, and another tab may have been open.
- A controlled pass (head still, 72 Hz, crowds of the knight 1.5 m away filling the view) showed
  the page is bound by fill rate, not by SH degree. At full resolution 245K ran at 32 fps with SH
  degree 3 and 34 fps with degree 1 or 0.

| Framebuffer scale, `maxStdDev` | 123K | 245K | 490K |
|---|---|---|---|
| 1.0, 2.83 (Spark's default) | – | 32 fps | 18 fps |
| 0.6, 2.83 | – | 56 fps | – |
| 0.6, 2.0 | 72 fps | 70 fps | 35 fps |

- So the VR page defaults to 72 Hz, scale 0.6 and `maxStdDev` 2.0, and the export budget is 150K
  Gaussians per subject. Published Quest budgets of 500K to 1M are for whole scenes; one subject
  filling the view at arm's length costs more per Gaussian.

## Gotchas

- Brush ignores `CUDA_VISIBLE_DEVICES`. Use `CUBECL_WGPU_DEFAULT_DEVICE='DiscreteGpu(1)'`.
- Brush treats any `.ply` in the dataset folder as the initial point cloud.
- Brush writes exports non-atomically, so a live preview must wait for the file size to settle.
- Brush logs nothing without a TTY unless `RUST_LOG=brush_cli=info` is set.
- COLMAP 4.x includes GLOMAP as `colmap global_mapper` and renames many options. Its mapper
  segfaulted once at random, so giro retries crashes.
- Two giro processes sharing a ComfyUI could kill each other's jobs. Stages now hold a file lock,
  and only the process that started an instance may stop it.
- `uvx ruff check --select F src tests` catches undefined names in 1 s. It would have caught a bug
  that only surfaced at the end of a 7-minute video.
- MapAnything's `fixed_mapping` crops, so giro uses `fixed_size`. Torch hub fetches unpinned
  DINOv2 code on first load.
