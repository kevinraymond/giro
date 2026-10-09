# Proxy orbit spike (Oct 3, 2026)

These became the proxy orbit mode on Oct 4 (`--model wan22-control`: `src/giro/stages/proxy.py`,
`src/giro/path.py`, `src/giro/stages/path_poses.py`); see docs/FINDINGS.md, "Proxy orbit". The
scripts stay as the record of the spike. `seedvr2_frames.py` is the SeedVR2 experiment that became
`orbit_video.upscale`.

Scratch scripts from the session that tried a proxy-guided orbit: TripoSplat makes a rough splat
of the hero, giro renders it as a depth video along a known camera path, and Wan 2.2 Fun Control
repaints that path with the hero as the appearance reference. They are kept to be turned into a
real `proxy` orbit mode; they are not part of the pipeline and take positional arguments only.

| Script | What it does |
|---|---|
| `proxy_orbit.py` | hero + hero mask -> TripoSplat -> depth path (flat ring, spiral or wave) -> Wan 2.2 Fun Control -> `frames_raw/`, `proxy.json` |
| `proxy_poses.py` | cameras from `proxy.json`, refined by COLMAP triangulation and bundle adjustment; writes `poses/fallback/` and `poses/frame.json` |
| `sheet.py` | 12-view review sheet of an attempt's canonical splat (no posed hero needed) |
| `triposplat_run.py`, `triposplat_compare.py` | TripoSplat alone on giro heroes, and side-by-side sheets against giro's splats |
| `sweep_train_analyze.py` | fill ratio and cropped-splat PSNR per `sweep_train.py` cell, and train-to-cap against decimation |

After `proxy_orbit.py`, run the stages with dedup off (its similarity test drops frames of a
small subject in a still room) and, until the hero is posed from the path, without the hero check:

    uv run giro stages ATTEMPT --from extract -p dedup.static_ssim=1.01 -p dedup.duplicate_ssim=1.01 \
        -p gate.require_hero=false -p gate.enforce=false

Results and numbers: docs/FINDINGS.md has the H3 and Wan orbit results; the proxy runs are in
`data/sweeps/20261003-proxy-orbit/` and on the board.
