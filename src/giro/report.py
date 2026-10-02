"""Export a job as one zip: an offline HTML report and the splats of every passing attempt.

    <job>/index.html                  the report (static images, inline SVG charts, no scripts)
    <job>/image.jpg                   the image the job orbits
    <job>/seed-<seed>/hero.jpg        the hero frame (the image at the video's aspect)
    <job>/seed-<seed>/result.jpg      the cropped splat beside the hero (crop/preview_hero.jpg)
    <job>/seed-<seed>/turnaround.jpg  the upright splat from several sides
    <job>/seed-<seed>/frames.jpg      every frame of the orbit, side by side
    <job>/seed-<seed>/orbit.mp4       the orbit video
    <job>/seed-<seed>/splat.{ply,sog,spz}

Passing attempts come in ranked order with result, orbit, cameras (top-down), training
(held-out PSNR) and the gate's numbers; the others get a line each with the reason.
"""

from __future__ import annotations

import html
import io
import json
import os
import subprocess
import threading
import time
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from giro import inspection, stages
from giro.job import Job

ROOT = Path(__file__).resolve().parents[2]
SPLATS = ("ply", "sog", "spz")
METRIC_LABELS = {  # the gate's checks, as the UI names them
    "reg_rate": "Images with a camera", "hero_registered": "Hero image placed", "n_views": "Frames with a camera",
    "reproj_err": "Reprojection error (px)", "azimuth_coverage": "Sweep around the subject (°)",
    "azimuth_monotonic": "Steps going forward", "max_step_deg": "Largest jump between frames (°)",
    "loop_closure": "Loop closure (radii)", "radius_cv": "Distance variation",
}


def _jpeg(src: Path, long_side: int) -> bytes:
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((long_side, long_side))
        out = io.BytesIO()
        im.save(out, format="JPEG", quality=88)
        return out.getvalue()


def _size(n: int) -> str:
    return f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{n / 2**10:.0f} KB"


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 10 else f"{v:.1f}"
    return html.escape(str(v))


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    """About n round tick values covering lo..hi."""
    span = max(hi - lo, 1e-9)
    raw = span / n
    step = min((m * 10 ** int(f"{raw:e}".split("e")[1]) for m in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    first = (lo // step) * step
    return [first + i * step for i in range(int((hi - first) / step) + 2) if lo - 1e-9 <= first + i * step <= hi + 1e-9]


def psnr_chart(series: list[dict[str, float]], total: int | None) -> str:
    """Held-out PSNR by iteration: one line, the last value labeled, a tooltip per point."""
    if not series:
        return '<p class="muted">No evaluation recorded.</p>'
    w, h, ml, mr, mt, mb = 640, 220, 44, 84, 12, 30
    xs = [p["step"] for p in series]
    ys = [p["value"] for p in series]
    x1 = max(total or 0, max(xs))
    y0, y1 = min(ys), max(ys)
    pad = max(0.5, (y1 - y0) * 0.1)
    y0, y1 = y0 - pad, y1 + pad
    sx = lambda x: ml + (w - ml - mr) * x / x1  # noqa: E731
    sy = lambda y: mt + (h - mt - mb) * (1 - (y - y0) / (y1 - y0))  # noqa: E731
    parts = []
    for t in _ticks(y0, y1):
        parts.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                     f'<text class="tick" x="{ml - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:g}</text>')
    for t in _ticks(0, x1):
        parts.append(f'<text class="tick" x="{sx(t):.1f}" y="{h - 10}" text-anchor="middle">{t / 1000:g}k</text>')
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys)))
    parts.append(f'<path class="line" d="{path}"/>')
    for x, y in zip(xs, ys):
        parts.append(f'<circle class="dot" cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4"><title>iteration {x:,}: {y:.2f} dB</title></circle>')
    parts.append(f'<text class="label" x="{sx(xs[-1]) + 8:.1f}" y="{sy(ys[-1]) + 4:.1f}">{ys[-1]:.2f} dB</text>')
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="Held-out PSNR by training iteration, ending at {ys[-1]:.2f} dB">{"".join(parts)}</svg>')


def camera_chart(cams: list[dict[str, Any]]) -> str:
    """The cameras seen from above (the upright frame: y up, the hero faces +z), each with a tick
    toward where it looks; the subject at the center."""
    if not cams:
        return '<p class="muted">No cameras.</p>'
    size, margin = 360, 28
    reach = max(max(abs(c["position"][0]), abs(c["position"][2])) for c in cams) or 1.0
    s = (size / 2 - margin) / reach
    px = lambda p: (size / 2 + p[0] * s, size / 2 + p[2] * s)  # noqa: E731 - +z (toward the hero view) is down
    parts = [f'<circle class="grid" cx="{size / 2}" cy="{size / 2}" r="{reach * s:.1f}" fill="none"/>',
             f'<path class="subject" d="M{size / 2 - 6},{size / 2}h12M{size / 2},{size / 2 - 6}v12"/>',
             f'<text class="tick" x="{size / 2 + 8}" y="{size / 2 - 8}">subject</text>']
    for c in sorted(cams, key=lambda c: c["hero"]):  # the hero last, on top
        x, y = px(c["position"])
        f = c["forward"]
        norm = (f[0] ** 2 + f[2] ** 2) ** 0.5 or 1.0
        tip = (x + 12 * f[0] / norm, y + 12 * f[2] / norm)
        err = f", {c['error']:.2f} px" if c.get("error") is not None else ""
        cls = "hero" if c["hero"] else "dot"
        parts.append(f'<g><title>{html.escape(c["name"])}{err}</title>'
                     f'<line class="look" x1="{x:.1f}" y1="{y:.1f}" x2="{tip[0]:.1f}" y2="{tip[1]:.1f}"/>'
                     f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{6 if c["hero"] else 4}"/></g>')
        if c["hero"]:
            parts.append(f'<text class="label" x="{x + 10:.1f}" y="{y + 16:.1f}">hero</text>')
    return (f'<svg class="chart cams" viewBox="0 0 {size} {size}" role="img" '
            f'aria-label="{len(cams)} cameras seen from above around the subject">{"".join(parts)}</svg>')


PERCENT = {"reg_rate", "azimuth_monotonic", "radius_cv"}  # fractions, shown as percentages


def _gate_table(gate: dict[str, Any]) -> str:
    rows = []
    for c in gate.get("checks", []):
        ok = c["pass"]
        pct = c["metric"] in PERCENT and isinstance(c["value"], (int, float)) and not isinstance(c["value"], bool)
        value, threshold = (f"{c['value']:.1%}", f"{c['threshold']:.0%}") if pct else (_fmt(c["value"]), _fmt(c["threshold"]))
        rows.append(f'<tr class="{"" if ok else "fail"}"><td>{"✓" if ok else "✗"}</td>'
                    f'<td>{METRIC_LABELS.get(c["metric"], c["metric"])}</td><td>{value}</td>'
                    f'<td class="muted">{html.escape(c["op"])} {threshold}</td></tr>')
    return f'<table class="kv"><tbody>{"".join(rows)}</tbody></table>'


def _attempt(job: Job, a: Any, rank: int, zf: zipfile.ZipFile, prefix: str) -> str:
    path = job.attempt_dir(a.seed)
    sub = f"seed-{a.seed}"
    metrics = json.loads((path / "metrics.json").read_text()) if (path / "metrics.json").exists() else {}
    gate = json.loads((path / "gate.json").read_text()) if (path / "gate.json").exists() else {}
    train = metrics.get("train", {})

    def add(src: Path | None, name: str, data: bytes | None = None) -> str | None:
        if data is None and (src is None or not src.exists()):
            return None
        if data is not None:
            zf.writestr(f"{prefix}/{sub}/{name}", data)
        else:
            zf.write(src, f"{prefix}/{sub}/{name}", compress_type=zipfile.ZIP_STORED)
        return f"{sub}/{name}"

    hero = add(None, "hero.jpg", _jpeg(path / "hero" / "hero.png", 1280)) if (path / "hero" / "hero.png").exists() else None
    result = add(path / "crop" / "preview_hero.jpg", "result.jpg")
    turnaround = add(path / "canonical" / "turnaround.jpg", "turnaround.jpg")
    frames = add(inspection.frame_sheet(path), "frames.jpg")
    video = add(path / "video.mp4", "orbit.mp4")
    downloads = []
    for ext in SPLATS:
        if link := add(path / "export" / f"splat.{ext}", f"splat.{ext}"):
            downloads.append(f'<a class="btn" href="{link}" download>{ext.upper()} '
                             f'<span class="muted">{_size((path / "export" / f"splat.{ext}").stat().st_size)}</span></a>')

    cams = (inspection.cameras(path) or {}).get("cameras", [])
    curve = inspection.training(path)
    ring = gate.get("ring", {})
    fig = lambda src, cap: f'<figure><img src="{src}" alt="{cap}" loading="lazy"><figcaption>{cap}</figcaption></figure>' if src else ""  # noqa: E731
    facts = [("Held-out PSNR", f"{a.eval_psnr:.2f} dB" if a.eval_psnr is not None else "—"),
             ("SSIM", f"{train['eval_ssim']:.3f}" if "eval_ssim" in train else "—"),
             ("Gaussians", f"{a.gaussians:,}" if a.gaussians else "—"),
             ("Cameras from", "the pose fallback (Depth Anything 3 + COLMAP)" if a.poses == "fallback" else "COLMAP"),
             ("Total time", f"{a.seconds / 60:.0f} min" if a.seconds else "—")]
    if a.filled:
        facts.append(("Gap filled", "; ".join(a.filled)))
    # What this seed actually ran with: the job's params under its own overrides (e.g. a new prompt).
    masks = stages.BY_NAME["masks"].defaults | a.stage_params(job.spec, "masks")
    canon = stages.BY_NAME["canonicalize"].defaults | a.stage_params(job.spec, "canonicalize")
    facts.insert(0, ("What to keep", str(masks["subject_prompt"])))
    facts.insert(1, ("Subject height", f'{canon["height_m"]} m'))
    if a.params:
        facts.append(("Changed for this seed", "; ".join(f"{st}.{k} = {v}" for st, vals in a.params.items() for k, v in vals.items())))
    fact_rows = "".join(f"<tr><td>{k}</td><td>{html.escape(v)}</td></tr>" for k, v in facts)
    return f"""
<section class="attempt">
  <h2><span class="rank">#{rank}</span> Seed {a.seed}</h2>
  <div class="downloads">{"".join(downloads)}</div>
  <h3>Result</h3>
  <div class="figs">{fig(result, "The cropped splat beside the hero image")}{fig(turnaround, "The upright splat from several sides")}</div>
  <h3>Orbit</h3>
  <div class="figs">{fig(hero, "Hero frame: the image at the video's aspect")}
  {f'<figure><video src="{video}" controls muted loop playsinline preload="metadata"></video><figcaption>The orbit video</figcaption></figure>' if video else ""}</div>
  {f'<div class="strip"><img src="{frames}" alt="Every frame of the orbit" loading="lazy"></div><p class="caption">Every extracted frame, in order.</p>' if frames else ""}
  <h3>Cameras</h3>
  <div class="split">
    {camera_chart(cams)}
    <div>
      <p>{len([c for c in cams if not c["hero"]])} frame cameras{", the hero placed" if any(c["hero"] for c in cams) else ""};
      the camera swept {_fmt(ring.get("azimuth_span", "—"))}° around the subject.
      Seen from above; each tick points where the camera looks. Hover a camera for its frame and reprojection error.</p>
      <h4>Gate</h4>
      {_gate_table(gate)}
    </div>
  </div>
  <h3>Training</h3>
  <p class="caption">Held-out PSNR (dB) by training iteration.</p>
  {psnr_chart(curve["psnr"], curve["total_iters"])}
  <h3>Numbers</h3>
  <table class="kv"><tbody>{fact_rows}</tbody></table>
</section>"""


def _git_version() -> str:
    r = subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def build(job: Job, out: Path) -> Path:
    """Write the job's archive to `out` (a .zip) and return it."""
    prefix = job.path.name
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp")  # two exports at once don't collide
    passed = [job.attempt(s) for s in job.ranking]
    others = [a for a in job.attempts if a.seed not in job.ranking and a.status != "discarded"]
    spec = job.spec
    orbit = spec.orbit
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        source = job.hero_source()
        zf.writestr(f"{prefix}/image.jpg", _jpeg(source, 1280))
        sections = "".join(_attempt(job, a, i + 1, zf, prefix) for i, a in enumerate(passed))
        other_rows = "".join(
            f'<tr><td>{a.seed}</td><td>{html.escape(a.status)}</td><td>{html.escape(a.stage)}</td>'
            f'<td>{html.escape(a.reason)}</td></tr>' for a in others)
        settings = [("Video", f'{orbit.get("width", 768)} × {orbit.get("height", 1024)}, {orbit.get("length", 158)} frames'),
                    ("What to keep", spec.params.get("masks", {}).get("subject_prompt", "main subject, held object:2")),
                    ("Subject height", f'{spec.params.get("canonicalize", {}).get("height_m", 1.7)} m'),
                    ("Background edit", (job.edit or {}).get("prompt", "") if spec.use_edit else "none"),
                    ("Crop", "chosen region" if spec.crop else "centered")]
        setting_rows = "".join(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>" for k, v in settings)
        name = html.escape(prefix.split("-", 2)[-1] if prefix.count("-") >= 2 else prefix)
        page = PAGE.format(
            title=name, name=name, created=html.escape(job.created.replace("T", " ")),
            summary=f"{len(passed)} good orbit{'s' if len(passed) != 1 else ''} of {len([a for a in job.attempts if a.status != 'discarded'])} seeds tried",
            settings=setting_rows, sections=sections or '<p class="muted">No attempt passed.</p>',
            others=(f'<section><h2>Other seeds</h2><table class="kv"><thead><tr><td>Seed</td><td>Status</td><td>Stopped at</td>'
                    f'<td>Reason</td></tr></thead><tbody>{other_rows}</tbody></table></section>') if others else "",
            made=time.strftime("%Y-%m-%d %H:%M"), version=html.escape(_git_version()),
        )
        zf.writestr(f"{prefix}/index.html", page)
    tmp.replace(out)
    return out


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{
  --bg: #ffffff; --panel: #f5f6f7; --line: #dfe2e6; --text: #1b1e22; --muted: #5f6873;
  --mark: #a8700f; --good: #1f8a4c; --bad: #c4392f;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #0f1113; --panel: #171a1d; --line: #2a2f35; --text: #e7e9ec; --muted: #8b939c;
          --mark: #b8841f; --good: #5cc98a; --bad: #ef6b63; }}
}}
body {{ margin: 0; background: var(--bg); color: var(--text);
       font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }}
header {{ display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; }}
header > div {{ flex: 1 1 260px; min-width: 0; }}
h1, td {{ overflow-wrap: break-word; }}
header img {{ width: 200px; border-radius: 10px; }}
h1 {{ margin: 0 0 4px; font-size: 28px; }}
h2 {{ margin: 40px 0 8px; font-size: 22px; border-top: 1px solid var(--line); padding-top: 24px; }}
h3 {{ margin: 28px 0 8px; font-size: 16px; }}
h4 {{ margin: 16px 0 6px; font-size: 14px; }}
.rank {{ color: var(--muted); font-weight: 500; }}
.muted, .caption, figcaption {{ color: var(--muted); }}
.caption, figcaption {{ font-size: 13px; margin: 4px 0 0; }}
.figs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
figure {{ margin: 0; }}
figure img, figure video {{ width: 100%; border-radius: 8px; background: var(--panel); display: block; }}
.strip {{ overflow-x: auto; margin-top: 12px; border-radius: 8px; background: var(--panel); }}
.strip img {{ height: 96px; display: block; }}
.split {{ display: grid; grid-template-columns: minmax(0, 360px) minmax(0, 1fr); gap: 24px; align-items: start; }}
@media (max-width: 720px) {{ .split {{ grid-template-columns: 1fr; }} }}
.downloads {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.btn {{ display: inline-block; padding: 6px 12px; border: 1px solid var(--line); border-radius: 8px; color: var(--text);
        text-decoration: none; background: var(--panel); font-weight: 600; }}
.btn:hover {{ border-color: var(--mark); }}
table.kv {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
.kv td {{ padding: 5px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }}
.kv thead td {{ color: var(--muted); font-weight: 600; }}
.kv tr.fail td {{ color: var(--bad); }}
.kv td:first-child {{ white-space: nowrap; }}
svg.chart {{ width: 100%; max-width: 640px; height: auto; display: block; }}
svg.cams {{ max-width: 360px; }}
.chart .grid {{ stroke: var(--line); stroke-width: 1; }}
.chart .tick {{ fill: var(--muted); font-size: 11px; }}
.chart .label {{ fill: var(--text); font-size: 12px; font-weight: 600; }}
.chart .line {{ fill: none; stroke: var(--mark); stroke-width: 2; stroke-linejoin: round; }}
.chart .dot {{ fill: var(--mark); stroke: var(--bg); stroke-width: 2; }}
.chart .hero {{ fill: var(--bg); stroke: var(--mark); stroke-width: 3; }}
.chart .look {{ stroke: var(--muted); stroke-width: 1.5; }}
.chart .subject {{ stroke: var(--text); stroke-width: 2; fill: none; }}
.chart circle:hover {{ stroke: var(--text); }}
footer {{ margin-top: 48px; color: var(--muted); font-size: 13px; }}
</style>
</head>
<body>
<main>
<header>
  <img src="image.jpg" alt="The input image">
  <div>
    <h1>{name}</h1>
    <p class="muted">{summary} · {created}</p>
    <table class="kv"><tbody>{settings}</tbody></table>
  </div>
</header>
{sections}
{others}
<footer>Exported {made} by giro {version}. Splats: PLY (3DGS, any viewer), SOG (PlayCanvas / SuperSplat), SPZ (Niantic, compact).</footer>
</main>
</body>
</html>
"""
