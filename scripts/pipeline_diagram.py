"""Draw the README's pipeline diagram as two SVGs, one per GitHub theme.

    uv run scripts/pipeline_diagram.py

Writes docs/images/pipeline-light.svg and pipeline-dark.svg. The README shows
the one that matches the viewer's theme through a <picture> element.
"""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
W, H = 1000, 530
TOP = 50  # nothing is drawn above this
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

THEMES = {
    "light": dict(
        node="#f6f8fa", node_stroke="#d0d7de", text="#1f2328", muted="#59636e",
        io="#fdf1dc", io_stroke="#d4861a", gate="#ddf4ff", gate_stroke="#0969da",
        retry="#fff1ec", retry_stroke="#cf5f3a", group="#8c959f", arrow="#6e7781",
    ),
    "dark": dict(
        node="#161b22", node_stroke="#3d444d", text="#e6edf3", muted="#9198a1",
        io="#3a2a10", io_stroke="#e8a33d", gate="#0c2d4f", gate_stroke="#4493f8",
        retry="#3d1d14", retry_stroke="#e0724f", group="#6e7681", arrow="#8b949e",
    ),
}

NODE_H = 62


def box(x, y, w, title, sub, kind="node", h=NODE_H, dashed=False, pill=False):
    r = h / 2 if pill else 10
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    cy = y + h / 2
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{kind}"{dash}/>'
        f'<text x="{x + w / 2}" y="{cy - 3 if sub else cy + 5}" class="title">{title}</text>'
        + (f'<text x="{x + w / 2}" y="{cy + 15}" class="sub">{sub}</text>' if sub else "")
    )


def diamond(cx, cy, r, title, sub):
    pts = f"{cx},{cy - r} {cx + r},{cy} {cx},{cy + r} {cx - r},{cy}"
    return (
        f'<polygon points="{pts}" class="gate"/>'
        f'<text x="{cx}" y="{cy - 2}" class="title">{title}</text>'
        f'<text x="{cx}" y="{cy + 15}" class="sub">{sub}</text>'
    )


def path(d, dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return f'<path d="{d}" class="edge"{dash} marker-end="url(#arrow)"/>'


def label(x, y, text, anchor="start"):
    return f'<text x="{x}" y="{y}" class="edge-label" style="text-anchor: {anchor}">{text}</text>'


def svg(t: dict) -> str:
    row1, row2 = 200, 432              # node tops
    c1 = row1 + NODE_H / 2             # row 1 center line
    gx, gr = 832, 52                   # gate center x, half-diagonal
    video, frames, masks, poses = 180, 330, 480, 630
    w = 118

    parts = [
        # the per-seed loop
        '<rect x="158" y="62" width="792" height="312" rx="16" class="group"/>',
        '<text x="178" y="88" class="group-label">FOR EACH SEED</text>',
        # input and optional edit
        box(20, row1, 112, "One image", "", "io", pill=True),
        box(10, 88, 138, "Background edit", "optional", dashed=True),
        path(f"M79 {row1} V{88 + NODE_H + 2}", dashed=True),
        path(f"M148 119 H{video + w / 2} V{row1 - 2}", dashed=True),
        path(f"M132 {c1} H{video - 2}"),
        # the main row
        box(video, row1, w, "Orbit video", "MiniMax H3"),
        box(frames, row1, w, "Frames", "trim · dedup"),
        box(masks, row1, w, "Masks", "SAM 3.1"),
        box(poses, row1, w, "Poses", "COLMAP · DA3"),  # DA3: the pose fallback
        diamond(gx, c1, gr, "Gate", "clean ring?"),
        path(f"M{video + w} {c1} H{frames - 2}"),
        path(f"M{frames + w} {c1} H{masks - 2}"),
        path(f"M{masks + w} {c1} H{poses - 2}"),
        path(f"M{poses + w} {c1} H{gx - gr - 2}"),
        # gap fill: regenerate one arc, then masks, poses and gate again
        box(670, 94, 124, "Gap fill", "regenerate the arc", "retry", h=54),
        path(f"M{gx} {c1 - gr} V121 H{794 + 2}"),
        label(gx + 10, 170, "one arc missing"),
        path(f"M670 121 H{masks + w / 2} V{row1 - 2}"),
        # reroll: a new seed
        box(gx - 55, 312, 110, "Reroll", "new seed", "retry", h=48),
        path(f"M{gx} {c1 + gr} V{312 - 2}"),
        label(gx + 10, 303, "fail"),
        path(f"M{gx - 55} 336 H{video + w / 2} V{row1 + NODE_H + 2}"),
        # pass: on to the splat
        path(f"M{gx + gr} {c1} H976 V402 H{video + 65} V{row2 - 2}"),
        label(gx + gr + 10, c1 - 8, "pass"),
        box(video, row2, 130, "Train", "Brush · transparent"),
        box(360, row2, 130, "Crop", "visual hull"),
        box(540, row2, 130, "Upright + scale", "real height"),
        box(720, row2, 150, "Splat", "SPZ · SOG · PLY", "io", pill=True),
        path(f"M310 {row2 + NODE_H / 2} H{360 - 2}"),
        path(f"M490 {row2 + NODE_H / 2} H{540 - 2}"),
        path(f"M670 {row2 + NODE_H / 2} H{720 - 2}"),
    ]
    style = f"""
      text {{ font-family: {FONT}; text-anchor: middle; }}
      .title {{ font-size: 14px; font-weight: 600; fill: {t['text']}; }}
      .sub {{ font-size: 11.5px; fill: {t['muted']}; }}
      .node {{ fill: {t['node']}; stroke: {t['node_stroke']}; stroke-width: 1.2; }}
      .io {{ fill: {t['io']}; stroke: {t['io_stroke']}; stroke-width: 1.5; }}
      .gate {{ fill: {t['gate']}; stroke: {t['gate_stroke']}; stroke-width: 1.5; }}
      .retry {{ fill: {t['retry']}; stroke: {t['retry_stroke']}; stroke-width: 1.3; }}
      .group {{ fill: none; stroke: {t['group']}; stroke-width: 1.2; stroke-dasharray: 6 5; }}
      .group-label {{ font-size: 11px; font-weight: 600; letter-spacing: 0.08em; fill: {t['muted']}; text-anchor: start; }}
      .edge {{ fill: none; stroke: {t['arrow']}; stroke-width: 1.5; stroke-linejoin: round; }}
      .edge-label {{ font-size: 12px; font-style: italic; fill: {t['muted']}; }}
    """
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {TOP} {W} {H - TOP}" width="{W}" height="{H - TOP}" '
        f'role="img" aria-label="giro pipeline">'
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{t["arrow"]}"/></marker></defs>'
        f"<style>{style}</style>" + "".join(parts) + "</svg>\n"
    )


if __name__ == "__main__":
    for name, theme in THEMES.items():
        (OUT / f"pipeline-{name}.svg").write_text(svg(theme))
        print(OUT / f"pipeline-{name}.svg")
