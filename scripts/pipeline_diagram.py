"""Draw the README's pipeline diagram as two SVGs, one per GitHub theme.

    uv run scripts/pipeline_diagram.py

Writes docs/images/pipeline-light.svg and pipeline-dark.svg. The README shows the one that
matches the viewer's theme through a <picture> element.

The per-seed loop is drawn as what it is: an orbit. The image sits in the middle, the stages
ride the ring (its dots are the camera positions the gate checks), the gate decides at the
end of a turn, and the splat falls out to the right. Colors: orange for what goes in and what
comes out, blue for the gate and its outcomes, neutral for everything else.
"""

import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"
W, H = 900, 560
FONT = "ui-monospace, 'JetBrains Mono', SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# fosfora's orange (#ff6040) and blue (#5090e0, #3a70c0): the same two accents everywhere.
THEMES = {
    "light": dict(
        node="#ffffff", node_stroke="#8c959f", text="#1f2328", muted="#59636e",
        io="#ffece6", io_stroke="#ff6040", gate="#e3edfa", gate_stroke="#3a70c0",
        retry="#eef4fc", retry_stroke="#5090e0", ring="#b6bec7", arrow="#6e7781", bg="#ffffff",
    ),
    "dark": dict(
        node="#0d1117", node_stroke="#6e7681", text="#e6edf3", muted="#9198a1",
        io="#4a2317", io_stroke="#ff6040", gate="#14305c", gate_stroke="#5090e0",
        retry="#1a2740", retry_stroke="#5090e0", ring="#484f58", arrow="#8b949e", bg="#0d1117",
    ),
}

CX, CY, R = 300, 300, 190     # the orbit
NODE_W, NODE_H = 112, 46
RING = {"orbit": 90, "frames": 162, "masks": 234, "poses": 306, "gate": 18}  # degrees, counterclockwise
GATE_R = 44


def pt(deg: float, r: float = R) -> tuple[float, float]:
    a = math.radians(deg)
    return CX + r * math.cos(a), CY - r * math.sin(a)


def f(v: float) -> str:
    return f"{v:.1f}".rstrip("0").rstrip(".")


def box(cx, cy, title, sub, kind="node", w=NODE_W, h=NODE_H, dashed=False, pill=False):
    x, y = cx - w / 2, cy - h / 2
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    r = h / 2 if pill else 2
    return (
        f'<rect x="{f(x)}" y="{f(y)}" width="{w}" height="{h}" rx="{r}" class="{kind}"{dash}/>'
        f'<text x="{f(cx)}" y="{f(cy - 3 if sub else cy + 4.5)}" class="title">{title}</text>'
        + (f'<text x="{f(cx)}" y="{f(cy + 13)}" class="sub">{sub}</text>' if sub else "")
    )


def diamond(cx, cy, r, title, sub):
    pts = f"{f(cx)},{f(cy - r)} {f(cx + r)},{f(cy)} {f(cx)},{f(cy + r)} {f(cx - r)},{f(cy)}"
    return (
        f'<polygon points="{pts}" class="gate"/>'
        f'<text x="{f(cx)}" y="{f(cy - 2)}" class="title">{title}</text>'
        f'<text x="{f(cx)}" y="{f(cy + 13)}" class="sub">{sub}</text>'
    )


def arc(a0: float, a1: float, r: float = R, cls: str = "edge", dashed=False) -> str:
    """An arrowed arc from angle a0 to a1 (degrees), the short way round."""
    x0, y0 = pt(a0, r)
    x1, y1 = pt(a1, r)
    ccw = (a1 - a0) % 360 < 180
    sweep = 0 if ccw else 1   # SVG's y points down, so counterclockwise on screen is sweep 0
    large = 0
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return (f'<path d="M{f(x0)} {f(y0)} A{r} {r} 0 {large} {sweep} {f(x1)} {f(y1)}" '
            f'class="{cls}"{dash} marker-end="url(#arrow)"/>')


def line(d: str, cls="edge", dashed=False) -> str:
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    return f'<path d="{d}" class="{cls}"{dash} marker-end="url(#arrow)"/>'


def label(x, y, text, anchor="start", cls="edge-label"):
    return f'<text x="{f(x)}" y="{f(y)}" class="{cls}" style="text-anchor: {anchor}">{text}</text>'


def svg(t: dict) -> str:
    gap = 17            # degrees an arc stops short of a node's center (about the node's half width)
    order = ["orbit", "frames", "masks", "poses", "gate"]
    parts = []

    # "for each seed" rides the outside of the ring, top left (text runs clockwise, so left to right)
    tx0, ty0 = pt(170, R + 18)
    tx1, ty1 = pt(100, R + 18)
    parts.append(f'<path id="seedpath" d="M{f(tx0)} {f(ty0)} A{R + 18} {R + 18} 0 0 1 {f(tx1)} {f(ty1)}" fill="none"/>')
    parts.append('<text class="eyebrow"><textPath href="#seedpath" startOffset="50%" style="text-anchor: middle">'
                 'FOR EACH SEED</textPath></text>')

    # the flow around the ring: four plain arcs, then the gate's two blue outcomes
    for a, b in zip(order, order[1:]):
        parts.append(arc(RING[a] + gap, RING[b] - gap))
    parts.append(arc(RING["gate"] + gap, RING["orbit"] - gap, cls="edge blue"))          # fail: reroll
    lx, ly = pt(54, R + 14)
    parts.append(label(lx + 2, ly, "fail: new seed", cls="edge-label blue"))
    # the ring's camera positions, every 6 degrees, drawn over the arcs like beads on the track
    for deg in range(0, 360, 6):
        x, y = pt(deg)
        parts.append(f'<circle cx="{f(x)}" cy="{f(y)}" r="2.4" class="cam"/>')
    # gap fill: back from the gate to the masks, inside the ring, through a small blue box
    gx, gy, gw = CX, 404, 104
    gcx0, gcy0 = pt(RING["gate"])
    mx, my = pt(RING["masks"])
    parts.append(line(f"M{f(gcx0)} {f(gcy0 + GATE_R + 2)} Q{f(gcx0 - 50)} {gy} {f(gx + gw / 2 + 2)} {gy}", cls="edge blue", dashed=True))
    parts.append(line(f"M{f(gx - gw / 2)} {gy} Q{f(mx + 10)} {gy} {f(mx + 10)} {f(my - NODE_H / 2 - 2)}", cls="edge blue", dashed=True))
    parts.append(box(gx, gy, "Gap fill", "ONE ARC MISSING", "retry", w=gw, h=38))

    # the image in the middle, the optional edit on its way up to the orbit
    ox, oy = pt(RING["orbit"])
    parts.append(box(CX, 330, "One image", "", "io", w=110, h=40, pill=True))
    parts.append(line(f"M{CX} 310 V254"))
    parts.append(box(CX, 230, "Background edit", "QWEN · OPTIONAL", dashed=True, w=130, h=44))
    parts.append(line(f"M{CX} 208 V{f(oy + NODE_H / 2 + 2)}"))

    # the stages on the ring
    parts.append(box(*pt(RING["orbit"]), "Orbit video", "PROXY · H3 · YOURS"))
    parts.append(box(*pt(RING["frames"]), "Frames", "TRIM · DEDUP"))
    parts.append(box(*pt(RING["masks"]), "Masks", "SAM 3.1"))
    parts.append(box(*pt(RING["poses"]), "Poses", "COLMAP"))
    gcx, gcy = pt(RING["gate"])
    parts.append(diamond(gcx, gcy, GATE_R, "Gate", "RING OK?"))

    # pass: out of the orbit and down the column to the splat
    col_x, w = 745, 170
    rows = [241, 330, 419, 508]
    parts.append(line(f"M{f(gcx + GATE_R + 2)} {f(gcy)} H{f(col_x - w / 2 - 2)}"))
    parts.append(label(gcx + GATE_R + 10, gcy - 8, "pass"))
    parts.append(box(col_x, rows[0], "Train", "BRUSH · HERO VIEW", w=w))
    parts.append(box(col_x, rows[1], "Crop", "VISUAL HULL", w=w))
    parts.append(box(col_x, rows[2], "Upright + scale", "REAL HEIGHT", w=w))
    parts.append(box(col_x, rows[3], "Splat", "SPZ · SOG · PLY · HTML", "io", w=w, h=44, pill=True))
    for y0, y1 in zip(rows, rows[1:]):
        parts.append(line(f"M{col_x} {f(y0 + NODE_H / 2)} V{f(y1 - NODE_H / 2 - 2)}"))

    style = f"""
      text {{ font-family: {FONT}; text-anchor: middle; }}
      .title {{ font-size: 13px; font-weight: 600; fill: {t['text']}; }}
      .sub {{ font-size: 9.5px; letter-spacing: 0.06em; fill: {t['muted']}; }}
      .eyebrow {{ font-size: 10.5px; font-weight: 600; letter-spacing: 0.18em; fill: {t['muted']}; }}
      .node {{ fill: {t['node']}; stroke: {t['node_stroke']}; stroke-width: 1.3; }}
      .io {{ fill: {t['io']}; stroke: {t['io_stroke']}; stroke-width: 1.6; }}
      .gate {{ fill: {t['gate']}; stroke: {t['gate_stroke']}; stroke-width: 1.6; }}
      .retry {{ fill: {t['retry']}; stroke: {t['retry_stroke']}; stroke-width: 1.3; }}
      .cam {{ fill: {t['bg']}; stroke: {t['arrow']}; stroke-width: 1.1; }}
      .edge {{ fill: none; stroke: {t['arrow']}; stroke-width: 1.5; stroke-linejoin: round; }}
      .edge.blue {{ stroke: {t['retry_stroke']}; marker-end: url(#arrow-blue); }}
      .edge-label {{ font-size: 11px; font-style: italic; fill: {t['muted']}; }}
      .edge-label.blue {{ fill: {t['retry_stroke']}; font-style: normal; font-weight: 600; letter-spacing: 0.04em; font-size: 10.5px; }}
    """
    marker = ('<marker id="{id}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
              'orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="{fill}"/></marker>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 60 {W} {H - 60}" width="{W}" height="{H - 60}" '
        f'role="img" aria-label="giro pipeline">'
        f'<defs>{marker.format(id="arrow", fill=t["arrow"])}{marker.format(id="arrow-blue", fill=t["retry_stroke"])}</defs>'
        f"<style>{style}</style>" + "".join(parts) + "</svg>\n"
    )


if __name__ == "__main__":
    for name, theme in THEMES.items():
        (OUT / f"pipeline-{name}.svg").write_text(svg(theme))
        print(OUT / f"pipeline-{name}.svg")
