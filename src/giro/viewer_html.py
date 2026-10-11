"""The all-in-one viewer: a single HTML file with the splat inside (base64 SPZ), the viewer script
inlined, no network. Built from ui/standalone/template.html and ui/dist/standalone/viewer.js
(`npm run build` in ui/, as for the rest of the UI)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "ui" / "standalone" / "template.html"
BUNDLE = ROOT / "ui" / "dist" / "standalone" / "viewer.js"


class ViewerNotBuilt(RuntimeError):
    pass


def write_standalone(spz: Path, out: Path, title: str, meta: dict[str, Any] | None = None,
                     template: Path = TEMPLATE, bundle: Path = BUNDLE) -> int:
    """Write `out` and return its size in bytes. `meta` reaches the page as JSON:
    units_per_meter (1), up ("y"), floor (grid height in meters, null for none), notes."""
    if not bundle.exists():
        raise ViewerNotBuilt(f"the standalone viewer is not built; run `npm run build` in ui/ ({bundle})")
    page_meta = {"title": title, "units_per_meter": 1, "up": "y", "floor": 0.0} | (meta or {})
    html = template.read_text()
    for key, value in (
        ("TITLE", title.replace("<", "&lt;")),
        ("META", json.dumps(page_meta).replace("</", "<\\/")),
        ("SPLAT_B64", base64.b64encode(spz.read_bytes()).decode("ascii")),
        ("VIEWER_JS", bundle.read_text().replace("</script", "<\\/script")),
    ):
        html = html.replace("{{" + key + "}}", value)
    out.write_text(html)
    return out.stat().st_size
