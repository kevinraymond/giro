import base64
import json
from pathlib import Path

import pytest

from giro import viewer_html


def test_standalone_inlines_everything(tmp_path: Path) -> None:
    bundle = tmp_path / "viewer.js"
    bundle.write_text('console.log("</script>")')
    spz = tmp_path / "splat.spz"
    spz.write_bytes(b"\x1f\x8b not really spz")
    out = tmp_path / "splat.html"
    size = viewer_html.write_standalone(spz, out, "Tree <1>", {"notes": "hi"}, bundle=bundle)
    html = out.read_text()
    assert size == out.stat().st_size
    assert base64.b64encode(spz.read_bytes()).decode() in html
    assert "<\\/script" in html and html.count("</script>") == 3  # the bundle's own tag is escaped
    meta = json.loads(html.split('id="giro-meta" type="application/json">')[1].split("</script>")[0])
    assert meta == {"title": "Tree <1>", "units_per_meter": 1, "up": "y", "floor": 0.0, "notes": "hi"}
    assert "<title>Tree &lt;1></title>" in html
    assert "{{" not in html


def test_unbuilt_viewer_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "x.spz").write_bytes(b"")
    with pytest.raises(viewer_html.ViewerNotBuilt):
        viewer_html.write_standalone(tmp_path / "x.spz", tmp_path / "x.html", "x", bundle=tmp_path / "missing.js")
