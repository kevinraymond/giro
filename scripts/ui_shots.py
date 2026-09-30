"""Screenshot giro UI routes in headless Chrome (the UI's visual check).

    uv run --with playwright python scripts/ui_shots.py OUTDIR ROUTE=NAME ...

ROUTE is the part after '#/' (e.g. job/<id>/<seed>/crop, or empty for the library);
a NAME starting with m- renders at phone width. PORT picks the server (default 8470).
Uses the system Chrome with GPU flags so Spark (WebGL2) renders.
"""
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome",
                          args=["--use-angle=vulkan", "--enable-gpu", "--ignore-gpu-blocklist", "--enable-unsafe-swiftshader"])
    for arg in sys.argv[2:]:
        route, _, name = arg.partition("=")
        page = b.new_page(viewport={"width": 390 if name.startswith("m-") else 1440, "height": 900})
        errors: list[str] = []
        page.on("console", lambda m: m.type in ("error", "warning") and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(f"http://127.0.0.1:{os.environ.get('PORT', '8470')}/#/{route}")
        page.wait_for_timeout(6000)  # splats load and sort
        page.screenshot(path=str(out / f"{name or 'home'}.png"), full_page=True)
        print(name or "home", "errors:", errors[:4])
        page.close()
    b.close()
