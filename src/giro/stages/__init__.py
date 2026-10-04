from giro.stages.base import Cancelled, Ctx, Rejected, Stage, StageFailed
from giro.stages.canonicalize import Canonicalize
from giro.stages.crop import Crop
from giro.stages.edit import EditImage
from giro.stages.export import Export
from giro.stages.fallback import PoseFallback
from giro.stages import gapfill
from giro.stages.frames import Dedup, Extract
from giro.stages.gapfill import GapFill
from giro.stages.gate import Gate
from giro.stages import poses
from giro.stages.masks import Masks
from giro.stages.orbit import OrbitVideo
from giro.stages.poses import ColmapPoses
from giro.stages.proxy import Proxy
from giro.stages.train import Dataset, Train

# Every stage of an attempt in order; the orbit video comes first.
ORBIT = OrbitVideo()
# Masks come before poses: COLMAP matches only the subject, which turns a video that spins
# the subject in front of a still room into a usable orbit (docs/FINDINGS.md, "Masks").
PIPELINE: list[Stage] = [ORBIT, Extract(), Dedup(), Masks(), ColmapPoses(), Gate(), Dataset(), Train(),
                         Crop(), Canonicalize(), Export()]
# The proxy orbit (orbit model "wan22-control"): a TripoSplat proxy of the hero comes first, its
# depth along a known camera path drives the video, and the path's cameras are the poses
# (docs/FINDINGS.md, "Proxy orbit").
PROXY = Proxy()
PROXY_MODEL = "wan22-control"
# What the proxy orbit sets under the user's own per-stage params: every frame has a known
# camera (no dedup), COLMAP refines the path's cameras instead of mapping, the hero, the
# video's exact first frame and its only real view, counts five times in training, and training
# keeps up to 150K splats for 60K iterations: its SeedVR2-sharpened frames carry detail an 80K
# cap throws away (docs/FINDINGS.md, "Proxy orbit").
PROXY_PARAMS: dict[str, dict] = {
    "dedup": {"keep_all": True},
    "poses_colmap": {"mapper": "path"},
    "dataset": {"hero_copies": 5},
    "train": {"max_splats": 150_000, "total_train_iters": 60_000},
}
BY_NAME: dict[str, Stage] = {s.name: s for s in [PROXY, *PIPELINE]}


# The orbit model new jobs get (Oct 4, 2026: the proxy orbit; docs/FINDINGS.md, "Proxy orbit").
# It is written into each new job's orbit params, so attempts from before the switch, which
# recorded no model, stay MiniMax H3 (orbit_video's own fallback).
DEFAULT_MODEL = PROXY_MODEL
# Each model's size and length where it differs from orbit_video's defaults (H3's).
MODEL_ORBIT_DEFAULTS: dict[str, dict] = {PROXY_MODEL: {"width": 576, "height": 768, "length": 81}}


def new_orbit(orbit: dict) -> dict:
    """A new job's orbit params: the default model unless one is given, and that model's size and
    length where the user set none."""
    model = orbit.get("model") or DEFAULT_MODEL
    return MODEL_ORBIT_DEFAULTS.get(model, {}) | {k: v for k, v in orbit.items() if v is not None} | {"model": model}


def pipeline(model: str | None) -> list[Stage]:
    """The stages of an attempt whose orbit video comes from `model` (None: H3)."""
    return [PROXY, *PIPELINE] if model == PROXY_MODEL else PIPELINE


def mode_params(model: str | None, stage: str) -> dict:
    """The params the orbit model sets for `stage`, under the user's."""
    return dict(PROXY_PARAMS.get(stage, {})) if model == PROXY_MODEL else {}

# Runs once per job, before any attempt, when the user asks for it (not part of PIPELINE).
EDIT = EditImage()
# Runs when the gate rejects COLMAP's cameras: Depth Anything 3 poses, refined by COLMAP; the
# gate then judges those (not part of PIPELINE).
FALLBACK = PoseFallback()
# Runs when the gate rejects an attempt only for a jump in the camera path; then masks,
# poses and gate run again on the filled frames (not part of PIPELINE).
GAPFILL = GapFill()

__all__ = ["BY_NAME", "DEFAULT_MODEL", "EDIT", "new_orbit", "FALLBACK", "GAPFILL", "ORBIT", "PIPELINE", "PROXY", "PROXY_MODEL", "Cancelled", "Ctx",
           "Rejected", "Stage", "StageFailed", "gapfill", "mode_params", "pipeline", "poses"]
