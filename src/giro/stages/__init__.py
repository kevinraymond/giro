from giro.stages.base import Cancelled, Ctx, Rejected, Stage, StageFailed
from giro.stages.canonicalize import Canonicalize
from giro.stages.crop import Crop
from giro.stages.edit import EditImage
from giro.stages.export import Export
from giro.stages import gapfill
from giro.stages.frames import Dedup, Extract
from giro.stages.gapfill import GapFill
from giro.stages.gate import Gate
from giro.stages.masks import Masks
from giro.stages.orbit import OrbitVideo
from giro.stages.poses import ColmapPoses
from giro.stages.train import Dataset, Train

# Every stage of an attempt in order; the orbit video comes first.
ORBIT = OrbitVideo()
# Masks come before poses: COLMAP matches only the subject, which turns a video that spins
# the subject in front of a still room into a usable orbit (docs/FINDINGS.md, "Masks").
PIPELINE: list[Stage] = [ORBIT, Extract(), Dedup(), Masks(), ColmapPoses(), Gate(), Dataset(), Train(),
                         Crop(), Canonicalize(), Export()]
BY_NAME: dict[str, Stage] = {s.name: s for s in PIPELINE}
# Runs once per job, before any attempt, when the user asks for it (not part of PIPELINE).
EDIT = EditImage()
# Runs when the gate rejects an attempt only for a jump in the camera path; then masks,
# poses and gate run again on the filled frames (not part of PIPELINE).
GAPFILL = GapFill()

__all__ = ["BY_NAME", "EDIT", "GAPFILL", "ORBIT", "PIPELINE", "Cancelled", "Ctx", "Rejected", "Stage",
           "StageFailed", "gapfill"]
