"""orbit_video (video in): an orbit video from anywhere, in place of a generated one.

The user brings the orbit: a video (MiniMax H3 or another video model run elsewhere, or a
phone walked around the subject) or a directory of frames (PNG saved from ComfyUI: no second
compression). It becomes the attempt's frames_raw and video.mp4, and the rest of the pipeline
runs as it does for an H3 orbit: COLMAP maps the cameras, the gate checks the ring, Brush
trains with the hero (the first frame unless an image is given) as the extra view. There is
no proxy, so no path cameras, and no video model to fill gaps with.

Frames larger than `max_side` are scaled down, and a long video is sampled evenly down to
`max_frames` (the dedup stage then picks its target from those): a 4K phone clip would
otherwise fill a disk with PNG and give COLMAP thousands of images.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from giro.stages.base import Ctx, Stage, StageFailed

MODEL = "video"  # the orbit "model" of an attempt built on the user's own video
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
MIN_FRAMES = 8


@dataclass(frozen=True)
class Probe:
    width: int
    height: int
    frames: int
    fps: float


def frame_files(directory: Path) -> list[Path]:
    """The images of a frames directory, in name order."""
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def probe(source: Path) -> Probe:
    """Size (as decoded: a phone's rotation flag applied), frame count and rate of a video, or of
    a directory of frames (its first image's size; 16 fps, the proxy orbit's). ValueError if
    `source` is neither. Any container and codec ffmpeg reads: MP4, MOV (HEVC included), WebM, MKV."""
    source = Path(source)
    if source.is_dir():
        files = frame_files(source)
        if not files:
            raise ValueError(f"{source} holds no PNG, JPEG or WebP frames")
        with Image.open(files[0]) as im:
            return Probe(im.width, im.height, len(files), 16.0)
    if not source.is_file():
        raise ValueError(f"{source} is not a video file or a directory of frames")
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,nb_frames,r_frame_rate,duration:stream_tags=rotate:stream_side_data=rotation",
           "-of", "json", str(source)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise ValueError(f"{source.name} is not a video ffmpeg can read") from e
    streams = json.loads(out).get("streams") or []
    if not streams:
        raise ValueError(f"{source.name} has no video stream")
    s = streams[0]
    num, _, den = str(s.get("r_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    frames = int(s["nb_frames"]) if str(s.get("nb_frames", "N/A")).isdigit() else 0
    if not frames and s.get("duration") and fps:
        frames = round(float(s["duration"]) * fps)
    width, height = int(s["width"]), int(s["height"])
    # A phone held upright records sideways and flags the rotation (a display matrix, or an older
    # rotate tag); ffmpeg decodes rotated, so the frames come out the other way round.
    rotation = next((sd.get("rotation") for sd in s.get("side_data_list") or [] if "rotation" in sd), None)
    if rotation is None:
        rotation = (s.get("tags") or {}).get("rotate")
    try:
        if rotation is not None and round(float(rotation) / 90) % 2:
            width, height = height, width
    except ValueError:
        pass
    return Probe(width, height, frames, fps)


def scaled(width: int, height: int, max_side: int) -> tuple[int, int]:
    """The frame size after fitting the longer side into `max_side` (even, for the mp4)."""
    f = min(1.0, max_side / max(width, height))
    return (max(2, round(width * f / 2) * 2), max(2, round(height * f / 2) * 2))


def orbit_params(source: Path, max_side: int | None = None) -> dict[str, Any]:
    """The orbit params of an attempt or job built on `source`: the video-in model, the source,
    and the frames' size (which the hero is fitted to)."""
    p = probe(source)
    width, height = scaled(p.width, p.height, max_side or VideoIn.defaults["max_side"])
    return {"model": MODEL, "source": str(Path(source).resolve()), "width": width, "height": height}


def first_frame(source: Path, out: Path) -> Path:
    """The source's first frame as PNG at `out` (the hero when the user gives no image: an H3
    orbit, and most generated ones, start on the image they were made from)."""
    source = Path(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        with Image.open(frame_files(source)[0]) as im:
            im.convert("RGB").save(out, format="PNG")
        return out
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-frames:v", "1", str(out)], check=True)
    if not out.exists():
        raise ValueError(f"ffmpeg decoded no frame from {source.name}")
    return out


class VideoIn(Stage):
    """The attempt's orbit from the user's video or frames: frames_raw/ (PNG, sampled and
    scaled), video.mp4 (re-encoded from them, for the UI and the report) and orbit.json."""

    name = "orbit_video"  # the same record as a generated orbit's: the attempt's model is read from it
    defaults = {"source": None, "max_frames": 400, "max_side": 1536, "fps": 16}
    # What a job's orbit params may carry over from the generated orbits' form; ignored here.
    extra_params = ("model", "width", "height", "length", "steps", "prompt", "keep_loaded", "seed")
    outputs = ("frames_raw", "video.mp4", "orbit.json")

    def _key(self, attempt: Path, params: dict[str, Any]) -> str:
        key = super()._key(attempt, params)
        source = self._source(attempt, params)
        if source is not None and source.exists():
            files = frame_files(source) if source.is_dir() else [source]
            stamp = ";".join(f"{f.name}:{f.stat().st_size}:{f.stat().st_mtime_ns}" for f in files)
            key = f"{key}-{hashlib.sha256(stamp.encode()).hexdigest()[:8]}"
        return key

    @staticmethod
    def _source(attempt: Path, params: dict[str, Any]) -> Path | None:
        if not params.get("source"):
            return None
        source = Path(params["source"])
        return source if source.is_absolute() else (attempt / source).resolve()

    def run(self, attempt: Path, params: dict[str, Any], ctx: Ctx) -> None:
        source = self._source(attempt, params)
        if source is None:
            raise StageFailed("video in: no source (orbit_video.source is the video or frames directory)")
        try:
            info = probe(source)
        except ValueError as e:
            raise StageFailed(f"video in: {e}") from None
        if info.frames and info.frames < MIN_FRAMES:
            raise StageFailed(f"video in: only {info.frames} frames in {source.name} (need {MIN_FRAMES})")
        width, height = scaled(info.width, info.height, int(params["max_side"]))
        out = attempt / "frames_raw"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        (attempt / "video.mp4").unlink(missing_ok=True)
        ctx.log(f"{source.name}: {info.width}x{info.height}, {info.frames or '?'} frames at {info.fps:.3g} fps"
                + (f"; scaled to {width}x{height}" if (width, height) != (info.width, info.height) else ""))
        ctx.progress(0.05, "reading the frames")
        if source.is_dir():
            n = self._from_frames(source, out, (width, height), int(params["max_frames"]), ctx)
        else:
            n = self._from_video(source, out, (width, height), info, int(params["max_frames"]))
        if n < MIN_FRAMES:
            raise StageFailed(f"video in: only {n} frames came out of {source.name} (need {MIN_FRAMES})")
        ctx.check_cancelled()
        ctx.progress(0.8, f"{n} frames; encoding the preview video")
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(params["fps"]), "-i", str(out / "%05d.png"),
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(attempt / "video.mp4")],
            check=True,
        )
        (attempt / "orbit.json").write_text(json.dumps({
            "params": {k: v for k, v in params.items() if k != "source"} | {"source": str(source)},
            "source_frames": info.frames, "source_fps": info.fps, "frames": n, "width": width, "height": height,
        }, indent=2) + "\n")
        ctx.metric("n_raw", n)
        ctx.metric("source_frames", info.frames)
        ctx.metric("width", width)
        ctx.metric("height", height)
        ctx.preview(attempt / "video.mp4")
        ctx.progress(1.0, f"{n} frames from {source.name}")

    @staticmethod
    def _from_video(source: Path, out: Path, size: tuple[int, int], info: Probe, max_frames: int) -> int:
        # Every `step`th frame, scaled: ffmpeg's select on the frame index keeps the sampling even.
        step = max(1, -(-info.frames // max_frames)) if info.frames else 1
        filters = [f"select=not(mod(n\\,{step}))", f"scale={size[0]}:{size[1]}:flags=lanczos"]
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(source), "-vf", ",".join(filters), "-fps_mode", "vfr",
             str(out / "%05d.png")],
            check=True,
        )
        return len(list(out.glob("*.png")))

    @staticmethod
    def _from_frames(source: Path, out: Path, size: tuple[int, int], max_frames: int, ctx: Ctx) -> int:
        files = frame_files(source)
        step = max(1, -(-len(files) // max_frames))
        kept = files[::step]
        for i, f in enumerate(kept, start=1):
            ctx.check_cancelled()
            with Image.open(f) as im:
                im = im.convert("RGB")
                if (im.width, im.height) != size:
                    im = im.resize(size, Image.LANCZOS)
                im.save(out / f"{i:05d}.png", format="PNG")
            if i % 20 == 0:
                ctx.progress(0.05 + 0.7 * i / len(kept), f"{i} of {len(kept)} frames")
        return len(kept)
