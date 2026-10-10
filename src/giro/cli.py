"""giro command line: headless access to each pipeline stage and to jobs."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from giro import gpu, report, stages, workflows
from giro import path as campath
from giro.comfy import server
from giro.job import Job, JobSpec, Runner, describe
from giro.stages.orbit import ORBIT_VRAM_MB, prepare_hero

JOBS_DIR = Path(__file__).resolve().parents[2] / "data" / "jobs"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _cli_ctx(gpu_index: int | None = None) -> stages.Ctx:
    last: dict[str, float] = {}

    def progress(stage: str, frac: float, msg: str) -> None:
        # Throttle to one line per 5% per stage.
        if frac >= 1.0 or frac - last.get(stage, -1.0) >= 0.05:
            last[stage] = frac
            _log(f"{stage}: {frac:4.0%} {msg}")

    return stages.Ctx(on_progress=progress, on_log=lambda stage, msg: _log(f"{stage}: {msg}"), gpu=gpu_index)


def _parse_params(pairs: list[str]) -> dict[str, dict[str, object]]:
    """'stage.key=value' pairs -> {stage: {key: value}}, values parsed as JSON when possible."""
    out: dict[str, dict[str, object]] = {}
    for pair in pairs:
        dotted, _, raw = pair.partition("=")
        stage, _, key = dotted.partition(".")
        try:
            value: object = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out.setdefault(stage, {})[key] = value
    return out


def _stage_params(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """-p STAGE.KEY=VALUE pairs, plus the shortcuts for what a user sets per subject."""
    params = _parse_params(args.param)
    if getattr(args, "height_m", None) is not None:
        params.setdefault("canonicalize", {}).setdefault("height_m", args.height_m)
    if getattr(args, "subject", None):
        params.setdefault("masks", {}).setdefault("subject_prompt", args.subject)
    if getattr(args, "kind", None):
        params.setdefault("proxy", {}).setdefault("model", stages.PROXY_BY_KIND[args.kind])
    return params


def _orbit_params(args: argparse.Namespace) -> dict[str, object]:
    """The orbit params a new attempt or job gets: the default model and its size and length unless
    given (stages.new_orbit), then orbit_video's own defaults."""
    given: dict[str, object] = {"prompt": args.prompt, "width": args.width, "height": args.height,
                                "length": args.length, "steps": args.steps, "keep_loaded": args.keep_loaded}
    for key in ("model", "path", "turns", "pitch_end"):
        given[key] = getattr(args, key, None)
    orbit = stages.new_orbit(given)
    return {k: orbit.get(k, stages.ORBIT.defaults.get(k)) for k in ("prompt", "width", "height", "length", "steps",
                                                                     "keep_loaded")} | orbit


def _attempt_model(attempt: Path) -> str | None:
    """The orbit model an attempt's video came from (None: H3, or no video yet)."""
    record = attempt / ".stages" / "orbit_video.json"
    return json.loads(record.read_text())["params"].get("model") if record.exists() else None


def _prepare_attempt(args: argparse.Namespace) -> dict[str, object]:
    """Hero and orbit params for a single attempt directory (`orbit` and `run`)."""
    orbit = _orbit_params(args)
    width, height = int(orbit["width"]), int(orbit["height"])  # type: ignore[call-overload]
    cropped = prepare_hero(args.image, args.out, width, height)
    if cropped > 0.01:
        _log(f"warning: hero cropped to {width}:{height}, {cropped:.0%} of the image removed")
    seed = args.seed
    if seed is None and (args.out / "orbit.json").exists():
        seed = json.loads((args.out / "orbit.json").read_text())["params"]["seed"]  # rerun the same video
    return orbit | {"seed": seed if seed is not None else random.randrange(2**32)}


def orbit(args: argparse.Namespace) -> int:
    params = _prepare_attempt(args)
    try:
        stages.ORBIT.execute(args.out, params, _cli_ctx(args.gpu))
    except stages.StageFailed as e:
        _log(f"orbit_video: FAILED: {e}")
        return 1
    _log(f"done: {args.out / 'video.mp4'}")
    return 0


def run_stages(args: argparse.Namespace, first_params: dict[str, object] | None = None) -> int:
    attempt: Path = args.attempt
    model = (first_params or {}).get("model") or _attempt_model(attempt)
    pipeline = stages.pipeline(model)
    names = [s.name for s in pipeline]
    first = names.index(args.start) if args.start else names.index("extract")
    last = names.index(args.stop) + 1 if args.stop else len(names)
    params = _stage_params(args)
    ctx = _cli_ctx(getattr(args, "gpu", None))
    for stage in pipeline[first:last]:
        t0 = time.monotonic()
        stage_params = (((first_params or {}) if stage is stages.ORBIT else stages.mode_params(model, stage.name))
                        | (params.get(stage.name) or {}))
        try:
            ran = stage.execute(attempt, stage_params, ctx, force=args.force)
        except stages.StageFailed as e:
            _log(f"{stage.name}: FAILED: {e}")
            return 1
        if ran:
            summary = ", ".join(f"{k}={v}" for k, v in ctx.metrics.items())
            _log(f"{stage.name}: done in {time.monotonic() - t0:.1f}s  {summary}")
    return 0


def run_all(args: argparse.Namespace) -> int:
    """Everything for one seed, from the image to train/final.ply."""
    t0 = time.monotonic()
    orbit_params = _prepare_attempt(args)
    args.attempt, args.stop, args.force = args.out, None, False
    args.start = stages.pipeline(orbit_params.get("model"))[0].name
    code = run_stages(args, orbit_params)
    _log(f"run: finished in {(time.monotonic() - t0) / 60:.1f} min" if code == 0 else "run: stopped")
    return code


def _int_list(text: str) -> list[int]:
    return [int(g) for g in text.split(",") if g.strip()]


def job_cmd(args: argparse.Namespace) -> int:
    if args.action == "export":
        job = Job.load(args.job)
        out = report.build(job, args.out or Path(f"{job.path.name}.zip"))
        _log(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MB): index.html, images and splats of {len(job.ranking)} passing attempts")
        return 0
    if args.action == "show":
        print(describe(Job.load(args.job)))
        return 0
    if args.action == "start":
        spec = JobSpec(
            image=str(args.image.resolve()), want=args.want, max_attempts=args.max_attempts,
            seeds=args.seeds, orbit=_orbit_params(args), params=_stage_params(args),
            video_gpus=args.video_gpus, post_gpus=args.post_gpus,
        )
        job = Job.create(args.root, spec, args.name)
        _log(f"job {job.path}")
    else:
        job = Job.load(args.job)
        for stage, overrides in _stage_params(args).items():
            job.spec.params[stage] = job.spec.params.get(stage, {}) | overrides
        for attempt in job.attempts:
            if attempt.seed in args.retry:
                # Finished stages are still skipped; only what failed or changed runs again.
                attempt.status, attempt.reason = "queued", ""
        _log(f"resuming {job.path}")
    log_file = open(job.path / "job.log", "a")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    t0 = time.monotonic()
    job = asyncio.run(Runner(job, log).run())
    log(f"job {job.status} in {(time.monotonic() - t0) / 60:.1f} min")
    print(describe(job))
    return 0 if job.status == "done" else 1


def serve_cmd(args: argparse.Namespace) -> int:
    import uvicorn

    from giro.api import create_app

    app = create_app(args.root, args.db, args.gpus)
    configs = [uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")]
    _log(f"giro serve: http://{args.host}:{args.port} (jobs in {args.root})")
    if args.https_port:
        # WebXR needs a secure context: https (or localhost, e.g. over adb reverse).
        cert, key = _self_signed_cert()
        configs.append(uvicorn.Config(app, host=args.host, port=args.https_port, log_level="warning",
                                      ssl_certfile=str(cert), ssl_keyfile=str(key), lifespan="off"))
        _log(f"giro serve: https://{args.host}:{args.https_port} (self-signed: accept the warning once per device)")

    async def serve_all() -> None:
        servers = [uvicorn.Server(c) for c in configs]
        await asyncio.gather(*(s.serve() for s in servers))

    asyncio.run(serve_all())
    return 0


def _self_signed_cert() -> tuple[Path, Path]:
    """A long-lived self-signed certificate for this host, made once in data/run/tls."""
    import socket
    import subprocess

    tls = JOBS_DIR.parent / "run" / "tls"
    cert, key = tls / "cert.pem", tls / "key.pem"
    if not cert.exists():
        tls.mkdir(parents=True, exist_ok=True)
        host = socket.gethostname()
        ips = sorted({ip for ip in socket.gethostbyname_ex(host)[2]} | {_lan_ip()} - {""})
        san = ",".join([f"DNS:{host}", "DNS:localhost", *(f"IP:{ip}" for ip in ips), "IP:127.0.0.1"])
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "3650",
                        "-subj", f"/CN={host}", "-addext", f"subjectAltName={san}",
                        "-keyout", str(key), "-out", str(cert)], check=True, capture_output=True)
    return cert, key


def _lan_ip() -> str:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
            return s.getsockname()[0]
        except OSError:
            return ""


def comfy_cmd(args: argparse.Namespace) -> int:
    if args.action == "up":
        device = args.gpu if args.gpu is not None else server.pick_gpu(ORBIT_VRAM_MB)
        print(f"GPU {device}: {server.start(device)}")
    elif args.action == "down":
        for device in [args.gpu] if args.gpu is not None else list(server.PORTS):
            print(f"GPU {device}: {'stopped' if server.stop(device) else 'not started by giro'}")
    else:
        for s in gpu.states():
            up = "up" if s.index in server.PORTS and server.is_up(s.index) else "down"
            print(f"GPU {s.index}: {s.free_mb}/{s.total_mb} MB free, giro comfy {up}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="giro")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def video_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--width", type=int, help="default 576 (proxy orbit) or 768 (h3)")
        p.add_argument("--height", type=int, help="default 768 (proxy orbit) or 1024 (h3)")
        p.add_argument("--length", type=int, help="frames: default 81 at 16 fps (proxy orbit, 4k+1) or 158 at 24 fps "
                                                  "(h3, 17k+5); see docs/FINDINGS.md")
        p.add_argument("--steps", type=int, default=20)
        p.add_argument("--prompt", default=workflows.ORBIT_PROMPT)
        p.add_argument("--keep-loaded", action="store_true", help="leave models in VRAM for the next run")
        p.add_argument("--model", choices=sorted(workflows.ORBIT_MODELS),
                       help=f"video model (default {stages.DEFAULT_MODEL}, the proxy orbit); h3 is MiniMax H3 with the "
                            "360 orbit LoRA, whose license excludes the US, EU, UK and Korea")
        p.add_argument("--path", choices=campath.PRESETS, help="proxy orbit: camera path (default spiral)")
        p.add_argument("--turns", type=float, help="proxy orbit: turns around the subject (default 2)")
        p.add_argument("--pitch-end", type=float, help="proxy orbit: final elevation in degrees (default 45)")

    def subject_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--height-m", type=float, help="subject height in meters for the exported splat (default 1.7)")
        p.add_argument("--subject", help="what SAM should segment (default 'main subject, held object:2')")
        p.add_argument("--kind", choices=sorted(stages.PROXY_BY_KIND),
                       help=f"proxy orbit: the subject's kind, which picks the proxy model (default {stages.DEFAULT_KIND}: "
                            + ", ".join(f"{k} {m}" for k, m in stages.PROXY_BY_KIND.items()) + ")")

    def attempt_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("image", type=Path)
        p.add_argument("-o", "--out", type=Path, required=True, help="attempt directory")
        p.add_argument("--seed", type=int)
        p.add_argument("--gpu", type=int, choices=sorted(server.PORTS))
        video_args(p)

    p = sub.add_parser("orbit", help="generate a 360-degree orbit video from one image")
    attempt_args(p)
    p.set_defaults(func=orbit)

    names = [st.name for st in stages.pipeline(stages.PROXY_MODEL)]
    e = sub.add_parser("run", help="one seed from image to trained splat, in one attempt directory")
    attempt_args(e)
    e.add_argument("-p", "--param", action="append", default=[], metavar="STAGE.KEY=VALUE")
    subject_args(e)
    e.set_defaults(func=run_all)

    r = sub.add_parser("stages", help="run the stages after the video on an attempt directory")
    r.add_argument("attempt", type=Path)
    r.add_argument("--from", dest="start", choices=names, help="default: extract")
    r.add_argument("--to", dest="stop", choices=names)
    r.add_argument("-p", "--param", action="append", default=[], metavar="STAGE.KEY=VALUE")
    r.add_argument("--force", action="store_true", help="rerun even when inputs and params are unchanged")
    r.add_argument("--gpu", type=int, choices=sorted(server.PORTS), help="default: the stage picks one")
    subject_args(r)
    r.set_defaults(func=run_stages)

    j = sub.add_parser("job", help="several seeds per image: gate, reroll, rank, both GPUs")
    jsub = j.add_subparsers(dest="action", required=True)
    js = jsub.add_parser("start", help="start a job for an image")
    js.add_argument("image", type=Path)
    js.add_argument("-n", "--want", type=int, default=3, help="passing attempts to aim for")
    js.add_argument("--max-attempts", type=int, default=6, help="seeds to try at most, rerolls included")
    js.add_argument("--seeds", type=_int_list, default=[], help="comma-separated seeds to try first")
    js.add_argument("--video-gpus", type=_int_list, default=[1], help="GPUs for video generation (default 1)")
    js.add_argument("--post-gpus", type=_int_list, default=[0, 1], help="GPUs for poses and training, preferred first")
    js.add_argument("--name", help="job name (default: the image name)")
    js.add_argument("--root", type=Path, default=JOBS_DIR)
    js.add_argument("-p", "--param", action="append", default=[], metavar="STAGE.KEY=VALUE")
    subject_args(js)
    video_args(js)
    for action, text in (("resume", "continue an interrupted job"), ("show", "print a job's attempts")):
        jr = jsub.add_parser(action, help=text)
        jr.add_argument("job", type=Path, help="job directory")
        if action == "resume":
            jr.add_argument("--retry", type=_int_list, default=[], metavar="SEEDS",
                            help="also rerun these finished attempts (e.g. after a crash or new params)")
            jr.add_argument("-p", "--param", action="append", default=[], metavar="STAGE.KEY=VALUE",
                            help="change a stage parameter for the rest of the job")
            subject_args(jr)
    je = jsub.add_parser("export", help="a zip with an offline HTML report and the splats (PLY, SOG, SPZ)")
    je.add_argument("job", type=Path, help="job directory")
    je.add_argument("-o", "--out", type=Path, help="zip to write (default: <job name>.zip here)")
    j.set_defaults(func=job_cmd)

    sv = sub.add_parser("serve", help="the orchestrator: API, event stream and UI, on the LAN")
    sv.add_argument("--host", default="0.0.0.0", help="default: every interface (LAN; ComfyUI stays on localhost)")
    sv.add_argument("--port", type=int, default=8470)
    sv.add_argument("--https-port", type=int, help="also serve https here (self-signed), for WebXR over the LAN")
    sv.add_argument("--root", type=Path, default=JOBS_DIR)
    sv.add_argument("--db", type=Path, default=JOBS_DIR.parent / "giro.sqlite")
    sv.add_argument("--gpus", type=_int_list, default=[0, 1], help="GPUs the server may use (default 0,1)")
    sv.set_defaults(func=serve_cmd)

    c = sub.add_parser("comfy", help="manage giro's ComfyUI instances")
    c.add_argument("action", choices=["up", "down", "status"])
    c.add_argument("--gpu", type=int, choices=sorted(server.PORTS))
    c.set_defaults(func=comfy_cmd)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
