"""Start and locate giro's vendored ComfyUI instances, one per GPU.

Each instance binds to 127.0.0.1 only. GPUs are addressed by PCI order
(nvidia-smi index) so "GPU 1" means the same card everywhere on the machine.

Several giro processes (the server, CLI runs) share the instances. A stage
holds a Lease while it talks to one: a shared flock on data/run/comfy-gpuN.lock.
A process stops an instance only if it started it and no process holds a
lease (stop_if_idle), so one giro never pulls ComfyUI from under another.
"""

from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from giro import gpu as gpu_mod

ROOT = Path(__file__).resolve().parents[3]
COMFY_DIR = ROOT / "vendor" / "comfyui"
MODEL_PATHS = ROOT / "scripts" / "extra_model_paths.yaml"
RUN_DIR = ROOT / "data" / "run"

PORTS = {1: 8190, 0: 8191}


def url_for(gpu: int) -> str:
    return f"http://127.0.0.1:{PORTS[gpu]}"


def is_up(gpu: int) -> bool:
    try:
        with urllib.request.urlopen(f"{url_for(gpu)}/system_stats", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


def pick_gpu(need_mb: int) -> int:
    """Placement for a Comfy job; a running giro instance counts as reclaimable."""
    return gpu_mod.pick(need_mb, reclaimable={g for g in PORTS if is_up(g)})


def start(gpu: int, timeout: float = 120.0) -> str:
    """Ensure an instance is serving on `gpu`; returns its base URL."""
    if is_up(gpu):
        return url_for(gpu)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log = open(RUN_DIR / f"comfy-gpu{gpu}.log", "ab")
    env = gpu_mod.cuda_env(gpu)
    proc = subprocess.Popen(
        [
            str(COMFY_DIR / ".venv" / "bin" / "python"), "main.py",
            "--listen", "127.0.0.1", "--port", str(PORTS[gpu]),
            "--extra-model-paths-config", str(MODEL_PATHS),
            "--preview-method", "latent2rgb", "--preview-size", "1024",
            "--disable-auto-launch",
        ],
        cwd=COMFY_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    (RUN_DIR / f"comfy-gpu{gpu}.pid").write_text(str(proc.pid))
    (RUN_DIR / f"comfy-gpu{gpu}.owner").write_text(str(os.getpid()))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"ComfyUI on GPU {gpu} exited with {proc.returncode}; see {log.name}")
        if is_up(gpu):
            return url_for(gpu)
        time.sleep(1)
    raise TimeoutError(f"ComfyUI on GPU {gpu} did not come up within {timeout:.0f}s")


class Lease:
    """Use the instance on `gpu` (starting it if needed) until close(); blocks while
    another process is stopping it."""

    def __init__(self, gpu: int, timeout: float = 120.0):
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = open(RUN_DIR / f"comfy-gpu{gpu}.lock", "a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_SH)
            self.url = start(gpu, timeout)
        except BaseException:
            self._lock.close()
            raise

    def close(self) -> None:
        self._lock.close()  # releases the flock

    def __enter__(self) -> Lease:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def stop_if_idle(gpu: int) -> bool:
    """Stop the instance on `gpu` if this process started it and nothing holds a Lease on it."""
    owner = RUN_DIR / f"comfy-gpu{gpu}.owner"
    if not owner.exists() or owner.read_text().strip() != str(os.getpid()):
        return False
    with open(RUN_DIR / f"comfy-gpu{gpu}.lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False  # in use; it stays up (`giro comfy down` stops it by hand)
        owner.unlink(missing_ok=True)
        return stop(gpu)


def stop(gpu: int) -> bool:
    pid_file = RUN_DIR / f"comfy-gpu{gpu}.pid"
    if not pid_file.exists():
        return False
    try:
        os.killpg(int(pid_file.read_text()), signal.SIGTERM)
    except ProcessLookupError:
        pass
    pid_file.unlink()
    return True
