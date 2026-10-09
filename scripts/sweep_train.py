"""Brush flag sweep: retrain finished attempts with different train params.

Every cell reuses an attempt's frames, masks and poses (symlinks) and reruns
dataset -> export, so cells differ only in the train params. Runs are spread
over the GPUs, one at a time per GPU.

    uv run scripts/sweep_train.py data/jobs/<job>/attempts/<seed> [...] \\
        --cells base alpha05:match_alpha_weight=0.5 cap120k:max_splats=120000

A cell is NAME[:KEY=VALUE[,KEY=VALUE...]]; the keys are train params.
Writes data/sweeps/<stamp>-train/: <subject>/<cell>/ (an attempt directory),
sheets/ (the 16-direction review sheet per run), summary.txt and results.json.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from m3_report import review  # noqa: E402

SHARED = ("frames", "hero", "masks", "poses")  # what a cell reuses from the source attempt
KEPT = ("crop", "canonicalize", "export")       # stages rerun with the source attempt's params


def source_params(attempt: Path) -> list[str]:
    """-p arguments that repeat what the source attempt used after training (height, box, tau)."""
    out = []
    for stage in KEPT:
        record = attempt / ".stages" / f"{stage}.json"
        if record.exists():
            for k, v in json.loads(record.read_text())["params"].items():
                out += ["-p", f"{stage}.{k}={json.dumps(v)}"]
    return out


def run_cell(src: Path, dest: Path, cell: dict[str, object], gpu: int) -> dict[str, object]:
    dest.mkdir(parents=True, exist_ok=True)
    for name in SHARED:
        if not (dest / name).exists():
            (dest / name).symlink_to((src / name).resolve())
    cmd = ["uv", "run", "giro", "stages", str(dest), "--from", "dataset", "--gpu", str(gpu), *source_params(src)]
    for k, v in cell.items():
        cmd += ["-p", f"train.{k}={json.dumps(v)}"]
    t0 = time.monotonic()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (dest / "sweep.log").write_text(r.stdout + r.stderr)
    row: dict[str, object] = {"ok": r.returncode == 0, "seconds": round(time.monotonic() - t0, 1), "gpu": gpu}
    if r.returncode != 0:
        row["error"] = (r.stdout + r.stderr).strip().splitlines()[-1:]
        return row
    m = json.loads((dest / "metrics.json").read_text())
    row |= {
        "eval_psnr": m["train"].get("eval_psnr"), "eval_ssim": m["train"].get("eval_ssim"),
        "train_seconds": m["train"].get("train_seconds"), "n_trained": m["crop"]["n_input"],
        "n_low_opacity": m["crop"]["n_low_opacity"], "n_kept": m["crop"]["n_kept"],
        "spz_mb": m["export"].get("splat_spz_mb"),
    }
    return row


def summarize(rows: list[dict[str, object]]) -> str:
    lines = [f"{'subject':<12} {'cell':<12} {'PSNR':>6} {'SSIM':>6} {'trained':>8} {'kept':>8} {'spz MB':>7} {'train s':>8}"]
    for r in rows:
        if not r["ok"]:
            lines.append(f"{r['subject']:<12} {r['cell']:<12} FAILED {r.get('error')}")
            continue
        lines.append(f"{r['subject']:<12} {r['cell']:<12} {r['eval_psnr']:>6.2f} {r['eval_ssim']:>6.3f} "
                     f"{r['n_trained']:>8,} {r['n_kept']:>8,} {r['spz_mb']:>7} {r['train_seconds']:>8}")
    return "\n".join(lines) + "\n"


def parse_cell(text: str) -> tuple[str, dict[str, object]]:
    name, _, rest = text.partition(":")
    params: dict[str, object] = {}
    for pair in filter(None, rest.split(",")):
        k, _, raw = pair.partition("=")
        try:
            params[k] = json.loads(raw)
        except json.JSONDecodeError:
            params[k] = raw
    return name, params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("attempts", nargs="+", type=Path, help="finished attempt directories; NAME=PATH names the subject")
    ap.add_argument("--cells", nargs="+", required=True, metavar="NAME[:KEY=VALUE,...]")
    ap.add_argument("--gpus", default="1,0", help="comma-separated, one run at a time on each (default 1,0)")
    ap.add_argument("-o", "--out", type=Path, help="default: data/sweeps/<stamp>-train")
    args = ap.parse_args()
    out = args.out or ROOT / "data" / "sweeps" / f"{time.strftime('%Y%m%d-%H%M%S')}-train"
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    subjects = []
    for a in args.attempts:
        name, sep, path = str(a).partition("=")
        src = Path(path if sep else name)
        subjects.append((name if sep else src.resolve().parents[1].name, src))
    cells = [parse_cell(c) for c in args.cells]
    todo: queue.Queue[tuple[int, str, Path, str, dict[str, object]]] = queue.Queue()
    for i, (cell, (subject, src)) in enumerate((c, s) for c in cells for s in subjects):
        todo.put((i, subject, src, cell[0], cell[1]))
    rows: dict[int, dict[str, object]] = {}
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        while True:
            try:
                i, subject, src, cell, params = todo.get_nowait()
            except queue.Empty:
                return
            dest = out / subject / cell
            row = {"subject": subject, "cell": cell, "params": params} | run_cell(src, dest, params, gpu)
            if row["ok"]:
                try:
                    review(dest, out / "sheets" / f"{subject}-{cell}.jpg")
                except Exception as e:  # a sheet is a convenience; keep the numbers
                    row["sheet_error"] = str(e)
            with lock:
                rows[i] = row
                done = [rows[k] for k in sorted(rows)]
                (out / "results.json").write_text(json.dumps(done, indent=2) + "\n")
                (out / "summary.txt").write_text(summarize(done))
                print(f"[{len(done)}/{len(cells) * len(subjects)}] {subject} {cell}: "
                      f"{'PSNR %.2f, kept %s' % (row['eval_psnr'], row['n_kept']) if row['ok'] else 'FAILED'}", flush=True)

    threads = [threading.Thread(target=worker, args=(int(g),)) for g in args.gpus.split(",")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(summarize([rows[k] for k in sorted(rows)]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
