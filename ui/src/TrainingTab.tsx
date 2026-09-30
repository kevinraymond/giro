// Attempt detail, training: the latest Brush checkpoint in 3D, refreshed as
// training writes new ones, and the held-out PSNR and splat count by iteration.

import { useEffect, useMemo, useState } from "react";
import { fileUrl, type AttemptDetail } from "./api";
import { SplatViewer } from "./SplatViewer";
import { key, onEvent, useStore } from "./store";
import { Progress } from "./ui";

interface Point {
  step: number;
  value: number;
}

export function TrainingTab({ detail }: { detail: AttemptDetail }) {
  const { job } = detail;
  const seed = detail.attempt.seed;
  const [psnr, setPsnr] = useState<Point[]>([]);
  const [splats, setSplats] = useState<Point[]>([]);
  const [checkpoint, setCheckpoint] = useState<string | null>(null);
  const [matrix, setMatrix] = useState<number[][] | null>(null);
  const [total, setTotal] = useState(30000);
  const progress = useStore((s) => s.progress[key(job, seed)]?.train);

  useEffect(() => {
    fetch(`/api/jobs/${encodeURIComponent(job)}/attempts/${seed}/cameras`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setMatrix(d?.world_to_viewer ?? null));
    // History from Brush's log; live events append to it.
    fetch(`/api/jobs/${encodeURIComponent(job)}/attempts/${seed}/training`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
      .then((t: { psnr: Point[]; splats: Point[]; total_iters: number | null; checkpoints: string[] }) => {
        setPsnr((live) => merge(t.psnr, live));
        setSplats((live) => merge(t.splats, live));
        if (t.total_iters) setTotal(t.total_iters);
        const last = t.checkpoints[t.checkpoints.length - 1];
        if (last) setCheckpoint((live) => live ?? `attempts/${seed}/train/${last}`);
      })
      .catch(() => {}); // no training yet: the live events fill in
    return onEvent((e) => {
      if (e.type === "start" && e.job === job && e.seed === seed && e.stage === "train") {
        setPsnr([]);
        setSplats([]);
        setCheckpoint(null);
      }
      if (!("seed" in e) || e.job !== job || e.seed !== seed || e.stage !== "train") return;
      if (e.type === "metric" && typeof e.step === "number") {
        const pt = { step: e.step, value: Number(e.value) };
        if (e.name === "eval_psnr") setPsnr((xs) => [...xs, pt]);
        if (e.name === "n_splats") setSplats((xs) => [...xs, pt]);
      }
      if (e.type === "preview" && e.path.endsWith(".ply")) setCheckpoint(e.path);
    });
  }, [job, seed]);

  const shown = checkpoint;
  const iter = shown?.match(/export_(\d+)\.ply$/)?.[1];
  const training = detail.attempt.status === "running" && detail.attempt.stage === "train";

  return (
    <div className="training">
      <div className="training-top">
        <div className="ring-view">
          {shown && matrix ? (
            <SplatViewer url={fileUrl(job, shown)} matrix={matrix} floor={false} />
          ) : (
            <div className="viewer-overlay">{training ? "The first checkpoint appears after 5,000 iterations." : "No checkpoints yet."}</div>
          )}
        </div>
        <aside className="ring-side">
          <h3>Checkpoint</h3>
          <p className="small muted">
            What Brush has trained so far, before the crop, in the orbit's frame (ring axis up).
          </p>
          <dl className="facts small-facts">
            <dt>Showing</dt>
            <dd>{iter ? `iteration ${Number(iter).toLocaleString()}` : shown?.endsWith("final.ply") ? "final" : "nothing yet"}</dd>
            {training && progress && (
              <>
                <dt>Training</dt>
                <dd>
                  <Progress frac={progress.frac} />
                  <small className="muted">{progress.msg}</small>
                </dd>
              </>
            )}
          </dl>
        </aside>
      </div>
      <div className="charts">
        <section>
          <h3>Held-out PSNR (dB)</h3>
          <LineChart points={psnr} maxX={total} fmt={(v) => v.toFixed(2)} empty="Evaluated every 1,000 iterations on every 8th frame." />
        </section>
        <section>
          <h3>Splats</h3>
          <LineChart points={splats} maxX={total} fmt={(v) => Math.round(v).toLocaleString()} empty="Counted as training refines the splat." />
        </section>
      </div>
    </div>
  );
}

/** Points from the log plus live ones it did not have yet. */
function merge(history: Point[], live: Point[]): Point[] {
  const last = history.length ? history[history.length - 1].step : -1;
  return [...history, ...live.filter((p) => p.step > last)];
}

/** One series against training iteration, with a hover readout. */
export function LineChart({ points, maxX, fmt, empty }: { points: Point[]; maxX: number; fmt: (v: number) => string; empty: string }) {
  const W = 600, H = 200, L = 58, R = 12, T = 14, B = 26;
  const [hi, setHi] = useState<number | null>(null);
  const sorted = useMemo(() => [...points].sort((a, b) => a.step - b.step), [points]);
  if (!sorted.length) return <div className="plot empty"><p className="muted small">{empty}</p></div>;
  const vals = sorted.map((p) => p.value);
  let lo = Math.min(...vals), top = Math.max(...vals);
  if (top - lo < 1e-9) { lo -= 1; top += 1; }
  const pad = (top - lo) * 0.08;
  lo -= pad; top += pad;
  const xMax = Math.max(maxX, sorted[sorted.length - 1].step);
  const x = (s: number) => L + ((W - L - R) * s) / xMax;
  const y = (v: number) => T + ((H - T - B) * (top - v)) / (top - lo);
  const yt = [0, 0.5, 1].map((f) => lo + pad + f * (top - lo - 2 * pad));
  const xt = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * xMax));
  const hp = hi !== null ? sorted[hi] : null;
  return (
    <div className="plot">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
        onMouseMove={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          const fx = ((e.clientX - box.left) / box.width) * W;
          let best = 0, bd = Infinity;
          sorted.forEach((p, k) => {
            const d = Math.abs(x(p.step) - fx);
            if (d < bd) { bd = d; best = k; }
          });
          setHi(best);
        }}
        onMouseLeave={() => setHi(null)}>
        {yt.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} className="grid" />
            <text x={L - 8} y={y(v) + 4} className="tick" textAnchor="end">{fmt(v)}</text>
          </g>
        ))}
        {xt.map((s) => (
          <text key={s} x={x(s)} y={H - 7} className="tick" textAnchor={s === 0 ? "start" : s === xMax ? "end" : "middle"}>
            {s >= 1000 ? `${s / 1000}k` : s}
          </text>
        ))}
        <polyline fill="none" className="series" points={sorted.map((p) => `${x(p.step)},${y(p.value)}`).join(" ")} vectorEffect="non-scaling-stroke" />
        {hp && (
          <>
            <line x1={x(hp.step)} x2={x(hp.step)} y1={T} y2={H - B} className="crosshair" vectorEffect="non-scaling-stroke" />
            <circle cx={x(hp.step)} cy={y(hp.value)} r={4.5} className="dot" vectorEffect="non-scaling-stroke" />
          </>
        )}
      </svg>
      {hp && (
        <div className="tooltip" style={{ left: `${(x(hp.step) / W) * 100}%` }}>
          <strong>iter {hp.step.toLocaleString()}</strong>
          <span>{fmt(hp.value)}</span>
        </div>
      )}
    </div>
  );
}
