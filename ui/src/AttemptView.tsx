import { useEffect, useState } from "react";
import { api, fileUrl, type AttemptDetail, type StageInfo } from "./api";
import { go } from "./App";
import { ATTEMPT_STATUS, METRICS, STAGES, bytes, checkText, jobName, reasons, stageLabel } from "./format";
import { CamerasTab, CropTab, SplatViewer, TrainingTab } from "./lazy";
import type { SplatInfo } from "./SplatViewer";
import { key, putJob, useStore } from "./store";
import { ActionButton, Pill, Progress } from "./ui";

const TABS = [
  { id: "result", label: "Result" },
  { id: "orbit", label: "Orbit" },
  { id: "cameras", label: "Cameras" },
  { id: "training", label: "Training" },
  { id: "crop", label: "Crop" },
  { id: "numbers", label: "Numbers" },
] as const;

export function AttemptView({ jobId, seed, tab }: { jobId: string; seed: number; tab?: string }) {
  const job = useStore((s) => s.jobs.find((j) => j.id === jobId));
  const attempt = job?.attempts.find((a) => a.seed === seed);
  const progress = useStore((s) => s.progress[key(jobId, seed)]);
  const [detail, setDetail] = useState<AttemptDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stages, setStages] = useState<StageInfo[] | null>(null);
  useEffect(() => {
    api.stages().then(setStages).catch(() => setStages([]));
  }, []);

  // Reload the detail whenever the attempt moves on (new stage, new status).
  const version = attempt ? `${attempt.status}/${attempt.stage}/${attempt.seconds}` : "";
  useEffect(() => {
    let live = true;
    api.attempt(jobId, seed).then((d) => live && setDetail(d)).catch((e) => live && setError(String(e)));
    return () => {
      live = false;
    };
  }, [jobId, seed, version]);

  if (!job || !attempt) return <div className="page"><p className="muted">{error ?? "Loading…"}</p></div>;
  const st = ATTEMPT_STATUS[attempt.status];
  const hasExport = !!detail?.files["export/splat.spz"];
  const current = tab && TABS.some((t) => t.id === tab) ? tab : hasExport ? "result" : "orbit";
  const rank = job.ranking.indexOf(seed);
  const active = attempt.status === "running" || attempt.status === "queued";
  const stageName = STAGES.find((s) => attempt.stage.startsWith(s.name))?.name;
  const p = stageName ? progress?.[stageName] : undefined;

  return (
    <div className="page">
      <nav className="crumbs">
        <a href={`#/job/${encodeURIComponent(jobId)}`}>{jobName(jobId)}</a>
        <span>/</span>
        <span>seed {seed}</span>
      </nav>
      <header className="attempt-head">
        <h1>
          Seed {seed} {rank >= 0 && <span className="rank inline">#{rank + 1}</span>}
        </h1>
        <Pill tone={st.tone}>{st.label}</Pill>
        {active && (
          <div className="attempt-progress">
            <span>{attempt.status === "queued" ? "Queued" : stageLabel(attempt.stage)}</span>
            {p && <Progress frac={p.frac} />}
            {p && <span className="muted small">{p.msg}</span>}
          </div>
        )}
        <div className="spacer" />
        {attempt.status === "rejected" && attempt.stage === "gate" && attempt.override !== "fail" && (
          <ActionButton kind="primary" onClick={() => api.override(jobId, seed, "pass").then(putJob)}>Use anyway</ActionButton>
        )}
        {attempt.status === "passed" && (
          <ActionButton onClick={() => api.override(jobId, seed, "fail").then(putJob)}>Reject</ActionButton>
        )}
        {(attempt.status === "error" || attempt.status === "cancelled") && (
          <ActionButton kind="primary" onClick={() => api.retry(jobId, seed).then(putJob)}>Retry</ActionButton>
        )}
      </header>
      {(attempt.status === "rejected" || attempt.status === "error") && attempt.reason && (
        <ul className="reasons big">
          {reasons(attempt.reason).map((r) => <li key={r}>{r}</li>)}
        </ul>
      )}
      {attempt.filled && attempt.filled.length > 0 && (
        <p className="muted small">
          Gap filled: the camera path jumped, so the video model regenerated that arc ({attempt.filled.join("; ")}).
          The new frames are marked in the Cameras tab.
        </p>
      )}

      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={current === t.id} className={current === t.id ? "on" : ""}
            onClick={() => go("job", jobId, seed, t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {!detail ? (
        <p className="muted">Loading…</p>
      ) : current === "result" ? (
        <ResultTab detail={detail} />
      ) : current === "orbit" ? (
        <OrbitTab detail={detail} />
      ) : current === "cameras" ? (
        <CamerasTab detail={detail} />
      ) : current === "training" ? (
        <TrainingTab detail={detail} />
      ) : current === "crop" ? (
        stages ? <CropTab detail={detail} stages={stages} /> : <p className="muted">Loading…</p>
      ) : (
        <NumbersTab detail={detail} />
      )}
    </div>
  );
}

function ResultTab({ detail }: { detail: AttemptDetail }) {
  const [info, setInfo] = useState<SplatInfo | null>(null);
  const { job, attempt } = detail;
  const base = `attempts/${attempt.seed}`;
  if (!detail.files["export/splat.spz"]) {
    return <p className="muted">No splat yet: it appears here once the attempt has been trained, cropped and exported.</p>;
  }
  const exp = detail.metrics.export ?? {};
  const canon = detail.metrics.canonicalize ?? {};
  const n = Number(exp.n_gaussians ?? info?.count ?? 0);
  const budget = Number(exp.vr_budget ?? 500000);
  const v = attempt.seconds ?? 0;
  return (
    <div className="result">
      <div className="result-view">
        <SplatViewer url={fileUrl(job, `${base}/export/splat.spz`, v)} onInfo={setInfo} />
      </div>
      <aside className="result-side">
        <dl className="facts">
          <dt>Size</dt>
          <dd>
            {canon.height_m !== undefined
              ? `${Number(canon.width_m).toFixed(2)} × ${Number(canon.height_m).toFixed(2)} × ${Number(canon.depth_m).toFixed(2)} m`
              : info && `${info.size.map((x) => x.toFixed(2)).join(" × ")} m`}
            <small className="muted">width × height × depth</small>
          </dd>
          <dt>Splats</dt>
          <dd>
            {n.toLocaleString()}
            <div className="budget">
              <Progress frac={n / budget} tone={n <= budget ? "good" : "warn"} />
              <small className="muted">
                {n <= budget ? `${Math.round((100 * n) / budget)}% of the Quest budget (${budget / 1000}K)` : `over the Quest budget of ${budget / 1000}K; a decimated VR file is included`}
              </small>
            </div>
          </dd>
          {attempt.eval_psnr !== null && (
            <>
              <dt>Held-out PSNR</dt>
              <dd>{attempt.eval_psnr.toFixed(2)} dB</dd>
            </>
          )}
        </dl>
        <div className="downloads col">
          {(["spz", "sog", "ply", "vr.sog"] as const).map((ext) => {
            const f = `export/splat.${ext}`;
            return detail.files[f] ? (
              <a key={ext} className="btn" href={fileUrl(job, `${base}/${f}`, v)} download={`${jobName(job)}-${attempt.seed}.${ext}`}>
                <span>{ext.toUpperCase()}</span>
                <span className="muted">{bytes(detail.files[f])}</span>
              </a>
            ) : null;
          })}
        </div>
        <a className="btn primary" href={`#/vr/${encodeURIComponent(job)}/${attempt.seed}`}>View in VR</a>
        <p className="muted small">
          Opens a page for the Quest browser: there, press Enter VR. SPZ and SOG load in Spark and SuperSplat;
          PLY is the full-quality master.
        </p>
      </aside>
    </div>
  );
}

function OrbitTab({ detail }: { detail: AttemptDetail }) {
  const { job, attempt } = detail;
  const base = `attempts/${attempt.seed}`;
  return (
    <div className="orbit">
      <div className="orbit-media">
        {detail.files["video.mp4"] ? (
          <video src={fileUrl(job, `${base}/video.mp4`)} controls muted loop autoPlay playsInline />
        ) : detail.previews.length ? (
          <img src={fileUrl(job, `${base}/previews/${detail.previews[detail.previews.length - 1]}`)} alt="latest preview" />
        ) : (
          <p className="muted">No video yet.</p>
        )}
      </div>
      <div className="orbit-side">
        <h3>Gate</h3>
        {detail.gate ? (
          <table className="checks-table">
            <tbody>
              {detail.gate.checks.map((c) => (
                <tr key={c.metric} className={c.pass ? "" : "fail"} title={METRICS[c.metric]?.help}>
                  <td>{c.pass ? "✓" : "✗"}</td>
                  <td>{METRICS[c.metric]?.help ?? c.metric}</td>
                  <td className="num">{checkText(c)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">Not checked yet.</p>
        )}
        {detail.files["crop/preview_hero.jpg"] && (
          <>
            <h3>Crop from the hero view</h3>
            <p className="muted small">The hero image, the trained splat, and what the crop kept.</p>
            <img className="wide" src={fileUrl(job, `${base}/crop/preview_hero.jpg`, attempt.seconds ?? 0)} alt="" />
          </>
        )}
      </div>
      {detail.files["canonical/turnaround.jpg"] && (
        <div className="orbit-full">
          <h3>Turnaround</h3>
          <img className="wide" src={fileUrl(job, `${base}/canonical/turnaround.jpg`, attempt.seconds ?? 0)} alt="" />
        </div>
      )}
    </div>
  );
}

function NumbersTab({ detail }: { detail: AttemptDetail }) {
  const stages = STAGES.filter((s) => detail.metrics[s.name]);
  const overrides = Object.entries(detail.attempt.params);
  return (
    <div className="numbers">
      {overrides.length > 0 && (
        <section>
          <h3>Changed for this seed</h3>
          <pre className="code">{JSON.stringify(detail.attempt.params, null, 2)}</pre>
        </section>
      )}
      {stages.map((s) => (
        <section key={s.name}>
          <h3>{s.label}</h3>
          <table className="kv">
            <tbody>
              {Object.entries(detail.metrics[s.name])
                .filter(([, v]) => typeof v !== "object" || v === null)
                .map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td>
                    <td className="num">{String(v)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </section>
      ))}
    </div>
  );
}
