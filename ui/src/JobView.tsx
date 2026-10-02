import { useState } from "react";
import { api, fileUrl, thumbUrl, type Attempt, type Job } from "./api";
import { go } from "./App";
import {
  ATTEMPT_STATUS, JOB_STATUS, STAGES, ago, headline, jobName, reasons, stageIndex, stageKey, stageLabel,
} from "./format";
import { DraftView } from "./DraftView";
import { SplatViewer } from "./lazy";
import { key, putJob, useStore } from "./store";
import { ActionButton, Pill, Progress } from "./ui";

const ORDER: Record<string, number> = { running: 0, queued: 1, passed: 2, rejected: 3, error: 4, cancelled: 5, discarded: 6 };

export function JobView({ jobId }: { jobId: string }) {
  const job = useStore((s) => s.jobs.find((j) => j.id === jobId));
  const connected = useStore((s) => s.connected);
  if (!job) return <div className="page"><p className="muted">{connected ? "No such job." : "Connecting…"}</p></div>;
  if (job.status === "draft") return <DraftView job={job} />;

  const st = JOB_STATUS[job.status] ?? { label: job.status, tone: "muted" as const };
  const passed = job.ranking.length;
  const tried = job.attempts.length;
  const rank = (seed: number) => job.ranking.indexOf(seed);
  const attempts = [...job.attempts].sort((a, b) => {
    const ra = rank(a.seed), rb = rank(b.seed);
    if (ra >= 0 || rb >= 0) return (ra < 0 ? 99 : ra) - (rb < 0 ? 99 : rb);
    return (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9);
  });
  const shown = attempts.filter((a) => a.status !== "discarded");
  const discarded = attempts.length - shown.length;
  const running = job.status === "running";
  const height = job.spec.params.canonicalize?.height_m;

  return (
    <div className="page">
      <header className="job-head">
        <img className="job-thumb" src={thumbUrl(job.id)} alt="" />
        <div className="job-title">
          <h1>{jobName(job.id)}</h1>
          <div className="job-sub">
            <Pill tone={st.tone}>{st.label}</Pill>
            <span>
              {passed} good orbit{passed === 1 ? "" : "s"}
              {passed < job.spec.want && <span className="muted"> of {job.spec.want} wanted</span>}
            </span>
            <span className="muted">
              · {tried} of at most {job.spec.max_attempts} seeds tried
            </span>
            {height !== undefined && <span className="muted">· {String(height)} m tall</span>}
            {job.spec.use_edit && <span className="muted" title={job.edit.prompt}>· edited image</span>}
            <span className="muted">· {ago(job.created)}</span>
          </div>
        </div>
        <div className="job-actions">
          {running ? (
            <ActionButton onClick={() => api.cancel(job.id).then(putJob)} title="Stop every running stage; Resume continues later">
              Stop
            </ActionButton>
          ) : (
            job.status !== "done" && (
              <ActionButton kind="primary" onClick={() => api.resume(job.id).then(putJob)} title="Finished stages are skipped">
                Resume
              </ActionButton>
            )
          )}
          <ActionButton onClick={() => api.addSeeds(job.id, 1).then(putJob)} title="Aim for one more good orbit">
            One more orbit
          </ActionButton>
          {job.ranking.length > 0 && (
            <a className="btn" href={`/api/jobs/${encodeURIComponent(job.id)}/export.zip`} download
              title="A zip with an offline report (result, orbit, cameras, training, numbers) and the splats as PLY, SOG and SPZ">
              Export job
            </a>
          )}
          {!running && (
            <ActionButton
              kind="ghost"
              onClick={async () => {
                if (!confirm(`Delete ${jobName(job.id)} and everything it made?`)) return;
                await api.deleteJob(job.id);
                go();
              }}
            >
              Delete
            </ActionButton>
          )}
        </div>
      </header>

      {job.best !== null && <BestResult job={job} seed={job.best} />}

      <h2 className="section">Orbits</h2>
      <div className="cards">
        {shown.map((a) => (
          <AttemptCard key={a.seed} job={job} attempt={a} rank={rank(a.seed)} />
        ))}
      </div>
      {discarded > 0 && <p className="muted small">{discarded} discarded seed{discarded > 1 ? "s" : ""} hidden.</p>}
    </div>
  );
}

function BestResult({ job, seed }: { job: Job; seed: number }) {
  const a = job.attempts.find((x) => x.seed === seed);
  // Version the URL by when the attempt last finished, so a re-export reloads.
  const version = a?.seconds ?? 0;
  const base = `attempts/${seed}/export`;
  return (
    <section className="best">
      <div className="best-view">
        <SplatViewer url={fileUrl(job.id, `${base}/splat.spz`, version)} autoRotate />
      </div>
      <div className="best-side">
        <div className="eyebrow">Best result</div>
        <h2>Seed {seed}</h2>
        <div className="badges">
          {a && headline(a).map((b) => (
            <span key={b.label} className="badge"><span>{b.label}</span>{b.value}</span>
          ))}
        </div>
        <div className="downloads">
          <a className="btn" href={fileUrl(job.id, `${base}/splat.spz`, version)} download={`${jobName(job.id)}.spz`}>SPZ</a>
          <a className="btn" href={fileUrl(job.id, `${base}/splat.sog`, version)} download={`${jobName(job.id)}.sog`}>SOG</a>
          <a className="btn" href={fileUrl(job.id, `${base}/splat.ply`, version)} download={`${jobName(job.id)}.ply`}>PLY</a>
        </div>
        <a className="btn primary" href={`#/job/${encodeURIComponent(job.id)}/${seed}`}>Open</a>
      </div>
    </section>
  );
}

function AttemptCard({ job, attempt: a, rank }: { job: Job; attempt: Attempt; rank: number }) {
  const progress = useStore((s) => s.progress[key(job.id, a.seed)]);
  const preview = useStore((s) => s.previews[key(job.id, a.seed)]);
  const [videoFailed, setVideoFailed] = useState(false);
  const st = ATTEMPT_STATUS[a.status] ?? { label: a.status, tone: "muted" as const };
  const current = stageIndex(a.stage);
  const stageName = STAGES[current]?.name;
  const live = stageKey(a.stage);
  const p = live ? progress?.[live] : undefined;
  const active = a.status === "running" || a.status === "queued";
  const failedAt = a.status === "rejected" || a.status === "error" ? current : -1;
  // The video exists once the orbit stage is behind the attempt.
  const hasVideo = !videoFailed && (a.status === "passed" || current > 0 || (current < 0 && a.gate.n_views !== undefined));
  const livePreview = active && stageName === "orbit_video" && preview && /\.(jpe?g|png|webp)$/.test(preview.path);
  const href = `#/job/${encodeURIComponent(job.id)}/${a.seed}`;

  return (
    <article className={`card ${st.tone} ${rank === 0 ? "top" : ""}`}>
      <a className="card-media" href={href}>
        {livePreview ? (
          <img src={fileUrl(job.id, preview.path)} alt="" />
        ) : hasVideo ? (
          <video src={fileUrl(job.id, `attempts/${a.seed}/video.mp4`)} muted loop autoPlay playsInline onError={() => setVideoFailed(true)} />
        ) : (
          <img src={thumbUrl(job.id)} alt="" className="dim" />
        )}
        {rank >= 0 && <span className="rank">#{rank + 1}</span>}
        {a.override === "pass" ? (
          <span className="flag" title="Passed by you over the gate's verdict">passed by hand</span>
        ) : a.filled && a.filled.length > 0 && (
          <span className="flag" title={`Regenerated where the camera path jumped:\n${a.filled.join("\n")}`}>gap filled</span>
        )}
      </a>
      <div className="card-body">
        <div className="card-head">
          <span className="seed">seed {a.seed}</span>
          <Pill tone={st.tone}>{st.label}</Pill>
        </div>
        <div className="pipeline" aria-label="stages">
          {STAGES.map((s, i) => {
            const done = a.status === "passed" || i < current;
            const cls = i === failedAt ? "fail" : done ? "done" : i === current && active ? "now" : "";
            return (
              <span key={s.name} className={`seg ${cls}`} title={s.label}>
                {i === current && active && p && <span style={{ width: `${p.frac * 100}%` }} />}
              </span>
            );
          })}
        </div>
        {active && (
          <div className="card-status">
            <div>{a.status === "queued" ? "Queued" : stageLabel(a.stage)}</div>
            {p && (
              <>
                <Progress frac={p.frac} />
                <div className="muted small ellipsis">{p.msg}</div>
              </>
            )}
          </div>
        )}
        {(a.status === "rejected" || a.status === "error") && a.reason && (
          <ul className="reasons">
            {reasons(a.reason).map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
        {a.status === "cancelled" && <p className="muted small">Stopped during {stageLabel(a.stage).toLowerCase()}.</p>}
        <div className="badges">
          {headline(a).map((b) => (
            <span key={b.label} className="badge"><span>{b.label}</span>{b.value}</span>
          ))}
        </div>
        <div className="card-actions">
          <a className="btn small" href={href}>Open</a>
          {a.status === "rejected" && a.stage === "gate" && a.override !== "fail" && (
            <ActionButton onClick={() => api.override(job.id, a.seed, "pass").then(putJob)} title="Train this orbit anyway">
              Use anyway
            </ActionButton>
          )}
          {a.status === "passed" && (
            <ActionButton onClick={() => api.override(job.id, a.seed, "fail").then(putJob)} title="Drop it from the ranking; a new seed replaces it if the budget allows">
              Reject
            </ActionButton>
          )}
          {(a.status === "error" || a.status === "cancelled") && (
            <ActionButton onClick={() => api.retry(job.id, a.seed).then(putJob)} title="Run it again; finished stages are skipped">
              Retry
            </ActionButton>
          )}
          {a.status !== "discarded" && (
            <ActionButton kind="ghost" onClick={() => api.discard(job.id, a.seed).then(putJob)} title="Stop it and hide it; a new seed replaces it if the budget allows">
              Discard
            </ActionButton>
          )}
        </div>
      </div>
    </article>
  );
}
