// Library: every finished splat. The thumbnail is the canonicalize stage's
// turnaround sheet (8 views around, then 8 from above and below); hovering spins through
// the first row.

import { fileUrl } from "./api";
import { jobName } from "./format";
import { useStore } from "./store";

export function Library() {
  const jobs = useStore((s) => s.jobs);
  const done = jobs.filter((j) => j.best !== null);
  const running = jobs.filter((j) => j.status === "running").length;
  return (
    <div className="page">
      <header className="page-head library-head">
        <div>
          <h1>Library</h1>
          <p className="muted">
            {done.length} finished splat{done.length === 1 ? "" : "s"}
            {running > 0 && `, ${running} job${running === 1 ? "" : "s"} running`}. Hover to turn one around.
          </p>
        </div>
        <a className="btn primary" href="#/new">New job</a>
      </header>
      {done.length === 0 && <p className="muted">Nothing finished yet. Start a job and its best result lands here.</p>}
      <div className="library">
        {done.map((j) => {
          const seed = j.best!;
          const a = j.attempts.find((x) => x.seed === seed);
          const v = a?.seconds ?? 0;
          const base = `attempts/${seed}`;
          const height = j.spec.params.canonicalize?.height_m;
          return (
            <article key={j.id} className="lib-card">
              <a href={`#/job/${encodeURIComponent(j.id)}/${seed}/result`} className="spin"
                style={{ backgroundImage: `url(${fileUrl(j.id, `${base}/canonical/turnaround.jpg`, v)})` }}
                aria-label={`Open ${jobName(j.id)}`} />
              <div className="lib-body">
                <a className="lib-name" href={`#/job/${encodeURIComponent(j.id)}`}>{jobName(j.id)}</a>
                <div className="muted small">
                  {[height !== undefined && `${String(height)} m`, a?.gaussians && `${Math.round(a.gaussians / 1000)}K splats`]
                    .filter(Boolean)
                    .join(" · ")}
                </div>
                <div className="downloads">
                  {(["spz", "sog", "ply"] as const).map((ext) => (
                    <a key={ext} className="btn small" href={fileUrl(j.id, `${base}/export/splat.${ext}`, v)} download={`${jobName(j.id)}.${ext}`}>
                      {ext.toUpperCase()}
                    </a>
                  ))}
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}
