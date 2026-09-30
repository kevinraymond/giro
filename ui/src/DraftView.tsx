// A job that has not started: try background edits, compare them with the
// original on a before/after slider, then start the orbits from either image.

import { useState } from "react";
import { api, fileUrl, thumbUrl, type Job } from "./api";
import { go } from "./App";
import { jobName } from "./format";
import { key, putJob, useStore } from "./store";
import { ActionButton, Progress } from "./ui";

export function DraftView({ job }: { job: Job }) {
  const e = job.edit;
  const running = e.status === "running";
  const progress = useStore((s) => s.progress[key(job.id, null)]?.edit);
  const preview = useStore((s) => s.previews[key(job.id, null)]);
  const [prompt, setPrompt] = useState(e.prompt ?? "");
  const [fast, setFast] = useState(e.fast ?? true);
  const [useEdit, setUseEdit] = useState(true);
  const edited = e.status === "done";
  // A new edit rewrites edited.png in place: version its URL by the edit's seed.
  const editedUrl = fileUrl(job.id, "input/edited.png", e.seed);
  const n = job.spec.want;

  return (
    <div className="page">
      <header className="job-head">
        <img className="job-thumb" src={thumbUrl(job.id)} alt="" />
        <div className="job-title">
          <h1>{jobName(job.id)}</h1>
          <p className="muted small">Not started. Change the background first if a full orbit would run into walls or clutter, then start the orbits.</p>
        </div>
        <div className="job-actions">
          <ActionButton kind="ghost" disabled={running} onClick={async () => {
            if (!confirm(`Delete ${jobName(job.id)}?`)) return;
            await api.deleteJob(job.id);
            go();
          }}>Delete</ActionButton>
        </div>
      </header>

      <div className="draft">
        <div className="draft-view">
          {running ? (
            <div className="compare">
              <img src={fileUrl(job.id, "input/source.png")} alt="original" />
              {preview && <img className="live" src={fileUrl(job.id, preview.path, preview.ts)} alt="edit in progress" />}
              <div className="compare-status">
                <span>{progress?.msg ?? "Starting the edit model"}</span>
                <Progress frac={progress?.frac ?? 0} />
              </div>
            </div>
          ) : edited ? (
            <BeforeAfter before={fileUrl(job.id, "input/source.png")} after={editedUrl} />
          ) : (
            <div className="compare"><img src={fileUrl(job.id, "input/source.png")} alt="original" /></div>
          )}
        </div>

        <aside className="draft-side">
          <label className="field">
            <span>Change the image</span>
            <textarea rows={4} value={prompt} onChange={(ev) => setPrompt(ev.target.value)}
              placeholder="Replace the background with a plain, softly lit studio backdrop. Keep the person exactly the same." />
            <small className="muted">Say what to keep as well as what to change; the edit model follows it closely.</small>
          </label>
          <label className="switch">
            <input type="checkbox" checked={fast} onChange={(ev) => setFast(ev.target.checked)} />
            <span>Quick (4 steps, about 20 s)</span>
          </label>
          {!fast && <small className="muted">40 steps: a few minutes, sometimes finer detail.</small>}
          <ActionButton disabled={running || !prompt.trim()} onClick={() => api.edit(job.id, prompt, fast).then(putJob)}>
            {edited || e.status === "error" ? "Edit again (new seed)" : "Edit"}
          </ActionButton>
          {e.status === "error" && <p className="note bad">{e.error}</p>}

          <div className="draft-start">
            {edited && (
              <div className="choice" role="radiogroup" aria-label="Image to orbit">
                <label><input type="radio" checked={useEdit} onChange={() => setUseEdit(true)} /> Orbit the edited image</label>
                <label><input type="radio" checked={!useEdit} onChange={() => setUseEdit(false)} /> Orbit the original</label>
              </div>
            )}
            <ActionButton kind="primary" disabled={running}
              onClick={() => api.start(job.id, edited && useEdit).then(putJob)}>
              Start {n} orbit{n > 1 ? "s" : ""}
            </ActionButton>
            {edited && useEdit && (
              <small className="muted">The edit is {e.seconds ? `done (${e.seconds.toFixed(0)} s)` : "done"}, at about 1 megapixel: enough for the video, a little softer than the original as the hero view.</small>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

/** Two images of one subject, the second revealed from the left by a slider. */
function BeforeAfter({ before, after }: { before: string; after: string }) {
  const [x, setX] = useState(50);
  return (
    <div className="compare">
      <img src={before} alt="original" />
      <img className="after" src={after} alt="edited" style={{ clipPath: `inset(0 ${100 - x}% 0 0)` }} />
      <div className="divider" style={{ left: `${x}%` }} />
      <span className="tag left">Edited</span>
      <span className="tag right">Original</span>
      <input type="range" min={0} max={100} value={x} onChange={(ev) => setX(Number(ev.target.value))} aria-label="Compare edited and original" />
    </div>
  );
}
