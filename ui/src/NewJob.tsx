import { useEffect, useMemo, useRef, useState } from "react";
import { api, ApiError, type StageInfo } from "./api";
import { go } from "./App";
import { putJob } from "./store";

interface Picked {
  file: File;
  url: string;
  width: number;
  height: number;
}

const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);

// The hero is center-cropped to the video's aspect ratio (giro/hero.py); show what survives.
function cropBox(w: number, h: number, vw: number, vh: number) {
  const target = vw / vh;
  const [cw, ch] = w / h > target ? [Math.round(h * target), h] : [w, Math.round(w / target)];
  return { left: (w - cw) / 2 / w, top: (h - ch) / 2 / h, width: cw / w, height: ch / h, removed: 1 - (cw * ch) / (w * h) };
}

export function NewJob() {
  const [stages, setStages] = useState<StageInfo[] | null>(null);
  const [picked, setPicked] = useState<Picked | null>(null);
  const [dragging, setDragging] = useState(false);
  const [name, setName] = useState("");
  const [height, setHeight] = useState("1.7");
  const [want, setWant] = useState(3);
  const [maxAttempts, setMaxAttempts] = useState(6);
  const [subject, setSubject] = useState("");
  const [length, setLength] = useState("");
  const [steps, setSteps] = useState("");
  const [seeds, setSeeds] = useState("");
  const [editOn, setEditOn] = useState(false);
  const [editPrompt, setEditPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api.stages().then(setStages).catch(() => setStages([]));
  }, []);
  useEffect(() => () => {
    if (picked) URL.revokeObjectURL(picked.url);
  }, [picked]);

  const defaults = (stage: string) => stages?.find((s) => s.name === stage)?.defaults ?? {};
  const orbit = defaults("orbit_video");
  const vw = Number(orbit.width ?? 768);
  const vh = Number(orbit.height ?? 1024);
  const crop = useMemo(() => picked && cropBox(picked.width, picked.height, vw, vh), [picked, vw, vh]);

  function pick(file: File | undefined) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError(`${file.name} is not an image`);
      return;
    }
    setError(null);
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setPicked({ file, url, width: img.naturalWidth, height: img.naturalHeight });
      if (!name) setName(file.name.replace(/\.[^.]+$/, "").replace(/[^A-Za-z0-9_.-]+/g, "-").slice(0, 64));
    };
    img.onerror = () => setError(`The browser cannot read ${file.name}`);
    img.src = url;
  }

  async function submit() {
    if (!picked) return;
    setBusy(true);
    setError(null);
    const orbitParams: Record<string, number> = {};
    if (length) orbitParams.length = Number(length);
    if (steps) orbitParams.steps = Number(steps);
    const editing = editOn && editPrompt.trim() !== "";
    try {
      let job = await api.createJob(picked.file, {
        start: !editing,
        name: name || undefined,
        want,
        max_attempts: Math.max(want, maxAttempts),
        height_m: height ? Number(height) : undefined,
        subject: subject || undefined,
        orbit: orbitParams,
        seeds: seeds.split(/[\s,]+/).filter(Boolean).map(Number),
      });
      // With an edit, the job waits as a draft: review the edit there, then start.
      if (editing) job = await api.edit(job.id, editPrompt, true);
      putJob(job);
      go("job", job.id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
      setBusy(false);
    }
  }

  const small = picked && (picked.width < vw || picked.height < vh);
  return (
    <div className="page newjob">
      <header className="page-head">
        <h1>New job</h1>
        <p className="muted">
          One image of a subject in, a cropped and upright Gaussian splat out. giro generates several orbit videos,
          keeps the ones whose cameras make a clean circle, and ranks them.
        </p>
      </header>

      <div className="newjob-grid">
        <div
          className={`dropzone ${dragging ? "over" : ""} ${picked ? "has" : ""}`}
          onClick={() => input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            pick(e.dataTransfer.files[0]);
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        >
          <input ref={input} type="file" accept="image/*" hidden onChange={(e) => pick(e.target.files?.[0])} />
          {picked && crop ? (
            <div className="hero-preview">
              <img src={picked.url} alt="" />
              {crop.removed > 0.005 && (
                <div
                  className="cropframe"
                  style={{
                    left: `${crop.left * 100}%`, top: `${crop.top * 100}%`,
                    width: `${crop.width * 100}%`, height: `${crop.height * 100}%`,
                  }}
                />
              )}
            </div>
          ) : (
            <div className="dropzone-empty">
              <svg viewBox="0 0 24 24" width="36" height="36" aria-hidden>
                <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <strong>Drop an image here</strong>
              <span className="muted">or click to choose one. One subject, fully in frame, works best.</span>
            </div>
          )}
        </div>

        <div className="form">
          {picked && crop && (
            <div className="checks">
              <div className="check-row">
                <span className="muted">Image</span>
                <span>
                  {picked.width} × {picked.height}
                </span>
              </div>
              <div className="check-row">
                <span className="muted">Video frames</span>
                <span>
                  {vw} × {vh} ({vw / gcd(vw, vh)}:{vh / gcd(vw, vh)})
                </span>
              </div>
              {crop.removed > 0.005 ? (
                <p className={`note ${crop.removed > 0.15 ? "bad" : "warn"}`}>
                  The image is cropped to the video's aspect ratio: the {(crop.removed * 100).toFixed(0)}% outside the
                  frame is dropped{crop.removed > 0.15 ? ". Check that the whole subject is inside it." : "."}
                </p>
              ) : (
                <p className="note good">The aspect ratio matches the video. Nothing is cropped.</p>
              )}
              {small && <p className="note warn">The image is smaller than a video frame, so the hero adds no extra detail.</p>}
            </div>
          )}

          <label className="field">
            <span>Name</span>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder="from the file name" pattern="[A-Za-z0-9_.-]+" />
          </label>
          <div className="field-row">
            <label className="field">
              <span>Subject height</span>
              <div className="suffixed">
                <input type="number" min="0.05" step="0.05" value={height} onChange={(e) => setHeight(e.target.value)} />
                <span>m</span>
              </div>
              <small className="muted">Sets the real-world size of the export.</small>
            </label>
            <label className="field">
              <span>Good orbits wanted</span>
              <input type="number" min={1} max={12} value={want} onChange={(e) => setWant(Number(e.target.value))} />
              <small className="muted">About 15 min of GPU each.</small>
            </label>
            <label className="field">
              <span>Seeds at most</span>
              <input type="number" min={want} max={32} value={Math.max(want, maxAttempts)} onChange={(e) => setMaxAttempts(Number(e.target.value))} />
              <small className="muted">Rejected videos are rerolled up to this.</small>
            </label>
          </div>
          <label className="field">
            <span>What to keep</span>
            <input value={subject} onChange={(e) => setSubject(e.target.value)} placeholder={String(defaults("masks").subject_prompt ?? "main subject")} />
            <small className="muted">What the segmenter should find, e.g. "person, held object:2". Leave empty for the default.</small>
          </label>
          <details className="advanced">
            <summary>Video settings</summary>
            <div className="field-row">
              <label className="field">
                <span>Frames</span>
                <input type="number" value={length} onChange={(e) => setLength(e.target.value)} placeholder={String(orbit.length ?? 158)} />
                <small className="muted">At 24 fps; longer gives more views.</small>
              </label>
              <label className="field">
                <span>Steps</span>
                <input type="number" value={steps} onChange={(e) => setSteps(e.target.value)} placeholder={String(orbit.steps ?? 20)} />
              </label>
              <label className="field">
                <span>Seeds to try first</span>
                <input value={seeds} onChange={(e) => setSeeds(e.target.value)} placeholder="random" />
              </label>
            </div>
          </details>

          <div className="edit-first">
            <label className="switch">
              <input type="checkbox" checked={editOn} onChange={(e) => setEditOn(e.target.checked)} />
              <span>Change the background first</span>
            </label>
            {editOn && (
              <label className="field">
                <textarea rows={3} value={editPrompt} onChange={(e) => setEditPrompt(e.target.value)}
                  placeholder="Replace the background with a plain, softly lit studio backdrop. Keep the person exactly the same." />
                <small className="muted">
                  For subjects near a wall or clutter that a full orbit would run into. You review the edit before any
                  orbit starts.
                </small>
              </label>
            )}
          </div>

          {error && <p className="note bad">{error}</p>}
          <button className="btn primary big" disabled={!picked || busy || (editOn && !editPrompt.trim())} onClick={submit}>
            {busy ? "Starting…" : editOn ? "Edit and review" : `Start ${want} orbit${want > 1 ? "s" : ""}`}
          </button>
        </div>
      </div>
    </div>
  );
}
