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

// Video shapes. Only 3:4 at 768x1024 has been tested end to end (docs/FINDINGS.md, "Orbit videos").
const SHAPES: { key: string; label: string; ratio?: number }[] = [
  { key: "3:4", label: "3:4 portrait (tested)", ratio: 3 / 4 },
  { key: "2:3", label: "2:3 portrait", ratio: 2 / 3 },
  { key: "9:16", label: "9:16 portrait", ratio: 9 / 16 },
  { key: "1:1", label: "1:1 square", ratio: 1 },
  { key: "4:3", label: "4:3 landscape", ratio: 4 / 3 },
  { key: "3:2", label: "3:2 landscape", ratio: 3 / 2 },
  { key: "16:9", label: "16:9 landscape", ratio: 16 / 9 },
  { key: "image", label: "Closest to the image" },
  { key: "custom", label: "Custom" },
];
// Pixel budgets: giro's tested 768x1024, and the video model's own default, 1344x768.
const SIZES: Record<string, { label: string; px: number }> = {
  draft: { label: "Draft, ~0.44 MP (proxy orbit's tested size)", px: 576 * 768 },
  standard: { label: "Standard, ~0.8 MP (tested)", px: 768 * 1024 },
  large: { label: "Large, ~1 MP (the model's default)", px: 1344 * 768 },
};
const TESTED_PX = 768 * 1024;
const snap32 = (v: number) => Math.min(2048, Math.max(256, Math.round(v / 32) * 32)); // the video model's grid
const frameSize = (ratio: number, px: number): [number, number] => [snap32(Math.sqrt(px * ratio)), snap32(Math.sqrt(px / ratio))];
const closestShape = (ratio: number) =>
  SHAPES.filter((s) => s.ratio).reduce((a, b) => (Math.abs(Math.log(ratio / b.ratio!)) < Math.abs(Math.log(ratio / a.ratio!)) ? b : a));

// The largest box of the video's aspect ratio that fits the image (giro/hero.py), as fractions of it.
function coverBox(w: number, h: number, vw: number, vh: number) {
  const target = vw / vh;
  const [cw, ch] = w / h > target ? [h * target, h] : [w, w / target];
  return { width: cw / w, height: ch / h };
}

interface View {
  zoom: number; // 1: the largest box; 2: half its width
  cx: number;   // center of the kept region, as fractions of the image
  cy: number;
}
const CENTERED: View = { zoom: 1, cx: 0.5, cy: 0.5 };
const MAX_ZOOM = 4;
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// SAM 3 keeps one match per comma-separated item unless it ends in ":N" (ComfyUI's sam3_clip),
// so "two women" finds one woman per frame, a different one from frame to frame.
const COUNTS: Record<string, number> = { two: 2, both: 2, pair: 2, couple: 2, three: 3, four: 4, five: 5, six: 6 };
function countHints(prompt: string): { said: string; fix: string }[] {
  return prompt.split(",").map((p) => p.trim()).filter((p) => p && !/:\s*\d+(\.\d+)?$/.test(p)).flatMap((p) => {
    const m = p.match(/^(\d+|two|three|four|five|six|both|pair of|couple of|pair|couple)\s+(.+)$/i);
    if (!m) return [];
    const word = m[1].toLowerCase().replace(/ of$/, "");
    const n = /^\d+$/.test(word) ? Number(word) : COUNTS[word];
    return n > 1 ? [{ said: p, fix: `${m[2]}:${n}` }] : [];
  });
}

// The region the hero keeps: the cover box shrunk by the zoom, moved to the view's center, kept inside the image.
function cropRegion(base: { width: number; height: number }, v: View) {
  const width = base.width / v.zoom;
  const height = base.height / v.zoom;
  const left = clamp(v.cx - width / 2, 0, 1 - width);
  const top = clamp(v.cy - height / 2, 0, 1 - height);
  return { left, top, width, height, removed: 1 - width * height };
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
  const [kind, setKind] = useState<"person" | "object">("person");
  const [length, setLength] = useState("");
  const [steps, setSteps] = useState("");
  const [mode, setMode] = useState<"h3" | "proxy">("proxy");
  const [path, setPath] = useState("spiral");
  const [seeds, setSeeds] = useState("");
  const [shape, setShape] = useState("3:4");
  const [size, setSize] = useState("draft");
  const [customW, setCustomW] = useState("768");
  const [customH, setCustomH] = useState("1024");
  const [view, setView] = useState<View>(CENTERED);
  const preview = useRef<HTMLDivElement>(null);
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
  const shapeRatio = shape === "image" ? (picked ? closestShape(picked.width / picked.height).ratio! : 3 / 4) : SHAPES.find((s) => s.key === shape)?.ratio;
  const [vw, vh] = shapeRatio ? frameSize(shapeRatio, SIZES[size].px) : [Number(customW), Number(customH)];
  const sizeOk = [vw, vh].every((v) => Number.isInteger(v) && v % 32 === 0 && v >= 256 && v <= 2048);
  const base = useMemo(() => picked && sizeOk ? coverBox(picked.width, picked.height, vw, vh) : null, [picked, vw, vh, sizeOk]);
  const crop = base && cropRegion(base, view);
  const moved = view.zoom !== 1 || (crop && (Math.abs(crop.left + crop.width / 2 - 0.5) > 1e-3 || Math.abs(crop.top + crop.height / 2 - 0.5) > 1e-3));

  // A new image or video shape starts from the centered crop again.
  useEffect(() => setView(CENTERED), [picked, vw, vh]);

  // Keep the view's center where cropRegion clamps it, so a drag past the edge does not build up slack.
  const settle = (v: View): View => {
    if (!base) return v;
    const r = cropRegion(base, v);
    return { zoom: v.zoom, cx: r.left + r.width / 2, cy: r.top + r.height / 2 };
  };

  // The wheel zooms the crop (a non-passive listener, so the page does not scroll instead).
  useEffect(() => {
    const el = preview.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      setView((v) => settle({ ...v, zoom: clamp(v.zoom * Math.exp(-e.deltaY * 0.0006), 1, MAX_ZOOM) }));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  });

  function drag(e: React.PointerEvent<HTMLDivElement>) {
    e.stopPropagation();
    const box = preview.current?.getBoundingClientRect();
    if (!box || !crop) return;
    const start = { x: e.clientX, y: e.clientY, cx: crop.left + crop.width / 2, cy: crop.top + crop.height / 2 };
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const move = (m: PointerEvent) =>
      setView((v) => settle({ ...v, cx: start.cx + (m.clientX - start.x) / box.width, cy: start.cy + (m.clientY - start.y) / box.height }));
    const up = () => {
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
  }

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
    if (vw !== Number(orbit.width ?? 768) || vh !== Number(orbit.height ?? 1024)) {
      orbitParams.width = vw;
      orbitParams.height = vh;
    }
    if (length) orbitParams.length = Number(length);
    if (steps) orbitParams.steps = Number(steps);
    // The model is always sent: the server's default (the proxy orbit) is not the only choice.
    const orbitAll: Record<string, number | string> = { ...orbitParams, model: mode === "proxy" ? "wan22-control" : "h3" };
    if (mode === "proxy") {
      orbitAll.path = path;
      orbitAll.width = vw;
      orbitAll.height = vh;
    }
    const editing = editOn && editPrompt.trim() !== "";
    try {
      let job = await api.createJob(picked.file, {
        start: !editing,
        name: name || undefined,
        want,
        max_attempts: Math.max(want, maxAttempts),
        height_m: height ? Number(height) : undefined,
        subject: subject || undefined,
        kind: mode === "proxy" ? kind : undefined,
        orbit: orbitAll,
        crop: moved && crop ? [crop.left, crop.top, crop.left + crop.width, crop.top + crop.height] : undefined,
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

  // The kept region in image pixels: zooming in past the video's resolution adds no detail.
  const keptW = picked && crop ? Math.round(picked.width * crop.width) : 0;
  const keptH = picked && crop ? Math.round(picked.height * crop.height) : 0;
  const small = picked && crop && (keptW < vw || keptH < vh);
  return (
    <div className="page newjob">
      <header className="page-head">
        <h1>New job</h1>
        <p className="muted">
          One image of a subject in, a cropped and upright Gaussian splat out. giro generates several orbit videos,
          keeps the ones whose cameras check out, and ranks them.
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
          {picked ? (
            <div className="hero-preview" ref={preview}>
              <img src={picked.url} alt="" draggable={false} />
              {crop && <div
                className={`cropframe ${crop.removed > 0.005 ? "" : "full"}`}
                title="Drag to move the crop; scroll to zoom"
                onPointerDown={drag}
                onClick={(e) => e.stopPropagation()}
                style={{
                  left: `${crop.left * 100}%`, top: `${crop.top * 100}%`,
                  width: `${crop.width * 100}%`, height: `${crop.height * 100}%`,
                }}
              />}
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
                  {vw} × {vh} ({vw / gcd(vw, vh)}:{vh / gcd(vw, vh)}, {((vw * vh) / 1e6).toFixed(2)} MP)
                </span>
              </div>
              <div className="check-row zoom-row">
                <span className="muted">Crop zoom</span>
                <input type="range" min={1} max={MAX_ZOOM} step={0.01} value={view.zoom}
                  onChange={(e) => setView((v) => settle({ ...v, zoom: Number(e.target.value) }))} />
                <button className="btn small" disabled={!moved} onClick={() => setView(CENTERED)}>Reset</button>
              </div>
              {crop.removed > 0.005 ? (
                <p className={`note ${crop.removed > 0.15 ? "bad" : "warn"}`}>
                  The image is cropped to the video's aspect ratio: the {(crop.removed * 100).toFixed(0)}% outside the
                  frame is dropped{crop.removed > 0.15 ? ". Check that the whole subject is inside it." : "."} Drag the
                  frame to move it; scroll over the image to zoom.
                </p>
              ) : (
                <p className="note good">The aspect ratio matches the video. Nothing is cropped.</p>
              )}
              {small && (
                <p className="note warn">
                  The kept region is {keptW} × {keptH}, smaller than a video frame, so the hero adds no extra detail.
                </p>
              )}
            </div>
          )}
          {picked && !crop && <p className="note bad">The video size must be multiples of 32 from 256 to 2048.</p>}

          <label className="field">
            <span>Orbit</span>
            <select value={mode} onChange={(e) => {
              const m = e.target.value as "h3" | "proxy";
              setMode(m);
              setSize(m === "proxy" ? "draft" : "standard");
            }}>
              <option value="proxy">Proxy orbit: a 3D proxy guides Wan 2.2 Fun Control (default)</option>
              <option value="h3">MiniMax H3 with the 360 orbit LoRA</option>
            </select>
            <small className="muted">
              {mode === "h3"
                ? "Keeps the hero's look around the subject a little better, but MiniMax H3's license excludes users in the US, EU, UK and South Korea."
                : "A rough 3D proxy of the subject sets the camera path, so the views from above are real. As good as H3 or better on the hero view, sharper, and its models allow commercial use everywhere."}
            </small>
          </label>
          {mode === "proxy" && (
            <label className="field">
              <span>Subject</span>
              <select value={kind} onChange={(e) => setKind(e.target.value as "person" | "object")}>
                <option value="person">A person or character (default)</option>
                <option value="object">An object: a vehicle, machine or prop</option>
              </select>
              <small className="muted">
                {kind === "person"
                  ? "TripoSplat makes the proxy: cleaner backs on people, clothes and armor."
                  : "Pixal3D makes the proxy: truer hard-surface shapes (hulls, frames, wheels), but worse backs on people."}
              </small>
            </label>
          )}
          <div className="field-row">
            <label className="field">
              <span>Video shape</span>
              <select value={shape} onChange={(e) => setShape(e.target.value)}>
                {SHAPES.map((s) => (
                  <option key={s.key} value={s.key}>
                    {s.key === "image" && picked ? `Closest to the image (${closestShape(picked.width / picked.height).key})` : s.label}
                  </option>
                ))}
              </select>
            </label>
            {shape === "custom" ? (
              <>
                <label className="field">
                  <span>Width</span>
                  <input type="number" step={32} min={256} max={2048} value={customW} onChange={(e) => setCustomW(e.target.value)} />
                </label>
                <label className="field">
                  <span>Height</span>
                  <input type="number" step={32} min={256} max={2048} value={customH} onChange={(e) => setCustomH(e.target.value)} />
                </label>
              </>
            ) : (
              <label className="field">
                <span>Video size</span>
                <select value={size} onChange={(e) => setSize(e.target.value)}>
                  {Object.entries(SIZES).map(([k, s]) => <option key={k} value={k}>{s.label}</option>)}
                </select>
              </label>
            )}
          </div>
          {mode === "h3" && (vw * vh !== TESTED_PX || vw / vh !== 3 / 4) && sizeOk && (
            <p className="note warn">
              Only 3:4 at 768 × 1024 has been tested.
              {vw * vh > TESTED_PX * 1.05 ? " A larger video may not fit in 24 GB of GPU memory." : ""}
            </p>
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
              <small className="muted">About {mode === "proxy" ? 20 : 15} min of GPU each.</small>
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
            <small className="muted">
              What the segmenter should find. Each comma-separated item keeps one match; add ":N" to keep up to N, e.g.
              "woman:2, bowl, held object:2". Leave empty for the default.
            </small>
          </label>
          {countHints(subject).map((h) => (
            <p key={h.said} className="note warn">
              "{h.said}" keeps only one match per frame, a different one from frame to frame. Write "{h.fix}" to keep {h.fix.split(":").pop()}.
            </p>
          ))}
          <details className="advanced">
            <summary>Video settings</summary>
            <div className="field-row">
              <label className="field">
                <span>Frames</span>
                <input type="number" value={length} onChange={(e) => setLength(e.target.value)} placeholder={mode === "proxy" ? "81" : String(orbit.length ?? 158)} />
                <small className="muted">{mode === "proxy" ? "At 16 fps (Wan's 4k+1 grid)." : "At 24 fps; longer gives more views."}</small>
              </label>
              {mode === "proxy" && (
                <label className="field">
                  <span>Camera path</span>
                  <select value={path} onChange={(e) => setPath(e.target.value)}>
                    <option value="spiral">Spiral: two turns, rising to 45°</option>
                    <option value="ring">Ring at the hero's height</option>
                    <option value="wave">Wave: up to 45° and back</option>
                    <option value="loop">Loop: up to 45° and back to the hero, which ends the clip too</option>
                  </select>
                </label>
              )}
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
          <button className="btn primary big" disabled={!picked || !sizeOk || busy || (editOn && !editPrompt.trim())} onClick={submit}>
            {busy ? "Starting…" : editOn ? "Edit and review" : `Start ${want} orbit${want > 1 ? "s" : ""}`}
          </button>
        </div>
      </div>
    </div>
  );
}
