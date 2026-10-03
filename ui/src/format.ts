// Plain-language names for what the pipeline does and measures.

import type { Attempt, AttemptStatus, GateCheck, JobStatus } from "./api";

export const STAGES: { name: string; label: string; short: string }[] = [
  { name: "orbit_video", label: "Generating the orbit video", short: "Video" },
  { name: "extract", label: "Extracting frames", short: "Frames" },
  { name: "dedup", label: "Picking frames", short: "Pick" },
  { name: "masks", label: "Masking the subject", short: "Masks" },
  { name: "poses_colmap", label: "Recovering the cameras", short: "Cameras" },
  { name: "gate", label: "Checking the orbit", short: "Gate" },
  { name: "dataset", label: "Preparing training data", short: "Data" },
  { name: "train", label: "Training the splat", short: "Train" },
  { name: "crop", label: "Cropping to the subject", short: "Crop" },
  { name: "canonicalize", label: "Standing it upright", short: "Upright" },
  { name: "export", label: "Exporting", short: "Export" },
];

// Steps that run only after the gate rejects an attempt, shown under the stage they redo.
const RECOVERY: Record<string, { label: string; under: string }> = {
  poses_fallback: { label: "Recovering the cameras another way", under: "poses_colmap" },
  gapfill: { label: "Regenerating a missing arc", under: "orbit_video" },
};

export const stageIndex = (name: string) => {
  const recovery = Object.entries(RECOVERY).find(([k]) => name.startsWith(k));
  return STAGES.findIndex((s) => (recovery ? recovery[1].under : name).startsWith(s.name));
};

// The key a stage's live progress is published under ("gate", "poses_fallback", ...).
export const stageKey = (stage: string): string | undefined =>
  Object.keys(RECOVERY).find((k) => stage.startsWith(k)) ?? STAGES.find((s) => stage.startsWith(s.name))?.name;

export function stageLabel(stage: string): string {
  const waiting = stage.endsWith("(waiting for a GPU)");
  const recovery = Object.entries(RECOVERY).find(([k]) => stage.startsWith(k));
  const s = STAGES[stageIndex(stage)];
  const label = recovery?.[1].label ?? s?.label ?? stage;
  return waiting ? `${label}: waiting for a GPU` : label;
}

export const ATTEMPT_STATUS: Record<AttemptStatus, { label: string; tone: Tone }> = {
  queued: { label: "Queued", tone: "muted" },
  running: { label: "Running", tone: "active" },
  passed: { label: "Passed", tone: "good" },
  rejected: { label: "Rejected", tone: "bad" },
  error: { label: "Error", tone: "bad" },
  cancelled: { label: "Stopped", tone: "muted" },
  discarded: { label: "Discarded", tone: "muted" },
};

export const JOB_STATUS: Record<JobStatus, { label: string; tone: Tone }> = {
  draft: { label: "Not started", tone: "warn" },
  running: { label: "Running", tone: "active" },
  done: { label: "Done", tone: "good" },
  failed: { label: "No usable orbit", tone: "bad" },
  error: { label: "Error", tone: "bad" },
  cancelled: { label: "Stopped", tone: "muted" },
  interrupted: { label: "Interrupted", tone: "warn" },
};

export type Tone = "good" | "bad" | "warn" | "active" | "muted";

// Gate metrics as the UI shows them: short name, unit, and how to print a value.
export const METRICS: Record<string, { label: string; fmt: (v: number) => string; help: string }> = {
  azimuth_coverage: { label: "Orbit", fmt: (v) => `${v.toFixed(0)}°`, help: "How far around the subject the camera got, start to end (net)" },
  n_views: { label: "Views", fmt: (v) => `${v}`, help: "Frames that got a camera pose" },
  reg_rate: { label: "Posed", fmt: (v) => `${(v * 100).toFixed(0)}%`, help: "Share of the images that got a camera pose" },
  reproj_err: { label: "Reproj.", fmt: (v) => `${v.toFixed(2)} px`, help: "Reprojection error: how consistent the frames are geometrically" },
  azimuth_monotonic: { label: "Steady", fmt: (v) => `${(v * 100).toFixed(0)}%`, help: "Share of camera steps that go forward" },
  max_step_deg: { label: "Max step", fmt: (v) => `${v.toFixed(0)}°`, help: "Largest jump between two consecutive frames" },
  loop_closure: { label: "Loop", fmt: (v) => v.toFixed(2), help: "Distance from the last frame back to the first, in orbit radii" },
  radius_cv: { label: "Distance", fmt: (v) => `${(v * 100).toFixed(0)}%`, help: "How much the camera distance to the subject varies" },
  hero_registered: { label: "Hero", fmt: (v) => (v ? "posed" : "missing"), help: "Whether the hero image got a camera pose" },
};

export function checkText(c: GateCheck, ring?: { azimuth_span?: number }): string {
  const m = METRICS[c.metric];
  if (c.value === null || c.value === undefined) return "not measured";
  if (typeof c.value === "boolean") return c.value ? "yes" : "no";
  let v = m ? m.fmt(c.value) : String(c.value);
  // The orbit check is on the net angle (where the camera ended up relative to the start). A camera
  // that swings out and back nets ~0 while having seen a wider range: say both, or the row reads as a bug.
  const span = ring?.azimuth_span;
  if (c.metric === "azimuth_coverage" && typeof span === "number" && Math.abs(span - c.value) >= 1) v = `${v} net, saw ${span.toFixed(0)}°`;
  if (typeof c.threshold === "boolean") return v;
  const t = m ? m.fmt(c.threshold) : String(c.threshold);
  return `${v} (${c.op === ">=" ? "need ≥" : "allowed ≤"} ${t})`;
}

/** The badges a card shows before any detail has loaded, from the attempt record. */
export function headline(a: Attempt): { label: string; value: string }[] {
  const out: { label: string; value: string }[] = [];
  const g = a.gate;
  if (typeof g.azimuth_coverage === "number") out.push({ label: "Orbit", value: `${g.azimuth_coverage.toFixed(0)}°` });
  if (typeof g.n_views === "number") out.push({ label: "Views", value: String(g.n_views) });
  if (a.eval_psnr !== null) out.push({ label: "PSNR", value: a.eval_psnr.toFixed(1) });
  if (a.gaussians !== null) out.push({ label: "Splats", value: `${Math.round(a.gaussians / 1000)}K` });
  return out;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "";
  if (seconds < 90) return `${Math.round(seconds)} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

export function ago(iso: string): string {
  const t = new Date(iso).getTime();
  const s = (Date.now() - t) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** "20260929-094954-m3-knight" -> "m3-knight" */
export const jobName = (id: string) => id.replace(/^\d{8}-\d{6}-/, "");

export const bytes = (n: number) =>
  n < 2 ** 20 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 2 ** 20).toFixed(1)} MB`;

/** Split a joined gate reason ("a; b") into its sentences, capitalized. */
export const reasons = (reason: string) =>
  reason
    .split("; ")
    .filter(Boolean)
    .map((r) => r.charAt(0).toUpperCase() + r.slice(1));
