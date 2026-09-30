// Types and calls for giro's API (src/giro/api.py).

export type AttemptStatus =
  | "queued" | "running" | "passed" | "rejected" | "error" | "cancelled" | "discarded";

export interface Attempt {
  seed: number;
  status: AttemptStatus;
  stage: string;
  reason: string;
  gpus: Record<string, number>;
  eval_psnr: number | null;
  gate: Record<string, number | boolean | string>;
  seconds: number | null;
  gaussians: number | null;
  params: Record<string, Record<string, unknown>>;
  override: "pass" | "fail" | null;
  filled?: string[]; // gaps of the orbit the video model regenerated (gapfill)
}

export interface JobSpec {
  image: string;
  want: number;
  max_attempts: number;
  seeds: number[];
  orbit: Record<string, unknown>;
  params: Record<string, Record<string, unknown>>;
  video_gpus: number[];
  post_gpus: number[];
  use_edit: boolean;
}

export interface EditState {
  status?: "running" | "done" | "error";
  prompt?: string;
  seed?: number;
  fast?: boolean;
  error?: string;
  seconds?: number;
}

export type JobStatus = "draft" | "running" | "done" | "failed" | "error" | "cancelled" | "interrupted";

export interface Job {
  id: string;
  created: string;
  status: JobStatus;
  spec: JobSpec;
  ranking: number[];
  best: number | null;
  attempts: Attempt[];
  edit: EditState;
}

export interface GateCheck {
  metric: string;
  value: number | boolean | null;
  op: string;
  threshold: number | boolean;
  pass: boolean;
}

export interface AttemptDetail {
  job: string;
  attempt: Attempt;
  rank: number | null;
  running: boolean;
  metrics: Record<string, Record<string, unknown>>;
  gate: { passed: boolean; reasons: string[]; checks: GateCheck[]; ring: { azimuths?: number[]; frames?: string[] } } | null;
  dedup: Record<string, unknown> | null;
  files: Record<string, number>;
  previews: string[];
  checkpoints: string[];
}

export interface StageInfo {
  name: string;
  defaults: Record<string, unknown>;
  gpu: boolean;
}

// Events on /api/events. `start` means a stage runs (was not skipped).
export type GiroEvent =
  | { type: "hello"; jobs: Job[]; progress: ProgressEvent[]; previews: PreviewEvent[]; gpus: number[] }
  | { type: "job"; job: string; data: Job; ts: number }
  | { type: "deleted"; job: string; ts: number }
  | ProgressEvent
  | { type: "metric"; job: string; seed: number; stage: string; name: string; value: unknown; step?: number; ts: number }
  | PreviewEvent
  | { type: "log"; job: string; seed: number; stage: string; msg: string; ts: number }
  | { type: "start"; job: string; seed: number | null; stage: string; ts: number };

export interface PreviewEvent {
  type: "preview";
  job: string;
  seed: number | null; // null: the job's image edit
  stage: string;
  path: string;
  ts: number;
}

export interface ProgressEvent {
  type: "progress";
  job: string;
  seed: number | null; // null: the job's image edit
  stage: string;
  frac: number;
  msg: string;
  ts: number;
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(method: string, url: string, body?: unknown): Promise<T> {
  const init: RequestInit = { method };
  if (body instanceof FormData) init.body = body;
  else if (body !== undefined) {
    init.body = JSON.stringify(body);
    init.headers = { "Content-Type": "application/json" };
  }
  const r = await fetch(url, init);
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const data = await r.json();
      detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* not JSON */
    }
    throw new ApiError(r.status, detail);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}

const attemptUrl = (job: string, seed: number) => `/api/jobs/${encodeURIComponent(job)}/attempts/${seed}`;

export const api = {
  stages: () => call<StageInfo[]>("GET", "/api/stages"),
  job: (id: string) => call<Job & { running: boolean; elsewhere: boolean }>("GET", `/api/jobs/${encodeURIComponent(id)}`),
  createJob: (image: File, spec: Record<string, unknown>) => {
    const form = new FormData();
    form.append("image", image);
    form.append("spec", JSON.stringify(spec));
    return call<Job>("POST", "/api/jobs", form);
  },
  edit: (id: string, prompt: string, fast: boolean, seed?: number) =>
    call<Job>("POST", `/api/jobs/${encodeURIComponent(id)}/edit`, { prompt, fast, seed }),
  start: (id: string, useEdit: boolean) =>
    call<Job>("POST", `/api/jobs/${encodeURIComponent(id)}/start`, { use_edit: useEdit }),
  resume: (id: string) => call<Job>("POST", `/api/jobs/${encodeURIComponent(id)}/resume`),
  cancel: (id: string) => call<Job>("POST", `/api/jobs/${encodeURIComponent(id)}/cancel`),
  addSeeds: (id: string, n: number) => call<Job>("POST", `/api/jobs/${encodeURIComponent(id)}/seeds`, { n }),
  deleteJob: (id: string) => call<void>("DELETE", `/api/jobs/${encodeURIComponent(id)}`),
  attempt: (job: string, seed: number) => call<AttemptDetail>("GET", attemptUrl(job, seed)),
  events: (job: string, seed: number, types: string[]) =>
    call<GiroEvent[]>("GET", `${attemptUrl(job, seed)}/events?types=${types.join(",")}`),
  override: (job: string, seed: number, verdict: "pass" | "fail") =>
    call<Job>("POST", `${attemptUrl(job, seed)}/override`, { verdict }),
  discard: (job: string, seed: number) => call<Job>("POST", `${attemptUrl(job, seed)}/discard`),
  retry: (job: string, seed: number) => call<Job>("POST", `${attemptUrl(job, seed)}/retry`),
  setParams: (job: string, seed: number, params: Record<string, Record<string, unknown>>) =>
    call<Job>("POST", `${attemptUrl(job, seed)}/params`, params),
};

/** URL of a file in a job directory (path relative to the job, e.g. "attempts/3/video.mp4"). */
export const fileUrl = (job: string, path: string, version?: string | number) =>
  `/files/${encodeURIComponent(job)}/${path}${version !== undefined ? `?v=${version}` : ""}`;

export const thumbUrl = (job: string) => `/api/jobs/${encodeURIComponent(job)}/thumb`;
