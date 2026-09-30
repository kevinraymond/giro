// Live state from /api/events: every job, the latest progress and preview of each attempt.
// The server sends a snapshot on connect, so reconnecting just starts over.

import { useSyncExternalStore } from "react";
import type { GiroEvent, Job, ProgressEvent } from "./api";

export interface State {
  connected: boolean;
  jobs: Job[]; // newest first
  // `${job}/${seed}` -> stage -> latest progress
  progress: Record<string, Record<string, ProgressEvent>>;
  // `${job}/${seed}` -> latest preview (path relative to the job dir) and when it arrived
  previews: Record<string, { path: string; stage: string; ts: number }>;
  gpus: number[];
}

let state: State = { connected: false, jobs: [], progress: {}, previews: {}, gpus: [] };
const listeners = new Set<() => void>();
const eventListeners = new Set<(e: GiroEvent) => void>();

function set(next: Partial<State>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

/** Progress and previews are per attempt; seed null is the job's image edit. */
export const key = (job: string, seed: number | null) => `${job}/${seed ?? "edit"}`;

function apply(e: GiroEvent) {
  switch (e.type) {
    case "hello": {
      const progress: State["progress"] = {};
      for (const p of e.progress) (progress[key(p.job, p.seed)] ??= {})[p.stage] = p;
      const previews: State["previews"] = {};
      for (const p of e.previews) previews[key(p.job, p.seed)] = { path: p.path, stage: p.stage, ts: p.ts };
      set({ jobs: e.jobs, progress, previews, gpus: e.gpus });
      break;
    }
    case "job": {
      const i = state.jobs.findIndex((j) => j.id === e.job);
      const jobs = i >= 0 ? state.jobs.map((j, k) => (k === i ? e.data : j)) : [e.data, ...state.jobs];
      jobs.sort((a, b) => (a.id < b.id ? 1 : -1));
      set({ jobs });
      break;
    }
    case "deleted":
      set({ jobs: state.jobs.filter((j) => j.id !== e.job) });
      break;
    case "progress": {
      const k = key(e.job, e.seed);
      set({ progress: { ...state.progress, [k]: { ...state.progress[k], [e.stage]: e } } });
      break;
    }
    case "start": {
      const k = key(e.job, e.seed);
      const stages = { ...state.progress[k] };
      delete stages[e.stage];
      set({ progress: { ...state.progress, [k]: stages } });
      break;
    }
    case "preview":
      set({ previews: { ...state.previews, [key(e.job, e.seed)]: { path: e.path, stage: e.stage, ts: e.ts } } });
      break;
  }
  eventListeners.forEach((l) => l(e));
}

let socket: WebSocket | null = null;
let retry = 0;

export function connect() {
  if (socket) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/events`);
  socket = ws;
  ws.onopen = () => {
    retry = 0;
    set({ connected: true });
  };
  ws.onmessage = (m) => apply(JSON.parse(m.data));
  ws.onclose = () => {
    socket = null;
    set({ connected: false });
    // Back off to at most 5 s; the server may be restarting.
    setTimeout(connect, Math.min(5000, 250 * 2 ** retry++));
  };
}

export function useStore<T>(select: (s: State) => T): T {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => select(state),
  );
}

/** Every raw event, for views that keep their own history (curves, logs). */
export function onEvent(listener: (e: GiroEvent) => void): () => void {
  eventListeners.add(listener);
  return () => eventListeners.delete(listener);
}

/** Apply a job returned by an action at once, before its event arrives. */
export function putJob(job: Job) {
  apply({ type: "job", job: job.id, data: job, ts: Date.now() / 1000 });
}
