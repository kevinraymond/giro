// Small shared pieces.

import { useState, type ReactNode } from "react";
import { ApiError } from "./api";
import type { Tone } from "./format";

export function Pill({ tone, children, title }: { tone: Tone; children: ReactNode; title?: string }) {
  return (
    <span className={`pill ${tone}`} title={title}>
      {children}
    </span>
  );
}

export function Progress({ frac, tone = "active" }: { frac: number; tone?: Tone }) {
  return (
    <div className={`bar ${tone}`} role="progressbar" aria-valuenow={Math.round(frac * 100)} aria-valuemin={0} aria-valuemax={100}>
      <div style={{ width: `${Math.max(0, Math.min(1, frac)) * 100}%` }} />
    </div>
  );
}

/** Runs an action, disables its button meanwhile, and reports a failure next to it. */
export function useAction(): [(fn: () => Promise<unknown>) => void, boolean, string | null] {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    fn()
      .catch((e) => setError(e instanceof ApiError ? e.message : String(e)))
      .finally(() => setBusy(false));
  };
  return [run, busy, error];
}

export function ActionButton({
  onClick, children, kind = "", title, disabled,
}: {
  onClick: () => Promise<unknown>;
  children: ReactNode;
  kind?: "" | "primary" | "danger" | "ghost";
  title?: string;
  disabled?: boolean;
}) {
  const [run, busy, error] = useAction();
  return (
    <span className="action">
      <button className={`btn ${kind}`} disabled={busy || disabled} title={title} onClick={() => run(onClick)}>
        {children}
      </button>
      {error && <span className="action-error">{error}</span>}
    </span>
  );
}
