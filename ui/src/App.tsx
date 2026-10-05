import { useEffect, useState } from "react";
import { fileUrl, thumbUrl } from "./api";
import { AttemptView } from "./AttemptView";
import { JOB_STATUS, ago, jobName } from "./format";
import { JobView } from "./JobView";
import { VrView } from "./lazy";
import type { Primer, VrItem } from "./VrView";
import { Library } from "./Library";
import { NewJob } from "./NewJob";
import { useStore } from "./store";
import { Pill } from "./ui";

// Routes: #/new, #/job/<id>, #/job/<id>/<seed>[/<tab>], #/vr/<id>/<seed>, #/vr-bench/<file>;
// anything else is the library.
function useRoute(): string[] {
  // Options after '?' (e.g. #/vr-bench/a.spz?hz=72) are read by the page itself.
  const read = () => location.hash.replace(/^#\/?/, "").split("?")[0].split("/").filter(Boolean).map(decodeURIComponent);
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export const go = (...parts: (string | number)[]) => {
  location.hash = "#/" + parts.map((p) => encodeURIComponent(String(p))).join("/");
};

// fosfora's depth-primer effect (board #3659): each primer mode twice, in mirrored order against drift.
const BENCH_PRESETS: Record<string, VrItem[]> = {
  primer: (["floor", "none", "opaque", "opaque", "none", "floor"] as Primer[])
    .map((primer) => ({ url: "/bench/crowd_245k.spz", label: `245k ${primer}`, primer }))
    .concat((["none", "opaque"] as Primer[]).map((primer) => ({ url: "/bench/crowd_490k.spz", label: `490k ${primer}`, primer }))),
};

export function App() {
  const route = useRoute();
  const [page, id, seed, tab] = route;
  // VR pages are the whole window: the Quest browser has little room and no use for the sidebar.
  if (page === "vr" && id && seed) return <VrView items={[{ url: fileUrl(id, `attempts/${seed}/export/splat.spz`), label: `${jobName(id)} · seed ${seed}` }]} />;
  // #/vr-bench/a.spz,b.spz,...: measure each in turn in one VR session.
  // An item may carry @std=N (Spark's maxStdDev) and @primer=floor|none|opaque (VrView's Primer),
  // e.g. crowd_245k.spz@std=2@primer=opaque.
  // Named sequences, so a URL is short enough to type in the headset: #/vr-bench/primer?scale=1.
  if (page === "vr-bench" && id && BENCH_PRESETS[id]) {
    const scale = new URLSearchParams(location.hash.split("?")[1] ?? "").get("scale") ?? "0.6";
    const next = scale === "1" ? { href: "#/vr-bench/primer?scale=0.6", label: "Run 2: scale 0.6" } : undefined;
    return <VrView key={scale} items={BENCH_PRESETS[id]} next={next} />;
  }
  if (page === "vr-bench" && id) return <VrView items={id.split(",").map((item) => {
    const [f, ...opts] = item.split("@");
    const opt = Object.fromEntries(opts.map((o) => o.split("=")));
    const std = opt.std ? Number(opt.std) : undefined;
    const primer = opt.primer as Primer | undefined;
    return { url: `/bench/${f}`, label: f.replace(/\.spz$/, "") + (std ? ` std ${std}` : "") + (primer ? ` ${primer}` : ""), maxStdDev: std, primer };
  })} />;
  let main;
  if (page === "job" && id && seed) main = <AttemptView key={`${id}/${seed}`} jobId={id} seed={Number(seed)} tab={tab} />;
  else if (page === "job" && id) main = <JobView key={id} jobId={id} />;
  else if (page === "new") main = <NewJob />;
  else main = <Home />;
  return (
    <div className="app">
      <Sidebar current={page === "job" ? id : undefined} />
      <main className="main">{main}</main>
    </div>
  );
}

/** The library once something is finished; before that, straight to a new job. */
function Home() {
  const jobs = useStore((s) => s.jobs);
  const connected = useStore((s) => s.connected);
  if (!connected && jobs.length === 0) return <div className="page"><p className="muted">Connecting…</p></div>;
  return jobs.length ? <Library /> : <NewJob />;
}

function Sidebar({ current }: { current?: string }) {
  const jobs = useStore((s) => s.jobs);
  const connected = useStore((s) => s.connected);
  return (
    <aside className="sidebar">
      <div className="brand">
        <svg viewBox="0 0 32 32" width="22" height="22" aria-hidden>
          <circle cx="16" cy="16" r="11" fill="none" stroke="var(--accent)" strokeWidth="3" strokeDasharray="52 18" />
        </svg>
        <a href="#/">giro</a>
        <span className={`conn ${connected ? "on" : "off"}`} title={connected ? "Connected" : "Reconnecting to the server…"} />
      </div>
      <a className="btn primary block" href="#/new">
        New job
      </a>
      <a className="navlink" href="#/">Library</a>
      <nav className="joblist">
        {jobs.length === 0 && <p className="muted small pad">No jobs yet.</p>}
        {jobs.map((j) => {
          const st = JOB_STATUS[j.status] ?? { label: j.status, tone: "muted" };
          const running = j.attempts.filter((a) => a.status === "running").length;
          return (
            <a key={j.id} href={`#/job/${encodeURIComponent(j.id)}`} className={`jobitem ${j.id === current ? "current" : ""}`}>
              <img src={thumbUrl(j.id)} alt="" loading="lazy" />
              <div className="jobitem-text">
                <div className="jobitem-name">{jobName(j.id)}</div>
                <div className="jobitem-meta">
                  <Pill tone={st.tone}>{j.status === "running" && running ? `${running} running` : st.label}</Pill>
                  <span className="muted">{ago(j.created)}</span>
                </div>
              </div>
            </a>
          );
        })}
      </nav>
    </aside>
  );
}
