// Attempt detail: the recovered camera ring in 3D, azimuth per frame, and the
// frame strip with what dedup dropped and what COLMAP could not place.

import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { fileUrl, type AttemptDetail } from "./api";

interface Cam {
  name: string;
  hero: boolean;
  position: number[];
  right: number[];
  down: number[];
  forward: number[];
  fx: number;
  fy: number;
  width: number;
  height: number;
  error: number | null;
}
interface CamerasData {
  frame: "canonical" | "orbit";
  cameras: Cam[];
  points: number[];
  colors: number[];
}
interface FrameInfo {
  name: string;
  kept: boolean;
  posed: boolean;
  azimuth: number | null;
  error: number | null;
  generated?: boolean; // made by a gap fill
  replaced?: boolean; // dropped for a gap fill's frames
}
interface FramesData {
  frames: FrameInfo[];
  static_start: number | null;
  static_end: number | null;
  hero: { posed: boolean; error: number | null };
  filled?: string[];
  thumb_height: number;
}

const get = <T,>(url: string) => fetch(url).then((r) => (r.ok ? (r.json() as Promise<T>) : Promise.reject(new Error(r.statusText))));

// Per-view reprojection error against the gate's limit on the mean (1.5 px).
export function errorTone(e: number | null): "good" | "warn" | "bad" | "muted" {
  if (e === null) return "muted";
  return e <= 1.0 ? "good" : e <= 1.5 ? "warn" : "bad";
}
const TONE_HEX = { good: 0x5cc98a, warn: 0xe8b04a, bad: 0xef6b63, muted: 0x8b939c } as const;

export function CamerasTab({ detail }: { detail: AttemptDetail }) {
  const base = `/api/jobs/${encodeURIComponent(detail.job)}/attempts/${detail.attempt.seed}`;
  const [cams, setCams] = useState<CamerasData | null>(null);
  const [frames, setFrames] = useState<FramesData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const version = `${detail.attempt.status}/${detail.attempt.stage}`;

  useEffect(() => {
    get<FramesData>(`${base}/frames`).then(setFrames).catch(() => setFrames(null));
    get<CamerasData>(`${base}/cameras`).then(setCams).catch((e) => setError(String(e.message)));
  }, [base, version]);

  const hasSplat = !!detail.files["export/splat.spz"] && cams?.frame === "canonical";
  return (
    <div className="cameras">
      <div className="cameras-top">
        <div className="ring-view">
          {cams ? (
            <RingView data={cams} hover={hover} splatUrl={hasSplat ? fileUrl(detail.job, `attempts/${detail.attempt.seed}/export/splat.spz`, detail.attempt.seconds ?? 0) : null} />
          ) : (
            <div className="viewer-overlay">{error ? "No camera poses yet." : "Loading cameras…"}</div>
          )}
        </div>
        <aside className="ring-side">
          <h3>Cameras</h3>
          {cams && (
            <>
              <p className="small muted">
                {cams.cameras.filter((c) => !c.hero).length} frames and the hero (outlined) placed by COLMAP, colored by
                how well each view fits the 3D points{cams.frame === "orbit" ? ". Orbit radius = 1." : ", in meters."}
              </p>
              <ul className="key">
                <li><i className="good" />≤ 1.0 px</li>
                <li><i className="warn" />1.0 to 1.5 px</li>
                <li><i className="bad" />&gt; 1.5 px</li>
                <li><i className="hero" />hero image</li>
              </ul>
            </>
          )}
          {frames && <FrameSummary data={frames} />}
        </aside>
      </div>
      {frames && frames.frames.some((f) => f.azimuth !== null) && (
        <section>
          <h3>Degrees around the subject, by frame</h3>
          <AzimuthPlot frames={frames.frames} need={Number(detail.gate?.checks.find((c) => c.metric === "azimuth_coverage")?.threshold ?? 330)} onHover={setHover} />
        </section>
      )}
      {frames && frames.frames.length > 0 && (
        <section>
          <h3>Frames</h3>
          <FrameStrip data={frames} sheet={`${base}/frames.jpg?v=${frames.frames.length}`} hover={hover} onHover={setHover} />
        </section>
      )}
    </div>
  );
}

function FrameSummary({ data }: { data: FramesData }) {
  const n = data.frames.filter((f) => !f.generated).length;
  const kept = data.frames.filter((f) => f.kept).length;
  const posed = data.frames.filter((f) => f.posed).length;
  return (
    <dl className="facts small-facts">
      <dt>Extracted</dt>
      <dd>{n} frames</dd>
      <dt>Kept</dt>
      <dd>
        {kept}
        <small className="muted">
          {data.static_start || data.static_end ? `${data.static_start ?? 0} still frames trimmed at the start, ${data.static_end ?? 0} at the end` : ""}
        </small>
      </dd>
      <dt>Placed</dt>
      <dd>
        {posed} of {kept}
        {kept > posed && <small className="muted">{kept - posed} kept frames got no camera (outlined red below)</small>}
      </dd>
      <dt>Hero</dt>
      <dd>{data.hero.posed ? `placed${data.hero.error !== null ? `, ${data.hero.error.toFixed(2)} px` : ""}` : "not placed"}</dd>
      {data.filled && data.filled.length > 0 && (
        <>
          <dt>Gap filled</dt>
          <dd>
            {data.filled.join("; ")}
            <small className="muted">regenerated by the video model where the camera path jumped (gold bar below)</small>
          </dd>
        </>
      )}
    </dl>
  );
}

function RingView({ data, hover, splatUrl }: { data: CamerasData; hover: string | null; splatUrl: string | null }) {
  const host = useRef<HTMLDivElement>(null);
  const highlight = useRef<(name: string | null) => void>(() => {});
  const splatRef = useRef<SplatMesh | null>(null);
  const pointsRef = useRef<THREE.Points | null>(null);
  const [showSplat, setShowSplat] = useState(true);
  const [showPoints, setShowPoints] = useState(!splatUrl);

  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121416);
    const camera = new THREE.PerspectiveCamera(40, 1, 0.01, 500);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const ring = data.cameras.filter((c) => !c.hero);
    const center = new THREE.Vector3();
    ring.forEach((c) => center.add(new THREE.Vector3(...c.position)));
    center.divideScalar(Math.max(1, ring.length));
    const radius = ring.reduce((s, c) => s + Math.hypot(c.position[0] - center.x, c.position[2] - center.z), 0) / Math.max(1, ring.length);
    const target = data.frame === "canonical" ? new THREE.Vector3(0, center.y * 0.6, 0) : new THREE.Vector3(0, 0, 0);
    controls.target.copy(target);
    camera.position.set(radius * 1.1, radius * 1.3, radius * 1.6);
    camera.near = radius / 200;
    camera.far = radius * 50;
    camera.updateProjectionMatrix();

    const grid = new THREE.GridHelper(radius * 2.6, 26, 0x3a4048, 0x24282d);
    grid.position.y = data.frame === "canonical" ? 0 : -radius * 0.4;
    scene.add(grid);

    // Frustums: apex at the camera, a rectangle at depth d.
    const d = radius * 0.12;
    const frustums = new Map<string, THREE.LineSegments>();
    for (const c of data.cameras) {
      const p = new THREE.Vector3(...c.position);
      const f = new THREE.Vector3(...c.forward), r = new THREE.Vector3(...c.right), dn = new THREE.Vector3(...c.down);
      const depth = c.hero ? d * 1.6 : d;
      const hw = (c.width / 2 / c.fx) * depth, hh = (c.height / 2 / c.fy) * depth;
      const corner = (sx: number, sy: number) => p.clone().addScaledVector(f, depth).addScaledVector(r, sx * hw).addScaledVector(dn, sy * hh);
      const [a, b, cc, e] = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
      const pts = [p, a, p, b, p, cc, p, e, a, b, b, cc, cc, e, e, a];
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const color = c.hero ? 0xffffff : TONE_HEX[errorTone(c.error)];
      const line = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: c.hero ? 1 : 0.85 }));
      line.userData.color = color;
      frustums.set(c.name, line);
      scene.add(line);
    }
    // The camera path, in time order.
    const path = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(ring.sort((x, y) => (x.name < y.name ? -1 : 1)).map((c) => new THREE.Vector3(...c.position))),
      new THREE.LineBasicMaterial({ color: 0x4a525b }),
    );
    scene.add(path);

    let points: THREE.Points | null = null;
    if (data.points.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(data.points, 3));
      geo.setAttribute("color", new THREE.Float32BufferAttribute(data.colors.map((v) => v / 255), 3));
      points = new THREE.Points(geo, new THREE.PointsMaterial({ size: radius * 0.006, vertexColors: true }));
      points.visible = showPoints;
      scene.add(points);
    }

    let mesh: SplatMesh | null = null;
    if (splatUrl) {
      scene.add(new SparkRenderer({ renderer }));
      mesh = new SplatMesh({ url: splatUrl });
      mesh.quaternion.set(0, 0, 1, 0); // file frame -> canonical frame (docs/FINDINGS.md, "Export and viewing")
      mesh.visible = showSplat;
      scene.add(mesh);
    }
    splatRef.current = mesh;
    pointsRef.current = points;

    let highlighted: THREE.LineSegments | null = null;
    highlight.current = (name) => {
      if (highlighted) (highlighted.material as THREE.LineBasicMaterial).color.setHex(highlighted.userData.color);
      highlighted = name ? frustums.get(`frames/${name}`) ?? null : null;
      if (highlighted) (highlighted.material as THREE.LineBasicMaterial).color.setHex(0x6aa8ff);
    };

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });
    return () => {
      renderer.setAnimationLoop(null);
      ro.disconnect();
      controls.dispose();
      scene.traverse((o) => {
        if (o instanceof THREE.LineSegments || o instanceof THREE.Line || o instanceof THREE.Points) {
          o.geometry.dispose();
          (o.material as THREE.Material).dispose();
        }
      });
      (mesh as unknown as { dispose?: () => void } | null)?.dispose?.();
      renderer.dispose();
      renderer.domElement.remove();
    };
    // showSplat / showPoints only toggle visibility (below); they must not rebuild the scene.
  }, [data, splatUrl]);

  useEffect(() => highlight.current(hover), [hover]);
  useEffect(() => {
    if (splatRef.current) splatRef.current.visible = showSplat;
    if (pointsRef.current) pointsRef.current.visible = showPoints;
  }, [showSplat, showPoints]);

  return (
    <div className="viewer" ref={host}>
      <div className="viewer-toggles">
        {splatUrl && (
          <label><input type="checkbox" checked={showSplat} onChange={(e) => setShowSplat(e.target.checked)} /> Splat</label>
        )}
        {data.points.length > 0 && (
          <label><input type="checkbox" checked={showPoints} onChange={(e) => setShowPoints(e.target.checked)} /> Sparse points</label>
        )}
      </div>
    </div>
  );
}

/** How far the camera has gone around the subject since the first posed frame, by frame number.
 *  Azimuths wrap at ±180°; unwrapped, signed so the orbit's own direction counts up. */
function AzimuthPlot({ frames, need, onHover }: { frames: FrameInfo[]; need: number; onHover: (name: string | null) => void }) {
  const W = 1000, H = 220, L = 44, R = 12, T = 12, B = 28;
  const pts = useMemo(() => {
    const raw = frames.map((f, i) => ({ i, f })).filter((p) => p.f.azimuth !== null) as { i: number; f: FrameInfo & { azimuth: number } }[];
    let total = 0;
    const out = raw.map((p, k) => {
      if (k > 0) total += ((((p.f.azimuth - raw[k - 1].f.azimuth) % 360) + 540) % 360) - 180;
      return { ...p, deg: total };
    });
    const sign = total < 0 ? -1 : 1;
    return out.map((p) => ({ ...p, deg: sign * p.deg }));
  }, [frames]);
  const [hi, setHi] = useState<number | null>(null);
  const n = frames.length;
  const lo = Math.min(0, ...pts.map((p) => p.deg)), top = Math.max(360, ...pts.map((p) => p.deg));
  const y0 = Math.floor(lo / 90) * 90, y1 = Math.ceil(top / 90) * 90;
  const x = (i: number) => L + ((W - L - R) * i) / Math.max(1, n - 1);
  const y = (a: number) => T + ((H - T - B) * (y1 - a)) / Math.max(1, y1 - y0);
  const ticks: number[] = [];
  for (let a = y0; a <= y1; a += 90) ticks.push(a);
  const xticks = Array.from({ length: Math.floor((n - 1) / 20) + 1 }, (_, k) => k * 20);
  const hp = hi !== null ? pts[hi] : null;

  function move(e: React.MouseEvent<SVGSVGElement>) {
    const box = e.currentTarget.getBoundingClientRect();
    const fx = ((e.clientX - box.left) / box.width) * W;
    let best = 0, bd = Infinity;
    pts.forEach((p, k) => {
      const dd = Math.abs(x(p.i) - fx);
      if (dd < bd) { bd = dd; best = k; }
    });
    setHi(best);
    onHover(pts[best]?.f.name ?? null);
  }

  return (
    <div className="plot">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onMouseMove={move}
        onMouseLeave={() => { setHi(null); onHover(null); }} role="img" aria-label="Camera azimuth by frame">
        {ticks.map((a) => (
          <g key={a}>
            <line x1={L} x2={W - R} y1={y(a)} y2={y(a)} className="grid" />
            <text x={L - 8} y={y(a) + 4} className="tick" textAnchor="end">{a}°</text>
          </g>
        ))}
        {xticks.map((i) => (
          <text key={i} x={x(i)} y={H - 8} className="tick" textAnchor="middle">{i}</text>
        ))}
        <line x1={L} x2={W - R} y1={y(need)} y2={y(need)} className="threshold" vectorEffect="non-scaling-stroke" />
        <text x={W - R - 4} y={y(need) - 5} className="tick" textAnchor="end">gate needs {need}°</text>
        <polyline fill="none" className="series" points={pts.map((p) => `${x(p.i)},${y(p.deg)}`).join(" ")} vectorEffect="non-scaling-stroke" />
        {hp && (
          <>
            <line x1={x(hp.i)} x2={x(hp.i)} y1={T} y2={H - B} className="crosshair" vectorEffect="non-scaling-stroke" />
            <circle cx={x(hp.i)} cy={y(hp.deg)} r={4.5} className="dot" vectorEffect="non-scaling-stroke" />
          </>
        )}
      </svg>
      {hp && (
        <div className="tooltip" style={{ left: `${(x(hp.i) / W) * 100}%` }}>
          <strong>Frame {hp.i}</strong>
          <span>{hp.deg.toFixed(0)}° around</span>
          {hp.f.error !== null && <span className="muted">{hp.f.error.toFixed(2)} px</span>}
        </div>
      )}
    </div>
  );
}

function FrameStrip({ data, sheet, hover, onHover }: { data: FramesData; sheet: string; hover: string | null; onHover: (n: string | null) => void }) {
  const h = 64;
  const w = Math.round((h * 3) / 4);
  const n = data.frames.length;
  const strip = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!hover || !strip.current) return;
    const i = data.frames.findIndex((f) => f.name === hover);
    const el = strip.current;
    const left = i * (w + 2);
    if (left < el.scrollLeft || left > el.scrollLeft + el.clientWidth - w) el.scrollTo({ left: left - el.clientWidth / 2, behavior: "smooth" });
  }, [hover, data.frames, w]);
  return (
    <>
      <div className="strip" ref={strip}>
        {data.frames.map((f, i) => {
          const cls = `${!f.kept ? "dropped" : !f.posed ? "unposed" : ""} ${f.generated ? "generated" : ""}`;
          const why = f.replaced ? " (replaced by a gap fill)" : !f.kept ? " (dropped: a near-duplicate or still frame)" : !f.posed ? " (kept, but got no camera)" : "";
          const title = `Frame ${i}${f.generated ? " (generated to fill a gap)" : ""}${why}${f.azimuth !== null ? `, ${f.azimuth.toFixed(0)}°` : ""}${f.error !== null ? `, ${f.error.toFixed(2)} px` : ""}`;
          return (
            <div key={f.name} className={`frame ${cls} ${hover === f.name ? "hover" : ""}`} title={title}
              onMouseEnter={() => f.posed && onHover(f.name)} onMouseLeave={() => onHover(null)}
              style={{ width: w, height: h, backgroundImage: `url(${sheet})`, backgroundSize: `${n * w}px ${h}px`, backgroundPosition: `-${i * w}px 0` }}>
              {f.posed && <i className={errorTone(f.error)} />}
            </div>
          );
        })}
      </div>
      <p className="muted small">Grayed: dropped before posing (still, near-duplicate, or replaced by a gap fill). Red outline: kept but got no camera. Gold bar on top: generated to fill a gap. The bar under each frame shows its fit.</p>
    </>
  );
}
