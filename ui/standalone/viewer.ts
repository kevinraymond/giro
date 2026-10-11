// giro's all-in-one viewer: one HTML file with the splat inside, nothing fetched.
// Orbit/fly around a splat, double-click to re-center, and measure distances by clicking
// two points on it (picked at the rendered depth under the click). Built by vite.standalone.config.ts
// into one script that src/giro/viewer_html.py inlines next to the base64 SPZ.
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";

interface Meta {
  title?: string;
  units_per_meter?: number;   // the splat's units per meter (1 = meters)
  up?: "y" | "z";             // the file's up axis after the PLY -> viewer flip (giro writes y)
  floor?: number | null;      // height of the floor grid in meters, null for none
  home?: { position: [number, number, number]; target: [number, number, number] };  // first view, meters
  notes?: string;
}

const meta: Meta = JSON.parse(document.getElementById("giro-meta")!.textContent || "{}");
const b64 = document.getElementById("giro-splat")!.textContent!.replace(/\s+/g, "");
const upm = meta.units_per_meter ?? 1;

// ---- DOM
const $ = (id: string) => document.getElementById(id)!;
const host = $("view");
const status = $("status");
const list = $("measures");

// ---- scene
const renderer = new THREE.WebGLRenderer({ antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
host.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14171c);
scene.add(new SparkRenderer({ renderer }));
const camera = new THREE.PerspectiveCamera(50, 1, 0.02, 500);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.screenSpacePanning = true;
const overlay = new THREE.Group();  // markers and lines, drawn on top of the splat
scene.add(overlay);

// Overlay materials are transparent with no depth test so they sort after the splat pass and
// draw on top of it: as opaque objects they would be painted over by the fuzzy splat surface.
const markerGeo = new THREE.SphereGeometry(1, 16, 12);
const markerMat = new THREE.MeshBasicMaterial({ color: 0xffb020, transparent: true, depthTest: false, depthWrite: false });
const outlineMat = new THREE.MeshBasicMaterial({ color: 0x101214, transparent: true, depthTest: false, depthWrite: false });
const lineMat = new LineMaterial({ color: 0xffb020, linewidth: 3, transparent: true, depthTest: false, depthWrite: false });
const lineOutlineMat = new LineMaterial({ color: 0x101214, linewidth: 7, transparent: true, depthTest: false, depthWrite: false });
const MARKER_PX = 6;  // marker radius on screen

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  for (const m of [lineMat, lineOutlineMat]) m.resolution.set(w, h);
}
new ResizeObserver(resize).observe(host);
resize();

// ---- load the embedded splat
function decode(s: string): Uint8Array {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
status.textContent = "Loading splat…";
const mesh = new SplatMesh({ fileBytes: decode(b64), fileType: "spz" as never });
if (meta.up === "z") mesh.quaternion.setFromEuler(new THREE.Euler(-Math.PI / 2, 0, Math.PI));
else mesh.quaternion.set(0, 0, 1, 0);  // PLY convention (y down) -> y up
mesh.scale.setScalar(1 / upm);
let grid: THREE.GridHelper | null = null;
let home = { position: new THREE.Vector3(), target: new THREE.Vector3() };

mesh.initialized.then(() => {
  scene.add(mesh);
  const { lo, hi, count } = bounds(mesh);
  const center = lo.clone().add(hi).multiplyScalar(0.5);
  const radius = Math.max(0.5 * lo.distanceTo(hi), 0.5);
  if (meta.floor !== null) {
    const span = Math.ceil(radius * 2);
    grid = new THREE.GridHelper(span, span, 0x3a4048, 0x24282d);  // 1 m cells
    grid.position.set(center.x, meta.floor ?? lo.y, center.z);
    scene.add(grid);
  }
  const dist = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 0.9;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0, 0.35, 1).normalize().multiplyScalar(dist));
  camera.near = Math.max(0.02, dist / 500);
  camera.far = dist * 40;
  camera.updateProjectionMatrix();
  controls.maxDistance = dist * 6;
  if (meta.home) {
    camera.position.fromArray(meta.home.position);
    controls.target.fromArray(meta.home.target);
  }
  home = { position: camera.position.clone(), target: controls.target.clone() };
  status.textContent = `${count.toLocaleString()} splats`;
}).catch((e: unknown) => { status.textContent = `Could not load the splat: ${String(e)}`; });

/** Robust bounds (2nd to 98th percentile per axis) of the splat centers in world units. */
function bounds(m: SplatMesh) {
  m.updateMatrixWorld();
  const xs: number[] = [], ys: number[] = [], zs: number[] = [];
  const v = new THREE.Vector3();
  let count = 0;
  m.packedSplats?.forEachSplat((_i, c, _s, _q, opacity) => {
    count++;
    if (opacity < 0.3 || count % 7) return;  // a sample is plenty
    v.copy(c).applyMatrix4(m.matrixWorld);
    xs.push(v.x); ys.push(v.y); zs.push(v.z);
  });
  const pct = (a: number[], p: number) => { a.sort((x, y) => x - y); return a[Math.floor((a.length - 1) * p)] ?? 0; };
  return {
    lo: new THREE.Vector3(pct(xs, 0.02), pct(ys, 0.02), pct(zs, 0.02)),
    hi: new THREE.Vector3(pct(xs, 0.98), pct(ys, 0.98), pct(zs, 0.98)),
    count,
  };
}

// ---- picking
const raycaster = new THREE.Raycaster();
function pick(ev: MouseEvent): THREE.Vector3 | null {
  const r = renderer.domElement.getBoundingClientRect();
  return pickNdc(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
}
/** The point on the splat under normalized device coordinates (x, y in -1..1).
 * Spark's raycast crosses only a handful of Gaussians per ray and a stray one in front of
 * the surface wins, so this picks the way the renderer sees it: every splat near the ray
 * is composited front to back and the pick is the depth window that contributes most of
 * what the eye sees there. One pass over the splats, a few tens of ms per click. */
function pickNdc(x: number, y: number): THREE.Vector3 | null {
  const prof = depthProfile(x, y);
  if (!prof) return null;
  const { ray, unit, windows } = prof;
  return windows.length ? ray.at(windows[0].t * unit, new THREE.Vector3()) : null;
}
/** Depth windows along the ray through (x, y), strongest contribution first. */
function depthProfile(x: number, y: number) {
  const splats = mesh.packedSplats;
  if (!splats) return null;
  raycaster.setFromCamera(new THREE.Vector2(x, y), camera);
  const ray = raycaster.ray;
  mesh.updateMatrixWorld();
  const toLocal = mesh.matrixWorld.clone().invert();
  const unit = mesh.scale.x;  // world meters per splat unit
  const o = ray.origin.clone().applyMatrix4(toLocal);
  const d = ray.direction.clone().transformDirection(toLocal);
  const k = (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / renderer.domElement.clientHeight;  // units per pixel per unit of depth
  const near = camera.near / unit;
  const cand: { t: number; a: number }[] = [];
  const v = new THREE.Vector3();
  splats.forEachSplat((_i, c, s, _q, opacity) => {
    if (opacity < 0.05) return;
    v.copy(c).sub(o);
    const t = v.dot(d);
    if (t < near) return;
    const click = 2 * k * t;  // a 2 px click radius at this depth
    const big = Math.max(s.x, s.y, s.z), mid = s.x + s.y + s.z - big - Math.min(s.x, s.y, s.z);
    const sigma = Math.max(Math.sqrt(big * mid), 0.5 * k * t);  // the splat's face, at least half a pixel
    const reach = 3 * sigma + click;
    const e2 = v.lengthSq() - t * t;
    if (e2 > reach * reach) return;
    const e = Math.max(0, Math.sqrt(e2) - click);
    cand.push({ t, a: opacity * Math.exp(-0.5 * (e / sigma) ** 2) });
  });
  cand.sort((p, q) => p.t - q.t);
  let through = 1;
  const seen: number[] = [];  // what each splat contributes to the pixel, front to back
  for (const c of cand) { seen.push(c.a * through); through *= 1 - c.a; }
  // The surface is the depth window that contributes most to the pixel: haze in front is
  // many faint contributions spread over depth, a surface is a few strong ones close together.
  const windows: { t: number; w: number }[] = [];
  for (let i = 0, j = 0, sum = 0, wsum = 0; i < cand.length; i++) {
    const width = Math.max(0.05, 0.02 * cand[i].t);
    while (j < cand.length && cand[j].t - cand[i].t <= width) { sum += seen[j]; wsum += seen[j] * cand[j].t; j++; }
    if (sum > 0.05) windows.push({ t: wsum / sum, w: sum });
    sum -= seen[i]; wsum -= seen[i] * cand[i].t;
  }
  windows.sort((p, q) => q.w - p.w);
  // keep the strongest window of each distinct depth
  const distinct: { t: number; w: number }[] = [];
  for (const w of windows) if (!distinct.some((o) => Math.abs(o.t - w.t) < Math.max(0.1, 0.04 * w.t))) distinct.push(w);
  return { ray, unit, windows: distinct, visible: 1 - through };
}

// ---- measuring
interface Measure { a: THREE.Vector3; b: THREE.Vector3; objects: THREE.Object3D[]; label: HTMLElement }
const measures: Measure[] = [];
let measuring = false;
let pending: { point: THREE.Vector3; marker: THREE.Group } | null = null;
let feet = false;

/** A dot with a dark rim at p; the frame loop keeps it MARKER_PX on screen. */
function marker(p: THREE.Vector3) {
  const g = new THREE.Group();
  const rim = new THREE.Mesh(markerGeo, outlineMat);
  rim.scale.setScalar(1.5);
  rim.renderOrder = 11;
  const dot = new THREE.Mesh(markerGeo, markerMat);
  dot.renderOrder = 12;
  g.add(rim, dot);
  g.position.copy(p);
  overlay.add(g);
  return g;
}
function segment(a: THREE.Vector3, b: THREE.Vector3): THREE.Object3D[] {
  const geo = new LineGeometry().setPositions([a.x, a.y, a.z, b.x, b.y, b.z]);
  const outline = new Line2(geo, lineOutlineMat);
  outline.renderOrder = 10;
  const line = new Line2(geo, lineMat);
  line.renderOrder = 11;
  overlay.add(outline, line);
  return [outline, line];
}
function fmt(meters: number) {
  if (!feet) return meters >= 10 ? `${meters.toFixed(2)} m` : `${(meters * 100).toFixed(1)} cm`;
  const inches = meters / 0.0254;
  return inches >= 24 ? `${Math.floor(inches / 12)}' ${(inches % 12).toFixed(1)}"` : `${inches.toFixed(1)}"`;
}
function addMeasure(a: THREE.Vector3, b: THREE.Vector3) {
  const label = document.createElement("div");
  label.className = "label";
  $("labels").appendChild(label);
  const m: Measure = { a, b, objects: [...segment(a, b), marker(a), marker(b)], label };
  measures.push(m);
  renderList();
}
function clearMeasures() {
  for (const m of measures) { m.objects.forEach((o) => overlay.remove(o)); m.label.remove(); }
  measures.length = 0;
  cancelPending();
  renderList();
}
function cancelPending() {
  if (pending) overlay.remove(pending.marker);
  pending = null;
}
function renderList() {
  list.innerHTML = "";
  measures.forEach((m, i) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span>${i + 1}</span><b>${fmt(m.a.distanceTo(m.b))}</b><span class="dims">Δx ${fmt(Math.abs(m.b.x - m.a.x))} · Δh ${fmt(Math.abs(m.b.y - m.a.y))} · Δz ${fmt(Math.abs(m.b.z - m.a.z))}</span>`;
    list.appendChild(row);
  });
  $("clear").toggleAttribute("disabled", measures.length === 0);
  $("panel").classList.toggle("hidden", measures.length === 0);
}
function setMeasuring(on: boolean) {
  measuring = on;
  $("measure").classList.toggle("on", on);
  host.style.cursor = on ? "crosshair" : "";
  if (!on) cancelPending();
}

// ---- input
let downAt: [number, number] | null = null;
renderer.domElement.addEventListener("pointerdown", (e) => { downAt = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener("click", (e) => {
  if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;  // a drag, not a click
  if (!measuring) return;
  const p = pick(e);
  if (!p) return;
  if (!pending) pending = { point: p, marker: marker(p) };
  else { overlay.remove(pending.marker); addMeasure(pending.point, p); pending = null; }
});
renderer.domElement.addEventListener("dblclick", (e) => {
  const p = pick(e);
  if (!p) return;
  const offset = camera.position.clone().sub(controls.target);
  controls.target.copy(p);
  camera.position.copy(p).add(offset);
});
const keys = new Set<string>();
window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  keys.add(e.key.toLowerCase());
  if (e.key === "m") setMeasuring(!measuring);
  if (e.key === "Escape") { cancelPending(); setMeasuring(false); }
  if (e.key === "r") resetView();
  if (e.key === "g" && grid) grid.visible = !grid.visible;
  if (e.key === "u") toggleUnits();
  if (e.key === "?" || e.key === "h") $("help").classList.toggle("hidden");
});
window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));
function fly(dt: number) {
  const speed = (keys.has("shift") ? 3 : 1) * dt * Math.max(0.5, camera.position.distanceTo(controls.target));
  const fwd = controls.target.clone().sub(camera.position).normalize();
  const right = fwd.clone().cross(camera.up).normalize();
  const move = new THREE.Vector3();
  if (keys.has("w")) move.add(fwd);
  if (keys.has("s")) move.sub(fwd);
  if (keys.has("d")) move.add(right);
  if (keys.has("a")) move.sub(right);
  if (keys.has("e")) move.y += 1;
  if (keys.has("q")) move.y -= 1;
  if (move.lengthSq() === 0) return;
  move.normalize().multiplyScalar(speed);
  camera.position.add(move);
  controls.target.add(move);
}
function resetView() {
  camera.position.copy(home.position);
  controls.target.copy(home.target);
}
function toggleUnits() {
  feet = !feet;
  $("units").textContent = feet ? "ft/in" : "m";
  renderList();
}
$("measure").onclick = () => setMeasuring(!measuring);
$("clear").onclick = clearMeasures;
$("reset").onclick = resetView;
$("units").onclick = toggleUnits;
$("helpbtn").onclick = () => $("help").classList.toggle("hidden");
$("help").onclick = () => $("help").classList.add("hidden");
if (meta.title) { document.title = meta.title; $("title").textContent = meta.title; }
if (meta.notes) $("notes").textContent = meta.notes;

// Scripting hook for tests and tools: window.giro.pick(x, y) picks at normalized device
// coordinates, window.giro.measure(a, b) adds a measurement between two points in meters.
(window as unknown as { giro: unknown }).giro = {
  camera, controls, mesh, measures, loaded: mesh.initialized,
  pick: pickNdc,
  profile: (x: number, y: number) => { const p = depthProfile(x, y); return p && { visible: p.visible, windows: p.windows.slice(0, 6).map((w) => ({ depth_m: w.t * p.unit, share: w.w })) }; },
  measure: (a: [number, number, number], b: [number, number, number]) =>
    addMeasure(new THREE.Vector3().fromArray(a), new THREE.Vector3().fromArray(b)),
  setUnits: (f: boolean) => { if (feet !== f) toggleUnits(); },
};

// ---- frame loop: labels follow their lines, markers keep a constant screen size
const clock = new THREE.Clock();
const tmp = new THREE.Vector3();
renderer.setAnimationLoop(() => {
  fly(Math.min(clock.getDelta(), 0.1));
  controls.update();
  const h = renderer.domElement.clientHeight;
  const px = (p: THREE.Vector3) => (2 * p.distanceTo(camera.position) * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) / h;
  for (const o of overlay.children) if (o instanceof THREE.Group) o.scale.setScalar(MARKER_PX * px(o.position));
  for (const m of measures) {
    tmp.copy(m.a).add(m.b).multiplyScalar(0.5).project(camera);
    const r = renderer.domElement.getBoundingClientRect();
    m.label.style.display = tmp.z < 1 ? "" : "none";
    m.label.style.left = `${((tmp.x + 1) / 2) * r.width}px`;
    m.label.style.top = `${((1 - tmp.y) / 2) * r.height}px`;
    m.label.textContent = fmt(m.a.distanceTo(m.b));
  }
  renderer.render(scene, camera);
});
