// Attempt detail, crop: the trained splat with the auto-crop previewed live.
// tau hides Gaussians that too few views see inside the subject mask (votes from the crop
// stage); the box trims what is left, in meters in the canonical frame. Apply reruns crop,
// canonicalize and export with the new values; everything before them is skipped.

import type { SplatMesh } from "@sparkjsdev/spark";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { api, fileUrl, type AttemptDetail, type StageInfo } from "./api";
import { SplatViewer } from "./SplatViewer";
import { putJob } from "./store";
import { ActionButton, Progress } from "./ui";

type Box = [number, number, number, number, number, number]; // min xyz, max xyz
const AXES = ["x", "y", "z"] as const;
const AXIS_NAME = { x: "Left–right", y: "Height", z: "Front–back" };

interface Loaded {
  mesh: SplatMesh;
  alpha: Uint32Array; // original first word (RGBA) of each splat
  centers: Float32Array; // viewer frame
  bounds: Box;
}

export function CropTab({ detail, stages }: { detail: AttemptDetail; stages: StageInfo[] }) {
  const { job, attempt } = detail;
  const seed = attempt.seed;
  const defaults = (stage: string) => stages.find((s) => s.name === stage)?.defaults ?? {};
  const applied = {
    tau: Number(attempt.params.crop?.tau ?? defaults("crop").tau ?? 0.8),
    box: (attempt.params.export?.box as Box | null | undefined) ?? null,
    height: Number(detail.metrics.canonicalize?.height_m ?? defaults("canonicalize").height_m ?? 1.7),
  };
  const [matrix, setMatrix] = useState<number[][] | null>(null);
  const [votes, setVotes] = useState<Float32Array | null | "missing">(null);
  const [tau, setTau] = useState(applied.tau);
  const [hull, setHull] = useState(true);
  const [boxOn, setBoxOn] = useState(applied.box !== null);
  const [box, setBox] = useState<Box | null>(applied.box);
  const [height, setHeight] = useState(String(applied.height));
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [visible, setVisible] = useState<number | null>(null);
  const helper = useRef<THREE.Box3Helper | null>(null);
  const version = attempt.seconds ?? 0;

  useEffect(() => {
    fetch(`/api/jobs/${encodeURIComponent(job)}/attempts/${seed}/cameras`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setMatrix(d?.frame === "canonical" ? d.world_to_viewer : null));
    fetch(fileUrl(job, `attempts/${seed}/crop/votes.f32`, version))
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject()))
      .then((b) => setVotes(new Float32Array(b)))
      .catch(() => setVotes("missing"));
  }, [job, seed, version]);

  // Hide what the current settings would crop away, by zeroing each splat's opacity byte.
  useEffect(() => {
    if (!loaded) return;
    const { mesh, alpha, centers } = loaded;
    const packed = mesh.packedSplats?.packedArray;
    if (!packed) return;
    const v = votes instanceof Float32Array && votes.length === alpha.length ? votes : null;
    const b = boxOn && box ? box : null;
    let count = 0;
    for (let i = 0; i < alpha.length; i++) {
      let show = !hull || !v || v[i] >= tau;
      if (show && b) {
        const x = centers[3 * i], y = centers[3 * i + 1], z = centers[3 * i + 2];
        show = x >= b[0] && y >= b[1] && z >= b[2] && x <= b[3] && y <= b[4] && z <= b[5];
      }
      packed[4 * i] = show ? alpha[i] : alpha[i] & 0x00ffffff;
      if (show) count++;
    }
    mesh.packedSplats!.needsUpdate = true;
    mesh.updateVersion(); // the mesh re-reads its splats only when its version changes
    setVisible(count);
  }, [loaded, votes, tau, hull, boxOn, box]);

  useEffect(() => {
    if (!helper.current) return;
    helper.current.visible = boxOn && !!box;
    if (box) helper.current.box.set(new THREE.Vector3(box[0], box[1], box[2]), new THREE.Vector3(box[3], box[4], box[5]));
  }, [box, boxOn, loaded]);

  function onLoad(mesh: SplatMesh, scene: THREE.Scene) {
    const packed = mesh.packedSplats!;
    const n = packed.numSplats;
    const alpha = new Uint32Array(n);
    const centers = new Float32Array(3 * n);
    mesh.updateMatrix();
    const m = mesh.matrix;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    const v = new THREE.Vector3();
    const kept = votes instanceof Float32Array && votes.length === n ? votes : null;
    packed.forEachSplat((i, c) => {
      alpha[i] = packed.packedArray![4 * i];
      v.copy(c).applyMatrix4(m);
      centers[3 * i] = v.x; centers[3 * i + 1] = v.y; centers[3 * i + 2] = v.z;
      if (!kept || kept[i] >= applied.tau) {
        lo[0] = Math.min(lo[0], v.x); lo[1] = Math.min(lo[1], v.y); lo[2] = Math.min(lo[2], v.z);
        hi[0] = Math.max(hi[0], v.x); hi[1] = Math.max(hi[1], v.y); hi[2] = Math.max(hi[2], v.z);
      }
    });
    const pad = 0.02;
    const bounds: Box = [lo[0] - pad, Math.max(lo[1], 0) - pad, lo[2] - pad, hi[0] + pad, hi[1] + pad, hi[2] + pad];
    if (helper.current?.parent !== scene) {
      helper.current = new THREE.Box3Helper(new THREE.Box3(), 0xe8b04a);
      scene.add(helper.current);
    }
    setBox((b) => b ?? bounds);
    setLoaded({ mesh, alpha, centers, bounds });
  }

  if (!detail.files["export/splat.spz"] && attempt.status !== "passed") {
    return <p className="muted">The crop can be adjusted once this seed has been trained and exported.</p>;
  }
  if (votes === "missing") {
    return (
      <div className="notice">
        <p>This seed was cropped before giro kept the per-Gaussian votes the live preview needs.</p>
        <ActionButton kind="primary" onClick={() => api.setParams(job, seed, { crop: { tau: applied.tau } }).then(putJob)}>
          Prepare the crop editor
        </ActionButton>
        <p className="muted small">Reruns the crop, upright and export steps (about 15 s).</p>
      </div>
    );
  }
  const exp = detail.metrics.export ?? {};
  const budget = Number(exp.vr_budget ?? 500000);
  const sentBox = boxOn && box ? box.map((x) => Math.round(x * 1000) / 1000) : null; // mm, as applied
  const changed = tau !== applied.tau || Number(height) !== applied.height ||
    JSON.stringify(sentBox) !== JSON.stringify(applied.box);
  const busy = attempt.status === "running" || attempt.status === "queued";
  const lim = loaded?.bounds;

  return (
    <div className="crop">
      <div className="ring-view">
        {matrix && votes ? (
          <SplatViewer url={fileUrl(job, `attempts/${seed}/train/final.ply`)} matrix={matrix} onLoad={onLoad} />
        ) : (
          <div className="viewer-overlay">Loading…</div>
        )}
      </div>
      <aside className="crop-side">
        <label className="switch">
          <input type="checkbox" checked={hull} onChange={(e) => setHull(e.target.checked)} />
          <span>Auto-crop to the subject</span>
        </label>
        <div className={`slider ${hull ? "" : "off"}`}>
          <div className="slider-head">
            <span>Strictness</span>
            <span className="num">{Math.round(tau * 100)}% of views</span>
          </div>
          <input type="range" min={0.5} max={1} step={0.05} value={tau} disabled={!hull}
            onChange={(e) => setTau(Number(e.target.value))} aria-label="Strictness (tau)" />
          <small className="muted">
            Keep a Gaussian if at least this share of the views that see it place it inside the subject mask. Higher trims
            more halo and floor, lower keeps more of thin parts.
          </small>
        </div>

        <label className="switch">
          <input type="checkbox" checked={boxOn} onChange={(e) => setBoxOn(e.target.checked)} />
          <span>Trim with a box</span>
        </label>
        {boxOn && box && lim && (
          <div className="box-edit">
            {AXES.map((a, k) => {
              const lo = lim[k] - 0.2, hi = lim[k + 3] + 0.2;
              return (
                <div key={a} className="box-axis">
                  <span className="small">{AXIS_NAME[a]}</span>
                  <input type="range" min={lo} max={hi} step={0.01} value={box[k]} aria-label={`${a} min`}
                    onChange={(e) => {
                      const b = [...box] as Box;
                      b[k] = Math.min(Number(e.target.value), b[k + 3] - 0.01);
                      setBox(b);
                    }} />
                  <input type="range" min={lo} max={hi} step={0.01} value={box[k + 3]} aria-label={`${a} max`}
                    onChange={(e) => {
                      const b = [...box] as Box;
                      b[k + 3] = Math.max(Number(e.target.value), b[k] + 0.01);
                      setBox(b);
                    }} />
                  <span className="num small muted">{box[k].toFixed(2)} to {box[k + 3].toFixed(2)} m</span>
                </div>
              );
            })}
            <button className="btn small ghost" onClick={() => setBox(lim)}>Fit to the subject</button>
          </div>
        )}

        <label className="field">
          <span>Subject height</span>
          <div className="suffixed">
            <input type="number" min="0.05" step="0.05" value={height} onChange={(e) => setHeight(e.target.value)} />
            <span>m</span>
          </div>
        </label>

        {visible !== null && (
          <div className="budget">
            <div className="slider-head">
              <span>Gaussians kept</span>
              <span className="num">{visible.toLocaleString()}</span>
            </div>
            <Progress frac={visible / budget} tone={visible <= budget ? "good" : "warn"} />
            <small className="muted">
              {visible <= budget ? `${Math.round((100 * visible) / budget)}% of the Quest budget (${budget / 1000}K)` : `over the Quest budget (${budget / 1000}K): the export adds a decimated VR file`}
            </small>
          </div>
        )}
        <ActionButton kind="primary" disabled={!changed || busy}
          onClick={() => api.setParams(job, seed, {
            crop: { tau },
            canonicalize: { height_m: Number(height) },
            export: { box: sentBox },
          }).then(putJob)}>
          {busy ? "Applying…" : "Apply and export"}
        </ActionButton>
        <p className="muted small">
          {changed ? "Reruns crop, upright and export (about 15 s); the preview is approximate at the box's edges." : "These are the settings of the current export."}
        </p>
      </aside>
    </div>
  );
}
