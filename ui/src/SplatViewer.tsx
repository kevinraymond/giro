// Orbit-controls splat viewer (Spark on three.js).
//
// giro's exports keep the 3DGS file convention (y down); Spark reads that as-is, so by
// default the mesh is turned 180 degrees about Z into the canonical frame: up +Y, feet on
// y = 0, the hero side facing +Z (docs/FINDINGS.md, "Export and viewing"). A splat still in COLMAP's world frame (a
// training checkpoint) passes `matrix` (world -> viewer, row-major 4x4) instead.
//
// Changing `url` swaps the splat in place once the new one has loaded, keeping the camera,
// so a sequence of checkpoints plays without the view jumping.

import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export interface SplatInfo {
  count: number;
  size: [number, number, number]; // width, height, depth in the viewer frame
}

interface Scene {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  mesh: SplatMesh | null;
  framed: boolean;
}

export function SplatViewer({
  url, matrix, autoRotate = false, floor = true, onInfo, onLoad,
}: {
  url: string;
  matrix?: number[][];
  autoRotate?: boolean;
  floor?: boolean;
  onInfo?: (info: SplatInfo) => void;
  /** Each time a splat has loaded and replaced the previous one: for editors that change it. */
  onLoad?: (mesh: SplatMesh, scene: THREE.Scene) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const world = useRef<Scene | null>(null);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");
  const infoRef = useRef(onInfo);
  infoRef.current = onInfo;
  const loadRef = useRef(onLoad);
  loadRef.current = onLoad;

  // The renderer, scene and camera live as long as the component.
  useEffect(() => {
    const el = host.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x121416);
    scene.add(new SparkRenderer({ renderer }));
    const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 200);
    camera.position.set(0, 1.2, 4);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 1.2;
    controls.target.set(0, 0.85, 0);
    const stopSpin = () => (controls.autoRotate = false);
    controls.addEventListener("start", stopSpin);
    let grid: THREE.GridHelper | null = null;
    if (floor) {
      grid = new THREE.GridHelper(4, 20, 0x3a4048, 0x24282d); // 20 cm cells
      scene.add(grid);
    }
    world.current = { scene, camera, controls, mesh: null, framed: false };

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
      controls.removeEventListener("start", stopSpin);
      controls.dispose();
      const mesh = world.current?.mesh;
      if (mesh) dispose(scene, mesh);
      world.current = null;
      grid?.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [autoRotate, floor]);

  // Load (or swap in) the splat.
  const matrixKey = matrix ? JSON.stringify(matrix) : "";
  useEffect(() => {
    const w = world.current;
    if (!w) return;
    let alive = true;
    if (!w.mesh) setState("loading");
    const mesh = new SplatMesh({ url });
    if (matrix) {
      const m = new THREE.Matrix4().set(...(matrix.flat() as Parameters<THREE.Matrix4["set"]>));
      m.decompose(mesh.position, mesh.quaternion, mesh.scale);
    } else {
      mesh.quaternion.set(0, 0, 1, 0);
    }
    mesh.initialized
      .then(() => {
        if (!alive || world.current !== w) {
          mesh.dispose();
          return;
        }
        w.scene.add(mesh);
        if (w.mesh) dispose(w.scene, w.mesh);
        w.mesh = mesh;
        const info = bounds(mesh);
        if (!w.framed) {
          frame(w, info.lo, info.hi);
          w.framed = true;
        }
        setState("ready");
        infoRef.current?.({ count: info.count, size: info.size });
        loadRef.current?.(mesh, w.scene);
      })
      .catch((e: unknown) => {
        if (!alive) return;
        setError(String(e));
        setState("error");
      });
    return () => {
      alive = false;
    };
    // matrixKey stands for `matrix`, compared by value.
  }, [url, matrixKey]);

  return (
    <div className="viewer" ref={host}>
      {state === "loading" && <div className="viewer-overlay">Loading splat…</div>}
      {state === "error" && <div className="viewer-overlay bad">Could not load the splat: {error}</div>}
    </div>
  );
}

function dispose(scene: THREE.Scene, mesh: SplatMesh) {
  scene.remove(mesh);
  mesh.dispose();
}

/** Bounds of the splat centers in the viewer frame. */
function bounds(mesh: SplatMesh) {
  mesh.updateMatrix();
  const m = mesh.matrix;
  const lo = new THREE.Vector3(Infinity, Infinity, Infinity);
  const hi = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
  const v = new THREE.Vector3();
  let count = 0;
  mesh.packedSplats?.forEachSplat((_i, c) => {
    count++;
    v.copy(c).applyMatrix4(m);
    lo.min(v);
    hi.max(v);
  });
  const size: [number, number, number] = [hi.x - lo.x, hi.y - lo.y, hi.z - lo.z];
  return { lo, hi, size, count };
}

/** Put the whole splat in view from +Z, slightly above. */
function frame(w: Scene, lo: THREE.Vector3, hi: THREE.Vector3) {
  const { camera, controls } = w;
  const center = lo.clone().add(hi).multiplyScalar(0.5);
  const radius = 0.5 * lo.distanceTo(hi);
  const dist = (radius / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.05;
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0, 0.25, 1).normalize().multiplyScalar(dist));
  camera.near = dist / 100;
  camera.far = dist * 20;
  camera.updateProjectionMatrix();
  controls.maxDistance = dist * 4;
  controls.minDistance = radius * 0.2;
}
