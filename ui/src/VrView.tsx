// View a splat in WebXR. Open this page in the Quest browser
// and press Enter VR. The canonical frame has the feet at y = 0 and the hero side facing +Z,
// so in a 'local-floor' space the subject stands on the real floor, 1.5 m ahead, facing you.
//
// Left stick walks (on the floor plane), right stick turns. The page measures its own frame
// times and posts them to the server (/api/xr/stats), which is how giro's Quest budget is
// measured rather than guessed (docs/FINDINGS.md, "VR on Quest 3").

import { SparkRenderer, SparkXr, SplatMesh, type XrGamepads } from "@sparkjsdev/spark";
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

export interface VrStats {
  label: string;
  primer: Primer;
  target_hz: number | null; // what the page asked the headset for (?hz=)
  scale: number; // WebXR framebuffer scale (?scale=)
  max_std_dev: number;
  splats: number;
  frames: number;
  seconds: number;
  fps: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  slow_frames: number; // frames over 1.5x the display's frame interval
  display_hz: number | null;
  xr: boolean;
  user_agent: string;
}

const DISTANCE = 1.5; // m from the viewer's start to the subject

export interface VrItem {
  url: string;
  label: string;
  maxStdDev?: number; // how far out each Gaussian is drawn (Spark default sqrt(8) = 2.83)
  primer?: Primer;
}

/** What else in the pass writes depth (Spark's splats never do). fosfora measured on the same
 *  Quest that blended sprites drew ~3x cheaper when any opaque depth-writing draw was in the pass
 *  (an Adreno pass-mode heuristic, presumably). "floor": the floor ring writes depth, as it always
 *  did; "none": nothing writes depth; "opaque": nothing but a 1 mm opaque quad under the floor. */
export type Primer = "floor" | "none" | "opaque";

/** One splat, or a benchmark sequence: after Enter VR each item is shown in turn, warmed up
 *  for 3 s and measured for 10 s, and its frame times are reported under its label. */
export default function VrView({ items, next }: { items: VrItem[]; next?: { href: string; label: string } }) {
  const host = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState("Loading the splat…");
  const [stats, setStats] = useState<VrStats | null>(null);
  const [supported, setSupported] = useState<boolean | null>(null);

  useEffect(() => {
    const el = host.current!;
    // Options: ?hz=90 asks the headset for that refresh rate; ?scale=1 renders every pixel.
    // The defaults are what held 72 fps on the Quest 3 (docs/FINDINGS.md, "VR on Quest 3").
    const opts = new URLSearchParams(location.hash.split("?")[1] ?? "");
    const targetHz = Number(opts.get("hz") ?? 72);
    const scale = Number(opts.get("scale") ?? 0.6);
    const renderer = new THREE.WebGLRenderer({ antialias: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);
    const spark = new SparkRenderer({ renderer });
    scene.add(spark);
    // How far out each Gaussian is drawn: Spark's default sqrt(8) costs fill for little visible gain.
    const defaultStdDev = Number(opts.get("std") ?? 2);
    spark.maxStdDev = defaultStdDev;

    // Controllers move the camera's parent: the rig.
    const rig = new THREE.Group();
    scene.add(rig);
    const camera = new THREE.PerspectiveCamera(70, 1, 0.05, 100);
    camera.position.set(0, 1.6, 0);
    rig.add(camera);

    const floor = new THREE.Mesh(
      new THREE.RingGeometry(0.45, 0.5, 64).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0x3a4048, transparent: true, opacity: 0.8 }),
    );
    floor.position.set(0, 0.002, -DISTANCE);
    scene.add(floor);
    const primer = new THREE.Mesh(new THREE.PlaneGeometry(0.001, 0.001).rotateX(-Math.PI / 2), new THREE.MeshBasicMaterial({ color: 0x000000 }));
    primer.position.set(0, -0.01, -DISTANCE);
    scene.add(primer);
    const setPrimer = (p: Primer) => {
      floor.material.depthWrite = p === "floor";
      primer.visible = p === "opaque";
    };
    setPrimer(items[0].primer ?? "floor");

    // Status inside VR (nobody wearing the headset sees the page): a text panel to the left of the
    // subject. It is blended and writes no depth, so it does not change what the primer test measures.
    const panelCanvas = document.createElement("canvas");
    panelCanvas.width = 1024;
    panelCanvas.height = 256;
    const panelTexture = new THREE.CanvasTexture(panelCanvas);
    panelTexture.colorSpace = THREE.SRGBColorSpace;
    const panel = new THREE.Mesh(new THREE.PlaneGeometry(0.8, 0.2),
      new THREE.MeshBasicMaterial({ map: panelTexture, transparent: true, depthWrite: false, depthTest: false }));
    panel.renderOrder = 10;
    panel.position.set(-0.9, 1.5, -DISTANCE + 0.2);
    panel.rotation.y = 0.5;
    scene.add(panel);
    const show = (lines: string[]) => {
      const g = panelCanvas.getContext("2d")!;
      g.clearRect(0, 0, panelCanvas.width, panelCanvas.height);
      g.fillStyle = "rgba(16, 18, 20, 0.85)";
      g.fillRect(0, 0, panelCanvas.width, panelCanvas.height);
      g.fillStyle = "#e8eaed";
      g.font = "40px sans-serif";
      lines.slice(0, 4).forEach((line, i) => g.fillText(line, 24, 60 + i * 60));
      panelTexture.needsUpdate = true;
    };
    const say = (text: string) => {
      setStatus(text);
      show([items.length > 1 ? `Benchmark: ${items.length} splats` : items[0].label, text]);
    };
    say("Loading the splat…");

    let mesh: SplatMesh | null = null;
    let splats = 0;
    let label = items[0].label;
    let alive = true;
    async function load(item: VrItem) {
      const next = new SplatMesh({ url: item.url });
      // File frame -> canonical frame (docs/FINDINGS.md, "Export and viewing"), then face the viewer: the hero side (+Z)
      // already faces the viewer at z = 0, so only move it ahead.
      next.quaternion.set(0, 0, 1, 0);
      next.position.set(0, 0, -DISTANCE);
      await next.initialized;
      if (!alive) return next.dispose();
      if (mesh) {
        scene.remove(mesh);
        mesh.dispose();
      }
      scene.add(next);
      mesh = next;
      splats = next.packedSplats?.numSplats ?? 0;
      label = item.label;
      say(`${item.label}: ${splats.toLocaleString()} splats`);
    }
    load(items[0]).catch((e: unknown) => say(`Could not load the splat: ${e}`));

    const sequence = items.length > 1;
    const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));
    async function runSequence() {
      for (const [i, item] of items.entries()) {
        if (!alive || !renderer.xr.isPresenting) return;
        say(`${i + 1}/${items.length} ${item.label}: loading`);
        try {
          await load(item);
        } catch (e) {
          // Skip it, but say so: in VR nobody sees the status line.
          fetch("/api/xr/stats", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ label: item.label, error: String(e), xr: renderer.xr.isPresenting }) }).catch(() => {});
          continue;
        }
        spark.maxStdDev = item.maxStdDev ?? defaultStdDev;
        setPrimer(item.primer ?? "floor");
        say(`${i + 1}/${items.length} ${item.label}: warming up`);
        await wait(3000); // sorting settles, GPU caches warm
        frames.length = 0;
        say(`${i + 1}/${items.length} ${item.label}: measuring 10 s`);
        await wait(10000);
        report();
      }
      say(next ? `Done. Exit VR, then tap "${next.label}" on the page.` : `Done: ${items.length} measurements sent. You can exit VR.`);
    }

    const xr = new SparkXr({
      renderer, // SparkXr adds its own Enter VR button to the page (removed on cleanup)
      mode: "vr",
      referenceSpaceType: "local-floor",
      fixedFoveation: 1,
      frameBufferScaleFactor: scale,
      enableHands: false,
      button: { enterVrText: "Enter VR", exitVrText: "Exit VR" },
      onReady: (ok) => setSupported(ok),
      onEnterXr: () => {
        frames.length = 0;
        const session = renderer.xr.getSession() as (XRSession & { updateTargetFrameRate?: (hz: number) => Promise<void> }) | null;
        if (targetHz && session?.updateTargetFrameRate) session.updateTargetFrameRate(targetHz).catch(() => {});
        rig.position.set(0, 0, 0);
        rig.quaternion.identity();
        if (sequence) runSequence().catch((e: unknown) => say(`Stopped: ${e}`));
      },
      controllers: {
        moveDirection: true,
        getMove: (g: XrGamepads) => {
          const a = g.left?.axes ?? [];
          return new THREE.Vector3(a[2] ?? 0, 0, a[3] ?? 0);
        },
        getRotate: (g: XrGamepads) => new THREE.Vector3(g.right?.axes?.[2] ?? 0, 0, 0),
      },
    });

    // Frame timing: intervals between animation frames, reported every 5 s.
    const frames: number[] = [];
    let last = 0;
    let reportAt = performance.now() + 5000;
    const report = () => {
      if (frames.length < 10) return;
      const sorted = [...frames].sort((a, b) => a - b);
      const q = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
      const session = renderer.xr.getSession();
      const hz = (session as unknown as { frameRate?: number } | null)?.frameRate ?? null;
      const interval = hz ? 1000 / hz : q(0.5);
      const total = frames.reduce((s, x) => s + x, 0);
      const s: VrStats = {
        label, primer: primer.visible ? "opaque" : floor.material.depthWrite ? "floor" : "none", target_hz: targetHz, scale, max_std_dev: Math.round(spark.maxStdDev * 100) / 100, splats, frames: frames.length, seconds: Math.round(total / 10) / 100,
        fps: Math.round((1000 * frames.length / total) * 10) / 10,
        p50_ms: Math.round(q(0.5) * 100) / 100, p95_ms: Math.round(q(0.95) * 100) / 100,
        p99_ms: Math.round(q(0.99) * 100) / 100,
        slow_frames: frames.filter((f) => f > 1.5 * interval).length,
        display_hz: hz, xr: renderer.xr.isPresenting, user_agent: navigator.userAgent,
      };
      setStats(s);
      fetch("/api/xr/stats", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(s) }).catch(() => {});
      frames.length = 0;
    };

    const resize = () => {
      if (renderer.xr.isPresenting) return;
      const w = el.clientWidth, h = el.clientHeight;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el);
    resize();
    renderer.setAnimationLoop((time) => {
      if (last) frames.push(time - last);
      last = time;
      if (renderer.xr.isPresenting) xr.updateControllers(camera);
      renderer.render(scene, camera);
      if (!sequence && time > reportAt) {
        report();
        reportAt = time + 5000;
      }
    });
    return () => {
      alive = false;
      renderer.setAnimationLoop(null);
      renderer.xr.getSession()?.end().catch(() => {});
      ro.disconnect();
      if (mesh) {
        scene.remove(mesh);
        mesh.dispose();
      }
      panelTexture.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      xr.element?.remove();
    };
    // items is a new array on every render of the parent: key the effect on its content.
  }, [JSON.stringify(items)]);

  return (
    <div className="vr">
      <div className="vr-view" ref={host} />
      <div className="vr-panel">
        <strong>{items.length > 1 ? `Benchmark: ${items.length} splats` : items[0].label}</strong>
        <span className="muted small">{status}</span>
        {supported === false && (
          <p className="note warn">
            This browser has no immersive VR here. Open the page in the Quest browser, over https or at
            localhost (adb reverse), then press Enter VR.
          </p>
        )}
        {next && <a href={next.href}>{next.label}</a>}
        {stats && (
          <span className="small num">
            {stats.fps} fps · p95 {stats.p95_ms} ms{stats.display_hz ? ` · ${stats.display_hz} Hz` : ""} · {stats.slow_frames} slow
          </span>
        )}
      </div>
    </div>
  );
}
