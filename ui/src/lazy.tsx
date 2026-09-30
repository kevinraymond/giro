// Spark and three.js are most of the bundle: load them with the first view that needs them.

import { lazy, Suspense, type ComponentProps, type ReactNode } from "react";

const Viewer = lazy(() => import("./SplatViewer").then((m) => ({ default: m.SplatViewer })));
const Cameras = lazy(() => import("./CamerasTab").then((m) => ({ default: m.CamerasTab })));
const Training = lazy(() => import("./TrainingTab").then((m) => ({ default: m.TrainingTab })));
const Crop = lazy(() => import("./CropTab").then((m) => ({ default: m.CropTab })));
const Vr = lazy(() => import("./VrView"));

const loading = (children: ReactNode) => <Suspense fallback={<div className="viewer"><div className="viewer-overlay">Loading the viewer…</div></div>}>{children}</Suspense>;

export const SplatViewer = (p: ComponentProps<typeof Viewer>) => loading(<Viewer {...p} />);
export const CamerasTab = (p: ComponentProps<typeof Cameras>) => loading(<Cameras {...p} />);
export const TrainingTab = (p: ComponentProps<typeof Training>) => loading(<Training {...p} />);
export const CropTab = (p: ComponentProps<typeof Crop>) => loading(<Crop {...p} />);
export const VrView = (p: ComponentProps<typeof Vr>) => loading(<Vr {...p} />);
