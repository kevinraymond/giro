import { defineConfig } from "vite";

// The all-in-one viewer: one classic script with three and Spark inside, which
// src/giro/viewer_html.py inlines into a single HTML file next to the splat.
export default defineConfig({
  build: {
    outDir: "dist/standalone",
    emptyOutDir: true,
    lib: { entry: "standalone/viewer.ts", name: "GiroViewer", formats: ["iife"], fileName: () => "viewer.js" },
    rollupOptions: { output: { inlineDynamicImports: true } },
    minify: true,
    chunkSizeWarningLimit: 8000,
  },
  worker: { format: "iife" },
});
