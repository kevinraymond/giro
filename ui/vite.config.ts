import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` serves the UI with hot reload and forwards the API to `giro serve`.
const api = process.env.GIRO_API ?? "http://127.0.0.1:8470";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      "/api/events": { target: api.replace(/^http/, "ws"), ws: true },
      "/api": api,
      "/files": api,
    },
  },
  build: { chunkSizeWarningLimit: 4000 },
});
