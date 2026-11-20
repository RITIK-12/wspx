import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The control plane serves index.html at / and hashed files at /assets/.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../src/wspx/control/ui",
    emptyOutDir: true,
    assetsDir: "assets",
    sourcemap: false,
  },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});
