import { defineConfig } from "vite";

export default defineConfig({
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 900 },
  test: { environment: "node" },
});
