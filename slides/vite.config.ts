import { defineConfig } from "vite";

// The deck reuses the repository's own files: the logo from assets/ (served as the public directory) and the
// figures from reports/figures/ (imported by relative path), so nothing is copied into slides/.
export default defineConfig({
  publicDir: "../assets",
  server: { fs: { allow: [".."] } },
});
