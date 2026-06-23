import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Lane console owns port 5180 ONLY. Forbidden ports (other projects / Ollama):
// 8000, 5173, 8765, 11434. Never change this without coordinating cross-lane.
//
// The recluster backend (lane stream) lives on :8010. We proxy `/api/*` to it so
// the browser talks same-origin (no CORS / host headaches). `/api/params` →
// `:8010/params`, `/api/recluster` → `:8010/recluster` (the `/api` prefix is
// stripped). If :8010 is down the front falls back to the static graph.json.
// A worker can point the proxy at its own backend (e.g. a worktree instance on a
// spare port) via VITE_API_TARGET, without touching this committed default.
const API_TARGET = process.env.VITE_API_TARGET || 'http://localhost:8010';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    host: true,
    // Allow reaching the dev server by the tailnet hostname `forge` (Vite blocks
    // unknown Host headers by default → "Blocked request. This host ... not allowed").
    allowedHosts: ['forge', 'localhost'],
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ''),
      },
    },
  },
  preview: {
    port: 5180,
    strictPort: true,
    allowedHosts: ['forge', 'localhost'],
  },
});
