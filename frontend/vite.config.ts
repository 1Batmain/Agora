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

// Proxy `/api/*` → backend (:8010), préfixe `/api` retiré. MÊME config en dev (`server`)
// ET en preview (build servi en prod) : sans ça, l'app publique ne joindrait pas l'API.
const apiProxy = {
  '/api': {
    target: API_TARGET,
    changeOrigin: true,
    rewrite: (p: string) => p.replace(/^\/api/, ''),
  },
};

// Hôtes autorisés : localhost, la machine `forge`, et le hostname public Tailscale Funnel
// (Vite bloque les Host inconnus par défaut → « Blocked request »).
const allowedHosts = ['forge', 'localhost', 'forge.tail0b8aa8.ts.net'];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    host: true,
    allowedHosts,
    proxy: apiProxy,
  },
  preview: {
    port: 5180,
    strictPort: true,
    host: true,
    allowedHosts,
    proxy: apiProxy,
  },
});
