import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Lane viz owns port 5180 ONLY. Forbidden ports (other projects / Ollama):
// 8000, 5173, 8765, 11434. Never change this without coordinating cross-lane.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5180,
    strictPort: true,
    host: true,
    // Allow reaching the dev server by the tailnet hostname `forge` (Vite blocks
    // unknown Host headers by default → "Blocked request. This host ... not allowed").
    allowedHosts: ['forge', 'localhost'],
    headers: {
      // SharedArrayBuffer for the force-layout worker (zero-copy position SAB).
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    port: 5180,
    strictPort: true,
    allowedHosts: ['forge', 'localhost'],
  },
  worker: {
    format: 'es',
  },
});
