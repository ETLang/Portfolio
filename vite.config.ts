import { defineConfig } from 'vite';
import mkcert from 'vite-plugin-mkcert';

export default defineConfig({
  // Tells Vite your live site will be served from a subfolder named after your repo
  base: '/Portfolio/',
  
  server: {
    host: true,
    port: 5173,
    https: true, // required for mkcert to work
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  // mkcert generates/installs a local HTTPS cert - a dev-server-only concern with no
  // business running under Vitest, where it's just a source of network/filesystem flakiness.
  // Vitest sets process.env.VITEST, so skip the plugin entirely under test.
  plugins: process.env.VITEST ? [] : [mkcert()],
});