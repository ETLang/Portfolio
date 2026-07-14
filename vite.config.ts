import { defineConfig, type Plugin } from 'vite';
import mkcert from 'vite-plugin-mkcert';

// Relays console.log/warn/error lines POSTed from the client (see main.ts's
// "CONSOLE LOG CAPTURE" section) into this terminal. Exists so mobile-device console
// output - which has no reliable path to a desktop devtools/logcat view, see CLAUDE.md's
// mobile-only-bug notes - shows up somewhere without needing the device physically tethered.
function consoleLogRelay(): Plugin {
  return {
    name: 'console-log-relay',
    configureServer(server) {
      server.middlewares.use('/__consolelog', (req, res) => {
        if (req.method !== 'POST') {
          res.statusCode = 405;
          res.end();
          return;
        }
        let body = '';
        req.on('data', (chunk) => { body += chunk; });
        req.on('end', () => {
          console.log(`[client] ${body}`);
          res.statusCode = 204;
          res.end();
        });
      });
    },
  };
}

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
  plugins: process.env.VITEST ? [] : [mkcert(), consoleLogRelay()],
});