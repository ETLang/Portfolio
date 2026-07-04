import { defineConfig } from 'vite';
import mkcert from 'vite-plugin-mkcert';

export default defineConfig({
  // Tells Vite your live site will be served from a subfolder named after your repo
  base: '/portfolio/',
  
  server: {
    host: true,
    port: 5173,
    https: true, // required for mkcert to work
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  plugins: [mkcert()],
});