import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import {defineConfig} from 'vite';

export default defineConfig(() => {
  const backendUrl = process.env.VITE_BACKEND_URL ?? 'http://127.0.0.1:8764';

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      host: '127.0.0.1',
      port: 8765,
      hmr: process.env.DISABLE_HMR !== 'true',
      proxy: {
        '/api': backendUrl,
      },
    },
  };
});
