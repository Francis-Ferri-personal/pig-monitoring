import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    proxy: {
      '/upload': {
        target: 'http://localhost:8008',
        changeOrigin: true,
      },
      '/videos': {
        target: 'http://localhost:8008',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8008',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
