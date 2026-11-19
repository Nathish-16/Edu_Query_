import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api to FastAPI so the browser always talks to the same origin.
const API_TARGET = process.env.VITE_API_HOST || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        secure: false,
      },
    },
  },
  preview: {
    port: 5173,
    proxy: {
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
      },
    },
  },
})
