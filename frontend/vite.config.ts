import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Frontend dev server; the backend remains at its own origin.
    host: true,
    port: 5173
  }
})
