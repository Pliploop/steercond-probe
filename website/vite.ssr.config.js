import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Renders the app to a string in Node: proves it mounts without a browser (effects do not run under SSR).
export default defineConfig({
  plugins: [react()],
  build: { ssr: 'src/ssr-check.jsx', outDir: 'dist-ssr', emptyOutDir: true },
})
