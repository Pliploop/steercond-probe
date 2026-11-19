import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// GitHub Pages serves the site from /<repo>/; override with BASE=/other/ npm run build
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE || '/steercond-probe/',
})
