import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    fs: {
      // exercises.json lives at the repo root and is shared with the Python
      // engine, so the dev server has to be allowed to read above frontend/.
      allow: ['..'],
    },
  },
})
