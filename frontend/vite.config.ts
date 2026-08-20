/// <reference types="vitest/config" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  test: {
    environment: 'node',
    include: ['src/**/*.test.ts'],
    // fake-indexeddb installs a working IndexedDB into the global scope so the
    // storage layer can be exercised for real rather than behind a mock.
    setupFiles: ['./src/storage/__tests__/setup.ts'],
  },
  server: {
    fs: {
      // exercises.json lives at the repo root and is shared with the Python
      // engine, so the dev server has to be allowed to read above frontend/.
      allow: ['..'],
    },
  },
})
