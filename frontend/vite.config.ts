import { resolve } from "node:path"
import { defineConfig } from "vite"
import vue from "@vitejs/plugin-vue"

const API_HOST = process.env.IMG_SEARCH_HOST ?? "127.0.0.1"
const API_PORT = Number(process.env.IMG_SEARCH_PORT ?? 8000)

export default defineConfig({
  base: "./",
  plugins: [vue()],
  server: {
    strictPort: true,
    proxy: {
      "/api": {
        target: `http://${API_HOST}:${API_PORT}`,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: resolve(__dirname, "../backend/frontend_dist"),
    emptyOutDir: true,
  },
})
