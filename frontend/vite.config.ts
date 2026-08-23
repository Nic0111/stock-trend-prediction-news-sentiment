import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

const overviewMediaCache: Plugin = {
  name: "stocksense-overview-media-cache",
  configureServer(server) {
    server.middlewares.use((request, response, next) => {
      if (request.originalUrl?.startsWith("/media/overview-world/")) {
        response.setHeader("Cache-Control", "public, max-age=3600");
      }
      next();
    });
  },
};

export default defineConfig({
  plugins: [react(), overviewMediaCache],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8000",
    },
  },
  build: {
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          react: ["react", "react-dom", "react-router-dom"],
          charts: ["recharts"],
          query: ["@tanstack/react-query"],
        },
      },
    },
  },
});
