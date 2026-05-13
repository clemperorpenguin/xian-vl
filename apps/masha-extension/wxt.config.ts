import { defineConfig } from 'wxt';

// See https://wxt.dev/api/config.html
export default defineConfig({
  srcDir: 'src',
  outDir: 'dist',
  manifest: {
    name: "MASHA Translate",
    description: "Multilingual Access & Site Handling Assistant",
    version: "1.0.0",
    permissions: ["storage", "activeTab", "scripting"],
    host_permissions: ["<all_urls>"]
  }
});
