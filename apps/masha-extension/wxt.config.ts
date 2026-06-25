/*
 * Masha — Browser extension selection translator.
 * Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)
 */

import { defineConfig } from 'wxt';

// See https://wxt.dev/api/config.html
export default defineConfig({
  srcDir: 'src',
  outDir: 'dist',
  manifest: {
    name: "MASHA Translate",
    description: "Multilingual Access & Site Handling Assistant",
    version: "1.0.0",
    permissions: ["storage", "contextMenus"],
    host_permissions: ["<all_urls>"],
    // Firefox refuses to install an unsigned MV3 extension without an explicit id.
    browser_specific_settings: {
      gecko: { id: "masha@pendragon.systems", strict_min_version: "121.0" }
    }
  }
});
