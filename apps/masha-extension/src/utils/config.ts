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

/* WXT-side realization of config persistence (PlatformBridge.getConfig/setConfig). */

import { normalizeLemonadeBaseUrl } from './lemonadeUrl';
import { DEFAULT_CONFIG, MashaConfig } from '../platform/bridge';

const STORAGE_KEY = 'config';

/** Load persisted config merged over defaults. */
export async function getConfig(): Promise<MashaConfig> {
  const stored = await chrome.storage.local.get([STORAGE_KEY]);
  const cfg = { ...DEFAULT_CONFIG, ...(stored[STORAGE_KEY] || {}) } as MashaConfig;
  cfg.serverUrl = normalizeLemonadeBaseUrl(cfg.serverUrl);
  return cfg;
}

/** Persist a partial config update. */
export async function setConfig(update: Partial<MashaConfig>): Promise<void> {
  const current = await getConfig();
  const next: MashaConfig = { ...current, ...update };
  next.serverUrl = normalizeLemonadeBaseUrl(next.serverUrl);
  await chrome.storage.local.set({ [STORAGE_KEY]: next });
}
