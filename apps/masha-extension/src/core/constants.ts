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

/*
 * Browser-free core. Nothing in `src/core/` may import `chrome.*`/`browser.*`
 * or any platform API — this is the unit that a future Falkon (Python) plugin
 * reimplements. See docs/FALKON.md for the porting contract.
 *
 * Mirror of packages/shared-types/src/shared_types/constants.py — keep in sync.
 */

/** Default OpenAI-compatible Lemonade base URL (already includes ``/v1``). */
export const DEFAULT_LEMONADE_URL = 'http://localhost:13305/v1';

/** Canonical default model. Mirror of shared_types ``DEFAULT_MODEL``. */
export const DEFAULT_MODEL = 'LMX-Omni-5.5B-Lite';

/** ``"Auto"`` lets the model detect the source language (general web use). */
export const DEFAULT_SOURCE_LANG = 'Auto';
export const DEFAULT_TARGET_LANG = 'English';

/** Sampling defaults mirror MAGE's text path (low temp for faithful output). */
export const DEFAULT_MAX_TOKENS = 1024;
export const DEFAULT_TEMPERATURE = 0.3;

/** Cap on surrounding page context sent as reference (chars). Page-level. */
export const MAX_CONTEXT_CHARS = 1800;
