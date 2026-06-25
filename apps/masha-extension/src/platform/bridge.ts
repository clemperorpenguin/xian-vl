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
 * The platform seam. Everything browser-specific lives behind this interface;
 * the browser-free `core/` knows nothing about it. A future Falkon plugin
 * reimplements ONLY this contract in Python (PyFalkon settings, its context-menu
 * hook, QWebEngine JS injection). See docs/FALKON.md.
 */

import { DEFAULT_LEMONADE_URL, DEFAULT_SOURCE_LANG, DEFAULT_TARGET_LANG } from '../core/constants';

/** Persisted user configuration. */
export interface MashaConfig {
  /** OpenAI-compatible Lemonade base URL (normalized to end in ``/v1``). */
  serverUrl: string;
  /** ``"Auto"`` or an explicit source language name. */
  sourceLang: string;
  targetLang: string;
  /** Optional stylistic register terms. */
  styles: string[];
}

export const DEFAULT_CONFIG: MashaConfig = {
  serverUrl: DEFAULT_LEMONADE_URL,
  sourceLang: DEFAULT_SOURCE_LANG,
  targetLang: DEFAULT_TARGET_LANG,
  styles: [],
};

/** A captured selection plus the surrounding page context for the model. */
export interface SelectionContext {
  /** The text to translate. */
  text: string;
  /** Surrounding page text (title + section), reference only. */
  context: string;
  /** True when the selection lives in an editable <input>/<textarea>. */
  isEditable: boolean;
}

export interface TranslationOutcome {
  selection: string;
  translation: string;
  /** Editable selections are replaced in place; others use the overlay. */
  isEditable: boolean;
}

/**
 * The full surface a platform must provide. Implement this once per platform
 * (WXT for Chrome/Firefox today; PyFalkon later) and the rest is reused.
 */
export interface PlatformBridge {
  /** Load persisted config, falling back to {@link DEFAULT_CONFIG}. */
  getConfig(): Promise<MashaConfig>;
  setConfig(config: Partial<MashaConfig>): Promise<void>;

  /** Register the user-facing trigger (context-menu entry / hotkey). */
  onTranslateCommand(handler: () => void | Promise<void>): void;

  /** Capture the current selection and surrounding page context. */
  getSelectionContext(): SelectionContext | null;

  /** Show a non-destructive overlay near the selection (read-only text). */
  showOverlay(outcome: TranslationOutcome): void;
  /** Show a transient error near the selection. */
  showError(message: string): void;

  /** Replace the selection in place (editable inputs only). */
  replaceSelection(text: string): void;
}
